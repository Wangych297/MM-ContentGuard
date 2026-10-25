"""
文件功能：语义模型训练脚本。实现 SemanticMemeModel 的训练流程，
使用 ClassBalancedFocalLoss 处理类别不平衡问题，支持多任务
联合训练和检查点保存。

核心类：
- ClassBalancedFocalLoss: 类别平衡的 Focal Loss

核心函数：
- train: 语义模型训练入口
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from common.config import load_yaml, resolve_path
from semantic_model.dataset import SemanticMemeDataset
from semantic_model.labels import BINARY_LABELS, SEMANTIC_TYPE_LABELS
from semantic_model.metrics import semantic_metrics
from semantic_model.model import build_semantic_model_from_config


TENSOR_INPUT_KEYS = {
    "text_input_ids",
    "text_attention_mask",
    "text_description_input_ids",
    "text_description_attention_mask",
    "meme_description_input_ids",
    "meme_description_attention_mask",
    "pixel_values",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for label_id in range(num_classes):
        count = counts.get(label_id, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _effective_number_weights(
    labels: list[int],
    num_classes: int,
    device: torch.device,
    beta: float = 0.999,
) -> torch.Tensor:
    counts = Counter(labels)
    weights = []
    for label_id in range(num_classes):
        count = max(1, counts.get(label_id, 1))
        effective_num = 1.0 - beta**count
        weights.append((1.0 - beta) / effective_num)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor / tensor.sum() * num_classes


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        labels: list[int],
        num_classes: int,
        device: torch.device,
        beta: float = 0.999,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "weights",
            _effective_number_weights(labels, num_classes, device, beta),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        weights = self.weights.gather(0, target)
        focal_factor = (1.0 - target_probs).clamp_min(1e-6).pow(self.gamma)
        return (-weights * focal_factor * target_log_probs).mean()


def _classification_criterion(
    labels: list[int],
    num_classes: int,
    device: torch.device,
    train_cfg: dict[str, Any],
    prefix: str,
) -> nn.Module:
    loss_kind = str(train_cfg.get(f"{prefix}_loss_kind", "cross_entropy")).lower()
    if loss_kind in {"ce", "cross_entropy", "weighted_cross_entropy"}:
        use_weights = bool(
            train_cfg.get(
                f"{prefix}_class_weight_loss",
                train_cfg.get("class_weight_loss", True),
            )
        )
        weights = _class_weights(labels, num_classes, device) if use_weights else None
        return nn.CrossEntropyLoss(weight=weights)
    if loss_kind in {"focal", "class_balanced_focal", "cb_focal"}:
        beta = float(train_cfg.get(f"{prefix}_focal_beta", train_cfg.get("focal_beta", 0.999)))
        gamma = float(train_cfg.get(f"{prefix}_focal_gamma", train_cfg.get("focal_gamma", 2.0)))
        return ClassBalancedFocalLoss(labels, num_classes, device, beta=beta, gamma=gamma)
    raise ValueError(f"Unsupported {prefix}_loss_kind: {loss_kind}")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in TENSOR_INPUT_KEYS}


def _loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    type_criterion: nn.Module,
    binary_criterion: nn.Module,
    modal_criterion: nn.Module,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    type_loss = type_criterion(outputs["type_logits"], batch["type_label"])
    binary_loss = binary_criterion(outputs["binary_logits"], batch["binary_label"])
    modal_loss = modal_criterion(outputs["modal_logits"], batch["modal_label"])
    total = (
        float(train_cfg.get("type_loss_weight", 1.0)) * type_loss
        + float(train_cfg.get("binary_loss_weight", 0.5)) * binary_loss
        + float(train_cfg.get("modal_loss_weight", 0.2)) * modal_loss
    )
    return total, {
        "loss": float(total.detach().item()),
        "type_loss": float(type_loss.detach().item()),
        "binary_loss": float(binary_loss.detach().item()),
        "modal_loss": float(modal_loss.detach().item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    train_cfg = train_cfg or {}
    type_criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.CrossEntropyLoss()
    modal_criterion = nn.BCEWithLogitsLoss()

    type_true: list[int] = []
    type_pred: list[int] = []
    binary_true: list[int] = []
    binary_pred: list[int] = []
    modal_true: list[list[int]] = []
    modal_pred: list[list[int]] = []
    losses: list[dict[str, float]] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(**_model_inputs(batch))
        _, loss_parts = _loss(
            outputs,
            batch,
            type_criterion,
            binary_criterion,
            modal_criterion,
            train_cfg,
        )
        losses.append(loss_parts)

        type_predictions = outputs["type_logits"].argmax(dim=-1)
        binary_predictions = outputs["binary_logits"].argmax(dim=-1)
        modal_predictions = (torch.sigmoid(outputs["modal_logits"]) >= 0.5).long()

        type_true.extend(batch["type_label"].detach().cpu().tolist())
        type_pred.extend(type_predictions.detach().cpu().tolist())
        binary_true.extend(batch["binary_label"].detach().cpu().tolist())
        binary_pred.extend(binary_predictions.detach().cpu().tolist())
        modal_true.extend(batch["modal_label"].long().detach().cpu().tolist())
        modal_pred.extend(modal_predictions.detach().cpu().tolist())

    metrics = semantic_metrics(type_true, type_pred, binary_true, binary_pred, modal_true, modal_pred)
    for key in ("loss", "type_loss", "binary_loss", "modal_loss"):
        metrics[key] = float(sum(part[key] for part in losses) / max(1, len(losses)))
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    best_metric: float,
    epochs_without_improvement: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "best_metric": best_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "type_labels": SEMANTIC_TYPE_LABELS,
            "binary_labels": BINARY_LABELS,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_metric = float(checkpoint.get("best_metric", -1.0))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    return start_epoch, best_metric, epochs_without_improvement


def _load_model_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def train(
    config: dict[str, Any],
    resume: bool = False,
    resume_checkpoint: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    epochs_override: int | None = None,
) -> Path:
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = _device()
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    processor_name = str(
        model_cfg.get("processor_name") or model_cfg.get("name", "OFA-Sys/chinese-clip-vit-large-patch14")
    )
    max_length = int(model_cfg.get("max_length", 64))
    use_descriptions = bool(model_cfg.get("use_descriptions", True))
    train_dataset = SemanticMemeDataset(
        resolve_path(data_cfg.get("train_file", "data/processed/semantic_meme/train.jsonl")),
        processor_name=processor_name,
        max_length=max_length,
        use_descriptions=use_descriptions,
    )
    val_dataset = SemanticMemeDataset(
        resolve_path(data_cfg.get("val_file", "data/processed/semantic_meme/val.jsonl")),
        processor_name=processor_name,
        max_length=max_length,
        use_descriptions=use_descriptions,
    )

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    sampler = _make_sampler(train_dataset.type_labels) if train_cfg.get("weighted_sampler", True) else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_semantic_model_from_config(config).to(device)
    freeze_epochs = int(train_cfg.get("freeze_encoder_epochs", 2))
    model.set_encoder_trainable(freeze_epochs <= 0)

    type_criterion = _classification_criterion(
        train_dataset.type_labels,
        len(SEMANTIC_TYPE_LABELS),
        device,
        train_cfg,
        "type",
    )
    binary_criterion = _classification_criterion(
        train_dataset.binary_labels,
        len(BINARY_LABELS),
        device,
        train_cfg,
        "binary",
    )
    modal_criterion = nn.BCEWithLogitsLoss()

    encoder_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("clip."):
            encoder_params.append(parameter)
        else:
            head_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(train_cfg.get("lr_encoder", 1e-5))},
            {"params": head_params, "lr": float(train_cfg.get("lr_head", 3e-4))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = resolve_path(train_cfg.get("output_dir", "checkpoints/semantic_model"))
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    progress_path = output_dir / "progress.json"
    metrics_log_path = output_dir / "metrics.jsonl"
    epochs = int(epochs_override or train_cfg.get("epochs", 15))
    patience = int(train_cfg.get("early_stop_patience", 5))
    monitor_metric = str(train_cfg.get("monitor_metric", "type_macro_f1"))
    best_metric = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    steps_per_epoch = len(train_loader)
    started_at = time.time()

    resume_path = resolve_path(resume_checkpoint) if resume_checkpoint else last_path
    if resume and resume_path.exists():
        start_epoch, best_metric, epochs_without_improvement = _load_checkpoint(
            resume_path, model, optimizer, scaler, device
        )
        print(f"Resumed from {resume_path} at epoch {start_epoch}.")
    elif resume:
        print(f"Resume requested, but checkpoint does not exist: {resume_path}. Starting fresh.")
    elif init_checkpoint:
        init_path = resolve_path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint does not exist: {init_path}")
        init_meta = _load_model_checkpoint(init_path, model, device)
        source_epoch = init_meta.get("epoch", "unknown") if isinstance(init_meta, dict) else "unknown"
        print(f"Initialized model weights from {init_path} (source epoch: {source_epoch}).")

    _write_json(
        progress_path,
        {
            "status": "running",
            "pid": os.getpid(),
            "device": str(device),
            "start_epoch": start_epoch,
            "current_epoch": start_epoch,
            "total_epochs": epochs,
            "step": 0,
            "steps_per_epoch": steps_per_epoch,
            "best_metric": best_metric,
            "monitor_metric": monitor_metric,
            "init_checkpoint": str(resolve_path(init_checkpoint)) if init_checkpoint else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    stopped_early = False
    final_progress: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs + 1):
        model.set_encoder_trainable(epoch > freeze_epochs)
        model.train()
        running = {"loss": 0.0, "type_loss": 0.0, "binary_loss": 0.0, "modal_loss": 0.0}

        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**_model_inputs(batch))
                loss, loss_parts = _loss(
                    outputs,
                    batch,
                    type_criterion,
                    binary_criterion,
                    modal_criterion,
                    train_cfg,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += loss_parts[key]

            if step % int(train_cfg.get("log_every", 20)) == 0:
                elapsed_seconds = time.time() - started_at
                percent = ((epoch - 1) * steps_per_epoch + step) / max(1, epochs * steps_per_epoch)
                progress = {
                    "status": "running",
                    "pid": os.getpid(),
                    "device": str(device),
                    "current_epoch": epoch,
                    "total_epochs": epochs,
                    "step": step,
                    "steps_per_epoch": steps_per_epoch,
                    "loss": running["loss"] / step,
                    "best_metric": best_metric,
                    "monitor_metric": monitor_metric,
                    "percent": round(percent * 100, 2),
                    "elapsed_minutes": round(elapsed_seconds / 60, 2),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                print(json.dumps(progress, ensure_ascii=False))
                _write_json(progress_path, progress)

        val_metrics = evaluate(model, val_loader, device, train_cfg)
        for key in running:
            val_metrics[f"train_{key}"] = running[key] / max(1, len(train_loader))
        print(json.dumps({"epoch": epoch, "val": val_metrics}, ensure_ascii=False, indent=2))

        metric_value = float(val_metrics.get(monitor_metric, val_metrics.get("type_macro_f1", -1.0)))
        if metric_value > best_metric:
            best_metric = metric_value
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                config,
                epoch,
                val_metrics,
                best_metric,
                epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            config,
            epoch,
            val_metrics,
            best_metric,
            epochs_without_improvement,
        )
        final_progress = {
            "status": "running",
            "pid": os.getpid(),
            "device": str(device),
            "current_epoch": epoch,
            "total_epochs": epochs,
            "step": steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "val_type_macro_f1": val_metrics["type_macro_f1"],
            "val_binary_macro_f1": val_metrics["binary_macro_f1"],
            "best_metric": best_metric,
            "monitor_metric": monitor_metric,
            "percent": round(epoch / max(1, epochs) * 100, 2),
            "checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(progress_path, final_progress)
        _append_jsonl(metrics_log_path, {"epoch": epoch, "val": val_metrics})

        if epochs_without_improvement >= patience:
            message = f"Early stopping at epoch {epoch}. Best {monitor_metric}={best_metric:.4f}"
            print(message)
            final_progress["status"] = "early_stopped"
            final_progress["message"] = message
            _write_json(progress_path, final_progress)
            stopped_early = True
            break

    if not stopped_early:
        final_progress = {
            "status": "finished",
            "pid": os.getpid(),
            "device": str(device),
            "total_epochs": epochs,
            "best_metric": best_metric,
            "monitor_metric": monitor_metric,
            "checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(progress_path, final_progress)

    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_meme.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    checkpoint = train(
        config,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        init_checkpoint=args.init_checkpoint,
        epochs_override=args.epochs,
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
