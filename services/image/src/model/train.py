"""
文件功能：图片模型训练脚本。实现图片风险分类模型的训练流程，
支持 mixup 数据增强、加权采样、学习率调度和检查点保存。

核心函数：
- train: 完整训练入口
- _mix_batch: mixup 批次混合
- _rand_bbox: CutMix 随机边界框生成
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from common.config import load_yaml, resolve_path
from model.dataset import (
    ImageDataset,
    build_eval_transform,
    build_train_transform,
)
from model.labels import ID_TO_LABEL, LABEL_TO_ID, IMAGE_LABELS
from model.metrics import classification_metrics
from model.model import build_model_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_sampler(labels: list[int], power: float = 1.0) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [(1.0 / counts[label]) ** power for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _class_weights(labels: list[int], device: torch.device, power: float = 1.0) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for label_id in range(len(IMAGE_LABELS)):
        count = counts.get(label_id, 1)
        weights.append((total / (len(IMAGE_LABELS) * count)) ** power)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _rand_bbox(images: torch.Tensor, lam: float) -> tuple[int, int, int, int]:
    height = images.shape[-2]
    width = images.shape[-1]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = int(torch.randint(width, (1,), device=images.device).item())
    center_y = int(torch.randint(height, (1,), device=images.device).item())
    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y2 = min(center_y + cut_height // 2, height)
    return x1, y1, x2, y2


def _mix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(train_cfg.get("cutmix_alpha", 0.0))
    mix_prob = float(train_cfg.get("mixup_prob", 0.0))
    if images.shape[0] < 2 or random.random() >= mix_prob or max(mixup_alpha, cutmix_alpha) <= 0:
        return images, labels, None, 1.0

    use_cutmix = cutmix_alpha > 0 and random.random() < float(train_cfg.get("cutmix_prob", 0.5))
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)
    labels_b = labels[permutation]

    if use_cutmix:
        x1, y1, x2, y2 = _rand_bbox(images, lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        patch_area = max(1, (x2 - x1) * (y2 - y1))
        lam = 1.0 - patch_area / float(images.shape[-1] * images.shape[-2])
        return mixed, labels, labels_b, lam

    mixed = lam * images + (1.0 - lam) * images[permutation]
    return mixed, labels, labels_b, lam


def _criterion_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor | None,
    lam: float,
) -> torch.Tensor:
    if labels_b is None:
        return criterion(logits, labels_a)
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def _lr_scale(global_step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and global_step <= warmup_steps:
        return max(global_step / warmup_steps, 1e-6)
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, (global_step - warmup_steps) / (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _set_lr(optimizer: torch.optim.Optimizer, scale: float) -> float:
    current_lr = 0.0
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = base_lr * scale
        current_lr = group["lr"]
    return current_lr


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=-1)
        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, ID_TO_LABEL)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
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
    config: dict,
    epoch: int,
    metrics: dict[str, Any],
    best_macro_f1: float,
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
            "best_macro_f1": best_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "labels": IMAGE_LABELS,
            "label_to_id": LABEL_TO_ID,
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
    best_macro_f1 = float(checkpoint.get("best_macro_f1", checkpoint.get("metrics", {}).get("macro_f1", -1.0)))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    return start_epoch, best_macro_f1, epochs_without_improvement


def train(
    config: dict,
    resume: bool = False,
    resume_checkpoint: str | Path | None = None,
    epochs_override: int | None = None,
) -> Path:
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = _device()
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    image_size = int(model_cfg.get("image_size", 224))
    train_dataset = ImageDataset(
        resolve_path(data_cfg.get("train_file", "data/processed/splits/image_train.jsonl")),
        transform=build_train_transform(image_size, train_cfg.get("augmentation", {})),
    )
    val_dataset = ImageDataset(
        resolve_path(data_cfg.get("val_file", "data/processed/splits/image_val.jsonl")),
        transform=build_eval_transform(image_size),
    )

    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    sampler = (
        _make_sampler(train_dataset.labels, float(train_cfg.get("sampler_weight_power", 1.0)))
        if train_cfg.get("weighted_sampler", True)
        else None
    )

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

    model = build_model_from_config(config, num_classes=len(IMAGE_LABELS)).to(device)
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 1))
    model.set_backbone_trainable(freeze_epochs <= 0)

    criterion_weight = (
        _class_weights(
            train_dataset.labels,
            device,
            float(train_cfg.get("class_weight_power", 1.0)),
        )
        if train_cfg.get("class_weight_loss", True)
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=criterion_weight,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(train_cfg.get("lr_backbone", 1e-5))},
            {"params": model.head.parameters(), "lr": float(train_cfg.get("lr_head", 3e-4))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    output_dir = resolve_path(train_cfg.get("output_dir", "checkpoints/image_model_dinov3_large"))
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    progress_path = output_dir / "progress.json"
    metrics_log_path = output_dir / "metrics.jsonl"
    epochs = int(epochs_override or train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("early_stop_patience", 4))
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(float(train_cfg.get("warmup_epochs", 0.0)) * steps_per_epoch)
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
    started_at = time.time()

    resume_path = resolve_path(resume_checkpoint) if resume_checkpoint else last_path
    if resume and resume_path.exists():
        start_epoch, best_macro_f1, epochs_without_improvement = _load_checkpoint(
            resume_path, model, optimizer, scaler, device
        )
        print(f"Resumed from {resume_path} at epoch {start_epoch}.")
    elif resume:
        print(f"Resume requested, but checkpoint does not exist: {resume_path}. Starting fresh.")

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
            "percent": 0.0,
            "best_macro_f1": best_macro_f1,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    for epoch in range(start_epoch, epochs + 1):
        if epoch == freeze_epochs + 1:
            model.set_backbone_trainable(True)
        elif epoch > freeze_epochs:
            model.set_backbone_trainable(True)
        else:
            model.set_backbone_trainable(False)

        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            images, labels_a, labels_b, lam = _mix_batch(images, labels, train_cfg)
            global_step = (epoch - 1) * steps_per_epoch + step
            current_lr = _set_lr(
                optimizer,
                _lr_scale(global_step, total_steps, warmup_steps, min_lr_ratio),
            )
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = _criterion_loss(criterion, logits, labels_a, labels_b, lam)

            scaler.scale(loss).backward()
            max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

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
                    "train_loss": running_loss / step,
                    "lr": current_lr,
                    "best_macro_f1": best_macro_f1,
                    "percent": round(percent * 100, 2),
                    "elapsed_minutes": round(elapsed_seconds / 60, 2),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                print(json.dumps(progress, ensure_ascii=False))
                _write_json(progress_path, progress)

        val_metrics = evaluate(model, val_loader, device)
        val_metrics["train_loss"] = running_loss / max(1, len(train_loader))
        print(json.dumps({"epoch": epoch, "val": val_metrics}, ensure_ascii=False, indent=2))

        macro_f1 = float(val_metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                config,
                epoch,
                val_metrics,
                best_macro_f1,
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
            best_macro_f1,
            epochs_without_improvement,
        )
        epoch_progress = {
            "status": "running",
            "pid": os.getpid(),
            "device": str(device),
            "current_epoch": epoch,
            "total_epochs": epochs,
            "step": steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_macro_f1": best_macro_f1,
            "percent": round(epoch / max(1, epochs) * 100, 2),
            "checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(progress_path, epoch_progress)
        _append_jsonl(metrics_log_path, {"epoch": epoch, "val": val_metrics})

        if epochs_without_improvement >= patience:
            message = f"Early stopping at epoch {epoch}. Best macro_f1={best_macro_f1:.4f}"
            print(message)
            epoch_progress["status"] = "early_stopped"
            epoch_progress["message"] = message
            _write_json(progress_path, epoch_progress)
            break

    final_progress = {
        "status": "finished",
        "pid": os.getpid(),
        "device": str(device),
        "total_epochs": epochs,
        "best_macro_f1": best_macro_f1,
        "checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(progress_path, final_progress)

    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_classifier_dinov3_large.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    checkpoint = train(
        config,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        epochs_override=args.epochs,
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()



