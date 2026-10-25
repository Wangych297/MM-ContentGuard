"""
文件功能：模型训练脚本。实现多任务文本审核模型的训练流程，包括
训练循环、验证评估、损失计算、学习率调度和检查点保存。

核心函数：
- train_epoch: 单轮训练循环
- evaluate: 验证集评估
- train: 完整训练入口，协调数据加载、训练和保存
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from risk_text_detection.augment.char_variant import CharVariantAugmenter, CharVariantIndex
from risk_text_detection.data.torch_dataset import JsonlRiskDataset, RiskBatchCollator
from risk_text_detection.evaluation.metrics import (
    compute_metrics,
    empty_prediction_store,
    update_predictions,
)
from risk_text_detection.models.multitask import MacBertMultiTaskClassifier, MultiTaskModelConfig
from risk_text_detection.training.labeling import DEFAULT_TASKS, LabelVocab, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed/quick20k/train.jsonl")
    parser.add_argument("--dev-file", default="data/processed/quick20k/dev.jsonl")
    parser.add_argument("--output-dir", default="runs/quick20k_macbert")
    parser.add_argument("--model-name", default="models/hfl/chinese-macbert-base")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-pinyin", action="store_true")
    parser.add_argument("--pinyin-max-length", type=int, default=256)
    parser.add_argument("--use-phonetic-glyph", action="store_true")
    parser.add_argument("--phonetic-glyph-index-file", default="data/raw/ast/hanzi.txt")
    parser.add_argument("--phonetic-glyph-neighbor-top-k", type=int, default=4)
    parser.add_argument("--phonetic-glyph-hidden-dim", type=int, default=256)
    parser.add_argument("--phonetic-glyph-embed-dim", type=int, default=48)
    parser.add_argument(
        "--char-augment-mode",
        choices=["none", "homophone", "glyph", "mixed"],
        default="none",
    )
    parser.add_argument("--char-augment-prob", type=float, default=0.15)
    parser.add_argument("--char-augment-max-replacements", type=int, default=2)
    parser.add_argument("--char-index-file", default="data/raw/ast/hanzi.txt")
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device) if isinstance(sub_value, torch.Tensor) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device) -> dict:
    model.eval()
    store = empty_prediction_store()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            pinyin_ids=batch.get("pinyin_ids"),
            phonetic_glyph=batch.get("phonetic_glyph"),
        )
        loss, _ = model.compute_loss(logits, batch["labels"])
        total_loss += float(loss.item())
        steps += 1
        update_predictions(store, logits, batch["labels"])
    metrics = compute_metrics(store)
    metrics["loss"] = {"mean": total_loss / max(steps, 1)}
    return metrics


def save_checkpoint(output_dir: Path, model, tokenizer, label_vocab, config, metrics: dict, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    label_vocab.save(output_dir / "label_vocab.json")
    (output_dir / "model_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "train_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dev_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)

    train_path = Path(args.train_file)
    dev_path = Path(args.dev_file)
    train_records = read_jsonl(train_path, limit=args.limit_train)
    label_vocab = LabelVocab.build(train_records, args.tasks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = JsonlRiskDataset(train_path, label_vocab=label_vocab, limit=args.limit_train)
    dev_dataset = JsonlRiskDataset(dev_path, label_vocab=label_vocab, limit=args.limit_dev)
    augmenter = None
    if args.char_augment_mode != "none":
        char_index = CharVariantIndex.from_hanzi_file(args.char_index_file)
        augmenter = CharVariantAugmenter(
            index=char_index,
            mode=args.char_augment_mode,
            replace_prob=args.char_augment_prob,
            max_replacements=args.char_augment_max_replacements,
            seed=args.seed,
        )

    collator = RiskBatchCollator(
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        max_length=args.max_length,
        use_pinyin=args.use_pinyin,
        pinyin_max_length=args.pinyin_max_length,
        use_phonetic_glyph=args.use_phonetic_glyph,
        phonetic_glyph_index_file=args.phonetic_glyph_index_file,
        phonetic_glyph_neighbor_top_k=args.phonetic_glyph_neighbor_top_k,
        augmenter=augmenter,
    )
    eval_collator = RiskBatchCollator(
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        max_length=args.max_length,
        use_pinyin=args.use_pinyin,
        pinyin_max_length=args.pinyin_max_length,
        use_phonetic_glyph=args.use_phonetic_glyph,
        phonetic_glyph_index_file=args.phonetic_glyph_index_file,
        phonetic_glyph_neighbor_top_k=args.phonetic_glyph_neighbor_top_k,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=args.num_workers,
    )

    config = MultiTaskModelConfig(
        model_name=args.model_name,
        task_num_labels=label_vocab.num_labels,
        task_weights={task: 1.0 for task in label_vocab.tasks},
        use_safetensors=args.use_safetensors,
        use_focal_loss=args.use_focal_loss,
        label_smoothing=args.label_smoothing,
        use_pinyin=args.use_pinyin,
        use_phonetic_glyph=args.use_phonetic_glyph,
        phonetic_glyph_hidden_dim=args.phonetic_glyph_hidden_dim,
        phonetic_glyph_embed_dim=args.phonetic_glyph_embed_dim,
    )
    model = MacBertMultiTaskClassifier(config)
    if args.init_checkpoint:
        checkpoint_dir = Path(args.init_checkpoint)
        checkpoint_file = checkpoint_dir / "pytorch_model.bin"
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"loaded_init_checkpoint={checkpoint_file}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro = -1.0
    epochs_without_improvement = 0
    global_step = 0
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "train_history.jsonl"
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    pinyin_ids=batch.get("pinyin_ids"),
                    phonetic_glyph=batch.get("phonetic_glyph"),
                )
                loss, _ = model.compute_loss(logits, batch["labels"])
                loss = loss / max(args.gradient_accumulation_steps, 1)

            scaler.scale(loss).backward()
            is_update_step = step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.max_steps and global_step >= args.max_steps:
                    break

        metrics = evaluate(model, dev_loader, device)
        task_scores = [
            values["macro_f1"]
            for task, values in metrics.items()
            if isinstance(values, dict) and "macro_f1" in values
        ]
        mean_macro = sum(task_scores) / max(len(task_scores), 1)
        epoch_summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "mean_macro_f1": mean_macro,
            "dev_loss": metrics.get("loss", {}).get("mean"),
            "elapsed_sec": round(time.time() - start, 2),
        }
        print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"summary": epoch_summary, "metrics": metrics}, ensure_ascii=False) + "\n")
        if mean_macro > best_macro:
            best_macro = mean_macro
            epochs_without_improvement = 0
            save_checkpoint(output_dir, model, tokenizer, label_vocab, config, metrics, args)
        else:
            epochs_without_improvement += 1

        if args.max_steps and global_step >= args.max_steps:
            break
        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            break

    print(f"best_mean_macro_f1={best_macro:.4f}", flush=True)
    print(f"saved_to={output_dir}", flush=True)


if __name__ == "__main__":
    main()
