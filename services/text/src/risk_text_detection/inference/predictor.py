"""
文件功能：文本风险推理器。加载训练好的 BERT 多任务模型检查点，
对输入文本执行推理，返回风险等级、风险概率和各标签预测结果。

核心类：
- RiskTextPredictor: 文本风险预测器，封装模型加载和推理逻辑
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from risk_text_detection.data.torch_dataset import RiskBatchCollator
from risk_text_detection.models.multitask import MacBertMultiTaskClassifier, MultiTaskModelConfig
from risk_text_detection.training.labeling import LabelVocab


DEFAULT_CHECKPOINT_DIR = "runs/ast_replay_continue_from_best_20260607_100730"

DISPLAY_LABELS = {
    "risk_binary": {"0": "safe", "1": "risk"},
    "toxic_binary": {"0": "non_toxic", "1": "toxic"},
}


@dataclass
class PredictorConfig:
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR
    device: str | None = None
    max_length: int | None = None
    risk_threshold: float = 0.7
    suspicious_threshold: float = 0.4
    phonetic_glyph_index_file: str = "data/raw/ast/hanzi.txt"
    phonetic_glyph_neighbor_top_k: int = 4


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
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


def _label_display(task: str, label: str) -> str:
    return DISPLAY_LABELS.get(task, {}).get(label, label)


def _risk_level(risk_prob: float, suspicious_threshold: float, risk_threshold: float) -> str:
    if risk_prob >= risk_threshold:
        return "risk"
    if risk_prob >= suspicious_threshold:
        return "suspicious"
    return "normal"


def _resolve_device(requested_device: str | None) -> torch.device:
    if requested_device:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RiskTextPredictor:
    def __init__(self, config: PredictorConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = PredictorConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a PredictorConfig or keyword arguments, not both.")

        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.label_vocab = LabelVocab.from_file(self.checkpoint_dir / "label_vocab.json")
        model_config_data = json.loads(
            (self.checkpoint_dir / "model_config.json").read_text(encoding="utf-8")
        )
        self.model_config = MultiTaskModelConfig(**model_config_data)
        self.max_length = int(config.max_length or 128)

        tokenizer_path = self.checkpoint_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path.exists() else self.model_config.model_name
        )
        self.collator = RiskBatchCollator(
            tokenizer=self.tokenizer,
            label_vocab=self.label_vocab,
            max_length=self.max_length,
            use_pinyin=self.model_config.use_pinyin,
            use_phonetic_glyph=self.model_config.use_phonetic_glyph,
            phonetic_glyph_index_file=config.phonetic_glyph_index_file,
            phonetic_glyph_neighbor_top_k=config.phonetic_glyph_neighbor_top_k,
        )

        self.device = _resolve_device(config.device)
        self.model = MacBertMultiTaskClassifier(self.model_config)
        state_dict = torch.load(self.checkpoint_dir / "pytorch_model.bin", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        text: str,
        risk_threshold: float | None = None,
        suspicious_threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.predict_batch(
            [text],
            risk_threshold=risk_threshold,
            suspicious_threshold=suspicious_threshold,
        )[0]

    @torch.no_grad()
    def predict_batch(
        self,
        texts: list[str],
        risk_threshold: float | None = None,
        suspicious_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if not texts:
            return []
        records = [self._record_for_text(text, idx) for idx, text in enumerate(texts)]
        batch = self.collator(records)
        batch = _move_batch_to_device(batch, self.device)
        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            pinyin_ids=batch.get("pinyin_ids"),
            phonetic_glyph=batch.get("phonetic_glyph"),
        )
        probabilities = {
            task: F.softmax(task_logits, dim=-1).detach().cpu()
            for task, task_logits in logits.items()
        }
        results = []
        for row_idx, text in enumerate(texts):
            results.append(
                self._format_prediction(
                    text,
                    row_idx,
                    probabilities,
                    risk_threshold=risk_threshold,
                    suspicious_threshold=suspicious_threshold,
                )
            )
        return results

    def _record_for_text(self, text: str, idx: int) -> dict[str, Any]:
        empty_labels = {task: None for task in self.label_vocab.tasks}
        return {
            "id": f"inference_{idx}",
            "text": str(text),
            "labels": self.label_vocab.encode_record({"labels": empty_labels}),
            "metadata": {},
        }

    def _format_prediction(
        self,
        text: str,
        row_idx: int,
        probabilities: dict[str, torch.Tensor],
        risk_threshold: float | None = None,
        suspicious_threshold: float | None = None,
    ) -> dict[str, Any]:
        task_outputs: dict[str, Any] = {}
        probability_outputs: dict[str, dict[str, float]] = {}
        raw_labels: dict[str, str] = {}

        for task, task_probs in probabilities.items():
            vocab = self.label_vocab.tasks[task]
            row = task_probs[row_idx]
            pred_idx = int(row.argmax().item())
            raw_label = vocab.id_to_label[pred_idx]
            raw_labels[task] = raw_label
            task_outputs[task] = _label_display(task, raw_label)
            probability_outputs[task] = {
                _label_display(task, label): float(row[label_idx].item())
                for label_idx, label in vocab.id_to_label.items()
            }

        risk_probs = probabilities.get("risk_binary")
        risk_prob = 0.0
        if risk_probs is not None and "1" in self.label_vocab.tasks["risk_binary"].label_to_id:
            risk_idx = self.label_vocab.tasks["risk_binary"].label_to_id["1"]
            risk_prob = float(risk_probs[row_idx][risk_idx].item())

        active_risk_threshold = (
            self.config.risk_threshold if risk_threshold is None else risk_threshold
        )
        active_suspicious_threshold = (
            self.config.suspicious_threshold
            if suspicious_threshold is None
            else suspicious_threshold
        )
        level = _risk_level(
            risk_prob,
            suspicious_threshold=active_suspicious_threshold,
            risk_threshold=active_risk_threshold,
        )
        system_risk = "risk" if level == "risk" else "safe"
        if level == "suspicious":
            system_risk = "suspicious"

        return {
            "text": text,
            "risk_binary": system_risk,
            "risk_level": level,
            "risk_probability": risk_prob,
            "primary_category": task_outputs.get("primary_category"),
            "toxic_binary": task_outputs.get("toxic_binary"),
            "camouflage_type": task_outputs.get("camouflage_type"),
            "raw_argmax": raw_labels,
            "probabilities": probability_outputs,
            "thresholds": {
                "suspicious": active_suspicious_threshold,
                "risk": active_risk_threshold,
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--risk-threshold", type=float, default=0.7)
    parser.add_argument("--suspicious-threshold", type=float, default=0.4)
    parser.add_argument("--phonetic-glyph-index-file", default="data/raw/ast/hanzi.txt")
    parser.add_argument("--phonetic-glyph-neighbor-top-k", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predictor = RiskTextPredictor(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        max_length=args.max_length,
        risk_threshold=args.risk_threshold,
        suspicious_threshold=args.suspicious_threshold,
        phonetic_glyph_index_file=args.phonetic_glyph_index_file,
        phonetic_glyph_neighbor_top_k=args.phonetic_glyph_neighbor_top_k,
    )
    print(
        json.dumps(
            predictor.predict_batch(
                args.text,
                risk_threshold=args.risk_threshold,
                suspicious_threshold=args.suspicious_threshold,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
