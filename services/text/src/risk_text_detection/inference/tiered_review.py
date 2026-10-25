"""
文件功能：分层审核决策。基于模型预测概率应用分层判定规则：
根据 risk_threshold 和 suspicious_threshold 将文本分为 normal（正常）、
suspicious（可疑）和 risk（风险）三个等级，返回结构化的审核结论。

核心函数：
- tiered_review_result: 对预测结果应用分层阈值判定
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_text_detection.inference.predictor import RiskTextPredictor


DEFAULT_GENERAL_CHECKPOINT_DIR = "runs/social_binary_v3_macbert_binary_bs128_no_online_aug"
DEFAULT_SOCIAL_CHECKPOINT_DIR = "runs/social_intent_binary_v1_from_general"
DEFAULT_REASON_CHECKPOINT_DIR = "runs/reason_category_v1_macbert"

REASON_LABELS_ZH = {
    "financial_illegal_promotion": "诈骗/违法服务/广告引流",
    "gambling": "赌博相关",
    "sexual_content": "色情/性相关风险",
    "abuse_hate": "辱骂/骚扰/仇恨歧视",
}


@dataclass(frozen=True)
class TieredReviewConfig:
    general_checkpoint_dir: str | Path = DEFAULT_GENERAL_CHECKPOINT_DIR
    social_checkpoint_dir: str | Path = DEFAULT_SOCIAL_CHECKPOINT_DIR
    reason_checkpoint_dir: str | Path = DEFAULT_REASON_CHECKPOINT_DIR
    device: str | None = None
    max_length: int = 128
    general_threshold: float = 0.5
    social_review_threshold: float = 0.5
    enable_reason: bool = True
    parallel_binary: bool = True
    parallel_reason: bool = True


def risk_score(prediction: dict[str, Any]) -> float:
    return float(prediction.get("risk_probability", 0.0))


def tiered_decision(
    general_score: float,
    social_score: float,
    general_threshold: float,
    social_review_threshold: float,
) -> tuple[str, str, str]:
    if general_score >= general_threshold:
        return "hard_risk", "risk", "通用模型达到硬风险阈值，建议判为风险。"
    if social_score >= social_review_threshold:
        return "review", "suspicious", "通用模型未触发硬风险，但口语模型触发复核阈值。"
    return "safe", "normal", "两个模型都未触发对应阈值，建议暂判安全。"


class TieredTextReviewPredictor:
    def __init__(self, config: TieredReviewConfig) -> None:
        self.config = config
        self.general_predictor = RiskTextPredictor(
            checkpoint_dir=config.general_checkpoint_dir,
            device=config.device,
            max_length=config.max_length,
            risk_threshold=config.general_threshold,
            suspicious_threshold=config.social_review_threshold,
        )
        self.social_predictor = RiskTextPredictor(
            checkpoint_dir=config.social_checkpoint_dir,
            device=config.device,
            max_length=config.max_length,
            risk_threshold=config.general_threshold,
            suspicious_threshold=config.social_review_threshold,
        )
        self.reason_predictor = None
        if config.enable_reason:
            self.reason_predictor = RiskTextPredictor(
                checkpoint_dir=config.reason_checkpoint_dir,
                device=config.device,
                max_length=config.max_length,
            )
        worker_count = 0
        if config.parallel_binary:
            worker_count = max(worker_count, 2)
        if config.parallel_reason and self.reason_predictor is not None:
            worker_count = max(worker_count, 3 if config.parallel_binary else 2)
        self._model_executor = (
            ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="risk-text-model")
            if worker_count > 0
            else None
        )

    @property
    def device(self) -> Any:
        return self.general_predictor.device

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

    def predict_batch(
        self,
        texts: list[str],
        risk_threshold: float | None = None,
        suspicious_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if not texts:
            return []

        general_threshold = (
            self.config.general_threshold if risk_threshold is None else risk_threshold
        )
        social_review_threshold = (
            self.config.social_review_threshold
            if suspicious_threshold is None
            else suspicious_threshold
        )

        general_predictions, social_predictions, reason_future = self._predict_front_models(texts)

        decisions: list[tuple[str, str, str]] = []
        reason_indices: list[int] = []
        reason_texts: list[str] = []
        for idx, (text, general_prediction, social_prediction) in enumerate(
            zip(texts, general_predictions, social_predictions)
        ):
            decision = tiered_decision(
                risk_score(general_prediction),
                risk_score(social_prediction),
                general_threshold,
                social_review_threshold,
            )
            decisions.append(decision)
            if decision[0] in {"hard_risk", "review"} and self.reason_predictor is not None:
                reason_indices.append(idx)
                reason_texts.append(text)

        reason_by_index: dict[int, dict[str, Any]] = {}
        if reason_indices and reason_future is not None:
            reason_predictions = reason_future.result()
            reason_by_index = {idx: reason_predictions[idx] for idx in reason_indices}
        elif reason_texts and self.reason_predictor is not None:
            reason_predictions = self.reason_predictor.predict_batch(reason_texts)
            reason_by_index = dict(zip(reason_indices, reason_predictions))
        elif reason_future is not None:
            try:
                reason_future.result()
            except Exception:
                # Safe texts do not expose reason output; keep that path independent.
                pass

        results = []
        for idx, text in enumerate(texts):
            decision, risk_level, decision_reason = decisions[idx]
            results.append(
                self._format_prediction(
                    text=text,
                    decision=decision,
                    risk_level=risk_level,
                    decision_reason=decision_reason,
                    general_prediction=general_predictions[idx],
                    social_prediction=social_predictions[idx],
                    reason_prediction=reason_by_index.get(idx),
                    general_threshold=general_threshold,
                    social_review_threshold=social_review_threshold,
                )
            )
        return results

    def _predict_front_models(
        self,
        texts: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Future | None]:
        reason_future = None
        if (
            self._model_executor is not None
            and self.config.parallel_reason
            and self.reason_predictor is not None
        ):
            reason_future = self._model_executor.submit(
                self.reason_predictor.predict_batch,
                texts,
            )

        if self._model_executor is None or not self.config.parallel_binary:
            return (
                self.general_predictor.predict_batch(texts),
                self.social_predictor.predict_batch(texts),
                reason_future,
            )

        general_future = self._model_executor.submit(
            self.general_predictor.predict_batch,
            texts,
        )
        social_future = self._model_executor.submit(
            self.social_predictor.predict_batch,
            texts,
        )
        return general_future.result(), social_future.result(), reason_future

    def _format_prediction(
        self,
        text: str,
        decision: str,
        risk_level: str,
        decision_reason: str,
        general_prediction: dict[str, Any],
        social_prediction: dict[str, Any],
        reason_prediction: dict[str, Any] | None,
        general_threshold: float,
        social_review_threshold: float,
    ) -> dict[str, Any]:
        general_score = risk_score(general_prediction)
        social_score = risk_score(social_prediction)
        trigger_score = max(general_score, social_score)
        if decision == "hard_risk":
            trigger_score = general_score
        elif decision == "review":
            trigger_score = social_score

        reason_info = None
        primary_category = "safe"
        if decision in {"hard_risk", "review"}:
            primary_category = "other_risk"
            if reason_prediction is not None:
                raw_reason = str(
                    reason_prediction.get("raw_argmax", {}).get("reason_category", "")
                )
                probs = reason_prediction.get("probabilities", {}).get("reason_category", {})
                confidence = float(probs.get(raw_reason, 0.0))
                if raw_reason:
                    primary_category = raw_reason
                reason_info = {
                    "category": raw_reason,
                    "label": REASON_LABELS_ZH.get(raw_reason, raw_reason),
                    "confidence": confidence,
                }

        risk_binary = "safe"
        if decision == "hard_risk":
            risk_binary = "risk"
        elif decision == "review":
            risk_binary = "suspicious"

        return {
            "text": text,
            "risk_binary": risk_binary,
            "risk_level": risk_level,
            "risk_probability": trigger_score,
            "primary_category": primary_category,
            "toxic_binary": "non_toxic" if decision == "safe" else None,
            "camouflage_type": "none" if decision == "safe" else "unknown",
            "decision": decision,
            "decision_reason": decision_reason,
            "reason": reason_info,
            "scores": {
                "general": general_score,
                "social": social_score,
            },
            "thresholds": {
                "general_hard_risk": general_threshold,
                "social_review": social_review_threshold,
                "risk": general_threshold,
                "suspicious": social_review_threshold,
            },
            "model_outputs": {
                "general_raw_argmax": general_prediction.get("raw_argmax", {}),
                "social_raw_argmax": social_prediction.get("raw_argmax", {}),
                "reason_raw_argmax": (
                    reason_prediction.get("raw_argmax", {}) if reason_prediction else {}
                ),
            },
            "probabilities": {
                "general": general_prediction.get("probabilities", {}),
                "social": social_prediction.get("probabilities", {}),
                "reason": reason_prediction.get("probabilities", {}) if reason_prediction else {},
            },
        }
