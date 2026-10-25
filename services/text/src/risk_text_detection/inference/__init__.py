"""推理子包。对外暴露 RiskTextPredictor 作为文本风险检测的统一推理入口。"""

from risk_text_detection.inference.predictor import RiskTextPredictor

__all__ = ["RiskTextPredictor"]
