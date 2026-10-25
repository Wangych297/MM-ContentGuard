"""
文件功能：图片审核决策。综合视觉模型的预测分数和规则检测器
的命中结果，做出最终的图片风险判定。

核心函数：
- decide_image_risk: 融合模型分数与规则命中，返回最终审核结论
"""

from __future__ import annotations

from typing import Any

from model.labels import is_risk_label


SPAM_TEXT_CATEGORIES = {
    "spam_general",
    "ad_promotion",
    "loan_ad",
    "fraud_phishing",
    "review_spam",
}


def decide_image_risk(
    image_model: dict[str, Any] | None,
    text_result: dict[str, Any] | None,
    rule_hits: list[dict[str, Any]],
    semantic_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    risk_category = "safe"
    route = "send_to_semantic_model"
    final_label = "safe_for_semantic_check"
    risk_level = "normal"

    image_model = image_model or {}
    text_result = text_result or {}
    model_label = image_model.get("model_label", "unknown")
    model_level = image_model.get("risk_level", "normal")

    if is_risk_label(str(model_label)) and model_level == "risk":
        final_label = "violation"
        route = "blocked_by_image_pipeline"
        risk_level = "risk"
        risk_category = str(model_label)
        reasons.append(f"image_model:{model_label}")

    text_level = str(text_result.get("risk_level", "normal"))
    primary_category = text_result.get("primary_category")
    if text_level == "risk":
        final_label = "spam_ad" if primary_category in SPAM_TEXT_CATEGORIES else "violation"
        route = "blocked_by_image_pipeline"
        risk_level = "risk"
        risk_category = str(primary_category or "text_risk")
        reasons.append(f"text:{primary_category or 'risk'}")
    elif text_level == "suspicious" and risk_level != "risk":
        final_label = "review"
        route = "manual_review"
        risk_level = "suspicious"
        risk_category = str(primary_category or "text_suspicious")
        reasons.append(f"text:{primary_category or 'suspicious'}")

    sensitive_hits = [hit for hit in rule_hits if hit.get("name") == "sensitive_word"]
    contact_hits = [hit for hit in rule_hits if hit.get("name") in {"phone", "email", "qq", "wechat", "url", "qrcode"}]
    if sensitive_hits:
        final_label = "violation"
        route = "blocked_by_image_pipeline"
        risk_level = "risk"
        risk_category = "rule_sensitive"
        reasons.extend(f"rule:{hit.get('name')}={hit.get('value')}" for hit in sensitive_hits[:5])
    elif len(contact_hits) >= 2 and risk_level != "risk":
        final_label = "spam_ad"
        route = "blocked_by_image_pipeline"
        risk_level = "risk"
        risk_category = "rule_contact_ad"
        reasons.extend(f"rule:{hit.get('name')}" for hit in contact_hits[:5])
    elif contact_hits and risk_level == "normal":
        final_label = "review"
        route = "manual_review"
        risk_level = "suspicious"
        risk_category = "rule_contact"
        reasons.extend(f"rule:{hit.get('name')}" for hit in contact_hits[:5])

    semantic_result = semantic_result or {}
    semantic_level = str(semantic_result.get("risk_level", "normal"))
    if route == "send_to_semantic_model" and semantic_level == "risk":
        semantic_type = str(semantic_result.get("type_label") or "semantic_risk")
        final_label = "semantic_violation"
        route = "blocked_by_semantic_model"
        risk_level = "risk"
        risk_category = semantic_type
        reasons.append(f"semantic_model:{semantic_type}")
    elif route == "send_to_semantic_model" and semantic_level == "suspicious":
        semantic_type = str(semantic_result.get("type_label") or "semantic_suspicious")
        final_label = "review"
        route = "manual_review"
        risk_level = "suspicious"
        risk_category = semantic_type
        reasons.append(f"semantic_model:{semantic_type}")

    return {
        "final_label": final_label,
        "risk_level": risk_level,
        "risk_category": risk_category,
        "route": route,
        "reasons": reasons,
    }


