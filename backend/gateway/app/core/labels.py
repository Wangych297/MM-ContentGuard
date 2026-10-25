"""
文件功能：风险标签映射表。定义文本、图片、视频三种模态的风险标签码到
中文显示名称的映射关系。

核心函数：
- display_label: 根据模态和标签码返回对应的中文显示名称
"""

from __future__ import annotations


TEXT_LABELS = {
    "financial_illegal_promotion": "诈骗/违法服务/广告引流",
    "gambling": "赌博相关",
    "sexual_content": "色情/性相关风险",
    "abuse_hate": "辱骂/骚扰/仇恨歧视",
    "spam_general": "广告 / 垃圾信息",
    "ad_promotion": "推广引流",
    "loan_ad": "贷款广告",
    "fraud_phishing": "诈骗 / 钓鱼",
    "review_spam": "刷评 / 垃圾推广",
    "porn_sexual": "色情内容",
    "toxic": "攻击性 / 侮辱性内容",
    "other_risk": "风险文本",
    "unknown": "风险文本",
}

IMAGE_LABELS = {
    "sexual": "色情 / 性暗示图片",
    "violence": "暴力内容",
    "weapon": "武器相关内容",
    "gambling": "赌博相关图片",
    "drug": "毒品相关内容",
    "regulated_goods": "受管制商品",
    "rule_sensitive": "命中敏感词规则",
    "rule_contact_ad": "疑似联系方式引流",
    "rule_contact": "疑似联系方式",
    "targeted_offense": "针对性冒犯",
    "general_offense": "一般冒犯",
    "dispirited": "消极有害表达",
    "text_risk": "图片文字存在风险",
}

VIDEO_LABELS = {
    "video_abnormal": "视频内容未通过多模态审核",
    "pcl_risk": "视频可能存在贬抑 / 冒犯性表达",
}


def display_label(modality: str, code: str) -> str:
    if modality == "text":
        return TEXT_LABELS.get(code, TEXT_LABELS["unknown"])
    if modality == "image":
        return IMAGE_LABELS.get(code, "图片内容存在风险")
    if modality == "video":
        return VIDEO_LABELS.get(code, VIDEO_LABELS["video_abnormal"])
    return code
