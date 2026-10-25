"""
文件功能：消息数据模型。定义消息记录、附件记录和消息发送响应的 Pydantic
数据模型，以及审核结果汇总构建函数。

核心类：
- MessageRecord: 消息记录模型，包含文本、状态、附件列表和审核结果
- AttachmentRecord: 附件记录模型，记录上传文件的元信息和存储路径
- SendMessageResponse: 消息发送接口的响应模型

核心函数：
- build_review_summary: 汇总各模态审核结果，生成统一的审核摘要
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.review import ModalityReviewResult, ReviewReason


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AttachmentRecord(BaseModel):
    id: str
    message_id: str
    modality: str
    original_name: str
    mime_type: str
    storage_path: str
    size_bytes: int


class MessageRecord(BaseModel):
    id: str
    client_message_id: str
    conversation_id: str
    sender_id: str = "me"
    text: Optional[str] = None
    status: str
    visible_to_others: bool
    attachments: List[AttachmentRecord] = Field(default_factory=list)
    review: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class SendMessageResponse(BaseModel):
    message_id: Optional[str]
    client_message_id: str
    status: str
    visible_to_others: bool
    message: Optional[MessageRecord] = None
    review: Dict[str, Any]


def build_review_summary(results: List[ModalityReviewResult]) -> Dict[str, Any]:
    allow = all(result.allow for result in results)
    reasons: List[ReviewReason] = []
    raw: Dict[str, Any] = {}
    for result in results:
        reasons.extend(result.reasons)
        raw[result.modality] = result.raw

    if allow:
        summary = "审核通过"
        status = "approved"
    elif reasons:
        summary = reasons[0].display
        status = "rejected"
    else:
        summary = "发送失败：内容未通过审核"
        status = "rejected"

    return {
        "allow": allow,
        "status": status,
        "summary": summary if allow else "发送失败：%s" % summary,
        "reasons": [reason.model_dump() for reason in reasons],
        "results": [result.model_dump() for result in results],
        "raw": raw,
    }
