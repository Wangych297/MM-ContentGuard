"""
文件功能：消息审核 API 路由。提供消息发送（含文本和附件上传）和消息列表
查询接口。接收用户提交的内容，协调文本/图片/视频三类适配器完成审核，
将审核结果和消息记录持久化存储。

核心接口：
- POST /api/conversations/{id}/messages: 提交消息审核请求
- GET /api/conversations/{id}/messages: 查询会话消息列表

核心函数：
- _review_message: 并发调用各模态适配器执行审核
- send_message: 消息发送端点的处理函数
- list_messages: 消息列表查询端点的处理函数
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.adapters.image_adapter import ImageReviewAdapter
from app.adapters.text_adapter import TextReviewAdapter
from app.adapters.video_adapter import VideoReviewAdapter
from app.core.config import get_settings
from app.core.labels import display_label
from app.models.message import (
    AttachmentRecord,
    MessageRecord,
    SendMessageResponse,
    build_review_summary,
)
from app.models.review import ModalityReviewResult, ReviewReason
from app.storage.files import SavedUpload, move_upload, save_upload_to_temp
from app.storage.message_store import JsonMessageStore


router = APIRouter(prefix="/api", tags=["messages"])


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=SendMessageResponse,
)
async def send_message(
    conversation_id: str,
    text: Optional[str] = Form(default=None),
    client_message_id: Optional[str] = Form(default=None),
    sender_id: str = Form(default="me"),
    files: Optional[List[UploadFile]] = File(default=None),
) -> SendMessageResponse:
    settings = get_settings()
    normalized_text = (text or "").strip()
    uploads = files or []
    if not normalized_text and not uploads:
        raise HTTPException(status_code=400, detail="Message text or attachment is required.")

    message_id = "msg_%s" % uuid.uuid4().hex
    local_client_id = client_message_id or "local_%s" % uuid.uuid4().hex
    tmp_dir = settings.storage_root / "uploads" / "tmp" / local_client_id
    saved_uploads: List[SavedUpload] = []

    try:
        for upload in uploads:
            saved = save_upload_to_temp(upload, tmp_dir)
            _validate_upload_size(saved, settings.max_image_mb, settings.max_video_mb)
            saved_uploads.append(saved)

        review_results = await _review_message(normalized_text, saved_uploads)
        review = build_review_summary(review_results)
        allow = bool(review["allow"])
        status = "approved" if allow else "rejected"
        visible_to_others = allow

        attachment_records = _finalize_uploads(
            saved_uploads=saved_uploads,
            settings_storage_root=settings.storage_root,
            conversation_id=conversation_id,
            message_id=message_id,
            approved=allow,
        )

        message = MessageRecord(
            id=message_id,
            client_message_id=local_client_id,
            conversation_id=conversation_id,
            sender_id=sender_id,
            text=normalized_text or None,
            status=status,
            visible_to_others=visible_to_others,
            attachments=attachment_records,
            review=review,
        )
        _message_store().add(message)

        return SendMessageResponse(
            message_id=message_id if allow else None,
            client_message_id=local_client_id,
            status=status,
            visible_to_others=visible_to_others,
            message=message,
            review=review,
        )
    except HTTPException:
        _cleanup_uploads(saved_uploads)
        raise
    except Exception as exc:
        _cleanup_uploads(saved_uploads)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[MessageRecord],
)
def list_messages(
    conversation_id: str,
    include_rejected: bool = Query(default=False),
) -> List[MessageRecord]:
    return _message_store().list_messages(
        conversation_id=conversation_id,
        include_rejected=include_rejected,
    )


async def _review_message(
    text: str,
    uploads: List[SavedUpload],
) -> List[ModalityReviewResult]:
    settings = get_settings()
    text_adapter = TextReviewAdapter(settings)
    image_adapter = ImageReviewAdapter(settings)
    video_adapter = VideoReviewAdapter(settings)

    results: List[ModalityReviewResult] = []
    async with httpx.AsyncClient(trust_env=False) as client:
        if text:
            try:
                results.append(await text_adapter.review_text(client, text))
            except Exception as exc:
                results.append(_service_error_result("text", exc))

        for upload in uploads:
            try:
                if upload.modality == "image":
                    results.append(
                        await image_adapter.review_image_path(
                            client,
                            str(upload.temp_path.resolve()),
                        )
                    )
                elif upload.modality == "video":
                    results.append(
                        await video_adapter.review_video_file(
                            client,
                            str(upload.temp_path.resolve()),
                        )
                    )
            except Exception as exc:
                results.append(_service_error_result(upload.modality, exc))

    return results


def _service_error_result(modality: str, exc: Exception) -> ModalityReviewResult:
    code = "%s_service_error" % modality
    display = "%s审核服务暂不可用" % _modality_display_name(modality)
    raw_reason = str(exc)
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        code = "%s_service_busy" % modality
        detail = _response_detail(exc.response)
        display = "%s审核服务正忙，请稍后重试" % _modality_display_name(modality)
        raw_reason = detail or display
    elif isinstance(exc, httpx.TimeoutException):
        timeout_seconds = {
            "text": get_settings().text_api_timeout,
            "image": get_settings().image_api_timeout,
            "video": get_settings().video_api_timeout,
        }.get(modality)
        raw_reason = (
            "%s review request timed out after %.0fs"
            % (modality, timeout_seconds)
            if timeout_seconds is not None
            else "%s review request timed out" % modality
        )
    elif not raw_reason:
        raw_reason = exc.__class__.__name__
    return ModalityReviewResult(
        modality=modality,
        allow=False,
        risk_level="error",
        category=code,
        summary="发送失败：%s" % display,
        reasons=[
            ReviewReason(
                modality=modality,
                code=code,
                display=display,
                source="%s_service" % modality,
                raw_reason=raw_reason,
            )
        ],
        raw={"error": raw_reason},
    )


def _response_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()
    detail = data.get("detail") if isinstance(data, dict) else data
    return str(detail).strip() if detail else ""


def _modality_display_name(modality: str) -> str:
    return {
        "text": "文本",
        "image": "图片",
        "video": "视频",
    }.get(modality, modality)


def _validate_upload_size(upload: SavedUpload, max_image_mb: int, max_video_mb: int) -> None:
    max_mb = max_image_mb if upload.modality == "image" else max_video_mb
    if upload.size_bytes > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="%s file is too large. Max size is %s MB."
            % (upload.modality, max_mb),
        )


def _finalize_uploads(
    saved_uploads: List[SavedUpload],
    settings_storage_root: Path,
    conversation_id: str,
    message_id: str,
    approved: bool,
) -> List[AttachmentRecord]:
    target_status = "approved" if approved else "rejected"
    target_dir = settings_storage_root / "uploads" / target_status / conversation_id / message_id
    records = []
    for upload in saved_uploads:
        attachment_id = "att_%s" % uuid.uuid4().hex
        target_path = move_upload(upload, target_dir, attachment_id)
        records.append(
            AttachmentRecord(
                id=attachment_id,
                message_id=message_id,
                modality=upload.modality,
                original_name=upload.original_name,
                mime_type=upload.mime_type,
                storage_path=str(target_path),
                size_bytes=upload.size_bytes,
            )
        )
    return records


def _cleanup_uploads(saved_uploads: List[SavedUpload]) -> None:
    for upload in saved_uploads:
        upload.temp_path.unlink(missing_ok=True)
        parent = upload.temp_path.parent
        try:
            parent.rmdir()
        except OSError:
            pass


def _message_store() -> JsonMessageStore:
    settings = get_settings()
    return JsonMessageStore(settings.storage_root / "db" / "messages.json")
