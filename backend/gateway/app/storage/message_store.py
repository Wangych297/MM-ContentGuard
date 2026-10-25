"""
文件功能：消息持久化存储。基于 JSON 文件的消息记录存储器，支持线程安全
的消息写入和按会话 ID 查询消息列表。

核心类：
- JsonMessageStore: JSON 文件消息存储，提供 add() 和 list_messages() 方法
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

from app.models.message import MessageRecord


class JsonMessageStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.write_text("[]\n", encoding="utf-8")

    def add(self, message: MessageRecord) -> None:
        with self.lock:
            records = self._read_unlocked()
            records.append(message)
            self._write_unlocked(records)

    def list_messages(
        self,
        conversation_id: str,
        include_rejected: bool = False,
    ) -> List[MessageRecord]:
        with self.lock:
            records = self._read_unlocked()
        filtered = [
            record
            for record in records
            if record.conversation_id == conversation_id
            and (include_rejected or record.status == "approved")
        ]
        return sorted(filtered, key=lambda record: record.created_at)

    def _read_unlocked(self) -> List[MessageRecord]:
        try:
            raw_records = json.loads(self.db_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_records = []
        return [MessageRecord.model_validate(raw) for raw in raw_records]

    def _write_unlocked(self, records: List[MessageRecord]) -> None:
        payload = [record.model_dump() for record in records]
        self.db_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
