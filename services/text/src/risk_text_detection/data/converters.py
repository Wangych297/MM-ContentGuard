"""
文件功能：数据格式转换器。将不同来源的原始标注格式（AST JSON、
垃圾短信 TSV）统一转换为 RiskRecord 结构化记录。

核心函数：
- load_ast: 加载 AST 标注数据并转换为 RiskRecord
- load_spam_message: 加载垃圾短信数据并转换为 RiskRecord
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .schema import RiskLabels, RiskRecord, clean_text, text_hash


FBS_CATEGORY_MAP = {
    "AD_Loan": ("loan_ad", ["spam", "ad", "loan"]),
    "AD_Network_service": ("ad_promotion", ["spam", "ad", "network_service"]),
    "AD_Other": ("ad_promotion", ["spam", "ad", "other_ad"]),
    "AD_Real_estate": ("ad_promotion", ["spam", "ad", "real_estate"]),
    "AD_Retail": ("ad_promotion", ["spam", "ad", "retail"]),
    "FR_Financial": ("fraud_phishing", ["spam", "fraud", "financial"]),
    "FR_Other": ("fraud_phishing", ["spam", "fraud", "other_fraud"]),
    "FR_Phishing(Bank)": ("fraud_phishing", ["spam", "fraud", "phishing", "bank"]),
    "FR_Phishing(Other)": ("fraud_phishing", ["spam", "fraud", "phishing"]),
    "IL_Escort_service": ("porn_sexual", ["spam", "sexual_service"]),
    "IL_Fake_ID_and_invoice": ("illegal_service", ["spam", "fake_id", "invoice"]),
    "IL_Gambling": ("gambling", ["spam", "gambling"]),
    "IL_Political_propaganda": ("political_propaganda", ["spam", "political_propaganda"]),
    "Other": ("other_risk", ["spam", "other_risk"]),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _record_id(source: str, text: str, idx: int) -> str:
    return f"{source}_{idx:07d}_{text_hash(text)}"


def deduplicate(records: Iterable[RiskRecord]) -> list[RiskRecord]:
    seen: set[str] = set()
    result: list[RiskRecord] = []
    for record in records:
        key = text_hash(record.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def load_spam_message(path: Path, max_chars: int = 512) -> list[RiskRecord]:
    records: list[RiskRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or parts[0] not in {"0", "1"}:
                continue
            label = int(parts[0])
            text = clean_text(parts[1], max_chars=max_chars)
            if not text:
                continue
            category = "spam_general" if label else "safe"
            tags = ["spam"] if label else []
            records.append(
                RiskRecord(
                    id=_record_id("spam_message", text, idx),
                    text=text,
                    labels=RiskLabels(label, category, tags),
                    metadata={
                        "source": "spam_message",
                        "original_label": parts[0],
                        "quality": "gold",
                    },
                )
            )
    return deduplicate(records)


def load_ast(path: Path, max_chars: int = 512) -> list[RiskRecord]:
    records: list[RiskRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] not in {"0", "1"}:
                continue
            label = int(parts[0])
            text = clean_text(parts[1], max_chars=max_chars)
            paired_text = clean_text(parts[2], max_chars=max_chars) if len(parts) > 2 else None
            if not text:
                continue
            category = "spam_general" if label else "safe"
            tags = ["spam", "camouflage"] if label else []
            records.append(
                RiskRecord(
                    id=_record_id("ast", text, idx),
                    text=text,
                    labels=RiskLabels(
                        risk_binary=label,
                        primary_category=category,
                        risk_tags=tags,
                        camouflage_type="unknown" if paired_text and paired_text != text else "none",
                    ),
                    metadata={
                        "source": "ast",
                        "original_label": parts[0],
                        "paired_text": paired_text,
                        "quality": "gold",
                    },
                )
            )
    return deduplicate(records)


def load_fbs(raw_dir: Path, max_chars: int = 512) -> list[RiskRecord]:
    records: list[RiskRecord] = []
    for file_path in sorted(raw_dir.iterdir()):
        if not file_path.is_file() or file_path.name.startswith(".") or file_path.name == "README.md":
            continue
        category, tags = FBS_CATEGORY_MAP.get(file_path.name, ("other_risk", ["spam", "other_risk"]))
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                text = clean_text(line, max_chars=max_chars)
                if not text:
                    continue
                idx = len(records)
                records.append(
                    RiskRecord(
                        id=_record_id("fbs", text, idx),
                        text=text,
                        labels=RiskLabels(1, category, tags),
                        metadata={
                            "source": "fbs",
                            "original_label": file_path.name,
                            "line_index": line_idx,
                            "quality": "gold",
                        },
                    )
                )
    return deduplicate(records)


def load_cold(raw_dir: Path, max_chars: int = 512) -> list[RiskRecord]:
    records: list[RiskRecord] = []
    for split_file in ("train_processed.csv", "dev_processed.csv", "test_processed.csv"):
        path = raw_dir / split_file
        if not path.exists():
            continue
        split = split_file.replace("_processed.csv", "")
        for row in _read_csv(path):
            text = clean_text(row.get("TEXT", ""), max_chars=max_chars)
            if not text:
                continue
            label = int(row.get("label", "0") or 0)
            topic = str(row.get("topic", "") or "").strip()
            if label:
                category = "hate_bias" if topic else "toxicity_abuse"
                tags = ["toxicity", "offensive"] + ([f"bias_{topic}"] if topic else [])
            else:
                category = "safe"
                tags = []
            idx = len(records)
            records.append(
                RiskRecord(
                    id=_record_id("cold", text, idx),
                    text=text,
                    labels=RiskLabels(
                        risk_binary=label,
                        primary_category=category,
                        risk_tags=tags,
                        toxic_binary=label,
                        toxic_type=topic or None,
                    ),
                    metadata={
                        "source": "cold",
                        "source_split": split,
                        "original_label": row.get("label"),
                        "topic": topic,
                        "quality": "gold",
                    },
                )
            )
    return deduplicate(records)


def load_toxicn(path: Path, max_chars: int = 512) -> list[RiskRecord]:
    records: list[RiskRecord] = []
    for row in _read_csv(path):
        text = clean_text(row.get("content", ""), max_chars=max_chars)
        if not text:
            continue
        label = int(row.get("toxic", "0") or 0)
        topic = str(row.get("topic", "") or "").strip()
        if label:
            category = "hate_bias" if topic and topic != "general" else "toxicity_abuse"
            tags = ["toxicity"] + ([f"bias_{topic}"] if topic else [])
        else:
            category = "safe"
            tags = []
        idx = len(records)
        records.append(
            RiskRecord(
                id=_record_id("toxicn", text, idx),
                text=text,
                labels=RiskLabels(
                    risk_binary=label,
                    primary_category=category,
                    risk_tags=tags,
                    toxic_binary=label,
                    toxic_type=str(row.get("toxic_type", "") or "") or None,
                ),
                metadata={
                    "source": "toxicn",
                    "platform": row.get("platform"),
                    "topic": topic,
                    "expression": row.get("expression"),
                    "target": row.get("target"),
                    "quality": "gold",
                },
            )
        )
    return deduplicate(records)


def _offensive_category(path: Path) -> tuple[str, list[str]]:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "sexharmset" in parts:
        return "sex_harm", ["sex_harm"]
    if "biasset" in parts or name.startswith("bias"):
        if "gender" in name or "genden" in name:
            return "hate_bias", ["bias_gender"]
        if "race" in name:
            return "hate_bias", ["bias_race"]
        if "region" in name:
            return "hate_bias", ["bias_region"]
        if "occupation" in name:
            return "hate_bias", ["bias_occupation"]
        return "hate_bias", ["bias"]
    return "toxicity_abuse", ["abuse"]


def load_chinese_offensive(raw_dir: Path, max_chars: int = 512) -> list[RiskRecord]:
    records: list[RiskRecord] = []
    for path in sorted(raw_dir.rglob("*.csv")):
        risk_category, risk_tags = _offensive_category(path)
        for row in _read_csv(path):
            text = clean_text(row.get("Sentence", ""), max_chars=max_chars)
            if not text:
                continue
            raw_type = str(row.get("Type", "") or "").strip()
            is_risk = 1 if raw_type.lower() in {"harmful", "1", "toxic"} else 0
            category = risk_category if is_risk else "safe"
            tags = list(risk_tags) if is_risk else []
            idx = len(records)
            records.append(
                RiskRecord(
                    id=_record_id("chinese_offensive", text, idx),
                    text=text,
                    labels=RiskLabels(
                        risk_binary=is_risk,
                        primary_category=category,
                        risk_tags=tags,
                        toxic_binary=is_risk,
                        toxic_type=raw_type or None,
                    ),
                    metadata={
                        "source": "chinese_offensive",
                        "source_file": str(path.relative_to(raw_dir)),
                        "keyword": row.get("Keyword"),
                        "original_label": raw_type,
                        "quality": "gold",
                    },
                )
            )
    return deduplicate(records)


def write_jsonl(path: Path, records: Iterable[RiskRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count

