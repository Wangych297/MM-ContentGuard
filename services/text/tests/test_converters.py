"""
文件功能：数据转换器单元测试。验证 load_ast 和 load_spam_message
能否将原始标注格式正确转换为 RiskRecord。
"""

from pathlib import Path

from risk_text_detection.data.converters import load_ast, load_spam_message


def _case_dir(name: str) -> Path:
    path = Path("tmp") / "pytest_cases" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_spam_message_converts_binary_labels() -> None:
    raw = _case_dir("converters") / "spam.txt"
    raw.write_text("1\t加我微信赚钱\n0\t今晚吃饭了吗\n", encoding="utf-8")

    records = load_spam_message(raw)

    assert len(records) == 2
    assert records[0].labels.risk_binary == 1
    assert records[0].labels.primary_category == "spam_general"
    assert records[1].labels.risk_binary == 0
    assert records[1].labels.primary_category == "safe"


def test_load_ast_marks_variant_presence() -> None:
    raw = _case_dir("converters") / "ast.txt"
    raw.write_text(
        "1\t筷伽我赚钱\t快加我赚钱\n0\t正常聊天\t正常聊天\n",
        encoding="utf-8",
    )

    records = load_ast(raw)

    assert len(records) == 2
    assert records[0].labels.camouflage_type == "unknown"
    assert records[0].metadata["paired_text"] == "快加我赚钱"
    assert records[1].labels.camouflage_type == "none"
