"""
文件功能：标签词汇单元测试。验证 LabelVocab 的编码行为，
包括已知标签编码和缺失标签的 IGNORE_INDEX 处理。
"""

from risk_text_detection.training.labeling import IGNORE_INDEX, LabelVocab


def test_label_vocab_encodes_static_and_missing_labels() -> None:
    vocab = LabelVocab.build(
        [
            {
                "labels": {
                    "risk_binary": 1,
                    "primary_category": "spam_general",
                    "toxic_binary": None,
                }
            }
        ],
        ["risk_binary", "primary_category", "toxic_binary"],
    )

    encoded = vocab.encode_record(
        {
            "labels": {
                "risk_binary": 1,
                "primary_category": "spam_general",
                "toxic_binary": None,
            }
        }
    )

    assert encoded["risk_binary"] == vocab.tasks["risk_binary"].label_to_id["1"]
    assert encoded["primary_category"] == vocab.tasks["primary_category"].label_to_id["spam_general"]
    assert encoded["toxic_binary"] == IGNORE_INDEX
