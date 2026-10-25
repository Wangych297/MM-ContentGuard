"""
文件功能：二维码检测器。使用 pyzbar 库检测图片中的二维码，
解码其中的文本内容并作为规则命中结果返回。

核心函数：
- detect_qrcodes: 检测图片中的二维码并返回解码文本
"""

from __future__ import annotations

from pathlib import Path

from common.image_io import load_rgb_image
from common.schemas import RuleHit


def detect_qrcodes(image_path: str | Path) -> list[RuleHit]:
    try:
        import cv2
    except ImportError:
        return []

    try:
        import numpy as np

        image = cv2.cvtColor(np.array(load_rgb_image(image_path)), cv2.COLOR_RGB2BGR)
    except Exception:
        return []
    detector = cv2.QRCodeDetector()
    try:
        ok, decoded_info, _points, _straight = detector.detectAndDecodeMulti(image)
        if ok:
            return [
                RuleHit(name="qrcode", value=value or "<empty>", metadata={"index": idx})
                for idx, value in enumerate(decoded_info)
            ]
        value, points, _straight = detector.detectAndDecode(image)
        if points is not None:
            return [RuleHit(name="qrcode", value=value or "<empty>")]
    except Exception:
        return []
    return []


