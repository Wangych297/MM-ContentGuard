"""
文件功能：API 端点测试。验证文本审核服务的 /health 端点是否正常返回 200。
"""

from fastapi.testclient import TestClient

from risk_text_detection.api.app import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "checkpoint_dir" in response.json()
