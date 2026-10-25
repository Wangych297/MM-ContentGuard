"""
文件功能：FastAPI 网关应用入口。创建 FastAPI 实例，配置 CORS 中间件，
注册健康检查和消息审核路由，并挂载上传文件的静态资源目录。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.messages import router as messages_router
from app.core.config import get_settings


app = FastAPI(title="Multimodal Risk Chat Gateway", version="0.1.0")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(messages_router)

(settings.storage_root / "uploads").mkdir(parents=True, exist_ok=True)
app.mount(
    "/media/uploads",
    StaticFiles(directory=settings.storage_root / "uploads"),
    name="uploads",
)
