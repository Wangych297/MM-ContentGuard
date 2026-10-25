"""
文件功能：前端静态文件服务。FastAPI 应用，将构建后的前端静态资源
（frontend/web/dist/）作为静态文件提供服务，同时将 /api 和 /media
请求代理转发至网关服务。用于生产模式下替代 Vite 开发服务器。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask


ROOT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", ROOT_DIR / "frontend" / "web" / "dist"))
GATEWAY_URL = os.getenv("GATEWAY_URL") or (
    "http://%s:%s" % (os.getenv("GATEWAY_HOST", "127.0.0.1"), os.getenv("GATEWAY_PORT", "8080"))
)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

if not DIST_DIR.exists():
    raise RuntimeError(f"Frontend dist directory does not exist: {DIST_DIR}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    timeout = httpx.Timeout(600.0, connect=10.0)
    app.state.client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    print(
        "Static frontend serving %s; proxying /api and /media to %s"
        % (DIST_DIR, GATEWAY_URL),
        flush=True,
    )
    try:
        yield
    finally:
        client: httpx.AsyncClient | None = getattr(app.state, "client", None)
        if client is not None:
            await client.aclose()


app = FastAPI(
    title="Multimodal Risk Chat Frontend",
    version="0.1.0",
    lifespan=lifespan,
)

assets_dir = DIST_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


def _proxy_request_headers(request: Request) -> dict[str, str]:
    headers = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower == "host" or lower in HOP_BY_HOP_HEADERS:
            continue
        headers[key] = value
    return headers


def _proxy_response_headers(response: httpx.Response) -> dict[str, str]:
    headers = {}
    for key, value in response.headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS:
            continue
        headers[key] = value
    return headers


async def _proxy(request: Request, upstream_path: str) -> Response:
    client: httpx.AsyncClient = app.state.client
    url = "%s/%s" % (GATEWAY_URL.rstrip("/"), upstream_path.lstrip("/"))
    body = await request.body()
    upstream_request = client.build_request(
        method=request.method,
        url=url,
        params=request.query_params,
        headers=_proxy_request_headers(request),
        content=body,
    )
    upstream_response = await client.send(upstream_request, stream=True)
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_proxy_response_headers(upstream_response),
        background=BackgroundTask(upstream_response.aclose),
    )


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_api(path: str, request: Request) -> Response:
    return await _proxy(request, f"api/{path}")


@app.api_route(
    "/media/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_media(path: str, request: Request) -> Response:
    return await _proxy(request, f"media/{path}")


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def spa(path: str) -> Response:
    requested = (DIST_DIR / path).resolve()
    try:
        requested.relative_to(DIST_DIR.resolve())
    except ValueError:
        requested = DIST_DIR / "index.html"

    if requested.is_file():
        return FileResponse(requested)
    return FileResponse(DIST_DIR / "index.html")


def main() -> None:
    uvicorn.run(
        app,
        host=os.getenv("FRONTEND_HOST", "127.0.0.1"),
        port=int(os.getenv("FRONTEND_PORT", "5173")),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
