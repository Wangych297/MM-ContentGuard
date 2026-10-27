# Web Frontend

微信式聊天前端 MVP。页面负责展示会话列表、聊天气泡、图片 / 视频预览、审核中状态和失败原因。

## 启动

```bash
cd /path/to/MM-ContentGuard/frontend/web
npm install
npm run dev
```

或从仓库根目录启动：

```bash
bash scripts/start_frontend.sh
```

浏览器打开：

```text
http://127.0.0.1:5173/
```

## 网关

开发服务器会把这些路径代理到 gateway：

```text
/api   -> http://127.0.0.1:8080
/media -> http://127.0.0.1:8080
```

因此本地开发时前端可以直接请求：

```text
/api/conversations/demo/messages
```

如果要改为直连其他 gateway 地址，可以设置：

```bash
VITE_GATEWAY_URL=http://127.0.0.1:8080 npm run dev
```

## 构建

```bash
npm run build
```

当前已通过 TypeScript 和 Vite 生产构建验证。

## 当前交互

- 文本、图片、视频都通过同一个发送入口提交给 gateway。
- 发送后先显示“审核中”。
- 审核通过后显示“已发送”，消息进入聊天记录。
- 审核失败后显示红色失败提示，并展示 gateway 返回的中文原因。
