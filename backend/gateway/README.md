# Gateway

统一审核网关。前端只调用这一层，gateway 负责保存上传文件、调用文字 / 图片 / 视频审核服务，并决定消息是否真正发出。

## 接口

```text
GET  /api/health
POST /api/conversations/{conversation_id}/messages
GET  /api/conversations/{conversation_id}/messages
```

## 启动

```bash
conda activate multimodal-risk-chat
cd /path/to/multimodal-risk-chat
bash scripts/start_gateway.sh
```

也可以直接启动：

```bash
PYTHON_BIN=python \
TEXT_API_URL=http://127.0.0.1:8000 \
TEXT_API_BACKEND=risk_text \
IMAGE_API_URL=http://127.0.0.1:8010 \
VIDEO_API_URL=http://127.0.0.1:8020 \
python -m uvicorn app.main:app --app-dir backend/gateway --host 127.0.0.1 --port 8080
```

文本后端默认使用本仓库的三层 BERT 服务：

```text
TEXT_API_BACKEND=risk_text -> POST {TEXT_API_URL}/text/check
```

## 健康检查

```bash
curl --noproxy '*' http://127.0.0.1:8080/api/health
```

返回中 `text.ok`、`image.ok`、`video.ok` 都为 `true` 时，三个底层审核服务都已经可用。

## 发送消息

请求使用 `multipart/form-data`。

字段：

```text
text: 可选，消息文字
files: 可选，可重复传入图片或 mp4 视频
client_message_id: 可选，前端本地消息 ID
sender_id: 可选，默认 me
```

发送文本：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'text=今天晚上一起吃饭吗' \
  -F 'client_message_id=local_text_1'
```

发送图片：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'client_message_id=local_image_1' \
  -F 'files=@/path/to/image.jpg;type=image/jpeg'
```

发送视频：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'client_message_id=local_video_1' \
  -F 'files=@/path/to/video.mp4;type=video/mp4'
```

审核通过时：

```json
{
  "status": "approved",
  "visible_to_others": true,
  "review": {
    "allow": true,
    "summary": "审核通过",
    "reasons": []
  }
}
```

审核失败时：

```json
{
  "status": "rejected",
  "visible_to_others": false,
  "review": {
    "allow": false,
    "summary": "发送失败：武器相关内容",
    "reasons": [
      {
        "modality": "image",
        "code": "weapon",
        "display": "武器相关内容"
      }
    ]
  }
}
```

## 获取聊天记录

默认只返回审核通过的正式消息：

```bash
curl --noproxy '*' http://127.0.0.1:8080/api/conversations/demo/messages
```

调试时可以包含未通过消息：

```bash
curl --noproxy '*' 'http://127.0.0.1:8080/api/conversations/demo/messages?include_rejected=true'
```

## 媒体文件

gateway 会把上传文件保存到：

```text
storage/uploads/tmp/
storage/uploads/approved/
storage/uploads/rejected/
```

审核通过或失败后，返回的 `message.attachments[].storage_path` 可以通过 `/media/uploads/...` 访问。前端已经内置了路径转换逻辑。
