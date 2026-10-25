# 多模态聊天审核系统整合规划

## 1. 目标

把当前三个独立项目整合成一个完整演示系统：

- `risk-text-detection-system/`：文字风险检测 API。
- `risk_image_full_package_for_teammate_20260607/`：图片风险检测 API。
- `video-pclmm-project/`：视频风险检测 API。

最终效果是一个类似微信聊天的软件页面：用户输入文字、发送图片或发送视频时，系统先进行审核；只有审核通过的消息才真正进入聊天记录并展示给对方。若审核失败，消息不发出，并在本端提示失败原因。若底层项目能给出具体类别或原因，则把原因转成用户能理解的中文提示。

页面可以尽量复刻微信的布局、密度、气泡、会话列表和发送体验，但不要直接使用微信官方 logo、商标图标或受版权保护的素材；使用自定义头像、图标和通用绿色主题即可。

## 2. 当前三个 API 的能力梳理

### 2.1 文字 API

项目路径：

```text
risk-text-detection-system/
```

启动后默认接口：

```text
GET  /health
POST /text/check
POST /text/batch-check
```

单条文本请求：

```json
{
  "text": "加我微信，稳赚不赔",
  "risk_threshold": 0.7,
  "suspicious_threshold": 0.4
}
```

核心返回字段：

```json
{
  "risk_binary": "risk",
  "risk_level": "risk",
  "risk_probability": 0.98,
  "primary_category": "spam_general",
  "toxic_binary": "non_toxic",
  "camouflage_type": "unknown",
  "probabilities": {},
  "thresholds": {
    "suspicious": 0.4,
    "risk": 0.7
  }
}
```

可用于前端解释的字段：

- `risk_level`: `normal` / `suspicious` / `risk`
- `risk_probability`: 风险概率
- `primary_category`: 主要风险类别
- `toxic_binary`: 是否有毒性内容
- `camouflage_type`: 是否存在谐音、异体字、字形伪装等辅助判断

### 2.2 图片 API

项目路径：

```text
risk_image_full_package_for_teammate_20260607/
```

启动后默认接口：

```text
GET  /health
POST /image/check
```

当前图片 API 接收的是本地图片路径，而不是 multipart 上传文件：

```json
{
  "image_path": "/path/to/image.jpg",
  "use_ocr": true
}
```

核心返回字段：

```json
{
  "final_label": "violation",
  "risk_level": "risk",
  "risk_category": "weapon",
  "route": "blocked_by_image_pipeline",
  "reasons": [
    "image_model:weapon"
  ],
  "detectors": {
    "image_model": {},
    "ocr": {},
    "text_risk": {},
    "rules": [],
    "semantic_model": {}
  }
}
```

可用于前端解释的字段：

- `risk_level`: `normal` / `suspicious` / `risk`
- `risk_category`: 图片风险类别，例如 `sexual`、`violence`、`weapon`、`gambling`、`drug`、`regulated_goods`
- `reasons`: 模型、OCR、规则、语义模型给出的触发原因
- `route`: 被哪一层拦截，例如图片模型、规则、语义模型、人工复核

注意：图片 API 内部还可能调用文字 API 检查 OCR 文本，因此整合部署时要先启动文字服务，或把图片服务里的 `text_api.url` 配置指向统一网关提供的文字检测地址。

### 2.3 视频 API

项目路径：

```text
video-pclmm-project/
```

启动后默认接口：

```text
GET  /health
POST /predict
```

请求格式：

```text
multipart/form-data
file: mp4 视频文件
threshold: 可选
debug: 可选
force_recompute: 可选
```

最小返回：

```json
{
  "is_normal": true
}
```

调试返回会包含：

```json
{
  "is_normal": true,
  "score": 0.001,
  "threshold": 0.32,
  "prediction": 0,
  "timings": {},
  "feature_backend": "resident",
  "extraction_mode": "parallel"
}
```

可用于前端解释的字段：

- `is_normal`: 是否正常
- `score`: 异常 / PCL 概率，只有 debug 或扩展返回时可见
- `threshold`: 当前阈值

限制：当前视频 API 主要输出布尔值和分数，不能像文字、图片一样给出细粒度类别。因此前端失败原因应使用较保守的表述，例如“视频内容未通过多模态审核”。

## 3. 总体架构

推荐新建一个整合项目，采用 monorepo 形式：

```text
multimodal-risk-chat/
  services/
    text/                 文字项目，可用子模块、复制或软链接接入
    image/                图片项目，可用子模块、复制或软链接接入
    video/                视频项目，可用子模块、复制或软链接接入
  backend/
    gateway/              统一审核网关和聊天业务后端
  frontend/
    web/                  微信式聊天前端
  storage/
    uploads/tmp/          待审核临时文件
    uploads/approved/     审核通过后正式文件
    db/                   SQLite 或其他轻量数据库
  scripts/
    start_text.sh
    start_image.sh
    start_video.sh
    start_gateway.sh
    start_frontend.sh
    start_all.sh
  deploy/
    docker-compose.yml
  doc/
    integration_plan.md
```

核心原则：

```text
前端只调用 gateway
gateway 负责保存临时文件、调用三个审核 API、统一结果、决定是否入库
文字 / 图片 / 视频服务保持独立，尽量少改原项目
```

服务端口建议：

```text
文字 API:    http://127.0.0.1:8000
图片 API:    http://127.0.0.1:8010
视频 API:    http://127.0.0.1:8020
审核网关:    http://127.0.0.1:8080
前端页面:    http://127.0.0.1:5173
```

视频 API 默认端口也是 8000，整合时要通过 `PORT=8020 scripts/run_api.sh` 避免和文字 API 冲突。

## 4. 统一审核网关设计

### 4.1 网关职责

网关是整个系统最关键的一层，负责：

- 对前端提供统一消息接口。
- 接收文字、图片、视频消息。
- 把上传文件先保存到临时目录。
- 根据消息类型调用对应模型 API。
- 把不同模型 API 的返回统一成一个审核结果。
- 审核通过后才写入聊天记录，并把临时文件移动到正式目录。
- 审核失败时删除或隔离临时文件，并返回失败原因。
- 记录每次审核的原始结果，便于调试。

### 4.2 推荐技术栈

后端建议使用 Python FastAPI：

- 三个已有模型 API 都是 Python / FastAPI 风格，调用和部署一致。
- 文件上传、异步任务、HTTP client、WebSocket 支持成熟。
- 方便复用现有 Python 环境和脚本。

推荐依赖：

```text
fastapi
uvicorn
httpx
python-multipart
sqlmodel 或 sqlalchemy
pydantic
aiofiles
```

### 4.3 网关核心接口

#### 健康检查

```text
GET /api/health
```

返回：

```json
{
  "ok": true,
  "services": {
    "text": {"ok": true, "url": "http://127.0.0.1:8000"},
    "image": {"ok": true, "url": "http://127.0.0.1:8010"},
    "video": {"ok": true, "url": "http://127.0.0.1:8020"}
  }
}
```

#### 发送消息

```text
POST /api/conversations/{conversation_id}/messages
```

请求格式使用 `multipart/form-data`，同时支持文字和文件：

```text
text: 可选
files: 可选，可包含图片或视频
client_message_id: 前端生成的临时消息 ID
```

返回分两种模式：

1. 文本和图片可以先做同步返回。
2. 视频耗时较长，建议采用异步审核返回。

同步通过：

```json
{
  "message_id": "msg_001",
  "status": "approved",
  "visible_to_others": true,
  "message": {
    "text": "你好",
    "attachments": []
  },
  "review": {
    "allow": true,
    "summary": "审核通过",
    "results": []
  }
}
```

同步拒绝：

```json
{
  "message_id": null,
  "status": "rejected",
  "visible_to_others": false,
  "review": {
    "allow": false,
    "summary": "发送失败：文本疑似广告或诈骗内容",
    "reasons": [
      {
        "modality": "text",
        "label": "spam_general",
        "display": "广告 / 垃圾信息",
        "score": 0.98
      }
    ]
  }
}
```

视频异步审核初始返回：

```json
{
  "client_message_id": "local_abc",
  "status": "reviewing",
  "visible_to_others": false,
  "review_job_id": "job_001"
}
```

#### 查询审核任务

```text
GET /api/review-jobs/{review_job_id}
```

返回：

```json
{
  "review_job_id": "job_001",
  "status": "approved",
  "message_id": "msg_002",
  "review": {
    "allow": true,
    "summary": "审核通过"
  }
}
```

前端也可以用 WebSocket / SSE 接收审核结果：

```text
GET /api/ws
```

MVP 阶段可以先用轮询，后续再上 WebSocket。

#### 获取聊天记录

```text
GET /api/conversations/{conversation_id}/messages
```

只返回审核通过的正式消息。被拒绝的消息默认只在发送者本地显示失败状态，不进入公共聊天记录。

## 5. 统一审核结果数据结构

三个模型返回字段不同，必须统一为 `ReviewDecision`：

```json
{
  "allow": false,
  "status": "rejected",
  "summary": "发送失败：图片包含武器相关内容",
  "reasons": [
    {
      "modality": "image",
      "code": "weapon",
      "display": "武器相关内容",
      "score": 0.96,
      "source": "image_model",
      "raw_reason": "image_model:weapon"
    }
  ],
  "raw": {
    "text": {},
    "image": {},
    "video": {}
  }
}
```

字段约定：

- `allow`: 是否允许发出。
- `status`: `approved` / `rejected` / `reviewing` / `error`。
- `summary`: 给用户看的中文提示。
- `reasons`: 结构化原因列表。
- `raw`: 底层 API 原始返回，只给调试页面或日志使用，不直接展示给普通用户。

### 5.1 审核策略

MVP 阶段建议：

```text
任一模态 risk      -> 拒绝发送
任一模态 suspicious -> 暂时拒绝发送，并提示“疑似风险”
所有模态 normal    -> 允许发送
模型服务异常       -> 拒绝发送或提示稍后重试，不允许绕过审核
```

后续如果需要人工审核，可以把 `suspicious` 改成进入人工审核队列。

### 5.2 类别中文映射

文字类别示例：

```text
spam_general      -> 广告 / 垃圾信息
ad_promotion      -> 推广引流
loan_ad           -> 贷款广告
fraud_phishing    -> 诈骗 / 钓鱼
review_spam       -> 刷评 / 垃圾推广
porn_sexual       -> 色情内容
gambling          -> 赌博相关
toxic             -> 攻击性 / 侮辱性内容
unknown           -> 风险文本
```

图片类别示例：

```text
sexual            -> 色情 / 性暗示图片
violence          -> 暴力内容
weapon            -> 武器相关内容
gambling          -> 赌博相关图片
drug              -> 毒品相关内容
regulated_goods   -> 受管制商品
rule_sensitive    -> 命中敏感词规则
rule_contact_ad   -> 疑似联系方式引流
targeted_offense  -> 针对性冒犯
general_offense   -> 一般冒犯
dispirited        -> 消极有害表达
```

视频类别当前较粗：

```text
video_abnormal    -> 视频内容未通过多模态审核
pcl_risk          -> 视频可能存在贬抑 / 冒犯性表达
```

如果底层 API 只能返回布尔值，则前端只展示通用原因，不编造细节。

## 6. 消息发送流程

### 6.1 文本消息

```text
用户点击发送
-> 前端生成本地 pending 气泡
-> POST gateway /messages
-> gateway 调用文字 API /text/check
-> risk_level=normal: 写入消息表，返回 approved
-> risk_level=risk/suspicious: 不写入正式消息，返回 rejected + 原因
-> 前端 approved: pending 气泡变成正式气泡
-> 前端 rejected: 气泡变红或消失，弹出失败原因
```

### 6.2 图片消息

```text
用户选择图片
-> 前端显示待发送缩略图
-> POST gateway /messages
-> gateway 保存到 storage/uploads/tmp
-> gateway 把本地路径传给图片 API /image/check
-> 图片 API 内部可 OCR 并调用文字 API
-> 通过: 文件移动到 storage/uploads/approved，消息入库
-> 拒绝: 删除或隔离临时文件，返回原因
```

注意：因为图片 API 当前只支持 `image_path`，所以 gateway 和图片 API 必须运行在同一台机器，或共享同一个上传目录。如果未来要容器化，需把上传目录挂载给 gateway 和 image service。

### 6.3 视频消息

```text
用户选择视频
-> 前端显示“上传中 / 审核中”
-> gateway 保存 mp4 到临时目录
-> gateway 调用视频 API /predict
-> 视频 API 抽取四类特征并融合判断
-> 通过: 移动视频到正式目录，消息入库
-> 拒绝: 不入库，提示“视频内容未通过审核”
```

视频审核可能需要数秒到几十秒。前端不应卡住整个页面，而是保留本地 pending 气泡和进度状态：

```text
uploading -> reviewing -> sent
uploading -> reviewing -> rejected
uploading -> failed
```

## 7. 前端规划

### 7.1 推荐技术栈

```text
React + Vite + TypeScript
CSS Modules 或 Tailwind
Lucide icons
fetch / axios
```

不建议一开始使用复杂状态管理库。MVP 可使用 React state + context；后续再按复杂度引入 Zustand。

### 7.2 页面结构

桌面端页面采用微信 PC 类布局：

```text
┌──────────┬──────────────┬──────────────────────────────┐
│ 左侧窄栏 │ 会话列表      │ 当前聊天窗口                   │
│ 头像/菜单│ 搜索/联系人   │ 顶部标题                       │
│          │ 最近消息      │ 消息列表                       │
│          │              │ 输入区：文字/图片/视频/发送按钮 │
└──────────┴──────────────┴──────────────────────────────┘
```

移动端可退化为单栏：

```text
会话列表 -> 聊天详情
```

### 7.3 视觉要求

- 整体接近微信的布局密度、聊天气泡、浅灰背景、右侧本人消息、左侧对方消息。
- 本人消息气泡使用自定义绿色，不直接使用微信品牌素材。
- 对方消息使用白色气泡。
- 图片消息显示缩略图，点击可预览。
- 视频消息显示封面、时长和播放按钮。
- 审核中的消息显示小型 loading 和“审核中”。
- 审核失败的消息显示红色感叹号，点击展示具体原因。
- 不要在页面上大段解释系统功能，交互应自然融入聊天体验。

### 7.4 前端组件拆分

```text
App
ConversationShell
Sidebar
ConversationList
ChatHeader
MessageList
MessageBubble
AttachmentPreview
MessageComposer
ReviewStatusBadge
RejectReasonDialog
MediaPreviewModal
```

### 7.5 前端状态模型

```ts
type MessageStatus =
  | "draft"
  | "uploading"
  | "reviewing"
  | "sent"
  | "rejected"
  | "failed";

type Message = {
  id: string;
  clientMessageId: string;
  conversationId: string;
  sender: "me" | "other";
  text?: string;
  attachments: Attachment[];
  status: MessageStatus;
  rejectReason?: string;
  review?: ReviewDecision;
  createdAt: string;
};
```

### 7.6 发送交互

前端不要等审核完才创建气泡，而是先创建本地 pending 气泡：

```text
点击发送
-> 本地立即出现右侧气泡，状态为 uploading/reviewing
-> 后端返回 approved: 状态变 sent
-> 后端返回 rejected: 状态变 rejected，并弹出原因
-> 后端返回 failed: 状态变 failed，可重试
```

对于审核失败消息：

- 默认不进入正式聊天记录。
- 发送者本地可以短暂看到失败气泡。
- 点击失败气泡展示原因。
- 对方永远看不到该消息。

## 8. 后端数据模型

MVP 使用 SQLite 即可：

```text
users
  id
  display_name
  avatar_url

conversations
  id
  title
  created_at

conversation_members
  conversation_id
  user_id

messages
  id
  conversation_id
  sender_id
  text
  status
  created_at

attachments
  id
  message_id
  modality
  original_name
  mime_type
  storage_path
  thumbnail_path
  size_bytes

review_jobs
  id
  client_message_id
  conversation_id
  status
  created_at
  completed_at

review_results
  id
  job_id
  modality
  allow
  risk_level
  category
  score
  display_reason
  raw_json
```

被拒绝的消息可以不写入 `messages`，只写入 `review_jobs` 和 `review_results`；也可以写入 `messages` 但 `status=rejected` 且只对发送者可见。MVP 推荐后者，前端调试更直观。

## 9. 服务启动方案

### 9.1 本地开发

先分别启动三个模型服务：

```bash
# text
cd risk-text-detection-system
PYTHONPATH=src RISK_TEXT_PORT=8000 python -m risk_text_detection.api.app

# image
cd risk_image_full_package_for_teammate_20260607
PYTHONPATH=src uvicorn api.app:app --host 127.0.0.1 --port 8010

# video
cd video-pclmm-project
PORT=8020 scripts/run_api.sh
```

再启动整合网关：

```bash
cd multimodal-risk-chat/backend/gateway
TEXT_API_URL=http://127.0.0.1:8000 \
IMAGE_API_URL=http://127.0.0.1:8010 \
VIDEO_API_URL=http://127.0.0.1:8020 \
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

最后启动前端：

```bash
cd multimodal-risk-chat/frontend/web
npm install
npm run dev
```

### 9.2 一键启动脚本

后续提供：

```text
scripts/start_all.sh
scripts/stop_all.sh
scripts/check_services.sh
```

`start_all.sh` 负责：

- 检查端口占用。
- 启动文字服务。
- 启动图片服务。
- 启动视频服务。
- 等待三个 `/health` 正常。
- 启动 gateway。
- 启动 frontend dev server。

### 9.3 Docker Compose

如果后续要交给更多成员部署，可提供 `docker-compose.yml`：

```text
text-service
image-service
video-service
gateway
frontend
```

注意 GPU 服务需要额外配置 NVIDIA runtime。图片服务和 gateway 要共享上传目录，否则图片 API 无法读取 gateway 保存的图片路径。

## 10. 错误处理策略

### 10.1 模型拒绝

返回 `rejected`，并展示模型原因：

```text
发送失败：文本疑似诈骗 / 广告内容
发送失败：图片包含武器相关内容
发送失败：视频内容未通过多模态审核
```

### 10.2 模型服务不可用

默认不允许绕过审核：

```text
发送失败：审核服务暂不可用，请稍后重试
```

网关日志记录具体服务：

```text
text service timeout
image service connection refused
video service 500
```

### 10.3 文件格式不支持

前端提前限制：

```text
图片: jpg / jpeg / png / webp
视频: mp4
```

后端再次校验 MIME 和扩展名。

### 10.4 超时

建议超时：

```text
text: 5 秒
image: 30 秒
video: 120 秒
```

视频超时后返回 `reviewing`，由后台任务继续处理；前端通过轮询获取结果。

## 11. 安全和隐私

- 前端只调用 gateway，不暴露三个模型 API 地址。
- 不把 API key、VLM key、真实服务器路径提交到仓库。
- 上传文件先进入临时目录，审核失败后删除或隔离。
- 日志中避免记录完整敏感文本；可记录 hash、类别、分数和 request id。
- 限制单个文件大小，例如图片 10MB、视频 200MB。
- 限制视频时长，例如 MVP 先限制 60 秒以内。
- CORS 只允许前端域名。
- 后续如有真实用户，需加入登录态和权限校验。

## 12. 实施阶段

### 阶段 1：接口验证和适配表

目标：确认三个子服务都能独立启动，并确定输入输出字段。

任务：

- 启动文字 API，测试 `/health` 和 `/text/check`。
- 启动图片 API，测试 `/health` 和 `/image/check`。
- 启动视频 API，测试 `/health` 和 `/predict`。
- 写 `backend/gateway/app/adapters/`：
  - `text_adapter.py`
  - `image_adapter.py`
  - `video_adapter.py`
- 每个 adapter 统一返回 `ModalityReviewResult`。

验收：

- 三个 adapter 可用 mock 或真实 API 单测通过。
- 任一 API 返回异常时，adapter 能转换为统一错误。

### 阶段 2：审核网关 MVP

目标：前端只调用一个后端接口即可完成文字、图片、视频审核。

任务：

- 建立 FastAPI gateway。
- 实现 `/api/health`。
- 实现 `/api/conversations/{id}/messages`。
- 实现临时上传目录和正式文件目录。
- 实现审核通过后入库，审核失败不公开展示。
- 实现类别中文映射。

验收：

- 正常文本能发送。
- 风险文本被拦截并返回原因。
- 正常图片能发送。
- 风险图片被拦截并返回原因。
- 正常视频能发送。
- 风险视频被拦截并返回通用原因。

### 阶段 3：微信式前端 MVP

目标：完成可演示的聊天页面。

任务：

- 创建 React + Vite 前端。
- 实现三栏桌面布局。
- 实现会话列表、消息列表、输入框。
- 实现文字发送。
- 实现图片上传预览和发送。
- 实现视频上传预览和发送。
- 实现 pending / reviewing / sent / rejected 状态。
- 实现失败原因弹窗。

验收：

- 页面视觉接近微信聊天主界面。
- 用户能发送三种模态消息。
- 审核失败时消息不会正式发出。
- 能看到具体失败原因。

### 阶段 4：异步任务和体验优化

目标：让视频等慢任务体验自然，不阻塞页面。

任务：

- 实现 `review_jobs` 表。
- 视频审核走后台任务。
- 前端轮询 `/api/review-jobs/{id}`。
- 可选实现 WebSocket 推送。
- 增加上传进度、审核中动画、失败重试。

验收：

- 视频审核期间页面仍可操作。
- 审核完成后气泡自动变成 sent 或 rejected。

### 阶段 5：部署和文档

目标：让小组成员能一键跑起来。

任务：

- 写 `README.md`。
- 写 `.env.example`。
- 写 `scripts/start_all.sh`。
- 写 `scripts/check_services.sh`。
- 补充端口、模型资产、VLM key、GPU 说明。
- 可选提供 Docker Compose。

验收：

- 新成员按 README 能完整启动系统。
- `/api/health` 能看到三个子服务均正常。
- 前端能完成一轮文本、图片、视频发送和拦截演示。

## 13. 推荐优先级

最高优先级：

```text
1. 统一审核网关
2. 微信式聊天前端基础界面
3. 文字 / 图片 / 视频发送审核闭环
4. 失败原因展示
```

第二优先级：

```text
1. 视频异步任务
2. WebSocket 状态推送
3. 图片和视频缩略图
4. 一键启动脚本
```

第三优先级：

```text
1. Docker Compose
2. 管理员审核后台
3. 审核日志可视化
4. 多用户登录
```

## 14. 关键风险

### 14.1 图片 API 当前只支持 image_path

影响：如果 gateway 和图片服务不在同一台机器或不共享目录，图片服务无法读取上传文件。

解决：

- MVP 让 gateway 和图片服务运行在同一台机器。
- 或给图片项目补一个 multipart `/image/check-upload` 接口。
- 容器化时共享 volume。

### 14.2 视频审核耗时较长

影响：用户发送视频时等待较久。

解决：

- 前端显示“上传中 / 审核中”。
- 后端用异步任务。
- 限制视频长度和大小。
- 当前视频服务使用 resident + parallel 模式。

### 14.3 三个项目环境可能冲突

影响：文字、图片、视频依赖不同，单一 conda 环境可能冲突。

解决：

- MVP 可分别用三个环境启动三个服务。
- gateway 和 frontend 独立环境。
- 后续用 Docker Compose 隔离。

### 14.4 图片语义模型依赖外部 VLM API

影响：没有 API key 或网络不通时，图片第二层语义检测能力下降。

解决：

- `.env.example` 只写变量名，不写 key。
- 健康检查显示 VLM 配置状态。
- VLM 不可用时仍保留第一层图片模型、OCR、规则检测。

### 14.5 不能编造审核原因

影响：视频 API 只有布尔值时无法给出细类。

解决：

- 对文字、图片展示细粒度原因。
- 对视频只展示“视频内容未通过多模态审核”。
- 调试模式可显示分数，但普通用户不展示内部模型细节。

## 15. MVP 完成定义

满足以下条件即可认为第一版整合项目完成：

- 前端页面有微信式会话列表和聊天窗口。
- 用户可以发送文字、图片、视频。
- 发送行为先进入审核状态。
- 审核通过后消息进入聊天记录。
- 审核失败后消息不对外发出。
- 文字风险能展示具体类别。
- 图片风险能展示具体类别或规则原因。
- 视频风险能展示通用审核失败原因。
- gateway 统一调用三个模型服务，前端不直连模型服务。
- README 能指导小组成员启动所有服务并完成演示。
