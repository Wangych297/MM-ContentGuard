# 组员运行手册

这份文档面向小组成员，目标是从 GitHub clone 后跑通完整演示系统：

```text
文字审核服务 + 图片审核服务 + 视频审核服务 + 统一 gateway + 微信式前端
```

## 1. Clone 仓库

```bash
git clone git@github.com:<YOUR_ORG_OR_USER>/multimodal-risk-chat.git
cd multimodal-risk-chat
```

如果使用 HTTPS：

```bash
git clone https://github.com/<YOUR_ORG_OR_USER>/multimodal-risk-chat.git
cd multimodal-risk-chat
```

## 2. 准备 conda 环境

推荐 Python 3.10。GPU 机器上先安装 CUDA 版 PyTorch，再安装统一 pip 依赖。

```bash
conda create -y -n multimodal-risk-chat python=3.10
conda activate multimodal-risk-chat

conda install -y -c pytorch -c nvidia \
  pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1

conda install -y -c conda-forge ffmpeg

python -m pip install "setuptools<70" wheel packaging
python -m pip install --no-build-isolation -r requirements-unified.txt
python -m pip install -e backend/gateway
python -m pip install -e services/text
python -m pip install -e 'services/image[ocr]'
```

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import fastapi, httpx, uvicorn, transformers, whisper, easyocr, timm, pypinyin; print('env ok')"
ffmpeg -version
```

## 3. 准备前端依赖

```bash
cd frontend/web
npm install
npm run build
cd ../..
```

说明：`scripts/start_frontend.sh` 默认会重新构建前端并用稳定的静态服务运行 `dist`，同时把 `/api` 和 `/media` 代理到 gateway。这样演示时不依赖 Vite dev server / HMR，远程端口转发更稳定。需要前端热更新调试时，可临时使用 `FRONTEND_MODE=dev bash scripts/start_frontend.sh`。

## 4. 准备运行资产

模型权重、缓存和大文件不提交进 Git，需要从本仓库 GitHub Release 下载。在仓库根目录执行：

```bash
VERSION=v0.1.0 bash scripts/download_runtime_assets.sh
```

下载后需要保持下面这些路径存在：

```text
services/text/runs/social_binary_v3_macbert_binary_bs128_no_online_aug/
services/text/runs/social_intent_binary_v1_from_general/
services/text/runs/reason_category_v1_macbert/
services/text/models/hfl/chinese-macbert-base/
services/text/data/raw/ast/

services/image/checkpoints/image_model_dinov3_large/best.pt
services/image/checkpoints/semantic_model_focal_binary/best.pt
services/image/.cache/manual_weights/dinov3_large/model.safetensors
services/image/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/
services/image/.EasyOCR/model/

services/video/outputs/checkpoints/multi_modal_cross_attention_model.pth
services/video/pretrained/googlevit-base-patch16-224-in21k/
services/video/pretrained/bert_chinese/
services/video/pretrained/FER-VT/
services/video/outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth
services/video/outputs/cache/whisper/large-v3.pt
```

原始训练数据集不需要放入本仓库，推理演示不依赖训练集。更详细的资产说明见：

```text
doc/runtime_assets.md
```

资产准备完成后检查：

```bash
bash scripts/check_runtime_assets.sh
```

## 5. 配置环境变量

```bash
cp .env.example .env
```

文本审核 API 默认使用本仓库内的三层 BERT 服务：

```text
TEXT_API_BACKEND=risk_text
```

说明：这个配置用于聊天文本消息的 gateway 审核。图片服务内部 OCR 文本联动仍由图片服务自己的 `text_api` 配置决定。

`.env.example` 不绑定具体 GPU 编号。默认配置会让各服务自动选择可用设备：

```text
TEXT_DEVICE=auto
TEXT_CUDA_VISIBLE_DEVICES=
IMAGE_CUDA_VISIBLE_DEVICES=
VIDEO_CUDA_VISIBLE_DEVICES=
PCLMM_API_DEVICE=auto
PCLMM_API_VIT_DEVICE=auto
PCLMM_API_WHISPER_DEVICE=auto
PCLMM_API_BERT_DEVICE=auto
PCLMM_API_FACE_DEVICE=auto
```

如果想指定某几张 GPU，再编辑 `.env`。例如只让视频服务看到 0、1 号卡：

```text
VIDEO_CUDA_VISIBLE_DEVICES=0,1
PCLMM_API_DEVICE=cuda:1
PCLMM_API_VIT_DEVICE=cuda:0
PCLMM_API_WHISPER_DEVICE=cuda:0
PCLMM_API_BERT_DEVICE=cuda:0
PCLMM_API_FACE_DEVICE=cuda:1
```

## 6. 启动服务

推荐一键后台启动：

```bash
bash scripts/start_all.sh
```

也可以分别开终端启动：

```bash
bash scripts/start_text.sh
bash scripts/start_image.sh
bash scripts/start_video.sh
bash scripts/start_gateway.sh
bash scripts/start_frontend.sh
```

服务端口：

```text
文字 API:    http://127.0.0.1:8000
图片 API:    http://127.0.0.1:8010
视频 API:    http://127.0.0.1:8020
审核网关:    http://127.0.0.1:8080
前端页面:    http://127.0.0.1:5173
```

## 7. 验证

```bash
bash scripts/check_services.sh
```

预期五项均为 `[ok]`：

```text
text
image
video
gateway
frontend
```

验证 gateway 文本审核：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'text=今天晚上一起吃饭吗' \
  -F 'client_message_id=local_safe_text'
```

风险文本会被拒发：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'text=出售枪支弹药，赌博彩票网站，微信联系' \
  -F 'client_message_id=local_risk_text'
```

打开前端：

```text
http://127.0.0.1:5173/
```

如果通过 VS Code / SSH 做端口转发，优先转发 `5173`。该端口现在是静态生产前端入口，不再依赖 Vite HMR。

## 8. 停止服务

```bash
bash scripts/stop_all.sh
```

## 9. 常见问题

### 9.1 端口已被占用

先检查：

```bash
ss -ltnp | grep -E ':(8000|8010|8020|8080|5173)\b'
```

停止旧服务：

```bash
bash scripts/stop_all.sh
```

### 9.2 代理导致 localhost 请求失败

`scripts/check_services.sh` 已经内置 no-proxy。手动 curl 命令建议统一加：

```bash
--noproxy '*'
```

gateway 内部已经使用 `trust_env=False`，不会把本地服务调用发到系统代理。

### 9.3 视频服务启动很慢

视频服务会常驻加载 ViT、Whisper、BERT、FER-VT 和融合模型。首次启动几十秒是正常现象。

### 9.4 GitHub 仓库里没有权重

这是预期行为。权重和缓存目录由 `.gitignore` 忽略，需要通过 Release、云盘或服务器共享目录补齐。
