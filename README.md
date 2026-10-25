# MM-ContentGuard

这是一个课程设计项目：系统模拟社交媒体/聊天发送界面，用户发送文本、图片或视频时，内容会先经过审核服务；审核通过才进入聊天记录，审核失败则在前端展示不通过原因。

本仓库是最终单仓库版本，不需要再 clone 文本、图片或视频子仓库。

> 项目初期我们基于课堂内容，实现了一个纯文本版本（仅使用传统机器学习方法，未使用BERT等深度学习方法），见 [text-detection-project](https://github.com/linsh63/text-detection-project)。

项目报告：[多模态内容审核系统报告](doc/project_report.md)

## demo



https://github.com/user-attachments/assets/603bf46f-0378-44cd-ae65-b82044edc32e



## 项目结构

```text
frontend/web/       聊天前端
backend/gateway/    统一审核网关，前端只调用这一层
services/text/      文本审核服务，三层 BERT 策略
services/image/     图片审核服务，图像模型 + OCR/语义模型
services/video/     视频审核服务，PCLMM 多模态模型
scripts/            下载资产、启动、检查和停止脚本
doc/                架构、指标、运行资产和参考文档
storage/            本地运行数据，运行时生成
```

## 复现范围

老师或助教复现时，不需要重新训练模型。

必需内容分两部分：

```text
代码：从 GitHub clone 本仓库获得
权重/缓存：从本仓库 GitHub Release 下载
```

数据集不是启动 demo 的必需项，只作为可选下载项提供给需要查看数据来源或复现实验的同学。

## 服务端口

```text
文本 API:    http://127.0.0.1:8000
图片 API:    http://127.0.0.1:8010
视频 API:    http://127.0.0.1:8020
审核网关:    http://127.0.0.1:8080
前端页面:    http://127.0.0.1:5173
```

## 1. Clone 仓库

```bash
git clone https://github.com/Wangych297/MM-ContentGuard.git
cd MM-ContentGuard
```

## 2. 准备 conda 环境

推荐 Python 3.10 和 CUDA 版 PyTorch：

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

验证 Python 环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import fastapi, httpx, uvicorn, transformers, whisper, easyocr, timm, pypinyin; print('env ok')"
ffmpeg -version
```

## 3. 准备前端依赖

```bash
cd frontend/web
npm ci
npm run build
cd ../..
```

## 4. 下载运行权重

模型权重、预训练模型缓存和 Whisper 权重不提交进 Git，需要从本仓库 Release 下载。

一键下载必需运行资产：

```bash
VERSION=v0.1.0 bash scripts/download_runtime_assets.sh
```

如果只想分开下载：

```bash
VERSION=v0.1.0 bash scripts/download_text_assets.sh
VERSION=v0.1.0 bash scripts/download_image_assets.sh
VERSION=v0.1.0 bash scripts/download_video_assets.sh
```

检查资产是否齐全：

```bash
bash scripts/check_runtime_assets.sh
```

看到下面输出即表示权重准备完成：

```text
Runtime assets are ready.
```

Release 文件清单见：

```text
doc/release_assets_checklist.md
doc/runtime_assets.md
```

## 5. 配置环境变量

```bash
cp .env.example .env
```

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

如果只想先用 CPU 或单卡调试，也可以把对应设备改成可用的本机配置，但视频和图片模型会明显变慢。

## 6. 启动系统

推荐一键后台启动：

```bash
bash scripts/start_all.sh
```

也可以分别启动：

```bash
bash scripts/start_text.sh
bash scripts/start_image.sh
bash scripts/start_video.sh
bash scripts/start_gateway.sh
bash scripts/start_frontend.sh
```

前端地址：

```text
http://127.0.0.1:5173/
```

## 7. 检查服务

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

## 8. 简单测试

安全文本：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'text=今天晚上一起吃饭吗' \
  -F 'client_message_id=local_safe_text'
```

风险文本：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'text=出售枪支弹药，赌博彩票网站，微信联系' \
  -F 'client_message_id=local_risk_text'
```

图片测试可以使用仓库内的小样例：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'files=@services/image/demo_samples/api_flow_samples/03_SIMAS_firearms_final_risk_stage1_weapon.jpg;type=image/jpeg' \
  -F 'client_message_id=local_risk_image'
```

视频 demo 素材体积较大，作为可选资产下载：

```bash
VERSION=v0.1.0 bash scripts/download_demo_assets.sh
```

下载后可用本地视频文件测试，也可以把路径换成任意本地 `.mp4`：

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'files=@demo/demo_short.mp4;type=video/mp4' \
  -F 'client_message_id=local_video'
```

## 9. 可选数据集

数据集不影响推理 demo。需要查看数据或复现实验时再下载：

```bash
VERSION=v0.1.0 bash scripts/download_dataset_assets.sh
```

训练和实验说明见：

```text
doc/bert_text_review/
doc/video_review/
doc/traditional_text_ml/
```

## 10. 停止服务

```bash
bash scripts/stop_all.sh
```

## 维护者打包 Release

在已经具备本机权重的机器上生成 Release 资产：

```bash
VERSION=v0.1.0 bash scripts/package_runtime_assets.sh
```

如果还要打包可选数据集和 demo 素材：

```bash
VERSION=v0.1.0 PACKAGE_DATASETS=true PACKAGE_DEMO=true bash scripts/package_runtime_assets.sh
```

生成的文件位于：

```text
release_assets/
```

上传清单见：

```text
doc/release_assets_checklist.md
```
