# 运行资产说明

本仓库只提交代码、配置、脚本和少量 demo 样例。模型权重、预训练缓存、Whisper 权重等大文件不进 Git，需要从 `MM-ContentGuard` 的 GitHub Release 下载。

## 1. 一键下载

在仓库根目录执行：

```bash
VERSION=v0.1.0 bash scripts/download_runtime_assets.sh
```

这个命令会依次下载并安装：

```text
文本审核运行资产
图片审核运行资产
视频审核运行资产
```

如果只想单独下载某一类资产：

```bash
VERSION=v0.1.0 bash scripts/download_text_assets.sh
VERSION=v0.1.0 bash scripts/download_image_assets.sh
VERSION=v0.1.0 bash scripts/download_video_assets.sh
```

默认下载地址：

```text
https://github.com/Wangych297/MM-ContentGuard/releases/download/<VERSION>
```

如果 Release 放在别的位置，可以覆盖：

```bash
RELEASE_DOWNLOAD_BASE_URL=https://example.com/assets \
VERSION=v0.1.0 \
bash scripts/download_runtime_assets.sh
```

## 2. 必需资产

### 文本服务

```text
services/text/runs/social_binary_v3_macbert_binary_bs128_no_online_aug/
services/text/runs/social_intent_binary_v1_from_general/
services/text/runs/reason_category_v1_macbert/
services/text/models/hfl/chinese-macbert-base/
services/text/data/raw/ast/
```

说明：

```text
通用二分类 BERT：social_binary_v3_macbert_binary_bs128_no_online_aug
口语二分类 BERT：social_intent_binary_v1_from_general
原因四分类 BERT：reason_category_v1_macbert
MacBERT backbone：models/hfl/chinese-macbert-base
字音字形索引：data/raw/ast
```

### 图片服务

```text
services/image/checkpoints/image_model_dinov3_large/best.pt
services/image/checkpoints/semantic_model_focal_binary/best.pt
services/image/.cache/manual_weights/dinov3_large/model.safetensors
services/image/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/
services/image/.EasyOCR/model/
```

### 视频服务

```text
services/video/outputs/checkpoints/multi_modal_cross_attention_model.pth
services/video/pretrained/googlevit-base-patch16-224-in21k/
services/video/pretrained/bert_chinese/
services/video/pretrained/FER-VT/
services/video/outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth
services/video/outputs/cache/whisper/large-v3.pt
```

## 3. Release 文件名

当前脚本期望 Release 中包含这些文件：

```text
text-runtime-assets-v0.1.0.sha256
text-runtime-core-v0.1.0.tar.gz

image-runtime-assets-v0.1.0.sha256
image-runtime-parts-v0.1.0.txt
image-runtime-core-v0.1.0.tar.gz.sha256
image-runtime-core-v0.1.0.tar.gz.part-*

runtime-assets-v0.1.0.sha256
video-pclmm-runtime-core-v0.1.0.tar.gz
whisper-large-v3.pt.part-aa
whisper-large-v3.pt.part-ab
whisper-large-v3.pt.sha256
```

其中图片运行资产和视频 Whisper 权重因为体积较大，按 GitHub Release 单文件限制切成多个 part。

可选数据集和 demo 素材：

```text
dataset-assets-v0.1.0.sha256
dataset-assets-v0.1.0.tar.gz
demo-assets-v0.1.0.sha256
demo-assets-v0.1.0.tar.gz
```

数据集和 demo 素材不影响默认推理流程。

## 4. 检查资产

下载完成后执行：

```bash
bash scripts/check_runtime_assets.sh
```

全部通过时会看到：

```text
Runtime assets are ready.
```

如果出现 `[miss]`，说明对应模型文件还没有下载或解压到正确位置。

## 5. 复现边界

公开仓库复现目标是：

```text
clone 仓库
安装环境
下载运行资产
启动文本/图片/视频审核服务
启动 gateway 和前端
跑通 demo
```

不要求助教或老师重新训练三个模型。
