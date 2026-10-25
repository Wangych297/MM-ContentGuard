# Release 资产清单

本项目公开复现采用“代码在 Git、权重在 Release、数据集可选”的方式。

## 1. 必需上传项

老师和助教只需要这些资产即可跑通完整 demo。

### 文本审核运行资产

```text
text-runtime-assets-v0.1.0.sha256
text-runtime-core-v0.1.0.tar.gz
```

内容包括：

```text
services/text/runs/social_binary_v3_macbert_binary_bs128_no_online_aug/
services/text/runs/social_intent_binary_v1_from_general/
services/text/runs/reason_category_v1_macbert/
services/text/models/hfl/chinese-macbert-base/
services/text/data/raw/ast/
```

### 图片审核运行资产

```text
image-runtime-assets-v0.1.0.sha256
image-runtime-parts-v0.1.0.txt
image-runtime-core-v0.1.0.tar.gz.sha256
image-runtime-core-v0.1.0.tar.gz.part-*
```

图片资产较大，因此按 part 分片上传。

内容包括：

```text
services/image/checkpoints/image_model_dinov3_large/best.pt
services/image/checkpoints/semantic_model_focal_binary/best.pt
services/image/.cache/manual_weights/dinov3_large/model.safetensors
services/image/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/
services/image/.EasyOCR/model/
```

### 视频审核运行资产

```text
runtime-assets-v0.1.0.sha256
video-pclmm-runtime-core-v0.1.0.tar.gz
whisper-large-v3.pt.part-aa
whisper-large-v3.pt.part-ab
whisper-large-v3.pt.sha256
```

内容包括：

```text
services/video/outputs/checkpoints/multi_modal_cross_attention_model.pth
services/video/pretrained/googlevit-base-patch16-224-in21k/
services/video/pretrained/bert_chinese/
services/video/pretrained/FER-VT/
services/video/outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth
services/video/outputs/cache/whisper/large-v3.pt
```

## 2. 可选上传项

这些文件不影响老师运行 demo。

### 可选数据集

```text
dataset-assets-v0.1.0.sha256
dataset-assets-v0.1.0.tar.gz
```

用途：

```text
查看数据来源
复现实验
后续继续训练
```

不用于默认推理流程。

### 可选 demo 媒体

```text
demo-assets-v0.1.0.sha256
demo-assets-v0.1.0.tar.gz
```

用途：

```text
提供较大的视频演示素材
```

文本和图片 demo 已经有小样例随代码仓库提供；视频素材较大，因此放在可选资产中。

## 3. 打包命令

只打包必需运行权重：

```bash
VERSION=v0.1.0 bash scripts/package_runtime_assets.sh
```

同时打包可选数据集和 demo 素材：

```bash
VERSION=v0.1.0 PACKAGE_DATASETS=true PACKAGE_DEMO=true bash scripts/package_runtime_assets.sh
```

输出目录：

```text
release_assets/
```

## 4. 下载命令

老师运行项目时只需要：

```bash
VERSION=v0.1.0 bash scripts/download_runtime_assets.sh
```

可选数据集：

```bash
VERSION=v0.1.0 bash scripts/download_dataset_assets.sh
```

可选 demo 素材：

```bash
VERSION=v0.1.0 bash scripts/download_demo_assets.sh
```

