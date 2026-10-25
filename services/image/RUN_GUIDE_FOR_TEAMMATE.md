# 风险图片检测系统整包运行指南

这个文件夹是一个干净的整包副本，已经包含图片项目代码和最终展示需要的模型资产。原始 Git 仓库没有被修改或删除。

建议把整个文件夹复制到本机一个路径较短的位置，例如：

```text
D:\risk_image_full_package_for_teammate_20260607
```

## 包里已经包含

```text
checkpoints/image_model_dinov3_large/best.pt
checkpoints/semantic_model_focal_binary/best.pt
.cache/manual_weights/dinov3_large/model.safetensors
.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/
.EasyOCR/model/craft_mlt_25k.pth
.EasyOCR/model/zh_sim_g2.pth
datasets/simas/
datasets/safe_extra/pass_1500/images/
datasets/ToxiCN_MM/data/meme/
datasets/ToxiCN_MM/data/*_discription*.json
data/processed/
demo_samples/
```

没有包含旧实验权重、`last.pt`、训练日志、`.pid`、`.git`、Python 缓存、ToxiCN 下载压缩包、PASS 下载日志/下载进程文件/URL 大列表。

## 数据集位置

本整包已经内置训练数据集，训练和评估默认走这些相对路径：

```text
datasets/
  simas/
  safe_extra/pass_1500/images/
  ToxiCN_MM/
    data/meme/
    data/train_data_discription_2.0.json
    data/test_data_discription_2.0.json
```

配置文件已经改成整包内路径：

```text
configs/image_classifier_dinov3_large.yaml
  simas_root: datasets/simas
  pass_safe_root: datasets/safe_extra/pass_1500/images

configs/semantic_meme_focal_binary.yaml
  source_root: datasets/ToxiCN_MM
  image_dir: datasets/ToxiCN_MM/data/meme
  name: .cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/snapshots/...
  processor_name: .cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/snapshots/...
```

`data/processed/` 里的图片路径也已经改成相对路径，不会指向原电脑的 `C:\Users\86155\...`。

ChineseCLIP 的本地 snapshot 已经补好 `processor_config.json`、`tokenizer_config.json`、`tokenizer.json`，可以离线加载。

## 额外依赖

图片项目会调用文本 API：

```text
http://127.0.0.1:8000/text/check
```

所以请先启动你已经拿到的 `risk-text-detection-system`。这个整包不再复制文本项目。

第二层语义检测会调用大模型生成图片描述，需要私下配置 API Key：

```powershell
$env:VLM_BASE_URL="https://api.vveai.com/v1"
$env:VLM_MODEL="gemini-3.1-flash-lite"
$env:VLM_API_KEY="填私下给你的key"
```

## 安装环境

建议 Python 3.10 或 3.11。进入本目录：

```powershell
cd D:\risk_image_full_package_for_teammate_20260607
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果有 NVIDIA 显卡，先装 CUDA 版 PyTorch：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[ocr]"
```

如果只用 CPU：

```powershell
pip install -e ".[ocr]"
```

## 启动文本 API

另开一个终端，进入文本项目：

```powershell
cd D:\risk-text-detection-system
$env:PYTHONPATH="src"
$env:RISK_TEXT_DEVICE="cuda"
python -m risk_text_detection.api.app
```

如果没有显卡，把 `cuda` 改成 `cpu`。

## 启动图片 API

回到这个整包目录：

```powershell
cd D:\risk_image_full_package_for_teammate_20260607
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"

$env:HF_HOME="$PWD\.cache\huggingface"
$env:EASYOCR_MODULE_PATH="$PWD\.EasyOCR"

$env:VLM_BASE_URL="https://api.vveai.com/v1"
$env:VLM_MODEL="gemini-3.1-flash-lite"
$env:VLM_API_KEY="填私下给你的key"

python -m api.app
```

图片 API 默认地址：

```text
http://127.0.0.1:8010
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
```

## 测试一张 demo 图片

```powershell
$body = @{
  image_path = "$PWD\demo_samples\labeled_api_demo_20260607\07_TRUE_harmful_targeted_offense_ToxiCN_MM_7340.jpg"
  use_ocr = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8010/image/check -Body $body -ContentType "application/json"
```

## 重新生成数据清单

如果需要从内置数据集重新生成第一层图片分类 split：

```powershell
cd D:\risk_image_full_package_for_teammate_20260607
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
python -m data.build_manifest --config configs/image_classifier_dinov3_large.yaml
```

如果需要重新生成第二层 ToxiCN_MM split：

```powershell
cd D:\risk_image_full_package_for_teammate_20260607
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
python -m semantic_model.prepare_data --config configs/semantic_meme_focal_binary.yaml
```

## 常见问题

1. 找不到 `best.pt`  
   确认你是在这个整包目录下启动的，不是在另一个空仓库目录里启动。

2. 找不到 `model.safetensors`  
   确认 `.cache/manual_weights/dinov3_large/model.safetensors` 存在。

3. ChineseCLIP 还在联网下载  
   确认启动图片 API 前设置了：

```powershell
$env:HF_HOME="$PWD\.cache\huggingface"
```

4. EasyOCR 还在下载模型  
   确认启动图片 API 前设置了：

```powershell
$env:EASYOCR_MODULE_PATH="$PWD\.EasyOCR"
```

5. 图片 API 里文本检测失败  
   先确认文本 API 已经在 `8000` 端口启动。

6. 第二层描述为空或失败  
   检查 `VLM_BASE_URL`、`VLM_MODEL`、`VLM_API_KEY`。

## 当前选用模型

第一层图片模型：

```text
checkpoints/image_model_dinov3_large/best.pt
```

第二层语义模型：

```text
checkpoints/semantic_model_focal_binary/best.pt
```

不用的旧模型没有放进这个包。
