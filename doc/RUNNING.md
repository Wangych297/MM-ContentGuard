# Running MM-ContentGuard

[Back to project overview](../README.md)

## Runtime Requirements

The provided launch scripts target a Linux Bash environment and use tools such as `setsid`, `curl`, and FFmpeg. The documented model setup uses Python 3.10 and CUDA-enabled PyTorch. On Windows, use a compatible Linux/WSL environment and configure GPU access for that environment. A CPU configuration is available, but image and video inference can be substantially slower.

Runtime assets are separate from source code. Pretrained caches and trained checkpoints are required to run the demo; retraining and dataset downloads are optional. The download scripts currently use the [original collaborative repository's v0.1.0 release](https://github.com/linsh63/multimodal-risk-chat/releases/tag/v0.1.0). This repository does not mirror those large assets.

## 1. Clone the Source

```bash
git clone https://github.com/Wangych297/MM-ContentGuard.git
cd MM-ContentGuard
```

## 2. Prepare the Python Environment

```bash
conda create -y -n mm-contentguard python=3.10
conda activate mm-contentguard

conda install -y -c pytorch -c nvidia \
  pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1
conda install -y -c conda-forge ffmpeg

python -m pip install "setuptools<70" wheel packaging
python -m pip install --no-build-isolation -r requirements-unified.txt
python -m pip install -e backend/gateway
python -m pip install -e services/text
python -m pip install -e 'services/image[ocr]'
```

Check the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import fastapi, httpx, uvicorn, transformers, whisper, easyocr, timm, pypinyin; print('env ok')"
ffmpeg -version
```

## 3. Prepare the Frontend

Node.js and npm are required for the React/Vite frontend.

```bash
cd frontend/web
npm ci
npm run build
cd ../..
```

## 4. Download Model Assets

```bash
VERSION=v0.1.0 bash scripts/download_runtime_assets.sh
bash scripts/check_runtime_assets.sh
```

The expected asset check ends with `Runtime assets are ready.` Individual downloads are also available:

```bash
VERSION=v0.1.0 bash scripts/download_text_assets.sh
VERSION=v0.1.0 bash scripts/download_image_assets.sh
VERSION=v0.1.0 bash scripts/download_video_assets.sh
```

See the [runtime asset inventory](runtime_assets.md) and [release checklist](release_assets_checklist.md) for archive names and extraction paths. These existing technical documents are in Chinese.

## 5. Configure Services

```bash
cp .env.example .env
```

Default devices are selected automatically. Set `TEXT_DEVICE`, `TEXT_CUDA_VISIBLE_DEVICES`, `IMAGE_CUDA_VISIBLE_DEVICES`, `VIDEO_CUDA_VISIBLE_DEVICES`, and the `PCLMM_API_*_DEVICE` settings for your machine if needed. Device indices inside a process are relative to its visible GPUs.

The second image stage can use a VLM to generate auxiliary descriptions. Configure `VLM_BASE_URL` or `VLM_ENDPOINT_URL`, `VLM_MODEL`, and `VLM_API_KEY` for your provider when using that feature. Keep keys in the local `.env` file.

The default video settings use resident models and parallel feature extraction:

```env
PCLMM_API_FEATURE_BACKEND=resident
PCLMM_API_EXTRACTION_MODE=parallel
VIDEO_THRESHOLD=0.32
```

The image semantic stage is enabled by default through `RISK_IMAGE_SEMANTIC_RUN_WHEN_STAGE1_SAFE=true`. Turning it off changes moderation coverage and should not be presented as the full two-stage system.

## 6. Start and Check the System

```bash
bash scripts/start_all.sh
bash scripts/check_services.sh
```

| Service | Default local address |
|---|---|
| Text API | http://127.0.0.1:8000 |
| Image API | http://127.0.0.1:8010 |
| Video API | http://127.0.0.1:8020 |
| Moderation gateway | http://127.0.0.1:8080 |
| Chat frontend | http://127.0.0.1:5173 |

Open **http://127.0.0.1:5173/**. The service check should report `[ok]` for text, image, video, gateway, and frontend.

For individual services:

```bash
bash scripts/start_text.sh
bash scripts/start_image.sh
bash scripts/start_video.sh
bash scripts/start_gateway.sh
bash scripts/start_frontend.sh
```

## 7. Exercise the Gateway

Send a text-only message:

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'text=Hello, shall we meet for dinner?' \
  -F 'client_message_id=local_safe_text'
```

The text models are trained for Chinese. Use representative Chinese inputs when evaluating their intended task; this English request is only a basic API connectivity example.

Send an existing image demo sample:

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'files=@services/image/demo_samples/api_flow_samples/03_SIMAS_firearms_final_risk_stage1_weapon.jpg;type=image/jpeg' \
  -F 'client_message_id=local_risk_image'
```

Download optional video demo media:

```bash
VERSION=v0.1.0 bash scripts/download_demo_assets.sh
```

Send a local MP4, substituting its actual path:

```bash
curl --noproxy '*' -X POST 'http://127.0.0.1:8080/api/conversations/demo/messages' \
  -F 'files=@/path/to/clip.mp4;type=video/mp4' \
  -F 'client_message_id=local_video'
```

The gateway accepts mixed messages as well. Every included content item must pass before the message is sent. Review records preserve modality-specific outcomes and reasons.

## 8. Optional Data and Training

```bash
VERSION=v0.1.0 bash scripts/download_dataset_assets.sh
```

Datasets are optional for inference. For training, data preparation, and model-specific evaluation, see:

- [Text service](../services/text/README.md) and [text evaluation records](bert_text_review/text_evaluation_sets_and_results.md).
- [Image service](../services/image/README.md), including separate visual and semantic model training.
- [Video service](../services/video/README.md) and [video experiment metrics](video_review/video_experiment_metrics.md).

## 9. Stop Services

```bash
bash scripts/stop_all.sh
```

## Maintainer: Package Runtime Assets

On a machine that already has the checkpoints and caches:

```bash
VERSION=v0.1.0 bash scripts/package_runtime_assets.sh
```

To include optional datasets and demo media:

```bash
VERSION=v0.1.0 PACKAGE_DATASETS=true PACKAGE_DEMO=true bash scripts/package_runtime_assets.sh
```

Generated archives are written to `release_assets/`. See the release checklist before publishing them.
