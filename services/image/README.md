# Risk Image Detection System

This is a separate image-side risk detection project. The image classifier and
the later semantic meme model are independently trainable, and the API connects
them only through the inference pipeline.

## Image Model

The current final model is the explicit DINOv3-Large image risk classifier. It
does not use a large vision-language model. It combines:

- DINOv3-Large visual classifier trained on SIMAS plus a small PASS safe subset.
- Optional OCR text path that calls the existing text detection API.
- Lightweight rules for URLs, contact details, QR codes, and sensitive words.

Image labels:

- `safe`
- `sexual`
- `violence`
- `weapon`
- `gambling`
- `drug`
- `regulated_goods`

## Data

Default local paths assume this layout:

- `../simas`
- `../safe_extra/pass_1500/images`

Build and split the manifest:

```powershell
python -m pip install -e .
python -m data.build_manifest --config configs/image_classifier_dinov3_large.yaml
python -m data.split --config configs/image_classifier_dinov3_large.yaml
```

Train and evaluate:

```powershell
python -m model.train --config configs/image_classifier_dinov3_large.yaml --resume
python -m model.evaluate --config configs/image_classifier_dinov3_large.yaml
```

The selected final checkpoint is:

```text
checkpoints/image_model_dinov3_large/best.pt
```

This checkpoint was selected from epoch 8 with validation accuracy
`0.9569230769` and macro F1 `0.9338556846`.

Large checkpoints and local data manifests are excluded from Git. See
`ARTIFACTS.md` for the local artifact paths and selected checkpoint summary.

Run training in the background:

```powershell
.\scripts\start_training_background.ps1
```

Check progress from the foreground:

```powershell
.\scripts\show_training_progress.ps1
```

Training writes:

- `checkpoints/image_model_dinov3_large/progress.json`
- `checkpoints/image_model_dinov3_large/train_stdout.log`
- `checkpoints/image_model_dinov3_large/train_stderr.log`
- `checkpoints/image_model_dinov3_large/metrics.jsonl`
- `checkpoints/image_model_dinov3_large/last.pt`
- `checkpoints/image_model_dinov3_large/best.pt`

If training stops, run the same training command with `--resume`. It resumes
from `last.pt`, which is saved after each completed epoch.

Run the API:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8010
```

## Semantic Model

The ToxiCN_MM-style meme and implicit semantics model lives under
`src/semantic_model`. It is trained separately from the explicit image
classifier.

It uses:

- `OFA-Sys/chinese-clip-vit-large-patch14` image encoder.
- Shared `OFA-Sys/chinese-clip-vit-large-patch14` text encoder for OCR text,
  text description, and meme description.
- Gated description fusion.
- Fusion features from image/text concat, product, and absolute difference.
- Three heads: `safe/harmful`, 5-class harmful type, and text/image modal
  auxiliary labels.

For the final demo pipeline, semantic inference uses
`configs/semantic_meme_focal_binary.yaml` and
`checkpoints/semantic_model_focal_binary/best.pt`. That checkpoint is the
epoch-3 binary-macro-F1 best model. The default API image config enables the
semantic model after stage-1 safe routing.

The semantic predictor can call an OpenAI-compatible vision-language API to
generate the two description fields used by the ToxiCN_MM-style model. Do not
commit keys. Provide them through the local runtime:

```powershell
$env:VLM_BASE_URL = "https://your-provider.example/v1"
$env:VLM_API_KEY = "..."
$env:VLM_MODEL = "your-vision-model"
```

If the provider exposes a full chat completions endpoint instead of a base URL,
set `VLM_ENDPOINT_URL` instead of `VLM_BASE_URL`. The expected model output is a
JSON object with `text_description` and `meme_description`.

Semantic labels:

- `safe`
- `targeted_offense`
- `sexual`
- `general_offense`
- `dispirited`

Prepare semantic data from the local ToxiCN_MM reference project:

```powershell
.\scripts\prepare_semantic_data.ps1
```

Train in the foreground:

```powershell
.\scripts\train_semantic_model.ps1
```

Train in the background and watch progress:

```powershell
.\scripts\start_semantic_training_background.ps1
.\scripts\show_semantic_training_progress.ps1
```

Evaluate a checkpoint:

```powershell
python -m semantic_model.evaluate --config configs/semantic_meme.yaml --split test
```
