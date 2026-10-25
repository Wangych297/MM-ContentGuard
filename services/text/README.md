# 基于 MacBERT 与字音字形鲁棒性的风险文本检测系统

This project organizes Chinese risk-text datasets for a staged content-safety detector:

- spam and ad detection
- fraud, gambling, sexual-service, and illegal-service categories
- toxic, abusive, biased, and sexual-harm categories
- robustness against phonetic, glyph, and mixed-form camouflage

The default data profile is `quick20k`, which builds about 20k training records for laptop experiments. The `lab_full` profile keeps a larger option for later GPU runs.

```powershell
python -m risk_text_detection.data.build_unified --root . --profile quick20k
```

Outputs are written to:

```text
data/processed/<profile>/
  train.jsonl
  dev.jsonl
  test.jsonl
  manifest.json
```

Training entrypoint:

```powershell
$env:PYTHONPATH="src"
python -m risk_text_detection.training.train `
  --train-file data/processed/quick20k/train.jsonl `
  --dev-file data/processed/quick20k/dev.jsonl `
  --output-dir runs/quick20k_macbert_pinyin `
  --model-name models/hfl/chinese-macbert-base `
  --batch-size 8 `
  --epochs 3 `
  --max-length 128 `
  --use-pinyin `
  --fp16
```

Optional character-level robustness augmentation uses `data/raw/ast/hanzi.txt` to sample same-sound or similar-shape character variants during training only:

```powershell
$env:PYTHONPATH="src"
python -m risk_text_detection.training.train `
  --train-file data/processed/quick20k/train.jsonl `
  --dev-file data/processed/quick20k/dev.jsonl `
  --output-dir runs/quick20k_macbert_pinyin_aug `
  --model-name models/hfl/chinese-macbert-base `
  --batch-size 8 `
  --epochs 3 `
  --max-length 128 `
  --use-pinyin `
  --char-augment-mode mixed `
  --char-augment-prob 0.15 `
  --fp16
```

Token-level phonetic-glyph fusion can be enabled with `--use-phonetic-glyph`. This adds per-token pronunciation, glyph, and similar-character neighbor features before the multi-task heads:

```powershell
$env:PYTHONPATH="src"
python -m risk_text_detection.training.train `
  --train-file data/processed/quick20k/train.jsonl `
  --dev-file data/processed/quick20k/dev.jsonl `
  --output-dir runs/quick20k_macbert_phonetic_glyph `
  --model-name models/hfl/chinese-macbert-base `
  --batch-size 8 `
  --epochs 3 `
  --max-length 128 `
  --use-phonetic-glyph `
  --phonetic-glyph-neighbor-top-k 4 `
  --char-augment-mode mixed `
  --char-augment-prob 0.15 `
  --fp16
```

For a short code-path check, add `--limit-train 32 --limit-dev 32 --max-steps 1`.

## Current deployed checkpoints

The current text backend uses a tiered policy:

```text
hard_risk = general >= 0.5
review    = general < 0.5 and social >= 0.5
safe      = otherwise
```

The deployed checkpoints are:

```text
runs/social_binary_v3_macbert_binary_bs128_no_online_aug
runs/social_intent_binary_v1_from_general
runs/reason_category_v1_macbert
```

The packaged MacBERT backbone is stored locally at:

```text
models/hfl/chinese-macbert-base
```

The reason classifier currently returns four rejection categories:

```text
financial_illegal_promotion
gambling
sexual_content
abuse_hate
```

## Response shape

The HTTP predictor returns:

```json
{
  "risk_binary": "suspicious",
  "risk_level": "suspicious",
  "risk_probability": 0.99,
  "primary_category": "sexual_content",
  "decision": "review",
  "reason": {
    "category": "sexual_content",
    "label": "色情/性相关风险",
    "confidence": 0.99
  },
  "scores": {
    "general": 0.15,
    "social": 0.99
  }
}
```

Default thresholds:

```text
general >= 0.5                       -> risk
general < 0.5 and social >= 0.5      -> suspicious
general < 0.5 and social < 0.5       -> normal
```

## HTTP API

Start the service:

```powershell
$env:PYTHONPATH="src"
python -m risk_text_detection.api.app
```

Endpoint:

```text
POST /text/check
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/text/check" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"加我微信，稳赚不赔\"}"
```

Health check:

```text
GET /health
```

## Report evaluation

Current deployment metrics were produced in the sibling `risk-text-detection` project. The most relevant reports are:

```text
risk-text-detection/reports/tiered_review_general_v3_social_v1_g050_s050_eval/
risk-text-detection/reports/reason_category_v1_macbert_test_eval/
```

This packaged service keeps the inference code and runtime API only.
