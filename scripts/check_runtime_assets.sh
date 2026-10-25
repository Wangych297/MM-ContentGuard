#!/usr/bin/env bash
#
# 文件功能：检查运行时资产是否齐全。逐一验证模型权重、缓存文件和
# 数据文件是否存在于预期路径，全部存在则输出 "Runtime assets are ready."。
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

required_paths=(
  "services/text/runs/social_binary_v3_macbert_binary_bs128_no_online_aug/pytorch_model.bin"
  "services/text/runs/social_intent_binary_v1_from_general/pytorch_model.bin"
  "services/text/runs/reason_category_v1_macbert/pytorch_model.bin"
  "services/text/models/hfl/chinese-macbert-base/pytorch_model.bin"
  "services/text/data/raw/ast/hanzi.txt"
  "services/text/data/raw/ast/similarity_matrix.pkl"
  "services/image/checkpoints/image_model_dinov3_large/best.pt"
  "services/image/checkpoints/semantic_model_focal_binary/best.pt"
  "services/image/.cache/manual_weights/dinov3_large/model.safetensors"
  "services/image/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14"
  "services/image/.EasyOCR/model/craft_mlt_25k.pth"
  "services/image/.EasyOCR/model/zh_sim_g2.pth"
  "services/video/outputs/checkpoints/multi_modal_cross_attention_model.pth"
  "services/video/pretrained/googlevit-base-patch16-224-in21k/model.safetensors"
  "services/video/pretrained/bert_chinese/model.safetensors"
  "services/video/pretrained/FER-VT/model/FERVT.py"
  "services/video/outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth"
  "services/video/outputs/cache/whisper/large-v3.pt"
)

status=0
for path in "${required_paths[@]}"; do
  if [[ -e "$path" ]]; then
    printf "[ok]   %s\n" "$path"
  else
    printf "[miss] %s\n" "$path"
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  printf "\nRuntime assets are ready.\n"
else
  printf "\nSome runtime assets are missing. See doc/runtime_assets.md.\n"
fi

exit "$status"
