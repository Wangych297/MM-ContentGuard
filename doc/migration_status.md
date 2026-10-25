# 服务迁移状态

## 当前结论

三个已有审核项目已经按服务目录迁入：

```text
services/text/
services/image/
services/video/
```

本轮已经完成结构验证，并在统一 conda 环境中完成三类底层 API、统一 gateway 和前端开发服务的真实启动验证。

## 1. 文字服务

来源：

```text
risk-text-detection-system/
```

迁入位置：

```text
services/text/
```

已迁入：

- `src/`
- `tests/`
- `configs/`
- `docs/`
- `README.md`
- `pyproject.toml`
- 当前推理 checkpoint：`runs/ast_replay_continue_from_best_20260607_100730/`
- MacBERT backbone：`models/hfl/chinese-macbert-base/`
- 字音字形索引：`data/raw/ast/`

验证结果：

```text
services/text/src/risk_text_detection/api/app.py 存在
services/text/runs/ast_replay_continue_from_best_20260607_100730/pytorch_model.bin 存在
services/text/models/hfl/chinese-macbert-base/pytorch_model.bin 存在
services/text/data/raw/ast/hanzi.txt 存在
源码 compileall 通过
基础 package import 通过
```

真实 API 验证：

```text
GET  http://127.0.0.1:8000/health 通过
POST http://127.0.0.1:8000/text/check 通过
风险文本样例返回 risk_level=risk, primary_category=spam_general
```

## 2. 图片服务

来源：

```text
risk_image_full_package_for_teammate_20260607/
```

迁入位置：

```text
services/image/
```

已迁入：

- `src/`
- `configs/`
- `scripts/`
- `demo_samples/`
- `data/`
- `README.md`
- `RUN_GUIDE_FOR_TEAMMATE.md`
- `ARTIFACTS.md`
- `pyproject.toml`
- 第一层图片模型：`checkpoints/image_model_dinov3_large/best.pt`
- 第二层语义模型：`checkpoints/semantic_model_focal_binary/best.pt`
- DINOv3 权重：`.cache/manual_weights/dinov3_large/model.safetensors`
- ChineseCLIP 缓存：`.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14/`
- EasyOCR 权重：`.EasyOCR/model/`

未迁入：

- 完整训练数据集 `datasets/`

说明：图片 API 推理不需要完整训练数据集；后续如果要恢复训练，再单独作为数据资产处理。

验证结果：

```text
services/image/src/api/app.py 存在
services/image/checkpoints/image_model_dinov3_large/best.pt 存在
services/image/checkpoints/semantic_model_focal_binary/best.pt 存在
services/image/.cache/manual_weights/dinov3_large/model.safetensors 存在
services/image/.EasyOCR/model 存在
ChineseCLIP snapshot 目录存在
/health 和 /image/check 路由静态确认
```

真实 API 验证：

```text
GET  http://127.0.0.1:8010/health 通过
POST http://127.0.0.1:8010/image/check 通过
demo weapon 图片返回 risk_level=risk, risk_category=weapon, reasons=image_model:weapon
```

兼容性修复：

```text
services/image/src/semantic_model/dataset.py
```

当前 transformers 版本下，`ChineseCLIPProcessor.from_pretrained` 读取本地 `processor_config.json` 时会出现 `image_processor` 参数重复。已增加 fallback：分别加载 `ChineseCLIPImageProcessor` 和 tokenizer，再手动组装 `ChineseCLIPProcessor`。

## 3. 视频服务

来源：

```text
video-pclmm-project/
```

迁入位置：

```text
services/video/
```

已迁入：

- `src/`
- `scripts/`
- `doc/`
- `docs/`
- `configs/`
- `third_party/`
- `README.md`
- `requirements.txt`
- 融合模型 checkpoint：`outputs/checkpoints/multi_modal_cross_attention_model.pth`
- ViT：`pretrained/googlevit-base-patch16-224-in21k/`
- BERT：`pretrained/bert_chinese/`
- FER-VT：`pretrained/FER-VT/`
- ResNet34 人脸检测权重：`outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth`
- Whisper large-v3：`outputs/cache/whisper/large-v3.pt`

未迁入：

- 原始视频数据：`data/raw/`
- 离线特征：`features/`
- API 运行缓存：`outputs/api/`
- GitHub Release 分片包：`outputs/release_assets/`

验证结果：

```text
services/video/src/api/app.py 存在
services/video/scripts/run_api.sh 存在
services/video/outputs/checkpoints/multi_modal_cross_attention_model.pth 存在
services/video/pretrained/googlevit-base-patch16-224-in21k/model.safetensors 存在
services/video/pretrained/bert_chinese/model.safetensors 存在
services/video/pretrained/FER-VT/model/FERVT.py 存在
services/video/outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth 存在
services/video/outputs/cache/whisper/large-v3.pt 存在
视频服务脚本 bash -n 通过
/health 和 /predict 路由静态确认
```

真实 API 验证：

```text
GET  http://127.0.0.1:8020/health 通过
POST http://127.0.0.1:8020/predict 通过
10 秒 disabled90.mp4 样例端到端推理约 7.6 秒
```

验证时使用：

```text
PCLMM_API_CUDA_VISIBLE_DEVICES=6,7
feature_backend=resident
extraction_mode=parallel
```

兼容性修复：

```text
services/video/pretrained/FER-VT/model/__init__.py
```

统一环境中图片项目存在顶层包 `model`，会和 FER-VT 的 `model.FERVT` 导入冲突。为 FER-VT 的 `model/` 增加 `__init__.py` 后，可优先导入视频服务自己的 FER-VT 包。

## 4. 后续验证顺序

建议后续统一使用 conda 环境按这个顺序验证：

1. `services/text`：启动 `/text/check`。
2. `services/image`：指向文字 API，启动 `/image/check`。
3. `services/video`：用 `PORT=8020` 启动 `/predict`。
4. `backend/gateway`：adapter、健康检查、统一消息发送接口已实现并验证。
5. `frontend/web`：微信式聊天 MVP 已实现并通过 Vite 构建验证。

## 4. Gateway 验证

真实 API 验证：

```text
GET http://127.0.0.1:8080/api/health 通过
POST http://127.0.0.1:8080/api/conversations/demo/messages 通过
GET http://127.0.0.1:8080/api/conversations/demo/messages 通过
```

返回中三个下游服务均为：

```text
text.ok=true
image.ok=true
video.ok=true
```

端到端消息验证：

```text
安全文本：approved，返回“审核通过”
风险文本：rejected，返回“发送失败：广告 / 垃圾信息”
武器图片：rejected，返回“发送失败：武器相关内容”
视频样例：approved，成功调用 video /predict
```

前端代理验证：

```text
GET  http://127.0.0.1:5173/ 通过
GET  http://127.0.0.1:5173/api/health 通过
POST http://127.0.0.1:5173/api/conversations/demo/messages 通过
```

## 5. 前端 MVP

已实现：

- 左侧窄导航和会话列表。
- 中间聊天消息区。
- 文本输入、图片上传、视频上传。
- 本地 pending 气泡。
- 审核通过后显示已发送。
- 审核失败后显示中文原因。
- 默认前端入口使用生产构建后的静态服务。
- `/api` 和 `/media` 通过静态前端服务代理到 gateway，避免 Vite dev server / HMR 在远程端口转发下卡住。

构建验证：

```text
cd frontend/web
npm install
npm run build
```

结果：

```text
TypeScript 编译通过
Vite production build 通过
```

## 6. 启停脚本

已实现：

```text
scripts/start_text.sh
scripts/start_image.sh
scripts/start_video.sh
scripts/start_gateway.sh
scripts/start_frontend.sh
scripts/start_all.sh
scripts/stop_all.sh
scripts/check_services.sh
```

验证结果：

```text
bash -n scripts/*.sh 通过
bash scripts/check_services.sh 五项通过
```

## 7. 统一 conda 环境

已验证环境：

```text
/data4/songxinshuai/conda/envs/multimodal-risk-chat
```

核心验证：

```text
torch==2.2.2+cu121
torch.cuda.is_available()=true
ffmpeg 可用
fastapi / httpx / uvicorn / transformers / whisper / easyocr / timm / pypinyin 可导入
```

## 8. 当前体积

```text
services/text   约 813M
services/image  约 7.7G
services/video  约 3.9G
```

这些模型资产后续不应直接提交进 Git，应统一打包到 Release 或外部资产包。

## 9. 上传前文档和 Git 策略

已补充：

```text
doc/teammate_runbook.md
requirements-unified.txt
```

`doc/teammate_runbook.md` 覆盖：

- clone 仓库。
- 创建统一 conda 环境。
- 安装 CUDA 版 PyTorch、ffmpeg 和 pip 依赖。
- 安装 gateway、text、image 本地包。
- 准备前端依赖。
- 补齐模型运行资产。
- 配置 `.env`。
- 启动五个服务。
- 验证 gateway 和前端。
- 常见问题。

`.gitignore` 已调整为：

- 忽略顶层模型权重、checkpoint、cache、运行输出和用户上传文件。
- 保留源码目录，例如 `services/video/src/models/`。
- 保留文字服务所需的小型字音字形索引 `services/text/data/raw/ast/`。
- 保留图片服务的小型 demo 样例 `services/image/demo_samples/`。

上传前临时 Git 检查结果：

```text
预计会提交文件约 24M
大模型、checkpoint、cache、node_modules、dist、运行上传文件均被忽略
```
