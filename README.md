<div align="center">

# MM-ContentGuard

### Understand the content. Explain the risk. Review before sending.

A multimodal content moderation system for text, images, memes, and video in social-media messaging.

**Text · Image · Video** &nbsp; / &nbsp; **3 specialist services** &nbsp; / &nbsp; **1 unified moderation gateway**

[Overview](#project-overview) · [Architecture](#how-it-works) · [Case Studies](#case-studies) · [Application Demo](#see-it-in-action) · [Results](#evaluation-results) · [Quick Start](#quick-start)

</div>

## Project Overview

**Moderation goes beyond matching a list of blocked words.** Harmful messages can use phonetic substitutions, implicit conversational attacks, text inside images, or image-text combinations that change the meaning of a meme. Video adds speech, sound, motion, and facial expression.

MM-ContentGuard brings specialist models into a single **review-before-send workflow**. A React chat interface sends message text and attachments to a FastAPI gateway, which routes each item to its moderation service and returns structured decisions and reasons. A message is sent only after all included items pass review.

- **Robust text understanding:** three MacBERT experts combine contextual semantics with phonetic and glyph features to handle general risk, conversational boundaries, and rejection reasons.
- **Two-stage image understanding:** visual classification, OCR/text checks, and rules address explicit risk; a ChineseCLIP semantic stage uses image-text evidence and auxiliary descriptions to assess memes and implicit meaning.
- **Four-modality video analysis:** fuses visual, audio, speech-text, and facial-expression features through cross-attention to detect patronizing and condescending language in videos.
- **A working application:** modality adapters, consistent result schemas, review records, and fail-closed handling connect separately evaluated models to one messaging workflow.

## How It Works

![System architecture: message intake, text/image/video review branches, and gateway aggregation leading to send or hold decisions](doc/assets/system-overview.svg)

| Route | Core design | What it handles |
|---|---|---|
| **Semantic text service** | Three MacBERT experts, phonetic/glyph fusion, tiered decision policy | General harmful text, implicit conversational attacks, and four user-facing reason categories |
| **Image service** | DINOv3 + OCR + text checks + rules, followed by ChineseCLIP with description augmentation | Explicit visual risk, text embedded in images, and harmful image-text combinations |
| **Video service** | ViT frames, MFCC audio, Whisper/BERT speech text, and FER-VT facial features with cross-attention fusion | Video language-risk detection using four complementary modalities |

**Routing and decisions.** The first image stage can stop on explicit risk; images that pass continue to the semantic stage. The online text service uses a tiered policy for hard risk, review, and safe outcomes. At the gateway, all included content must pass. Timeouts, unavailable services, and parsing failures keep a message from being sent without review.

The gateway connects the MacBERT text, two-stage image, and four-modality video services through a shared review interface.

## Case Studies

**What makes the content risky?** These examples show the input, its meaning in context, and how the system handles it. English translations accompany the original Chinese content.

![Three case studies: a gambling solicitation disguised with symbols and character variants; a floor-plan meme whose caption creates sexual innuendo; a video sketch mocking a person's shaking behavior](doc/assets/technical-evidence.png)

<details>
<summary><strong>Read the examples and their English explanations</strong></summary>

### Text — obfuscated gambling solicitation

**Original, contact omitted:** `揭秘彩票漏洞专业套%俐零风险每天纷红100倒500来打狗庄蒖㊣菂技术联 繋……`

**Translation:** “Exploit a lottery loophole for risk-free profit. Earn 100–500 daily, beat the bookmaker—real techniques. Get in touch…”

**What is disguised:** `套%俐 → 套利` (profit), `纷红 → 分红` (returns), `倒 → 到` (to), `蒖㊣菂 → 真正的` (genuine), and `繋 → 系` in `联系` (contact). The message combines a symbol insertion, sound-alike substitutions, and unusual character forms.

**Why it is risky:** It advertises supposedly risk-free lottery profits and daily returns, uses slang for beating the bookmaker, and directs readers to make contact. The dataset contains both this altered text and its normalized counterpart, with a spam label. Phonetic/glyph features address the spelling changes; semantic experts assess the underlying solicitation. No new per-example prediction score is claimed.

### Images — a double meaning created by image and caption

**Caption translation:** “Most people cannot find the stairs and passageways on their first visit.”

**Why it is risky:** The diagram resembles female genital anatomy. Combined with that shape, “first time” and difficulty finding an entrance become a sexual double entendre. The literal floor-plan text alone does not convey that meaning.

**Selected example:** sample 45. The previously tested outcome, confirmed by the project owner, is **image: safe · text: safe · combined: harmful**. No per-example probability is assigned here.

### Video — ridicule conveyed through performance

**Subtitle translation:** “Hey, that guy keeps shaking.” The frame also contains “mocking laughter,” while the sketch's caption identifies a character as a psychiatric patient.

**Why it is risky:** The person's shaking is treated as a punchline. The caption, mocking delivery, and portrayal together frame the behavior as something to ridicule, reinforcing stigma toward people with mental-health conditions. Shaking or a mental-health condition alone is not harmful content.

**Recorded outcome:** The application recording later shows that the video fails moderation and is not sent. The service combines visual, audio, speech-text, and facial features; this example explains the content rather than assigning a separate score to each modality.

</details>

[Case notes and original material →](doc/SHOWCASE.md)

## See It in Action

**Application integration demo.** The recording shows ordinary messages being sent, risky text receiving rejection reasons, image attachments receiving review outcomes, and a video attachment being uploaded and previewed.

[![Recorded chat interface showing accepted messages and rejected promotional text with review reasons](doc/assets/demo-text.webp)](doc/assets/demo.mp4)

*Text moderation in the recorded application: accepted messages remain visible alongside rejected test inputs and their reasons. Click the image to open the complete 58-second demo.*

<table>
<tr>
<td width="50%"><a href="doc/assets/demo-image.webp"><img src="doc/assets/demo-image.webp" alt="Recorded image-moderation view with approved images and an image rejected as regulated content" width="100%"></a></td>
<td width="50%"><a href="doc/assets/demo-video.webp"><img src="doc/assets/demo-video.webp" alt="Recorded application showing an uploaded video attachment with its preview" width="100%"></a></td>
</tr>
<tr>
<td><strong>Image review.</strong> Approved images and a rejected image are shown with their application feedback.</td>
<td><strong>Video workflow.</strong> A video attachment is uploaded and opened in the in-app preview.</td>
</tr>
</table>

[Watch or download the complete demo →](doc/assets/demo.mp4)

*Frames extracted from the provided recording preserve the original Chinese interface and test inputs. Evaluation results are reported separately below.*

## Evaluation Results

![Separate result panels for text risk recall, explicit-image classification, meme classification, and video ROC AUC, with dataset sizes](doc/assets/evaluation.png)

| Evaluated component | Test scope | Reported result |
|---|---|---|
| **Tiered MacBERT text system** | 7,800 regular-test messages | **96.63% risk recall**, 94.74% accuracy, 94.59% macro F1 |
| **Text social-boundary evaluation** | 400 manually constructed sealed examples | **99.00% accuracy, macro F1, and risk recall** |
| **DINOv3 explicit-image classifier** | 975 SIMAS + PASS test images; 7 classes | **97.23% accuracy**, 96.12% macro F1 |
| **ChineseCLIP meme semantic classifier** | 2,400 ToxiCN_MM test examples | **83.41% binary accuracy**, 80.29% binary macro F1 |
| **Four-modality video classifier** | 31 feature-complete test clips | **0.8846 ROC AUC**; at threshold 0.32, 87.10% accuracy and 71.30% macro F1 |

**Different tasks, separate evidence.** These results describe each component on its own dataset. They are not a pooled overall accuracy or an end-to-end gateway score. The sealed text set is constructed, and the video results use a small feature-complete subset with a threshold selected from test-set comparisons. [Protocols, trade-offs, and full results →](doc/EVALUATION.md)

### Video Inference Engineering

![Reported video processing times: about 65 seconds for serial subprocess extraction, 47 seconds for parallel subprocess extraction, and 7.4 seconds for resident parallel extraction](doc/assets/video-latency.png)

The report records an approximately **88.6% reduction in reference processing time**, from about 65 to 7.4 seconds for an approximately ten-second clip, after keeping models resident and extracting features in parallel. Feature caching supports repeated requests. These are configuration-specific reference timings, not a standardized throughput benchmark.


## Quick Start

The supplied scripts target **Linux/Bash**, with **Python 3.10**, **FFmpeg**, **Node.js/npm**, and the documented PyTorch/CUDA environment. Trained checkpoints and model caches are downloaded separately; no retraining is required for the demo.

```bash
git clone https://github.com/Wangych297/MM-ContentGuard.git
cd MM-ContentGuard
```

Follow the [environment and asset setup](doc/RUNNING.md), then start the services:

```bash
cp .env.example .env
VERSION=v0.1.0 bash scripts/download_runtime_assets.sh
bash scripts/check_runtime_assets.sh
bash scripts/start_all.sh
bash scripts/check_services.sh
```

Open **http://127.0.0.1:5173/**. Configure GPU devices and any VLM description endpoint in `.env` for your machine.

[Complete installation, service ports, API examples, training references, and shutdown instructions →](doc/RUNNING.md)

<details>
<summary><strong>Repository map</strong></summary>

| Directory | Responsibility |
|---|---|
| [`frontend/web/`](frontend/web/) | React/Vite chat interface and moderation feedback |
| [`backend/gateway/`](backend/gateway/) | FastAPI gateway, adapters, message state, and review records |
| [`services/text/`](services/text/) | MacBERT experts, training, and text moderation API |
| [`services/image/`](services/image/) | Explicit-image and semantic-meme models, OCR, and image API |
| [`services/video/`](services/video/) | Four-modality fusion, feature extraction, and video API |
| [`scripts/`](scripts/) | Runtime asset preparation and service lifecycle |
| [`doc/`](doc/) | Project report, architecture, evaluation, and reproduction guides |

</details>

## Scope

This is a research and engineering prototype for Chinese-language social-media content. Video evaluation focuses on patronizing and condescending language; it does not cover every category of harmful video. A moderation decision is a model/policy outcome; deployment thresholds and review practices need to match the intended use.

The project builds on MacBERT, DINOv3, ChineseCLIP, and other model dependencies. Dataset and third-party asset use remains subject to their original licenses and terms.
