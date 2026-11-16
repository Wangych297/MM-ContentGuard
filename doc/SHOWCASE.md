# Technical Showcase: Evidence Details

[Back to the project](../README.md#technical-highlights)

## Text: the role of the review expert

The regular test contains 7,800 messages: 4,650 safe and 3,150 risk. General-model hard-risk detection reaches 94.73% risk recall. Adding the social-boundary review expert recovers 60 risk messages and sends 181 safe messages into review. Combined risk recall is 96.63%; accuracy changes from 96.29% to 94.74%. Review is a non-safe outcome, not an automatic final risk label.

The 400-example sealed set is manually constructed, evaluation-only, and covers conversational boundaries and character obfuscation. Combined risk recall is 99.00%. Its records provide aggregate metrics, so the showcase does not assign recorded predictions to invented example sentences. The results do not establish an isolated gain from phonetic or glyph features.

Full tables: [text evaluation records](bert_text_review/text_evaluation_sets_and_results.md).

## Images: a real second-stage decision

![Original recorded meme](../services/image/demo_samples/api_flow_samples/08_ToxiCN_11273_general_final_risk_stage2_general_offense.jpg)

Caption gloss: “I'm telling you, you'll get dissed for that.” The gloss aids English readers; the model received the original Chinese material. This example is labeled `general_offense` in the sample summary. Contextual interpretation may be ambiguous; the displayed outputs are the recorded model decisions.

The DINOv3 visual classifier returns `safe`, the OCR text-risk check returns `safe`, and no first-stage rule fires. The semantic stage returns `harmful`, `general_offense`, and harmful probability 0.8603323698043823. The final route is `blocked_by_semantic_model` with final label `semantic_violation`.

The semantic stage combines image/text representations with auxiliary descriptions. This trace establishes the routing and decision change. It is not a controlled comparison of modalities or a demonstration of causal explanation.

The original filename contains historical result wording. The outputs above follow the actual JSON fields, not the filename.

Recorded evidence: [API summary](../services/image/demo_samples/api_flow_samples/latest_11_api_summary.csv) and [full API records](../services/image/demo_samples/api_flow_samples/latest_11_api_results.json), example 08 (eighth record). The showcase rounds the harmful probability to four decimal places. No input image, prediction, or threshold was changed.

## Video: architecture and measured results

Video processing combines ViT frame embeddings, MFCC audio features, Whisper/BERT speech-text features, and FER-VT facial features using cross-attention. These are feature streams; the showcase does not present per-stream predictions or attention heatmaps.

The experiment uses 31 feature-complete test clips (26 normal, five risk). At threshold 0.32, accuracy is 87.10%, macro F1 is 71.30%, and ROC AUC is 0.8846. The threshold was recommended from the test-set comparison, not validated on a separate untouched set. This is a small-subset evaluation of patronizing and condescending language.

The report records approximately 7.4 seconds to process an approximately ten-second clip using resident models and parallel feature extraction. Timing is configuration-specific, not a standardized throughput result. The supplied application video shows upload and preview; it does not provide a complete per-clip model trace.

Details: [evaluation protocols](EVALUATION.md), [video metrics](video_review/video_experiment_metrics.md), and [video architecture](video_review/video_model_architecture.md).
