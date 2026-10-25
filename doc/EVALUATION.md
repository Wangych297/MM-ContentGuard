# Evaluation and Evidence

[Back to project overview](../README.md)

## What the Numbers Describe

This page summarizes the evaluation datasets, component-level results, and operating thresholds.

The system combines independently trained and evaluated components. Their datasets, labels, and metrics differ. There is **no pooled overall moderation accuracy** in the report, and component scores should not be treated as an end-to-end gateway benchmark.

## Tiered MacBERT Text Moderation

The gateway-facing text policy treats both `hard_risk` and `review` as non-safe. General-risk, social-boundary, and reason-category experts have distinct roles.

| Evaluation set | Samples | Accuracy | Macro F1 | Risk recall |
|---|---:|---:|---:|---:|
| Regular test | 7,800 | 94.74% | 94.59% | 96.63% |
| Golden v1, manually constructed | 100 | 98.00% | 98.00% | 96.00% |
| Golden v2 diagnostic | 400 | 99.50% | 99.50% | 99.50% |
| Golden v2 sealed | 400 | 99.00% | 99.00% | 99.00% |

The regular test contains 4,650 safe and 3,150 risk messages. The constructed golden sets exercise conversational boundaries and character obfuscation; they are not a random sample of all social-media traffic. The sealed set is recorded as evaluation-only, with aggregate results rather than exported per-example predictions.

The benefit has a measurable trade-off. On the regular test, hard-risk detection alone has 96.29% accuracy and 94.73% risk recall. Adding the review layer raises risk recall to 96.63% while reducing accuracy to 94.74%; it recovers 60 risk messages and sends 181 safe messages into review.

The separate four-category reason classifier reports 99.43% accuracy and 98.64% macro F1 on 3,331 risk examples. This is a component result, not the full text pipeline's accuracy.


## Two-Stage Image Moderation

| Component | Test set | Samples | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| DINOv3-Large visual classifier | SIMAS + PASS | 975 | 97.23% | 96.12% |
| ChineseCLIP semantic classifier, safe/harmful | ToxiCN_MM | 2,400 | 83.41% | 80.29% |
| ChineseCLIP semantic classifier, five categories | ToxiCN_MM | 2,400 | 79.74% | 62.39% |

The explicit-image dataset has 6,500 samples (4,550 train / 975 validation / 975 test). The semantic dataset has 12,000 samples (8,159 train / 1,441 validation / 2,400 test).

**Validation versus test:** the 95.69% accuracy and 93.39% macro F1 shown in the image README/selected-checkpoint notes are validation scores used to select epoch 8. The project report separately records final test scores of 97.23% and 96.12%; the main README uses those test results with their source and sample count.

The first stage combines visual classification, OCR/text moderation, and rules. The second stage checks images that pass the first stage, using ChineseCLIP with OCR and auxiliary descriptions. The semantic model's standalone test result is not an end-to-end cascade score. VLM-generated descriptions are auxiliary inputs, not the final moderation classifier.

Recorded API examples are available in the [image demo CSV](../services/image/demo_samples/api_flow_samples/latest_11_api_summary.csv) and [full JSON results](../services/image/demo_samples/api_flow_samples/latest_11_api_results.json). These are illustrative examples, not an additional benchmark.

## Four-Modality Video PCL Detection

The video service is based on the third-party PCLMM framework. PCL means **patronizing and condescending language**; this task does not establish broad video detection of every harmful-content category.

The full annotation file contains 715 clips (571 train / 144 test). The reported experiment uses only the feature-complete `disabled` subset: **116 clips, 85 train / 31 test**. The test set contains 26 normal and five PCL clips.

| Test threshold | Accuracy | Macro F1 | PCL precision | PCL recall | ROC AUC |
|---|---:|---:|---:|---:|---:|
| 0.5 | 83.87% | 59.74% | 50.00% | 20.00% | 0.8846 |
| 0.32 | 87.10% | 71.30% | 66.67% | 40.00% | 0.8846 |
| 0.008 | 83.87% | 75.67% | 50.00% | 80.00% | 0.8846 |

The API default threshold of 0.32 was recommended from this test-set comparison. It is not a separately validated threshold on an untouched evaluation set. These small-subset results support engineering verification, not a broad generalization claim.

### Resident Inference Timings

Report Table 5-4 gives reference times for an approximately ten-second clip:

| Backend | Feature extraction | Approximate time |
|---|---|---:|
| Subprocess | Serial | 65 s |
| Subprocess | Parallel | 47 s |
| Resident | Parallel | 7.4 s |

The reduction from 65 to 7.4 seconds is approximately 88.6%. These are reported configuration timings; the report does not fully specify a standardized hardware, warm-up, cache, or repeated-trial protocol. They should not be interpreted as a universal speedup or latency guarantee.
