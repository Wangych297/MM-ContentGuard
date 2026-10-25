# 文本审核评测集与最终结果

本文记录最后一版文本审核模型使用过的评测集合、构造方式和聚合评测结果。

## 最终模型

```text
通用二分类 BERT: runs/social_binary_v3_macbert_binary_bs128_no_online_aug
口语二分类 BERT: runs/social_intent_binary_v1_from_general
原因四分类 BERT: runs/reason_category_v1_macbert
```

二分类策略：

```text
hard_risk = general_score >= 0.5
review    = general_score < 0.5 且 social_score >= 0.5
safe      = otherwise
```

评测报告来源：

```text
reports/tiered_review_general_v3_social_v1_g050_s050_eval
reports/social_intent_binary_v1_from_general_social_intent_test_eval
reports/reason_category_v1_macbert_test_eval
```

## 评测集合构造方式

| 集合 | 路径 | 规模 | 构造方式 |
| --- | --- | ---: | --- |
| 常规 dev | `data/processed/social_binary_v3/dev.jsonl` | 7800 | 从 `normal_chat_lccc`、`spam_message`、`ast`、`fbs`、`cold`、`toxicn`、`chinese_offensive` 抽样；4650 safe / 3150 risk。用于开发期观察。 |
| 常规 test | `data/processed/social_binary_v3/test.jsonl` | 7800 | 与 dev 同源但独立抽样；4650 safe / 3150 risk。用于常规分布测试。 |
| 口语 test | `data/processed/social_intent_binary_v1/test.jsonl` | 400 | 口语社交场景合成集；200 safe / 200 risk；构造时排除与 golden 集和 `social_binary_v3` 的归一化重复。 |
| golden v1 | `data/eval/golden_social_binary_v1.jsonl` | 100 | 人工构造 eval-only 盲测集；50 safe / 50 risk；风险样本含普通表达、变音、拆字、数字、空格、拼音混排等变体。 |
| golden v2 diag | `data/eval/golden_social_binary_v2_diag.jsonl` | 400 | 诊断集，可打开文本做 bad case；200 safe / 200 risk；包含口语风险、变体风险、上下文风险、正常聊天和敏感词安全语境。 |
| golden v2 sealed | `data/eval/golden_social_binary_v2_sealed.jsonl` | 400 | 与 diag 相似但不同的封存集；只看聚合指标，不导出文本或逐条预测；200 safe / 200 risk。 |
| 原因四分类 test | `data/processed/reason_category_v1/test.jsonl` | 3331 | 从 `social_binary_v3/test` 和 `social_intent_binary_v1/test` 的风险样本构造；safe 不参与；细标签合并成四个不通过原因。 |

`data/eval/` 下的 golden 集均为 eval-only，不用于训练、验证、阈值选择或构造训练样本。

## 二分类三层策略结果

下表中：

```text
hard_risk：只看通用模型硬判风险。
risk_or_review：把 hard_risk 和 review 都视为非安全。
```

| 集合 | hard acc | hard macro F1 | hard risk recall | risk_or_review acc | risk_or_review macro F1 | risk_or_review risk recall | review 队列 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 常规 dev | 0.9583 | 0.9567 | 0.9444 | 0.9385 | 0.9368 | 0.9606 | 257 / 7800 |
| 常规 test | 0.9629 | 0.9614 | 0.9473 | 0.9474 | 0.9459 | 0.9663 | 241 / 7800 |
| golden v1 | 0.7700 | 0.7572 | 0.5400 | 0.9800 | 0.9800 | 0.9600 | 21 / 100 |
| golden v2 diag | 0.6250 | 0.5655 | 0.2550 | 0.9950 | 0.9950 | 0.9950 | 148 / 400 |
| golden v2 sealed | 0.6375 | 0.5861 | 0.2850 | 0.9900 | 0.9900 | 0.9900 | 141 / 400 |

复核层补抓效果：

| 集合 | 通用模型漏判风险 | review 补抓风险 | 安全文本进 review |
| --- | ---: | ---: | ---: |
| 常规 dev | 175 | 51 | 206 |
| 常规 test | 166 | 60 | 181 |
| golden v1 | 23 | 21 | 0 |
| golden v2 diag | 149 | 148 | 0 |
| golden v2 sealed | 143 | 141 | 0 |

## 组件评测结果

口语二分类模型在口语 test 上：

```text
support  = 400
accuracy = 1.0000
macro F1 = 1.0000
```

原因四分类模型在原因 test 上：

```text
support  = 3331
accuracy = 0.9943
macro F1 = 0.9864
```

四个原因类别的 F1：

| 类别 | F1 |
| --- | ---: |
| 诈骗/违法服务/广告引流 | 0.9959 |
| 赌博/博彩 | 0.9818 |
| 色情/性相关风险 | 0.9719 |
| 辱骂/骚扰/仇恨歧视 | 0.9958 |

## 简要结论

常规测试集上，`hard_risk` 维持较高精度和较低误杀；golden 口语/变体集上，单靠通用模型召回不足。加入 `review` 层后，golden v1/v2 的风险召回显著提升，同时 sealed 集保持只看聚合指标，避免针对封存样本调参。
