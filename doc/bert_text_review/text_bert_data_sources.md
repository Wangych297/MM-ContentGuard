# 三个文本 BERT 的数据来源

模型架构和推理策略见：[tiered_text_review_strategy.md](tiered_text_review_strategy.md)。

当前文本审核后端包含三个 BERT：

```text
通用二分类 BERT
口语二分类 BERT
原因四分类 BERT
```

三者使用同一个 MacBERT backbone，但训练任务和数据来源不同。

## 1. 通用二分类 BERT

模型路径：

```text
risk-text-detection/runs/social_binary_v3_macbert_binary_bs128_no_online_aug
```

训练数据：

```text
risk-text-detection/data/processed/social_binary_v3/train.jsonl
```

任务：

```text
risk_binary: safe / risk
```

这个模型负责识别整体上的明确风险文本，是主审核模型。

训练集规模：

```text
train: 71000
safe:  36000
risk:  35000
```

主要数据来源：

```text
normal_chat_lccc       正常中文聊天数据，作为 safe 主来源
spam_message           垃圾短信 / 广告 / 诈骗类文本
ast                    短信风险文本及变体文本
fbs                    广告、诈骗、赌博、违法服务等细分类风险短信
cold                   中文攻击 / 冒犯类数据
toxicn                 中文仇恨 / 偏见 / 攻击类数据
chinese_offensive      中文冒犯、辱骂、性伤害、仇恨类数据
social_risk_augmented  为提升短文本、社交媒体风险召回构造的风险增强数据
```

训练集来源数量：

```text
normal_chat_lccc       22000
social_risk_augmented  14000
spam_message            8000
fbs                     7000
ast                     6000
cold                    6000
chinese_offensive       5000
toxicn                  3000
```

## 2. 口语二分类 BERT

模型路径：

```text
risk-text-detection/runs/social_intent_binary_v1_from_general
```

训练数据：

```text
risk-text-detection/data/processed/social_intent_binary_v1/train.jsonl
```

任务：

```text
risk_binary: safe / risk
```

这个模型是在通用二分类模型基础上继续训练的，专门补充识别自然聊天、隐晦表达、短句口语中的风险。

训练集规模：

```text
train: 1600
safe:   800
risk:   800
```

数据来源：

```text
social_intent_synthetic_v1
```

这是一份面向社交媒体口语场景构造的合成数据集，包含：

```text
risk_plain_social        直接口语风险表达
risk_variant_social      变体、谐音、拆字等风险表达
risk_contextual_social   需要上下文理解的风险表达
benign_sensitive_context 含敏感词但语义安全的边界样本
normal_short_chat        正常短聊天
normal_daily_context     正常日常语境
hard_negative_boundary   容易误判的安全边界样本
```

构造时排除了以下参考集的完全归一化重复文本，避免和评测集或通用模型数据重复：

```text
golden_social_binary_v1
golden_social_binary_v2_diag
golden_social_binary_v2_sealed
social_binary_v3 train/dev/test
```

## 3. 原因四分类 BERT

模型路径：

```text
risk-text-detection/runs/reason_category_v1_macbert
```

训练数据：

```text
risk-text-detection/data/processed/reason_category_v1/train.jsonl
```

任务：

```text
reason_category
```

这个模型只负责给风险文本输出不通过原因。`safe` 文本不参与原因分类训练。

四个原因类别：

```text
financial_illegal_promotion  诈骗/违法服务/广告引流
gambling                     赌博/博彩
sexual_content               色情/性相关风险
abuse_hate                   辱骂/骚扰/仇恨歧视
```

训练数据来自前两类数据集中的风险样本：

```text
social_binary_v3/train.jsonl
social_intent_binary_v1/train.jsonl
```

然后把原来的细标签合并成四个产品侧原因标签：

```text
fraud_phishing / illegal_service / spam_general / ad_promotion / loan_ad / review_spam / other_risk
  -> financial_illegal_promotion

gambling
  -> gambling

porn_sexual / sex_harm
  -> sexual_content

toxicity_abuse / hate_bias
  -> abuse_hate
```

训练集规模：

```text
train: 14074
```

按四类分布：

```text
financial_illegal_promotion  5000
abuse_hate                   5000
sexual_content               2172
gambling                     1902
```

原始来源分布：

```text
social_risk_augmented        5910
fbs                          2171
cold                         1505
chinese_offensive            1429
spam_message                 1030
ast                           781
toxicn                        768
social_intent_synthetic_v1    480
```

## 总结

三个 BERT 的数据分工如下：

```text
通用二分类 BERT：
  使用大规模混合数据，负责整体风险识别。

口语二分类 BERT：
  使用小规模高质量口语合成数据，负责补抓自然聊天中的隐晦风险。

原因四分类 BERT：
  使用前两类数据中的风险样本，合并细标签后学习不通过原因。
```

因此，前两个 BERT 解决“是否违规”，第三个 BERT 解决“为什么违规”。
