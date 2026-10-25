# 数据集说明

本目录用于组织系统训练、验证、测试所需的数据。仓库只保留数据结构说明和少量测试样例；原始数据、处理后数据和训练产物不建议直接提交到代码仓库。

## 当前文本主线数据集

| 数据集名称 | 当前用途 | 标签类型 | 是否进入 quick20k |
| --- | --- | --- | --- |
| SpamMessage | 中文垃圾短信与正常短信检测 | 二分类：正常 / 垃圾 | 是 |
| AST 中文垃圾文本变体数据集 | 垃圾文本检测、字音字形伪装识别、鲁棒性评估 | 二分类、原文/变体文本、伪装类型 | 是 |
| FBS_SMS_Dataset | 广告、诈骗、赌博、违法服务等短信细分类 | 多分类风险类别 | 是 |
| COLDataset | 中文攻击性、仇恨与正常文本检测 | 二分类：正常 / 有害 | 是 |
| ToxiCN | 中文毒性与仇恨文本检测 | 二分类、毒性相关标签 | 是 |
| Chinese Offensive Language 数据集 | 辱骂、性危害、偏见类文本检测 | 多分类风险类别 | 是 |

## 辅助与后续扩展数据集

| 数据集名称 | 当前用途 | 备注 |
| --- | --- | --- |
| Review Spam 数据集 | 评价垃圾文本检测候选数据 | 暂未进入默认 quick20k |
| ToxiCN-MM | 多模态有害内容检测扩展 | 后续用于图文联合检测 |
| PCLMM | 多模态有害内容检测扩展 | 后续用于图文联合检测 |

## 数据切分配置

默认使用 `configs/dataset_profiles.json` 管理切分方案：

| profile | 说明 |
| --- | --- |
| quick20k | 笔记本可快速训练的默认版本，训练集约 2 万条 |
| lab_full | 实验室 GPU 使用的大规模版本，保留后续完整训练选项 |

当前 `quick20k` 使用的数据量配置如下：

| split | SpamMessage | AST | FBS_SMS_Dataset | COLDataset | ToxiCN | Chinese Offensive Language |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8000 | 4000 | 2500 | 2500 | 1500 | 1500 |
| dev | 1000 | 800 | 500 | 500 | 300 | 300 |
| test | 1000 | 800 | 500 | 500 | 300 | 300 |

## 统一标签设计

处理后的样本统一为 JSONL，每行包含：

```json
{
  "id": "sample_id",
  "text": "文本内容",
  "labels": {
    "risk_binary": 0,
    "primary_category": "safe",
    "risk_tags": [],
    "toxic_binary": null,
    "toxic_type": null,
    "camouflage_type": null
  },
  "metadata": {}
}
```

主要训练任务包括：

| 字段 | 含义 |
| --- | --- |
| risk_binary | 是否为风险文本 |
| primary_category | 风险主类别 |
| toxic_binary | 是否为毒性、攻击性或仇恨文本 |
| camouflage_type | 是否存在字音、字形、数字、拉丁字符等伪装变体 |

## 目录约定

```text
data/
  raw/        原始数据
  interim/    中间处理文件
  processed/  统一格式后的训练、验证、测试数据
  manifests/  数据构建统计与记录
```
