# Titanic Baseline

## Competition

Kaggle: Titanic - Machine Learning from Disaster

## Goal

根据乘客的基础信息预测是否生还。

## Highlights

- 使用 `ColumnTransformer` 统一处理数值和类别特征
- 用 `Pipeline` 串联预处理和模型
- 输出可直接提交到 Kaggle 的 `submission.csv`

## Expected Data Files

把 Kaggle 原始文件放到：

- `data/raw/train.csv`
- `data/raw/test.csv`

如果没有真实数据，脚本会自动生成小样本演示数据。

## Run

```bash
python train.py
```

## Output

- `outputs/metrics.json`
- `outputs/submission.csv`
