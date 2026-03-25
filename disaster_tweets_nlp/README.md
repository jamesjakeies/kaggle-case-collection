# Disaster Tweets NLP

## Competition

Kaggle: Natural Language Processing with Disaster Tweets

## Goal

根据推文文本判断是否与真实灾难事件相关。

## Highlights

- 文本清洗
- TF-IDF 特征
- Logistic Regression 文本分类 baseline

## Expected Data Files

- `data/raw/train.csv`
- `data/raw/test.csv`

缺少真实数据时脚本会自动生成演示样本。

## Run

```bash
python train.py
```

## Output

- `outputs/metrics.json`
- `outputs/submission.csv`
