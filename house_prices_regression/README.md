# House Prices Regression

## Competition

Kaggle: House Prices - Advanced Regression Techniques

## Goal

根据房屋属性预测最终销售价格。

## Highlights

- 对目标变量做 `log1p`
- 使用数值特征和类别特征联合建模
- 集成 `RandomForestRegressor` 与 `GradientBoostingRegressor`

## Expected Data Files

- `data/raw/train.csv`
- `data/raw/test.csv`

缺少真实数据时会自动生成小型演示数据。

## Run

```bash
python train.py
```

## Output

- `outputs/metrics.json`
- `outputs/submission.csv`
