# kaggle-case-collection

A GitHub-ready collection of Kaggle starter cases for classification, regression, and NLP competitions.

## Included Cases

- Titanic baseline: binary classification with preprocessing, pipelines, and cross-validation
- House Prices regression: tabular regression with log target transform and ensemble models
- Disaster Tweets NLP: TF-IDF plus Logistic Regression text classification baseline

## Structure

```text
kaggle_cases/
  competition_template/
  titanic_baseline/
  house_prices_regression/
  disaster_tweets_nlp/
  requirements.txt
  .gitignore
  PUBLISH_TO_GITHUB.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

Then run any case:

```bash
python train.py
```

If real Kaggle data is not present, each case can fall back to a very small demo dataset so the repository remains runnable.

## Next Steps

1. Put the original Kaggle files into each case's `data/raw/` directory.
2. Run the training script to generate `outputs/submission.csv`.
3. Add your leaderboard scores and experiment notes to the case README files.
4. Push updates to GitHub.
