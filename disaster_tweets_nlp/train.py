from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s#@]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        [
            [1, "Our house is on fire please help", 1],
            [2, "I love this sunny day in Shanghai", 0],
            [3, "Flood warning issued for the river area", 1],
            [4, "Concert tonight will be amazing", 0],
            [5, "Earthquake damaged several roads downtown", 1],
            [6, "My phone battery is a disaster", 0],
            [7, "Rescue teams are searching collapsed buildings", 1],
            [8, "Coffee spilled on my laptop again", 0],
            [9, "Wildfire smoke covers the city skyline", 1],
            [10, "This homework is killing me", 0],
        ],
        columns=["id", "text", "target"],
    )
    test_df = pd.DataFrame(
        [
            [11, "Emergency shelters opened after the storm"],
            [12, "Movie night with friends was fun"],
            [13, "Train delay ruined my schedule"],
            [14, "Volunteers deliver food after tornado"],
        ],
        columns=["id", "text"],
    )
    return train_df, test_df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return build_demo_data()


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_data()

    X = train_df["text"].fillna("")
    y = train_df["target"].astype(int)
    X_test = test_df["text"].fillna("")

    pipeline = make_pipeline()
    splits = min(5, y.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, splits), shuffle=True, random_state=42)
    oof_pred = cross_val_predict(pipeline, X, y, cv=cv)

    metrics = {
        "accuracy": round(float(accuracy_score(y, oof_pred)), 4),
        "f1": round(float(f1_score(y, oof_pred)), 4),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
    }

    pipeline.fit(X, y)
    submission = pd.DataFrame({"id": test_df["id"], "target": pipeline.predict(X_test).astype(int)})

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
