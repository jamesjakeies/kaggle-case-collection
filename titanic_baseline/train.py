from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        [
            [1, 0, 3, "Allen", "male", 22.0, 1, 0, 7.25, "S"],
            [2, 1, 1, "Bonnell", "female", 58.0, 0, 0, 26.55, "S"],
            [3, 1, 3, "Cumings", "female", 38.0, 1, 0, 71.28, "C"],
            [4, 1, 1, "Futrelle", "female", 35.0, 1, 0, 53.10, "S"],
            [5, 0, 3, "Moran", "male", np.nan, 0, 0, 8.05, "S"],
            [6, 0, 3, "McCarthy", "male", 54.0, 0, 0, 51.86, "S"],
            [7, 0, 3, "Palsson", "male", 2.0, 3, 1, 21.07, "S"],
            [8, 1, 2, "Heikkinen", "female", 26.0, 0, 0, 13.00, "S"],
            [9, 1, 2, "Johnson", "female", 27.0, 0, 0, 11.13, "S"],
            [10, 0, 3, "Nasser", "male", 14.0, 1, 0, 30.07, "C"],
        ],
        columns=[
            "PassengerId",
            "Survived",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
    )
    test_df = pd.DataFrame(
        [
            [11, 3, "Kelly", "male", 34.5, 0, 0, 7.83, "Q"],
            [12, 3, "Wilkes", "female", 47.0, 1, 0, 7.00, "S"],
            [13, 2, "Myles", "male", 62.0, 0, 0, 9.69, "Q"],
            [14, 1, "Ryerson", "female", 21.0, 1, 0, 82.17, "S"],
        ],
        columns=["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    )
    return train_df, test_df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return build_demo_data()


def make_pipeline() -> Pipeline:
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_data()

    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = train_df[feature_cols]
    y = train_df["Survived"]
    X_test = test_df[feature_cols]

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
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": pipeline.predict(X_test).astype(int),
        }
    )

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
