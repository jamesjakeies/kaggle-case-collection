from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        [
            [1, "RL", "Inside", 8450, "Gtl", 7, 5, 2003, 1710, 208500],
            [2, "RL", "FR2", 9600, "Gtl", 6, 8, 1976, 1262, 181500],
            [3, "RL", "Inside", 11250, "Gtl", 7, 5, 2001, 1786, 223500],
            [4, "RL", "Corner", 9550, "Gtl", 7, 5, 1915, 1717, 140000],
            [5, "RL", "FR2", 14260, "Gtl", 8, 5, 2000, 2198, 250000],
            [6, "RM", "Inside", 14115, "Gtl", 5, 5, 1993, 1362, 143000],
            [7, "RL", "Inside", 10084, "Gtl", 8, 5, 2004, 1694, 307000],
            [8, "RM", "Corner", 10382, "Gtl", 7, 6, 1973, 2090, 200000],
            [9, "FV", "Inside", 6120, "Gtl", 7, 5, 1931, 1774, 129900],
            [10, "RL", "Inside", 7420, "Gtl", 5, 6, 1939, 1077, 118000],
        ],
        columns=[
            "Id",
            "MSZoning",
            "LotConfig",
            "LotArea",
            "LandSlope",
            "OverallQual",
            "OverallCond",
            "YearBuilt",
            "GrLivArea",
            "SalePrice",
        ],
    )
    test_df = pd.DataFrame(
        [
            [1461, "RH", "Inside", 11622, "Gtl", 5, 6, 1961, 896],
            [1462, "RL", "Corner", 14267, "Gtl", 6, 6, 1958, 1329],
            [1463, "RL", "Inside", 13830, "Gtl", 5, 5, 1997, 1629],
            [1464, "RL", "Inside", 9978, "Gtl", 6, 6, 1998, 1604],
        ],
        columns=[
            "Id",
            "MSZoning",
            "LotConfig",
            "LotArea",
            "LandSlope",
            "OverallQual",
            "OverallCond",
            "YearBuilt",
            "GrLivArea",
        ],
    )
    return train_df, test_df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return build_demo_data()


def make_pipeline() -> Pipeline:
    numeric_features = ["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea"]
    categorical_features = ["MSZoning", "LotConfig", "LandSlope"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
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

    model = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=80, random_state=42)),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_data()

    feature_cols = ["MSZoning", "LotConfig", "LotArea", "LandSlope", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea"]
    X = train_df[feature_cols]
    y = np.log1p(train_df["SalePrice"])
    X_test = test_df[feature_cols]

    pipeline = make_pipeline()
    cv = KFold(n_splits=min(5, len(train_df)), shuffle=True, random_state=42)
    oof_pred = cross_val_predict(pipeline, X, y, cv=cv)

    metrics = {
        "rmse_log": round(rmse(y.to_numpy(), oof_pred), 4),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
    }

    pipeline.fit(X, y)
    sale_price_pred = np.expm1(pipeline.predict(X_test))
    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": np.round(sale_price_pred, 2)})

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
