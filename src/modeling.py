from pathlib import Path
import json
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
    "stars",
    "review_count_log",
    "is_open",
    "merchant_quality",
    "is_cafe_query",
    "is_cheap_query",
    "is_nightlife_query",
    "query_category_match",
    "keyword_score",
    "vector_score",
    "hybrid_score",
    "sponsored_flag",
    "bid",
]


def train_ctr_cvr_models():
    print("Loading training features...")
    data_path = PROCESSED_DIR / "training_features.csv"
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLS].fillna(0)

    y_ctr = df["clicked"]
    y_cvr = df["converted"]

    X_train, X_test, y_ctr_train, y_ctr_test, y_cvr_train, y_cvr_test = train_test_split(
        X,
        y_ctr,
        y_cvr,
        test_size=0.2,
        random_state=42,
        stratify=y_ctr,
    )

    print("Training CTR baseline model...")
    ctr_logreg = LogisticRegression(max_iter=1000)
    ctr_logreg.fit(X_train, y_ctr_train)

    ctr_logreg_pred = ctr_logreg.predict_proba(X_test)[:, 1]

    print("Training CTR main model...")
    ctr_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    ctr_model.fit(X_train, y_ctr_train)

    ctr_pred = ctr_model.predict_proba(X_test)[:, 1]

    print("Training CVR model...")
    cvr_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    cvr_model.fit(X_train, y_cvr_train)

    cvr_pred = cvr_model.predict_proba(X_test)[:, 1]
    cvr_pred_label = (cvr_pred >= 0.5).astype(int)

    metrics = {
        "ctr_logreg_auc": roc_auc_score(y_ctr_test, ctr_logreg_pred),
        "ctr_logreg_log_loss": log_loss(y_ctr_test, ctr_logreg_pred),
        "ctr_random_forest_auc": roc_auc_score(y_ctr_test, ctr_pred),
        "ctr_random_forest_log_loss": log_loss(y_ctr_test, ctr_pred),
        "cvr_random_forest_auc": roc_auc_score(y_cvr_test, cvr_pred),
        "cvr_random_forest_f1": f1_score(y_cvr_test, cvr_pred_label),
    }

    print("Model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    joblib.dump(ctr_model, MODELS_DIR / "ctr_model.pkl")
    joblib.dump(cvr_model, MODELS_DIR / "cvr_model.pkl")

    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved CTR model to models/ctr_model.pkl")
    print("Saved CVR model to models/cvr_model.pkl")
    print("Saved metrics to reports/model_metrics.json")


if __name__ == "__main__":
    train_ctr_cvr_models()