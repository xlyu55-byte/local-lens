from pathlib import Path

import joblib
import pandas as pd


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

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


def build_ranking_results():
    print("Loading features and models...")

    features_path = PROCESSED_DIR / "training_features.csv"
    data = pd.read_csv(features_path)

    ctr_model = joblib.load(MODELS_DIR / "ctr_model.pkl")
    cvr_model = joblib.load(MODELS_DIR / "cvr_model.pkl")

    X = data[FEATURE_COLS].fillna(0)

    print("Predicting pCTR and pCVR...")
    data["pCTR"] = ctr_model.predict_proba(X)[:, 1]
    data["pCVR"] = cvr_model.predict_proba(X)[:, 1]

    # Simple personalization proxy
    data["personalization_score"] = data["query_category_match"]

    # Organic ranking score
    data["organic_score"] = (
        0.35 * data["hybrid_score"]
        + 0.25 * data["pCTR"]
        + 0.15 * data["pCVR"]
        + 0.15 * data["merchant_quality"]
        + 0.10 * data["personalization_score"]
    )

    # Sponsored ads score
    data["ad_score"] = data["bid"] * data["pCTR"] * data["merchant_quality"]

    # Final blended score
    data["final_score"] = (
        0.80 * data["organic_score"]
        + 0.20 * data["ad_score"]
    )

    # One row per business for demo ranking
    ranking = (
        data.sort_values("final_score", ascending=False)
        .drop_duplicates(subset=["business_id"])
        .copy()
    )

    ranking["rank"] = range(1, len(ranking) + 1)

    final_cols = [
        "rank",
        "business_id",
        "name",
        "query",
        "categories",
        "stars",
        "review_count",
        "pCTR",
        "pCVR",
        "hybrid_score",
        "merchant_quality",
        "personalization_score",
        "sponsored_flag",
        "bid",
        "organic_score",
        "ad_score",
        "final_score",
    ]

    ranking = ranking[final_cols]

    output_path = PROCESSED_DIR / "ranking_results.csv"
    ranking.to_csv(output_path, index=False)

    print("Ranking rows:", len(ranking))
    print("Top 10 results:")
    print(
        ranking[
            [
                "rank",
                "name",
                "stars",
                "pCTR",
                "pCVR",
                "sponsored_flag",
                "final_score",
            ]
        ].head(10)
    )
    print("Saved to:", output_path)


if __name__ == "__main__":
    build_ranking_results()