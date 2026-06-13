from pathlib import Path
import json

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_k(df, score_col, target_col="clicked", k=10):
    top_k = df.sort_values(score_col, ascending=False).head(k)
    return top_k[target_col].mean()


def ndcg_at_k(df, score_col, target_col="clicked", k=10):
    ranked = df.sort_values(score_col, ascending=False).head(k).copy()

    dcg = 0
    for i, rel in enumerate(ranked[target_col], start=1):
        dcg += rel / np.log2(i + 1)

    ideal = df.sort_values(target_col, ascending=False).head(k)

    idcg = 0
    for i, rel in enumerate(ideal[target_col], start=1):
        idcg += rel / np.log2(i + 1)

    if idcg == 0:
        return 0

    return dcg / idcg


def run_ab_testing_simulation():
    features_path = PROCESSED_DIR / "training_features.csv"
    ranking_path = PROCESSED_DIR / "ranking_results.csv"

    print("Loading data...")
    features = pd.read_csv(features_path)
    ranking = pd.read_csv(ranking_path)

    features["rating_only_score"] = features["stars"] / 5.0
    features["relevance_only_score"] = features["hybrid_score"]

    ranking_scores = ranking[["business_id", "final_score"]]

    features = features.merge(
        ranking_scores,
        on="business_id",
        how="left",
    )

    features["final_score"] = features["final_score"].fillna(0)

    results = {
        "rating_only_precision_at_10": precision_at_k(features, "rating_only_score"),
        "relevance_only_precision_at_10": precision_at_k(features, "relevance_only_score"),
        "blended_precision_at_10": precision_at_k(features, "final_score"),
        "rating_only_ndcg_at_10": ndcg_at_k(features, "rating_only_score"),
        "relevance_only_ndcg_at_10": ndcg_at_k(features, "relevance_only_score"),
        "blended_ndcg_at_10": ndcg_at_k(features, "final_score"),
    }

    baseline_top = features.sort_values("relevance_only_score", ascending=False).head(1000)
    blended_top = features.sort_values("final_score", ascending=False).head(1000)

    baseline_ctr = baseline_top["clicked"].mean()
    blended_ctr = blended_top["clicked"].mean()

    baseline_cvr = baseline_top["converted"].mean()
    blended_cvr = blended_top["converted"].mean()

    baseline_revenue = baseline_top["bid"].mean()
    blended_revenue = blended_top["bid"].mean()

    results["baseline_ctr_top_1000"] = baseline_ctr
    results["blended_ctr_top_1000"] = blended_ctr
    results["ctr_lift"] = (blended_ctr - baseline_ctr) / baseline_ctr if baseline_ctr > 0 else 0

    results["baseline_cvr_top_1000"] = baseline_cvr
    results["blended_cvr_top_1000"] = blended_cvr
    results["cvr_lift"] = (blended_cvr - baseline_cvr) / baseline_cvr if baseline_cvr > 0 else 0

    results["baseline_revenue_proxy"] = baseline_revenue
    results["blended_revenue_proxy"] = blended_revenue
    results["revenue_proxy_lift"] = (
        (blended_revenue - baseline_revenue) / baseline_revenue
        if baseline_revenue > 0 else 0
    )

    output_json = REPORTS_DIR / "ab_testing_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    output_csv = PROCESSED_DIR / "ab_testing_results.csv"
    pd.DataFrame([results]).to_csv(output_csv, index=False)

    print("A/B testing results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("Saved to:", output_json)
    print("Saved to:", output_csv)


if __name__ == "__main__":
    run_ab_testing_simulation()