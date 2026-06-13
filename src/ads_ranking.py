from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")


def build_sponsored_ads_analysis():
    ranking_path = PROCESSED_DIR / "ranking_results.csv"

    print("Loading ranking results...")
    ranking = pd.read_csv(ranking_path)

    # Organic-only ranking: ignore ad_score
    organic_only = ranking.copy()
    organic_only["organic_only_final_score"] = organic_only["organic_score"]
    organic_only = organic_only.sort_values(
        "organic_only_final_score",
        ascending=False
    ).copy()
    organic_only["organic_rank"] = range(1, len(organic_only) + 1)

    # Sponsored blended ranking: use existing final_score
    sponsored_blended = ranking.copy()
    sponsored_blended = sponsored_blended.sort_values(
        "final_score",
        ascending=False
    ).copy()
    sponsored_blended["sponsored_blended_rank"] = range(
        1, len(sponsored_blended) + 1
    )

    comparison = sponsored_blended.merge(
        organic_only[["business_id", "organic_rank", "organic_only_final_score"]],
        on="business_id",
        how="left",
    )

    comparison["rank_change"] = (
        comparison["organic_rank"] - comparison["sponsored_blended_rank"]
    )

    final_cols = [
        "business_id",
        "name",
        "categories",
        "stars",
        "review_count",
        "pCTR",
        "pCVR",
        "sponsored_flag",
        "bid",
        "organic_score",
        "ad_score",
        "final_score",
        "organic_rank",
        "sponsored_blended_rank",
        "rank_change",
    ]

    comparison = comparison[final_cols]

    output_path = PROCESSED_DIR / "sponsored_ranking_results.csv"
    comparison.to_csv(output_path, index=False)

    print("Sponsored ranking rows:", len(comparison))
    print("Top sponsored businesses:")
    print(
        comparison[comparison["sponsored_flag"] == 1][
            [
                "name",
                "pCTR",
                "bid",
                "ad_score",
                "organic_rank",
                "sponsored_blended_rank",
                "rank_change",
            ]
        ].head(10)
    )
    print("Saved to:", output_path)


if __name__ == "__main__":
    build_sponsored_ads_analysis()