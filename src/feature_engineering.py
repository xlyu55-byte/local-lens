from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")


def build_training_features():
    business_path = PROCESSED_DIR / "business_clean.csv"
    interactions_path = PROCESSED_DIR / "interactions.csv"

    print("Loading data...")
    business = pd.read_csv(business_path)
    interactions = pd.read_csv(interactions_path)

    keep_business_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "review_count_log",
        "is_open",
        "categories",
        "business_profile",
    ]

    business = business[keep_business_cols]

    print("Merging interaction and business features...")
    features = interactions.merge(
        business,
        on="business_id",
        how="left",
    )

    # Basic business quality feature
    features["merchant_quality"] = (
        0.7 * (features["stars"] / 5.0)
        + 0.3 * (features["review_count_log"] / features["review_count_log"].max())
    )

    # Simple query simulation
    # For this first version, every training row uses the same demo query.
    features["query"] = "quiet cafe for studying"

    # Simple query intent features
    features["is_cafe_query"] = features["query"].str.contains(
        "cafe|coffee|study|studying",
        case=False,
        na=False,
    ).astype(int)

    features["is_cheap_query"] = features["query"].str.contains(
        "cheap|budget|affordable",
        case=False,
        na=False,
    ).astype(int)

    features["is_nightlife_query"] = features["query"].str.contains(
        "night|bar|bars|club|drink",
        case=False,
        na=False,
    ).astype(int)

    # Category match feature
    features["category_text"] = features["categories"].fillna("").str.lower()

    features["query_category_match"] = (
        (
            (features["is_cafe_query"] == 1)
            & features["category_text"].str.contains("cafe|coffee|tea", na=False)
        )
        | (
            (features["is_nightlife_query"] == 1)
            & features["category_text"].str.contains("bar|nightlife", na=False)
        )
    ).astype(int)

    # Placeholder retrieval scores.
    # Later these can be replaced with per-query retrieval scores.
    features["keyword_score"] = features["query_category_match"] * 0.6
    features["vector_score"] = features["query_category_match"] * 0.7
    features["hybrid_score"] = 0.5 * features["keyword_score"] + 0.5 * features["vector_score"]

    # Sponsored ads simulation
    features["sponsored_flag"] = (
        features["business_id"].astype(str).str[-1].isin(["a", "b", "c", "d"])
    ).astype(int)

    features["bid"] = features["sponsored_flag"].apply(lambda x: 1.5 if x == 1 else 0.0)

    final_cols = [
        "user_id",
        "business_id",
        "name",
        "query",
        "rating",
        "clicked",
        "converted",
        "stars",
        "review_count",
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
        "categories",
    ]

    features = features[final_cols]

    output_path = PROCESSED_DIR / "training_features.csv"
    features.to_csv(output_path, index=False)

    print("Training feature rows:", len(features))
    print("Clicked rate:", features["clicked"].mean())
    print("Converted rate:", features["converted"].mean())
    print("Saved to:", output_path)


if __name__ == "__main__":
    build_training_features()