import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


LOCAL_SERVICE_KEYWORDS = [
    "Restaurants",
    "Food",
    "Coffee & Tea",
    "Cafes",
    "Bars",
    "Nightlife",
    "Desserts",
    "Bakeries",
    "Breakfast & Brunch",
    "American",
    "Italian",
    "Japanese",
    "Chinese",
    "Mexican",
    "Thai",
    "Pizza",
    "Sushi",
    "Ice Cream",
]


def load_json_lines(file_path, max_rows=None):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


def clean_business_data():
    business_path = RAW_DIR / "yelp_academic_dataset_business.json"

    print("Loading business data...")
    business = load_json_lines(business_path)

    print("Original business rows:", len(business))

    keep_cols = [
        "business_id",
        "name",
        "address",
        "city",
        "state",
        "postal_code",
        "latitude",
        "longitude",
        "stars",
        "review_count",
        "is_open",
        "categories",
    ]

    business = business[keep_cols]

    business = business.dropna(subset=["business_id", "name", "city", "categories"])

    # Keep local service categories only
    pattern = "|".join(LOCAL_SERVICE_KEYWORDS)
    business = business[
        business["categories"].str.contains(pattern, case=False, na=False)
    ]

    # Pick top city by number of businesses
    top_city = business["city"].value_counts().index[0]
    business = business[business["city"] == top_city].copy()

    # Limit project size
        # Limit project size
    business = business.sort_values("review_count", ascending=False).head(5000)

    business["review_count_log"] = business["review_count"].apply(
        lambda x: pd.NA if pd.isna(x) else __import__("math").log1p(x)
    )

    def create_business_profile(row):
        return (
            f"{row['name']} is a local business in {row['city']}, {row['state']}. "
            f"It has {row['stars']} stars and {row['review_count']} reviews. "
            f"Categories include {row['categories']}."
        )

    business["business_profile"] = business.apply(create_business_profile, axis=1)

    output_path = PROCESSED_DIR / "business_clean.csv"
    business.to_csv(output_path, index=False)

    print("Selected city:", top_city)
    print("Cleaned business rows:", len(business))
    print("Saved to:", output_path)


if __name__ == "__main__":
    clean_business_data()