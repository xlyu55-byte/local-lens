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

def clean_review_data(max_reviews=50000):
    business_path = PROCESSED_DIR / "business_clean.csv"

    review_path = Path("/Users/lvxinyi/Desktop/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json")

    print("Loading cleaned business ids...")
    business = pd.read_csv(business_path)
    business_ids = set(business["business_id"])

    rows = []

    print("Sampling review data...")
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)

            if review["business_id"] in business_ids:
                rows.append({
                    "review_id": review["review_id"],
                    "user_id": review["user_id"],
                    "business_id": review["business_id"],
                    "stars": review["stars"],
                    "text": review["text"],
                    "date": review["date"],
                    "useful": review["useful"],
                    "funny": review["funny"],
                    "cool": review["cool"],
                })

            if len(rows) >= max_reviews:
                break

    reviews = pd.DataFrame(rows)

    output_path = PROCESSED_DIR / "reviews_clean.csv"
    reviews.to_csv(output_path, index=False)

    print("Cleaned review rows:", len(reviews))
    print("Saved to:", output_path)

def create_interaction_labels(negative_ratio=1):
    reviews_path = PROCESSED_DIR / "reviews_clean.csv"
    business_path = PROCESSED_DIR / "business_clean.csv"

    print("Creating interaction labels...")

    reviews = pd.read_csv(reviews_path)
    business = pd.read_csv(business_path)

    # Positive samples: user actually reviewed this business
    positive = reviews[["user_id", "business_id", "stars", "date"]].copy()
    positive = positive.rename(columns={"stars": "rating"})
    positive["clicked"] = 1
    positive["converted"] = (positive["rating"] >= 4).astype(int)

    # Negative samples: same users, random businesses they did not review
    business_ids = business["business_id"].tolist()
    interacted_pairs = set(zip(positive["user_id"], positive["business_id"]))

    negative_rows = []
    target_negative_count = len(positive) * negative_ratio

    print("Sampling negative interactions...")

    sampled_users = positive["user_id"].sample(
        n=target_negative_count,
        replace=True,
        random_state=42
    ).tolist()

    sampled_businesses = pd.Series(business_ids).sample(
        n=target_negative_count,
        replace=True,
        random_state=42
    ).tolist()

    for user_id, business_id in zip(sampled_users, sampled_businesses):
        if (user_id, business_id) not in interacted_pairs:
            negative_rows.append({
                "user_id": user_id,
                "business_id": business_id,
                "rating": 0,
                "date": None,
                "clicked": 0,
                "converted": 0,
            })

    negative = pd.DataFrame(negative_rows)

    interactions = pd.concat([positive, negative], ignore_index=True)

    output_path = PROCESSED_DIR / "interactions.csv"
    interactions.to_csv(output_path, index=False)

    print("Positive rows:", len(positive))
    print("Negative rows:", len(negative))
    print("Total interaction rows:", len(interactions))
    print("Clicked rate:", interactions["clicked"].mean())
    print("Converted rate:", interactions["converted"].mean())
    print("Saved to:", output_path)


if __name__ == "__main__":
    clean_business_data()
    clean_review_data()
    create_interaction_labels()