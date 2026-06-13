from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_DIR = Path("data/processed")


class KeywordRetriever:
    def __init__(self, business_path=PROCESSED_DIR / "business_clean.csv"):
        self.business = pd.read_csv(business_path)

        self.business["business_profile"] = self.business["business_profile"].fillna("")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        self.business_tfidf = self.vectorizer.fit_transform(
            self.business["business_profile"]
        )

    def search(self, query, top_k=10):
        query_tfidf = self.vectorizer.transform([query])

        scores = cosine_similarity(query_tfidf, self.business_tfidf).flatten()

        results = self.business.copy()
        results["keyword_score"] = scores

        results = results.sort_values("keyword_score", ascending=False).head(top_k)

        return results[
            [
                "business_id",
                "name",
                "categories",
                "city",
                "stars",
                "review_count",
                "keyword_score",
                "business_profile",
            ]
        ]


if __name__ == "__main__":
    retriever = KeywordRetriever()

    query = "quiet cafe for studying"
    results = retriever.search(query, top_k=10)

    print("Query:", query)
    print(
        results[
            ["name", "categories", "stars", "review_count", "keyword_score"]
        ]
    )