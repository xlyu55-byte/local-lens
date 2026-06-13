from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("data/processed")


class HybridRetriever:
    def __init__(self, business_path=PROCESSED_DIR / "business_clean.csv"):
        self.business = pd.read_csv(business_path)
        self.business["business_profile"] = self.business["business_profile"].fillna("")

        # Keyword retrieval
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.business_tfidf = self.vectorizer.fit_transform(
            self.business["business_profile"]
        )

        # Vector retrieval
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Encoding business profiles...")
        self.business_embeddings = self.embedding_model.encode(
            self.business["business_profile"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def search(self, query, top_k=10):
        # Keyword score
        query_tfidf = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.business_tfidf).flatten()

        # Vector score
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
        vector_scores = cosine_similarity(
            query_embedding,
            self.business_embeddings,
        ).flatten()

        results = self.business.copy()
        results["keyword_score"] = keyword_scores
        results["vector_score"] = vector_scores

        # Hybrid score
        results["hybrid_score"] = (
            0.5 * results["keyword_score"] + 0.5 * results["vector_score"]
        )

        results = results.sort_values("hybrid_score", ascending=False).head(top_k)

        return results[
            [
                "business_id",
                "name",
                "categories",
                "city",
                "stars",
                "review_count",
                "keyword_score",
                "vector_score",
                "hybrid_score",
                "business_profile",
            ]
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "quiet cafe for studying"
    results = retriever.search(query, top_k=10)

    print("Query:", query)
    print(
        results[
            [
                "name",
                "categories",
                "stars",
                "review_count",
                "keyword_score",
                "vector_score",
                "hybrid_score",
            ]
        ]
    )