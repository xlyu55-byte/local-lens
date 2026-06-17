from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

sys.path.append("src")
from retrieval import HybridRetriever


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


st.set_page_config(
    page_title="LocalLens",
    layout="wide",
)


@st.cache_data
def load_data():
    ranking = pd.read_csv(PROCESSED_DIR / "ranking_results.csv")
    sponsored = pd.read_csv(PROCESSED_DIR / "sponsored_ranking_results.csv")

    with open(REPORTS_DIR / "model_metrics.json", "r") as f:
        metrics = json.load(f)

    with open(REPORTS_DIR / "ab_testing_results.json", "r") as f:
        ab_results = json.load(f)

    return ranking, sponsored, metrics, ab_results


@st.cache_resource
def load_retriever():
    return HybridRetriever()


ranking, sponsored, metrics, ab_results = load_data()
retriever = load_retriever()


st.title("LocalLens")
st.caption("Local Services Search, Recommendation, CTR/CVR, and Ads Ranking System")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Search Demo",
        "Ranking Explanation",
        "Model Performance",
        "A/B Testing",
    ]
)


with tab1:
    st.header("Search Demo")

    query = st.text_input(
        "Enter a local services query",
        value="quiet cafe for studying",
    )

    include_sponsored = st.toggle("Include sponsored results", value=True)
    top_k = st.slider("Top K results", min_value=5, max_value=30, value=10)

    retrieval_results = retriever.search(query, top_k=200)

    demo_results = retrieval_results.merge(
        ranking[
            [
                "business_id",
                "pCTR",
                "pCVR",
                "merchant_quality",
                "personalization_score",
                "sponsored_flag",
                "bid",
                "organic_score",
                "ad_score",
                "final_score",
            ]
        ],
        on="business_id",
        how="left",
    )

    demo_results = demo_results.fillna(
        {
            "pCTR": 0,
            "pCVR": 0,
            "merchant_quality": 0,
            "personalization_score": 0,
            "sponsored_flag": 0,
            "bid": 0,
            "organic_score": 0,
            "ad_score": 0,
            "final_score": 0,
        }
    )

    demo_results["final_score"] = (
        0.35 * demo_results["hybrid_score"]
        + 0.25 * demo_results["pCTR"]
        + 0.15 * demo_results["pCVR"]
        + 0.15 * demo_results["merchant_quality"]
        + 0.10 * demo_results["personalization_score"]
        + 0.20 * demo_results["ad_score"]
    )

    if not include_sponsored:
        demo_results = demo_results[demo_results["sponsored_flag"] == 0]

    demo_results = (
        demo_results.sort_values("final_score", ascending=False)
        .drop_duplicates(subset=["name"])
        .head(top_k)
        .copy()
    )

    demo_results["rank"] = range(1, len(demo_results) + 1)

    st.dataframe(
        demo_results[
            [
                "rank",
                "name",
                "categories",
                "city",
                "stars",
                "review_count",
                "keyword_score",
                "vector_score",
                "hybrid_score",
                "pCTR",
                "pCVR",
                "sponsored_flag",
                "final_score",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Why recommended?")

    if len(demo_results) > 0:
        top_result = demo_results.iloc[0]

        st.write(
            f"**{top_result['name']}** is recommended because it has strong search relevance "
            f"(hybrid score={top_result['hybrid_score']:.2f}), predicted engagement "
            f"(pCTR={top_result['pCTR']:.2f}), predicted conversion "
            f"(pCVR={top_result['pCVR']:.2f}), and final ranking score "
            f"({top_result['final_score']:.2f})."
        )
    else:
        st.warning("No results found. Try another query.")


with tab2:
    st.header("Ranking Explanation")

    selected_business = st.selectbox(
        "Choose a business",
        ranking["name"].drop_duplicates().head(100).tolist(),
    )

    row = ranking[ranking["name"] == selected_business].iloc[0]

    explanation = pd.DataFrame(
        {
            "Component": [
                "Hybrid relevance",
                "Predicted CTR",
                "Predicted CVR",
                "Merchant quality",
                "Personalization",
                "Ad score",
                "Final score",
            ],
            "Value": [
                row["hybrid_score"],
                row["pCTR"],
                row["pCVR"],
                row["merchant_quality"],
                row["personalization_score"],
                row["ad_score"],
                row["final_score"],
            ],
        }
    )

    st.dataframe(explanation, use_container_width=True)

    st.write(
        "Final score is a weighted combination of hybrid relevance, predicted engagement, "
        "predicted conversion, merchant quality, personalization, and sponsored ad value."
    )


with tab3:
    st.header("Model Performance")

    metric_table = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )

    st.dataframe(metric_table, use_container_width=True)

    st.subheader("Key Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CTR Random Forest AUC",
        f"{metrics['ctr_random_forest_auc']:.3f}",
    )

    col2.metric(
        "CTR Log Loss",
        f"{metrics['ctr_random_forest_log_loss']:.3f}",
    )

    col3.metric(
        "CVR Random Forest AUC",
        f"{metrics['cvr_random_forest_auc']:.3f}",
    )


with tab4:
    st.header("A/B Testing Simulation")

    ab_table = pd.DataFrame(
        {
            "Metric": list(ab_results.keys()),
            "Value": list(ab_results.values()),
        }
    )

    st.dataframe(ab_table, use_container_width=True)

    st.subheader("Business Impact")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CTR Lift",
        f"{ab_results['ctr_lift'] * 100:.1f}%",
    )

    col2.metric(
        "CVR Lift",
        f"{ab_results['cvr_lift'] * 100:.1f}%",
    )

    col3.metric(
        "Revenue Proxy Lift",
        f"{ab_results['revenue_proxy_lift'] * 100:.1f}%",
    )