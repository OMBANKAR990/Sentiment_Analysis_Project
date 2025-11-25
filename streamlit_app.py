# streamlit_app.py
import streamlit as st
from transformers import pipeline
from typing import Dict

# Cache the model so it loads only once across reruns
@st.cache_resource
def load_model(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    # default uses a small, fast sentiment model
    return pipeline("sentiment-analysis", model=model_name)

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("📊 Sentiment Analyzer (Hugging Face + Streamlit)")

st.markdown(
    """
Enter text below and click **Analyze**.
- Uses Hugging Face `transformers.pipeline("sentiment-analysis")`.
- Default model: `distilbert-base-uncased-finetuned-sst-2-english` (small & fast).
"""
)

# Sidebar options
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model",
        (
            "distilbert-base-uncased-finetuned-sst-2-english",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "cardiffnlp/twitter-roberta-base-sentiment"
        ),
        index=0,
        help="Pick a smaller model for faster startup. You can also enter a custom model name."
    )
    custom_model = st.text_input("Custom model (optional)", value="")
    if custom_model.strip():
        model_choice = custom_model.strip()

# Load model (cached)
with st.spinner("Loading model — this may take 10–40 seconds the first time..."):
    classifier = load_model(model_choice)

# Text input
text = st.text_area("Enter text to analyze", value="The product quality is terrible. I'm disappointed.", height=140)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Classifying..."):
                try:
                    result = classifier(text)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    result = None

            if result:
                # pipeline returns a list e.g. [{'label': 'NEGATIVE', 'score': 0.998}]
                r = result[0] if isinstance(result, list) else result
                label = r.get("label", "")
                score = r.get("score", None)
                st.metric(label=f"Label: {label}", value=f"{score:.3f}" if score is not None else "N/A")
                st.write("Raw output:")
                st.json(result)

with col2:
    st.markdown("### Examples")
    st.write("- I love this product! 😍")
    st.write("- The food was awful and cold.")
    st.write("- It's okay, not the best but functional.")
    st.write("### Notes")
    st.write("- First model load downloads weights and can be slow. Subsequent calls are fast.")
    st.write("- To reduce startup time use a smaller model or the Hugging Face Inference API (requires token).")

# Option: process CSV upload
st.markdown("---")
st.header("Batch mode (CSV)")
st.write("Upload a CSV with a column named `text` to analyze multiple rows.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must contain a column named `text`.")
    else:
        st.write("Preview:")
        st.dataframe(df.head())
        if st.button("Analyze batch"):
            with st.spinner("Classifying rows..."):
                preds = []
                for t in df["text"].astype(str).tolist():
                    preds.append(classifier(t)[0])
                # expand preds to columns
                df["label"] = [p.get("label") for p in preds]
                df["score"] = [p.get("score") for p in preds]
                st.success("Done — preview:")
                st.dataframe(df.head())
                st.download_button("Download results as CSV", df.to_csv(index=False).encode("utf-8"), "sentiment_results.csv")
