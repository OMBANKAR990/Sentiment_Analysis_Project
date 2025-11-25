import streamlit as st
from transformers import pipeline

# Load model once
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

# Beautiful title
st.markdown(
    """
    <h1 style='text-align: center; color: #4a4a4a;'>🧠 Sentiment Analyzer</h1>
    <p style='text-align: center; color: gray;'>Enter any sentence or review below — the model will detect the sentiment instantly.</p>
    """,
    unsafe_allow_html=True
)

# Input box
text = st.text_area(
    "Enter your text here 👇",
    placeholder="Example: The product quality is terrible, I'm disappointed.",
    height=160
)

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]
        label = result["label"]
        score = round(result["score"], 3)

        # Beautiful output
        st.markdown(
            f"""
            <div style="padding:18px; border-radius:10px; background-color:#f8f9fa; border:1px solid #e2e2e2;">
                <h3 style="color:#2b2b2b; text-align:center;">Result</h3>
                <p style="font-size:22px; text-align:center; margin-top:-5px;">
                    <b>Sentiment:</b> <span style="color:#0073e6;">{label}</span> <br>
                    <b>Confidence Score:</b> {score}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠ Please enter some text.")
