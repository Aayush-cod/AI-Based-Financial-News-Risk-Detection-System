import streamlit as st
import pickle
import re
import numpy as np

# ===============================
# Load Saved Model & Vectorizer
# ===============================

with open("financial_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="AI Financial Risk Detector",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Based Financial News Risk Detection System")
st.markdown("Analyze financial news sentiment and detect potential market manipulation risks.")


# ===============================
# Risk Scoring Logic
# ===============================

def calculate_risk(sentiment, confidence):
    if sentiment == "negative" and confidence > 0.75:
        return "HIGH"
    elif sentiment == "positive" and confidence > 0.80:
        return "MEDIUM"
    else:
        return "LOW"


# ===============================
# Suspicious Words Detection
# ===============================

suspicious_keywords = [
    "surge", "plunge", "collapse", "soar",
    "record high", "crash", "doubled",
    "skyrocket", "drop", "volatile"
]

def highlight_keywords(text):
    found_words = []
    for word in suspicious_keywords:
        if word in text.lower():
            found_words.append(word)
    return found_words


# ===============================
# Summary Generation
# ===============================

def generate_summary(sentiment, confidence, risk):
    return f"""
This financial article shows a **{sentiment.upper()}** sentiment 
with a confidence level of **{confidence:.2%}**.

Based on linguistic signals and financial indicators, 
the market risk level is assessed as **{risk}**.

Investors should carefully evaluate the credibility 
and potential volatility implications of this news.
"""


# ===============================
# User Input
# ===============================

user_input = st.text_area("📰 Enter Financial News Text Below:", height=200)

if st.button("Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter financial news text.")
    else:

        # Vectorize input
        input_vector = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vector)[0]

        # Confidence (if Logistic Regression)
        try:
            probabilities = model.predict_proba(input_vector)
            confidence = np.max(probabilities)
        except:
            confidence = 0.75  # fallback for SVM

        sentiment_label = label_encoder.inverse_transform([prediction])[0]

        risk_level = calculate_risk(sentiment_label, confidence)

        suspicious_words = highlight_keywords(user_input)

        summary = generate_summary(sentiment_label, confidence, risk_level)

        # ===============================
        # Display Results
        # ===============================

        st.subheader("📌 Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Sentiment", sentiment_label.capitalize())
        col2.metric("Confidence", f"{confidence:.2%}")
        col3.metric("Risk Level", risk_level)

        st.markdown("---")

        st.subheader("🧠 AI Generated Summary")
        st.markdown(summary)

        st.markdown("---")

        st.subheader("⚠ Suspicious Words Detected")

        if suspicious_words:
            st.write(", ".join(suspicious_words))
        else:
            st.write("No high-risk financial keywords detected.")