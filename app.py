import streamlit as st
import joblib

# Load model & vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_rf_model.pkl")

# --- Page Config ---
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# --- App Title ---
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📧 Email Spam Detection</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Enter an email below and instantly find out if it's SPAM or SAFE ✅</p>",
    unsafe_allow_html=True,
)

# --- Input Box ---
email_text = st.text_area("✉️ Paste Email Text Here", height=200, placeholder="Type or paste email content...")

# --- Prediction Button ---
if st.button("🔍 Check Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text to check.")
    else:
        vectorized = tfidf.transform([email_text])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 **This email is classified as SPAM / harmful!**")
        else:
            st.success("✅ **This email seems SAFE. No malicious intent detected.**")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px; color: gray;'>Made with ❤️ using Streamlit | AI-Powered Email Classification</p>",
    unsafe_allow_html=True,
)

