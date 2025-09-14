import streamlit as st
import joblib

# Load model & vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_rf_model.pkl")

st.title("📧 Email Spam Detection App")
st.write("Paste an email text and check if it's spam or safe!")

email_text = st.text_area("✉️ Enter email text:")

if st.button("🔍 Check Email"):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Transform input
        vectorized = tfidf.transform([email_text])

        # ✅ Get spam probability
        spam_prob = model.predict_proba(vectorized)[:, 1][0]

        # ✅ Use custom threshold (0.6)
        prediction = 1 if spam_prob >= 0.6 else 0

        if prediction == 1:
            st.error(f"🚨 This email is classified as SPAM / harmful! (Spam Probability: {spam_prob:.2f})")
        else:
            st.success(f"✅ This email seems SAFE. (Spam Probability: {spam_prob:.2f})")

