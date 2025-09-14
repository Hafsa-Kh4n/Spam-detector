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
        vectorized = tfidf.transform([email_text])
        prediction = model.predict(vectorized)[0]
        
        if prediction == 1:
            st.error("🚨 This email is classified as SPAM / harmful!")
        else:
            st.success("✅ This email seems SAFE.")
