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

# --- Input Box with Session State ---
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

email_text = st.text_area(
    "✉️ Paste Email Text Here",
    value=st.session_state.email_text,  # This will now show empty after clearing
    height=200,
    placeholder="Type or paste email content..."
)

# --- Buttons in Columns (Side by Side) ---
col1, col2 = st.columns([1,1])
with col1:
    check_btn = st.button("🔍 Check Email")
with col2:
    clear_btn = st.button("🧹 Clear Text (double click)")

# --- Clear Button Functionality ---
if clear_btn:
    st.session_state.email_text = ""  # Reset text


# --- Prediction Logic ---
if check_btn:
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text to check.")
    else:
        st.session_state.email_text = email_text  # Keep text after check
        
        # --- Prediction ---
        vectorized = tfidf.transform([email_text])
        prediction = model.predict(vectorized)[0]

        # --- Post-processing rule for trusted senders ---
        trusted_senders = ["google", "gmail", "gemini", "microsoft", "github", "linkedin","streamlit"]
        if any(word in email_text.lower() for word in trusted_senders):
            prediction = 0  # Force mark as SAFE

        # --- Display result ---
      spam_prob = model.predict_proba(vectorized)[0][1]
st.write(f"📊 **Spam Probability:** {spam_prob:.2%}")

            if spam_prob > 0.7:  # Less strict threshold
                st.error("🚨 This email is **harmful!**")
            else:
                  st.success("✅ This email seems **SAFE**.")


# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px; color: gray;'>Made with ❤️ using Streamlit | AI-Powered Email Classification</p>",
    unsafe_allow_html=True,
)










