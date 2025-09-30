import streamlit as st
import joblib
from newspaper import Article
import pandas as pd
import datetime
import random
import nltk

# --- NLTK punkt download (for newspaper3k) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Fake News Detector Pro 📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model & vectorizer
try:
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: Model or Vectorizer files not found. Please check paths: 'model/fake_news_model.pkl' and 'model/vectorizer.pkl'")
    st.stop()

# --- SESSION STATE for History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- MODEL PREDICTION FUNCTION ---
@st.cache_data(show_spinner=False)
def predict_news(text):
    """Predicts news class and confidence."""
    if not text or not text.strip():
        return "N/A", 0.0
    try:
        vec_input = vectorizer.transform([text])
        prediction = model.predict(vec_input)[0]
        proba = model.predict_proba(vec_input)[0]
        confidence = round(max(proba)*100, 2)
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return "Error", 0.0

# --- UI ENHANCEMENTS & COMPONENTS ---
st.markdown(
    """
    <style>
    .big-font {
        font-size:36px !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="big-font">📰 Fake News Detector Pro</p>', unsafe_allow_html=True)
st.caption("A robust system utilizing Machine Learning to classify news articles as **Fake** or **Real**.")
st.markdown("---")

# --- Sidebar: Theme & Info ---
with st.sidebar:
    st.header("⚙️ Settings")
    theme = st.radio("App Theme:", ["Light", "Dark"], index=0)

    if theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp {background-color: #0E1117; color: #FAFAFA;}
            h1, h2, h3, h4, h5, p, label {color: #FAFAFA !important;}
            .stTextArea > div > textarea, .stTextInput > div > div > input, .stSelectbox > div > div {
                background-color: #1F2430; color: #FAFAFA; border-color: #333;
            }
            .stButton > button {color: #FAFAFA; background-color: #1F2430; border-color: #333;}
            .st-emotion-cache-1wq4rvf {background-color: #1F2430;}
            </style>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Model Info")
    st.info("This application uses a pre-trained ML classifier trained on a large dataset of fake and real news articles.")

# --- Input Selection ---
input_options = ["Enter News Text", "Enter URL", "Upload File", "Try Random Example"]
option = st.selectbox("Select your Input Method:", input_options, key="input_method_select")

result_placeholder = st.empty()

# --- Enter Text ---
if option == "Enter News Text":
    st.info("Paste the full text of the article below.")
    news_text = st.text_area("News Content:", height=250, placeholder="Paste your news article text here...")
    if st.button("🔍 Analyze Text", use_container_width=True):
        if news_text.strip():
            with st.spinner('Analyzing content...'):
                pred, conf = predict_news(news_text)
            with result_placeholder.container():
                if pred != "Error":
                    st.success(f"✅ Analysis Complete: The news is likely **{pred}** with **{conf}%** confidence.")
                    st.session_state.history.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Text", news_text[:60]+"...", pred, conf])
        else:
            st.warning("Please paste some text to analyze.")

# --- Enter URL ---
elif option == "Enter URL":
    st.info("Provide the web link to the news article. Uses the `newspaper3k` library to extract content.")
    url = st.text_input("Article URL:", placeholder="e.g., https://www.nytimes.com/...")
    if st.button("🔗 Fetch & Analyze URL", use_container_width=True):
        if url.strip():
            try:
                with st.spinner('Fetching and analyzing article...'):
                    article = Article(url)
                    article.download()
                    article.parse()
                    article_text = article.text
                    if not article_text.strip():
                        st.error("❌ Failed to extract content from the URL.")
                        st.stop()
                    pred, conf = predict_news(article_text)
                with result_placeholder.container():
                    st.success(f"✅ Analysis Complete: The news is likely **{pred}** with **{conf}%** confidence.")
                    with st.expander("Extracted Article Snippet"):
                        st.text(article_text[:600]+"...")
                    st.session_state.history.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "URL", url, pred, conf])
            except Exception as e:
                st.error(f"❌ Failed to fetch or parse article. Error: {e}")
        else:
            st.warning("Please enter a valid URL.")

# --- File Upload ---
elif option == "Upload File":
    uploaded = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"])
    if uploaded:
        with st.spinner('Processing file...'):
            if uploaded.name.endswith(".txt"):
                text = uploaded.read().decode("utf-8")
                pred, conf = predict_news(text)
                with result_placeholder.container():
                    st.success(f"✅ Analysis Complete: The news is likely **{pred}** with **{conf}%** confidence.")
                    st.session_state.history.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File (TXT)", uploaded.name, pred, conf])
            elif uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
                if "text" in df.columns:
                    texts_to_predict = df["text"].fillna("").astype(str).tolist()
                    vec_input = vectorizer.transform(texts_to_predict)
                    predictions = model.predict(vec_input)
                    probabilities = model.predict_proba(vec_input)
                    df["Prediction"] = predictions
                    df["Confidence (%)"] = [round(max(p)*100,2) for p in probabilities]
                    with result_placeholder.container():
                        st.subheader("CSV Batch Analysis Results")
                        st.dataframe(df, use_container_width=True)
                        for _, row in df.head(5).iterrows():
                            snippet = str(row["text"])[:60]+"..." if len(str(row["text"]))>60 else str(row["text"])
                            st.session_state.history.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File (CSV)", snippet, row["Prediction"], row["Confidence (%)"]])
                else:
                    st.warning("CSV must contain a column named 'text' for analysis.")

# --- Random Example ---
elif option == "Try Random Example":
    st.info("Click the buttons to see how the model classifies pre-selected examples.")
    examples = {
        "Fake": [
            "COVID-19 vaccines contain microchips that control humans via 5G.",
            "Eating garlic cures all types of flu and common cold immediately.",
            "A secret cabal of global elites is planning to drastically reduce the world population."
        ],
        "Real": [
            "NASA’s Perseverance rover collected its 25th rock sample on Mars.",
            "The Federal Reserve announced a quarter-point rate hike to combat rising inflation.",
            "A new study published in 'Nature' details a breakthrough in gene-editing technology."
        ]
    }
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Test a Fake Example", use_container_width=True):
            ex_text = random.choice(examples["Fake"])
            pred, conf = predict_news(ex_text)
            with result_placeholder.container():
                st.info(f"**Example:** {ex_text}")
                st.error(f"🔴 Prediction: **{pred}** ({conf}% confidence)")
                st.session_state.history.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Example", "Fake Example", pred, conf])
    with col2:
        if st.button("✅ Test a Real Example", use_container_width=True):
            ex_text = random.choice(examples["Real"])
            pred, conf = predict_news(ex_text)
            with result_placeholder.container():
                st.info(f"**Example:** {ex_text}")
                st.success(f"🟢 Prediction: **{pred}** ({conf}% confidence)")
                st.session_state.history.append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Example", "Real Example", pred, conf])

# --- History Section ---
st.markdown("---")
st.subheader("📜 Check History & Summary")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history, columns=["Time","Type","Input Snippet","Prediction","Confidence (%)"])
    st.dataframe(hist_df.iloc[::-1], use_container_width=True, height=200)
    st.markdown("#### Prediction Distribution")
    col3, col4 = st.columns([1,2])
    with col3:
        summary_counts = hist_df["Prediction"].value_counts().reset_index()
        summary_counts.columns = ["Prediction","Count"]
        st.dataframe(summary_counts, hide_index=True)
    with col4:
        st.bar_chart(hist_df["Prediction"].value_counts(), use_container_width=True)
else:
    st.info("No checks recorded yet. Start testing some news articles!")

st.markdown("---")
