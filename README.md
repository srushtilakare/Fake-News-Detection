# 📰 Fake News Detector Pro

**Fake News Detector Pro** is an interactive web application built using **Streamlit** and **Machine Learning** to classify news articles as **Fake** or **Real**. It supports multiple input methods including text, URLs, uploaded files, and random examples for testing.

This project leverages a pre-trained ML model with **TF-IDF vectorization**, providing high-confidence predictions and real-time analysis.

---

## 🚀 Features

- ✅ **Multiple Input Methods**: Text, URL, File (.txt/.csv), or Random Examples
- ✅ **Real-time News Classification**: Detects Fake vs Real news with confidence score
- ✅ **History Logging**: Tracks all analyzed articles with timestamp
- ✅ **Prediction Summary**: View prediction distribution in table & charts
- ✅ **File Upload**: Batch prediction for CSV files
- ✅ **Expandable Article Preview**: Check extracted text from URLs
- ✅ **Random Example Testing**: Try pre-selected real/fake news examples

---

## 🛠 Tech Stack

- **Frontend & App**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: scikit-learn (Any classifier + TF-IDF)
- **Text Processing**: NLTK, Newspaper3k
- **Data Handling**: Pandas
- **Model Storage**: Joblib

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/srushtilakare/fake-news-detection.git
cd fake-news-detecton
````

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download necessary NLTK data:

```python
import nltk
nltk.download('punkt')
```

5. Download SpaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## ⚙ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and start analyzing news articles.

---

## 📝 File Structure

```
fake-news-detector/
│
├── app.py                 # Main Streamlit application
├── model/
│   ├── fake_news_model.pkl  # Pre-trained ML model
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── README.md
└── examples/              # Optional: Example news text files
```

---

## 💡 How It Works

1. User inputs news via **text, URL, or file**.
2. The app pre-processes the text and applies **TF-IDF vectorization**.
3. The pre-trained ML model predicts **Fake** or **Real** with confidence.
4. Results are displayed along with a **history log** and **charts** for distribution.

---

## 📈 Screenshots

*(Add screenshots of your app here for visual reference)*

---

## 🔗 References

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Newspaper3k Documentation](https://newspaper.readthedocs.io/)
* [NLTK Documentation](https://www.nltk.org/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## ⚡ Future Enhancements

* Add **multi-language support** for international news.
* Integrate **advanced NLP models** (BERT / Transformers) for higher accuracy.
* Enable **user authentication** to save personal analysis history.
* Export **history & analysis results** as CSV/PDF.

---
