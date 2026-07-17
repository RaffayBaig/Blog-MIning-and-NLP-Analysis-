# 🔎 Blog Mining & NLP Analysis System

An end-to-end pipeline that scrapes blogs for a given keyword, processes and analyzes them using NLP techniques, and generates AI-powered insights — wrapped in a futuristic Streamlit dashboard.


## 🚀 Features

- **Web Scraping** — discovers and scrapes blogs for any keyword using DuckDuckGo search + BeautifulSoup, with a Selenium fallback for JS-rendered comment sections
- **Text Preprocessing** — cleans and normalizes scraped content (stopword removal, lemmatization via spaCy)
- **Keyword Extraction** — TF-IDF based top-keyword identification
- **Topic Modeling** — LDA (Latent Dirichlet Allocation) to surface underlying themes across blogs
- **Motive Inference** — classifies each blog's intent (Tutorial / Promotional / Opinion / Informative)
- **Sentiment Analysis** — VADER-based sentiment scoring and classification
- **AI-Generated Comments** — uses Groq's LLM API (Llama 3.1) to generate human-like comments on each blog
- **Interactive Visualizations** — Plotly-powered sentiment distribution, motive breakdown, and score histograms, plus a neon-styled word cloud
- **Streamlit Frontend** — a full HUD-style dashboard to run the entire pipeline from the browser

<img width="1600" height="434" alt="image" src="https://github.com/user-attachments/assets/5a8df9d0-f33c-4f09-83c1-23abddeddbd9" />

## 🗂️ Project Structure

```
DM-FINAL/
├── src/
│   ├── app.py                  # Streamlit frontend
│   ├── main.py                 # CLI entry point
│   ├── scraper.py              # Blog discovery + scraping (requests + Selenium fallback)
│   ├── preprocess.py           # Text cleaning and normalization
│   ├── topic_modelling.py      # Keyword extraction + LDA topic modeling
│   ├── motive_inference.py     # Rule-based motive classification
│   ├── sentiment_analysis.py   # VADER sentiment scoring
│   ├── comment_generator.py    # Groq LLM-powered comment generation
│   └── visualisation.py        # Plotly + word cloud visualizations
├── data/
│   ├── raw/                    # Scraped blog data
│   └── processed/              # Cleaned + final datasets
├── visuals/                    # Saved chart exports
├── requirements.txt
└── .env                        # GROQ_API_KEY (not committed)
```

## 🛠️ Tech Stack

- **Language:** Python
- **Scraping:** `requests`, `BeautifulSoup`, `selenium`, `ddgs`
- **NLP:** `nltk`, `spaCy`, `scikit-learn` (TF-IDF, LDA)
- **LLM:** Groq API (Llama 3.1 8B Instant)
- **Visualization:** `plotly`, `matplotlib`, `wordcloud`
- **Frontend:** `Streamlit`

## ⚙️ Setup

1. Clone the repo:
```bash
git clone https://github.com/RaffayBaig/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## ▶️ Usage

**Run via CLI:**
```bash
python src/main.py
```

**Run via Streamlit dashboard:**
```bash
streamlit run src/app.py
```

<img width="1600" height="415" alt="image" src="https://github.com/user-attachments/assets/cd3370b0-7f50-4994-99ba-4d02467562d1" />


## 📊 Pipeline Overview

```
Keyword Input
   ↓
Blog Scraping (requests + Selenium fallback)
   ↓
Text Preprocessing (cleaning, lemmatization)
   ↓
Keyword Extraction (TF-IDF) + Topic Modeling (LDA)
   ↓
Motive Inference + Sentiment Analysis
   ↓
Visualization (Plotly charts + word cloud)
   ↓
AI Comment Generation (Groq LLM)
   ↓
Final Dataset Export (CSV)
```

## 📌 Status

🔄 Actively developed — this project was built as part of a Data Mining course, exploring the full pipeline from raw web data to structured NLP insights.

## 📄 License

This project is for educational purposes.

---

*Feel free to open issues or suggest improvements!*
