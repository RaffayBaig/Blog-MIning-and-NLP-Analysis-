import io
import contextlib
import pandas as pd
import streamlit as st

from scraper import scrape_multiple_blogs, save_to_csv
from preprocess import preprocess_csv
from topic_modelling import extract_keywords, perform_topic_modeling
from motive_inference import analyze_motives
from sentiment_analysis import analyze_sentiment
from comment_generator import generate_comments_for_blogs
from visualisation import (
    plot_sentiment_distribution,
    plot_motive_distribution,
    generate_wordcloud,
    plot_sentiment_scores
)

st.set_page_config(
    page_title="Blog Mining & NLP Analysis",
    page_icon="◈",
    layout="wide"
)

# ============================================================
# THEME -- HUD / data-terminal identity
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void: #05070d;
    --bg-panel: #0b0f1a;
    --cyan: #00e5ff;
    --magenta: #ff3d81;
    --violet: #8b5cf6;
    --text: #eaf6ff;
    --muted: #5c7088;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(0,229,255,0.07), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(139,92,246,0.07), transparent 40%),
        var(--bg-void);
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Chakra Petch', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: 0.02em;
}

p, span, label, .stMarkdown, div[data-testid="stMarkdownContainer"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* HUD corner-bracket panel -- the signature element */
.hud-panel {
    position: relative;
    border: 1px solid rgba(0, 229, 255, 0.25);
    background: rgba(11, 15, 26, 0.6);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border-radius: 2px;
}
.hud-panel::before, .hud-panel::after {
    content: "";
    position: absolute;
    width: 14px;
    height: 14px;
    border-color: var(--cyan);
    border-style: solid;
    opacity: 0.9;
}
.hud-panel::before {
    top: -1px; left: -1px;
    border-width: 2px 0 0 2px;
}
.hud-panel::after {
    bottom: -1px; right: -1px;
    border-width: 0 2px 2px 0;
}

.hud-title {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hud-title::before {
    content: "";
    width: 6px; height: 6px;
    background: var(--cyan);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--cyan);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* Metric strip */
.metric-box {
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
    padding: 0.8rem 0;
}
.metric-value {
    font-size: 1.9rem;
    color: var(--cyan);
    font-weight: 500;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Inputs */
.stTextInput input {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    font-family: 'JetBrains Mono', monospace;
}

.stButton button {
    background: transparent !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    font-family: 'Chakra Petch', sans-serif !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.stButton button:hover {
    background: rgba(0, 229, 255, 0.1) !important;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.35);
}

div[data-testid="stExpander"] {
    background: var(--bg-panel);
    border: 1px solid rgba(139, 92, 246, 0.25);
}

hr { border-color: rgba(0, 229, 255, 0.15); }
</style>
""", unsafe_allow_html=True)


def hud_section(title):
    st.markdown(f'<div class="hud-title">{title}</div>', unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown('<h1 style="margin-bottom:0;">◈ BLOG MINING &amp; NLP ANALYSIS</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#5c7088;font-family:JetBrains Mono, monospace;font-size:0.85rem;">'
    'SCRAPE // PREPROCESS // MODEL // INFER // GENERATE</p>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

col_input, col_slider = st.columns([2, 1])
with col_input:
    keyword = st.text_input("TARGET KEYWORD", placeholder="e.g. cat")
with col_slider:
    num_results = st.slider("BLOG COUNT", min_value=5, max_value=30, value=10)

run_button = st.button("▶ RUN ANALYSIS")

if run_button and keyword:

    # ---------------- SCRAPING ----------------
    with st.spinner("Scanning sources..."):
        blogs = scrape_multiple_blogs(keyword=keyword, num_results=num_results)

    if not blogs or len(blogs) == 0:
        st.error("No blogs scraped. Try a different keyword.")
        st.stop()

    save_to_csv(blogs)

    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    hud_section("Data Collection")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-box"><div class="metric-value">{len(blogs)}</div><div class="metric-label">Blogs Scraped</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-value">{len(set(b["domain"] for b in blogs))}</div><div class="metric-label">Domains</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="metric-value">{sum(b["comment_count"] for b in blogs)}</div><div class="metric-label">Comments Found</div></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(blogs)[["title", "domain", "comment_count"]], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PREPROCESSING ----------------
    with st.spinner("Cleaning signal..."):
        processed_df = preprocess_csv()

    if processed_df is None or processed_df.empty:
        st.error("Preprocessing failed or returned no data.")
        st.stop()

    st.info(f"Remaining blogs after cleaning: {len(processed_df)}")

    # ---------------- KEYWORD EXTRACTION ----------------
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    hud_section("Keyword Extraction")
    with st.spinner("Extracting keywords..."):
        keywords = extract_keywords(processed_df)
        st.markdown(
            " &nbsp;·&nbsp; ".join([f"`{k}`" for k in keywords])
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TOPIC MODELING ----------------
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    hud_section("Topics Discovered")
    with st.spinner("Running topic modeling..."):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            perform_topic_modeling(processed_df)
        st.code(buffer.getvalue(), language=None)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- MOTIVE INFERENCE ----------------
    with st.spinner("Inferring motives..."):
        motive_df = analyze_motives(processed_df)

    # ---------------- SENTIMENT ANALYSIS ----------------
    with st.spinner("Analyzing sentiment..."):
        sentiment_df = analyze_sentiment(processed_df)

    # ---------------- VISUALIZATIONS ----------------
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    hud_section("Visual Intelligence")

    with st.spinner("Rendering visuals..."):
        sentiment_fig = plot_sentiment_distribution(sentiment_df)
        motive_fig = plot_motive_distribution(motive_df)
        score_fig = plot_sentiment_scores(sentiment_df)
        wordcloud_path = generate_wordcloud(processed_df)

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(sentiment_fig, use_container_width=True)
    with row1_col2:
        st.plotly_chart(motive_fig, use_container_width=True)

    st.plotly_chart(score_fig, use_container_width=True)
    st.image(wordcloud_path, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- MOTIVE / SENTIMENT TABLES ----------------
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    hud_section("Motive & Sentiment Detail")
    tbl_col1, tbl_col2 = st.columns(2)
    with tbl_col1:
        st.dataframe(motive_df[["title", "motive"]], use_container_width=True)
    with tbl_col2:
        st.dataframe(sentiment_df[["title", "sentiment", "sentiment_score"]], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- AI COMMENT GENERATION ----------------
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    hud_section("AI-Generated Comments")
    with st.spinner("Generating AI comments (this can take a bit)..."):
        comments_df = generate_comments_for_blogs(processed_df)

    for _, row in comments_df.iterrows():
        with st.expander(row["title"]):
            st.write(row["ai_generated_comment"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- DOWNLOAD ----------------
    csv = comments_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ DOWNLOAD FULL DATASET (CSV)",
        csv,
        "final_blog_data.csv",
        "text/csv"
    )

elif run_button and not keyword:
    st.warning("Enter a keyword first.")