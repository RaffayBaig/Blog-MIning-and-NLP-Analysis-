import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================
# THEME TOKENS -- keep these in sync with the CSS in app.py
# ============================================================

BG_VOID = "#05070d"
BG_PANEL = "#0b0f1a"
CYAN = "#00e5ff"
MAGENTA = "#ff3d81"
VIOLET = "#8b5cf6"
GREEN = "#39ff88"
TEXT = "#eaf6ff"
MUTED = "#5c7088"
GRID = "rgba(0, 229, 255, 0.08)"

FONT_FAMILY = "JetBrains Mono, monospace"

os.makedirs("visuals", exist_ok=True)


def _base_layout(title):
    return dict(
        title=dict(
            text=title,
            font=dict(family="Chakra Petch, sans-serif", size=20, color=TEXT),
            x=0.02,
            xanchor="left"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=TEXT, size=13),
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=12)
        )
    )


# =========================================
# SENTIMENT DISTRIBUTION (donut)
# =========================================

def plot_sentiment_distribution(sentiment_df):

    counts = sentiment_df["sentiment"].value_counts()

    color_map = {"Positive": CYAN, "Negative": MAGENTA, "Neutral": VIOLET}
    colors = [color_map.get(label, MUTED) for label in counts.index]

    fig = go.Figure(
        data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=BG_VOID, width=3)),
            textfont=dict(family=FONT_FAMILY, color=TEXT, size=13),
            hoverinfo="label+percent+value"
        )]
    )

    fig.update_layout(**_base_layout("SENTIMENT DISTRIBUTION"))
    fig.add_annotation(
        text=f"{len(sentiment_df)}<br><span style='font-size:11px;color:{MUTED}'>BLOGS</span>",
        showarrow=False,
        font=dict(family=FONT_FAMILY, size=26, color=TEXT),
        x=0.5, y=0.5
    )

    try:
        fig.write_image("visuals/sentiment_pie_chart.png", width=900, height=600, scale=2)
    except Exception:
        pass  # kaleido not installed -- safe to skip static export

    return fig


# =========================================
# MOTIVE DISTRIBUTION (horizontal bar)
# =========================================

def plot_motive_distribution(motive_df):

    counts = motive_df["motive"].value_counts().sort_values()

    fig = go.Figure(
        data=[go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker=dict(
                color=counts.values,
                colorscale=[[0, VIOLET], [0.5, CYAN], [1, GREEN]],
                line=dict(color=BG_VOID, width=1)
            ),
            text=counts.values,
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, color=TEXT)
        )]
    )

    layout = _base_layout("MOTIVE DISTRIBUTION")
    layout["showlegend"] = False
    layout["xaxis"] = dict(showgrid=True, gridcolor=GRID, color=MUTED, title="Count")
    layout["yaxis"] = dict(showgrid=False, color=TEXT)
    fig.update_layout(**layout)

    try:
        fig.write_image("visuals/motive_bar_chart.png", width=900, height=500, scale=2)
    except Exception:
        pass

    return fig


# =========================================
# SENTIMENT SCORE DISTRIBUTION (gradient histogram)
# =========================================

def plot_sentiment_scores(sentiment_df):

    scores = sentiment_df["sentiment_score"]

    fig = go.Figure(
        data=[go.Histogram(
            x=scores,
            nbinsx=12,
            marker=dict(
                color=scores,
                colorscale=[[0, MAGENTA], [0.5, VIOLET], [1, CYAN]],
                cmin=-1, cmax=1,
                line=dict(color=BG_VOID, width=1)
            )
        )]
    )

    layout = _base_layout("SENTIMENT SCORE DISTRIBUTION")
    layout["showlegend"] = False
    layout["xaxis"] = dict(
        showgrid=True, gridcolor=GRID, color=MUTED,
        title="Compound Score", range=[-1, 1]
    )
    layout["yaxis"] = dict(showgrid=True, gridcolor=GRID, color=MUTED, title="Frequency")
    fig.update_layout(**layout)

    try:
        fig.write_image("visuals/sentiment_histogram.png", width=900, height=500, scale=2)
    except Exception:
        pass

    return fig


# =========================================
# WORD CLOUD (dark, neon colormap)
# =========================================

def _neon_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    palette = [CYAN, MAGENTA, VIOLET, GREEN, "#5ce1ff"]
    return palette[random_state.randint(0, len(palette) - 1)] if random_state else palette[0]


def generate_wordcloud(df):

    text = " ".join(df["cleaned_content"])

    wc = WordCloud(
        width=1400,
        height=700,
        background_color=BG_VOID,
        color_func=_neon_color_func,
        prefer_horizontal=0.9,
        max_words=120,
        random_state=42
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_VOID)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.savefig("visuals/wordcloud.png", facecolor=BG_VOID, bbox_inches="tight")
    plt.close(fig)

    return "visuals/wordcloud.png"