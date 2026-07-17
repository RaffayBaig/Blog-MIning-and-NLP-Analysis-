import pandas as pd
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download("vader_lexicon")


sia = SentimentIntensityAnalyzer()

def get_sentiment(text):

    score = sia.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"



def analyze_sentiment(df):

    df["sentiment"] = df["cleaned_content"].apply(get_sentiment)

    # ADD THIS LINE (IMPORTANT)
    df["sentiment_score"] = df["cleaned_content"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    print("\nSENTIMENT ANALYSIS RESULT:\n")

    print(df[["title", "sentiment", "sentiment_score"]])

    return df
