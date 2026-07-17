import re
import pandas as pd
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STOPWORDS
stop_words = set(stopwords.words("english"))


def clean_text(text):

    # HANDLE NON-STRING VALUES
    if not isinstance(text, str):
        return ""

    # CONVERT TO LOWERCASE
    text = text.lower()

    # REMOVE URLS
    text = re.sub(r"http\S+", "", text)

    # REMOVE NUMBERS
    text = re.sub(r"\d+", "", text)

    # REMOVE SPECIAL CHARACTERS
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # REMOVE EXTRA SPACES
    text = re.sub(r"\s+", " ", text).strip()

    # TOKENIZATION
    tokens = word_tokenize(text)

    # REMOVE STOPWORDS + SHORT WORDS
    custom_stopwords = {
    "get", "use", "make", "know", "let", "take",
    "want", "need", "like", "also", "one", "two"
}

    tokens = [
      word for word in tokens
      if word not in stop_words
      and word not in custom_stopwords
      and len(word) > 3
]
       

    # LEMMATIZATION
    doc = nlp(" ".join(tokens))

    cleaned_tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ != "-PRON-"
    ]

    return " ".join(cleaned_tokens)


def preprocess_csv():

    # LOAD RAW DATA
    df = pd.read_csv("data/raw/blogs.csv")

    print("Raw data loaded")

    # HANDLE EMPTY VALUES
    df["content"] = df["content"].fillna("")
    df["title"] = df["title"].fillna("")

    # REMOVE INVALID TITLES
    df = df[df["title"] != "No Title"]

    # REMOVE SHORT TITLES
    df = df[df["title"].str.len() > 10]

    # REMOVE CLOUDFLARE / BLOCKED PAGES
    df = df[
        ~df["title"].str.contains(
            "Cloudflare",
            case=False,
            na=False
        )
    ]

    # REMOVE EMPTY CONTENT
    df = df[df["content"].str.strip() != ""]

    # REMOVE VERY SHORT CONTENT
    df = df[
        df["content"].apply(
            lambda x: len(str(x).split()) > 50
        )
    ]

    # CLEAN CONTENT
    df["cleaned_content"] = df["content"].apply(clean_text)

    # REMOVE LOW-INFORMATION CLEANED TEXT
    df = df[
        df["cleaned_content"].apply(
            lambda x: len(x.split()) > 30
        )
    ]

    # RESET INDEX
    df = df.reset_index(drop=True)

    # SAVE PROCESSED DATA
    output_path = "data/processed/processed_blogs.csv"

    df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")

    print(f"Remaining blogs after cleaning: {len(df)}")

    return df