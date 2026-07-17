import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.decomposition import (
    LatentDirichletAllocation
)



def extract_keywords(df):

    vectorizer = TfidfVectorizer(
        max_features=20
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["cleaned_content"]
    )

    keywords = vectorizer.get_feature_names_out()

    print("\nTOP KEYWORDS:\n")

    for word in keywords:

        print(word)

    return keywords

def perform_topic_modeling(df):

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=1,
        stop_words="english"
    )

    document_term_matrix = vectorizer.fit_transform(
        df["cleaned_content"]
    )

    lda_model = LatentDirichletAllocation(
        n_components=3,
        random_state=42
    )

    lda_model.fit(document_term_matrix)

    words = vectorizer.get_feature_names_out()

    print("\nTOPICS DISCOVERED:\n")

    for topic_idx, topic in enumerate(lda_model.components_):

        print(f"\nTopic {topic_idx + 1}:")

        top_words = [
            words[i]
            for i in topic.argsort()[-10:]
        ]

        print(top_words)
        
        