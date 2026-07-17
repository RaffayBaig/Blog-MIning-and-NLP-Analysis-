from sentiment_analysis import analyze_sentiment
from motive_inference import analyze_motives
from topic_modelling import extract_keywords, perform_topic_modeling
from scraper import scrape_multiple_blogs, save_to_csv
from preprocess import preprocess_csv
from visualisation import (
    plot_sentiment_distribution,
    plot_motive_distribution,
    generate_wordcloud,
    plot_sentiment_scores
)
from comment_generator import generate_comments_for_blogs

import pandas as pd
from tabulate import tabulate

pd.set_option('display.max_colwidth', None)


def main():

    print("\n" + "="*60)
    print(" BLOG MINING & NLP ANALYSIS SYSTEM ")
    print("="*60)

    keyword = input("\nEnter keyword: ")

    # ============================================================
    # DATA COLLECTION PHASE
    # ============================================================
    print("\n" + "="*60)
    print(" DATA COLLECTION PHASE ")
    print("="*60)

    blogs = scrape_multiple_blogs(keyword=keyword, num_results=10)

    if not blogs or len(blogs) == 0:
        print("❌ No blogs scraped. Exiting...")
        return

    print(f"✅ Scraped {len(blogs)} blogs")

    save_to_csv(blogs)
    print("✅ Scraping Completed")

    # ============================================================
    # TEXT PREPROCESSING
    # ============================================================
    print("\n" + "="*60)
    print(" TEXT PREPROCESSING ")
    print("="*60)

    processed_df = preprocess_csv()

    if processed_df is None or processed_df.empty:
        print("❌ Preprocessing failed or empty data. Exiting...")
        return

    print(f"✅ Preprocessing Completed | Remaining blogs: {len(processed_df)}")

    # ============================================================
    # KEYWORD EXTRACTION
    # ============================================================
    print("\n" + "="*60)
    print(" KEYWORD EXTRACTION ")
    print("="*60)

    extract_keywords(processed_df)
    print("✅ Keyword Extraction Completed")

    # ============================================================
    # TOPIC MODELING
    # ============================================================
    print("\n" + "="*60)
    print(" TOPIC MODELING ")
    print("="*60)

    perform_topic_modeling(processed_df)
    print("✅ Topic Modeling Completed")

    # ============================================================
    # MOTIVE INFERENCE
    # ============================================================
    print("\n" + "="*60)
    print(" MOTIVE INFERENCE ")
    print("="*60)

    motive_df = analyze_motives(processed_df)

    print("✅ Motive Analysis Completed")

    print(
        tabulate(
            motive_df[["title", "motive"]],
            headers="keys",
            tablefmt="fancy_grid"
        )
    )

    # ============================================================
    # SENTIMENT ANALYSIS
    # ============================================================
    print("\n" + "="*60)
    print(" SENTIMENT ANALYSIS ")
    print("="*60)

    sentiment_df = analyze_sentiment(processed_df)

    print("✅ Sentiment Analysis Completed")

    print(
        tabulate(
            sentiment_df[["title", "sentiment"]],
            headers="keys",
            tablefmt="fancy_grid"
        )
    )

    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\n" + "="*60)
    print(" VISUALIZATION ")
    print("="*60)

    plot_sentiment_distribution(sentiment_df)
    plot_motive_distribution(motive_df)
    generate_wordcloud(processed_df)
    plot_sentiment_scores(sentiment_df)

    print("✅ Visualizations Saved")

    # ============================================================
    # LLM COMMENT GENERATION
    # ============================================================
    print("\n" + "="*60)
    print(" LLM COMMENT GENERATION ")
    print("="*60)

    comments_df = generate_comments_for_blogs(processed_df)

    for index, row in comments_df.iterrows():
        print("\n-----------------------------------")
        print(f"Blog {index + 1}")
        print(f"Title: {row['title']}")
        print(
            f"Generated Comment: "
            f"{row['ai_generated_comment'][:120]}..."
        )

    print("✅ AI Comment Generation Completed")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print(" FINAL PROJECT SUMMARY ")
    print("="*60)

    print(f"Keyword Entered: {keyword}")
    print(f"Blogs Scraped: {len(blogs)}")
    print(f"Processed Blogs: {len(processed_df)}")
    print(f"Topics Generated: 3")
    print("Visualizations Saved in outputs/ folder")
    print("Final Dataset Saved:")
    print("data/processed/final_blog_data.csv")

    print("\n✅ PROJECT EXECUTION COMPLETED 🚀")


if __name__ == "__main__":
    main()