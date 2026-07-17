import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_comment(blog_text):
    try:
        short_text = blog_text[:1500]

        prompt = f"""
        Read the following blog content and generate
        a short human-like comment about it.

        Blog Content:
        {short_text}

        Comment:
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating comment: {e}")
        return "Comment generation failed"


def generate_comments_for_blogs(df):
    df = df.copy()  # ✅ avoid SettingWithCopyWarning
    generated_comments = []

    print("\nGENERATING AI COMMENTS...\n")

    for index, row in df.iterrows():
        print(f"Processing Blog {index + 1}")
        comment = generate_comment(row["cleaned_content"])  # ✅ fixed
        generated_comments.append(comment)

    df["ai_generated_comment"] = generated_comments

    output_path = "data/processed/final_blog_data.csv"
    os.makedirs("data/processed", exist_ok=True)  # ✅ ensure folder exists
    df.to_csv(output_path, index=False)

    print(f"\nFinal dataset saved to {output_path}")
    print(df[["title", "ai_generated_comment"]])

    return df