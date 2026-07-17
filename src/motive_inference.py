import pandas as pd


def infer_motive(text):
    text = text.lower()
    word_count = max(len(text.split()), 1)  # avoid divide by zero

    tutorial_keywords = ["tutorial", "step by step", "how to", "build", "guide", "learn", "implementation"]
    promotional_keywords = ["buy", "pricing", "subscribe", "product", "service", "free trial", "demo", "platform"]
    opinion_keywords = ["i think", "in my opinion", "i believe", "should", "we think", "analysis suggests"]

    score = {"Tutorial": 0, "Promotional": 0, "Opinion": 0}

    for word in tutorial_keywords:
        score["Tutorial"] += text.count(word) * 2
    for word in promotional_keywords:
        score["Promotional"] += text.count(word) * 2
    for word in opinion_keywords:
        score["Opinion"] += text.count(word) * 2

    # normalize by length so long articles don't just win by volume
    for key in score:
        score[key] = score[key] / word_count

    best = max(score, key=score.get)

    # explicit tie handling instead of silently defaulting to Tutorial
    top_values = sorted(score.values(), reverse=True)
    if top_values[0] == 0:
        return "Informative"
    if len(set(top_values)) < len(top_values) and top_values.count(top_values[0]) > 1:
        return "Mixed"  # or "Informative" — your call

    return best

def analyze_motives(df):

    df["motive"] = df["cleaned_content"].apply(
        infer_motive
    )

    print("\nBLOG MOTIVE ANALYSIS:\n")

    print(
        df[["title", "motive"]]
    )

    return df



