"""
📦 Module: 03_glovo_nlp_identification.py

🔹 Description:
This is the third module in the Glovo NLP project pipeline.
It performs **sentiment analysis** on previously cleaned English reviews using three independent approaches:

1. **VADER** — rule-based model optimized for social media text.
2. **TextBlob** — lexicon-based polarity scoring.
3. **RoBERTa** — transformer-based sentiment classification model from CardiffNLP.

The goal is to enrich the dataset with **multiple sentiment perspectives** for more robust ensemble modeling.

Additionally, the module performs a frequency breakdown of top words per rating (1 to 5) to provide basic exploratory
insight.

🔸 Output:
- Final enriched CSV: `glovo_sentiment_vader_textblob_roberta.csv`
- Adds **12 new columns** (6 per method, across both cleaning pipelines).
"""

from collections import Counter
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 🔹 Load the cleaned review dataset
df = pd.read_csv("glovo_en_combined_clean.csv")

# 🔍 Null check on both cleaning methods
print("🧼 Null Check:")
print(df[["cleaned_review_nltk", "cleaned_review_spacy"]].isnull().sum())

# 🔹 Top 20 words by rating (exploratory insight)
for rating in sorted(df['rating'].unique()):
    print(f"\n📊 Rating {rating} — Top 20 Words:")
    reviews = df[df['rating'] == rating]["cleaned_review_nltk"].dropna()
    words = ' '.join(reviews).split()
    top_words = Counter(words).most_common(20)
    for word, count in top_words:
        print(f"{word:<15} {count}")

# --- Sentiment Analysis Setup ---
# nltk.download('vader_lexicon')  # Uncomment if running for the first time

sia = SentimentIntensityAnalyzer()
tqdm.pandas()

# --- VADER ---
def get_vader_sentiment_and_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series(["unknown", 0.0])
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.3:
        sentiment = "positive"
    elif score <= -0.3:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return pd.Series([sentiment, score])

# --- TextBlob ---
def get_textblob_sentiment_and_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series(["unknown", 0.0])
    score = TextBlob(text).sentiment.polarity
    if score >= 0.3:
        sentiment = "positive"
    elif score <= -0.3:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return pd.Series([sentiment, score])

# --- RoBERTa ---
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
model.eval()

def get_roberta_sentiment_and_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series(["unknown", 0.0])
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = torch.nn.functional.softmax(output.logits, dim=1).detach().numpy()[0]
        labels = ['negative', 'neutral', 'positive']
        label = labels[np.argmax(scores)]
        confidence = float(np.max(scores))
        return pd.Series([label, confidence])
    except Exception:
        return pd.Series(["error", 0.0])

# 🔹 Apply sentiment models to NLTK-cleaned text
df[["vader_sentiment_nltk", "vader_score_nltk"]] = df["cleaned_review_nltk"].progress_apply(get_vader_sentiment_and_score)
df[["textblob_sentiment_nltk", "textblob_score_nltk"]] = df["cleaned_review_nltk"].progress_apply(get_textblob_sentiment_and_score)
df[["roberta_sentiment_nltk", "roberta_score_nltk"]] = df["cleaned_review_nltk"].progress_apply(get_roberta_sentiment_and_score)

# 🔹 Apply sentiment models to spaCy-cleaned text
df[["vader_sentiment_spacy", "vader_score_spacy"]] = df["cleaned_review_spacy"].progress_apply(get_vader_sentiment_and_score)
df[["textblob_sentiment_spacy", "textblob_score_spacy"]] = df["cleaned_review_spacy"].progress_apply(get_textblob_sentiment_and_score)
df[["roberta_sentiment_spacy", "roberta_score_spacy"]] = df["cleaned_review_spacy"].progress_apply(get_roberta_sentiment_and_score)

# 💾 Save enriched dataset
df.to_csv("glovo_sentiment_vader_textblob_roberta.csv", index=False, encoding="utf-8-sig")
print("✅ Saved 12 new columns to: glovo_sentiment_vader_textblob_roberta.csv")