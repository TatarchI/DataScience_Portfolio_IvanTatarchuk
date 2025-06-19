"""
📦 Module: 05_glovo_text_explainer.py

🔹 Description:
This is the fifth and final module in the Glovo NLP project pipeline.
It performs **exploratory textual analysis (EDA)** on the sentiment-labeled dataset.
The focus is on understanding word-level patterns in both positive and negative reviews.

Included features:
- Frequency analysis of unigrams, bigrams, and trigrams by sentiment class
- WordCloud visualizations
- Time-series trend plots for top frequent words in negative reviews (monthly)

This analysis helps reveal the linguistic drivers behind user satisfaction or dissatisfaction over time.

🔸 Output:
- Word frequency barplots and word clouds
- N-gram visualizations (bigrams and trigrams)
- Line chart showing monthly trends of top 3 negative words
- All visualizations are saved to `results_last/`
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from wordcloud import WordCloud
from nltk.util import ngrams

import warnings
warnings.filterwarnings("ignore")

# 🔹 Prepare output directory
os.makedirs("results_last", exist_ok=True)

# 🔹 Load final sentiment-labeled dataset
df = pd.read_csv("glovo_final_sentiment_dataset.csv")

# 🔹 Filter for only positive and negative reviews
df_filtered = df[df["final_sentiment"].isin(["positive", "negative"])]

# 🔹 Tokenize cleaned NLTK text
df_filtered["tokens"] = df_filtered["cleaned_review_nltk"].str.split()

# --- 🔍 Top Unigrams by Sentiment ---
def get_top_words(df, sentiment, n=20):
    all_words = df[df["final_sentiment"] == sentiment]["tokens"].explode()
    freq = Counter(all_words)
    return pd.DataFrame(freq.most_common(n), columns=["word", "count"])

top_pos = get_top_words(df_filtered, "positive")
top_neg = get_top_words(df_filtered, "negative")

# Unigram visualization
def plot_top_words(data, sentiment):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x="count", y="word", palette="crest" if sentiment=="positive" else "rocket")
    plt.title(f"Top 20 Words in {sentiment.capitalize()} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(f"results_last/top20_words_{sentiment}.png", dpi=300)
    plt.show()

plot_top_words(top_pos, "positive")
plot_top_words(top_neg, "negative")

# WordClouds
def generate_wordcloud(df, sentiment):
    words = df[df["final_sentiment"] == sentiment]["tokens"].explode()
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud — {sentiment.capitalize()} Reviews")
    plt.tight_layout()
    plt.savefig(f"results_last/wordcloud_{sentiment}.png", dpi=300)
    plt.show()

generate_wordcloud(df_filtered, "positive")
generate_wordcloud(df_filtered, "negative")

# --- 🔗 N-gram Frequency (Bigrams & Trigrams) ---
def get_top_ngrams(df, sentiment, ngram_size=2, top_n=20):
    tokens_series = df[df["final_sentiment"] == sentiment]["tokens"]
    ngram_counter = Counter()
    for tokens in tokens_series:
        ngrams_list = list(ngrams(tokens, ngram_size))
        ngram_counter.update(ngrams_list)
    top = ngram_counter.most_common(top_n)
    return pd.DataFrame(top, columns=["ngram", "count"])

# Bigrams & Trigrams visualization
def plot_top_ngrams(data, sentiment, ngram_size):
    data["ngram"] = data["ngram"].apply(lambda x: " ".join(x))
    plt.figure(figsize=(10, 5))
    palette = "crest" if sentiment == "positive" else "rocket"
    sns.barplot(data=data, x="count", y="ngram", palette=palette)
    plt.title(f"Top {len(data)} {ngram_size}-grams in {sentiment.capitalize()} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel(f"{ngram_size}-gram")
    plt.tight_layout()
    plt.savefig(f"results_last/top{ngram_size}grams_{sentiment}.png", dpi=300)
    plt.show()

# 🔹 Bigrams
top_bigrams_pos = get_top_ngrams(df_filtered, "positive", ngram_size=2)
top_bigrams_neg = get_top_ngrams(df_filtered, "negative", ngram_size=2)

plot_top_ngrams(top_bigrams_pos, "positive", ngram_size=2)
plot_top_ngrams(top_bigrams_neg, "negative", ngram_size=2)

# 🔹 Trigrams
top_trigrams_pos = get_top_ngrams(df_filtered, "positive", ngram_size=3)
top_trigrams_neg = get_top_ngrams(df_filtered, "negative", ngram_size=3)

plot_top_ngrams(top_trigrams_pos, "positive", ngram_size=3)
plot_top_ngrams(top_trigrams_neg, "negative", ngram_size=3)

# --- 📈 Temporal Trends of Top Words in Negative Reviews ---
df_neg = df[df["final_sentiment"] == "negative"].copy()
df_neg["tokens"] = df_neg["cleaned_review_nltk"].astype(str).str.split()
df_neg["month"] = pd.to_datetime(df_neg["date"], errors="coerce").dt.to_period("M")

# Top 3 frequent unigrams
all_words = df_neg["tokens"].explode()
top_3_words = [w for w, _ in Counter(all_words).most_common(3)]

# Frequency per month
freq_data = []
for month, group in df_neg.groupby("month"):
    word_counts = Counter(group["tokens"].explode())
    for word in top_3_words:
        freq_data.append({
            "month": str(month),
            "word": word,
            "count": word_counts[word]
        })

# Lineplot
freq_df = pd.DataFrame(freq_data)
freq_df = freq_df.sort_values("month")

plt.figure(figsize=(14, 7))
sns.lineplot(data=freq_df, x="month", y="count", hue="word", marker="o")
plt.title("Monthly Frequency of Top 3 Words in Negative Reviews (2023–2025)")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results_last/unigram_trend_negative_top3.png", dpi=300)
plt.show()