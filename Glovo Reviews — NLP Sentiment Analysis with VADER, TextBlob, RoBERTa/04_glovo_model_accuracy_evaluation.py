"""
📦 Module: 04_glovo_model_accuracy_evaluation.py

🔹 Description:
This is the fourth module in the Glovo NLP pipeline.
It evaluates the **performance of sentiment models** (VADER, TextBlob, RoBERTa) by comparing their predictions against
the true sentiment inferred from user star ratings (1–5).

It includes:
- Overall model accuracy comparison
- Per-class (positive / neutral / negative) accuracy evaluation
- Visualization of all results
- A mini-ensemble logic (RoBERTa NLTK + spaCy) for the negative class
- Selection of the best-performing model (RoBERTa on NLTK-cleaned text)
- Export of the final streamlined dataset `glovo_final_sentiment_dataset.csv` for downstream modules

🔸 Output:
- Accuracy tables and plots in `/results_first`
- Final dataset with selected sentiment model
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs("results_first", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

# 🔹 Load dataset with sentiment results
df = pd.read_csv("glovo_sentiment_vader_textblob_roberta.csv")

# 🔹 Ground truth sentiment based on rating
def get_true_sentiment(rating):
    if rating in [1, 2]:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["true_sentiment"] = df["rating"].apply(get_true_sentiment)

# 🔹 List of all sentiment prediction columns
sentiment_columns = [
    "vader_sentiment_nltk",
    "textblob_sentiment_nltk",
    "roberta_sentiment_nltk",
    "vader_sentiment_spacy",
    "textblob_sentiment_spacy",
    "roberta_sentiment_spacy"
]

# 🔹 Compute accuracy for each model (overall)
accuracy_results = {}
for col in sentiment_columns:
    accuracy = (df[col] == df["true_sentiment"]).mean()
    accuracy_results[col] = round(accuracy * 100, 2)

# Table with results
acc_df = pd.DataFrame(list(accuracy_results.items()), columns=["model", "accuracy"])
acc_df = acc_df.sort_values(by="accuracy", ascending=False)

# Console output
print("\n📊 Overall model accuracy:")
print(acc_df.to_string(index=False))

# 💾 Save to CSV
acc_df.to_csv("results_first/sentiment_model_accuracy_summary.csv", index=False)

# 📈 Bar plot
plt.figure(figsize=(10, 5))
bars = sns.barplot(data=acc_df, x="model", y="accuracy",
                   palette=sns.color_palette("viridis", n_colors=len(acc_df)))
plt.title("Model Accuracy vs True Rating Sentiment", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy (%)")

for bar in bars.patches:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("results_first/sentiment_model_accuracy_summary.png", dpi=300)
plt.show()

# --- 🔍 Accuracy per Sentiment Class ---
classes = ["negative", "neutral", "positive"]

for sentiment_class in classes:
    print(f"\n🔍 Accuracy for class '{sentiment_class}':")

    class_results = {}
    for col in sentiment_columns:
        class_df = df[df["true_sentiment"] == sentiment_class]
        class_accuracy = (class_df[col] == class_df["true_sentiment"]).mean()
        class_results[col] = round(class_accuracy * 100, 2)

    class_acc_df = pd.DataFrame(list(class_results.items()), columns=["model", "accuracy"])
    class_acc_df = class_acc_df.sort_values(by="accuracy", ascending=False)

    print(class_acc_df.to_string(index=False))

    # Save class-level accuracy
    class_acc_df.to_csv(f"results_first/sentiment_accuracy_{sentiment_class}.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    bars = sns.barplot(data=class_acc_df, x="model", y="accuracy",
                       palette=sns.color_palette("mako", n_colors=len(class_acc_df)))
    plt.title(f"Accuracy per Model — Class: {sentiment_class.capitalize()}", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Accuracy (%)")

    for bar in bars.patches:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.2f}%",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"results_first/sentiment_accuracy_{sentiment_class}.png", dpi=300)
    plt.show()

# --- 🧪 Ensemble Strategy (RoBERTa NLTK + spaCy) for NEGATIVE class ---
print("\n🧪 RoBERTa ensemble for class negative")

df_neg = df[df["true_sentiment"] == "negative"].copy()

def ensemble_roberta(row):
    label1 = row["roberta_sentiment_nltk"]
    score1 = row["roberta_score_nltk"]
    label2 = row["roberta_sentiment_spacy"]
    score2 = row["roberta_score_spacy"]
    if label1 == label2:
        return label1
    return label1 if score1 > score2 else label2

df_neg["ensemble_roberta"] = df_neg.apply(ensemble_roberta, axis=1)

ens_accuracy = (df_neg["ensemble_roberta"] == df_neg["true_sentiment"]).mean() * 100
print(f"📊 Ensemble Accuracy (Negative class): {ens_accuracy:.2f}%")

# Save to CSV
ens_df = pd.DataFrame({"model": ["ensemble_roberta_negative"], "accuracy": [round(ens_accuracy, 2)]})
ens_df.to_csv("results_first/ensemble_roberta_negative_accuracy.csv", index=False)

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(data=ens_df, x="model", y="accuracy", palette="rocket")
plt.ylim(0, 100)
plt.title("Ensemble Accuracy on Negative Class")
plt.ylabel("Accuracy (%)")
for bar in plt.gca().patches:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha='center',
        va='bottom',
        fontsize=9
    )
plt.tight_layout()
plt.savefig("results_first/ensemble_roberta_negative_accuracy.png", dpi=300)
plt.show()

# --- 🔚 Final Dataset Creation: Select Best Performing Model (RoBERTa NLTK) ---
columns_to_drop = [
    "cleaned_review_spacy", "true_sentiment",
    "vader_sentiment_nltk", "vader_score_nltk",
    "textblob_sentiment_nltk", "textblob_score_nltk",
    "vader_sentiment_spacy", "vader_score_spacy",
    "textblob_sentiment_spacy", "textblob_score_spacy",
    "roberta_sentiment_spacy", "roberta_score_spacy"
]

df_final = df.drop(columns=columns_to_drop)

df_final = df_final.rename(columns={
    "roberta_sentiment_nltk": "final_sentiment",
    "roberta_score_nltk": "final_score"
})

# Save final simplified dataset
df_final.to_csv("glovo_final_sentiment_dataset.csv", index=False, encoding="utf-8-sig")
print("✅ Final sentiment dataset saved to: glovo_final_sentiment_dataset.csv")