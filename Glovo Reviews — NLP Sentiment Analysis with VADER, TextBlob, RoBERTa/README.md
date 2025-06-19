# 📦 Glovo Reviews — NLP Sentiment Analysis with VADER, TextBlob, RoBERTa

This repository presents a complete **NLP pipeline for sentiment analysis of Glovo app reviews**.  
It combines scraping, language filtering, dual-mode text cleaning, sentiment modeling with three classifiers, model evaluation, and interpretability.

---

## 🔍 Project Goal

To **analyze user sentiment** expressed in Glovo app reviews from Google Play using a multi-model NLP pipeline, and to **identify key textual indicators** of satisfaction or frustration.  
The project aims to answer:

- What do users talk about in negative vs positive reviews?
- Which sentiment model is most reliable for Glovo-like data?
- How do top negative concerns evolve over time?

---

## 💡 Business Value

This project simulates a real-world NLP use case for **customer experience analysis in mobile applications**:

- 📉 Detect churn and pain points based on review language
- 📈 Inform product teams with interpretable sentiment trends
- 🧠 Benchmark multiple NLP models (VADER, TextBlob, RoBERTa) on domain-specific user feedback
- 📊 Visualize changes in negative word frequency over time for proactive mitigation

Applicable in **B2C apps, e-commerce, food delivery, fintech**, and any scenario involving public reviews.

---

## 🧭 Project Structure

The pipeline is modularized into 5 stages:

### `01_glovo_review_scraper.py`
- Scrapes reviews from Google Play using `google-play-scraper`
- Detects language of each review
- Saves incrementally in batches with autosave and final merge

### `02_glovo_nlp_preprocessing.py`
- Performs robust language detection with fallback heuristics
- Filters for English reviews
- Applies two independent text cleaning approaches:
  - **NLTK** (lemmatization, stopword removal)
  - **spaCy** (POS-aware, model-driven cleaning)
- Saves merged dataset with both versions

### `03_glovo_nlp_identification.py`
- Applies **three sentiment models** on both cleaned versions:
  - VADER (rule-based)
  - TextBlob (lexicon-based)
  - RoBERTa (transformer-based, via HuggingFace)
- Appends 12 new sentiment-related columns (label + score)

### `04_glovo_model_accuracy_evaluation.py`
- Derives **true sentiment** from user ratings (1–5 stars)
- Compares model predictions vs ground truth:
  - Computes overall and class-specific accuracy
  - Plots model performance and saves CSVs
- Implements **RoBERTa-based ensemble** logic for `negative` class
- Selects the best model (`roberta_sentiment_nltk`) as final output

### `05_glovo_text_explainer.py`
- Performs EDA on positive and negative reviews:
  - Top unigrams, bigrams, trigrams
  - WordClouds
- Tracks **monthly frequency of top 3 negative words** over time
- Saves all results to `/results_last/`

---

## 📁 Output Files and Folders

| File / Folder                                 | Description                                 |
|---------------------------------------------- |---------------------------------------------|
| `glovo_reviews_full.csv`                      | Raw scraped data with language detection    |
| `glovo_full_with_detected_lang.csv`           | Dataset with classified language            |
| `glovo_en_combined_clean.csv`                 | Cleaned text (NLTK + spaCy)                 |
| `glovo_sentiment_vader_textblob_roberta.csv`  | All sentiment predictions                   |
| `glovo_final_sentiment_dataset.csv`           | Final version with best sentiment model     |
| `/results_first/`                             | Accuracy plots and ensemble results         |
| `/results_last/`                              | EDA plots: top words, n-grams, word trends  |
| `/partials/`                                  | Autosaved review batches during scraping    |
| `/console_log/`                               | Files with logs from console                |

---

## 🔮 Future Improvements

- 🧠 **Model fine-tuning**: Fine-tune RoBERTa on Glovo-specific review corpus for even higher accuracy.
- 🌍 **Multi-language support**: Extend language detection and sentiment classification to Spanish, Romanian, Ukrainian, etc.
- ⚖️ **Weighted ensemble**: Build a confidence-weighted voting system that combines VADER, TextBlob, and RoBERTa outputs for more stable predictions.
- 📲 **Deployment-ready API**: Convert the entire pipeline into a production-ready Flask/FastAPI service for real-time sentiment inference.
- 📈 **Interactive dashboards**: Integrate with Streamlit or Plotly Dash for dynamic filtering, exploration, and trend monitoring.
- 🧩 **Code modularization**: Refactor all scripts into reusable Python modules with functions and a unified `main()` entry point to improve maintainability and product-readiness.
- 🚀 **Scalable cloud execution**: Leverage affordable GPU rental services like [Vast.ai](https://vast.ai/) to scale RoBERTa inference across millions of reviews, significantly reducing processing time.

---

## 🛠 Tech Stack

- Python, Pandas, Matplotlib, Seaborn
- NLTK, spaCy, TextBlob
- VADER, HuggingFace Transformers
- tqdm, langdetect, WordCloud

---

## 📬 Contact

Feel free to reach out or fork the project for adaptation to your own city or business sector.

© 2025 Ivan Tatarchuk (Telegram - @Ivan_Tatarchuk; LinkedIn - https://www.linkedin.com/in/ivan-tatarchuk/)