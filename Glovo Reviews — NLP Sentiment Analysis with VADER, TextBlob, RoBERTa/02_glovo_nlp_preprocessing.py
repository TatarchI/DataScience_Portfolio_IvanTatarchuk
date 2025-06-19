"""
📦 Module: 02_glovo_nlp_preprocessing.py

🔹 Description:
This is the second module in the Glovo NLP project pipeline.
It performs initial **data validation**, **language detection**, and **text cleaning**.

Key functionality includes:
- Detecting the actual language of each review using both `langdetect` and English word ratio heuristics.
- Filtering for clean English-only reviews.
- Text preprocessing using two parallel methods: **NLTK** and **spaCy**.
- Saving the final cleaned dataset to `glovo_en_combined_clean.csv` for downstream analysis.

This module ensures all reviews used for further NLP modeling are valid, language-consistent, and normalized.

🔸 Output:
- Intermediate dataset: `glovo_full_with_detected_lang.csv`
- Final cleaned dataset (dual preprocessing): `glovo_en_combined_clean.csv`
"""

import pandas as pd
from langdetect import detect, LangDetectException
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional
from tqdm import tqdm
from nltk.tokenize import TreebankWordTokenizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# 🔹 Display Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)
pd.set_option('display.max_colwidth', 100)

# 🔹 Load Raw Dataset
df = pd.read_csv("glovo_reviews_full.csv")

# 🔍 Preview and Initial Checks
print("🔎 First 5 rows:")
print(df.head())

print("\n🔎 Last 5 rows:")
print(df.tail())

print("\n📐 Shape:", df.shape)
print("\n🧱 Columns:", df.columns.tolist())
print("\n📊 Rating Distribution:", df['rating'].value_counts())
print("\n🧼 Null Value Check:")
print(df.isnull().sum())

print("\n🌍 Unique Language Codes:")
print(df['lang'].value_counts())

# 🔹 Language Detection Function with Heuristic Filtering
tqdm.pandas()

def safe_detect_language(text: str) -> str:
    """
    Detect language using:
    - English word ratio (based on ASCII a-zA-Z)
    - langdetect applied to raw text (to catch Arabic, Cyrillic, etc.)
    - Robust fallback logic for mixed or ambiguous texts
    """
    try:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "others"

        original_text = text.strip()

        # Primary langdetect detection
        lang_raw = "unknown"
        try:
            lang_raw = detect(original_text)
        except LangDetectException:
            pass

        # Clean for English ratio
        cleaned = re.sub(r'[^\x00-\x7F]+', '', original_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        all_words = re.findall(r'\b\w+\b', cleaned)
        en_words = [w for w in all_words if re.fullmatch(r'[a-zA-Z]{2,}', w)]
        ratio = len(en_words) / max(len(all_words), 1)

        # Conditions for classification
        if lang_raw != "en":
            return "others"
        elif ratio > 0.75:
            return "en"
        elif 0.25 <= ratio <= 0.75:
            return "mixed_en"
        else:
            return "others"

    except Exception:
        return "others"

# 🔍 Apply Language Detection
print("🌐 Detecting language for each review...")
df["detected_lang"] = df["review"].astype(str).progress_apply(safe_detect_language)

# 💾 Save for manual inspection
df.to_csv("glovo_full_with_detected_lang.csv", index=False, encoding="utf-8-sig")
print("💾 Saved full dataset with detected_lang column: glovo_full_with_detected_lang.csv")

# 🔹 Filter English-only Reviews
df_en = df[df["detected_lang"] == "en"].copy()
print(f"✅ Retained {len(df_en)} reviews with detected_lang='en'")

# 🔹 Text Cleaning with NLTK
def clean_text_nltk(text: str) -> str:
    """
    Clean and lemmatize text using NLTK.
    Steps:
    - Lowercase
    - Remove non-alphabetic characters
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)

    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# 🔹 Text Cleaning with spaCy
def clean_text_spacy(text: str, nlp) -> str:
    """
    Clean and lemmatize text using spaCy pipeline.
    - Remove stopwords and punctuation
    - Keep only useful POS (optional)
    """
    doc = nlp(text.lower())
    cleaned = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(cleaned)

# 🔹 Wrapper to Apply Cleaning Method
def clean_reviews(df: pd.DataFrame, method: str = "nltk", spacy_model: Optional[str] = "en_core_web_sm") -> pd.DataFrame:
    """
    Apply text cleaning to all reviews using either NLTK or spaCy.

    Args:
        df (pd.DataFrame): Must contain 'review' column.
        method (str): 'nltk' or 'spacy'.
        spacy_model (str): Name of spaCy model to load (if method == 'spacy').

    Returns:
        pd.DataFrame: Original df with new column 'cleaned_review'.
    """
    tqdm.pandas()

    if method == "nltk":
        df["cleaned_review"] = df["review"].astype(str).progress_apply(clean_text_nltk)
    elif method == "spacy":
        print(f"🔁 Loading spaCy model: {spacy_model}...")
        nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        df["cleaned_review"] = df["review"].astype(str).progress_apply(lambda x: clean_text_spacy(x, nlp))
    else:
        raise ValueError("Invalid method. Choose 'nltk' or 'spacy'.")

    return df

# 🔹 Apply NLTK Cleaning
df_en_nltk = clean_reviews(df_en.copy(), method='nltk')
df_en["cleaned_review_nltk"] = df_en_nltk["cleaned_review"]

# 🔹 Apply spaCy Cleaning
df_en_spacy = clean_reviews(df_en.copy(), method='spacy', spacy_model='en_core_web_sm')
df_en["cleaned_review_spacy"] = df_en_spacy["cleaned_review"]

# 💾 Save Cleaned Combined Output
df_en.to_csv("glovo_en_combined_clean.csv", index=False, encoding="utf-8-sig")
print("💾 Saved combined cleaned dataset: glovo_en_combined_clean.csv")

# 🔍 Preview Random Sample
print("🔍 Sample of 10 random cleaned reviews:")
print(df_en.sample(10))