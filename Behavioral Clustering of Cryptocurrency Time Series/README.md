# 📊 Behavioral Clustering of Cryptocurrency Time Series

This project performs unsupervised clustering of historical price time series for three major cryptocurrencies — **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Solana (SOL)** — using statistical features extracted from sliding windows.  
It explores two main clustering methods (KMeans and Hierarchical) to identify different **market regimes** such as **growth**, **decline**, and **consolidation**.

---

## 🧠 Project Overview

- **Data Source**: Historical price data from [Coingecko](https://www.coingecko.com/)
- **Period Covered**: 2021 to present
- **Assets Analyzed**: BTC, ETH, SOL
- **Approach**:
  - Initial clustering based on raw 2D vectors: `[index, price]`
  - Advanced clustering using engineered features from sliding time windows:
    - `mean`, `std`, `delta`, `slope`, `range`

---

## 💼 Business Value

- **Unsupervised Market Structure Discovery**: Enables identification of typical price behavior patterns (growth, decline, consolidation) in major cryptocurrencies without relying on labeled data or manual annotation.

- **Exploratory Crypto Analytics**: Provides a visual and quantitative way to segment time periods by behavioral regimes, supporting further research and hypothesis generation.

- **Risk & Trend Awareness for Retail Investors**: Helps non-professional investors better understand when markets are stable vs. volatile, trending vs. flat — based on clustering of historical dynamics.

- **Lightweight Analytical Framework**: Offers a reusable Python-based framework for time-series segmentation that can be applied to other volatile assets or financial instruments.

- **Educational and Interpretability Use Case**: Demonstrates how statistical feature engineering (e.g., slope, volatility) can be used in tandem with unsupervised learning to model real-world financial phenomena in an interpretable way.

---

## ⚙️ Technologies Used

- `Python 3.11`
- `NumPy`, `Pandas`
- `Matplotlib`, `Seaborn`
- `Scikit-learn (KMeans, AgglomerativeClustering)`

---

## 🔍 Clustering Methods

### 1. **KMeans Clustering**

- Classical unsupervised algorithm based on distance to centroids.
- Applied both on `[index, price]` and `[slope, std]` features.

### 2. **Hierarchical Clustering (Agglomerative)**

- Tree-based method useful for detecting nested or overlapping patterns.
- Offers an interpretable alternative to KMeans, especially in behavioral segmentation.

---

## 🧬 Feature Engineering Strategy

We extract statistical indicators over rolling windows of price data (default = 30 days):

| Feature  | Description                          |
|----------|--------------------------------------|
| Mean     | Average price in the window          |
| Std      | Volatility of the price              |
| Delta    | Price difference (end - start)       |
| Slope    | Linear trend over the window         |
| Range    | Difference between max and min price |

After analysis, **`slope`** and **`std`** were selected as the most informative and least correlated pair (corr ≈ 0.3).

---

## 🧠 Result Analysis

- Raw `[index, price]` clustering gave visually interpretable but limited segmentation.
- Custom features like `slope` and `std` added **behavioral meaning**.
- KMeans and Hierarchical methods consistently grouped market windows into:
  - 📉 **Bear phases**: negative slope + high volatility
  - 📊 **Sideways periods**: near-zero slope + low volatility
  - 🚀 **Bull phases**: steep slope + increased std

---

## 🚀 Future Improvements

- Add more assets (BNB, ADA, XRP, etc.)
- Apply DTW or t-SNE for time-sensitive clustering
- Automate cluster labeling with fuzzy logic
- Deploy as interactive web dashboard (e.g. Streamlit)

---

## 📬 Contact

Feel free to reach out or fork the project for adaptation to your own city or business sector.

© 2025 Ivan Tatarchuk (Telegram - @Ivan_Tatarchuk; LinkedIn - https://www.linkedin.com/in/ivan-tatarchuk/)