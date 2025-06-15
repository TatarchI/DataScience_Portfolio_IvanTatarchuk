# ------------------------------- 🔹 Project: Cryptocurrency Time Series Clustering --------------------------------

"""
Author: Ivan Tatarchuk

Project Description:
--------------------
This project implements unsupervised clustering techniques to analyze the historical price behavior
of three major cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), and Solana (SOL).

Using historical price data from 2021 onward, the script performs time series clustering using
both KMeans and Hierarchical (Agglomerative) methods. The clustering is applied to two different
feature sets:
    ▸ Raw [index, price] pairs
    ▸ Engineered statistical features extracted from sliding time windows, including:
        - Mean
        - Standard deviation
        - Delta (end - start)
        - Slope (trend)
        - Range (max - min)

The goal is to detect and classify different behavioral market regimes such as uptrends, downtrends,
and periods of consolidation based on the shape and volatility of the price curves.

Data Sources (Historical Prices):
---------------------------------
    • Bitcoin  → https://www.coingecko.com/en/coins/bitcoin/historical_data
    • Ethereum → https://www.coingecko.com/en/coins/ethereum/historical_data
    • Solana   → https://www.coingecko.com/en/coins/solana/historical_data

Dependencies:
-------------
    - numpy        1.26.4
    - pandas       2.2.3
    - matplotlib   3.7.5
    - seaborn      0.13.2
    - scikit-learn 1.6.1
    - xlrd         2.0.1
"""

# -------------------------------------- 🔹 Imports and Setup --------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------- 🔹 Data Loading and Preprocessing --------------------------------------

def load_real_data(filename):
    """
    Loads and preprocesses historical cryptocurrency price data from a CSV/Excel file.

    The function reads price data exported from Coingecko, parses dates, filters from 2021 onward,
    and returns both price and date arrays.

    Parameters:
    -----------
    filename : str
        Name of the input file containing historical crypto prices (CSV or mislabeled Excel format).

    Returns:
    --------
    prices : np.ndarray
        Array of float prices since 2021.
    dates : pd.Series
        Corresponding array of datetime values.
    """
    print(f"🔹 Loading data from file: {filename}...")

    # Read file as CSV regardless of .xls extension — some files are mislabeled CSVs
    try:
        df = pd.read_csv(filename, delimiter=",", encoding="utf-8", usecols=["snapped_at", "price"])
    except Exception as e:
        raise ValueError(f"❌ Error: Failed to read file {filename} as CSV. Details: {e}")

    # Convert date column to datetime and sort chronologically
    df["snapped_at"] = pd.to_datetime(df["snapped_at"], errors="coerce")
    df = df.sort_values(by="snapped_at", ascending=True)

    # Filter only data from January 1, 2021 onward
    df = df[df["snapped_at"] >= "2021-01-01"]

    # Convert to NumPy arrays
    prices = np.array(df["price"])
    dates = df["snapped_at"].reset_index(drop=True)

    return prices, dates

# ------------------------------------------------------------------------------------
# 📌 Rationale Behind Clustering Method Selection
#
# Out of the available machine learning algorithms, we selected two true clustering methods:
#
# ▸ KMeans — a baseline and widely-used centroid-based clustering algorithm.
# ▸ Hierarchical (Agglomerative) Clustering — useful for discovering nested or structural groupings.
#
# ❌ Support Vector Machines (SVM) and ❌ K-Nearest Neighbors (KNN) were intentionally excluded,
#     as they are classification algorithms, not clustering ones.
#     To apply them here, we would have to artificially simulate labels (e.g., via price range bins),
#     which contradicts the goal of unsupervised clustering and leads to pseudo-clustering.
#
# Therefore, we retained only those methods that implement genuine unsupervised learning
# without requiring prior labeling or manual segmentation.
# This ensures logical consistency and meaningful interpretation of the results.
# ------------------------------------------------------------------------------------

# -------------------------------------- 🔹 KMeans Clustering --------------------------------------

def run_kmeans_clustering(real_data_dict, n_clusters=4, label_x="Feature 1", label_y="Feature 2", title=None):
    """
    Performs KMeans clustering on each asset in the input dataset.

    Supports both basic [index, price] representation and advanced feature vectors
    such as [slope, std] extracted from sliding windows.

    Parameters:
    -----------
    real_data_dict : dict[str, tuple[np.ndarray, np.ndarray]]
        Dictionary where:
            - key is the asset symbol (e.g., 'BTC', 'ETH', 'SOL'),
            - value is a tuple (X, dates),
              where X is a (n_samples, 2)-shaped array of features,
              and dates is a datetime index (can be unused here).

    n_clusters : int, optional (default=4)
        Number of clusters to form.

    label_x : str
        Label for the x-axis (used in the plot).

    label_y : str
        Label for the y-axis (used in the plot).

    title : str, optional
        Overall plot title (optional).
    """
    print("\n🔷 Running KMeans clustering on input data...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (crypto, (X, _)) in enumerate(real_data_dict.items()):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        ax = axes[i]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=10)
        ax.set_title(f"{crypto} — KMeans")
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.grid(True)

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

# -------------------------------------- 🔹 Hierarchical Clustering --------------------------------------

def run_hierarchical_clustering(real_data_dict, n_clusters=4, label_x="Feature 1", label_y="Feature 2", title=None):
    """
    Performs agglomerative (hierarchical) clustering on 2D feature vectors for each crypto asset.

    Supports both simple inputs like [index, price] and engineered features like [std, slope].

    Parameters:
    -----------
    real_data_dict : dict[str, tuple[np.ndarray, np.ndarray]]
        Dictionary where:
            - key is the asset symbol (e.g., 'BTC', 'ETH', 'SOL'),
            - value is a tuple (X, dates),
              where X is a (n_samples, 2)-shaped array of features.

    n_clusters : int, optional (default=4)
        Number of clusters to compute.

    label_x : str
        Label for the x-axis (used in the plot).

    label_y : str
        Label for the y-axis (used in the plot).

    title : str, optional
        Overall plot title (optional).
    """
    print("\n🔺 Running Hierarchical (Agglomerative) Clustering...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (crypto, (X, _)) in enumerate(real_data_dict.items()):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)

        ax = axes[i]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=10)
        ax.set_title(f"{crypto} — Hierarchical")
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.grid(True)

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

# -------------------------------------- 🔹 Feature Engineering from Sliding Windows --------------------------------------

def generate_features_from_windows(prices: np.ndarray, window_size: int = 30) -> np.ndarray:
    """
    Generates statistical features from fixed-size sliding windows over a price time series.

    For each window of N consecutive days, the following features are calculated:
        - mean  : average price within the window
        - std   : standard deviation (volatility)
        - delta : difference between last and first price (end - start)
        - slope : trend slope via linear regression (captures direction and strength)
        - range : difference between max and min price within the window

    Parameters:
    -----------
    prices : np.ndarray
        1D array of historical prices.

    window_size : int, optional (default=30)
        Size of the sliding window in days.

    Returns:
    --------
    features : np.ndarray
        Feature matrix of shape (n_windows, 5), where each row is:
        [mean, std, delta, slope, range]
    """
    features = []

    for i in range(len(prices) - window_size + 1):
        window = prices[i:i + window_size]

        mean = np.mean(window)
        std = np.std(window)
        delta = window[-1] - window[0]

        # Fit linear regression: y = a * x + b → slope = a
        x = np.arange(window_size)
        a, b = np.polyfit(x, window, deg=1)
        slope = a

        # slope > 0 → uptrend; slope < 0 → downtrend; slope ≈ 0 → flat
        # Note: slope is in price units per day, not percent

        rnge = np.max(window) - np.min(window)

        features.append([mean, std, delta, slope, rnge])

    return np.array(features)

# -------------------------------------- 🔹 Main Execution Flow --------------------------------------

if __name__ == "__main__":

    print("\n--- Loading cryptocurrency price data from Coingecko files ---")

    file_map = {
        "BTC": "btc-usd-max.xls",
        "ETH": "eth-usd-max.xls",
        "SOL": "sol-usd-max.xls"
    }

    # 🔹 Load raw prices for each crypto asset
    raw_prices_dict = {
        crypto: load_real_data(filename) for crypto, filename in file_map.items()
    }

    # 🔹 Construct basic 2D data: [index, price]
    real_data_dict = {}
    for crypto in raw_prices_dict:
        prices, dates = raw_prices_dict[crypto]
        X = np.array([[i, price] for i, price in enumerate(prices)])
        real_data_dict[crypto] = (X, dates)

    # 🔹 Plot time series of BTC, ETH, SOL prices
    print("\n📊 Plotting historical price trends (2021–present)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (crypto, (prices, dates)) in enumerate(raw_prices_dict.items()):
        ax = axes[i]
        ax.plot(dates, prices, label="Price", color="blue", linewidth=1.5)
        ax.set_title(f"{crypto} — Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid(True)
        ax.legend()

        # Show year ticks only
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle("Price History of BTC, ETH, SOL Since 2021", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 🔹 Run clustering on [index, price]
    run_kmeans_clustering(real_data_dict)
    run_hierarchical_clustering(real_data_dict)

    print("\n✅ Proceeding to advanced clustering based on time-window statistical features.")
    print("This allows us to detect market regimes (growth, decline, consolidation) based on behavior, not just shape.")

    # 🔹 Window size selection
    print("\n📐 Choose the sliding window size for feature extraction:")
    print("1 - Default size (30 days)")
    print("2 - Enter custom size")

    sub_choice = input("👉 Enter 1 or 2: ")
    if sub_choice == "2":
        try:
            window_size = int(input("🔢 Enter window size in days (e.g., 20 or 60): "))
            print(f"✅ Using custom window size: {window_size} days")
        except:
            print("❌ Invalid input. Falling back to default (30 days).")
            window_size = 30
    else:
        window_size = 30
        print("✅ Default window size selected — 30 days.")

    # 🔹 Extract price series again
    prices_btc, dates_btc = raw_prices_dict["BTC"]
    prices_eth, dates_eth = raw_prices_dict["ETH"]
    prices_sol, dates_sol = raw_prices_dict["SOL"]

    # 🔹 Generate feature vectors from sliding windows
    X_btc = generate_features_from_windows(prices_btc, window_size)
    X_eth = generate_features_from_windows(prices_eth, window_size)
    X_sol = generate_features_from_windows(prices_sol, window_size)

    features_dict = {
        "BTC": (X_btc, dates_btc[window_size - 1:]),
        "ETH": (X_eth, dates_eth[window_size - 1:]),
        "SOL": (X_sol, dates_sol[window_size - 1:])
    }

    # 🔹 Correlation heatmap of all extracted features (example: BTC)
    sns.heatmap(
        pd.DataFrame(X_btc, columns=["mean", "std", "delta", "slope", "range"]).corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Correlation Matrix of Features (BTC)")
    plt.show()

    # 📌 We choose slope and std as the key clustering dimensions because:
    # ▸ Slope captures the direction and strength of price trends.
    # ▸ Std represents market volatility (price variability).
    # 🔍 These features are weakly correlated with each other, but together reveal meaningful market regimes.

    print("\n📊 Clustering based on slope and std features...")

    # 🔹 KMeans clustering on selected features [slope, std]
    run_kmeans_clustering(
        {
            crypto: (X[:, [3, 1]], dates)
            for crypto, (X, dates) in features_dict.items()
        },
        label_x="Slope (trend direction)",
        label_y="Std (volatility)",
        title="Market Structure Clustering using [slope + std]"
    )

    # 🔹 Hierarchical clustering on same features
    run_hierarchical_clustering(
        {
            crypto: (X[:, [3, 1]], dates)
            for crypto, (X, dates) in features_dict.items()
        },
        label_x="Slope (trend direction)",
        label_y="Std (volatility)",
        title="Hierarchical Clustering (slope + std)"
    )

    # 📈 Interpreting resulting clusters:
    # ▸ Low slope + low std → market flat / sideways movement
    # ▸ High slope + high std → strong uptrends or panic sell-offs
    # ▸ High std + negative slope → downtrends or high uncertainty
    # 🔍 This allows us to classify dynamic market behavior per asset — not just price segments.

"""
📈 Analysis of Clustering Results and Interpretation
----------------------------------------------------

🔹 The first stage of clustering was based on simple two-dimensional features: [index, price],
representing each observation as a point in the space (day, price).

🔹 Two clustering algorithms were applied:
    1. KMeans — centroid-based clustering
    2. Hierarchical (Agglomerative) clustering

📊 Visually, the results were consistent: clusters aligned with typical market phases
such as uptrends, downtrends, and periods of consolidation.
This behavior was observed across all three assets: BTC, ETH, and SOL.

⚠️ However, there is a key limitation:
The "index" feature is a synthetic proxy for time and carries no financial meaning,
while "price" alone may not fully capture underlying market structure or dynamics.

🔎 As a result, the clustering at this stage reflects geometric segmentation rather than
behavioral or statistical insights about market regimes.

----------------------------------------------------

🧠 To address this, we developed a custom feature engineering approach
based on sliding windows (e.g., 30-day sequences) instead of pointwise prices.

🔹 In the second part of the project, we created feature vectors for each time window,
including the following statistical characteristics:
    - mean
    - standard deviation (std)
    - delta (end - start)
    - slope (linear trend)
    - range (max - min)

Among them, slope and std were selected due to their low correlation and strong explanatory power.

📊 Clustering on [slope + std] revealed distinct market regimes:
• low slope + low std → flat/consolidation phase  
• high slope + high std → aggressive growth or panic sell-offs  
• negative slope + high std → bear markets with high volatility

📈 The correlation matrix confirmed that slope and std were only weakly correlated (~0.3),
making them ideal for separating behavioral clusters in 2D space.

🔍 In this form, clustering goes beyond geometry — it models dynamic market structure
based on price behavior patterns rather than raw values.

✅ Both clustering methods (KMeans and Hierarchical) produced stable, interpretable results,
and the feature engineering approach is scalable and generalizable across assets.
"""