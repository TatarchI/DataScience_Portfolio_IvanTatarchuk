# --------------------------- 🔹 Kalman Filtering of Cryptocurrency Prices  ------------------------------------

"""
Author: Ivan Tatarchuk

🔹 Project Description:
This project implements two recursive filtering algorithms designed for real-world cryptocurrency time series:
- Alpha-Beta (ABF) Filter
- Alpha-Beta-Gamma (ABGF) Filter

The objective is to enhance signal smoothness and track price trends for major cryptocurrencies (Bitcoin, Ethereum,
Solana) from 2021 onward. The analysis focuses exclusively on real market data to evaluate model robustness in a
non-synthetic environment.

In addition to custom filters, the project includes a comparative implementation using the `filterpy` Kalman Filter
library, applied in chunk-wise mode to avoid long-term drift and overfitting.

📌 Data Sources:
- Bitcoin: https://www.coingecko.com/en/coins/bitcoin/historical_data
- Ethereum: https://www.coingecko.com/en/coins/ethereum/historical_data
- Solana: https://www.coingecko.com/en/coins/solana/historical_data

📦 Dependencies (tested configuration):
--------------------------------------------------
pip                23.2.1
numpy              1.26.4
pandas             2.2.3
xlrd               2.0.1
matplotlib         3.10.1
scikit-learn       1.6.1
scipy              1.15.2
seaborn            0.13.2
pmdarima           2.0.4
filterpy           1.4.5
"""

# --------------------------- 🔹 Imports and Dependencies  ------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from filterpy.kalman import KalmanFilter

"""
Required Python libraries for:
- numerical computations (NumPy),
- data manipulation (Pandas),
- visualization (Matplotlib),
- evaluation metrics (Scikit-learn),
- and Kalman filtering via FilterPy.
"""

# ------------------------ 🔹 Data Loading and Preprocessing Function ------------------------

def load_real_data(filename):
    """
    Load and preprocess cryptocurrency price data from a Coingecko CSV/Excel file.

    The function:
    - Reads the file using pandas (assumes CSV content, even if extension is .xls)
    - Converts date column to datetime format
    - Filters data starting from 2021
    - Returns sorted price values as a NumPy array

    Parameters
    ----------
    filename : str
        Path to the CSV or Excel file containing historical price data.
        The file must include columns: 'snapped_at' and 'price'.

    Returns
    -------
    np.ndarray
        NumPy array of price values sorted chronologically since 2021.
    """
    print(f"🔹 Loading data from file {filename}...")

    # Read as CSV (despite .xls extension — file actually contains CSV structure)
    df = pd.read_csv(filename, delimiter=",", encoding="utf-8", usecols=["snapped_at", "price"])

    # Convert 'snapped_at' to datetime and sort chronologically
    df["snapped_at"] = pd.to_datetime(df["snapped_at"], errors='coerce')
    df = df.sort_values(by="snapped_at", ascending=True)

    # Filter from January 2021 onward
    df = df[df["snapped_at"] >= "2021-01-01"]

    # Return price values as NumPy array
    prices = np.array(df["price"])

    return prices

# ------------------------ 🔹 Alpha-Beta Filter Implementation ------------------------

def ABF(S0):
    """
    Alpha-Beta Filter (ABF) — a recursive smoothing method that estimates
    both position and its rate of change (velocity) over time.

    The filter is initialized with a simple velocity estimate and then recursively
    updates the smoothed value using dynamic α and β coefficients.

    Parameters
    ----------
    S0 : list or np.ndarray
        One-dimensional array of input values (e.g., cryptocurrency prices)

    Returns
    -------
    np.ndarray
        Smoothed time series of the same length as input
    """
    iter = len(S0)
    Yin = np.zeros((iter, 1))      # Column vector of input values
    YoutAB = np.zeros((iter, 1))   # Output smoothed values
    T0 = 1                         # Time step (can be adjusted)

    # Convert input to float
    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # ---------- Initial values
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0      # Initial velocity estimate
    Yextra = Yin[0, 0] + Yspeed_retro                # Initial extrapolated value

    # Initial α and β coefficients
    alfa = (2 * (2 * 1 - 1)) / (1 * (1 + 1))
    beta = 6 / (1 * (1 + 1))

    # First smoothing step
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])

    # ---------- Recursive filtering loop
    for i in range(1, iter):
        YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)  # Smoothed value
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)  # Velocity update
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i, 0] + Yspeed_retro                  # Next prediction

        # Update α and β dynamically
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))

    return YoutAB

# ------------------------ 🔹 Alpha-Beta-Gamma Filter Implementation ------------------------

def ABGF(S0):
    """
    Alpha-Beta-Gamma Filter (ABGF) — a recursive smoothing method that estimates
    position, velocity, and acceleration of a signal over time.

    This extension of the AB filter incorporates second-order dynamics and adapts
    α, β, and γ coefficients at each iteration to refine the prediction.

    Parameters
    ----------
    S0 : list or np.ndarray
        One-dimensional array of input values (e.g., cryptocurrency prices)

    Returns
    -------
    np.ndarray
        Smoothed time series of the same length as input
    """
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    Yout_ABG = np.zeros((iter, 1))
    T0 = 1  # Discretization period (time unit)

    # --- Step 0: convert input to float
    for i in range(iter):
        Yin[i, 0] = float(S0[i])

    # --- Initial estimates (position, velocity, acceleration)
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yaccel_retro = (Yin[2, 0] - 2 * Yin[1, 0] + Yin[0, 0]) / (T0 * T0)
    Yextra = Yin[0, 0] + (Yspeed_retro * T0) + (0.5 * Yaccel_retro * T0 * T0)

    # --- First smoothed value
    Yout_ABG[0, 0] = Yin[0, 0]

    # --- Recursive filtering loop
    for i in range(1, iter):
        # Dynamic coefficients (adaptive)
        alpha = 3 * (3 * i - 1) / (i * (i + 1) * (i + 2))
        beta = 18 * (i - 1) / (T0 * (i + 1) * (i + 2) * i)
        gamma = 60 / (T0 * T0 * (i + 1) * (i + 2) * i)

        # Position correction
        Yout_ABG[i, 0] = Yextra + alpha * (Yin[i, 0] - Yextra)

        # Update velocity and acceleration
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
        Yaccel = Yaccel_retro + (gamma / (T0 * T0)) * (Yin[i, 0] - Yextra)

        # Update prediction for the next step
        Yspeed_retro = Yspeed
        Yaccel_retro = Yaccel
        Yextra = Yout_ABG[i, 0] + (Yspeed * T0) + (0.5 * Yaccel * T0 * T0)

    return Yout_ABG

# ------------------------ 🔹 Chunk-Wise Filtering with Manual and FilterPy Kalman Filters ------------------------

def apply_manual_filter_in_chunks(data: np.ndarray, mode: str, chunk_size: int = 100) -> np.ndarray:
    """
    Applies the custom Alpha-Beta or Alpha-Beta-Gamma filter in chunks to prevent long-term drift.

    Each chunk is filtered independently and reset after a specified number of days.
    This method avoids excessive accumulation of error over long sequences.

    Recommended values (based on testing):
    - ABF: chunk_size = 365
    - ABGF: chunk_size = 100

    Parameters
    ----------
    data : np.ndarray
        Input 1D array of values (e.g., prices)
    mode : str
        Either 'ABF' (Alpha-Beta Filter) or 'ABGF' (Alpha-Beta-Gamma Filter)
    chunk_size : int, default=100
        Number of time steps per chunk (e.g., days)

    Returns
    -------
    np.ndarray
        Smoothed full-length array obtained by stitching together all chunks
    """
    smoothed_chunks = []
    n = len(data)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = data[start:end]

        if mode == "ABF":
            smoothed = ABF(chunk).flatten()
        elif mode == "ABGF":
            smoothed = ABGF(chunk).flatten()
        else:
            raise ValueError("Unknown filter mode. Use 'ABF' or 'ABGF'.")

        smoothed_chunks.extend(smoothed)

    return np.array(smoothed_chunks)


def apply_filterpy_filter_in_chunks(data: np.ndarray, mode: str, chunk_size: int = 365) -> np.ndarray:
    """
    Applies the FilterPy-based Kalman filter in chunks with periodic reinitialization.

    This method mitigates filter "sticking" or overfitting to long sequences
    by processing each window independently.

    Parameters
    ----------
    data : np.ndarray
        Input 1D array of values (e.g., prices)
    mode : str
        Filter mode: "ABF" or "ABGF"
    chunk_size : int, default=365
        Number of time steps (e.g., days) per chunk

    Returns
    -------
    np.ndarray
        Smoothed array concatenated from all independently filtered chunks
    """
    smoothed_chunks = []
    n = len(data)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = data[start:end]

        # Apply FilterPy Kalman filter to each chunk independently
        filtered_chunk = apply_filterpy_filter(chunk, mode)

        smoothed_chunks.extend(filtered_chunk)

    return np.array(smoothed_chunks)

# ------------------------ 🔹 Kalman Filter Implementation using FilterPy ------------------------

def apply_filterpy_filter(data: np.ndarray, mode: str) -> np.ndarray:
    """
    Applies a Kalman filter (using the FilterPy library) to a given time series
    in either Alpha-Beta (ABF) or Alpha-Beta-Gamma (ABGF) configuration.

    Parameters
    ----------
    data : np.ndarray
        Input 1D time series data (e.g., cryptocurrency prices)
    mode : str
        Filtering mode: "ABF" (2D state: position + velocity) or "ABGF" (3D state: + acceleration)

    Returns
    -------
    np.ndarray
        Smoothed output series after Kalman filtering
    """
    n = len(data)
    dt = 1.0  # Time step (1 day)

    if mode == "ABF":
        # Kalman filter with 2 state variables: position and velocity
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([data[0], data[1] - data[0]])  # Initial position and velocity
        kf.F = np.array([[1, dt],
                         [0, 1]])                      # State transition model
        kf.H = np.array([[1, 0]])                     # Observation model
        kf.P *= 1                                     # Initial uncertainty
        kf.R *= 1_000_00                              # High measurement noise
        kf.Q = np.array([[5.0, 0.0],                  # Process noise
                         [0.0, 0.5]])

    elif mode == "ABGF":
        # Kalman filter with 3 state variables: position, velocity, and acceleration
        kf = KalmanFilter(dim_x=3, dim_z=1)
        kf.x = np.array([
            data[0],
            data[1] - data[0],
            data[2] - 2 * data[1] + data[0]
        ])  # Initial position, velocity, acceleration

        kf.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])  # State transition model

        kf.H = np.array([[1, 0, 0]])  # Observation model
        kf.P *= 1
        kf.R *= 1_000_00              # High measurement noise
        q = 1
        kf.Q = q * np.array([
            [dt**4 / 4, dt**3 / 2, dt**2 / 2],
            [dt**3 / 2, dt**2, dt],
            [dt**2 / 2, dt, 1]
        ])  # Process noise matrix

    else:
        raise ValueError("Unknown mode. Use 'ABF' or 'ABGF'.")

    results = []

    for z in data:
        kf.predict()
        kf.update([z])
        results.append(kf.x[0])  # Store position estimate (first state variable)

    return np.array(results)

# ------------------------ 🔹 Visualization of Filtered Results per Cryptocurrency ------------------------

def visualize_all_methods_per_crypto(crypto_name, real_data, filtered_dict, dates, mode):
    """
    Visualizes real vs. smoothed data for a given cryptocurrency using various filtering methods.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., "Bitcoin")
    real_data : np.ndarray
        Original (raw) price data
    filtered_dict : dict[str, np.ndarray]
        Dictionary mapping method names to their smoothed time series
    dates : pd.DatetimeIndex
        Array of datetime values matching the time series
    mode : str
        Filtering mode used ("ABF" or "ABGF")
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates, real_data, label="Real Data", color="black", linewidth=1.5)

    colors = ["darkorange", "green", "purple"]
    for i, (method_name, filtered_series) in enumerate(filtered_dict.items()):
        r2 = r2_score(real_data, filtered_series)
        plt.plot(dates, filtered_series,
                 label=f"{method_name} (R² = {r2:.3f})",
                 color=colors[i],
                 linewidth=2)

    # 🔹 Adjust y-axis range based on raw values
    min_y = np.min(real_data)
    max_y = np.max(real_data)
    margin = 0.1 * (max_y - min_y)
    plt.ylim(min_y - margin, max_y + margin)

    plt.title(f"{crypto_name} – {mode} Filtering", fontsize=12)
    plt.xlabel("Year")
    plt.ylabel("Price, USD")
    plt.xticks(
        pd.date_range(start="2021-01-01", end=dates[-1], freq="YS"),
        labels=[str(d.year) for d in pd.date_range(start="2021-01-01", end=dates[-1], freq="YS")],
        rotation=45
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------ 🔹 Quality Metrics Reporting ------------------------

def print_metrics_table(metrics, method_name):
    """
    Prints a formatted table with quality metrics for each cryptocurrency
    and a given filtering method.

    Parameters
    ----------
    metrics : list of tuples
        Each tuple should contain (name, R², MAE, RMSE)
    method_name : str
        Name of the smoothing method ('ABF' or 'ABGF')
    """
    print(f"\n📊 Quality Metrics for {method_name} Method:")
    print(f"{'Cryptocurrency':<15} | {'R²':>6} | {'MAE':>10} | {'RMSE':>10}")
    print("-" * 47)
    for name, r2, mae, rmse in metrics:
        print(f"{name:<15} | {r2:>6.4f} | {mae:>10.4f} | {rmse:>10.4f}")

# ------------------------ 🔹 Main Execution Block ------------------------

if __name__ == "__main__":
    # --- Load cryptocurrency data ---
    BTC = load_real_data("btc-usd-max.xls")
    ETH = load_real_data("eth-usd-max.xls")
    SOL = load_real_data("sol-usd-max.xls")
    print("✅ Data successfully loaded for all 3 cryptocurrencies.")

    # --- Mode selection ---
    print("\nSelect filtering mode:")
    print("1 - Alpha-Beta Filtering (ABF)")
    print("2 - Alpha-Beta-Gamma Filtering (ABGF)")
    mode_input = input("Your choice: ")

    if mode_input == "1":
        mode = "ABF"
        print("\n🔹 Running mode: ABF (Alpha-Beta Filtering)\n")
    elif mode_input == "2":
        mode = "ABGF"
        print("\n🔹 Running mode: ABGF (Alpha-Beta-Gamma Filtering)\n")
    else:
        print("❌ Invalid mode selected. Exiting.")
        exit()

    # --- Filter reset period selection ---
    print("\nSelect filter reset policy:")
    print("1 - Default recommended (ABF: 365 days, ABGF: 100 days)")
    print("2 - Custom (enter number of days manually)")
    reset_choice = input("Your choice: ")

    if reset_choice == "1":
        chunk_size = 365 if mode == "ABF" else 100
        print(f"🔹 Using recommended reset interval: {chunk_size} days\n")
    elif reset_choice == "2":
        user_value = input("Enter desired reset interval (in days): ")
        if user_value.isdigit():
            chunk_size = int(user_value)
            print(f"🔹 Using custom reset interval: {chunk_size} days\n")
        else:
            print("❌ Invalid value. Exiting.")
            exit()
    else:
        print("❌ Invalid selection. Exiting.")
        exit()

    # --- Prepare input dataset ---
    crypto_dict = {
        "Bitcoin": BTC,
        "Ethereum": ETH,
        "Solana": SOL
    }

    # --- Initialize results ---
    filtered_manual = {}
    metrics_manual = {}
    dates_dict = {}

    # --- Apply custom filter (manual implementation) ---
    for name, data in crypto_dict.items():
        smoothed = apply_manual_filter_in_chunks(data, mode, chunk_size=chunk_size)
        filtered_manual[name] = smoothed

        r2 = r2_score(data, smoothed)
        mae = mean_absolute_error(data, smoothed)
        rmse = np.sqrt(mean_squared_error(data, smoothed))
        metrics_manual[name] = (r2, mae, rmse)

        dates_dict[name] = pd.date_range(start="2021-01-01", periods=len(data), freq="D")

    # --- Apply FilterPy-based Kalman filter ---
    filtered_filterpy = {}
    metrics_filterpy = {}

    for name, data in crypto_dict.items():
        smoothed_fp = apply_filterpy_filter_in_chunks(data, mode, chunk_size)
        filtered_filterpy[name] = smoothed_fp

        r2 = r2_score(data, smoothed_fp)
        mae = mean_absolute_error(data, smoothed_fp)
        rmse = np.sqrt(mean_squared_error(data, smoothed_fp))
        metrics_filterpy[name] = (r2, mae, rmse)

    # --- Visualization per cryptocurrency ---
    for name in crypto_dict:
        visualize_all_methods_per_crypto(
            crypto_name=name,
            real_data=crypto_dict[name],
            filtered_dict={
                "Manual": filtered_manual[name],
                "FilterPy": filtered_filterpy[name]
            },
            dates=dates_dict[name],
            mode=mode
        )

    # --- Print summary metrics ---
    metrics_list = []

    for name in crypto_dict:
        metrics_list.append((f"{name} (Manual)", *metrics_manual[name]))
        metrics_list.append((f"{name} (FilterPy)", *metrics_filterpy[name]))

    print_metrics_table(metrics_list, method_name=mode)

# ============================================================================================
# 🔍 Analysis of Filtering Results – Model Verification and Insights
# ============================================================================================

"""
In this project, two Kalman filter modes were implemented:

1) Alpha-Beta Filter (ABF)
2) Alpha-Beta-Gamma Filter (ABGF)

Each of them was tested using:
- a custom manual implementation
- the FilterPy library

Data was sourced from Coingecko for Bitcoin, Ethereum, and Solana (2021–present).

Main observations:

🔹 Manual Implementation:
- Delivered the best performance in ABF mode (especially for Bitcoin and Solana).
- By using chunk-based reinitialization (e.g., every 365 days), the model avoided overfitting
  and better adapted to structural changes in the data.
- ABGF mode was less stable and prone to either mimicking input or over-smoothing.

🔹 FilterPy:
- Initially showed overfitting ("sticking" to true values).
- Improved after:
    • Proper initialization of kf.x
    • Manual tuning of kf.P, kf.R, and kf.Q
    • Applying the filter in segments (chunk-based)
- Still, ABF remained more robust than ABGF across all tested cryptocurrencies.
- ABGF required fine-tuned parameters (e.g., kf.P=1000, kf.R=1e8, kf.Q=0.1) to behave acceptably.

💡 Additional Notes:
- ABF is generally more appropriate for financial time series, which often exhibit trend-based 
  movement rather than uniform acceleration.
- Bitcoin showed the best R² and visual quality, while Ethereum was more erratic and difficult to filter.
- R² should not be overinterpreted in Kalman filtering — the aim is tracking, not prediction.

✅ Conclusion:
- The best-performing method: Manual ABF with chunk_size = 365
- FilterPy is usable, but only with careful tuning
- Model complexity does not guarantee better results
- Filter performance depends heavily on data characteristics (volatility, trend, noise)
"""