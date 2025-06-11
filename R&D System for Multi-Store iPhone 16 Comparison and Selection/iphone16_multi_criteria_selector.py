# --------- 🔹 Multi-Criteria Selection of iPhone 16 Models from Ukrainian Online Retailers (R&D Project) ------------
'''
Author: Ivan Tatarchuk

Project Overview:
This project implements a complete data-driven pipeline for selecting the most suitable
iPhone 16 model across three major Ukrainian online retailers: Rozetka, Citrus, and Comfy.

The solution covers:
- Real-time and offline web scraping
- Data preprocessing and enrichment
- Optional filtering by subjective preferences (e.g., color)
- Multi-criteria scoring using expert-based weighted scales
- 3D OLAP-style visualization for decision support

The project is designed to simulate a realistic product selection experience,
taking into account key user-relevant criteria: price, memory, model, delivery speed, and store reputation.

Dependencies:
------------------------------------------------
pip                          24.0
numpy                        1.26.4
pandas                       2.2.3
xlrd                         2.0.1
matplotlib                   3.10.1
requests                     2.32.3
selenium                     4.31.0
regex                        2024.11.6
webdriver-manager            4.0.2
bs4                          0.0.2
'''

# 🔧 Required Libraries

import os
import time
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt
import requests
import json

# 🍊 Citrus Parser — JSON-LD Product Extraction

def parse_citrus_from_jsonld(urls):
    """
    Parses iPhone product listings from the Citrus.ua website using embedded JSON-LD metadata.

    Each page is scanned for <script type="application/ld+json"> blocks containing structured product data.
    Extracted fields include:
    - Model name
    - Memory size
    - Price (UAH)
    - Product link
    - Full name string for further parsing

    :param urls: List of Citrus product listing URLs
    :return: Pandas DataFrame with structured product data
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    results = []

    for url in urls:
        print(f"📥 Downloading Citrus: {url}")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")

        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "ItemList":
                    for item in data.get("itemListElement", []):
                        name = item.get("name", "").strip()
                        model = next(
                            (m for m in ["iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16"]
                             if m.lower() in name.lower()),
                            "Unknown"
                        )
                        memory_match = re.search(r"(1[\s]?(ТБ|TB|Tb)|512|256|128)", name)
                        memory = memory_match.group(1).replace(" ", "") if memory_match else "Unknown"
                        if memory in ["1ТБ", "1TB", "1Tb"]:
                            memory = "1024"
                        price = int(item.get("offers", {}).get("price", 0))
                        link = item.get("url", "")

                        results.append({
                            "Магазин": "Citrus",
                            "Модель": model,
                            "Памʼять": memory,
                            "Ціна": price,
                            "Посилання": link,
                            "Деталі": name
                        })
            except:
                continue

    return pd.DataFrame(results)

# 🛒 Comfy Parser — Offline HTML Parsing

def parse_comfy_from_html():
    """
    Parses iPhone product data from saved local HTML files obtained from Comfy.ua.

    This function is used when live scraping is not possible due to dynamic JavaScript content.
    It extracts:
    - Product title
    - URL
    - Price (UAH)
    - Memory size
    - Model category

    :return: Pandas DataFrame with structured product data from Comfy
    """
    folder = "Comfy_HTML"
    files = [
        "manual_comfy_iphone16.html",
        "manual_comfy_iphone16plus.html",
        "manual_comfy_iphone16pro.html",
        "manual_comfy_iphone16promax.html"
    ]
    data = []

    for file in files:
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            cards = soup.select("a.prdl-item__name")
            for card in cards:
                try:
                    name = card.get("title") or card.text.strip()
                    url = card.get("href")
                    price_tag = card.find_next("div", class_="products-list-item-price__actions-price-current")
                    price = int(price_tag.text.strip().replace("₴", "").replace(" ", "")
                                .replace("\xa0", ""))
                    model = next((m for m in ["iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16"]
                                  if m.lower() in name.lower()), "Unknown")
                    memory_match = re.search(r"(1[\s]?(ТБ|TB|Tb)|512|256|128)", name)
                    memory = memory_match.group(1).replace(" ", "") if memory_match else "Unknown"
                    if memory in ["1ТБ", "1TB", "1Tb"]:
                        memory = "1024"
                    data.append({
                        "Магазин": "Comfy",
                        "Модель": model,
                        "Памʼять": memory,
                        "Ціна": price,
                        "Посилання": url,
                        "Деталі": name
                    })
                except:
                    continue

    return pd.DataFrame(data)

# 🛍️ Rozetka HTML Downloader (Selenium-Based)

def download_rozetka_html(url, filename):
    """
    Downloads a full page from Rozetka using headless Selenium automation and saves it as an HTML file.

    Used when dynamic content requires JavaScript rendering (infinite scroll simulation included).

    :param url: Target Rozetka product listing URL
    :param filename: Local file path to save the downloaded HTML
    """
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    print(f"📥 Downloading Rozetka: {url}")
    driver.get(url)
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    driver.quit()

# 🛍️ Rozetka Parser — Offline HTML Processing

def parse_rozetka_from_html(file_paths):
    """
    Parses saved local HTML pages from Rozetka to extract product information.

    Each file is expected to contain a listing of iPhones. The parser locates:
    - Product name
    - Price
    - Memory size
    - Model type
    - Direct product link

    :param file_paths: List of paths to saved Rozetka HTML files
    :return: Pandas DataFrame with extracted product details
    """
    results = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            links = soup.select("a.tile-title")
            for link_tag in links:
                try:
                    name = link_tag.get("title") or link_tag.text.strip()
                    link = link_tag.get("href")
                    price_tag = link_tag.find_next("div", class_="price")
                    price = int(price_tag.text.strip().replace("₴", "").replace(" ", "")
                                .replace(" ", ""))
                    model = next((m for m in ["iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16"]
                                  if m.lower() in name.lower()), "Unknown")
                    memory_match = re.search(r"(1[\s]?(ТБ|TB|Tb)|512|256|128)", name)
                    memory = memory_match.group(1).replace(" ", "") if memory_match else "Unknown"
                    if memory in ["1ТБ", "1TB", "1Tb"]:
                        memory = "1024"
                    results.append({
                        "Магазин": "Rozetka",
                        "Модель": model,
                        "Памʼять": memory,
                        "Ціна": price,
                        "Посилання": link,
                        "Деталі": name
                    })
                except:
                    continue

    return pd.DataFrame(results)

# 📊 Matrix Generation for Multi-Criteria Analysis

def transform_csv_to_matrix_format(csv_path: str, output_path: str = "iphone16_matrix_ready.xlsx") -> str:
    """
    Builds an Excel matrix for multi-criteria decision analysis (MCDA) using the parsed iPhone dataset.

    Args:
        csv_path (str): Path to the CSV file containing parsed iPhone listings.
        output_path (str): Output Excel file path in matrix format (for use in Voronin / OLAP analysis).

    Output format:
        - Column 1: Criterion names
        - Column 2: Criterion weights
        - Columns 3..N: Alternatives (products), labeled as "<Store> <Index>"
        - Last column: Strategy for each criterion ("min" / "max")
    """

    # 🧠 Criterion Breakdown and Weight Justification:

    # Price (weight 0.4, min): Price remains the dominant factor for most Ukrainian customers,
    # who seek the best price-to-feature ratio.

    # Memory (0.2, max): Storage capacity strongly influences user experience (apps, media).
    # Larger memory is generally preferred, especially since iPhones lack SD card support.

    # Store Rating (0.1, max): Based on reviews from Google, Hotline.ua, Trustpilot.
    # Rozetka is rated highest; Citrus lowest.

    # Delivery Speed (0.1, min): Customers care about shipping speed.
    # Based on user feedback, Rozetka delivers within 1–2 days, Comfy ~2–3, Citrus up to 5.

    # Model Score (0.2, max): Derived from comparisons on GSM Arena, ITC.ua, and MacRumors.
    # Scores are assigned on a scale: iPhone 16 = 1.0 → 16 Plus = 1.25 → Pro = 1.5 → Pro Max = 2.0.

    df = pd.read_csv(csv_path)
    df.columns = ["Store", "Model", "Memory", "Price", "Link", "Details", "Color"]

    # --- Add derived criteria: ratings and delivery ---
    rating_map = {"Rozetka": 4.8, "Comfy": 4.5, "Citrus": 4.3}
    delivery_map = {"Rozetka": 2, "Comfy": 3, "Citrus": 4}

    df["Store Rating"] = df["Store"].map(rating_map)
    df["Delivery Time"] = df["Store"].map(delivery_map)

    def model_score(text: str) -> float:
        if "Pro Max" in text:
            return 2.0
        elif "Pro" in text:
            return 1.5
        elif "Plus" in text:
            return 1.25
        else:
            return 1.0

    df["Model Score"] = df["Details"].apply(model_score)

    # 📌 Price is not normalized directly, but mapped to a scale that reflects consumer perception.
    # Apple uses price tiers (e.g., 39999, 44999, 54999), so minor differences are less meaningful.
    # Standard normalization leads to unstable results; a stepped scoring system works better.

    def price_score(price):
        if price <= 40000:
            return 0.5  # lowest = best
        elif price <= 60000:
            return 0.75
        elif price <= 80000:
            return 0.9
        else:
            return 1.0  # highest = worst (for 'min')

    df["Price Score"] = df["Price"].apply(price_score)

    # Memory scoring is treated similarly — discrete stepwise scale
    memory_mapping = {128: 0.5, 256: 0.75, 512: 0.9, 1024: 1.0}
    df["Memory Score"] = df["Memory"].astype(int).map(memory_mapping)

    # --- Define criteria, weights, and strategies ---
    criteria = ["Price Score", "Memory Score", "Model Score", "Delivery Time", "Store Rating"]
    weights = [0.4, 0.2, 0.2, 0.1, 0.1]
    strategies = ["min", "max", "max", "min", "max"]

    # --- Construct matrix: each row = criterion, columns = products ---
    matrix = []
    for i, criterion in enumerate(criteria):
        row = [criterion, weights[i]]
        for _, row_data in df.iterrows():
            row.append(row_data[criterion])
        row.append(strategies[i])
        matrix.append(row)

    # --- Generate column headers: "Criterion", "Weight", "Store 1", ..., "Store N", "Strategy" ---
    alt_names = [f"{row['Store']} {idx + 1}" for idx, row in df.iterrows()]
    columns = ["Criterion", "Weight"] + alt_names + ["Strategy"]

    result_df = pd.DataFrame(matrix, columns=columns)
    result_df.to_excel(output_path, index=False)

    print(f"📊 Criterion matrix saved to: {output_path}")
    return output_path

# 🧮 Matrix Utilities and Scoring (Voronin Method)

def matrix_generation(file_name):
    """
    Loads a decision matrix from an Excel file and extracts:
    - Criterion values (matrix: criteria x alternatives)
    - Weights array
    - Strategy list ('min' / 'max')

    :param file_name: Path to the Excel file containing the matrix
    :return: (values_matrix, weights_array, strategy_list)
    """
    sample_data = pd.read_excel(file_name, header=None, skiprows=1)  # Skip column headers

    weights = sample_data.iloc[:, 1].to_numpy(dtype=np.float64)  # Column 2: weights
    values_matrix = sample_data.iloc[:, 2:-1].to_numpy(dtype=np.float64)  # Columns 3 to N-1: values
    strategies = sample_data.iloc[:, -1].astype(str).str.strip().str.lower().tolist()  # Last column: strategy

    return values_matrix, weights, strategies

def matrix_adapter(matrix, row_index):
    """
    Extracts a single criterion vector from the full decision matrix.

    :param matrix: 2D NumPy array (criteria x products)
    :param row_index: Index of the criterion to extract
    :return: 1D NumPy array of values for the selected criterion
    """
    return matrix[row_index, :]

def Voronin(matrix_file: str) -> tuple:
    """
    Computes an integral performance score for each product using the Voronin method.
    This assumes that all criteria have already been normalized or scaled (e.g., [0.5...1.0]).

    :param matrix_file: Path to Excel file in matrix format
    :return: (original_matrix, normalized_matrix, score_vector, product_names)
    """
    df = pd.read_excel(matrix_file)
    weights = df.iloc[:, 1].to_numpy(dtype=np.float64)
    strategies = df.iloc[:, -1].astype(str).str.strip().str.lower().tolist()
    names = df.columns[2:-1].tolist()
    matrix = df.iloc[:, 2:-1].to_numpy(dtype=np.float64)

    num_criteria, num_products = matrix.shape

    # ❗ No normalization needed — values already represent final [0.5–1.0] scores
    normalized_matrix = matrix.copy()

    scor = np.zeros(num_products)
    for j in range(num_products):
        for i in range(num_criteria):
            scor[j] += weights[i] * normalized_matrix[i, j]

    return matrix, normalized_matrix, scor, names

# 📊 OLAP Cube Visualization — 3D Criteria Impact Graph

def OLAP_cube(matrix, normalized_matrix, scor, file_name):
    """
    Builds a 3D OLAP-style bar chart that visualizes the contribution of each criterion
    to the final integrated score (scor) for each product.

    Key Features:
    - Supports any number of criteria and alternatives
    - Dynamically scales axes based on input matrix
    - Renders both normalized criteria bars and real scor values
    - Highlights the best and worst products by score

    :param matrix: Original value matrix (criteria x products)
    :param normalized_matrix: Matrix after normalization (e.g., via Voronin pipeline)
    :param scor: Final scoring vector (integral performance indicator)
    :param file_name: Excel file path (used to read criterion labels)
    """
    num_criteria, num_products = matrix.shape
    xg = np.arange(num_products)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    clr = ['#4bb2c5', '#c5b47f', '#EAA228', '#579575', '#839557',
           '#958c12', '#953579', '#4b5de4', '#ff8c00', '#1e90ff']

    # 🔁 Read criterion labels from Excel column
    df_labels = pd.read_excel(file_name)
    criteria_labels = df_labels["Criterion"].tolist()

    # 🔁 Normalize each criterion row (MinMax) to avoid scale distortion
    for i in range(num_criteria):
        norm_row = normalized_matrix[i] / np.max(normalized_matrix[i])
        ax.bar(xg, norm_row, zs=i, zdir='y', color=clr[i % len(clr)])

    # ⚫ Add final integral score (non-normalized) as black bar
    ax.bar(xg, scor, zs=num_criteria, zdir='y', color='black')

    # 🏷️ Annotate best and worst scores
    best_idx = np.argmin(scor)
    worst_idx = np.argmax(scor)

    ax.text(xg[best_idx], num_criteria + 0.5, scor[best_idx], f"{scor[best_idx]:.4f}",
            ha='center', va='bottom', fontsize=9, color='green')
    ax.text(xg[worst_idx], num_criteria + 0.5, scor[worst_idx], f"{scor[worst_idx]:.4f}",
            ha='center', va='bottom', fontsize=9, color='red')

    # 🧾 Set Y-axis labels to criterion names + scor
    ax.set_yticks(np.arange(num_criteria + 1))
    ax.set_yticklabels(criteria_labels + ['Total_Scor'], fontsize=9)

    ax.set_xlabel("Phones (Top 5 alternatives)")
    ax.set_zlabel("Value")

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plt.show()

# 🚀 MAIN EXECUTION LOGIC

if __name__ == "__main__":
    print("🔰 Choose mode:")
    print("1 — Full parsing of all iPhone 16 models from Citrus, Comfy, and Rozetka (may take up to 1 minute)\n")
    print("2 — Skip parsing and proceed to optimal selection using a previously saved CSV\n"
          "⏩ Recommended if you've already parsed or if the HTML structure changed\n"
          "⏩ The CSV file will still be refreshed and reprocessed")
    mode = input("Enter mode number: ")

    if mode == "1":
        print("⏳ Please wait... Fetching data from three online stores. This may take up to 1 minute.\n")

        # --- Citrus ---
        citrus_urls = [
            "https://www.ctrs.com.ua/smartfony/brand-apple/seriya_iphone-16_iphone-16-plus_iphone-16-pro_iphone-16-pro-max/",
            "https://www.ctrs.com.ua/smartfony/brand-apple/seriya_iphone-16_iphone-16-plus_iphone-16-pro_iphone-16-pro-max/page_2/"
        ]
        df_citrus = parse_citrus_from_jsonld(citrus_urls)
        print(f"✅ Citrus: {len(df_citrus)} items")

        # --- Comfy ---
        df_comfy = parse_comfy_from_html()
        print(f"✅ Comfy: {len(df_comfy)} items")

        # --- Rozetka ---
        os.makedirs("Rozetka_HTML", exist_ok=True)
        rozetka_urls = [
            "https://rozetka.com.ua/ua/mobile-phones/c80003/producer=apple;series=iphone-16/",
            "https://rozetka.com.ua/ua/mobile-phones/c80003/producer=apple;series=iphone-16-plus/",
            "https://rozetka.com.ua/ua/mobile-phones/c80003/producer=apple;series=iphone-16-pro/",
            "https://rozetka.com.ua/ua/mobile-phones/c80003/producer=apple;series=iphone-16-pro-max/"
        ]
        rozetka_paths = []
        for i, url in enumerate(rozetka_urls, start=1):
            fname = f"Rozetka_HTML/debug_rozetka_{i}.html"
            download_rozetka_html(url, fname)
            rozetka_paths.append(fname)

        df_rozetka = parse_rozetka_from_html(rozetka_paths)
        print(f"✅ Rozetka: {len(df_rozetka)} items\n")

        # --- Merge datasets ---
        df_all = pd.concat([df_citrus, df_comfy, df_rozetka], ignore_index=True)
        df_all.sort_values(by="Ціна", inplace=True)

        # Rename columns to English for consistency
        df_all.columns = ["Store", "Model", "Memory", "Price", "Link", "Details"]

        # --- Save combined dataset ---
        df_all.to_csv("iphone16_all_prices_final.csv", index=False, encoding="utf-8-sig")
        print(f"📦 Saved {len(df_all)} entries to: iphone16_all_prices_final.csv")
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_columns", None)
        print(df_all.head(10).to_string(index=False))

    elif mode == "2":
        print("📂 Loading saved CSV file: iphone16_all_prices_final.csv\n"
              "⏩ Data is sorted by ascending price")
        df_all = pd.read_csv("iphone16_all_prices_final.csv")
        print(f"✅ Loaded {len(df_all)} items\n")
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_columns", None)
        print(df_all.head(10).to_string(index=False))

    else:
        print("❌ Invalid mode. Exiting.")

    # --- Continue with matrix generation and OLAP analysis ---
    if mode in ["1", "2"]:

        # --- Extract color from product details ---
        def extract_color(text):
            colors = {
                "Black Titanium": "Black Titanium",
                "Natural Titanium": "Natural Titanium",
                "Desert Titanium": "Desert Titanium",
                "White Titanium": "White Titanium",
                "Ultramarine": "Ultramarine",
                "Black": "Black",
                "Teal": "Teal",
                "Pink": "Pink",
                "White": "White",
            }
            for key in colors:
                if key.lower() in text.lower():
                    return colors[key]
            return "Unknown"

        df_all["Color"] = df_all["Details"].apply(extract_color)

        # --- Ask user to optionally filter by color ---
        print("\n🎨 Color is a subjective but important filter in product selection.")
        print("📌 Choose a preferred color below (optional):")
        color_options = {
            "1": "Black",
            "2": "Teal",
            "3": "Pink",
            "4": "White",
            "5": "Ultramarine",
            "6": "Black Titanium",
            "7": "Natural Titanium",
            "8": "Desert Titanium",
            "9": "White Titanium"
        }
        for k, v in color_options.items():
            print(f"{k} — {v}")
        color_choice = input("Enter the number of your preferred color or press Enter to skip: ").strip()

        if color_choice in color_options:
            selected_color = color_options[color_choice]
            df_all = df_all[df_all["Color"] == selected_color]
            print(f"✅ Selected color: {selected_color}")
        else:
            print("✅ No color filter applied — using all products.")

        # --- Save filtered dataset ---
        filtered_csv_path = "iphone16_filtered_by_color.csv"
        df_all.columns = ["Store", "Model", "Memory", "Price", "Link", "Details", "Color"]
        df_all.to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")

        matrix_file = transform_csv_to_matrix_format(filtered_csv_path)

        # --- Plot price steps chart ---
        df_temp = pd.read_csv("iphone16_all_prices_final.csv")
        plt.figure(figsize=(10, 4))
        sorted_prices = df_temp["Price"].sort_values().reset_index(drop=True)
        plt.plot(sorted_prices.index, sorted_prices.values, drawstyle='steps-post', marker='o')
        plt.title("iPhone 16 Price Structure — Tiered Pricing Effect")
        plt.xlabel("Product Index (sorted)")
        plt.ylabel("Price (UAH)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Run scoring pipeline ---
        matrix, norm_matrix, scor, names = Voronin(matrix_file)

        # --- Display scoring results ---
        print("\nFinal scores (scor) — product ranking:")
        sorted_indices = np.argsort(scor)  # Lower = better

        for rank, idx in enumerate(sorted_indices, 1):
            print(f"{rank:3}. {names[idx]}: {scor[idx]:.6f}")

        # --- Top 3 recommendations ---
        print("\n🔍 Top 3 recommended models (filtered by color):\n")
        labels = ["🥇 Best Pick", "🥈 Alternative 1", "🥉 Alternative 2"]
        for i, idx in enumerate(sorted_indices[:3]):
            row = df_all.iloc[idx]
            print(f"{labels[i]}: {names[idx]}")
            print(f"📱 Model: {row['Model']}")
            print(f"💾 Memory: {row['Memory']} GB")
            print(f"💰 Price: {row['Price']} UAH")
            print(f"🏪 Store: {row['Store']}")
            print(f"📝 Details: {row['Details']}")
            print(f"🔗 Link: {row['Link']}")

        # --- Highlight best product separately ---
        print("✅ Best overall choice:")
        idx = sorted_indices[0]
        row = df_all.iloc[idx]
        print(f"📱 {row['Model']} / {row['Memory']}GB — {row['Price']} UAH ({row['Store']})")
        print(f"🔗 {row['Link']}")

        # --- Display top 5 in OLAP Cube ---
        sorted_indices = np.argsort(scor)
        selected_indices = sorted_indices[:5]

        matrix_vis = matrix[:, selected_indices]
        norm_vis = norm_matrix[:, selected_indices]
        scor_vis = scor[selected_indices]

        OLAP_cube(matrix_vis, norm_vis, scor_vis, file_name=matrix_file)

'''
📊 Results Analysis — Validation of the Mathematical Model and Decision Framework
--------------------------------------------------------------------------------

✅ This project delivers a full-featured multi-criteria selection system for iPhone 16 models,
leveraging OLAP-style visualization, data mining, flexible scaling, and a custom expert scoring system 
to simulate realistic consumer decision-making.

🔹 1. Data Parsing from Three Sources (Citrus, Comfy, Rozetka):
- Citrus: The simplest case — clean JSON-LD (`application/ld+json`) structure, easy access to product attributes 
(model, price, memory, color, link).
- Rozetka: More complex DOM structure, required Selenium and dynamic HTML loading.
- Comfy: Most challenging — content hidden in nested structures, required manual HTML saving and local parsing.
- All datasets were normalized into a unified format, resulting in the final dataset `iphone16_all_prices_final.csv`.

🔹 2. Dual Execution Modes:
- Mode 1 — Full pipeline with live parsing from all sources. Recommended for fresh analysis or when data changes.
- Mode 2 — Works from saved CSV, ideal for quick re-analysis or offline use.

🔹 3. Color Filtering — Subjective but User-Driven:
- Product color is auto-extracted from the `Деталі` column using keyword detection.
- The user can filter by color (via 1–9 selection); a new filtered CSV is created.
- This avoids repetition of the same models in different colors and improves result clarity.
- Filtering by color does not affect `scor`, only narrows the candidate pool.

🔹 4. Criterion Matrix Construction:
- The function `transform_csv_to_matrix_format()` converts the parsed CSV into a decision matrix.
- A deliberate choice was made **not** to apply standard min-max normalization.
- 📌 Instead, prices are scored using a stepped expert scale that reflects Apple’s tiered pricing logic:
  Apple uses fixed pricing levels (e.g., 39999, 44999, 54999), and stores rarely diverge from them.
  As a result, classic normalization caused large score jumps from minor price shifts.
  Consumers are especially sensitive to entry-level affordability, while differences at higher tiers
  are perceived less critically — hence the following manual scale:

    - ≤ 39999 → 0.5
    - ≤ 59999 → 0.75
    - ≤ 79999 → 0.9
    - > 80000 → 1.0

- Memory is scored similarly: 128 → 0.5, 256 → 0.75, 512 → 0.9, 1024 → 1.0
- Model scoring: iPhone 16 = 1.0 → Plus = 1.25 → Pro = 1.5 → Pro Max = 2.0

🔹 5. Model Selection:
- Using the `Voronin()` function, a weighted score `scor` is calculated per product.
- This score is used as the single-ranking metric for evaluation.
- Full ranking output is shown with numerical values.

🔹 6. Top-3 Recommendation Output:
- The top 3 models are displayed based on **user-selected color**, not the full dataset.
- The best product is shown as “Best Pick”, followed by “Alternative 1” and “Alternative 2”.
- Each listing includes model, memory, price, store name, product link, and details.

🔹 7. OLAP Cube Visualization:
- A 3D cube visualizes criterion influence per product.
- Unlike previous implementations, this version shows **only the top-5 ranked models**.
- Each criterion is mapped to a Y-axis slice; each product forms a vertical column stack.
- The integral score (`scor`) is shown as a **black bar**, scaled in line with other axes.
- This provides a clear and distortion-free comparison of scores and criterion values.

---

🔚 Summary:

- ✅ End-to-end system from real-world data collection to multi-criteria evaluation
- ✅ Modular, extensible architecture with color filtering, flexible scoring, and visualization
- ✅ Realism enhanced via expert scoring and subjective UX-driven parameters
- ✅ Easily adaptable to other product categories: Samsung, MacBook, Xiaomi, etc.
'''