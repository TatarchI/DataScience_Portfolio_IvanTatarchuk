
# 📱 Multi-Criteria Selection of iPhone 16 Models from Ukrainian Online Retailers

Author: Ivan Tatarchuk  
Type: Data Science R&D Project  
Category: Product Decision Support System  
Technologies: Web Scraping, Data Cleaning, Expert Scoring, MCDA, Voronin Method, OLAP 3D Visualization  

---

## 🧠 Project Overview

This project implements an end-to-end pipeline for selecting the optimal iPhone 16 model across three Ukrainian e-commerce platforms: Rozetka, Citrus, and Comfy.  
It is built around real-world data collection, flexible filtering, expert-defined scoring, and OLAP-based multi-criteria analysis.

---

## 🛠 Features

- 🌐 Live & offline web scraping from Citrus, Comfy, and Rozetka
- 🧹 Data cleaning and unification across all platforms
- 🎨 Optional filtering by user-preferred color
- 🧮 Multi-criteria scoring:
  - Price (minimize)
  - Memory (maximize)
  - Model quality (maximize)
  - Delivery speed (minimize)
  - Store rating (maximize)
- 🧾 Expert weight system & manual stepwise normalization
- 📊 Final Voronin-based score ranking
- 📦 3D OLAP-style visualization of score decomposition

---

## 📂 Project Structure

```
├── Comfy_HTML/                                   ← Offline HTML files for Comfy
├── Rozetka_HTML/                                 ← Dynamically scraped HTML files via Selenium
├── iphone16_all_prices_final.csv                 ← Unified raw dataset
├── iphone16_filtered_by_color.csv                ← Optional filtered dataset with final_result (only the last selected color)
├── iphone16_matrix_ready.xlsx                    ← Scoring matrix for Voronin method
├── README.md                                     ← Full description of project
├── visualizations_logs                           ← List of output graphics and logs
└── iphone16_multi_criteria_selector.py           ← Full project pipeline script
```

## 📦 Skills Demonstrated

- Web Scraping (Live & Offline HTML)
- JSON-LD Parsing (Structured Data)
- Multi-Criteria Decision Analysis (MCDA)
- Expert-Based Weighting System
- Data Cleaning & Transformation
- Stepwise Scoring (Nonlinear Normalization)
- Custom OLAP-style Visualization (3D)
- Product Ranking & Recommendation Logic

## 🧠 Technologies Used

| Component             | Library / Tool                  |
|----------------------|-------------------------------   |
| Web Scraping         | Selenium 4.31.0, Requests 2.32.3 |
| HTML Parsing         | BeautifulSoup 4 (bs4)            |
| Data Handling        | Pandas 2.2.3, NumPy 1.26.4       |
| Visualization        | Matplotlib 3.10.1                |
| Browser Driver Mgmt  | webdriver-manager 4.0.2          |
| Excel Support        | xlrd 2.0.1                       |

---

## ⚙️ Execution Modes

- Mode 1: Full live parsing of product listings (Citrus, Comfy, Rozetka)
- Mode 2: Load pre-saved CSV dataset for fast re-analysis

---

## 📈 Scoring System (Weights + Strategy)

| Criterion      | Weight | Strategy | Details                                                   |
|----------------|--------|----------|---------------------------------------------------------- |
| Price          | 0.40   | Min      | Stepwise: ≤40k→0.5, ≤60k→0.75, ≤80k→0.9, >80k→1.0        |
| Memory         | 0.20   | Max      | Stepwise: 128→0.5, 256→0.75, 512→0.9, 1024→1.0            |
| Model Score    | 0.20   | Max      | iPhone 16 = 1.0 → Plus = 1.25 → Pro = 1.5 → Pro Max = 2.0 |
| Delivery Time  | 0.10   | Min      | Rozetka=2d, Comfy=3d, Citrus=4d                           |
| Store Rating   | 0.10   | Max      | Based on reviews: Rozetka (4.8), Comfy (4.5), Citrus (4.3)|

---

## 🧮 Scoring Method: Voronin Approach

The weighted sum `scor` is calculated for each product using:
```python
scor[j] += ∑ (weight_i × value_ij)
```
Where:
- i = criterion index
- j = product index
- weight_i = predefined weight per criterion
- value_ij = normalized or expert-scaled score

---

## 🏆 Output

- 🥇 Top 3 recommended models (based on filtered data if selected)
- 🧾 Full product rankings with individual scores
- 📊 3D OLAP Cube for top 5 models

---

## 💡 Business Value

This project simulates a realistic consumer product choice system that goes beyond price comparison.  
It demonstrates a flexible R&D architecture for supporting multi-criteria decisions based on dynamic or offline data sources.

Applicable to:
- Smartphones, laptops, TVs
- Cross-store offer comparisons
- Automated retail decision agents

---

## 🚀 Future Improvements

- Add real user reviews scraping (e.g., Trustpilot / Hotline.ua)
- Extend to Android flagships (Samsung S series, Pixel)
- GUI or web dashboard for interactive use
- API version for enterprise integration

## 📬 Contact

Feel free to reach out or fork the project for adaptation to your own city or business sector.

© 2025 Ivan Tatarchuk (Telegram - @Ivan_Tatarchuk; LinkedIn - https://www.linkedin.com/in/ivan-tatarchuk/)
