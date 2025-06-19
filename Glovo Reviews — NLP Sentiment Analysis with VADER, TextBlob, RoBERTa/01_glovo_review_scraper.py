"""
📦 Module: 01_glovo_review_scraper.py

🔹 Description:
This is the first module in the Glovo NLP project pipeline.
It is responsible for collecting user reviews of the Glovo app from the Google Play Store using the
`google-play-scraper` library.

The module implements randomized batching, autosaving to prevent data loss, and automatic language detection for each
review.
Reviews are saved incrementally in a `partials/` folder and later merged into a single dataset
(`glovo_reviews_full.csv`).

This data serves as the input for all subsequent text preprocessing, sentiment analysis, and modeling steps.

🔸 Output:
- Partial CSV files in /partials
- Final merged file: glovo_reviews_full.csv
"""

import pandas as pd
import time
import os
import random
from langdetect import detect
from google_play_scraper import reviews, Sort

# 🔹 Main Function: Fetch Reviews from Google Play with Randomized Batches
def fetch_reviews_with_random_batch_and_autosave(
    max_reviews: int = 10000,
    min_batch: int = 140,
    max_batch: int = 200,
    min_sleep: int = 7,
    max_sleep: int = 13,
    autosave_step: int = 1000
) -> None:
    """
    Fetches Google Play reviews for the Glovo app with randomized batch sizes and sleep intervals.
    Reviews are saved incrementally every autosave_step reviews.

    Args:
        max_reviews (int): Maximum number of reviews to fetch.
        min_batch (int): Minimum number of reviews to fetch per request.
        max_batch (int): Maximum number of reviews to fetch per request.
        min_sleep (int): Minimum delay between requests (in seconds).
        max_sleep (int): Maximum delay between requests (in seconds).
        autosave_step (int): Number of accumulated reviews before saving to disk.
    """
    all_reviews = []
    continuation_token = None
    total_fetched = 0
    autosave_buffer_count = 0
    part_number = 1

    os.makedirs("partials", exist_ok=True)
    print(f"🟡 Starting download of up to {max_reviews} reviews...")

    while total_fetched < max_reviews:
        batch_size = random.randint(min_batch, max_batch)
        print(f"\n📦 Fetching batch {total_fetched // batch_size + 1} with size {batch_size}...")

        batch, continuation_token = reviews(
            'com.glovo',
            lang='en',
            country='us',
            count=batch_size,
            sort=Sort.NEWEST,
            continuation_token=continuation_token
        )

        if not batch:
            print("❗ No more reviews returned. Stopping.")
            break

        for entry in batch:
            content = entry['content'].strip()
            if not content:
                continue  # Skip empty reviews
            try:
                language = detect(content)
            except:
                language = "unknown"
            all_reviews.append({
                'review': content,
                'rating': entry['score'],
                'date': entry['at'].isoformat(),  # Safer datetime format
                'lang': language
            })

        total_fetched += len(batch)
        autosave_buffer_count += len(batch)
        print(f"✅ Total fetched so far: {total_fetched} (Buffer: {autosave_buffer_count})")

        if autosave_buffer_count >= autosave_step:
            df_part = pd.DataFrame(all_reviews)
            filename = f"partials/glovo_reviews_part_{part_number}.csv"
            df_part.to_csv(filename, index=False)
            print(f"💾 Saved {len(df_part)} reviews to {filename}")
            all_reviews.clear()
            autosave_buffer_count = 0
            part_number += 1

        sleep_duration = random.uniform(min_sleep, max_sleep)
        print(f"⏳ Sleeping for {sleep_duration:.2f} seconds...")
        time.sleep(sleep_duration)

    if all_reviews:
        df_part = pd.DataFrame(all_reviews)
        filename = f"partials/glovo_reviews_part_{part_number}.csv"
        df_part.to_csv(filename, index=False)
        print(f"💾 Saved final partial ({len(df_part)} reviews) to {filename}")

    print("📎 Merging all partial files into one CSV...")
    csv_files = [f"partials/{f}" for f in os.listdir("partials") if f.endswith(".csv")]
    all_dfs = [pd.read_csv(file) for file in sorted(csv_files)]
    df_merged = pd.concat(all_dfs, ignore_index=True)
    df_merged.to_csv("glovo_reviews_full.csv", index=False)
    print("✅ Merged CSV saved as glovo_reviews_full.csv")


# -------------------- Script Entry Point --------------------

if __name__ == "__main__":
    fetch_reviews_with_random_batch_and_autosave(
        max_reviews=20000,    # You can increase this if needed
        max_batch=200,
        min_sleep=7,
        max_sleep=13,
        autosave_step=1000
    )