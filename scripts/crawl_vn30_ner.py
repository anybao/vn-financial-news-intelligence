import json
import re
import time
import datetime
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser

VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

OUT_FILE = "data/processed/ner.json"

def tokenize(text):
    # Split text considering Vietnamese words and punctuation
    text = re.sub(r'([.,!?()\[\]{}":;])', r' \1 ', text)
    tokens = text.split()
    return tokens

def label_ner(tokens, ticker):
    labels = []
    # Sometimes tickers appear as lowercase or with other forms, but typically in uppercase
    ticker_upper = ticker.upper()
    for token in tokens:
        if token.upper() == ticker_upper:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def clean_title(title):
    # Google news titles often end with " - Publisher Name"
    # We strip that part if it matches the pattern
    parts = title.rsplit(' - ', 1)
    if len(parts) > 1:
        return parts[0].strip()
    return title.strip()

def fetch_feed(ticker, start_date, end_date):
    query = f"{ticker} after:{start_date} before:{end_date}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=vi&gl=VN&ceid=VN:vi"
    
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries:
        title = clean_title(entry.title)
        
        # Only keep titles that actually mention the ticker to ensure quality
        if ticker.upper() in title.upper():
            tokens = tokenize(title)
            ner_tags = label_ner(tokens, ticker)
            
            # Additional validation: Ensure at least one token is labeled as 1
            if 1 in ner_tags:
                results.append({
                    "tokens": tokens,
                    "ner_tags": ner_tags
                })
    return results

def generate_date_ranges(start_year, end_year):
    ranges = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = datetime.date(year, month, 1)
            # Find the last day of the month by going to the 1st of next month and subtracting 1 day
            if month == 12:
                end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            
            ranges.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    return ranges

def main():
    date_ranges = generate_date_ranges(2020, 2024) # 5 years
    # plus 2025 up to march
    for month in range(1, 4):
        start_date = datetime.date(2025, month, 1)
        if month == 3:
            end_date = datetime.date(2025, month, 31)
        else:
            end_date = datetime.date(2025, month + 1, 1) - datetime.timedelta(days=1)
        date_ranges.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

    total_items = 0
    all_results = []
    
    tasks = []
    for ticker in VN30_TICKERS:
        for start_date, end_date in date_ranges:
            tasks.append((ticker, start_date, end_date))
            
    print(f"Total tasks (queries) to execute: {len(tasks)}")
    
    # Use ThreadPoolExecutor for concurrent requests
    # Limit max_workers to avoid getting blocked by Google
    max_workers = 10
    
    # Read existing data if possible to append properly
    # However we will just append directly to the file to save memory
    out_file = open(OUT_FILE, "a", encoding="utf-8")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(fetch_feed, t, s, e): (t, s, e) for t, s, e in tasks}
        
        for i, future in enumerate(as_completed(future_to_task)):
            t, s, e = future_to_task[future]
            try:
                data = future.result()
                if data:
                    for item in data:
                        out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_items += len(data)
                
                if (i + 1) % 50 == 0:
                    print(f"Completed {i + 1}/{len(tasks)} queries. Total items collected so far: {total_items}")
                    out_file.flush()
            except Exception as exc:
                print(f"Query {t} {s} to {e} generated an exception: {exc}")
                
            # Sleep tiny bit to be gentle
            time.sleep(0.05)
            
    out_file.close()
    print(f"Scraping completed. Total customized NER items saved: {total_items}")

if __name__ == "__main__":
    main()
