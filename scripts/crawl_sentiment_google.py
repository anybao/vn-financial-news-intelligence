import csv
import urllib.parse
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
from transformers import pipeline

VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

OUT_FILE = "data/processed/sentiment.csv"

# Make classifier global so threads can share it
print("Loading multilingual sentiment model (this might take a minute)...")
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', device=-1)

def get_sentence_sentiment(sentence):
    """
    nlptown/bert-base-multilingual-uncased-sentiment returns 1 star to 5 stars.
    We map it to our labels:
    1 or 2 stars -> Negative (0)
    3 stars      -> Neutral (1)
    4 or 5 stars -> Positive (2)
    """
    try:
        # Avoid sequence length issues
        if len(sentence) > 500:
            sentence = sentence[:500]
            
        result = classifier(sentence)[0]
        label_str = result['label']
        stars = int(label_str.split()[0])
        score = result['score']
        
        # Only accept if the model is somewhat confident
        if score < 0.4:
            return None
            
        if stars in [1, 2]:
            return 0
        elif stars == 3:
            return 1
        elif stars in [4, 5]:
            return 2
    except Exception as e:
        return None
    return None


def clean_title(title):
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
        
        # Keep relevant financial titles
        if ticker.upper() in title.upper():
            label = get_sentence_sentiment(title)
            if label is not None:
                results.append([title, label])
                
    return results

def generate_date_ranges(start_year, end_year):
    ranges = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = datetime.date(year, month, 1)
            if month == 12:
                end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            
            ranges.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    return ranges

def main():
    date_ranges = generate_date_ranges(2020, 2024)
    for month in range(1, 4):
        start_date = datetime.date(2025, month, 1)
        if month == 3:
            end_date = datetime.date(2025, month, 31)
        else:
            end_date = datetime.date(2025, month + 1, 1) - datetime.timedelta(days=1)
        date_ranges.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

    existing_texts = set()
    try:
        with open(OUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if row:
                    existing_texts.add(row[0])
    except:
        pass
        
    print(f"Loaded {len(existing_texts)} existing items.")

    tasks = []
    for ticker in VN30_TICKERS:
        for start_date, end_date in date_ranges:
            tasks.append((ticker, start_date, end_date))
            
    print(f"Total time batches to execute: {len(tasks)}")
    
    # We write directly to CSV
    out_file = open(OUT_FILE, "a", encoding="utf-8", newline='')
    writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL)
    
    # Needs to be smaller threads since model inference is being run
    max_workers = 4 
    total_added = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(fetch_feed, t, s, e): (t, s, e) for t, s, e in tasks}
        
        for i, future in enumerate(as_completed(future_to_task)):
            t, s, e = future_to_task[future]
            try:
                data = future.result()
                if data:
                    for text, label in data:
                        if text not in existing_texts:
                            writer.writerow([text, label])
                            existing_texts.add(text)
                            total_added += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(tasks)} batches. New Sentiment Rows added so far: {total_added}")
                    out_file.flush()
            except Exception as exc:
                print(f"Failed {t} between {s} and {e}: {exc}")
                
            time.sleep(0.01)
            
    out_file.close()
    print(f"Scraping completed. Added {total_added} new sentiment items.")

if __name__ == "__main__":
    main()
