import csv
import feedparser
import time
from bs4 import BeautifulSoup
import re
from transformers import pipeline

VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

OUT_FILE = "data/processed/sentiment.csv"

def get_sentence_sentiment(classifier, sentence):
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
        print(f"Error classifying: {e}")
        return None
    return None

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=' ', strip=True)

def split_sentences(text):
    # Basic sentence splitter for Vietnamese
    text = re.sub(r'([.!?])\s+', r'\1|||', text)
    sentences = [s.strip() for s in text.split('|||') if s.strip()]
    return sentences

def main():
    print("Loading multilingual sentiment model (this might take a minute)...")
    classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    
    existing_texts = set()
    try:
        with open(OUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if row:
                    existing_texts.add(row[0])
    except FileNotFoundError:
        with open(OUT_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            
    print(f"Found {len(existing_texts)} existing items in {OUT_FILE}")
    
    out_file = open(OUT_FILE, "a", encoding="utf-8", newline='')
    writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL)
    
    total_added = 0
    
    print("Starting CafeF RSS crawl for VN30 tickers...")
    
    # CafeF provides RSS feeds for specific categories
    # We will use the stock market category (Chung khoan) and business (Doanh nghiep)
    rss_urls = [
        "https://cafef.vn/trang-chu.rss",
        "https://cafef.vn/thi-truong-chung-khoan.rss",
        "https://cafef.vn/doanh-nghiep.rss",
        "https://cafef.vn/tai-chinh-ngan-hang.rss",
        "https://cafef.vn/vi-mo-dau-tu.rss",
        "https://cafef.vn/bat-dong-san.rss"
    ]
    
    for rss_url in rss_urls:
        print(f"Fetching RSS: {rss_url}")
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries:
            title = entry.title
            summary_html = entry.get('summary', '')
            summary_text = clean_html(summary_html)
            
            full_text = title + ". " + summary_text
            sentences = split_sentences(full_text)
            
            for sentence in sentences:
                # We only want sentences that are somewhat related to our VN30 stocks
                # or financial terms to keep the domain relevant
                is_relevant = False
                for ticker in VN30_TICKERS:
                    if ticker in sentence:
                        is_relevant = True
                        break
                        
                # Additional financial keywords to expand dataset
                financial_keywords = ["cổ phiếu", "lợi nhuận", "doanh thu", "lãi", "lỗ", "cổ tức", "VN-Index", "thị trường", "tăng", "giảm", "điều chỉnh"]
                if not is_relevant:
                    for kw in financial_keywords:
                        if kw.lower() in sentence.lower():
                            is_relevant = True
                            break
                            
                if is_relevant and len(sentence.split()) >= 5 and sentence not in existing_texts:
                    label = get_sentence_sentiment(classifier, sentence)
                    if label is not None:
                        writer.writerow([sentence, label])
                        existing_texts.add(sentence)
                        total_added += 1
                        
                        if total_added % 50 == 0:
                            print(f"Added {total_added} new sentiment items")
                            out_file.flush()
                            
        time.sleep(1) # Be gentle to CafeF
        
    out_file.close()
    print(f"Crawl completed. Added {total_added} new items to sentiment.csv")
    print(f"Note: This RSS approach yields recent news. To hit 10,000 items, you should set this up as a cronjob or use the historical Google News scrape strategy with the sentiment model.")

if __name__ == "__main__":
    main()
