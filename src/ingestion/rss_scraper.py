import feedparser
import html
import re
import logging
from bs4 import BeautifulSoup
from typing import List, Dict

logger = logging.getLogger(__name__)


def decode_html_entities(text: str) -> str:
    """Decode HTML entities including broken ones (e.g. #225; without &)."""
    if not text:
        return ""
    # Fix broken numeric entities: #225; -> &#225;
    text = re.sub(r'(?<!&)#(\d+);', r'&#\1;', text)
    # Fix broken named entities: e.g. nbsp; without &
    text = re.sub(r'(?<!&)((?:amp|lt|gt|quot|apos|nbsp|mdash|ndash|laquo|raquo|hellip));', r'&\1;', text)
    # Decode all HTML entities
    text = html.unescape(text)
    return text


class RSSScraper:
    """Scrapes financial news from RSS feeds."""
    
    def __init__(self, feed_urls: List[str]):
        self.feed_urls = feed_urls

    def clean_html(self, raw_html: str) -> str:
        """Removes HTML tags and decodes HTML entities from a string."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = decode_html_entities(text)
        return text

    def scrape(self) -> List[Dict]:
        """Scrapes all configured feeds and returns a list of article dictionaries."""
        articles = []
        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    article = {
                        "title": self.clean_html(entry.get("title", "")),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": self.clean_html(entry.get("summary", "")),
                        "source_url": url
                    }
                    if article["title"]: # Only add if it has a title
                        articles.append(article)
            except Exception as e:
                logger.error(f"Error scraping feed {url}: {e}")
        
        return articles

if __name__ == "__main__":
    # Example usage
    sample_feeds = [
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://vneconomy.vn/chung-khoan.rss"
    ]
    scraper = RSSScraper(sample_feeds)
    data = scraper.scrape()
    print(f"Scraped {len(data)} articles.")
    if data:
        print(f"Sample: {data[0]['title']}")
