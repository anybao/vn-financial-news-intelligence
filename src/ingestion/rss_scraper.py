import feedparser
import logging
from bs4 import BeautifulSoup
from typing import List, Dict

logger = logging.getLogger(__name__)

class RSSScraper:
    """Scrapes financial news from RSS feeds."""
    
    def __init__(self, feed_urls: List[str]):
        self.feed_urls = feed_urls

    def clean_html(self, raw_html: str) -> str:
        """Removes HTML tags from a string."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

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
        "https://cafef.vn/trang-chu.rss"
    ]
    scraper = RSSScraper(sample_feeds)
    data = scraper.scrape()
    print(f"Scraped {len(data)} articles.")
    if data:
        print(f"Sample: {data[0]['title']}")
