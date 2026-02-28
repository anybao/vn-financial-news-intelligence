import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from src.ingestion.rss_scraper import RSSScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with actual database/storage logic later
def save_articles(articles):
    logger.info(f"Saving {len(articles)} articles to storage...")
    # TODO: Connect to DB or save to file

def run_ingestion_job():
    """Main job function to scrape and save articles."""
    logger.info("Starting ingestion job...")
    
    feeds = [
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://cafef.vn/trang-chu.rss"
    ]
    
    scraper = RSSScraper(feeds)
    articles = scraper.scrape()
    
    if articles:
        save_articles(articles)
    else:
        logger.warning("No articles scraped.")
        
    logger.info("Ingestion job completed.")

def start_scheduler(interval_minutes: int = 60):
    """Starts a background scheduler to run the ingestion job periodically."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_ingestion_job, 'interval', minutes=interval_minutes)
    scheduler.start()
    logger.info(f"Scheduler started. Job will run every {interval_minutes} minutes.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shutdown.")

if __name__ == "__main__":
    start_scheduler(interval_minutes=30)
