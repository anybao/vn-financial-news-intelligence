import time
import logging
import requests
from apscheduler.schedulers.background import BackgroundScheduler

from src.ingestion.rss_scraper import RSSScraper
from src.database import SessionLocal, engine
from src.models import Base, Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure tables exist
Base.metadata.create_all(bind=engine)

def save_articles(articles):
    logger.info(f"Processing and saving {len(articles)} articles...")
    db = SessionLocal()
    try:
        new_count = 0
        for article_data in articles:
            # Check if exists
            exists = db.query(Article).filter(Article.link == article_data["link"]).first()
            if exists:
                continue
                
            # Perform NLP processing via local API
            nlp_data = {
                "summary": "",
                "sentiment": "Neutral",
                "stocks": [],
                "is_duplicate": False
            }
            try:
                # Use raw summary for NLP; title is stored separately
                text_to_process = article_data['summary'] or article_data['title']
                res = requests.post("http://localhost:8000/api/v1/predict_event", json={"text": text_to_process}, timeout=10)
                if res.status_code == 200:
                    nlp_res = res.json()
                    nlp_data["summary"] = nlp_res.get("summary", "")
                    nlp_data["sentiment"] = nlp_res.get("sentiment", "Neutral")
                    nlp_data["stocks"] = nlp_res.get("stocks", [])
                    nlp_data["is_duplicate"] = nlp_res.get("is_duplicate", False)
            except Exception as e:
                logger.error(f"NLP API call failed: {e}")

            # Create DB Record
            db_article = Article(
                title=article_data["title"],
                link=article_data["link"],
                published=article_data["published"],
                source=article_data["source_url"],
                raw_summary=article_data["summary"],
                nlp_summary=nlp_data["summary"],
                sentiment=nlp_data["sentiment"],
                stocks=",".join(nlp_data["stocks"]),
                is_duplicate=nlp_data["is_duplicate"]
            )
            db.add(db_article)
            db.commit() # Commit individually to stream live updates
            new_count += 1
            
        logger.info(f"Successfully added {new_count} new articles to database.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save articles: {e}")
    finally:
        db.close()

def run_ingestion_job():
    """Main job function to scrape, process and save articles."""
    logger.info("Starting ingestion job...")
    
    feeds = [
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://vneconomy.vn/chung-khoan.rss"
    ]
    
    try:
        scraper = RSSScraper(feeds)
        articles = scraper.scrape()
        
        if articles:
            save_articles(articles)
        else:
            logger.warning("No articles scraped.")
    except Exception as e:
        logger.error(f"Ingestion job failed: {e}")
        
    logger.info("Ingestion job completed.")

def start_scheduler(interval_minutes: int = 3):
    """Starts a background scheduler to run the ingestion job periodically."""
    # Run once immediately on startup
    run_ingestion_job()
    
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
    start_scheduler(interval_minutes=3)
