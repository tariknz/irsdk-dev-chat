import sys
import asyncio

from main_forum_scraper import main as main_forum
from jforum_scraper import scrape_jforum as jforum

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "jforum":
        asyncio.run(jforum())
    else:
        asyncio.run(main_forum())