import asyncio
import json
from crawl4ai import AsyncWebCrawler, JsonCssExtractionStrategy, CrawlerRunConfig, BrowserConfig
from milvus import setup_database, save_post_with_embedding

import os

async def scrape_forum_page(crawler, page_num, extraction_strategy):
    """Scrape a single forum page and return the extracted data"""
    url = f"https://forums.iracing.com/discussion/62/iracing-sdk/p{page_num}"
    
    crawl_config = CrawlerRunConfig(
        wait_for="css:li.ItemComment",  # Wait for forum posts to load
        extraction_strategy=extraction_strategy
    )
    
    print(f"Scraping page {page_num}: {url}")
    result = await crawler.arun(url=url, config=crawl_config)
    
    if result.success and result.extracted_content:
        posts_data = json.loads(result.extracted_content)
        print(f"  → Extracted {len(posts_data)} posts from page {page_num}")
        return posts_data
    else:
        print(f"  → Failed to extract data from page {page_num}")
        if hasattr(result, 'error_message'):
            print(f"  → Error: {result.error_message}")
        return []

async def main():
    # Define the extraction schema for forum posts
    schema = {
        "name": "iRacing Forum Posts",
        "baseSelector": "li.ItemComment",  # Each forum post container
        "baseFields": [
            {
                "name": "comment_id",
                "type": "attribute",
                "attribute": "id",
                "default": ""
            }
        ],
        "fields": [
            {
                "name": "author_name",
                "selector": ".Username",  # Author name from the username link
                "type": "text",
                "default": "Unknown"
            },
            {
                "name": "comment_text",
                "selector": ".Message.userContent",  # The actual comment content
                "type": "text",
                "default": ""
            },
            {
                "name": "post_date",
                "selector": "time[datetime]",  # Post date from the time element
                "type": "attribute",
                "attribute": "datetime",
                "default": ""
            }
        ]
    }

    # Create the extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=False)  # Less verbose for multiple pages

    browser_config = BrowserConfig(
        headless=False,             # 'True' for automated runs
        verbose=False,              # Less verbose for multiple pages
        use_managed_browser=True,  # Enables persistent browser strategy
        browser_type="chromium",
        user_data_dir=os.path.join(os.path.dirname(__file__), "chromium-profile")
    )

    # Setup database once
    client = setup_database()
    
    total_posts = 0
    total_saved = 0
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Loop through all 64 pages
        for page_num in range(1, 65):  # Pages 1 to 64
            try:
                # Scrape the page
                posts_data = await scrape_forum_page(crawler, page_num, extraction_strategy)
                total_posts += len(posts_data)
                
                # Save posts with embeddings
                page_saved = 0
                for post in posts_data:
                    if post.get('comment_text'):  # Only save posts with content
                        save_post_with_embedding(client, post)
                        page_saved += 1
                
                total_saved += page_saved
                print(f"  → Saved {page_saved} posts from page {page_num}")
                
                # Add a small delay to be respectful to the server
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"  → Error processing page {page_num}: {e}")
                continue
    
    # Final summary
    print("\n=== SCRAPING COMPLETE ===")
    print(f"Total posts extracted: {total_posts}")
    print(f"Total posts saved: {total_saved}")
    print(f"Pages processed: {page_num}")
    
    # No need to close Milvus client explicitly

if __name__ == "__main__":
    asyncio.run(main())
