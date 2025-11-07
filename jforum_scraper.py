import asyncio
from crawl4ai import AsyncWebCrawler, JsonCssExtractionStrategy, CrawlerRunConfig, BrowserConfig
from milvus import setup_database, save_post_with_embedding

import os

def hybrid_extract_jforum_posts(html_content):
    """
    Hybrid extraction approach: Use CSS extraction for posts structure,
    then regex to extract timestamps and match them to posts.
    """
    import re
    
    # Schema for CSS extraction (without problematic post_date selector)
    jforum_schema = {
        "name": "iRacing JForum Posts",
        "baseSelector": "tr.trPosts",  # Each forum post row
        "fields": [
            {
                "name": "author_name",
                "selector": ".tdPostAuthor strong",  # Author name from the strong tag
                "type": "text",
                "default": "Unknown"
            },
            {
                "name": "comment_text",
                "selector": ".postBody",  # The actual comment content
                "type": "text",
                "default": ""
            },
            {
                "name": "comment_id",
                "selector": ".tdMessage",  # Get the comment ID from the message cell
                "type": "attribute",
                "attribute": "id",
                "default": ""
            }
        ]
    }
    
    # 1. Extract posts using CSS
    css_strategy = JsonCssExtractionStrategy(jforum_schema, verbose=False)
    posts = css_strategy.extract('jforum', html_content)
    
    # 2. Extract timestamps using regex
    timestamp_pattern = r'getDateAndTime\((\d+)\)'
    timestamps = re.findall(timestamp_pattern, html_content)
    
    print(f"  → Found {len(posts)} posts and {len(timestamps)} timestamps")
    
    # 3. Match timestamps to posts (assuming they appear in the same order)
    for i, post in enumerate(posts):
        if i < len(timestamps):
            try:
                from datetime import datetime
                timestamp_ms = int(timestamps[i])
                timestamp_s = timestamp_ms / 1000  # Convert to seconds
                dt = datetime.fromtimestamp(timestamp_s)
                post['post_date'] = dt.isoformat() + "+00:00"  # Standard ISO with UTC
            except Exception as e:
                print(f"Warning: Failed to convert timestamp {timestamps[i]}: {e}")
                post['post_date'] = timestamps[i]  # Fallback to raw timestamp
        else:
            post['post_date'] = ""
    
    return posts

async def scrape_jforum_page(crawler, offset, thread_id, extraction_strategy=None):
    """Scrape a single JForum page and return the extracted data using hybrid extraction"""
    if offset > 0:
        url = f"https://members.iracing.com/jforum/posts/list/{offset}/{thread_id}.page"
    else:
        url = f"https://members.iracing.com/jforum/posts/list/{thread_id}.page"
    
    crawl_config = CrawlerRunConfig(
        wait_for="css:.navLink",  # Wait for navigation links to appear
        # Don't use extraction_strategy here - we'll do hybrid extraction manually
    )
    
    print(f"Scraping JForum offset {offset}: {url}")
    result = await crawler.arun(url=url, config=crawl_config)
    
    if result.success:
        # Use hybrid extraction approach
        posts_data = hybrid_extract_jforum_posts(result.html)
        print(f"  → Extracted {len(posts_data)} posts from offset {offset}")
        return posts_data
    else:
        print(f"  → Failed to load page from offset {offset}")
        if hasattr(result, 'error_message'):
            print(f"  → Error: {result.error_message}")
        return []

async def scrape_jforum():
    """Scrape the JForum (members.iracing.com) and append to existing database"""
    # Now using hybrid extraction approach (CSS + regex) - no need for extraction strategy

    browser_config = BrowserConfig(
        headless=False,
        verbose=False,
        use_managed_browser=True,
        browser_type="chromium",
        user_data_dir=os.path.join(os.path.dirname(__file__), "chromium-profile")
    )

    # Setup database (will append to existing data)
    client = setup_database()
    
    total_posts = 0
    total_saved = 0
    thread_id = 1470675
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Loop through all 151 pages (0 to 150, each with 25 posts)
        for i in range(151):
            offset = i * 25
            try:
                # Scrape the page using hybrid extraction
                posts_data = await scrape_jforum_page(crawler, offset, thread_id)
                total_posts += len(posts_data)
                
                # Save posts with embeddings
                page_saved = 0
                for post in posts_data:
                    if post.get('comment_text'):  # Only save posts with content
                        save_post_with_embedding(client, post, source="members.iracing.com")
                        page_saved += 1
                
                total_saved += page_saved
                print(f"  → Saved {page_saved} posts from offset {offset}")
                
                # Add a small delay to be respectful to the server
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"  → Error processing offset {offset}: {e}")
                continue
    
    # Final summary
    print("\n=== JFORUM SCRAPING COMPLETE ===")
    print(f"Total posts extracted: {total_posts}")
    print(f"Total posts saved: {total_saved}")
    print(f"Pages processed: {i + 1}")
    

if __name__ == "__main__":
    asyncio.run(scrape_jforum())
