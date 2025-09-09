import asyncio
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite_vss
from crawl4ai import *

# Database setup
DB_PATH = "iracing_forum.db"

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

def setup_database():
    """Setup SQLite database with VSS extension"""
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    cur = conn.cursor()
    
    # Create tables if they don't exist (don't drop existing ones)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forum_posts_meta (
        id INTEGER PRIMARY KEY,
        source TEXT,
        author TEXT,
        date TEXT,
        text TEXT,
        comment_id TEXT
    )
    """)
    
    # Check if VSS table exists, if not create it
    cur.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name='forum_posts_embeddings'
    """)
    
    if not cur.fetchone():
        # VSS table doesn't exist, try to create it
        try:
            cur.execute(f"""
            CREATE VIRTUAL TABLE forum_posts_embeddings USING vss0(
                embedding vector({dim})
            )
            """)
            print("Created new VSS table for embeddings")
        except Exception as e:
            print(f"VSS table creation failed: {e}")
            # Fallback: create a regular table instead
            cur.execute(f"""
            CREATE TABLE forum_posts_embeddings (
                id INTEGER PRIMARY KEY,
                embedding BLOB
            )
            """)
            print("Created fallback regular table for embeddings")
    else:
        print("Using existing embeddings table")
    
    return conn, cur

def save_post_with_embedding(cur, post_data, source="forums.iracing.com", raw_html=""):
    """Save a forum post and its embedding to the database"""
    # Generate embedding for the comment text
    comment_text = post_data.get('comment_text', '')
    if comment_text:
        embedding = model.encode(comment_text).astype(np.float32).tolist()
        
        # Use post_date directly (now standardized in extraction)
        post_date = post_data.get('post_date', '')
        
        # Insert metadata
        cur.execute("""
        INSERT INTO forum_posts_meta (source, author, date, text, comment_id) 
        VALUES (?, ?, ?, ?, ?)
        """, (
            source,
            post_data.get('author_name', 'Unknown'),
            post_date,
            comment_text,
            post_data.get('comment_id', '')
        ))
        
        # Insert embedding - try VSS first, fallback to regular table
        try:
            # Convert embedding to bytes for VSS
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            cur.execute("""
            INSERT INTO forum_posts_embeddings (embedding) VALUES (?)
            """, (embedding_bytes,))
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            # Fallback for regular table structure - store as JSON
            cur.execute("""
            INSERT INTO forum_posts_embeddings (embedding) VALUES (?)
            """, (json.dumps(embedding),))
