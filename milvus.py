import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

DB_PATH = "iracing_forum.db"
COLLECTION_NAME = "forum_posts"

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

def setup_database():
    client = MilvusClient(DB_PATH)
    
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=dim,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE"
        )
    
    return client

def save_post_with_embedding(client, post_data, source="forums.iracing.com", raw_html=""):
    comment_text = post_data.get('comment_text', '')
    if not comment_text:
        return
    
    embedding = model.encode(comment_text).astype(np.float32).tolist()
    post_date = post_data.get('post_date', '')
    
    data = {
        "vector": embedding,
        "source": source,
        "author": post_data.get('author_name', 'Unknown'),
        "date": post_date,
        "text": comment_text,
        "comment_id": post_data.get('comment_id', '')
    }
    
    client.insert(collection_name=COLLECTION_NAME, data=[data])
