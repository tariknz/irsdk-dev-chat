import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite_vss
from openai import OpenAI
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ForumQuerySystem:
    def __init__(self, db_path: str = "iracing_forum.db", openai_api_key: str = None):
        """
        Initialize the forum query system with database and OpenAI client.
        
        Args:
            db_path: Path to the SQLite database
            openai_api_key: OpenAI API key (if None, will try to get from environment)
        """
        self.db_path = db_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            # Try to load from .env file first
            load_dotenv()
            
            # Try to get from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file, environment variable, or pass it directly.")
            self.openai_client = OpenAI(api_key=api_key)
        
        # Setup database connection
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        
        # Try to load VSS extension
        try:
            sqlite_vss.load(self.conn)
            self.vss_available = True
        except Exception as e:
            print(f"VSS extension not available: {e}")
            self.vss_available = False
        
        self.cur = self.conn.cursor()
    
    def search_similar_posts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for posts similar to the query using embeddings.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing post metadata and similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query).astype(np.float32)
        
        # Try VSS search first if available, fallback to manual similarity
        if self.vss_available:
            try:
                # Convert query embedding to bytes for VSS
                query_embedding_bytes = query_embedding.tobytes()
                
                # VSS search
                self.cur.execute("""
                SELECT 
                    m.id, m.source, m.author, m.date, m.text, m.comment_id,
                    vss_distance_l2(embedding, ?) as distance
                FROM forum_posts_meta m
                JOIN forum_posts_embeddings e ON m.id = e.rowid
                ORDER BY distance
                LIMIT ?
                """, (query_embedding_bytes, limit))
                
                results = self.cur.fetchall()
                
                # Convert to list of dictionaries
                posts = []
                for row in results:
                    posts.append({
                        'id': row[0],
                        'source': row[1],
                        'author': row[2],
                        'date': self._clean_date(row[3]),
                        'text': row[4],
                        'comment_id': row[5],
                        'similarity_score': 1 - row[6]  # Convert distance to similarity
                    })
                
                return posts
                
            except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
                print(f"VSS search failed, falling back to manual similarity: {e}")
                return self._manual_similarity_search(query, limit)
        else:
            # VSS not available, use manual similarity
            return self._manual_similarity_search(query, limit)
    
    def _manual_similarity_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fallback method for similarity search when VSS is not available.
        """
        print("Warn: Manual similarity search")
        # Generate query embedding
        query_embedding = self.model.encode(query).astype(np.float32)
        
        # Get all posts with their embeddings
        self.cur.execute("""
        SELECT m.id, m.source, m.author, m.date, m.text, m.comment_id, e.embedding
        FROM forum_posts_meta m
        JOIN forum_posts_embeddings e ON m.id = e.id
        """)
        
        all_posts = self.cur.fetchall()
        
        # Calculate similarities
        similarities = []
        for row in all_posts:
            post_id, source, author, date, text, comment_id, embedding_data = row
            
            # Parse embedding (could be JSON or bytes)
            try:
                if isinstance(embedding_data, bytes):
                    # Try to parse as bytes first
                    post_embedding = np.frombuffer(embedding_data, dtype=np.float32)
                else:
                    # Parse as JSON
                    post_embedding = np.array(json.loads(embedding_data), dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, post_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(post_embedding)
                )
                
                similarities.append({
                    'id': post_id,
                    'source': source,
                    'author': author,
                    'date': self._clean_date(date),
                    'text': text,
                    'comment_id': comment_id,
                    'similarity_score': float(similarity)
                })
                
            except Exception as e:
                print(f"Error processing embedding for post {post_id}: {e}")
                continue
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:limit]
    
    def ask_question(self, question: str, max_context_posts: int = 5, stream: bool = False):
        """
        Ask a question about the forum posts using OpenAI.
        
        Args:
            question: The question to ask
            max_context_posts: Maximum number of relevant posts to include as context
            
        Returns:
            The AI's answer based on the forum posts
        """
        # Search for relevant posts
        relevant_posts = self.search_similar_posts(question, limit=max_context_posts)
        
        if not relevant_posts:
            return "I couldn't find any relevant posts in the database to answer your question."
        
        # Build context from relevant posts
        context = "Here are some relevant forum posts that might help answer your question:\n\n"
        
        for i, post in enumerate(relevant_posts, 1):
            context += f"Post {i} (by {post['author']}, {post['date']}):\n"
            context += f"{post['text'][:500]}{'...' if len(post['text']) > 500 else ''}\n\n"
        
        # Create the prompt for OpenAI
        prompt = f"""You are an AI assistant helping users find information from iRacing forum posts. 
Based on the following forum posts, please answer the user's question. If the posts don't contain enough information to answer the question, say so.

Forum Posts Context:
{context}

User Question: {question}

Please provide a helpful answer based on the forum posts above. If you reference specific posts, mention the author and date."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on forum post content. You will provide references to the forum posts in your answer."},
                    {"role": "user", "content": prompt}
                ],
                stream=stream
            )
            
            if stream:
                return response
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error getting response from OpenAI: {e}"
    
    def get_post_by_id(self, post_id: int) -> Dict[str, Any]:
        """Get a specific post by its ID."""
        self.cur.execute("""
        SELECT id, source, author, date, text, comment_id
        FROM forum_posts_meta
        WHERE id = ?
        """, (post_id,))
        
        row = self.cur.fetchone()
        if row:
            return {
                'id': row[0],
                'source': row[1],
                'author': row[2],
                'date': self._clean_date(row[3]),
                'text': row[4],
                'comment_id': row[5]
            }
        return None
    
    def _clean_date(self, date_str: str) -> str:
        """Clean up date string, removing JavaScript code and formatting properly."""
        if not date_str:
            return "Unknown date"
        
        # If it contains JavaScript code, return a generic message
        if "function" in date_str or "var loc" in date_str:
            return "Date not available"
        
        # If it's a proper ISO date, format it nicely
        if "T" in date_str and "+" in date_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d %H:%M")
            except:
                return date_str
        
        return date_str
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

def main():
    """Simple CLI interface for querying the forum database."""
    import sys
    
    # Check if OpenAI API key is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--api-key":
        if len(sys.argv) < 3:
            print("Usage: python query_system.py --api-key YOUR_OPENAI_API_KEY")
            sys.exit(1)
        api_key = sys.argv[2]
        # Remove the api-key arguments for the rest of the script
        sys.argv = [sys.argv[0]] + sys.argv[3:]
    else:
        api_key = None
    
    try:
        query_system = ForumQuerySystem(openai_api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key:")
        print("1. Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("2. Or pass it directly: python query_system.py --api-key your-key-here")
        sys.exit(1)
    
    print("iRacing Forum Query System")
    print("=" * 40)
    print("Type 'quit' to exit, 'help' for commands")
    print()
    
    while True:
        try:
            user_input = input("Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  help - Show this help message")
                print("  search <query> - Search for similar posts")
                print("  ask <question> - Ask a question (uses AI)")
                print("  post <id> - Get a specific post by ID")
                print("  quit - Exit the program")
                print()
                continue
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if not query:
                    print("Please provide a search query.")
                    continue
                
                print(f"\nSearching for: '{query}'")
                posts = query_system.search_similar_posts(query, limit=5)
                
                if posts:
                    print(f"\nFound {len(posts)} relevant posts:")
                    for i, post in enumerate(posts, 1):
                        print(f"\n{i}. {post['author']} ({post['date']}) - Score: {post['similarity_score']:.3f}")
                        print(f"   {post['text'][:200]}{'...' if len(post['text']) > 200 else ''}")
                else:
                    print("No relevant posts found.")
                print()
                
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if not question:
                    print("Please provide a question.")
                    continue
                
                print(f"\nQuestion: {question}")
                print("Thinking...")
                response = query_system.ask_question(question, stream=True)
                print("\nAnswer: ", end='')
                answer = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end='', flush=True)
                    answer += delta
                print()
                
            elif user_input.lower().startswith('post '):
                try:
                    post_id = int(user_input[5:].strip())
                    post = query_system.get_post_by_id(post_id)
                    if post:
                        print(f"\nPost {post_id}:")
                        print(f"Author: {post['author']}")
                        print(f"Date: {post['date']}")
                        print(f"Source: {post['source']}")
                        print(f"Text: {post['text']}")
                    else:
                        print(f"Post {post_id} not found.")
                except ValueError:
                    print("Please provide a valid post ID number.")
                print()
                
            else:
                # Default: treat as a question
                print("Thinking...")
                response = query_system.ask_question(user_input, stream=True)
                print("\nAnswer: ", end='')
                answer = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end='', flush=True)
                    answer += delta
                print()
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    query_system.close()

if __name__ == "__main__":
    main()
