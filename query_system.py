import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from milvus import DB_PATH, COLLECTION_NAME, model, dim
from pymilvus import MilvusClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ForumQuerySystem:
    def __init__(self, db_path: str = DB_PATH, openai_api_key: str = None):
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
        
        # Setup Milvus connection
        self.client = MilvusClient(db_path)
        
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
        query_embedding = model.encode(query).astype(np.float32).tolist()
        
        res = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=limit,
            output_fields=["source", "author", "date", "text", "comment_id"]
        )
        
        posts = []
        for hit in res[0]:
            entity = hit["entity"]
            posts.append({
                'id': hit['id'],
                'source': entity.get('source'),
                'author': entity.get('author'),
                'date': self._clean_date(entity.get('date')),
                'text': entity.get('text'),
                'comment_id': entity.get('comment_id'),
                'similarity_score': 1 - hit['distance']  # Assuming COSINE metric where distance = 1 - similarity
            })
        
        return posts
    
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
            context += f"{post['text']}{'...' if len(post['text']) > 500 else ''}\n\n"
        
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
        res = self.client.query(
            collection_name=COLLECTION_NAME,
            filter=f"id == {post_id}",
            output_fields=["source", "author", "date", "text", "comment_id"]
        )
        
        if res:
            entity = res[0]
            return {
                'id': entity['id'],
                'source': entity.get('source'),
                'author': entity.get('author'),
                'date': self._clean_date(entity.get('date')),
                'text': entity.get('text'),
                'comment_id': entity.get('comment_id')
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
        self.client.close()

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
                        print(f"   {post['text']}")
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
