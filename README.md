c# iRacing Forum Query System

A simple system to query your scraped iRacing forum posts using embeddings and OpenAI's GPT models.

## Features

- **Semantic Search**: Find relevant forum posts using sentence embeddings
- **AI-Powered Q&A**: Ask questions and get answers based on forum content
- **Interactive CLI**: Easy-to-use command-line interface
- **Vector Search**: Uses SQLite VSS extension for fast similarity search

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key** (choose one method):

   **Option A: Create a .env file** (recommended):
   ```bash
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
   ```

   **Option B: Set environment variable**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

## Usage

### Interactive CLI

Run the interactive query system:

```bash
python query_system.py
```

Available commands:
- `ask <question>` - Ask a question about the forum posts
- `search <query>` - Search for similar posts
- `post <id>` - Get a specific post by ID
- `help` - Show available commands
- `quit` - Exit the program

### Example Usage

```python
from query_system import ForumQuerySystem

# Initialize the system
query_system = ForumQuerySystem(openai_api_key="your-key")

# Search for similar posts
posts = query_system.search_similar_posts("telemetry data", limit=5)

# Ask a question
answer = query_system.ask_question("How do I get started with the iRacing SDK?")

# Get a specific post
post = query_system.get_post_by_id(123)

# Don't forget to close the connection
query_system.close()
```

### Run Examples

```bash
python example_usage.py
```

## How It Works

1. **Embeddings**: Each forum post is converted to a vector embedding using the `all-MiniLM-L6-v2` model
2. **Similarity Search**: When you search or ask a question, the system finds the most similar posts using cosine similarity
3. **Context Building**: Relevant posts are used as context for the OpenAI model
4. **AI Response**: GPT-3.5-turbo generates answers based on the forum content

## Database Structure

The system expects a SQLite database with these tables:

- `forum_posts_meta`: Contains post metadata (author, date, text, etc.)
- `forum_posts_embeddings`: Contains vector embeddings for each post

## API Key Options

You can provide your OpenAI API key in four ways:

1. **.env file** (recommended):
   ```bash
   echo "OPENAI_API_KEY=your-key" > .env
   ```

2. **Environment variable**:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. **Command line argument**:
   ```bash
   python query_system.py --api-key your-key
   ```

4. **Direct parameter** (in code):
   ```python
   query_system = ForumQuerySystem(openai_api_key="your-key")
   ```

## Troubleshooting

- **VSS Extension**: If SQLite VSS extension is not available, the system will fall back to manual similarity calculation
  - Make sure you have `sqlite-vss>=0.1.2` installed: `pip install sqlite-vss`
  - The system uses `vss_distance_l2` function for vector similarity search
  - If you get "no such function: vss_distance" error, ensure you're using the correct conda environment
- **API Limits**: Be mindful of OpenAI API usage and costs
- **Database**: Make sure your database file exists and contains the expected tables

## Files

- `query_system.py` - Main query system and CLI
- `example_usage.py` - Example usage script
- `scraper.py` - Original scraper (creates the database)
- `requirements.txt` - Python dependencies
