import os
import psycopg2
from pgvector.psycopg2 import register_vector

def get_db_connection():
    """Establishes a connection to the Dockerized PostgreSQL database."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres-db"), 
        database=os.getenv("POSTGRES_DB"),        
        user=os.getenv("POSTGRES_USER"),          
        password=os.getenv("POSTGRES_PASSWORD")   
    )
    return conn

def init_db():
    """Initializes the database, enables pgvector, and creates necessary tables."""
    print("Initializing Database...")
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Enable the pgvector extension for AI embeddings
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Register the vector type with psycopg2
    register_vector(conn)

    # 2. Create the Structured Data Table (Trading Strategies)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_logs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT NOT NULL,
            generated_concept TEXT NOT NULL,
            backtest_results JSONB,
            is_approved BOOLEAN
        );
    """)

    # 3. Create the Unstructured Data Table (Financial News Vectors for RAG)
    # We use a vector size of 384, which is standard for lightweight embedding models like all-MiniLM-L6-v2
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_news_vectors (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding vector(384) 
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database tables and pgvector extension initialized successfully.")

if __name__ == "__main__":
    init_db()