"""
For Observability
SQLite database to record every interaction, latency, confidence score, and final answer.
"""

import sqlite3
import time
import ollama
from advanced_retrieval import get_hybrid_pool, rerank_results
from hybrid_search import setup_hybrid_system, HDB_POLICY_TEXT

def setup_database():
    """Creates SQLite DB and table"""
    conn = sqlite3.connect("rag_observability.db")
    cursor = conn.cursor()
    # Create table if not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            top_context_score REAL,
            latency_seconds REAL,
            llm_answer TEXT
        ) 
    ''')
    conn.commit()
    return conn

def observable_generate(query: str, context_chunk: str) -> str:
    """
    Uses local model to generate answer strictly based on context
    """
    system_prompt = f"""
    You are a highly accurate, strict HDB policy assistant.
    CRITICAL RULES:
    1. Answer the question USING ONLY the information in the provided context.
    2. Keep your answer concise and direct.
    3. If the context does NOT contain the answer, output EXACTLY: "I do not have enough information."
    
    CONTEXT:    
    {context_chunk}
    """

    response = ollama.chat(model="qwen2.5:3b", messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query},
    ])
    return response['message']['content'] # retrieve the answer

def log_interaction(conn, query, score, latency, answer):
    """Inserts the exact details of the interaction into the database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO query_logs (user_query, top_context_score, latency_seconds, llm_answer)
        VALUES (?, ?, ?, ?)
    ''', (query, score, latency, answer))
    conn.commit()
    

if __name__ == "__main__": 
    # S1: Initialize DB and setup hybrid system
    db_conn = setup_database()
    collection, bm25, chunks = setup_hybrid_system(HDB_POLICY_TEXT)

    test_queries = [
        "What is the income ceiling for singles buying a resale flat?",
        "How far away can my parents live for the PHG grant?",
        "Hi what can u do?"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"USER QUERY: {query}")
        print(f"{'='*70}")

        # Start timer
        start_time = time.time()

        # S2: Advanced retrieval (Hybrid + Rerank)
        pooled_docs = get_hybrid_pool(query, collection, bm25, chunks, top_k=10)
        final_docs, final_scores = rerank_results(query, pooled_docs)

        # Get the top 1 result
        top_context = final_docs[0]
        top_score = final_scores[0]

        # Generate answer
        final_answer = observable_generate(query, top_context)

        # End timer
        end_time = time.time()
        latency = end_time - start_time

        print(f"[Retrieved Context Score: {top_score:.2f}]")
        print(f"Context Injected into LLM:\n{top_context}")
        print(f"\n{'='*70}")
        print(f"FINAL ANSWER: {final_answer}")
        print(f"{'='*70}")
        print(f"Latency: {latency:.4f} seconds")

        # S4: Log to SQLite
        cursor = db_conn.cursor()
        cursor.execute('''
            INSERT INTO query_logs (user_query, top_context_score, latency_seconds, llm_answer)
            VALUES (?, ?, ?, ?)
        ''', (query, top_score, latency, final_answer)) 
        db_conn.commit()

    db_conn.close()
    
   