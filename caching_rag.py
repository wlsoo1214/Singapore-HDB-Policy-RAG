import sqlite3
import time
import ollama
from advanced_retrieval import get_hybrid_pool, rerank_results
from hybrid_search import setup_hybrid_system, HDB_POLICY_TEXT
from observable_rag import setup_database, log_interaction

# --- Initialize In-Memory Cache ---
# In production, Redis will be used, but for local, we use a simple dictionary
query_cache = {}

def hardened_generate(query: str, context_chunk: str, score: float) -> str:
    """ Generates an answer with a strict relevance threshold"""

    # avoid chatting
    if score < 0.50:
        return "I do not have enough information in my policy documents to answer that."

    system_prompt = f"""
    You are a strict HDB policy assistant.
    CRITICAL RULES:
    1. Answer the question USING ONLY the information in the context.
    2. Keep your answer concise.
    3. If the answer is missing, output EXACTLY: "I do not have enough information."
    
    CONTEXT:
    {context_chunk}
    """
    
    response = ollama.chat(model='qwen2.5:3b', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ])
    
    return response['message']['content']


if __name__ == "__main__":
    db_conn = setup_database()
    collection, bm25, chunks = setup_hybrid_system(HDB_POLICY_TEXT)
    
    # We will ask the exact same question twice to test the cache
    test_queries = [
        "How far away can my parents live for the PHG grant?",
        "Hi what can u do?",
        "How far away can my parents live for the PHG grant?" # Identical repeat
    ]
    
    for i, query in enumerate(test_queries):
        print("\n" + "="*70)
        print(f"USER QUERY {i+1}: '{query}'")
        
        start_time = time.time()
        
        # --- THE CACHE CHECK ---
        # If we already answered this exact question, skip the heavy AI lifting
        if query in query_cache:
            answer = query_cache[query]
            latency = time.time() - start_time
            print(f"\n[⚡ CACHE HIT] Latency: {latency:.4f}s")
            print(f"FINAL ANSWER: {answer}")
            log_interaction(db_conn, query, 1.0, latency, answer)
            continue
            
        # --- IF CACHE MISS: Run the full pipeline ---
        print("[!] Cache miss. Running full RAG pipeline...")
        
        pooled_docs = get_hybrid_pool(query, collection, bm25, chunks, top_k=3)
        final_docs, final_scores = rerank_results(query, pooled_docs)
        
        top_context = final_docs[0]
        top_score = final_scores[0]
        
        # Pass the score into the generator so it can short-circuit if needed
        answer = hardened_generate(query, top_context, top_score)
        
        # Save the heavy AI work to the cache for next time
        query_cache[query] = answer
        
        total_latency = time.time() - start_time
        print(f"\n[🐌 FULL RAG RUN] Latency: {total_latency:.4f}s")
        print(f"FINAL ANSWER: {answer}")
        
        log_interaction(db_conn, query, top_score, total_latency, answer)

    db_conn.close()