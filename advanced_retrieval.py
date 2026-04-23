import numpy as np
from sentence_transformers import CrossEncoder

# Import from hybrid_search.py
from hybrid_search import setup_hybrid_system,HDB_POLICY_TEXT

def get_hybrid_pool(query, collection, bm25, chunks, top_k=10):
    """
    Returns a pool of candidate chunks from both Vector and BM25 search.
    """
    # 1. Vector Search (Vector DB)
    vector_results = collection.query(query_texts=[query], n_results=top_k)
    vector_docs = vector_results['documents'][0]

    # 2. Keyword Search (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top K chunks from BM25
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_docs = [chunks[i] for i in bm25_top_indices]

    # Combine and deduplicate
    combined_pool = list(set(vector_docs + bm25_docs))

    return combined_pool

def rerank_results(query, document_pool):
    """
    Uses a Cross-Encoder to judge relevance.
    """
    print(f"Reranking {len(document_pool)} combined documents...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    
    # Create pairs of (query, document)
    # Same query pair with the different docs
    pairs = [(query, doc) for doc in document_pool]
    
    # Get scores
    scores = reranker.predict(pairs)
    
    # Sort by score (descending order)
    sorted_indices = np.argsort(scores)[::-1] # reverser -> descending 
    
    # Return sorted results
    reranked_docs = [document_pool[i] for i in sorted_indices]
    reranked_scores = [scores[i] for i in sorted_indices]
    
    # return a tuple with 2 lists list[str], list[float]
    return reranked_docs, reranked_scores

if __name__ == "__main__":
    collection, bm25, chunks = setup_hybrid_system(HDB_POLICY_TEXT)    
    test_queries = [
        "What is the income ceiling for singles buying a resale flat?",
        "How far away can my parents live for the PHG grant?"
    ]
    
    print("\n--- RERANKED RESULTS ---")
    # Show the top 3 finalists
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")

        # Get hybrid pool
    pool_docs = get_hybrid_pool(query, collection, bm25, chunks, top_k=10)
    
    # Rerank the pool
    reranked_docs, reranked_scores = rerank_results(query, pool_docs)  
    print("\n🏆 THE RERANKED TOP RESULT (What the LLM will actually see):")
    # only print the absolute best result (Rank 1)
    print(f"\n[Score: {reranked_scores[0]:.2f}]")
    print(reranked_docs[0])