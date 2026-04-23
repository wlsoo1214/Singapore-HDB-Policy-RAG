import ollama
from advanced_retrieval import get_hybrid_pool, rerank_results 
from hybrid_search import setup_hybrid_system, HDB_POLICY_TEXT

def generate_rag_answer(query: str, context_chunk: str) -> str:
    """
    Uses local model to generate answer strictly based on context

    Args:
        query (str): The user's question.
        context_chunk (str): The context to use for generating the answer.
    
    Returns:
        str: The generated answer.
    """
    
    prompt = f"""
    You are a highly accurate, strict HDB policy assistant.
    You will be provided with a user question and an official policy context.
    
    CRITICAL RULES:
    1. Answer the question USING ONLY the information in the provided context.
    2. Do not use outside knowledge or prior training data.
    3. Keep your answer concise and direct.
    4. If the context does NOT contain the answer, you must output EXACTLY: "I do not have enough information in my policy documents to answer that."
    
    CONTEXT:
    {context_chunk}
    """
    
    # Call the local model
    response = ollama.chat(model="qwen2.5:3b", messages=[
        {'role': 'system', 'content': prompt}, # system prompt
        {'role': 'user', 'content': query}, # user prompt
    ])
    return response['message']['content']

if __name__ == "__main__":
    # S1: setup database
    collection, bm25, chunks = setup_hybrid_system(HDB_POLICY_TEXT)
    
    test_query = "Hi what can u do?"

    print("\n" + "="*70)
    print(f"Query: {test_query}")
    print("="*70)

    # S2: Advanced retrieval (Hybrid + Rerank)

    # Hybrid Search (Vector + Keyword)
    pooled_docs = get_hybrid_pool(test_query, collection, bm25, chunks, top_k=10)
    final_docs, final_scores = rerank_results(test_query, pooled_docs)

    # S3: Synthesis (Generate answer)

    # only take the top 1 result
    top_context = final_docs[0]
    print(f"\n[Retrieved Context Score: {final_scores[0]:.2f}]")
    print(f"Context Injected into LLM:\n{top_context}")

    # Generate answer
    final_answer = generate_rag_answer(test_query, top_context)

    print(f"\n{'='*70}")
    print(f"FINAL ANSWER: {final_answer}")
    print(f"{'='*70}")
    