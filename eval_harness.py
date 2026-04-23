import ollama

def run_llm_judge(query: str, context: str, generated_answer: str) -> str:
    """Uses the LLM as an automated judge to score RAG outputs."""
    
    judge_prompt = f"""
    You are an impartial, strict AI auditor evaluating a RAG system.
    
    You will be provided with:
    1. USER QUESTION
    2. RETRIEVED CONTEXT
    3. GENERATED ANSWER
    
    YOUR TASK: Evaluate the "Faithfulness" of the Generated Answer.
    
    - PASS CRITERIA (SCORE: 1): The Generated Answer is fully supported by the Retrieved Context. Being concise, summarizing, or omitting extra details from the context is ALLOWED and should PASS.
    - FAIL CRITERIA (SCORE: 0): The Generated Answer introduces NEW facts, numbers, or external knowledge NOT found in the Retrieved Context.
    
    USER QUESTION: {query}
    RETRIEVED CONTEXT: {context}
    GENERATED ANSWER: {generated_answer}
    
    OUTPUT FORMAT EXACTLY AS FOLLOWS:
    SCORE: [1 or 0]
    REASON: [Brief 1-sentence explanation]
    """
    
    response = ollama.chat(model='qwen2.5:3b', messages=[
        {'role': 'system', 'content': judge_prompt}
    ])
    
    return response['message']['content']

if __name__ == "__main__":
    print("[*] Starting Automated RAG Evaluation...\n")
    
    # ---------------------------------------------------------
    # TEST CASE 1: A perfect RAG interaction
    # ---------------------------------------------------------
    t1_query = "What is the income ceiling for a single buying a 2-room Flexi flat?"
    t1_context = "Under the Single Singapore Citizen Scheme, to buy a new 2-room Flexi flat, your monthly income ceiling is $7,000."
    t1_answer = "The income ceiling is $7,000 per month."
    
    print("="*60)
    print("TEST CASE 1: The Good Answer")
    print("="*60)
    print(run_llm_judge(t1_query, t1_context, t1_answer))
    
    # ---------------------------------------------------------
    # TEST CASE 2: The LLM Hallucinates external knowledge
    # ---------------------------------------------------------
    t2_query = "What is the income ceiling for a single buying a 2-room Flexi flat?"
    t2_context = "Under the Single Singapore Citizen Scheme, to buy a new 2-room Flexi flat, your monthly income ceiling is $7,000."
    # We simulate the LLM bringing in outside knowledge about BTO wait times
    t2_answer = "The income ceiling is $7,000 per month, but keep in mind BTO flats take 4 years to build."
    
    print("\n" + "="*60)
    print("TEST CASE 2: The Hallucinated Answer")
    print("="*60)
    print(run_llm_judge(t2_query, t2_context, t2_answer))