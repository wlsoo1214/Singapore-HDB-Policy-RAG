"""
Hybrid Search = Vector DB (Dense/vector search) + BM25 (Sparse/keyword search)
"""

import os
import transformers

transformers.logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

# Hardcoded text
HDB_POLICY_TEXT = """
To qualify for the Enhanced CPF Housing Grant (EHG) (Singles), your average gross monthly household income for the 12 months prior to your flat application must not exceed $7,000.
You must be a first-timer Singapore Citizen, 35 years old or above, buying a 2-room to 5-room resale flat.
The remaining lease of the flat must be at least 20 years and can cover you to the age of 95.

For the Proximity Housing Grant (PHG), singles buying a resale flat to live with or within 4km of their parents are eligible for a $15,000 grant. 
If you are buying a resale flat to live near your parents (within 4km), but not with them, the PHG is $10,000.

Under the Single Singapore Citizen Scheme, you can buy a 2-room Flexi flat in non-mature estates from HDB, or any resale flat on the open market.
To buy a new 2-room Flexi flat, your monthly income ceiling is $7,000. For resale flats, there is no income ceiling to buy the flat itself, but there are income ceilings for the grants.
"""

# --- SETUP DATABASE & BM25 ---
def setup_hybrid_system(text):
    print("[*] Chunking data...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=150,
        chunk_overlap=20
    )
    # split_text is the function of the splitter class
    chunks = splitter.split_text(text)

    # Vector Search (Vector DB)
    print("[*] Setting up ChromaDB...(Vector/Dense Search)")
    client = chromadb.PersistentClient(path="./hdb_vector_db")
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name="hdb_hybrid",
        embedding_function=hf_ef
    )
    
    """
    --- Content based hashing ---
    """

    # 1. Create a unique ID for each chunk based on its content
    print("[*] Generating Hash IDs and Upserting...")
    ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in chunks]
    # 2. Use .upsert() instead of .add() to avoid errors when running the script multiple times
    collection.upsert(documents=chunks, ids=ids)

    print(f"[*] Total unique chunks in a database: {collection.count()}")

    # Keyword Search (BM25)

    print("[*] Initializing BM25...(Keyword/Sparse Search)")
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]  
    bm25 = BM25Okapi(tokenized_chunks)

    return collection, bm25, chunks

def hybrid_search(query, collection, bm25, chunks, top_k=3):
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")

    # 1. Vector Search (Vector DB)
    vector_results = collection.query(query_texts=[query], n_results=top_k)
    vector_docs = vector_results['documents'][0]

    # 2. Keyword Search (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top K chunks from BM25
    """
    Sort in ascending -> [::-1] to reverse -> [:top_k] to get top K
    """
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    # Get the actual chunks based on the top indices
    bm25_docs = [chunks[i] for i in bm25_top_indices]

    print("\n--- VECTOR SEARCH TOP RESULT ---")
    print(vector_docs[0])
    
    print("\n--- BM25 KEYWORD SEARCH TOP RESULT ---")
    print(bm25_docs[0])

if __name__ == "__main__":
    collection, bm25, chunks = setup_hybrid_system(HDB_POLICY_TEXT)
    
    # Let's test the exact query that failed you earlier
    hybrid_search(
    chunks=chunks, 
    bm25=bm25, 
    collection=collection, 
    query="What is the income ceiling?"
    )

    hybrid_search(
        chunks=chunks, 
        bm25=bm25, 
        collection=collection, 
        query="How far away can my parents live for the PHG grant?"
    )
    
    

    

    
    
        
    
    
    