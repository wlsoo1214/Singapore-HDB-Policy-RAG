import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError

"""
Chroma DB 

Store four things:
- ID (unique indentifier)
- Document (text)
- Metadata (custom tags) ->  is a dict 
- Embedding (vector)

Meta filtering: 
    - attach tags to the documents
    - filter the documents based on the tags

SQL -> math and list
NoSQL -> flexible schema
ChromaDB -> metadata + vector -> semantic similar lookup

"""

def setup_metadata_db():
    print("[*] Initializing Metadata Database...")
    # 1. Initialize ChromaDB
    # Save to hard disk
    client = chromadb.PersistentClient(path="./hdb_metadata_db")

    # local embedding model
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Clean up old collection
    """
    if else -> checking and then acting -> Look before you leap
    try except -> try to do something, if error, do something else -> Easier to ask for forgiveness than permission
    
    - Efficiency & API calls
    if.. else ->  fetch list -> delete
    try.. except -> delete
    
    """
    try:
        client.delete_collection(name="hdb_metadata")
    except (ValueError, NotFoundError): 
        print("[!] Collection didn't exist, skipping deletion.")
        pass
    # Create new collection
    collection = client.create_collection(
        name="hdb_metadata",
        embedding_function=hf_ef,
        metadata={"hnsw:space": "cosine"} # use cosine similarity
    )

    return collection

if __name__ == "__main__":
    collection = setup_metadata_db()
    
    # 1. The Conflicting Data
    docs = [
        "To buy a 2-room Flexi flat, your monthly income ceiling is $7,000.",
        "To buy a 4-room flat, your monthly income ceiling is $14,000."
    ]
    
    # attach tags to the documents
    metadatas = [
        {"scheme": "single"},
        {"scheme": "family"}
    ]
    ids = ["doc_single", "doc_family"]
    
    print("[*] Inserting conflicting policies with metadata tags...")
    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    
    # ---------------------------------------------------------
    # TEST 1: The Blind Search (The Problem)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 1: 'What is my income ceiling?' (NO FILTER)")
    print("="*60)
    
    bad_results = collection.query(
        query_texts=["What is my income ceiling?"],
        n_results=2
    )
    for i, doc in enumerate(bad_results['documents'][0]):
        # Notice how it pulls BOTH policies because they both match the "vibe" of the query
        print(f"Result {i+1} [Metadata: {bad_results['metadatas'][0][i]}]: {doc}")

    # ---------------------------------------------------------
    # TEST 2: The Filtered Search (The Solution)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2: 'What is my income ceiling?' (WITH 'SINGLE' FILTER)")
    print("="*60)
    
    good_results = collection.query(
        query_texts=["What is my income ceiling?"],
        n_results=1,
        where={"scheme": "single"} # THE HARD GUARDRAIL
    )
    for i, doc in enumerate(good_results['documents'][0]):
        print(f"Result {i+1} [Metadata: {good_results['metadatas'][0][i]}]: {doc}")

    
    
