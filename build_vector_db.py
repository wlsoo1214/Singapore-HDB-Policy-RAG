"""
RAG Pipeline
Load -> Transform -> Embed -> Store
Load -> Chunk -> Embed -> Store
"""


import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from ingest_policies import scrape_hdb_policy, test_semantic_chunking

def setup_vector_db():
    """
    Sets up the vector database and returns the collection.
    """
    print("[*] Initializing Vector DB")
    # save to hard drive
    client = chromadb.PersistentClient(path="./hdb_vector_db")
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    # create collection. cosine distance is default for text similarity
    # Vector DB is like a database, collection is like a table
    collection = client.get_or_create_collection(
        name="hdb_grants_singles",
        embedding_function = hf_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection 

def populate_db(collection, chunks):
    """
    Populates the vector DB with chunks of text.
    """
    # check if collection is empty
    # to avoid re-ingesting the same data
    print(f"[*] Preparing to sync {len(chunks)} chunks...")

    # Generate IDs based on a hash of content.
    # prevents same text from being added twice
    ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in chunks]

    metadatas = [{"source": "hdb_singles_page", "timestamp": "2026-04-11"} for _ in range(len(chunks))]

    # .upsert is safer than .add for appending
    collection.upsert(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"[+] Sync complete! Collection now has {collection.count()} total chunks.")
        
def run_test_query(collection, query_text):
    print(f"\n{'='*50}")
    print(f"Query: {query_text}")
    print(f"{'='*50}")

    # query the vector DB. Get top 3 most semantically similar chunks. 
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )

    # Chroma returns distances. Lower distance = higher similarity
    for i in range(len(results["documents"][0])):
        # results is a nested dictionary
        # doc is the actual text
        doc = results['documents'][0][i]
        dist = results['distances'][0][i]
        print(f"Match {i+1} (Distance: {dist:.4f})")
        print(doc)

def inspect_chunks(chunks):
    """
    Prints chunks in a structured, readable format for debugging.
    """
    print("\n" + "="*60)
    print(f"CHUNKING INSPECTION REPORT ({len(chunks)} Chunks Found)")
    print("="*60)

    for i, chunk in enumerate(chunks):
        # Generate the hash just for the preview
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        
        print(f"\n[CHUNK #{i}] | ID: {chunk_id[:8]}... | Length: {len(chunk)} chars")
        print("-" * 30)
        # Indent the text so it looks like a clean block
        indented_text = "    " + chunk.replace("\n", "\n    ")
        print(indented_text)
        print("-" * 30)

    print("\n" + "="*60)

if __name__ == "__main__":
    target_url = "https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/singles"
    # scrape the data
    raw_text = scrape_hdb_policy(target_url)
    # chunk the data
    semantic_chunks = test_semantic_chunking(raw_text)
    # inspect chunk
    inspect_chunks(semantic_chunks)
    # setup vector db
    collection = setup_vector_db()
    # populate the vector db
    populate_db(collection, semantic_chunks)
    

    # Verification
    print("\n" + "="*50)
    print("FINAL VERIFICATION")
    print("="*50)
    print(f"Total Chunks in DB: {collection.count()}")

    # Test query
    run_test_query(collection, "What is the income ceiling for singles buying a resale flat?")