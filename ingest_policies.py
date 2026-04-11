"""
Handles the raw HTML extraction and demonstrates the critical difference between naive and semantic chunking
"""

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def scrape_hdb_policy(url: str) -> str:
    """Scrapes paragraph and list text from an HDB policy page."""
    print(f"[*] Fetching data from: {url}")
    # HDB and government sites often block requests without a standard User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Fail fast if the page blocks us
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = []
    # Policy rules usually live in paragraphs and bulleted lists
    for tag in soup.find_all(['p', 'li']):
        text = tag.get_text(strip=True)
        if text:
            content.append(text)
            
    return "\n\n".join(content)

def test_fixed_size_chunking(text: str) -> list:
    """Splits text rigidly by character count (Anti-pattern for RAG)."""
    splitter = CharacterTextSplitter(
        separator="", 
        chunk_size=150, 
        chunk_overlap=20
    )
    return splitter.split_text(text)

def test_semantic_chunking(text: str) -> list:
    """Splits text logically, preserving sentences and paragraphs (Best Practice)."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=150,
        chunk_overlap=20
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    # Target: HDB Single Citizen Scheme Eligibility
    target_url = "https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/singles"
    
    try:
        raw_text = scrape_hdb_policy(target_url)
        print(f"[*] Successfully extracted {len(raw_text)} characters.\n")
        
        print("="*50)
        print("METHOD 1: FIXED-SIZE CHUNKING (NAIVE)")
        print("="*50)
        naive_chunks = test_fixed_size_chunking(raw_text)
        for i, chunk in enumerate(naive_chunks[5:8]):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)
            
        print("\n" + "="*50)
        print("METHOD 2: RECURSIVE CHUNKING (SEMANTIC)")
        print("="*50)
        semantic_chunks = test_semantic_chunking(raw_text)
        for i, chunk in enumerate(semantic_chunks[5:8]):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)
            
    except Exception as e:
        print(f"[!] Error during ingestion: {e}")