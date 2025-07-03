"""
chunking&embedding.py

This script:
1. Loads the cleaned complaint dataset
2. Splits long narratives into smaller chunks
3. Generates embeddings using all-MiniLM-L6-v2
4. Builds and saves a FAISS index
5. Saves metadata for traceability
"""

import os
import pickle
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
CHUNK_SIZE = 500        # Number of characters per chunk
CHUNK_OVERLAP = 50      # Overlap between chunks
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Model from sentence-transformers
DATA_PATH = 'data/filtered_complaints.csv'
VECTOR_STORE_DIR = 'vector_store/'
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, 'faiss_index.bin')
METADATA_FILE = os.path.join(VECTOR_STORE_DIR, 'chunk_metadata.pkl')

# Ensure Output Directory Exists
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Step 1: Load Cleaned Data
print("Step 1: Loading cleaned complaints...")
df = pd.read_csv(DATA_PATH)
print(f"Total complaints loaded: {len(df)}")
print("Columns:", df.columns.tolist())
print("\n")

# Step 2: Text Chunking
print("Step 2: Splitting narratives into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

all_chunks = []
metadata = []

for idx, row in df.iterrows():
    narrative = row['cleaned_narrative']
    product = row['Product']
    company = row['Company']

    if isinstance(narrative, str):
        chunks = text_splitter.split_text(narrative)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                'original_index': idx,
                'product': product,
                'company': company,
                'chunk_length': len(chunk),
                'chunk_id': i
            })

print(f"Total chunks generated: {len(all_chunks)}")
print("\n")

# Step 3: Generate Embeddings
print("Step 3: Generating embeddings...")

model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(all_chunks, show_progress_bar=True)

print(f"Embedding dimension: {embeddings.shape[1]}")
print("\n")

# Step 4: Build FAISS Index
print("Step 4: Building FAISS index...")

embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings).astype('float32')

# Add to FAISS index
faiss_index.add(embeddings_np)

# Save FAISS index to disk
faiss.write_index(faiss_index, INDEX_FILE)
print(f"FAISS index saved to: {INDEX_FILE}")
print("\n")

# Step 5: Save Metadata
print("Step 5: Saving metadata...")

with open(METADATA_FILE, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Metadata saved to: {METADATA_FILE}")
print("\n")

# Final Summary
print("✅ Chunking and embedding completed successfully!")
print(f"- Total original complaints: {len(df)}")
print(f"- Total chunks created: {len(all_chunks)}")
print(f"- Average chunk length: {sum(m['chunk_length'] for m in metadata) // len(metadata)}")
print(f"- Embedding model used: {EMBEDDING_MODEL}")
print(f"- FAISS index dimension: {embedding_dim}")
print("- Files saved:")
print(f"  - FAISS index: {INDEX_FILE}")
print(f"  - Metadata: {METADATA_FILE}")