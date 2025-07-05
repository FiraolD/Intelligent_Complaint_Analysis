"""
rag_pipeline.py

Implements the RAG pipeline using TinyLlama for fast local testing.
Includes fixes for:
- Input length limits
- Garbled output
- Tokenization overflow
"""

import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from transformers import pipeline


class RAGPipeline:
    def __init__(self,
                index_path="vector_store/faiss_index.bin",
                metadata_path="vector_store/chunk_metadata.pkl",
                data_path="Data/filtered_complaints.csv",
                embedding_model_name="all-MiniLM-L6-v2",
                llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        
        # Load FAISS index
        self.index = self._load_faiss_index(index_path)
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_path)
        
        # Load original dataset
        self.df = pd.read_csv(data_path)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load LLM
        self.llm = self._load_llm(llm_name)

        # Prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _load_faiss_index(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found at {path}")
        return faiss.read_index(path)
    
    def _load_metadata(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def _load_llm(self, model_name):
        try:
            print(f"[INFO] Loading LLM: {model_name}")
            return pipeline("text-generation", model=model_name, max_new_tokens=200, truncation=True)
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}. Falling back to TinyLlama...")
            return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=200, truncation=True)
    
    def _create_prompt_template(self):
        template = """
You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def retrieve_chunks(self, query, k=3):  # Reduced from 5 to 3
        """Retrieve top-k most similar complaint chunks to the given query."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), k)

        retrieved_chunks = [
            self.df.iloc[self.metadata[i]['original_index']]['cleaned_narrative'][:500]  # Limit to 500 chars
            for i in indices[0]
        ]
        retrieved_metadata = [self.metadata[i] for i in indices[0]]
        
        return retrieved_chunks, retrieved_metadata
    
    def generate_answer(self, question, k=3):
        """Generate an answer to the given question using RAG."""
        retrieved_chunks, retrieved_metadata = self.retrieve_chunks(question, k=k)
        
        # Format context (limit total length)
        context = "\n\n".join(retrieved_chunks[:k])[:1500]  # Max 1500 characters
        
        # Format prompt
        formatted_prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate response (with truncation)
        raw_response = self.llm(formatted_prompt)
        answer = raw_response[0]['generated_text'][len(formatted_prompt):].strip()
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "metadata": retrieved_metadata
        }