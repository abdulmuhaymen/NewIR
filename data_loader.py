import os
import requests
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from io import BytesIO
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class PolicyDataLoader:
    def __init__(self, pdf_path: str):  # 👈 Keep name as pdf_path
        self.pdf_path = pdf_path
        self.excel_url = "https://docs.google.com/spreadsheets/d/198nb2GUCXZufC7hNy1DpsFeXrxeKfUcAurh9krZUtoM/export?format=xlsx"
        self.vectorstore = None
        self.retriever = None
        self.all_chunks = []
        self.chunk_embeddings = None
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _load_pdf_documents(self) -> List[Document]:
        """Load and chunk TXT file instead of PDF (name unchanged)."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Text file not found at {self.pdf_path}")
        
        with open(self.pdf_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"source": "txt"}) for chunk in chunks]

        print(f"📄 TXT Chunks Created: {len(documents)}")
        return documents

    def _load_excel_documents(self) -> List[Document]:
        """Download and convert Google Sheet to document chunks."""
        print("🔗 Downloading Excel (Google Sheet)...")
        response = requests.get(self.excel_url)
        if response.status_code != 200:
            raise Exception("Failed to download Excel file from Google Sheets")

        excel_file = BytesIO(response.content)
        df_sheets = pd.read_excel(excel_file, sheet_name=None)

        documents = []
        for sheet_name, df in df_sheets.items():
            if df.empty:
                continue
            text = df.to_string(index=False)
            doc = Document(page_content=text, metadata={"source": f"sheet:{sheet_name}"})
            documents.append(doc)

        print(f"📊 Excel Sheets Loaded: {len(documents)}")
        return documents

    def rerank_chunks(self, query, retrieved_chunks, top_k=3):
        """Custom reranking function using cosine similarity"""
        if not retrieved_chunks:
            return []
        
        chunk_texts = [chunk.page_content for chunk in retrieved_chunks]
        chunk_embeddings = self.embedder.encode(chunk_texts)
        query_vec = self.embedder.encode([query])
        scores = cosine_similarity(query_vec, chunk_embeddings)[0]
        ranked = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked[:top_k]]

    def load_all_documents(self):
        """Combine TXT and Excel documents, embed, and store."""
        try:
            txt_docs = self._load_pdf_documents()  # 👈 Still calling this name
            excel_docs = self._load_excel_documents()

            all_docs = txt_docs + excel_docs
            self.all_chunks = all_docs
            print(f"📚 Total Documents for Embedding: {len(all_docs)}")

            chunk_texts = [doc.page_content for doc in all_docs]
            self.chunk_embeddings = self.embedder.encode(chunk_texts)
            print(f"🧮 Precomputed embeddings shape: {self.chunk_embeddings.shape}")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = FAISS.from_documents(all_docs, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

            print("✅ Vectorstore and retriever successfully created.")
        except Exception as e:
            raise Exception(f"❌ Failed to load documents: {str(e)}")

    def get_retriever(self):
        return self.retriever

    def get_reranked_chunks(self, query, top_k=3):
        initial_chunks = self.retriever.invoke(query)
        return self.rerank_chunks(query, initial_chunks, top_k)