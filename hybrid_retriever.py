from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import os

load_dotenv()

CHROMA_DIR = "chroma_db"
TOP_K_VECTOR = 20 #fetch more candidates initially
TOP_K_BM25 = 20 #bm25 candidates
TOP_K_FINAL = 4 # final chunks after re-ranking

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

#cross-encoder re-ranker - read query + chunk together for preciese scoring
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def load_all_chunks() -> list[Document]:
  """ Load every chunk stored in the ChromaDB"""
  vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
  )
  results = vectorstore.get(include=["documents","metadatas"])
  docs = []
  for text, meta in zip(results["documents"], results["metadatas"]):
    docs.append(Document(page_content=text, metadata = meta or {}))
  return docs

load_all_chunks()

def bm25_search(query: str, all_chunks: list[Document], k:int) -> list[Document]:
  """ Keyword search using BM25 - finds exact term matches. """
  tokenized = [doc.page_content.lower().split() for doc in all_chunks]
  bm25 = BM25Okapi(tokenized)
  scores = bm25.get_scores(query.lower().split())
  top_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse=True)[:k]
  return [all_chunks[i] for i in top_indices]

def vector_search(query: str, k: int) -> list[Document]:
  """semantic search using vector similarity - finds meaning matches."""
  vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function = embeddings,
  )
  return vectorstore.similarity_search(query, k = k)

def deduplicate(docs: list[Document]) -> list[Document]:
  """ remove duplicate chunks that appears both in BM25 and vector results."""

  seen = set()
  unique = []
  for doc in docs:
    key = doc.page_content[:100]
    if key not in seen:
      seen.add(key)
      unique.append(doc)
  return unique

def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
  pairs = [[query, doc.page_content] for doc in docs]
  scores = reranker.predict(pairs)
  ranked = sorted(zip(scores, docs), key = lambda x: x[0], reverse=True)
  return [doc for _, doc in ranked[:top_k]]


def hybrid_retrieve(query: str) -> list[Document]:
  """
  full hybrid pipeline:
  1. BM25 keyword search (finds exact terms)
  2. Vector semantic search (finds meaning)
  3. Merge + deduplicate
  4. Cross-encoder re-rank (precision cleanup)
  """
  print(f"\n [BM25] searching for exact keyword matches...")
  all_chunks = load_all_chunks()
  bm25_results = bm25_search(query, all_chunks, k=TOP_K_BM25)

  print(f"  [Vector] searching for semantic matches...")
  vector_results = vector_search(query, k=TOP_K_VECTOR)

  print(f"  [Merge] combining and deduplicating...")
  combined = deduplicate(bm25_results + vector_results)
  print(f"  Combined candidates: {len(combined)}")

  print(f"  [Re-rank] scoring with cross-encoder...")
  final = rerank(query, combined, top_k=TOP_K_FINAL)
  print(f"  Final chunks after re-ranking: {len(final)}\n")

  return final


if __name__ == "__main__":
  query = input("Test hybird retrieval - enter query: ")
  chunks = hybrid_retrieve(query)
  print(f"\Top {TOP_K_FINAL} chunks after hybrid search + re-ranking:\n")
  for i, chunk in enumerate(chunks, 1):
    source = chunk.metadata.get("source","unknown")
    page = chunk.metadata.get("page","?")
    print(f"[{i}] {source} - page {page}")
    print(f" {chunk.page_content[:200]}")
    print()