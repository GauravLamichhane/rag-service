import os
os.environ.setdefault("USER_AGENT", "rag-service-ingest/1.0")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

#config
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
  chunk_size = CHUNK_SIZE,
  chunk_overlap = CHUNK_OVERLAP,
  length_function = len,

)

#Loaders i.e PDF, MarkDown or text, URLS

def load_pdfs(folder = "docs"):
  docs = []
  for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
      path = os.path.join(folder, filename)
      print(f"Loading PDF: {filename}")
      loader = PyPDFLoader(path)
      docs.extend(loader.load())
  return docs

def load_markdown(folder = "docs"):
  docs = []
  for filename in os.listdir(folder):
    if filename.endswith(".txt") or filename.endswith(".md"):
      path = os.path.join(folder, filename)
      print(f"Loading Markdown/text: {filename}")
      loader = TextLoader(path, encoding="utf-8")
      docs.extend(loader.load())
  return docs

def load_url(urls: list[str]):
  docs = []
  for url in urls:
    print(f"Loading  URL: {url}")
    try:
      loader = WebBaseLoader(url)
      docs.extend(loader.load())
    except Exception as exc:
      print(f"Skipping URL due to load error: {url} ({exc})")
  return docs

def ingest(use_urls = False, urls: list[str] | None = None):
  print("\n--Loading documents ---")
  all_docs = []
  all_docs.extend(load_pdfs())
  all_docs.extend(load_markdown())
  urls = urls or []
  if use_urls and urls:
    all_docs.extend(load_url(urls))
  if not all_docs:
    print("No documents found. Add files to the docs/ folder or pass URLS.")
    return
  print(f"\n Splitting into chunks (size = {CHUNK_SIZE}, overlap = {CHUNK_OVERLAP})")
  chunks = splitter.split_documents(all_docs)
  print(f"Total chunks created: {len(chunks)}")

  print("\n Embedding and storing in ChromaDB")
  vectorstore = Chroma.from_documents(
      documents=chunks,
      embedding=embeddings,
      persist_directory= CHROMA_DIR,
    )

  print(f"{len(chunks)} chunks stored in {CHROMA_DIR}")
  return vectorstore


if __name__ == "__main__":
#   urls = [
#         "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
# ]
#   ingest(urls=urls)
  ingest(use_urls=False)