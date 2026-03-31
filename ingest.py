import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
load_dotenv()

#config
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
    loader = WebBaseLoader(url)
    docs.extend(loader.load())
  return docs

def ingest(urls: list[str] = []):
  print("\n--Loading documents ---")
  all_docs = []
  all_docs.extend(load_pdfs())
  all_docs.extend(load_markdown())
  if urls:
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
  urls = [
        "https://example.com/"
  ]
  ingest(urls=urls)