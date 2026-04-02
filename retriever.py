from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
TOP_K = 4

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_retriever():
  vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
  )

  return vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":TOP_K},
  )

retriever = get_retriever()

def retrieve(query: str):
  return retriever.invoke(query)


if __name__ == "__main__":
  query = input("Enter a test query:")
  chunks = retrieve(query)
  print(f"\nTop {TOP_K} chunks retrieved:\n")
  for i, chunk in enumerate(chunks, 1):
    source = chunk.metadata.get("source","unknown")
    page = chunk.metadata.get("page","")
    print(f"[{i}] Source: {source} Page: {page}")
    print(f" {chunk.page_content[:200]}...")
    print()