import os
import yaml
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from hybrid_retriever import hybrid_retrieve

load_dotenv()

with open("prompts.yaml","r") as f:
  config = yaml.safe_load(f)

prompt = PromptTemplate(
  input_variables=["context","question"],
  template=config["rag_prompt"]
)

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025")

llm = ChatCohere(
  model = COHERE_MODEL,
  temperature=0,
  cohere_api_key=os.getenv("COHERE_API_KEY")
)

RELEVANCE_THRESHOLD = 0.0 #cross - encoder scores below this = not relevant (score range from -10 to +10)

def check_relevance(chunks) -> bool:
  """
  If all retrieved chunks are bibilography / reference pages
  refuse to answer than hallucinate
  """
  for chunk in chunks:
    text = chunk.page_content.lower()
    urls_count = text.count("http")
    word_count = len(text.split())
    url_density = urls_count / max(word_count, 1)
    if url_density < 0.05: #less than 5% URLs = real content chunk
      return True

  return False #all chunks look like bibliography


def ask(question: str) -> dict:
  #1. hybrid retreival (BM25 + vector + re-rank)
  chunks = hybrid_retrieve(question)

  if not chunks:
    return {
      "answer": "No relevant document found.",
      "sources": [],
    }
  #2. citation enforcement - check if chunks are actually useful
  if not check_relevance(chunks):
    return {
      "answer": "I don't have enough information in the provided documents to answer this.",
      "sources": [],
    }
  
  #3. format context with source labels
  context_parts = []
  sources = []
  for i, chunk in enumerate(chunks,1):
    source = chunk.metadata.get("source","unknown")
    page = chunk.metadata.get("page","N/A")
    context_parts.append(
      f"[{i}] Source: {source} | Page: {page} \n {chunk.page_content}"
    )
    sources.append({"source": source, "page": page})
  
  context = "\n\n".join(context_parts)

  #4. generate ans using versioned prompt
  formatted = prompt.format(context = context, question = question)
  response = llm.invoke(formatted)

  return {
    "answer": response.content,
    "sources": sources,
    "contexts": [chunk.page_content for chunk in chunks],
  }



if __name__ == "__main__":
  print("RAG v2 ready (hybrid search + re-ranking). Type 'quit' to exit.\n")
  while True:
    question = input("Your question:").strip()
    if question.lower() in ("quit","exit","q"):
      break
    if not question:
      continue

    result = ask(question)

    print("\n── Answer ──────────────────────────────────────────────")
    print(result["answer"])
    print("\n── Sources retrieved ────────────────────────────────────")
    for s in result["sources"]:
      source = s.get("source", s.get("Source", "unknown"))
      page = s.get("page", "N/A")
      print(f"  • {source}  (page {page})")
    print()