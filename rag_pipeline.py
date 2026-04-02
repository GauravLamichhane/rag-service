from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from retriever import retrieve
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025")

llm = ChatCohere(
  model = COHERE_MODEL,
  temperature=0,
  cohere_api_key=os.getenv("COHERE_API_KEY")
)

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
For every claim you make, cite the source by writing [Source: <filename>, Page: <page>] 
at the end of the sentence.

If the context does not contain enough information to answer the question,
say exactly: "I don't have enough information in the provided documents to answer this."
Do NOT make up information.

────────────────────────────────────────────────────────
CONTEXT:
{context}
────────────────────────────────────────────────────────

QUESTION: {question}

ANSWER:
"""

prompt = PromptTemplate(
  input_variables = ["context","question"],
  template = PROMPT_TEMPLATE
)

# main pipeline

def ask(question:str) -> dict:
  #1. retrieve relevant chunks
  chunks = retrieve(question)

  if not chunks:
    return {
      "answer": "No relevant documents found in the vector store.",
      "sources": [],
    }
  
  # 2. format chunks into a context block with source labels
  context_parts = []
  sources = []
  for i, chunk in enumerate(chunks, 1):
    source = chunk.metadata.get("source","unknown")
    page = chunk.metadata.get("page","N/A")
    context_parts.append(
      f"[{i}] Source: {source} | Page: {page}\n {chunk.page_content}"
    )
    sources.append({"source": source, "page": page})
  
  context = "\n\n".join(context_parts)

  #3. build and send the prompt
  formatted_prompt = prompt.format(context = context, question = question)
  response = llm.invoke(formatted_prompt)

  return {
    "answer": response.content,
    "sources": sources
  }


if __name__ == "__main__":
  print("RAG System ready. Type 'quit' to exit. \n")
  while True:
    question = input("Enter your question: ").strip()
    if question.lower() in ("quit","exit","q"):
      break
    if not question:
      continue
    result = ask(question)

    print("\n --Answer--------------")
    print(result["answer"])
    print("\n--Sources retrieved--------")
    for s in result["sources"]:
      print(f" .{s['source']} (page {s['page']})")
      print()