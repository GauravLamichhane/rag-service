import json
import os
import sys
from statistics import mean
from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from rag_pipeline_v2 import ask


GOLDEN_DATASET = "golden_dataset.json"
MIN_FAITHFULNESS = 0.7 #CI fails if score drops below this

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

if not NVIDIA_API_KEY:
  raise RuntimeError("Set NVIDIA_API_KEY in your environment before running evaluate.py")

eval_llm = ChatOpenAI(
  model=NVIDIA_MODEL,
  temperature=0,
  api_key=NVIDIA_API_KEY,
  openai_api_base="https://integrate.api.nvidia.com/v1",
  max_tokens=4096,
  timeout=120,
)
eval_embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def metric_mean(value) -> float:
  """Normalize ragas metric output (list or scalar) into a float mean."""
  if isinstance(value, list):
    numeric = [v for v in value if isinstance(v, (int, float))]
    return mean(numeric) if numeric else 0.0
  if isinstance(value, (int, float)):
    return float(value)
  return 0.0

def run_evaluation():
  print("\n Loading golden datset")
  with open(GOLDEN_DATASET) as f:
    golden = json.load(f)
  print(f"{len(golden)} question-answer pairs loaded--------------")
  print("\n Running questions through RAG pipeline----------------")

  questions = []
  answers = []
  contexts = []
  ground_truths = []

  for i, item in enumerate(golden, 1):
    print(f" [{i} / {len(golden)}] {item['question'][:60]}...")
    result = ask(item['question'])
    questions.append(item['question'])
    answers.append(result['answer'])
    ground_truths.append(item["ground_truth"])

    # RAGAS needs the actual retrieved text, not source metadata.
    contexts.append(result.get("contexts", []))

  print("\n--Scoring with RAGAS --------------------")
  dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths,
  })

  results = evaluate(
     dataset,
      metrics=[faithfulness, answer_relevancy],
      llm=eval_llm,
      embeddings=eval_embeddings,
      batch_size=1,
      raise_exceptions=False,
  )

  faithfulness_score = metric_mean(results["faithfulness"])
  answer_relevancy_score = metric_mean(results["answer_relevancy"])

  print("\n── Results ─────────────────────────────────────────")
  print(f"  Faithfulness   : {faithfulness_score:.3f}  (min required: {MIN_FAITHFULNESS})")
  print(f"  Answer relevancy: {answer_relevancy_score:.3f}")

  # ── CI gate ──────────────────────────────────────────────
  if faithfulness_score < MIN_FAITHFULNESS:
    print(f"\n  FAILED — faithfulness {faithfulness_score:.3f} < {MIN_FAITHFULNESS}")
    print("  This would block a pull request in CI.")
    sys.exit(1)   # non-zero exit = CI failure
  else:
    print(f"\n  PASSED — quality gate cleared.")
    sys.exit(0)   # zero exit = CI success


if __name__ == "__main__":
  run_evaluation()