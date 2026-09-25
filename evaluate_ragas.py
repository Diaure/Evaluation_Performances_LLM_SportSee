# eval_ragas_prototype.py

import json
import csv
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
    # groundedness,
    # semantic_similarity,
    # noise_sensitivity,
)

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from mistralai import MistralAIEmbeddings
# embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY, 
#                                 model="mistral-embed")

# from mistralai import ChatMistralAI
# from ragas.llms import LangchainLLMWrapper
# from langchain_openai import ChatOpenAI

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.vector_store import VectorStoreManager
# from openai import OpenAI
# client = OpenAI(
#     api_key=MISTRAL_API_KEY,
#     base_url="https://api.mistral.ai/v1")

# def llm(prompt):
#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0
#     )
#     return response.choices[0].message.content


# LLM Mistral (prototype)
client = MistralClient(api_key=MISTRAL_API_KEY)

from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import LLMResult, Generation

# LLM pour RAGAS
class MistralRagasLLM(BaseRagasLLM):

    def generate_text(
        self,
        prompt,
        n=1,
        temperature=1e-8,
        stop=None,
        callbacks=None,
    ):
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                ChatMessage(
                    role="user",
                    content=prompt.to_string()
                )
            ],
            temperature=temperature,
        )

        text = response.choices[0].message.content

        return LLMResult(
            generations=[
                [Generation(text=text)]
            ]
        )

    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=1e-8,
        stop=None,
        callbacks=None,
    ):
        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

class MistralRagasEmbeddings(BaseRagasEmbeddings):

    def embed_query(self, text):
        response = client.embeddings(
            model="mistral-embed",
            input=[text],
        )
        return response.data[0].embedding

    def embed_documents(self, texts):
        response = client.embeddings(
            model="mistral-embed",
            input=texts,
        )
        return [item.embedding for item in response.data]


# LLM pour RAGAS
# def ragas_llm(prompt):
#     response = client.chat(
#         model=MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0
#     )
#     return response.choices[0].message.content

# wrapped_llm = LangchainLLMWrapper(ragas_llm)

# Prompt RAG du prototype
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


# Charger le dataset d’évaluation
def load_evaluation_set(path="./evaluation/validation_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Charger FAISS + chunks du prototype
def load_vectorstore():
    manager = VectorStoreManager()
    if manager.index is None or not manager.document_chunks:
        raise RuntimeError("VectorStoreManager n'a pas pu charger l'index ou les chunks.")
    return manager


# Génération de réponse comme le prototype
def generate_answer(question, context_str):
    final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)
    messages = [ChatMessage(role="user", content=final_prompt)]
    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,)
    return response.choices[0].message.content


# Construction du dataset RAGAS
def build_ragas_dataset(eval_set, vector_store_manager):
    dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "metadata": [],
    }

    for item in eval_set:
        q = item["question"]

        # Retrieval prototype
        search_results = vector_store_manager.search(q, k=SEARCH_K)

        # Format du contexte prototype
        if search_results:
            context_str = "\n\n---\n\n".join([
                f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
                for res in search_results])
        else:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances."

        # Génération EXACTEMENT comme prototype
        answer = generate_answer(q, context_str)

        # RAGAS dataset
        dataset["question"].append(q)
        dataset["answer"].append(answer)
        dataset["contexts"].append([r["text"] for r in search_results])
        dataset["ground_truth"].append(item["ground_truth"])
        dataset["metadata"].append(item)

    return dataset


# Évaluation RAGAS
def run_ragas_evaluation(dataset):

    ragas_dataset = Dataset.from_dict({
    "question": dataset["question"],
    "answer": dataset["answer"],
    "contexts": dataset["contexts"],
    "ground_truth": dataset["ground_truth"],
})

    evaluator_llm = MistralRagasLLM(run_config=RunConfig())
    evaluator_embeddings = MistralRagasEmbeddings()

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
            # groundedness,
            # semantic_similarity,
            # noise_sensitivity,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings)
    
    return result


# Export JSON + CSV
def export_results(result, dataset, output_dir="ragas_results"):
    Path(output_dir).mkdir(exist_ok=True)

    # JSON complet
    with open(Path(output_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "ragas_scores": result,
            "metadata": dataset["metadata"]
        }, f, indent=4)

    # CSV simple
    with open(Path(output_dir) / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "score"])
        for metric, score in result.items():
            writer.writerow([metric, score])


# MAIN
if __name__ == "__main__":
    print("Chargement du VectorStore prototype…")
    vector_store_manager = load_vectorstore()

    print("Chargement du dataset d’évaluation…")
    eval_set = load_evaluation_set()

    print("Construction du dataset RAGAS…")
    dataset = build_ragas_dataset(eval_set, vector_store_manager)

    print("Évaluation RAGAS…")
    result = run_ragas_evaluation(dataset)

    print("Export des résultats…")
    export_results(result, dataset)

    print("Évaluation terminée.")
