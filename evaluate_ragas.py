# eval_ragas_prototype.py

import json
import csv
from pathlib import Path
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

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

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,)

from utils.vector_store import VectorStoreManager


# Charger le dataset d’évaluation
def load_evaluation_set(path="./evaluation/validation_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# LLM Mistral (prototype)
client = MistralClient(api_key=MISTRAL_API_KEY)

# Prompt RAG du prototype
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


# Charger FAISS + chunks du prototype
def load_vectorstore():
    manager = VectorStoreManager()
    if manager.index is None or not manager.document_chunks:
        raise RuntimeError("VectorStoreManager n'a pas pu charger l'index ou les chunks.")
    return manager


# Génération de réponse EXACTEMENT comme le prototype
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
        "metadata": [],  # On ajoute les métadonnées du dataset  
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
        # Ajouter les métadonnées
        dataset["metadata"].append(item)
    return dataset


# Évaluation RAGAS
def run_ragas_evaluation(dataset):

    os.environ["OPENAI_API_KEY"] = "dummy"          # empêche l’erreur
    os.environ["RAGAS_USE_OPENAI"] = "false"        # désactive OpenAI
    os.environ["RAGAS_LLM_BACKEND"] = "custom"      # force l’usage de ton LLM

    ragas_dataset = Dataset.from_dict({
    "question": dataset["question"],
    "answer": dataset["answer"],
    "contexts": dataset["contexts"],
    "ground_truth": dataset["ground_truth"],
})

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
        llm=lambda prompt: client.chat(
            model=MODEL_NAME,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.0,
        ).choices[0].message.content
    )
    return result


# Export JSON + CSV
def export_results(result, dataset, output_dir="./evaluation/ragas_results"):
    Path(output_dir).mkdir(exist_ok=True)

    # JSON complet
    with open(Path(output_dir) / "./evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "ragas_scores": result,
            "metadata": dataset["metadata"]
        }, f, indent=4)

    # CSV simple
    with open(Path(output_dir) / "./evaluation/results.csv", "w", newline="", encoding="utf-8") as f:
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
