# eval_ragas_prototype.py

import json
import csv
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Any
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent


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

# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
from mistralai import Mistral

from utils.vector_store import VectorStoreManager

from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import LLMResult, Generation

logfire.configure(
    service_name="sportsee-rag-eval",
    environment="development",)

# Modèles pydantic
from utils.pydantic_validation import (
    DocumentModel,
    PreparedDocumentModel,
    QuestionModel,
    ChunkModel,
    AnswerModel,
    RAGResultModel,)

# LLM Mistral (prototype)
client = Mistral(api_key=MISTRAL_API_KEY)

# Agent pydanticai
answer_agent = Agent(
    model=MODEL_NAME,
    client=client,
    response_model=AnswerModel
)

# LLM pour RAGAS
class MistralRagasLLM(BaseRagasLLM):
    def generate_text(
        self,
        prompt,
        n=1,
        temperature=1e-8,
        stop=None,
        callbacks=None,):

        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt.to_string()}] #[ChatMessage(role="user", content=prompt.to_string())],
            temperature=temperature,)

        text = response.output_text #choices[0].message.content
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None,  callbacks=None,):
        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,)

class MistralRagasEmbeddings(BaseRagasEmbeddings):
    def embed_query(self, text):
        response = client.embeddings.create(
            model="mistral-embed",
            input=[text],)
        return response.data[0].embedding

    def embed_documents(self, texts):
        response = client.embeddings(
            model="mistral-embed",
            input=texts,
        )
        return [item.embedding for item in response.data]


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
    messages = [{"role": "user", "content": final_prompt.to_string()}] #[ChatMessage(role="user", content=final_prompt)]
    response = client.chat.complet(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,)

    # Validation PydanticAI
    validated = answer_agent.run({
        "question": question,
        "context": context_str,
        "answer": response
    })

    return validated.choices[0].message.content



# Construction du dataset RAGAS
def build_ragas_dataset(eval_set, vector_store_manager):
    dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "metadata": [],}

    # Validation de la question avant son entrée dans le pipeline RAG
    for item in eval_set:
        validated_question = QuestionModel(
        question=item["question"])

        q = validated_question.question

        # Retrieval prototype
        search_results = vector_store_manager.search(q, k=SEARCH_K)

        # Validation des chunks retournés par le retrieval
        validated_chunks = [
        ChunkModel(
            text=res["text"],
            metadata=res.get("metadata", {}),
            score=res.get("score"))
            for res in search_results]


        if validated_chunks:
            context_str = "\n\n---\n\n".join([
                f"Source: {chunk.metadata.get('source', 'Inconnue')} "
                f"(Score: {chunk.score:.1f}%)\n"
                f"Contenu: {chunk.text}"
                for chunk in validated_chunks])
        else:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances."

        # Génération du prototype
        answer = generate_answer(q, context_str)

        # Validation Pydantic de la réponse du LLM
        validated_answer = AnswerModel(
            question=q,
            context=context_str,
            answer=answer)

        # Validation de l'ensemble des données transmises à RAGAS
        rag_result = RAGResultModel(
            question=validated_answer.question,
            contexts=[
                chunk.text
                for chunk in validated_chunks],

            answer=validated_answer.answer,
            ground_truth=item["ground_truth"])

        # Ajout des données validées au dataset
        dataset["question"].append(rag_result.question)
        dataset["answer"].append(rag_result.answer)
        dataset["contexts"].append(rag_result.contexts)
        dataset["ground_truth"].append(rag_result.ground_truth)
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

    # Construction du fichier détaillé par question
    detailed = []

    for i in range(len(dataset["question"])):
        detailed.append({
            "question": dataset["question"][i],
            "contexts": dataset["contexts"][i],
            "answer": dataset["answer"][i],
            "ground_truth": dataset["ground_truth"][i],
            "scores": result  # scores globaux répétés pour chaque question
        })

    with open(Path(output_dir) / "results_detaille.json", "w", encoding="utf-8") as f:json.dump(detailed, f, indent=4)



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
