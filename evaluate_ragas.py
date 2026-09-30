# eval_ragas_prototype.py

import json
import csv
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any

from utils.vector_store import VectorStoreManager

# Configuration projet mistral
from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,)

import logfire
# Configuration environnement logfire avec pydantic
logfire.configure(
    service_name="sportsee-rag-eval",
    environment="development",
    send_to_logfire="if-token-present",)

logfire.instrument_pydantic()
logfire.instrument_pydantic_ai()


from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

# Modèles pydantic
from utils.pydantic_validation import (
    DocumentModel,
    PreparedDocumentModel,
    QuestionModel,
    ChunkModel,
    AnswerModel,
    RAGResultModel,
    ControlledAnswerModel,)
from mistralai.client import Mistral

# LLM Mistral (prototype)
client = Mistral(api_key=MISTRAL_API_KEY)
mistral_provider = MistralProvider(mistral_client=client,)

# Modèle Mistral utilisé par Pydantic AI
pydantic_ai_model = MistralModel(MODEL_NAME, provider="mistral",)

# Agent pydanticai
answer_agent = Agent(
    model=pydantic_ai_model,
    output_type=ControlledAnswerModel,
    instructions=(
        "Tu es un contrôleur de réponses pour un assistant NBA. "
        "\n\n"
        "Ta tâche est de vérifier une réponse générée par un LLM "
        "à partir d'un contexte documentaire fourni. "
        "\n\n"
        "Règles : "
        "\n"
        "1. Utilise uniquement les informations présentes dans le contexte. "
        "\n"
        "2. Ne complète jamais les informations manquantes avec tes connaissances. "
        "\n"
        "3. Si une information de la réponse n'est pas présente ou "
        "pas justifiable par le contexte, considère-la comme non fondée. "
        "\n"
        "4. Conserve les informations correctement justifiées. "
        "\n"
        "5. Si la réponse ne peut pas être justifiée par le contexte, "
        "indique-le clairement dans `grounded_in_context`. "
        "\n"
        "6. Liste dans `unsupported_claims` les affirmations non justifiées."),)


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

from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import (LLMResult, Generation,)


# Prompt RAG du prototype
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


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
            messages=[{"role": "user", "content": prompt.to_string()}],
            temperature=temperature,)

        text = response.choices[0].message.content
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
            inputs=[text],)
        return response.data[0].embedding

    def embed_documents(self, texts):
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=texts,
        )
        return [item.embedding for item in response.data]


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

# générer la réponse du prototype (réponse évaluée avant contrôle pydandic)
def generate_reponse_prototype(question, context_str):
    final_prompt = SYSTEM_PROMPT.format(
        context_str=context_str,
        question=question,)

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt,}], 
        temperature=0.1,)

    return response.choices[0].message.content


# Contrôle pydantic AI
def control_reponse_pydantic_ai(
    question,
    context_str,
    raw_answer,):

    control_prompt = f"""
        QUESTION :
        {question}

        CONTEXTE :
        {context_str}

        RÉPONSE PRODUITE PAR LE PROTOTYPE :
        {raw_answer}

        Analyse cette réponse en appliquant strictement les règles
        définies dans tes instructions.
        """

    result = answer_agent.run_sync(control_prompt)

    return result.output

# Mini prompt pour la validation des réponses avec pydantic
def validation_reponse_pydantic_ai(question, context_str, answer):
    prompt = f"""
        Tu es chargé de contrôler une réponse produite par un assistant NBA.

        CONTEXTE DISPONIBLE:
        {context_str}

        QUESTION:
        {question}

        RÉPONSE PRODUITE:
        {answer}

        Analyse la réponse uniquement à partir du contexte fourni.

        Si une information de la réponse n'est pas suffisamment justifiée par le contexte,
        considère-la comme potentiellement non fondée.
        """

    result = answer_agent.run_sync(prompt)

    return result.output

# Génération de réponse du prototype avec validation pydantic ai
def generate_answer(question, context_str):
    reponse_brute = generate_reponse_prototype(question, context_str,)

    # Validation PydanticAI
    validated = validation_reponse_pydantic_ai(
        question,
        context_str,
        reponse_brute,)

    return validated.answer


# Construction du dataset RAGAS
def build_ragas_dataset(eval_set, vector_store_manager):
    baseline_dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "metadata": [],}

    controlled_dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "metadata": [],}

    # Validation de la question avant son entrée dans le pipeline RAG
    for index, item in enumerate(eval_set):
        print(
            f"Question {index + 1}/{len(eval_set)} : "
            f"{item['question']}")

        # validation de la question
        validated_question = QuestionModel(
        question=item["question"])

        question = validated_question.question

        # Retrieval prototype
        search_results = vector_store_manager.search(question, k=SEARCH_K)

        # Validation des chunks retournés par le retrieval
        validated_chunks = [ChunkModel(
            text=result["text"],
            metadata=result.get("metadata", {}),
            score=result.get("score"))
            for result in search_results]

        # Construction du contexte
        if validated_chunks:
            context_str = "\n\n---\n\n".join([(
                f"Source: "
                f"{chunk.metadata.get('source', 'Inconnue')} "
                f"(Score: {chunk.score:.1f}%)\n"
                f"Contenu: {chunk.text}")
                for chunk in validated_chunks])
        else:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances."

        # Génération du prototype avant controle pydantic
        reponse_raw = generate_reponse_prototype(question, context_str)

        # Validation Pydantic de la réponse du LLM
        validated_answer_brute = AnswerModel(
            question=question,
            context=context_str,
            answer=reponse_raw)

        # RAGAS avant validation AI
        baseline_result = RAGResultModel(
            question=validated_answer_brute.question,
            contexts=[chunk.text for chunk in validated_chunks],
            answer=validated_answer_brute.answer,
            ground_truth=item["ground_truth"],)


        baseline_dataset["question"].append(baseline_result.question)
        baseline_dataset["answer"].append(baseline_result.answer)
        baseline_dataset["contexts"].append(baseline_result.contexts)
        baseline_dataset["ground_truth"].append(baseline_result.ground_truth)
        baseline_dataset["metadata"].append(item)

        # Validation pydantic AI
        controlled_answer = control_reponse_pydantic_ai(
            question=question,
            context_str=context_str,
            raw_answer=reponse_raw,)

        # Réponse après validation
        reponse_finale = controlled_answer.answer

        # validation
        reponse_validee_controleer = AnswerModel(
            question=question,
            context=context_str,
            answer=reponse_finale,)

        # résultats ragas après controle
        results_controlled = RAGResultModel(
            question=reponse_validee_controleer.question,
            contexts=[
                chunk.text
                for chunk in validated_chunks],

            answer=reponse_validee_controleer.answer,
            ground_truth=item["ground_truth"])

        # dataset après validation
        controlled_dataset["question"].append(results_controlled.question)
        controlled_dataset["answer"].append(results_controlled.answer)
        controlled_dataset["contexts"].append(results_controlled.contexts)
        controlled_dataset["ground_truth"].append(results_controlled.ground_truth)
        controlled_dataset["metadata"].append({**item,
                "pydantic_ai_control": {
                "grounded_in_context": controlled_answer.grounded_in_context,
                "unsupported_claims": controlled_answer.unsupported_claims,
                "raw_answer": reponse_raw,
                },})

        # Ajout des données validées au dataset
        # dataset["question"].append(rag_result.question)
        # dataset["answer"].append(rag_result.answer)
        # dataset["contexts"].append(rag_result.contexts)
        # dataset["ground_truth"].append(rag_result.ground_truth)
        # dataset["metadata"].append(item)

    return baseline_dataset, controlled_dataset


# Évaluation RAGAS
def run_ragas_evaluation(dataset):

    ragas_dataset = Dataset.from_dict({
    "question": dataset["question"],
    "answer": dataset["answer"],
    "contexts": dataset["contexts"],
    "ground_truth": dataset["ground_truth"],
})

    evaluateur_llm = MistralRagasLLM(run_config=RunConfig())
    evaluateur_embeddings = MistralRagasEmbeddings()

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
        llm=evaluateur_llm,
        embeddings=evaluateur_embeddings)
    
    return result


# Export JSON + CSV
# def export_results(result, dataset, output_dir="ragas_results"):
#     Path(output_dir).mkdir(exist_ok=True)

#     # JSON complet
#     with open(Path(output_dir) / "results.json", "w", encoding="utf-8") as f:
#         json.dump({
#             "ragas_scores": result,
#             "metadata": dataset["metadata"]
#         }, f, indent=4)

#     # CSV simple
#     with open(Path(output_dir) / "results.csv", "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["metric", "score"])
#         for metric, score in result.items():
#             writer.writerow([metric, score])

#     # Construction du fichier détaillé par question
#     detailed = []

#     for i in range(len(dataset["question"])):
#         detailed.append({
#             "question": dataset["question"][i],
#             "contexts": dataset["contexts"][i],
#             "answer": dataset["answer"][i],
#             "ground_truth": dataset["ground_truth"][i],
#             "scores": result  # scores globaux répétés pour chaque question
#         })

#     with open(Path(output_dir) / "results_detaille.json", "w", encoding="utf-8") as f:json.dump(detailed, f, indent=4)

def export_results(
    result,
    dataset,
    output_dir,
    evaluation_name,):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True,)

    # Scores globaux JSON
    with open(output_path / "results.json", "w", encoding="utf-8",) as f:
        json.dump({"evaluation": evaluation_name, "ragas_scores": dict(result), "metadata": dataset["metadata"], }, f, indent=4, ensure_ascii=False, default=str,)

    # Scores globaux CSV
    with open(output_path / "results.csv", "w", newline="", encoding="utf-8",) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "score",])

        for metric, score in result.items():
            writer.writerow([metric, score,])

    # Résultats détaillés
    detailed = []
    for i in range(len(dataset["question"])):
        detailed.append({
                "question": dataset["question"][i],
                "contexts": dataset["contexts"][i],
                "answer": dataset["answer"][i],
                "ground_truth": dataset["ground_truth"][i],
                "metadata": dataset["metadata"][i],})

    with open(output_path / "results_detaille.json", "w", encoding="utf-8",) as f:
        json.dump(detailed, f, indent=4, ensure_ascii=False, default=str,)

# Scores comparés
def export_comparison(
    baseline_result,
    controlled_result,
    output_dir="ragas_results",):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True,)

    comparison = []
    metrics = set(list(baseline_result.keys()) + list(controlled_result.keys()))
    for metric in metrics:
        before = baseline_result.get(metric)
        after = controlled_result.get(metric)

        difference = None
        if (before is not None
            and after is not None):
            difference = after - before


        comparison.append({
                "metric": metric,
                "before_pydantic_ai": before,
                "after_pydantic_ai": after,
                "difference": difference,})

    with open(output_path / "comparison.json", "w", encoding="utf-8",) as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False,)

    with open(output_path / "comparison.csv", "w", newline="", encoding="utf-8", ) as f:
        writer = csv.writer(f)
        writer.writerow([
                "metric",
                "before_pydantic_ai",
                "after_pydantic_ai",
                "difference",])

        for row in comparison:
            writer.writerow([
                    row["metric"],
                    row["before_pydantic_ai"],
                    row["after_pydantic_ai"],
                    row["difference"],])

# MAIN
if __name__ == "__main__":

    print("Chargement du VectorStore prototype…")
    vector_store_manager = load_vectorstore()

    print("Chargement du dataset d’évaluation…")
    eval_set = load_evaluation_set()

    print("Construction du dataset RAGAS…")
    baseline_dataset, controlled_dataset = build_ragas_dataset(eval_set, vector_store_manager,)

    print("\nÉvaluation RAGAS AVANT contrôle Pydantic AI…")
    baseline_result = run_ragas_evaluation(baseline_dataset)

    print("\nScores AVANT contrôle :")
    for metric, score in baseline_result.items():
        print(f"  {metric}: {score}")

    print("\nÉvaluation RAGAS APRÈS contrôle Pydantic AI…")
    controlled_result = run_ragas_evaluation(controlled_dataset)

    print("\nScores APRÈS contrôle :")
    for metric, score in controlled_result.items():
        print(f"  {metric}: {score}")

    print("\nExport des résultats…")

    export_results(
        baseline_result,
        baseline_dataset,
        output_dir="ragas_results/before_pydantic_ai",
        evaluation_name="before_pydantic_ai",)

    export_results(
        controlled_result,
        controlled_dataset,
        output_dir="ragas_results/after_pydantic_ai",
        evaluation_name="after_pydantic_ai",)

    export_comparison(
        baseline_result,
        controlled_result,
        output_dir="ragas_results",)

    print("\nÉvaluation terminée.")
