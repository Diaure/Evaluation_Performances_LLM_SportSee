
import json
import csv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from mistralai.client import Mistral
from datasets import Dataset

from utils.vector_store_initial import VectorStoreManager
from utils.config_initial import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
    answer_similarity,
    context_entity_recall,
)

from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import BaseRagasEmbeddings

from langchain_core.outputs import LLMResult, Generation


# ============================================================
# CONFIGURATION MISTRAL
# ============================================================

client = Mistral(api_key=MISTRAL_API_KEY)


# ============================================================
# PROMPT RAG DU PROTOTYPE INITIAL
# ============================================================

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""


# ============================================================
# LLM MISTRAL POUR RAGAS
# ============================================================

class MistralRagasLLM(BaseRagasLLM):

    def generate_text(
        self,
        prompt,
        n=1,
        temperature=1e-8,
        stop=None,
        callbacks=None,
    ):
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt.to_string(),
                }
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
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=[text],
        )

        return response.data[0].embedding

    def embed_documents(self, texts):
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]


# ============================================================
# CHARGEMENT DU DATASET D'ÉVALUATION
# ============================================================

def load_evaluation_set(
    path="./evaluation/validation_set_initial.json",
):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CHARGEMENT DU VECTOR STORE INITIAL
# ============================================================

def load_vectorstore():
    manager = VectorStoreManager()

    if manager.index is None or not manager.document_chunks:
        raise RuntimeError(
            "VectorStoreManager n'a pas pu charger "
            "l'index ou les chunks."
        )

    return manager


# ============================================================
# GÉNÉRATION DE LA RÉPONSE DU PROTOTYPE INITIAL
# ============================================================

def generate_reponse_prototype(question, context_str):
    final_prompt = SYSTEM_PROMPT.format(
        context_str=context_str,
        question=question,
    )

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# ============================================================
# CONSTRUCTION DU DATASET RAGAS
# ============================================================

def build_ragas_dataset(eval_set, vector_store_manager):

    dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "metadata": [],
    }

    for index, item in enumerate(eval_set):

        question = item["question"]

        print(
            f"\nQuestion {index + 1}/{len(eval_set)} : "
            f"{question}"
        )

        # ----------------------------------------------------
        # 1. RETRIEVAL DU PROTOTYPE INITIAL
        # ----------------------------------------------------

        search_results = vector_store_manager.search(
            question,
            k=SEARCH_K,
        )

        # Extraction directe des chunks récupérés.
        # Aucun contrôle ni transformation par un modèle
        # de validation n'est effectué.
        contexts = [
            result["text"]
            for result in search_results
        ]

        # ----------------------------------------------------
        # 2. CONSTRUCTION DU CONTEXTE POUR MISTRAL
        # ----------------------------------------------------

        if search_results:

            context_str = "\n\n---\n\n".join(
                (
                    f"Source: "
                    f"{result.get('metadata', {}).get('source', 'Inconnue')} "
                    f"(Score: {result.get('score', 0):.1f}%)\n"
                    f"Contenu: {result['text']}"
                )
                for result in search_results
            )

        else:

            context_str = (
                "Aucune information pertinente trouvée "
                "dans la base de connaissances."
            )

        # ----------------------------------------------------
        # 3. GÉNÉRATION DIRECTE DE LA RÉPONSE
        # ----------------------------------------------------

        answer = generate_reponse_prototype(
            question=question,
            context_str=context_str,
        )

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "Le prototype a retourné une réponse vide "
                f"pour la question : {question}"
            )

        # ----------------------------------------------------
        # 4. AJOUT AU DATASET RAGAS
        # ----------------------------------------------------

        dataset["question"].append(question)
        dataset["answer"].append(answer)
        dataset["contexts"].append(contexts)
        dataset["ground_truth"].append(item["ground_truth"])
        dataset["metadata"].append(item)

        print("Réponse du prototype :", answer)

    return dataset


# ============================================================
# ÉVALUATION RAGAS
# ============================================================

def run_ragas_evaluation(dataset):

    ragas_dataset = Dataset.from_dict({
        "question": dataset["question"],
        "answer": dataset["answer"],
        "contexts": dataset["contexts"],
        "ground_truth": dataset["ground_truth"],
    })

    evaluateur_llm = MistralRagasLLM(
        run_config=RunConfig(),
    )

    evaluateur_embeddings = MistralRagasEmbeddings()

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
            answer_similarity,
            context_entity_recall,
        ],
        llm=evaluateur_llm,
        embeddings=evaluateur_embeddings,
        raise_exceptions=False,
    )

    return result


# ============================================================
# EXPORT DES RÉSULTATS
# ============================================================

def export_results(
    result,
    dataset,
    output_dir="ragas_results/before_pydantic_ai",
    evaluation_name="before_pydantic_ai",
):

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. SCORES GLOBAUX JSON
    # --------------------------------------------------------

    with open(
        output_path / "results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "evaluation": evaluation_name,
                "ragas_scores": dict(result),
                "metadata": dataset["metadata"],
            },
            f,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------------
    # 2. SCORES GLOBAUX CSV
    # --------------------------------------------------------

    with open(
        output_path / "results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["metric", "score"])

        for metric, score in result.items():
            writer.writerow([metric, score])

    # --------------------------------------------------------
    # 3. SCORES PAR QUESTION
    # --------------------------------------------------------

    result_df = result.to_pandas()

    metrics = [
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_similarity",
        "context_entity_recall",
    ]

    scores_par_question = []

    for i in range(len(dataset["question"])):

        metadata = dataset["metadata"][i]

        question_id = metadata.get(
            "question_id",
            metadata.get("id", f"Q{i + 1:03d}"),
        )

        row = {
            "question_id": question_id,
            "question": dataset["question"][i],
        }

        for metric in metrics:

            if metric in result_df.columns:

                value = result_df.iloc[i][metric]

                row[metric] = (
                    None
                    if pd.isna(value)
                    else float(value)
                )

            else:
                row[metric] = None

        scores_par_question.append(row)

    scores_df = pd.DataFrame(scores_par_question)

    scores_df.to_csv(
        output_path / "scores_par_question.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4. RÉSULTATS DÉTAILLÉS
    # --------------------------------------------------------

    detailed = []

    for i in range(len(dataset["question"])):

        detailed.append({
            "question": dataset["question"][i],
            "contexts": dataset["contexts"][i],
            "answer": dataset["answer"][i],
            "ground_truth": dataset["ground_truth"][i],
            "metadata": dataset["metadata"][i],
        })

    with open(
        output_path / "results_detaille.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            detailed,
            f,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Chargement du VectorStore du prototype initial...")

    vector_store_manager = load_vectorstore()

    print("\nChargement du dataset d'évaluation...")

    eval_set = load_evaluation_set()

    print("\nConstruction du dataset RAGAS...")

    dataset = build_ragas_dataset(
        eval_set,
        vector_store_manager,
    )

    print("\nÉvaluation RAGAS du prototype initial...")

    result = run_ragas_evaluation(dataset)

    print("\nScores du prototype initial :")

    for metric, score in result.items():
        print(f"  {metric}: {score}")

    print("\nExport des résultats...")

    export_results(
        result=result,
        dataset=dataset,
        output_dir="ragas_results/before_pydantic_ai",
        evaluation_name="before_pydantic_ai",
    )

    print(
        "\nÉvaluation du prototype initial terminée."
    )
