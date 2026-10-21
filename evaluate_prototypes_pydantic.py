
"""
Évaluation du prototype NBA amélioré avec Pydantic AI et RAGAS.

Pipeline :
1. Charger le jeu de validation.
2. Router chaque question vers SQL, RAG ou BOTH.
3. Récupérer les informations nécessaires.
4. Générer une réponse avec Mistral.
5. Contrôler la réponse avec Pydantic AI.
6. Évaluer uniquement la réponse contrôlée avec RAGAS.
7. Exporter les résultats.

Aucune évaluation avant Pydantic AI.
Aucune comparaison avec l'ancienne évaluation.

Exécution depuis la racine du dépôt :
    python evaluate_prototype_ameliore_pydantic.py
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any
import ast

import logfire
import pandas as pd

from datasets import Dataset
from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

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

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from utils.pydantic_validation import (
    QuestionModel,
    ChunkModel,
    AnswerModel,
    RAGResultModel,
    ControlledAnswerModel,
)
from sql_tool import sql_tool


# Configuration

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

VALIDATION_SET_PATH = (BASE_DIR / "evaluation" / "validation_set_new.json")

OUTPUT_DIR = (
    BASE_DIR
    / "ragas_results"
    / "prototype_ameliore_pydantic")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",)
logger = logging.getLogger(__name__)
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY est absente de la configuration.")

# Instrumentation
logfire.configure(
    service_name="sportsee-rag-eval",
    environment="development",
    send_to_logfire="if-token-present",)

logfire.instrument_pydantic()
logfire.instrument_pydantic_ai()

client = Mistral(api_key=MISTRAL_API_KEY)
mistral_provider = MistralProvider( mistral_client=client)
pydantic_ai_model = MistralModel(MODEL_NAME, provider=mistral_provider,)

# Aget pydantic AI
answer_agent = Agent(
    model=pydantic_ai_model,
    output_type=ControlledAnswerModel,
    instructions=(
        "Tu es un contrôleur de réponses pour un assistant NBA. "
        "Tu vérifies une réponse générée à partir d'un contexte "
        "documentaire et/ou de résultats SQL fournis. "
        "Utilise exclusivement les informations présentes dans "
        "le contexte. Ne complète jamais les informations "
        "manquantes avec tes propres connaissances. "
        "Conserve les affirmations justifiées. "
        "Signale les affirmations non justifiées dans "
        "unsupported_claims. Si le contexte est insuffisant, "
        "indique-le clairement dans answer et dans "
        "grounded_in_context. Ne produis jamais une réponse vide."
    ),
)

# Prompts prototype amélioré
ROUTING_PROMPT = """
Tu es un routeur de questions pour une application d'analyse des statistiques NBA.

Tu dois choisir exactement UNE source parmi :
- SQL
- RAG
- BOTH

RÈGLES DE DÉCISION

1. SQL
Choisis SQL si la question nécessite principalement:
- des statistiques numériques présentes dans la base relationnelle
- un classement, un maximum, un minimum ou un top N
- une comparaison entre joueurs ou équipes
- une comparaison portant sur plusieurs statistiques
- une question demandant quelle équipe ou quel joueur arrive en tête pour une ou plusieurs statistiques
- un calcul, une agrégation ou un filtrage sur des données structurées
- des statistiques sur les derniers matchs d'un joueur, si les données nécessaires sont disponibles dans la base 
- des pourcentages tels que FG%, 3P% ou FT%.

Règles pour les comparaisons statistiques:
- Lorsqu'une question demande de comparer des équipes sur plusieurs statistiques, calcule chaque statistique séparément et identifie le maximum ou le classement correspondant à chacune.
- Lorsqu'une question demande quelle équipe arrive en tête pour plusieurs indicateurs, retourne un résultat distinct pour chaque indicateur demandé.
- Utilise la table teams pour les statistiques d'équipe déjà enregistrées.
- Pour les statistiques disponibles uniquement au niveau des joueurs, relie teams, players et stats, puis agrège les valeurs par équipe.
- Vérifie le niveau de granularité des données avant d'utiliser SUM: ne somme pas plusieurs fois des statistiques déjà agrégées et ne mélange pas des périodes différentes.
- Ne somme jamais une statistique d'équipe déjà agrégée après une jointure qui multiplie les lignes.
- Respecte exactement les statistiques demandées et les colonnes réellement présentes dans le schéma.
- Conserve les ex æquo lorsque plusieurs équipes partagent le maximum.
- N'invente aucune colonne, aucune valeur et aucun résultat.

RÈGLES POUR LES COMPARAISONS DOMICILE / EXTÉRIEUR
- Ne déduis jamais le lieu d'un match à partir de son résultat, du nombre de victoires ou de défaites, du score ou de toute autre statistique indirecte.
- Une comparaison domicile / extérieur n'est autorisée que si le schéma de la base contient un champ explicite permettant d'identifier le lieu du match, ou si une autre relation documentée permet de déterminer ce lieu sans ambiguïté.
- N'invente jamais de catégories telles que 'Domicile' et 'Extérieur' dans une requête SQL pour compenser l'absence d'une information dans les données.
- N'utilise pas un CASE WHEN pour attribuer un lieu à un match à partir d'une condition qui ne représente pas explicitement ce lieu.
- Avant de générer la requête, vérifie que les tables et colonnes nécessaires existent et que leur signification permet réellement de répondre à la question.
- Si aucune donnée fiable ne permet d'identifier le lieu des matchs, ne génère pas de résultats chiffrés pour cette comparaison. Indique que la base ne contient pas les informations nécessaires.
- Ne présente jamais une estimation, une hypothèse ou une classification construite par le modèle comme une donnée observée dans la base.

Si toutes les informations nécessaires peuvent être obtenues depuis SQL,
choisis SQL, même si la question est formulée en langage naturel.

2. RAG
Choisis RAG si la question porte principalement sur:
- des informations descriptives issues des documents
- des explications, des commentaires ou des éléments textuels
- des opinions ou des analyses issues des documents
- des informations qui ne sont pas disponibles dans la base SQL, mais qui sont présentes dans les documents récupérables.

3. BOTH
Choisis BOTH uniquement si la question nécessite réellement à la fois les données SQL et des informations issues des documents RAG.
Ne choisis jamais BOTH simplement parce que la question est complexe.

Exemples :
- « Quelle équipe a marqué le plus de points ? » → SQL
- « Compare les statistiques de deux joueurs » → SQL
- « Quelle équipe arrive en tête pour les points, rebonds et passes ? » → SQL
- « Que pensent les internautes de l'efficacité de ce joueur ? » → RAG
- « Compare les statistiques de ce joueur et les critiques formulées à son sujet sur Reddit » → BOTH

IMPORTANT
- Ne confonds pas complexité et besoin de plusieurs sources.
- Un classement ou une comparaison statistique ne nécessite pas RAG si SQL fournit toutes les données nécessaires.
- Une question générale ne nécessite pas automatiquement BOTH.
- Ne choisis pas BOTH simplement parce que plusieurs statistiques ou plusieurs entités sont concernées.
- Le choix des statistiques pertinentes et la construction des requêtes relèvent des étapes de traitement suivantes.
- Retourne uniquement le label: SQL, RAG ou BOTH.

QUESTION :
{question}
"""

SYSTEM_PROMPT = """
Tu es NBA Analyst AI, un assistant spécialisé dans l'analyse NBA.

Réponds précisément à la question en utilisant exclusivement les informations fournies dans le contexte SQL et/ou les documents RAG.
N'invente aucune donnée, statistique, source ou conclusion qui ne soit pas étayée par le contexte.

RÈGLES DE RÉPONSE:

1. REPONDRE DIRECTEMENT 
- Réponds directement à la question dès la première phrase. 
- Sois concis et adapte la longueur de la réponse à la complexité de la question.
- N'ajoute ni introduction, ni titre, ni conclusion générale inutile.
- Ne fournis pas d'informations sans rapport avec la question.

2. RESPECT DE LA QUESTION
- Identifie précisément les joueurs, équipes, matchs, périodes et statistiques demandés.
- Respecte le périmètre de la question: un joueur, une équipe, plusieurs entités, un match particulier ou l'ensemble des données.
- Ne suppose jamais qu'une équipe, un joueur, un match ou une période est implicitement désigné si le contexte ne permet pas de l'établir.
- Si une information indispensable manque et qu'aucune réponse fiable ne peut être calculée, indique clairement cette limite.
- Ne transforme pas une question ambiguë en une question différente pour produire malgré tout une réponse.

3. SÉLECTIONNER LES STATISTIQUES PERTINENTES
- Si la question demande une statistique précise, utilise cette statistique uniquement, sauf si une précision supplémentaire est indispensable.
- Si la question demande une comparaison générale de joueurs ou d'équipes sans préciser de statistique, utilise les statistiques pertinentes disponibles dans les résultats SQL pour caractériser la comparaison.
- Pour les joueurs, examine selon leur disponibilité les points, rebonds, passes décisives, pourcentages de réussite, tirs à trois points, lancers francs et autres indicateurs effectivement présents dans le contexte.
- Pour les équipes, examine selon leur disponibilité les points, victoires, défaites et autres statistiques d'équipe effectivement présentes dans le contexte.
- Ne sélectionne pas uniquement les points lorsqu'une comparaison plus large est demandée et que d'autres statistiques pertinentes sont disponibles.
- Ne prétends pas qu'une entité est globalement meilleure qu'une autre sur la seule base d'une statistique, sauf si la question définit ce critère.
- Ne construis pas de score global en additionnant des statistiques de natures ou d'unités différentes.
- Ne mentionne que les indicateurs utiles pour répondre, sans produire systématiquement une liste exhaustive de toutes les colonnes disponibles.

4. CLASSEMENTS ET RÉSULTATS MULTIPLES
- Respecte le nombre de résultats explicitement demandé.
- Si un classement est demandé sans nombre précisé, restitue le classement fourni par le contexte SQL dans la limite des résultats disponibles.
- Ne réduis pas arbitrairement un classement à un ou deux résultats lorsque le contexte en fournit davantage.
- Respecte l'ordre de classement indiqué par les résultats SQL.
- Préserve les ex æquo lorsqu'ils sont présents dans les résultats.
- N'ajoute pas de joueurs ou d'équipes absents des résultats SQL pour compléter un classement.
- Ne présente pas un classement partiel comme un classement exhaustif.
- Si le contexte ne contient qu'un extrait ou un nombre limité de résultats, précise cette limite lorsque cela change l'interprétation de la réponse.

5. FIABILITÉ DES STATISTIQUES ET DES CALCULS
- Les valeurs statistiques et les calculs quantitatifs doivent être fondés sur les résultats SQL fournis.
- Vérifie que la statistique utilisée correspond exactement à celle demandée. Par exemple, le pourcentage de réussite aux tirs à trois points ne doit pas être confondu avec le pourcentage de réussite aux tirs toutes catégories.
- Ne déduis pas le lieu d'un match, la période concernée ou le dernier match d'un joueur à partir d'un indicateur qui ne permet pas de l'établir.
- Ne confonds pas le dernier match présent dans la base avec le dernier match joué par chaque joueur.
- Ne suppose pas que les données sont disponibles au niveau individuel et par match si le contexte ne fournit que des statistiques agrégées.
- N'effectue pas de calcul si les données nécessaires sont absentes, incomplètes ou à un niveau de détail incompatible avec ce calcul.
- Ne présente pas une valeur calculée comme une valeur directement observée dans la base.
- Ne corrige pas silencieusement une incohérence dans les données.

6. UTILISER SQL ET RAG CORRECTEMENT
- Les résultats SQL constituent la référence pour les statistiques, les classements et les calculs quantitatifs.
- Les documents RAG constituent la source pour les informations qualitatives, les opinions, les analyses et les commentaires issus des documents récupérés.
- Pour une question exclusivement statistique, réponds à partir des résultats SQL pertinents.
- Pour une question exclusivement qualitative, réponds à partir des documents RAG pertinents.
- Pour une question combinant statistiques et informations qualitatives, utilise les résultats SQL pour les données quantitatives et les documents RAG pour les éléments qualitatifs.
- Dans une réponse mixte, mets en relation les informations pertinentes sans attribuer aux auteurs des opinions qu'ils n'ont pas exprimées.
- N'utilise pas un document RAG pour contredire une valeur SQL sans élément explicite justifiant cette contradiction.
- Ne présente pas une opinion issue du RAG comme un fait statistique établi.
- N'ajoute pas de métriques, de commentaires ou de conclusions absents des sources fournies.
- Si une partie de la question est documentée mais qu'une autre ne l'est pas, réponds à la partie étayée et signale clairement la limite restante.

7. PERTINENCE DES DOCUMENTS RAG
- Utilise uniquement les passages qui apportent des éléments utiles à la question.
- Ne tire pas de conclusion à partir d'un passage simplement parce qu'il contient des mots-clés similaires à ceux de la question.
- Ne fusionne pas des opinions provenant de contextes différents en leur attribuant un consensus qui n'est pas établi.
- Si les documents sont contradictoires, mentionne cette divergence lorsque cela est pertinent.
- Si les passages récupérés ne permettent pas de répondre, indique que les documents disponibles ne fournissent pas suffisamment d'informations.
  
8. INFORMATIONS INSUFFISANTES OU ABSENTES
- Si les données disponibles ne permettent pas de répondre de façon fiable, dis-le explicitement.
- Distingue une donnée absente du contexte d'une donnée qui n'existe pas dans la base.
- Ne prétends pas qu'une question est impossible à résoudre si les résultats fournis permettent effectivement de la traiter.
- Ne fabrique pas de réponse pour éviter de reconnaître une limite.
- Si une clarification est indispensable, indique précisément l'information manquante.

9. FORMAT FINAL
- Fournis une réponse claire, factuelle et proportionnée à la question.
- Pour une question simple, une à trois phrases suffisent généralement.
- Pour une comparaison générale, plusieurs indicateurs peuvent être nécessaires.
- Pour un classement, utilise une liste si elle améliore la lisibilité.
- N'ajoute pas de commentaire général, de conseil ou de spéculation qui ne soit pas demandé ou directement justifié par le contexte.

QUESTION :
{question}

INFORMATIONS RÉCUPÉRÉES :
{context}

RÉPONSE :
"""

# Chargement du jeu de validation
def load_evaluation_set():
    if not VALIDATION_SET_PATH.exists():
        raise FileNotFoundError(
            f"Jeu de validation introuvable : "
            f"{VALIDATION_SET_PATH}")

    with VALIDATION_SET_PATH.open("r", encoding="utf-8") as file:
        eval_set = json.load(file)

    if not isinstance(eval_set, list) or not eval_set:
        raise ValueError("Le jeu de validation doit être une liste JSON non vide.")

    for i, item in enumerate(eval_set, start=1):
        item.setdefault("id", f"Q{i:03d}")
        if not str(item.get("question", "")).strip():
            raise ValueError(f"Question vide pour {item['id']}.")

        if not str(item.get("ground_truth", "")).strip():
            raise ValueError(f"Ground truth absente pour {item['id']}.")
    return eval_set

# Chargement du vector store
def load_vectorstore():
    manager = VectorStoreManager()
    if (manager.index is None or not manager.document_chunks):
        raise RuntimeError("VectorStoreManager n'a pas pu charger l'index FAISS ou les chunks.")
    
    return manager

# Routing
def route_question(question: str) -> str:
    prompt = ROUTING_PROMPT.format(question=question)
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,}],
        temperature=0.0,)

    if not response.choices:
        raise RuntimeError("Mistral n'a retourné aucun résultat pour le routage.")

    raw_route = str(
        response.choices[0].message.content or "").strip().upper()
    match = re.match(r"^\s*(BOTH|SQL|RAG)\b", raw_route,)
    if match:
        return match.group(1)

    logger.warning("Route non reconnue : %r. Repli sur RAG.", raw_route,)
    return "RAG"

# Récupération RAG
def retrieve_rag(vector_store, question):
    search_results = (vector_store.search(question, k=SEARCH_K) or [])

    display_contexts = []
    ragas_contexts = []
    chunks_details = []

    for result in search_results:
        chunk = ChunkModel(
            text=result["text"],
            metadata=result.get("metadata", {}),
            score=result.get("score"),)

        chunk_text = str(chunk.text or "").strip()
        if not chunk_text:
            continue

        source = chunk.metadata.get("source", "Inconnue")
        display_contexts.append(
            f"Source : {source}\n"
            f"Contenu : {chunk_text}")

        # récupération des chunks
        ragas_contexts.append(chunk_text)
        chunks_details.append({
            "source": source,
            "text": chunk_text,
            "score": chunk.score,
            "metadata": chunk.metadata,})

    return (display_contexts, ragas_contexts, chunks_details,)

# Choix des sources SQL / RAG / BOTH
def retrieve_sources(item, vector_store):
    validated_question = QuestionModel(question=item["question"])
    question = validated_question.question
    route = route_question(question)
    expected_route = infer_expected_route(item)
    logger.info("Routage %s | attendu=%s | sélectionné=%s | question=%s", item["id"], expected_route, route, question,)

    context_sections = []
    ragas_contexts = []
    chunks_details = []
    sql_result = None
    source_errors = []

    # Source SQL
    if route in {"SQL", "BOTH"}:
        try:
            sql_result = str(sql_tool.invoke({"question": question}))
            context_sections.append("RÉSULTATS SQL :\n" + sql_result)
            ragas_contexts.append("RÉSULTATS SQL :\n" + sql_result)

        except Exception as exc:
            logger.exception("Erreur SQL pour %s", item["id"],)
            source_errors.append(f"SQL : {type(exc).__name__}: {exc}")
            context_sections.append("RÉSULTATS SQL : erreur pendant l'interrogation de la base.")
            ragas_contexts.append("Erreur SQL : aucune donnée exploitable n'a été obtenue.")

    # Source RAG
    if route in {"RAG", "BOTH"}:
        try:
            (display_contexts, retrieved_contexts, chunks_details,) = retrieve_rag(vector_store, question,)

            if display_contexts:
                context_sections.append(
                    "CONTEXTE DOCUMENTAIRE RAG :\n"
                    + "\n\n---\n\n".join(display_contexts))

                ragas_contexts.extend(retrieved_contexts)

            else:
                context_sections.append("CONTEXTE DOCUMENTAIRE RAG : aucun résultat pertinent trouvé.")
                ragas_contexts.append("Aucun contexte documentaire pertinent n'a été retrouvé.")

        except Exception as exc:
            logger.exception("Erreur RAG pour %s", item["id"],)
            source_errors.append(f"RAG : {type(exc).__name__}: {exc}")
            context_sections.append("CONTEXTE DOCUMENTAIRE RAG : erreur pendant la recherche.")
            ragas_contexts.append("Erreur RAG : aucun contexte documentaire exploitable n'a été obtenu.")

    if not context_sections:
        context_sections.append("Aucune source n'a été interrogée.")

    if not ragas_contexts:
        ragas_contexts.append("Aucune information n'a été récupérée.")

    return {
        "question": question,
        "route": route,
        "context_str": "\n\n".join(context_sections),
        "ragas_contexts": ragas_contexts,
        "sql_result": sql_result,
        "chunks_details": chunks_details,
        "source_errors": source_errors,}

# Génération de réponse du prototype
def generate_reponse_prototype(question, context_str):
    final_prompt = SYSTEM_PROMPT.format(question=question, context=context_str,)
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": final_prompt,}],
        temperature=0.1,)

    if not response.choices:
        raise RuntimeError("Mistral n'a retourné aucune réponse.")
    
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("La réponse brute du prototype est vide.")

    return answer.strip()


# Contrôle pydantic AI
def validation_reponse_pydantic_ai(question, context_str, raw_answer,):
    prompt = f"""
QUESTION :
{question}

CONTEXTE DISPONIBLE :
{context_str}

RÉPONSE PRODUITE PAR LE PROTOTYPE :
{raw_answer}

Vérifie chaque affirmation à partir du contexte fourni.

- Conserve toutes les affirmations factuelles justifiées par le contexte.
- Conserve les chiffres, les unités, les noms, les classements et les explications factuelles corrects présents dans la réponse initiale.
- Signale les affirmations non justifiées et corrige-les ou supprime-les si nécessaire.
- Si le contexte est insuffisant, indique-le clairement sans inventer d'information.
- N'utilise aucune connaissance extérieure au contexte.
- Préserve autant que possible la structure et le niveau de détail de la réponse initiale.
- Conserve les noms, les statistiques, les valeurs numériques, les unités, les pourcentages, les classements et les explications factuelles justifiés.
- Ne supprime pas une information correcte au seul motif qu'elle serait facultative.
- Ne remplace pas une réponse détaillée et justifiée par une simple liste de noms ou une réponse plus générale.
- Ne raccourcis pas la réponse lorsque toutes ses affirmations sont justifiées.
- Si une seule partie d'une phrase est incorrecte ou non justifiée, corrige uniquement cette partie et conserve le reste.
- Ne modifie pas les valeurs numériques correctes.
- Réponds directement à la question.
- Ne produis jamais une réponse vide.

PRINCIPE PRIORITAIRE:

Si toutes les affirmations de la réponse initiale sont justifiées par le contexte, restitue la réponse initiale en conservant son contenu et son niveau de détail.
La concision ne constitue pas un motif pour supprimer des informations correctes.
"""

    result = answer_agent.run_sync(prompt)
    controlled_answer = result.output

    if not isinstance(
        controlled_answer,
        ControlledAnswerModel,):
        controlled_answer = (
            ControlledAnswerModel.model_validate(
                controlled_answer))

    if not controlled_answer.answer.strip():
        raise ValueError(
            "Pydantic AI a retourné une réponse vide "
            f"pour la question : {question}")

    return controlled_answer


# Construction du dataset
def build_controlled_dataset(eval_set, vector_store,):
    controlled_dataset = []
    for index, item in enumerate(eval_set, start=1):
        logger.info(
            "Question %s/%s : %s",
            index,
            len(eval_set),
            item["question"],)

        # Routage et récupération des sources
        retrieved = retrieve_sources(item, vector_store,)
        question = retrieved["question"]
        context_str = retrieved["context_str"]

        # Génération de la réponse brute
        raw_answer = generate_reponse_prototype(question, context_str,)

        # Validation structurelle Pydantic, comme dans l'ancien script
        validated_answer = AnswerModel(
            question=question,
            context=context_str,
            answer=raw_answer,)

        # Contrôle Pydantic AI
        controlled_answer = validation_reponse_pydantic_ai(
            question=question,
            context_str=context_str,
            raw_answer=raw_answer,)

        final_answer = controlled_answer.answer.strip()

        # Validation de la réponse finale
        validated_answer = AnswerModel(
            question=question,
            context=context_str,
            answer=final_answer,)

        # RAGAS reçoit uniquement la réponse après contrôle
        rag_result = RAGResultModel(
            question=validated_answer.question,
            contexts=retrieved["ragas_contexts"],
            answer=validated_answer.answer,
            ground_truth=item["ground_truth"],)

        controlled_dataset.append({
            "id": item["id"],
            "question": rag_result.question,
            "answer": rag_result.answer,
            "contexts": rag_result.contexts,
            "ground_truth": rag_result.ground_truth,
            "metadata": {
                **item,
                "prototype": "prototype_ameliore_hybride",
                "route": retrieved["route"],
                "sql_result": retrieved["sql_result"],
                "rag_chunks": retrieved["chunks_details"],
                "source_errors": retrieved["source_errors"],
                "pydantic_ai_control": {
                    "grounded_in_context": (
                        controlled_answer.grounded_in_context
                    ),
                    "unsupported_claims": (
                        controlled_answer.unsupported_claims
                    ),
                    "raw_answer": raw_answer,
                },
            },
        })

    return controlled_dataset


# 11. LLM & EMBEDDINGS RAGAS
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

# Evaluation RAGAS
def run_ragas_evaluation(dataset):
    ragas_dataset = Dataset.from_dict({
        "question": [
            row["question"]
            for row in dataset
        ],
        "answer": [
            row["answer"]
            for row in dataset
        ],
        "contexts": [
            row["contexts"]
            for row in dataset
        ],
        "ground_truth": [
            row["ground_truth"]
            for row in dataset
        ],
    })

    evaluator_llm = MistralRagasLLM(
        run_config=RunConfig()
    )

    evaluator_embeddings = MistralRagasEmbeddings()

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
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    return result

# Export des résultats
def json_safe(value):
    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(val)
            for val in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(data),
            file,
            indent=4,
            ensure_ascii=False,
        )


def export_results(
    result,
    dataset,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = result.to_pandas()

    if len(result_df) != len(dataset):
        raise ValueError(
            f"RAGAS a produit {len(result_df)} scores "
            f"pour {len(dataset)} réponses."
        )

    metrics = [
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_similarity",
        "context_entity_recall",
    ]

    # Scores individuels
    scores_df = result_df.copy()

    question_ids = [row["id"] for row in dataset]
    if "question_id" in scores_df.columns:
        scores_df["question_id"] = question_ids
    else:
        scores_df.insert(0, "question_id", question_ids)

    # Ajouter ou mettre à jour le texte de la question
    questions = [row["question"] for row in dataset]
    if "question" in scores_df.columns:
        scores_df["question"] = questions
    else:
        scores_df.insert(1, "question", questions)

    # Placer les colonnes d'identification en premier
    scores_df = scores_df[
        ["question_id", "question"]
        + [
            col for col in scores_df.columns
            if col not in ["question_id", "question"]]]

    scores_df.to_csv(
        output_dir / "scores_par_question.csv",
        index=False,
        encoding="utf-8-sig",)
    
    save_json(
        output_dir / "scores_par_question.json",
        scores_df.to_dict(orient="records"),)

    # Moyennes des métriques
    means = {}

    for metric in metrics:
        if metric in scores_df.columns:
            values = pd.to_numeric(
                scores_df[metric],
                errors="coerce",
            )

            means[metric] = (
                float(values.mean())
                if values.notna().any()
                else None
            )
        else:
            means[metric] = None

    save_json(
        output_dir / "scores_moyens.json",
        {
            "evaluation": (
                "prototype_ameliore_apres_pydantic_ai"
            ),
            "nombre_questions": len(dataset),
            "scores_moyens": means,
        },
    )

    with (
        output_dir / "scores_moyens.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "score"])

        for metric, score in means.items():
            writer.writerow([metric, score])

    # Réponses, contextes, ground truth et métadonnées
    save_json(
        output_dir / "resultats_detailles.json",
        dataset,
    )

    return means

# Métriques routage
def infer_expected_route(metadata):
    explicit_route = str(
        metadata.get("expected_route") or "").strip().upper()

    if explicit_route in {"SQL", "RAG", "BOTH"}:
        return explicit_route

    source = str(metadata.get("source_expected") or "").strip().lower()
    question_type = str(metadata.get("type") or "").strip().lower()

    if ("reddit" in source or "qualitative" in question_type):
        return "RAG"

    if any(
        value in question_type
        for value in ("mixte", "mixed", "hybrid")):
        return "BOTH"

    if source in {
        "données nba",
        "donnees nba",
        "analyse",}:
        return "SQL"
    return None


def export_routing_metrics(
    dataset,
    output_dir,):
    rows = []

    for item in dataset:
        metadata = item["metadata"]
        expected = infer_expected_route(metadata)
        selected = metadata.get("route")

        rows.append({
            "id": item["id"],
            "question": item["question"],
            "expected_route": expected,
            "selected_route": selected,
            "route_correct": (
                selected == expected
                if expected is not None
                else None
            ),
            "source_errors": " | ".join(
                metadata.get("source_errors", [])
            ),
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "evaluation_routage.csv",
        index=False,
        encoding="utf-8-sig",)

    labelled = [
        row for row in rows
        if row["expected_route"] in {
            "SQL", "RAG", "BOTH"}]

    correct = sum(
        row["route_correct"] is True
        for row in labelled)

    confusion = {
        expected: {
            selected: sum(
                row["expected_route"] == expected
                and row["selected_route"] == selected
                for row in labelled
            )
            for selected in ("SQL", "RAG", "BOTH")
        }
        for expected in ("SQL", "RAG", "BOTH")
    }

    metrics = {
        "questions_avec_route_attendue": len(labelled),
        "routes_correctes": correct,
        "accuracy_routage": (
            correct / len(labelled)
            if labelled
            else None
        ),
        "matrice_confusion": confusion,
    }

    save_json(
        output_dir / "metriques_routage.json",
        metrics,
    )

    return metrics


# main
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True,)

    logger.info("Chargement du jeu de validation...")
    eval_set = load_evaluation_set()
    logger.info("Chargement du VectorStore...")
    vector_store_manager = load_vectorstore()

    controlled_dataset = []
    errors = []

    # Une seule exécution du prototype amélioré.
    for index, item in enumerate(
        eval_set,
        start=1,):
        logger.info(
            "Question %s/%s : %s",
            index,
            len(eval_set),
            item["id"],)

        try:
            # Génération et contrôle Pydantic AI
            row = build_controlled_dataset(
                [item],
                vector_store_manager,)[0]
            controlled_dataset.append(row)

        except Exception as exc:
            logger.exception(
                "Échec pour la question %s",
                item["id"],)

            errors.append({
                "id": item["id"],
                "question": item.get("question"),
                "error_type": type(exc).__name__,
                "error": str(exc),})

    save_json(OUTPUT_DIR / "erreurs_execution.json", errors,)
    if not controlled_dataset:
        raise RuntimeError("Aucune réponse exploitable. Consultez erreurs_execution.json.")

    # Enregistrement préalable des réponses contrôlées
    save_json(OUTPUT_DIR / "reponses_apres_pydantic.json", controlled_dataset,)
    logger.info("Évaluation RAGAS des réponses après Pydantic AI...")
    ragas_result = run_ragas_evaluation(controlled_dataset)
    means = export_results(ragas_result, controlled_dataset, OUTPUT_DIR,)
    routing_metrics = export_routing_metrics(controlled_dataset, OUTPUT_DIR,)
    
    logger.info("Évaluation terminée : %s", OUTPUT_DIR,)
    logger.info("Scores moyens : %s", means,)
    logger.info("Accuracy du routage : %s", routing_metrics["accuracy_routage"],)
    logger.info("Questions réussies : %s/%s ; échecs : %s", len(controlled_dataset), len(eval_set), len(errors),)