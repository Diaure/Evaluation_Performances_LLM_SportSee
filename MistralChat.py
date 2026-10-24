# MistralChat.py (version RAG)
import streamlit as st
import os
import ast
import re
import logging
from mistralai.client import Mistral
# from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME)
    
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from utils.pydantic_validation import (
    QuestionModel,
    ChunkModel,
    AnswerModel,
    RAGResultModel,
    ControlledAnswerModel,)

from sql_tool import sql_tool

from evaluate_prototypes_pydantic import generate_reponse_prototype, validation_reponse_pydantic_ai, SYSTEM_PROMPT, reconstruire_classement_sql

# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    client = Mistral(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# Confoguration de l'Agent pydantic AI (identique à celui de l'évaluation RAGAS)
mistral_provider = MistralProvider( mistral_client=client)
pydantic_ai_model = MistralModel(MODEL_NAME, provider=mistral_provider,)
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
        "grounded_in_context. Ne produis jamais une réponse vide."),)  

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# --- Prompt Système pour RAG ---
SYSTEM_PROMPT = SYSTEM_PROMPT
# f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
# Ta mission est de répondre aux questions des fans en animant le débat.

# ---
# {{context_str}}
# ---

# QUESTION DU FAN:
# {{question}}

# RÉPONSE DE L'ANALYSTE NBA:"""


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Fonctions ---
def generer_reponse(prompt_messages: list[dict]) -> str: #[ChatMessage]
    """
    Envoie le prompt (qui inclut maintenant le contexte) à l'API Mistral.
    """
    if not prompt_messages:
         logging.warning("Tentative de génération de réponse avec un prompt vide.")
         return "Je ne peux pas traiter une demande vide."
    try:
        logging.info("Début de la synthèse Mistral")
        logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")
        # Log le contenu du prompt (peut être long) - commenter si trop verbeux
        # logging.debug(f"Prompt envoyé à l'API: {prompt_messages}")

        response = client.chat.complete(
            model=model,
            messages=prompt_messages,
            temperature=0.1, # Température basse pour des réponses factuelles basées sur le contexte
        )
        logging.info("Fin de l'appel Mistral")
        if response.choices and len(response.choices) > 0:
            logging.info("Réponse reçue de l'API Mistral.")
            return response.choices[0].message.content
        else:
            logging.warning("L'API n'a pas retourné de choix valide.")
            return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        logging.exception("Erreur API Mistral pendant client.chat")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."

# fonction de routage vers sql, RAG ou les deux selon le type de question
def determiner_source(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Tu routes les questions d'un assistant NBA.\n"
                "SQL: statistiques, classements, calculs, agrégations et comparaisons chiffrées disponibles en base.\n"
                "RAG: avis de fans, commentaires Reddit et analyses qualitatives.\n"
                "BOTH: les deux sources sont nécessaires.\n"
                "Un chiffre dans la question ne suffit pas à choisir SQL.\n"
                "Réponds uniquement par SQL, RAG ou BOTH."),},
        {"role": "user",
        "content": question,},]

    try:
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0,)

        choix = response.choices[0].message.content.strip().upper()

        if choix.startswith("BOTH"):
            return "BOTH"
        if choix.startswith("SQL"):
            return "SQL"
        if choix.startswith("RAG"):
            return "RAG"

        logging.warning("Routage non reconnu : %s", choix)
        return "RAG"

    except Exception:
        logging.exception("Erreur lors de la détermination de la source.")
        raise

# def formater_classement_sql(question: str, resultat_sql: str):
#     """
#     Formate un classement à partir des résultats SQL, pour les équipes
#     comme pour les joueurs.

#     - Détecte l'entité, la statistique et le nombre de places demandées.
#     - Trie les résultats par ordre décroissant.
#     - Conserve tous les ex aequo à la dernière place demandée.
#     - Attribue le même rang aux valeurs égales.
#     - Retourne None si la question ou les données ne sont pas reconnues.
#     """

#     question_min = question.lower()

#     # Détecter l'entité concernée.
#     entite_equipe = bool(re.search(
#         r"\b(équipe|équipes|team|teams)\b",
#         question_min,
#     ))

#     entite_joueur = bool(re.search(
#         r"\b(joueur|joueurs|player|players)\b",
#         question_min,
#     ))

#     # La fonction ne traite que les classements d'équipes ou de joueurs.
#     if not entite_equipe and not entite_joueur:
#         return None

#     # Si les deux entités sont mentionnées, ne pas deviner laquelle classer.
#     if entite_equipe and entite_joueur:
#         return None

#     # Vérifier qu'il s'agit bien d'une demande de classement.
#     termes_classement = (
#         "plus de", "plus d'", "plus d’", "le plus", "la plus",
#         "les plus", "top", "classement", "meilleur", "meilleure",
#         "meilleurs", "meilleures", "maximum", "maximal",
#     )

#     if not any(terme in question_min for terme in termes_classement):
#         return None

#     # Déterminer le nombre de places demandées.
#     nombres = {
#         "un": 1,
#         "une": 1,
#         "deux": 2,
#         "trois": 3,
#         "quatre": 4,
#         "cinq": 5,
#         "six": 6,
#         "sept": 7,
#         "huit": 8,
#         "neuf": 9,
#         "dix": 10,
#         "onze": 11,
#         "douze": 12,
#         "quinze": 15,
#         "vingt": 20,
#     }

#     match_nombre = re.search(r"\b(\d+)\b", question_min)
#     n = int(match_nombre.group(1)) if match_nombre else None

#     if n is None:
#         for mot, valeur in nombres.items():
#             if re.search(rf"\b{mot}\b", question_min):
#                 n = valeur
#                 break

#     if n is None or n < 1:
#         return None

#     # Identifier la statistique demandée.
#     correspondances = {
#         "interception": (
#             "total_interceptions",
#             "interceptions",
#         ),
#         "rebond": (
#             "total_rebonds",
#             "rebonds",
#         ),
#         "passe décisive": (
#             "total_passes",
#             "passes décisives",
#         ),
#         "passe": (
#             "total_passes",
#             "passes",
#         ),
#         "point": (
#             "total_points",
#             "points",
#         ),
#         "contre": (
#             "total_contres",
#             "contres",
#         ),
#     }

#     champ_stat = None

#     # Tester les expressions les plus précises en premier.
#     for terme in sorted(correspondances, key=len, reverse=True):
#         if re.search(rf"\b{re.escape(terme)}", question_min):
#             champ_stat = correspondances[terme][0]
#             break

#     if champ_stat is None:
#         return None

#     # Extraire les dictionnaires présents dans le résultat SQL.
#     if isinstance(resultat_sql, list):
#         rows = [row for row in resultat_sql if isinstance(row, dict)]
#     else:
#         lignes_sql = re.findall(r"\{[^{}]*\}", str(resultat_sql or ""))
#         rows = []

#         for ligne in lignes_sql:
#             try:
#                 row = ast.literal_eval(ligne)
#                 if isinstance(row, dict):
#                     rows.append(row)
#             except (ValueError, SyntaxError):
#                 continue

#     if not rows:
#         return None

#     # Déterminer les colonnes de nom selon l'entité.
#     champs_nom_possibles = (["nom_complet_equipe"] if entite_equipe else ["nom_du_joueur"])
#     champ_nom = next(
#         (
#             champ
#             for champ in champs_nom_possibles
#             if all(champ in row for row in rows)
#         ), None,)

#     if champ_nom is None:
#         return None

#     # Vérifier que chaque ligne contient une statistique numérique exploitable.
#     if not all(
#         champ_stat in row
#         and isinstance(row[champ_stat], (int, float))
#         and not isinstance(row[champ_stat], bool)
#         for row in rows
#     ):
#         return None

#     # Trier par ordre décroissant.
#     rows.sort(key=lambda row: row[champ_stat], reverse=True)

#     # Ne pas afficher moins de N résultats si le SQL en fournit moins.
#     # Conserver tous les ex aequo au seuil de la dernière place demandée.
#     seuil = rows[min(n, len(rows)) - 1][champ_stat]
#     selection = [
#         row for row in rows
#         if row[champ_stat] >= seuil
#     ]

#     # Construire le classement avec des rangs partagés.
#     resultat = []
#     valeur_precedente = None
#     rang_precedent = None

#     for position, row in enumerate(selection, start=1):
#         valeur = row[champ_stat]

#         if position == 1 or valeur != valeur_precedente:
#             rang = position
#         else:
#             rang = rang_precedent

#         resultat.append(
#             f"{rang}. **{row[champ_nom]}** — {valeur:g}"
#         )

#         valeur_precedente = valeur
#         rang_precedent = rang

#     return "\n".join(resultat)


# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Début de la logique RAG ===
    # Vérifier si le Vector Store est disponible
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        # On arrête ici car on ne peut pas faire de RAG
        st.stop()

    # Rechercher le contexte dans le Vector Store
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Analyse de la question...")

    try:
        source = determiner_source(prompt)
        logging.info("Source sélectionnée : %s", source)

        contextes = []

        # SQL: appel du tool LangChain existant
        if source in ("SQL", "BOTH"):
            message_placeholder.write("Interrogation de la base SQL...")
            logging.info("Début de l'appel sql_tool")
            resultat_sql = sql_tool.invoke({"question": prompt})
            logging.info("Fin de l'appel sql_tool") 
            logging.info("Résultat SQL : %s", resultat_sql)
            contextes.append("RÉSULTATS SQL :\n" + str(resultat_sql))

        # RAG: recherche dans les chunks indexés
        if source in ("RAG", "BOTH"):
            message_placeholder.write("Recherche dans les documents...")
            search_results = vector_store_manager.search(prompt, k=SEARCH_K, )
            if search_results:
                contexte_rag = "\n\n---\n\n".join(
                    f"Source : {res.get('metadata', {}).get('source', 'Inconnue')}\n"
                    f"Contenu : {res.get('text', '')}"
                    for res in search_results)
                contextes.append("CONTEXTE REDDIT :\n" + contexte_rag)
            else:
                contextes.append("CONTEXTE REDDIT : aucun résultat pertinent trouvé.")

        # Synthèse finale
        contexte_final = "\n\n".join(contextes)

        prompt_synthese = f"""
Tu es NBA Analyst AI, un assistant spécialisé dans la NBA.

Réponds à la question en utilisant uniquement les informations ci-dessous.

Règles :
- Les statistiques doivent provenir des résultats SQL.
- Les opinions doivent provenir des commentaires retrouvés.
- Pour une question mixte, distingue les faits des opinions.
- N'invente aucune statistique ni aucun commentaire.
- Si les informations sont insuffisantes, indique-le.
- Rédige une réponse claire et synthétique.

QUESTION :
{prompt}

INFORMATIONS :
{contexte_final}

RÉPONSE :
"""
        message_placeholder.write("Synthèse en cours...")

        # Validation de la question
        question_validee = QuestionModel(question=prompt)
        
        # Génération brute par Mistral
        raw_answer = generate_reponse_prototype(
            question=question_validee.question,
            context_str=contexte_final,)
        
        # Validation structurelle de la réponse brute
        AnswerModel(
            question=question_validee.question,
            context=contexte_final,
            answer=raw_answer,)
        
        # Contrôle sémantique par Pydantic AI
        controlled_answer = validation_reponse_pydantic_ai(
            question=question_validee.question,
            context_str=contexte_final,
            raw_answer=raw_answer, )
        
        # Réponse contrôlée
        response_content = controlled_answer.answer.strip()
        
        # Conserver le traitement spécifique des classements déjà présent dans MistralChat.py
        if source in ("SQL", "BOTH"):
            ranking_answer = reconstruire_classement_sql(
                prompt,
                resultat_sql,)
        
            if ranking_answer is not None:
                response_content = ranking_answer
        
        # Validation structurelle de la réponse finale
        AnswerModel(
            question=question_validee.question,
            context=contexte_final,
            answer=response_content,)
        
        # Affichage de la réponse finale
        message_placeholder.markdown(response_content)
        
        # response_content = None
        # if source in ("SQL", "BOTH"):
        #     response_content = formater_classement_sql(
        #         question=prompt,
        #         resultat_sql=str(resultat_sql),)

        # if response_content is None:    
        #     response_content = generer_reponse([{"role":"user", "content":prompt_synthese}])  
        
        # message_placeholder.markdown(response_content)   
    except Exception as e:
        logging.exception("Erreur pendant le traitement de la question.")
        response_content = (
                "Une erreur technique empêche le traitement de la question. "
                f"Détail : {e}")

        message_placeholder.write(response_content)

    # Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")