# MistralChat.py (version RAG)
import streamlit as st
import os
import logging
from mistralai.client import Mistral
# from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

from sql_tool import sql_tool

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
# Adaptez ce prompt selon vos besoins
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


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
    # 2. Vérifier si le Vector Store est disponible
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        # logging.error("VectorStoreManager non disponible pour la recherche.")
        # On arrête ici car on ne peut pas faire de RAG
        st.stop()

    # 3. Rechercher le contexte dans le Vector Store
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
        response_content = generer_reponse([{"role":"user", "content":prompt_synthese}])  
        message_placeholder.markdown(response_content)   
        # logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
        # search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        # logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
    except Exception as e:
        logging.exception("Erreur pendant le traitement de la question.")
        response_content = (
                "Une erreur technique empêche le traitement de la question. "
                f"Détail : {e}")

        message_placeholder.write(response_content)
        # st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
        # logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
        # search_results = [] # On continue sans contexte si la recherche échoue

    # 4. Formater le contexte pour le prompt LLM
    # context_str = "\n\n---\n\n".join([
    #     f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
    #     for res in search_results
    # ])

    # if not search_results:
    #     context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
    #     logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

    # 5. Construire le prompt final pour l'API Mistral en utilisant le System Prompt RAG
    # final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

    # Créer la liste de messages pour l'API (juste le prompt système/utilisateur combiné)
    # messages_for_api = [
    #     # On pourrait séparer system et user, mais Mistral gère bien un long message user structuré
    #     ChatMessage(role="user", content=final_prompt_for_llm)
    # ]

    # === Fin de la logique RAG ===


    # 6. Afficher indicateur + Générer la réponse de l'assistant via LLM
    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt augmenté
        # response_content = generer_reponse(messages_for_api)

        # Affichage de la réponse complète
        # message_placeholder.write(response_content)

    # 7. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")