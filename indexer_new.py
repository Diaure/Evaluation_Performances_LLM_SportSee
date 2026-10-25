# indexer.py
import argparse
import logging
from typing import Optional

from pathlib import Path

from utils.config import (
    INPUT_DIR,
    FAISS_INDEX_FILE,
    DOCUMENT_CHUNKS_FILE,
)

from utils.config import INPUT_DIR
from utils.data_loader_reviewed import load_and_parse_files
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# chemins des nouveaux fichiers
DEFAULT_NEW_FAISS_INDEX = str(Path(FAISS_INDEX_FILE).with_name("faiss_index_new.idx"))
DEFAULT_NEW_CHUNKS = str(Path(DOCUMENT_CHUNKS_FILE).with_name("document_chunks_new.pkl"))


def run_indexing(input_directory: str): # , data_url: Optional[str] = None
    """Exécute le processus complet d'indexation."""
    logging.info("--- Démarrage du processus d'indexation ---")

    logging.info("Chargement et parsing des fichiers depuis : %s", input_directory,)

    documents = load_and_parse_files(input_directory)

    if not documents:
        logging.warning("Aucun document n'a été chargé ou parsé. Vérifiez le contenu du dossier d'entrée.")
        logging.info("--- Processus d'indexation terminé (aucun document traité) ---")
        return

    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager(
        faiss_index_file=DEFAULT_NEW_FAISS_INDEX,
        document_chunks_file=DEFAULT_NEW_CHUNKS,) # Le constructeur ne fait que charger s'il existe

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: %s", len(documents))

    if vector_store.index is not None:
        logging.info(f"Nombre de chunks indexés: %s", vector_store.index.ntotal)
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL optionnelle pour télécharger et extraire un fichier inputs.zip"
    )
    args = parser.parse_args()

    final_data_url = args.data_url

    run_indexing(input_directory=args.input_dir) # , data_url=final_data_url