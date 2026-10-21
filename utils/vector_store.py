# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import Any, List, Dict, Tuple, Optional
from httpx import HTTPError
from mistralai.client import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Utilisé pour le format attendu par le splitter

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(
        self,
        faiss_index_file: Optional[str] = None,
        document_chunks_file: Optional[str] = None,
    ):
        self.faiss_index_file = faiss_index_file or FAISS_INDEX_FILE
        self.document_chunks_file = (
            document_chunks_file or DOCUMENT_CHUNKS_FILE
        )

        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, Any]] = []

        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)

        self._load_index_and_chunks()

        # Vérifier la cohérence uniquement si un index a été chargé.
        if self.index is not None:
            if self.index.ntotal != len(self.document_chunks):
                raise ValueError(
                    f"Index incohérent : {self.index.ntotal} vecteurs "
                    f"pour {len(self.document_chunks)} chunks."
                )

    
    def _load_index_and_chunks(self):
        """Charge l'index FAISS et les chunks si les fichiers existent."""

        if (
            os.path.exists(self.faiss_index_file)
            and os.path.exists(self.document_chunks_file)
        ):
            try:
                logging.info(
                    f"Chargement de l'index FAISS depuis "
                    f"{self.faiss_index_file}..."
                )

                self.index = faiss.read_index(self.faiss_index_file)

                logging.info(
                    f"Chargement des chunks depuis "
                    f"{self.document_chunks_file}..."
                )

                with open(self.document_chunks_file, "rb") as f:
                    self.document_chunks = pickle.load(f)

                # Vérifier la cohérence entre les vecteurs et les chunks.
                if self.index.ntotal != len(self.document_chunks):
                    raise ValueError(
                        f"Index incohérent : {self.index.ntotal} vecteurs "
                        f"pour {len(self.document_chunks)} chunks."
                    )

                logging.info(
                    f"Index ({self.index.ntotal} vecteurs) et "
                    f"{len(self.document_chunks)} chunks chargés."
                )

            except Exception:
                logging.exception(
                    "Erreur lors du chargement de l'index et des chunks."
                )
                self.index = None
                self.document_chunks = []

        else:
            logging.info(
                "Aucun index existant trouvé aux chemins configurés. "
                "Un nouvel index pourra être créé."
            )

    def _split_documents_to_chunks(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Découpe les documents en chunks avec métadonnées."""
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len, # Important: mesure en caractères
            add_start_index=True, # Ajoute la position de début du chunk dans le document original
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            # Convertit notre format de document en format Langchain Document pour le splitter
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc['metadata'].get('filename', 'N/A')}' découpé en {len(chunks)} chunks.")

            # Enrichit chaque chunk avec des métadonnées supplémentaires
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": f"{doc_counter}_{i}", # Identifiant unique du chunk (doc_index_chunk_index)
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata, # Métadonnées héritées du document (source, category, etc.)
                        "chunk_id_in_doc": i, # Position du chunk dans son document d'origine
                        "start_index": chunk.metadata.get("start_index", -1) # Position de début (en caractères)
                    }
                }
                all_chunks.append(chunk_dict)
            doc_counter += 1

        logging.info(f"Total de {len(all_chunks)} chunks créés.")
        return all_chunks


    def _generate_embeddings(
            self,
            chunks: List[Dict[str, Any]]
        ) -> Optional[np.ndarray]:
            """Génère les embeddings Mistral pour les chunks."""

            if not MISTRAL_API_KEY:
                logging.error("MISTRAL_API_KEY manquante.")
                return None

            if not chunks:
                logging.warning("Aucun chunk à vectoriser.")
                return None

            if EMBEDDING_BATCH_SIZE <= 0:
                logging.error("EMBEDDING_BATCH_SIZE doit être positif.")
                return None

            all_embeddings = []
            total_batches = (
                len(chunks) + EMBEDDING_BATCH_SIZE - 1
            ) // EMBEDDING_BATCH_SIZE

            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch_num = i // EMBEDDING_BATCH_SIZE + 1
                batch = chunks[i:i + EMBEDDING_BATCH_SIZE]

                try:
                    response = self.mistral_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        inputs=[chunk["text"] for chunk in batch],
                    )

                    embeddings = [
                        item.embedding for item in response.data
                    ]

                    if len(embeddings) != len(batch):
                        logging.error(
                            f"Lot {batch_num} : nombre d'embeddings incorrect."
                        )
                        return None

                    batch_array = np.asarray(embeddings, dtype=np.float32)

                    if batch_array.ndim != 2 or batch_array.shape[1] == 0:
                        logging.error(
                            f"Lot {batch_num} : dimensions invalides."
                        )
                        return None

                    if not np.isfinite(batch_array).all():
                        logging.error(
                            f"Lot {batch_num} : valeurs non finies détectées."
                        )
                        return None

                    if all_embeddings and (
                        batch_array.shape[1] != len(all_embeddings[0])
                    ):
                        logging.error("Dimensions d'embeddings incohérentes.")
                        return None

                    all_embeddings.extend(batch_array)

                    logging.info(
                        f"Lot {batch_num}/{total_batches} traité."
                    )

                except Exception:
                    logging.exception(
                        f"Échec du lot {batch_num}/{total_batches}. "
                        "La génération est interrompue."
                    )
                    return None

            result = np.asarray(all_embeddings, dtype=np.float32)

            if result.shape[0] != len(chunks):
                logging.error("Nombre final d'embeddings incorrect.")
                return None

            logging.info(f"Embeddings générés : {result.shape}")
            return result
    
    
    def build_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Construit et sauvegarde un nouvel index FAISS."""

        if not documents:
            logging.warning("Aucun document fourni.")
            return False

        new_chunks = self._split_documents_to_chunks(documents)

        if not new_chunks:
            logging.error("Aucun chunk créé.")
            return False

        embeddings = self._generate_embeddings(new_chunks)

        if (
            embeddings is None
            or embeddings.shape[0] != len(new_chunks)
        ):
            logging.error("Échec des embeddings. Indexation annulée.")
            return False

        faiss.normalize_L2(embeddings)

        new_index = faiss.IndexFlatIP(embeddings.shape[1])
        new_index.add(embeddings)

        if new_index.ntotal != len(new_chunks):
            logging.error("Incohérence entre index et chunks.")
            return False

        # Sauvegarder les fichiers de la nouvelle version.
        ancien_index = self.index
        anciens_chunks = self.document_chunks

        self.index = new_index
        self.document_chunks = new_chunks

        if not self._save_index_and_chunks():
            logging.error(
                "Échec de la sauvegarde du nouvel index."
            )
            self.index = ancien_index
            self.document_chunks = anciens_chunks
            return False

        logging.info(
            f"Nouvel index créé : {self.index.ntotal} vecteurs.")

        return True

    
    def _save_index_and_chunks(self) -> bool:
        """Sauvegarde l'index FAISS et ses chunks."""

        if self.index is None or not self.document_chunks:
            logging.error("Index ou chunks vides : sauvegarde impossible.")
            return False

        index_dir = os.path.dirname(self.faiss_index_file) or "."
        chunks_dir = os.path.dirname(self.document_chunks_file) or "."

        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(chunks_dir, exist_ok=True)

        try:
            faiss.write_index(self.index, self.faiss_index_file)

            with open(self.document_chunks_file, "wb") as f:
                pickle.dump(self.document_chunks, f)

            logging.info("Index et chunks sauvegardés.")
            return True

        except Exception:
            logging.exception("Erreur pendant la sauvegarde.")
            return False

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des chunks pertinents avec leurs scores
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
             logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
             return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text] # La requête doit être une liste
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            # Normaliser l'embedding de la requête pour la similarité cosinus
            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            # Pour IndexFlatIP: scores = produit scalaire (plus grand = meilleur)
            # indices: index des chunks correspondants dans self.document_chunks
            # Demander plus de résultats si un score minimum est spécifié
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats
            results = []
            if indices.size > 0: # Vérifier s'il y a des résultats
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks): # Vérifier la validité de l'index
                        chunk = self.document_chunks[idx]
                        # Convertir le score en similarité (0-1)
                        # Pour IndexFlatIP avec vecteurs normalisés, le score est déjà entre -1 et 1
                        # On le convertit en pourcentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filtrer les résultats en fonction du score minimum
                        # Le min_score est entre 0 et 1, mais similarity est en pourcentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity, # Score de similarité en pourcentage
                            "raw_score": raw_score, # Score brut pour débogage
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contient source, category, chunk_id_in_doc, start_index etc.
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except HTTPError as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []