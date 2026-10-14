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

    # def __init__(self):
    #     self.index: Optional[faiss.Index] = None
    #     self.document_chunks: List[Dict[str, any]] = []
    #     self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    #     self._load_index_and_chunks()

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

    # def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
    #     """Génère les embeddings pour une liste de chunks via l'API Mistral."""
    #     if not MISTRAL_API_KEY:
    #         logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
    #         return None
    #     if not chunks:
    #         logging.warning("Aucun chunk fourni pour générer les embeddings.")
    #         return None

    #     logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
    #     all_embeddings = []
    #     total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    #     for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
    #         batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
    #         batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
    #         texts_to_embed = [chunk["text"] for chunk in batch_chunks]

    #         logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
    #         try:
    #             response = self.mistral_client.embeddings.create(
    #                 model=EMBEDDING_MODEL,
    #                 input=texts_to_embed
    #             )
    #             batch_embeddings = [data.embedding for data in response.data]
    #             all_embeddings.extend(batch_embeddings)
    #         except HTTPError as e:
    #             logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
    #             logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
    #         except Exception as e:
    #             logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
    #              # Gérer l'erreur: ici on ajoute des vecteurs nuls pour ne pas bloquer
    #             num_failed = len(texts_to_embed)
    #             if all_embeddings: # Si on a déjà des embeddings, on prend la dimension du premier
    #                 dim = len(all_embeddings[0])
    #             else: # Sinon, on ne peut pas déterminer la dimension, on saute ce lot
    #                  logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
    #                  continue
    #             logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
    #             all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)

    #         except Exception as e:
    #             logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
    #             # Gérer comme ci-dessus
    #             num_failed = len(texts_to_embed)
    #             if all_embeddings:
    #                 dim = len(all_embeddings[0])
    #             else:
    #                  logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
    #                  continue
    #             logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
    #             all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)


    #     if not all_embeddings:
    #          logging.error("Aucun embedding n'a pu être généré.")
    #          return None

    #     embeddings_array = np.array(all_embeddings).astype('float32')
    #     logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
    #     return embeddings_array

    
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

    # def build_index(self, documents: List[Dict[str, any]]):
    #     """Construit l'index Faiss à partir des documents."""
    #     if not documents:
    #         logging.warning("Aucun document fourni pour construire l'index.")
    #         return

    #     # 1. Découper en chunks
    #     self.document_chunks = self._split_documents_to_chunks(documents)
    #     if not self.document_chunks:
    #         logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
    #         return

    #     # 2. Générer les embeddings
    #     embeddings = self._generate_embeddings(self.document_chunks)
    #     if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
    #         logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
    #         # Nettoyer pour éviter un état incohérent
    #         self.document_chunks = []
    #         self.index = None
    #         # Supprimer les fichiers potentiellement corrompus
    #         if os.path.exists(self.self.FAISS_INDEX_FILE): os.remove(self.self.FAISS_INDEX_FILE)
    #         if os.path.exists(self.self.DOCUMENT_CHUNKS_FILE): os.remove(self.self.DOCUMENT_CHUNKS_FILE)
    #         return


    #     # 3. Créer l'index Faiss optimisé pour la similarité cosinus
    #     dimension = embeddings.shape[1]
    #     logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

    #     # Normaliser les embeddings pour la similarité cosinus
    #     faiss.normalize_L2(embeddings)

    #     # Créer un index pour la similarité cosinus (IndexFlatIP = produit scalaire)
    #     self.index = faiss.IndexFlatIP(dimension)
    #     self.index.add(embeddings)
    #     logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

    #     # 4. Sauvegarder l'index et les chunks
    #     self._save_index_and_chunks()

    
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

        # try:
        #     self._save_index_and_chunks()

        #     # Vérifier que les deux fichiers existent.
        #     if not (
        #         os.path.isfile(self.faiss_index_file)
        #         and os.path.isfile(self.document_chunks_file)
        #     ):
        #         raise OSError("Un ou plusieurs fichiers n'ont pas été créés.")

        if not self._save_index_and_chunks():
            logging.error(
                "Échec de la sauvegarde du nouvel index."
            )
            self.index = ancien_index
            self.document_chunks = anciens_chunks
            return False

        logging.info(
            f"Nouvel index créé : {self.index.ntotal} vecteurs.")

        # except Exception:
        #     logging.exception("Échec de la sauvegarde du nouvel index.")
        #     self.index = ancien_index
        #     self.document_chunks = anciens_chunks
        #     return False

        # logging.info(
        #     f"Nouvel index créé : {self.index.ntotal} vecteurs."
        # )
        return True

    # def _save_index_and_chunks(self):
    #     """Sauvegarde l'index Faiss et la liste des chunks."""
    #     if self.index is None or not self.document_chunks:
    #         logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
    #         return

    #     os.makedirs(os.path.dirname(self.FAISS_INDEX_FILE), exist_ok=True)
    #     os.makedirs(os.path.dirname(self.DOCUMENT_CHUNKS_FILE), exist_ok=True)

    #     try:
    #         logging.info(f"Sauvegarde de l'index Faiss dans {self.FAISS_INDEX_FILE}...")
    #         faiss.write_index(self.index, self.FAISS_INDEX_FILE)
    #         logging.info(f"Sauvegarde des chunks dans {self.DOCUMENT_CHUNKS_FILE}...")
    #         with open(self.DOCUMENT_CHUNKS_FILE, 'wb') as f:
    #             pickle.dump(self.document_chunks, f)
    #         logging.info("Index et chunks sauvegardés avec succès.")
    #     except Exception as e:
    #         logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    
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