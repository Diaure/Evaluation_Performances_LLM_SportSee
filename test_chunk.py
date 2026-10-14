import pickle

# Charger le fichier
with open("vector_db/document_chunks_new.pkl", "rb") as f:
    chunks = pickle.load(f)

# Vérifier le nombre total
print(f"Nombre total de chunks : {len(chunks)}")

# Afficher les 10 derniers
for i, chunk in enumerate(chunks[-10:], start=len(chunks)-9):
    print(f"\nChunk {i}:")
    print(chunk)

from utils.data_loader_reviewed import extraction_pdf

texte = extraction_pdf("inputs_test/Reddit 1.pdf")

print(texte)


# from pathlib import Path
# import pickle
# import re
# from collections import Counter

# FICHIER_CHUNKS = Path("vector_db/document_chunks_new.pkl")

# if not FICHIER_CHUNKS.exists():
#     raise FileNotFoundError(
#         f"Fichier introuvable : {FICHIER_CHUNKS.resolve()}"
#     )

# with FICHIER_CHUNKS.open("rb") as fichier:
#     chunks = pickle.load(fichier)

# if not isinstance(chunks, (list, tuple)):
#     raise TypeError(
#         f"Format inattendu : {type(chunks).__name__}"
#     )

# print(f"Nombre de chunks : {len(chunks)}")

# longueurs = []
# bruit = Counter()
# exemples_bruit = {}

# motifs_bruit = {
#     "date_et_titre_repetes": r"12/06/2025.*Who are teams in the playoffs",
#     "url": r"https?://|https?[:/]{1,}",
#     "reponses_supplementaires": r"\d+\s+réponses? supplémentaires",
#     "interface_reddit": r"(?i)^(partager|trier par meilleurs|afficher plus de commentaires)$",
#     "suggestions_connexes": r"(?i)discussions connexes|what team do you absolutely not want",
# }

# for numero, chunk in enumerate(chunks, start=1):
#     if isinstance(chunk, dict):
#         texte = chunk.get("page_content", "")
#         metadata = chunk.get("metadata", {})
#     else:
#         texte = getattr(chunk, "page_content", "")
#         metadata = getattr(chunk, "metadata", {})

#     if not isinstance(texte, str):
#         texte = ""

#     longueurs.append(len(texte))

#     for nom, motif in motifs_bruit.items():
#         if re.search(motif, texte):
#             bruit[nom] += 1
#             exemples_bruit.setdefault(nom, (numero, texte[:350]))

#     if numero <= 3:
#         print(f"\n--- CHUNK {numero} ---")
#         print(f"Longueur : {len(texte)} caractères")
#         print(f"Source : {metadata.get('source', 'inconnue')}")
#         print(texte[:700])

# print("\n--- BILAN ---")
# print(f"Chunks non vides : {sum(n > 0 for n in longueurs)} / {len(chunks)}")

# if longueurs:
#     print(f"Longueur minimale : {min(longueurs)} caractères")
#     print(f"Longueur maximale : {max(longueurs)} caractères")
#     print(f"Longueur moyenne : {sum(longueurs) // len(longueurs)} caractères")

# print("\nOccurrences de bruit par chunk :")
# for nom, nombre in bruit.items():
#     print(f"- {nom} : {nombre}")
#     numero, extrait = exemples_bruit[nom]
#     print(f"  Exemple chunk {numero} : {extrait!r}")

# if not chunks:
#     print("ATTENTION : aucun chunk dans le fichier.")
# elif not any(longueurs):
#     print("ATTENTION : tous les chunks sont vides.")
# else:
#     print("\nContrôle terminé.")



# import pickle
# from pathlib import Path

# fichier = Path("vector_db/document_chunks_new.pkl")

# with fichier.open("rb") as f:
#     chunks = pickle.load(f)

# questions = [
# ("Orlando Magic", ["orlando", "magic"]),
# ("Julius Randle", ["randle"]),
# ("Tirs à trois points", ["three", "shooting", "3pt"]),
# ("Blessures", ["injury", "injuries", "health"]),
# ]

# print(f"Nombre de chunks chargés : {len(chunks)}")

# for nom, mots_cles in questions:
#     print("\n" + "=" * 70)
#     print(f"RECHERCHE : {nom}")
#     print("=" * 70)


# resultats = []

# for chunk in chunks:
#     texte = chunk.get("text", "")
#     texte_minuscule = texte.lower()

#     occurrences = sum(
#         texte_minuscule.count(mot.lower())
#         for mot in mots_cles
#     )

#     if occurrences > 0:
#         resultats.append((occurrences, chunk))

# resultats.sort(key=lambda element: element[0], reverse=True)

# if not resultats:
#     print("Aucun résultat.")

# for score, chunk in resultats[:3]:
#     print(f"\nID : {chunk.get('id')}")
#     print(f"Occurrences des mots-clés : {score}")
#     print(chunk.get("text", "")[:500])

