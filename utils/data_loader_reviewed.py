# utils/data_loader.py
import os
import re
import requests
import zipfile
import io
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
import logging
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Extraction PDF > OCR
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import easyocr
except ImportError:
    easyocr = None


# Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")


# Connexion à PostgreSQL
def recuperer_database_engine():
    load_dotenv()

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    required_variables = {
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
        "DB_HOST": db_host,
        "DB_PORT": db_port,
        "DB_NAME": db_name,}
    
    missing = [name for name, value in required_variables.items() if not value]
    if missing:
        raise ValueError(
            "Variables PostgreSQL manquantes dans le fichier .env : "
            + ", ".join(missing))

    database_url = URL.create(
        drivername="postgresql+psycopg2",
        username=db_user,
        password=db_password,
        host=db_host,
        port=int(db_port),
        database=db_name,)
    
    return create_engine(database_url)

# chargement automatique des noms d'équipes et joueurs depuis la base postgresql
def chargement_entites_alias(engine):
    with engine.connect() as connection:
        team_rows = connection.execute(text("""
            SELECT code_equipe, nom_complet_equipe
            FROM teams
            WHERE code_equipe IS NOT NULL
        """)).mappings().all()

        player_rows = connection.execute(text("""
            SELECT nom_du_joueur
            FROM players
            WHERE nom_du_joueur IS NOT NULL
        """)).mappings().all()

    team_aliases = {}
    for row in team_rows:
        code = str(row["code_equipe"]).strip()
        full_name = row["nom_complet_equipe"]
        full_name = str(full_name).strip() if full_name else code
        
        team_aliases[full_name] = [code, full_name]

    player_aliases = {}
    for row in player_rows:
        player_name = str(row["nom_du_joueur"]).strip()
        if player_name:
            player_aliases[player_name] = []

    logging.info("Référentiels PostgreSQL chargés : %s équipes, %s joueurs", len(team_aliases), len(player_aliases),)

    print("Première ligne PostgreSQL :", dict(team_rows[0]))
    print("Premier alias construit :", list(team_aliases.items())[:1])

    return team_aliases, player_aliases

# type de ichiers pouvant alimenter le RAG
SUPPORTED_RAG_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"}

# fichiers exclus du RAG
EXCLUDED_RAG_EXTENSIONS = {
    ".xlsx",
    ".xls",
    ".csv"}

# Elements pouvant apporter du bruit aux exports REDDI
REDDIT_NOISE_PATTERNS = [
    # Navigation et interface
    r"(?i)^accéder au contenu principal$",
    r"(?i)^se connecter$",
    r"(?i)^créer un compte$",
    r"(?i)^rechercher$",
    r"(?i)^rechercher dans r/\S+$",
    r"(?i)^menu$",
    r"(?i)^ouvrir le menu$",
    r"(?i)^fermer le menu$",
    r"(?i)^accéder à la communauté$",
    r"(?i)^r/nba$",

    # Actions
    r"(?i)^partager$",
    r"(?i)^répondre$",
    r"(?i)^commenter$",
    r"(?i)^rejoindre la conversation$",
    r"(?i)^rejoindre$",
    r"(?i)^en savoir plus$",
    r"(?i)^voir plus$",
    r"(?i)^afficher plus(?: de commentaires)?$",
    r"(?i)^\d+\s+réponses? supplémentaires$",
    r"(?i)^masquer$",
    r"(?i)^ouvrir dans un nouvel onglet$",

    # Publicités et abonnements
    r"(?i)^sponsorisé(?:e)?$",
    r"(?i)^publicité$",
    r"(?i)^abonné(?:e)?$",

    # Tri et recherche de commentaires
    r"(?i)^trier par$",
    r"(?i)^meilleurs commentaires$",
    r"(?i)^meilleurs$",
    r"(?i)^nouveaux commentaires$",
    r"(?i)^plus anciens commentaires$",
    r"(?i)^rechercher des commentaires$",

    # Métadonnées Reddit
    r"(?i)^comm\.\s*du\s*(?:top\s*)?\d+\s*%?$",
    r"(?i)^auteur(?:-rice)?\s+top.*$",

    # Date et heure de publication affichées ensemble
    r"(?i)^\d{2}/\d{2}/\d{4}\s+\d{1,2}[.:]\d{2}$",

    # Ligne ne contenant que des nombres séparés par des espaces
    r"^\d+\s+\d+$",
]

# Normalisation du texte
# Fonction pour normaliser les espaces et retours à la ligne
def _normalisation_espaces(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # Maximum deux retours à la ligne consécutifs
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Déterminer si une ligne constitue du bruit à l'information utile 
def _ligne_avec_bruit(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    # Éléments d'interface connus
    for pattern in REDDIT_NOISE_PATTERNS:
        if re.match(pattern, stripped):
            return True

    # URL complète, même si elle est isolée sur une ligne
    if re.fullmatch(r"(?i)https?://\S+", stripped):
        return True

    # Pagination : 1/15, 2 / 15, etc.
    if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
        return True

    # Séparateurs inutiles
    if re.fullmatch(r"[-_=•·|]+", stripped):
        return True

    return False

# supprimer les éléments d'interface reddit sans filtrer les mots courts
def nettoyer_interface_reddit(texte: str) -> str:
    """Supprime les éléments d'interface Reddit sans filtrer les mots courts."""

    if not texte:
        return ""

    motifs_interface = [
        r"(?i)\baccéder au contenu principal\b",
        r"(?i)\brechercher dans r/nba\b",
        r"(?i)\bse connecter\b",
        r"(?i)\brejoindre la conversation\b",
        r"(?i)\btrier par\b",
        r"(?i)\bmeilleurs commentaires\b",
        r"(?i)\bmeilleurs\b",
        r"(?i)\brechercher des commentaires\b",
        r"(?i)\bpartager\b",
        r"(?i)\brépondre\b",
        r"(?i)\ben savoir plus\b",
        r"(?i)\bsponsorisé(?:e)?\b",
    ]

    lignes_nettoyees = []

    for ligne in texte.splitlines():
        ligne = ligne.strip()

        if not ligne:
            continue

        for motif in motifs_interface:
            ligne = re.sub(motif, " ", ligne)

        ligne = re.sub(r"\s+", " ", ligne).strip()

        if not ligne:
            continue

        # Supprimer les lignes contenant uniquement des chiffres
        # et des symboles de vote, de ponctuation ou de séparation.
        if re.fullmatch(r"[\d\s.,%+\-&|/]+", ligne):
            continue

        # Supprimer les métadonnées de classement Reddit.
        if re.search(
            r"(?i)\bcomm\.\s*du\s*top\s*1\s*%",
            ligne,
        ):
            continue

        # Supprimer les URL et les fragments de pied de page.
        if re.search(r"(?i)https?://|reddit\.com", ligne):
            continue

        lignes_nettoyees.append(ligne)

    return "\n".join(lignes_nettoyees)

# 
def filtrer_entete_reddit(resultats, hauteur_image: int, proportion_entete: float = 0.08):
    seuil_y = hauteur_image * proportion_entete

    return [
        (boite, texte, confiance)
        for boite, texte, confiance in resultats
        if max(point[1] for point in boite) >= seuil_y]

# retirer les éléments d'entête fusionnés par l'ocr
def nettoyer_entete_reddit(texte: str) -> str:
    lignes_nettoyees = []

    for ligne in texte.splitlines():
        ligne = re.sub(
            r"(?i)accéder au contenu principal",
            "",
            ligne,
        )
        ligne = re.sub(
            r"(?i)rechercher dans r/\S+",
            "",
            ligne,
        )
        ligne = re.sub(
            r"(?i)\bse connecter\b",
            "",
            ligne,
        )
        ligne = re.sub(
            r"(?i)\bY[A-Z]\b",
            "",
            ligne,
        )
        ligne = ligne.strip()

        if ligne:
            lignes_nettoyees.append(ligne)

    return "\n".join(lignes_nettoyees)

# reconstruction du texte à partir des éléments détectés par easyocr en utilisant leur coordonnées

def reconstruire_texte_ocr(resultats, tolerance_y: int = 18) -> str:
    detections = []

    for boite, texte, confiance in resultats:
        texte = texte.strip()
        if not texte:
            continue

        y_min = min(point[1] for point in boite)
        y_max = max(point[1] for point in boite)
        x_min = min(point[0] for point in boite)

        detections.append({
            "texte": texte,
            "x": x_min,
            "y_min": y_min,
            "y_max": y_max,
            "centre_y": (y_min + y_max) / 2,
        })

    detections.sort(key=lambda d: (d["centre_y"], d["x"]))

    lignes = []

    for detection in detections:
        ligne_cible = None
        meilleur_ecart = None

        for ligne in lignes:
            y_min_ligne = min(d["y_min"] for d in ligne)
            y_max_ligne = max(d["y_max"] for d in ligne)
            centre_ligne = (
                y_min_ligne + y_max_ligne
            ) / 2

            # Chevauchement vertical entre la détection et la ligne.
            chevauchement = max(
                0,
                min(detection["y_max"], y_max_ligne)
                - max(detection["y_min"], y_min_ligne),
            )

            hauteur_min = min(
                detection["y_max"] - detection["y_min"],
                y_max_ligne - y_min_ligne,
            )

            ecart = abs(detection["centre_y"] - centre_ligne)

            meme_ligne = (
                (hauteur_min > 0 and chevauchement / hauteur_min >= 0.5)
                or ecart <= tolerance_y
            )

            if meme_ligne:
                if meilleur_ecart is None or ecart < meilleur_ecart:
                    ligne_cible = ligne
                    meilleur_ecart = ecart

        if ligne_cible is None:
            lignes.append([detection])
        else:
            ligne_cible.append(detection)

    # Trier les lignes du haut vers le bas, puis les mots de gauche à droite.
    lignes.sort(
        key=lambda ligne: min(d["centre_y"] for d in ligne)
    )

    textes_lignes = []

    for ligne in lignes:
        ligne.sort(key=lambda d: d["x"])
        textes_lignes.append(
            " ".join(d["texte"] for d in ligne)
        )

    return "\n".join(textes_lignes)

# suppression du texte de navigation, boutons, publicité, urls, pagination, interface
# conservation volontaire d'éléments utiles: nombres, %, nom/codes d'équipes joueurs, commentaires, opinions, stats

def nettoyage_texte_reddit(text: str) -> str:
    """Nettoie les éléments d'interface et les pieds de page Reddit."""

    if not text:
        return ""

    text = _normalisation_espaces(text)
    lines = text.splitlines()

    # Supprimer le nom du compte publicitaire avant « Sponsorisé ».
    for i, line in enumerate(lines):
        if re.search(r"(?i)\bsponsorisé(?:e)?\b", line.strip()):
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1

            if j >= 0 and re.fullmatch(
                r"[A-Za-z0-9_.-]{2,40}", lines[j].strip()
            ):
                lines[j] = ""

    cleaned_lines = []
    in_advertisement = False
    in_related_content = False

    ad_end_markers = {
        "rejoindre la conversation",
        "trier par",
        "meilleurs",
        "meilleurs commentaires",
        "nouveaux commentaires",
        "plus anciens commentaires",
        "rechercher des commentaires",
    }

    # Éléments d'interface reconnus, y compris certaines erreurs OCR.
    noise_patterns = [
        r"(?i)^accéder au contenu principal$",
        r"(?i)^se connecter$",
        r"(?i)^créer un compte$",
        r"(?i)^rechercher$",
        r"(?i)^rechercher dans r/\S+$",
        r"(?i)^r/nba$",
        r"(?i)^partager$",
        r"(?i)^répondre$",
        r"(?i)^commenter$",
        r"(?i)^rejoindre(?: la conversation)?$",
        r"(?i)^en savoir plus$",
        r"(?i)^voir plus$",
        r"(?i)^réponses? supplémentaires?$",
        r"(?i)^comm\.?\s*du\s*top(?:\s+\d+\s*%?)?$",
        r"(?i)^auteur(?:-rice)?\s+top.*$",
        r"(?i)^trier par$",
        r"(?i)^meilleurs commentaires$",
        r"(?i)^nouveaux commentaires$",
        r"(?i)^plus anciens commentaires$",
        r"(?i)^\d+\s+upvotes?\s+\d+\s+commentaires?$",
        r"(?i)^\d+\s+commentaires?$",
        r"(?i)^discussions connexes$",
        r"(?i)^règles de reddit.*$",
        r"(?i)^politique de confidentialité.*$",
        r"(?i)^contrat d'utilisation.*$",
        r"(?i)^reddit,? inc\..*$",
    ]

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        lower = stripped.lower()

        # Les contenus associés et le pied de page commencent ici.
        if re.match(r"(?i)^discussions connexes\b", stripped):
            in_related_content = True
            continue

        if in_related_content:
            continue

        # Supprimer le bloc publicitaire sans conserver son contenu.
        if re.search(r"(?i)\bsponsorisé(?:e)?\b", stripped):
            in_advertisement = True
            continue

        if in_advertisement:
            if lower in ad_end_markers:
                in_advertisement = False
            continue

        # Éléments d'interface isolés.
        if any(re.fullmatch(pattern, stripped) for pattern in noise_patterns):
            continue

        # En-têtes de sous-forums suivis d'une date relative,
        # souvent mal reconnus par l'OCR.
        if re.fullmatch(
            r"(?i)r[/|\\]?\w+\s+\w+\s*"
            r"(?:il\s*y\s*a\s*)?\d+\s*"
            r"(?:s|m|h|j|d|sem|semaine|semaines|mois|an|ans)\b.*",
            stripped,
        ):
            continue

        # URL isolée, pagination et lignes composées uniquement
        # de chiffres et de symboles.
        if re.fullmatch(r"(?i)https?://\S+", stripped):
            continue

        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
            continue

        if re.fullmatch(r"[\d\s.,%+&|/=_-]+", stripped):
            continue

        # Retirer certains boutons lorsqu'ils sont fusionnés au texte.
        stripped = re.sub(
            r"(?i)\b(?:accéder au contenu principal|"
            r"rechercher dans r/\S+|se connecter|"
            r"rejoindre la conversation|"
            r"rechercher des commentaires)\b",
            "",
            stripped,
        )

        stripped = re.sub(r"[ \t]+", " ", stripped).strip()

        if stripped:
            cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"[ \t]+([,.;:!?])", r"\1", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
  

# extraction du texte du pdf: essai PyMuPDF prioritaire + OCR si insuffisant

def extraction_pdf(file_path: Union[str, Path]) -> str:
    """Extrait et nettoie le texte d'un PDF, avec OCR si nécessaire."""
    file_path = Path(file_path)

    if fitz is None:
        logging.warning("PyMuPDF n'est pas disponible : %s", file_path)
        return ""

    def nettoyer_texte_extrait(texte: str) -> str:
        """Applique le même nettoyage, quelle que soit la méthode d'extraction."""
        texte = nettoyer_entete_reddit(texte)
        texte = nettoyage_texte_reddit(texte)
        texte = nettoyer_interface_reddit(texte)
        return texte.strip()

    # 1. Extraction textuelle avec PyMuPDF
    texte_pdf = ""

    try:
        with fitz.open(file_path) as document:
            pages = [
                page.get_text("text")
                for page in document
            ]

        texte_pdf = "\n\n".join(
            page for page in pages if page.strip()
        )

        # Si le texte extrait est suffisamment riche, éviter l'OCR.
        if len(texte_pdf.strip()) >= 100:
            logging.info(
                "Extraction textuelle utilisée pour %s",
                file_path.name,
            )
            return nettoyer_texte_extrait(texte_pdf)

    except Exception as exc:
        logging.warning(
            "Erreur extraction PDF %s : %s",
            file_path,
            exc,
        )

    # 2. OCR de secours
    if easyocr is None or Image is None:
        logging.warning(
            "OCR indisponible pour %s. "
            "Retour du texte extrait par PyMuPDF.",
            file_path,
        )
        return nettoyer_texte_extrait(texte_pdf)

    try:
        reader = easyocr.Reader(["fr", "en"], gpu=False)
        pages_ocr = []

        with fitz.open(file_path) as document:
            total_pages = len(document)

            for numero_page, page in enumerate(document):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                resultats = reader.readtext(
                    np.array(image),
                    detail=1,
                )

                # L'en-tête du document n'est filtré que sur la première page.
                if numero_page == 0:
                    resultats = filtrer_entete_reddit(
                        resultats,
                        hauteur_image=image.height,
                    )

                if resultats:
                    texte_page = reconstruire_texte_ocr(resultats)
                    pages_ocr.append(texte_page)

                logging.info(
                    "OCR page %s/%s : %s",
                    numero_page + 1,
                    total_pages,
                    file_path.name,
                )

        texte_ocr = "\n\n".join(pages_ocr)

        # Si l'OCR n'a rien extrait, conserver le texte natif disponible.
        if not texte_ocr.strip():
            logging.warning(
                "OCR vide pour %s ; conservation du texte natif.",
                file_path.name,
            )
            return nettoyer_texte_extrait(texte_pdf)

        logging.info(
            "Extraction OCR utilisée pour %s",
            file_path.name,
        )
        return nettoyer_texte_extrait(texte_ocr)

    except Exception as exc:
        logging.error(
            "Erreur OCR %s : %s",
            file_path,
            exc,
        )
        return nettoyer_texte_extrait(texte_pdf)

# extraction type docs si présent
def extraction_docx(file_path: Union[str, Path]) -> str:
    try:
        from docx import Document
        document = Document(file_path)
        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()]
        return "\n".join(paragraphs).strip()

    except Exception as exc:
        logging.error("Erreur lecture DOCX %s : %s", file_path, exc)
        return ""


# extraction du type txt
def extraction_txt(file_path: Union[str, Path]) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read().strip()
    except Exception as exc:
        logging.error("Erreur lecture TXT %s : %s", file_path, exc)

        return ""

# téléchargement de fichiers depuis un site
def chargement_url(url: str, output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        file.write(response.content)

    return output_path

# chargement et lecteure depuis un zip
def extraction_zip(zip_path: Union[str, Path], output_dir: Union[str, Path]) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
        for name in archive.namelist():
            extracted_path = (output_dir / name)
            if extracted_path.is_file():
                extracted_files.append(extracted_path)

    return extracted_files


# lecture et transformation des fichiers en format compatible avec le pipeline RAG
def parsing_fichier(file_path: Union[str, Path]) -> str:
    file_path = Path(file_path)
    
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return extraction_pdf(file_path)

    if extension == ".docx":
        return extraction_docx(file_path)

    if extension == ".txt":
        return extraction_txt(file_path)

    if extension in EXCLUDED_RAG_EXTENSIONS:
        logging.info("Fichier structuré ignoré du corpus RAG : %s", file_path)
        return ""
    logging.warning("Extension non prise en charge : %s", file_path)

    return ""

# normalisation des noms d'équipe et/ou joueur
def normalize_entity_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(
        r"[^\w\s-]",
        " ", value)

    value = re.sub(
        r"\s+",
        " ",  value)
    return value

# fonction pour détecter les équipes et joueurs dans les textes:
def detect_player_team(
    text: str,
    team_aliases: Optional[Dict[str, List[str]]] = None,
    player_aliases: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:

    if not text:
        return {
            "teams": [],
            "team_codes": [],
            "players": []}

    normalized_text = normalize_entity_name(text)
    detected_teams = set()
    detected_team_codes = set()
    detected_players = set()

    # équipes
    if team_aliases:
            for team_name, aliases in (team_aliases or {}).items():
                for alias in set([team_name] + aliases):
                    if not alias:
                        continue
                    normalized_alias = normalize_entity_name(str(alias))
                    if not normalized_alias:
                        continue
                    pattern = rf"(?<!\w){re.escape(normalized_alias)}(?!\w)"
                    if re.search(pattern, normalized_text):
                        detected_teams.add(team_name)

                        # récupérer le code officiel dès que l'équipe est détectée
                        code = aliases[0] if aliases else None

                        if code and len(code.strip()) <= 3: # and alias.strip().isalpha() and alias.strip().upper() == alias.strip():
                            detected_team_codes.add(code.strip().upper())

    # joueurs
    if player_aliases:
            for player_name, aliases in (player_aliases or {}).items():
                for alias in set([player_name] + aliases):
                    if not alias:
                        continue
                    normalized_alias = normalize_entity_name(str(alias))
                    if not normalized_alias:
                        continue
                    pattern = rf"(?<!\w){re.escape(normalized_alias)}(?!\w)"
                    if re.search(pattern, normalized_text):
                        detected_players.add(player_name)
                        break

    return {
            "teams": sorted(detected_teams),
            "team_codes": sorted(detected_team_codes),
            "players": sorted(detected_players),}

# chargement des fichies depuis le dossier input et construit le format du RAG 
def load_and_parse_files(input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logging.warning("Dossier introuvable : %s", input_dir)
        return []

    # Charger une seule fois les référentiels PostgreSQL avant de parcourir les fichiers.
    engine = recuperer_database_engine()
    try:
        team_aliases, player_aliases = chargement_entites_alias(engine)
    finally:
        engine.dispose()

    documents = []
    files = sorted(path for path in input_dir.rglob("*") if path.is_file())

    for file_path in tqdm(files, desc="Chargement des documents"):
        extension = file_path.suffix.lower()

        # Les fichiers de données structurées restent exclus du corpus RAG.
        if extension in EXCLUDED_RAG_EXTENSIONS:
            logging.info("Exclusion du RAG : %s", file_path)
            continue

        if extension not in SUPPORTED_RAG_EXTENSIONS:
            continue

        content = parsing_fichier(file_path)
        if not content.strip():
            logging.warning("Aucun texte exploitable : %s", file_path)
            continue

        # Détecter les entités avec les noms chargés depuis PostgreSQL.
        entities = detect_player_team(
            content,
            team_aliases=team_aliases,
            player_aliases=player_aliases,)

        relative_path = str(file_path.relative_to(input_dir))
        is_reddit = extension == ".pdf"

        metadata = {
            "source": file_path.name,
            "filename": file_path.name,
            "full_path": str(file_path),
            "relative_path": relative_path,
            "category": "reddit" if is_reddit else "text",
            "content_type": "reddit" if is_reddit else "text",
            "file_extension": extension,
            "teams": entities["teams"],
            "team_codes": entities["team_codes"],
            "players": entities["players"],}

        documents.append({"page_content": content, "metadata": metadata})

    logging.info("Nombre de documents chargés pour le RAG : %s", len(documents))
    return documents


# alias conservé 
def load_documents(input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    return load_and_parse_files(input_dir)