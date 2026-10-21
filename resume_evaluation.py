# -*- coding: utf-8 -*-
"""Construit un CSV consolidé à partir des exports RAGAS déjà produits."""
import json
import csv
from pathlib import Path
import pandas as pd

# configuration des chemins
RESULTS_DIR = Path("ragas_results")

AFTER_DIR = RESULTS_DIR / "after_pydantic_ai"
PROTOTYPE_AMELIORE_DIR = RESULTS_DIR / "prototype_ameliore_pydantic" 

AFTER_DETAILS = AFTER_DIR / "results_detaille.json"
PRO_AMELIORE_DETAILS = PROTOTYPE_AMELIORE_DIR / "resultats_detailles.json"

AFTER_SCORES = AFTER_DIR / "scores_par_question.csv"
PRO_AMELIORE_SCORES = PROTOTYPE_AMELIORE_DIR / "scores_par_question.csv"

OUTPUT_FILE = RESULTS_DIR / "evaluation_prototypes.csv"
METRICS = [
    "answer_relevancy", 
    "answer_correctness", 
    "context_precision", 
    "context_recall", 
    "faithfulness", 
    "answer_similarity", 
    "context_entity_recall"]


# chargement des résultats détaillés json
def load_json(path):
    if not path.exists(): raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

# chargement des scores ragas
def load_scores(path):
    if not path.exists(): raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path)

# vérification du format valide des jsons
def normalize_details(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        if isinstance(data.get("results"), list): return data["results"]
        if isinstance(data.get("data"), list): return data["data"]
    raise ValueError("Format inattendu dans results_detaille.json.")

# conversion d'un dictionnaire ou d'une liste en texte pour le CSV
def json_to_cell(value):
    if value is None: return ""
    if isinstance(value, (dict, list)): return json.dumps(value, ensure_ascii=False)
    return str(value)

# extraire une clé de données parmi plusieurs
def get_value(item, *keys, default=""):
    for key in keys:
        if key in item and item[key] is not None: return item[key]
    return default

# récupérer les métadonnées
def get_metadata(item):
    value = item.get("metadata", {})
    return value if isinstance(value, dict) else {}

# récupère les ids des questions
def get_question_id(item, index):
    metadata = get_metadata(item)
    qid = metadata.get("id") or metadata.get("question_id") or item.get("question_id")
    return str(qid) if qid else f"Q{index + 1:03d}"

# conversion des iden texte
def normalize_question_id(value):
    return "" if pd.isna(value) else str(value).strip()

def safe_float(value):
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError): return None

# récupérer les scores
def extract_scores(row):
    return {m: safe_float(row[m]) for m in METRICS if m in row.index and safe_float(row[m]) is not None}

# convertir les scores en nombre
def format_score(v): return "N/A" if v is None else f"{v:.2f}"

# conversion des contextes en texte
def context_to_text(context):
    if context is None: return ""
    if isinstance(context, list):
        out=[]
        for x in context:
            if isinstance(x, dict): out.append(str(x.get("text") or x.get("page_content") or x.get("content") or ""))
            else: out.append(str(x))
        return "\n".join(out)
    if isinstance(context, dict): return json.dumps(context, ensure_ascii=False)
    return str(context)

# construction automatique des commentaires
def build_comment(question, ground_truth_prototype_initial, ground_truth_prototype_ameliore, metadata, context_prototype, context_prototype_ameliore, answer_prototype, answer_prototype_ameliore, 
                  scores_prototype, scores_prototype_ameliore):
    observations=[]
    diagnostics=[]

    # Vérifier si les références attendues sont différentes
    truth_b = str(ground_truth_prototype_initial or "").strip()
    truth_a = str(ground_truth_prototype_ameliore or "").strip()
    ground_truth_modifie = (bool(truth_b) and bool(truth_a) and truth_b != truth_a)
    
    # Récupérer et normaliser le statut de répondabilité
    answerable = metadata.get("answerable")
    if isinstance(answerable, str):
        v = answerable.lower().strip()

        if v in {"false", "non", "no", "0"}:
            answerable = False
        elif v in {"true", "oui", "yes", "1"}:
            answerable = True
        else:
            answerable = None

    if ground_truth_modifie:
        observations.append("Le ground truth a été révisé entre les deux évaluations.")
        diagnostics.append("Les écarts de answer_correctness doivent être interprétés avec "
            "prudence, car ils peuvent provenir du changement de référence "
            "et du changement de réponse")
    elif not truth_b or not truth_a:
        diagnostics.append(
            "un ground truth est absent pour au moins un des deux prototypes ; "
            "la comparaison des scores de correction peut être limitée")
    else:
        observations.append("Le ground truth est identique entre les deux évaluations.")

    # dictionnaire d'aide à la formulation des phrases
    labels={
        "answer_relevancy":"la pertinence de la réponse",
        "answer_correctness":"la correction par rapport à la réponse attendue",
        "context_precision":"la précision des contextes récupérés",
        "context_recall":"la couverture du contexte",
        "faithfulness":"l'alignement de la réponse avec les contextes",
        "answer_similarity":"la similarité sémantique avec la réponse attendue",
        "context_entity_recall":"la couverture des entités attendues dans les contextes récupérés",}

    # comparaison des scores
    for metric in METRICS:
        b=scores_prototype.get(metric); a=scores_prototype_ameliore.get(metric)
        if b is None or a is None: continue

        # calcul de la différence entre les 2 prototypes
        d=a-b 
        if d>=.20: observations.append(f"{labels[metric]} augmente nettement ({format_score(b)} → {format_score(a)}).")
        elif d>=.05: observations.append(f"{labels[metric]} augmente légèrement ({format_score(b)} → {format_score(a)}).")
        elif d<=-.20: observations.append(f"{labels[metric]} diminue nettement ({format_score(b)} → {format_score(a)}).")
        elif d<=-.05: observations.append(f"{labels[metric]} diminue légèrement ({format_score(b)} → {format_score(a)}).")

    # remarques en fonction des seuils
    fb=scores_prototype.get("faithfulness"); fa=scores_prototype_ameliore.get("faithfulness")
    if fb is not None and fb < .40:
        diagnostics.append("la réponse du prototype brut est faiblement alignée avec les contextes récupérés, signalant un problème de grounding ou des informations non justifiées")
    if fb is not None and fa is not None and fa-fb >= .20 and fa >= .80:
        diagnostics.append("le prototype amélioré réduit fortement les informations non justifiées et améliore le grounding")

    rb=scores_prototype.get("context_recall"); ra=scores_prototype_ameliore.get("context_recall")
    if rb is not None and ra is not None and abs(ra-rb)<.05:
        diagnostics.append("la couverture du contexte reste globalement inchangée; le prototype amélioré ne corrige donc pas une donnée absente du retrieval")

    if answerable is False:
        diagnostics.append("la question est identifiée comme non répondable avec les données disponibles; le prototype amélioré devrait idéalement le signaler explicitement plutôt que produire une réponse spéculative")

    cb = scores_prototype.get("answer_correctness")
    ca = scores_prototype_ameliore.get("answer_correctness")
    if cb is not None and cb < .40:
        diagnostics.append("le score de answer_correctness du prototype brut est faible")

    if cb is not None and ca is not None:
        delta_correctness = ca - cb
        if ground_truth_modifie:
            if delta_correctness >= .15:
                diagnostics.append(
                    "le score de answer_correctness augmente sensiblement, "
                    "mais le changement de ground truth empêche d'attribuer "
                    "directement cette hausse à une amélioration du prototype")
            elif delta_correctness <= -.15:
                diagnostics.append(
                    "le score de answer_correctness diminue sensiblement ; "
                    "la modification du ground truth doit être prise en compte "
                    "avant de conclure à une régression du prototype")
        else:
            if delta_correctness >= .15:
                diagnostics.append("le prototype amélioré augmente sensiblement la conformité à la même référence attendue")
            elif delta_correctness <= -.15:
                diagnostics.append("le prototype amélioré diminue sensiblement la conformité à la même référence attendue")

    lb=scores_prototype.get("answer_relevancy"); la=scores_prototype_ameliore.get("answer_relevancy")
    if lb is not None and la is not None and la-lb <= -.20:
        diagnostics.append("le prototype amélioré rend la réponse nettement moins directement pertinente ; il faut éviter une sécurisation qui remplace une réponse utile par une réponse excessivement prudente")

    # comparaison des réponses 
    if not str(answer_prototype or "").strip() and str(answer_prototype_ameliore or "").strip():
        diagnostics.append("une réponse est disponible dans le prototype amélioré alors que la réponse du prototype brut est vide")
    elif str(answer_prototype or "").strip() != str(answer_prototype_ameliore or "").strip():
        observations.append("La réponse est modifiée entre les deux prototypes.")
    else:
        observations.append("La réponse reste identique entre les deux prototypes.")

    # vérification des contextes
    if not context_to_text(context_prototype).strip():
        diagnostics.append("aucun contexte exploitable n'est disponible dans le prototype brut")

    if not context_to_text(context_prototype_ameliore).strip():
        diagnostics.append("aucun contexte exploitable n'est disponible dans le prototype amélioré")
    
    # commentaire final
    return f"{str(question).strip()} — " + " ".join(observations) + " Diagnostic : " + ". ".join(diagnostics) + "."

# construction du csv pour avoir les questions, metadata, contexte/reponses/scores avant/après pour chaque question
def build_complete_csv():
    print("Lecture des exports RAGAS...")
    before_details = normalize_details(load_json(AFTER_DETAILS))
    after_details = normalize_details(load_json(PRO_AMELIORE_DETAILS))

    bs=load_scores(AFTER_SCORES)
    a_s=load_scores(PRO_AMELIORE_SCORES)

    bd={get_question_id(x,i):x for i,x in enumerate(before_details)}; ad={get_question_id(x,i):x for i,x in enumerate(after_details)}
    for df,name in [(bs,"prototype"),(a_s,"pro_ameliore")]:
        if "question_id" not in df.columns: raise ValueError(f"Colonne question_id absente du fichier scores {name}.")
        df["question_id"]=df["question_id"].apply(normalize_question_id)
    bs=bs.set_index("question_id"); a_s=a_s.set_index("question_id")
    ids=[]
    for qid in list(bd)+list(ad):
        if qid not in ids: ids.append(qid)
    rows=[]
    for qid in ids:
        b=bd.get(qid,{})
        a=ad.get(qid,{})
        
        question=get_value(b,"question", "questions", default=get_value(a, "question", "questions", default=""))

        # groud truth prototype initial
        truth_b = get_value(b, "ground_truth", default="")

        # groud truth prototype initial
        truth_a = get_value(a, "ground_truth", default="")
        
        metadata = get_metadata(b) or get_metadata(a)

        cb = get_value(b,"contexts", "context", default=""); ca=get_value(a, "contexts", "context", default="")

        ab = get_value(b, "answer", "response", "reponse", default="")
        aa = get_value(a, "answer", "response", "reponse", default="")

        sb = extract_scores(bs.loc[qid].iloc[0] if isinstance(bs.loc[qid], pd.DataFrame) else bs.loc[qid]) if qid in bs.index else {}
        sa = extract_scores(a_s.loc[qid].iloc[0] if isinstance(a_s.loc[qid], pd.DataFrame) else a_s.loc[qid]) if qid in a_s.index else {}
        comment = build_comment(
            question,
            truth_b,
            truth_a,
            metadata,
            cb,
            ca,
            ab,
            aa,
            sb,
            sa)
        rows.append({"question_id":qid,
                     "questions":question,
                     "metadata":json_to_cell(metadata),
                     "ground_truth_prototype_initial":truth_b,
                     "ground_truth_prototype_ameliore":truth_a,
                     "contexte_prototype_initial":json_to_cell(cb),
                     "contexte_prototype_ameliore":json_to_cell(ca),
                     "reponse_prototype_initial":ab,
                     "reponse_prototype_ameliore":aa,
                     "scores_prototype_initial":json_to_cell(sb),
                     "scores_prototype_ameliore":json_to_cell(sa),
                     "commentaire":comment})

    # export du fichier
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig",quoting=csv.QUOTE_MINIMAL)
    print(f"CSV créé : {OUTPUT_FILE}"); print(f"Questions : {len(rows)}")

if __name__ == "__main__": build_complete_csv()
