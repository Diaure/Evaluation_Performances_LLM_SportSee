# -*- coding: utf-8 -*-
"""Construit un CSV consolidé à partir des exports RAGAS déjà produits."""
import json
import csv
from pathlib import Path
import pandas as pd

# configuration des chemins
RESULTS_DIR = Path("ragas_results") # dossier de resultats

BEFORE_DIR = RESULTS_DIR / "before_pydantic_ai" 
AFTER_DIR = RESULTS_DIR / "after_pydantic_ai"

BEFORE_DETAILS = BEFORE_DIR / "results_detaille.json"
AFTER_DETAILS = AFTER_DIR / "results_detaille.json"

BEFORE_SCORES = BEFORE_DIR / "scores_par_question.csv"
AFTER_SCORES = AFTER_DIR / "scores_par_question.csv"

OUTPUT_FILE = RESULTS_DIR / "evaluation_complete_avant_apres.csv"
METRICS = ["answer_relevancy", "answer_correctness", "context_precision", "context_recall", "faithfulness"]

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
def build_comment(question, ground_truth, metadata, context_before, context_after, answer_before, answer_after, scores_before, scores_after):
    observations=[]; diagnostics=[]
    answerable=metadata.get("answerable") # vérifie si la question est répondable selon les metadoées
    if isinstance(answerable, str): # convertit les valeurs en booléens
        v=answerable.lower().strip()
        if v in {"false","non","no","0"}: answerable=False
        elif v in {"true","oui","yes","1"}: answerable=True

    # dictionnaire d'aide à la formulation des phrases
    labels={
        "answer_relevancy":"la pertinence de la réponse",
        "answer_correctness":"la correction par rapport à la réponse attendue",
        "context_precision":"la précision des contextes récupérés",
        "context_recall":"la couverture du contexte",
        "faithfulness":"l'alignement de la réponse avec les contextes",}

    # comparaison des scores avant vs après
    for metric in METRICS:
        b=scores_before.get(metric); a=scores_after.get(metric)
        if b is None or a is None: continue
        d=a-b # calcul de la différence entre avant et après
        if d>=.20: observations.append(f"{labels[metric]} augmente nettement ({format_score(b)} → {format_score(a)}).")
        elif d>=.05: observations.append(f"{labels[metric]} augmente légèrement ({format_score(b)} → {format_score(a)}).")
        elif d<=-.20: observations.append(f"{labels[metric]} diminue nettement ({format_score(b)} → {format_score(a)}).")
        elif d<=-.05: observations.append(f"{labels[metric]} diminue légèrement ({format_score(b)} → {format_score(a)}).")

    # remarques en fonction des seuils
    fb=scores_before.get("faithfulness"); fa=scores_after.get("faithfulness")
    if fb is not None and fb < .40:
        diagnostics.append("la réponse avant contrôle est faiblement alignée avec les contextes récupérés, signalant un problème de grounding ou des informations non justifiées")
    if fb is not None and fa is not None and fa-fb >= .20 and fa >= .80:
        diagnostics.append("le contrôle Pydantic AI réduit fortement les informations non justifiées et améliore le grounding")

    rb=scores_before.get("context_recall"); ra=scores_after.get("context_recall")
    if rb is not None and ra is not None and abs(ra-rb)<.05:
        diagnostics.append("la couverture du contexte reste globalement inchangée ; le contrôle Pydantic AI ne corrige donc pas une donnée absente du retrieval")
    if answerable is False:
        diagnostics.append("la question est identifiée comme non répondable avec les données disponibles ; le prototype devrait idéalement le signaler explicitement plutôt que produire une réponse spéculative")

    cb=scores_before.get("answer_correctness"); ca=scores_after.get("answer_correctness")
    if cb is not None and cb < .40: diagnostics.append("la réponse avant contrôle est peu conforme à la réponse attendue")
    if cb is not None and ca is not None and ca-cb >= .15: diagnostics.append("le contrôle améliore sensiblement la conformité à la réponse attendue")
    if cb is not None and ca is not None and ca-cb <= -.15: diagnostics.append("le contrôle dégrade sensiblement la conformité à la réponse attendue")

    lb=scores_before.get("answer_relevancy"); la=scores_after.get("answer_relevancy")
    if lb is not None and la is not None and la-lb <= -.20:
        diagnostics.append("le contrôle rend la réponse nettement moins directement pertinente ; il faut éviter une sécurisation qui remplace une réponse utile par une réponse excessivement prudente")

    # comparaison des réponses avant vs après
    if not str(answer_before or "").strip() and str(answer_after or "").strip():
        diagnostics.append("une réponse est disponible après contrôle alors que la réponse avant contrôle est vide dans l'export")
    elif str(answer_before or "").strip() != str(answer_after or "").strip():
        observations.append("La réponse est modifiée après le contrôle Pydantic AI.")
    else:
        observations.append("La réponse reste identique avant et après le contrôle.")

    if not context_to_text(context_before).strip(): diagnostics.append("aucun contexte exploitable n'est disponible avant contrôle")
    if not context_to_text(context_after).strip(): diagnostics.append("aucun contexte exploitable n'est disponible après contrôle")

    if not diagnostics:
        diagnostics.append("aucun défaut majeur ne ressort des scores et des exports disponibles pour cette question")
    
    # commentaire final
    return f"{str(question).strip()} — " + " ".join(observations) + " Diagnostic : " + ". ".join(diagnostics) + "."

# construction du csv pour avoir les questions, metadata, contexte/reponses/scores avant/après pour chaque question
def build_complete_csv():
    print("Lecture des exports RAGAS...")
    before_details=normalize_details(load_json(BEFORE_DETAILS)); after_details=normalize_details(load_json(AFTER_DETAILS))
    bs=load_scores(BEFORE_SCORES); a_s=load_scores(AFTER_SCORES)
    bd={get_question_id(x,i):x for i,x in enumerate(before_details)}; ad={get_question_id(x,i):x for i,x in enumerate(after_details)}
    for df,name in [(bs,"avant"),(a_s,"après")]:
        if "question_id" not in df.columns: raise ValueError(f"Colonne question_id absente du fichier scores {name}.")
        df["question_id"]=df["question_id"].apply(normalize_question_id)
    bs=bs.set_index("question_id"); a_s=a_s.set_index("question_id")
    ids=[]
    for qid in list(bd)+list(ad):
        if qid not in ids: ids.append(qid)
    rows=[]
    for qid in ids:
        b=bd.get(qid,{}); a=ad.get(qid,{})
        question=get_value(b,"question","questions",default=get_value(a,"question","questions",default=""))
        truth=get_value(b,"ground_truth",default=get_value(a,"ground_truth",default=""))
        metadata=get_metadata(b) or get_metadata(a)
        cb=get_value(b,"contexts","context",default=""); ca=get_value(a,"contexts","context",default="")
        ab=get_value(b,"answer","response","reponse",default=""); aa=get_value(a,"answer","response","reponse",default="")
        sb=extract_scores(bs.loc[qid].iloc[0] if isinstance(bs.loc[qid],pd.DataFrame) else bs.loc[qid]) if qid in bs.index else {}
        sa=extract_scores(a_s.loc[qid].iloc[0] if isinstance(a_s.loc[qid],pd.DataFrame) else a_s.loc[qid]) if qid in a_s.index else {}
        comment=build_comment(question,truth,metadata,cb,ca,ab,aa,sb,sa)
        rows.append({"question_id":qid,"questions":question,"metadata":json_to_cell(metadata),"ground_truth":truth,"contexte_avant_pydantic":json_to_cell(cb),"contexte_apres_pydantic":json_to_cell(ca),"reponse_avant":ab,"reponse_apres":aa,"scores_avant":json_to_cell(sb),"scores_apres":json_to_cell(sa),"commentaire":comment})

    # export du fichier
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig",quoting=csv.QUOTE_MINIMAL)
    print(f"CSV créé : {OUTPUT_FILE}"); print(f"Questions : {len(rows)}")

if __name__ == "__main__": build_complete_csv()
