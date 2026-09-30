
# Centralisation des modèles de validation à utiliser dans le pipeline
from typing import Any, Optional
from pydantic import BaseModel, Field

# validation du document extrait depuis la source
class DocumentModel(BaseModel):
    page_content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


# validation nettoyage & préparation du document
class PreparedDocumentModel(BaseModel):
    page_content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Validation des chunks utilisés par le vector store
class ChunkModel(BaseModel):
    id: str | None = None
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


# Validation de la question
class QuestionModel(BaseModel):
    question: str = Field(..., min_length=1)
    

# Validation de la réponse du LLM
class AnswerModel(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

# Validation de l'ensemble des données RAGAS
class RAGResultModel(BaseModel):
    question: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    ground_truth: str | None = None

# Contrôle des réponses du model
class ControlledAnswerModel(BaseModel):
    answer: str = Field(description="Réponse finale à la question de l'utilisateur.")
    grounded_in_context: bool = Field(
        description=(
            "True si la réponse est entièrement justifiée "
            "par le contexte fourni, sinon False."))

    unsupported_claims: list[str] = Field(default_factory=list,
                                          description=(
            "Liste des affirmations présentes dans la réponse "
            "qui ne sont pas justifiées par le contexte."))


# Pour base de données
class Team(BaseModel):
    code_equipe: str
    nom_complet_equipe: Optional[str] = None
    nombre_joueurs_par_equipe: Optional[int] = None
    nombre_points_total_par_equipe: Optional[int] = None


class Player(BaseModel):
    nom_du_joueur: str
    equipe_du_joueur: str
    age_du_joueur: int


class Match(BaseModel):
    nom_du_joueur: str
    nombre_matchs_joues: int
    victoires: int
    defaites: int
    minutes_moyennes: float
    points_moyens: float
    tirs_reussis: float
    tirs_tentes: float
    pourcentage_reussite: float
    minutes_apres_15: float
    tirs_3_points_tentes: float


class Stats(BaseModel):
    nom_du_joueur: str
    pourcentage_3_points: float
    lancers_francs_reussis: float
    lancers_francs_tentes: float
    pourcentage_lancers_francs: float
    rebonds_offensifs: float
    rebonds_defensifs: float
    rebonds_totaux: float
    passes_decisives: float
    balles_perdues: float
    interceptions: float
    contres: float
    fautes_personnelles: float
    fantasy_points: float
    double_doubles: int
    triple_doubles: int
    plus_minus: float
    offensive_rating: float
    defensive_rating: float
    net_rating: float
    pourcentage_assists: float
    ratio_assists_pertes: float
    ratio_assists_100_possessions: float
    pourcentage_rebonds_offensifs: float
    pourcentage_rebonds_defensifs: float
    pourcentage_rebonds_totaux: float
    turnover_ratio: float
    efg_percent: float
    true_shooting_percent: float
    usage_rate: float
    rythme_de_jeu: float
    player_impact_estimate: float
    possessions_totales: float


class Top15Player(BaseModel):
    nom_du_joueur: str
    nombre_points_total: float
    tirs_reussis: float
    pourcentage_tirs_reussis: float
    pourcentage_tirs_3_points: float
    pourcentage_lancers_francs: float
    rebonds_offensifs: float
    impact_estime: float