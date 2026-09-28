
# Centralisation des modèles de validation à utiliser dans le pipeline
from typing import Any
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