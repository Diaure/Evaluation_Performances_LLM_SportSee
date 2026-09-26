from typing import Any

from pydantic import BaseModel, Field


class DocumentModel(BaseModel):
    page_content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuestionModel(BaseModel):
    question: str = Field(..., min_length=1)


class ChunkModel(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class AnswerModel(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class RAGResultModel(BaseModel):
    question: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    ground_truth: str | None = None