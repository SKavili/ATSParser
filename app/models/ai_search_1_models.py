"""Pydantic models for AI search v1 (query-driven IT / NON_IT / category classification)."""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

from app.models.ai_search_models import (
    _normalize_mastercategory,
    CandidateResult,
)


class AISearch1Request(BaseModel):
    """Request for /ai-search-1: optional overrides; otherwise classification is inferred from query."""

    query: str = Field(..., description="Natural language search query")
    mastercategory: Optional[str] = Field(
        None,
        description="Optional override: IT or NON_IT — if BOTH mastercategory and category are set, inference is skipped",
    )
    category: Optional[str] = Field(
        None,
        description="Optional override: exact category name — used only with mastercategory",
    )
    user_id: Optional[int] = Field(None, description="Optional user ID for tracking")
    top_k: Optional[int] = Field(100, description="Number of results (default: 100)")

    @field_validator("mastercategory", mode="before")
    @classmethod
    def normalize_mastercategory(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_mastercategory(v)


class AISearch1Response(BaseModel):
    """Response for /ai-search-1: same candidate shape as AI search plus classification metadata."""

    query: str
    mastercategory: Optional[str] = Field(
        None, description="Mastercategory used for search (None if broad / name search)"
    )
    category: Optional[str] = Field(
        None, description="Category namespace used for search (None if broad / name search)"
    )
    total_results: int
    results: List[CandidateResult]
    search_type: Optional[str] = Field(None, description="'semantic' | 'name' | 'hybrid'")
    results_from: Optional[str] = None
    inferred_mastercategory: Optional[str] = Field(
        None,
        description="IT / NON_IT inferred from query when classification ran (null if explicit payload or not inferred)",
    )
    inferred_category: Optional[str] = Field(
        None,
        description="Category inferred from query when classification ran",
    )
    classification_mode: Optional[str] = Field(
        None,
        description="'explicit_payload' | 'inferred' | 'broad' | 'name'",
    )
