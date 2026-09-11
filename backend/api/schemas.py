from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationResponse(BaseModel):
    valid: bool
    input_configuration: str
    image_count: int
    modalities_declared: List[str]
    crs_declared: List[str]
    spatial_resolutions: List[Dict[str, float]]
    co_registration_overlap_pct: Optional[float] = None
    dimension_match: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query: str
    task_type: str
    answer: str
    confidence: float
    visual_overlays: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace: Dict[str, Any] = Field(default_factory=dict)
    status: str
    error: Optional[str] = None


class SampleDataset(BaseModel):
    id: str
    name: str
    category: str  # "single", "bitemporal", "cross_modal"
    description: str
    recommended_queries: List[str]
    images: List[Dict[str, Any]]
