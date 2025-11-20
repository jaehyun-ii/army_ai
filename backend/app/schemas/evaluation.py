"""
Pydantic schemas for evaluation endpoints.
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class EvalStatus(str, Enum):
    """Evaluation run status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class EvalPhase(str, Enum):
    """Evaluation phase."""
    PRE_ATTACK = "pre_attack"
    POST_ATTACK = "post_attack"


class EvalDatasetType(str, Enum):
    """Evaluation dataset type."""
    BASE = "base"
    ATTACK = "attack"


# ========== Evaluation Run Schemas ==========

class EvalRunBase(BaseModel):
    """Base schema for evaluation run (aligned with DB model)."""
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    phase: EvalPhase
    model_id: UUID
    # 2D datasets only (3D removed, dimension inferred from which dataset_id fields are set)
    base_dataset_id: Optional[UUID] = None
    attack_dataset_id: Optional[UUID] = None
    # Experiment linkage
    experiment_id: Optional[UUID] = None
    params: Optional[Dict[str, Any]] = Field(None, description="Evaluation parameters (threshold, NMS, IoU, etc.)")


class EvalRunCreate(EvalRunBase):
    """Schema for creating evaluation run."""

    @field_validator("params")
    @classmethod
    def validate_params(cls, v):
        """Validate params is a dict if provided."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("params must be a JSON object")
        return v


class EvalRunUpdate(BaseModel):
    """Schema for updating evaluation run."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[EvalStatus] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    params: Optional[Dict[str, Any]] = None


class EvalRunResponse(EvalRunBase):
    """Schema for evaluation run response."""
    id: UUID
    status: EvalStatus
    metrics_summary: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class EvalRunListResponse(BaseModel):
    """Schema for paginated evaluation run list."""
    items: List[EvalRunResponse]
    total: int
    page: int
    page_size: int


# ========== Evaluation Item Schemas ==========

class EvalItemBase(BaseModel):
    """
    Base schema for evaluation item (2D only).

    PHASE 2 UPDATE: file_name and storage_key removed (use image_2d relationship).
    """
    run_id: UUID
    image_2d_id: Optional[UUID] = None
    dataset_type: Optional[EvalDatasetType] = Field(None, description="Dataset type: 'base' or 'attack'")
    ground_truth: Optional[Union[Dict[str, Any], List[Any]]] = Field(None, description="GT bounding boxes/classes")
    prediction: Optional[Union[Dict[str, Any], List[Any]]] = Field(None, description="Model predictions")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Per-item metrics")
    notes: Optional[str] = None


class EvalItemCreate(EvalItemBase):
    """Schema for creating evaluation item."""
    pass


class EvalItemUpdate(BaseModel):
    """Schema for updating evaluation item."""
    ground_truth: Optional[Union[Dict[str, Any], List[Any]]] = None
    prediction: Optional[Union[Dict[str, Any], List[Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class EvalItemResponse(EvalItemBase):
    """
    Schema for evaluation item response.

    PHASE 2 UPDATE: file_name and storage_key are computed from image_2d relationship.
    """
    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    # Computed fields from image_2d relationship (for backward compatibility)
    file_name: Optional[str] = Field(None, description="Computed from image_2d.file_name")
    storage_key: Optional[str] = Field(None, description="Computed from image_2d.storage_key")

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class EvalItemListResponse(BaseModel):
    """Schema for paginated evaluation item list."""
    items: List[EvalItemResponse]
    total: int
    page: int
    page_size: int


# ========== Evaluation List Schemas ==========

class EvalListBase(BaseModel):
    """Base schema for evaluation list."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class EvalListCreate(EvalListBase):
    """Schema for creating evaluation list."""
    pass


class EvalListUpdate(BaseModel):
    """Schema for updating evaluation list."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None


class EvalListResponse(EvalListBase):
    """Schema for evaluation list response."""
    id: UUID
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class EvalListWithItemsResponse(EvalListResponse):
    """Schema for evaluation list with items."""
    items: List["EvalListItemResponse"]


class EvalListListResponse(BaseModel):
    """Schema for paginated evaluation list."""
    items: List[EvalListResponse]
    total: int
    page: int
    page_size: int


# ========== Evaluation List Item Schemas ==========

class EvalListItemBase(BaseModel):
    """Base schema for evaluation list item."""
    list_id: UUID
    run_id: UUID
    sort_order: int = 0


class EvalListItemCreate(EvalListItemBase):
    """Schema for creating evaluation list item."""
    pass


class EvalListItemUpdate(BaseModel):
    """Schema for updating evaluation list item."""
    sort_order: Optional[int] = None


class EvalListItemResponse(EvalListItemBase):
    """Schema for evaluation list item response."""
    id: UUID
    created_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}


# ========== Comparison Schemas ==========

class EvalRunPairResponse(BaseModel):
    """Schema for pre/post evaluation run pair comparison (aligned with DB model)."""
    model_config = ConfigDict(protected_namespaces=())

    pre_run_id: UUID
    post_run_id: UUID
    model_id: UUID
    base_dataset_id: UUID
    attack_dataset_id: UUID
    pre_metrics: Optional[Dict[str, Any]] = None
    post_metrics: Optional[Dict[str, Any]] = None
    pre_created_at: datetime
    post_created_at: datetime


class EvalRunPairDeltaResponse(EvalRunPairResponse):
    """Schema for pre/post evaluation run pair with delta metrics."""
    pre_map: Optional[float] = None
    post_map: Optional[float] = None
    delta_map: Optional[float] = None


class RobustnessMetrics(BaseModel):
    """Robustness degradation metrics."""
    delta_map: float = Field(..., description="Absolute drop in mAP (clean - adv)")
    delta_map50: float = Field(..., description="Absolute drop in mAP@50")
    drop_percentage: float = Field(..., description="Percentage drop in mAP ((clean-adv)/clean * 100)")
    robustness_ratio: float = Field(..., description="Adversarial robustness ratio (adv/clean, 1.0 = fully robust)")
    delta_recall: float = Field(..., description="Recall drop (clean - adv)")
    delta_precision: float = Field(..., description="Precision drop (clean - adv)")

    # Clean vs Adversarial comparison
    ap_clean: float = Field(..., description="Clean mAP")
    ap_adv: float = Field(..., description="Adversarial mAP")
    ap50_clean: float = Field(..., description="Clean mAP@50")
    ap50_adv: float = Field(..., description="Adversarial mAP@50")
    recall_clean: float = Field(..., description="Clean recall")
    recall_adv: float = Field(..., description="Adversarial recall")
    precision_clean: float = Field(..., description="Clean precision")
    precision_adv: float = Field(..., description="Adversarial precision")


class AttackInfo(BaseModel):
    """Attack dataset information."""
    id: str
    name: str
    attack_type: str
    parameters: Optional[Dict[str, Any]] = None
    target_class: Optional[str] = None


class VisualizationData(BaseModel):
    """Data structure optimized for frontend visualization."""
    # 1. Overall comparison for bar charts
    overall_comparison: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {metric, clean, adversarial} for overall comparison chart"
    )

    # 2. Performance drops for horizontal bar charts
    drops: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {metric, drop_percentage, delta} for drop visualization"
    )

    # 3. Per-class comparison
    per_class_comparison: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {class_name, clean_map, adv_map, drop_percentage, robustness_ratio}"
    )

    # 4. Summary cards
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary metrics for dashboard cards"
    )


class EvalRunComparisonResponse(BaseModel):
    """Schema for detailed evaluation run comparison."""
    pre_attack_run: EvalRunResponse = Field(..., description="Pre-attack (clean) evaluation results")
    post_attack_run: EvalRunResponse = Field(..., description="Post-attack (adversarial) evaluation results")
    overall_robustness: RobustnessMetrics = Field(..., description="Overall robustness metrics")
    per_class_robustness: Dict[str, Union[RobustnessMetrics, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Per-class robustness metrics"
    )
    attack_info: Optional[AttackInfo] = Field(None, description="Attack dataset information")
    visualization_data: VisualizationData = Field(..., description="Data optimized for charts and graphs")

    model_config = {"from_attributes": True}


# Legacy schema - keeping for backward compatibility
class EvalRunComparisonResponseLegacy(BaseModel):
    """Schema for detailed evaluation run comparison (legacy)."""
    pre_run: EvalRunResponse
    post_run: EvalRunResponse
    delta_metrics: Dict[str, Any] = Field(default_factory=dict, description="Computed metric deltas")
