"""
CRUD operations for evaluation.
"""
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy import select, func, and_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.evaluation import (
    EvalRun,
    EvalDatasetResult,
    EvalItem,
    EvalList,
    EvalListItem,
    EvalClassMetrics,
    EvalPhase,
    EvalStatus,
)
from app.schemas.evaluation import (
    EvalRunCreate,
    EvalRunUpdate,
    EvalDatasetResultCreate,
    EvalDatasetResultUpdate,
    EvalItemCreate,
    EvalItemUpdate,
    EvalListCreate,
    EvalListUpdate,
    EvalListItemCreate,
    EvalClassMetricsCreate,
    EvalClassMetricsUpdate,
    EvalDatasetType,
)


# ========== Evaluation Run CRUD ==========

async def create_eval_run(
    db: AsyncSession,
    eval_run: EvalRunCreate,
    created_by: Optional[UUID] = None,
) -> EvalRun:
    """Create a new evaluation run."""
    # Exclude None values to avoid constraint violations
    eval_run_data = eval_run.model_dump(exclude_none=True)
    db_eval_run = EvalRun(
        **eval_run_data,
        created_by=created_by,
    )
    db.add(db_eval_run)
    await db.flush()
    await db.refresh(db_eval_run)
    
    # Re-fetch with eager loading to ensure relationships are available
    return await get_eval_run(db, db_eval_run.id)


async def get_eval_run(db: AsyncSession, eval_run_id: UUID) -> Optional[EvalRun]:
    """Get evaluation run by ID."""
    result = await db.execute(
        select(EvalRun)
        .options(selectinload(EvalRun.dataset_results))
        .where(
            and_(
                EvalRun.id == eval_run_id,
                EvalRun.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_eval_runs(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    phase: Optional[EvalPhase] = None,
    status: Optional[EvalStatus] = None,
    model_id: Optional[UUID] = None,
    base_dataset_id: Optional[UUID] = None,
    attack_dataset_id: Optional[UUID] = None,
    base_dataset_3d_id: Optional[UUID] = None,
    attack_dataset_3d_id: Optional[UUID] = None,
) -> tuple[List[EvalRun], int]:
    """Get list of evaluation runs with filters and pagination."""
    # Build filters
    filters = [EvalRun.deleted_at.is_(None)]
    if phase:
        filters.append(EvalRun.phase == phase)
    if status:
        filters.append(EvalRun.status == status)
    if model_id:
        filters.append(EvalRun.model_id == model_id)
    if base_dataset_id:
        filters.append(EvalRun.base_dataset_id == base_dataset_id)
    if attack_dataset_id:
        filters.append(EvalRun.attack_dataset_id == attack_dataset_id)
    if base_dataset_3d_id:
        filters.append(EvalRun.base_dataset_3d_id == base_dataset_3d_id)
    if attack_dataset_3d_id:
        filters.append(EvalRun.attack_dataset_3d_id == attack_dataset_3d_id)

    # Get total count
    count_query = select(func.count()).select_from(EvalRun).where(and_(*filters))
    total = await db.scalar(count_query)

    # Get items
    query = (
        select(EvalRun)
        .options(selectinload(EvalRun.dataset_results))
        .where(and_(*filters))
        .order_by(EvalRun.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    items = list(result.scalars().all())

    return items, total or 0


async def update_eval_run(
    db: AsyncSession,
    eval_run_id: UUID,
    eval_run_update: EvalRunUpdate,
) -> Optional[EvalRun]:
    """Update evaluation run."""
    db_eval_run = await get_eval_run(db, eval_run_id)
    if not db_eval_run:
        return None

    update_data = eval_run_update.model_dump(exclude_unset=True)
    # Filter out None values for JSONB fields to prevent JSON 'null' constraint violations
    # If you want to clear a JSONB field, don't include it in the update
    update_data = {k: v for k, v in update_data.items() if v is not None or k not in ['params', 'metrics_summary', 'iou_distribution']}
    for field, value in update_data.items():
        setattr(db_eval_run, field, value)

    await db.flush()
    await db.refresh(db_eval_run)
    
    # Re-fetch to ensure eager loading
    return await get_eval_run(db, eval_run_id)


async def delete_eval_run(db: AsyncSession, eval_run_id: UUID) -> bool:
    """Soft delete evaluation run."""
    db_eval_run = await get_eval_run(db, eval_run_id)
    if not db_eval_run:
        return False

    db_eval_run.deleted_at = func.now()
    await db.flush()
    return True


# ========== Evaluation Dataset Result CRUD ==========

async def create_eval_dataset_result(
    db: AsyncSession,
    dataset_result: EvalDatasetResultCreate,
) -> EvalDatasetResult:
    """Create a new evaluation dataset result."""
    # Exclude None values to prevent JSON 'null' from being stored in JSONB columns
    # which would violate the check constraint (must be NULL or object type)
    db_dataset_result = EvalDatasetResult(**dataset_result.model_dump(exclude_none=True))
    db.add(db_dataset_result)
    await db.flush()
    await db.refresh(db_dataset_result)
    return db_dataset_result


async def get_eval_dataset_result(
    db: AsyncSession,
    dataset_result_id: UUID,
) -> Optional[EvalDatasetResult]:
    """Get evaluation dataset result by ID."""
    result = await db.execute(
        select(EvalDatasetResult).where(
            and_(
                EvalDatasetResult.id == dataset_result_id,
                EvalDatasetResult.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_eval_dataset_results_by_run(
    db: AsyncSession,
    eval_run_id: UUID,
    dataset_type: Optional[EvalDatasetType] = None,
) -> List[EvalDatasetResult]:
    """Get all dataset results for an evaluation run, optionally filtered by type."""
    filters = [
        EvalDatasetResult.eval_run_id == eval_run_id,
        EvalDatasetResult.deleted_at.is_(None),
    ]
    if dataset_type:
        filters.append(EvalDatasetResult.dataset_type == dataset_type)

    result = await db.execute(
        select(EvalDatasetResult)
        .where(and_(*filters))
        .order_by(EvalDatasetResult.dataset_type)  # BASE before ATTACK
    )
    return list(result.scalars().all())


async def get_eval_dataset_result_by_run_and_type(
    db: AsyncSession,
    eval_run_id: UUID,
    dataset_type: EvalDatasetType,
) -> Optional[EvalDatasetResult]:
    """Get specific dataset result for a run by type (BASE or ATTACK)."""
    result = await db.execute(
        select(EvalDatasetResult).where(
            and_(
                EvalDatasetResult.eval_run_id == eval_run_id,
                EvalDatasetResult.dataset_type == dataset_type,
                EvalDatasetResult.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def update_eval_dataset_result(
    db: AsyncSession,
    dataset_result_id: UUID,
    dataset_result_update: EvalDatasetResultUpdate,
) -> Optional[EvalDatasetResult]:
    """Update evaluation dataset result."""
    db_dataset_result = await get_eval_dataset_result(db, dataset_result_id)
    if not db_dataset_result:
        return None

    update_data = dataset_result_update.model_dump(exclude_unset=True)
    # Filter out None values for JSONB fields to prevent JSON 'null' constraint violations
    # If you want to clear a JSONB field, don't include it in the update
    update_data = {k: v for k, v in update_data.items() if v is not None or k not in ['metrics_summary', 'iou_distribution']}
    for field, value in update_data.items():
        setattr(db_dataset_result, field, value)

    await db.flush()
    await db.refresh(db_dataset_result)
    return db_dataset_result


async def delete_eval_dataset_result(
    db: AsyncSession,
    dataset_result_id: UUID,
) -> bool:
    """Soft delete evaluation dataset result."""
    db_dataset_result = await get_eval_dataset_result(db, dataset_result_id)
    if not db_dataset_result:
        return False

    db_dataset_result.deleted_at = func.now()
    await db.flush()
    return True


# ========== Evaluation Item CRUD ==========

async def create_eval_item(
    db: AsyncSession,
    eval_item: EvalItemCreate,
) -> EvalItem:
    """Create a new evaluation item."""
    db_eval_item = EvalItem(**eval_item.model_dump())
    db.add(db_eval_item)
    await db.flush()
    await db.refresh(db_eval_item)
    return db_eval_item


async def create_eval_items_bulk(
    db: AsyncSession,
    eval_items: List[EvalItemCreate],
) -> List[EvalItem]:
    """Create multiple evaluation items in bulk."""
    db_eval_items = [EvalItem(**item.model_dump()) for item in eval_items]
    db.add_all(db_eval_items)
    await db.flush()
    return db_eval_items


async def get_eval_item(db: AsyncSession, eval_item_id: UUID) -> Optional[EvalItem]:
    """
    Get evaluation item by ID.

    PHASE 2 UPDATE: Eagerly loads image relationships for file_name/storage_key.
    """
    from sqlalchemy.orm import joinedload

    result = await db.execute(
        select(EvalItem)
        .options(joinedload(EvalItem.image_2d), joinedload(EvalItem.image_3d))
        .where(
            and_(
                EvalItem.id == eval_item_id,
                EvalItem.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_eval_items(
    db: AsyncSession,
    run_id: UUID,
    skip: int = 0,
    limit: int = 100,
    dataset_type: Optional[EvalDatasetType] = None,  # NEW: Filter by EvalDatasetType enum
) -> tuple[List[EvalItem], int]:
    """
    Get list of evaluation items for a run with pagination.

    PHASE 2 UPDATE: Eagerly loads image_2d relationship for file_name/storage_key.
    """
    from sqlalchemy.orm import joinedload

    filters = [
        EvalItem.run_id == run_id,
        EvalItem.deleted_at.is_(None),
    ]

    # NEW: Add dataset_type filter if provided
    if dataset_type:
        filters.append(EvalItem.dataset_type == dataset_type.value)

    # Get total count
    count_query = select(func.count()).select_from(EvalItem).where(and_(*filters))
    total = await db.scalar(count_query)

    # Get items with image relationships
    query = (
        select(EvalItem)
        .options(joinedload(EvalItem.image_2d), joinedload(EvalItem.image_3d))
        .where(and_(*filters))
        .order_by(EvalItem.created_at)
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    items = list(result.scalars().all())

    return items, total or 0


async def get_all_eval_items(
    db: AsyncSession,
    run_id: UUID,
    dataset_type: Optional[EvalDatasetType] = None,
) -> List[EvalItem]:
    """
    Get all evaluation items for a run without pagination.

    PHASE 2 UPDATE: Eagerly loads image_2d relationship for file_name/storage_key.
    """
    from sqlalchemy.orm import joinedload

    filters = [
        EvalItem.run_id == run_id,
        EvalItem.deleted_at.is_(None),
    ]

    # Add dataset_type filter if provided
    if dataset_type:
        filters.append(EvalItem.dataset_type == dataset_type.value)

    # Get all items with image relationships
    query = (
        select(EvalItem)
        .options(joinedload(EvalItem.image_2d), joinedload(EvalItem.image_3d))
        .where(and_(*filters))
        .order_by(EvalItem.created_at)
    )
    result = await db.execute(query)
    items = list(result.scalars().all())

    return items


async def update_eval_item(
    db: AsyncSession,
    eval_item_id: UUID,
    eval_item_update: EvalItemUpdate,
) -> Optional[EvalItem]:
    """Update evaluation item."""
    db_eval_item = await get_eval_item(db, eval_item_id)
    if not db_eval_item:
        return None

    update_data = eval_item_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_eval_item, field, value)

    await db.flush()
    await db.refresh(db_eval_item)
    return db_eval_item


async def delete_eval_item(db: AsyncSession, eval_item_id: UUID) -> bool:
    """Soft delete evaluation item."""
    db_eval_item = await get_eval_item(db, eval_item_id)
    if not db_eval_item:
        return False

    db_eval_item.deleted_at = func.now()
    await db.flush()
    return True


# ========== Evaluation List CRUD ==========

async def create_eval_list(
    db: AsyncSession,
    eval_list: EvalListCreate,
    created_by: Optional[UUID] = None,
) -> EvalList:
    """Create a new evaluation list."""
    db_eval_list = EvalList(
        **eval_list.model_dump(),
        created_by=created_by,
    )
    db.add(db_eval_list)
    await db.flush()
    await db.refresh(db_eval_list)
    return db_eval_list


async def get_eval_list(db: AsyncSession, list_id: UUID) -> Optional[EvalList]:
    """Get evaluation list by ID."""
    result = await db.execute(
        select(EvalList).where(
            and_(
                EvalList.id == list_id,
                EvalList.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_eval_lists(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
) -> tuple[List[EvalList], int]:
    """Get list of evaluation lists with pagination."""
    filters = [EvalList.deleted_at.is_(None)]

    # Get total count
    count_query = select(func.count()).select_from(EvalList).where(and_(*filters))
    total = await db.scalar(count_query)

    # Get items
    query = (
        select(EvalList)
        .where(and_(*filters))
        .order_by(EvalList.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    items = list(result.scalars().all())

    return items, total or 0


async def update_eval_list(
    db: AsyncSession,
    list_id: UUID,
    list_update: EvalListUpdate,
) -> Optional[EvalList]:
    """Update evaluation list."""
    db_eval_list = await get_eval_list(db, list_id)
    if not db_eval_list:
        return None

    update_data = list_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_eval_list, field, value)

    await db.flush()
    await db.refresh(db_eval_list)
    return db_eval_list


async def delete_eval_list(db: AsyncSession, list_id: UUID) -> bool:
    """Soft delete evaluation list."""
    db_eval_list = await get_eval_list(db, list_id)
    if not db_eval_list:
        return False

    db_eval_list.deleted_at = func.now()
    await db.flush()
    return True


# ========== Evaluation List Item CRUD ==========

async def add_run_to_list(
    db: AsyncSession,
    list_item: EvalListItemCreate,
) -> EvalListItem:
    """Add an evaluation run to a list."""
    db_list_item = EvalListItem(**list_item.model_dump())
    db.add(db_list_item)
    await db.flush()
    await db.refresh(db_list_item)
    return db_list_item


async def remove_run_from_list(
    db: AsyncSession,
    list_id: UUID,
    run_id: UUID,
) -> bool:
    """Remove an evaluation run from a list (soft delete)."""
    result = await db.execute(
        select(EvalListItem).where(
            and_(
                EvalListItem.list_id == list_id,
                EvalListItem.run_id == run_id,
                EvalListItem.deleted_at.is_(None),
            )
        )
    )
    db_list_item = result.scalar_one_or_none()
    if not db_list_item:
        return False

    db_list_item.deleted_at = func.now()
    await db.flush()
    return True


async def get_list_items(
    db: AsyncSession,
    list_id: UUID,
) -> List[EvalListItem]:
    """Get all items in an evaluation list."""
    result = await db.execute(
        select(EvalListItem)
        .where(
            and_(
                EvalListItem.list_id == list_id,
                EvalListItem.deleted_at.is_(None),
            )
        )
        .order_by(EvalListItem.sort_order)
    )
    return list(result.scalars().all())


# ========== Comparison Queries ==========

async def get_eval_run_pairs(
    db: AsyncSession,
    model_id: Optional[UUID] = None,
    base_dataset_id: Optional[UUID] = None,
    attack_dataset_id: Optional[UUID] = None,
) -> List[Dict[str, Any]]:
    """
    Get pre/post evaluation run pairs for comparison.
    Uses the eval_run_pairs view.
    """
    filters = []
    if model_id:
        filters.append("model_id = :model_id")
    if base_dataset_id:
        filters.append("base_dataset_id = :base_dataset_id")
    if attack_dataset_id:
        filters.append("attack_dataset_id = :attack_dataset_id")

    where_clause = " AND ".join(filters) if filters else "1=1"
    query = text(f"SELECT * FROM eval_run_pairs WHERE {where_clause}")

    params = {}
    if model_id:
        params["model_id"] = str(model_id)
    if base_dataset_id:
        params["base_dataset_id"] = str(base_dataset_id)
    if attack_dataset_id:
        params["attack_dataset_id"] = str(attack_dataset_id)

    result = await db.execute(query, params)
    return [dict(row._mapping) for row in result.fetchall()]


async def get_eval_run_pairs_delta(
    db: AsyncSession,
    model_id: Optional[UUID] = None,
) -> List[Dict[str, Any]]:
    """
    Get pre/post evaluation run pairs with delta metrics.
    Uses the eval_run_pairs_delta view.
    """
    where_clause = ""
    params = {}

    if model_id:
        where_clause = "WHERE model_id = :model_id"
        params["model_id"] = str(model_id)

    query = text(f"SELECT * FROM eval_run_pairs_delta {where_clause}")
    result = await db.execute(query, params)
    return [dict(row._mapping) for row in result.fetchall()]


# ==================== EvalClassMetrics CRUD ====================

async def create_eval_class_metrics(
    db: AsyncSession,
    obj_in: EvalClassMetricsCreate
) -> EvalClassMetrics:
    """Create a new per-class metrics record."""
    from app.models.evaluation import EvalClassMetrics
    # Exclude None values to prevent JSON 'null' from being stored in JSONB columns
    # which would violate the check constraint (must be NULL or object type)
    db_obj = EvalClassMetrics(**obj_in.model_dump(exclude_none=True))
    db.add(db_obj)
    await db.flush()
    await db.refresh(db_obj)
    return db_obj


async def create_eval_class_metrics_bulk(
    db: AsyncSession,
    objs_in: List[EvalClassMetricsCreate]
) -> List[EvalClassMetrics]:
    """Create multiple per-class metrics records in bulk."""
    from app.models.evaluation import EvalClassMetrics
    # Exclude None values to prevent JSON 'null' from being stored in JSONB columns
    db_objs = [EvalClassMetrics(**obj_in.model_dump(exclude_none=True)) for obj_in in objs_in]
    db.add_all(db_objs)
    await db.flush()
    for obj in db_objs:
        await db.refresh(obj)
    return db_objs


async def get_eval_class_metrics(
    db: AsyncSession,
    id: UUID
) -> Optional[EvalClassMetrics]:
    """Get per-class metrics by ID."""
    from app.models.evaluation import EvalClassMetrics
    result = await db.execute(
        select(EvalClassMetrics).where(
            EvalClassMetrics.id == id,
            EvalClassMetrics.deleted_at.is_(None)
        )
    )
    return result.scalar_one_or_none()


async def get_eval_class_metrics_by_run(
    db: AsyncSession,
    run_id: UUID,
    dataset_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[EvalClassMetrics]:
    """Get all per-class metrics for a specific evaluation run."""
    from app.models.evaluation import EvalClassMetrics, EvalDatasetType
    query = select(EvalClassMetrics).where(
        EvalClassMetrics.run_id == run_id,
        EvalClassMetrics.deleted_at.is_(None)
    )

    if dataset_type:
        query = query.where(EvalClassMetrics.dataset_type == EvalDatasetType(dataset_type))

    query = query.order_by(EvalClassMetrics.class_name).offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def update_eval_class_metrics(
    db: AsyncSession,
    id: UUID,
    obj_in: EvalClassMetricsUpdate
) -> Optional[EvalClassMetrics]:
    """Update per-class metrics."""
    from app.models.evaluation import EvalClassMetrics
    db_obj = await get_eval_class_metrics(db, id)
    if not db_obj:
        return None

    update_data = obj_in.model_dump(exclude_unset=True)
    # Filter out None values for JSONB fields to prevent JSON 'null' constraint violations
    # If you want to clear a JSONB field, don't include it in the update
    update_data = {k: v for k, v in update_data.items() if v is not None or k not in ['metrics', 'iou_distribution']}
    for field, value in update_data.items():
        setattr(db_obj, field, value)

    await db.flush()
    await db.refresh(db_obj)
    return db_obj


async def delete_eval_class_metrics(
    db: AsyncSession,
    id: UUID
) -> bool:
    """Soft delete per-class metrics."""
    from app.models.evaluation import EvalClassMetrics
    db_obj = await get_eval_class_metrics(db, id)
    if not db_obj:
        return False

    db_obj.deleted_at = func.now()
    await db.flush()
    return True
