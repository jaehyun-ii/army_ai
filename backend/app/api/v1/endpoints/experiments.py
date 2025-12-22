"""
Experiment API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from uuid import UUID

from app.database import get_db
from app.crud.experiment import experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
)
from app.models.experiment import ExperimentStatus

router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    experiment_in: ExperimentCreate,
    db: Session = Depends(get_db),
):
    """Create a new experiment."""
    return experiment.create(db, obj_in=experiment_in)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Get an experiment by ID."""
    db_experiment = experiment.get(db, id=experiment_id)
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return db_experiment


@router.get("/", response_model=List[ExperimentResponse])
def list_experiments(
    skip: int = 0,
    limit: int = 100,
    status_filter: ExperimentStatus = None,
    db: Session = Depends(get_db),
):
    """List all experiments with optional status filter."""
    if status_filter:
        return experiment.get_by_status(db, status=status_filter, skip=skip, limit=limit)
    return experiment.get_multi(db, skip=skip, limit=limit)


@router.put("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment_put(
    experiment_id: UUID,
    experiment_update: ExperimentUpdate,
    db: Session = Depends(get_db),
):
    """Update an experiment (PUT method)."""
    db_experiment = experiment.get(db, id=experiment_id)
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return experiment.update(db, db_obj=db_experiment, obj_in=experiment_update)


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment_patch(
    experiment_id: UUID,
    experiment_update: ExperimentUpdate,
    db: Session = Depends(get_db),
):
    """Update an experiment (PATCH method - frontend compatible)."""
    db_experiment = experiment.get(db, id=experiment_id)
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return experiment.update(db, db_obj=db_experiment, obj_in=experiment_update)


@router.post("/{experiment_id}/start", response_model=ExperimentResponse)
def start_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Start an experiment."""
    db_experiment = experiment.start_experiment(db, experiment_id=experiment_id)
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start experiment (not in draft status or not found)",
        )
    return db_experiment


@router.post("/{experiment_id}/complete", response_model=ExperimentResponse)
def complete_experiment(
    experiment_id: UUID,
    results_summary: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
):
    """Complete an experiment with results summary."""
    db_experiment = experiment.complete_experiment(
        db, experiment_id=experiment_id, results_summary=results_summary
    )
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot complete experiment (not in running status or not found)",
        )
    return db_experiment


@router.post("/{experiment_id}/fail", response_model=ExperimentResponse)
def fail_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Mark an experiment as failed."""
    db_experiment = experiment.fail_experiment(db, experiment_id=experiment_id)
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot fail experiment (not in running status or not found)",
        )
    return db_experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete an experiment (soft delete)."""
    db_experiment = experiment.get(db, id=experiment_id)
    if not db_experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    experiment.remove(db, id=experiment_id)
    return None
