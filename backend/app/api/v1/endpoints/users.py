from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.user import UserResponse, UserUpdate
from app.crud import user as crud_user
from app.models.user import User
from app.services.auth_service import get_current_active_admin

router = APIRouter()

@router.get("/", response_model=List[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """
    Retrieve users.
    """
    users = await crud_user.get_users(db, skip=skip, limit=limit)
    return users

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user_status(
    user_id: UUID,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """
    Update a user's status.
    """
    user = await crud_user.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    user = await crud_user.update_user(db=db, user=user, user_update=user_in)
    return user

@router.delete("/{user_id}", response_model=UserResponse)
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin)
):
    """
    Delete a user.
    """
    user = await crud_user.delete_user(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user
