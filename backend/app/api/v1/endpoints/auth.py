"""
Authentication endpoints.

SOLID Improvements:
- SRP: Endpoints delegate business logic to AuthService
- DIP: Depends on AuthService abstraction
- DRY: Eliminates code duplication between login endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.user import UserCreate, UserResponse, Token, UserLogin
from app.crud.user import create_user, get_user_by_email
from app.services.auth_service import auth_service

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user (using username as unique identifier).
    """
    # Check if username already exists
    from app.crud.user import get_user_by_username
    existing_user = await get_user_by_username(db, user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Create new user
    user = await create_user(db, user_create)

    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Login with form data and get access token.

    Uses AuthService for authentication business logic.
    """
    return await auth_service.authenticate_and_create_token(
        db, form_data.username, form_data.password
    )


@router.post("/login-json", response_model=Token)
async def login_json(
    user_login: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with JSON body and get access token (using username).

    Uses AuthService for authentication business logic.
    """
    return await auth_service.authenticate_and_create_token(
        db, user_login.username, user_login.password
    )
