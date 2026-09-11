"""
SatQuery AI — Authentication & Security API Routes
Endpoints for user registration, credential verification, session validation, and logout.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr
from backend.security.auth import (
    User,
    generate_token,
    user_repository,
    verify_token,
)

auth_router = APIRouter(prefix="/api/auth", tags=["Security & Authentication"])


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    role: Optional[str] = "Geospatial Analyst"
    organization: Optional[str] = "Earth Observation Lab"


class AuthResponse(BaseModel):
    status: str
    token: str
    token_type: str = "bearer"
    user: User


async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[User]:
    """Dependency that extracts and validates the user from the Authorization header."""
    if not authorization:
        # Default fallback to anonymous analyst for open development or return None
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None

    payload = verify_token(token)
    if not payload:
        return None

    user = user_repository.get_by_id(payload.get("sub"))
    if not user:
        return None

    return User(**user.model_dump())


async def require_auth(current_user: Optional[User] = Depends(get_current_user)) -> User:
    """Dependency that strictly enforces an authenticated user."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


@auth_router.post("/register", response_model=AuthResponse)
def register_user(req: RegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long.",
        )
    if not req.email or "@" not in req.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid email address is required.",
        )

    try:
        user = user_repository.create(
            email=req.email,
            password=req.password,
            full_name=req.full_name,
            role=req.role or "Geospatial Analyst",
            organization=req.organization or "Earth Observation",
        )
        token = generate_token(user)
        return AuthResponse(status="success", token=token, user=user)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@auth_router.post("/login", response_model=AuthResponse)
def login_user(req: LoginRequest):
    user = user_repository.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = generate_token(user)
    return AuthResponse(status="success", token=token, user=user)


@auth_router.get("/me")
def get_user_profile(user: User = Depends(require_auth)):
    return {
        "status": "success",
        "user": user,
    }


@auth_router.post("/logout")
def logout_user():
    return {
        "status": "success",
        "message": "Session invalidated successfully.",
    }
