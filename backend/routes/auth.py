import logging
import bcrypt
from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

from backend.db import users_collection, is_mongo_available
from backend.auth import generate_token
from backend.schemas import RegisterRequest, LoginRequest, AuthResponse

logger = logging.getLogger("driveiq.routes.auth")
auth_router = APIRouter(prefix="/api/v1/auth")

@auth_router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> dict:
    email = payload.email.strip().lower()
    password = payload.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    mongo_ok, mongo_error = is_mongo_available()
    if not mongo_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Database unavailable",
                "message": "MongoDB is not reachable. Start MongoDB and retry.",
                "details": mongo_error,
            }
        )

    try:
        if users_collection.find_one({"email": email}):
            raise HTTPException(status_code=400, detail="User exists")

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user_id = users_collection.insert_one({"email": email, "password": hashed}).inserted_id

        token = generate_token(str(user_id))
        return {"token": token}
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="User exists")
    except ServerSelectionTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Database unavailable",
                "message": "MongoDB connection timed out.",
                "details": str(e),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Registration failed: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Registration service unavailable")


@auth_router.post("/login", response_model=AuthResponse, status_code=status.HTTP_200_OK)
def login(payload: LoginRequest) -> dict:
    email = payload.email.strip().lower()
    password = payload.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    mongo_ok, mongo_error = is_mongo_available()
    if not mongo_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Database unavailable",
                "message": "MongoDB is not reachable. Start MongoDB and retry.",
                "details": mongo_error,
            }
        )

    try:
        user = users_collection.find_one({"email": email})
        if not user or not bcrypt.checkpw(password.encode(), user["password"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        token = generate_token(str(user["_id"]))
        return {"token": token}
    except ServerSelectionTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Database unavailable",
                "message": "MongoDB connection timed out.",
                "details": str(e),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Login failed: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Login service unavailable")
