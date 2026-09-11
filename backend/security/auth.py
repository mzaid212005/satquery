"""
SatQuery AI — Security & Authentication Subsystem
Provides cryptographic password hashing (PBKDF2-SHA256), signed session tokens,
role-based user management, and persistent storage.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

SECRET_KEY = os.environ.get("SATQUERY_SECRET_KEY", "satquery_super_secret_ai_key_2026_geospatial_secure")
TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days
USERS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "users.json"


class User(BaseModel):
    id: str
    email: str
    full_name: str
    role: str = "Analyst"  # Administrator, Geospatial Analyst, Researcher, Farmer
    organization: str = "SatQuery Earth Observation"
    is_active: bool = True
    created_at: float


class UserInDB(User):
    password_hash: str
    salt: str


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    key = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000)
    return key.hex(), salt


def verify_password(plain_password: str, password_hash: str, salt: str) -> bool:
    """Verifies a plain password against stored hash and salt."""
    key, _ = hash_password(plain_password, salt=salt)
    return hmac.compare_digest(key, password_hash)


def generate_token(user: User) -> str:
    """Generates a cryptographic HMAC-SHA256 signed session token."""
    payload = {
        "sub": user.id,
        "email": user.email,
        "name": user.full_name,
        "role": user.role,
        "org": user.organization,
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS,
        "iat": int(time.time()),
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decodes a signed session token."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        
        # Verify HMAC signature
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        
        # Add padding back if necessary
        padding = "=" * (4 - len(payload_b64) % 4) if len(payload_b64) % 4 != 0 else ""
        payload_json = base64.urlsafe_b64decode((payload_b64 + padding).encode("utf-8")).decode("utf-8")
        payload = json.loads(payload_json)
        
        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None
            
        return payload
    except Exception:
        return None


class UserRepository:
    """Persistent storage for registered users."""

    def __init__(self, storage_file: Path = USERS_FILE):
        self.storage_file = storage_file
        self._mem_db = None
        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._ensure_seed_users()

    def _load_db(self) -> Dict[str, Dict[str, Any]]:
        if self._mem_db is not None:
            return self._mem_db
        if not self.storage_file.exists():
            return {}
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_db(self, data: Dict[str, Dict[str, Any]]):
        self._mem_db = data
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Fall back to in-memory on read-only environments like Vercel Lambda
            pass

    def _ensure_seed_users(self):
        db = self._load_db()
        changed = False

        # Seed 1: Administrator
        if "admin@satquery.ai" not in db:
            pwd_hash, salt = hash_password("Admin@1234")
            db["admin@satquery.ai"] = {
                "id": "usr_admin_001",
                "email": "admin@satquery.ai",
                "full_name": "SatQuery System Administrator",
                "role": "Administrator",
                "organization": "SatQuery AI Core Lab",
                "is_active": True,
                "created_at": time.time(),
                "password_hash": pwd_hash,
                "salt": salt,
            }
            changed = True

        # Seed 2: Geospatial Analyst
        if "analyst@satquery.ai" not in db:
            pwd_hash, salt = hash_password("Analyst@1234")
            db["analyst@satquery.ai"] = {
                "id": "usr_analyst_002",
                "email": "analyst@satquery.ai",
                "full_name": "Dr. Elena Rostova",
                "role": "Geospatial Analyst",
                "organization": "Earth Observation Research",
                "is_active": True,
                "created_at": time.time(),
                "password_hash": pwd_hash,
                "salt": salt,
            }
            changed = True

        if changed:
            self._save_db(db)

    def get_by_email(self, email: str) -> Optional[UserInDB]:
        db = self._load_db()
        raw = db.get(email.lower().strip())
        if not raw:
            return None
        return UserInDB(**raw)

    def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        db = self._load_db()
        for raw in db.values():
            if raw.get("id") == user_id:
                return UserInDB(**raw)
        return None

    def create(self, email: str, password: str, full_name: str, role: str = "Analyst", organization: str = "SatQuery EO") -> User:
        email_clean = email.lower().strip()
        db = self._load_db()
        if email_clean in db:
            raise ValueError(f"User with email '{email_clean}' already exists.")

        user_id = f"usr_{secrets.token_hex(6)}"
        pwd_hash, salt = hash_password(password)

        record = {
            "id": user_id,
            "email": email_clean,
            "full_name": full_name.strip(),
            "role": role.strip(),
            "organization": organization.strip() or "Earth Observation",
            "is_active": True,
            "created_at": time.time(),
            "password_hash": pwd_hash,
            "salt": salt,
        }
        db[email_clean] = record
        self._save_db(db)
        return User(**record)

    def authenticate(self, email: str, password: str) -> Optional[User]:
        user_db = self.get_by_email(email)
        if not user_db or not user_db.is_active:
            return None
        if not verify_password(password, user_db.password_hash, user_db.salt):
            return None
        return User(**user_db.model_dump())

    def get_or_create_google_user(self, email: str, full_name: str, picture: Optional[str] = None) -> User:
        email_clean = email.lower().strip()
        db = self._load_db()
        if email_clean in db:
            raw = db[email_clean]
            return User(**raw)

        user_id = f"usr_g_{secrets.token_hex(6)}"
        pwd_hash, salt = hash_password(secrets.token_hex(16))

        record = {
            "id": user_id,
            "email": email_clean,
            "full_name": full_name.strip() or "Google User",
            "role": "Geospatial Analyst",
            "organization": "Google Earth Observation",
            "is_active": True,
            "created_at": time.time(),
            "password_hash": pwd_hash,
            "salt": salt,
        }
        db[email_clean] = record
        self._save_db(db)
        return User(**record)


user_repository = UserRepository()
