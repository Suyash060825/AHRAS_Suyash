from __future__ import annotations
"""
AHRAS Authentication & Token Management
----------------------------------------
Implements secure password hashing, JWT token creation, and user management.
"""

import os
import time
import hmac
import hashlib
import base64
import json
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

SECRET_KEY = os.getenv("AHRAS_SECRET_KEY", "ahras-enterprise-secret-key-2026-production-hardening")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("AHRAS_TOKEN_EXPIRE_MINUTES", "480"))


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _b64decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode('utf-8'))


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """PBKDF2-HMAC-SHA256 password hash with salt."""
    if salt is None:
        salt = base64.b64encode(os.urandom(16)).decode('utf-8')
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"pbkdf2:sha256:100000${salt}${base64.b64encode(key).decode('utf-8')}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored PBKDF2 hash."""
    try:
        parts = hashed_password.split('$')
        if len(parts) != 3:
            return False
        salt = parts[1]
        expected_hash = hash_password(plain_password, salt=salt)
        return hmac.compare_digest(expected_hash, hashed_password)
    except Exception as e:
        log.error(f"[AUTH] Password verification error: {e}")
        return False


# In-memory user database
DEFAULT_USERS: Dict[str, Dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "email": "admin@ahras.security",
        "hashed_password": hash_password("AdminSecurePass2026!"),
        "role": "admin",
        "is_active": True,
    },
    "analyst": {
        "username": "analyst",
        "email": "analyst@ahras.security",
        "hashed_password": hash_password("AnalystPass2026!"),
        "role": "soc_analyst",
        "is_active": True,
    },
    "hunter": {
        "username": "hunter",
        "email": "hunter@ahras.security",
        "hashed_password": hash_password("HunterPass2026!"),
        "role": "threat_hunter",
        "is_active": True,
    },
    "responder": {
        "username": "responder",
        "email": "responder@ahras.security",
        "hashed_password": hash_password("ResponderPass2026!"),
        "role": "incident_responder",
        "is_active": True,
    },
    "manager": {
        "username": "manager",
        "email": "manager@ahras.security",
        "hashed_password": hash_password("ManagerPass2026!"),
        "role": "manager",
        "is_active": True,
    },
}

_user_db = dict(DEFAULT_USERS)


def get_user(username: str) -> Optional[Dict[str, Any]]:
    return _user_db.get(username)


def list_users() -> List[Dict[str, Any]]:
    return [
        {
            "username": u["username"],
            "email": u["email"],
            "role": u["role"],
            "is_active": u["is_active"],
        }
        for u in _user_db.values()
    ]


def register_user(username: str, password: str, role: str, email: str = "") -> Dict[str, Any]:
    if username in _user_db:
        raise ValueError(f"User '{username}' already exists")
    user_record = {
        "username": username,
        "email": email or f"{username}@ahras.security",
        "hashed_password": hash_password(password),
        "role": role,
        "is_active": True,
    }
    _user_db[username] = user_record
    return {"username": username, "email": user_record["email"], "role": role, "is_active": True}


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if not user.get("is_active", True):
        return None
    return user


def create_access_token(data: dict, expires_delta_seconds: Optional[int] = None) -> str:
    """Generates standard HMAC-SHA256 JWT access token."""
    to_encode = data.copy()
    now = int(time.time())
    if expires_delta_seconds:
        expire = now + expires_delta_seconds
    else:
        expire = now + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        
    to_encode.update({"iat": now, "exp": expire})
    
    header = {"alg": ALGORITHM, "typ": "JWT"}
    encoded_header = _b64encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    encoded_payload = _b64encode(json.dumps(to_encode, separators=(',', ':')).encode('utf-8'))
    
    signing_input = f"{encoded_header}.{encoded_payload}".encode('utf-8')
    signature = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    encoded_signature = _b64encode(signature)
    
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decodes a JWT token."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        encoded_header, encoded_payload, encoded_signature = parts
        
        signing_input = f"{encoded_header}.{encoded_payload}".encode('utf-8')
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        
        if not hmac.compare_digest(_b64encode(expected_sig), encoded_signature):
            return None
            
        payload_bytes = _b64decode(encoded_payload)
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        # Verify expiration
        exp = payload.get("exp")
        if exp and int(exp) < int(time.time()):
            return None
            
        return payload
    except Exception as e:
        log.debug(f"[AUTH] Token verification failed: {e}")
        return None
