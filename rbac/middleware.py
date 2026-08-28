from __future__ import annotations
"""
AHRAS RBAC Middleware & FastAPI Dependency Guards
--------------------------------------------------
Provides zero-boilerplate dependency factories for route protection.
"""

from typing import Callable, Set
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from rbac.permissions import Perm, Role, ROLE_PERMISSIONS, role_has_permission

security_scheme = HTTPBearer(auto_error=False)


def get_user_permissions(role_name: str) -> Set[str]:
    try:
        r = Role(role_name.lower())
        return {p.value for p in ROLE_PERMISSIONS.get(r, set())}
    except ValueError:
        return set()


def require_permission(required_perm: Perm) -> Callable:
    """
    Factory creating a FastAPI dependency that checks whether the current
    user's token/role possesses `required_perm`.
    """
    def _dependency(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
        from auth.manager import verify_token
        
        # If no auth header supplied in open dev mode, allow fallback or require auth
        if credentials is None:
            # Check if dev token bypass is enabled or raise 401
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials required",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        token = credentials.credentials
        user = verify_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        role = user.get("role", "soc_analyst")
        if not role_has_permission(role, required_perm):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: role '{role}' lacks required permission '{required_perm.value}'",
            )
        return user

    return _dependency
