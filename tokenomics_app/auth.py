"""
Authentication module for Tokenomics App.

Password stored as PBKDF2-HMAC-SHA256 hash (600 000 iterations) — never in plain text.
Sessions stored server-side in memory; tokens are 256-bit URL-safe random strings.
"""

import hashlib
import secrets
import time
from typing import Optional

# ── Credentials (only the hash is stored, never the plain-text password) ──
_ADMIN_USERNAME: str = "admin"
_SALT: str = "fc3ef034df7443c1de5774d880d32709"
_PASSWORD_HASH: str = "5445f8e8a5f90964a4f2f0223e58a44c3a8828a999759dcdfe65e48382f26133"
_ITERATIONS: int = 600_000

# ── Session store: { token -> expiry_unix_timestamp } ──
_sessions: dict[str, float] = {}
_SESSION_TTL: int = 86_400   # 24 hours
COOKIE_NAME: str = "tkn_session"  # exported so main.py can import it


def verify_credentials(username: str, password: str) -> bool:
    """Constant-time credential check — resistant to timing attacks."""
    username_ok = secrets.compare_digest(
        username.lower().strip(), _ADMIN_USERNAME
    )
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        _SALT.encode("utf-8"),
        _ITERATIONS,
    )
    password_ok = secrets.compare_digest(dk.hex(), _PASSWORD_HASH)
    # Evaluate BOTH checks so timing doesn't leak which one failed.
    return username_ok and password_ok


def create_session() -> str:
    """Generate a new session token and store it with an expiry."""
    token = secrets.token_urlsafe(32)   # 256-bit entropy
    _sessions[token] = time.time() + _SESSION_TTL
    _cleanup_sessions()
    return token


def verify_session(token: Optional[str]) -> bool:
    """Return True if the token is valid and not expired."""
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if time.time() > expiry:
        del _sessions[token]
        return False
    return True


def delete_session(token: str) -> None:
    """Invalidate a session (logout)."""
    _sessions.pop(token, None)


def _cleanup_sessions() -> None:
    """Purge expired sessions to prevent unbounded memory growth."""
    now = time.time()
    expired = [t for t, exp in list(_sessions.items()) if now > exp]
    for t in expired:
        del _sessions[t]
