import logging
import os
import threading
from typing import List
from keycloak import KeycloakOpenID
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt as jose_jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError
from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status
from restapi.config import (
    AUTH_DISABLE_TESTS,
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL,
)

logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var (avoids the "False"-string-is-truthy trap)."""
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Token validation tuning (all optional, safe defaults) ------------------
# DEV ESCAPE HATCH: when true, tokens are decoded WITHOUT verifying signature
# or expiry (real claims still flow through). Defaults to False so production,
# which does not set it, stays fully verified. Set it only in dev values files.
AUTH_DISABLE_VERIFICATION = _env_bool("AUTH_DISABLE_VERIFICATION")
# Expected audience. Leave UNSET until the Keycloak audience mapper is in place
# (dashboard tokens currently carry aud="account"). Once the mapper adds the
# API as an audience, set this to the API client id (e.g. "synthema-fl-restapi")
# and audience validation is enforced automatically.
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE")
# Verify the token issuer only when explicitly configured, because in-cluster
# the API may reach Keycloak through an internal URL while tokens carry the
# public issuer (identity.gatv.es). Opt-in avoids a URL split breaking auth.
KEYCLOAK_EXPECTED_ISSUER = os.getenv("KEYCLOAK_EXPECTED_ISSUER")
# Clock-skew tolerance (seconds) for exp/nbf. Access tokens are short-lived
# (~300s), so even small skew between the API host and Keycloak can wrongly
# reject fresh tokens. Matches python-keycloak's own default of 60s.
KEYCLOAK_LEEWAY = int(os.getenv("KEYCLOAK_LEEWAY", "60"))

keycloak_openid = KeycloakOpenID(
    server_url=KEYCLOAK_SERVER_URL,
    client_id=KEYCLOAK_CLIENT_ID,
    realm_name=KEYCLOAK_REALM_NAME,
    verify=False,
)

# Cached JWKS (signing keys) fetched from Keycloak. Keycloak rotates its signing
# keys, so we refresh the cache whenever a token references a `kid` we have not
# seen yet instead of pinning a single hardcoded public key.
_jwks_lock = threading.Lock()
_jwks_cache: Optional[dict] = None


def _fetch_jwks(force_refresh: bool = False) -> dict:
    """Return the realm JWKS, fetching from Keycloak on first use / refresh."""
    global _jwks_cache
    with _jwks_lock:
        if _jwks_cache is None or force_refresh:
            _jwks_cache = keycloak_openid.certs()
        return _jwks_cache


def _find_key(jwks: dict, kid: str) -> Optional[dict]:
    for key in jwks.get("keys", []):
        if key.get("kid") == kid and key.get("use", "sig") == "sig":
            return key
    return None


def _get_signing_key(kid: str) -> Optional[dict]:
    """Find the signing JWK matching `kid`, refreshing the cache on a miss.

    A miss usually means Keycloak rotated its signing keys, so we refetch the
    JWKS once before giving up.
    """
    key = _find_key(_fetch_jwks(), kid)
    if key is None:
        key = _find_key(_fetch_jwks(force_refresh=True), kid)
    return key

class UserClaims(BaseModel):
    exp: int
    iat: int
    jti: str
    iss: str
    sub: str
    typ: str
    azp: str
    session_state: Optional[str] = None
    scope: str
    sid: Optional[str] = None
    synthema_roles: List[str] = Field(alias="synthemaRoles", default_factory=list)
    name: str = Field(alias="given_name", default="")
    last_name: str = Field(alias="family_name", default="")
    username: str = Field(alias="preferred_username", default="")

    def has_organization_role(self, organization, role) -> bool:
        for syn_role in self.synthema_roles:
            org, rol = syn_role.split(":")
            if role == rol and organization == org:
                return True

        return False

    def has_role(self, role) -> bool:
        for syn_role in self.synthema_roles:
            org, rol = syn_role.split(":")
            if role == rol:
                return True

        return False

oauth2_scheme = HTTPBearer(auto_error=False)


def get_user_data_from_token(token: str) -> UserClaims:
    """Verify a Keycloak access token and return its claims.

    The signing key is resolved from the realm JWKS by the token's ``kid`` so
    Keycloak key rotation is handled automatically. Signature and expiry are
    always verified. Audience/issuer checks are opt-in (see the KEYCLOAK_*
    settings above) because dashboard tokens currently carry ``aud="account"``
    until the Keycloak audience mapper is configured.
    """
    if AUTH_DISABLE_VERIFICATION:
        logger.warning(
            "AUTH_DISABLE_VERIFICATION is set - accepting token WITHOUT "
            "signature/expiry verification. DO NOT use in production."
        )
        return UserClaims(**jose_jwt.get_unverified_claims(token))

    unverified_header = jose_jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise JWTError("Token header is missing a 'kid'")

    signing_key = _get_signing_key(kid)
    if signing_key is None:
        raise JWTError(f"No signing key found for kid={kid!r}")

    options = {
        "verify_aud": bool(KEYCLOAK_AUDIENCE),
        "verify_at_hash": False,
        "leeway": KEYCLOAK_LEEWAY,
    }
    decoded_token = jose_jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        options=options,
        audience=KEYCLOAK_AUDIENCE if KEYCLOAK_AUDIENCE else None,
        issuer=KEYCLOAK_EXPECTED_ISSUER if KEYCLOAK_EXPECTED_ISSUER else None,
    )

    user_claims = UserClaims(**decoded_token)
    return user_claims


def get_mock_user() -> UserClaims:
    return UserClaims(
            exp=9999999999,
            iat=0,
            jti="test",
            iss="test",
            sub="test-user",
            typ="Bearer",
            azp="test",
            session_state="test",
            scope="openid",
            sid="test",
            synthemaRoles=["test:Admin"],
            given_name="E2E",
            family_name="Test",
            preferred_username="e2e-test",
        )


async def get_current_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(oauth2_scheme),
    ],
) -> UserClaims:

    # E2E auth bypass
    print("AUTH_DISABLE_TESTS =", AUTH_DISABLE_TESTS)
    if AUTH_DISABLE_TESTS:
        print(
            "AUTH_DISABLE_TESTS detected - bypassing authentication"
        )

        return get_mock_user()

    # Normal authentication flow
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        user = get_user_data_from_token(token)

    except (
        ExpiredSignatureError,
        JWTClaimsError,
        JWTError,
    ) as e:

        # Log the concrete failure so 401s are diagnosable (expired vs bad
        # signature vs unknown kid vs audience/issuer mismatch).
        logger.error(
            "JWT validation failed: %s: %s", type(e).__name__, str(e)
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {type(e).__name__}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def require_authentication(current_user: Optional[UserClaims] = Depends(get_current_user)):
    """Require user to be authenticated"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return current_user
