import os
from typing import List
from keycloak import KeycloakOpenID
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwcrypto.jws import InvalidJWSSignature, InvalidJWSObject
from jwcrypto.jwt import JWTExpired
from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status
from restapi.config import AUTH_DISABLE_TESTS



KEYCLOAK_SERVER_URL=os.getenv("KEYCLOAK_SERVER_URL", "https://users.k8s.synthema.rid-intrasoft.eu" )
KEYCLOAK_CLIENT_ID=os.getenv("KEYCLOAK_CLIENT_ID", "synthema")
KEYCLOAK_REALM=os.getenv("KEYCLOAK_REALM", "Synthema")

keycloak_openid = KeycloakOpenID(server_url=KEYCLOAK_SERVER_URL,client_id=KEYCLOAK_CLIENT_ID,realm_name=KEYCLOAK_REALM,verify=False)

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
    sid: str
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

public_key= """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAw0ycG6p5R0qM37giBV1XejJBoweEOgkmP/U8kjLUKcpbPYIXuTBb2TMCK3p7vHJ8GQokco+qiBd4qiOx4B+fy0qeleNoOWw2jnTpOUdExzjANrGDfh8OB3tmoZUd1yHbfLtnF2A47vrO2KKX5E9zFrZ5pjFaDA8806X1qCGnecBPv9Oi52lZLpH0r/e5SyPfAIOAO/+DT8OPxlizShGlzg5R+Ll1nsJgipBexs1peKgOTrYBmdsgP1SiaCiVamf9xIcghWAX2LPZd2AjynBNyux9ZaDq34lckA20ybpBe+QNpmYvpQScRJNNTqJ3UqD8NkV1W3C1I6M87+pUTeLFnQIDAQAB-----END PUBLIC KEY-----
"""

def get_user_data_from_token(token: str) -> UserClaims:
    decoded_token = keycloak_openid.decode_token(token,
                                                 key=public_key,
                                                 #key=None,
                                                 validate=False
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
        JWTExpired,
        InvalidJWSSignature,
        InvalidJWSObject,
    ) as e:

        logger.error("JWT validation error: %s", str(e))

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
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
