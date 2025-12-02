import os
from typing import List, Optional, Annotated

from keycloak import KeycloakOpenID
from pydantic import BaseModel, Field
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
from jwcrypto.jws import InvalidJWSSignature, InvalidJWSObject
from jwcrypto.jwt import JWTExpired

# Environment variables
KEYCLOAK_SERVER_URL = os.getenv(
    "KEYCLOAK_SERVER_URL",
    "https://users.k8s.synthema.rid-intrasoft.eu"
)
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "synthema")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "Synthema")

# Keycloak client
keycloak_openid = KeycloakOpenID(
    server_url=KEYCLOAK_SERVER_URL,
    client_id=KEYCLOAK_CLIENT_ID,
    realm_name=KEYCLOAK_REALM
)

# User claims model
class UserClaims(BaseModel):
    exp: int
    iat: int
    jti: str
    iss: str
    sub: str
    typ: str
    azp: str
    session_state: str
    scope: str
    sid: str
    synthema_roles: List[str] = Field(alias="synthemaRoles", default_factory=list)
    name: str = Field(alias="firstName")
    last_name: str = Field(alias="lastName")
    username: str

    def has_organization_role(self, organization: str, role: str) -> bool:
        for syn_role in self.synthema_roles:
            org, rol = syn_role.split(":")
            if organization == org and role == rol:
                return True
        return False

    def has_role(self, role: str) -> bool:
        for syn_role in self.synthema_roles:
            _, rol = syn_role.split(":")
            if role == rol:
                return True
        return False


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_user_data_from_token(token: str) -> UserClaims:
    decoded_token = keycloak_openid.decode_token(token, validate=True)
    return UserClaims(**decoded_token)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> UserClaims:
    try:
        user = get_user_data_from_token(token)
    except (JWTExpired, InvalidJWSSignature, InvalidJWSObject):
        user = None

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


async def require_authentication(
    current_user: Optional[UserClaims] = Depends(get_current_user)
):
    """Require user to be authenticated"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return current_user
