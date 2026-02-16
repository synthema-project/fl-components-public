import os
from typing import List
from keycloak import KeycloakOpenID
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwcrypto.jws import InvalidJWSSignature, InvalidJWSObject
from jwcrypto.jwt import JWTExpired
from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status



#KEYCLOAK_SERVER_URL=os.getenv("KEYCLOAK_SERVER_URL", "https://users.k8s.synthema.rid-intrasoft.eu" )
#KEYCLOAK_CLIENT_ID=os.getenv("KEYCLOAK_CLIENT_ID", "synthema")
#KEYCLOAK_REALM=os.getenv("KEYCLOAK_REALM", "Synthema")

KEYCLOAK_SERVER_URL="https://users.k8s.synthema.rid-intrasoft.eu"#os.getenv("KEYCLOAK_SERVER_URL", "https://users.k8s.synthema.rid-intrasoft.eu" )
KEYCLOAK_CLIENT_ID="synthema"#os.getenv("KEYCLOAK_CLIENT_ID", "synthema")
KEYCLOAK_REALM="Synthema"#os.getenv("KEYCLOAK_REALM", "Synthema")

keycloak_openid = KeycloakOpenID(server_url=KEYCLOAK_SERVER_URL,
                                 client_id=KEYCLOAK_CLIENT_ID,
                                 realm_name=KEYCLOAK_REALM)

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

oauth2_scheme = HTTPBearer()

def get_user_data_from_token(token: str) -> UserClaims:
    decoded_token = keycloak_openid.decode_token(token,
                                                 #key=public_key,
                                                 validate=True
                                                )
    user_claims = UserClaims(**decoded_token)
    return user_claims


async def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)]) -> UserClaims:
    token = credentials.credentials
    try:
        user = get_user_data_from_token(token)
    except (JWTExpired, InvalidJWSSignature, InvalidJWSObject):
        user = None

    if not user:
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
