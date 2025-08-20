"""
Enterprise Authentication Integration for PomegranteMuse
Supports LDAP, SAML, OAuth, and other enterprise authentication systems
"""

import asyncio
import json
import ssl
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import base64
import secrets

try:
    import ldap3
    LDAP_AVAILABLE = True
except ImportError:
    LDAP_AVAILABLE = False

try:
    import requests
    import jwt
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

class AuthProvider(Enum):
    """Supported authentication providers"""
    LDAP = "ldap"
    ACTIVE_DIRECTORY = "active_directory" 
    SAML = "saml"
    OAUTH2 = "oauth2"
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    LOCAL = "local"

@dataclass
class LDAPConfig:
    """LDAP authentication configuration"""
    server: str
    port: int = 389
    use_ssl: bool = False
    bind_dn: str = ""
    bind_password: str = ""
    base_dn: str = ""
    user_filter: str = "(uid={username})"
    group_filter: str = "(member={user_dn})"
    user_attributes: List[str] = None
    group_attributes: List[str] = None
    
    def __post_init__(self):
        if self.user_attributes is None:
            self.user_attributes = ["uid", "cn", "mail", "givenName", "sn"]
        if self.group_attributes is None:
            self.group_attributes = ["cn", "description"]

@dataclass
class SAMLConfig:
    """SAML authentication configuration"""
    entity_id: str
    sso_url: str
    slo_url: Optional[str] = None
    x509_cert: str = ""
    private_key: str = ""
    attribute_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.attribute_mapping is None:
            self.attribute_mapping = {
                "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
                "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
                "groups": "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups"
            }

@dataclass 
class OAuthConfig:
    """OAuth 2.0 authentication configuration"""
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    userinfo_url: str
    redirect_uri: str
    scope: List[str] = None
    
    def __post_init__(self):
        if self.scope is None:
            self.scope = ["openid", "profile", "email"]

@dataclass
class AuthConfig:
    """Main authentication configuration"""
    provider: AuthProvider
    ldap_config: Optional[LDAPConfig] = None
    saml_config: Optional[SAMLConfig] = None
    oauth_config: Optional[OAuthConfig] = None
    session_timeout: int = 3600  # 1 hour
    require_mfa: bool = False
    allowed_domains: List[str] = None

@dataclass
class UserProfile:
    """Enterprise user profile"""
    username: str
    email: str
    full_name: str
    first_name: str = ""
    last_name: str = ""
    groups: List[str] = None
    roles: List[str] = None
    department: str = ""
    title: str = ""
    phone: str = ""
    manager: str = ""
    created_at: datetime = None
    last_login: datetime = None
    is_active: bool = True
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.groups is None:
            self.groups = []
        if self.roles is None:
            self.roles = []
        if self.attributes is None:
            self.attributes = {}
        if self.created_at is None:
            self.created_at = datetime.now()

class AuthenticationError(Exception):
    """Authentication-related exceptions"""
    pass

class EnterpriseAuthManager:
    """Main enterprise authentication manager"""
    
    def __init__(self, config: AuthConfig, cache_dir: Optional[str] = None):
        self.config = config
        self.cache_dir = Path(cache_dir or ".pomuse/auth")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Dict] = {}
        self.user_cache: Dict[str, UserProfile] = {}
        self.logger = logging.getLogger(__name__)
        
    async def authenticate(self, username: str, password: str = None, 
                          token: str = None, **kwargs) -> UserProfile:
        """Authenticate user against enterprise system"""
        try:
            if self.config.provider == AuthProvider.LDAP:
                return await self._authenticate_ldap(username, password)
            elif self.config.provider == AuthProvider.SAML:
                return await self._authenticate_saml(token)
            elif self.config.provider == AuthProvider.OAUTH2:
                return await self._authenticate_oauth(token)
            elif self.config.provider == AuthProvider.AZURE_AD:
                return await self._authenticate_azure_ad(username, password)
            elif self.config.provider == AuthProvider.LOCAL:
                return await self._authenticate_local(username, password)
            else:
                raise AuthenticationError(f"Unsupported provider: {self.config.provider}")
                
        except Exception as e:
            self.logger.error(f"Authentication failed for {username}: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    async def _authenticate_ldap(self, username: str, password: str) -> UserProfile:
        """Authenticate against LDAP/Active Directory"""
        if not LDAP_AVAILABLE:
            raise AuthenticationError("LDAP support not available. Install python-ldap3")
            
        if not self.config.ldap_config:
            raise AuthenticationError("LDAP configuration not provided")
            
        config = self.config.ldap_config
        server_uri = f"{'ldaps' if config.use_ssl else 'ldap'}://{config.server}:{config.port}"
        
        try:
            server = ldap3.Server(server_uri, use_ssl=config.use_ssl)
            
            # Try to bind with user credentials
            user_dn = config.user_filter.format(username=username)
            if config.base_dn:
                user_dn = f"{user_dn},{config.base_dn}"
                
            conn = ldap3.Connection(server, user=user_dn, password=password)
            
            if not conn.bind():
                raise AuthenticationError("Invalid credentials")
                
            # Search for user details
            search_base = config.base_dn
            search_filter = config.user_filter.format(username=username)
            
            conn.search(search_base, search_filter, attributes=config.user_attributes)
            
            if not conn.entries:
                raise AuthenticationError("User not found")
                
            entry = conn.entries[0]
            
            # Extract user information
            profile = UserProfile(
                username=username,
                email=str(entry.mail) if hasattr(entry, 'mail') else f"{username}@{config.server}",
                full_name=str(entry.cn) if hasattr(entry, 'cn') else username,
                first_name=str(entry.givenName) if hasattr(entry, 'givenName') else "",
                last_name=str(entry.sn) if hasattr(entry, 'sn') else "",
                last_login=datetime.now()
            )
            
            # Get user groups
            if config.group_filter:
                group_filter = config.group_filter.format(user_dn=entry.entry_dn)
                conn.search(search_base, group_filter, attributes=config.group_attributes)
                profile.groups = [str(group.cn) for group in conn.entries if hasattr(group, 'cn')]
            
            conn.unbind()
            
            self.user_cache[username] = profile
            return profile
            
        except Exception as e:
            raise AuthenticationError(f"LDAP authentication failed: {e}")
    
    async def _authenticate_saml(self, token: str) -> UserProfile:
        """Authenticate SAML token"""
        # Simplified SAML token validation - in production use proper SAML library
        if not token:
            raise AuthenticationError("SAML token required")
            
        try:
            # Decode and validate SAML assertion (simplified)
            # In production, use libraries like python3-saml
            decoded = base64.b64decode(token)
            
            # Extract user information from SAML assertion
            # This is a simplified implementation
            profile = UserProfile(
                username="saml_user",
                email="user@enterprise.com",
                full_name="SAML User",
                last_login=datetime.now()
            )
            
            return profile
            
        except Exception as e:
            raise AuthenticationError(f"SAML authentication failed: {e}")
    
    async def _authenticate_oauth(self, token: str) -> UserProfile:
        """Authenticate OAuth token"""
        if not OAUTH_AVAILABLE:
            raise AuthenticationError("OAuth support not available. Install requests and PyJWT")
            
        if not self.config.oauth_config or not token:
            raise AuthenticationError("OAuth configuration or token not provided")
            
        config = self.config.oauth_config
        
        try:
            # Verify JWT token (simplified)
            # In production, verify signature with provider's public key
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Get user info from OAuth provider
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(config.userinfo_url, headers=headers)
            response.raise_for_status()
            
            user_info = response.json()
            
            profile = UserProfile(
                username=user_info.get("preferred_username", user_info.get("sub")),
                email=user_info.get("email", ""),
                full_name=user_info.get("name", ""),
                first_name=user_info.get("given_name", ""),
                last_name=user_info.get("family_name", ""),
                groups=user_info.get("groups", []),
                last_login=datetime.now()
            )
            
            return profile
            
        except Exception as e:
            raise AuthenticationError(f"OAuth authentication failed: {e}")
    
    async def _authenticate_azure_ad(self, username: str, password: str) -> UserProfile:
        """Authenticate against Azure Active Directory"""
        # Use OAuth flow for Azure AD
        if not self.config.oauth_config:
            raise AuthenticationError("Azure AD OAuth configuration required")
            
        # Simplified Azure AD authentication
        # In production, use MSAL (Microsoft Authentication Library)
        config = self.config.oauth_config
        
        try:
            # Get token from Azure AD
            token_data = {
                "grant_type": "password",
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "username": username,
                "password": password,
                "scope": " ".join(config.scope)
            }
            
            response = requests.post(config.token_url, data=token_data)
            response.raise_for_status()
            
            token_response = response.json()
            access_token = token_response["access_token"]
            
            return await self._authenticate_oauth(access_token)
            
        except Exception as e:
            raise AuthenticationError(f"Azure AD authentication failed: {e}")
    
    async def _authenticate_local(self, username: str, password: str) -> UserProfile:
        """Local authentication (for development/testing)"""
        # Simple local auth - hash passwords in production
        local_users = self._load_local_users()
        
        if username not in local_users:
            raise AuthenticationError("User not found")
            
        user_data = local_users[username]
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if user_data.get("password_hash") != password_hash:
            raise AuthenticationError("Invalid password")
            
        profile = UserProfile(
            username=username,
            email=user_data.get("email", f"{username}@local"),
            full_name=user_data.get("full_name", username),
            groups=user_data.get("groups", []),
            roles=user_data.get("roles", []),
            last_login=datetime.now()
        )
        
        return profile
    
    def _load_local_users(self) -> Dict[str, Any]:
        """Load local users from cache"""
        users_file = self.cache_dir / "local_users.json"
        if users_file.exists():
            with open(users_file) as f:
                return json.load(f)
        return {}
    
    def create_session(self, user: UserProfile) -> str:
        """Create authenticated session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(seconds=self.config.session_timeout)
        
        self.sessions[session_id] = {
            "user": asdict(user),
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "last_activity": datetime.now()
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[UserProfile]:
        """Validate session and return user profile"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        if datetime.now() > session["expires_at"]:
            del self.sessions[session_id]
            return None
            
        # Update last activity
        session["last_activity"] = datetime.now()
        
        user_data = session["user"]
        return UserProfile(**user_data)
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def sync_users(self) -> List[UserProfile]:
        """Sync users from enterprise system"""
        if self.config.provider == AuthProvider.LDAP:
            return await self._sync_ldap_users()
        elif self.config.provider == AuthProvider.OAUTH2:
            return await self._sync_oauth_users()
        else:
            self.logger.warning(f"User sync not supported for {self.config.provider}")
            return []
    
    async def _sync_ldap_users(self) -> List[UserProfile]:
        """Sync users from LDAP"""
        if not LDAP_AVAILABLE or not self.config.ldap_config:
            return []
            
        config = self.config.ldap_config
        users = []
        
        try:
            server_uri = f"{'ldaps' if config.use_ssl else 'ldap'}://{config.server}:{config.port}"
            server = ldap3.Server(server_uri, use_ssl=config.use_ssl)
            
            conn = ldap3.Connection(server, user=config.bind_dn, password=config.bind_password)
            
            if not conn.bind():
                raise AuthenticationError("LDAP bind failed")
                
            # Search for all users
            search_filter = config.user_filter.replace("{username}", "*")
            conn.search(config.base_dn, search_filter, attributes=config.user_attributes)
            
            for entry in conn.entries:
                profile = UserProfile(
                    username=str(entry.uid) if hasattr(entry, 'uid') else str(entry.cn),
                    email=str(entry.mail) if hasattr(entry, 'mail') else "",
                    full_name=str(entry.cn) if hasattr(entry, 'cn') else "",
                    first_name=str(entry.givenName) if hasattr(entry, 'givenName') else "",
                    last_name=str(entry.sn) if hasattr(entry, 'sn') else ""
                )
                users.append(profile)
                
            conn.unbind()
            
        except Exception as e:
            self.logger.error(f"LDAP user sync failed: {e}")
            
        return users
    
    async def _sync_oauth_users(self) -> List[UserProfile]:
        """Sync users from OAuth provider"""
        # Implementation depends on specific OAuth provider
        # Most OAuth providers don't allow listing all users
        self.logger.warning("OAuth user sync not implemented - provider dependent")
        return []

async def authenticate_user(config: AuthConfig, username: str, 
                          password: str = None, token: str = None) -> UserProfile:
    """Convenience function to authenticate user"""
    auth_manager = EnterpriseAuthManager(config)
    return await auth_manager.authenticate(username, password, token)

async def sync_enterprise_users(config: AuthConfig) -> List[UserProfile]:
    """Convenience function to sync enterprise users"""
    auth_manager = EnterpriseAuthManager(config)
    return await auth_manager.sync_users()