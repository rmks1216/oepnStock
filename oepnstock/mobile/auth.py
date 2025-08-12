"""
모바일 API 인증 시스템
JWT 기반 인증 및 권한 관리
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from dataclasses import dataclass

from .models import User, UserRole

logger = logging.getLogger(__name__)


@dataclass
class JWTSettings:
    """JWT 설정"""
    secret_key: str = "oepnstock_secret_key_change_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7


class AuthManager:
    """인증 관리자"""
    
    def __init__(self, jwt_settings: JWTSettings = None):
        self.jwt_settings = jwt_settings or JWTSettings()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # 임시 사용자 저장소 (실제 구현에서는 데이터베이스 사용)
        self.users_db = {
            "admin": {
                "id": 1,
                "username": "admin",
                "email": "admin@oepnstock.com",
                "hashed_password": self.get_password_hash("admin123!"),
                "role": UserRole.ADMIN,
                "created_at": datetime.now(),
                "is_active": True
            },
            "demo": {
                "id": 2,
                "username": "demo",
                "email": "demo@oepnstock.com",
                "hashed_password": self.get_password_hash("demo123!"),
                "role": UserRole.USER,
                "created_at": datetime.now(),
                "is_active": True
            }
        }
        
        logger.info("Auth manager initialized")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """비밀번호 해시화"""
        return self.pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """사용자 조회"""
        return self.users_db.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """사용자 인증"""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user["hashed_password"]):
            return None
        if not user["is_active"]:
            return None
        
        # 마지막 로그인 시간 업데이트
        user["last_login"] = datetime.now()
        
        return user
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """액세스 토큰 생성"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.jwt_settings.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.jwt_settings.secret_key, 
            algorithm=self.jwt_settings.algorithm
        )
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """리프레시 토큰 생성"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.jwt_settings.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.jwt_settings.secret_key,
            algorithm=self.jwt_settings.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """토큰 검증"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_settings.secret_key,
                algorithms=[self.jwt_settings.algorithm]
            )
            
            username: str = payload.get("sub")
            if username is None:
                return None
            
            # 토큰 타입 확인
            token_type = payload.get("type", "access")
            
            return {
                "username": username,
                "type": token_type,
                "exp": payload.get("exp"),
                "payload": payload
            }
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """액세스 토큰 갱신"""
        token_data = self.verify_token(refresh_token)
        if not token_data or token_data["type"] != "refresh":
            return None
        
        username = token_data["username"]
        user = self.get_user(username)
        if not user or not user["is_active"]:
            return None
        
        # 새 액세스 토큰 생성
        access_token_data = {
            "sub": username,
            "role": user["role"]
        }
        return self.create_access_token(access_token_data)


class JWTAuth:
    """JWT 인증 의존성"""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(None)):
        """인증 확인"""
        if credentials is None:
            credentials = await self.security(request=None)
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = self.auth_manager.verify_token(credentials.credentials)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        username = token_data["username"]
        user = self.auth_manager.get_user(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return User(
            id=user["id"],
            username=user["username"],
            email=user.get("email"),
            role=user["role"],
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            is_active=user["is_active"]
        )


class RoleChecker:
    """권한 확인"""
    
    def __init__(self, allowed_roles: list[UserRole]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(None)) -> User:
        """권한 확인"""
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user


# 권한별 의존성 팩토리
def admin_required():
    """관리자 권한 필요"""
    return RoleChecker([UserRole.ADMIN])


def user_required():
    """사용자 권한 필요"""
    return RoleChecker([UserRole.ADMIN, UserRole.USER])


def viewer_allowed():
    """뷰어 권한 허용"""
    return RoleChecker([UserRole.ADMIN, UserRole.USER, UserRole.VIEWER])


class APIKeyAuth:
    """API 키 인증 (선택적)"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {
            "mobile_app": "oepnstock_mobile_api_key_2024",
            "web_client": "oepnstock_web_api_key_2024"
        }
    
    def verify_api_key(self, api_key: str) -> bool:
        """API 키 검증"""
        return api_key in self.api_keys.values()
    
    def get_client_name(self, api_key: str) -> Optional[str]:
        """클라이언트 이름 조회"""
        for name, key in self.api_keys.items():
            if key == api_key:
                return name
        return None


class SessionManager:
    """세션 관리자"""
    
    def __init__(self):
        # 실제 구현에서는 Redis 등 사용
        self.active_sessions = {}
    
    def create_session(self, user_id: int, token: str) -> str:
        """세션 생성"""
        session_id = f"session_{user_id}_{datetime.now().timestamp()}"
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 조회"""
        return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """세션 활동 시간 업데이트"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.now()
    
    def invalidate_session(self, session_id: str):
        """세션 무효화"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def cleanup_expired_sessions(self, max_inactive_hours: int = 24):
        """만료된 세션 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_inactive_hours)
        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if session_data["last_activity"] < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# 전역 인스턴스
auth_manager = AuthManager()
session_manager = SessionManager()
api_key_auth = APIKeyAuth()
jwt_auth = JWTAuth(auth_manager)