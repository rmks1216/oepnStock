"""
FastAPI 기반 모바일 API 서버
RESTful API 및 WebSocket 지원
"""
from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
import uvicorn
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback
from contextlib import asynccontextmanager

from .models import (
    # Authentication models
    LoginRequest, LoginResponse, RefreshTokenRequest, User, UserRole,
    
    # Dashboard models
    DashboardOverview, Position, Trade, Alert, PerformanceSummary,
    
    # Trading control models
    TradingControlRequest, TradingControlResponse,
    
    # Chart models
    ChartDataRequest, ChartData,
    
    # WebSocket models
    WebSocketMessage, LiveUpdate,
    
    # API response models
    APIResponse, ErrorResponse, PaginationParams, PaginatedResponse,
    
    # Filter models
    TradeFilter, AlertFilter,
    
    # Statistics models
    TradingStatistics, RiskMetrics,
    
    # System models
    SystemStatus, HealthCheck
)
from .auth import auth_manager, jwt_auth, admin_required, user_required, viewer_allowed
from ..dashboard.data_manager import DashboardDataManager

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """연결 수락"""
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """연결 해제"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, user_id: str):
        """개인 메시지 발송"""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """전체 브로드캐스트"""
        disconnected_connections = []
        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)
                else:
                    disconnected_connections.append(connection)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                disconnected_connections.append(connection)
        
        # 끊어진 연결 정리
        for conn in disconnected_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


class MobileAPI:
    """모바일 API 메인 클래스"""
    
    def __init__(self, data_manager: DashboardDataManager = None):
        self.data_manager = data_manager or DashboardDataManager()
        self.websocket_manager = WebSocketManager()
        
        # 실시간 업데이트 태스크
        self.update_task = None
        self.is_running = False
        
        logger.info("Mobile API initialized")
    
    async def start_background_tasks(self):
        """백그라운드 태스크 시작"""
        self.is_running = True
        self.update_task = asyncio.create_task(self._live_update_worker())
        logger.info("Background tasks started")
    
    async def stop_background_tasks(self):
        """백그라운드 태스크 중지"""
        self.is_running = False
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("Background tasks stopped")
    
    async def _live_update_worker(self):
        """실시간 업데이트 워커"""
        while self.is_running:
            try:
                live_data = self.data_manager.get_live_data()
                update_message = WebSocketMessage(
                    type="live_update",
                    data=live_data
                )
                await self.websocket_manager.broadcast(update_message.json())
                await asyncio.sleep(5)  # 5초마다 업데이트
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Live update worker error: {e}")
                await asyncio.sleep(10)


def create_mobile_api_app(data_manager: DashboardDataManager = None) -> FastAPI:
    """FastAPI 앱 생성"""
    
    mobile_api = MobileAPI(data_manager)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 시작시 실행
        await mobile_api.start_background_tasks()
        yield
        # 종료시 실행
        await mobile_api.stop_background_tasks()
    
    app = FastAPI(
        title="oepnStock Mobile API",
        description="자동매매 시스템 모바일 API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 실제 환경에서는 제한
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 예외 핸들러
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=str(exc.status_code),
                message=exc.detail
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                message="Internal server error"
            ).dict()
        )
    
    # === 인증 엔드포인트 ===
    
    @app.post("/api/v1/auth/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        """로그인"""
        user = auth_manager.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # 토큰 생성
        token_data = {
            "sub": user["username"],
            "role": user["role"]
        }
        access_token = auth_manager.create_access_token(token_data)
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,
            user_info={
                "id": user["id"],
                "username": user["username"],
                "email": user.get("email"),
                "role": user["role"]
            }
        )
    
    @app.post("/api/v1/auth/refresh")
    async def refresh_token(request: RefreshTokenRequest):
        """토큰 갱신"""
        new_token = auth_manager.refresh_access_token(request.refresh_token)
        if not new_token:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
        
        return {"access_token": new_token, "token_type": "bearer"}
    
    @app.get("/api/v1/auth/me", response_model=User)
    async def get_current_user(current_user: User = Depends(jwt_auth)):
        """현재 사용자 정보"""
        return current_user
    
    # === 대시보드 엔드포인트 ===
    
    @app.get("/api/v1/dashboard/overview", response_model=DashboardOverview)
    async def get_dashboard_overview(current_user: User = Depends(viewer_allowed())):
        """대시보드 개요"""
        try:
            live_data = mobile_api.data_manager.get_live_data()
            
            return DashboardOverview(
                total_asset=live_data["current_capital"],
                daily_return=live_data["daily_return"],
                daily_pnl=live_data["daily_pnl"],
                monthly_return=live_data["monthly_return"],
                risk_level=live_data["risk_level"],
                market_score=live_data["market_score"],
                positions_count=live_data["positions_count"],
                win_rate=live_data["win_rate"],
                consecutive_losses=mobile_api.data_manager.get_consecutive_losses(),
                is_trading_active=live_data["is_trading_active"],
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"Failed to get dashboard overview: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch dashboard data")
    
    @app.get("/api/v1/positions", response_model=List[Position])
    async def get_positions(current_user: User = Depends(viewer_allowed())):
        """현재 포지션 목록"""
        try:
            positions_data = mobile_api.data_manager.get_position_details()
            return [Position(**pos) for pos in positions_data]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch positions")
    
    @app.get("/api/v1/trades", response_model=List[Trade])
    async def get_trades(
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        symbol: Optional[str] = Query(None),
        current_user: User = Depends(viewer_allowed())
    ):
        """거래 내역"""
        try:
            trades_data = mobile_api.data_manager.get_recent_trades(limit + offset)
            
            # 심볼 필터링
            if symbol:
                trades_data = [t for t in trades_data if t["symbol"] == symbol]
            
            # 페이징
            paginated_trades = trades_data[offset:offset + limit]
            
            return [
                Trade(
                    id=i,
                    date=datetime.fromisoformat(trade["date"]),
                    symbol=trade["symbol"],
                    action=trade["action"],
                    quantity=trade["quantity"],
                    price=trade["price"],
                    trade_value=trade["trade_value"],
                    costs=trade["costs"],
                    pnl=trade.get("pnl"),
                    reason=trade["reason"]
                )
                for i, trade in enumerate(paginated_trades)
            ]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch trades")
    
    @app.get("/api/v1/alerts", response_model=List[Alert])
    async def get_alerts(
        limit: int = Query(10, ge=1, le=50),
        level: Optional[AlertLevel] = Query(None),
        current_user: User = Depends(viewer_allowed())
    ):
        """최근 알림"""
        try:
            alerts_data = mobile_api.data_manager.get_recent_alerts(limit)
            
            # 레벨 필터링
            if level:
                alerts_data = [a for a in alerts_data if a["level"] == level.value]
            
            return [
                Alert(
                    id=i,
                    timestamp=datetime.fromisoformat(alert["timestamp"]),
                    level=AlertLevel(alert["level"]),
                    title=alert["title"],
                    message=alert["message"],
                    is_read=False,  # 임시값
                    data=alert["data"]
                )
                for i, alert in enumerate(alerts_data)
            ]
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch alerts")
    
    @app.get("/api/v1/performance", response_model=PerformanceSummary)
    async def get_performance_summary(current_user: User = Depends(viewer_allowed())):
        """성과 요약"""
        try:
            summary_data = mobile_api.data_manager.get_performance_summary()
            
            return PerformanceSummary(
                period_start=datetime.now() - timedelta(days=30),
                period_end=datetime.now(),
                total_return=summary_data["total_return"],
                annual_return=summary_data.get("annual_return", 0),
                volatility=summary_data["volatility"],
                sharpe_ratio=summary_data["sharpe_ratio"],
                max_drawdown=summary_data["max_drawdown"],
                win_rate=summary_data["win_rate"],
                total_trades=summary_data["total_trades"],
                profit_factor=summary_data.get("profit_factor", 1.0)
            )
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch performance data")
    
    # === 거래 제어 엔드포인트 ===
    
    @app.post("/api/v1/trading/control", response_model=TradingControlResponse)
    async def control_trading(
        request: TradingControlRequest,
        current_user: User = Depends(user_required())
    ):
        """거래 제어"""
        try:
            if request.action == "pause":
                duration = request.duration_hours or 1
                success = mobile_api.data_manager.pause_trading(duration)
                if success:
                    return TradingControlResponse(
                        status="paused",
                        message=f"거래가 {duration}시간 동안 중단됩니다.",
                        active_until=datetime.now() + timedelta(hours=duration)
                    )
            
            elif request.action == "resume":
                success = mobile_api.data_manager.resume_trading()
                if success:
                    return TradingControlResponse(
                        status="active",
                        message="거래가 재개되었습니다."
                    )
            
            else:
                raise HTTPException(status_code=400, detail="Invalid action")
            
            raise HTTPException(status_code=500, detail="Failed to control trading")
            
        except Exception as e:
            logger.error(f"Failed to control trading: {e}")
            raise HTTPException(status_code=500, detail="Trading control failed")
    
    @app.get("/api/v1/trading/status")
    async def get_trading_status(current_user: User = Depends(viewer_allowed())):
        """거래 상태 조회"""
        return {
            "is_active": mobile_api.data_manager.is_trading_active(),
            "timestamp": datetime.now().isoformat()
        }
    
    # === 차트 데이터 엔드포인트 ===
    
    @app.get("/api/v1/charts/equity", response_model=ChartData)
    async def get_equity_chart(
        period: str = Query("1M", regex="^(1D|1W|1M|3M|6M|1Y|ALL)$"),
        current_user: User = Depends(viewer_allowed())
    ):
        """자산 곡선 차트 데이터"""
        try:
            days_map = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "ALL": None}
            days = days_map.get(period, 30)
            
            equity_data = mobile_api.data_manager.get_equity_curve(days) if days else mobile_api.data_manager.get_equity_curve()
            
            data_points = [
                ChartDataPoint(
                    timestamp=timestamp,
                    value=value,
                    label=f"₩{value:,.0f}"
                )
                for timestamp, value in equity_data.items()
            ]
            
            return ChartData(
                title="자산 곡선",
                chart_type="line",
                data_points=data_points,
                y_axis_label="자산 (원)",
                period=period,
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"Failed to get equity chart data: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch chart data")
    
    # === 시스템 상태 엔드포인트 ===
    
    @app.get("/api/v1/system/status", response_model=SystemStatus)
    async def get_system_status(current_user: User = Depends(admin_required())):
        """시스템 상태"""
        try:
            import psutil
            
            return SystemStatus(
                is_running=True,
                is_trading_active=mobile_api.data_manager.is_trading_active(),
                last_heartbeat=datetime.now(),
                uptime_seconds=int((datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds()),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                market_connection=True,  # 실제 구현 필요
                broker_connection=True   # 실제 구현 필요
            )
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch system status")
    
    @app.get("/api/v1/health", response_model=HealthCheck)
    async def health_check():
        """헬스 체크"""
        try:
            checks = {
                "database": True,  # 실제 DB 연결 확인 필요
                "data_manager": mobile_api.data_manager is not None,
                "websocket": len(mobile_api.websocket_manager.active_connections) >= 0
            }
            
            all_healthy = all(checks.values())
            status = "healthy" if all_healthy else "degraded"
            
            return HealthCheck(
                status=status,
                checks=checks,
                message="All systems operational" if all_healthy else "Some components are degraded"
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheck(
                status="unhealthy",
                checks={},
                message=f"Health check failed: {str(e)}"
            )
    
    # === WebSocket 엔드포인트 ===
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket 연결"""
        await mobile_api.websocket_manager.connect(websocket)
        try:
            while True:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        # 핑 응답
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif message_type == "subscribe":
                        # 구독 요청 처리
                        subscription_type = message.get("data", {}).get("subscription")
                        await websocket.send_text(json.dumps({
                            "type": "subscription_ack",
                            "data": {"subscription": subscription_type},
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                    
        except WebSocketDisconnect:
            mobile_api.websocket_manager.disconnect(websocket)
    
    return app


# 서버 실행 함수
def run_mobile_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    data_manager: DashboardDataManager = None,
    debug: bool = False
):
    """모바일 API 서버 실행"""
    app = create_mobile_api_app(data_manager)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info" if not debug else "debug",
        reload=debug
    )


if __name__ == "__main__":
    run_mobile_api_server(debug=True)