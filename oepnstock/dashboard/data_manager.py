"""
대시보드 데이터 관리자
실시간 거래 데이터 및 성과 지표 관리
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    purchase_date: datetime


@dataclass
class Trade:
    """거래 정보"""
    date: datetime
    symbol: str
    action: str  # buy/sell
    quantity: int
    price: float
    trade_value: float
    costs: float
    pnl: Optional[float] = None
    reason: str = ""


@dataclass
class Alert:
    """알림 정보"""
    timestamp: datetime
    level: str
    title: str
    message: str
    data: Dict[str, Any]


class DashboardDataManager:
    """대시보드 데이터 관리자"""
    
    def __init__(self, db_path: str = "data/dashboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # 현재 상태
        self.current_capital = 10000000  # 초기 자본 1000만원
        self.initial_capital = 10000000
        self.is_trading_paused = False
        self.pause_until = None
        
        # 메모리 캐시
        self.positions_cache = {}
        self.trades_cache = []
        self.alerts_cache = []
        self.equity_curve_cache = pd.Series()
        self.daily_returns_cache = pd.Series()
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 샘플 데이터 로드
        self._load_sample_data()
        
        logger.info("Dashboard data manager initialized")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        trade_value REAL NOT NULL,
                        costs REAL NOT NULL,
                        pnl REAL,
                        reason TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        quantity INTEGER NOT NULL,
                        avg_price REAL NOT NULL,
                        purchase_date TEXT NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS equity_history (
                        date TEXT PRIMARY KEY,
                        equity REAL NOT NULL,
                        daily_return REAL,
                        positions_value REAL,
                        cash REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        data TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _load_sample_data(self):
        """샘플 데이터 로드"""
        try:
            # 최근 30일 자산 곡선 생성
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                end=datetime.now(),
                freq='D'
            )
            
            # 시뮬레이션된 자산 곡선
            np.random.seed(42)
            returns = np.random.normal(0.0005, 0.015, len(dates))  # 일평균 0.05%, 변동성 1.5%
            
            equity_values = [self.initial_capital]
            for ret in returns[1:]:
                equity_values.append(equity_values[-1] * (1 + ret))
            
            self.equity_curve_cache = pd.Series(equity_values, index=dates)
            self.daily_returns_cache = pd.Series(returns, index=dates)
            
            # 현재 자본 업데이트
            self.current_capital = equity_values[-1]
            
            # 샘플 포지션
            self.positions_cache = {
                '005930': Position(
                    symbol='005930',
                    quantity=100,
                    avg_price=70000,
                    current_price=71500,
                    market_value=7150000,
                    unrealized_pnl=150000,
                    unrealized_pnl_pct=0.0214,
                    purchase_date=datetime.now() - timedelta(days=5)
                ),
                '000660': Position(
                    symbol='000660',
                    quantity=50,
                    avg_price=120000,
                    current_price=118000,
                    market_value=5900000,
                    unrealized_pnl=-100000,
                    unrealized_pnl_pct=-0.0167,
                    purchase_date=datetime.now() - timedelta(days=3)
                )
            }
            
            # 샘플 거래 내역
            self.trades_cache = [
                Trade(
                    date=datetime.now() - timedelta(days=5),
                    symbol='005930',
                    action='buy',
                    quantity=100,
                    price=70000,
                    trade_value=7000000,
                    costs=10500,
                    reason="기술적 지지선 반등"
                ),
                Trade(
                    date=datetime.now() - timedelta(days=3),
                    symbol='000660',
                    action='buy',
                    quantity=50,
                    price=120000,
                    trade_value=6000000,
                    costs=9000,
                    reason="반도체 업종 강세"
                )
            ]
            
            # 샘플 알림
            self.alerts_cache = [
                Alert(
                    timestamp=datetime.now() - timedelta(minutes=30),
                    level="INFO",
                    title="일일 목표 달성",
                    message="일일 수익률 0.15%로 목표 달성",
                    data={"daily_return": 0.0015}
                ),
                Alert(
                    timestamp=datetime.now() - timedelta(hours=2),
                    level="WARNING",
                    title="시장 변동성 증가",
                    message="VIX 지수 상승으로 변동성 주의",
                    data={"vix": 28.5}
                )
            ]
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
    
    # === 기본 정보 조회 ===
    
    def get_current_capital(self) -> float:
        """현재 자본금"""
        return self.current_capital
    
    def get_initial_capital(self) -> float:
        """초기 자본금"""
        return self.initial_capital
    
    def get_daily_return(self) -> float:
        """일일 수익률"""
        if len(self.daily_returns_cache) > 0:
            return self.daily_returns_cache.iloc[-1]
        return 0.0
    
    def get_daily_pnl(self) -> float:
        """일일 손익"""
        daily_return = self.get_daily_return()
        return self.current_capital * daily_return
    
    def get_monthly_return(self) -> float:
        """월간 수익률"""
        if len(self.equity_curve_cache) >= 30:
            start_value = self.equity_curve_cache.iloc[-30]
            end_value = self.equity_curve_cache.iloc[-1]
            return (end_value / start_value) - 1
        return 0.0
    
    def get_current_positions(self) -> int:
        """현재 포지션 수"""
        return len(self.positions_cache)
    
    def get_max_positions(self) -> int:
        """최대 포지션 수"""
        return 5
    
    def get_risk_level(self) -> str:
        """리스크 레벨"""
        daily_return = self.get_daily_return()
        consecutive_losses = self.get_consecutive_losses()
        
        if daily_return <= -0.02 or consecutive_losses >= 3:
            return "위험"
        elif daily_return <= -0.01 or consecutive_losses >= 2:
            return "주의"
        else:
            return "안전"
    
    def get_market_score(self) -> int:
        """시장 점수 (0-100)"""
        # 간단한 시장 점수 시뮬레이션
        base_score = 70
        daily_return = self.get_daily_return()
        volatility = self.get_current_volatility()
        
        # 수익률에 따른 조정
        return_adjustment = daily_return * 1000
        
        # 변동성에 따른 조정
        volatility_adjustment = -volatility * 500
        
        score = base_score + return_adjustment + volatility_adjustment
        return max(0, min(100, int(score)))
    
    def get_win_rate(self) -> float:
        """승률"""
        if not self.trades_cache:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades_cache if trade.pnl and trade.pnl > 0)
        total_trades = len([trade for trade in self.trades_cache if trade.pnl is not None])
        
        if total_trades == 0:
            return 0.0
        
        return winning_trades / total_trades
    
    def get_today_trade_count(self) -> int:
        """오늘 거래 횟수"""
        today = datetime.now().date()
        return sum(1 for trade in self.trades_cache if trade.date.date() == today)
    
    def get_consecutive_losses(self) -> int:
        """연속 손실 횟수"""
        consecutive = 0
        for trade in reversed(self.trades_cache):
            if trade.pnl and trade.pnl < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def get_current_volatility(self) -> float:
        """현재 변동성"""
        if len(self.daily_returns_cache) >= 20:
            recent_returns = self.daily_returns_cache.iloc[-20:]
            return recent_returns.std() * np.sqrt(252)
        return 0.15  # 기본값
    
    # === 차트 데이터 ===
    
    def get_equity_curve(self, days: int = 30) -> pd.Series:
        """자산 곡선"""
        if len(self.equity_curve_cache) >= days:
            return self.equity_curve_cache.iloc[-days:]
        return self.equity_curve_cache
    
    def get_daily_returns(self, days: int = 30) -> pd.Series:
        """일일 수익률"""
        if len(self.daily_returns_cache) >= days:
            return self.daily_returns_cache.iloc[-days:]
        return self.daily_returns_cache
    
    def get_drawdown_series(self) -> pd.Series:
        """드로다운 시리즈"""
        if len(self.equity_curve_cache) == 0:
            return pd.Series()
        
        cumulative = self.equity_curve_cache / self.initial_capital
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        return drawdown
    
    # === 상세 정보 ===
    
    def get_position_details(self) -> List[Dict[str, Any]]:
        """포지션 상세 정보"""
        positions = []
        for position in self.positions_cache.values():
            positions.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'purchase_date': position.purchase_date.isoformat(),
                'days_held': (datetime.now() - position.purchase_date).days
            })
        return positions
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """최근 거래 내역"""
        trades = sorted(self.trades_cache, key=lambda x: x.date, reverse=True)[:limit]
        
        result = []
        for trade in trades:
            result.append({
                'date': trade.date.isoformat(),
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': trade.price,
                'trade_value': trade.trade_value,
                'costs': trade.costs,
                'pnl': trade.pnl,
                'reason': trade.reason
            })
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약"""
        total_return = (self.current_capital / self.initial_capital) - 1
        
        # 기본 지표
        summary = {
            'total_return': total_return,
            'daily_return': self.get_daily_return(),
            'monthly_return': self.get_monthly_return(),
            'volatility': self.get_current_volatility(),
            'win_rate': self.get_win_rate(),
            'total_trades': len(self.trades_cache),
            'current_positions': len(self.positions_cache),
            'consecutive_losses': self.get_consecutive_losses()
        }
        
        # 고급 지표
        if len(self.daily_returns_cache) > 0:
            returns = self.daily_returns_cache
            
            # 샤프 비율
            excess_returns = returns.mean() - 0.03/252
            volatility_daily = returns.std()
            summary['sharpe_ratio'] = (excess_returns / volatility_daily * np.sqrt(252)) if volatility_daily > 0 else 0
            
            # 최대 드로다운
            drawdown = self.get_drawdown_series()
            summary['max_drawdown'] = drawdown.min() if len(drawdown) > 0 else 0
            
            # VaR
            summary['var_95'] = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        return summary
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 알림"""
        alerts = sorted(self.alerts_cache, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        result = []
        for alert in alerts:
            result.append({
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'title': alert.title,
                'message': alert.message,
                'data': alert.data
            })
        
        return result
    
    # === 제어 기능 ===
    
    def pause_trading(self, duration_hours: int = 1) -> bool:
        """거래 중지"""
        try:
            self.is_trading_paused = True
            self.pause_until = datetime.now() + timedelta(hours=duration_hours)
            
            # 알림 추가
            self.add_alert(
                level="WARNING",
                title="거래 중단",
                message=f"거래가 {duration_hours}시간 동안 중단됩니다.",
                data={"duration": duration_hours}
            )
            
            logger.info(f"Trading paused for {duration_hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause trading: {e}")
            return False
    
    def resume_trading(self) -> bool:
        """거래 재개"""
        try:
            self.is_trading_paused = False
            self.pause_until = None
            
            # 알림 추가
            self.add_alert(
                level="INFO",
                title="거래 재개",
                message="거래가 재개되었습니다.",
                data={}
            )
            
            logger.info("Trading resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume trading: {e}")
            return False
    
    def is_trading_active(self) -> bool:
        """거래 활성 상태"""
        if self.is_trading_paused and self.pause_until:
            if datetime.now() > self.pause_until:
                self.resume_trading()
        
        return not self.is_trading_paused
    
    # === 데이터 업데이트 ===
    
    def add_trade(self, trade: Trade):
        """거래 추가"""
        self.trades_cache.append(trade)
        
        # 자산 곡선 업데이트
        self._update_equity_curve()
        
        logger.info(f"Trade added: {trade.action} {trade.symbol} {trade.quantity}")
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """포지션 업데이트"""
        if symbol in self.positions_cache:
            position = self.positions_cache[symbol]
            # 평균 단가 계산 (추가 매수 시)
            total_quantity = position.quantity + quantity
            if total_quantity > 0:
                total_cost = (position.quantity * position.avg_price) + (quantity * price)
                new_avg_price = total_cost / total_quantity
                
                position.quantity = total_quantity
                position.avg_price = new_avg_price
            else:
                # 모두 매도
                del self.positions_cache[symbol]
        else:
            # 신규 포지션
            self.positions_cache[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                purchase_date=datetime.now()
            )
    
    def update_market_prices(self, price_data: Dict[str, float]):
        """시장가 업데이트"""
        for symbol, current_price in price_data.items():
            if symbol in self.positions_cache:
                position = self.positions_cache[symbol]
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_price)
                position.unrealized_pnl_pct = position.unrealized_pnl / (position.quantity * position.avg_price)
    
    def add_alert(self, level: str, title: str, message: str, data: Dict[str, Any] = None):
        """알림 추가"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            title=title,
            message=message,
            data=data or {}
        )
        
        self.alerts_cache.append(alert)
        
        # 최대 100개 알림만 유지
        if len(self.alerts_cache) > 100:
            self.alerts_cache = self.alerts_cache[-100:]
        
        logger.info(f"Alert added: {level} - {title}")
    
    def _update_equity_curve(self):
        """자산 곡선 업데이트"""
        today = datetime.now().date()
        
        # 현재 포지션 가치
        positions_value = sum(pos.market_value for pos in self.positions_cache.values())
        
        # 현금 (간소화)
        cash = self.current_capital - positions_value
        
        # 총 자산
        total_equity = cash + positions_value
        
        # 일일 수익률 계산
        if len(self.equity_curve_cache) > 0:
            previous_equity = self.equity_curve_cache.iloc[-1]
            daily_return = (total_equity / previous_equity) - 1
        else:
            daily_return = (total_equity / self.initial_capital) - 1
        
        # 시리즈 업데이트
        self.equity_curve_cache.loc[today] = total_equity
        self.daily_returns_cache.loc[today] = daily_return
        self.current_capital = total_equity
    
    def get_live_data(self) -> Dict[str, Any]:
        """실시간 데이터 (WebSocket용)"""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_capital': self.get_current_capital(),
            'daily_return': self.get_daily_return(),
            'daily_pnl': self.get_daily_pnl(),
            'monthly_return': self.get_monthly_return(),
            'positions_count': self.get_current_positions(),
            'risk_level': self.get_risk_level(),
            'market_score': self.get_market_score(),
            'win_rate': self.get_win_rate(),
            'volatility': self.get_current_volatility(),
            'is_trading_active': self.is_trading_active()
        }