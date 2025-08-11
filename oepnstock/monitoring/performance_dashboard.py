"""
성과 모니터링 대시보드
실시간 성과 추적 및 리포팅 시스템
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from ..config.realistic_targets import RealisticTargets, PerformanceMonitor
from ..core.risk_management import EnhancedRiskManager, RiskLevel
from ..core.adaptive import AutoAdjustmentEngine

logger = logging.getLogger(__name__)


@dataclass
class DailyMetrics:
    """일일 성과 지표"""
    date: str
    daily_return: float
    daily_pnl: float
    portfolio_value: float
    positions_count: int
    trades_count: int
    win_rate: float
    risk_level: str
    market_score: int
    volatility: float
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "date": self.date,
            "daily_return": round(self.daily_return, 4),
            "daily_pnl": int(self.daily_pnl),
            "portfolio_value": int(self.portfolio_value),
            "positions_count": self.positions_count,
            "trades_count": self.trades_count,
            "win_rate": round(self.win_rate, 3),
            "risk_level": self.risk_level,
            "market_score": self.market_score,
            "volatility": round(self.volatility, 2)
        }


@dataclass
class WeeklyReport:
    """주간 리포트"""
    week_start: str
    week_end: str
    weekly_return: float
    total_trades: int
    win_rate: float
    avg_daily_return: float
    max_daily_loss: float
    sharpe_ratio: float
    best_day: Tuple[str, float]  # (date, return)
    worst_day: Tuple[str, float]
    risk_events: List[str]
    adjustments_made: List[str]


@dataclass
class MonthlyReport:
    """월간 리포트"""
    month: str
    monthly_return: float
    target_achievement: bool
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    trading_days: int
    best_week: Tuple[str, float]
    worst_week: Tuple[str, float]
    risk_incidents: int
    strategy_adjustments: int
    cost_ratio: float  # 거래비용 비율


class PerformanceDashboard:
    """성과 모니터링 대시보드"""
    
    def __init__(self, initial_capital: float, dashboard_path: str = "dashboard"):
        self.initial_capital = initial_capital
        self.dashboard_path = Path(dashboard_path)
        self.dashboard_path.mkdir(exist_ok=True)
        
        # 모니터링 컴포넌트
        self.targets = RealisticTargets()
        self.performance_monitor = PerformanceMonitor()
        
        # 데이터 저장
        self.daily_metrics: List[DailyMetrics] = []
        self.weekly_reports: List[WeeklyReport] = []
        self.monthly_reports: List[MonthlyReport] = []
        
        # 현재 상태
        self.current_capital = initial_capital
        self.last_update = datetime.now()
        
        logger.info(f"Performance Dashboard initialized at {self.dashboard_path}")
    
    def update_daily_metrics(self, portfolio_value: float, daily_pnl: float,
                           positions: Dict, trades_today: List, 
                           market_score: int, volatility: float,
                           risk_manager: EnhancedRiskManager) -> DailyMetrics:
        """일일 지표 업데이트"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        daily_return = daily_pnl / self.current_capital if self.current_capital > 0 else 0.0
        
        # 승률 계산
        if trades_today:
            winning_trades = sum(1 for trade in trades_today if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / len(trades_today)
        else:
            win_rate = 0.0
        
        # 리스크 상태
        risk_status = risk_manager.get_current_risk_status()
        
        daily_metric = DailyMetrics(
            date=today,
            daily_return=daily_return,
            daily_pnl=daily_pnl,
            portfolio_value=portfolio_value,
            positions_count=len(positions),
            trades_count=len(trades_today),
            win_rate=win_rate,
            risk_level=risk_status.level.value,
            market_score=market_score,
            volatility=volatility
        )
        
        self.daily_metrics.append(daily_metric)
        self.current_capital = portfolio_value
        self.last_update = datetime.now()
        
        # 파일로 저장
        self._save_daily_metrics(daily_metric)
        
        return daily_metric
    
    def generate_daily_checklist(self) -> Dict[str, bool]:
        """일일 체크리스트 생성"""
        today_metrics = self._get_today_metrics()
        
        if not today_metrics:
            return {}
        
        checklist = {
            "Market Score 확인 (75점 이상)": today_metrics.market_score >= 75,
            "일일 목표 달성 (0.05% 이상)": today_metrics.daily_return >= self.targets.daily_target_min,
            "리스크 한도 준수 (-2% 이상)": today_metrics.daily_return >= -0.02,
            "포지션 적정 보유 (4개 이하)": today_metrics.positions_count <= 4,
            "변동성 정상 범위 (30 이하)": today_metrics.volatility <= 30
        }
        
        return checklist
    
    def generate_weekly_report(self) -> Optional[WeeklyReport]:
        """주간 리포트 생성"""
        if len(self.daily_metrics) < 7:
            return None
        
        # 최근 7일 데이터
        recent_metrics = self.daily_metrics[-7:]
        week_start = recent_metrics[0].date
        week_end = recent_metrics[-1].date
        
        # 주간 수익률
        weekly_return = sum(m.daily_return for m in recent_metrics)
        
        # 통계 계산
        total_trades = sum(m.trades_count for m in recent_metrics)
        win_rates = [m.win_rate for m in recent_metrics if m.trades_count > 0]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
        
        avg_daily_return = weekly_return / len(recent_metrics)
        daily_returns = [m.daily_return for m in recent_metrics]
        max_daily_loss = min(daily_returns) if daily_returns else 0.0
        
        # 샤프 비율 (간단 계산)
        if len(daily_returns) > 1:
            import numpy as np
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 최고/최악의 날
        best_day = max(recent_metrics, key=lambda x: x.daily_return)
        worst_day = min(recent_metrics, key=lambda x: x.daily_return)
        
        # 리스크 이벤트
        risk_events = [
            f"{m.date}: {m.risk_level}" 
            for m in recent_metrics 
            if m.risk_level in ["위험", "경고", "비상"]
        ]
        
        weekly_report = WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            weekly_return=weekly_return,
            total_trades=total_trades,
            win_rate=avg_win_rate,
            avg_daily_return=avg_daily_return,
            max_daily_loss=max_daily_loss,
            sharpe_ratio=sharpe_ratio,
            best_day=(best_day.date, best_day.daily_return),
            worst_day=(worst_day.date, worst_day.daily_return),
            risk_events=risk_events,
            adjustments_made=[]  # TODO: 조정 히스토리 연동
        )
        
        self.weekly_reports.append(weekly_report)
        self._save_weekly_report(weekly_report)
        
        return weekly_report
    
    def generate_monthly_report(self) -> Optional[MonthlyReport]:
        """월간 리포트 생성"""
        current_month = datetime.now().strftime('%Y-%m')
        
        # 현재 월 데이터 필터링
        monthly_metrics = [
            m for m in self.daily_metrics 
            if m.date.startswith(current_month)
        ]
        
        if len(monthly_metrics) < 10:  # 최소 10일 데이터 필요
            return None
        
        # 월간 수익률
        monthly_return = sum(m.daily_return for m in monthly_metrics)
        
        # 목표 달성 여부
        target_achievement = monthly_return >= self.targets.monthly_target
        
        # 통계 계산
        total_trades = sum(m.trades_count for m in monthly_metrics)
        win_rates = [m.win_rate for m in monthly_metrics if m.trades_count > 0]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
        
        # 샤프 비율 및 최대 낙폭
        daily_returns = [m.daily_return for m in monthly_metrics]
        if len(daily_returns) > 1:
            import numpy as np
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0.0
            
            # 최대 낙폭 계산 (누적 수익률에서)
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        # 연속 승/패 (간단 계산)
        consecutive_wins = 0
        consecutive_losses = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for metric in monthly_metrics:
            if metric.daily_return > 0:
                current_win_streak += 1
                current_loss_streak = 0
                consecutive_wins = max(consecutive_wins, current_win_streak)
            elif metric.daily_return < 0:
                current_loss_streak += 1
                current_win_streak = 0
                consecutive_losses = max(consecutive_losses, current_loss_streak)
        
        # 리스크 사건 수
        risk_incidents = sum(
            1 for m in monthly_metrics 
            if m.risk_level in ["위험", "경고", "비상"]
        )
        
        monthly_report = MonthlyReport(
            month=current_month,
            monthly_return=monthly_return,
            target_achievement=target_achievement,
            total_trades=total_trades,
            win_rate=avg_win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            trading_days=len(monthly_metrics),
            best_week=("", 0.0),  # TODO: 주간 데이터 연동
            worst_week=("", 0.0),
            risk_incidents=risk_incidents,
            strategy_adjustments=0,  # TODO: 조정 히스토리 연동
            cost_ratio=0.093  # 기본 거래비용 9.3%
        )
        
        self.monthly_reports.append(monthly_report)
        self._save_monthly_report(monthly_report)
        
        return monthly_report
    
    def get_performance_summary(self) -> Dict:
        """성과 요약"""
        if not self.daily_metrics:
            return {}
        
        recent_metrics = self.daily_metrics[-30:] if len(self.daily_metrics) >= 30 else self.daily_metrics
        
        # 기본 통계
        total_return = sum(m.daily_return for m in recent_metrics)
        avg_daily_return = total_return / len(recent_metrics)
        
        # 성과 평가
        evaluation = self.performance_monitor.evaluate_performance(
            daily_return=avg_daily_return,
            monthly_return=total_return,
            sharpe_ratio=self._calculate_sharpe_ratio(recent_metrics),
            win_rate=self._calculate_avg_win_rate(recent_metrics)
        )
        
        return {
            "period_days": len(recent_metrics),
            "total_return": round(total_return, 4),
            "avg_daily_return": round(avg_daily_return, 4),
            "current_capital": int(self.current_capital),
            "capital_change": int(self.current_capital - self.initial_capital),
            "evaluation": evaluation,
            "last_update": self.last_update.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def export_to_json(self, filename: Optional[str] = None) -> str:
        """JSON으로 내보내기"""
        if filename is None:
            filename = f"performance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "export_date": datetime.now().isoformat(),
            "daily_metrics": [m.to_dict() for m in self.daily_metrics],
            "performance_summary": self.get_performance_summary()
        }
        
        filepath = self.dashboard_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Performance data exported to {filepath}")
        return str(filepath)
    
    def _get_today_metrics(self) -> Optional[DailyMetrics]:
        """오늘 지표 조회"""
        today = datetime.now().strftime('%Y-%m-%d')
        for metric in reversed(self.daily_metrics):
            if metric.date == today:
                return metric
        return None
    
    def _calculate_sharpe_ratio(self, metrics: List[DailyMetrics]) -> float:
        """샤프 비율 계산"""
        if len(metrics) < 2:
            return 0.0
        
        returns = [m.daily_return for m in metrics]
        import numpy as np
        returns_array = np.array(returns)
        return np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0.0
    
    def _calculate_avg_win_rate(self, metrics: List[DailyMetrics]) -> float:
        """평균 승률 계산"""
        win_rates = [m.win_rate for m in metrics if m.trades_count > 0]
        return sum(win_rates) / len(win_rates) if win_rates else 0.0
    
    def _save_daily_metrics(self, metric: DailyMetrics):
        """일일 지표 저장"""
        daily_file = self.dashboard_path / "daily_metrics.json"
        
        try:
            if daily_file.exists():
                with open(daily_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
            
            data.append(metric.to_dict())
            
            with open(daily_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save daily metrics: {e}")
    
    def _save_weekly_report(self, report: WeeklyReport):
        """주간 리포트 저장"""
        weekly_file = self.dashboard_path / "weekly_reports.json"
        # TODO: 구현
        pass
    
    def _save_monthly_report(self, report: MonthlyReport):
        """월간 리포트 저장"""
        monthly_file = self.dashboard_path / "monthly_reports.json"
        # TODO: 구현
        pass


class AlertSystem:
    """알림 시스템"""
    
    def __init__(self):
        self.alert_history: List[Dict] = []
        self.targets = RealisticTargets()
    
    def check_alerts(self, daily_metric: DailyMetrics, 
                    risk_status: Dict) -> List[Dict[str, str]]:
        """알림 체크"""
        alerts = []
        
        # 일일 손실 경고
        if daily_metric.daily_return <= -0.015:  # -1.5% 이상 손실
            alerts.append({
                "level": "경고",
                "type": "손실",
                "message": f"일일 손실 {daily_metric.daily_return:.2%} 기록",
                "action": "포지션 재검토 필요"
            })
        
        # 목표 달성
        if daily_metric.daily_return >= self.targets.daily_target_optimal:
            alerts.append({
                "level": "정보",
                "type": "달성",
                "message": f"일일 목표 달성 {daily_metric.daily_return:.2%}",
                "action": "목표 수익 확보 고려"
            })
        
        # 시장 점수 경고
        if daily_metric.market_score < 70:
            alerts.append({
                "level": "주의",
                "type": "시장",
                "message": f"Market Score 낮음 ({daily_metric.market_score}점)",
                "action": "신규 진입 자제"
            })
        
        # 변동성 경고
        if daily_metric.volatility > 25:
            alerts.append({
                "level": "주의",
                "type": "변동성",
                "message": f"시장 변동성 높음 ({daily_metric.volatility})",
                "action": "포지션 크기 축소 고려"
            })
        
        return alerts