"""
알림 관리자 - 통합 알림 시스템
다양한 알림 채널을 통한 리스크 관리 및 상태 알림
"""
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

from .telegram_notifier import TelegramNotifier
from .email_notifier import EmailNotifier

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """알림 타입"""
    # 긴급 알림
    DAILY_LOSS_LIMIT = "일일 손실 한도"
    CONSECUTIVE_LOSS = "연속 손실"
    MONTHLY_DRAWDOWN = "월간 드로다운"
    SYSTEM_ERROR = "시스템 오류"
    EMERGENCY_STOP = "긴급 중단"
    
    # 경고 알림
    POSITION_STOP_LOSS = "포지션 손절"
    MARKET_VOLATILITY = "시장 변동성"
    RISK_THRESHOLD = "리스크 임계값"
    CONCENTRATION_RISK = "집중 리스크"
    
    # 정보 알림
    TARGET_ACHIEVED = "목표 달성"
    DAILY_REPORT = "일일 리포트"
    WEEKLY_REPORT = "주간 리포트"
    MONTHLY_REPORT = "월간 리포트"
    TRADE_EXECUTED = "거래 체결"
    
    # 시스템 상태
    TRADING_PAUSED = "거래 중단"
    TRADING_RESUMED = "거래 재개"
    SYSTEM_STARTUP = "시스템 시작"
    SYSTEM_SHUTDOWN = "시스템 종료"


@dataclass
class AlertRule:
    """알림 규칙"""
    alert_type: AlertType
    condition: str              # 조건식 (예: "daily_loss <= threshold")
    threshold: float            # 임계값
    level: str                  # EMERGENCY, WARNING, INFO, SUCCESS
    channels: List[str]         # telegram, email, slack 등
    cooldown_minutes: int = 60  # 재알림 방지 (분)
    enabled: bool = True        # 규칙 활성화 여부
    
    # 메시지 템플릿
    title_template: str = ""
    message_template: str = ""
    
    # 조건부 설정
    market_conditions: List[str] = field(default_factory=list)  # 특정 시장 상황에서만 적용
    time_conditions: List[str] = field(default_factory=list)    # 특정 시간대에서만 적용


@dataclass 
class AlertHistory:
    """알림 발송 이력"""
    alert_type: AlertType
    timestamp: datetime
    level: str
    channels_sent: List[str]
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """통합 알림 관리자"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/alert_config.json"
        
        # 알림 제공자들
        self.notifiers = {}
        
        # 알림 규칙 및 이력
        self.rules: List[AlertRule] = []
        self.last_alert_times: Dict[AlertType, datetime] = {}
        self.alert_history: List[AlertHistory] = []
        
        # 알림 큐 (비동기 처리)
        self.alert_queue = asyncio.Queue()
        self.is_running = False
        
        # 설정 로드
        self._load_config()
        self._initialize_default_rules()
        
        logger.info("Alert manager initialized")
    
    def _load_config(self):
        """설정 파일 로드"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 설정 적용 로직
                    logger.info(f"Alert config loaded from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load alert config: {e}")
    
    def _initialize_default_rules(self):
        """기본 알림 규칙 초기화"""
        self.rules = [
            # === 긴급 알림 ===
            AlertRule(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                condition="daily_loss <= threshold",
                threshold=-0.02,  # -2%
                level="EMERGENCY",
                channels=["telegram", "email"],
                cooldown_minutes=30,
                title_template="🚨 일일 손실 한도 도달",
                message_template="일일 손실 {daily_loss:.2%}가 한도 {threshold:.2%}에 도달했습니다."
            ),
            
            AlertRule(
                alert_type=AlertType.CONSECUTIVE_LOSS,
                condition="consecutive_losses >= threshold",
                threshold=3,
                level="EMERGENCY", 
                channels=["telegram", "email"],
                cooldown_minutes=60,
                title_template="⚡ 연속 손실 경고",
                message_template="연속 {consecutive_losses}회 손실이 발생했습니다. 거래를 중단합니다."
            ),
            
            AlertRule(
                alert_type=AlertType.MONTHLY_DRAWDOWN,
                condition="monthly_drawdown <= threshold",
                threshold=-0.10,  # -10%
                level="EMERGENCY",
                channels=["telegram", "email"],
                cooldown_minutes=360,  # 6시간
                title_template="📉 월간 드로다운 한도 초과",
                message_template="월간 최대 낙폭 {monthly_drawdown:.2%}가 한도를 초과했습니다."
            ),
            
            # === 경고 알림 ===
            AlertRule(
                alert_type=AlertType.MARKET_VOLATILITY,
                condition="volatility >= threshold",
                threshold=30.0,  # VIX 30 이상
                level="WARNING",
                channels=["telegram"],
                cooldown_minutes=120,
                title_template="🌪️ 시장 변동성 증가",
                message_template="시장 변동성(VIX {volatility:.1f})이 높아졌습니다. 신중한 거래가 필요합니다."
            ),
            
            AlertRule(
                alert_type=AlertType.CONCENTRATION_RISK,
                condition="sector_concentration >= threshold",
                threshold=0.40,  # 40%
                level="WARNING", 
                channels=["telegram"],
                cooldown_minutes=240,
                title_template="⚠️ 섹터 집중 위험",
                message_template="단일 섹터 집중도 {sector_concentration:.1%}가 한도를 초과했습니다."
            ),
            
            # === 정보 알림 ===
            AlertRule(
                alert_type=AlertType.TARGET_ACHIEVED,
                condition="daily_return >= threshold",
                threshold=0.001,  # 0.1%
                level="SUCCESS",
                channels=["telegram"],
                cooldown_minutes=1440,  # 하루 1회
                title_template="🎯 일일 목표 달성",
                message_template="일일 수익률 {daily_return:.2%}로 목표를 달성했습니다! 🎉"
            ),
            
            AlertRule(
                alert_type=AlertType.TRADE_EXECUTED,
                condition="always",
                threshold=0,
                level="INFO",
                channels=["telegram"],
                cooldown_minutes=0,  # 쿨다운 없음
                title_template="📊 거래 체결",
                message_template="{action} {symbol} {quantity}주 @ {price:,}원"
            ),
            
            # === 시스템 상태 ===
            AlertRule(
                alert_type=AlertType.SYSTEM_ERROR,
                condition="always",
                threshold=0,
                level="EMERGENCY",
                channels=["telegram", "email"],
                cooldown_minutes=15,
                title_template="❌ 시스템 오류",
                message_template="시스템 오류가 발생했습니다: {error_message}"
            )
        ]
    
    def add_notifier(self, name: str, notifier):
        """알림 제공자 추가"""
        self.notifiers[name] = notifier
        logger.info(f"Notifier added: {name}")
    
    def add_telegram_notifier(self, bot_token: str, chat_id: str):
        """텔레그램 알림자 추가"""
        telegram = TelegramNotifier(bot_token, chat_id)
        self.add_notifier("telegram", telegram)
    
    def add_email_notifier(self, smtp_server: str, smtp_port: int, 
                          email: str, password: str):
        """이메일 알림자 추가"""
        email_notifier = EmailNotifier(smtp_server, smtp_port, email, password)
        self.add_notifier("email", email_notifier)
    
    async def start_worker(self):
        """알림 워커 시작"""
        self.is_running = True
        logger.info("Alert manager worker started")
        
        while self.is_running:
            try:
                alert_data = await asyncio.wait_for(
                    self.alert_queue.get(),
                    timeout=1.0
                )
                await self._process_alert(alert_data)
                self.alert_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Alert worker error: {e}")
                await asyncio.sleep(5)
    
    async def stop_worker(self):
        """알림 워커 중지"""
        self.is_running = False
        logger.info("Alert manager worker stopped")
    
    async def check_and_send_alerts(self, metrics: Dict[str, Any]):
        """지표 확인 및 알림 발송"""
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            if self._should_trigger_alert(rule, metrics):
                await self._queue_alert(rule, metrics)
    
    def _should_trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """알림 발송 조건 확인"""
        # 쿨다운 체크
        if not self._check_cooldown(rule):
            return False
        
        # 시장 조건 체크
        if rule.market_conditions and not self._check_market_conditions(rule, metrics):
            return False
        
        # 시간 조건 체크
        if rule.time_conditions and not self._check_time_conditions(rule):
            return False
        
        # 메인 조건 평가
        return self._evaluate_condition(rule, metrics)
    
    def _check_cooldown(self, rule: AlertRule) -> bool:
        """쿨다운 시간 확인"""
        if rule.alert_type not in self.last_alert_times:
            return True
        
        last_time = self.last_alert_times[rule.alert_type]
        elapsed = datetime.now() - last_time
        return elapsed.total_seconds() >= rule.cooldown_minutes * 60
    
    def _check_market_conditions(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """시장 조건 확인"""
        if not rule.market_conditions:
            return True
        
        market_state = metrics.get('market_state', 'normal')
        return market_state in rule.market_conditions
    
    def _check_time_conditions(self, rule: AlertRule) -> bool:
        """시간 조건 확인"""
        if not rule.time_conditions:
            return True
        
        current_time = datetime.now()
        current_hour = current_time.hour
        current_day = current_time.strftime('%A').lower()
        
        for condition in rule.time_conditions:
            if condition == "trading_hours" and 9 <= current_hour <= 15:
                return True
            elif condition == "weekdays" and current_day not in ['saturday', 'sunday']:
                return True
            elif condition == "after_hours" and (current_hour < 9 or current_hour > 15):
                return True
        
        return False
    
    def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """조건식 평가"""
        try:
            if rule.condition == "always":
                return True
            
            # 간단한 조건 평가 로직
            condition = rule.condition
            threshold = rule.threshold
            
            if "daily_loss" in condition:
                daily_loss = metrics.get('daily_return', 0)
                if "daily_loss <= threshold" in condition:
                    return daily_loss <= threshold
            
            elif "consecutive_losses" in condition:
                consecutive_losses = metrics.get('consecutive_losses', 0)
                if "consecutive_losses >= threshold" in condition:
                    return consecutive_losses >= threshold
            
            elif "monthly_drawdown" in condition:
                monthly_drawdown = metrics.get('monthly_drawdown', 0)
                if "monthly_drawdown <= threshold" in condition:
                    return monthly_drawdown <= threshold
            
            elif "volatility" in condition:
                volatility = metrics.get('volatility', 0)
                if "volatility >= threshold" in condition:
                    return volatility >= threshold
            
            elif "daily_return" in condition:
                daily_return = metrics.get('daily_return', 0)
                if "daily_return >= threshold" in condition:
                    return daily_return >= threshold
            
            elif "sector_concentration" in condition:
                sector_concentration = metrics.get('sector_concentration', 0)
                if "sector_concentration >= threshold" in condition:
                    return sector_concentration >= threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition: {rule.condition}, error: {e}")
            return False
    
    async def _queue_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """알림을 큐에 추가"""
        alert_data = {
            'rule': rule,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        await self.alert_queue.put(alert_data)
    
    async def _process_alert(self, alert_data: Dict):
        """알림 처리"""
        rule = alert_data['rule']
        metrics = alert_data['metrics']
        timestamp = alert_data['timestamp']
        
        try:
            # 메시지 생성
            title = self._format_message(rule.title_template, metrics)
            message = self._format_message(rule.message_template, metrics)
            
            # 각 채널에 발송
            sent_channels = []
            for channel in rule.channels:
                if channel in self.notifiers:
                    try:
                        success = await self._send_to_channel(
                            channel, rule.level, title, message, metrics
                        )
                        if success:
                            sent_channels.append(channel)
                    except Exception as e:
                        logger.error(f"Failed to send alert to {channel}: {e}")
            
            # 발송 이력 기록
            self._record_alert_history(rule, timestamp, sent_channels, len(sent_channels) > 0, metrics)
            
            # 마지막 알림 시간 업데이트
            self.last_alert_times[rule.alert_type] = timestamp
            
            logger.info(f"Alert processed: {rule.alert_type.value} -> {sent_channels}")
            
        except Exception as e:
            logger.error(f"Failed to process alert: {e}")
    
    def _format_message(self, template: str, metrics: Dict[str, Any]) -> str:
        """메시지 템플릿 포매팅"""
        try:
            return template.format(**metrics)
        except KeyError as e:
            logger.warning(f"Missing metric in template: {e}")
            return template
        except Exception as e:
            logger.error(f"Template formatting error: {e}")
            return template
    
    async def _send_to_channel(self, channel: str, level: str, title: str, 
                              message: str, data: Dict) -> bool:
        """특정 채널로 알림 발송"""
        notifier = self.notifiers.get(channel)
        if not notifier:
            logger.warning(f"Notifier not found: {channel}")
            return False
        
        try:
            if channel == "telegram":
                return await notifier.send_alert(level, title, message, data)
            elif channel == "email":
                # 이메일은 더 상세한 데이터 포함
                return await notifier.send_risk_alert("recipient@example.com", level, {
                    "title": title,
                    "message": message,
                    "details": data
                })
            else:
                logger.warning(f"Unknown channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send to {channel}: {e}")
            return False
    
    def _record_alert_history(self, rule: AlertRule, timestamp: datetime,
                             channels_sent: List[str], success: bool, data: Dict):
        """알림 이력 기록"""
        history = AlertHistory(
            alert_type=rule.alert_type,
            timestamp=timestamp,
            level=rule.level,
            channels_sent=channels_sent,
            success=success,
            data=data
        )
        
        self.alert_history.append(history)
        
        # 이력 크기 제한 (최근 1000개)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    # === 편의 메서드들 ===
    
    async def send_immediate_alert(self, alert_type: AlertType, data: Dict):
        """즉시 알림 발송 (규칙 무시)"""
        rule = next((r for r in self.rules if r.alert_type == alert_type), None)
        if not rule:
            logger.warning(f"No rule found for alert type: {alert_type}")
            return
        
        await self._queue_alert(rule, data)
    
    async def send_daily_report(self, report_data: Dict):
        """일일 리포트 발송"""
        await self.send_immediate_alert(AlertType.DAILY_REPORT, report_data)
    
    async def send_trade_notification(self, action: str, symbol: str, 
                                    price: float, quantity: int, reason: str):
        """거래 알림 발송"""
        data = {
            'action': '매수' if action == 'buy' else '매도',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'reason': reason
        }
        await self.send_immediate_alert(AlertType.TRADE_EXECUTED, data)
    
    async def send_system_status(self, status: str, details: Dict = None):
        """시스템 상태 알림"""
        status_mapping = {
            'startup': AlertType.SYSTEM_STARTUP,
            'shutdown': AlertType.SYSTEM_SHUTDOWN,
            'pause': AlertType.TRADING_PAUSED,
            'resume': AlertType.TRADING_RESUMED,
            'error': AlertType.SYSTEM_ERROR
        }
        
        alert_type = status_mapping.get(status)
        if alert_type:
            data = details or {}
            data['status'] = status
            await self.send_immediate_alert(alert_type, data)
    
    def get_alert_statistics(self) -> Dict:
        """알림 통계 조회"""
        if not self.alert_history:
            return {}
        
        total_alerts = len(self.alert_history)
        success_alerts = sum(1 for h in self.alert_history if h.success)
        
        # 레벨별 통계
        level_stats = {}
        for history in self.alert_history:
            level = history.level
            level_stats[level] = level_stats.get(level, 0) + 1
        
        # 채널별 통계
        channel_stats = {}
        for history in self.alert_history:
            for channel in history.channels_sent:
                channel_stats[channel] = channel_stats.get(channel, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'success_rate': success_alerts / total_alerts if total_alerts > 0 else 0,
            'level_breakdown': level_stats,
            'channel_breakdown': channel_stats,
            'last_24h_alerts': len([h for h in self.alert_history 
                                  if h.timestamp > datetime.now() - timedelta(hours=24)])
        }
    
    def enable_rule(self, alert_type: AlertType):
        """알림 규칙 활성화"""
        for rule in self.rules:
            if rule.alert_type == alert_type:
                rule.enabled = True
                logger.info(f"Alert rule enabled: {alert_type.value}")
                break
    
    def disable_rule(self, alert_type: AlertType):
        """알림 규칙 비활성화"""
        for rule in self.rules:
            if rule.alert_type == alert_type:
                rule.enabled = False
                logger.info(f"Alert rule disabled: {alert_type.value}")
                break
    
    def update_rule_threshold(self, alert_type: AlertType, new_threshold: float):
        """알림 규칙 임계값 변경"""
        for rule in self.rules:
            if rule.alert_type == alert_type:
                old_threshold = rule.threshold
                rule.threshold = new_threshold
                logger.info(f"Alert threshold updated: {alert_type.value} "
                          f"{old_threshold} -> {new_threshold}")
                break
    
    async def test_all_channels(self):
        """모든 알림 채널 테스트"""
        test_results = {}
        
        for name, notifier in self.notifiers.items():
            try:
                if hasattr(notifier, 'test_connection'):
                    result = await notifier.test_connection()
                    test_results[name] = result
                else:
                    # 간단한 테스트 메시지 발송
                    result = await notifier.send_alert(
                        "INFO", 
                        "연결 테스트",
                        f"{name} 알림 채널 테스트 메시지입니다."
                    )
                    test_results[name] = result
            except Exception as e:
                logger.error(f"Channel test failed for {name}: {e}")
                test_results[name] = False
        
        return test_results