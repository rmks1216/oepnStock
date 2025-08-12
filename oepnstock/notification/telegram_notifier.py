"""
텔레그램 알림 시스템
실시간 거래 상황 및 리스크 알림
"""
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TelegramMessage:
    """텔레그램 메시지"""
    level: str
    title: str
    message: str
    data: Optional[Dict] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramNotifier:
    """텔레그램 알림 시스템"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # 알림 레벨별 이모지
        self.alert_levels = {
            "EMERGENCY": "🚨",  # 긴급 (거래 중단, 큰 손실)
            "WARNING": "⚠️",    # 경고 (리스크 한도 근접)
            "INFO": "ℹ️",       # 정보 (일반적인 상태)
            "SUCCESS": "✅",    # 성공 (목표 달성)
            "TRADE": "📊",      # 거래 (매수/매도)
            "MARKET": "📈"      # 시장 (시장 상황)
        }
        
        # 메시지 큐 (비동기 전송)
        self.message_queue = asyncio.Queue()
        self.is_running = False
        
        logger.info(f"Telegram notifier initialized for chat: {chat_id}")
    
    async def start_worker(self):
        """알림 워커 시작"""
        self.is_running = True
        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                await self._send_message_internal(message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Telegram worker error: {e}")
                await asyncio.sleep(5)
    
    async def stop_worker(self):
        """알림 워커 중지"""
        self.is_running = False
    
    async def send_alert(self, level: str, title: str, message: str, 
                        data: Dict = None) -> bool:
        """알림 발송 (큐에 추가)"""
        telegram_message = TelegramMessage(
            level=level,
            title=title, 
            message=message,
            data=data
        )
        
        await self.message_queue.put(telegram_message)
        logger.info(f"Alert queued: {level} - {title}")
        return True
    
    async def send_daily_report(self, report_data: Dict) -> bool:
        """일일 리포트 발송"""
        emoji = "📊"
        title = "일일 거래 리포트"
        
        # 수익률에 따른 이모지 결정
        daily_return = report_data.get('daily_return', 0)
        if daily_return > 0.001:  # 0.1% 이상
            emoji = "🎉"
        elif daily_return < -0.01:  # -1% 이하
            emoji = "😰"
        elif daily_return > 0:
            emoji = "😊"
        
        message = f"""
{emoji} *일일 거래 결과*

📈 *수익률*: {daily_return:.2%}
💰 *손익*: {report_data.get('daily_pnl', 0):,}원
🎯 *목표 달성*: {'✅' if daily_return >= 0.0005 else '❌'}

📊 *거래 정보*
• 거래 횟수: {report_data.get('trade_count', 0)}회
• 승률: {report_data.get('win_rate', 0):.1%}
• 현재 포지션: {report_data.get('positions', 0)}개

⚠️ *리스크 상태*
• 리스크 레벨: {report_data.get('risk_level', 'N/A')}
• 연속 손실: {report_data.get('consecutive_losses', 0)}회
• Market Score: {report_data.get('market_score', 0)}점

💡 *다음 거래일 전망*
• 시장 변동성: {report_data.get('volatility', 0):.1f}
• 추천 액션: {report_data.get('recommendation', '정상 운영')}
        """
        
        return await self.send_alert("INFO", title, message.strip())
    
    async def send_risk_alert(self, risk_type: str, current_value: float, 
                             threshold: float, recommendation: str) -> bool:
        """리스크 알림 발송"""
        
        risk_messages = {
            "daily_loss": {
                "emoji": "🔴",
                "title": "일일 손실 경고",
                "format": "일일 손실 {:.2%} (한도: {:.2%})"
            },
            "consecutive_loss": {
                "emoji": "⚡",
                "title": "연속 손실 경고", 
                "format": "연속 손실 {}회 (한도: {}회)"
            },
            "monthly_drawdown": {
                "emoji": "📉",
                "title": "월간 낙폭 경고",
                "format": "월간 낙폭 {:.2%} (한도: {:.2%})"
            },
            "volatility": {
                "emoji": "🌪️",
                "title": "시장 변동성 경고",
                "format": "VIX {:.1f} (기준: {:.1f})"
            }
        }
        
        alert_info = risk_messages.get(risk_type, {
            "emoji": "⚠️",
            "title": "리스크 경고",
            "format": "위험값: {} (기준: {})"
        })
        
        message = f"""
{alert_info['emoji']} *{alert_info['title']}*

🚨 {alert_info['format'].format(current_value, threshold)}

💡 *권장 조치*: {recommendation}

⏰ 발생 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return await self.send_alert("WARNING", alert_info['title'], message.strip())
    
    async def send_trade_notification(self, action: str, symbol: str, 
                                    price: float, quantity: int, 
                                    reason: str) -> bool:
        """거래 알림 발송"""
        action_emoji = "🟢" if action == "buy" else "🔴"
        action_name = "매수" if action == "buy" else "매도"
        
        message = f"""
{action_emoji} *{action_name} 체결*

📋 종목: {symbol}
💰 가격: {price:,}원
📦 수량: {quantity:,}주
💵 금액: {price * quantity:,}원

💡 사유: {reason}
        """
        
        return await self.send_alert("TRADE", f"{action_name} 알림", message.strip())
    
    async def send_target_achievement(self, target_type: str, 
                                    achieved_value: float, 
                                    target_value: float) -> bool:
        """목표 달성 알림"""
        
        target_messages = {
            "daily": {
                "emoji": "🎯",
                "title": "일일 목표 달성!",
                "format": "일일 수익률 {:.2%} 달성! (목표: {:.2%})"
            },
            "monthly": {
                "emoji": "🏆", 
                "title": "월간 목표 달성!",
                "format": "월간 수익률 {:.2%} 달성! (목표: {:.2%})"
            },
            "sharpe": {
                "emoji": "⭐",
                "title": "샤프 비율 목표 달성!",
                "format": "샤프 비율 {:.2f} 달성! (목표: {:.2f})"
            }
        }
        
        alert_info = target_messages.get(target_type, {
            "emoji": "✨",
            "title": "목표 달성!",
            "format": "목표값 {} 달성! (기준: {})"
        })
        
        message = f"""
{alert_info['emoji']} *{alert_info['title']}*

🎉 {alert_info['format'].format(achieved_value, target_value)}

축하합니다! 목표를 달성했습니다! 🎊
        """
        
        return await self.send_alert("SUCCESS", alert_info['title'], message.strip())
    
    async def send_system_status(self, status: str, details: Dict = None) -> bool:
        """시스템 상태 알림"""
        status_messages = {
            "startup": "🚀 시스템 시작",
            "shutdown": "🔴 시스템 종료",
            "error": "❌ 시스템 오류",
            "maintenance": "🔧 시스템 점검",
            "resume": "▶️ 거래 재개",
            "pause": "⏸️ 거래 중단"
        }
        
        title = status_messages.get(status, f"시스템 상태: {status}")
        
        message = f"""
{title.split(' ')[0]} *{title.split(' ', 1)[1]}*

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if details:
            message += "\n\n📋 *상세 정보*:\n"
            for key, value in details.items():
                message += f"• {key}: {value}\n"
        
        level = "WARNING" if status in ["error", "shutdown"] else "INFO"
        return await self.send_alert(level, title, message.strip())
    
    async def _send_message_internal(self, telegram_message: TelegramMessage) -> bool:
        """내부 메시지 전송 함수"""
        try:
            emoji = self.alert_levels.get(telegram_message.level, "📢")
            
            text = f"{emoji} *{telegram_message.title}*\n\n"
            text += f"{telegram_message.message}\n"
            
            if telegram_message.data:
                text += "\n📊 *상세 정보*:\n"
                for key, value in telegram_message.data.items():
                    text += f"• {key}: {value}\n"
            
            text += f"\n⏰ {telegram_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # API 호출
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': True
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Telegram message sent: {telegram_message.title}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """텔레그램 봇 연결 테스트"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/getMe"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        bot_name = data['result']['username']
                        logger.info(f"Telegram bot connection successful: @{bot_name}")
                        
                        # 테스트 메시지 발송
                        await self.send_alert(
                            "INFO",
                            "연결 테스트",
                            "텔레그램 봇이 성공적으로 연결되었습니다! 🎉"
                        )
                        return True
                    else:
                        logger.error(f"Telegram bot connection failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def format_number(self, value: float, format_type: str = "currency") -> str:
        """숫자 포매팅"""
        if format_type == "currency":
            return f"{value:,}원"
        elif format_type == "percent":
            return f"{value:.2%}"
        elif format_type == "ratio":
            return f"{value:.2f}"
        else:
            return str(value)