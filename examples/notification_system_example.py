"""
알림 시스템 사용 예제
텔레그램, 이메일을 통한 실시간 알림 및 리포트 발송
"""
import asyncio
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from oepnstock.notification.alert_manager import AlertManager, AlertType
from oepnstock.notification.telegram_notifier import TelegramNotifier
from oepnstock.notification.email_notifier import EmailNotifier

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationSystemDemo:
    """알림 시스템 데모"""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.setup_notifiers()
    
    def setup_notifiers(self):
        """알림 제공자 설정"""
        # 환경변수에서 설정 로드 (실제 사용시)
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID')
        
        email_address = os.getenv('EMAIL_ADDRESS', 'your_email@gmail.com')
        email_password = os.getenv('EMAIL_PASSWORD', 'your_app_password')
        
        # 텔레그램 알림자 추가
        if telegram_token != 'YOUR_BOT_TOKEN':
            self.alert_manager.add_telegram_notifier(telegram_token, telegram_chat_id)
            logger.info("Telegram notifier configured")
        else:
            logger.warning("Telegram credentials not provided - using mock notifier")
            self.alert_manager.add_notifier("telegram", MockNotifier("telegram"))
        
        # 이메일 알림자 추가
        if email_address != 'your_email@gmail.com':
            self.alert_manager.add_email_notifier(
                'smtp.gmail.com', 587, email_address, email_password
            )
            logger.info("Email notifier configured")
        else:
            logger.warning("Email credentials not provided - using mock notifier")
            self.alert_manager.add_notifier("email", MockNotifier("email"))
    
    async def run_demo(self):
        """데모 실행"""
        logger.info("=== 알림 시스템 데모 시작 ===")
        
        # 알림 워커 시작
        worker_task = asyncio.create_task(self.alert_manager.start_worker())
        
        try:
            # 1. 시스템 시작 알림
            await self.demo_system_startup()
            await asyncio.sleep(2)
            
            # 2. 거래 알림
            await self.demo_trade_notifications()
            await asyncio.sleep(2)
            
            # 3. 리스크 알림
            await self.demo_risk_alerts()
            await asyncio.sleep(2)
            
            # 4. 목표 달성 알림
            await self.demo_target_achievement()
            await asyncio.sleep(2)
            
            # 5. 일일 리포트
            await self.demo_daily_report()
            await asyncio.sleep(2)
            
            # 6. 알림 통계 조회
            await self.demo_alert_statistics()
            
        finally:
            # 워커 중지
            await self.alert_manager.stop_worker()
            worker_task.cancel()
            
        logger.info("=== 알림 시스템 데모 완료 ===")
    
    async def demo_system_startup(self):
        """시스템 시작 알림 데모"""
        logger.info("1. 시스템 시작 알림 발송")
        
        await self.alert_manager.send_system_status('startup', {
            'version': '1.0.0',
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_capital': 10_000_000,
            'active_strategies': ['conservative_enhanced']
        })
    
    async def demo_trade_notifications(self):
        """거래 알림 데모"""
        logger.info("2. 거래 알림 발송")
        
        # 매수 알림
        await self.alert_manager.send_trade_notification(
            action='buy',
            symbol='005930',  # 삼성전자
            price=70000,
            quantity=100,
            reason='지지선 반등 + 매수 신호 확인'
        )
        
        await asyncio.sleep(1)
        
        # 매도 알림
        await self.alert_manager.send_trade_notification(
            action='sell',
            symbol='000660',  # SK하이닉스
            price=85000,
            quantity=50,
            reason='목표가 달성 (손익: +250,000원)'
        )
    
    async def demo_risk_alerts(self):
        """리스크 알림 데모"""
        logger.info("3. 리스크 알림 발송")
        
        # 시장 변동성 경고
        market_metrics = {
            'volatility': 32.5,
            'market_state': 'volatile',
            'recommendation': '포지션 크기 축소 권장'
        }
        await self.alert_manager.check_and_send_alerts(market_metrics)
        
        await asyncio.sleep(1)
        
        # 일일 손실 한도 근접 (실제로는 발송되지 않음 - 임계값 미달)
        loss_metrics = {
            'daily_return': -0.015,  # -1.5% (임계값 -2%보다 높음)
            'daily_pnl': -150000,
            'recommendation': '신중한 거래 필요'
        }
        await self.alert_manager.check_and_send_alerts(loss_metrics)
    
    async def demo_target_achievement(self):
        """목표 달성 알림 데모"""
        logger.info("4. 목표 달성 알림 발송")
        
        # 일일 목표 달성
        achievement_metrics = {
            'daily_return': 0.0025,  # 0.25%
            'target_type': '일일 목표',
            'achieved_value': 0.0025,
            'target_value': 0.001
        }
        await self.alert_manager.check_and_send_alerts(achievement_metrics)
    
    async def demo_daily_report(self):
        """일일 리포트 데모"""
        logger.info("5. 일일 리포트 발송")
        
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'daily_return': 0.0025,
            'daily_pnl': 25000,
            'win_rate': 0.67,
            'trade_count': 3,
            'positions': 2,
            'risk_level': '안전',
            'consecutive_losses': 0,
            'market_score': 75,
            'volatility': 18.5,
            'recommendations': [
                '현재 포트폴리오는 안정적입니다',
                '시장 변동성이 낮아 추가 진입 기회를 모색하세요',
                '목표 수익률을 달성했습니다'
            ],
            'trades': [
                {
                    'time': '09:35',
                    'action': '매수',
                    'symbol': '005930',
                    'price': 70000,
                    'quantity': 100,
                    'pnl': 15000
                },
                {
                    'time': '14:20',
                    'action': '매도',
                    'symbol': '000660',
                    'price': 85000,
                    'quantity': 50,
                    'pnl': 10000
                }
            ]
        }
        
        await self.alert_manager.send_daily_report(report_data)
    
    async def demo_alert_statistics(self):
        """알림 통계 데모"""
        logger.info("6. 알림 통계 조회")
        
        stats = self.alert_manager.get_alert_statistics()
        logger.info(f"알림 통계: {stats}")
        
        # 채널 테스트
        test_results = await self.alert_manager.test_all_channels()
        logger.info(f"채널 테스트 결과: {test_results}")


class MockNotifier:
    """테스트용 가짜 알림자"""
    
    def __init__(self, channel_name: str):
        self.channel_name = channel_name
    
    async def send_alert(self, level: str, title: str, message: str, data: Dict = None):
        """가짜 알림 발송"""
        logger.info(f"[{self.channel_name.upper()}] {level}: {title}")
        logger.info(f"  Message: {message}")
        if data:
            logger.info(f"  Data: {data}")
        return True
    
    async def send_risk_alert(self, recipient: str, level: str, alert_data: Dict):
        """가짜 리스크 알림"""
        logger.info(f"[{self.channel_name.upper()}] Risk Alert to {recipient}")
        logger.info(f"  Level: {level}")
        logger.info(f"  Data: {alert_data}")
        return True
    
    async def test_connection(self):
        """가짜 연결 테스트"""
        logger.info(f"[{self.channel_name.upper()}] Connection test - OK")
        return True


async def run_notification_tests():
    """종합 알림 테스트"""
    logger.info("=== 알림 시스템 종합 테스트 ===")
    
    demo = NotificationSystemDemo()
    await demo.run_demo()


def setup_environment_example():
    """환경변수 설정 예제"""
    example_env = """
# .env 파일에 다음과 같이 설정하세요:

# 텔레그램 봇 설정
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# 이메일 설정 (Gmail 앱 비밀번호 사용)
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
ALERT_EMAIL_RECIPIENT=recipient@gmail.com

# 알림 설정
ALERT_LEVEL=INFO
ENABLE_TELEGRAM=true
ENABLE_EMAIL=true
    """
    
    print("환경변수 설정 예제:")
    print(example_env)


if __name__ == "__main__":
    # 환경변수 설정 예제 출력
    setup_environment_example()
    print("\n" + "="*50 + "\n")
    
    # 알림 시스템 데모 실행
    asyncio.run(run_notification_tests())