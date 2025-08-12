"""
이메일 알림 시스템
일일/주간/월간 리포트 및 중요 알림 발송
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import io
import base64

logger = logging.getLogger(__name__)


class EmailNotifier:
    """이메일 알림 시스템"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 email: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        
        # 이메일 템플릿들
        self.templates = self._initialize_templates()
        
        logger.info(f"Email notifier initialized: {email}")
    
    def _initialize_templates(self) -> Dict[str, str]:
        """이메일 템플릿 초기화"""
        return {
            "daily_report": """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }
                        .header { background: #f8f9fa; padding: 20px; text-align: center; }
                        .content { padding: 20px; }
                        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
                        .metric { text-align: center; padding: 15px; background: #e9ecef; border-radius: 5px; }
                        .metric h3 { margin: 0; color: #495057; }
                        .metric p { margin: 5px 0 0 0; font-size: 24px; font-weight: bold; }
                        .positive { color: #28a745; }
                        .negative { color: #dc3545; }
                        .neutral { color: #6c757d; }
                        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        .table th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>📈 일일 거래 리포트</h1>
                        <p>{{ date }}</p>
                    </div>
                    
                    <div class="content">
                        <div class="metrics">
                            <div class="metric">
                                <h3>일일 수익률</h3>
                                <p class="{{ 'positive' if daily_return > 0 else 'negative' if daily_return < 0 else 'neutral' }}">
                                    {{ "%.2f" | format(daily_return * 100) }}%
                                </p>
                            </div>
                            <div class="metric">
                                <h3>일일 손익</h3>
                                <p class="{{ 'positive' if daily_pnl > 0 else 'negative' if daily_pnl < 0 else 'neutral' }}">
                                    {{ "{:,}".format(daily_pnl|int) }}원
                                </p>
                            </div>
                            <div class="metric">
                                <h3>승률</h3>
                                <p>{{ "%.1f" | format(win_rate * 100) }}%</p>
                            </div>
                            <div class="metric">
                                <h3>포지션</h3>
                                <p>{{ positions }}개</p>
                            </div>
                        </div>
                        
                        <h2>📊 거래 내역</h2>
                        {% if trades %}
                        <table class="table">
                            <tr>
                                <th>시간</th>
                                <th>구분</th>
                                <th>종목</th>
                                <th>가격</th>
                                <th>수량</th>
                                <th>손익</th>
                            </tr>
                            {% for trade in trades %}
                            <tr>
                                <td>{{ trade.time }}</td>
                                <td>{{ trade.action }}</td>
                                <td>{{ trade.symbol }}</td>
                                <td>{{ "{:,}".format(trade.price|int) }}</td>
                                <td>{{ trade.quantity }}</td>
                                <td class="{{ 'positive' if trade.pnl > 0 else 'negative' if trade.pnl < 0 else 'neutral' }}">
                                    {{ "{:,}".format(trade.pnl|int) }}
                                </td>
                            </tr>
                            {% endfor %}
                        </table>
                        {% else %}
                        <p>오늘은 거래가 없었습니다.</p>
                        {% endif %}
                        
                        <h2>⚠️ 리스크 상태</h2>
                        <ul>
                            <li>리스크 레벨: <strong>{{ risk_level }}</strong></li>
                            <li>연속 손실: {{ consecutive_losses }}회</li>
                            <li>Market Score: {{ market_score }}점</li>
                            <li>시장 변동성: {{ "%.1f" | format(volatility) }}</li>
                        </ul>
                        
                        {% if recommendations %}
                        <h2>💡 권장 사항</h2>
                        <ul>
                            {% for rec in recommendations %}
                            <li>{{ rec }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        <hr>
                        <p><small>이 리포트는 자동으로 생성되었습니다. {{ timestamp }}</small></p>
                    </div>
                </body>
                </html>
            """,
            
            "weekly_report": """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }
                        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 color: white; padding: 30px; text-align: center; }
                        .content { padding: 30px; }
                        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                                       gap: 20px; margin: 30px 0; }
                        .summary-card { background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }
                        .chart-container { margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>📊 주간 성과 리포트</h1>
                        <p>{{ week_start }} ~ {{ week_end }}</p>
                    </div>
                    
                    <div class="content">
                        <div class="summary-grid">
                            <div class="summary-card">
                                <h3>주간 수익률</h3>
                                <h2 class="{{ 'positive' if weekly_return > 0 else 'negative' }}">
                                    {{ "%.2f" | format(weekly_return * 100) }}%
                                </h2>
                            </div>
                            <div class="summary-card">
                                <h3>총 거래</h3>
                                <h2>{{ total_trades }}회</h2>
                            </div>
                            <div class="summary-card">
                                <h3>평균 승률</h3>
                                <h2>{{ "%.1f" | format(avg_win_rate * 100) }}%</h2>
                            </div>
                            <div class="summary-card">
                                <h3>샤프 비율</h3>
                                <h2>{{ "%.2f" | format(sharpe_ratio) }}</h2>
                            </div>
                        </div>
                        
                        <h2>📈 일별 성과</h2>
                        <!-- 차트 데이터 삽입 예정 -->
                        
                        <h2>🏆 주간 하이라이트</h2>
                        <ul>
                            <li><strong>최고 수익일:</strong> {{ best_day.0 }} ({{ "%.2f" | format(best_day.1 * 100) }}%)</li>
                            <li><strong>최저 수익일:</strong> {{ worst_day.0 }} ({{ "%.2f" | format(worst_day.1 * 100) }}%)</li>
                            <li><strong>평균 일일 수익률:</strong> {{ "%.2f" | format(avg_daily_return * 100) }}%</li>
                            <li><strong>최대 일일 손실:</strong> {{ "%.2f" | format(max_daily_loss * 100) }}%</li>
                        </ul>
                        
                        {% if risk_events %}
                        <h2>⚠️ 리스크 이벤트</h2>
                        <ul>
                            {% for event in risk_events %}
                            <li>{{ event }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        <h2>🔧 전략 조정 사항</h2>
                        {% if adjustments %}
                        <ul>
                            {% for adj in adjustments %}
                            <li>{{ adj }}</li>
                            {% endfor %}
                        </ul>
                        {% else %}
                        <p>이번 주에는 전략 조정이 없었습니다.</p>
                        {% endif %}
                    </div>
                </body>
                </html>
            """,
            
            "risk_alert": """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }
                        .alert { background: #f8d7da; border: 1px solid #f5c6cb; 
                                color: #721c24; padding: 20px; border-radius: 5px; margin: 20px; }
                        .warning { background: #fff3cd; border: 1px solid #ffeaa7; 
                                  color: #856404; padding: 20px; border-radius: 5px; margin: 20px; }
                        .info { background: #d1ecf1; border: 1px solid #bee5eb; 
                               color: #0c5460; padding: 20px; border-radius: 5px; margin: 20px; }
                    </style>
                </head>
                <body>
                    <div class="{{ alert_class }}">
                        <h2>{{ alert_icon }} {{ title }}</h2>
                        <p><strong>{{ message }}</strong></p>
                        
                        {% if details %}
                        <h3>상세 정보:</h3>
                        <ul>
                            {% for key, value in details.items() %}
                            <li><strong>{{ key }}:</strong> {{ value }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        {% if recommendations %}
                        <h3>권장 조치:</h3>
                        <ul>
                            {% for rec in recommendations %}
                            <li>{{ rec }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        <hr>
                        <p><small>알림 시간: {{ timestamp }}</small></p>
                    </div>
                </body>
                </html>
            """,
            
            "monthly_summary": """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; }
                        .header { background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%); 
                                 color: white; padding: 40px; text-align: center; }
                        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                                   gap: 25px; margin: 40px 0; }
                        .kpi-card { background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                                   padding: 25px; border-radius: 10px; text-align: center; }
                        .performance-table { width: 100%; border-collapse: collapse; margin: 30px 0; }
                        .performance-table th, .performance-table td { 
                            border: 1px solid #ddd; padding: 12px; text-align: center; }
                        .performance-table th { background-color: #f2f2f2; font-weight: bold; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🏆 월간 성과 리포트</h1>
                        <p>{{ month_name }} 종합 분석</p>
                    </div>
                    
                    <div style="padding: 40px;">
                        <div class="kpi-grid">
                            <div class="kpi-card">
                                <h3>월간 수익률</h3>
                                <h1 class="{{ 'positive' if monthly_return > 0 else 'negative' }}">
                                    {{ "%.2f" | format(monthly_return * 100) }}%
                                </h1>
                                <p>목표: 2.5%</p>
                            </div>
                            <div class="kpi-card">
                                <h3>샤프 비율</h3>
                                <h1>{{ "%.2f" | format(sharpe_ratio) }}</h1>
                                <p>목표: 1.0</p>
                            </div>
                            <div class="kpi-card">
                                <h3>최대 낙폭</h3>
                                <h1 class="{{ 'positive' if max_drawdown < 0.1 else 'negative' }}">
                                    {{ "%.2f" | format(max_drawdown * 100) }}%
                                </h1>
                                <p>한도: 10%</p>
                            </div>
                            <div class="kpi-card">
                                <h3>거래 승률</h3>
                                <h1>{{ "%.1f" | format(win_rate * 100) }}%</h1>
                                <p>목표: 55%</p>
                            </div>
                        </div>
                        
                        <h2>📊 주간별 성과</h2>
                        <table class="performance-table">
                            <tr>
                                <th>주차</th>
                                <th>수익률</th>
                                <th>거래횟수</th>
                                <th>승률</th>
                                <th>리스크 사건</th>
                            </tr>
                            {% for week in weekly_data %}
                            <tr>
                                <td>{{ week.week }}</td>
                                <td class="{{ 'positive' if week.return > 0 else 'negative' }}">
                                    {{ "%.2f" | format(week.return * 100) }}%
                                </td>
                                <td>{{ week.trades }}</td>
                                <td>{{ "%.1f" | format(week.win_rate * 100) }}%</td>
                                <td>{{ week.risk_events }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                        
                        <h2>🎯 목표 달성도</h2>
                        <ul>
                            <li>월간 수익률: {{ "✅ 달성" if target_achieved else "❌ 미달성" }}</li>
                            <li>리스크 관리: {{ "✅ 양호" if risk_incidents < 5 else "⚠️ 주의" }}</li>
                            <li>거래 효율성: {{ "✅ 우수" if win_rate > 0.55 else "📈 개선 필요" }}</li>
                        </ul>
                        
                        <h2>📈 다음 월 전망</h2>
                        {% for forecast in next_month_forecast %}
                        <li>{{ forecast }}</li>
                        {% endfor %}
                    </div>
                </body>
                </html>
            """
        }
    
    async def send_daily_report(self, recipient: str, report_data: Dict) -> bool:
        """일일 리포트 발송"""
        try:
            subject = f"📈 일일 거래 리포트 - {datetime.now().strftime('%Y-%m-%d')}"
            
            # 템플릿 데이터 준비
            template_data = {
                "date": datetime.now().strftime('%Y년 %m월 %d일'),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **report_data
            }
            
            html_content = self._render_template("daily_report", template_data)
            
            return await self._send_email(
                recipient=recipient,
                subject=subject,
                html_content=html_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
            return False
    
    async def send_weekly_report(self, recipient: str, report_data: Dict) -> bool:
        """주간 리포트 발송"""
        try:
            subject = f"📊 주간 성과 리포트 - {report_data.get('week_start')} ~ {report_data.get('week_end')}"
            
            html_content = self._render_template("weekly_report", report_data)
            
            return await self._send_email(
                recipient=recipient,
                subject=subject,
                html_content=html_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")
            return False
    
    async def send_monthly_summary(self, recipient: str, summary_data: Dict) -> bool:
        """월간 종합 리포트 발송"""
        try:
            month_name = datetime.now().strftime('%Y년 %m월')
            subject = f"🏆 월간 성과 리포트 - {month_name}"
            
            template_data = {
                "month_name": month_name,
                **summary_data
            }
            
            html_content = self._render_template("monthly_summary", template_data)
            
            # 성과 차트를 첨부파일로 추가
            chart_attachment = self._create_performance_chart(summary_data)
            
            return await self._send_email(
                recipient=recipient,
                subject=subject,
                html_content=html_content,
                attachments=[chart_attachment] if chart_attachment else None
            )
            
        except Exception as e:
            logger.error(f"Failed to send monthly summary: {e}")
            return False
    
    async def send_risk_alert(self, recipient: str, alert_type: str, 
                            alert_data: Dict) -> bool:
        """리스크 알림 발송"""
        try:
            alert_configs = {
                "EMERGENCY": {
                    "subject": "🚨 긴급 리스크 알림",
                    "icon": "🚨",
                    "class": "alert"
                },
                "WARNING": {
                    "subject": "⚠️ 리스크 경고",
                    "icon": "⚠️",
                    "class": "warning"  
                },
                "INFO": {
                    "subject": "ℹ️ 리스크 정보",
                    "icon": "ℹ️",
                    "class": "info"
                }
            }
            
            config = alert_configs.get(alert_type, alert_configs["INFO"])
            
            template_data = {
                "title": config["subject"],
                "alert_icon": config["icon"],
                "alert_class": config["class"],
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **alert_data
            }
            
            html_content = self._render_template("risk_alert", template_data)
            
            return await self._send_email(
                recipient=recipient,
                subject=config["subject"],
                html_content=html_content,
                priority="high" if alert_type == "EMERGENCY" else "normal"
            )
            
        except Exception as e:
            logger.error(f"Failed to send risk alert: {e}")
            return False
    
    async def _send_email(self, recipient: str, subject: str, 
                         html_content: str, attachments: List = None,
                         priority: str = "normal") -> bool:
        """이메일 발송 내부 함수"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email
            msg['To'] = recipient
            msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
            
            # 우선순위 설정
            if priority == "high":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'
            
            # HTML 본문
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # 첨부파일
            if attachments:
                for attachment in attachments:
                    msg.attach(attachment)
            
            # SMTP 발송
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                text = msg.as_string()
                server.sendmail(self.email, recipient, text)
            
            logger.info(f"Email sent successfully: {subject} to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _render_template(self, template_name: str, data: Dict) -> str:
        """템플릿 렌더링"""
        try:
            from jinja2 import Template
            template = Template(self.templates[template_name])
            return template.render(**data)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            # 간단한 fallback 템플릿
            return f"""
                <html>
                <body>
                    <h1>{data.get('title', 'Report')}</h1>
                    <p>Report data: {str(data)}</p>
                </body>
                </html>
            """
    
    def _create_performance_chart(self, data: Dict) -> Optional[MIMEBase]:
        """성과 차트 생성 (첨부파일용)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 한글 폰트 설정
            plt.rcParams['font.family'] = ['DejaVu Sans']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 일별 수익률
            if 'daily_returns' in data:
                ax1.plot(data['daily_returns'])
                ax1.set_title('Daily Returns')
                ax1.grid(True)
            
            # 2. 누적 수익률
            if 'cumulative_returns' in data:
                ax2.plot(data['cumulative_returns'])
                ax2.set_title('Cumulative Returns')
                ax2.grid(True)
            
            # 3. 드로다운
            if 'drawdown' in data:
                ax3.fill_between(range(len(data['drawdown'])), data['drawdown'], alpha=0.3)
                ax3.set_title('Drawdown')
                ax3.grid(True)
            
            # 4. 월간 수익률
            if 'monthly_returns' in data:
                ax4.bar(range(len(data['monthly_returns'])), data['monthly_returns'])
                ax4.set_title('Monthly Returns')
                ax4.grid(True)
            
            plt.tight_layout()
            
            # BytesIO로 저장
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # MIMEBase 첨부파일 생성
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(img_buffer.read())
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="performance_chart_{datetime.now().strftime("%Y%m%d")}.png"'
            )
            
            return attachment
            
        except Exception as e:
            logger.error(f"Failed to create performance chart: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """이메일 연결 테스트"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                logger.info("Email connection test successful")
                return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False