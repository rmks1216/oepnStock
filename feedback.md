# 자동매매 시스템 추가 개선 사항 구현 가이드

## 1. 📱 알림 시스템 구현

### 1.1 알림 채널 구성

#### Telegram Bot 연동
```python
# oepnstock/notification/telegram_notifier.py
import telegram
from telegram.ext import Updater
from typing import Dict, List
import asyncio
from datetime import datetime

class TelegramNotifier:
    """텔레그램 알림 시스템"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id
        self.alert_levels = {
            "EMERGENCY": "🚨",  # 긴급
            "WARNING": "⚠️",    # 경고
            "INFO": "ℹ️",       # 정보
            "SUCCESS": "✅"     # 성공
        }
    
    async def send_alert(self, level: str, title: str, message: str, data: Dict = None):
        """알림 발송"""
        emoji = self.alert_levels.get(level, "📢")
        
        text = f"{emoji} *{title}*\n\n"
        text += f"{message}\n"
        
        if data:
            text += "\n📊 *상세 정보:*\n"
            for key, value in data.items():
                text += f"• {key}: {value}\n"
        
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode='Markdown'
        )
```

#### Email 알림
```python
# oepnstock/notification/email_notifier.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template

class EmailNotifier:
    """이메일 알림 시스템"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 email: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        
        # 이메일 템플릿
        self.templates = {
            "daily_report": """
                <h2>📈 일일 거래 리포트</h2>
                <table>
                    <tr><td>일일 수익률:</td><td>{{ daily_return }}%</td></tr>
                    <tr><td>거래 횟수:</td><td>{{ trade_count }}회</td></tr>
                    <tr><td>승률:</td><td>{{ win_rate }}%</td></tr>
                    <tr><td>현재 포지션:</td><td>{{ positions }}</td></tr>
                </table>
            """,
            "risk_alert": """
                <h2>⚠️ 리스크 경고</h2>
                <p>{{ risk_message }}</p>
                <ul>
                    <li>현재 손실: {{ current_loss }}%</li>
                    <li>권장 조치: {{ recommended_action }}</li>
                </ul>
            """
        }
    
    def send_email(self, recipient: str, subject: str, 
                   template_name: str, data: Dict):
        """이메일 발송"""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.email
        msg['To'] = recipient
        
        # 템플릿 렌더링
        template = Template(self.templates[template_name])
        html = template.render(**data)
        
        part = MIMEText(html, 'html')
        msg.attach(part)
        
        # 발송
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
```

### 1.2 알림 트리거 설정

```python
# oepnstock/notification/alert_manager.py
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

class AlertType(Enum):
    """알림 타입"""
    DAILY_LOSS_LIMIT = "일일 손실 한도"
    CONSECUTIVE_LOSS = "연속 손실"
    MONTHLY_DRAWDOWN = "월간 드로다운"
    TARGET_ACHIEVED = "목표 달성"
    SYSTEM_ERROR = "시스템 오류"
    MARKET_VOLATILITY = "시장 변동성"
    POSITION_STOP_LOSS = "포지션 손절"
    TRADING_PAUSED = "거래 중단"

@dataclass
class AlertRule:
    """알림 규칙"""
    alert_type: AlertType
    condition: str
    threshold: float
    level: str  # EMERGENCY, WARNING, INFO
    channels: List[str]  # telegram, email, slack
    cooldown_minutes: int = 60  # 재알림 방지

class AlertManager:
    """통합 알림 관리자"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.notifiers = {}
        self.last_alert_times = {}
    
    def _initialize_rules(self) -> List[AlertRule]:
        """알림 규칙 초기화"""
        return [
            # 긴급 알림
            AlertRule(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                condition="daily_loss <= threshold",
                threshold=-0.02,
                level="EMERGENCY",
                channels=["telegram", "email"],
                cooldown_minutes=30
            ),
            AlertRule(
                alert_type=AlertType.CONSECUTIVE_LOSS,
                condition="consecutive_losses >= threshold",
                threshold=3,
                level="EMERGENCY",
                channels=["telegram", "email"],
                cooldown_minutes=60
            ),
            
            # 경고 알림
            AlertRule(
                alert_type=AlertType.MARKET_VOLATILITY,
                condition="vix >= threshold",
                threshold=30,
                level="WARNING",
                channels=["telegram"],
                cooldown_minutes=120
            ),
            
            # 정보 알림
            AlertRule(
                alert_type=AlertType.TARGET_ACHIEVED,
                condition="daily_return >= threshold",
                threshold=0.001,
                level="SUCCESS",
                channels=["telegram"],
                cooldown_minutes=1440  # 하루 1회
            )
        ]
    
    async def check_and_send_alerts(self, metrics: Dict):
        """지표 확인 및 알림 발송"""
        for rule in self.rules:
            if self._should_trigger_alert(rule, metrics):
                await self._send_alert(rule, metrics)
    
    def _should_trigger_alert(self, rule: AlertRule, metrics: Dict) -> bool:
        """알림 발송 조건 확인"""
        # 쿨다운 체크
        if not self._check_cooldown(rule):
            return False
        
        # 조건 평가
        return self._evaluate_condition(rule, metrics)
```

---

## 2. 🧪 백테스트 검증 시스템

### 2.1 고급 백테스트 프레임워크

```python
# oepnstock/backtest/advanced_backtester.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import concurrent.futures

@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy_name: str
    capital_range: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    best_trade: Dict
    worst_trade: Dict
    monthly_returns: List[float]
    risk_metrics: Dict

class AdvancedBacktester:
    """고급 백테스트 엔진"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        self.results_cache = {}
    
    def run_walk_forward_analysis(self, strategy, 
                                 window_size: int = 252,
                                 step_size: int = 63) -> Dict:
        """Walk-Forward Analysis"""
        results = []
        data = self.data_source.get_historical_data()
        
        for i in range(0, len(data) - window_size, step_size):
            # 훈련 기간
            train_data = data[i:i+window_size]
            
            # 테스트 기간
            test_start = i + window_size
            test_end = min(test_start + step_size, len(data))
            test_data = data[test_start:test_end]
            
            # 파라미터 최적화
            optimal_params = self._optimize_parameters(strategy, train_data)
            
            # 테스트
            test_result = self._run_backtest(strategy, test_data, optimal_params)
            results.append(test_result)
        
        return self._aggregate_walk_forward_results(results)
    
    def run_monte_carlo_simulation(self, strategy, 
                                  n_simulations: int = 1000) -> Dict:
        """몬테카를로 시뮬레이션"""
        results = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(n_simulations):
                future = executor.submit(
                    self._run_single_simulation,
                    strategy, 
                    random_seed=i
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return self._analyze_monte_carlo_results(results)
    
    def compare_capital_strategies(self) -> pd.DataFrame:
        """투자금액별 전략 비교"""
        capital_levels = [1_000_000, 3_000_000, 5_000_000, 10_000_000]
        results = []
        
        for capital in capital_levels:
            for strategy_name in ['conservative_100k', 'conservative_300k', 
                                 'conservative_500k', 'conservative_1000k']:
                result = self._run_capital_backtest(capital, strategy_name)
                results.append({
                    'capital': capital,
                    'strategy': strategy_name,
                    **result
                })
        
        return pd.DataFrame(results)
```

### 2.2 성과 검증 메트릭

```python
# oepnstock/backtest/performance_metrics.py
import numpy as np
from scipy import stats

class PerformanceMetrics:
    """고급 성과 지표 계산"""
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.array, 
                              max_drawdown: float) -> float:
        """Calmar Ratio = 연간 수익률 / 최대 낙폭"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.array, 
                               target_return: float = 0) -> float:
        """Sortino Ratio (하방 리스크만 고려)"""
        downside_returns = returns[returns < target_return]
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return np.inf
        
        return (returns.mean() - target_return) / downside_std
    
    @staticmethod
    def calculate_omega_ratio(returns: np.array, 
                            threshold: float = 0) -> float:
        """Omega Ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf
            
        return gains.sum() / losses.sum()
    
    @staticmethod
    def calculate_tail_ratio(returns: np.array, 
                           percentile: float = 0.05) -> float:
        """Tail Ratio (극단적 수익/손실 비율)"""
        sorted_returns = np.sort(returns)
        n = len(returns)
        
        tail_size = int(n * percentile)
        right_tail = sorted_returns[-tail_size:].mean()
        left_tail = sorted_returns[:tail_size].mean()
        
        return abs(right_tail / left_tail)
```

---

## 3. 🖥️ UI/UX 개선

### 3.1 웹 대시보드 구현

```python
# oepnstock/dashboard/web_dashboard.py
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class WebDashboard:
    """웹 기반 대시보드"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.setup_routes()
    
    def setup_routes(self):
        """라우트 설정"""
        
        @app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @app.route('/api/overview')
        def get_overview():
            """대시보드 개요 데이터"""
            return jsonify({
                'current_capital': self.data_manager.get_current_capital(),
                'daily_return': self.data_manager.get_daily_return(),
                'monthly_return': self.data_manager.get_monthly_return(),
                'positions': self.data_manager.get_positions(),
                'risk_level': self.data_manager.get_risk_level()
            })
        
        @app.route('/api/chart/equity')
        def get_equity_chart():
            """자산 곡선 차트"""
            equity_data = self.data_manager.get_equity_curve()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_data['date'],
                y=equity_data['equity'],
                mode='lines',
                name='자산가치',
                line=dict(color='#00C853', width=2)
            ))
            
            # 목표선 추가
            fig.add_hline(
                y=self.data_manager.initial_capital * 1.02,
                line_dash="dash",
                line_color="gray",
                annotation_text="월간 목표 (2%)"
            )
            
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return graphJSON
        
        @socketio.on('connect')
        def handle_connect():
            """실시간 연결"""
            emit('connected', {'data': 'Connected to dashboard'})
        
        @socketio.on('request_live_data')
        def handle_live_data_request():
            """실시간 데이터 요청"""
            while True:
                data = self.data_manager.get_live_data()
                emit('live_update', data)
                socketio.sleep(5)  # 5초마다 업데이트
```

### 3.2 반응형 웹 인터페이스

```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>자동매매 대시보드</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <!-- 헤더 -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800">
                📈 자동매매 대시보드
            </h1>
            <div class="flex items-center mt-4">
                <span class="px-3 py-1 rounded-full text-sm font-semibold
                    {{ 'bg-green-100 text-green-800' if risk_level == '안전' else 'bg-red-100 text-red-800' }}">
                    리스크: {{ risk_level }}
                </span>
            </div>
        </div>
        
        <!-- 주요 지표 카드 -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <!-- 일일 수익률 -->
            <div class="bg-white rounded-lg shadow p-6">
                <div class="flex items-center">
                    <div class="flex-1">
                        <p class="text-gray-600 text-sm">일일 수익률</p>
                        <p class="text-2xl font-bold 
                            {{ 'text-green-600' if daily_return > 0 else 'text-red-600' }}">
                            {{ daily_return }}%
                        </p>
                    </div>
                    <div class="text-4xl">
                        {{ '📈' if daily_return > 0 else '📉' }}
                    </div>
                </div>
            </div>
            
            <!-- 월간 수익률 -->
            <div class="bg-white rounded-lg shadow p-6">
                <div class="flex items-center">
                    <div class="flex-1">
                        <p class="text-gray-600 text-sm">월간 수익률</p>
                        <p class="text-2xl font-bold">{{ monthly_return }}%</p>
                        <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-blue-600 h-2 rounded-full" 
                                style="width: {{ (monthly_return/3)*100 }}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 현재 포지션 -->
            <div class="bg-white rounded-lg shadow p-6">
                <p class="text-gray-600 text-sm">현재 포지션</p>
                <p class="text-2xl font-bold">{{ positions }}/{{ max_positions }}</p>
            </div>
            
            <!-- 자본금 -->
            <div class="bg-white rounded-lg shadow p-6">
                <p class="text-gray-600 text-sm">현재 자본</p>
                <p class="text-2xl font-bold">₩{{ current_capital }}</p>
            </div>
        </div>
        
        <!-- 차트 영역 -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">자산 곡선</h2>
                <div id="equity-chart"></div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">일일 수익률</h2>
                <div id="daily-returns-chart"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Socket.IO 연결
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to server');
            socket.emit('request_live_data');
        });
        
        socket.on('live_update', function(data) {
            updateDashboard(data);
        });
        
        function updateDashboard(data) {
            // 실시간 데이터 업데이트
            document.getElementById('daily-return').innerText = data.daily_return + '%';
            // ... 기타 업데이트
        }
        
        // 차트 로드
        fetch('/api/chart/equity')
            .then(response => response.json())
            .then(fig => Plotly.plot('equity-chart', fig));
    </script>
</body>
</html>
```

### 3.3 모바일 앱 연동

```python
# oepnstock/mobile/api_server.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import jwt
from datetime import datetime, timedelta

app = FastAPI(title="자동매매 API")
security = HTTPBearer()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MobileAPI:
    """모바일 앱 API"""
    
    @app.post("/api/v1/auth/login")
    async def login(username: str, password: str):
        """로그인"""
        # 인증 로직
        if authenticate_user(username, password):
            token = create_jwt_token(username)
            return {"access_token": token, "token_type": "bearer"}
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    @app.get("/api/v1/dashboard")
    async def get_dashboard(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """대시보드 데이터"""
        return {
            "overview": {
                "total_asset": 10500000,
                "daily_return": 0.08,
                "monthly_return": 2.3,
                "risk_level": "안전"
            },
            "positions": [
                {"symbol": "005930", "name": "삼성전자", "quantity": 100, "pnl": 50000},
                {"symbol": "000660", "name": "SK하이닉스", "quantity": 50, "pnl": -20000}
            ],
            "recent_trades": [
                {"date": "2024-01-15", "action": "buy", "symbol": "005930", "price": 70000}
            ]
        }
    
    @app.post("/api/v1/trading/pause")
    async def pause_trading(duration: int = 1):
        """거래 일시 중지"""
        # 거래 중지 로직
        return {"status": "paused", "duration": duration}
    
    @app.get("/api/v1/alerts")
    async def get_alerts():
        """알림 조회"""
        return {
            "alerts": [
                {
                    "id": 1,
                    "type": "WARNING",
                    "title": "일일 손실 경고",
                    "message": "일일 손실 -1.5% 도달",
                    "timestamp": datetime.now()
                }
            ]
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """실시간 웹소켓 연결"""
        await websocket.accept()
        try:
            while True:
                # 실시간 데이터 전송
                data = get_realtime_data()
                await websocket.send_json(data)
                await asyncio.sleep(1)
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await websocket.close()
```

---

## 4. 🚀 구현 우선순위 및 일정

### Phase 1: 알림 시스템 (1주일)
- [ ] Telegram Bot 설정 및 연동
- [ ] 기본 알림 규칙 구현
- [ ] 일일 리포트 자동화

### Phase 2: 백테스트 고도화 (2주일)
- [ ] Walk-Forward Analysis 구현
- [ ] 몬테카를로 시뮬레이션
- [ ] 투자금액별 성과 비교

### Phase 3: 웹 대시보드 (2주일)
- [ ] Flask 기반 웹서버 구축
- [ ] 실시간 차트 구현
- [ ] 반응형 UI 개발

### Phase 4: 모바일 연동 (3주일)
- [ ] REST API 서버 구축
- [ ] JWT 인증 시스템
- [ ] React Native 앱 개발

---

## 💡 기대 효과

### 알림 시스템
- **즉각적 대응**: 위험 상황 실시간 인지
- **성과 추적**: 일일/주간/월간 자동 리포트
- **심리적 안정**: 시스템 정상 작동 확인

### 백테스트 검증
- **신뢰도 향상**: 다양한 시장 상황 검증
- **최적화**: 투자금액별 최적 파라미터 도출
- **리스크 파악**: 극단적 시나리오 대비

### UI/UX 개선
- **접근성**: 언제 어디서나 모니터링
- **시각화**: 직관적인 성과 파악
- **제어력**: 원격 거래 제어 가능

---

*작성일: 2025-08-12*  
*예상 완료일: 2025-09-30*