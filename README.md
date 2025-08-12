# 🚀 oepnStock - 종합 자동매매 플랫폼

**한국 주식 시장을 위한 완전한 Full-Stack 자동매매 생태계**

## 🌟 시스템 개요

oepnStock은 4단계 체크리스트 기반 트레이딩 엔진을 중심으로 한 **종합적인 자동매매 플랫폼**입니다. 실시간 알림, 웹 대시보드, 모바일 API, 고급 백테스트까지 포함한 프로덕션 레디 시스템입니다.

### ✨ 핵심 시스템

- **🎯 4단계 트레이딩 엔진**: 시장분석 → 지지선탐지 → 신호확인 → 리스크관리
- **📱 실시간 알림 시스템**: 텔레그램/이메일 멀티채널 알림 with HTML 템플릿
- **🌐 웹 대시보드**: Flask + Socket.IO 실시간 모니터링 & 제어
- **📱 모바일 API**: FastAPI + WebSocket + JWT 인증 완전 지원
- **🧪 고급 백테스트**: Walk-Forward, 몬테카를로, 다중시나리오 분석
- **🛡️ 엔터프라이즈 보안**: JWT 인증, 역할기반 접근제어, 입력검증

## 📁 시스템 아키텍처

```
oepnStock/
├── 🎯 핵심 트레이딩 엔진
│   ├── core/stage1_market_flow/     # 시장 흐름 분석
│   ├── core/stage2_support_detection/  # 지지선 감지
│   ├── core/stage3_signal_confirmation/  # 신호 확인
│   └── core/stage4_risk_management/     # 리스크 관리
├── 📱 알림 시스템
│   ├── notification/telegram_notifier.py    # 텔레그램 봇
│   ├── notification/email_notifier.py       # HTML 이메일
│   └── notification/alert_manager.py        # 통합 알림 관리
├── 🌐 웹 플랫폼
│   ├── dashboard/web_dashboard.py           # Flask 서버
│   ├── dashboard/data_manager.py            # 데이터 관리
│   └── dashboard/templates/dashboard.html   # 반응형 UI
├── 📱 모바일 API
│   ├── mobile/api_server.py                # FastAPI 서버
│   ├── mobile/auth.py                      # JWT 인증
│   └── mobile/models.py                    # API 모델
├── 🧪 백테스트 엔진
│   ├── backtest/advanced_backtester.py     # 다중시나리오
│   ├── backtest/walk_forward_analyzer.py   # 시계열 검증
│   ├── backtest/monte_carlo_simulator.py   # 리스크 분석
│   └── backtest/performance_metrics.py     # 고급 지표
└── 📚 설정 & 예제
    ├── config/alert_config.json            # 알림 규칙
    ├── examples/                           # 실행 예제
    └── SETUP_GUIDE.md                      # 종합 설정 가이드
```

## 🚀 즉시 실행하기

### 1️⃣ 환경 설정
```bash
# 환경 변수 설정 (.env 파일)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

### 2️⃣ 시스템 실행
```bash
# 🌐 웹 대시보드 (localhost:5000)
python examples/web_dashboard_example.py

# 📱 모바일 API 서버 (localhost:8000)
python examples/mobile_api_example.py

# 📱 알림 시스템 테스트
python examples/notification_system_example.py

# 🧪 고급 백테스트 실행
python examples/advanced_backtest_example.py
```

### 3️⃣ API 사용 예제
```bash
# 로그인 (demo/demo123!)
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo","password":"demo123!"}'

# 대시보드 조회 (토큰 필요)
curl -X GET http://localhost:8000/api/v1/dashboard/overview \
  -H 'Authorization: Bearer <TOKEN>'

# 거래 제어
curl -X POST http://localhost:8000/api/v1/trading/control \
  -H 'Authorization: Bearer <TOKEN>' \
  -d '{"action":"pause","duration_hours":1}'
```

## 🎯 4-Phase 트레이딩 시스템

### Phase 1: 알림 시스템 📱
**완전 비동기 멀티채널 알림**
- **텔레그램**: 실시간 거래/리스크 알림 with 메시지 큐
- **이메일**: HTML 템플릿 + 차트 첨부 자동 리포트
- **통합 관리**: JSON 규칙 엔진 + 쿨다운 제어

### Phase 2: 고급 백테스트 🧪
**다중 시나리오 & 시계열 검증**
- **4개 시장 환경**: 강세/약세/횡보/고변동성 자동 분석
- **Walk-Forward**: 252일 훈련 → 63일 테스트 롤링 검증
- **몬테카를로**: 1000회 시뮬레이션 리스크 분석
- **15개 고급 지표**: Sharpe, Sortino, Calmar, VaR, CVaR

### Phase 3: 웹 대시보드 🌐
**실시간 모니터링 & 제어**
- **Flask + Socket.IO**: 5초마다 라이브 데이터 스트리밍
- **인터랙티브 차트**: Plotly.js 자산곡선/수익률/드로다운
- **반응형 UI**: TailwindCSS 모바일 최적화
- **원격 제어**: 실시간 거래 일시정지/재개

### Phase 4: 모바일 API 📱
**FastAPI + WebSocket + JWT 보안**
- **15개 REST 엔드포인트**: 대시보드/포지션/거래/알림/제어
- **실시간 WebSocket**: 양방향 통신 + 자동 재연결
- **JWT 인증**: Access/Refresh 토큰 + 역할기반 접근제어
- **자동 문서화**: Swagger UI + ReDoc

## 💡 주요 기술적 성취

### 🚀 완전한 비동기 아키텍처
- **알림 시스템**: asyncio 논블로킹 + 메시지 큐
- **웹 대시보드**: Socket.IO 실시간 양방향 통신
- **API 서버**: FastAPI async/await 고성능
- **백테스트**: concurrent.futures 병렬 처리

### 🔒 엔터프라이즈급 보안
- **JWT 인증**: Access/Refresh 토큰 자동 갱신
- **역할 기반 접근**: Admin/User/Viewer 권한 분리
- **입력 검증**: Pydantic 완전한 타입 안전성
- **세션 관리**: 만료 세션 자동 정리

### 🏗️ 확장 가능한 설계
- **모듈화**: 각 시스템 완전 독립 실행 가능
- **설정 기반**: JSON/YAML 파일로 유연한 구성
- **플러그인 구조**: 새로운 알림 채널 쉬운 추가
- **마이크로서비스**: API 서버와 대시보드 분리

### 📊 실시간 모든 것
- **5초 업데이트**: 모든 데이터 실시간 스트리밍
- **즉각적 피드백**: 사용자 액션에 즉시 반응
- **라이브 차트**: 자산곡선/수익률 실시간 업데이트
- **푸시 알림**: 중요 이벤트 즉시 전송

## 🎛️ 시스템 사용 예제

### 📱 알림 시스템 통합 예제
```python
from oepnstock.notification import AlertManager

# 통합 알림 매니저 초기화
alert_manager = AlertManager()

# 거래 성공 알림
await alert_manager.send_trade_alert(
    symbol="005930", action="BUY", 
    price=65000, quantity=10, profit_pct=2.5
)

# 리스크 경고 알림
await alert_manager.send_risk_alert(
    alert_type="position_limit", 
    message="포지션 한도 90% 도달", 
    severity="high"
)

# 일일 리포트 자동 발송
await alert_manager.send_daily_report()
```

### 🌐 웹 대시보드 실시간 제어
```python
from oepnstock.dashboard import WebDashboard

# 대시보드 서버 초기화
dashboard = WebDashboard(data_manager, host='0.0.0.0', port=5000)

# 실시간 데이터 브로드캐스트
@dashboard.socketio.on('request_update')
def handle_update_request():
    live_data = dashboard.data_manager.get_live_data()
    dashboard.socketio.emit('live_update', live_data)

# 거래 제어 핸들러
@dashboard.socketio.on('trading_control')
def handle_trading_control(data):
    action = data['action']  # 'pause' or 'resume'
    result = dashboard.data_manager.control_trading(action)
    dashboard.socketio.emit('control_response', result)
```

### 📱 모바일 API 인증 & 사용
```python
import requests

# 1. 로그인 토큰 획득
login_response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    json={"username": "demo", "password": "demo123!"}
)
token = login_response.json()["access_token"]

# 2. 인증 헤더 설정
headers = {"Authorization": f"Bearer {token}"}

# 3. 실시간 대시보드 조회
dashboard_data = requests.get(
    "http://localhost:8000/api/v1/dashboard/overview", 
    headers=headers
).json()

# 4. 거래 제어 요청
control_response = requests.post(
    "http://localhost:8000/api/v1/trading/control",
    headers=headers,
    json={"action": "pause", "duration_hours": 2}
)
```

## 🧪 고급 백테스트 활용

### 다중 시나리오 백테스트
```python
from oepnstock.backtest import AdvancedBacktester

# 여러 투자금액으로 성과 비교
backtester = AdvancedBacktester()
results = backtester.run_comprehensive_backtest(
    strategy=YourStrategy(),
    capital_levels=[1_000_000, 3_000_000, 5_000_000, 10_000_000]
)

# 시장 상황별 가중 결과
weighted_result = backtester.calculate_weighted_performance(results)
print(f"종합 샤프 비율: {weighted_result.sharpe_ratio:.2f}")
print(f"최대 드로다운: {weighted_result.max_drawdown:.2%}")
```

### Walk-Forward 시계열 검증
```python
from oepnstock.backtest import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer()
wf_results = analyzer.run_walk_forward_analysis(
    strategy=YourStrategy(),
    window_size=252,  # 1년 훈련
    step_size=63      # 3개월 테스트
)

# 시간대별 성과 분석
for period_result in wf_results.period_results:
    print(f"{period_result.period}: 수익률 {period_result.return_pct:.2%}")
```

### 몬테카를로 리스크 분석
```python
from oepnstock.backtest import MonteCarloSimulator

simulator = MonteCarloSimulator()
mc_results = simulator.run_simulation(
    strategy=YourStrategy(), 
    n_simulations=1000
)

print(f"95% 신뢰구간 VaR: {mc_results.var_95:.2%}")
print(f"CVaR (조건부 VaR): {mc_results.cvar_95:.2%}")
print(f"꼬리 위험 비율: {mc_results.tail_ratio:.2f}")
```

## ⚙️ 고급 설정 & 커스터마이징

### 알림 규칙 커스터마이징
```json
// config/alert_config.json
{
  "risk_alerts": {
    "drawdown_threshold": 0.05,
    "position_limit_threshold": 0.9,
    "volatility_spike_threshold": 2.0
  },
  "notification_settings": {
    "telegram_enabled": true,
    "email_enabled": true,
    "cooldown_minutes": 30
  }
}
```

### 대시보드 설정 커스터마이징
```python
from oepnstock.dashboard import WebDashboard

dashboard = WebDashboard(
    data_manager=data_manager,
    host='0.0.0.0',
    port=5000,
    update_interval=5,  # 5초마다 업데이트
    chart_history_days=30,  # 30일 차트 히스토리
    enable_remote_control=True  # 원격 제어 활성화
)
```

### API 보안 설정 강화
```python
# mobile/auth.py 커스터마이징
JWT_SETTINGS = {
    "secret_key": "your_super_secure_key",
    "access_token_expire_minutes": 30,
    "refresh_token_expire_days": 7,
    "algorithm": "HS256"
}

# IP 화이트리스트 설정
ALLOWED_IPS = ["192.168.1.0/24", "10.0.0.0/8"]
```

## 🔮 Next Phase 로드맵

### Phase 5: 실제 브로커 연동 🏦
- **키움증권 OpenAPI**: 실제 주문/체결 시스템 구축
- **실시간 데이터**: 호가/체결 데이터 스트리밍
- **주문 관리**: 지정가/시장가 주문 완전 지원
- **계좌 동기화**: 실제 잔고와 시스템 실시간 동기화

### Phase 6: AI 기능 강화 🤖
- **GPT 기반 분석**: 뉴스/공시 자동 해석 및 투자 의견
- **강화학습 최적화**: 매개변수 자동 튜닝 시스템
- **패턴 인식**: 차트 패턴 자동 감지 및 분류
- **감정 분석**: 시장 심리 지표 실시간 모니터링

### Phase 7: 확장 생태계 🌐
- **모바일 앱**: React Native 네이티브 앱 개발
- **클라우드 배포**: AWS/Azure 완전 관리형 서비스
- **멀티 브로커**: 여러 증권사 통합 지원
- **커뮤니티**: 전략 공유 및 백테스트 경쟁 플랫폼

## 📚 문서 & 설정 가이드

### 🛠️ 핵심 설정 문서
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: 완전한 시스템 설정 가이드
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)**: 현재 구현 상태 및 히스토리
- **[CLAUDE.md](CLAUDE.md)**: 프로젝트 아키텍처 및 개발 가이드

### 📊 실행 예제 모음
```bash
# 🌐 웹 대시보드 (http://localhost:5000)
python examples/web_dashboard_example.py

# 📱 모바일 API 서버 (http://localhost:8000/docs)
python examples/mobile_api_example.py

# 📱 알림 시스템 통합 테스트
python examples/notification_system_example.py

# 🧪 고급 백테스트 (4시나리오 + Walk-Forward + Monte Carlo)
python examples/advanced_backtest_example.py
```

### 🔧 설정 파일 구조
```
config/
├── alert_config.json          # 알림 규칙 설정
├── .env                       # 환경 변수 (토큰, 비밀번호)
└── settings.py                # 시스템 전역 설정

data/
├── oepnstock.db              # SQLite 데이터베이스
└── backtest_cache/           # 백테스트 결과 캐시

logs/
├── oepnstock.log             # 시스템 로그
├── trading.log               # 거래 로그
└── notifications.log         # 알림 로그
```

### 🔗 API 문서 링크
- **Swagger UI**: http://localhost:8000/docs (API 서버 실행 시)
- **ReDoc**: http://localhost:8000/redoc (자동 생성 API 문서)
- **WebSocket**: ws://localhost:8000/ws (실시간 통신)

## 🎯 핵심 성과 요약

### ✅ 완료된 구현 사항
- **📱 실시간 알림 시스템**: 텔레그램/이메일 멀티채널 with HTML 템플릿
- **🌐 웹 대시보드**: Flask + Socket.IO 실시간 모니터링 & 원격 제어
- **📱 모바일 API**: FastAPI + JWT + WebSocket 완전한 모바일 지원
- **🧪 고급 백테스트**: 4시나리오 + Walk-Forward + Monte Carlo 리스크 분석
- **🛡️ 엔터프라이즈 보안**: JWT 인증, 역할기반 접근제어, 완전한 입력검증
- **⚡ 완전한 비동기**: asyncio + 병렬처리로 고성능 실시간 시스템

### 🚀 즉시 사용 가능
```bash
# 1. 웹 대시보드 실행
python examples/web_dashboard_example.py

# 2. API 서버 실행 
python examples/mobile_api_example.py

# 3. 시스템 상태 확인
curl http://localhost:8000/api/v1/system/status
```

## ⚠️ 면책 조항

이 시스템은 교육 및 연구 목적으로 제공됩니다. 실제 투자에는 항상 리스크가 따르므로:

1. **충분한 검증**: 페이퍼 트레이딩으로 시스템 검증 필수
2. **리스크 관리**: 개인 투자 가능 범위 내에서만 사용
3. **지속적 모니터링**: 시장 환경 변화에 따른 조정 필요
4. **분산 투자**: 단일 시스템에 의존하지 말고 포트폴리오 분산

---

🎉 **oepnStock** - 한국 주식 시장을 위한 **완전한 종합 자동매매 플랫폼** 🎉

*실시간 알림부터 모바일 API까지, 프로덕션 레디 Full-Stack 시스템*