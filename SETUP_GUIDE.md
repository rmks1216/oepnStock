# 🚀 oepnStock 자동매매 시스템 설정 가이드

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [설치 방법](#설치-방법)
3. [환경 설정](#환경-설정)
4. [알림 시스템 설정](#알림-시스템-설정)
5. [웹 대시보드 실행](#웹-대시보드-실행)
6. [모바일 API 서버](#모바일-api-서버)
7. [백테스트 실행](#백테스트-실행)
8. [문제 해결](#문제-해결)

## 🎯 시스템 개요

oepnStock은 한국 주식 시장을 위한 포괄적인 자동매매 시스템입니다:

### 주요 구성 요소
- **4단계 트레이딩 엔진**: 시장분석 → 지지선탐지 → 신호확인 → 리스크관리
- **실시간 알림 시스템**: 텔레그램, 이메일을 통한 즉각적인 상황 알림
- **웹 대시보드**: 실시간 모니터링 및 제어 인터페이스
- **모바일 API**: REST API + WebSocket 지원으로 모바일 앱 연동
- **고급 백테스트**: Walk-Forward, 몬테카를로 시뮬레이션 지원
- **리스크 관리**: 다단계 리스크 제어 및 자동 대응

## 🛠️ 설치 방법

### 1. 시스템 요구사항
- Python 3.8 이상
- 최소 4GB RAM
- 안정적인 인터넷 연결

### 2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

주요 패키지:
```text
pandas>=1.5.0
numpy>=1.21.0
ta-lib>=0.4.25
matplotlib>=3.5.0
plotly>=5.0.0
flask>=2.0.0
flask-socketio>=5.0.0
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
websockets>=10.0
aiohttp>=3.8.0
python-telegram-bot>=20.0
jinja2>=3.0.0
bcrypt>=3.2.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
```

### 3. 프로젝트 설정
```bash
# 프로젝트 루트에서
python -m oepnstock.setup
```

## ⚙️ 환경 설정

### 1. 환경 변수 설정 (.env 파일)
```bash
# .env 파일 생성
touch .env
```

`.env` 파일 내용:
```bash
# === 텔레그램 봇 설정 ===
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# === 이메일 설정 (Gmail) ===
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
ALERT_EMAIL_RECIPIENT=recipient@gmail.com

# === 데이터베이스 설정 ===
DATABASE_URL=sqlite:///data/oepnstock.db

# === API 보안 설정 ===
JWT_SECRET_KEY=your_super_secret_jwt_key_here
API_KEY=your_api_key_here

# === 로깅 설정 ===
LOG_LEVEL=INFO
LOG_FILE=logs/oepnstock.log
```

### 2. 디렉토리 구조 생성
```bash
mkdir -p data logs backtest_cache walk_forward_results
```

## 📱 알림 시스템 설정

### 1. 텔레그램 봇 생성
1. @BotFather에게 `/newbot` 명령 전송
2. 봇 이름과 사용자명 설정
3. 받은 토큰을 `TELEGRAM_BOT_TOKEN`에 설정
4. 봇과 대화하여 Chat ID 확인:
   ```bash
   curl https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```

### 2. Gmail 앱 비밀번호 설정
1. Google 계정 → 보안 → 2단계 인증 활성화
2. 앱 비밀번호 생성
3. 16자리 비밀번호를 `EMAIL_PASSWORD`에 설정

### 3. 알림 테스트
```bash
python examples/notification_system_example.py
```

## 🌐 웹 대시보드 실행

### 1. 대시보드 서버 시작
```bash
python examples/web_dashboard_example.py
```

### 2. 접속 주소
- 로컬: http://localhost:5000
- 네트워크: http://[YOUR_IP]:5000

### 3. 주요 기능
- **실시간 자산 곡선**: 포트폴리오 가치 변화 추적
- **일일 수익률 차트**: 일별 성과 시각화
- **포지션 현황**: 현재 보유 종목 및 손익
- **리스크 모니터링**: 실시간 리스크 지표
- **거래 제어**: 일시정지/재개 기능

## 📱 모바일 API 서버

### 1. API 서버 시작
```bash
python examples/mobile_api_example.py
```

### 2. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. 인증 정보
- **관리자**: username=`admin`, password=`admin123!`
- **일반사용자**: username=`demo`, password=`demo123!`

### 4. 주요 엔드포인트
```bash
# 로그인
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo","password":"demo123!"}'

# 대시보드 개요 (토큰 필요)
curl -X GET http://localhost:8000/api/v1/dashboard/overview \
  -H 'Authorization: Bearer <TOKEN>'

# 현재 포지션
curl -X GET http://localhost:8000/api/v1/positions \
  -H 'Authorization: Bearer <TOKEN>'
```

### 5. WebSocket 연결
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    // 실시간 업데이트 구독
    ws.send(JSON.stringify({
        type: 'subscribe',
        data: {subscription: 'live_updates'}
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('실시간 데이터:', data);
};
```

## 🧪 백테스트 실행

### 1. 기본 백테스트
```bash
python examples/advanced_backtest_example.py
```

### 2. 투자금액별 성과 비교
```python
from oepnstock.backtest import AdvancedBacktester

backtester = AdvancedBacktester()
results = backtester.run_comprehensive_backtest(
    strategy=YourStrategy(),
    capital_levels=[1_000_000, 3_000_000, 5_000_000, 10_000_000]
)

# 결과 저장
backtester.save_results(results, "backtest_results.json")
```

### 3. Walk-Forward Analysis
```python
from oepnstock.backtest import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer()
wf_results = analyzer.run_walk_forward_analysis(
    strategy=YourStrategy(),
    window_size=252,  # 1년 훈련 기간
    step_size=63      # 3개월 테스트 기간
)
```

## 🔧 문제 해결

### 1. 일반적인 문제

#### 텔레그램 알림이 안 됨
```bash
# 토큰과 채팅 ID 확인
curl https://api.telegram.org/bot<TOKEN>/getMe
curl https://api.telegram.org/bot<TOKEN>/getUpdates
```

#### 이메일 알림 실패
- Gmail: 2단계 인증 활성화 필요
- 앱 비밀번호 사용 (일반 비밀번호 아님)
- SMTP 설정 확인: smtp.gmail.com:587

#### 웹 대시보드 접속 불가
```bash
# 포트 사용 확인
netstat -tulpn | grep 5000

# 방화벽 설정 확인
sudo ufw allow 5000
```

#### API 서버 오류
```bash
# 로그 확인
tail -f logs/oepnstock.log

# 의존성 재설치
pip install --upgrade -r requirements.txt
```

### 2. 성능 최적화

#### 메모리 사용량 최적화
```python
# config/settings.py
CACHE_SIZE_LIMIT = 1000  # 캐시 크기 제한
CLEANUP_INTERVAL = 3600  # 1시간마다 정리
```

#### 백테스트 속도 개선
```python
# 병렬 처리 활성화
backtester = AdvancedBacktester()
backtester.enable_parallel_processing(n_workers=4)
```

### 3. 보안 강화

#### JWT 토큰 보안
```python
# 강력한 시크릿 키 생성
import secrets
jwt_secret = secrets.token_urlsafe(32)
```

#### API 접근 제한
```python
# IP 화이트리스트 설정
ALLOWED_IPS = ['192.168.1.0/24', '10.0.0.0/8']
```

## 📞 지원 및 문의

### 문서 및 예제
- [프로젝트 위키](https://github.com/your-repo/oepnstock/wiki)
- [API 문서](http://localhost:8000/docs)
- [예제 코드](/examples/)

### 로그 위치
- 시스템 로그: `logs/oepnstock.log`
- 거래 로그: `logs/trading.log`
- 알림 로그: `logs/notifications.log`

### 디버그 모드 실행
```bash
# 상세 로그 출력
LOG_LEVEL=DEBUG python examples/web_dashboard_example.py

# 백테스트 디버그
python -m pdb examples/advanced_backtest_example.py
```

---

🎉 **축하합니다!** oepnStock 시스템 설정이 완료되었습니다.
실제 거래 전에 충분한 백테스트와 페이퍼 트레이딩을 통해 시스템을 검증하시기 바랍니다.