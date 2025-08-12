#!/usr/bin/env python3
"""
모바일 API 서버 실행 예제
FastAPI 기반 REST API + WebSocket 서버
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from oepnstock.mobile.api_server import run_mobile_api_server, create_mobile_api_app
from oepnstock.dashboard.data_manager import DashboardDataManager


def main():
    """메인 실행 함수"""
    print("🚀 모바일 API 서버 시작")
    print("=" * 50)
    
    try:
        # 데이터 매니저 생성
        print("📊 데이터 매니저 초기화 중...")
        data_manager = DashboardDataManager()
        
        print("✅ 초기화 완료!")
        print("\n📋 API 서버 정보:")
        print("  - 서버 주소: http://localhost:8000")
        print("  - API 문서: http://localhost:8000/docs")
        print("  - WebSocket: ws://localhost:8000/ws")
        
        print("\n🔐 인증 정보:")
        print("  - 관리자: username=admin, password=admin123!")
        print("  - 데모: username=demo, password=demo123!")
        
        print("\n🛠️ 주요 엔드포인트:")
        print("  POST /api/v1/auth/login - 로그인")
        print("  GET  /api/v1/dashboard/overview - 대시보드 개요")
        print("  GET  /api/v1/positions - 현재 포지션")
        print("  GET  /api/v1/trades - 거래 내역")
        print("  GET  /api/v1/alerts - 최근 알림")
        print("  POST /api/v1/trading/control - 거래 제어")
        print("  GET  /api/v1/charts/equity - 자산 곡선")
        print("  GET  /api/v1/system/status - 시스템 상태")
        
        print("\n🔗 사용 예제:")
        print("  curl -X POST http://localhost:8000/api/v1/auth/login \\")
        print("    -H 'Content-Type: application/json' \\")
        print("    -d '{\"username\":\"demo\",\"password\":\"demo123!\"}'")
        
        print("\n🚨 주의사항:")
        print("  - 실제 환경에서는 JWT 시크릿 키를 변경하세요")
        print("  - CORS 설정을 프로덕션에 맞게 조정하세요")
        print("  - 현재는 시뮬레이션 데이터를 사용합니다")
        print("  - Ctrl+C로 서버를 중단할 수 있습니다")
        
        print("\n🌐 서버 시작 중...")
        
        # API 서버 실행
        run_mobile_api_server(
            host="0.0.0.0",
            port=8000,
            data_manager=data_manager,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 서버가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def test_api_client():
    """API 클라이언트 테스트"""
    import requests
    import json
    
    print("🧪 API 클라이언트 테스트")
    print("=" * 30)
    
    base_url = "http://localhost:8000/api/v1"
    
    try:
        # 1. 로그인
        print("1. 로그인 테스트")
        login_data = {
            "username": "demo",
            "password": "demo123!"
        }
        
        response = requests.post(f"{base_url}/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data["access_token"]
            print(f"✅ 로그인 성공: {access_token[:20]}...")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            # 2. 대시보드 개요 조회
            print("\n2. 대시보드 개요 조회")
            response = requests.get(f"{base_url}/dashboard/overview", headers=headers)
            if response.status_code == 200:
                overview = response.json()
                print("✅ 대시보드 데이터:")
                print(f"  총 자산: ₩{overview['total_asset']:,.0f}")
                print(f"  일일 수익률: {overview['daily_return']:.2%}")
                print(f"  리스크 레벨: {overview['risk_level']}")
            else:
                print(f"❌ 대시보드 조회 실패: {response.status_code}")
            
            # 3. 포지션 조회
            print("\n3. 현재 포지션 조회")
            response = requests.get(f"{base_url}/positions", headers=headers)
            if response.status_code == 200:
                positions = response.json()
                print(f"✅ 현재 포지션 수: {len(positions)}")
                for pos in positions:
                    print(f"  {pos['symbol']}: {pos['quantity']}주 ({pos['unrealized_pnl_pct']:.2%})")
            else:
                print(f"❌ 포지션 조회 실패: {response.status_code}")
            
            # 4. 거래 제어 테스트
            print("\n4. 거래 제어 테스트")
            control_data = {
                "action": "pause",
                "duration_hours": 1,
                "reason": "API 테스트"
            }
            response = requests.post(f"{base_url}/trading/control", headers=headers, json=control_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 거래 제어: {result['status']} - {result['message']}")
            else:
                print(f"❌ 거래 제어 실패: {response.status_code}")
                
        else:
            print(f"❌ 로그인 실패: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ API 서버에 연결할 수 없습니다.")
        print("   먼저 'python mobile_api_example.py'로 서버를 시작하세요.")
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")


def websocket_test():
    """WebSocket 클라이언트 테스트"""
    try:
        import websocket
        import json
        import threading
        import time
        
        print("\n🔌 WebSocket 클라이언트 테스트")
        print("=" * 30)
        
        def on_message(ws, message):
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            
            if msg_type == "live_update":
                live_data = data["data"]
                print(f"📊 실시간 업데이트: 자산 ₩{live_data['current_capital']:,.0f}, "
                      f"수익률 {live_data['daily_return']:.2%}")
            elif msg_type == "pong":
                print("🏓 Pong received")
            else:
                print(f"📨 메시지 수신: {data}")
        
        def on_error(ws, error):
            print(f"❌ WebSocket 오류: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("🔌 WebSocket 연결 종료")
        
        def on_open(ws):
            print("✅ WebSocket 연결 성공")
            
            def run():
                time.sleep(1)
                # 핑 전송
                ws.send(json.dumps({"type": "ping"}))
                
                time.sleep(2)
                # 구독 요청
                ws.send(json.dumps({
                    "type": "subscribe",
                    "data": {"subscription": "live_updates"}
                }))
                
                # 10초 후 연결 종료
                time.sleep(10)
                ws.close()
            
            thread = threading.Thread(target=run)
            thread.start()
        
        # WebSocket 연결
        ws = websocket.WebSocketApp(
            "ws://localhost:8000/ws",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        ws.run_forever()
        
    except ImportError:
        print("❌ websocket-client 패키지가 필요합니다:")
        print("   pip install websocket-client")
    except Exception as e:
        print(f"❌ WebSocket 테스트 오류: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_api_client()
        elif sys.argv[1] == "websocket":
            websocket_test()
        else:
            print("사용법:")
            print("  python mobile_api_example.py        # API 서버 실행")
            print("  python mobile_api_example.py test   # API 클라이언트 테스트")
            print("  python mobile_api_example.py websocket  # WebSocket 테스트")
    else:
        main()