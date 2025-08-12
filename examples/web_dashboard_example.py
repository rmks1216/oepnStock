#!/usr/bin/env python3
"""
웹 대시보드 실행 예제
Flask 기반 실시간 모니터링 대시보드
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from oepnstock.dashboard.web_dashboard import WebDashboard
from oepnstock.dashboard.data_manager import DashboardDataManager


def main():
    """메인 실행 함수"""
    print("🚀 웹 대시보드 시작")
    print("=" * 50)
    
    try:
        # 데이터 매니저 생성
        print("📊 데이터 매니저 초기화 중...")
        data_manager = DashboardDataManager()
        
        # 웹 대시보드 생성
        print("🌐 웹 대시보드 초기화 중...")
        dashboard = WebDashboard(
            data_manager=data_manager,
            host='0.0.0.0',
            port=5000
        )
        
        print("✅ 초기화 완료!")
        print("\n📋 접속 정보:")
        print("  - 로컬 접속: http://localhost:5000")
        print("  - 네트워크 접속: http://[YOUR_IP]:5000")
        print("\n🔧 주요 기능:")
        print("  - 실시간 자산 곡선")
        print("  - 일일 수익률 차트")
        print("  - 현재 포지션 현황")
        print("  - 최근 거래 내역")
        print("  - 리스크 지표 모니터링")
        print("  - 거래 제어 (일시정지/재개)")
        
        print("\n🚨 주의사항:")
        print("  - 실제 거래 시스템과 연결하려면 데이터 소스를 설정하세요")
        print("  - 현재는 시뮬레이션 데이터를 사용합니다")
        print("  - Ctrl+C로 서버를 중단할 수 있습니다")
        
        print("\n🌐 서버 시작 중...")
        
        # 대시보드 서버 실행
        dashboard.run(debug=True)
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 서버가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()