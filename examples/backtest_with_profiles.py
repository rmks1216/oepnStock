"""
백테스팅 프로파일 활용 예제
다양한 전략 설정으로 백테스트 실행하여 성과 비교
"""

import sys
import os
from datetime import date
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.backtesting_example import SimpleBacktester
from oepnstock.utils.config import ConfigManager
from oepnstock.utils import get_logger

logger = get_logger(__name__)


def run_profile_comparison(profiles: List[str] = None) -> Dict[str, Dict]:
    """
    여러 프로파일로 백테스트 실행하여 성과 비교
    
    Args:
        profiles: 비교할 프로파일 리스트 (None이면 모든 프로파일)
    
    Returns:
        Dict: 각 프로파일별 백테스트 결과
    """
    if profiles is None:
        profiles = ['default', 'aggressive', 'conservative', 'scalping', 'swing']
    
    config_manager = ConfigManager()
    results = {}
    
    print("=" * 80)
    print("📊 백테스팅 프로파일 성과 비교")
    print("=" * 80)
    print()
    
    for profile_name in profiles:
        print(f"🔄 실행 중: {profile_name} 프로파일...")
        
        try:
            # 프로파일 적용
            custom_config = config_manager.apply_backtest_profile(profile_name)
            
            # 백테스터 생성
            backtester = SimpleBacktester(custom_config=custom_config)
            
            # 백테스트 실행
            result = backtester.run_backtest()
            
            if 'error' not in result:
                results[profile_name] = {
                    'config_name': custom_config.backtest.__dict__.get('name', profile_name),
                    'result': result
                }
                
                # 간단한 결과 출력
                print(f"✅ {profile_name}: 총 수익률 {result['total_return']:.2%}, "
                      f"거래 {result['total_trades']}회, 승률 {result['win_rate']:.1%}")
            else:
                print(f"❌ {profile_name}: 오류 - {result['error']}")
                
        except Exception as e:
            logger.error(f"Error running profile {profile_name}: {e}")
            print(f"❌ {profile_name}: 실행 오류 - {str(e)}")
        
        print()
    
    return results


def print_comparison_report(results: Dict[str, Dict]):
    """프로파일 비교 리포트 출력"""
    
    if not results:
        print("❌ 비교할 결과가 없습니다.")
        return
    
    print("=" * 80)
    print("📈 성과 비교 리포트")
    print("=" * 80)
    print()
    
    # 성과 지표별 비교표
    metrics = [
        ('총 수익률', 'total_return', '{:.2%}'),
        ('연환산 수익률', 'annualized_return', '{:.2%}'),
        ('샤프 비율', 'sharpe_ratio', '{:.2f}'),
        ('최대 낙폭', 'max_drawdown', '{:.2%}'),
        ('총 거래 횟수', 'total_trades', '{:,}회'),
        ('승률', 'win_rate', '{:.1%}'),
        ('거래당 평균 수익', 'avg_profit_per_trade', '{:,.0f}원')
    ]
    
    # 헤더 출력
    header = f"{'지표':<15}"
    for profile in results.keys():
        header += f"{profile:<15}"
    print(header)
    print("-" * len(header))
    
    # 각 지표별 비교
    for metric_name, metric_key, format_str in metrics:
        row = f"{metric_name:<15}"
        for profile, data in results.items():
            value = data['result'].get(metric_key, 0)
            formatted_value = format_str.format(value) if value != 0 else "N/A"
            row += f"{formatted_value:<15}"
        print(row)
    
    print()
    
    # 최고 성과 프로파일 찾기
    best_return = max(results.items(), key=lambda x: x[1]['result'].get('total_return', -999))
    best_sharpe = max(results.items(), key=lambda x: x[1]['result'].get('sharpe_ratio', -999))
    best_trades = max(results.items(), key=lambda x: x[1]['result'].get('total_trades', 0))
    
    print("🏆 최고 성과:")
    print(f"  • 최고 수익률: {best_return[0]} ({best_return[1]['result']['total_return']:.2%})")
    print(f"  • 최고 샤프 비율: {best_sharpe[0]} ({best_sharpe[1]['result']['sharpe_ratio']:.2f})")
    print(f"  • 최다 거래: {best_trades[0]} ({best_trades[1]['result']['total_trades']:,}회)")
    print()
    
    # 권장사항
    print("💡 권장사항:")
    
    profitable_profiles = [name for name, data in results.items() 
                          if data['result'].get('total_return', 0) > 0]
    
    if profitable_profiles:
        print(f"  • 수익 창출 프로파일: {', '.join(profitable_profiles)}")
        
        stable_profiles = [name for name, data in results.items() 
                          if data['result'].get('sharpe_ratio', 0) > 0.5 
                          and data['result'].get('total_return', 0) > 0]
        
        if stable_profiles:
            print(f"  • 안정적 수익 프로파일: {', '.join(stable_profiles)}")
        else:
            print("  • 모든 프로파일의 리스크 조정 수익률이 낮습니다. 전략 개선이 필요합니다.")
    else:
        print("  • 모든 프로파일에서 손실이 발생했습니다. 시장 상황 또는 전략 재검토가 필요합니다.")


def main():
    """메인 실행 함수"""
    
    # 사용 가능한 프로파일 확인
    config_manager = ConfigManager()
    
    # 모든 프로파일로 백테스트 실행
    print("🚀 다중 프로파일 백테스팅을 시작합니다...")
    print()
    
    results = run_profile_comparison()
    
    # 비교 리포트 출력
    print_comparison_report(results)
    
    print()
    print("✅ 프로파일 비교 백테스팅 완료!")


if __name__ == "__main__":
    main()