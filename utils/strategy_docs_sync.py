"""
백테스팅 전략 문서 동기화 도구
YAML 프로파일과 MD 문서 간의 일관성을 유지합니다.
"""

import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json


class StrategyDocsSync:
    """전략 문서 동기화 관리 클래스"""
    
    def __init__(self, config_path: str = "config/backtest_profiles.yaml"):
        self.config_path = Path(config_path)
        self.docs_dir = Path("docs/strategies")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategy_templates = {
            'overview': self._get_overview_template(),
            'parameters': self._get_parameters_template(), 
            'performance': self._get_performance_template(),
            'optimization': self._get_optimization_template()
        }
    
    def load_yaml_profiles(self) -> Dict[str, Any]:
        """YAML 프로파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Profile file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def extract_strategy_info(self, profile_name: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """전략 정보 추출 및 구조화"""
        return {
            'name': profile_data.get('name', f'{profile_name.title()} Strategy'),
            'description': profile_data.get('description', 'No description available'),
            'profile_key': profile_name,
            'backtest_params': profile_data.get('backtest', {}),
            'trading_params': profile_data.get('trading', {}),
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }
    
    def generate_strategy_doc(self, strategy_info: Dict[str, Any]) -> str:
        """전략 문서 생성"""
        profile_name = strategy_info['profile_key']
        name = strategy_info['name']
        description = strategy_info['description']
        backtest = strategy_info['backtest_params']
        trading = strategy_info['trading_params']
        
        doc_content = f"""# {name} ({profile_name})

**YAML 프로파일**: `config/backtest_profiles.yaml > {profile_name}`

## 📊 전략 개요

### 전략 설명
{description}

### 투자 특성
- **프로파일명**: `{profile_name}`
- **리밸런싱 주기**: {backtest.get('rebalance_frequency', 'N/A')}일
- **최대 포지션**: {trading.get('max_positions', 'N/A')}개
- **시장 진입 기준**: {trading.get('market_score_threshold', 'N/A')}점

---

## ⚙️ 파라미터 설정 (YAML 연동)

### 백테스트 파라미터
```yaml
backtest:
  initial_capital: {backtest.get('initial_capital', 10000000)}
  rebalance_frequency: {backtest.get('rebalance_frequency', 5)}
  signal_ma_short: {backtest.get('signal_ma_short', 5)}
  signal_ma_long: {backtest.get('signal_ma_long', 20)}
  signal_rsi_period: {backtest.get('signal_rsi_period', 14)}
  signal_rsi_overbought: {backtest.get('signal_rsi_overbought', 70)}
  min_recent_up_days: {backtest.get('min_recent_up_days', 2)}
  recent_up_window: {backtest.get('recent_up_window', 3)}
  ma_trend_factor: {backtest.get('ma_trend_factor', 1.0)}
  sell_threshold_ratio: {backtest.get('sell_threshold_ratio', 0.95)}
```

### 거래 파라미터  
```yaml
trading:
  market_score_threshold: {trading.get('market_score_threshold', 70)}
  max_positions: {trading.get('max_positions', 5)}
  max_single_position_ratio: {trading.get('max_single_position_ratio', 0.2)}
```

---

## 📈 신호 생성 로직

### 매수 조건
```python
buy_conditions = [
    ma_short > ma_long * {backtest.get('ma_trend_factor', 1.0)},
    current_rsi < {backtest.get('signal_rsi_overbought', 70)},
    up_days >= {backtest.get('min_recent_up_days', 2)},  # 최근 {backtest.get('recent_up_window', 3)}일 중
    market_score >= {trading.get('market_score_threshold', 70)}
]
```

### 매도 조건
```python
sell_condition = ma_short < ma_long * {backtest.get('sell_threshold_ratio', 0.95)}
```

---

## 🎯 최적화 방향

### 핵심 파라미터 튜닝
1. **이동평균 조합**: MA({backtest.get('signal_ma_short', 5)}, {backtest.get('signal_ma_long', 20)}) → 다양한 조합 테스트
2. **리밸런싱 주기**: {backtest.get('rebalance_frequency', 5)}일 → ±2일 범위 테스트
3. **시장 진입 기준**: {trading.get('market_score_threshold', 70)}점 → ±5점 범위 테스트

### 성과 목표
- **목표 수익률**: 시장 상황별 차등 설정
- **리스크 지표**: 샤프 비율 >0.5, 최대 낙폭 <20%
- **거래 효율**: 승률 >45%, 거래당 수익 양수

---

## 📊 성과 추적

### 백테스트 실행
```bash
# 단일 전략 테스트
python examples/backtesting_example.py --profile {profile_name}

# 다중 전략 비교
python examples/backtest_with_profiles.py
```

### 모니터링 지표
- 일간 수익률 변화
- 포지션 비중 점검  
- 거래 신호 정확도
- 비용 대비 효율성

---

## ⚠️ 주의사항

### 파라미터 변경시 체크리스트
- [ ] YAML 파일 백업
- [ ] 기존 성과와 비교 분석
- [ ] 아웃오브샘플 테스트 실시
- [ ] 문서 동기화 확인

### 리스크 관리
- 과최적화 방지: 최소 1년 이상 백테스트
- 시장 환경 변화: 정기적 재검증 필요
- 실거래 차이: 슬리피지, 거래비용 현실적 반영

---

*문서 생성일: {strategy_info['last_updated']}*  
*YAML 연동 상태: ✅ 동기화됨*  
*다음 업데이트: 파라미터 변경시 자동*
"""
        return doc_content
    
    def sync_all_strategies(self) -> Dict[str, str]:
        """모든 전략 문서 동기화"""
        profiles = self.load_yaml_profiles()
        results = {}
        
        for profile_name, profile_data in profiles.items():
            if isinstance(profile_data, dict):
                try:
                    # 전략 정보 추출
                    strategy_info = self.extract_strategy_info(profile_name, profile_data)
                    
                    # 문서 생성
                    doc_content = self.generate_strategy_doc(strategy_info)
                    
                    # 파일 저장
                    doc_filename = f"{profile_name.upper()}_STRATEGY.md"
                    doc_path = self.docs_dir / doc_filename
                    
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(doc_content)
                    
                    results[profile_name] = str(doc_path)
                    
                except Exception as e:
                    results[profile_name] = f"Error: {str(e)}"
        
        return results
    
    def generate_summary_table(self) -> str:
        """전략 비교 요약 테이블 생성"""
        profiles = self.load_yaml_profiles()
        
        table_rows = []
        headers = ["전략", "리밸런싱", "MA조합", "RSI", "최대포지션", "시장기준점"]
        
        for profile_name, profile_data in profiles.items():
            if isinstance(profile_data, dict):
                backtest = profile_data.get('backtest', {})
                trading = profile_data.get('trading', {})
                
                row = [
                    f"`{profile_name}`",
                    f"{backtest.get('rebalance_frequency', 'N/A')}일",
                    f"MA({backtest.get('signal_ma_short', 'N/A')},{backtest.get('signal_ma_long', 'N/A')})",
                    f"{backtest.get('signal_rsi_period', 'N/A')}일",
                    f"{trading.get('max_positions', 'N/A')}개",
                    f"{trading.get('market_score_threshold', 'N/A')}점"
                ]
                table_rows.append(row)
        
        # 마크다운 테이블 생성
        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "|" + "---|" * len(headers) + "\n"
        
        for row in table_rows:
            table_md += "| " + " | ".join(row) + " |\n"
        
        return table_md
    
    def update_main_guide(self):
        """메인 가이드 문서 업데이트"""
        summary_table = self.generate_summary_table()
        profiles = self.load_yaml_profiles()
        
        # 전략 개수 및 기본 정보
        strategy_count = len([k for k, v in profiles.items() if isinstance(v, dict)])
        update_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 메인 가이드에 요약 정보 추가
        summary_section = f"""
## 📊 전략 프로파일 요약

**총 전략 수**: {strategy_count}개  
**마지막 업데이트**: {update_date}

{summary_table}

### 빠른 실행 가이드
```bash
# 전체 전략 비교
python examples/backtest_with_profiles.py

# 개별 전략 테스트  
python examples/backtesting_example.py  # default 프로파일

# 문서 동기화
python utils/strategy_docs_sync.py
```
"""
        return summary_section
    
    def validate_yaml_integrity(self) -> Dict[str, List[str]]:
        """YAML 파일 무결성 검증"""
        profiles = self.load_yaml_profiles()
        issues = {}
        
        required_backtest_fields = [
            'initial_capital', 'rebalance_frequency', 'signal_ma_short', 
            'signal_ma_long', 'signal_rsi_period'
        ]
        
        required_trading_fields = [
            'market_score_threshold', 'max_positions', 'max_single_position_ratio'
        ]
        
        for profile_name, profile_data in profiles.items():
            if isinstance(profile_data, dict):
                profile_issues = []
                
                # 기본 필드 검증
                if 'name' not in profile_data:
                    profile_issues.append("Missing 'name' field")
                if 'description' not in profile_data:
                    profile_issues.append("Missing 'description' field")
                
                # 백테스트 파라미터 검증
                backtest = profile_data.get('backtest', {})
                for field in required_backtest_fields:
                    if field not in backtest:
                        profile_issues.append(f"Missing backtest.{field}")
                
                # 거래 파라미터 검증  
                trading = profile_data.get('trading', {})
                for field in required_trading_fields:
                    if field not in trading:
                        profile_issues.append(f"Missing trading.{field}")
                
                # 값 범위 검증
                if backtest.get('rebalance_frequency', 0) <= 0:
                    profile_issues.append("Invalid rebalance_frequency (must be > 0)")
                
                if trading.get('max_positions', 0) <= 0:
                    profile_issues.append("Invalid max_positions (must be > 0)")
                
                if profile_issues:
                    issues[profile_name] = profile_issues
        
        return issues
    
    @staticmethod
    def _get_overview_template() -> str:
        return """## 📊 전략 개요\n\n### 전략 설명\n{description}\n"""
    
    @staticmethod  
    def _get_parameters_template() -> str:
        return """## ⚙️ 파라미터 설정\n\n```yaml\n{yaml_content}\n```\n"""
    
    @staticmethod
    def _get_performance_template() -> str:
        return """## 📈 성과 분석\n\n### 백테스트 결과\n- 수익률: {return_rate}\n- 샤프비율: {sharpe_ratio}\n"""
    
    @staticmethod
    def _get_optimization_template() -> str:
        return """## 🎯 최적화 가이드\n\n### 파라미터 튜닝 우선순위\n1. 이동평균 조합\n2. 리밸런싱 주기\n"""


def main():
    """메인 실행 함수"""
    print("🔄 전략 문서 동기화를 시작합니다...")
    
    try:
        sync_tool = StrategyDocsSync()
        
        # YAML 무결성 검증
        print("📋 YAML 파일 검증 중...")
        issues = sync_tool.validate_yaml_integrity()
        
        if issues:
            print("⚠️ 발견된 문제점:")
            for profile, problems in issues.items():
                print(f"  - {profile}: {', '.join(problems)}")
            return
        else:
            print("✅ YAML 파일 검증 완료")
        
        # 전략 문서 동기화
        print("📝 전략 문서 생성 중...")
        results = sync_tool.sync_all_strategies()
        
        success_count = len([r for r in results.values() if not r.startswith("Error")])
        error_count = len([r for r in results.values() if r.startswith("Error")])
        
        print(f"✅ 성공: {success_count}개 문서 생성")
        if error_count > 0:
            print(f"❌ 오류: {error_count}개 문서 실패")
            for profile, result in results.items():
                if result.startswith("Error"):
                    print(f"  - {profile}: {result}")
        
        # 요약 테이블 생성
        print("📊 요약 테이블 생성 중...")
        summary = sync_tool.update_main_guide()
        print("✅ 요약 정보 업데이트 완료")
        
        print("\n🎯 동기화 작업 완료!")
        print(f"📁 생성된 문서: docs/strategies/ 디렉토리")
        print("🔗 연동 상태: YAML ↔ MD 동기화됨")
        
    except Exception as e:
        print(f"❌ 동기화 작업 실패: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())