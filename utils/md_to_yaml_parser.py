"""
MD 파일에서 YAML 데이터를 추출하는 파서
전략 문서의 파라미터 정보를 구조화된 데이터로 변환합니다.
"""

import re
import yaml
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from datetime import datetime


class MarkdownYAMLParser:
    """마크다운 파일에서 YAML 데이터를 추출하는 파서"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # YAML 코드 블록 패턴
        self.yaml_pattern = re.compile(r'```yaml\s*\n(.*?)\n```', re.DOTALL)
        
        # 파라미터 추출 패턴들
        self.param_patterns = {
            'profile_name': re.compile(r'#\s+(.*?)\s*\(([^)]+)\)', re.MULTILINE),
            'name': re.compile(r'###?\s*전략\s*설명[^#]*?([^\n]+)', re.MULTILINE),
            'description': re.compile(r'###?\s*전략\s*설명\s*\n([^\n#]+)', re.MULTILINE),
            'rebalance_frequency': re.compile(r'리밸런싱\s*주기[:\s]*(\d+)일?', re.IGNORECASE),
            'max_positions': re.compile(r'최대\s*포지션[:\s]*(\d+)개?', re.IGNORECASE),
            'market_score_threshold': re.compile(r'시장\s*진입\s*기준[:\s]*(\d+)점?', re.IGNORECASE),
        }
        
        # 숫자 값 패턴
        self.numeric_patterns = {
            'initial_capital': re.compile(r'initial_capital:\s*(\d+)', re.IGNORECASE),
            'signal_ma_short': re.compile(r'signal_ma_short:\s*(\d+)', re.IGNORECASE),
            'signal_ma_long': re.compile(r'signal_ma_long:\s*(\d+)', re.IGNORECASE),
            'signal_rsi_period': re.compile(r'signal_rsi_period:\s*(\d+)', re.IGNORECASE),
            'signal_rsi_overbought': re.compile(r'signal_rsi_overbought:\s*(\d+)', re.IGNORECASE),
            'min_recent_up_days': re.compile(r'min_recent_up_days:\s*(\d+)', re.IGNORECASE),
            'recent_up_window': re.compile(r'recent_up_window:\s*(\d+)', re.IGNORECASE),
            'ma_trend_factor': re.compile(r'ma_trend_factor:\s*([\d.]+)', re.IGNORECASE),
            'sell_threshold_ratio': re.compile(r'sell_threshold_ratio:\s*([\d.]+)', re.IGNORECASE),
            'max_single_position_ratio': re.compile(r'max_single_position_ratio:\s*([\d.]+)', re.IGNORECASE),
        }
    
    def extract_yaml_blocks(self, content: str) -> List[Dict[str, Any]]:
        """마크다운에서 YAML 코드 블록들을 추출"""
        yaml_blocks = []
        matches = self.yaml_pattern.findall(content)
        
        for match in matches:
            try:
                yaml_data = yaml.safe_load(match)
                if yaml_data:
                    yaml_blocks.append(yaml_data)
            except yaml.YAMLError as e:
                self.logger.warning(f"Invalid YAML block found: {e}")
                continue
        
        return yaml_blocks
    
    def extract_profile_name(self, content: str) -> Optional[str]:
        """제목에서 프로파일명 추출 (예: "기본 전략 (default)" → "default")"""
        match = self.param_patterns['profile_name'].search(content)
        if match:
            return match.group(2).strip()
        return None
    
    def extract_basic_info(self, content: str) -> Dict[str, str]:
        """기본 전략 정보 추출"""
        info = {}
        
        # 프로파일명 추출
        profile_match = self.param_patterns['profile_name'].search(content)
        if profile_match:
            info['strategy_title'] = profile_match.group(1).strip()
            info['profile_key'] = profile_match.group(2).strip()
        
        # 설명 추출 - 더 정교한 패턴
        desc_patterns = [
            re.compile(r'###?\s*전략\s*설명\s*\n([^\n#]+)', re.MULTILINE),
            re.compile(r'설명[:\s]*([^\n]+)', re.IGNORECASE),
            re.compile(r'description[:\s]*["\']?([^"\'\\n]+)["\']?', re.IGNORECASE)
        ]
        
        for pattern in desc_patterns:
            match = pattern.search(content)
            if match:
                info['description'] = match.group(1).strip()
                break
        
        return info
    
    def extract_parameters_from_yaml(self, yaml_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """YAML 블록에서 파라미터 추출"""
        merged_params = {}
        
        for block in yaml_blocks:
            if isinstance(block, dict):
                # backtest 섹션
                if 'backtest' in block:
                    if 'backtest' not in merged_params:
                        merged_params['backtest'] = {}
                    merged_params['backtest'].update(block['backtest'])
                
                # trading 섹션
                if 'trading' in block:
                    if 'trading' not in merged_params:
                        merged_params['trading'] = {}
                    merged_params['trading'].update(block['trading'])
                
                # 직접 포함된 파라미터들
                for key in ['initial_capital', 'rebalance_frequency', 'signal_ma_short', 
                           'signal_ma_long', 'signal_rsi_period', 'signal_rsi_overbought',
                           'min_recent_up_days', 'recent_up_window', 'ma_trend_factor', 'sell_threshold_ratio',
                           'market_score_threshold', 'max_positions', 'max_single_position_ratio']:
                    if key in block:
                        # 적절한 섹션에 배치
                        if key in ['market_score_threshold', 'max_positions', 'max_single_position_ratio']:
                            if 'trading' not in merged_params:
                                merged_params['trading'] = {}
                            merged_params['trading'][key] = block[key]
                        else:
                            if 'backtest' not in merged_params:
                                merged_params['backtest'] = {}
                            merged_params['backtest'][key] = block[key]
        
        return merged_params
    
    def extract_parameters_from_text(self, content: str) -> Dict[str, Any]:
        """텍스트에서 직접 파라미터 추출 (YAML 블록 외부)"""
        params = {'backtest': {}, 'trading': {}}
        
        # 숫자 값 파라미터들
        for param_name, pattern in self.numeric_patterns.items():
            match = pattern.search(content)
            if match:
                try:
                    value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                    
                    # 적절한 섹션에 배치
                    if param_name in ['market_score_threshold', 'max_positions', 'max_single_position_ratio']:
                        params['trading'][param_name] = value
                    else:
                        params['backtest'][param_name] = value
                except ValueError:
                    continue
        
        # 텍스트 패턴에서 추가 정보 추출
        for param_name, pattern in self.param_patterns.items():
            if param_name in ['profile_name', 'name', 'description']:
                continue
                
            match = pattern.search(content)
            if match:
                try:
                    value = int(match.group(1))
                    if param_name == 'rebalance_frequency':
                        params['backtest'][param_name] = value
                    elif param_name in ['max_positions', 'market_score_threshold']:
                        params['trading'][param_name] = value
                except ValueError:
                    continue
        
        return params
    
    def parse_strategy_document(self, md_path: Path) -> Optional[Dict[str, Any]]:
        """전략 문서를 파싱하여 구조화된 데이터 반환"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (FileNotFoundError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to read {md_path}: {e}")
            return None
        
        # 기본 정보 추출
        basic_info = self.extract_basic_info(content)
        if not basic_info.get('profile_key'):
            self.logger.warning(f"Could not extract profile key from {md_path}")
            return None
        
        # YAML 블록에서 파라미터 추출
        yaml_blocks = self.extract_yaml_blocks(content)
        yaml_params = self.extract_parameters_from_yaml(yaml_blocks)
        
        # 텍스트에서 파라미터 추출
        text_params = self.extract_parameters_from_text(content)
        
        # 파라미터 병합 (YAML 블록 우선)
        merged_backtest = {}
        merged_backtest.update(text_params.get('backtest', {}))
        merged_backtest.update(yaml_params.get('backtest', {}))
        
        merged_trading = {}
        merged_trading.update(text_params.get('trading', {}))
        merged_trading.update(yaml_params.get('trading', {}))
        
        # 결과 구조화
        result = {
            'profile_key': basic_info['profile_key'],
            'name': basic_info.get('strategy_title', f"{basic_info['profile_key'].title()} Strategy"),
            'description': basic_info.get('description', 'No description available'),
            'backtest': merged_backtest,
            'trading': merged_trading,
            'metadata': {
                'source_file': str(md_path),
                'parsed_at': datetime.now().isoformat(),
                'yaml_blocks_found': len(yaml_blocks),
                'extraction_method': 'md_to_yaml_parser'
            }
        }
        
        # 빈 섹션 제거
        if not result['backtest']:
            del result['backtest']
        if not result['trading']:
            del result['trading']
        
        return result
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """추출된 데이터의 유효성 검증"""
        issues = []
        
        # 필수 필드 확인
        required_fields = ['profile_key', 'name']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append(f"Missing required field: {field}")
        
        # 백테스트 파라미터 검증
        if 'backtest' in data:
            backtest = data['backtest']
            
            # 숫자 범위 검증
            numeric_validations = {
                'initial_capital': (1000000, 100000000),  # 100만 ~ 1억
                'rebalance_frequency': (1, 30),           # 1~30일
                'signal_ma_short': (1, 50),               # 1~50일
                'signal_ma_long': (5, 100),               # 5~100일
                'signal_rsi_period': (5, 50),             # 5~50일
                'signal_rsi_overbought': (50, 90),        # 50~90
                'min_recent_up_days': (1, 10),            # 1~10일
                'recent_up_window': (1, 10),              # 1~10일
                'ma_trend_factor': (0.8, 1.2),           # 0.8~1.2
                'sell_threshold_ratio': (0.8, 1.0),      # 0.8~1.0
            }
            
            for param, (min_val, max_val) in numeric_validations.items():
                if param in backtest:
                    value = backtest[param]
                    if not isinstance(value, (int, float)):
                        issues.append(f"Invalid type for {param}: expected number, got {type(value)}")
                    elif not (min_val <= value <= max_val):
                        issues.append(f"Value out of range for {param}: {value} (expected {min_val}-{max_val})")
        
        # 거래 파라미터 검증
        if 'trading' in data:
            trading = data['trading']
            
            trading_validations = {
                'market_score_threshold': (50, 90),       # 50~90점
                'max_positions': (1, 20),                 # 1~20개
                'max_single_position_ratio': (0.05, 0.5), # 5%~50%
            }
            
            for param, (min_val, max_val) in trading_validations.items():
                if param in trading:
                    value = trading[param]
                    if not isinstance(value, (int, float)):
                        issues.append(f"Invalid type for {param}: expected number, got {type(value)}")
                    elif not (min_val <= value <= max_val):
                        issues.append(f"Value out of range for {param}: {value} (expected {min_val}-{max_val})")
        
        return len(issues) == 0, issues
    
    def parse_all_strategy_docs(self, docs_dir: Path = None) -> Dict[str, Dict[str, Any]]:
        """모든 전략 문서를 파싱"""
        if docs_dir is None:
            docs_dir = Path("docs/strategies")
        
        if not docs_dir.exists():
            self.logger.error(f"Strategy docs directory not found: {docs_dir}")
            return {}
        
        results = {}
        strategy_files = docs_dir.glob("*_STRATEGY.md")
        
        for md_file in strategy_files:
            self.logger.info(f"Parsing {md_file.name}...")
            
            parsed_data = self.parse_strategy_document(md_file)
            if parsed_data:
                # 데이터 검증
                is_valid, issues = self.validate_extracted_data(parsed_data)
                
                if is_valid:
                    results[parsed_data['profile_key']] = parsed_data
                    self.logger.info(f"✅ Successfully parsed {parsed_data['profile_key']}")
                else:
                    self.logger.warning(f"⚠️ Validation issues for {parsed_data['profile_key']}: {', '.join(issues)}")
                    # 검증 실패해도 결과에 포함 (이슈와 함께)
                    parsed_data['validation_issues'] = issues
                    results[parsed_data['profile_key']] = parsed_data
            else:
                self.logger.error(f"❌ Failed to parse {md_file.name}")
        
        return results


def main():
    """테스트용 메인 함수"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    parser = MarkdownYAMLParser()
    
    # 단일 파일 테스트
    test_file = Path("docs/strategies/DEFAULT_STRATEGY.md")
    if test_file.exists():
        print(f"🧪 Testing single file: {test_file}")
        result = parser.parse_strategy_document(test_file)
        
        if result:
            print("✅ Parsing successful!")
            print(f"Profile: {result['profile_key']}")
            print(f"Name: {result['name']}")
            print(f"Description: {result['description']}")
            
            if 'backtest' in result:
                print(f"Backtest params: {len(result['backtest'])} found")
            if 'trading' in result:
                print(f"Trading params: {len(result['trading'])} found")
            
            print("\n📊 Extracted YAML structure:")
            print(yaml.dump(result, default_flow_style=False, allow_unicode=True))
        else:
            print("❌ Parsing failed")
    
    # 전체 디렉토리 테스트
    print(f"\n🔄 Testing all strategy docs...")
    all_results = parser.parse_all_strategy_docs()
    
    print(f"✅ Parsed {len(all_results)} strategy documents:")
    for profile_key, data in all_results.items():
        status = "⚠️" if 'validation_issues' in data else "✅"
        print(f"  {status} {profile_key}: {data['name']}")


if __name__ == "__main__":
    main()