"""
양방향 동기화 시스템 (YAML ↔ MD)
전략 문서와 설정 파일 간의 일관성을 유지합니다.
"""

import yaml
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import hashlib
import json

from .md_to_yaml_parser import MarkdownYAMLParser
from .strategy_docs_sync import StrategyDocsSync


class SyncDirection(Enum):
    YAML_TO_MD = "yaml_to_md"
    MD_TO_YAML = "md_to_yaml" 
    BIDIRECTIONAL = "bidirectional"


class ConflictResolution(Enum):
    YAML_WINS = "yaml_wins"        # YAML 파일이 우선
    MD_WINS = "md_wins"           # MD 파일이 우선
    INTERACTIVE = "interactive"    # 사용자가 선택
    BACKUP_BOTH = "backup_both"    # 둘 다 백업 후 병합


class BidirectionalSyncManager:
    """양방향 동기화 관리자"""
    
    def __init__(self, 
                 yaml_path: str = "config/backtest_profiles.yaml",
                 docs_dir: str = "docs/strategies",
                 backup_dir: str = "backups"):
        
        self.yaml_path = Path(yaml_path)
        self.docs_dir = Path(docs_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.md_parser = MarkdownYAMLParser()
        self.docs_sync = StrategyDocsSync(str(yaml_path))
        
        # 상태 추적 파일
        self.state_file = Path("config/.sync_state.json")
        self.load_sync_state()
    
    def load_sync_state(self):
        """동기화 상태 로드"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.sync_state = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.sync_state = {}
        else:
            self.sync_state = {}
    
    def save_sync_state(self):
        """동기화 상태 저장"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.sync_state, f, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return ""
    
    def detect_changes(self) -> Dict[str, Dict[str, Any]]:
        """변경사항 감지"""
        changes = {
            'yaml_changed': False,
            'md_changed': [],
            'yaml_hash': '',
            'md_hashes': {},
            'conflicts': []
        }
        
        # YAML 파일 변경 확인
        current_yaml_hash = self.get_file_hash(self.yaml_path)
        last_yaml_hash = self.sync_state.get('yaml_hash', '')
        
        if current_yaml_hash != last_yaml_hash:
            changes['yaml_changed'] = True
            changes['yaml_hash'] = current_yaml_hash
        
        # MD 파일들 변경 확인
        if self.docs_dir.exists():
            for md_file in self.docs_dir.glob("*_STRATEGY.md"):
                current_md_hash = self.get_file_hash(md_file)
                last_md_hash = self.sync_state.get('md_hashes', {}).get(str(md_file), '')
                
                if current_md_hash != last_md_hash:
                    changes['md_changed'].append(str(md_file))
                
                changes['md_hashes'][str(md_file)] = current_md_hash
        
        # 충돌 감지
        if changes['yaml_changed'] and changes['md_changed']:
            changes['conflicts'] = self.detect_conflicts()
        
        return changes
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """충돌 상황 감지"""
        conflicts = []
        
        try:
            # YAML에서 현재 데이터 로드
            yaml_data = self.docs_sync.load_yaml_profiles()
            
            # MD에서 데이터 추출
            md_data = self.md_parser.parse_all_strategy_docs(self.docs_dir)
            
            # 공통 전략들에 대해 충돌 검사
            common_strategies = set(yaml_data.keys()) & set(md_data.keys())
            
            for strategy in common_strategies:
                yaml_profile = yaml_data[strategy]
                md_profile = md_data[strategy]
                
                strategy_conflicts = self.compare_strategy_data(strategy, yaml_profile, md_profile)
                if strategy_conflicts:
                    conflicts.extend(strategy_conflicts)
        
        except Exception as e:
            conflicts.append({
                'type': 'parsing_error',
                'message': f"Error detecting conflicts: {str(e)}"
            })
        
        return conflicts
    
    def compare_strategy_data(self, strategy_name: str, yaml_data: Dict, md_data: Dict) -> List[Dict]:
        """전략 데이터 비교 및 충돌 발견"""
        conflicts = []
        
        # 기본 정보 비교
        yaml_name = yaml_data.get('name', '')
        md_name = md_data.get('name', '')
        if yaml_name != md_name:
            conflicts.append({
                'type': 'field_conflict',
                'strategy': strategy_name,
                'field': 'name',
                'yaml_value': yaml_name,
                'md_value': md_name
            })
        
        yaml_desc = yaml_data.get('description', '')
        md_desc = md_data.get('description', '')
        if yaml_desc != md_desc:
            conflicts.append({
                'type': 'field_conflict', 
                'strategy': strategy_name,
                'field': 'description',
                'yaml_value': yaml_desc,
                'md_value': md_desc
            })
        
        # 백테스트 파라미터 비교
        yaml_backtest = yaml_data.get('backtest', {})
        md_backtest = md_data.get('backtest', {})
        
        for param in set(yaml_backtest.keys()) | set(md_backtest.keys()):
            yaml_val = yaml_backtest.get(param)
            md_val = md_backtest.get(param)
            
            if yaml_val != md_val:
                conflicts.append({
                    'type': 'parameter_conflict',
                    'strategy': strategy_name,
                    'section': 'backtest',
                    'parameter': param,
                    'yaml_value': yaml_val,
                    'md_value': md_val
                })
        
        # 거래 파라미터 비교
        yaml_trading = yaml_data.get('trading', {})
        md_trading = md_data.get('trading', {})
        
        for param in set(yaml_trading.keys()) | set(md_trading.keys()):
            yaml_val = yaml_trading.get(param)
            md_val = md_trading.get(param)
            
            if yaml_val != md_val:
                conflicts.append({
                    'type': 'parameter_conflict',
                    'strategy': strategy_name,
                    'section': 'trading',
                    'parameter': param,
                    'yaml_value': yaml_val,
                    'md_value': md_val
                })
        
        return conflicts
    
    def create_backup(self, label: str = None) -> str:
        """백업 생성"""
        if label is None:
            label = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_subdir = self.backup_dir / f"sync_backup_{label}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # YAML 파일 백업
        if self.yaml_path.exists():
            shutil.copy2(self.yaml_path, backup_subdir / "backtest_profiles.yaml")
        
        # MD 파일들 백업
        if self.docs_dir.exists():
            md_backup_dir = backup_subdir / "strategies"
            md_backup_dir.mkdir(exist_ok=True)
            
            for md_file in self.docs_dir.glob("*_STRATEGY.md"):
                shutil.copy2(md_file, md_backup_dir / md_file.name)
        
        # 백업 정보 저장
        backup_info = {
            'created_at': datetime.now().isoformat(),
            'yaml_file': str(self.yaml_path),
            'docs_dir': str(self.docs_dir),
            'backup_path': str(backup_subdir)
        }
        
        with open(backup_subdir / "backup_info.json", 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        return str(backup_subdir)
    
    def sync_yaml_to_md(self) -> Dict[str, Any]:
        """YAML → MD 동기화"""
        print("🔄 Syncing YAML to MD files...")
        
        try:
            results = self.docs_sync.sync_all_strategies()
            
            success_count = len([r for r in results.values() if not r.startswith("Error")])
            error_count = len([r for r in results.values() if r.startswith("Error")])
            
            return {
                'success': error_count == 0,
                'results': results,
                'success_count': success_count,
                'error_count': error_count,
                'direction': 'yaml_to_md'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'direction': 'yaml_to_md'
            }
    
    def sync_md_to_yaml(self) -> Dict[str, Any]:
        """MD → YAML 동기화"""
        print("🔄 Syncing MD files to YAML...")
        
        try:
            # MD 파일들에서 데이터 추출
            md_data = self.md_parser.parse_all_strategy_docs(self.docs_dir)
            
            if not md_data:
                return {
                    'success': False,
                    'error': 'No valid strategy documents found',
                    'direction': 'md_to_yaml'
                }
            
            # 새로운 YAML 구조 생성
            yaml_profiles = {}
            
            for profile_key, profile_data in md_data.items():
                yaml_profile = {
                    'name': profile_data['name'],
                    'description': profile_data['description']
                }
                
                if 'backtest' in profile_data and profile_data['backtest']:
                    yaml_profile['backtest'] = profile_data['backtest']
                
                if 'trading' in profile_data and profile_data['trading']:
                    yaml_profile['trading'] = profile_data['trading']
                
                yaml_profiles[profile_key] = yaml_profile
            
            # YAML 파일 업데이트
            self.update_yaml_file(yaml_profiles)
            
            return {
                'success': True,
                'profiles_updated': list(yaml_profiles.keys()),
                'total_profiles': len(yaml_profiles),
                'direction': 'md_to_yaml'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'direction': 'md_to_yaml'
            }
    
    def update_yaml_file(self, profiles_data: Dict[str, Any]):
        """YAML 파일 업데이트"""
        # 헤더 코멘트 보존
        header_comment = "# 백테스팅 설정 프로파일\\n# 다양한 시나리오별 설정값들\\n"
        
        # YAML 문자열 생성
        yaml_content = yaml.dump(profiles_data, 
                                default_flow_style=False, 
                                allow_unicode=True,
                                sort_keys=False,
                                indent=2)
        
        # 파일 쓰기
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("# 백테스팅 설정 프로파일\\n")
            f.write("# 다양한 시나리오별 설정값들\\n")
            f.write("\\n")
            f.write(yaml_content)
    
    def resolve_conflicts_interactive(self, conflicts: List[Dict[str, Any]]) -> Dict[str, str]:
        """대화형 충돌 해결"""
        resolutions = {}
        
        print(f"\\n⚠️ Found {len(conflicts)} conflicts that need resolution:")
        print("="*60)
        
        for i, conflict in enumerate(conflicts, 1):
            print(f"\\nConflict {i}/{len(conflicts)}:")
            print(f"Strategy: {conflict.get('strategy', 'N/A')}")
            
            if conflict['type'] == 'field_conflict':
                print(f"Field: {conflict['field']}")
                print(f"YAML value: '{conflict['yaml_value']}'")
                print(f"MD value: '{conflict['md_value']}'")
            
            elif conflict['type'] == 'parameter_conflict':
                print(f"Section: {conflict['section']}")
                print(f"Parameter: {conflict['parameter']}")
                print(f"YAML value: {conflict['yaml_value']}")
                print(f"MD value: {conflict['md_value']}")
            
            print("\\nChoose resolution:")
            print("1. Keep YAML value")
            print("2. Keep MD value") 
            print("3. Skip this conflict")
            
            while True:
                try:
                    choice = input("Your choice (1-3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                    print("Invalid choice. Please enter 1, 2, or 3.")
                except KeyboardInterrupt:
                    print("\\n\\nOperation cancelled by user.")
                    return {}
            
            conflict_key = f"{conflict.get('strategy', 'unknown')}_{conflict.get('field', conflict.get('parameter', 'unknown'))}"
            
            if choice == '1':
                resolutions[conflict_key] = 'yaml'
            elif choice == '2':
                resolutions[conflict_key] = 'md'
            else:
                resolutions[conflict_key] = 'skip'
        
        return resolutions
    
    def perform_sync(self, 
                    direction: SyncDirection,
                    conflict_resolution: ConflictResolution = ConflictResolution.INTERACTIVE,
                    create_backup: bool = True) -> Dict[str, Any]:
        """동기화 수행"""
        
        # 백업 생성
        backup_path = None
        if create_backup:
            backup_path = self.create_backup()
            print(f"📁 Backup created: {backup_path}")
        
        # 변경사항 감지
        changes = self.detect_changes()
        
        if not changes['yaml_changed'] and not changes['md_changed']:
            return {
                'success': True,
                'message': 'No changes detected. Files are already in sync.',
                'backup_path': backup_path
            }
        
        # 충돌 처리
        if changes['conflicts']:
            print(f"⚠️ Detected {len(changes['conflicts'])} conflicts")
            
            if conflict_resolution == ConflictResolution.INTERACTIVE:
                resolutions = self.resolve_conflicts_interactive(changes['conflicts'])
                if not resolutions:  # 사용자가 취소한 경우
                    return {'success': False, 'message': 'Operation cancelled by user'}
            elif conflict_resolution == ConflictResolution.YAML_WINS:
                direction = SyncDirection.YAML_TO_MD
            elif conflict_resolution == ConflictResolution.MD_WINS:
                direction = SyncDirection.MD_TO_YAML
        
        # 동기화 실행
        result = {'backup_path': backup_path}
        
        if direction == SyncDirection.YAML_TO_MD:
            sync_result = self.sync_yaml_to_md()
            result.update(sync_result)
            
        elif direction == SyncDirection.MD_TO_YAML:
            sync_result = self.sync_md_to_yaml()
            result.update(sync_result)
            
        elif direction == SyncDirection.BIDIRECTIONAL:
            # 양방향의 경우 변경사항에 따라 결정
            if changes['yaml_changed'] and not changes['md_changed']:
                sync_result = self.sync_yaml_to_md()
            elif changes['md_changed'] and not changes['yaml_changed']:
                sync_result = self.sync_md_to_yaml()
            else:
                # 둘 다 변경된 경우 충돌 해결 필요
                sync_result = {'success': False, 'message': 'Bidirectional conflicts require explicit resolution'}
            
            result.update(sync_result)
        
        # 동기화 상태 업데이트
        if result.get('success', False):
            self.sync_state.update({
                'last_sync': datetime.now().isoformat(),
                'yaml_hash': self.get_file_hash(self.yaml_path),
                'md_hashes': {str(f): self.get_file_hash(f) for f in self.docs_dir.glob("*_STRATEGY.md") if f.exists()}
            })
            self.save_sync_state()
        
        return result


def main():
    """CLI 테스트"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bidirectional Strategy Documentation Sync")
    parser.add_argument('--direction', choices=['yaml-to-md', 'md-to-yaml', 'bidirectional'], 
                       default='bidirectional', help='Sync direction')
    parser.add_argument('--conflict-resolution', choices=['yaml-wins', 'md-wins', 'interactive'],
                       default='interactive', help='Conflict resolution strategy')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--test-parse', action='store_true', help='Test MD parsing only')
    
    args = parser.parse_args()
    
    # 동기화 관리자 초기화
    sync_manager = BidirectionalSyncManager()
    
    if args.test_parse:
        # MD 파싱 테스트
        print("🧪 Testing MD file parsing...")
        md_data = sync_manager.md_parser.parse_all_strategy_docs()
        
        print(f"✅ Parsed {len(md_data)} strategy documents:")
        for profile_key, data in md_data.items():
            status = "⚠️" if 'validation_issues' in data else "✅"
            backtest_count = len(data.get('backtest', {}))
            trading_count = len(data.get('trading', {}))
            print(f"  {status} {profile_key}: {backtest_count} backtest + {trading_count} trading params")
        return
    
    # 동기화 실행
    direction_map = {
        'yaml-to-md': SyncDirection.YAML_TO_MD,
        'md-to-yaml': SyncDirection.MD_TO_YAML,
        'bidirectional': SyncDirection.BIDIRECTIONAL
    }
    
    resolution_map = {
        'yaml-wins': ConflictResolution.YAML_WINS,
        'md-wins': ConflictResolution.MD_WINS,
        'interactive': ConflictResolution.INTERACTIVE
    }
    
    print("🚀 Starting bidirectional sync...")
    result = sync_manager.perform_sync(
        direction=direction_map[args.direction],
        conflict_resolution=resolution_map[args.conflict_resolution],
        create_backup=not args.no_backup
    )
    
    if result['success']:
        print("✅ Sync completed successfully!")
        if 'profiles_updated' in result:
            print(f"📊 Updated profiles: {', '.join(result['profiles_updated'])}")
        if 'success_count' in result:
            print(f"📝 Generated {result['success_count']} documents")
    else:
        print("❌ Sync failed!")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    if result.get('backup_path'):
        print(f"💾 Backup available at: {result['backup_path']}")


if __name__ == "__main__":
    main()