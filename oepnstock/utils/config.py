"""
Configuration utilities and helpers
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from ..config.settings import config as main_config


class ConfigManager:
    """
    Configuration management utility
    """
    
    def __init__(self, config_dir: str = "oepnstock/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config_to_file(self, filename: str, config_data: Dict[str, Any]) -> None:
        """Save configuration to YAML file"""
        config_path = self.config_dir / filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def load_config_from_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def save_trading_session(self, session_data: Dict[str, Any]) -> str:
        """Save current trading session configuration"""
        timestamp = session_data.get('timestamp', 'unknown')
        filename = f"session_{timestamp}.json"
        session_path = self.config_dir / "sessions" / filename
        session_path.parent.mkdir(exist_ok=True)
        
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        return str(session_path)
    
    def load_trading_session(self, session_file: str) -> Dict[str, Any]:
        """Load trading session configuration"""
        session_path = Path(session_file)
        
        if not session_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def export_current_config(self) -> Dict[str, Any]:
        """Export current configuration as dictionary"""
        return main_config.to_dict()
    
    def validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration data
        Returns dict with 'valid' bool and 'errors' list
        """
        errors = []
        
        # Check required fields
        required_fields = [
            'trading.market_score_threshold',
            'trading.max_positions', 
            'trading.max_single_position_ratio',
            'technical.ma_short',
            'technical.ma_medium', 
            'technical.ma_long'
        ]
        
        for field in required_fields:
            if not self._get_nested_value(config_data, field):
                errors.append(f"Missing required field: {field}")
        
        # Validate value ranges
        trading_config = config_data.get('trading', {})
        
        if trading_config.get('market_score_threshold', 0) < 50 or trading_config.get('market_score_threshold', 100) > 100:
            errors.append("market_score_threshold must be between 50-100")
        
        if trading_config.get('max_single_position_ratio', 0) > 0.5:
            errors.append("max_single_position_ratio should not exceed 50%")
        
        if trading_config.get('initial_risk_per_trade', 0) > 0.1:
            errors.append("initial_risk_per_trade should not exceed 10%")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None


class EnvironmentConfig:
    """
    Environment-specific configuration management
    """
    
    @staticmethod
    def is_development() -> bool:
        """Check if running in development environment"""
        return os.getenv('DEBUG', 'false').lower() == 'true'
    
    @staticmethod
    def is_testing() -> bool:
        """Check if running in test environment"""
        return os.getenv('TESTING', 'false').lower() == 'true'
    
    @staticmethod
    def is_paper_trading() -> bool:
        """Check if paper trading is enabled"""
        return os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    @staticmethod
    def get_api_keys() -> Dict[str, Optional[str]]:
        """Get all API keys from environment"""
        return {
            'kiwoom': os.getenv('KIWOOM_API_KEY'),
            'korea_investment': os.getenv('KOREA_INVESTMENT_API_KEY'),
            'data_provider': os.getenv('DATA_PROVIDER_API_KEY')
        }
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """Validate environment configuration"""
        errors = []
        warnings = []
        
        # Check database connection
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            errors.append("DATABASE_URL not set")
        
        # Check Redis connection  
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            warnings.append("REDIS_URL not set - using default")
        
        # Check API keys in production
        if not EnvironmentConfig.is_development() and not EnvironmentConfig.is_testing():
            api_keys = EnvironmentConfig.get_api_keys()
            missing_keys = [k for k, v in api_keys.items() if not v]
            
            if missing_keys:
                errors.append(f"Missing API keys in production: {missing_keys}")
        
        # Check paper trading in production
        if not EnvironmentConfig.is_development() and EnvironmentConfig.is_paper_trading():
            warnings.append("Paper trading enabled in non-development environment")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


# Global instances
config_manager = ConfigManager()
env_config = EnvironmentConfig()