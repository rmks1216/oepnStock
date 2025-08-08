"""
Portfolio Concentration Manager - Critical Phase 1 Module
포트폴리오 집중도 관리 시스템: 과도한 집중 투자 방지 및 리스크 분산
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict

from ...config import config
from ...utils import get_logger
from ...utils.korean_market import KoreanMarketUtils

logger = get_logger(__name__)


@dataclass
class Position:
    """Individual position information"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    weight: float
    sector: str
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_date: datetime
    days_held: int


@dataclass
class Portfolio:
    """Portfolio summary"""
    total_value: float
    cash: float
    positions: List[Position]
    position_count: int
    largest_position_weight: float
    sector_weights: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame] = None


@dataclass
class ConcentrationAnalysis:
    """Portfolio concentration analysis result"""
    portfolio: Portfolio
    concentration_score: float  # 0-100, 높을수록 집중도 높음
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    violations: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    diversification_ratio: float
    max_correlation_exposure: float
    position_limits_ok: bool
    sector_limits_ok: bool
    correlation_limits_ok: bool


@dataclass
class AddPositionResult:
    """새 포지션 추가 결과"""
    can_add: bool
    max_allowed_size: float
    recommended_size: float
    blocking_reasons: List[str]
    warnings: List[str]
    position_adjustment_factor: float


class PortfolioConcentrationManager:
    """
    포트폴리오 집중도 관리 시스템
    
    주요 기능:
    - 포지션 수 제한 (최대 5개)
    - 단일 종목 비중 제한 (20%)
    - 섹터 집중도 제한 (40%)
    - 상관관계 리스크 관리 (상관계수 0.7+ 종목들 합계 60% 제한)
    - 현금 비중 유지 (최소 10%)
    - 포트폴리오 재조정 제안
    """
    
    def __init__(self):
        self.config = config.trading
        
        # Concentration limits
        self.limits = {
            'max_positions': self.config.max_positions,  # 5개
            'max_single_position': self.config.max_single_position_ratio,  # 20%
            'max_sector_exposure': self.config.max_sector_exposure,  # 40%
            'max_correlation_exposure': self.config.max_correlation_exposure,  # 60%
            'min_cash_ratio': self.config.min_cash_ratio  # 10%
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'position_count': 0.2,
            'single_position': 0.3,
            'sector_concentration': 0.25,
            'correlation_risk': 0.25
        }
        
        # Correlation threshold for high-correlation grouping
        self.high_correlation_threshold = 0.7
        
        # Portfolio rebalancing thresholds
        self.rebalance_thresholds = {
            'position_weight': 0.05,  # 5% deviation triggers rebalance suggestion
            'sector_weight': 0.1      # 10% deviation
        }
    
    def analyze_portfolio_concentration(self, portfolio_data: Dict[str, Any]) -> ConcentrationAnalysis:
        """
        포트폴리오 집중도 종합 분석
        
        Args:
            portfolio_data: {
                'positions': [{'symbol', 'quantity', 'avg_price', 'current_price', 'sector', ...}],
                'cash': float,
                'total_value': float
            }
            
        Returns:
            ConcentrationAnalysis: 집중도 분석 결과
        """
        logger.info("Starting portfolio concentration analysis")
        
        try:
            # 1. 포트폴리오 객체 생성
            portfolio = self._create_portfolio_object(portfolio_data)
            
            # 2. 상관관계 매트릭스 계산
            correlation_matrix = self._calculate_correlation_matrix(portfolio.positions)
            portfolio.correlation_matrix = correlation_matrix
            
            # 3. 집중도 점수 계산
            concentration_score = self._calculate_concentration_score(portfolio)
            
            # 4. 리스크 레벨 결정
            risk_level = self._determine_risk_level(concentration_score)
            
            # 5. 제한 위반 사항 검사
            violations = self._check_limit_violations(portfolio)
            
            # 6. 다각화 비율 계산
            diversification_ratio = self._calculate_diversification_ratio(portfolio)
            
            # 7. 최대 상관관계 노출 계산
            max_correlation_exposure = self._calculate_max_correlation_exposure(portfolio)
            
            # 8. 개선 제안 생성
            recommendations = self._generate_recommendations(portfolio, violations)
            
            analysis = ConcentrationAnalysis(
                portfolio=portfolio,
                concentration_score=concentration_score,
                risk_level=risk_level,
                violations=violations,
                recommendations=recommendations,
                diversification_ratio=diversification_ratio,
                max_correlation_exposure=max_correlation_exposure,
                position_limits_ok=len(violations) == 0 or all(v['type'] != 'position_count' for v in violations),
                sector_limits_ok=all(v['type'] != 'sector_exposure' for v in violations),
                correlation_limits_ok=all(v['type'] != 'correlation_risk' for v in violations)
            )
            
            logger.info(f"Portfolio concentration analysis complete - Score: {concentration_score:.1f}, "
                       f"Risk: {risk_level}, Violations: {len(violations)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in portfolio concentration analysis: {e}")
            raise
    
    def can_add_position(self, symbol: str, planned_investment: float,
                        current_portfolio: Dict[str, Any]) -> AddPositionResult:
        """
        새 포지션 추가 가능 여부 및 적정 규모 판단
        
        Args:
            symbol: 추가할 종목 코드
            planned_investment: 계획된 투자 금액
            current_portfolio: 현재 포트폴리오 정보
            
        Returns:
            AddPositionResult: 추가 가능 여부 및 권장 사항
        """
        logger.info(f"Checking if can add position {symbol} with {planned_investment:,.0f}")
        
        try:
            portfolio = self._create_portfolio_object(current_portfolio)
            total_value = portfolio.total_value
            
            blocking_reasons = []
            warnings = []
            adjustment_factor = 1.0
            
            # 1. 포지션 수 제한 검사
            if portfolio.position_count >= self.limits['max_positions']:
                blocking_reasons.append(f"최대 포지션 수 {self.limits['max_positions']}개 초과")
            
            # 2. 현금 부족 검사
            available_cash = portfolio.cash
            if planned_investment > available_cash:
                blocking_reasons.append(f"현금 부족 (가용: {available_cash:,.0f}, 필요: {planned_investment:,.0f})")
            
            # 3. 단일 포지션 비중 검사
            planned_weight = planned_investment / total_value
            if planned_weight > self.limits['max_single_position']:
                max_allowed = total_value * self.limits['max_single_position']
                warnings.append(f"단일 포지션 한도 {self.limits['max_single_position']:.0%} 초과, "
                              f"최대 가능: {max_allowed:,.0f}")
                adjustment_factor = min(adjustment_factor, max_allowed / planned_investment)
            
            # 4. 섹터 집중도 검사
            symbol_sector = self._get_symbol_sector(symbol)
            current_sector_weight = portfolio.sector_weights.get(symbol_sector, 0.0)
            new_sector_weight = current_sector_weight + planned_weight
            
            if new_sector_weight > self.limits['max_sector_exposure']:
                max_sector_addition = (self.limits['max_sector_exposure'] - current_sector_weight) * total_value
                if max_sector_addition <= 0:
                    blocking_reasons.append(f"{symbol_sector} 섹터 한도 {self.limits['max_sector_exposure']:.0%} 초과")
                else:
                    warnings.append(f"섹터 집중도 주의 (현재: {current_sector_weight:.1%}, "
                                  f"추가후: {new_sector_weight:.1%})")
                    adjustment_factor = min(adjustment_factor, max_sector_addition / planned_investment)
            
            # 5. 상관관계 리스크 검사
            correlation_risk = self._assess_correlation_risk(symbol, portfolio, planned_weight)
            if correlation_risk['high_risk']:
                if correlation_risk['blocking']:
                    blocking_reasons.append("높은 상관관계 종목들의 비중 한도 초과")
                else:
                    warnings.append(f"상관관계 리스크 증가: {correlation_risk['message']}")
                    adjustment_factor = min(adjustment_factor, correlation_risk['max_ratio'])
            
            # 6. 현금 비중 유지 검사
            remaining_cash = available_cash - planned_investment
            remaining_cash_ratio = remaining_cash / total_value
            
            if remaining_cash_ratio < self.limits['min_cash_ratio']:
                max_investment_for_cash = available_cash - (total_value * self.limits['min_cash_ratio'])
                if max_investment_for_cash <= 0:
                    blocking_reasons.append(f"최소 현금 비중 {self.limits['min_cash_ratio']:.0%} 유지 불가")
                else:
                    warnings.append("최소 현금 비중 유지를 위한 투자 금액 조정")
                    adjustment_factor = min(adjustment_factor, max_investment_for_cash / planned_investment)
            
            # 7. 최종 결정
            can_add = len(blocking_reasons) == 0
            max_allowed_size = min(available_cash, planned_investment * adjustment_factor) if can_add else 0
            recommended_size = max_allowed_size * 0.9 if can_add else 0  # 10% 안전 마진
            
            result = AddPositionResult(
                can_add=can_add,
                max_allowed_size=max_allowed_size,
                recommended_size=recommended_size,
                blocking_reasons=blocking_reasons,
                warnings=warnings,
                position_adjustment_factor=adjustment_factor
            )
            
            logger.info(f"Position addition check complete for {symbol} - "
                       f"Can add: {can_add}, Recommended: {recommended_size:,.0f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking position addition for {symbol}: {e}")
            raise
    
    def get_rebalancing_suggestions(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """포트폴리오 재조정 제안"""
        suggestions = []
        
        try:
            portfolio = self._create_portfolio_object(portfolio_data)
            violations = self._check_limit_violations(portfolio)
            
            # 1. 과대 포지션 축소 제안
            for position in portfolio.positions:
                if position.weight > self.limits['max_single_position']:
                    target_weight = self.limits['max_single_position'] * 0.95  # 5% 안전 마진
                    reduction_amount = (position.weight - target_weight) * portfolio.total_value
                    
                    suggestions.append({
                        'type': 'reduce_position',
                        'symbol': position.symbol,
                        'current_weight': position.weight,
                        'target_weight': target_weight,
                        'reduction_amount': reduction_amount,
                        'priority': 'high',
                        'reason': f'단일 포지션 한도 {self.limits["max_single_position"]:.0%} 초과'
                    })
            
            # 2. 섹터 집중도 조정 제안
            for sector, weight in portfolio.sector_weights.items():
                if weight > self.limits['max_sector_exposure']:
                    sector_positions = [p for p in portfolio.positions if p.sector == sector]
                    sector_positions.sort(key=lambda x: x.weight, reverse=True)
                    
                    # 가장 큰 포지션부터 축소
                    cumulative_reduction = 0
                    target_sector_weight = self.limits['max_sector_exposure'] * 0.95
                    
                    for position in sector_positions:
                        if weight - cumulative_reduction <= target_sector_weight:
                            break
                        
                        reduction_needed = min(
                            position.weight * 0.3,  # 최대 30%씩 축소
                            weight - target_sector_weight - cumulative_reduction
                        )
                        
                        suggestions.append({
                            'type': 'reduce_sector_exposure',
                            'symbol': position.symbol,
                            'sector': sector,
                            'current_sector_weight': weight,
                            'target_sector_weight': target_sector_weight,
                            'reduction_ratio': reduction_needed / position.weight,
                            'priority': 'medium',
                            'reason': f'{sector} 섹터 집중도 {self.limits["max_sector_exposure"]:.0%} 초과'
                        })
                        
                        cumulative_reduction += reduction_needed
            
            # 3. 상관관계 리스크 분산 제안
            high_corr_groups = self._find_high_correlation_groups(portfolio)
            for group in high_corr_groups:
                if group['total_weight'] > self.limits['max_correlation_exposure']:
                    # 그룹 내에서 성과가 가장 낮은 종목부터 축소
                    group_positions = sorted(group['positions'], key=lambda x: x.unrealized_pnl_pct)
                    
                    for position in group_positions[:2]:  # 하위 2개 종목
                        suggestions.append({
                            'type': 'reduce_correlation_risk',
                            'symbol': position.symbol,
                            'correlation_group': [p.symbol for p in group['positions']],
                            'group_weight': group['total_weight'],
                            'reduction_ratio': 0.25,  # 25% 축소
                            'priority': 'medium',
                            'reason': f'높은 상관관계 그룹 비중 {self.limits["max_correlation_exposure"]:.0%} 초과'
                        })
            
            # 4. 우선순위별 정렬
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            suggestions.sort(key=lambda x: (priority_order[x['priority']], -x.get('reduction_amount', 0)))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating rebalancing suggestions: {e}")
            return []
    
    def calculate_optimal_position_size(self, symbol: str, available_capital: float,
                                      current_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """최적 포지션 크기 계산"""
        try:
            portfolio = self._create_portfolio_object(current_portfolio)
            
            # 1. 기본 제한 사항 적용
            constraints = {
                'max_single_position': available_capital * self.limits['max_single_position'],
                'available_cash': portfolio.cash,
                'min_cash_reserve': portfolio.total_value * self.limits['min_cash_ratio']
            }
            
            # 2. 섹터 제한 적용
            symbol_sector = self._get_symbol_sector(symbol)
            current_sector_weight = portfolio.sector_weights.get(symbol_sector, 0.0)
            max_sector_addition = max(0, (self.limits['max_sector_exposure'] - current_sector_weight) * portfolio.total_value)
            constraints['max_sector_addition'] = max_sector_addition
            
            # 3. 상관관계 제한 적용
            correlation_constraint = self._calculate_correlation_constraint(symbol, portfolio)
            constraints['max_correlation_addition'] = correlation_constraint
            
            # 4. 최종 최적 크기 계산
            max_investment = min(
                constraints['max_single_position'],
                constraints['available_cash'] - constraints['min_cash_reserve'],
                constraints['max_sector_addition'],
                constraints['max_correlation_addition'],
                available_capital
            )
            
            # 5. 권장 크기 (최대 크기의 80%)
            recommended_size = max_investment * 0.8 if max_investment > 0 else 0
            
            return {
                'max_size': max(max_investment, 0),
                'recommended_size': max(recommended_size, 0),
                'constraints': constraints,
                'limiting_factor': self._identify_limiting_factor(constraints),
                'risk_assessment': self._assess_position_risk(symbol, recommended_size, portfolio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size for {symbol}: {e}")
            return {
                'max_size': 0,
                'recommended_size': 0,
                'constraints': {},
                'limiting_factor': 'calculation_error',
                'risk_assessment': 'high'
            }
    
    # Private helper methods
    
    def _create_portfolio_object(self, portfolio_data: Dict[str, Any]) -> Portfolio:
        """포트폴리오 데이터를 Portfolio 객체로 변환"""
        positions = []
        total_value = portfolio_data.get('total_value', 0)
        cash = portfolio_data.get('cash', 0)
        
        for pos_data in portfolio_data.get('positions', []):
            market_value = pos_data['quantity'] * pos_data['current_price']
            weight = market_value / total_value if total_value > 0 else 0
            
            unrealized_pnl = market_value - (pos_data['quantity'] * pos_data['avg_price'])
            unrealized_pnl_pct = unrealized_pnl / (pos_data['quantity'] * pos_data['avg_price'])
            
            entry_date = pos_data.get('entry_date', datetime.now() - timedelta(days=30))
            days_held = (datetime.now() - entry_date).days
            
            position = Position(
                symbol=pos_data['symbol'],
                quantity=pos_data['quantity'],
                avg_price=pos_data['avg_price'],
                current_price=pos_data['current_price'],
                market_value=market_value,
                weight=weight,
                sector=pos_data.get('sector', 'unknown'),
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                entry_date=entry_date,
                days_held=days_held
            )
            positions.append(position)
        
        # 섹터별 가중치 계산
        sector_weights = defaultdict(float)
        for position in positions:
            sector_weights[position.sector] += position.weight
        
        largest_position_weight = max([p.weight for p in positions]) if positions else 0
        
        return Portfolio(
            total_value=total_value,
            cash=cash,
            positions=positions,
            position_count=len(positions),
            largest_position_weight=largest_position_weight,
            sector_weights=dict(sector_weights)
        )
    
    def _calculate_correlation_matrix(self, positions: List[Position]) -> Optional[pd.DataFrame]:
        """포지션 간 상관관계 매트릭스 계산"""
        if len(positions) < 2:
            return None
        
        try:
            symbols = [p.symbol for p in positions]
            correlation_data = {}
            
            # Mock correlation data (실제로는 가격 데이터로 계산)
            for i, symbol1 in enumerate(symbols):
                correlation_data[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_data[symbol1][symbol2] = 1.0
                    else:
                        # Mock correlation based on sector similarity
                        pos1 = positions[i]
                        pos2 = positions[j]
                        if pos1.sector == pos2.sector:
                            correlation_data[symbol1][symbol2] = 0.6 + np.random.normal(0, 0.1)
                        else:
                            correlation_data[symbol1][symbol2] = 0.1 + np.random.normal(0, 0.1)
                        
                        # Ensure symmetry
                        correlation_data[symbol1][symbol2] = max(-1, min(1, correlation_data[symbol1][symbol2]))
            
            return pd.DataFrame(correlation_data)
            
        except Exception as e:
            logger.warning(f"Error calculating correlation matrix: {e}")
            return None
    
    def _calculate_concentration_score(self, portfolio: Portfolio) -> float:
        """포트폴리오 집중도 점수 계산 (0-100)"""
        score = 0.0
        
        # 1. 포지션 수 점수 (적을수록 높은 점수)
        position_score = max(0, (self.limits['max_positions'] - portfolio.position_count + 1) / self.limits['max_positions']) * 100
        score += position_score * self.risk_weights['position_count']
        
        # 2. 단일 포지션 집중도 점수
        single_position_score = min(100, (portfolio.largest_position_weight / self.limits['max_single_position']) * 100)
        score += single_position_score * self.risk_weights['single_position']
        
        # 3. 섹터 집중도 점수
        max_sector_weight = max(portfolio.sector_weights.values()) if portfolio.sector_weights else 0
        sector_score = min(100, (max_sector_weight / self.limits['max_sector_exposure']) * 100)
        score += sector_score * self.risk_weights['sector_concentration']
        
        # 4. 상관관계 리스크 점수
        correlation_score = self._calculate_correlation_risk_score(portfolio)
        score += correlation_score * self.risk_weights['correlation_risk']
        
        return min(100, score)
    
    def _calculate_correlation_risk_score(self, portfolio: Portfolio) -> float:
        """상관관계 리스크 점수 계산"""
        if portfolio.correlation_matrix is None or len(portfolio.positions) < 2:
            return 0.0
        
        high_corr_groups = self._find_high_correlation_groups(portfolio)
        if not high_corr_groups:
            return 0.0
        
        max_group_weight = max(group['total_weight'] for group in high_corr_groups)
        return min(100, (max_group_weight / self.limits['max_correlation_exposure']) * 100)
    
    def _find_high_correlation_groups(self, portfolio: Portfolio) -> List[Dict]:
        """높은 상관관계 그룹 찾기"""
        if portfolio.correlation_matrix is None:
            return []
        
        groups = []
        processed = set()
        
        for symbol1 in portfolio.correlation_matrix.index:
            if symbol1 in processed:
                continue
            
            group_members = [symbol1]
            for symbol2 in portfolio.correlation_matrix.columns:
                if (symbol1 != symbol2 and symbol2 not in processed and
                    portfolio.correlation_matrix.loc[symbol1, symbol2] > self.high_correlation_threshold):
                    group_members.append(symbol2)
            
            if len(group_members) >= 2:
                group_positions = [p for p in portfolio.positions if p.symbol in group_members]
                total_weight = sum(p.weight for p in group_positions)
                
                groups.append({
                    'symbols': group_members,
                    'positions': group_positions,
                    'total_weight': total_weight,
                    'size': len(group_members)
                })
                
                processed.update(group_members)
        
        return sorted(groups, key=lambda x: x['total_weight'], reverse=True)
    
    def _determine_risk_level(self, concentration_score: float) -> str:
        """집중도 점수로부터 리스크 레벨 결정"""
        if concentration_score >= 80:
            return 'critical'
        elif concentration_score >= 60:
            return 'high'
        elif concentration_score >= 40:
            return 'medium'
        else:
            return 'low'
    
    def _check_limit_violations(self, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """제한 위반 사항 검사"""
        violations = []
        
        # 포지션 수 제한
        if portfolio.position_count > self.limits['max_positions']:
            violations.append({
                'type': 'position_count',
                'severity': 'high',
                'current': portfolio.position_count,
                'limit': self.limits['max_positions'],
                'message': f"포지션 수 {portfolio.position_count}개가 한도 {self.limits['max_positions']}개 초과"
            })
        
        # 단일 포지션 비중 제한
        for position in portfolio.positions:
            if position.weight > self.limits['max_single_position']:
                violations.append({
                    'type': 'single_position',
                    'severity': 'high',
                    'symbol': position.symbol,
                    'current': position.weight,
                    'limit': self.limits['max_single_position'],
                    'message': f"{position.symbol} 비중 {position.weight:.1%}가 한도 {self.limits['max_single_position']:.0%} 초과"
                })
        
        # 섹터 집중도 제한
        for sector, weight in portfolio.sector_weights.items():
            if weight > self.limits['max_sector_exposure']:
                violations.append({
                    'type': 'sector_exposure',
                    'severity': 'medium',
                    'sector': sector,
                    'current': weight,
                    'limit': self.limits['max_sector_exposure'],
                    'message': f"{sector} 섹터 비중 {weight:.1%}가 한도 {self.limits['max_sector_exposure']:.0%} 초과"
                })
        
        # 현금 비중 제한
        cash_ratio = portfolio.cash / portfolio.total_value if portfolio.total_value > 0 else 0
        if cash_ratio < self.limits['min_cash_ratio']:
            violations.append({
                'type': 'cash_ratio',
                'severity': 'medium',
                'current': cash_ratio,
                'limit': self.limits['min_cash_ratio'],
                'message': f"현금 비중 {cash_ratio:.1%}가 최소 한도 {self.limits['min_cash_ratio']:.0%} 미달"
            })
        
        return violations
    
    def _calculate_diversification_ratio(self, portfolio: Portfolio) -> float:
        """다각화 비율 계산 (1에 가까울수록 잘 분산됨)"""
        if len(portfolio.positions) <= 1:
            return 0.0
        
        # 간단한 허핀달 지수 기반 다각화 측정
        sum_of_squares = sum(pos.weight ** 2 for pos in portfolio.positions)
        max_concentration = 1.0  # 모든 자금이 한 종목에 집중된 경우
        
        # 다각화 비율 = 1 - (실제 집중도 / 최대 집중도)
        diversification = 1 - (sum_of_squares / max_concentration)
        
        return max(0, min(1, diversification))
    
    def _calculate_max_correlation_exposure(self, portfolio: Portfolio) -> float:
        """최대 상관관계 노출 계산"""
        high_corr_groups = self._find_high_correlation_groups(portfolio)
        if not high_corr_groups:
            return 0.0
        
        return max(group['total_weight'] for group in high_corr_groups)
    
    def _generate_recommendations(self, portfolio: Portfolio, violations: List[Dict]) -> List[Dict[str, Any]]:
        """개선 제안 생성"""
        recommendations = []
        
        # 위반 사항별 권고
        for violation in violations:
            if violation['type'] == 'position_count':
                recommendations.append({
                    'type': 'reduce_positions',
                    'priority': 'high',
                    'action': f"{violation['current'] - violation['limit']}개 포지션 정리",
                    'benefit': '포트폴리오 관리 복잡성 감소'
                })
            
            elif violation['type'] == 'single_position':
                recommendations.append({
                    'type': 'reduce_concentration',
                    'priority': 'high',
                    'symbol': violation['symbol'],
                    'action': f"{violation['symbol']} 비중을 {violation['limit']:.0%}로 축소",
                    'benefit': '단일 종목 리스크 감소'
                })
            
            elif violation['type'] == 'sector_exposure':
                recommendations.append({
                    'type': 'sector_diversification',
                    'priority': 'medium',
                    'sector': violation['sector'],
                    'action': f"{violation['sector']} 섹터 비중 축소 또는 다른 섹터 확대",
                    'benefit': '섹터 리스크 분산'
                })
        
        # 일반적인 개선 제안
        if portfolio.position_count < 3:
            recommendations.append({
                'type': 'increase_diversification',
                'priority': 'medium',
                'action': '포트폴리오 종목 수 확대 (3-5개 권장)',
                'benefit': '분산 효과 향상'
            })
        
        return recommendations
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """종목의 섹터 조회 (Mock)"""
        # Mock implementation - 실제로는 종목 정보 DB에서 조회
        sector_map = {
            'technology': ['005930', '000660', '035420'],  # 삼성전자, SK하이닉스, NAVER
            'finance': ['055550', '086790', '032830'],      # 신한지주, 하나금융지주, 삼성생명
            'manufacturing': ['005380', '012330', '000270'], # 현대차, 현대모비스, 기아
        }
        
        for sector, symbols in sector_map.items():
            if symbol in symbols:
                return sector
        
        return 'other'
    
    def _assess_correlation_risk(self, symbol: str, portfolio: Portfolio, planned_weight: float) -> Dict:
        """상관관계 리스크 평가"""
        # Mock implementation
        symbol_sector = self._get_symbol_sector(symbol)
        
        high_corr_weight = 0.0
        for position in portfolio.positions:
            if position.sector == symbol_sector:  # 같은 섹터는 높은 상관관계로 가정
                high_corr_weight += position.weight
        
        new_high_corr_weight = high_corr_weight + planned_weight
        
        return {
            'high_risk': new_high_corr_weight > self.limits['max_correlation_exposure'],
            'blocking': new_high_corr_weight > self.limits['max_correlation_exposure'] * 1.1,
            'message': f"높은 상관관계 그룹 비중: {new_high_corr_weight:.1%}",
            'max_ratio': min(1.0, (self.limits['max_correlation_exposure'] - high_corr_weight) / planned_weight) if planned_weight > 0 else 0
        }
    
    def _calculate_correlation_constraint(self, symbol: str, portfolio: Portfolio) -> float:
        """상관관계 제약으로 인한 최대 투자 가능 금액"""
        symbol_sector = self._get_symbol_sector(symbol)
        current_sector_weight = portfolio.sector_weights.get(symbol_sector, 0.0)
        
        max_additional_weight = self.limits['max_correlation_exposure'] - current_sector_weight
        max_additional_amount = max(0, max_additional_weight * portfolio.total_value)
        
        return max_additional_amount
    
    def _identify_limiting_factor(self, constraints: Dict) -> str:
        """제약 요소 중 가장 제한적인 요소 식별"""
        constraint_values = {
            'single_position': constraints['max_single_position'],
            'cash': constraints['available_cash'] - constraints['min_cash_reserve'],
            'sector': constraints['max_sector_addition'],
            'correlation': constraints['max_correlation_addition']
        }
        
        return min(constraint_values, key=constraint_values.get)
    
    def _assess_position_risk(self, symbol: str, position_size: float, portfolio: Portfolio) -> str:
        """포지션 리스크 평가"""
        if position_size <= 0:
            return 'none'
        
        position_weight = position_size / portfolio.total_value
        
        if position_weight > self.limits['max_single_position'] * 0.8:
            return 'high'
        elif position_weight > self.limits['max_single_position'] * 0.5:
            return 'medium'
        else:
            return 'low'