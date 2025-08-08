"""
Fundamental Event Filter - Critical Phase 1 Module
뉴스/공시 필터링 시스템: 펀더멘털 이벤트로 인한 급격한 가격 변동 리스크 사전 차단
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, date
import re
from enum import Enum

from ...config import config
from ...utils import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Event risk levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FundamentalEvent:
    """Individual fundamental event"""
    event_type: str
    title: str
    date: datetime
    risk_level: RiskLevel
    description: str
    impact_assessment: str
    blackout_start: Optional[datetime] = None
    blackout_end: Optional[datetime] = None
    confidence: float = 1.0


@dataclass
class FilterDecision:
    """Final filtering decision"""
    can_buy: bool
    reason: str
    risk_events: List[FundamentalEvent]
    retry_after: Optional[datetime] = None
    warnings: List[str] = None
    position_adjustment: float = 1.0  # Multiplier for position size


class FundamentalEventFilter:
    """
    뉴스/공시 필터링 시스템
    
    주요 기능:
    - 실적 발표 일정 체크 (D-3 ~ D+1 블랙아웃)
    - 중요 공시 모니터링 및 위험도 분류
    - 뉴스 감성 분석
    - 배당락일 체크
    - 종합적 매수 가능 여부 판단
    """
    
    def __init__(self):
        self.config = config.trading
        
        # Event blackout periods (days)
        self.blackout_periods = {
            'earnings': {'before': 3, 'after': 1},        # 실적발표 D-3 ~ D+1
            'dividend': {'before': 1, 'after': 1},        # 배당락 D-1 ~ D+1  
            'major_disclosure': {'before': 0, 'after': 2}, # 중요공시 D ~ D+2
            'shareholder_meeting': {'before': 2, 'after': 1}, # 주주총회 D-2 ~ D+1
            'rights_offering': {'before': 5, 'after': 2}   # 유상증자 D-5 ~ D+2
        }
        
        # Major disclosure keywords (Korean)
        self.major_disclosure_keywords = [
            # 자본 관련
            '유상증자', '무상증자', '감자', '분할', '합병',
            # 거래 관련  
            '상장폐지', '거래정지', '관리종목', '투자주의', '투자경고',
            # 실적 관련
            '영업정지', '특별손실', '자산손상', '대손충당금',
            # 지배구조 관련
            '최대주주변경', '경영권', '적대적인수',
            # 기타 중요 사항
            '상한가', '하한가', '기업회생', '워크아웃', '법정관리',
            '특별관계자거래', '내부자거래',
            # 실적 변화
            '흑자전환', '적자전환', '영업손실'
        ]
        
        # News sentiment keywords
        self.negative_news_keywords = [
            '하락', '급락', '폭락', '부진', '악화', '위험', '우려',
            '손실', '적자', '감소', '축소', '중단', '연기', '취소',
            '조사', '수사', '기소', '제재', '처분', '과징금'
        ]
        
        self.positive_news_keywords = [
            '상승', '급등', '폭등', '호조', '개선', '성장', '확대',
            '수익', '흑자', '증가', '신규', '출시', '계약', '수주'
        ]
        
        # News sentiment analysis cache
        self.sentiment_cache = {}
        self.sentiment_cache_ttl = 3600  # 1 hour
    
    def check_fundamental_events(self, symbol: str, 
                               check_date: Optional[datetime] = None) -> Tuple[bool, List[FundamentalEvent]]:
        """
        종합적인 펀더멘털 이벤트 체크
        
        Args:
            symbol: 종목 코드
            check_date: 체크할 날짜 (기본: 현재)
            
        Returns:
            (is_safe, risk_events): 안전 여부와 위험 이벤트 목록
        """
        if check_date is None:
            check_date = datetime.now()
        
        logger.info(f"Checking fundamental events for {symbol}")
        
        risk_events = []
        
        try:
            # 1. 실적 발표 체크
            earnings_events = self._check_earnings_schedule(symbol, check_date)
            risk_events.extend(earnings_events)
            
            # 2. 공시 분석
            disclosure_events = self._check_recent_disclosures(symbol, check_date)
            risk_events.extend(disclosure_events)
            
            # 3. 뉴스 감성 분석
            news_events = self._analyze_news_sentiment(symbol, check_date)
            risk_events.extend(news_events)
            
            # 4. 배당락일 체크
            dividend_events = self._check_dividend_schedule(symbol, check_date)
            risk_events.extend(dividend_events)
            
            # 5. 기타 주요 이벤트
            other_events = self._check_other_events(symbol, check_date)
            risk_events.extend(other_events)
            
            # 6. 안전성 평가
            is_safe = self._assess_overall_safety(risk_events)
            
            logger.info(f"Fundamental check complete for {symbol} - "
                       f"Safe: {is_safe}, Events: {len(risk_events)}")
            
            return is_safe, risk_events
            
        except Exception as e:
            logger.error(f"Error checking fundamental events for {symbol}: {e}")
            # Conservative approach - assume not safe
            return False, [FundamentalEvent(
                event_type='system_error',
                title='시스템 오류로 인한 예방적 차단',
                date=check_date,
                risk_level=RiskLevel.HIGH,
                description=str(e),
                impact_assessment='시스템 오류'
            )]
    
    def get_filter_decision(self, symbol: str, 
                          check_date: Optional[datetime] = None) -> FilterDecision:
        """
        최종 필터링 결정
        
        Args:
            symbol: 종목 코드
            check_date: 체크할 날짜
            
        Returns:
            FilterDecision: 매수 가능 여부 및 상세 정보
        """
        is_safe, events = self.check_fundamental_events(symbol, check_date)
        
        if not is_safe:
            # 고위험 이벤트 우선 처리
            high_risk_events = [e for e in events if e.risk_level == RiskLevel.HIGH]
            critical_events = [e for e in events if e.risk_level == RiskLevel.CRITICAL]
            
            if critical_events:
                primary_event = critical_events[0]
                return FilterDecision(
                    can_buy=False,
                    reason=f"치명적 리스크 이벤트: {primary_event.event_type}",
                    risk_events=events,
                    retry_after=primary_event.blackout_end,
                    warnings=[f"즉시 매매 중단 권고: {primary_event.title}"]
                )
            
            elif high_risk_events:
                primary_event = high_risk_events[0]
                return FilterDecision(
                    can_buy=False,
                    reason=f"고위험 이벤트 감지: {primary_event.event_type}",
                    risk_events=events,
                    retry_after=primary_event.blackout_end or 
                               (check_date or datetime.now()) + timedelta(days=2),
                    warnings=[primary_event.impact_assessment]
                )
        
        # 중간 위험 이벤트들 평가
        medium_risk_count = len([e for e in events if e.risk_level == RiskLevel.MEDIUM])
        
        if medium_risk_count >= 2:
            return FilterDecision(
                can_buy=False,
                reason="복수의 중간 위험 이벤트 감지",
                risk_events=events,
                retry_after=(check_date or datetime.now()) + timedelta(days=1),
                warnings=["다수 위험 요소로 인한 신중 대기"]
            )
        
        # 낮은 위험 또는 안전
        position_adjustment = 1.0
        warnings = []
        
        if medium_risk_count == 1:
            position_adjustment = 0.7  # 포지션 30% 축소
            warnings.append("중간 위험 요소로 인한 포지션 축소 권고")
        
        low_risk_count = len([e for e in events if e.risk_level == RiskLevel.LOW])
        if low_risk_count > 0:
            position_adjustment *= 0.9  # 추가 10% 축소
            warnings.append("경미한 위험 요소 존재")
        
        return FilterDecision(
            can_buy=True,
            reason="펀더멘털 이벤트 체크 통과",
            risk_events=events,
            warnings=warnings,
            position_adjustment=position_adjustment
        )
    
    def _check_earnings_schedule(self, symbol: str, check_date: datetime) -> List[FundamentalEvent]:
        """실적 발표 일정 체크"""
        events = []
        
        try:
            # 실제 구현에서는 실적 발표 일정 API 호출
            earnings_date = self._get_earnings_date(symbol, check_date)
            
            if earnings_date:
                days_until = (earnings_date - check_date.date()).days
                blackout = self.blackout_periods['earnings']
                
                if -blackout['after'] <= days_until <= blackout['before']:
                    risk_level = RiskLevel.HIGH if abs(days_until) <= 1 else RiskLevel.MEDIUM
                    
                    events.append(FundamentalEvent(
                        event_type='earnings',
                        title=f'실적발표 예정 ({earnings_date.strftime("%Y-%m-%d")})',
                        date=datetime.combine(earnings_date, datetime.min.time()),
                        risk_level=risk_level,
                        description=f'실적발표까지 {days_until}일',
                        impact_assessment='실적 서프라이즈 위험',
                        blackout_start=datetime.combine(
                            earnings_date - timedelta(days=blackout['before']), 
                            datetime.min.time()
                        ),
                        blackout_end=datetime.combine(
                            earnings_date + timedelta(days=blackout['after']), 
                            datetime.max.time()
                        )
                    ))
            
        except Exception as e:
            logger.warning(f"Error checking earnings schedule for {symbol}: {e}")
            
        return events
    
    def _check_recent_disclosures(self, symbol: str, check_date: datetime) -> List[FundamentalEvent]:
        """최근 공시 분석"""
        events = []
        
        try:
            # 실제 구현에서는 공시 API 호출 (KIND, DART 등)
            disclosures = self._get_recent_disclosures(symbol, days=3)
            
            for disclosure in disclosures:
                if self._is_major_disclosure(disclosure):
                    risk_level = self._assess_disclosure_risk(disclosure)
                    
                    # 블랙아웃 기간 계산
                    disclosure_date = disclosure['date']
                    blackout = self.blackout_periods['major_disclosure']
                    
                    events.append(FundamentalEvent(
                        event_type='disclosure',
                        title=disclosure['title'],
                        date=disclosure_date,
                        risk_level=risk_level,
                        description=disclosure.get('summary', ''),
                        impact_assessment=self._generate_disclosure_impact(disclosure),
                        blackout_start=disclosure_date,
                        blackout_end=disclosure_date + timedelta(days=blackout['after'])
                    ))
                    
        except Exception as e:
            logger.warning(f"Error checking disclosures for {symbol}: {e}")
            
        return events
    
    def _analyze_news_sentiment(self, symbol: str, check_date: datetime) -> List[FundamentalEvent]:
        """뉴스 감성 분석"""
        events = []
        
        try:
            # Cache check
            cache_key = f"{symbol}_{check_date.date()}"
            if cache_key in self.sentiment_cache:
                cached_result = self.sentiment_cache[cache_key]
                if (datetime.now() - cached_result['timestamp']).total_seconds() < self.sentiment_cache_ttl:
                    return cached_result['events']
            
            # 실제 구현에서는 뉴스 API 호출
            news_items = self._get_recent_news(symbol, days=2)
            
            if news_items:
                sentiment_score = self._calculate_news_sentiment(news_items)
                
                if sentiment_score < -0.3:  # 강한 부정적 감정
                    negative_headlines = [
                        item['title'] for item in news_items 
                        if self._is_negative_news(item['title'])
                    ]
                    
                    events.append(FundamentalEvent(
                        event_type='negative_news',
                        title='부정적 뉴스 감성 감지',
                        date=check_date,
                        risk_level=RiskLevel.MEDIUM,
                        description=f'감성 점수: {sentiment_score:.2f}',
                        impact_assessment=f'부정적 헤드라인 {len(negative_headlines)}건',
                        confidence=abs(sentiment_score)
                    ))
                
                elif sentiment_score < -0.1:  # 약한 부정적 감정
                    events.append(FundamentalEvent(
                        event_type='weak_negative_news',
                        title='약한 부정적 뉴스 감성',
                        date=check_date,
                        risk_level=RiskLevel.LOW,
                        description=f'감성 점수: {sentiment_score:.2f}',
                        impact_assessment='주의 관찰 필요',
                        confidence=abs(sentiment_score)
                    ))
            
            # Cache result
            self.sentiment_cache[cache_key] = {
                'timestamp': datetime.now(),
                'events': events
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing news sentiment for {symbol}: {e}")
            
        return events
    
    def _check_dividend_schedule(self, symbol: str, check_date: datetime) -> List[FundamentalEvent]:
        """배당락일 체크"""
        events = []
        
        try:
            # 실제 구현에서는 배당 일정 API 호출
            ex_dividend_date = self._get_ex_dividend_date(symbol)
            
            if ex_dividend_date:
                days_until = (ex_dividend_date - check_date.date()).days
                blackout = self.blackout_periods['dividend']
                
                if -blackout['after'] <= days_until <= blackout['before']:
                    dividend_yield = self._get_dividend_yield(symbol)
                    
                    risk_level = RiskLevel.MEDIUM if dividend_yield > 0.03 else RiskLevel.LOW
                    
                    events.append(FundamentalEvent(
                        event_type='dividend',
                        title=f'배당락일 임박 ({ex_dividend_date.strftime("%Y-%m-%d")})',
                        date=datetime.combine(ex_dividend_date, datetime.min.time()),
                        risk_level=risk_level,
                        description=f'배당수익률: {dividend_yield:.2%}',
                        impact_assessment='배당락 가격 조정 예상'
                    ))
                    
        except Exception as e:
            logger.warning(f"Error checking dividend schedule for {symbol}: {e}")
            
        return events
    
    def _check_other_events(self, symbol: str, check_date: datetime) -> List[FundamentalEvent]:
        """기타 주요 이벤트 체크"""
        events = []
        
        try:
            # 주주총회
            agm_date = self._get_shareholders_meeting_date(symbol)
            if agm_date:
                days_until = (agm_date - check_date.date()).days
                blackout = self.blackout_periods['shareholder_meeting']
                
                if -blackout['after'] <= days_until <= blackout['before']:
                    events.append(FundamentalEvent(
                        event_type='shareholders_meeting',
                        title=f'주주총회 ({agm_date.strftime("%Y-%m-%d")})',
                        date=datetime.combine(agm_date, datetime.min.time()),
                        risk_level=RiskLevel.LOW,
                        description='주주총회 관련 이슈 가능성',
                        impact_assessment='일반적으로 영향 제한적'
                    ))
            
            # 기타 기업 행동 (분할, 합병 등)은 공시에서 이미 감지됨
            
        except Exception as e:
            logger.warning(f"Error checking other events for {symbol}: {e}")
            
        return events
    
    def _assess_overall_safety(self, events: List[FundamentalEvent]) -> bool:
        """전체 안전성 평가"""
        if not events:
            return True
        
        # 위험도별 개수 계산
        critical_count = sum(1 for e in events if e.risk_level == RiskLevel.CRITICAL)
        high_count = sum(1 for e in events if e.risk_level == RiskLevel.HIGH)
        medium_count = sum(1 for e in events if e.risk_level == RiskLevel.MEDIUM)
        
        # 위험 이벤트가 하나라도 있으면 불안전
        if critical_count > 0 or high_count > 0:
            return False
        
        # 중간 위험 이벤트가 2개 이상이면 불안전
        if medium_count >= 2:
            return False
        
        return True
    
    # Mock data methods (실제 구현에서는 실제 API로 대체)
    
    def _get_earnings_date(self, symbol: str, check_date: datetime) -> Optional[date]:
        """실적 발표일 조회 (Mock)"""
        # Mock: 분기별 실적 발표 (3, 6, 9, 12월)
        current_quarter = ((check_date.month - 1) // 3) + 1
        
        # 다음 분기 첫 달 15일경 실적 발표
        next_quarter_month = current_quarter * 3 + 1
        if next_quarter_month > 12:
            next_quarter_month -= 12
            earnings_year = check_date.year + 1
        else:
            earnings_year = check_date.year
            
        earnings_date = date(earnings_year, next_quarter_month, 15)
        
        # 현재로부터 30일 이내만 반환
        if (earnings_date - check_date.date()).days <= 30:
            return earnings_date
        
        return None
    
    def _get_recent_disclosures(self, symbol: str, days: int = 7) -> List[Dict]:
        """최근 공시 목록 조회 (Mock)"""
        # Mock data
        return [
            {
                'title': '2024년 3분기 실적 공시',
                'date': datetime.now() - timedelta(days=1),
                'summary': '분기 실적 발표'
            },
            {
                'title': '주요사항보고서(신규시설투자)',
                'date': datetime.now() - timedelta(days=2), 
                'summary': '신규 설비 투자 계획'
            }
        ]
    
    def _get_recent_news(self, symbol: str, days: int = 2) -> List[Dict]:
        """최근 뉴스 조회 (Mock)"""
        return [
            {'title': f'{symbol} 실적 개선 전망', 'date': datetime.now() - timedelta(hours=12)},
            {'title': f'{symbol} 신제품 출시 발표', 'date': datetime.now() - timedelta(hours=24)}
        ]
    
    def _get_ex_dividend_date(self, symbol: str) -> Optional[date]:
        """배당락일 조회 (Mock)"""
        # Mock: 12월 말 배당락
        current_year = datetime.now().year
        return date(current_year, 12, 28)
    
    def _get_dividend_yield(self, symbol: str) -> float:
        """배당수익률 조회 (Mock)"""
        return 0.025  # 2.5%
    
    def _get_shareholders_meeting_date(self, symbol: str) -> Optional[date]:
        """주주총회일 조회 (Mock)"""
        return date(datetime.now().year, 3, 28)  # 3월 말 일반적
    
    # Analysis helper methods
    
    def _is_major_disclosure(self, disclosure: Dict) -> bool:
        """주요 공시 여부 판별"""
        title = disclosure['title']
        return any(keyword in title for keyword in self.major_disclosure_keywords)
    
    def _assess_disclosure_risk(self, disclosure: Dict) -> RiskLevel:
        """공시 위험도 평가"""
        title = disclosure['title']
        
        # Critical risk keywords
        critical_keywords = ['상장폐지', '거래정지', '기업회생', '법정관리', '워크아웃']
        if any(keyword in title for keyword in critical_keywords):
            return RiskLevel.CRITICAL
        
        # High risk keywords  
        high_risk_keywords = ['유상증자', '감자', '합병', '분할', '최대주주변경']
        if any(keyword in title for keyword in high_risk_keywords):
            return RiskLevel.HIGH
        
        # Medium risk keywords
        medium_risk_keywords = ['특별손실', '영업정지', '자산손상', '적자전환']
        if any(keyword in title for keyword in medium_risk_keywords):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _generate_disclosure_impact(self, disclosure: Dict) -> str:
        """공시 영향 평가 메시지 생성"""
        title = disclosure['title']
        
        if '실적' in title:
            return '실적 변동성 가능'
        elif '증자' in title:
            return '지분 희석 우려'
        elif '합병' in title or '분할' in title:
            return '기업 구조 변경'
        else:
            return '상세 분석 필요'
    
    def _calculate_news_sentiment(self, news_items: List[Dict]) -> float:
        """뉴스 감성 점수 계산"""
        if not news_items:
            return 0.0
        
        total_score = 0.0
        for item in news_items:
            title = item['title']
            
            positive_count = sum(1 for keyword in self.positive_news_keywords if keyword in title)
            negative_count = sum(1 for keyword in self.negative_news_keywords if keyword in title)
            
            # 단순 키워드 기반 감성 점수
            item_score = (positive_count - negative_count) / max(len(title.split()), 1)
            total_score += item_score
        
        return total_score / len(news_items)
    
    def _is_negative_news(self, title: str) -> bool:
        """부정적 뉴스 여부 판별"""
        return any(keyword in title for keyword in self.negative_news_keywords)