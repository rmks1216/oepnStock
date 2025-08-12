"""
Korean stock market specific utilities and constants
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime, time, date
from enum import Enum
import pandas as pd

from ..config import config


class MarketSession(Enum):
    """Korean stock market trading sessions"""
    PRE_OPENING = "pre_opening"          # 08:30-09:00
    OPENING_AUCTION = "opening_auction"   # 09:00-09:10  
    MORNING_VOLATILITY = "morning_volatility"  # 09:10-09:30
    MORNING_TREND = "morning_trend"       # 09:30-11:00
    LUNCH_LULL = "lunch_lull"            # 11:00-13:00
    AFTERNOON_TREND = "afternoon_trend"   # 13:00-14:30
    CLOSING_VOLATILITY = "closing_volatility"  # 14:30-15:20
    CLOSING_AUCTION = "closing_auction"   # 15:20-15:30


class KoreanMarketUtils:
    """
    Korean stock market specific utility functions
    """
    
    # Korean market sectors
    SECTORS = {
        'technology': ['IT', '소프트웨어', '반도체', '전자부품', '컴퓨터'],
        'finance': ['은행', '증권', '보험', '종합금융'],
        'manufacturing': ['자동차', '철강', '화학', '기계', '조선'],
        'consumer': ['유통', '식음료', '의류', '화장품', '생활용품'],
        'healthcare': ['제약', '의료기기', '바이오', '의료서비스'],
        'energy': ['전력', '가스', '석유화학', '에너지'],
        'materials': ['건설', '부동산', '건자재', '종이'],
        'telecom': ['통신서비스', '미디어']
    }
    
    # Market cap categories (KRW)
    MARKET_CAP_CATEGORIES = {
        'large': 2_000_000_000_000,      # 2조원 이상
        'mid': 300_000_000_000,          # 3천억원 이상
        'small': 50_000_000_000,         # 500억원 이상
        'micro': 0                       # 500억원 미만
    }
    
    # Trading hours
    TRADING_HOURS = {
        MarketSession.PRE_OPENING: (time(8, 30), time(9, 0)),
        MarketSession.OPENING_AUCTION: (time(9, 0), time(9, 10)),
        MarketSession.MORNING_VOLATILITY: (time(9, 10), time(9, 30)),
        MarketSession.MORNING_TREND: (time(9, 30), time(11, 0)),
        MarketSession.LUNCH_LULL: (time(11, 0), time(13, 0)),
        MarketSession.AFTERNOON_TREND: (time(13, 0), time(14, 30)),
        MarketSession.CLOSING_VOLATILITY: (time(14, 30), time(15, 20)),
        MarketSession.CLOSING_AUCTION: (time(15, 20), time(15, 30))
    }
    
    @staticmethod
    def get_current_market_session(current_time: Optional[datetime] = None) -> Optional[MarketSession]:
        """
        Get current market session based on time
        
        Args:
            current_time: Current time (default: now)
            
        Returns:
            Current market session or None if market is closed
        """
        if current_time is None:
            current_time = datetime.now()
        
        current_time_only = current_time.time()
        
        for session, (start_time, end_time) in KoreanMarketUtils.TRADING_HOURS.items():
            if start_time <= current_time_only <= end_time:
                return session
        
        return None
    
    @staticmethod
    def is_market_open(current_time: Optional[datetime] = None) -> bool:
        """Check if Korean stock market is currently open"""
        return KoreanMarketUtils.get_current_market_session(current_time) is not None
    
    @staticmethod
    def is_trading_day(check_date: Optional[date] = None) -> bool:
        """
        Check if given date is a trading day (excluding weekends and Korean holidays)
        """
        if check_date is None:
            check_date = date.today()
        
        # Exclude weekends
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check Korean holidays
        if KoreanMarketUtils._is_korean_holiday(check_date):
            return False
        
        return True
    
    @staticmethod
    def _is_korean_holiday(check_date: date) -> bool:
        """
        Check if the given date is a Korean public holiday
        Includes major holidays that affect stock market trading
        """
        year = check_date.year
        month = check_date.month
        day = check_date.day
        
        # Fixed holidays
        fixed_holidays = [
            (1, 1),   # New Year's Day (신정)
            (3, 1),   # Independence Movement Day (삼일절)
            (5, 5),   # Children's Day (어린이날)
            (6, 6),   # Memorial Day (현충일)
            (8, 15),  # Liberation Day (광복절)
            (10, 3),  # National Foundation Day (개천절)
            (10, 9),  # Hangeul Day (한글날)
            (12, 25), # Christmas Day (성탄절)
        ]
        
        if (month, day) in fixed_holidays:
            return True
        
        # Variable holidays (Lunar calendar based - simplified approximation)
        # Note: For production use, consider using a proper lunar calendar library
        lunar_holidays = KoreanMarketUtils._get_lunar_holidays(year)
        
        if check_date in lunar_holidays:
            return True
        
        # Buddha's Birthday (부처님오신날) - 8th day of 4th lunar month
        # Chuseok (추석) - 15th day of 8th lunar month (and surrounding days)
        # These are approximated - for exact dates, use lunar calendar conversion
        
        return False
    
    @staticmethod
    def _get_lunar_holidays(year: int) -> List[date]:
        """
        Get approximate dates for lunar calendar based holidays
        Note: This is a simplified approximation. For exact dates, 
        use a proper lunar calendar conversion library.
        """
        # Approximate lunar holidays for 2024-2026
        # In production, this should be replaced with proper lunar calendar calculation
        lunar_holidays_by_year = {
            2024: [
                date(2024, 2, 9),   # Lunar New Year's Day
                date(2024, 2, 10),  # Lunar New Year 2nd day
                date(2024, 2, 11),  # Lunar New Year 3rd day
                date(2024, 2, 12),  # Lunar New Year alternative holiday
                date(2024, 5, 15),  # Buddha's Birthday
                date(2024, 9, 16),  # Chuseok 1st day
                date(2024, 9, 17),  # Chuseok (Mid-Autumn Festival)
                date(2024, 9, 18),  # Chuseok 3rd day
            ],
            2025: [
                date(2025, 1, 28),  # Lunar New Year's Day
                date(2025, 1, 29),  # Lunar New Year 2nd day
                date(2025, 1, 30),  # Lunar New Year 3rd day
                date(2025, 5, 5),   # Buddha's Birthday (overlaps with Children's Day)
                date(2025, 10, 5),  # Chuseok 1st day
                date(2025, 10, 6),  # Chuseok (Mid-Autumn Festival)
                date(2025, 10, 7),  # Chuseok 3rd day
                date(2025, 10, 8),  # Chuseok alternative holiday
            ],
            2026: [
                date(2026, 2, 16),  # Lunar New Year's Day
                date(2026, 2, 17),  # Lunar New Year 2nd day
                date(2026, 2, 18),  # Lunar New Year 3rd day
                date(2026, 5, 24),  # Buddha's Birthday
                date(2026, 9, 24),  # Chuseok 1st day
                date(2026, 9, 25),  # Chuseok (Mid-Autumn Festival)
                date(2026, 9, 26),  # Chuseok 3rd day
            ]
        }
        
        return lunar_holidays_by_year.get(year, [])
    
    @staticmethod
    def get_market_cap_category(market_cap_krw: int) -> str:
        """Categorize stock by market capitalization"""
        for category, threshold in sorted(KoreanMarketUtils.MARKET_CAP_CATEGORIES.items(), 
                                        key=lambda x: x[1], reverse=True):
            if market_cap_krw >= threshold:
                return category
        return 'micro'
    
    @staticmethod
    def calculate_round_figure_price(price: float) -> float:
        """
        Calculate appropriate round figure price for Korean market
        
        Args:
            price: Current price
            
        Returns:
            Nearest round figure price below current price
        """
        if price < 10000:
            round_unit = 100      # 100원 단위
        elif price < 100000:
            round_unit = 1000     # 1,000원 단위  
        else:
            round_unit = 5000     # 5,000원 단위
        
        return int(price / round_unit) * round_unit
    
    @staticmethod
    def get_price_tick_size(price: float) -> float:
        """
        Get minimum price tick size based on price level
        Korean market has different tick sizes for different price ranges
        """
        if price < 1000:
            return 1        # 1원
        elif price < 5000:
            return 5        # 5원
        elif price < 10000:
            return 10       # 10원
        elif price < 50000:
            return 50       # 50원
        elif price < 100000:
            return 100      # 100원
        elif price < 500000:
            return 500      # 500원
        else:
            return 1000     # 1,000원
    
    @staticmethod
    def classify_sector(company_name: str, business_type: str = "") -> Optional[str]:
        """
        Classify company sector based on name and business type
        
        Args:
            company_name: Company name
            business_type: Business type description
            
        Returns:
            Sector classification or None
        """
        text_to_check = f"{company_name} {business_type}".lower()
        
        for sector, keywords in KoreanMarketUtils.SECTORS.items():
            for keyword in keywords:
                if keyword.lower() in text_to_check:
                    return sector
        
        return None
    
    @staticmethod
    def calculate_daily_limit(previous_close: float, market_type: str = "kospi") -> Tuple[float, float]:
        """
        Calculate daily price limits (상한가/하한가)
        
        Args:
            previous_close: Previous day's closing price
            market_type: "kospi" or "kosdaq"
            
        Returns:
            Tuple of (upper_limit, lower_limit)
        """
        if market_type.lower() == "kospi":
            limit_ratio = 0.30  # 30%
        else:  # KOSDAQ
            limit_ratio = 0.30  # 30%
        
        upper_limit = previous_close * (1 + limit_ratio)
        lower_limit = previous_close * (1 - limit_ratio)
        
        # Apply tick size rounding
        upper_limit = KoreanMarketUtils._round_to_tick_size(upper_limit)
        lower_limit = KoreanMarketUtils._round_to_tick_size(lower_limit)
        
        return upper_limit, lower_limit
    
    @staticmethod
    def _round_to_tick_size(price: float) -> float:
        """Round price to appropriate tick size"""
        tick_size = KoreanMarketUtils.get_price_tick_size(price)
        return round(price / tick_size) * tick_size
    
    @staticmethod
    def is_lunch_break(current_time: Optional[datetime] = None) -> bool:
        """Check if current time is during lunch break"""
        current_session = KoreanMarketUtils.get_current_market_session(current_time)
        return current_session == MarketSession.LUNCH_LULL
    
    @staticmethod
    def get_session_characteristics(session: MarketSession) -> Dict[str, any]:
        """Get trading characteristics for each session"""
        characteristics = {
            MarketSession.PRE_OPENING: {
                'volatility': 'low',
                'volume': 'low', 
                'suitable_for': 'preparation',
                'avoid': ['market_orders']
            },
            MarketSession.OPENING_AUCTION: {
                'volatility': 'extreme',
                'volume': 'high',
                'suitable_for': 'gap_analysis',
                'avoid': ['immediate_entry', 'large_positions']
            },
            MarketSession.MORNING_VOLATILITY: {
                'volatility': 'high',
                'volume': 'high',
                'suitable_for': ['trend_following', 'breakout_trading'],
                'avoid': ['contrarian_plays']
            },
            MarketSession.MORNING_TREND: {
                'volatility': 'medium',
                'volume': 'high',
                'suitable_for': ['main_trading', 'trend_following'],
                'avoid': []
            },
            MarketSession.LUNCH_LULL: {
                'volatility': 'low',
                'volume': 'low',
                'suitable_for': ['position_management'],
                'avoid': ['new_entries', 'large_trades']
            },
            MarketSession.AFTERNOON_TREND: {
                'volatility': 'medium',
                'volume': 'medium',
                'suitable_for': ['trend_continuation', 'mean_reversion'],
                'avoid': []
            },
            MarketSession.CLOSING_VOLATILITY: {
                'volatility': 'high',
                'volume': 'high',
                'suitable_for': ['day_trading_exits', 'momentum_plays'],
                'avoid': ['overnight_holds']
            },
            MarketSession.CLOSING_AUCTION: {
                'volatility': 'high',
                'volume': 'high',
                'suitable_for': ['final_adjustments'],
                'avoid': ['new_entries']
            }
        }
        
        return characteristics.get(session, {})
    
    @staticmethod
    def format_korean_number(number: int) -> str:
        """Format number in Korean style (만, 억, 조)"""
        if number >= 1_000_000_000_000:  # 조
            return f"{number / 1_000_000_000_000:.1f}조"
        elif number >= 100_000_000:      # 억
            return f"{number / 100_000_000:.1f}억"
        elif number >= 10_000:           # 만
            return f"{number / 10_000:.0f}만"
        else:
            return f"{number:,}"
    
    @staticmethod
    def get_trading_calendar(year: int) -> List[date]:
        """
        Get Korean stock market trading calendar for given year
        Note: This is a simplified version. Real implementation should use official calendar.
        """
        trading_days = []
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            if KoreanMarketUtils.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date = date.fromordinal(current_date.toordinal() + 1)
        
        return trading_days