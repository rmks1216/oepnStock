"""
Quick Paper Trading Test
빠른 페이퍼 트레이딩 테스트 (5분 실행)
"""

import sys
import os
from datetime import datetime, timedelta
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oepnstock.utils.krx_data_source import get_data_provider
from oepnstock.utils import get_logger

logger = get_logger(__name__)


class QuickPaperTest:
    """빠른 페이퍼 트레이딩 테스트"""
    
    def __init__(self):
        self.data_provider = get_data_provider()
        self.portfolio = {
            'cash': 10000000,  # 1천만원
            'positions': [],
            'total_value': 10000000
        }
        
        # 테스트할 종목들
        self.test_symbols = ['005930', '000660', '035420']
        self.symbol_names = {
            '005930': '삼성전자',
            '000660': 'SK하이닉스',
            '035420': 'NAVER'
        }
        
        logger.info("Quick Paper Trading Test initialized")
    
    def _get_recent_data(self, symbol: str, days: int = 60):
        """최근 N일 KRX 일봉 데이터 조회 (현재가는 최근 종가로 근사)"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        return self.data_provider.get_historical_data(symbol, start_date, end_date)

    def analyze_opportunity(self, symbol: str) -> dict:
        """간단한 매매 기회 분석"""
        try:
            # 과거 데이터로 간단한 기술분석 (현재가는 최근 종가로 근사)
            hist_data = self._get_recent_data(symbol)
            if hist_data is None or hist_data.empty:
                return {'signal': 'NO_DATA', 'reason': 'Historical data unavailable'}

            current_price = float(hist_data['close'].iloc[-1])

            # 단순 이동평균 계산
            hist_data['ma5'] = hist_data['close'].rolling(5).mean()
            hist_data['ma20'] = hist_data['close'].rolling(20).mean()
            
            latest = hist_data.iloc[-1]
            ma5 = latest['ma5']
            ma20 = latest['ma20']
            
            # 간단한 매매 신호 생성
            if current_price > ma5 > ma20:
                signal = 'BUY'
                reason = f'상승 추세 (가격: {current_price:,.0f} > MA5: {ma5:,.0f} > MA20: {ma20:,.0f})'
            elif current_price < ma5 < ma20:
                signal = 'SELL'
                reason = f'하락 추세 (가격: {current_price:,.0f} < MA5: {ma5:,.0f} < MA20: {ma20:,.0f})'
            else:
                signal = 'HOLD'
                reason = f'횡보 (가격: {current_price:,.0f}, MA5: {ma5:,.0f}, MA20: {ma20:,.0f})'
            
            return {
                'signal': signal,
                'current_price': current_price,
                'ma5': ma5,
                'ma20': ma20,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}
    
    def execute_paper_trade(self, symbol: str, signal: str, price: float):
        """페이퍼 트레이딩 주문 실행"""
        if signal == 'BUY' and len(self.portfolio['positions']) < 3:
            # 매수: 포트폴리오의 10% 투자
            investment = self.portfolio['total_value'] * 0.10
            shares = int(investment / price)
            cost = shares * price
            
            if cost <= self.portfolio['cash']:
                position = {
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'cost': cost
                }
                
                self.portfolio['positions'].append(position)
                self.portfolio['cash'] -= cost
                
                logger.info(f"📝 BUY: {shares}주 of {symbol} at {price:,.0f}원")
                return f"매수 완료: {shares}주 ({cost:,.0f}원)"
        
        elif signal == 'SELL':
            # 매도: 해당 종목 보유 시에만
            for i, pos in enumerate(self.portfolio['positions']):
                if pos['symbol'] == symbol:
                    proceeds = pos['shares'] * price
                    pnl = proceeds - pos['cost']
                    pnl_pct = pnl / pos['cost'] * 100
                    
                    self.portfolio['cash'] += proceeds
                    del self.portfolio['positions'][i]
                    
                    logger.info(f"📝 SELL: {pos['shares']}주 of {symbol} at {price:,.0f}원")
                    logger.info(f"   P&L: {pnl:+,.0f}원 ({pnl_pct:+.1f}%)")
                    
                    return f"매도 완료: {pos['shares']}주, 손익: {pnl:+,.0f}원 ({pnl_pct:+.1f}%)"
        
        return f"거래 없음: {signal}"
    
    def run_test(self):
        """테스트 실행"""
        print("🚀 Quick Paper Trading Test 시작")
        print(f"초기 자본: {self.portfolio['cash']:,.0f}원")
        print("=" * 60)
        
        for symbol in self.test_symbols:
            name = self.symbol_names[symbol]
            print(f"\n📊 {symbol} ({name}) 분석 중...")
            
            # 분석 실행
            analysis = self.analyze_opportunity(symbol)
            
            print(f"신호: {analysis['signal']}")
            print(f"사유: {analysis.get('reason', 'N/A')}")
            
            # 매매 실행
            if analysis['signal'] in ['BUY', 'SELL']:
                result = self.execute_paper_trade(
                    symbol, 
                    analysis['signal'], 
                    analysis['current_price']
                )
                print(f"결과: {result}")
        
        # 최종 포트폴리오 상태
        print("\n" + "=" * 60)
        print("📊 최종 포트폴리오 상태")
        print(f"현금: {self.portfolio['cash']:,.0f}원")
        print(f"보유 종목: {len(self.portfolio['positions'])}개")
        
        total_position_value = 0
        for pos in self.portfolio['positions']:
            recent = self._get_recent_data(pos['symbol'], days=10)
            current_price = float(recent['close'].iloc[-1]) if recent is not None and not recent.empty else None
            if current_price:
                current_value = pos['shares'] * current_price
                pnl = current_value - pos['cost']
                pnl_pct = pnl / pos['cost'] * 100
                
                print(f"  - {pos['symbol']}: {pos['shares']}주, "
                      f"현재가치: {current_value:,.0f}원, "
                      f"손익: {pnl:+,.0f}원 ({pnl_pct:+.1f}%)")
                total_position_value += current_value
        
        total_value = self.portfolio['cash'] + total_position_value
        total_pnl = total_value - 10000000
        total_pnl_pct = total_pnl / 10000000 * 100
        
        print(f"\n총 자산: {total_value:,.0f}원")
        print(f"총 손익: {total_pnl:+,.0f}원 ({total_pnl_pct:+.2f}%)")
        
        if total_pnl > 0:
            print("🎉 수익 달성!")
        else:
            print("📉 손실 발생")
        
        print("\n✅ 테스트 완료!")


def main():
    """메인 실행"""
    test = QuickPaperTest()
    test.run_test()


if __name__ == "__main__":
    main()