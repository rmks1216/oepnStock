"""
Paper Trading Dashboard
페이퍼 트레이딩 성과 모니터링 대시보드
"""

import sys
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oepnstock.utils import get_logger

logger = get_logger(__name__)


class PaperTradingDashboard:
    """페이퍼 트레이딩 대시보드"""
    
    def __init__(self, data_file: str = "paper_trading_history.json"):
        self.data_file = data_file
        self.trades_history = []
        self.daily_portfolio = []
        
        # 데이터 파일 로드
        self.load_data()
        
        logger.info("Paper Trading Dashboard initialized")
    
    def load_data(self):
        """거래 이력 데이터 로드"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trades_history = data.get('trades', [])
                    self.daily_portfolio = data.get('daily_portfolio', [])
                    logger.info(f"Loaded {len(self.trades_history)} trades")
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
    
    def save_data(self):
        """거래 이력 데이터 저장"""
        try:
            data = {
                'trades': self.trades_history,
                'daily_portfolio': self.daily_portfolio,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def add_trade(self, trade: Dict[str, Any]):
        """거래 기록 추가"""
        trade['timestamp'] = datetime.now().isoformat()
        self.trades_history.append(trade)
        self.save_data()
    
    def add_daily_snapshot(self, portfolio: Dict[str, Any]):
        """일일 포트폴리오 스냅샷 추가"""
        snapshot = {
            'date': datetime.now().date().isoformat(),
            'total_value': portfolio.get('total_value', 0),
            'cash': portfolio.get('cash', 0),
            'positions_value': portfolio.get('positions_value', 0),
            'positions_count': len(portfolio.get('positions', [])),
            'daily_pnl': portfolio.get('daily_pnl', 0)
        }
        
        self.daily_portfolio.append(snapshot)
        self.save_data()
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """성과 리포트 생성"""
        if not self.trades_history:
            return {"error": "No trading data available"}
        
        df_trades = pd.DataFrame(self.trades_history)
        
        # 기본 통계
        total_trades = len(df_trades)
        buy_trades = len(df_trades[df_trades['action'] == 'BUY'])
        sell_trades = len(df_trades[df_trades['action'] == 'SELL'])
        
        # 손익 계산 (매도 거래만)
        sell_df = df_trades[df_trades['action'] == 'SELL'].copy()
        if not sell_df.empty:
            total_pnl = sell_df['pnl'].sum()
            winning_trades = len(sell_df[sell_df['pnl'] > 0])
            losing_trades = len(sell_df[sell_df['pnl'] < 0])
            win_rate = winning_trades / len(sell_df) if len(sell_df) > 0 else 0
            
            avg_win = sell_df[sell_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = sell_df[sell_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            best_trade = sell_df['pnl'].max()
            worst_trade = sell_df['pnl'].min()
            
            # 수익률 계산
            if self.daily_portfolio:
                initial_capital = 10000000  # 초기 자본
                current_value = self.daily_portfolio[-1]['total_value']
                total_return = (current_value - initial_capital) / initial_capital
            else:
                total_return = total_pnl / 10000000  # 근사치
        else:
            total_pnl = 0
            win_rate = 0
            avg_win = avg_loss = 0
            best_trade = worst_trade = 0
            total_return = 0
        
        # 활동 통계
        if df_trades['timestamp'].dtype == 'object':
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        
        trading_days = df_trades['timestamp'].dt.date.nunique()
        trades_per_day = total_trades / max(trading_days, 1)
        
        return {
            'summary': {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'trading_days': trading_days,
                'trades_per_day': trades_per_day
            },
            'performance': {
                'total_pnl': total_pnl,
                'total_return_pct': total_return * 100,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }
        }
    
    def plot_performance(self, save_file: str = "paper_trading_performance.png"):
        """성과 차트 생성"""
        if not self.daily_portfolio:
            print("No daily portfolio data to plot")
            return
        
        df_daily = pd.DataFrame(self.daily_portfolio)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Paper Trading Performance Dashboard', fontsize=16)
        
        # 1. 포트폴리오 가치 변화
        axes[0, 0].plot(df_daily['date'], df_daily['total_value'], 'b-', linewidth=2)
        axes[0, 0].axhline(y=10000000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Value (KRW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 현금 vs 포지션 가치
        axes[0, 1].plot(df_daily['date'], df_daily['cash'], 'g-', label='Cash', linewidth=2)
        axes[0, 1].plot(df_daily['date'], df_daily['positions_value'], 'r-', label='Positions', linewidth=2)
        axes[0, 1].set_title('Cash vs Positions Value')
        axes[0, 1].set_ylabel('Value (KRW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 일일 손익
        if 'daily_pnl' in df_daily.columns:
            colors = ['green' if x >= 0 else 'red' for x in df_daily['daily_pnl']]
            axes[1, 0].bar(df_daily['date'], df_daily['daily_pnl'], color=colors, alpha=0.7)
            axes[1, 0].set_title('Daily P&L')
            axes[1, 0].set_ylabel('P&L (KRW)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 보유 종목 수
        axes[1, 1].plot(df_daily['date'], df_daily['positions_count'], 'purple', marker='o', linewidth=2)
        axes[1, 1].set_title('Number of Positions')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 날짜 축 포매팅
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df_daily)//10)))
        
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance chart saved as {save_file}")
    
    def print_summary(self):
        """요약 리포트 출력"""
        report = self.generate_performance_report()
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        summary = report['summary']
        performance = report['performance']
        
        print("=" * 60)
        print("📊 PAPER TRADING PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"📈 거래 활동:")
        print(f"   총 거래: {summary['total_trades']}건")
        print(f"   매수: {summary['buy_trades']}건, 매도: {summary['sell_trades']}건")
        print(f"   거래 일수: {summary['trading_days']}일")
        print(f"   일평균 거래: {summary['trades_per_day']:.1f}건")
        
        print(f"\n💰 투자 성과:")
        print(f"   총 손익: {performance['total_pnl']:+,.0f}원")
        print(f"   수익률: {performance['total_return_pct']:+.2f}%")
        print(f"   승률: {performance['win_rate']:.1%}")
        
        if performance['avg_win'] > 0 or performance['avg_loss'] < 0:
            print(f"   평균 수익: {performance['avg_win']:+,.0f}원")
            print(f"   평균 손실: {performance['avg_loss']:+,.0f}원")
            print(f"   최고 수익: {performance['best_trade']:+,.0f}원")
            print(f"   최대 손실: {performance['worst_trade']:+,.0f}원")
        
        # 전체 등급 매기기
        if performance['total_return_pct'] > 5:
            grade = "🌟 우수"
        elif performance['total_return_pct'] > 0:
            grade = "✅ 양호"
        elif performance['total_return_pct'] > -3:
            grade = "⚠️  보통"
        else:
            grade = "❌ 부진"
        
        print(f"\n🎯 종합 평가: {grade}")
        print("=" * 60)


def demo_dashboard():
    """대시보드 데모"""
    dashboard = PaperTradingDashboard()
    
    # 샘플 거래 데이터 생성 (실제로는 거래 시 자동 기록)
    sample_trades = [
        {
            'symbol': '005930',
            'action': 'BUY',
            'shares': 10,
            'price': 70000,
            'amount': 700000,
            'timestamp': '2024-08-01T09:30:00'
        },
        {
            'symbol': '005930',
            'action': 'SELL',
            'shares': 10,
            'price': 72000,
            'amount': 720000,
            'pnl': 20000,
            'pnl_pct': 2.86,
            'timestamp': '2024-08-02T14:15:00'
        },
        {
            'symbol': '000660',
            'action': 'BUY',
            'shares': 5,
            'price': 250000,
            'amount': 1250000,
            'timestamp': '2024-08-03T10:00:00'
        },
        {
            'symbol': '000660',
            'action': 'SELL',
            'shares': 5,
            'price': 245000,
            'amount': 1225000,
            'pnl': -25000,
            'pnl_pct': -2.0,
            'timestamp': '2024-08-05T15:30:00'
        }
    ]
    
    # 샘플 일일 포트폴리오 데이터
    sample_daily = [
        {
            'date': '2024-08-01',
            'total_value': 10000000,
            'cash': 9300000,
            'positions_value': 700000,
            'positions_count': 1,
            'daily_pnl': 0
        },
        {
            'date': '2024-08-02',
            'total_value': 10020000,
            'cash': 10020000,
            'positions_value': 0,
            'positions_count': 0,
            'daily_pnl': 20000
        },
        {
            'date': '2024-08-03',
            'total_value': 10020000,
            'cash': 8770000,
            'positions_value': 1250000,
            'positions_count': 1,
            'daily_pnl': 0
        },
        {
            'date': '2024-08-05',
            'total_value': 9995000,
            'cash': 9995000,
            'positions_value': 0,
            'positions_count': 0,
            'daily_pnl': -25000
        }
    ]
    
    # 데이터 추가
    dashboard.trades_history = sample_trades
    dashboard.daily_portfolio = sample_daily
    dashboard.save_data()
    
    # 리포트 출력
    dashboard.print_summary()
    
    # 차트 생성 (matplotlib 사용 가능한 환경에서)
    try:
        dashboard.plot_performance()
    except Exception as e:
        print(f"Chart generation failed: {e}")
        print("(This is normal in environments without display)")


if __name__ == "__main__":
    demo_dashboard()