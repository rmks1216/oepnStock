"""
Performance Analysis Tool
실시간 거래 성과 분석 및 리포팅
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

class PerformanceAnalyzer:
    """거래 성과 분석기"""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = []
        
    def add_trade(self, trade: Dict[str, Any]):
        """거래 기록 추가"""
        trade['timestamp'] = datetime.now()
        self.trades.append(trade)
    
    def generate_report(self) -> Dict[str, Any]:
        """성과 리포트 생성"""
        if not self.trades:
            return {"message": "No trades to analyze"}
        
        df = pd.DataFrame(self.trades)
        
        # 기본 통계
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 수익률 통계
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        
        # 최대 드로우다운
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 샤프 비율 (일간 수익률 기준)
        daily_returns = df.groupby(df['timestamp'].dt.date)['pnl'].sum()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min()
        }
    
    def plot_performance(self):
        """성과 차트 생성"""
        if not self.trades:
            print("No trades to plot")
            return
        
        df = pd.DataFrame(self.trades)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 누적 수익률
        cumulative_pnl = df['pnl'].cumsum()
        axes[0,0].plot(cumulative_pnl.index, cumulative_pnl.values)
        axes[0,0].set_title('Cumulative P&L')
        axes[0,0].set_ylabel('P&L (KRW)')
        
        # 거래별 수익률
        axes[0,1].bar(range(len(df)), df['pnl'], color=['green' if x > 0 else 'red' for x in df['pnl']])
        axes[0,1].set_title('Trade P&L Distribution')
        axes[0,1].set_ylabel('P&L (KRW)')
        
        # 승률 추이
        win_loss = (df['pnl'] > 0).astype(int)
        rolling_win_rate = win_loss.rolling(window=10, min_periods=1).mean()
        axes[1,0].plot(rolling_win_rate.index, rolling_win_rate.values)
        axes[1,0].set_title('Rolling Win Rate (10 trades)')
        axes[1,0].set_ylabel('Win Rate')
        
        # 드로우다운
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        axes[1,1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[1,1].set_title('Drawdown (%)')
        axes[1,1].set_ylabel('Drawdown %')
        
        plt.tight_layout()
        plt.savefig('trading_performance.png')
        plt.show()

# 사용 예제
if __name__ == "__main__":
    # 성과 분석기 초기화
    analyzer = PerformanceAnalyzer()
    
    # 샘플 거래 데이터
    sample_trades = [
        {'symbol': '005930', 'pnl': 50000, 'pnl_pct': 0.02},
        {'symbol': '000660', 'pnl': -30000, 'pnl_pct': -0.015},
        {'symbol': '035420', 'pnl': 80000, 'pnl_pct': 0.035},
        {'symbol': '055550', 'pnl': -20000, 'pnl_pct': -0.01},
        {'symbol': '005380', 'pnl': 120000, 'pnl_pct': 0.04},
    ]
    
    for trade in sample_trades:
        analyzer.add_trade(trade)
    
    # 리포트 생성
    report = analyzer.generate_report()
    
    print("=== Trading Performance Report ===")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate']:.1%}")
    print(f"Total P&L: {report['total_pnl']:,.0f}원")
    print(f"Profit Factor: {report['profit_factor']:.2f}")
    print(f"Max Drawdown: {report['max_drawdown']:.1%}")
    print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")