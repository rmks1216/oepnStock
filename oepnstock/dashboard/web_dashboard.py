"""
Flask 기반 웹 대시보드
실시간 거래 상황 모니터링 및 시각화
"""
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import asyncio
from threading import Thread
import time

from .data_manager import DashboardDataManager

logger = logging.getLogger(__name__)


class WebDashboard:
    """웹 기반 대시보드"""
    
    def __init__(self, data_manager: DashboardDataManager, host: str = '0.0.0.0', port: int = 5000):
        self.data_manager = data_manager
        self.host = host
        self.port = port
        
        # Flask 앱 설정
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'oepnstock_dashboard_secret'
        
        # Socket.IO 설정
        self.socketio = SocketIO(self.app, 
                               cors_allowed_origins="*",
                               async_mode='threading')
        
        # 실시간 업데이트 플래그
        self.is_running = False
        self.update_thread = None
        
        # 라우트 설정
        self._setup_routes()
        self._setup_socketio_events()
        
        logger.info(f"Web dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 대시보드 페이지"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/overview')
        def get_overview():
            """대시보드 개요 데이터 API"""
            try:
                overview_data = {
                    'current_capital': self.data_manager.get_current_capital(),
                    'daily_return': self.data_manager.get_daily_return(),
                    'daily_pnl': self.data_manager.get_daily_pnl(),
                    'monthly_return': self.data_manager.get_monthly_return(),
                    'positions': self.data_manager.get_current_positions(),
                    'max_positions': self.data_manager.get_max_positions(),
                    'risk_level': self.data_manager.get_risk_level(),
                    'market_score': self.data_manager.get_market_score(),
                    'win_rate': self.data_manager.get_win_rate(),
                    'trade_count_today': self.data_manager.get_today_trade_count(),
                    'consecutive_losses': self.data_manager.get_consecutive_losses(),
                    'volatility': self.data_manager.get_current_volatility(),
                    'timestamp': datetime.now().isoformat()
                }
                return jsonify(overview_data)
            except Exception as e:
                logger.error(f"Failed to get overview data: {e}")
                return jsonify({'error': 'Failed to fetch data'}), 500
        
        @self.app.route('/api/chart/equity')
        def get_equity_chart():
            """자산 곡선 차트 데이터"""
            try:
                equity_data = self.data_manager.get_equity_curve()
                
                fig = go.Figure()
                
                # 자산가치 곡선
                fig.add_trace(go.Scatter(
                    x=equity_data.index,
                    y=equity_data.values,
                    mode='lines',
                    name='자산가치',
                    line=dict(color='#00C853', width=2),
                    hovertemplate='%{x}<br>자산: %{y:,.0f}원<extra></extra>'
                ))
                
                # 목표선 추가
                initial_capital = self.data_manager.get_initial_capital()
                monthly_target = initial_capital * 1.02
                
                fig.add_hline(
                    y=monthly_target,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="월간 목표 (2%)",
                    annotation_position="bottom right"
                )
                
                # 손익분기점
                fig.add_hline(
                    y=initial_capital,
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="손익분기점",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title='자산 곡선',
                    xaxis_title='날짜',
                    yaxis_title='자산 (원)',
                    hovermode='x unified',
                    showlegend=True,
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graphJSON
                
            except Exception as e:
                logger.error(f"Failed to generate equity chart: {e}")
                return jsonify({'error': 'Failed to generate chart'}), 500
        
        @self.app.route('/api/chart/daily-returns')
        def get_daily_returns_chart():
            """일일 수익률 차트"""
            try:
                daily_returns = self.data_manager.get_daily_returns()
                
                colors = ['green' if x > 0 else 'red' for x in daily_returns.values]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=daily_returns.index,
                    y=daily_returns.values * 100,  # 퍼센트로 변환
                    marker_color=colors,
                    name='일일 수익률',
                    hovertemplate='%{x}<br>수익률: %{y:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title='일일 수익률',
                    xaxis_title='날짜',
                    yaxis_title='수익률 (%)',
                    showlegend=False,
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graphJSON
                
            except Exception as e:
                logger.error(f"Failed to generate daily returns chart: {e}")
                return jsonify({'error': 'Failed to generate chart'}), 500
        
        @self.app.route('/api/chart/drawdown')
        def get_drawdown_chart():
            """드로다운 차트"""
            try:
                drawdown = self.data_manager.get_drawdown_series()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    mode='lines',
                    fill='tonexty',
                    name='드로다운',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)',
                    hovertemplate='%{x}<br>드로다운: %{y:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title='드로다운 분석',
                    xaxis_title='날짜',
                    yaxis_title='드로다운 (%)',
                    showlegend=False,
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graphJSON
                
            except Exception as e:
                logger.error(f"Failed to generate drawdown chart: {e}")
                return jsonify({'error': 'Failed to generate chart'}), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            """현재 포지션 조회"""
            try:
                positions = self.data_manager.get_position_details()
                return jsonify(positions)
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")
                return jsonify({'error': 'Failed to fetch positions'}), 500
        
        @self.app.route('/api/trades/recent')
        def get_recent_trades():
            """최근 거래 내역"""
            try:
                limit = request.args.get('limit', 20, type=int)
                trades = self.data_manager.get_recent_trades(limit)
                return jsonify(trades)
            except Exception as e:
                logger.error(f"Failed to get recent trades: {e}")
                return jsonify({'error': 'Failed to fetch trades'}), 500
        
        @self.app.route('/api/performance/summary')
        def get_performance_summary():
            """성과 요약"""
            try:
                summary = self.data_manager.get_performance_summary()
                return jsonify(summary)
            except Exception as e:
                logger.error(f"Failed to get performance summary: {e}")
                return jsonify({'error': 'Failed to fetch performance data'}), 500
        
        @self.app.route('/api/alerts/recent')
        def get_recent_alerts():
            """최근 알림"""
            try:
                limit = request.args.get('limit', 10, type=int)
                alerts = self.data_manager.get_recent_alerts(limit)
                return jsonify(alerts)
            except Exception as e:
                logger.error(f"Failed to get recent alerts: {e}")
                return jsonify({'error': 'Failed to fetch alerts'}), 500
        
        @self.app.route('/api/control/pause', methods=['POST'])
        def pause_trading():
            """거래 일시 중지"""
            try:
                duration = request.json.get('duration', 1) if request.json else 1
                success = self.data_manager.pause_trading(duration)
                
                if success:
                    return jsonify({'status': 'paused', 'duration': duration})
                else:
                    return jsonify({'error': 'Failed to pause trading'}), 500
                    
            except Exception as e:
                logger.error(f"Failed to pause trading: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/control/resume', methods=['POST'])
        def resume_trading():
            """거래 재개"""
            try:
                success = self.data_manager.resume_trading()
                
                if success:
                    return jsonify({'status': 'resumed'})
                else:
                    return jsonify({'error': 'Failed to resume trading'}), 500
                    
            except Exception as e:
                logger.error(f"Failed to resume trading: {e}")
                return jsonify({'error': 'Internal server error'}), 500
    
    def _setup_socketio_events(self):
        """Socket.IO 이벤트 설정"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """클라이언트 연결"""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'data': 'Connected to dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """클라이언트 연결 해제"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_live_data')
        def handle_live_data_request():
            """실시간 데이터 요청"""
            logger.info(f"Live data requested by: {request.sid}")
            self._send_live_update()
        
        @self.socketio.on('subscribe_alerts')
        def handle_alert_subscription():
            """알림 구독"""
            logger.info(f"Alert subscription requested by: {request.sid}")
            # 클라이언트를 알림 그룹에 추가 (구현 필요시)
    
    def _send_live_update(self):
        """실시간 업데이트 발송"""
        try:
            live_data = {
                'timestamp': datetime.now().isoformat(),
                'current_capital': self.data_manager.get_current_capital(),
                'daily_return': self.data_manager.get_daily_return(),
                'daily_pnl': self.data_manager.get_daily_pnl(),
                'positions': self.data_manager.get_current_positions(),
                'risk_level': self.data_manager.get_risk_level(),
                'market_score': self.data_manager.get_market_score()
            }
            self.socketio.emit('live_update', live_data)
        except Exception as e:
            logger.error(f"Failed to send live update: {e}")
    
    def send_alert_notification(self, alert_data: Dict[str, Any]):
        """알림 푸시"""
        try:
            self.socketio.emit('alert_notification', alert_data)
            logger.info(f"Alert notification sent: {alert_data.get('title', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    def _live_update_worker(self):
        """실시간 업데이트 워커"""
        while self.is_running:
            try:
                self._send_live_update()
                time.sleep(5)  # 5초마다 업데이트
            except Exception as e:
                logger.error(f"Live update worker error: {e}")
                time.sleep(10)  # 오류 시 10초 대기
    
    def start_live_updates(self):
        """실시간 업데이트 시작"""
        if not self.is_running:
            self.is_running = True
            self.update_thread = Thread(target=self._live_update_worker)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Live updates started")
    
    def stop_live_updates(self):
        """실시간 업데이트 중지"""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        logger.info("Live updates stopped")
    
    def run(self, debug: bool = False):
        """대시보드 서버 실행"""
        try:
            self.start_live_updates()
            
            logger.info(f"Starting web dashboard on {self.host}:{self.port}")
            self.socketio.run(
                self.app, 
                host=self.host, 
                port=self.port, 
                debug=debug,
                use_reloader=False  # 실시간 업데이트와 충돌 방지
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
        finally:
            self.stop_live_updates()
    
    def create_custom_chart(self, chart_type: str, data: Dict[str, Any]) -> str:
        """사용자 정의 차트 생성"""
        try:
            if chart_type == "portfolio_allocation":
                return self._create_portfolio_allocation_chart(data)
            elif chart_type == "monthly_performance":
                return self._create_monthly_performance_chart(data)
            elif chart_type == "risk_metrics":
                return self._create_risk_metrics_chart(data)
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")
                
        except Exception as e:
            logger.error(f"Failed to create custom chart: {e}")
            return json.dumps({'error': 'Failed to generate chart'})
    
    def _create_portfolio_allocation_chart(self, data: Dict[str, Any]) -> str:
        """포트폴리오 배분 차트"""
        positions = data.get('positions', [])
        
        if not positions:
            return json.dumps({'error': 'No position data available'})
        
        symbols = [pos['symbol'] for pos in positions]
        values = [pos['market_value'] for pos in positions]
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.4,
            hovertemplate='%{label}<br>%{value:,.0f}원<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='포트폴리오 배분',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_monthly_performance_chart(self, data: Dict[str, Any]) -> str:
        """월별 성과 차트"""
        monthly_returns = data.get('monthly_returns', {})
        
        if not monthly_returns:
            return json.dumps({'error': 'No monthly returns data available'})
        
        months = list(monthly_returns.keys())
        returns = list(monthly_returns.values())
        colors = ['green' if x > 0 else 'red' for x in returns]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=months,
            y=[r * 100 for r in returns],
            marker_color=colors,
            name='월간 수익률',
            hovertemplate='%{x}<br>수익률: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='월별 성과',
            xaxis_title='월',
            yaxis_title='수익률 (%)',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_risk_metrics_chart(self, data: Dict[str, Any]) -> str:
        """리스크 지표 차트"""
        metrics = data.get('risk_metrics', {})
        
        if not metrics:
            return json.dumps({'error': 'No risk metrics data available'})
        
        # 레이더 차트로 리스크 지표 표시
        categories = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 
                     'Win Rate', 'Profit Factor']
        values = [
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0), 
            metrics.get('calmar_ratio', 0),
            metrics.get('win_rate', 0) * 2,  # 0-2 범위로 스케일링
            metrics.get('profit_factor', 0) / 2  # 0-2 범위로 스케일링
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='현재 성과'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2]
                )
            ),
            title='리스크 지표 분석',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)