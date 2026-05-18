"""
台股 AI 決策系統 — Streamlit 互動儀表板
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock import (
    fetch_data, add_all_indicators, add_ma, add_rsi, add_macd,
    add_bollinger, add_atr, build_features, train_random_forest,
    fetch_institutional, fetch_margin, backtest,
)

st.set_page_config(page_title='台股 AI 決策系統', page_icon='📈', layout='wide')
st.title('📈 台股 AI 量化決策系統')


# ── 側邊欄 ──

with st.sidebar:
    ticker = st.text_input('股票代號', '2330.TW')
    period = st.selectbox('資料區間', ['1y', '2y', '3y', '5y', 'max'], index=2)
    run = st.button('🚀 執行分析', type='primary')


# ── 主流程 ──

if run or 'df' not in st.session_state:
    with st.spinner('下載資料中...'):
        df = fetch_data(ticker, period)
        if df.empty:
            st.error(f'無法取得 {ticker} 資料')
            st.stop()
        df = add_all_indicators(df)
        st.session_state.df = df

        with st.spinner('訓練 AI 模型中...'):
            ml_df = build_features(df)
            model, features, _, X_test, _, y_test = train_random_forest(ml_df)
            st.session_state.ml_df = ml_df
            st.session_state.model = model
            st.session_state.features = features

        with st.spinner('抓取籌碼資料...'):
            num = ticker.replace('.TW', '')
            st.session_state.inst_df = fetch_institutional(num)
            st.session_state.margin_df = fetch_margin(num)

        with st.spinner('執行回測...'):
            X_full = ml_df[[c for c in ml_df.columns if c not in
                            ['Target', 'open', 'high', 'low', 'close', 'volume',
                             'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240']]]
            bt_df = df.loc[ml_df.index].copy()
            bt_df['AI_Signal'] = model.predict(X_full)
            st.session_state.bt_result = backtest(bt_df)

        st.success('分析完成！')

df = st.session_state.get('df')
if df is None:
    st.info('請在左側輸入股票代號並點擊「執行分析」')
    st.stop()


# ── K 線圖 + 均線 ──

st.subheader('📊 K 線圖與均線')
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df['open'], high=df['high'],
    low=df['low'], close=df['close'], name='K線'
))
for ma, color, name in [('MA20', 'orange', '月線'), ('MA60', 'blue', '季線'),
                         ('MA120', 'green', '半年線'), ('MA240', 'red', '年線')]:
    if ma in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], line=dict(color=color, width=1), name=name))
fig.update_layout(template='plotly_white', xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)


# ── RSI + MACD ──

st.subheader('📉 RSI 與 MACD')
fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                      vertical_spacing=0.05)
fig2.add_trace(go.Scatter(x=df.index, y=df['close'], name='收盤價'), row=1, col=1)
fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI(14)', line=dict(color='purple')),
               row=2, col=1)
fig2.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
fig2.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
fig2.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color='gray'),
               row=3, col=1)
fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
               row=3, col=1)
fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')),
               row=3, col=1)
fig2.update_layout(template='plotly_white', height=600, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)


# ── 回測績效 ──

bt = st.session_state.get('bt_result')
if bt is not None:
    st.subheader('💰 回測績效')
    col1, col2, col3, col4 = st.columns(4)
    last = bt.iloc[-1]
    init = 1_000_000
    col1.metric('累積報酬', f'{(last["Cumulative_Strategy"] / init - 1):+.2%}')
    col2.metric('大盤報酬', f'{(last["Cumulative_Market"] / init - 1):+.2%}')
    sr = np.sqrt(252) * bt['Strategy_Return'].mean() / bt['Strategy_Return'].std()
    col3.metric('Sharpe 比', f'{sr:.2f}')
    dd = (bt['Cumulative_Strategy'] / bt['Cumulative_Strategy'].cummax() - 1).min()
    col4.metric('最大回撤', f'{dd:+.2%}')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=bt.index, y=bt['Cumulative_Market'], name='大盤 (買入持有)', line=dict(color='gray')))
    fig3.add_trace(go.Scatter(x=bt.index, y=bt['Cumulative_Strategy'], name='AI 策略', line=dict(color='blue')))
    fig3.update_layout(template='plotly_white', height=400, title='累積權益曲線')
    st.plotly_chart(fig3, use_container_width=True)

    # 布林通道
    st.subheader('🔵 布林通道')
    if 'Upper_Band' in df.columns:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index, y=df['close'], name='收盤價', line=dict(color='blue')))
        fig4.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], name='上軌', line=dict(color='red', dash='dash')))
        fig4.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], name='下軌', line=dict(color='green', dash='dash')))
        fig4.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='中軌(MA20)', line=dict(color='orange')))
        fig4.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig4, use_container_width=True)
