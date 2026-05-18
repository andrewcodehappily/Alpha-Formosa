"""
台股 AI 波段盲測系統 — Gradio 互動版
基於 test.py 的嚴格樣本外回測 + 趨勢濾網
"""
import warnings; warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd, numpy as np, gradio as gr
from datetime import datetime, timedelta
from test import run_analysis


def plot_backtest(bt_result, ticker):
    """繪製回測績效圖"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

    # 上圖：權益曲線
    ax = axes[0]
    ax.plot(bt_result.index, bt_result['Cumulative_Market'], color='gray', alpha=0.5,
            linewidth=1.5, label='大盤 (買入持有)')
    ax.plot(bt_result.index, bt_result['Cumulative_Strategy'], color='blue',
            linewidth=2, label='AI 策略')

    # 標註交易點
    trades = bt_result[bt_result['Traded']]
    longs = trades[trades['Position'] == 1]
    shorts = trades[trades['Position'] == -1]
    closes = trades[trades['Position'] == 0]
    if not longs.empty:
        ax.scatter(longs.index, longs['close'] * 1_000_000 / bt_result['close'].iloc[0] / 10,
                   marker='^', s=120, color='green', zorder=5, label='做多', alpha=0.8)
    if not shorts.empty:
        ax.scatter(shorts.index, shorts['close'] * 1_000_000 / bt_result['close'].iloc[0] / 10,
                   marker='v', s=120, color='red', zorder=5, label='放空', alpha=0.8)
    if not closes.empty:
        ax.scatter(closes.index, closes['close'] * 1_000_000 / bt_result['close'].iloc[0] / 10,
                   marker='o', s=60, color='gray', zorder=5, label='平倉', alpha=0.6)

    ax.axhline(y=1_000_000, color='black', linestyle=':', alpha=0.3)
    ax.set_title(f'{ticker} 盲測回測 — 累積權益曲線', fontsize=14, fontweight='bold')
    ax.set_ylabel('帳戶價值 (元)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # 中圖：每日報酬
    ax = axes[1]
    colors = ['red' if r < 0 else 'green' for r in bt_result['Strategy_Return']]
    ax.bar(bt_result.index, bt_result['Strategy_Return'], color=colors, alpha=0.6, width=1)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('日報酬')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1%}'))

    # 下圖：回撤
    ax = axes[2]
    cummax = bt_result['Cumulative_Strategy'].cummax()
    drawdown = bt_result['Cumulative_Strategy'] / cummax - 1
    ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax.set_ylabel('回撤')
    ax.set_xlabel('日期')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1%}'))

    fig.tight_layout()
    return fig


def plot_feature_importance(rf_model, feature_names, top_n=10):
    """繪製特徵重要性"""
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel('重要性')
    ax.set_title('Top 10 關鍵特徵', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    fig.tight_layout()
    return fig


def analyze(ticker, period, trend_filter):
    """Gradio 入口函數"""
    if ticker.isdigit():
        ticker = f'{ticker}.TW'

    try:
        results = run_analysis(
            ticker=ticker,
            period=period,
            with_xgb=False,
            with_lstm=False,
            trend_filter=trend_filter,
        )
    except Exception as e:
        return None, None, f'❌ 錯誤：{e}'

    df = results['df']
    bt = results['bt_result']
    rf_model = results['rf_model']

    # 排除特徵
    feat_names = list(rf_model.feature_names_in_)

    # 圖表
    fig1 = plot_backtest(bt, ticker)
    fig2 = plot_feature_importance(rf_model, feat_names)

    # 績效摘要
    init = 1_000_000
    total_r = bt['Cumulative_Strategy'].iloc[-1] / init - 1
    market_r = bt['Cumulative_Market'].iloc[-1] / init - 1
    sr = np.sqrt(252) * bt['Strategy_Return'].mean() / bt['Strategy_Return'].std()
    dd = (bt['Cumulative_Strategy'] / bt['Cumulative_Strategy'].cummax() - 1).min()
    trades = int(bt['Traded'].sum())

    # 交易勝率（含做多+放空）
    entry_price = None
    entry_type = None  # 'long' or 'short'
    trade_results = []
    for i in range(len(bt)):
        if not bt['Traded'].iloc[i]:
            continue
        pos = int(bt['Position'].iloc[i])
        if entry_price is None and pos != 0:
            entry_price = bt['close'].iloc[i]
            entry_type = 'long' if pos == 1 else 'short'
        elif entry_price is not None:
            exit_price = bt['close'].iloc[i]
            if entry_type == 'long':
                ret = exit_price / entry_price - 1 - 0.008
            else:
                ret = entry_price / exit_price - 1 - 0.008
            trade_results.append(ret)
            if pos == 0:
                entry_price, entry_type = None, None
            else:
                entry_price = exit_price
                entry_type = 'long' if pos == 1 else 'short'
    win_rate = np.mean([r > 0 for r in trade_results]) if trade_results else 0

    # 最近一筆訊號
    last_signal = bt['Signal_Category'].iloc[-1] if 'Signal_Category' in bt.columns else '無'
    last_price = float(df['close'].iloc[-1])
    last_date = str(df.index[-1].date())

    # 法人動態
    inst = results.get('inst_df', pd.DataFrame())
    inst_summary = ''
    if not inst.empty and 'name' in inst.columns:
        foreign = inst[inst['name'] == 'Foreign_Investor']
        if not foreign.empty:
            recent = foreign.sort_values('date').tail(5)
            diff_col = [c for c in recent.columns if 'diff' in c.lower() or 'net' in c.lower() or 'buy' in c.lower()]
            if diff_col:
                avg = recent[diff_col[0]].astype(float).mean()
                inst_summary = f'近5日外資平均買賣超: {avg:+.0f} 張'

    # 訊號分佈
    signal_dist = ''
    if 'Signal_Category' in bt.columns:
        signal_counts = bt['Signal_Category'].value_counts()
        signal_dist = ' | '.join(f'{k}: {v}' for k, v in signal_counts.items())

    market_note = '💰 策略打敗大盤！ 🎉' if total_r > market_r else '📉 策略落後大盤'
    trend_note = '✅ 啟用（多空雙向）' if trend_filter else '❌ 關閉'
    summary = (
        f'## 📊 {ticker} 盲測績效報告\n\n'
        f'**分析區間**: {df.index[0].date()} ~ {last_date}\n\n'
        f'### 績效指標\n\n'
        f'| 指標 | 策略 | 大盤 |\n'
        f'|------|------|------|\n'
        f'| 累積報酬 | **{total_r:.2%}** | {market_r:.2%} |\n'
        f'| Sharpe 比 | **{sr:.2f}** | — |\n'
        f'| 最大回撤 | {dd:.2%} | — |\n'
        f'| 交易次數 | {trades} 次 | — |\n'
        f'| 交易勝率 | **{win_rate:.1%}** | — |\n\n'
        f'{market_note}\n\n'
        f'### 最新狀態\n\n'
        f'- 最後交易日: **{last_date}**\n'
        f'- 收盤價: **{last_price:,.2f}**\n'
        f'- 目前訊號: **{last_signal}**\n'
        f'- 趨勢濾網: **{trend_note}**\n\n'
        + (f'- {inst_summary}\n\n' if inst_summary else '')
        + (f'### 訊號分佈\n{signal_dist}' if signal_dist else '')
    )

    return fig1, fig2, summary


# ── Gradio 介面 ──

with gr.Blocks(title='台股 AI 波段盲測系統', theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 台股 AI 波段盲測系統
    嚴格樣本外回測 + 隨機森林預測 5 日波段趨勢
    """)

    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(label='股票代號', value='2330.TW', placeholder='2330.TW 或 2330')
            period_input = gr.Dropdown(label='資料區間', choices=['1y', '2y', '3y', '5y'], value='3y')
            trend_input = gr.Checkbox(label='啟用趨勢濾網 (MA60>MA120 做多, MA60<MA120 放空)', value=True)
            run_btn = gr.Button('🚀 執行盲測', variant='primary', size='lg')

        with gr.Column(scale=2):
            summary_output = gr.Markdown()

    with gr.Row():
        with gr.Tab('📈 回測績效'):
            chart1 = gr.Plot(label='累積權益曲線')
        with gr.Tab('🌟 特徵重要性'):
            chart2 = gr.Plot(label='Top 10 關鍵特徵')

    run_btn.click(
        fn=analyze,
        inputs=[ticker_input, period_input, trend_input],
        outputs=[chart1, chart2, summary_output],
    )

    gr.Markdown("""
    ---
    ### 💡 使用說明
    - **模型**: RandomForest (max_depth=3, min_samples_leaf=10) — 經 30+ 配置盲測驗證最佳
    - **預測**: 5 日後波段漲跌
    - **策略**: 機率 ≥ 0.48 做多, ≤ 0.35 放空, 中間空手
    - **趨勢濾網**: MA60>MA120 才做多, MA60<MA120 才放空, 避免逆勢交易
    - **交易成本**: 單趟 0.4% (含手續費 + 滑價)
    - **訓練/測試**: 80/20 時間序列分割，嚴格樣本外盲測
    """)


if __name__ == '__main__':
    demo.launch(share=True)
