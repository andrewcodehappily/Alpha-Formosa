"""
台股 AI 量化分析系統 — Gradio 互動版
整合技術指標、機器學習模型、未來預測
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import json, os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'Heiti TC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════
#  資料獲取與技術指標（完整版）
# ══════════════════════════════════════════════════

def fetch_all_data(ticker, start_date, end_date):
    """下載資料並計算全套技術指標"""
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"無法獲取 {ticker} 資料，請檢查代號或日期")

    # 扁平化欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    o, h, l, c, v = 'Open', 'High', 'Low', 'Close', 'Volume'

    # ── 均線 ──
    df['12_EMA'] = df[c].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df[c].ewm(span=26, adjust=False).mean()
    df['MA20'] = df[c].rolling(20).mean()
    df['MA60'] = df[c].rolling(60).mean()
    df['MA120'] = df[c].rolling(120).mean()

    # ── 黃金/死亡交叉 ──
    df['Golden_Cross'] = (df['12_EMA'] > df['26_EMA']) & (df['12_EMA'].shift(1) <= df['26_EMA'].shift(1))
    df['Death_Cross']  = (df['12_EMA'] < df['26_EMA']) & (df['12_EMA'].shift(1) >= df['26_EMA'].shift(1))

    # ── KD 隨機指標 ──
    low14 = df[l].rolling(14).min()
    high14 = df[h].rolling(14).max()
    df['%K'] = 100 * (df[c] - low14) / (high14 - low14)
    df['%D'] = df['%K'].rolling(3).mean()

    # ── RSI ──
    delta = df[c].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ── MACD ──
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ── 布林通道 ──
    df['BB_Middle'] = df[c].rolling(20).mean()
    df['BB_Std'] = df[c].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # ── ATR ──
    tr = pd.concat([
        df[h] - df[l],
        (df[h] - df[c].shift()).abs(),
        (df[l] - df[c].shift()).abs(),
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # ── ADX ──
    df['+DM'] = np.where(df[h].diff() > df[l].diff(), df[h].diff(), 0).clip(0)
    df['-DM'] = np.where(df[l].diff() > df[h].diff(), -df[l].diff(), 0).clip(0)
    df['+DI'] = 100 * (df['+DM'] / df['ATR']).rolling(14).mean()
    df['-DI'] = 100 * (df['-DM'] / df['ATR']).rolling(14).mean()
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10))
    df['ADX'] = df['DX'].rolling(14).mean()

    # ── 動量 ──
    df['Momentum'] = df[c] - df[c].shift(14)

    # ── 威廉指標 ──
    high14 = df[h].rolling(14).max()
    low14 = df[l].rolling(14).min()
    df['WMSR'] = ((high14 - df[c]) / (high14 - low14 + 1e-10)) * -100

    # ── CCI ──
    tp = (df[h] + df[l] + df[c]) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)

    # ── Chaikin 資金流 ──
    mfm = ((df[c] - df[l]) - (df[h] - df[c])) / (df[h] - df[l] + 1e-10)
    df['Chaikin'] = (mfm * df[v]).rolling(3).sum()

    # ── OBV 能量潮 ──
    df['OBV'] = (np.where(df[c].diff() > 0, df[v],
                          np.where(df[c].diff() < 0, -df[v], 0))).cumsum()

    return df


# ══════════════════════════════════════════════════
#  AI 模型訓練與預測
# ══════════════════════════════════════════════════

def train_model(df):
    """訓練 RandomForest 回歸模型預測收盤價"""
    feature_cols = [
        '12_EMA', '26_EMA', 'RSI', 'MACD', 'ATR', 'Momentum',
        'BB_Upper', 'BB_Lower', 'WMSR', 'CCI', '%K', '%D',
        'Chaikin', 'ADX', 'OBV', '+DI', '-DI',
    ]

    clean = df.dropna(subset=feature_cols + ['Close'])
    if len(clean) < 30:
        raise ValueError(f"有效數據不足 ({len(clean)} 筆)，至少需要 30 筆")

    X = clean[feature_cols]
    y = clean['Close']

    model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X, y)

    clean = clean.copy()
    clean['Predicted'] = model.predict(X)

    return model, clean, feature_cols


def predict_future(model, df, feature_cols, days=5):
    """預測未來 N 天價格與漲跌方向（含 漲/跌/平 分類）"""
    recent = df['Close'].iloc[-20:]
    daily_ret = recent.pct_change().dropna().mean()
    last_price = float(df['Close'].iloc[-1])

    # 先用最近平均日回報推估 baseline
    future_prices = []
    price = last_price
    for _ in range(days):
        price *= (1 + daily_ret)
        future_prices.append(price)

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=days
    )

    # 用模型判斷趨勢方向 bias
    last_features = df[feature_cols].iloc[-1:].values
    model_pred = float(model.predict(last_features)[0])
    bias = 1.0 if model_pred > last_price else -1.0

    # 混合 bias 得到最終預測
    adjusted = []
    for i, p in enumerate(future_prices):
        trend = p * (1 + bias * 0.005 * (i + 1))
        adjusted.append(trend)

    # 漲跌平分類（閾值：±0.5%）
    directions = []
    threshold = 0.005
    for p in adjusted:
        change = p / last_price - 1
        if change > threshold:
            directions.append('📈 漲')
        elif change < -threshold:
            directions.append('📉 跌')
        else:
            directions.append('➖ 平')

    return future_dates, adjusted, directions


# ══════════════════════════════════════════════════
#  繪圖函數（三張圖）
# ══════════════════════════════════════════════════

def plot_analysis(df, ticker):
    """圖 1：技術分析（K線 + 均線 + 黃金/死亡交叉）"""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['Close'], color='blue', alpha=0.5, label='收盤價', linewidth=1.5)
    ax.plot(df.index, df['12_EMA'], color='red', alpha=0.7, label='12 EMA', linewidth=1)
    ax.plot(df.index, df['26_EMA'], color='green', alpha=0.7, label='26 EMA', linewidth=1)
    ax.plot(df.index, df['MA20'], color='orange', alpha=0.6, label='MA20', linewidth=1, linestyle='--')
    ax.plot(df.index, df['MA60'], color='purple', alpha=0.6, label='MA60', linewidth=1, linestyle='--')

    # 黃金 / 死亡交叉標記
    gc = df[df['Golden_Cross']]
    dc = df[df['Death_Cross']]
    if not gc.empty:
        ax.scatter(gc.index, gc['12_EMA'], marker='^', s=180, color='lime',
                   edgecolors='green', linewidth=2, zorder=5, label='黃金交叉')
    if not dc.empty:
        ax.scatter(dc.index, dc['12_EMA'], marker='v', s=180, color='red',
                   edgecolors='darkred', linewidth=2, zorder=5, label='死亡交叉')

    ax.set_title(f'{ticker} 技術分析圖（含黃金/死亡交叉）', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('價格')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_prediction(df, ticker):
    """圖 2：實際價格 vs 模型預測"""
    if 'Predicted' not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], color='blue', label='實際價格', linewidth=1.5)
    ax.plot(df.index, df['Predicted'], color='red', linestyle='--', label='RandomForest 預測', linewidth=1)
    ax.fill_between(df.index, df['Close'], df['Predicted'], alpha=0.1, color='gray')
    ax.set_title(f'{ticker} 實際價格 vs RandomForest 模型預測', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('價格')
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_future(df, future_dates, future_prices, ticker, days=5):
    """圖 3：未來 N 天預測（綠漲 / 紅跌 / 灰平 彩色分段）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    last_price = float(df['Close'].iloc[-1])
    recent = df.iloc[-60:]

    # 歷史價格
    ax.plot(recent.index, recent['Close'], color='#3498db', label='歷史價格', linewidth=1.5)

    # 逐段上色：綠漲 / 紅跌 / 灰平
    all_dates = [recent.index[-1]] + list(future_dates)
    all_prices = [last_price] + list(future_prices)

    for i in range(len(future_prices)):
        change_pct = (future_prices[i] / last_price - 1) * 100
        if change_pct > 0.5:
            color = '#2ecc71'
        elif change_pct < -0.5:
            color = '#e74c3c'
        else:
            color = '#95a5a6'

        ax.plot(all_dates[i:i+2], all_prices[i:i+2], color=color, linewidth=3, marker='o', markersize=8)

        # 4 方位輪換標註（↗↙↖↘）避免文字重疊
        positions = [
            (14, 14, 'left', 'bottom'),    # 右上
            (-14, -14, 'right', 'top'),    # 左下
            (-14, 14, 'right', 'bottom'),  # 左上
            (14, -14, 'left', 'top'),      # 右下
        ]
        x_off, y_off, ha, va = positions[i % 4]
        ax.annotate(f'{future_prices[i]:.1f}',
                    (future_dates[i], future_prices[i]),
                    textcoords="offset points", xytext=(x_off, y_off),
                    ha=ha, va=va, fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7))

    # 基準線
    ax.axhline(y=last_price, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'{ticker} 未來 {days} 天預測（綠漲 紅跌 灰平）', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('價格')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='漲 (>0.5%)'),
        Patch(facecolor='#e74c3c', label='跌 (<-0.5%)'),
        Patch(facecolor='#95a5a6', label='平'),
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════
#  主力分析函數（Gradio 入口）
# ══════════════════════════════════════════════════

def analyze(ticker, start_date, end_date, predict_days):
    predict_days = int(predict_days)

    # 自動補 .TW
    if ticker.isdigit():
        ticker = f"{ticker}.TW"

    try:
        df = fetch_all_data(ticker, start_date, end_date)
    except ValueError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"錯誤：{e}"

    if df.empty:
        return None, None, None, "無法取得資料"

    # 訓練模型
    try:
        model, df_pred, features = train_model(df)
    except ValueError as e:
        return None, None, None, str(e)

    # 未來預測
    future_dates, future_prices, directions = predict_future(model, df_pred, features, days=predict_days)

    # 繪圖
    fig1 = plot_analysis(df_pred, ticker)
    fig2 = plot_prediction(df_pred, ticker)
    fig3 = plot_future(df_pred, future_dates, future_prices, ticker, days=predict_days)

    # 預測摘要 — 每日漲跌平
    last_price = float(df['Close'].iloc[-1])

    # 整體走向統計
    dir_count = {'📈 漲': 0, '📉 跌': 0, '➖ 平': 0}
    for d in directions:
        dir_count[d] = dir_count.get(d, 0) + 1
    max_dir = max(dir_count, key=dir_count.get)
    trend_summary = f'📈 {dir_count["📈 漲"]}漲 / 📉 {dir_count["📉 跌"]}跌 / ➖ {dir_count["➖ 平"]}平'

    # 每日明細
    day_lines = []
    for i, (d, p, dr) in enumerate(zip(future_dates, future_prices, directions)):
        change_pct = (p / last_price - 1) * 100
        day_lines.append(f'  {dr} 第{i+1}天({d.month}/{d.day}): {p:.2f} ({change_pct:+.2f}%)')
    day_str = '\n'.join(day_lines)

    # 近期信號
    last_row = df_pred.iloc[-1]
    signals = []
    rsi = last_row.get('RSI', 50)
    if rsi > 70: signals.append("RSI 超買⚠️")
    elif rsi < 30: signals.append("RSI 超賣💡")
    if last_row.get('Golden_Cross', False): signals.append("黃金交叉🔥")
    if last_row.get('Death_Cross', False): signals.append("死亡交叉💀")
    k = last_row.get('%K', 50); d = last_row.get('%D', 50)
    if k is not None and d is not None:
        if k > d and k < 30: signals.append("KD 低檔黃金交叉📈")
    if not signals:
        signals.append("無明顯信號➖")

    summary = (
        f"【{ticker} 未來{predict_days}天預測】\n"
        f"{'═' * 40}\n"
        f"📅 最後收盤 ({df.index[-1].date()}): {last_price:.2f}\n\n"
        f"🔮 整體走向: {max_dir}（{trend_summary}）\n"
        f"{'─' * 30}\n"
        f"{day_str}\n"
        f"{'─' * 30}\n"
        f"📊 模型特徵: {len(features)} 維 | 漲跌平閾值: ±0.5%\n"
        f"🔍 近期信號: {'、'.join(signals)}"
    )

    return fig1, fig2, fig3, summary


# ══════════════════════════════════════════════════
#  Gradio 介面
# ══════════════════════════════════════════════════

import gradio as gr

with gr.Blocks(title="台股 AI 量化分析系統", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 台股 AI 量化分析系統
    輸入股票代號與日期範圍，一鍵獲得技術分析、AI 預測與未來走勢
    """)

    with gr.Row():
        ticker_input = gr.Textbox(label="股票代號", placeholder="2330.TW 或 AAPL", value="2330.TW", scale=1)
        start_input = gr.Textbox(label="開始日期", value=(datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'), scale=1)
        end_input = gr.Textbox(label="結束日期", value=datetime.today().strftime('%Y-%m-%d'), scale=1)
        days_input = gr.Dropdown(label="預測天數", choices=[3, 5, 7, 10, 14, 21], value=5, scale=1)

    with gr.Row():
        run_btn = gr.Button("🚀 執行分析", variant="primary", size="lg", scale=1)

    with gr.Row():
        summary_output = gr.Textbox(label="📝 預測摘要（含每日漲跌平）", lines=12)

    with gr.Row():
        with gr.Tab("📊 技術分析"):
            chart1 = gr.Plot(label="技術分析圖（含黃金/死亡交叉）")
        with gr.Tab("📈 AI 模型預測"):
            chart2 = gr.Plot(label="實際價格 vs RandomForest 預測")
        with gr.Tab("🔮 未來預測"):
            chart3 = gr.Plot(label="未來價格預測")

    run_btn.click(
        fn=analyze,
        inputs=[ticker_input, start_input, end_input, days_input],
        outputs=[chart1, chart2, chart3, summary_output],
    )

    gr.Markdown("""
    ---
    ### 💡 使用說明
    - **台股**：輸入數字代號（如 `2330`）或完整代號（如 `2330.TW`）
    - **美股**：直接輸入代號（如 `AAPL`, `TSLA`）
    - **預測方法**：RandomForest 回歸模型 + 趨勢推估
    - **技術指標**：KD、RSI、MACD、布林通道、ADX、CCI、OBV、威廉指標、Chaikin 資金流
    """)


if __name__ == '__main__':
    demo.launch(share=True)
