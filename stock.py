"""
台股 AI 量化分析系統
整合自 stock.ipynb 核心功能，提供 CLI 一鍵分析
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ── 資料獲取與清洗 ─────────────────────────────────────────────

def fetch_data(ticker='2330.TW', period='3y'):
    """下載股價資料並清洗欄位"""
    df = yf.download(ticker, period=period)
    df = fix_columns(df)
    df.columns = [c.lower() for c in df.columns]
    return df


def fix_columns(df):
    """處理 yfinance MultiIndex 欄位"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]
    return df


def fetch_institutional(ticker='2330', years=1):
    """從 FinMind 抓三大法人買賣超"""
    start = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": ticker,
        "start_date": start,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except Exception:
        return pd.DataFrame()


def fetch_margin(ticker='2330', years=1):
    """從 FinMind 抓融資融券資料"""
    start = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": ticker,
        "start_date": start,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except Exception:
        return pd.DataFrame()


# ── 技術指標計算 ─────────────────────────────────────────────

def add_ma(df, windows=[20, 60, 120, 240]):
    """加入移動平均線"""
    for w in windows:
        df[f'MA{w}'] = df['close'].rolling(window=w).mean()
    return df


def add_rsi(df, period=14):
    """計算 RSI 指標"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    """計算 MACD 指標"""
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def add_bollinger(df, period=20, std=2):
    """計算布林通道"""
    df['MA20'] = df['close'].rolling(period).mean()
    rstd = df['close'].rolling(period).std()
    df['Upper_Band'] = df['MA20'] + rstd * std
    df['Lower_Band'] = df['MA20'] - rstd * std
    return df


def add_atr(df, period=14):
    """計算 ATR 真實波動幅度"""
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    return df

def add_ema_cross(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['Golden_Cross'] = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1))
    df['Death_Cross'] = (ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))
    df['EMA_Ratio'] = ema12 / ema26
    return df

def add_stochastic(df, period=14):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['PctK'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['PctD'] = df['PctK'].rolling(window=3).mean()
    return df

def add_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    df['Plus_DM'] = np.where(high.diff() > low.diff(), high.diff(), 0).clip(min=0)
    df['Minus_DM'] = np.where(low.diff() > high.diff(), -low.diff(), 0).clip(min=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(period).mean() / atr)
    df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(period).mean() / atr)
    df['ADX'] = 100 * (abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])).rolling(period).mean()
    return df

def add_momentum(df, period=14):
    df['Momentum_14d'] = df['close'] - df['close'].shift(period)
    return df

def add_williams_r(df, period=14):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    df['WMSR'] = ((high_max - df['close']) / (high_max - low_min)) * -100
    return df

def add_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).std())
    return df

def add_chaikin(df):
    df['Chaikin'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['Chaikin'] = df['Chaikin'].rolling(window=3).sum()
    return df

def add_obv(df):
    condition = df['close'].diff() > 0
    df['OBV'] = (np.where(condition, df['volume'],
                          np.where(df['close'].diff() < 0, -df['volume'], 0))).cumsum()
    return df

def add_dema_tema(df, period=20):
    ema1 = df['close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    df['DEMA'] = 2 * ema1 - ema2
    df['TEMA'] = 3 * ema1 - 3 * ema2 + ema3
    return df

def add_hma(df, period=16):
    def _wma(series, n):
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    w1 = _wma(df['close'], half)
    w2 = _wma(df['close'], period)
    raw_hma = 2 * w1 - w2
    df['HMA'] = _wma(raw_hma, sqrt_n)
    return df

def add_supertrend(df, period=10, multiplier=3):
    med_price = (df['high'] + df['low']) / 2
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    basic_upper = med_price + multiplier * atr
    basic_lower = med_price - multiplier * atr
    final_upper = basic_upper.values.copy()
    final_lower = basic_lower.values.copy()
    direction = np.ones(len(df))
    supertrend = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper.iloc[i]
            final_lower[i] = basic_lower.iloc[i]
            continue
        final_upper[i] = basic_upper.iloc[i] if (basic_upper.iloc[i] < final_upper[i-1] or df['close'].iloc[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = basic_lower.iloc[i] if (basic_lower.iloc[i] > final_lower[i-1] or df['close'].iloc[i-1] < final_lower[i-1]) else final_lower[i-1]
    for i in range(1, len(df)):
        if np.isnan(final_lower[i - 1]):
            direction[i] = 1
        elif df['close'].iloc[i] > final_lower[i-1]:
            direction[i] = 1
        elif df['close'].iloc[i] < final_upper[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
    df['Supertrend'] = supertrend
    df['ST_Direction'] = direction
    return df

def add_donchian(df, period=20):
    df['Donchian_Upper'] = df['high'].rolling(period).max()
    df['Donchian_Lower'] = df['low'].rolling(period).min()
    df['Donchian_Mid'] = (df['Donchian_Upper'] + df['Donchian_Lower']) / 2
    return df

def add_keltner(df, ema_period=20, atr_period=10, multiplier=2):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    df['Keltner_Mid'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['Keltner_Upper'] = df['Keltner_Mid'] + multiplier * atr
    df['Keltner_Lower'] = df['Keltner_Mid'] - multiplier * atr
    return df

def add_zigzag(df, pct=5):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    zz = np.zeros(n)
    direction = 0
    pivot_idx = 0
    pivot_price = close[0]
    for i in range(1, n):
        if direction >= 0 and close[i] > pivot_price:
            pivot_price = close[i]
            pivot_idx = i
        elif direction <= 0 and close[i] < pivot_price:
            pivot_price = close[i]
            pivot_idx = i
        elif direction >= 0 and close[i] < pivot_price * (1 - pct / 100):
            zz[pivot_idx] = pivot_price
            pivot_price = close[i]
            pivot_idx = i
            direction = -1
        elif direction <= 0 and close[i] > pivot_price * (1 + pct / 100):
            zz[pivot_idx] = pivot_price
            pivot_price = close[i]
            pivot_idx = i
            direction = 1
    zz[pivot_idx] = pivot_price
    df['ZigZag'] = np.where(zz > 0, zz, np.nan)
    return df

def add_all_indicators(df):
    df = add_ma(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_ema_cross(df)
    df = add_stochastic(df)
    df = add_adx(df)
    df = add_momentum(df)
    df = add_williams_r(df)
    df = add_cci(df)
    df = add_chaikin(df)
    df = add_obv(df)
    df = add_dema_tema(df)
    df = add_hma(df)
    df = add_supertrend(df)
    df = add_donchian(df)
    df = add_keltner(df)
    return df


# ── 特徵工程 ─────────────────────────────────────────────

def build_features(df, label_shifts=5):
    """建立 ML 特徵與標籤（預測 5 日後波段方向）"""
    ml = df.copy()

    ml['Return_1d'] = ml['close'].pct_change(1)
    ml['Return_5d'] = ml['close'].pct_change(5)
    ml['Return_20d'] = ml['close'].pct_change(20)
    ml['Bias_MA20'] = (ml['close'] - ml['MA20']) / ml['MA20'] * 100
    ml['Bias_MA60'] = (ml['close'] - ml['MA60']) / ml['MA60'] * 100
    ml['Volume_Ratio'] = ml['volume'] / ml['volume'].rolling(20).mean()
    ml['High_Low_Ratio'] = (ml['high'] - ml['low']) / ml['close']
    ml['Close_Open_Ratio'] = (ml['close'] - ml['open']) / ml['open']

    ml['RSI_Change'] = ml['RSI'].diff(3)
    ml['RSI_High'] = ml['RSI'] > 70
    ml['RSI_Low'] = ml['RSI'] < 30

    ml['MACD_Cross'] = ml['MACD'] - ml['MACD_Signal']
    ml['MACD_Positive'] = ml['MACD_Cross'] > 0

    ml['ATR_Ratio'] = ml['ATR'] / ml['close']
    ml['BB_Width'] = (ml['Upper_Band'] - ml['Lower_Band']) / ml['close']
    ml['BB_Position'] = (ml['close'] - ml['Lower_Band']) / (ml['Upper_Band'] - ml['Lower_Band'])

    ml['MA20_above_MA60'] = ml['MA20'] > ml['MA60']
    ml['MA60_above_MA120'] = ml['MA60'] > ml['MA120']
    ml['MA20_above_MA120'] = ml['MA20'] > ml['MA120']
    ml['Momentum_14d'] = ml['close'] - ml['close'].shift(14)

    ml['DEMA_Ratio'] = ml['close'] / ml['DEMA']
    ml['TEMA_Ratio'] = ml['close'] / ml['TEMA']
    ml['HMA_Slope'] = ml['HMA'] - ml['HMA'].shift(3)
    ml['ST_Bullish'] = ml['ST_Direction'] == 1
    ml['Donchian_Position'] = (ml['close'] - ml['Donchian_Lower']) / (ml['Donchian_Upper'] - ml['Donchian_Lower'])
    ml['Donchian_Breakout'] = ml['close'] > ml['Donchian_Upper']
    ml['Keltner_Position'] = (ml['close'] - ml['Keltner_Lower']) / (ml['Keltner_Upper'] - ml['Keltner_Lower'])

    ml['Target'] = (ml['close'].shift(-label_shifts) > ml['close']).astype(int)
    ml.dropna(inplace=True)
    return ml


# ── 隨機森林模型 ─────────────────────────────────────────────

def train_random_forest(ml_df, test_size=0.2, n_estimators=200):
    """訓練 RandomForest 漲跌預測模型"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
               'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
               'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
               'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
               'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
    features = [c for c in ml_df.columns if c not in exclude]

    X = ml_df[features]
    y = ml_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=3,
        min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'  RF 訓練準確率: {train_acc:.2%}')
    print(f'  RF 測試準確率: {test_acc:.2%}')

    return model, features, X_train, X_test, y_train, y_test


# ── XGBoost 模型 ─────────────────────────────────────────────

def train_xgboost(ml_df, test_size=0.2):
    """訓練 XGBoost 模型（含超參數搜尋）"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import xgboost as xgb

    exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
               'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
               'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
               'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
               'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
    features = [c for c in ml_df.columns if c not in exclude]

    X = ml_df[features]
    y = ml_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'  XGB 訓練準確率: {train_acc:.2%}')
    print(f'  XGB 測試準確率: {test_acc:.2%}')

    return model, features, X_train, X_test, y_train, y_test


# ── LSTM 模型 ─────────────────────────────────────────────

def train_lstm(df, seq_len=20, epochs=30, lr=0.001):
    """訓練 LSTM 深度學習模型（PyTorch）"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  使用裝置: {device}')

    # 資料準備
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i - seq_len:i])
        y.append(data_scaled[i, 3])  # close
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    # 模型定義
    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 訓練
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # 評估
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        test_pred = model(X_test_t)
        train_loss = criterion(train_pred, y_train_t).item()
        test_loss = criterion(test_pred, y_test_t).item()
    print(f'  LSTM 訓練損失: {train_loss:.6f}')
    print(f'  LSTM 測試損失: {test_loss:.6f}')

    return model, scaler


# ── 回測框架 ─────────────────────────────────────────────

def backtest(backtest_df, initial_capital=1_000_000):
    BUY_RATE = 0.001425; SELL_RATE = 0.004425
    df = backtest_df.copy()
    if 'Position' not in df.columns: df['Position'] = 0.0

    df['Daily_Return'] = df['close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    pos_change = df['Position'].diff().fillna(0)
    df['Strategy_Return'] -= pos_change.clip(lower=0) * BUY_RATE
    df['Strategy_Return'] -= (-pos_change).clip(lower=0) * SELL_RATE

    df['Cumulative_Market'] = (1 + df['Daily_Return']).cumprod() * initial_capital
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod() * initial_capital

    print('\n  📜 交易明細')
    print('  ' + '-' * 80)
    entry_price, entry_date, entry_type, entry_pos = 0.0, None, None, 0.0
    trade_count = win_trades = 0
    last_idx = len(df) - 1

    for idx, (date, row) in enumerate(df.iterrows()):
        pos = row['Position']
        prev_pos = df['Position'].iloc[idx - 1] if idx > 0 else 0.0
        is_last = (idx == last_idx)

        if entry_type is None and abs(pos) > 0.05 and abs(prev_pos) < 0.05:
            entry_price, entry_date = row['close'], date
            entry_type = 'long' if pos > 0 else 'short'
            action = '做多' if pos > 0 else '放空'
            print(f"  {'🟢' if pos > 0 else '🔴'} [{action}] {date.date()}  價格: {entry_price:>8.2f}  倉位: {abs(pos):.0%}")

        elif entry_type is not None and ((abs(pos) < 0.05 and abs(prev_pos) >= 0.05) or is_last):
            exit_price = row['close']
            if entry_type == 'long':
                trade_ret = (exit_price / entry_price - 1) - (BUY_RATE + SELL_RATE)
            else:
                trade_ret = (entry_price / exit_price - 1) - (BUY_RATE + SELL_RATE)
            trade_count += 1
            if trade_ret > 0: win_trades += 1
            flag = '✅' if trade_ret > 0 else '❌'
            action = '賣出' if entry_type == 'long' else '回補'
            suffix = ' ⏹ 強制結算' if is_last else ''
            print(f"  {'🔵' if entry_type == 'long' else '🟣'} [{action}] {date.date()}  價格: {exit_price:>8.2f}  淨報酬: {trade_ret:>+7.2%}  {flag}{suffix}")
            entry_date, entry_type = None, None

        elif entry_type is not None and abs(pos - prev_pos) > 0.15:
            action = '加碼' if pos > prev_pos else '減碼'
            print(f"  {'⬆' if pos > prev_pos else '⬇'} [{action}] {date.date()}  價格: {row['close']:>8.2f}  倉位: {abs(pos):.0%}")

    print('  ' + '-' * 80)

    total_return = df['Cumulative_Strategy'].iloc[-1] / initial_capital - 1
    market_return = df['Cumulative_Market'].iloc[-1] / initial_capital - 1
    std = df['Strategy_Return'].std()
    sharpe = np.sqrt(252) * df['Strategy_Return'].mean() / std if std != 0 else 0
    max_dd = (df['Cumulative_Strategy'] / df['Cumulative_Strategy'].cummax() - 1).min()
    win_rate = win_trades / trade_count if trade_count > 0 else 0

    print(f'\n  📊 回測績效（{initial_capital:,} 元，含真實交易成本）')
    print(f'  {'=' * 42}')
    print(f'  累積報酬:   {total_return:>+7.2%}')
    print(f'  大盤報酬:   {market_return:>+7.2%}')
    print(f'  Sharpe 比:  {sharpe:>7.2f}')
    print(f'  最大回撤:   {max_dd:>+7.2%}')
    print(f'  交易次數:   {trade_count} 趟')
    print(f'  交易勝率:   {win_rate:>7.2%}')

    return df

# ── 一鍵分析 ─────────────────────────────────────────────

def run_analysis(ticker='2330.TW', period='3y', with_xgb=True, with_lstm=False, trend_filter=True):
    """完整分析流程：資料 → 指標 → 模型 → 回測"""
    import platform
    print(f'╔{"═" * 45}╗')
    print(f'║  台股 AI 量化分析系統')
    print(f'║  標的: {ticker}  區間: {period}')
    print(f'╚{"═" * 45}╝')
    print()

    # 1. 下載資料
    print('📥 [1/5] 下載股價資料...')
    df = fetch_data(ticker, period)
    print(f'  共 {len(df)} 筆資料 ({df.index[0].date()} ~ {df.index[-1].date()})')

    # 2. 技術指標
    print('📊 [2/5] 計算技術指標...')
    df = add_all_indicators(df)
    print(f'  均線: MA20/MA60/MA120/MA240')
    print(f'  RSI(14) | MACD | 布林通道 | ATR(14)')

    # 3. 法人籌碼
    print('🏦 [3/5] 抓取三大法人資料...')
    ticker_num = ticker.replace('.TW', '')
    inst_df = fetch_institutional(ticker_num)
    margin_df = fetch_margin(ticker_num)
    if not inst_df.empty:
        print(f'  法人資料 {len(inst_df)} 筆 | 融資融券 {len(margin_df)} 筆')
    else:
        print('  ⚠️ 法人資料抓取失敗（FinMind 可能無資料）')

    # 4. ML 模型
    print('🤖 [4/5] 訓練 AI 模型...')
    ml_df = build_features(df)
    print(f'  特徵維度: {ml_df.shape[1] - 8} 維, 樣本數: {len(ml_df)}')

    # RandomForest
    print(f'  ── Random Forest ──')
    rf_model, rf_features, _, X_test, _, y_test = train_random_forest(ml_df)

    # XGBoost
    if with_xgb:
        print(f'  ── XGBoost ──')
        xgb_model, xgb_features, _, _, _, _ = train_xgboost(ml_df)

    # LSTM
    if with_lstm:
        print(f'  ── LSTM ──')
        lstm_model, scaler = train_lstm(df)

    print('📈 [5/5] 樣本外回測...')
    test_indices = X_test.index
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    backtest_df = df.loc[test_indices].copy()

    # 信心度分級
    conditions = [
        rf_probs >= 0.70,
        rf_probs >= 0.55,
        rf_probs >= 0.48,
        rf_probs >= 0.30,
        rf_probs >= 0.20,
        rf_probs >= 0.10,
        rf_probs <  0.10,
    ]
    categories = ['強力買進', '買進', '偏多', '中立', '偏空', '賣出', '強力賣出']
    backtest_df['Signal_Category'] = np.select(conditions, categories, default='中立')

    min_hold = 10
    raw_long = rf_probs >= 0.48
    raw_short = rf_probs <= 0.35
    short_tech = (
        (backtest_df['close'] < backtest_df['MA20']) &
        (backtest_df['RSI'] < 45) &
        (backtest_df['MACD'] < backtest_df['MACD_Signal'])
    )

    hold = 0
    position = np.zeros(len(backtest_df), dtype=float)
    for i in range(len(backtest_df)):
        if hold > 0:
            hold -= 1
            position[i] = position[i-1]
        elif raw_long[i]:
            position[i] = rf_probs[i]
            hold = min_hold
        elif raw_short[i] and short_tech[i]:
            position[i] = -rf_probs[i]
            hold = min_hold

    backtest_df['Position'] = position

    if trend_filter:
        is_uptrend = backtest_df['MA60'] > backtest_df['MA120']
        backtest_df['Position'] = backtest_df['Position'].where(
            ~((backtest_df['Position'] > 0) & ~is_uptrend), 0
        )
        print(f'  📐 趨勢濾網 + 最低持有{min_hold}天')

    bt_result = backtest(backtest_df)

    print()
    print(f'╔{"═" * 45}╗')
    print(f'║  分析完成！')
    print(f'║  df, inst_df, margin_df, bt_result 等變數已就緒')
    print(f'╚{"═" * 45}╝')

    return {
        'df': df,
        'inst_df': inst_df,
        'margin_df': margin_df,
        'ml_df': ml_df,
        'rf_model': rf_model,
        'bt_result': bt_result,
    }


# ── CLI 入口 ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='台股 AI 量化分析系統')
    parser.add_argument('ticker', nargs='?', default='2330.TW', help='股票代號 (預設 2330.TW)')
    parser.add_argument('--period', default='3y', help='資料區間 (預設 3y)')
    parser.add_argument('--no-xgb', action='store_true', help='跳過 XGBoost')
    parser.add_argument('--lstm', action='store_true', help='加入 LSTM')
    parser.add_argument('--no-trend', action='store_true', help='關閉趨勢濾網')
    args = parser.parse_args()

    results = run_analysis(
        ticker=args.ticker,
        period=args.period,
        with_xgb=not args.no_xgb,
        with_lstm=args.lstm,
        trend_filter=not args.no_trend,
    )
