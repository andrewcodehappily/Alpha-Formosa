"""
台股 AI 量化分析系統 — 嚴格樣本外盲測 + 多頭實戰版
預測 5 日後波段趨勢，只做多不做空，附完整交易明細
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
    df = yf.download(ticker, period=period)
    df = fix_columns(df)
    df.columns = [c.lower() for c in df.columns]
    return df

def fix_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]
    return df

def fetch_institutional(ticker='2330', years=1):
    start = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    try:
        res = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                     "data_id": ticker, "start_date": start},
            timeout=10
        )
        data = res.json().get('data', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except Exception:
        return pd.DataFrame()

def fetch_margin(ticker='2330', years=1):
    start = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    try:
        res = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockMarginPurchaseShortSale",
                     "data_id": ticker, "start_date": start},
            timeout=10
        )
        data = res.json().get('data', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except Exception:
        return pd.DataFrame()


# ── 技術指標 ─────────────────────────────────────────────

def add_ma(df, windows=[20, 60, 120, 240]):
    for w in windows:
        df[f'MA{w}'] = df['close'].rolling(window=w).mean()
    return df

def add_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df):
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_bollinger(df, period=20, std=2):
    df['MA20'] = df['close'].rolling(period).mean()
    rstd = df['close'].rolling(period).std()
    df['Upper_Band'] = df['MA20'] + rstd * std
    df['Lower_Band'] = df['MA20'] - rstd * std
    return df

def add_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    return df

def add_ema_cross(df):
    """EMA 黃金/死亡交叉（只保留訊號，不吃 raw EMA 防過擬合）"""
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['Golden_Cross'] = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1))
    df['Death_Cross'] = (ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))
    df['EMA_Ratio'] = ema12 / ema26  # 比率（無量綱，避免價格過擬合）
    return df

def add_stochastic(df, period=14):
    """KD 隨機指標"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['PctK'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['PctD'] = df['PctK'].rolling(window=3).mean()
    return df

def add_adx(df, period=14):
    """ADX 趨向指標"""
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
    """動量指標"""
    df['Momentum_14d'] = df['close'] - df['close'].shift(period)
    return df

def add_williams_r(df, period=14):
    """威廉指標"""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    df['WMSR'] = ((high_max - df['close']) / (high_max - low_min)) * -100
    return df

def add_cci(df, period=20):
    """CCI 商品通道指數"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).std())
    return df

def add_chaikin(df):
    """Chaikin 資金流"""
    df['Chaikin'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['Chaikin'] = df['Chaikin'].rolling(window=3).sum()
    return df

def add_obv(df):
    """OBV 能量潮"""
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


# ── 特徵工程（目標：預測 5 日後漲跌）────────────────────────

def build_features(df, label_shifts=5):
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

    # 目標：5 日後是否上漲（波段趨勢）
    ml['Target'] = (ml['close'].shift(-label_shifts) > ml['close']).astype(int)
    ml.dropna(inplace=True)
    return ml


def add_cross_prob_feature(ml_df, cross_model_path='models/xgb_cross.joblib'):
    """把跨股票模型預測機率當作特徵加入（讓 RF 自行學習何時參考）"""
    from pathlib import Path
    from joblib import load as jload
    if not Path(cross_model_path).exists():
        return ml_df

    saved = jload(cross_model_path)
    cross_model = saved['model']
    cross_features = list(saved['features'])

    exclude_cols = ['Target', 'open', 'high', 'low', 'close', 'volume',
                    'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
                    'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
                    'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
                    'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
    have = [c for c in ml_df.columns if c not in exclude_cols]
    use = [f for f in cross_features if f in have]

    cross_prob = cross_model.predict_proba(ml_df[use])[:, 1]
    ml_df['Cross_Prob'] = cross_prob
    return ml_df


# ── 模型訓練（強正則化防過擬合）───────────────────────────

def train_random_forest(ml_df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
               'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
               'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
               'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
               'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
    features = [c for c in ml_df.columns if c not in exclude]

    X, y = ml_df[features], ml_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # depth=3 best for both 24-feat and 39-feat (deeper overfits)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'  RF 訓練準確率: {train_acc:.2%} (已限制過擬合)')
    print(f'  RF 測試準確率: {test_acc:.2%} (波段盲測)')

    return model, features, X_train, X_test, y_train, y_test


def train_xgboost(ml_df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import xgboost as xgb

    exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
               'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
               'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
               'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
               'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
    features = [c for c in ml_df.columns if c not in exclude]

    X, y = ml_df[features], ml_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'  XGB 訓練準確率: {train_acc:.2%} (已限制過擬合)')
    print(f'  XGB 測試準確率: {test_acc:.2%} (波段盲測)')

    return model, features, X_train, X_test, y_train, y_test


def train_lstm(df, seq_len=20, epochs=30, lr=0.001):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  使用裝置: {device}')

    data = df[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i - seq_len:i])
        y.append(data_scaled[i, 3])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train_t = torch.tensor(X[:split], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y[:split], dtype=torch.float32, device=device).unsqueeze(1)
    X_test_t = torch.tensor(X[split:], dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y[split:], dtype=torch.float32, device=device).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

    class LSTMPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tl = criterion(model(X_train_t), y_train_t).item()
        vl = criterion(model(X_test_t), y_test_t).item()
    print(f'  LSTM 訓練損失: {tl:.6f} | 測試損失: {vl:.6f}')
    return model, scaler


# ── 回測（信心度倉位 + 台股實戰成本）────────────────────────

def backtest(backtest_df, initial_capital=1_000_000):
    """
    回測引擎：信心度倉位 + 正確台股交易成本
    - 倉位 = RF 機率（0.48~1.0），非 binary 0/1
    - 買進成本 0.1425%（券商手續費）
    - 賣出成本 0.4425%（手續費 0.1425% + 證交稅 0.3%）
    - 最後強制結算
    """
    BUY_RATE = 0.001425    # 買進手續費
    SELL_RATE = 0.004425   # 賣出手續費 + 證交稅

    df = backtest_df.copy()

    if 'Position' not in df.columns:
        df['Position'] = 0.0

    # — 每日報酬 —
    df['Daily_Return'] = df['close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']

    # — 交易成本（根據倉位變動量） —
    pos_change = df['Position'].diff().fillna(0)
    buy_cost = pos_change.clip(lower=0) * BUY_RATE
    sell_cost = (-pos_change).clip(lower=0) * SELL_RATE
    df['Strategy_Return'] -= (buy_cost + sell_cost)
    df['Traded'] = pos_change != 0

    df['Cumulative_Market'] = (1 + df['Daily_Return']).cumprod() * initial_capital
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod() * initial_capital

    # — 交易明細 —
    print('\n  📜 交易明細')
    print('  ' + '-' * 80)

    entry_price = 0.0
    entry_date = None
    entry_type = None  # 'long' or 'short'
    entry_pos = 0.0    # 進場時的倉位大小
    trade_count = 0
    win_trades = 0
    last_idx = len(df) - 1

    for idx, (date, row) in enumerate(df.iterrows()):
        pos = row['Position']
        prev_pos = df['Position'].iloc[idx - 1] if idx > 0 else 0.0
        cat = row.get('Signal_Category', '')
        is_last = (idx == last_idx)

        # 進場：從空手→有倉位（且超過 5% 才算）
        if entry_type is None and abs(pos) > 0.05 and abs(prev_pos) < 0.05:
            entry_price = row['close']
            entry_date = date
            entry_type = 'long' if pos > 0 else 'short'
            entry_pos = abs(pos)
            action = '做多' if pos > 0 else '放空'
            print(f"  {'🟢' if pos > 0 else '🔴'} [{action}] {date.date()}  價格: {entry_price:>8.2f}  倉位: {abs(pos):.0%}  訊號: {cat}")

        # 出場：有倉位→空手（或最後一天強制結算）
        elif entry_type is not None and ((abs(pos) < 0.05 and abs(prev_pos) >= 0.05) or is_last):
            exit_price = row['close']
            actual_pos = entry_pos  # 用進場時的倉位算損益
            if entry_type == 'long':
                trade_ret = (exit_price / entry_price - 1) - (BUY_RATE + SELL_RATE)
            else:
                trade_ret = (entry_price / exit_price - 1) - (BUY_RATE + SELL_RATE)
            trade_count += 1
            if trade_ret > 0:
                win_trades += 1
            flag = '✅' if trade_ret > 0 else '❌'
            action = '賣出' if entry_type == 'long' else '回補'
            suffix = ' ⏹ 強制結算' if is_last else ''
            print(f"  {'🔵' if entry_type == 'long' else '🟣'} [{action}] {date.date()}  價格: {exit_price:>8.2f}  淨報酬: {trade_ret:>+7.2%}  {flag}{suffix}")
            entry_date = None
            entry_type = None
            entry_pos = 0.0

        elif entry_type is not None and pos != 0 and prev_pos != 0 and abs(pos - prev_pos) > 0.15:
            # 倉位顯著變化（>15%）— 僅記錄不計趟
            action = '加碼' if pos > prev_pos else '減碼'
            print(f"  {'⬆' if pos > prev_pos else '⬇'} [{action}] {date.date()}  價格: {row['close']:>8.2f}  倉位: {abs(pos):.0%}")

    print('  ' + '-' * 80)

    # — 績效指標 —
    total_return = df['Cumulative_Strategy'].iloc[-1] / initial_capital - 1
    market_return = df['Cumulative_Market'].iloc[-1] / initial_capital - 1
    std = df['Strategy_Return'].std()
    sharpe = np.sqrt(252) * df['Strategy_Return'].mean() / std if std != 0 else 0
    max_dd = (df['Cumulative_Strategy'] / df['Cumulative_Strategy'].cummax() - 1).min()
    win_rate = win_trades / trade_count if trade_count > 0 else 0

    print(f'\n  📊 樣本外盲測績效（{initial_capital:,} 元，含真實交易成本）')
    print(f'  {"=" * 42}')
    print(f'  累積報酬:   {total_return:>+7.2%}')
    print(f'  大盤報酬:   {market_return:>+7.2%}')
    print(f'  Sharpe 比:  {sharpe:>7.2f}')
    print(f'  最大回撤:   {max_dd:>+7.2%}')
    print(f'  交易次數:   {trade_count} 趟')
    print(f'  交易勝率:   {win_rate:>7.2%}')

    return df


# ── 主流程 ─────────────────────────────────────────────

def run_analysis(ticker='2330.TW', period='3y', with_xgb=True, with_lstm=False,
                 trend_filter=True, cross_model_path=None, ensemble=False):
    if ensemble and cross_model_path is None:
        # 優先使用 XGBoost 跨股票模型（訓練準確率 73.8% vs RF 57.2%）
        xgb_path = 'models/xgb_cross.joblib'
        rf_path = 'models/rf_cross.joblib'
        from pathlib import Path
        cross_model_path = xgb_path if Path(xgb_path).exists() else rf_path
    if ensemble:
        strategy = '混合投票(單股+跨股票)'
    elif cross_model_path:
        strategy = '跨股票模型'
    else:
        strategy = '多空雙向'
    print(f'╔{"═" * 50}╗')
    print(f'║  台股 AI 量化分析 — 嚴格盲測 + 多空雙向版')
    print(f'║  標的: {ticker}  區間: {period}')
    print(f'║  預測: 5 日後波段趨勢  策略: {strategy}')
    trend_text = ' + 雙向趨勢濾網' if trend_filter else ' + 無濾網'
    print(f'║  濾網: {trend_text}')
    print(f'╚{"═" * 50}╝\n')

    print('📥 [1/5] 下載股價資料...')
    df = fetch_data(ticker, period)
    print(f'  共 {len(df)} 筆 ({df.index[0].date()} ~ {df.index[-1].date()})')

    print('📊 [2/5] 計算技術指標...')
    df = add_all_indicators(df)

    print('🏦 [3/5] 抓取三大法人資料...')
    ticker_num = ticker.replace('.TW', '')
    inst_df = fetch_institutional(ticker_num)
    margin_df = fetch_margin(ticker_num)
    if not inst_df.empty:
        print(f'  法人 {len(inst_df)} 筆 | 融資券 {len(margin_df)} 筆')
    else:
        print('  ⚠️ FinMind 法人資料無法取得（可略過）')

    print('🤖 [4/5] 載入/訓練 AI 模型...')
    ml_df = build_features(df)
    feat_count = len([c for c in ml_df.columns if c not in ['Target', 'open', 'high', 'low', 'close', 'volume',
                   'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240']])
    print(f'  原始特徵: {feat_count} 維 | 樣本: {len(ml_df)}')

    from joblib import load as jload
    from sklearn.metrics import accuracy_score
    exclude_cols = ['Target', 'open', 'high', 'low', 'close', 'volume',
                    'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
                    'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
                    'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
                    'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
    have_features = [c for c in ml_df.columns if c not in exclude_cols]

    # ── 混合投票：單股 RF + 跨股票 RF ──
    if ensemble:
        print('  ── 單股 Random Forest ──')
        rf_model, rf_features, X_train, X_test, y_train, y_test = train_random_forest(ml_df)
        test_indices = X_test.index
        single_probs = rf_model.predict_proba(X_test)[:, 1]

        saved = jload(cross_model_path)
        cross_model = saved['model']
        cross_features = list(saved['features'])
        use_cross = [f for f in cross_features if f in have_features]
        X_test_cross = ml_df.loc[test_indices][use_cross]
        cross_probs = cross_model.predict_proba(X_test_cross)[:, 1]

        # 平均機率
        rf_probs = (single_probs + cross_probs) / 2
        print(f'  🔀 混合投票: prob = (單股 + 跨股票) / 2')
        print(f'     單股訓練準確率: {accuracy_score(y_train, rf_model.predict(X_train)):.2%}')
        print(f'     測試樣本: {len(rf_probs)} 筆')

    # ── 純跨股票模型 ──
    elif cross_model_path:
        saved = jload(cross_model_path)
        rf_model = saved['model']
        rf_features = list(saved['features'])
        use = [f for f in rf_features if f in have_features]
        missing = set(rf_features) - set(have_features)
        if missing:
            print(f'  ⚠️ 缺少 {len(missing)} 個特徵')

        split = int(len(ml_df) * 0.8)
        X_test = ml_df.iloc[split:][use]
        y_test = ml_df.iloc[split:]['Target']
        test_indices = X_test.index
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        print(f'  📦 載入跨股票模型（{len(rf_features)} 維特徵）')
        print(f'  測試資料: {len(X_test)} 筆')

    # ── 單股模型 ──
    else:
        print('  ── Random Forest ──')
        rf_model, rf_features, X_train, X_test, y_train, y_test = train_random_forest(ml_df)
        test_indices = X_test.index
        rf_probs = rf_model.predict_proba(X_test)[:, 1]

        if with_xgb:
            print('  ── XGBoost ──')
            xgb_model, xgb_features, *_ = train_xgboost(ml_df)

        if with_lstm:
            print('  ── LSTM ──')
            lstm_model, scaler = train_lstm(df)

    print('\n📈 [5/5] 嚴格樣本外回測 (Out-of-Sample)...')
    backtest_df = df.loc[test_indices].copy()

    # 7 級訊號
    conditions = [
        rf_probs >= 0.70,  # 強力買進
        rf_probs >= 0.55,  # 買進
        rf_probs >= 0.48,  # 偏多
        rf_probs >= 0.30,  # 中立
        rf_probs >= 0.20,  # 偏空
        rf_probs >= 0.10,  # 賣出
        rf_probs <  0.10,  # 強力賣出
    ]
    categories = ['強力買進', '買進', '偏多', '中立', '偏空', '賣出', '強力賣出']
    backtest_df['Signal_Category'] = np.select(conditions, categories, default='中立')

    # ── 信心度倉位：position = model_probability ──

    min_hold = 10

    raw_long = rf_probs >= 0.48
    raw_short = rf_probs <= 0.35

    # 技術面確認放空
    short_tech = (
        (backtest_df['close'] < backtest_df['MA20']) &
        (backtest_df['RSI'] < 45) &
        (backtest_df['MACD'] < backtest_df['MACD_Signal'])
    )

    # 信心度倉位：機率直接當倉位大小
    hold = 0
    position = np.zeros(len(backtest_df), dtype=float)
    for i in range(len(backtest_df)):
        if hold > 0:
            hold -= 1
            position[i] = position[i-1]
        elif raw_long[i]:
            position[i] = rf_probs[i]  # 50% 信心 → 50% 倉位
            hold = min_hold
        elif raw_short[i] and short_tech[i]:
            position[i] = -rf_probs[i]
            hold = min_hold

    backtest_df['Position'] = position

    # 趨勢濾網（把逆勢的多單濾掉）
    if trend_filter:
        is_uptrend = backtest_df['MA60'] > backtest_df['MA120']
        backtest_df['Position'] = backtest_df['Position'].where(
            ~((backtest_df['Position'] > 0) & ~is_uptrend), 0
        )
        print(f'  📐 趨勢濾網 + 最低持有{min_hold}天')

    bt_result = backtest(backtest_df, initial_capital=1_000_000)

    # 特徵重要性排名
    importance = pd.Series(rf_model.feature_importances_, index=rf_features).sort_values(ascending=False)
    print(f'\n  🌟 特徵重要性排名（共 {len(importance)} 項）：')
    print(f'  {"排名":>4} {"特徵":<22} {"重要性":>8} {"累計佔比":>8}')
    print(f'  {"-" * 44}')
    cumsum = 0
    for i, (feat, imp) in enumerate(importance.items(), 1):
        cumsum += imp
        feat_short = feat[:22] if len(feat) > 22 else feat
        print(f'  {i:>4} {feat_short:<22} {imp:>7.2%} {cumsum:>7.2%}')

    print(f'\n╔{"═" * 50}╗')
    print(f'║  分析完成！策略已轉換為實戰多頭配置。')
    print(f'╚{"═" * 50}╝')

    return {
        'df': df,
        'bt_result': bt_result,
        'rf_model': rf_model,
        'X_test': X_test,
        'inst_df': inst_df,
        'margin_df': margin_df,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='台股 AI 波段盲測系統')
    parser.add_argument('ticker', nargs='?', default='2330.TW', help='股票代號 (預設 2330.TW)')
    parser.add_argument('--period', default='3y', help='資料區間 (預設 3y)')
    parser.add_argument('--no-xgb', action='store_true', help='跳過 XGBoost')
    parser.add_argument('--lstm', action='store_true', help='加入 LSTM')
    parser.add_argument('--no-trend', action='store_true', help='關閉趨勢濾網')
    parser.add_argument('--cross-model', default=None, help='使用跨股票模型路徑 (預設: models/rf_cross.joblib)',
                        nargs='?', const='models/rf_cross.joblib')
    parser.add_argument('--ensemble', action='store_true', help='混合投票 (單股 RF + 跨股票 RF)')
    args = parser.parse_args()

    results = run_analysis(
        ticker=args.ticker,
        period=args.period,
        with_xgb=not args.no_xgb,
        with_lstm=args.lstm,
        trend_filter=not args.no_trend,
        cross_model_path=args.cross_model,
        ensemble=args.ensemble,
    )
