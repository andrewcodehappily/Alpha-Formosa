"""
掃描 watchlist 所有股票，比較單股 RF vs 跨股票 XGBoost
用法：
  python scan.py                           # 完整掃描
  python scan.py --quick                   # 只跑 10 檔測試
  python scan.py --cross-only              # 只用跨股票模型
"""
import warnings; warnings.filterwarnings('ignore')
import json, sys, time
from pathlib import Path
import pandas as pd, numpy as np
from joblib import load

from tw_stocks import TW_STOCKS
from test import fetch_data, add_all_indicators, build_features, train_random_forest

WATCHLIST = 'watchlist.json'
CROSS_MODEL = 'models/xgb_cross.joblib'
LGB_MODEL = 'models/lgb_cross.joblib'
DATA_DIR = Path('data_scan')
DATA_DIR.mkdir(exist_ok=True)


def load_watchlist():
    with open(WATCHLIST) as f:
        data = json.load(f)
    return data['stocks']


def load_cross_model(use_lgb=False):
    path = LGB_MODEL if use_lgb else CROSS_MODEL
    print(f'  📦 載入跨股票模型：{path}')
    saved = load(path)
    return saved['model'], saved['features']


def run_single_stock(ticker):
    """跑單股 RF，回傳回測結果"""
    df = fetch_data(ticker, '3y')
    if df is None or len(df) < 150:
        return None

    df = add_all_indicators(df)
    ml_df = build_features(df)
    if len(ml_df) < 50:
        return None

    rf_model, rf_features, X_train, X_test, y_train, y_test = train_random_forest(ml_df)

    # 樣本外回測
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    backtest_df = df.loc[X_test.index].copy()

    min_hold = 10
    raw_long = rf_probs >= 0.48
    position = np.zeros(len(backtest_df), dtype=float)
    hold = 0
    for i in range(len(backtest_df)):
        if hold > 0:
            hold -= 1
            position[i] = position[i-1]
        elif raw_long[i]:
            position[i] = rf_probs[i]
            hold = min_hold

    is_uptrend = backtest_df['MA60'] > backtest_df['MA120']
    position = np.where(~((position > 0) & ~is_uptrend), position, 0)
    backtest_df['Position'] = position

    ret = calc_return(backtest_df)
    return {
        'ticker': ticker,
        'return': ret,
        'test_acc': accuracy_score(y_test, rf_model.predict(X_test)),
        'trades': count_trades(position),
    }


def run_cross_stock(ticker, model, features):
    """跑跨股票 XGBoost，回傳回測結果"""
    df = fetch_data(ticker, '3y')
    if df is None or len(df) < 150:
        return None

    df = add_all_indicators(df)
    ml_df = build_features(df)
    if len(ml_df) < 50:
        return None

    # 找出 features 跟 ml_df 共有的欄位
    common = [c for c in features if c in ml_df.columns]
    if len(common) < len(features) * 0.5:
        return None  # 太少特徵對不起來

    X = ml_df[common].values
    probs = model.predict_proba(X)[:, 1]

    split = int(len(ml_df) * 0.8)
    test_idx = ml_df.index[split:]
    test_probs = probs[split:]
    backtest_df = df.loc[test_idx].copy()

    min_hold = 10
    raw_long = test_probs >= 0.48
    position = np.zeros(len(backtest_df), dtype=float)
    hold = 0
    for i in range(len(backtest_df)):
        if hold > 0:
            hold -= 1
            position[i] = position[i-1]
        elif raw_long[i]:
            position[i] = test_probs[i]
            hold = min_hold

    is_uptrend = backtest_df['MA60'] > backtest_df['MA120']
    position = np.where(~((position > 0) & ~is_uptrend), position, 0)
    backtest_df['Position'] = position

    ret = calc_return(backtest_df)
    return {
        'ticker': ticker,
        'return': ret,
        'trades': count_trades(position),
    }


def calc_return(df):
    BUY_RATE = 0.001425; SELL_RATE = 0.004425
    df['Daily_Return'] = df['close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    pos_change = df['Position'].diff().fillna(0)
    df['Strategy_Return'] -= pos_change.clip(lower=0) * BUY_RATE
    df['Strategy_Return'] -= (-pos_change).clip(lower=0) * SELL_RATE
    return (1 + df['Strategy_Return']).prod() - 1


def count_trades(position):
    """計算交易趟數"""
    pos_prev = np.roll(position, 1)
    pos_prev[0] = 0
    entries = np.sum((position > 0.05) & (pos_prev < 0.05))
    return entries


def print_ranking(results, label):
    """印出排名"""
    results = sorted(results, key=lambda x: -x['return'])
    print(f'\n  {"排名":>4} {"股票":>10} {"報酬率":>10} {"交易次數":>8}')
    print(f'  {"-" * 36}')
    for i, r in enumerate(results[:20], 1):
        flag = ' 💰' if r['return'] > 0.2 else ''
        print(f'  {i:>4} {r["ticker"]:>10} {r["return"]:>+9.2%} {r["trades"]:>6}{flag}')

    avg = np.mean([r['return'] for r in results])
    pos = sum(1 for r in results if r['return'] > 0)
    print(f'\n  平均報酬：{avg:.2%}  |  正報酬：{pos}/{len(results)} ({pos/len(results):.0%})')
    return avg


if __name__ == '__main__':
    import argparse
    from sklearn.metrics import accuracy_score

    parser = argparse.ArgumentParser(description='掃描 watchlist 股票')
    parser.add_argument('--quick', action='store_true', help='只跑 10 檔測試')
    parser.add_argument('--cross-only', action='store_true', help='只用跨股票模型')
    parser.add_argument('--single-only', action='store_true', help='只用單股 RF')
    parser.add_argument('--lgb', action='store_true', help='使用 LightGBM 取代 XGBoost')
    parser.add_argument('--lgb-only', action='store_true', help='只用 LightGBM 跨股票模型')
    args = parser.parse_args()

    stocks = load_watchlist()
    if args.quick:
        stocks = stocks[:10]
        print(f'⚡ 快速模式：只掃 {len(stocks)} 檔')

    print(f'🔍 開始掃描 {len(stocks)} 檔股票...\n')

    single_results = []
    cross_results = []

    cross_model, cross_features = load_cross_model(use_lgb=args.lgb or args.lgb_only)
    model_label = 'LGB' if (args.lgb or args.lgb_only) else 'XGB'
    cross_only = args.cross_only or args.lgb_only

    for i, s in enumerate(stocks, 1):
        ticker = s['ticker']
        print(f'  [{i:>3}/{len(stocks)}] {ticker} ... ', end='', flush=True)

        try:
            if not cross_only:
                sr = run_single_stock(ticker)
                if sr:
                    single_results.append(sr)
                    print(f'RF: {sr["return"]:+.2%}', end='', flush=True)
                else:
                    print(f'RF: ❌', end='', flush=True)

            if not args.single_only:
                cr = run_cross_stock(ticker, cross_model, cross_features)
                if cr:
                    cross_results.append(cr)
                    action = '  ' if not cross_only else ''
                    print(f'{action}{model_label}: {cr["return"]:+.2%}', end='', flush=True)
                else:
                    print(f' {model_label}: ❌', end='', flush=True)

        except Exception as e:
            print(f'⚠️  錯誤: {e}', end='', flush=True)

        print()

    print(f'\n{"=" * 50}')
    print(f'📊 掃描完成！')
    print(f'{"=" * 50}')

    if single_results:
        print(f'\n🔥 單股 RF 排名 (共 {len(single_results)} 檔)：')
        single_avg = print_ranking(single_results, '單股 RF')

    if cross_results:
        cross_label_full = f'跨股票 {model_label}'
        print(f'\n🔥 {cross_label_full} 排名 (共 {len(cross_results)} 檔)：')
        cross_avg = print_ranking(cross_results, cross_label_full)

    if single_results and cross_results:
        print(f'\n{"=" * 50}')
        print(f'🏆 終極對決')
        print(f'{"=" * 50}')
        print(f'  單股 RF 平均：{single_avg:.2%}')
        print(f'  {cross_label_full} 平均：{cross_avg:.2%}')
        winner = '單股 RF' if single_avg > cross_avg else cross_label_full
        print(f'  贏家：{winner}')

        # 同場比較
        single_map = {r['ticker']: r for r in single_results}
        cross_map = {r['ticker']: r for r in cross_results}
        common = set(single_map) & set(cross_map)
        wins_single = sum(1 for t in common if single_map[t]['return'] > cross_map[t]['return'])
        print(f'  同場對決 {len(common)} 檔：RF 贏 {wins_single} 檔, {model_label} 贏 {len(common)-wins_single} 檔')
