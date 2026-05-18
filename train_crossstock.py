"""
跨股票 RF 訓練：200 檔台股 → 通用波段預測模型
用法：
  python train_crossstock.py              # 完整流程（下載→訓練→儲存）
  python train_crossstock.py --skip-download  # 只訓練（已有快取）
  python train_crossstock.py --list-only      # 只顯示股票代號
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time, sys
from joblib import dump

from tw_stocks import TW_STOCKS
from test import fetch_data, add_all_indicators, build_features

DATA_DIR = Path('data_cross')
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# XGBoost 跨股票模型
XGB_MODEL_PATH = 'models/xgb_cross.joblib'


def download_all(tickers, max_workers=8):
    """下載所有股票並計算指標"""
    downloaded = []
    for i, ticker in enumerate(tickers, 1):
        fpath = DATA_DIR / f'{ticker}.parquet'
        if fpath.exists():
            downloaded.append(ticker)
            continue

        try:
            df = fetch_data(ticker, '3y')
            if df is None or len(df) < 100:
                continue
            df = add_all_indicators(df)
            ml_df = build_features(df)
            if len(ml_df) < 20:
                continue
            # 保存完整資料（含技術指標）
            df.to_parquet(DATA_DIR / f'{ticker}_raw.parquet')
            ml_df.to_parquet(fpath)
            downloaded.append(ticker)
        except Exception as e:
            pass  # skip failed downloads

        if i % 20 == 0:
            print(f'  📥 [{i}/{len(tickers)}] 已下載 {len(downloaded)} 檔')

    print(f'  ✅ 成功下載 {len(downloaded)}/{len(tickers)} 檔')
    return downloaded


def train_cross(tickers, model_path='models/rf_cross.joblib', max_depth=5):
    """訓練跨股票 RF 模型"""
    all_X, all_y, feat_list = [], [], None

    for ticker in tickers:
        fpath = DATA_DIR / f'{ticker}.parquet'
        if not fpath.exists():
            continue

        ml_df = pd.read_parquet(fpath)
        exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
                   'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240']
        features = [c for c in ml_df.columns if c not in exclude]

        if feat_list is None:
            feat_list = features

        # 取前 80% 作為訓練資料（時間序列分割）
        split = int(len(ml_df) * 0.8)
        train_df = ml_df.iloc[:split]
        all_X.append(train_df[features].values)
        all_y.append(train_df['Target'].values)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    print(f'\n  📊 訓練資料：{len(X):,} 筆, {X.shape[1]} 特徵')
    print(f'  正標籤比率：{y.mean():.1%}')

    model = RandomForestClassifier(
        n_estimators=200, max_depth=max_depth, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    print(f'   參數: n_estimators=200, max_depth={max_depth}, min_samples_leaf=10')
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    print(f'  訓練準確率：{train_acc:.2%}')

    # 儲存模型 + 特徵清單
    dump({'model': model, 'features': feat_list}, model_path)
    print(f'  ✅ 模型已儲存：{model_path}')
    return model, feat_list


def train_cross_xgb(tickers, model_path='models/xgb_cross.joblib'):
    """訓練跨股票 XGBoost 模型"""
    import xgboost as xgb
    all_X, all_y, feat_list = [], [], None

    for ticker in tickers:
        fpath = DATA_DIR / f'{ticker}.parquet'
        if not fpath.exists():
            continue

        ml_df = pd.read_parquet(fpath)
        exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
                   'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240']
        features = [c for c in ml_df.columns if c not in exclude]

        if feat_list is None:
            feat_list = features

        split = int(len(ml_df) * 0.8)
        train_df = ml_df.iloc[:split]
        all_X.append(train_df[features].values)
        all_y.append(train_df['Target'].values)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    print(f'\n  📊 訓練資料：{len(X):,} 筆, {X.shape[1]} 特徵')
    print(f'  正標籤比率：{y.mean():.1%}')

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0,
        enable_categorical=False
    )
    print(f'   參數: n_estimators=500, max_depth=6, lr=0.05')
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    print(f'  訓練準確率：{train_acc:.2%}')

    dump({'model': model, 'features': feat_list}, model_path)
    print(f'  ✅ XGBoost 模型已儲存：{model_path}')
    return model, feat_list


def test_one(ticker, model_path='models/rf_cross.joblib'):
    """對單一股票進行樣本外測試"""
    from joblib import load
    saved = load(model_path)
    model = saved['model']
    features = saved['features']

    df = pd.read_parquet(DATA_DIR / f'{ticker}.parquet')
    exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
               'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240']
    feat_actual = [c for c in df.columns if c not in exclude]

    split = int(len(df) * 0.8)
    test_df = df.iloc[split:]
    X_test = test_df[feat_actual]
    y_test = test_df['Target']

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f'  {ticker} 樣本外準確率：{acc:.2%} ({len(X_test)} 筆)')
    return acc


def show_accuracy_summary(tickers, model_path='models/rf_cross.joblib'):
    """顯示所有股票樣本外準確率"""
    from joblib import load
    saved = load(model_path)
    model = saved['model']
    features = saved['features']

    results = []
    for ticker in tickers:
        fpath = DATA_DIR / f'{ticker}.parquet'
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
                   'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240']
        feat_actual = [c for c in df.columns if c not in exclude]
        split = int(len(df) * 0.8)
        test_df = df.iloc[split:]
        y_test = test_df['Target']
        if len(y_test) < 5:
            continue
        acc = accuracy_score(y_test, model.predict(test_df[feat_actual]))
        results.append((ticker, acc, len(y_test)))

    results.sort(key=lambda x: -x[1])
    print('\n📊 各股票樣本外準確率排名：')
    print(f'  {"股票":>12} {"準確率":>8} {"測試筆數":>10}')
    print(f'  {"-"*32}')
    for t, a, n in results:
        flag = ' ✅' if a > 0.55 else ''
        print(f'  {t:>12} {a:>7.2%} {n:>10}{flag}')
    print(f'\n  平均：{np.mean([r[1] for r in results]):.2%}')
    print(f'  最佳：{results[0][0]} {results[0][1]:.2%}')
    print(f'  最差：{results[-1][0]} {results[-1][1]:.2%}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='跨股票 RF 訓練')
    parser.add_argument('--skip-download', action='store_true', help='跳過下載')
    parser.add_argument('--list-only', action='store_true', help='只顯示股票代號')
    parser.add_argument('--test', type=str, help='測試單一股票')
    parser.add_argument('--summary', action='store_true', help='顯示全部測試準確率')
    parser.add_argument('--depth', type=int, default=5, help='RF 樹深度 (預設 5)')
    parser.add_argument('--xgb', action='store_true', help='使用 XGBoost 訓練')
    args = parser.parse_args()

    if args.list_only:
        for t in TW_STOCKS:
            print(t)
        sys.exit(0)

    if not args.skip_download:
        print(f'📥 開始下載 {len(TW_STOCKS)} 檔台股資料...')
        t0 = time.time()
        ok = download_all(TW_STOCKS)
        print(f'⏱ 下載耗時：{time.time()-t0:.0f}秒')

        if args.xgb:
            print(f'\n🧠 開始訓練跨股票 XGBoost 模型...')
            train_cross_xgb(ok, XGB_MODEL_PATH)
        else:
            print(f'\n🧠 開始訓練跨股票 RF 模型 (depth={args.depth})...')
            train_cross(ok, 'models/rf_cross.joblib', max_depth=args.depth)
    else:
        cached = sorted([f.stem for f in DATA_DIR.glob('*.parquet') if not f.stem.endswith('_raw')])
        print(f'📂 從快取載入 {len(cached)} 檔股票資料')
        if args.xgb:
            train_cross_xgb(cached, XGB_MODEL_PATH)
        else:
            train_cross(cached, 'models/rf_cross.joblib', max_depth=args.depth)

    if args.test:
        print(f'\n🔍 測試單一股票：{args.test}')
        test_one(args.test)

    if args.summary:
        cached = sorted([f.stem for f in DATA_DIR.glob('*.parquet') if f.stem.endswith('.TW') or f.stem.endswith('.TWO')])
        cached = [c for c in cached if not c.endswith('_raw')]
        show_accuracy_summary(cached)
