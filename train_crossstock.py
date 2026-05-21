"""
跨股票模型訓練 + 優化
用法：
  python train_crossstock.py                         # 完整流程（下載→訓練→儲存）
  python train_crossstock.py --skip-download         # 只訓練（已有快取）
  python train_crossstock.py --tune                  # 參數搜尋 + 訓練
  python train_crossstock.py --prune                 # 特徵剪枝 + 重新訓練
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import time, sys
from joblib import dump

from tw_stocks import TW_STOCKS
from test import fetch_data, add_all_indicators, build_features

DATA_DIR = Path('data_cross')
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

XGB_MODEL_PATH = 'models/xgb_cross.joblib'
LGB_MODEL_PATH = 'models/lgb_cross.joblib'


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


EXCLUDE = ['Target', 'open', 'high', 'low', 'close', 'volume',
           'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
           'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
           'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
           'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']


def load_data(tickers):
    """載入所有股票資料，回傳 X, y, features"""
    all_X, all_y, feat_list = [], [], None
    for ticker in tickers:
        fpath = DATA_DIR / f'{ticker}.parquet'
        if not fpath.exists():
            continue
        ml_df = pd.read_parquet(fpath)
        features = [c for c in ml_df.columns if c not in EXCLUDE]
        if feat_list is None:
            feat_list = features
        split = int(len(ml_df) * 0.8)
        all_X.append(ml_df.iloc[:split][features].values)
        all_y.append(ml_df.iloc[:split]['Target'].values)
    return np.concatenate(all_X), np.concatenate(all_y), feat_list


def tune_xgb(tickers):
    """Grid Search + TimeSeriesSplit 找最佳 XGBoost 參數"""
    import xgboost as xgb
    print('\n🔧 參數搜尋（取 30 檔代表，TimeSeriesSplit 3 折）...')
    subset = tickers[:30] if len(tickers) > 30 else tickers
    X, y, features = load_data(subset)
    print(f'  搜尋資料：{len(X):,} 筆, {X.shape[1]} 特徵')

    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [3, 5],
    }

    model = xgb.XGBClassifier(
        n_estimators=500, random_state=42,
        verbosity=0, n_jobs=1
    )

    search = GridSearchCV(
        model, param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1,
        verbose=1
    )
    search.fit(X, y)

    print(f'\n  🏆 最佳參數：{search.best_params_}')
    print(f'  最佳 3 折平均準確率：{search.best_score_:.2%}')

    # 把沒搜到的補回預設值
    best = search.best_params_.copy()
    best.setdefault('n_estimators', 500)
    best.setdefault('random_state', 42)
    best.setdefault('verbosity', 0)
    best.setdefault('n_jobs', -1)
    best.setdefault('enable_categorical', False)

    print(f'  完整參數：{best}')
    return best


def prune_features(model, features, threshold=0.01):
    """砍掉重要性低於 threshold 的特徵"""
    imp = pd.Series(model.feature_importances_, index=features)
    keep = imp[imp >= threshold]
    dropped = imp[imp < threshold]
    if len(dropped) > 0:
        print(f'\n  ✂️  剪枝：砍掉 {len(dropped)} 個低貢獻特徵')
        for feat, val in dropped.sort_values(ascending=False).items():
            print(f'    - {feat}: {val:.3%}')
        print(f'  保留 {len(keep)} / {len(features)} 個特徵')
    return list(keep.index)


def train_cross(tickers, model_path='models/rf_cross.joblib', max_depth=5):
    """訓練跨股票 RF 模型"""
    all_X, all_y, feat_list = [], [], None

    for ticker in tickers:
        fpath = DATA_DIR / f'{ticker}.parquet'
        if not fpath.exists():
            continue

        ml_df = pd.read_parquet(fpath)
        exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
                   'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
                   'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
                   'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
                   'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
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


def train_cross_xgb(tickers, model_path='models/xgb_cross.joblib', params=None, prune=False):
    """訓練跨股票 XGBoost 模型"""
    import xgboost as xgb
    X, y, features = load_data(tickers)

    print(f'\n  📊 訓練資料：{len(X):,} 筆, {X.shape[1]} 特徵')
    print(f'  正標籤比率：{y.mean():.1%}')

    if params is None:
        params = {
            'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
            'enable_categorical': False,
        }

    print(f'   參數: {params}')
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    print(f'  訓練準確率：{train_acc:.2%}')

    # 特徵重要性
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f'\n  🌟 特徵重要性 Top 10：')
    for feat, val in imp.head(10).items():
        print(f'    {feat:<20s} {val:.3%}')

    if prune:
        keep = prune_features(model, features, threshold=0.01)
        print(f'\n  🔄 剪枝後重新訓練...')
        X_pruned = pd.DataFrame(X, columns=features)[keep].values
        model2 = xgb.XGBClassifier(**params)
        model2.fit(X_pruned, y)
        acc2 = accuracy_score(y, model2.predict(X_pruned))
        print(f'  剪枝後訓練準確率：{acc2:.2%}')
        model = model2
        features = keep

    dump({'model': model, 'features': features}, model_path)
    print(f'  ✅ 模型已儲存：{model_path}')
    return model, features


def tune_lgb(tickers):
    """Grid Search + TimeSeriesSplit 找最佳 LightGBM 參數"""
    import lightgbm as lgb
    print('\n🔧 LightGBM 參數搜尋（取 30 檔代表，TimeSeriesSplit 3 折）...')
    subset = tickers[:30] if len(tickers) > 30 else tickers
    X, y, features = load_data(subset)
    print(f'  搜尋資料：{len(X):,} 筆, {X.shape[1]} 特徵')

    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        'num_leaves': [15, 31],
        'max_depth': [4, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'min_child_samples': [10, 20],
    }

    model = lgb.LGBMClassifier(
        n_estimators=500, random_state=42, verbose=-1
    )

    search = GridSearchCV(
        model, param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1,
        verbose=1
    )
    search.fit(X, y)

    print(f'\n  🏆 最佳 LightGBM 參數：{search.best_params_}')
    print(f'  最佳 3 折平均準確率：{search.best_score_:.2%}')

    best = search.best_params_.copy()
    best.setdefault('n_estimators', 500)
    best.setdefault('random_state', 42)
    best.setdefault('verbose', -1)

    print(f'  完整參數：{best}')
    return best


def train_cross_lgb(tickers, model_path='models/lgb_cross.joblib', params=None, prune=False):
    """訓練跨股票 LightGBM 模型"""
    import lightgbm as lgb
    X, y, features = load_data(tickers)

    print(f'\n  📊 訓練資料：{len(X):,} 筆, {X.shape[1]} 特徵')
    print(f'  正標籤比率：{y.mean():.1%}')

    if params is None:
        params = {
            'n_estimators': 500, 'num_leaves': 31, 'max_depth': 6,
            'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_samples': 10,
            'random_state': 42, 'verbose': -1,
        }

    print(f'   參數: {params}')
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    print(f'  訓練準確率：{train_acc:.2%}')

    # 特徵重要性
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f'\n  🌟 特徵重要性 Top 10：')
    for feat, val in imp.head(10).items():
        print(f'    {feat:<20s} {val:.3%}')

    if prune:
        keep = prune_features(model, features, threshold=0.01)
        print(f'\n  🔄 剪枝後重新訓練...')
        X_pruned = pd.DataFrame(X, columns=features)[keep].values
        model2 = lgb.LGBMClassifier(**params)
        model2.fit(X_pruned, y)
        acc2 = accuracy_score(y, model2.predict(X_pruned))
        print(f'  剪枝後訓練準確率：{acc2:.2%}')
        model = model2
        features = keep

    dump({'model': model, 'features': features}, model_path)
    print(f'  ✅ 模型已儲存：{model_path}')
    return model, features


def test_one(ticker, model_path='models/rf_cross.joblib'):
    """對單一股票進行樣本外測試"""
    from joblib import load
    saved = load(model_path)
    model = saved['model']
    features = saved['features']

    df = pd.read_parquet(DATA_DIR / f'{ticker}.parquet')
    exclude = ['Target', 'open', 'high', 'low', 'close', 'volume',
               'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
               'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
               'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
               'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
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
                   'Upper_Band', 'Lower_Band', 'MA20', 'MA60', 'MA120', 'MA240',
                   'DEMA', 'TEMA', 'HMA', 'Supertrend', 'ST_Direction',
                   'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid',
                   'Keltner_Mid', 'Keltner_Upper', 'Keltner_Lower']
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
    parser = argparse.ArgumentParser(description='跨股票模型訓練 + 優化')
    parser.add_argument('--skip-download', action='store_true', help='跳過下載')
    parser.add_argument('--list-only', action='store_true', help='只顯示股票代號')
    parser.add_argument('--test', type=str, help='測試單一股票')
    parser.add_argument('--summary', action='store_true', help='顯示全部測試準確率')
    parser.add_argument('--depth', type=int, default=5, help='RF 樹深度 (預設 5)')
    parser.add_argument('--xgb', action='store_true', help='使用 XGBoost 訓練')
    parser.add_argument('--lgb', action='store_true', help='使用 LightGBM 訓練')
    parser.add_argument('--tune', action='store_true', help='Grid Search 找最佳參數')
    parser.add_argument('--prune', action='store_true', help='剪枝低貢獻特徵後重新訓練')
    args = parser.parse_args()

    if args.list_only:
        for t in TW_STOCKS:
            print(t)
        sys.exit(0)

    # 決定股票清單
    if not args.skip_download:
        print(f'📥 開始下載 {len(TW_STOCKS)} 檔台股資料...')
        t0 = time.time()
        ok = download_all(TW_STOCKS)
        print(f'⏱ 下載耗時：{time.time()-t0:.0f}秒')
    else:
        ok = sorted([f.stem for f in DATA_DIR.glob('*.parquet') if not f.stem.endswith('_raw')])
        print(f'📂 從快取載入 {len(ok)} 檔股票資料')

    # LightGBM
    if args.lgb:
        best_params = None
        if args.tune:
            best_params = tune_lgb(ok)
            print(f'\n🧠 使用最佳 LightGBM 參數重新訓練...')
        else:
            print(f'\n🧠 訓練跨股票 LightGBM 模型...')

        train_cross_lgb(ok, LGB_MODEL_PATH, params=best_params, prune=args.prune)

    # XGBoost
    elif args.xgb:
        best_params = None
        if args.tune:
            best_params = tune_xgb(ok)
            print(f'\n🧠 使用最佳參數重新訓練完整模型...')
        else:
            print(f'\n🧠 訓練跨股票 XGBoost 模型...')

        train_cross_xgb(ok, XGB_MODEL_PATH, params=best_params, prune=args.prune)

    # RF
    else:
        print(f'\n🧠 訓練跨股票 RF 模型 (depth={args.depth})...')
        train_cross(ok, 'models/rf_cross.joblib', max_depth=args.depth)

    if args.test:
        print(f'\n🔍 測試單一股票：{args.test}')
        test_one(args.test)

    if args.summary:
        cached = sorted([f.stem for f in DATA_DIR.glob('*.parquet') if f.stem.endswith('.TW') or f.stem.endswith('.TWO')])
        cached = [c for c in cached if not c.endswith('_raw')]
        show_accuracy_summary(cached)
