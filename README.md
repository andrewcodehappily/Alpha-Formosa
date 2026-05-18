# Alpha-Formosa

台股 AI 波段預測系統 — 用機器學習預測台股波段方向，支援單股 RandomForest 與跨股票 XGBoost 模型。

## 功能

- **技術指標全自動計算** — RSI, MACD, 布林通道, ATR, KD, ADX, CCI, OBV, Williams %R 等 13 種指標
- **隨機森林波段預測** — 每檔股票獨立訓練，預測 5 日後漲跌方向
- **跨股票 XGBoost 模型** — 200 檔台股聯合訓練，泛化能力強於單股模型
- **信心度倉位管理** — 以模型預測機率決定進場倉位，不 all-in
- **真實台灣交易成本** — 買進 0.1425% 手續費，賣出 0.4425% 手續費+稅
- **趨勢濾網** — MA60 > MA120 確認多頭趨勢才做多
- **最低持有天數** — 預設持有 10 天，避免頻繁進出
- **樣本外回測** — 嚴格時間序列分割，不偷看未來資料

## 安裝

```bash
pip install pandas numpy scikit-learn xgboost yfinance joblib
```

## 使用方法

### 單股分析

```bash
python stock.py 2330.TW
python stock.py 2330.TW --no-xgb       # 跳過 XGBoost
python stock.py 2330.TW --no-trend     # 關閉趨勢濾網
python stock.py 2330.TW --lstm         # 加入 LSTM
```

### 全掃描（比較單股 RF vs 跨股票 XGBoost）

```bash
python scan.py                          # 掃描 watchlist 全部股票
python scan.py --quick                  # 只跑 10 檔測試
python scan.py --cross-only             # 只用跨股票 XGBoost
```

### 跨股票模型訓練

```bash
python train_crossstock.py                                    # 下載 200 檔 → 訓練 RF
python train_crossstock.py --xgb                              # 使用 XGBoost 訓練
python train_crossstock.py --skip-download                    # 跳過下載，直接用快取
python train_crossstock.py --summary                          # 顯示所有股票樣本外準確率
```

### 圖形化介面

```bash
python test_ui.py                    # Gradio Web UI
```

## 模型策略參數

| 參數 | 值 | 說明 |
|------|-----|------|
| 預測週期 | 5 日 | `Target = close.shift(-5) > close` |
| 最低持有 | 10 天 | 進場後至少持有 10 個交易日 |
| 做多門檻 | RF 機率 ≥ 0.48 | 信心度低於 48% 不進場 |
| 放空門檻 | RF 機率 ≤ 0.35 + 技術確認 | 需 RSI < 45 + MACD 死亡交叉 |
| 倉位大小 | = RF 機率 | 50% 信心 → 50% 倉位 |
| 趨勢濾網 | MA60 > MA120 | 空頭趨勢不抱多單 |
| RF 深度 | max_depth=3 | 限制過擬合 |
| RF 樹數 | n_estimators=200 | |
| XGBoost | n_estimators=500, lr=0.05 | 跨股票模型專用 |

## 掃描結果（109 檔電子股 + 航運）

跨股票 XGBoost 平均樣本外報酬 **+20.32%**，勝過單股 RandomForest 的 **+17.37%**（101 檔可交易股票，含真實交易成本）。
