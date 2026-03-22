"""
lead_lag_paper_v3.py
====================
Phase 2+3: LightGBM + テクニカル指標 + 動的ポジションサイジング

追加機能:
  - RSI / MACD / ボリンジャーバンド
  - LightGBM でシグナル統合・精度向上
  - シグナル強度に応じたポジションサイズ変動
  - VIX・為替・日経によるリスクフィルター継続
  - 仮想資金: 1億円
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ============================================================
# 定数
# ============================================================
US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = [f"{t}.T" for t in range(1617, 1634)]
US_CYC  = ["XLB","XLE","XLF","XLRE"]
US_DEF  = ["XLK","XLP","XLU","XLV"]
JP_CYC  = ["1618.T","1625.T","1629.T","1631.T"]
JP_DEF  = ["1617.T","1621.T","1627.T","1630.T"]

MACRO_TICKERS = {
    "USDJPY": "USDJPY=X",
    "VIX":    "^VIX",
    "NIKKEI": "^N225",
}

LAM       = 0.9
K         = 3
L         = 60
Q         = 0.3
PRIOR_END = "2014-12-31"
TRAIN_END = "2018-12-31"   # LightGBM 学習期間の終端
LGBM_SKIP_THRESHOLD = 0.35  # この確率以下の日はスキップ
CAPITAL   = 1e8            # 1億円

# リスクフィルター閾値
VIX_HIGH    = 25.0
VIX_EXTREME = 35.0
NIKKEI_DROP = -0.02
USDJPY_MOVE = 0.015


# ============================================================
# Step 1: データ取得
# ============================================================
def load_data(start="2010-01-01", end=None, cache_dir="data_cache"):
    from datetime import date
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    import yfinance as yf
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)

    fp_etf = cache / "raw.parquet"
    all_tickers = US_TICKERS + JP_TICKERS
    if fp_etf.exists():
        raw = pd.read_parquet(fp_etf)
        cached_end = raw.index.max().strftime("%Y-%m-%d")
        today = date.today().strftime("%Y-%m-%d")
        if cached_end >= today:
            log.info(f"ETFキャッシュからロード（最新: {cached_end}）")
        else:
            log.info(f"キャッシュ更新中（{cached_end} → {today}）")
            new = yf.download(US_TICKERS + JP_TICKERS,
                              start=cached_end, end=end,
                              auto_adjust=True, progress=False, threads=True)
            raw = pd.concat([raw, new]).sort_index()
            raw = raw[~raw.index.duplicated(keep="last")]
            raw.to_parquet(fp_etf)
    else:
        log.info(f"ETFダウンロード: {start} 〜 {end}")
        raw = yf.download(all_tickers, start=start, end=end,
                          auto_adjust=True, progress=True, threads=True)
        raw.to_parquet(fp_etf)

    def get_price(field):
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[field].copy()
        else:
            df = raw[[c for c in raw.columns if field in c]].copy()
        df.columns = [c if isinstance(c, str) else c[1] for c in df.columns]
        return df.ffill().bfill()

    close = get_price("Close")[all_tickers]
    open_ = get_price("Open")[all_tickers]
    mask  = close.notna().all(axis=1) & open_.notna().all(axis=1)
    close, open_ = close[mask], open_[mask]

    r_cc    = close.pct_change().dropna(how="all")
    r_oc_jp = (close[JP_TICKERS] / open_[JP_TICKERS] - 1)
    idx     = r_cc.index.intersection(r_oc_jp.index)
    r_cc    = r_cc.loc[idx].where(r_cc.abs() < 0.5)
    r_oc_jp = r_oc_jp.loc[idx].where(r_oc_jp.abs() < 0.5)

    macro_df = load_macro_data(start, end, cache_dir)
    log.info(f"データ準備完了: {len(idx)} 営業日")
    return r_cc, r_oc_jp, macro_df, close


def load_macro_data(start="2010-01-01", end=None, cache_dir="data_cache"):
    from datetime import date
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    import yfinance as yf
    fp = Path(cache_dir) / "macro.parquet"
    if fp.exists():
        macro = pd.read_parquet(fp)
        cached_end = macro.index.max().strftime("%Y-%m-%d")
        today = date.today().strftime("%Y-%m-%d")
        if cached_end >= today:
            log.info(f"マクロデータキャッシュからロード（最新: {cached_end}）")
            return macro
        else:
            log.info(f"マクロデータ更新中（{cached_end} → {today}）")
            # 以降は新規ダウンロード処理に続く

    log.info("マクロデータダウンロード中...")
    tickers = list(MACRO_TICKERS.values())
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[[c for c in raw.columns if "Close" in c]].copy()

    rename = {v: k for k, v in MACRO_TICKERS.items()}
    close.columns = [rename.get(c, c) if isinstance(c, str) else rename.get(c[1], c[1])
                     for c in close.columns]
    close = close.ffill().bfill()
    macro_ret = close.pct_change()

    macro = pd.DataFrame({
        "USDJPY":     close.get("USDJPY"),
        "VIX":        close.get("VIX"),
        "NIKKEI":     close.get("NIKKEI"),
        "USDJPY_ret": macro_ret.get("USDJPY"),
        "NIKKEI_ret": macro_ret.get("NIKKEI"),
    })
    macro.to_parquet(fp)
    log.info(f"マクロデータ保存: {len(macro)} 日")
    return macro


# ============================================================
# Step 2: テクニカル指標計算（新機能）
# ============================================================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI（相対力指数）"""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD"""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": histogram})


def calc_bollinger(series: pd.Series, period: int = 20, std: int = 2) -> pd.DataFrame:
    """ボリンジャーバンド"""
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    pct_b = (series - lower) / (upper - lower + 1e-10)   # %B
    width = (upper - lower) / (mid + 1e-10)               # バンド幅
    return pd.DataFrame({"mid": mid, "upper": upper,
                          "lower": lower, "pct_b": pct_b, "width": width})


def compute_technical_features(close_prices: pd.DataFrame,
                                 tickers: list) -> pd.DataFrame:
    """
    全ティッカーのテクニカル指標を計算して結合する。
    戻り値: DataFrame (index=date, columns=ticker_指標名)
    """
    log.info("テクニカル指標を計算中...")
    feats = {}

    for ticker in tickers:
        if ticker not in close_prices.columns:
            continue
        price = close_prices[ticker].dropna()

        # RSI
        feats[f"{ticker}_rsi"] = calc_rsi(price)

        # MACD
        macd_df = calc_macd(price)
        feats[f"{ticker}_macd_hist"] = macd_df["hist"]

        # ボリンジャー %B
        bb_df = calc_bollinger(price)
        feats[f"{ticker}_bb_pctb"] = bb_df["pct_b"]
        feats[f"{ticker}_bb_width"] = bb_df["width"]

        # モメンタム（5日・20日）
        feats[f"{ticker}_mom5"]  = price.pct_change(5)
        feats[f"{ticker}_mom20"] = price.pct_change(20)

    feat_df = pd.DataFrame(feats)
    log.info(f"テクニカル指標完了: {feat_df.shape[1]} 特徴量")
    return feat_df


# ============================================================
# Step 3: 事前部分空間・PCAシグナル
# ============================================================
def gram_schmidt(v, basis):
    for b in basis:
        v = v - np.dot(v, b) * b
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def build_V0():
    all_tickers = US_TICKERS + JP_TICKERS
    N   = len(all_tickers)
    idx = {t: i for i, t in enumerate(all_tickers)}
    v1  = np.ones(N) / np.sqrt(N)
    v2_raw = np.zeros(N)
    for t in US_TICKERS: v2_raw[idx[t]] = +1.0
    for t in JP_TICKERS: v2_raw[idx[t]] = -1.0
    v2 = gram_schmidt(v2_raw, [v1])
    v3_raw = np.zeros(N)
    for t in US_CYC + JP_CYC: v3_raw[idx[t]] = +1.0
    for t in US_DEF + JP_DEF: v3_raw[idx[t]] = -1.0
    v3 = gram_schmidt(v3_raw, [v1, v2])
    V0  = np.column_stack([v1, v2, v3])
    log.info(f"V0 構築完了: shape={V0.shape}")
    return V0


def build_C0(r_cc_prior, V0):
    all_tickers = US_TICKERS + JP_TICKERS
    r_prior = r_cc_prior[all_tickers].dropna(how="all")
    Z       = (r_prior - r_prior.mean()) / r_prior.std()
    Z       = Z.fillna(0.0)
    C_full  = np.nan_to_num(Z.corr().values, nan=0.0)
    np.fill_diagonal(C_full, 1.0)
    D0      = np.diag(V0.T @ C_full @ V0)
    C0_raw  = V0 @ np.diag(D0) @ V0.T
    diag_sqrt = np.sqrt(np.diag(C0_raw))
    diag_sqrt[diag_sqrt < 1e-12] = 1.0
    C0 = C0_raw / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(C0, 1.0)
    log.info(f"C0 構築完了: shape={C0.shape}")
    return C0


def compute_pca_sub_signals(r_cc, r_oc_jp, C0,
                             lam=LAM, L=L, K=K,
                             cache_path="cache_pca_sub.parquet"):
    fp = Path(cache_path)
    if fp.exists():
        log.info("PCA_SUB: キャッシュロード")
        return pd.read_parquet(fp)

    n_us = len(US_TICKERS); n_jp = len(JP_TICKERS)
    all_tickers = US_TICKERS + JP_TICKERS
    all_ret = r_cc[all_tickers].values
    dates = r_cc.index; T = len(dates)
    signals = np.full((T, n_jp), np.nan)

    log.info(f"PCA_SUB 計算開始: {T} 日")
    for t in range(L, T):
        W = all_ret[t-L:t, :]
        mu = np.nanmean(W, axis=0)
        sigma = np.nanstd(W, axis=0, ddof=0)
        sigma[sigma < 1e-10] = 1.0
        Z = np.nan_to_num((W - mu) / sigma, nan=0.0)

        std = Z.std(axis=0); valid = std > 1e-10
        if valid.sum() < 3: continue
        Ct_s = np.corrcoef(Z[:, valid].T)
        Ct = np.eye(len(std)); iv = np.where(valid)[0]
        for i, vi in enumerate(iv):
            for j, vj in enumerate(iv): Ct[vi, vj] = Ct_s[i, j]
        Ct = np.nan_to_num(Ct, nan=0.0)
        np.fill_diagonal(Ct, 1.0); Ct = (Ct + Ct.T) / 2

        C_reg = np.nan_to_num((1-lam)*Ct + lam*C0, nan=0.0)
        np.fill_diagonal(C_reg, 1.0); C_reg = (C_reg+C_reg.T)/2
        np.fill_diagonal(C_reg, 1.0)

        try:
            N = len(all_tickers)
            vals, vecs = eigh(C_reg, subset_by_index=[N-K, N-1])
            V_K = vecs[:, np.argsort(vals)[::-1]]
        except Exception: continue

        V_U = V_K[:n_us]; V_J = V_K[n_us:]
        z_us  = np.nan_to_num((all_ret[t,:n_us]-mu[:n_us])/sigma[:n_us], nan=0.0)
        z_hat = V_J @ (V_U.T @ z_us)
        signals[t, :] = z_hat

        if (t-L) % 500 == 0:
            log.info(f"  進捗: {(t-L)/(T-L)*100:.0f}% ({dates[t].date()})")

    sig_df = pd.DataFrame(signals, index=dates, columns=JP_TICKERS)
    sig_df.to_parquet(fp)
    log.info(f"PCA_SUB 完了: {sig_df.dropna(how='all').shape[0]} 日分")
    return sig_df


# ============================================================
# Step 4: LightGBM 特徴量エンジニアリング
# ============================================================
def build_lgbm_features(pca_signals:  pd.DataFrame,
                         tech_feats:   pd.DataFrame,
                         macro_df:     pd.DataFrame,
                         r_cc:         pd.DataFrame) -> pd.DataFrame:
    """
    LightGBM の入力特徴量を構築する。

    特徴量:
      - PCA_SUB シグナル（各日本ETFの予測値）
      - テクニカル指標（RSI, MACD, BB）
      - マクロ指標（VIX, 為替, 日経）
      - 米国ETFのリターン（当日）
    """
    all_feats = []

    # PCA シグナルを1日シフト（t日シグナル → t+1日特徴量）
    pca_shifted = pca_signals.shift(1)
    pca_shifted.columns = [f"pca_{c}" for c in pca_shifted.columns]
    all_feats.append(pca_shifted)

    # テクニカル指標（JP ETF）
    jp_tech_cols = [c for c in tech_feats.columns
                    if any(t in c for t in JP_TICKERS)]
    if jp_tech_cols:
        jp_tech = tech_feats[jp_tech_cols].shift(1)
        all_feats.append(jp_tech)

    # US ETF テクニカル指標（当日）
    us_tech_cols = [c for c in tech_feats.columns
                    if any(t in c for t in US_TICKERS)]
    if us_tech_cols:
        all_feats.append(tech_feats[us_tech_cols])

    # マクロ指標
    if macro_df is not None:
        macro_feats = macro_df[["VIX", "USDJPY", "NIKKEI_ret", "USDJPY_ret"]].copy()
        # VIX の変化率
        macro_feats["VIX_ret"] = macro_df["VIX"].pct_change()
        # VIX レベル区分
        macro_feats["VIX_regime"] = pd.cut(
            macro_df["VIX"],
            bins=[0, 15, 20, 25, 35, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        all_feats.append(macro_feats)

    # 米国ETFの当日リターン
    us_ret = r_cc[US_TICKERS].copy()
    us_ret.columns = [f"us_ret_{c}" for c in us_ret.columns]
    all_feats.append(us_ret)

    # 結合
    feat_df = pd.concat(all_feats, axis=1)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    log.info(f"特徴量構築完了: {feat_df.shape[1]} 特徴量")
    return feat_df


# ============================================================
# Step 5: LightGBM 学習・予測（新機能）
# ============================================================
def train_lgbm_signals(feat_df, r_oc_jp, train_end=TRAIN_END):
    """
    LightGBM をリスク管理用に使う。
    「この日は取引しない（損失リスクが高い）」を予測する。
    戻り値: {ticker: model}
    """
    import lightgbm as lgb
    models = {}
    train_mask = feat_df.index <= train_end
    log.info(f"LightGBM リスク管理モデル学習: 〜{train_end}")
    log.info(f"  学習期間: {train_mask.sum()}日")

    for ticker in JP_TICKERS:
        if ticker not in r_oc_jp.columns:
            continue

        # 目標変数: 翌日リターンが正かどうか（取引して良い日=1）
        y_full = (r_oc_jp[ticker].shift(-1) > 0).astype(int)
        common = feat_df.index.intersection(y_full.dropna().index)
        X = feat_df.loc[common].fillna(0)
        y = y_full.loc[common]

        train_idx = common[common <= train_end]
        if len(train_idx) < 200:
            continue

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            verbose=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        models[ticker] = model

    log.info(f"LightGBM 学習完了: {len(models)}銘柄")
    return models


def predict_lgbm_signals(models, feat_df, test_start="2019-01-01"):
    """
    各銘柄の「取引OK確率」を予測する。
    低い銘柄はポートフォリオから除外する。
    """
    test_idx = feat_df.index[feat_df.index >= test_start]
    X_test   = feat_df.loc[test_idx].fillna(0)
    trade_ok = pd.DataFrame(index=test_idx, columns=JP_TICKERS, dtype=float)

    for ticker, model in models.items():
        try:
            probs = model.predict_proba(X_test)[:, 1]
            trade_ok[ticker] = probs
        except Exception:
            trade_ok[ticker] = 0.5

    log.info(f"LightGBM 予測完了: {len(trade_ok)} 日")
    return trade_ok


# ============================================================
# Step 6: リスクフィルター（v2から継承）
# ============================================================
def compute_risk_multiplier(date, macro_df):
    if macro_df is None or date not in macro_df.index:
        return {"multiplier": 1.0, "reason": "マクロデータなし"}

    row = macro_df.loc[date]
    mult, reasons = 1.0, []

    vix = row.get("VIX", np.nan)
    if pd.notna(vix):
        if vix >= VIX_EXTREME:
            mult = 0.0
            reasons.append(f"VIX極高({vix:.1f})→停止")
        elif vix >= VIX_HIGH:
            scale = 1.0 - (vix - VIX_HIGH) / (VIX_EXTREME - VIX_HIGH)
            mult  = min(mult, max(0.2, scale))
            reasons.append(f"VIX高({vix:.1f})→{mult*100:.0f}%")

    nikkei_ret = row.get("NIKKEI_ret", np.nan)
    if pd.notna(nikkei_ret) and nikkei_ret <= NIKKEI_DROP:
        mult = min(mult, 0.5)
        reasons.append(f"日経下落({nikkei_ret*100:.1f}%)")

    usdjpy_ret = row.get("USDJPY_ret", np.nan)
    if pd.notna(usdjpy_ret) and abs(usdjpy_ret) >= USDJPY_MOVE:
        mult = min(mult, 0.7)
        reasons.append(f"円急変({usdjpy_ret*100:.1f}%)")

    reason = "、".join(reasons) if reasons else "通常"
    return {"multiplier": round(mult, 2), "reason": reason}


# ============================================================
# Step 7: ポートフォリオ構築（動的ポジションサイジング）
# ============================================================
def build_portfolio_v3(signal_df:  pd.DataFrame,
                        r_oc_jp:   pd.DataFrame,
                        macro_df:  pd.DataFrame,
                        q:         float = Q,
                        capital:   float = CAPITAL,
                        use_dynamic_sizing: bool = True,
                        lgbm_filter=None) -> tuple:
    """
    動的ポジションサイジング付きポートフォリオ。

    シグナルが強いほど大きくポジションを張る。
    """
    sig_shifted = signal_df.shift(1) if signal_df.index.equals(r_oc_jp.index) \
                  else signal_df  # LightGBMは既にシフト済み

    rets, dates, risk_log = [], [], []

    for date in sig_shifted.index:
        s = sig_shifted.loc[date].dropna()
        # LGBMフィルター: 取引OK確率が低い銘柄を除外
        if lgbm_filter is not None and date in lgbm_filter.index:
            ok_prob = lgbm_filter.loc[date]
            s = s[ok_prob.reindex(s.index).fillna(0.5) >= LGBM_SKIP_THRESHOLD]
        
        if len(s) < 4:
            continue

        # リスクフィルター
        risk  = compute_risk_multiplier(date, macro_df)
        mult  = risk["multiplier"]
        if mult == 0.0:
            risk_log.append({"date": date, "multiplier": 0.0,
                              "reason": risk["reason"]})
            continue

        n_pos      = max(1, int(np.ceil(len(s) * q)))
        sorted_idx = s.sort_values(ascending=False).index
        long_set   = sorted_idx[:n_pos]
        short_set  = sorted_idx[-n_pos:]

        if use_dynamic_sizing:
            # シグナル強度に応じてウェイトを調整
            sig_abs   = s.abs()
            sig_mean  = sig_abs.mean()
            sig_std   = sig_abs.std() + 1e-10

            w = pd.Series(0.0, index=s.index)
            MAX_WEIGHT = 0.20  # 1銘柄最大20%
            for t in long_set:
                strength = min(2.0, max(0.5, sig_abs[t] / (sig_mean + 1e-10)))
                w[t]     = +mult * strength
            for t in short_set:
                strength = min(2.0, max(0.5, sig_abs[t] / (sig_mean + 1e-10)))
                w[t]     = -mult * strength


            # 合計が2になるよう正規化
            pos_sum = w[w > 0].sum()
            neg_sum = w[w < 0].abs().sum()
            if pos_sum > 0: w[w > 0] = w[w > 0] / pos_sum
            if neg_sum > 0: w[w < 0] = w[w < 0] / neg_sum
        else:
            w = pd.Series(0.0, index=s.index)
            w[long_set]  = +mult / len(long_set)
            w[short_set] = -mult / len(short_set)

        r = r_oc_jp.loc[date, s.index]
        R = (w * r).sum()

        if np.isfinite(R):
            rets.append(R)
            dates.append(date)
            risk_log.append({"date": date, "multiplier": mult,
                             "reason": risk["reason"]})

    port_ret = pd.Series(rets, index=pd.DatetimeIndex(dates))
    risk_df  = pd.DataFrame(risk_log).set_index("date") if risk_log else pd.DataFrame()
    return port_ret, risk_df


# ============================================================
# Step 8: パフォーマンス評価
# ============================================================
def performance(r, ann=252, capital=CAPITAL):
    r = r.dropna()
    if len(r) == 0:
        return {k: np.nan for k in ["AR(%)", "RISK(%)", "R/R", "MDD(%)", "N", "最終資産(億円)"]}
    ar   = r.mean() * ann * 100
    risk = r.std()  * np.sqrt(ann) * 100
    rr   = ar / risk if risk > 0 else np.nan
    cum  = (1 + r).cumprod()
    mdd  = ((cum / cum.cummax()) - 1).min() * 100
    final = capital * cum.iloc[-1] / 1e8
    return {"AR(%)": round(ar,2), "RISK(%)": round(risk,2),
            "R/R": round(rr,3), "MDD(%)": round(mdd,2),
            "N": len(r), "最終資産(億円)": round(final,2)}


# ============================================================
# Step 9: グラフ
# ============================================================
def plot_results_v3(strategies: dict, macro_df, risk_df, save_dir="output"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    Path(save_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 14))

    # ① 累積リターン
    ax = axes[0]
    colors = {
        "PCA_SUB v1":     "#9B59B6",
        "PCA_SUB v2":     "#6C3483",
        "LightGBM v3":    "#E74C3C",
        "Ensemble v3":    "#E67E22",
    }
    for name, r in strategies.items():
        if r is None or len(r.dropna()) == 0: continue
        cum = (1 + r.dropna()).cumprod()
        lw  = 2.5 if "v3" in name else 1.5
        ls  = "-" if "v3" in name else "--"
        ax.plot(cum.index, cum.values, label=name,
                color=colors.get(name, "gray"),
                linewidth=lw, linestyle=ls, alpha=0.9)
    ax.set_title("累積リターン比較（1億円スタート）")
    ax.set_ylabel("累積倍率")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.3)

    # ② VIX 推移
    ax = axes[1]
    if macro_df is not None and "VIX" in macro_df.columns:
        vix = macro_df["VIX"].dropna()
        ax.plot(vix.index, vix.values, color="#E74C3C", linewidth=0.8)
        ax.fill_between(vix.index, vix.values, VIX_HIGH,
                        where=(vix.values >= VIX_HIGH),
                        alpha=0.3, color="orange", label=f"VIX≥{VIX_HIGH}")
        ax.fill_between(vix.index, vix.values, VIX_EXTREME,
                        where=(vix.values >= VIX_EXTREME),
                        alpha=0.5, color="red", label=f"VIX≥{VIX_EXTREME}")
        ax.axhline(VIX_HIGH,    color="orange", linestyle="--", linewidth=0.8)
        ax.axhline(VIX_EXTREME, color="red",    linestyle="--", linewidth=0.8)
        ax.set_title("VIX推移（リスクフィルター）")
        ax.set_ylabel("VIX")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

    # ③ ポジション乗数
    ax = axes[2]
    if risk_df is not None and not risk_df.empty and "multiplier" in risk_df.columns:
        mult = risk_df["multiplier"]
        ax.fill_between(mult.index, mult.values, alpha=0.4, color="#7B2D8B")
        ax.plot(mult.index, mult.values, color="#7B2D8B", linewidth=0.8)
        ax.set_title("ポジションサイズ乗数（1.0=フル、0.0=停止）")
        ax.set_ylabel("乗数")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(risk_df.index.min(), risk_df.index.max())
        ax.grid(alpha=0.3)

    fig.tight_layout()
    p = f"{save_dir}/comparison_v3.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    log.info(f"保存: {p}")


# ============================================================
# メイン
# ============================================================
def main():
    log.info("="*60)
    log.info("日米リードラグ v3（LightGBM + テクニカル + 動的サイジング）")
    log.info(f"仮想資金: {CAPITAL/1e8:.0f}億円")
    log.info("="*60)

    # ── 1. データ取得 ──
    log.info("[Step 1] データ取得")
    r_cc, r_oc_jp, macro_df, close_prices = load_data()

    # ── 2. 事前部分空間 ──
    log.info("[Step 2] 事前部分空間構築")
    V0 = build_V0()
    C0 = build_C0(r_cc[r_cc.index <= PRIOR_END], V0)

    # ── 3. PCA_SUB シグナル ──
    log.info("[Step 3] PCA_SUBシグナル計算")
    sig_pca = compute_pca_sub_signals(r_cc, r_oc_jp, C0)

    # ── 4. テクニカル指標 ──
    log.info("[Step 4] テクニカル指標計算")
    all_tickers = US_TICKERS + JP_TICKERS
    tech_feats  = compute_technical_features(close_prices, all_tickers)

    # ── 5. LightGBM 特徴量 & 学習 ──
    log.info("[Step 5] LightGBM 特徴量構築・学習")
    feat_df = build_lgbm_features(sig_pca, tech_feats, macro_df, r_cc)
    models  = train_lgbm_signals(feat_df, r_oc_jp, train_end=TRAIN_END)

    # ── 6. LightGBM 予測シグナル ──
    log.info("[Step 6] LightGBM 予測")
    lgbm_signals = predict_lgbm_signals(models, feat_df, test_start="2019-01-01")

    # ── 7. アンサンブルシグナル（PCA + LightGBM）──
    log.info("[Step 7] アンサンブルシグナル構築")
    common_idx  = sig_pca.index.intersection(lgbm_signals.index)
    common_cols = sig_pca.columns.intersection(lgbm_signals.columns)

    # 各シグナルをランクに変換して統合
    pca_rank  = sig_pca.loc[common_idx, common_cols].rank(axis=1, pct=True)
    lgbm_rank = lgbm_signals.loc[common_idx, common_cols].rank(axis=1, pct=True)
    ensemble  = 0.5 * pca_rank + 0.5 * lgbm_rank   # 50:50 ブレンド

    # ── 8. ポートフォリオ構築 ──
    log.info("[Step 8] ポートフォリオ構築")

    # v1: 基本版（比較用）
    from lead_lag_paper import build_portfolio
    port_v1, _ = build_portfolio(sig_pca, r_oc_jp), None
    port_v1    = build_portfolio(sig_pca, r_oc_jp)

    # v2: リスクフィルター付き
    port_v2, risk_df_v2 = build_portfolio_v3(
        sig_pca, r_oc_jp, macro_df,
        use_dynamic_sizing=False
    )

    # v3: PCA_SUB + LGBMフィルター
    port_lgbm, risk_df_lgbm = build_portfolio_v3(
        sig_pca, r_oc_jp, macro_df,
        use_dynamic_sizing=True,
        lgbm_filter=lgbm_signals   # ← フィルターとして使う
    )

    # アンサンブルはそのまま
    port_ens, risk_df_ens = build_portfolio_v3(
        ensemble, r_oc_jp, macro_df,
        use_dynamic_sizing=True
    )

    strategies = {
        "PCA_SUB v1":       port_v1,
        "PCA_SUB v2":       port_v2,
        "PCA+LGBM filter":  port_lgbm,   # ← 名前変更
        "Ensemble v3":      port_ens,
    }

    # ── 9. 結果表示 ──
    log.info("[Step 9] パフォーマンス評価")
    rows = []
    for name, r in strategies.items():
        if r is None: continue
        m = performance(r)
        m["Strategy"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "="*70)
    print(f"  パフォーマンス比較（仮想資金 {CAPITAL/1e8:.0f}億円）")
    print("="*70)
    print(df.to_string())

    # LightGBM 重要特徴量（PCA_SUB の代表銘柄）
    print("\n" + "="*70)
    print("  LightGBM 重要特徴量 TOP10（代表銘柄: 1625.T）")
    print("="*70)
    if "1625.T" in models:
        model = models["1625.T"]
        imp   = pd.Series(model.feature_importances_,
                          index=feat_df.columns).sort_values(ascending=False)
        print(imp.head(10).to_string())

    # CSV 保存
    Path("output").mkdir(exist_ok=True)
    pd.DataFrame(strategies).to_csv("output/returns_v3.csv")
    if not risk_df_ens.empty:
        risk_df_ens.to_csv("output/risk_log_v3.csv")
    log.info("CSV保存完了")

    # グラフ
    log.info("[Step 10] グラフ描画")
    plot_results_v3(strategies, macro_df, risk_df_ens)

    log.info("完了！output/comparison_v3.png を確認してください。")
    return strategies, df


if __name__ == "__main__":
    main()
