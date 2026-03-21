"""
lead_lag_paper.py
=================
論文「部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略」
の完全再現スクリプト。単一ファイルで完結。

実行: python lead_lag_paper.py
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
# 定数（論文§4.1・§4.2）
# ============================================================
US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = [f"{t}.T" for t in range(1617, 1634)]  # 1617〜1633

# シクリカル・ディフェンシブ（論文§4.1）
US_CYC  = ["XLB","XLE","XLF","XLRE"]
US_DEF  = ["XLK","XLP","XLU","XLV"]
JP_CYC  = ["1618.T","1625.T","1629.T","1631.T"]
JP_DEF  = ["1617.T","1621.T","1627.T","1630.T"]

# ハイパーパラメータ（論文デフォルト）
LAM   = 0.9   # 正則化強度 λ（式13）
K     = 3     # 固有ベクトル数
K0    = 3     # 事前部分空間次元
L     = 60    # ローリングウィンドウ（営業日）
Q     = 0.3   # 分位点
PRIOR_END = "2014-12-31"   # C_full 推定期間の終端


# ============================================================
# Step 1: データ取得
# ============================================================
def load_data(start="2010-01-01", end="2025-12-31", cache_dir="data_cache"):
    """yfinance で Open/Close を取得し CC・OC リターンを返す"""
    import yfinance as yf

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    fp = cache / "raw.parquet"

    all_tickers = US_TICKERS + JP_TICKERS

    if fp.exists():
        log.info("キャッシュからロード")
        raw = pd.read_parquet(fp)
    else:
        log.info(f"yfinance ダウンロード: {start} 〜 {end}")
        raw = yf.download(all_tickers, start=start, end=end,
                          auto_adjust=True, progress=True, threads=True)
        raw.to_parquet(fp)
        log.info("キャッシュ保存完了")

    # Open / Close を抽出（MultiIndex or flat）
    def get_price(field):
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[field].copy()
        else:
            df = raw[[c for c in raw.columns if field in c]].copy()
        df.columns = [c if isinstance(c, str) else c[1] for c in df.columns]
        # 欠損を前後補完
        df = df.ffill().bfill()
        return df

    close = get_price("Close")[all_tickers]
    open_ = get_price("Open")[all_tickers]

    # 共通取引日（両方 NaN でない行）
    mask = close.notna().all(axis=1) & open_.notna().all(axis=1)
    close = close[mask]
    open_ = open_[mask]

    # CC リターン（式1）: 全銘柄
    r_cc = close.pct_change().dropna(how="all")

    # OC リターン（式2）: 日本のみ
    r_oc_jp = (close[JP_TICKERS] / open_[JP_TICKERS] - 1)

    # 共通日に揃える
    idx = r_cc.index.intersection(r_oc_jp.index)
    r_cc    = r_cc.loc[idx]
    r_oc_jp = r_oc_jp.loc[idx]

    # 極端な外れ値除去（±50%超）
    r_cc    = r_cc.where(r_cc.abs() < 0.5)
    r_oc_jp = r_oc_jp.where(r_oc_jp.abs() < 0.5)

    log.info(f"データ準備完了: {len(idx)} 営業日 | US={len(US_TICKERS)} JP={len(JP_TICKERS)}")
    return r_cc, r_oc_jp


# ============================================================
# Step 2: 事前部分空間 V0 と C0 の構築（論文§3.1, 式10〜12）
# ============================================================
def gram_schmidt(v, basis):
    """v を basis の各ベクトルに直交化"""
    for b in basis:
        v = v - np.dot(v, b) * b
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def build_V0():
    """
    事前固有ベクトル V0 ∈ R^{N×3} を構築する。
    v1: グローバル（全銘柄均等）
    v2: 国スプレッド（US+, JP-）
    v3: シクリカル・ディフェンシブ
    """
    all_tickers = US_TICKERS + JP_TICKERS
    N = len(all_tickers)
    idx = {t: i for i, t in enumerate(all_tickers)}

    # v1: グローバル
    v1 = np.ones(N) / np.sqrt(N)

    # v2: 国スプレッド → v1 に直交化
    v2_raw = np.zeros(N)
    for t in US_TICKERS: v2_raw[idx[t]] = +1.0
    for t in JP_TICKERS: v2_raw[idx[t]] = -1.0
    v2 = gram_schmidt(v2_raw, [v1])

    # v3: シクリカル・ディフェンシブ → v1, v2 に直交化
    v3_raw = np.zeros(N)
    for t in US_CYC + JP_CYC: v3_raw[idx[t]] = +1.0
    for t in US_DEF + JP_DEF: v3_raw[idx[t]] = -1.0
    v3 = gram_schmidt(v3_raw, [v1, v2])

    V0 = np.column_stack([v1, v2, v3])  # (N, 3)

    # 直交性チェック
    err = np.abs(V0.T @ V0 - np.eye(3)).max()
    log.info(f"V0 構築完了: shape={V0.shape}, 直交性誤差={err:.2e}")
    return V0


def build_C0(r_cc_prior: pd.DataFrame, V0: np.ndarray) -> np.ndarray:
    """
    事前エクスポージャー行列 C0 を構築する（式10〜12）。
    r_cc_prior: 事前期間（〜2014年）の CC リターン（US+JP）
    """
    all_tickers = US_TICKERS + JP_TICKERS

    # 標準化リターン → 相関行列（式 Cfull）
    r_prior = r_cc_prior[all_tickers].dropna(how='all')
    Z = (r_prior - r_prior.mean()) / r_prior.std()
    Z = Z.fillna(0.0)  # NaNを0で埋める
    C_full = Z.corr().values
    C_full = np.nan_to_num(C_full, nan=0.0)
    np.fill_diagonal(C_full, 1.0)

    # 式(10): D0 = diag(V0^T Cfull V0)
    D0 = np.diag(V0.T @ C_full @ V0)

    # 式(11): C0_raw = V0 D0 V0^T
    C0_raw = V0 @ np.diag(D0) @ V0.T

    # 式(12): 対角要素で正規化 → 相関行列
    diag_sqrt = np.sqrt(np.diag(C0_raw))
    diag_sqrt[diag_sqrt < 1e-12] = 1.0
    C0 = C0_raw / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(C0, 1.0)

    log.info(f"C0 構築完了: shape={C0.shape}, diag mean={np.diag(C0).mean():.4f}")
    return C0


# ============================================================
# Step 3: 部分空間正則化 PCA シグナル（論文§3.2〜3.3）
# ============================================================
def compute_pca_sub_signals(r_cc: pd.DataFrame,
                             r_oc_jp: pd.DataFrame,
                             C0: np.ndarray,
                             lam: float = LAM,
                             L: int = L,
                             K: int = K,
                             cache_path: str = "cache_pca_sub.parquet") -> pd.DataFrame:
    """
    全期間のローリング PCA SUB シグナルを計算する。
    シグナル zbJ[t] → t+1 日の OC に投資
    """
    fp = Path(cache_path)
    if fp.exists():
        log.info(f"PCA_SUB シグナル: キャッシュロード ({cache_path})")
        return pd.read_parquet(fp)

    n_us = len(US_TICKERS)
    n_jp = len(JP_TICKERS)
    all_tickers = US_TICKERS + JP_TICKERS

    # US+JP の CC リターンを結合（列順を固定）
    all_ret = r_cc[all_tickers].values   # (T, N)
    dates   = r_cc.index
    T       = len(dates)

    signals = np.full((T, n_jp), np.nan)

    log.info(f"PCA_SUB シグナル計算開始: {T} 日")
    for t in range(L, T):
        # ── ウィンドウ (t-L, t-1) ──
        W = all_ret[t-L:t, :]       # (L, N)

        # ── 標準化（式8〜9）──
        mu    = np.nanmean(W, axis=0)
        sigma = np.nanstd(W, axis=0, ddof=0)
        sigma[sigma < 1e-10] = 1.0
        Z = (W - mu) / sigma         # (L, N): NaN は 0 で扱う
        Z = np.nan_to_num(Z, nan=0.0)

        # ── 標本相関行列 Ct（式13の Ct）──
        # pandas の corr() が最も正確だが速度のため numpy で実装
        # 標準偏差がゼロの列を除いて相関行列を計算
        std = Z.std(axis=0)
        valid = std > 1e-10
        if valid.sum() < 3:
            continue
        Z_valid = Z[:, valid]
        Ct_small = np.corrcoef(Z_valid.T)
        # フルサイズに戻す
        Ct = np.eye(len(std))
        idx_v = np.where(valid)[0]
        for i, vi in enumerate(idx_v):
            for j, vj in enumerate(idx_v):
                Ct[vi, vj] = Ct_small[i, j]
        Ct = np.nan_to_num(Ct, nan=0.0)
        np.fill_diagonal(Ct, 1.0)
        Ct = (Ct + Ct.T) / 2

        # ── 正則化（式13）──
        C_reg = (1 - lam) * Ct + lam * C0
        C_reg = (C_reg + C_reg.T) / 2
        np.fill_diagonal(C_reg, 1.0)

        # ── 上位 K 固有ベクトル（式14〜16）──
        try:
            # eigh: 対称行列専用、昇順で返る
            vals, vecs = eigh(C_reg, subset_by_index=[len(C_reg)-K, len(C_reg)-1])
            # 降順に並べ替え
            order = np.argsort(vals)[::-1]
            V_K = vecs[:, order]     # (N, K)
        except Exception:
            continue

        V_U = V_K[:n_us, :]         # (n_us, K)
        V_J = V_K[n_us:, :]         # (n_jp, K)

        # ── 米国当日標準化リターン（式17）──
        r_us_today = all_ret[t, :n_us]
        z_us_t = (r_us_today - mu[:n_us]) / sigma[:n_us]
        z_us_t = np.nan_to_num(z_us_t, nan=0.0)

        # ── ファクタースコア（式18）──
        f_t = V_U.T @ z_us_t        # (K,)

        # ── 日本シグナル（式19〜20）──
        z_hat_J = V_J @ f_t         # (n_jp,)

        signals[t, :] = z_hat_J

        if (t - L) % 500 == 0:
            pct = (t - L) / (T - L) * 100
            log.info(f"  進捗: {pct:.0f}% ({dates[t].date()})")

    sig_df = pd.DataFrame(signals, index=dates, columns=JP_TICKERS)
    sig_df.to_parquet(fp)
    log.info(f"PCA_SUB 完了: {sig_df.dropna(how='all').shape[0]} 日分")
    return sig_df


def compute_pca_plain_signals(r_cc, L=L, K=K,
                               cache_path="cache_pca_plain.parquet"):
    fp = Path(cache_path)
    if fp.exists():
        log.info(f"PCA_PLAIN シグナル: キャッシュロード")
        return pd.read_parquet(fp)
    N = len(US_TICKERS) + len(JP_TICKERS)
    C0_dummy = np.eye(N)
    sig = compute_pca_sub_signals(r_cc, None, C0_dummy,
                                   lam=0.0, L=L, K=K,
                                   cache_path="cache_pca_plain_tmp.parquet")
    sig.to_parquet(fp)
    return sig


def compute_mom_signals(r_cc_jp: pd.DataFrame, L: int = L) -> pd.DataFrame:
    """MOM: 日本株 CC リターンのローリング平均（式31）"""
    sig = r_cc_jp[JP_TICKERS].rolling(L).mean()
    log.info("MOM シグナル計算完了")
    return sig


# ============================================================
# Step 4: ロング・ショートポートフォリオ（論文§2.2, 式3〜7）
# ============================================================
def build_portfolio(signal_df: pd.DataFrame,
                    r_oc_jp: pd.DataFrame,
                    q: float = Q) -> pd.Series:
    """
    シグナル t 日 → t+1 日 OC リターンに投資するロング・ショート戦略。
    """
    # シグナルを 1 日先送り（t 日シグナル → t+1 日に執行）
    sig_shifted = signal_df.shift(1)

    rets, dates = [], []
    for date in sig_shifted.index:
        s = sig_shifted.loc[date].dropna()
        if len(s) < 4 or date not in r_oc_jp.index:
            continue

        n = len(s)
        n_pos = max(1, int(np.ceil(n * q)))

        sorted_idx = s.sort_values(ascending=False).index
        long_set  = sorted_idx[:n_pos]     # 式(3): 上位 q
        short_set = sorted_idx[-n_pos:]    # 式(4): 下位 q

        # 等ウェイト（式5〜6）
        w = pd.Series(0.0, index=s.index)
        w[long_set]  = +1.0 / len(long_set)
        w[short_set] = -1.0 / len(short_set)

        r = r_oc_jp.loc[date, s.index]
        R = (w * r).sum()               # 式(7)

        if np.isfinite(R):
            rets.append(R)
            dates.append(date)

    return pd.Series(rets, index=pd.DatetimeIndex(dates))


def build_double_portfolio(sig_pca: pd.DataFrame,
                            sig_mom: pd.DataFrame,
                            r_oc_jp: pd.DataFrame) -> pd.Series:
    """DOUBLE: PCA_SUB × MOM の 2×2 ダブルソート（論文§4.3）"""
    # 各シグナルを 1 日先送り
    pca_sh = sig_pca.shift(1)
    mom_sh = sig_mom.shift(1)

    rets, dates = [], []
    for date in pca_sh.index:
        pca = pca_sh.loc[date].dropna()
        mom = mom_sh.loc[date].dropna()
        common = pca.index.intersection(mom.index)
        if len(common) < 4 or date not in r_oc_jp.index:
            continue

        pca, mom = pca[common], mom[common]

        # メジアンで High/Low に二分割
        pca_high = pca >= pca.median()
        mom_high = mom >= mom.median()

        long_set  = common[pca_high.values & mom_high.values]
        short_set = common[(~pca_high.values) & (~mom_high.values)]

        if len(long_set) == 0 or len(short_set) == 0:
            continue

        w = pd.Series(0.0, index=common)
        w[long_set]  = +1.0 / len(long_set)
        w[short_set] = -1.0 / len(short_set)

        R = (w * r_oc_jp.loc[date, common]).sum()
        if np.isfinite(R):
            rets.append(R)
            dates.append(date)

    return pd.Series(rets, index=pd.DatetimeIndex(dates))


# ============================================================
# Step 5: パフォーマンス評価（論文§4.2, 式27〜30）
# ============================================================
def performance(r: pd.Series, ann: int = 252) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"AR": np.nan, "RISK": np.nan, "R/R": np.nan, "MDD": np.nan, "N": 0}

    ar   = r.mean() * ann * 100                          # 式(27)
    risk = r.std()  * np.sqrt(ann) * 100                 # 式(28)
    rr   = ar / risk if risk > 0 else np.nan             # 式(29)

    cum  = (1 + r).cumprod()
    mdd  = ((cum / cum.cummax()) - 1).min() * 100        # 式(30)

    return {"AR(%)": round(ar,2), "RISK(%)": round(risk,2),
            "R/R": round(rr,3), "MDD(%)": round(mdd,2), "N": len(r)}


def print_table(strategies: dict):
    rows = []
    for name, r in strategies.items():
        m = performance(r)
        m["Strategy"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Strategy")
    cols = ["AR(%)", "RISK(%)", "R/R", "MDD(%)", "N"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "="*60)
    print("  パフォーマンス評価（論文 Table 2 相当）")
    print("="*60)
    print(df[cols].to_string())
    print("="*60)


# ============================================================
# Step 6: グラフ
# ============================================================
def plot_results(strategies: dict, save_dir: str = "output"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    # 日本語フォント警告を抑制
    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    Path(save_dir).mkdir(exist_ok=True)

    colors = {"MOM": "#aaaaaa", "PCA_PLAIN": "#4EA8DE",
              "PCA_SUB": "#7B2D8B", "DOUBLE": "#E67E22"}
    styles = {"MOM": ":", "PCA_PLAIN": "--", "PCA_SUB": "-", "DOUBLE": "-"}

    # 累積リターン
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in strategies.items():
        cum = (1 + r.dropna()).cumprod()
        ax.plot(cum.index, cum.values, label=name,
                color=colors.get(name, "gray"),
                linestyle=styles.get(name, "-"), linewidth=1.8)
    ax.set_title("Cumulative returns by strategy (paper replication)")
    ax.set_ylabel("Cumulative wealth (start=1)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    p = f"{save_dir}/cumulative.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    log.info(f"保存: {p}")

    # 年次リターン
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, (name, r) in zip(axes.flat, strategies.items()):
        annual = r.dropna().resample("YE").sum() * 100
        clrs = ["#7B2D8B" if v >= 0 else "#E74C3C" for v in annual.values]
        ax.bar(annual.index.year, annual.values, color=clrs, width=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(name)
        ax.set_ylabel("Annual return (%)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Annual returns by strategy")
    fig.tight_layout()
    p = f"{save_dir}/annual.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    log.info(f"保存: {p}")


# ============================================================
# メイン
# ============================================================
def main():
    log.info("="*60)
    log.info("日米業種リードラグ バックテスト（論文完全再現版）")
    log.info("="*60)

    # ── 1. データ取得 ──
    log.info("[Step 1] データ取得")
    r_cc, r_oc_jp = load_data()

    # ── 2. 事前部分空間 ──
    log.info("[Step 2] 事前部分空間 V0 / C0 構築")
    V0 = build_V0()
    prior_mask = r_cc.index <= PRIOR_END
    C0 = build_C0(r_cc[prior_mask], V0)

    # ── 3. シグナル生成 ──
    log.info("[Step 3] シグナル生成")

    log.info("  MOM...")
    sig_mom   = compute_mom_signals(r_cc[JP_TICKERS])

    log.info("  PCA_PLAIN（正則化なし）...")
    sig_plain = compute_pca_plain_signals(r_cc)

    log.info("  PCA_SUB（提案手法、最長数分）...")
    sig_sub   = compute_pca_sub_signals(r_cc, r_oc_jp, C0)

    # ── 4. ポートフォリオ構築 ──
    log.info("[Step 4] ポートフォリオ構築")
    port_mom   = build_portfolio(sig_mom,   r_oc_jp)
    port_plain = build_portfolio(sig_plain, r_oc_jp)
    port_sub   = build_portfolio(sig_sub,   r_oc_jp)
    port_dbl   = build_double_portfolio(sig_sub, sig_mom, r_oc_jp)

    strategies = {
        "MOM":       port_mom,
        "PCA_PLAIN": port_plain,
        "PCA_SUB":   port_sub,
        "DOUBLE":    port_dbl,
    }

    # ── 5. 評価 ──
    log.info("[Step 5] パフォーマンス評価")
    print_table(strategies)

    # CSV 保存
    Path("output").mkdir(exist_ok=True)
    pd.DataFrame(strategies).to_csv("output/returns.csv")
    log.info("戦略リターン保存: output/returns.csv")

    # ── 6. グラフ ──
    log.info("[Step 6] グラフ描画")
    plot_results(strategies)

    log.info("完了！output/ フォルダにグラフを保存しました。")


if __name__ == "__main__":
    main()
