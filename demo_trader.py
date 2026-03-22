"""
demo_trader.py - 日米業種リードラグ デモトレードシステム
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, logging, os, webbrowser
from datetime import datetime, date
from pathlib import Path
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

VIRTUAL_CAPITAL     = 10_000_000
TRADE_LOG_PATH      = Path("demo_trade_log.json")
REPORT_PATH         = Path("demo_report.html")
SIGNAL_PATH         = Path("demo_signal_latest.json")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

JP_NAMES = {
    "1617.T":"食品","1618.T":"エネルギー資源","1619.T":"建設・資材",
    "1620.T":"素材・化学","1621.T":"医薬品","1622.T":"自動車・輸送機",
    "1623.T":"鉄鋼・非鉄","1624.T":"機械","1625.T":"電機・精密",
    "1626.T":"情報通信・サービス","1627.T":"電力・ガス","1628.T":"運輸・物流",
    "1629.T":"商社・卸売","1630.T":"小売","1631.T":"銀行",
    "1632.T":"金融(除く銀行)","1633.T":"不動産",
}

def compute_today_signal():
    from lead_lag_paper_v3 import (
        load_data, build_V0, build_C0,
        compute_technical_features, build_lgbm_features,
        train_lgbm_signals, predict_lgbm_signals,
        US_TICKERS, JP_TICKERS,
        LAM, K, L, PRIOR_END, Q, TRAIN_END,
        LGBM_SKIP_THRESHOLD
    )
    from scipy.linalg import eigh
    log.info("データ取得中...")
    r_cc, r_oc_jp, macro_df, close_prices = load_data()
    V0 = build_V0()
    C0 = build_C0(r_cc[r_cc.index <= PRIOR_END], V0)
    all_tickers = US_TICKERS + JP_TICKERS
    n_us, n_jp  = len(US_TICKERS), len(JP_TICKERS)
    all_ret     = r_cc[all_tickers].values
    T = len(all_ret); t = T - 1
    signal_date = r_cc.index[t]
    log.info(f"シグナル計算日: {signal_date.date()}")
    W = all_ret[t-L:t, :]
    mu = np.nanmean(W, axis=0); sigma = np.nanstd(W, axis=0, ddof=0)
    sigma[sigma < 1e-10] = 1.0
    Z = np.nan_to_num((W - mu) / sigma, nan=0.0)
    std = Z.std(axis=0); valid = std > 1e-10
    Ct_s = np.corrcoef(Z[:, valid].T)
    Ct = np.eye(len(std)); iv = np.where(valid)[0]
    for i,vi in enumerate(iv):
        for j,vj in enumerate(iv): Ct[vi,vj] = Ct_s[i,j]
    Ct = np.nan_to_num(Ct, nan=0.0); np.fill_diagonal(Ct, 1.0); Ct = (Ct+Ct.T)/2
    C_reg = np.nan_to_num((1-LAM)*Ct + LAM*C0, nan=0.0)
    np.fill_diagonal(C_reg, 1.0); C_reg = (C_reg+C_reg.T)/2; np.fill_diagonal(C_reg, 1.0)
    N = len(all_tickers)
    vals, vecs = eigh(C_reg, subset_by_index=[N-K, N-1])
    V_K = vecs[:, np.argsort(vals)[::-1]]
    V_U, V_J = V_K[:n_us], V_K[n_us:]
    z_us = np.nan_to_num((all_ret[t,:n_us] - mu[:n_us]) / sigma[:n_us], nan=0.0)
    z_hat = V_J @ (V_U.T @ z_us)
    # LightGBMフィルター適用
    log.info("LightGBMフィルター計算中...")
    tech_feats  = compute_technical_features(close_prices, US_TICKERS + JP_TICKERS)
    feat_df     = build_lgbm_features(
                    pd.DataFrame([z_hat], index=[signal_date], columns=JP_TICKERS),
                    tech_feats, macro_df, r_cc)
    models      = train_lgbm_signals(feat_df, r_oc_jp, train_end=TRAIN_END)
    lgbm_filter = predict_lgbm_signals(models, feat_df,
                                        test_start=signal_date.strftime("%Y-%m-%d"))

    # フィルターを適用してシグナルを絞る
    ok_prob = lgbm_filter.iloc[-1] if len(lgbm_filter) > 0 else pd.Series()
    z_hat_series = pd.Series(z_hat, index=JP_TICKERS)
    if len(ok_prob) > 0:
        z_hat_series = z_hat_series[
            ok_prob.reindex(z_hat_series.index).fillna(0.5) >= LGBM_SKIP_THRESHOLD
        ]
    signals = {t: round(float(v), 6) for t, v in z_hat_series.items()}
    sig_s   = pd.Series(signals); n_pos = max(1, int(np.ceil(len(sig_s)*Q)))
    sorted_idx = sig_s.sort_values(ascending=False).index
    result = {
        "date":         signal_date.strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signals":      signals,
        "long":         sorted_idx[:n_pos].tolist(),
        "short":        sorted_idx[-n_pos:].tolist(),
        "us_ret_today": {US_TICKERS[i]: round(float(r_cc[US_TICKERS[i]].iloc[-1])*100,2) for i in range(n_us)},
    }
    SIGNAL_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info(f"シグナル保存: {SIGNAL_PATH}")
    return result

def load_trade_log(): return json.loads(TRADE_LOG_PATH.read_text()) if TRADE_LOG_PATH.exists() else []
def save_trade_log(d): TRADE_LOG_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2))

def record_position(signal, capital=VIRTUAL_CAPITAL):
    tl = [r for r in load_trade_log() if r["signal_date"] != signal["date"]]
    alloc = capital / max(len(signal["long"]), len(signal["short"]), 1)
    positions = [{"ticker":t,"name":JP_NAMES.get(t,t),"direction":"LONG","signal":signal["signals"][t],"alloc":alloc} for t in signal["long"]] + \
                [{"ticker":t,"name":JP_NAMES.get(t,t),"direction":"SHORT","signal":signal["signals"][t],"alloc":alloc} for t in signal["short"]]
    tl.append({"signal_date":signal["date"],"generated_at":signal["generated_at"],"positions":positions,"pnl":None,"pnl_pct":None,"status":"OPEN"})
    save_trade_log(tl)
    log.info(f"ポジション記録: LONG {len(signal['long'])}銘柄 / SHORT {len(signal['short'])}銘柄")

def update_pnl():
    from lead_lag_paper import load_data
    tl = load_trade_log(); open_r = [r for r in tl if r["status"]=="OPEN"]
    if not open_r: log.info("更新対象なし"); return
    r_cc, r_oc_jp = load_data()
    for record in open_r:
        fd = r_oc_jp.index[r_oc_jp.index > pd.Timestamp(record["signal_date"])]
        if not len(fd): continue
        r_next = r_oc_jp.loc[fd[0]]; total = 0.0
        for pos in record["positions"]:
            t = pos["ticker"]
            if t not in r_next.index or np.isnan(r_next[t]): continue
            pnl = pos["alloc"]*r_next[t] if pos["direction"]=="LONG" else -pos["alloc"]*r_next[t]
            pos["realized_return"] = round(float(r_next[t])*100,3); pos["pnl_yen"] = round(float(pnl),0); total += pnl
        record.update({"pnl":round(total,0),"pnl_pct":round(total/VIRTUAL_CAPITAL*100,4),"close_date":fd[0].strftime("%Y-%m-%d"),"status":"CLOSED"})
    save_trade_log(tl); log.info("損益更新完了")

def send_discord(signal, trade_log):
    if not DISCORD_WEBHOOK_URL: log.warning("DISCORD_WEBHOOK_URL未設定"); return
    closed    = [r for r in trade_log if r["status"]=="CLOSED" and r["pnl"] is not None]
    total_pnl = sum(r["pnl"] for r in closed)
    win_rate  = sum(1 for r in closed if r["pnl"]>0)/len(closed)*100 if closed else 0
    today_pnl = closed[-1]["pnl"] if closed else 0
    today_pct = closed[-1]["pnl_pct"] if closed else 0
    long_str  = "\n".join([f"　🔼 {JP_NAMES.get(t,t)}（{t}）" for t in signal.get("long",[])])
    short_str = "\n".join([f"　🔽 {JP_NAMES.get(t,t)}（{t}）" for t in signal.get("short",[])])
    us_top    = sorted(signal.get("us_ret_today",{}).items(), key=lambda x:-x[1])[:3]
    us_str    = "  ".join([f"{t}: {'▲' if v>0 else '▼'}{abs(v):.2f}%" for t,v in us_top])
    payload   = {"embeds":[{"title":"🇯🇵🇺🇸 日米リードラグ デモトレード","description":f"**シグナル日: {signal['date']}**","color":0x4a148c,
        "fields":[
            {"name":"🔼 LONG（買い）","value":long_str or "なし","inline":True},
            {"name":"🔽 SHORT（売り）","value":short_str or "なし","inline":True},
            {"name":"🇺🇸 米国ETF上位","value":us_str or "データなし","inline":False},
            {"name":f"{'✅' if today_pnl>=0 else '❌'} 前回損益","value":f"¥{today_pnl:,.0f}（{today_pct:+.3f}%）","inline":True},
            {"name":f"{'📈' if total_pnl>=0 else '📉'} 累積損益","value":f"¥{total_pnl:,.0f}（勝率{win_rate:.1f}%）","inline":True},
        ],"footer":{"text":f"生成: {signal['generated_at']}"}}]}
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        log.info("Discord通知送信完了" if r.status_code==204 else f"Discord失敗: {r.status_code}")
    except Exception as e: log.error(f"Discordエラー: {e}")

def generate_html_report(open_browser=True):
    tl = load_trade_log(); signal = json.loads(SIGNAL_PATH.read_text()) if SIGNAL_PATH.exists() else {}
    closed = [r for r in tl if r["status"]=="CLOSED" and r["pnl"] is not None]
    total_pnl = sum(r["pnl"] for r in closed); total_pct = total_pnl/VIRTUAL_CAPITAL*100
    win_rate  = sum(1 for r in closed if r["pnl"]>0)/len(closed)*100 if closed else 0
    running   = 0; cum = []
    for r in sorted(closed, key=lambda x:x["signal_date"]):
        running += r["pnl"]; cum.append({"date":r["close_date"],"v":running})

    sig_rows = ""
    for ticker, val in pd.Series(signal.get("signals",{})).sort_values(ascending=False).items():
        d = "🔼 LONG" if ticker in signal.get("long",[]) else ("🔽 SHORT" if ticker in signal.get("short",[]) else "　")
        c = "#e8f5e9" if "LONG" in d else ("#fce4ec" if "SHORT" in d else "white")
        sig_rows += f'<tr style="background:{c}"><td>{ticker}</td><td>{JP_NAMES.get(ticker,ticker)}</td><td>{val:.4f}</td><td><b>{d}</b></td></tr>'

    us_rows = ""
    for ticker, ret in sorted(signal.get("us_ret_today",{}).items(), key=lambda x:-x[1]):
        c = "#e8f5e9" if ret>0 else "#fce4ec"
        us_rows += f'<tr style="background:{c}"><td>{ticker}</td><td>{"▲" if ret>0 else "▼"} {abs(ret):.2f}%</td></tr>'

    hist = ""
    for r in reversed(tl[-30:]):
        _pc  = "#2e7d32" if r.get("pnl",0) and r["pnl"]>0 else "#c62828"
        _ps  = ("¥" + f'{r["pnl"]:,.0f}') if r["pnl"] is not None else "-"
        _nl  = len([p for p in r["positions"] if p["direction"]=="LONG"])
        _ns  = len([p for p in r["positions"] if p["direction"]=="SHORT"])
        _st  = "🟢 CLOSED" if r["status"]=="CLOSED" else "🟡 OPEN"
        hist += (f'<tr><td>{r["signal_date"]}</td>'
                 f'<td>{r.get("close_date","-")}</td>'
                 f'<td>{_nl}L/{_ns}S</td>'
                 f'<td style="color:{_pc};font-weight:bold">{_ps}</td>'
                 f'<td>{r.get("pnl_pct","-")}%</td>'
                 f'<td>{_st}</td></tr>')

    html = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>デモトレード レポート</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:0;background:#f5f5f5;color:#333}}
.hd{{background:linear-gradient(135deg,#1a237e,#4a148c);color:white;padding:24px 32px}}
.hd h1{{margin:0;font-size:24px}}.hd p{{margin:4px 0 0;opacity:.8;font-size:14px}}
.ct{{max-width:1200px;margin:24px auto;padding:0 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.card h3{{margin:0 0 8px;font-size:13px;color:#666}}.card .val{{font-size:28px;font-weight:bold}}
.pos{{color:#2e7d32}}.neg{{color:#c62828}}.neu{{color:#1a237e}}
.sec{{background:white;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px}}
.sec h2{{margin:0 0 16px;font-size:18px;color:#1a237e}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{background:#f3f4f6;padding:10px 12px;text-align:left;font-weight:600;color:#555}}
td{{padding:9px 12px;border-bottom:1px solid #f0f0f0}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.sd{{font-size:13px;color:#888;margin-bottom:16px}}
@media(max-width:768px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="hd"><h1>🇯🇵🇺🇸 日米業種リードラグ デモトレード</h1>
<p>部分空間正則化PCA戦略 | 仮想資金: ¥{VIRTUAL_CAPITAL:,.0f} | 更新: {datetime.now().strftime("%Y/%m/%d %H:%M")}</p></div>
<div class="ct">
<div class="grid">
  <div class="card"><h3>累積損益</h3><div class="val {'pos' if total_pnl>=0 else 'neg'}">¥{total_pnl:,.0f}</div></div>
  <div class="card"><h3>累積損益率</h3><div class="val {'pos' if total_pct>=0 else 'neg'}">{total_pct:+.2f}%</div></div>
  <div class="card"><h3>取引回数</h3><div class="val neu">{len(closed)}回</div></div>
  <div class="card"><h3>勝率</h3><div class="val {'pos' if win_rate>=50 else 'neg'}">{win_rate:.1f}%</div></div>
</div>
<div class="sec"><h2>📈 累積損益推移</h2><canvas id="c" height="80"></canvas></div>
<div class="two">
  <div class="sec"><h2>📊 本日のシグナル（日本ETF）</h2>
  <p class="sd">シグナル日: {signal.get("date","未計算")} | 翌営業日に執行</p>
  <table><tr><th>ティッカー</th><th>業種</th><th>シグナル値</th><th>方向</th></tr>
  {sig_rows or '<tr><td colspan="4">シグナル未計算</td></tr>'}</table></div>
  <div class="sec"><h2>🇺🇸 米国ETF 本日リターン</h2>
  <p class="sd">当日CCリターン（シグナルの入力情報）</p>
  <table><tr><th>ティッカー</th><th>リターン</th></tr>
  {us_rows or '<tr><td colspan="2">データなし</td></tr>'}</table></div>
</div>
<div class="sec"><h2>📋 取引履歴（直近30件）</h2>
<table><tr><th>シグナル日</th><th>クローズ日</th><th>ポジション</th><th>損益</th><th>損益率</th><th>状態</th></tr>
{hist or '<tr><td colspan="6" style="text-align:center;color:#aaa">取引履歴なし</td></tr>'}</table></div>
</div>
<script>
new Chart(document.getElementById('c').getContext('2d'),{{
  type:'line',data:{{labels:{json.dumps([d["date"] for d in cum])},
  datasets:[{{label:'累積損益',data:{json.dumps([d["v"] for d in cum])},
  borderColor:'#4a148c',backgroundColor:'rgba(74,20,140,0.1)',fill:true,tension:0.3,pointRadius:3}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}}}},
  scales:{{y:{{ticks:{{callback:v=>'¥'+v.toLocaleString()}}}}}}}}
}});
</script></body></html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info(f"レポート生成: {REPORT_PATH}")
    if open_browser: webbrowser.open(REPORT_PATH.resolve().as_uri())

def run_scheduler():
    try: import schedule, time
    except ImportError: log.error("pip install schedule が必要"); return
    def job():
        if date.today().weekday()>=5: return
        try:
            s = compute_today_signal(); record_position(s); update_pnl()
            send_discord(s, load_trade_log()); generate_html_report(False)
        except Exception as e: log.error(f"エラー: {e}")
    schedule.every().day.at("17:00").do(job)
    log.info("スケジューラ起動: 毎日17:00 自動実行（Ctrl+C で停止）")
    import time
    while True: schedule.run_pending(); time.sleep(60)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report",     action="store_true")
    p.add_argument("--schedule",   action="store_true")
    p.add_argument("--update",     action="store_true")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    ob   = not args.no_browser

    if args.schedule: run_scheduler(); return
    if args.update:   update_pnl(); generate_html_report(ob); return
    if args.report:   generate_html_report(ob); return

    log.info("=== デモトレード実行 ===")
    signal = compute_today_signal()
    print(f"\n{'='*50}\n  シグナル日: {signal['date']}\n{'='*50}")
    print(f"\n🔼 LONG  ({len(signal['long'])}銘柄):")
    for t in signal["long"]:   print(f"   {t} {JP_NAMES.get(t,'')}")
    print(f"\n🔽 SHORT ({len(signal['short'])}銘柄):")
    for t in signal["short"]:  print(f"   {t} {JP_NAMES.get(t,'')}")
    record_position(signal); update_pnl()
    send_discord(signal, load_trade_log())
    generate_html_report(ob)
    log.info("完了！")

if __name__ == "__main__":
    main()
 
