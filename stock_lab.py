#!/usr/bin/env python3
"""
STOCK LAB - swing trading toolkit with a selectable menu
=========================================================
Combines ideas from these open-source projects into one app:
  * RyanJHamby/stock-screener  -> Minervini Trend Template scanner + RS rating
  * chand1012/stonks           -> EMA pullback strategy, EMA / time / % trailing exits, Alpaca paper trading
  * sofus-nl/swing-trading-strategies -> strategy library with written entry/exit rules (Strategy guide)
  * kernc/backtesting.py       -> backtesting engine
  * polakowo/vectorbt          -> fast settings optimizer
  * pkjmesra/PKScreener        -> launcher for Indian (NSE) screening
(Ideas re-implemented here; no code copied from those repos.)

Install:   pip install yfinance pandas numpy matplotlib backtesting vectorbt alpaca-py lxml requests
           (vectorbt and alpaca-py are optional; pkscreener optional: pip install pkscreener)
Run:       python stock_lab.py            (real data)
           python stock_lab.py --demo     (simulated data, no internet)

Educational tool only - not financial advice.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
import zlib
from io import StringIO

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")   # hide backtest progress bars
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

DEMO = False
SETTINGS_FILE = "stock_lab_settings.json"
WATCHLIST_FILE = "watchlist.txt"

DEFAULTS = {
    "market": "US",            # US or INDIA
    "account": 10000.0,
    "risk_pct": 1.0,
    "strategy": "pullback",
    "exits": ["targets", "atr_trail"],
    "target_r": 2.0,
    "max_days": 14,
    "trail_atr": 3.0,
    "pct_activate": 5.0,
    "pct_trail": 2.0,
    "years": 3,
    "commission": 0.0005,
}

US_LARGE = ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B JPM LLY V UNH XOM MA COST HD PG JNJ "
            "ABBV NFLX CRM AMD ORCL BAC KO PEP MRK ADBE WMT CVX TMO LIN ACN MCD CSCO ABT QCOM "
            "INTU DHR TXN GE CAT AMAT ISRG IBM NOW UBER PLTR PANW").split()
NIFTY50 = ("RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL ITC SBIN LT HINDUNILVR BAJFINANCE "
           "KOTAKBANK AXISBANK MARUTI SUNPHARMA M&M HCLTECH TITAN ULTRACEMCO NTPC ASIANPAINT "
           "POWERGRID TATASTEEL ONGC ADANIENT ADANIPORTS COALINDIA BAJAJFINSV NESTLEIND WIPRO "
           "JSWSTEEL GRASIM TECHM HINDALCO CIPLA DRREDDY SBILIFE HDFCLIFE BRITANNIA EICHERMOT "
           "APOLLOHOSP TATACONSUM BAJAJ-AUTO HEROMOTOCO SHRIRAMFIN TRENT BEL INDUSINDBK ETERNAL JIOFIN").split()

GREEN, YELLOW, RED, BOLD, END = "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"


# =================================================================== settings
def load_settings():
    s = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                s.update(json.load(f))
        except Exception:
            pass
    return s


def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)


def cur(s):
    return "Rs " if s["market"] == "INDIA" else "$"


def sym(t, s):
    """Add .NS for Indian stocks."""
    t = t.strip().upper().replace(".", "-") if s["market"] == "US" else t.strip().upper()
    if s["market"] == "INDIA" and not t.endswith((".NS", ".BO")) and not t.startswith("^"):
        t += ".NS"
    return t


def benchmark(s):
    return "^NSEI" if s["market"] == "INDIA" else "SPY"


# ======================================================================= data
def _demo_df(ticker, years):
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    n = int(252 * years)
    drift = rng.uniform(-0.0003, 0.0012)
    # regime switching so strategies get trends and pullbacks
    regimes = np.repeat(rng.normal(drift, 0.0015, n // 60 + 1), 60)[:n]
    close = 100 * np.exp(np.cumsum(rng.normal(regimes, 0.017, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    openp = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high, low = np.maximum(high, openp), np.minimum(low, openp)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(2_000_000, 9_000_000, n).astype(float)}, index=idx)


def fetch(tickers, years=3):
    """Return {ticker: OHLCV DataFrame}. Skips tickers with no / short data."""
    tickers = list(dict.fromkeys(tickers))
    out = {}
    if DEMO:
        return {t: _demo_df(t, years) for t in tickers}
    import yfinance as yf
    print(f"  Downloading {len(tickers)} symbol(s)...")
    raw = yf.download(tickers, period=f"{years}y", interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 220:
                out[t] = df
        except Exception:
            pass
    missing = [t for t in tickers if t not in out]
    if missing:
        print(f"  {YELLOW}Skipped (no data or < 220 days): {', '.join(missing[:15])}"
              f"{' ...' if len(missing) > 15 else ''}{END}")
    return out


def wiki_symbols(url, col_names):
    import requests
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 stock-lab"}, timeout=20).text
    for tab in pd.read_html(StringIO(html)):
        for c in col_names:
            if c in tab.columns:
                return [str(x).replace(".", "-") for x in tab[c].dropna()]
    raise ValueError("symbol column not found")


# ================================================================ indicators
def indicators(df, bench=None):
    d = df.copy()
    c, h, l = d.Close, d.High, d.Low
    for n in (5, 20, 50, 150, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()
    for n in (10, 21, 50):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()

    def rsi(n):
        dl = c.diff()
        g = dl.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        ls = (-dl.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + g / ls.replace(0, np.nan))
    d["RSI"], d["RSI2"] = rsi(14), rsi(2)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACDhist"] = macd - macd.ewm(span=9, adjust=False).mean()
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["High20"] = h.rolling(20).max().shift(1)        # prior 20-day high
    d["Low10"] = l.rolling(10).min()
    d["Low5"] = l.rolling(5).min()
    d["High252"] = h.rolling(252, min_periods=120).max()
    d["Low252"] = l.rolling(252, min_periods=120).min()
    d["Vol20"] = d.Volume.rolling(20).mean()
    d["Ret126"] = c.pct_change(126)
    d["RSscore"] = (0.4 * c.pct_change(63) + 0.2 * c.pct_change(126) +
                    0.2 * c.pct_change(189) + 0.2 * c.pct_change(252))
    if bench is not None:
        b = bench.Close.reindex(d.index).ffill()
        d["BenchRet126"] = b.pct_change(126)
    else:
        d["BenchRet126"] = 0.0
    return d


# ================================================================ strategies
# Each strategy returns (entry_signal, stop_price, exit_signal) Series and a plan_now(row) function.
def _uptrend(d):
    return (d.Close > d.SMA50) & (d.SMA50 > d.SMA200)


def _cap_stop(d, stop, min_atr=1.0, max_atr=3.0):
    return stop.clip(lower=d.Close - max_atr * d.ATR, upper=d.Close - min_atr * d.ATR)


def strat_pullback(d):
    near = (d.Close >= d.SMA20 - d.ATR) & (d.Close <= d.SMA20 + 0.5 * d.ATR)
    sig = _uptrend(d) & near & d.RSI.between(38, 60) & (d.Close > d.Open)
    stop = _cap_stop(d, d.Low10 - 0.25 * d.ATR)
    return sig, stop, pd.Series(False, d.index)


def plan_pullback(x):
    e = min(x.Close, x.SMA20 + 0.25 * x.ATR)
    st = min(x.Low10 - 0.25 * x.ATR, e - x.ATR)
    return e, max(st, e - 3 * x.ATR), "Limit buy near the 20-day average while the uptrend holds"


def strat_breakout(d):
    sig = (d.Close > d.SMA50) & (d.Close > d.High20) & (d.Volume > 1.2 * d.Vol20)
    return sig, d.Close - 2 * d.ATR, pd.Series(False, d.index)


def plan_breakout(x):
    e = x.High20 + 0.1 * x.ATR
    return e, e - 2 * x.ATR, "Buy-stop just above the 20-day high, needs volume"


def minervini_template(d):
    """Mark Minervini Trend Template (8 rules) - returns DataFrame of rule checks."""
    return pd.DataFrame({
        "Price>150&200MA": (d.Close > d.SMA150) & (d.Close > d.SMA200),
        "150MA>200MA": d.SMA150 > d.SMA200,
        "200MA rising 1m": d.SMA200 > d.SMA200.shift(21),
        "50>150>200MA": (d.SMA50 > d.SMA150) & (d.SMA150 > d.SMA200),
        "Price>50MA": d.Close > d.SMA50,
        ">=30% above 52w low": d.Close >= 1.3 * d.Low252,
        "within 25% of 52w high": d.Close >= 0.75 * d.High252,
        "beats index (6m)": d.Ret126 > d.BenchRet126,
    })


def strat_minervini(d):
    tt = minervini_template(d).all(axis=1)
    sig = tt & (d.Close > d.High20) & (d.Volume > 1.4 * d.Vol20)
    stop = np.maximum(d.Close * 0.92, d.Close - 2 * d.ATR)          # never risk more than 8%
    return sig, pd.Series(stop, d.index), pd.Series(False, d.index)


def plan_minervini(x):
    e = x.High20 + 0.1 * x.ATR
    return e, max(e * 0.92, e - 2 * x.ATR), "Buy pivot breakout only if all 8 Trend Template rules pass; max 8% stop"


def strat_ema_pullback(d):
    stack = (d.EMA10 > d.EMA21) & (d.EMA21 > d.EMA50) & (d.Close > d.EMA50)
    touch = (d.Low <= d.EMA21 * 1.01) & (d.Close > d.EMA21)
    stop = _cap_stop(d, d.Low5 - 0.25 * d.ATR, 0.75)
    return stack & touch, stop, pd.Series(False, d.index)


def plan_ema_pullback(x):
    e = x.EMA21 * 1.005
    return e, max(min(x.Low5 - 0.25 * x.ATR, e - 0.75 * x.ATR), e - 3 * x.ATR), \
        "Limit buy at the 21-day EMA when 10>21>50 EMA are stacked"


def strat_rsi2(d):
    sig = (d.Close > d.SMA200) & (d.RSI2 < 10)
    return sig, d.Close - 3 * d.ATR, d.Close > d.SMA5


def plan_rsi2(x):
    return x.Close, x.Close - 3 * x.ATR, "Buy when RSI(2) < 10 above the 200MA; sell on close above the 5-day average"


def strat_golden(d):
    up = (d.SMA50 > d.SMA200)
    sig = up & ~up.shift(1, fill_value=False)
    return sig, d.Close - 3 * d.ATR, (~up) & up.shift(1, fill_value=False)


def plan_golden(x):
    return x.Close, x.Close - 3 * x.ATR, "Buy when 50MA crosses above 200MA; sell when it crosses back below"


STRATEGIES = {
    "pullback": dict(name="Pullback to 20-day MA", fn=strat_pullback, plan=plan_pullback,
                     src="original swing_trader",
                     rules="ENTRY: uptrend (price > 50MA > 200MA), price within 1 ATR of the 20MA, RSI 38-60, "
                           "green candle.\nSTOP: below 10-day low (1-3 ATR).  EXIT: by selected exit modes."),
    "breakout": dict(name="20-day breakout", fn=strat_breakout, plan=plan_breakout, src="original swing_trader",
                     rules="ENTRY: close above prior 20-day high, above 50MA, volume > 1.2x average.\n"
                           "STOP: 2 ATR below entry.  EXIT: by selected exit modes."),
    "minervini": dict(name="Minervini Trend Template (SEPA)", fn=strat_minervini, plan=plan_minervini,
                      src="RyanJHamby/stock-screener",
                      rules="ENTRY: all 8 Trend Template rules pass + breakout above 20-day high on 1.4x volume.\n"
                            "STOP: 2 ATR, never more than 8% below entry.  EXIT: by selected exit modes."),
    "ema_pullback": dict(name="EMA pullback (10/21/50)", fn=strat_ema_pullback, plan=plan_ema_pullback,
                         src="chand1012/stonks",
                         rules="ENTRY: 10 EMA > 21 EMA > 50 EMA, low touches the 21 EMA and closes above it.\n"
                               "STOP: below 5-day low.  EXIT: try the EMA exit or 14-day time exit."),
    "rsi2": dict(name="RSI(2) mean reversion", fn=strat_rsi2, plan=plan_rsi2,
                 src="swing-trading-strategies library (Connors)",
                 rules="ENTRY: price > 200MA and RSI(2) < 10 (sharp short-term dip in an uptrend).\n"
                       "EXIT: close above 5-day MA (built in). STOP: 3 ATR. Works best with targets OFF."),
    "golden": dict(name="Golden cross (50/200)", fn=strat_golden, plan=plan_golden,
                   src="swing-trading-strategies library",
                   rules="ENTRY: 50MA crosses above 200MA.  EXIT: 50MA crosses below 200MA (built in).\n"
                         "STOP: 3 ATR. Slow, long-hold strategy - use with targets OFF."),
}

EXITS = {
    "targets": "Fixed stop + profit target (Target = N x risk)",
    "atr_trail": "ATR trailing stop (highest close - N x ATR)",
    "ema_exit": "EMA exit (sell when close < 10 EMA)",
    "time": "Time exit (sell after N trading days)",
    "pct_trail": "% trailing stop (activate at +X%, trail by Y%)",
}


# ============================================================== menu helpers
def choose(title, options, allow_back=True):
    """options: list of (key, label). Returns key or None."""
    print(f"\n{BOLD}{title}{END}")
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}. {label}")
    if allow_back:
        print("  0. Back")
    while True:
        a = input("Select: ").strip()
        if allow_back and a in ("0", ""):
            return None
        if a.isdigit() and 1 <= int(a) <= len(options):
            return options[int(a) - 1][0]
        print("  Enter a number from the list.")


def choose_multi(title, options, current):
    print(f"\n{BOLD}{title}{END}  (comma-separated, e.g. 1,3)")
    for i, (k, label) in enumerate(options, 1):
        print(f"  {i}. [{'x' if k in current else ' '}] {label}")
    a = input("Select (Enter = keep current): ").strip()
    if not a:
        return current
    picks = [options[int(p) - 1][0] for p in a.replace(" ", "").split(",")
             if p.isdigit() and 1 <= int(p) <= len(options)]
    return picks or current


def ask_num(prompt, default):
    a = input(f"{prompt} [{default}]: ").strip()
    try:
        return type(default)(a) if a else default
    except ValueError:
        return default


def pick_tickers(s, multi=True):
    opts = [("type", "Type ticker(s)"), ("watch", f"My watchlist ({WATCHLIST_FILE})")]
    if multi:
        if s["market"] == "US":
            opts += [("large", "Top 50 US large caps (built-in)"),
                     ("sp500", "S&P 500 (downloads list from Wikipedia)"),
                     ("ndx", "Nasdaq 100 (downloads list from Wikipedia)")]
        else:
            opts += [("nifty", "Nifty 50 (built-in)")]
    k = choose("Which stocks?", opts)
    if k is None:
        return []
    if k == "type":
        raw = input("Ticker(s), space separated: ").split()
    elif k == "watch":
        if not os.path.exists(WATCHLIST_FILE):
            print(f"  {YELLOW}{WATCHLIST_FILE} not found - add tickers in Settings > Edit watchlist.{END}")
            return []
        raw = open(WATCHLIST_FILE).read().split()
    elif k == "large":
        raw = US_LARGE
    elif k == "nifty":
        raw = NIFTY50
    else:
        try:
            if DEMO:
                raise RuntimeError("demo mode")
            if k == "sp500":
                raw = wiki_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"])
            else:
                raw = wiki_symbols("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Symbol"])
        except Exception as e:
            print(f"  {YELLOW}Could not download list ({e}); using built-in top 50.{END}")
            raw = US_LARGE
    return [sym(t, s) for t in raw][: (None if multi else 1)]


def get_bench(s, years):
    b = fetch([benchmark(s)], years)
    return b.get(benchmark(s))


# ================================================================ 1. analyze
def score_row(d):
    x = d.iloc[-1]
    sc, why = 0, []
    if x.Close > x.SMA50 > x.SMA200:
        sc += 30; why.append("uptrend")
    elif x.Close > x.SMA200:
        sc += 15; why.append("above 200MA")
    else:
        why.append("below 200MA")
    if d.SMA50.iloc[-1] > d.SMA50.iloc[-10]:
        sc += 10; why.append("50MA rising")
    if 40 <= x.RSI <= 65:
        sc += 20; why.append(f"RSI {x.RSI:.0f} ok")
    elif x.RSI > 70:
        why.append(f"RSI {x.RSI:.0f} overbought")
    elif x.RSI < 30:
        sc += 5; why.append(f"RSI {x.RSI:.0f} oversold")
    else:
        sc += 10
    if x.MACDhist > 0:
        sc += 15; why.append("MACD bullish")
    if x.Volume > 1.2 * x.Vol20 and x.Close > d.Close.iloc[-2]:
        sc += 10; why.append("volume up-day")
    if x.Close > x.SMA50 > x.SMA200 and -1 <= (x.Close - x.SMA20) / x.ATR <= 0.75:
        sc += 15; why.append("in pullback zone")
    verdict = "BUY ZONE" if sc >= 65 else "WATCH" if sc >= 45 else "AVOID"
    return sc, verdict, why


def build_plan(d, s, strat_key):
    x = d.iloc[-1]
    st = STRATEGIES[strat_key]
    e, stop, note = st["plan"](x)
    r = e - stop
    shares = int((s["account"] * s["risk_pct"] / 100) // r) if r > 0 else 0
    shares = min(shares, int(s["account"] // e)) if e > 0 else 0
    sig, _, _ = st["fn"](d)
    last = sig[sig].index[-1].date() if sig.any() else "never"
    return dict(strategy=st["name"], entry=e, stop=stop, t1=e + s["target_r"] * r, t2=e + (s["target_r"] + 1) * r,
                shares=shares, cost=shares * e, max_loss=shares * r, signal_today=bool(sig.iloc[-1]),
                last_signal=last, note=note)


def analyze_menu(s):
    tickers = pick_tickers(s)
    if not tickers:
        return
    k = choose("Show plans for which strategies?", [("sel", f"Selected strategy only ({STRATEGIES[s['strategy']]['name']})"),
                                                    ("all", "All 6 strategies side by side")])
    if k is None:
        return
    keys = [s["strategy"]] if k == "sel" else list(STRATEGIES)
    chart = input("Save chart PNGs? (y/N): ").strip().lower() == "y"
    data = fetch(tickers, max(2, s["years"]))
    bench = get_bench(s, max(2, s["years"]))
    c, rows = cur(s), []
    for t, df in data.items():
        d = indicators(df, bench)
        sc, verdict, why = score_row(d)
        x = d.iloc[-1]
        colr = {"BUY ZONE": GREEN, "WATCH": YELLOW, "AVOID": RED}[verdict]
        tt = minervini_template(d).iloc[-1]
        print("\n" + "=" * 92)
        print(f" {BOLD}{t}{END}  {c}{x.Close:.2f}  ({d.index[-1].date()})   {colr}{verdict}  score {sc}/100{END}"
              f"   RSI {x.RSI:.0f}  ATR {c}{x.ATR:.2f}")
        print(f" 20MA {x.SMA20:.2f} | 50MA {x.SMA50:.2f} | 200MA {x.SMA200:.2f} | 52w high {x.High252:.2f}"
              f" | Trend Template {int(tt.sum())}/8")
        print(" Why: " + ", ".join(why))
        print(f"\n {'STRATEGY':<32}{'SIGNAL':>7}{'ENTRY':>10}{'STOP':>10}{'TARGET1':>10}{'TARGET2':>10}"
              f"{'SHARES':>8}{'MAX LOSS':>10}  LAST SIGNAL")
        for key in keys:
            p = build_plan(d, s, key)
            flag = f"{GREEN}  YES {END}" if p["signal_today"] else "   no  "
            print(f" {p['strategy']:<32}{flag}{p['entry']:>10.2f}{p['stop']:>10.2f}{p['t1']:>10.2f}{p['t2']:>10.2f}"
                  f"{p['shares']:>8}{p['max_loss']:>10,.0f}  {p['last_signal']}")
            rows.append(dict(Ticker=t, Score=sc, Verdict=verdict, Strategy=p["strategy"], SignalToday=p["signal_today"],
                             Entry=round(p["entry"], 2), Stop=round(p["stop"], 2), T1=round(p["t1"], 2),
                             T2=round(p["t2"], 2), Shares=p["shares"]))
        trail = d.Close.iloc[-22:].max() - s["trail_atr"] * x.ATR
        broken = f" {RED}(price is already below it - exit signal){END}" if x.Close < trail else ""
        print(f"\n Holding it already? Exit on a close below {c}{trail:.2f} (ATR trail){broken} or below 10 EMA {c}{x.EMA10:.2f}.")
        print(f" Plan: sell half at Target 1, move stop to entry, let the rest run. Risk {s['risk_pct']}% of "
              f"{c}{s['account']:,.0f}.")
        if chart:
            save_chart(t, d, build_plan(d, s, keys[0]), c)
    if rows and input("\nSave results to CSV? (y/N): ").strip().lower() == "y":
        pd.DataFrame(rows).to_csv("analysis.csv", index=False)
        print("  Saved analysis.csv")


def save_chart(t, d, p, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = d.iloc[-150:]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(v.index, v.Close, color="#222", lw=1.4, label="Close")
    for col, clr in (("SMA20", "#1f77b4"), ("SMA50", "#ff7f0e"), ("SMA200", "#9467bd")):
        ax.plot(v.index, v[col], lw=1, color=clr, label=col)
    for y, lab, clr in ((p["entry"], "Entry", "#2b8a3e"), (p["stop"], "Stop", "#c92a2a"),
                        (p["t1"], "T1", "#1971c2"), (p["t2"], "T2", "#1971c2")):
        ax.axhline(y, ls="--", lw=1, color=clr)
        ax.text(v.index[-1], y, f" {lab} {y:.2f}", va="center", fontsize=8, color=clr)
    ax.set_title(f"{t} - {p['strategy']}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.3)
    ax2.plot(v.index, v.RSI, color="#6741d9"); ax2.axhline(70, ls="--", c="grey"); ax2.axhline(30, ls="--", c="grey")
    ax2.set_ylabel("RSI"); ax2.grid(alpha=.3)
    fn = f"{t.replace('^', '')}_chart.png"
    fig.tight_layout(); fig.savefig(fn, dpi=110); plt.close(fig)
    print(f"  Chart saved: {fn}")


# =================================================================== 2. scan
def scan_menu(s):
    k = choose("Scan type", [("tt", "Minervini Trend Template (all 8 rules) + RS rating"),
                             ("sig", f"Signals today - selected strategy ({STRATEGIES[s['strategy']]['name']})"),
                             ("all", "Signals today - all 6 strategies"),
                             ("score", "Rank by overall score")])
    if k is None:
        return
    tickers = pick_tickers(s)
    if not tickers:
        return
    data = fetch(tickers, 2)
    bench = get_bench(s, 2)
    ds = {t: indicators(df, bench) for t, df in data.items()}
    if not ds:
        return
    rs = pd.Series({t: d.RSscore.iloc[-1] for t, d in ds.items()})
    rs_rank = (rs.rank(pct=True) * 99).round().fillna(0).astype(int)    # 1-99 like IBD RS
    rows = []
    for t, d in ds.items():
        x = d.iloc[-1]
        sc, verdict, _ = score_row(d)
        tt = minervini_template(d).iloc[-1]
        row = dict(Ticker=t, Price=round(x.Close, 2), Score=sc, Verdict=verdict, TT=f"{int(tt.sum())}/8",
                   RS=int(rs_rank[t]), RSI=round(x.RSI, 0),
                   PctFromHigh=round((x.Close / x.High252 - 1) * 100, 1))
        hits = [STRATEGIES[key]["name"] for key in (STRATEGIES if k == "all" else [s["strategy"]])
                if STRATEGIES[key]["fn"](d)[0].iloc[-1]]
        row["Signals"] = ", ".join(hits)
        if k == "tt" and not (tt.all() and rs_rank[t] >= 70):
            continue
        if k in ("sig", "all") and not hits:
            continue
        p = build_plan(d, s, s["strategy"])
        row.update(Entry=round(p["entry"], 2), Stop=round(p["stop"], 2), Target=round(p["t1"], 2), Shares=p["shares"])
        rows.append(row)
    if not rows:
        print(f"\n  No stocks passed this scan today ({len(ds)} checked).")
        return
    tab = pd.DataFrame(rows).sort_values(["Score", "RS"], ascending=False)
    print(f"\n{BOLD}{len(tab)} of {len(ds)} stocks passed{END}  (Entry/Stop/Target use the selected strategy)\n")
    print(tab.to_string(index=False))
    if input("\nSave to scan_results.csv? (y/N): ").strip().lower() == "y":
        tab.to_csv("scan_results.csv", index=False)
        print("  Saved scan_results.csv")
    if input("Add these to your watchlist? (y/N): ").strip().lower() == "y":
        old = open(WATCHLIST_FILE).read().split() if os.path.exists(WATCHLIST_FILE) else []
        with open(WATCHLIST_FILE, "w") as f:
            f.write("\n".join(dict.fromkeys(old + list(tab.Ticker))))
        print(f"  Watchlist now has {len(set(old + list(tab.Ticker)))} tickers.")


# =============================================================== 3. backtest
def prep_bt(df, bench, strat_key):
    d = indicators(df, bench)
    sig, stop, ex = STRATEGIES[strat_key]["fn"](d)
    d["Signal"] = sig.astype(float)
    d["StopPx"] = stop
    d["ExitSig"] = ex.astype(float)
    return d.dropna(subset=["SMA200", "ATR", "StopPx"])


def make_bt_class(s, stop_mult=None):
    from backtesting import Strategy

    class Swing(Strategy):
        def init(self):
            pass

        def next(self):
            i = len(self.data) - 1
            price, atr = self.data.Close[-1], self.data.ATR[-1]
            for tr in list(self.trades):
                eb = tr.entry_bar
                if "atr_trail" in s["exits"]:
                    new = self.data.Close[eb:].max() - s["trail_atr"] * atr
                    if tr.sl is None or new > tr.sl:
                        tr.sl = new
                if "pct_trail" in s["exits"]:
                    hh = self.data.High[eb:].max()
                    if hh >= tr.entry_price * (1 + s["pct_activate"] / 100):
                        new = hh * (1 - s["pct_trail"] / 100)
                        if tr.sl is None or new > tr.sl:
                            tr.sl = new
                if ("ema_exit" in s["exits"] and price < self.data.EMA10[-1]) or \
                   ("time" in s["exits"] and i - eb >= s["max_days"]) or self.data.ExitSig[-1] > 0:
                    tr.close()
            if not self.position and self.data.Signal[-1] > 0:
                stop = price - stop_mult * atr if stop_mult else self.data.StopPx[-1]
                r = price - stop
                if r <= 0:
                    return
                units = int(self.equity * s["risk_pct"] / 100 / r)
                units = min(units, int(self.equity * 0.98 / price))
                if units < 1:
                    return
                tp = price + s["target_r"] * r if "targets" in s["exits"] else None
                self.buy(size=units, sl=stop, tp=tp)
    return Swing


def run_bt(d, s, stop_mult=None):
    from backtesting import Backtest
    bt = Backtest(d, make_bt_class(s, stop_mult), cash=s["account"], commission=s["commission"],
                  exclusive_orders=False, finalize_trades=True)
    return bt, bt.run()


def backtest_menu(s):
    try:
        import backtesting  # noqa
    except ImportError:
        print(f"  {RED}Install backtesting.py first: pip install backtesting{END}")
        return
    k = choose("Backtest what?", [("one", f"One stock, selected strategy ({STRATEGIES[s['strategy']]['name']})"),
                                  ("cmp", "One stock, compare all 6 strategies"),
                                  ("many", "Many stocks, selected strategy")])
    if k is None:
        return
    tickers = pick_tickers(s, multi=(k == "many"))
    if not tickers:
        return
    data = fetch(tickers, s["years"])
    bench = get_bench(s, s["years"])
    print(f"\n  Exits: {', '.join(s['exits']) or 'stop only'} | risk {s['risk_pct']}%/trade | "
          f"{s['years']} years | commission {s['commission']*100:.2f}%")
    keys = list(STRATEGIES) if k == "cmp" else [s["strategy"]]
    rows = []
    for t, df in data.items():
        for key in keys:
            d = prep_bt(df, bench, key)
            bt, st = run_bt(d, s)
            rows.append(dict(Ticker=t, Strategy=STRATEGIES[key]["name"], Trades=st["# Trades"],
                             WinRate=round(st["Win Rate [%]"], 1) if st["# Trades"] else 0,
                             Return=round(st["Return [%]"], 1), BuyHold=round(st["Buy & Hold Return [%]"], 1),
                             MaxDD=round(st["Max. Drawdown [%]"], 1),
                             ProfitFactor=round(st["Profit Factor"], 2) if st["# Trades"] else 0,
                             AvgTrade=round(st["Avg. Trade [%]"], 2) if st["# Trades"] else 0,
                             Exposure=round(st["Exposure Time [%]"], 0)))
            if k == "one":
                print(f"\n{BOLD}{t} - {STRATEGIES[key]['name']}{END}")
                print(st.drop(["_strategy", "_equity_curve", "_trades"]).to_string())
                if st["# Trades"]:
                    tr = st["_trades"][["EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size", "PnL", "ReturnPct"]].copy()
                    tr["ReturnPct"] *= 100
                    print(f"\nLast 10 trades:\n{tr.tail(10).round(2).to_string(index=False)}")
                if input("\nSave interactive HTML chart? (y/N): ").strip().lower() == "y":
                    fn = f"backtest_{t.replace('^', '')}_{key}.html"
                    bt.plot(filename=fn, open_browser=False)
                    print(f"  Saved {fn}")
    if k != "one":
        tab = pd.DataFrame(rows)
        print("\n" + tab.to_string(index=False))
        if k == "many" and len(tab):
            print(f"\n Average return {tab.Return.mean():.1f}% vs buy & hold {tab.BuyHold.mean():.1f}% | "
                  f"avg win rate {tab.WinRate.mean():.1f}% | {int(tab.Trades.sum())} trades")
        if input("\nSave to backtest_results.csv? (y/N): ").strip().lower() == "y":
            tab.to_csv("backtest_results.csv", index=False)
            print("  Saved backtest_results.csv")
    print(f"\n {YELLOW}Past results do not guarantee future returns. Check enough trades (30+) before trusting it.{END}")


# =============================================================== 4. optimize
def optimize_menu(s):
    print(f"\n{BOLD}Optimize stop distance and profit target{END}  for {STRATEGIES[s['strategy']]['name']}")
    print(" Tests stop = 1-3 x ATR and target = 1.5-4 x risk. Uses vectorbt if installed, else backtesting.py.")
    tickers = pick_tickers(s, multi=False)
    if not tickers:
        return
    data = fetch(tickers, s["years"])
    if not data:
        return
    t, df = next(iter(data.items()))
    d = prep_bt(df, get_bench(s, s["years"]), s["strategy"])
    stops, targets = [1.0, 1.5, 2.0, 2.5, 3.0], [1.5, 2.0, 3.0, 4.0]
    split = int(len(d) * 0.7)
    rows = []
    try:
        import vectorbt as vbt
        engine = "vectorbt"
        for part, dd in (("train", d.iloc[:split]), ("test", d.iloc[split:])):
            for sm in stops:
                for tm in targets:
                    sl = (sm * dd.ATR / dd.Close).values
                    pf = vbt.Portfolio.from_signals(dd.Close, entries=dd.Signal > 0, exits=dd.ExitSig > 0,
                                                    sl_stop=sl, tp_stop=sl * tm, init_cash=s["account"],
                                                    fees=s["commission"], freq="1D")
                    st = pf.stats()
                    rows.append(dict(Part=part, StopATR=sm, TargetR=tm, Trades=int(st["Total Trades"]),
                                     Return=round(st["Total Return [%]"], 1),
                                     WinRate=round(st["Win Rate [%]"], 1) if st["Total Trades"] else 0,
                                     MaxDD=round(st["Max Drawdown [%]"], 1)))
    except ImportError:
        engine = "backtesting.py"
        base = dict(s)
        for part, dd in (("train", d.iloc[:split]), ("test", d.iloc[split:])):
            for sm in stops:
                for tm in targets:
                    s2 = dict(base, target_r=tm, exits=["targets"])
                    _, st = run_bt(dd, s2, stop_mult=sm)
                    rows.append(dict(Part=part, StopATR=sm, TargetR=tm, Trades=st["# Trades"],
                                     Return=round(st["Return [%]"], 1),
                                     WinRate=round(st["Win Rate [%]"], 1) if st["# Trades"] else 0,
                                     MaxDD=round(st["Max. Drawdown [%]"], 1)))
    tab = pd.DataFrame(rows)
    tr = tab[tab.Part == "train"].drop(columns="Part").set_index(["StopATR", "TargetR"])
    te = tab[tab.Part == "test"].drop(columns="Part").set_index(["StopATR", "TargetR"])
    both = tr.join(te, lsuffix="_train", rsuffix="_test").reset_index()
    both = both.sort_values("Return_train", ascending=False)
    print(f"\n {t} - engine: {engine}. Train = first 70% of history, Test = last 30% (unseen).")
    print(" Each trade uses the full account here, so compare settings with each other, not with the backtest.\n")
    print(both.head(10).to_string(index=False))
    best = both.iloc[0]
    print(f"\n Best on train: stop {best.StopATR} x ATR, target {best.TargetR} x risk -> "
          f"train {best.Return_train}% / test {best.Return_test}%")
    print(f" {YELLOW}If the test return is much worse than train, the setting is over-fitted - don't use it.{END}")
    if input("Use this target in your settings? (y/N): ").strip().lower() == "y":
        s["target_r"] = float(best.TargetR)
        save_settings(s)
        print("  Saved.")


# ================================================================== 5. alpaca
def alpaca_menu(s):
    if s["market"] != "US":
        print(f"  {YELLOW}Alpaca trades US stocks only. Switch market to US in Settings.{END}")
        return
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
    except ImportError:
        print(f"  {RED}Install first: pip install alpaca-py{END}")
        return
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("  Set your PAPER keys first (free at alpaca.markets > Paper Trading > API Keys):")
        print("    Windows:   set ALPACA_API_KEY=xxx   and   set ALPACA_SECRET_KEY=yyy")
        print("    Mac/Linux: export ALPACA_API_KEY=xxx   and   export ALPACA_SECRET_KEY=yyy")
        return
    client = TradingClient(key, secret, paper=True)       # PAPER ONLY - no real money
    k = choose("Alpaca PAPER trading (practice money)", [("acct", "Show account and open positions"),
                                                         ("trade", "Scan and place bracket orders for today's signals"),
                                                         ("orders", "Show open orders")])
    if k == "acct":
        a = client.get_account()
        print(f"\n  Equity ${float(a.equity):,.2f} | Cash ${float(a.cash):,.2f} | Buying power ${float(a.buying_power):,.2f}")
        pos = client.get_all_positions()
        for p in pos:
            print(f"  {p.symbol:<6} {p.qty:>6} @ {float(p.avg_entry_price):.2f}  now {float(p.current_price):.2f}  "
                  f"P/L ${float(p.unrealized_pl):,.2f}")
        if not pos:
            print("  No open positions.")
    elif k == "orders":
        for o in client.get_orders():
            print(f"  {o.symbol:<6} {o.side} {o.qty} {o.type} {o.status}")
    elif k == "trade":
        tickers = pick_tickers(s)
        data = fetch(tickers, 2)
        bench = get_bench(s, 2)
        held = {p.symbol for p in client.get_all_positions()}
        equity = float(client.get_account().equity)
        orders = []
        for t, df in data.items():
            d = indicators(df, bench)
            sig, stop, _ = STRATEGIES[s["strategy"]]["fn"](d)
            if not sig.iloc[-1] or t in held:
                continue
            px, st = d.Close.iloc[-1], stop.iloc[-1]
            r = px - st
            qty = min(int(equity * s["risk_pct"] / 100 / r), int(equity * 0.2 / px)) if r > 0 else 0
            if qty >= 1:
                orders.append((t, qty, px, st, px + s["target_r"] * r))
        if not orders:
            print(f"\n  No {STRATEGIES[s['strategy']]['name']} signals today (or already held).")
            return
        print(f"\n{BOLD}Proposed PAPER bracket orders{END} ({STRATEGIES[s['strategy']]['name']}):")
        for t, q, px, st, tp in orders:
            print(f"  BUY {q:>5} {t:<6} ~{px:.2f}   stop {st:.2f}   target {tp:.2f}")
        if input("Send these to your Alpaca PAPER account? type YES: ").strip() != "YES":
            print("  Cancelled.")
            return
        for t, q, px, st, tp in orders:
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=t, qty=q, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
                    stop_loss=StopLossRequest(stop_price=round(st, 2))))
                print(f"  {GREEN}Sent {t}{END}")
            except Exception as e:
                print(f"  {RED}{t} failed: {e}{END}")


# =============================================================== 6. guide
def guide_menu(s):
    k = choose("Strategy guide", [(key, f"{v['name']}  (from {v['src']})") for key, v in STRATEGIES.items()]
               + [("exits", "Exit modes explained")])
    if k is None:
        return
    if k == "exits":
        for key, v in EXITS.items():
            print(f"  - {key:<10} {v}")
        print("  Tip: combine 'targets' with a trailing stop, or use 'ema_exit' + 'time' for short swings.")
        return
    v = STRATEGIES[k]
    print(f"\n{BOLD}{v['name']}{END}\n{v['rules']}\n")
    if input("Make this my selected strategy? (y/N): ").strip().lower() == "y":
        s["strategy"] = k
        save_settings(s)


# ============================================================= 7. pkscreener
def pkscreener_menu(s):
    exe = shutil.which("pkscreener")
    if not exe:
        print("  PKScreener (NSE India screener) is not installed. Install it with:  pip install pkscreener")
        print("  Or use Stock Lab with Market = INDIA in Settings (Nifty 50 scan, all strategies, backtests).")
        return
    print("  Launching PKScreener... (close it to come back here)")
    subprocess.call([exe])


# ============================================================== 8. settings
def settings_menu(s):
    while True:
        c = cur(s)
        k = choose("Settings", [
            ("market", f"Market: {s['market']}"),
            ("strategy", f"Strategy: {STRATEGIES[s['strategy']]['name']}"),
            ("exits", f"Exit modes: {', '.join(s['exits']) or 'stop only'}"),
            ("account", f"Account size: {c}{s['account']:,.0f}"),
            ("risk", f"Risk per trade: {s['risk_pct']}%"),
            ("exitp", f"Exit settings: target {s['target_r']}R, ATR trail {s['trail_atr']}x, "
                      f"time {s['max_days']}d, % trail +{s['pct_activate']}%/{s['pct_trail']}%"),
            ("years", f"Backtest history: {s['years']} years"),
            ("watch", "Edit watchlist"),
            ("reset", "Reset to defaults")])
        if k is None:
            save_settings(s)
            return
        if k == "market":
            m = choose("Market", [("US", "United States (NYSE / Nasdaq)"), ("INDIA", "India (NSE)")])
            s["market"] = m or s["market"]
        elif k == "strategy":
            s["strategy"] = choose("Strategy", [(key, v["name"]) for key, v in STRATEGIES.items()]) or s["strategy"]
        elif k == "exits":
            s["exits"] = choose_multi("Exit modes", list(EXITS.items()), s["exits"])
        elif k == "account":
            s["account"] = ask_num("Account size", float(s["account"]))
        elif k == "risk":
            s["risk_pct"] = ask_num("Risk % per trade (0.5-2 is typical)", float(s["risk_pct"]))
        elif k == "exitp":
            s["target_r"] = ask_num("Target = N x risk", float(s["target_r"]))
            s["trail_atr"] = ask_num("ATR trailing multiple", float(s["trail_atr"]))
            s["max_days"] = ask_num("Time exit after N days", int(s["max_days"]))
            s["pct_activate"] = ask_num("% trail activates at +%", float(s["pct_activate"]))
            s["pct_trail"] = ask_num("% trail distance", float(s["pct_trail"]))
        elif k == "years":
            s["years"] = ask_num("Years of history", int(s["years"]))
        elif k == "watch":
            old = open(WATCHLIST_FILE).read().split() if os.path.exists(WATCHLIST_FILE) else []
            print(f"  Current: {' '.join(old) or '(empty)'}")
            a = input("  Type tickers to ADD, or '-TICKER' to remove: ").upper().split()
            new = [t for t in old if f"-{t}" not in a] + [t for t in a if not t.startswith("-")]
            with open(WATCHLIST_FILE, "w") as f:
                f.write("\n".join(dict.fromkeys(new)))
            print(f"  Watchlist: {' '.join(dict.fromkeys(new))}")
        elif k == "reset":
            s.clear(); s.update(DEFAULTS)
        save_settings(s)


# =================================================================== main
def main():
    global DEMO
    ap = argparse.ArgumentParser(description="Stock Lab - swing trading toolkit")
    ap.add_argument("--demo", action="store_true", help="simulated data, no internet needed")
    DEMO = ap.parse_args().demo
    s = load_settings()
    menu = [("analyze", "Analyze stocks - entry, stop & target prices"),
            ("scan", "Scan the market (Minervini / signals / score)"),
            ("backtest", "Backtest a strategy (backtesting.py)"),
            ("optimize", "Optimize stop & target (vectorbt)"),
            ("alpaca", "Paper trade with Alpaca (US)"),
            ("guide", "Strategy guide - rules for each strategy"),
            ("pk", "PKScreener - India NSE screener"),
            ("settings", "Settings")]
    funcs = dict(analyze=analyze_menu, scan=scan_menu, backtest=backtest_menu, optimize=optimize_menu,
                 alpaca=alpaca_menu, guide=guide_menu, pk=pkscreener_menu, settings=settings_menu)
    while True:
        print(f"\n{BOLD}{'=' * 60}\n STOCK LAB {'(DEMO DATA)' if DEMO else ''}   market {s['market']} | "
              f"{STRATEGIES[s['strategy']]['name']} | risk {s['risk_pct']}%\n{'=' * 60}{END}")
        k = choose("Main menu", menu + [("quit", "Quit")], allow_back=False)
        if k == "quit":
            print("Bye. Educational tool only - not financial advice.")
            return
        try:
            funcs[k](s)
        except KeyboardInterrupt:
            print("\n  (cancelled)")
        except Exception as e:
            print(f"  {RED}Error: {e}{END}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
