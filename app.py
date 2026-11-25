"""
STOCK LAB - web app (runs in your browser)
==========================================
Run:   pip install -r requirements.txt
       streamlit run app.py
Opens http://localhost:8501 in your browser.
Needs stock_lab.py in the same folder (it is the analysis engine).
Educational tool only - not financial advice.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import stock_lab as sl

st.set_page_config(page_title="Stock Lab - Swing Trading", page_icon="📈", layout="wide")

VERDICT_COLOR = {"BUY ZONE": "#2b8a3e", "WATCH": "#e67700", "AVOID": "#c92a2a"}

# ------------------------------------------------------------------ settings (sidebar)
if "s" not in st.session_state:
    st.session_state.s = sl.load_settings()
s = st.session_state.s

with st.sidebar:
    st.title("📈 Stock Lab")
    demo = st.toggle("Demo data (no internet)", value=os.environ.get("STOCKLAB_DEMO") == "1",
                     help="Simulated prices for practice. Turn off for real Yahoo Finance data.")
    s["market"] = st.radio("Market", ["US", "INDIA"], key="market", index=["US", "INDIA"].index(s["market"]), horizontal=True)
    keys = list(sl.STRATEGIES)
    s["strategy"] = st.selectbox("Strategy", keys, index=keys.index(s["strategy"]),
                                 format_func=lambda k: sl.STRATEGIES[k]["name"])
    s["exits"] = st.multiselect("Exit modes", list(sl.EXITS), default=s["exits"],
                                format_func=lambda k: sl.EXITS[k].split(" (")[0])
    c1, c2 = st.columns(2)
    s["account"] = c1.number_input("Account", 100.0, 1e9, float(s["account"]), step=1000.0)
    s["risk_pct"] = c2.number_input("Risk %", 0.1, 10.0, float(s["risk_pct"]), step=0.25)
    with st.expander("Exit settings"):
        s["target_r"] = st.number_input("Target = N x risk", 0.5, 10.0, float(s["target_r"]), 0.5)
        s["trail_atr"] = st.number_input("ATR trailing multiple", 0.5, 10.0, float(s["trail_atr"]), 0.5)
        s["max_days"] = st.number_input("Time exit (days)", 1, 200, int(s["max_days"]))
        s["pct_activate"] = st.number_input("% trail activates at +%", 0.5, 50.0, float(s["pct_activate"]), 0.5)
        s["pct_trail"] = st.number_input("% trail distance", 0.5, 30.0, float(s["pct_trail"]), 0.5)
        s["years"] = st.slider("Backtest history (years)", 2, 10, int(s["years"]))
    wl_old = open(sl.WATCHLIST_FILE).read() if os.path.exists(sl.WATCHLIST_FILE) else ""
    wl = st.text_area("Watchlist (one ticker per line)", wl_old, height=110)
    if wl != wl_old:
        with open(sl.WATCHLIST_FILE, "w") as f:
            f.write(wl)
    sl.save_settings(s)
    st.caption("Educational tool only — not financial advice.")

CUR = sl.cur(s)


# ------------------------------------------------------------------ data helpers
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(tickers: tuple, years: int, demo_mode: bool):
    sl.DEMO = demo_mode
    return sl.fetch(list(tickers), years)


def load(tickers, years):
    with st.spinner(f"Downloading {len(tickers)} symbol(s)..."):
        data = get_data(tuple(tickers), years, demo)
        bench = get_data((sl.benchmark(s),), years, demo).get(sl.benchmark(s))
    missing = [t for t in tickers if t not in data]
    if missing:
        st.warning(f"No data / too little history for: {', '.join(missing[:20])}")
    return data, bench


@st.cache_data(ttl=86400, show_spinner=False)
def wiki_list(which):
    if which == "sp500":
        return sl.wiki_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"])
    return sl.wiki_symbols("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Symbol"])


def universe_picker(key, multi=True, default="AAPL NVDA MSFT"):
    opts = {"type": "Type tickers", "watch": "My watchlist"}
    if multi:
        opts.update({"large": "Top 50 US large caps", "sp500": "S&P 500", "ndx": "Nasdaq 100"}
                    if s["market"] == "US" else {"nifty": "Nifty 50"})
    if s["market"] == "INDIA" and default == "AAPL NVDA MSFT":
        default = "RELIANCE TCS HDFCBANK"
    c1, c2 = st.columns([1, 2])
    src = c1.selectbox("Stocks", list(opts), format_func=opts.get, key=f"{key}_src")
    raw = []
    if src == "type":
        raw = c2.text_input("Tickers (space separated)" if multi else "Ticker", default if multi else default.split()[0],
                            key=f"{key}_txt").split()
    elif src == "watch":
        raw = wl.split()
        c2.caption(f"{len(raw)} tickers from your watchlist" if raw else "Your watchlist is empty (sidebar).")
    elif src == "large":
        raw = sl.US_LARGE
    elif src == "nifty":
        raw = sl.NIFTY50
    else:
        try:
            if demo:
                raise RuntimeError("demo mode")
            raw = wiki_list(src)
        except Exception as e:
            c2.warning(f"Could not download list ({e}); using top 50 US.")
            raw = sl.US_LARGE
        c2.caption(f"{len(raw)} stocks")
    out = [sl.sym(t, s) for t in raw]
    return out if multi else out[:1]


# ------------------------------------------------------------------ chart
def price_chart(t, d, plan, days=180):
    v = d.iloc[-days:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name=t,
                                 increasing_line_color="#2b8a3e", decreasing_line_color="#c92a2a"), 1, 1)
    for col, clr in (("SMA20", "#1c7ed6"), ("SMA50", "#f08c00"), ("SMA200", "#7048e8")):
        fig.add_trace(go.Scatter(x=v.index, y=v[col], name=col, line=dict(width=1.3, color=clr)), 1, 1)
    if plan:
        for y, lab, clr, dash in ((plan["entry"], "Entry", "#2b8a3e", "dash"), (plan["stop"], "Stop", "#c92a2a", "dash"),
                                  (plan["t1"], "Target 1", "#1971c2", "dot"), (plan["t2"], "Target 2", "#1971c2", "dot")):
            fig.add_hline(y=y, line=dict(color=clr, dash=dash, width=1.2), row=1, col=1,
                          annotation_text=f"{lab} {y:.2f}", annotation_position="right",
                          annotation_font=dict(color=clr, size=11))
    fig.add_trace(go.Bar(x=v.index, y=v.Volume, name="Volume", marker_color="#adb5bd"), 2, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.RSI, name="RSI", line=dict(color="#6741d9", width=1.3)), 3, 1)
    for lvl in (70, 30):
        fig.add_hline(y=lvl, line=dict(color="grey", dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=90, t=30, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    return fig


# ------------------------------------------------------------------ pages
tabs = st.tabs(["🔍 Analyze", "📡 Scanner", "🧪 Backtest", "⚙️ Optimize", "💼 Paper trade", "📘 Strategy guide"])

# ===== Analyze
with tabs[0]:
    st.subheader("Entry, stop & target prices")
    tickers = universe_picker("an")
    which = st.multiselect("Strategies to show", keys, default=keys,
                           format_func=lambda k: sl.STRATEGIES[k]["name"], key="an_strats")
    if st.button("Analyze", type="primary", key="an_go") and tickers:
        st.session_state.an_res = (tickers, which)
    if "an_res" in st.session_state:
        tickers, which = st.session_state.an_res
        data, bench = load(tickers[:25], max(2, s["years"]))
        if len(tickers) > 25:
            st.info("Showing the first 25 stocks. Use the Scanner for bigger lists.")
        rows = []
        for t, df in data.items():
            d = sl.indicators(df, bench)
            sc, verdict, why = sl.score_row(d)
            x = d.iloc[-1]
            tt = sl.minervini_template(d).iloc[-1]
            plans = {k: sl.build_plan(d, s, k) for k in (which or [s["strategy"]])}
            with st.container(border=True):
                h1, h2 = st.columns([2, 3])
                h1.markdown(f"### {t} &nbsp; <span style='color:{VERDICT_COLOR[verdict]}'>{verdict}</span>",
                            unsafe_allow_html=True)
                h1.caption(f"{d.index[-1].date()} · " + ", ".join(why))
                m = h2.columns(4)
                m[0].metric("Price", f"{CUR}{x.Close:,.2f}", f"{(x.Close / d.Close.iloc[-2] - 1) * 100:+.2f}%")
                m[1].metric("Score", f"{sc}/100")
                m[2].metric("RSI", f"{x.RSI:.0f}")
                m[3].metric("Trend Template", f"{int(tt.sum())}/8")
                tab = pd.DataFrame([{"Strategy": p["strategy"], "Signal today": "✅ YES" if p["signal_today"] else "—",
                                     "Entry": p["entry"], "Stop": p["stop"], "Target 1": p["t1"], "Target 2": p["t2"],
                                     "Shares": p["shares"], "Cost": p["cost"], "Max loss": p["max_loss"],
                                     "Last signal": str(p["last_signal"]), "How": p["note"]} for p in plans.values()])
                st.dataframe(tab, hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.2f")
                                            for c in ("Entry", "Stop", "Target 1", "Target 2")} |
                             {"Cost": st.column_config.NumberColumn(format="%.0f"),
                              "Max loss": st.column_config.NumberColumn(format="%.0f")})
                ck = st.selectbox("Draw levels for", list(plans), format_func=lambda k: sl.STRATEGIES[k]["name"],
                                  key=f"chart_{t}")
                st.plotly_chart(price_chart(t, d, plans[ck]), width="stretch", key=f"pc_{t}")
                trail = d.Close.iloc[-22:].max() - s["trail_atr"] * x.ATR
                warn = " — **price is already below it (exit signal)**" if x.Close < trail else ""
                st.markdown(f"**Already holding?** Exit on a close below **{CUR}{trail:.2f}** (ATR trail){warn} "
                            f"or below the 10 EMA **{CUR}{x.EMA10:.2f}**. Sell half at Target 1, then move the stop "
                            f"to your entry price.")
                with st.expander("Minervini Trend Template checklist"):
                    st.dataframe(pd.DataFrame({"Rule": tt.index, "Pass": ["✅" if v else "❌" for v in tt.values]}),
                                 hide_index=True)
            for p in plans.values():
                rows.append(dict(Ticker=t, Score=sc, Verdict=verdict, Strategy=p["strategy"],
                                 Signal=p["signal_today"], Entry=round(p["entry"], 2), Stop=round(p["stop"], 2),
                                 T1=round(p["t1"], 2), T2=round(p["t2"], 2), Shares=p["shares"]))
        if rows:
            st.download_button("Download results (CSV)", pd.DataFrame(rows).to_csv(index=False), "analysis.csv")

# ===== Scanner
with tabs[1]:
    st.subheader("Market scanner")
    scan = st.radio("Scan type", ["tt", "sig", "all", "score"], horizontal=True, key="scan_type", format_func={
        "tt": "Minervini Trend Template + RS ≥ 70", "sig": "Signals today (selected strategy)",
        "all": "Signals today (all strategies)", "score": "Rank all by score"}.get)
    tickers = universe_picker("sc")
    if st.button("Run scan", type="primary", key="sc_go") and tickers:
        data, bench = load(tickers, 2)
        ds = {t: sl.indicators(df, bench) for t, df in data.items()}
        rs = pd.Series({t: d.RSscore.iloc[-1] for t, d in ds.items()})
        rs_rank = (rs.rank(pct=True) * 99).round().fillna(0).astype(int)
        rows = []
        prog = st.progress(0.0)
        for i, (t, d) in enumerate(ds.items()):
            prog.progress((i + 1) / len(ds))
            x = d.iloc[-1]
            sc, verdict, _ = sl.score_row(d)
            tt = sl.minervini_template(d).iloc[-1]
            hits = [sl.STRATEGIES[k]["name"] for k in (keys if scan == "all" else [s["strategy"]])
                    if sl.STRATEGIES[k]["fn"](d)[0].iloc[-1]]
            if scan == "tt" and not (tt.all() and rs_rank[t] >= 70):
                continue
            if scan in ("sig", "all") and not hits:
                continue
            p = sl.build_plan(d, s, s["strategy"])
            rows.append(dict(Ticker=t, Price=x.Close, Score=sc, Verdict=verdict, TT=int(tt.sum()), RS=int(rs_rank[t]),
                             RSI=x.RSI, FromHigh=(x.Close / x.High252 - 1) * 100, Signals=", ".join(hits),
                             Entry=p["entry"], Stop=p["stop"], Target=p["t1"], Shares=p["shares"]))
        prog.empty()
        st.session_state.scan_res = (pd.DataFrame(rows), len(ds))
    if "scan_res" in st.session_state:
        tab, n = st.session_state.scan_res
        if tab.empty:
            st.info(f"No stocks passed this scan today ({n} checked).")
        else:
            tab = tab.sort_values(["Score", "RS"], ascending=False)
            st.success(f"{len(tab)} of {n} stocks passed. Entry / Stop / Target use your selected strategy.")
            st.dataframe(tab, hide_index=True, width="stretch", height=min(600, 38 + 35 * len(tab)),
                         column_config={
                             "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                             "RS": st.column_config.ProgressColumn("RS rating", min_value=0, max_value=99, format="%d"),
                             "TT": st.column_config.NumberColumn("Trend Tmpl /8"),
                             "FromHigh": st.column_config.NumberColumn("% from 52w high", format="%.1f%%"),
                             **{c: st.column_config.NumberColumn(format="%.2f") for c in ("Price", "Entry", "Stop", "Target")},
                             "RSI": st.column_config.NumberColumn(format="%.0f")})
            c1, c2 = st.columns(2)
            c1.download_button("Download CSV", tab.to_csv(index=False), "scan_results.csv")
            if c2.button("Add all to watchlist"):
                with open(sl.WATCHLIST_FILE, "w") as f:
                    f.write("\n".join(dict.fromkeys(wl.split() + list(tab.Ticker))))
                st.rerun()

# ===== Backtest
with tabs[2]:
    st.subheader("Backtest (backtesting.py)")
    mode = st.radio("Mode", ["one", "cmp", "many"], horizontal=True, key="bt_mode", format_func={
        "one": "One stock · selected strategy", "cmp": "One stock · compare all strategies",
        "many": "Many stocks · selected strategy"}.get)
    tickers = universe_picker("bt", multi=(mode == "many"))
    st.caption(f"Exits: {', '.join(s['exits']) or 'stop only'} · risk {s['risk_pct']}% per trade · "
               f"{s['years']} years · commission {s['commission'] * 100:.2f}%")
    if st.button("Run backtest", type="primary", key="bt_go") and tickers:
        data, bench = load(tickers, s["years"])
        res = []
        prog = st.progress(0.0)
        strat_keys = keys if mode == "cmp" else [s["strategy"]]
        total, i = max(1, len(data) * len(strat_keys)), 0
        for t, df in data.items():
            for k in strat_keys:
                bt, stt = sl.run_bt(sl.prep_bt(df, bench, k), s)
                res.append((t, k, stt))
                i += 1
                prog.progress(i / total)
        prog.empty()
        st.session_state.bt_res = (mode, res)
    if "bt_res" in st.session_state:
        mode_r, res = st.session_state.bt_res
        if mode_r == "one" and res:
            t, k, r = res[0]
            st.markdown(f"#### {t} — {sl.STRATEGIES[k]['name']}")
            m = st.columns(6)
            m[0].metric("Return", f"{r['Return [%]']:.1f}%")
            m[1].metric("Buy & hold", f"{r['Buy & Hold Return [%]']:.1f}%")
            m[2].metric("Trades", int(r["# Trades"]))
            m[3].metric("Win rate", f"{r['Win Rate [%]']:.0f}%" if r["# Trades"] else "–")
            m[4].metric("Max drawdown", f"{r['Max. Drawdown [%]']:.1f}%")
            m[5].metric("Profit factor", f"{r['Profit Factor']:.2f}" if r["# Trades"] else "–")
            eq = r["_equity_curve"]["Equity"]
            fig = go.Figure(go.Scatter(x=eq.index, y=eq, line=dict(color="#1c7ed6"), name="Equity"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Account equity")
            st.plotly_chart(fig, width="stretch")
            if r["# Trades"]:
                tr = r["_trades"][["EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size", "PnL", "ReturnPct"]].copy()
                tr["ReturnPct"] *= 100
                st.dataframe(tr.round(2), hide_index=True, width="stretch")
            with st.expander("All statistics"):
                st.dataframe(r.drop(["_strategy", "_equity_curve", "_trades"]).astype(str), width="stretch")
        else:
            tab = pd.DataFrame([dict(Ticker=t, Strategy=sl.STRATEGIES[k]["name"], Trades=int(r["# Trades"]),
                                     WinRate=r["Win Rate [%]"] if r["# Trades"] else 0, Return=r["Return [%]"],
                                     BuyHold=r["Buy & Hold Return [%]"], MaxDD=r["Max. Drawdown [%]"],
                                     ProfitFactor=r["Profit Factor"] if r["# Trades"] else 0,
                                     AvgTrade=r["Avg. Trade [%]"] if r["# Trades"] else 0) for t, k, r in res])
            if len(tab):
                if mode_r == "many":
                    m = st.columns(4)
                    m[0].metric("Avg return", f"{tab.Return.mean():.1f}%")
                    m[1].metric("Avg buy & hold", f"{tab.BuyHold.mean():.1f}%")
                    m[2].metric("Avg win rate", f"{tab.WinRate.mean():.0f}%")
                    m[3].metric("Total trades", int(tab.Trades.sum()))
                xcol = "Strategy" if mode_r == "cmp" else "Ticker"
                fig = go.Figure([go.Bar(x=tab[xcol], y=tab.Return, name="Strategy return",
                                        marker_color=np.where(tab.Return >= 0, "#2b8a3e", "#c92a2a")),
                                 go.Bar(x=tab[xcol], y=tab.BuyHold, name="Buy & hold", marker_color="#adb5bd")])
                fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=30, b=10), yaxis_title="%")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(tab.round(2), hide_index=True, width="stretch")
                st.download_button("Download CSV", tab.to_csv(index=False), "backtest_results.csv")
        st.warning("Past results don't guarantee future returns. Look for 30+ trades before trusting a result.")

# ===== Optimize
with tabs[3]:
    st.subheader("Optimize stop & target")
    st.caption("Tests stop = 1–3 × ATR and target = 1.5–4 × risk on the first 70% of history (train), then checks "
               "the last 30% (test, unseen). Uses vectorbt if installed. Each trade uses the full account, so compare "
               "settings with each other.")
    tickers = universe_picker("op", multi=False)
    if st.button("Optimize", type="primary", key="op_go") and tickers:
        data, bench = load(tickers, s["years"])
        if data:
            t, df = next(iter(data.items()))
            d = sl.prep_bt(df, bench, s["strategy"])
            split = int(len(d) * 0.7)
            stops, targets = [1.0, 1.5, 2.0, 2.5, 3.0], [1.5, 2.0, 3.0, 4.0]
            rows = []
            try:
                import vectorbt as vbt
                engine = "vectorbt"
                for part, dd in (("train", d.iloc[:split]), ("test", d.iloc[split:])):
                    for sm in stops:
                        for tm in targets:
                            slp = (sm * dd.ATR / dd.Close).values
                            stt = vbt.Portfolio.from_signals(dd.Close, dd.Signal > 0, dd.ExitSig > 0, sl_stop=slp,
                                                             tp_stop=slp * tm, init_cash=s["account"],
                                                             fees=s["commission"], freq="1D").stats()
                            rows.append(dict(Part=part, Stop=sm, Target=tm, Trades=int(stt["Total Trades"]),
                                             Return=stt["Total Return [%]"]))
            except ImportError:
                engine = "backtesting.py"
                for part, dd in (("train", d.iloc[:split]), ("test", d.iloc[split:])):
                    for sm in stops:
                        for tm in targets:
                            _, stt = sl.run_bt(dd, dict(s, target_r=tm, exits=["targets"]), stop_mult=sm)
                            rows.append(dict(Part=part, Stop=sm, Target=tm, Trades=int(stt["# Trades"]),
                                             Return=stt["Return [%]"]))
            st.session_state.op_res = (t, engine, pd.DataFrame(rows))
    if "op_res" in st.session_state:
        t, engine, tab = st.session_state.op_res
        st.markdown(f"#### {t} — {sl.STRATEGIES[s['strategy']]['name']} · engine: {engine}")
        cols = st.columns(2)
        for col, part in zip(cols, ("train", "test")):
            pv = tab[tab.Part == part].pivot(index="Stop", columns="Target", values="Return")
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{c}R" for c in pv.columns], y=[f"{i} ATR" for i in pv.index],
                                       colorscale="RdYlGn", zmid=0, text=np.round(pv.values, 1),
                                       texttemplate="%{text}%", colorbar=dict(title="%")))
            fig.update_layout(title=f"{part.title()} return %", height=360, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_title="Target", yaxis_title="Stop")
            col.plotly_chart(fig, width="stretch")
        tr = tab[tab.Part == "train"].sort_values("Return", ascending=False).iloc[0]
        te = tab[(tab.Part == "test") & (tab.Stop == tr.Stop) & (tab.Target == tr.Target)].iloc[0]
        st.markdown(f"**Best on train:** stop {tr.Stop} × ATR, target {tr.Target} × risk → "
                    f"train **{tr.Return:.1f}%**, test **{te.Return:.1f}%**")
        if te.Return < tr.Return * 0.3:
            st.error("Test result is much worse than train — this setting is probably over-fitted. Don't trust it.")
        if st.button(f"Use target {tr.Target}R in my settings"):
            s["target_r"] = float(tr.Target)
            sl.save_settings(s)
            st.rerun()

# ===== Paper trade
with tabs[4]:
    st.subheader("Alpaca paper trading (practice money, US stocks)")
    if s["market"] != "US":
        st.info("Alpaca trades US stocks only. Switch Market to US in the sidebar.")
    else:
        st.caption("Get free paper keys at alpaca.markets → Paper Trading → API Keys. Keys stay in this browser "
                   "session only. This app is locked to the PAPER account.")
        c1, c2 = st.columns(2)
        key = c1.text_input("API key", os.environ.get("ALPACA_API_KEY", ""), type="password")
        secret = c2.text_input("Secret key", os.environ.get("ALPACA_SECRET_KEY", ""), type="password")
        if key and secret:
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
                client = TradingClient(key, secret, paper=True)
                a = client.get_account()
                m = st.columns(3)
                m[0].metric("Equity", f"${float(a.equity):,.2f}")
                m[1].metric("Cash", f"${float(a.cash):,.2f}")
                m[2].metric("Buying power", f"${float(a.buying_power):,.2f}")
                pos = client.get_all_positions()
                if pos:
                    st.dataframe(pd.DataFrame([dict(Symbol=p.symbol, Qty=p.qty, Entry=float(p.avg_entry_price),
                                                    Now=float(p.current_price), PnL=float(p.unrealized_pl))
                                               for p in pos]).round(2), hide_index=True)
                st.markdown(f"**Find today's {sl.STRATEGIES[s['strategy']]['name']} signals and propose bracket orders**")
                tickers = universe_picker("al")
                if st.button("Scan for orders", key="al_go"):
                    data, bench = load(tickers, 2)
                    held, equity, orders = {p.symbol for p in pos}, float(a.equity), []
                    for t, df in data.items():
                        d = sl.indicators(df, bench)
                        sig, stop, _ = sl.STRATEGIES[s["strategy"]]["fn"](d)
                        if not sig.iloc[-1] or t in held:
                            continue
                        px, stp = d.Close.iloc[-1], stop.iloc[-1]
                        r = px - stp
                        q = min(int(equity * s["risk_pct"] / 100 / r), int(equity * 0.2 / px)) if r > 0 else 0
                        if q >= 1:
                            orders.append(dict(Symbol=t, Qty=q, Price=round(px, 2), Stop=round(stp, 2),
                                               Target=round(px + s["target_r"] * r, 2)))
                    st.session_state.orders = orders
                orders = st.session_state.get("orders")
                if orders is not None:
                    if not orders:
                        st.info("No signals today (or you already hold them).")
                    else:
                        st.dataframe(pd.DataFrame(orders), hide_index=True)
                        ok = st.checkbox("I understand these are PAPER orders with a stop and target attached")
                        if st.button("Send orders", type="primary", disabled=not ok):
                            for o in orders:
                                try:
                                    client.submit_order(MarketOrderRequest(
                                        symbol=o["Symbol"], qty=o["Qty"], side=OrderSide.BUY,
                                        time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
                                        take_profit=TakeProfitRequest(limit_price=o["Target"]),
                                        stop_loss=StopLossRequest(stop_price=o["Stop"])))
                                    st.success(f"Sent {o['Symbol']}")
                                except Exception as e:
                                    st.error(f"{o['Symbol']} failed: {e}")
                            st.session_state.orders = None
            except ImportError:
                st.error("Install first: pip install alpaca-py")
            except Exception as e:
                st.error(f"Alpaca error: {e}")

# ===== Guide
with tabs[5]:
    st.subheader("Strategy guide")
    for k, v in sl.STRATEGIES.items():
        with st.expander(f"{v['name']}  ·  from {v['src']}", expanded=(k == s["strategy"])):
            st.text(v["rules"])
    st.markdown("**Exit modes**")
    for k, v in sl.EXITS.items():
        st.markdown(f"- **{k}** — {v}")
    st.markdown("Tip: combine *targets* with a trailing stop, or use *EMA exit* + *time exit* for short swings.  \n"
                "For Indian stocks you can also install **PKScreener** (`pip install pkscreener`) and run `pkscreener` "
                "in a terminal.")
