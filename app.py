"""Streamlit dashboard for the four-agent paper-trading simulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trading_sim.agents import AGENT_DESCRIPTIONS  # noqa: E402
from trading_sim.config import (  # noqa: E402
    AGENT_LABELS,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK,
    DEFAULT_SYMBOL,
    OPENAI_API_KEY,
    STARTING_CASH,
)
from trading_sim.db import Database  # noqa: E402
from trading_sim.engine import SimulationEngine  # noqa: E402
from trading_sim.research_agent import ResearchAgent  # noqa: E402

st.set_page_config(
    page_title="Trend Arena",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

INTERVALS = ["15m", "1h", "4h", "1d"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


@st.cache_resource
def get_engine() -> SimulationEngine:
    return SimulationEngine()


def fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def agent_snapshot(db: Database, mark_price: float | None) -> pd.DataFrame:
    agents = db.get_agents()
    if agents.empty:
        return agents
    price = mark_price or 0.0
    agents["label"] = agents["id"].map(AGENT_LABELS)
    agents["mark_equity"] = agents["cash"] + agents["position_qty"] * price
    agents["return_pct"] = (agents["mark_equity"] / STARTING_CASH - 1) * 100
    agents["exposure"] = agents.apply(
        lambda r: "LONG" if r["position_qty"] > 0 else "FLAT",
        axis=1,
    )
    return agents.sort_values("mark_equity", ascending=False)


def main() -> None:
    engine = get_engine()
    db = engine.db

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .block-container { padding-top: 1.4rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Trend Arena")
    st.caption(
        "Four deterministic trend agents paper-trade Binance public market data. "
        "No market-data API keys. Optional AI research runs once on demand."
    )

    with st.sidebar:
        st.header("Controls")
        symbol = st.selectbox("Symbol", SYMBOLS, index=SYMBOLS.index(DEFAULT_SYMBOL) if DEFAULT_SYMBOL in SYMBOLS else 0)
        interval = st.selectbox(
            "Interval",
            INTERVALS,
            index=INTERVALS.index(DEFAULT_INTERVAL) if DEFAULT_INTERVAL in INTERVALS else 1,
        )
        lookback = st.slider("Lookback candles", 100, 1000, DEFAULT_LOOKBACK, 50)

        run_bt = st.button("Run full backtest", type="primary", use_container_width=True)
        step = st.button("Fetch latest & step agents", use_container_width=True)
        research = st.button("Daily research report", use_container_width=True)
        reset = st.button("Reset paper books", use_container_width=True)

        st.divider()
        st.markdown("**Cost posture**")
        st.write("Market data · agents · SQLite · dashboard → **$0**")
        st.write(
            "Optional LLM research → "
            + ("API key detected" if OPENAI_API_KEY else "rule-based fallback (no key)")
        )

    if reset:
        engine.reset()
        engine.db.ensure_agents(AGENT_LABELS, AGENT_DESCRIPTIONS)
        st.success("Paper books reset to starting cash.")
        st.cache_resource.clear()
        st.rerun()

    if run_bt:
        with st.spinner(f"Backtesting {symbol} {interval} on Binance public data…"):
            try:
                result = engine.backtest(symbol=symbol, interval=interval, limit=lookback)
                st.success(
                    f"Processed {result.bars_processed} bars · {result.trades} fills · "
                    f"last price {fmt_money(next(iter(result.latest_prices.values())))}"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Backtest failed: {exc}")

    if step:
        with st.spinner("Syncing latest candles…"):
            try:
                result = engine.step_live(symbol=symbol, interval=interval, limit=lookback)
                st.success(f"Stepped agents on latest bar · {result.trades} new fills")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Live step failed: {exc}")

    if research:
        with st.spinner("Writing research note…"):
            report = ResearchAgent(db).run(prefer_llm=bool(OPENAI_API_KEY))
            st.info(f"**{report.source}** — {report.summary}")
            for p in report.proposals:
                st.write(f"- {p}")

    candles = db.load_candles(symbol, interval)
    mark = float(candles["close"].iloc[-1]) if not candles.empty else None
    snap = agent_snapshot(db, mark)
    equity = db.get_equity()
    trades = db.get_trades()
    notes = db.get_research_notes(5)

    if snap.empty or (equity.empty and trades.empty and candles.empty):
        st.info("No simulation data yet. Choose a symbol and run a full backtest from the sidebar.")
        with st.expander("Agent roster", expanded=True):
            for aid, desc in AGENT_DESCRIPTIONS.items():
                st.markdown(f"**{AGENT_LABELS[aid]}** (`{aid}`) — {desc}")
        return

    # KPI row
    cols = st.columns(4)
    for i, (_, row) in enumerate(snap.iterrows()):
        with cols[i % 4]:
            st.metric(
                row["label"],
                fmt_money(row["mark_equity"]),
                fmt_pct(row["return_pct"]),
            )

    left, right = st.columns([1.55, 1])

    with left:
        st.subheader("Equity curves")
        if equity.empty:
            st.write("Equity history appears after a backtest or live step.")
        else:
            plot_df = equity.copy()
            plot_df["label"] = plot_df["agent_id"].map(AGENT_LABELS)
            plot_df["ts"] = pd.to_datetime(plot_df["ts"], utc=True)
            fig = px.line(
                plot_df,
                x="ts",
                y="equity",
                color="label",
                labels={"ts": "Time", "equity": "Equity (USDT)", "label": "Agent"},
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(248,249,251,0.9)",
            )
            fig.add_hline(y=STARTING_CASH, line_dash="dot", line_color="#888")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price")
        if candles.empty:
            st.write("No candles stored.")
        else:
            price_fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=candles["open_time"],
                        open=candles["open"],
                        high=candles["high"],
                        low=candles["low"],
                        close=candles["close"],
                        name=symbol,
                    )
                ]
            )
            price_fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_rangeslider_visible=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(price_fig, use_container_width=True)

    with right:
        st.subheader("Leaderboard")
        board = snap[
            ["label", "mark_equity", "return_pct", "exposure", "position_qty", "cash"]
        ].rename(
            columns={
                "label": "Agent",
                "mark_equity": "Equity",
                "return_pct": "Return %",
                "exposure": "Side",
                "position_qty": "Qty",
                "cash": "Cash",
            }
        )
        st.dataframe(
            board.style.format(
                {"Equity": "${:,.2f}", "Return %": "{:+.2f}", "Qty": "{:.6f}", "Cash": "${:,.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Strategies")
        for aid, desc in AGENT_DESCRIPTIONS.items():
            st.markdown(f"**{AGENT_LABELS[aid]}** — {desc}")

        st.subheader("Research notes")
        if notes.empty:
            st.write("No research notes yet. Generate a daily report from the sidebar.")
        else:
            for _, note in notes.iterrows():
                with st.expander(f"{note['created_at'][:19]} · {note['source']}", expanded=False):
                    st.write(note["summary"])
                    for p in json.loads(note["proposals_json"]):
                        st.write(f"- {p}")

    st.subheader("Recent fills")
    if trades.empty:
        st.write("No trades yet.")
    else:
        show = trades.head(40).copy()
        show["agent"] = show["agent_id"].map(AGENT_LABELS)
        st.dataframe(
            show[["ts", "agent", "side", "qty", "price", "fee", "reason"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
