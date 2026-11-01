"""Streamlit dashboard: 4 panels over the autopilot ledger.

Reads SQLite directly (sync sqlite3) rather than calling the FastAPI app:
the dashboard refreshes on its own cadence and stays usable when the API is
down. The file is opened read-only so a misclick cannot alter the ledger.

Run with:
    uv run --extra dashboard streamlit run src/llm_cost_autopilot/dashboard/app.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from llm_cost_autopilot.config import get_settings

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="🤖",
    layout="wide",
)


def _read_only_uri(path: str) -> str:
    """Return a `file:` URI that opens the DB in read-only mode."""
    abs_path = Path(path).expanduser().resolve()
    return f"file:{abs_path}?mode=ro"


@st.cache_data(ttl=15, show_spinner=False)
def _load_routing(db_path: str) -> pd.DataFrame:
    """Routing log ordered by time, with a derived ``saved_usd`` column (cached 15 s)."""
    with sqlite3.connect(_read_only_uri(db_path), uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM routing_log ORDER BY created_at",
            conn,
            parse_dates=["created_at"],
        )
    if not df.empty:
        df["saved_usd"] = df["baseline_cost_usd"] - df["cost_usd"]
    return df


@st.cache_data(ttl=15, show_spinner=False)
def _load_corrections(db_path: str) -> pd.DataFrame:
    """Correction rows ordered by time (cached 15 s)."""
    with sqlite3.connect(_read_only_uri(db_path), uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM corrections ORDER BY created_at",
            conn,
            parse_dates=["created_at"],
        )
    return df


settings = get_settings()

with st.sidebar:
    st.markdown("### LLM Cost Autopilot")
    st.caption("Two-stage zero-shot routing • verifier-driven feedback loop")
    db_path = st.text_input("SQLite path", value=settings.sqlite_path)
    if st.button("Refresh data", width="stretch"):
        _load_routing.clear()
        _load_corrections.clear()
        st.rerun()

if not Path(db_path).exists():
    st.warning(
        f"No ledger at `{db_path}` yet. Send some traffic to "
        "`POST /v1/chat/completions` and refresh."
    )
    st.stop()

routing = _load_routing(db_path)
corrections = _load_corrections(db_path)

if routing.empty:
    st.info("Ledger is empty: no routing decisions recorded yet.")
    st.stop()


total_actual = float(routing["cost_usd"].sum())
total_baseline = float(routing["baseline_cost_usd"].sum())
total_saved = total_baseline - total_actual
savings_pct = (total_saved / total_baseline * 100.0) if total_baseline > 0 else 0.0

standard_model = settings.moderate_model
verifiable = int((routing["model_used"] != standard_model).sum())
correction_count = len(corrections)
accuracy_pct = (1.0 - correction_count / verifiable) * 100.0 if verifiable > 0 else 100.0
accuracy_pct = max(0.0, min(100.0, accuracy_pct))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Requests routed", f"{len(routing):,}")
m2.metric("Total saved", f"${total_saved:,.4f}", f"{savings_pct:.1f}%")
m3.metric("Routing accuracy", f"{accuracy_pct:.1f}%")
m4.metric("Corrections logged", f"{correction_count:,}")


tab_cost, tab_acc, tab_corr, tab_compare = st.tabs(
    ["💰 Cost savings", "🎯 Routing accuracy", "🔍 Corrections", "⚖️ Model comparison"]
)


# Panel 1: Cost savings ----------------------------------------------------
with tab_cost:
    st.subheader("Cumulative savings vs. routing everything to the standard model")

    cum = routing[["created_at", "cost_usd", "baseline_cost_usd", "saved_usd"]].copy()
    cum = cum.set_index("created_at").sort_index()
    cum["cum_actual"] = cum["cost_usd"].cumsum()
    cum["cum_baseline"] = cum["baseline_cost_usd"].cumsum()
    cum["cum_saved"] = cum["saved_usd"].cumsum()

    fig = px.area(
        cum.reset_index(),
        x="created_at",
        y="cum_saved",
        title="Cumulative USD saved over time",
        labels={"cum_saved": "Saved (USD)", "created_at": "Time"},
    )
    fig.update_layout(height=380, margin={"t": 40, "b": 20, "l": 0, "r": 0})
    st.plotly_chart(fig, width="stretch")

    tier_counts = routing["classified_tier"].value_counts().reset_index()
    tier_counts.columns = ["tier", "count"]
    donut = px.pie(
        tier_counts,
        values="count",
        names="tier",
        title="Requests by complexity tier",
        hole=0.55,
    )
    donut.update_layout(height=320, margin={"t": 40, "b": 0, "l": 0, "r": 0})
    st.plotly_chart(donut, width="stretch")


# Panel 2: Routing accuracy trend -----------------------------------------
with tab_acc:
    st.subheader("Routing accuracy (verifier agreement) over time")

    # About 20 equal windows whatever the ledger size, so the trend stays readable.
    window_size = max(1, len(routing) // 20)
    routing_sorted = routing.sort_values("created_at").reset_index(drop=True)
    routing_sorted["window"] = routing_sorted.index // window_size

    correction_request_ids = (
        set(corrections["request_id"].tolist()) if not corrections.empty else set()
    )
    routing_sorted["was_correction"] = (
        routing_sorted["request_id"].isin(correction_request_ids).astype(int)
    )
    routing_sorted["verifiable"] = (routing_sorted["model_used"] != standard_model).astype(int)

    agg = (
        routing_sorted.groupby("window")
        .agg(
            verifiable=("verifiable", "sum"),
            corrections=("was_correction", "sum"),
            t=("created_at", "max"),
        )
        .reset_index()
    )
    agg["accuracy_pct"] = agg.apply(
        lambda r: (
            (1.0 - r["corrections"] / r["verifiable"]) * 100.0 if r["verifiable"] > 0 else 100.0
        ),
        axis=1,
    )

    fig = px.line(
        agg,
        x="t",
        y="accuracy_pct",
        markers=True,
        title=f"Routing accuracy: windows of {window_size} requests",
        labels={"t": "Time", "accuracy_pct": "Accuracy (%)"},
    )
    fig.update_yaxes(range=[0, 105])
    fig.update_layout(height=380, margin={"t": 40, "b": 20, "l": 0, "r": 0})
    st.plotly_chart(fig, width="stretch")

    if len(agg) >= 2:
        first_acc = float(agg["accuracy_pct"].iloc[0])
        last_acc = float(agg["accuracy_pct"].iloc[-1])
        delta = last_acc - first_acc
        st.caption(
            f"Started at **{first_acc:.1f}%**, now at **{last_acc:.1f}%** "
            f"({delta:+.1f} pp after {correction_count} corrections)."
        )


# Panel 3: Correction dataset explorer ------------------------------------
with tab_corr:
    st.subheader("Routing failures: the prompts the verifier flagged")

    if corrections.empty:
        st.info("No routing failures yet: the verifier hasn't disagreed.")
    else:
        max_div = float(corrections["divergence_threshold"].max())
        min_sim = st.slider(
            "Show failures with similarity below…",
            min_value=0.0,
            max_value=max_div,
            value=max_div,
            step=0.05,
        )
        filtered = corrections[corrections["similarity_score"] <= min_sim].copy()
        filtered = filtered.sort_values("created_at", ascending=False)
        filtered["prompt_preview"] = filtered["prompt"].str.slice(0, 80) + "…"

        st.dataframe(
            filtered[
                [
                    "created_at",
                    "request_id",
                    "cheap_tier",
                    "classifier_confidence",
                    "similarity_score",
                    "prompt_preview",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(f"{len(filtered)} of {len(corrections)} corrections shown.")


# Panel 4: Model comparison -----------------------------------------------
with tab_compare:
    st.subheader("Side-by-side: budget model vs. standard model")

    if corrections.empty:
        st.info("No comparisons yet: the verifier hasn't produced any.")
    else:
        recent = corrections.sort_values("created_at", ascending=False).head(20)
        labels = recent.apply(
            lambda r: (
                f"{r['created_at']:%H:%M:%S} · {r['request_id'][:12]}… "
                f"(sim {r['similarity_score']:.2f})"
            ),
            axis=1,
        ).tolist()
        choice = st.selectbox(
            "Pick a correction to inspect",
            options=list(range(len(recent))),
            format_func=lambda i: labels[i],
        )
        row = recent.iloc[choice]

        st.markdown(f"**Prompt:** {row['prompt']}")
        st.markdown(
            f"Similarity **{row['similarity_score']:.3f}** "
            f"(threshold {row['divergence_threshold']:.2f}): "
            f"classifier confidence **{row['classifier_confidence']:.2f}** "
            f"for tier **{row['cheap_tier']}**"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### Budget · `{row['cheap_model']}`")
            st.text_area(
                "Cheap response",
                value=row["cheap_response"],
                height=300,
                label_visibility="collapsed",
            )
        with c2:
            st.markdown(f"#### Standard · `{row['standard_model']}`")
            st.text_area(
                "Standard response",
                value=row["standard_response"],
                height=300,
                label_visibility="collapsed",
            )
