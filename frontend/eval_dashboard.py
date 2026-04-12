"""Streamlit eval dashboard — retrieval & generation metrics across RAG variants.

Visualises experiment results from the eval harness that compares three
retrieval configurations (librarian / raptor / bedrock).

Data sources (in priority order):
    1. Uploaded JSON file (from ``uv run python -m eval.experiment run --export results.json``)
    2. Langfuse API (when LANGFUSE_ENABLED=true)

Run:
    streamlit run frontend/eval_dashboard.py
"""

from __future__ import annotations

import json

import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from eval.dashboard_data import (
    DashboardData,
    load_dashboard_data,
    load_from_dict,
    load_from_langfuse,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_COLOURS: dict[str, str] = {
    "librarian": "#636EFA",
    "raptor": "#EF553B",
    "bedrock": "#00CC96",
}

VARIANT_EMOJI: dict[str, str] = {
    "librarian": "🟦",
    "raptor": "🟥",
    "bedrock": "🟩",
}

# Regression thresholds (match test_retrieval_metrics.py)
HIT_RATE_FLOOR = 0.6
MRR_FLOOR = 0.4

CHART_HEIGHT = 400
CHART_HEIGHT_SM = 350

# Session state key for cached dashboard data
_DATA_KEY = "dashboard_data"


def _colour(variant: str) -> str:
    """Return colour for a variant, with fallback for unknown names."""
    return VARIANT_COLOURS.get(variant, "#AB63FA")


def _emoji(variant: str) -> str:
    """Return emoji badge for a variant."""
    return VARIANT_EMOJI.get(variant, "🔷")


# ---------------------------------------------------------------------------
# Sidebar — data source selector
# ---------------------------------------------------------------------------


def _sidebar_data_source() -> DashboardData:
    """Sidebar controls for loading data. Returns loaded DashboardData."""
    st.sidebar.title("📊 RAG Eval Dashboard")
    st.sidebar.divider()

    source = st.sidebar.radio(
        "Data source",
        ["Upload JSON", "Langfuse API"],
        help="Upload a results.json from `--export`, or pull live from Langfuse.",
    )

    if source == "Upload JSON":
        return _sidebar_upload_json()

    return _sidebar_langfuse()


def _sidebar_upload_json() -> DashboardData:
    """Handle the Upload JSON data source."""
    uploaded = st.sidebar.file_uploader(
        "Results JSON",
        type=["json"],
        help="From: `uv run python -m eval.experiment run --export results.json`",
    )
    if uploaded is not None:
        raw = json.loads(uploaded.getvalue())
        data = load_from_dict(raw)
        st.session_state[_DATA_KEY] = data
        return data

    # Return cached data if available
    if _DATA_KEY in st.session_state:
        return st.session_state[_DATA_KEY]

    st.sidebar.info("Upload a JSON file to get started.")
    return DashboardData(source="none")


def _sidebar_langfuse() -> DashboardData:
    """Handle the Langfuse API data source."""
    st.sidebar.caption("Reads from LANGFUSE_* environment variables.")
    dataset_name = st.sidebar.text_input(
        "Dataset name",
        value="golden_eval",
        help="Langfuse dataset name to fetch.",
    )

    if st.sidebar.button("🔄 Fetch from Langfuse"):
        with st.spinner("Fetching from Langfuse..."):
            data = load_from_langfuse(dataset_name=dataset_name)
            st.session_state[_DATA_KEY] = data
            return data

    # Return cached data if available
    if _DATA_KEY in st.session_state:
        return st.session_state[_DATA_KEY]

    # Auto-fetch on first load only
    data = load_dashboard_data(prefer_langfuse=True, dataset_name=dataset_name)
    if not data.is_empty:
        st.session_state[_DATA_KEY] = data
    return data


# ---------------------------------------------------------------------------
# Panel 1 — Variant comparison overview (headline KPIs)
# ---------------------------------------------------------------------------


def _panel_overview(data: DashboardData) -> None:
    """Headline metric cards for each variant."""
    st.header("Variant Comparison")

    cols = st.columns(max(len(data.variants), 1))
    for col, name in zip(cols, data.variant_names, strict=False):
        v = data.variants[name]
        with col:
            st.subheader(f"{_emoji(name)} {name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Hit Rate", f"{v.hit_rate:.1%}")
            c2.metric("MRR", f"{v.mrr:.3f}")
            c3.metric("Avg Latency", f"{v.avg_latency_ms:.0f}ms")
            st.caption(f"{v.n_hits}/{v.n_queries} queries hit • run: {v.run_name}")

            cs = v.config_snapshot
            if cs:
                with st.expander("Config"):
                    st.json(cs)


# ---------------------------------------------------------------------------
# Panel 2 — Retrieval metrics deep-dive
# ---------------------------------------------------------------------------


def _chart_hit_rate(
    names: list[str], hit_rates: list[float], colours: list[str]
) -> None:
    """Render hit rate bar chart with threshold floor line."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=hit_rates,
            marker_color=colours,
            text=[f"{v:.1%}" for v in hit_rates],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=HIT_RATE_FLOOR,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Floor ({HIT_RATE_FLOOR})",
    )
    fig.update_layout(
        title="Hit Rate @ k",
        yaxis_range=[0, 1.05],
        yaxis_title="Hit Rate",
        showlegend=False,
        height=CHART_HEIGHT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_mrr(names: list[str], mrrs: list[float], colours: list[str]) -> None:
    """Render MRR bar chart with threshold floor line."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=mrrs,
            marker_color=colours,
            text=[f"{v:.3f}" for v in mrrs],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=MRR_FLOOR,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Floor ({MRR_FLOOR})",
    )
    fig.update_layout(
        title="Mean Reciprocal Rank",
        yaxis_range=[0, 1.05],
        yaxis_title="MRR",
        showlegend=False,
        height=CHART_HEIGHT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _panel_retrieval(data: DashboardData) -> None:
    """Bar charts + threshold lines for retrieval metrics."""
    st.header("Retrieval Metrics")

    names = data.variant_names
    hit_rates = [data.variants[n].hit_rate for n in names]
    mrrs = [data.variants[n].mrr for n in names]
    colours = [_colour(n) for n in names]

    col_left, col_right = st.columns(2)
    with col_left:
        _chart_hit_rate(names, hit_rates, colours)
    with col_right:
        _chart_mrr(names, mrrs, colours)

    st.subheader("Latency Distribution")
    _panel_latency(data)


def _panel_latency(data: DashboardData) -> None:
    """Box plots of per-query latency for each variant."""
    fig = go.Figure()
    for name in data.variant_names:
        v = data.variants[name]
        latencies = [qr.latency_ms for qr in v.query_results]
        if latencies:
            fig.add_trace(
                go.Box(
                    y=latencies,
                    name=name,
                    marker_color=_colour(name),
                    boxmean="sd",
                )
            )

    fig.update_layout(
        title="Per-Query Latency (ms)",
        yaxis_title="Latency (ms)",
        height=CHART_HEIGHT_SM,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 3 — Generation quality (answer judge dimensions)
# ---------------------------------------------------------------------------


def _panel_generation(data: DashboardData) -> None:
    """Generation quality radar chart — placeholder until answer judge is wired."""
    st.header("Generation Quality")
    st.info(
        "💡 Generation metrics (faithfulness, relevance, completeness) require "
        "running the **AnswerJudge** grader with `CONFIRM_EXPENSIVE_OPS=True`. "
        "This panel will populate when LLM-as-judge scores are logged to Langfuse."
    )

    st.markdown("""
    **Planned metrics** (from `src/eval/graders/`):
    | Metric | Grader | Source |
    |--------|--------|--------|
    | Faithfulness | `AnswerJudge`, `RagasGrader`, `DeepEvalGrader` | LLM-as-judge |
    | Relevance | `AnswerJudge`, `RagasGrader` | LLM-as-judge |
    | Completeness | `AnswerJudge` | LLM-as-judge |
    | Hallucination | `DeepEvalGrader` | LLM-as-judge |
    | Context Precision | `RagasGrader`, `DeepEvalGrader` | Automated |
    | Context Recall | `RagasGrader`, `DeepEvalGrader` | Automated |
    | Retrieval Lift | `ClosedBookBaseline` vs RAG | Comparative |
    """)


# ---------------------------------------------------------------------------
# Panel 4 — Per-query drill-down table
# ---------------------------------------------------------------------------


def _build_query_rows(
    data: DashboardData, selected: list[str], show_only: str
) -> list[dict]:
    """Build row dicts for the query drill-down table."""
    rows: list[dict] = []
    for name in selected:
        v = data.variants[name]
        for qr in v.query_results:
            if show_only == "Hits only" and not qr.hit:
                continue
            if show_only == "Misses only" and qr.hit:
                continue
            rows.append(
                {
                    "variant": name,
                    "query_id": qr.query_id,
                    "query": qr.query[:80] + ("…" if len(qr.query) > 80 else ""),
                    "hit": "✅" if qr.hit else "❌",
                    "reciprocal_rank": qr.reciprocal_rank,
                    "latency_ms": qr.latency_ms,
                    "expected_url": qr.expected_url,
                    "trace_id": qr.trace_id or "—",
                }
            )
    return rows


def _panel_query_table(data: DashboardData) -> None:
    """Filterable table of per-query results across all variants."""
    st.header("Per-Query Drill-Down")

    selected = st.multiselect(
        "Variants",
        data.variant_names,
        default=data.variant_names,
    )

    col1, col2 = st.columns(2)
    with col1:
        show_only = st.radio(
            "Filter",
            ["All queries", "Hits only", "Misses only"],
            horizontal=True,
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["query_id", "latency_ms", "reciprocal_rank"],
        )

    rows = _build_query_rows(data, selected, show_only)

    if rows:
        reverse = sort_by in ("latency_ms", "reciprocal_rank")
        rows.sort(key=lambda r: r.get(sort_by, ""), reverse=reverse)
        st.dataframe(rows, use_container_width=True, height=CHART_HEIGHT)
        st.caption(f"{len(rows)} queries shown")
    else:
        st.warning("No queries match the current filter.")


# ---------------------------------------------------------------------------
# Panel 5 — Failure analysis
# ---------------------------------------------------------------------------


def _failure_treemap(name: str, v: DashboardData) -> None:
    """Render treemap of failure types for a single variant."""
    variant = v.variants[name]
    types = [fc.failure_type for fc in variant.failure_clusters]
    counts = [fc.count for fc in variant.failure_clusters]

    fig = px.treemap(
        names=types,
        parents=["" for _ in types],
        values=counts,
        title=f"{name} — Failure Taxonomy ({sum(counts)} failures)",
        color=counts,
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=CHART_HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    for fc in variant.failure_clusters:
        with st.expander(f"{fc.failure_type} ({fc.count}×)"):
            if fc.common_patterns:
                st.markdown("**Common patterns:**")
                for p in fc.common_patterns:
                    st.markdown(f"- {p}")
            else:
                st.caption("No patterns extracted")


def _failure_comparison_table(data: DashboardData) -> None:
    """Cross-variant failure type comparison table."""
    st.subheader("Cross-Variant Failure Comparison")
    all_types: set[str] = set()
    for v in data.variants.values():
        for fc in v.failure_clusters:
            all_types.add(fc.failure_type)

    if not all_types:
        return

    comparison_rows: list[dict] = []
    for ft in sorted(all_types):
        row: dict[str, str | int] = {"failure_type": ft}
        for name in data.variant_names:
            v = data.variants[name]
            count = next(
                (fc.count for fc in v.failure_clusters if fc.failure_type == ft),
                0,
            )
            row[name] = count
        comparison_rows.append(row)
    st.dataframe(comparison_rows, use_container_width=True)


def _panel_failures(data: DashboardData) -> None:
    """Failure cluster breakdown by variant."""
    st.header("Failure Analysis")

    has_clusters = any(v.failure_clusters for v in data.variants.values())
    if not has_clusters:
        st.info(
            "No failure clusters found. "
            "Run the experiment with the full golden dataset to see failure analysis."
        )
        return

    tab_names = [
        name for name in data.variant_names if data.variants[name].failure_clusters
    ]
    if not tab_names:
        return

    tabs = st.tabs(tab_names)
    for tab, name in zip(tabs, tab_names, strict=False):
        with tab:
            _failure_treemap(name, data)

    _failure_comparison_table(data)


# ---------------------------------------------------------------------------
# Panel 6 — Variant config comparison
# ---------------------------------------------------------------------------


def _panel_config(data: DashboardData) -> None:
    """Side-by-side config comparison table."""
    st.header("Configuration Comparison")

    configs: list[dict] = []
    for name in data.variant_names:
        cs = data.variants[name].config_snapshot
        if cs:
            configs.append({"variant": name, **cs})

    if configs:
        st.dataframe(configs, use_container_width=True)
    else:
        st.info("No configuration snapshots available.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Dashboard entry point — loads data and renders all panels."""
    data = _sidebar_data_source()

    if data.is_empty:
        st.markdown("---")
        _empty_state()
        return

    # Source badge
    st.sidebar.divider()
    st.sidebar.success(f"Source: **{data.source}**")
    st.sidebar.caption(f"Exported: {data.exported_at or '—'}")
    st.sidebar.caption(f"Variants: {', '.join(data.variant_names)}")

    # Render all panels
    _panel_overview(data)
    st.divider()
    _panel_retrieval(data)
    st.divider()
    _panel_generation(data)
    st.divider()
    _panel_query_table(data)
    st.divider()
    _panel_failures(data)
    st.divider()
    _panel_config(data)

    # Footer
    st.divider()
    st.caption(
        "Data from `uv run python -m eval.experiment run --export results.json` "
        "or Langfuse API. "
        "See `src/eval/` for the full evaluation harness."
    )


def _empty_state() -> None:
    """Show instructions when no data is loaded."""
    st.title("📊 RAG Eval Dashboard")
    st.markdown("""
    ### Getting Started

    This dashboard visualises experiment results from the **eval harness**
    that compares three RAG retrieval configurations:

    | Variant | Strategy | Reranker | k |
    |---------|----------|----------|---|
    | **librarian** | Hybrid BM25+dense | CrossEncoder | 10 |
    | **raptor** | Pure knn | Passthrough | 5 |
    | **bedrock** | Pure knn | Passthrough | 5 |

    #### Option 1: Upload exported JSON
    ```bash
    # Run the eval harness and export results
    uv run python -m eval.experiment run --export results.json
    ```
    Then upload `results.json` via the sidebar.

    #### Option 2: Connect to Langfuse
    Set these environment variables:
    ```bash
    LANGFUSE_ENABLED=true
    LANGFUSE_PUBLIC_KEY=pk-...
    LANGFUSE_SECRET_KEY=sk-...
    LANGFUSE_HOST=https://cloud.langfuse.com
    ```
    Then click **Fetch from Langfuse** in the sidebar.
    """)


if __name__ == "__main__":
    main()
