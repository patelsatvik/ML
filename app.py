"""Interactive Sales Analytics Dashboard (Streamlit).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import data_processing as dp
import visuals as viz

SAMPLE_PATH = Path(__file__).parent / "data" / "experiment10_raw_dataset.csv"

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Cached wrappers (keep the UI fast when filters change)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading file...")
def cached_load(raw: bytes, filename: str) -> pd.DataFrame:
    return dp.load_data(raw, filename)


@st.cache_data(show_spinner="Cleaning data...")
def cached_clean(raw_df: pd.DataFrame, money: str, other: str, outliers: bool):
    return dp.clean_data(raw_df, money_strategy=money, other_numeric_strategy=other,
                         remove_outliers=outliers)


@st.cache_data
def cached_validate(raw_df: pd.DataFrame) -> dict:
    return dp.validate_data(raw_df)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def show_chart(builder, *args, **kwargs) -> None:
    """Render a chart; a failure in one chart never breaks the whole page."""
    try:
        st.plotly_chart(builder(*args, **kwargs), width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"This chart could not be drawn: {exc}")


def money(value: float) -> str:
    return f"${value:,.0f}" if abs(value) >= 1000 else f"${value:,.2f}"


def get_source() -> tuple[bytes, str, bool]:
    """Return (raw bytes, filename, is_sample) from the uploader or the bundled sample."""
    uploaded = st.sidebar.file_uploader(
        "Upload a dataset", type=["csv", "xlsx", "json"],
        help="CSV, Excel (.xlsx) or JSON with columns like Order ID, Order Date, Sales, Quantity, Profit.")
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name, False
    if SAMPLE_PATH.exists():
        st.sidebar.info("No file uploaded - using the bundled sample dataset.")
        return SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name, True
    st.info("Upload a CSV, Excel or JSON file in the sidebar to begin.")
    st.stop()


def sidebar_filters(df: pd.DataFrame, signature: str):
    """Render the interactive filters and return (date_range, selections)."""
    st.sidebar.header("🔎 Filters")
    min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
    keys = {"date": f"date_{signature}", "Region": f"region_{signature}",
            "Category": f"category_{signature}", "Segment": f"segment_{signature}",
            "Ship Mode": f"ship_{signature}"}

    def reset() -> None:
        st.session_state[keys["date"]] = (min_d, max_d)
        for col in ("Region", "Category", "Segment", "Ship Mode"):
            st.session_state[keys[col]] = sorted(df[col].unique())

    date_value = st.sidebar.date_input("Order date range", value=(min_d, max_d),
                                       min_value=min_d, max_value=max_d, key=keys["date"])
    selections = {}
    for col in ("Region", "Category", "Segment", "Ship Mode"):
        options = sorted(df[col].unique())
        selections[col] = st.sidebar.multiselect(col, options, default=options, key=keys[col])
    st.sidebar.button("Reset filters", on_click=reset, width="stretch")

    if not isinstance(date_value, (tuple, list)) or len(date_value) != 2:
        st.sidebar.warning("Pick both a start and an end date.")
        date_value = (min_d, max_d)
    return date_value, selections


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def render_validation(raw_df: pd.DataFrame, report: dict, filename: str) -> None:
    st.subheader(f"Uploaded file: {filename}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{report['rows']:,}")
    c2.metric("Columns", report["columns"])
    c3.metric("Missing cells", f"{report['total_missing']:,}")
    c4.metric("Duplicate records", report["duplicates"])

    if report["missing_required"]:
        st.error("Required columns missing: " + ", ".join(report["missing_required"]))
    else:
        st.success("All required columns are present.")

    st.markdown("**Column names, data types and missing values**")
    st.dataframe(report["schema"], hide_index=True, width="stretch")

    if not report["problems"].empty:
        st.markdown("**Type problems found** (fixed in the cleaning step)")
        st.dataframe(report["problems"], hide_index=True, width="stretch")

    with st.expander(f"Duplicate rows ({report['duplicates']} extra copies)"):
        if report["duplicate_rows"].empty:
            st.write("No duplicate records.")
        else:
            st.dataframe(report["duplicate_rows"], width="stretch")
    with st.expander("Preview of raw data"):
        st.dataframe(raw_df.head(50), width="stretch")


def render_cleaning(log: pd.DataFrame, raw_rows: int, clean_df: pd.DataFrame) -> None:
    st.subheader("Preprocessing log")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{raw_rows:,}")
    c2.metric("Clean rows", f"{len(clean_df):,}")
    c3.metric("Rows removed", f"{raw_rows - len(clean_df):,}")
    st.dataframe(log[log["Rows / cells affected"] > 0], hide_index=True, width="stretch")
    st.caption("Steps that changed nothing are hidden. Cleaning options are in the sidebar.")
    st.download_button("⬇️ Download cleaned dataset (CSV)", clean_df.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_sales_data.csv", mime="text/csv")


def render_dashboard(df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    kpis = dp.compute_kpis(df)
    cols = st.columns(5)
    cols[0].metric("Total Sales", money(kpis["Total Sales"]))
    cols[1].metric("Total Profit", money(kpis["Total Profit"]))
    cols[2].metric("Total Orders", f"{kpis['Total Orders']:,}", help="Unique Order IDs")
    cols[3].metric("Total Quantity Sold", f"{kpis['Total Quantity Sold']:,}")
    cols[4].metric("Profit Margin", f"{kpis['Profit Margin (%)']:.2f}%", help="Total Profit / Total Sales")
    st.caption(f"Showing {len(df):,} of {len(full_df):,} cleaned records.")
    st.divider()

    show_chart(viz.sales_profit_trend, df)
    left, right = st.columns([3, 2])
    with left:
        dim = st.selectbox("Compare Sales & Profit by", ["Category", "Segment", "Region", "Ship Mode"])
        show_chart(viz.category_performance, df, dim)
    with right:
        st.write("")
        st.write("")
        show_chart(viz.region_share, df)
    left, right = st.columns(2)
    with left:
        show_chart(viz.subcategory_profit, df)
    with right:
        show_chart(viz.discount_vs_profit, df)
    show_chart(viz.margin_heatmap, df)


def render_statistics(df: pd.DataFrame) -> None:
    st.subheader("Numeric summary")
    st.dataframe(dp.numeric_summary(df), width="stretch")
    st.subheader("Categorical summary")
    st.dataframe(dp.categorical_summary(df), hide_index=True, width="stretch")
    st.subheader("Group summary")
    by = st.selectbox("Group by", ["Category", "Sub-Category", "Region", "State", "Segment", "Ship Mode", "Year"],
                      key="group_by")
    st.dataframe(dp.group_summary(df, by), hide_index=True, width="stretch")
    show_chart(viz.correlation_heatmap, dp.correlation_matrix(df))


def render_data(df: pd.DataFrame) -> None:
    st.subheader("Filtered records")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("⬇️ Download filtered data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_sales_data.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📊 Sales Analytics Dashboard")
    st.caption("Upload → validate → clean → filter → explore.")

    st.sidebar.header("📁 Data")
    raw_bytes, filename, _ = get_source()

    try:
        raw_df = cached_load(raw_bytes, filename)
    except dp.DataLoadError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading the file: {exc}")
        st.stop()

    report = cached_validate(raw_df)
    if report["missing_required"]:
        st.error("This file cannot be analysed - required columns are missing: "
                 + ", ".join(report["missing_required"]))
        st.write("Columns found:", ", ".join(report["column_names"]))
        st.stop()

    with st.sidebar.expander("⚙️ Cleaning options"):
        money_choice = st.radio("Missing Sales / Profit", ["Drop rows", "Fill with median"], index=0,
                                help="Money values are not guessed by default, so KPIs stay trustworthy.")
        other_choice = st.radio("Missing Quantity / Discount", ["Fill with median", "Drop rows"], index=0)
        outliers = st.checkbox("Remove extreme outliers", value=True,
                               help="Values beyond Q3 + 10×IQR (or Q1 − 10×IQR) are treated as entry errors.")

    try:
        clean_df, log = cached_clean(
            raw_df, "drop" if money_choice == "Drop rows" else "median",
            "drop" if other_choice == "Drop rows" else "median", outliers)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cleaning failed: {exc}")
        st.stop()
    if clean_df.empty:
        st.error("No rows remain after cleaning. Try less strict cleaning options.")
        st.stop()

    signature = f"{filename}_{len(raw_bytes)}"
    date_range, selections = sidebar_filters(clean_df, signature)
    filtered = dp.apply_filters(clean_df, date_range, selections)

    tab_val, tab_clean, tab_dash, tab_stats, tab_data = st.tabs(
        ["1️⃣ Upload & Validation", "2️⃣ Cleaning", "3️⃣ Dashboard", "4️⃣ Statistics", "5️⃣ Data"])
    with tab_val:
        render_validation(raw_df, report, filename)
    with tab_clean:
        render_cleaning(log, report["rows"], clean_df)
    if filtered.empty:
        for tab in (tab_dash, tab_stats, tab_data):
            with tab:
                st.warning("No records match the current filters. Widen the selection or press **Reset filters**.")
        return
    with tab_dash:
        render_dashboard(filtered, clean_df)
    with tab_stats:
        render_statistics(filtered)
    with tab_data:
        render_data(filtered)


if __name__ == "__main__":
    main()
