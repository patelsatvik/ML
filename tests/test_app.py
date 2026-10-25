"""End-to-end smoke tests of the Streamlit UI using Streamlit's AppTest runner."""
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import visuals as viz  # noqa: E402

APP = str(ROOT / "app.py")


def kpi_values(at):
    return {m.label: m.value for m in at.metric if m.label in
            ("Total Sales", "Total Profit", "Total Orders", "Total Quantity Sold", "Profit Margin")}


def test_app_runs_and_shows_five_kpis():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(kpi_values(at)) == 5


def test_filter_changes_kpis():
    at = AppTest.from_file(APP, default_timeout=60).run()
    before = kpi_values(at)
    region = next(m for m in at.multiselect if m.label == "Region")
    region.set_value(["East"]).run()
    assert not at.exception
    after = kpi_values(at)
    assert after["Total Sales"] != before["Total Sales"]
    assert int(after["Total Orders"].replace(",", "")) < int(before["Total Orders"].replace(",", ""))


def test_empty_selection_shows_warning_not_crash():
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(m for m in at.multiselect if m.label == "Category").set_value([]).run()
    assert not at.exception
    assert any("No records match" in w.value for w in at.warning)


def test_charts_handle_empty_dataframe():
    empty = pd.DataFrame(columns=["Month", "Sales", "Profit", "Category", "Region", "Sub-Category",
                                  "Discount", "Order ID", "Quantity", "Product Name"])
    for fn in (viz.sales_profit_trend, viz.category_performance, viz.region_share,
               viz.subcategory_profit, viz.discount_vs_profit, viz.margin_heatmap):
        assert fn(empty) is not None
