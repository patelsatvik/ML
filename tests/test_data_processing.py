import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_processing as dp  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "experiment10_raw_dataset.csv"


@pytest.fixture(scope="module")
def raw():
    return dp.load_data(SAMPLE.read_bytes(), SAMPLE.name)


@pytest.fixture(scope="module")
def cleaned(raw):
    return dp.clean_data(raw)


# ---------------------------- loading ---------------------------------- #
def test_load_csv(raw):
    assert raw.shape == (186, 18)


def test_load_json_records_and_wrapped():
    records = [{"Order ID": "A", "Sales": 1}, {"Order ID": "B", "Sales": 2}]
    assert len(dp.load_data(json.dumps(records).encode(), "x.json")) == 2
    assert len(dp.load_data(json.dumps({"data": records}).encode(), "x.json")) == 2


def test_load_excel_roundtrip(raw):
    buf = io.BytesIO()
    raw.head(20).to_excel(buf, index=False)
    assert dp.load_data(buf.getvalue(), "x.xlsx").shape[0] == 20


@pytest.mark.parametrize("name,content", [("a.txt", b"x"), ("a.csv", b""), ("a.json", b"{bad")])
def test_load_errors(name, content):
    with pytest.raises(dp.DataLoadError):
        dp.load_data(content, name)


# ---------------------------- validation ------------------------------- #
def test_validation_report(raw):
    rep = dp.validate_data(raw)
    assert (rep["rows"], rep["columns"]) == (186, 18)
    assert rep["duplicates"] == 4
    assert rep["missing_required"] == []
    assert set(rep["problems"]["Column"]) >= {"Sales", "Quantity", "Discount", "Profit", "Order Date"}
    assert {"Column", "Data Type", "Missing"} <= set(rep["schema"].columns)


def test_validation_flags_missing_required_columns():
    rep = dp.validate_data(pd.DataFrame({"a": [1]}))
    assert "Sales" in rep["missing_required"]


# ---------------------------- helpers ---------------------------------- #
def test_parse_dates_mixed_formats():
    s = pd.Series(["2025-01-02", "26/12/2024", "21-Mar-2025", "not-a-date", None])
    out = dp.parse_dates(s)
    assert out.iloc[0] == pd.Timestamp("2025-01-02")
    assert out.iloc[1] == pd.Timestamp("2024-12-26")
    assert out.iloc[2] == pd.Timestamp("2025-03-21")
    assert out.iloc[3:].isna().all()


def test_to_numeric_clean():
    s = pd.Series(["five", "1,200.5", "$30", "?", "inf", "7"])
    out = dp.to_numeric_clean(s)
    assert out.iloc[0] == 5 and out.iloc[1] == 1200.5 and out.iloc[2] == 30 and out.iloc[5] == 7
    assert out.iloc[3:5].isna().all()


def test_discount_percent_words():
    out = dp.to_numeric_clean(pd.Series(["ten percent", "15%", "0.2"]), allow_percent=True)
    assert out.tolist() == pytest.approx([0.10, 0.15, 0.2])


def test_normalize_categories():
    out = dp.normalize_categories(pd.Series([" technology ", "Office supply", "gadgets", None]),
                                  dp.VALID_VALUES["Category"], dp.ALIASES["Category"])
    assert out.iloc[0] == "Technology" and out.iloc[1] == "Office Supplies"
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])


def test_ambiguous_ship_date_read_month_first():
    order = pd.Series([pd.Timestamp("2025-03-08")])
    ship = dp.parse_dates(pd.Series(["03-12-2025"]))  # read day-first = 3 Dec (270 days)
    fixed = dp.fix_ambiguous_ship_dates(order, pd.Series(["03-12-2025"]), ship)
    assert fixed.iloc[0] == pd.Timestamp("2025-03-12")


# ---------------------------- cleaning --------------------------------- #
def test_clean_removes_duplicates_and_junk(cleaned):
    df, _ = cleaned
    assert not df.duplicated().any()
    assert not df["City"].str.contains("Test").any()


def test_clean_standardises_categories(cleaned):
    df, _ = cleaned
    assert set(df["Segment"]) <= {"Consumer", "Corporate", "Home Office", "Unknown"}
    assert set(df["Region"]) == {"East", "West", "Central", "South"}
    assert set(df["Category"]) == {"Furniture", "Office Supplies", "Technology"}
    assert set(df["Ship Mode"]) <= {"Same Day", "First Class", "Second Class", "Standard Class", "Unknown"}


def test_clean_numeric_sanity(cleaned):
    df, _ = cleaned
    assert df["Sales"].between(0, 20000).all()
    assert df["Quantity"].between(1, 100).all()
    assert df["Discount"].between(0, 1).all()
    assert np.isfinite(df["Profit"]).all()
    assert df[["Sales", "Quantity", "Profit", "Discount", "Order ID", "Order Date"]].notna().all().all()
    assert df["Shipping Days"].dropna().between(0, 60).all()
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])


def test_clean_log_has_result_row(cleaned):
    _, log = cleaned
    assert log.iloc[-1]["Step"] == "Result"


def test_median_strategy_keeps_more_rows(raw):
    drop_df, _ = dp.clean_data(raw, money_strategy="drop")
    fill_df, _ = dp.clean_data(raw, money_strategy="median")
    assert len(fill_df) > len(drop_df)


# ---------------------------- KPIs & stats ----------------------------- #
def test_kpis_small_frame():
    df = pd.DataFrame({"Order ID": ["A", "A", "B"], "Sales": [100.0, 50.0, 50.0],
                       "Profit": [10.0, 5.0, -5.0], "Quantity": [1, 2, 3]})
    k = dp.compute_kpis(df)
    assert k["Total Sales"] == 200 and k["Total Profit"] == 10
    assert k["Total Orders"] == 2          # unique Order IDs, not rows
    assert k["Total Quantity Sold"] == 6
    assert k["Profit Margin (%)"] == pytest.approx(5.0)


def test_kpis_empty_frame_no_division_error():
    empty = pd.DataFrame({"Order ID": [], "Sales": [], "Profit": [], "Quantity": []})
    assert dp.compute_kpis(empty)["Profit Margin (%)"] == 0.0


def test_filters(cleaned):
    df, _ = cleaned
    out = dp.apply_filters(df, (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
                           {"Region": ["East"], "Category": ["Technology", "Furniture"]})
    assert (out["Region"] == "East").all() and (out["Order Date"].dt.year == 2025).all()
    assert dp.apply_filters(df, None, {"Region": []}).empty


def test_summaries(cleaned):
    df, _ = cleaned
    assert {"mean", "median", "std", "IQR", "skew"} <= set(dp.numeric_summary(df).columns)
    g = dp.group_summary(df, "Category")
    assert g["Sales"].sum() == pytest.approx(df["Sales"].sum())
    assert dp.correlation_matrix(df).shape == (4, 4)
