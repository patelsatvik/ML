"""Reusable data loading, validation, cleaning and statistics functions.

Everything here is plain pandas/numpy (no Streamlit), so it can be unit-tested
and reused outside the dashboard.
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".json")

REQUIRED_COLUMNS = ["Order ID", "Order Date", "Sales", "Quantity", "Profit"]
NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]
MONEY_COLUMNS = ["Sales", "Profit"]
DATE_COLUMNS = ["Order Date", "Ship Date"]
TEXT_COLUMNS = [
    "Order ID", "Ship Mode", "Customer ID", "Segment", "Country", "City",
    "State", "Region", "Product ID", "Category", "Sub-Category", "Product Name",
]

# Allowed values for low-cardinality columns (used to repair inconsistent spelling).
VALID_VALUES = {
    "Segment": ["Consumer", "Corporate", "Home Office"],
    "Region": ["East", "West", "Central", "South"],
    "Ship Mode": ["Same Day", "First Class", "Second Class", "Standard Class"],
    "Category": ["Furniture", "Office Supplies", "Technology"],
}
# Known alternative spellings (normalised key -> canonical value).
ALIASES = {
    "Category": {"officesupply": "Office Supplies"},
}

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "?", "-", "unknown",
               "not available", "not-a-date"}
DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%d %b %Y",
    "%Y/%m/%d", "%m/%d/%Y", "%b %d, %Y", "%d.%m.%Y",
]
PLACEHOLDER_PATTERN = r"(?i)\b(?:test|dummy|mystery)"
MONTH_FIRST_FORMATS = ["%m-%d-%Y", "%m/%d/%Y"]
MAX_SHIPPING_DAYS = 60  # longer gaps are treated as a mis-read date
OUTLIER_IQR_MULTIPLIER = 10  # "extreme" fence: beyond Q3 + 10*IQR (or Q1 - 10*IQR)


class DataLoadError(ValueError):
    """Raised when an uploaded file cannot be read or is unusable."""


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_data(file: BinaryIO | bytes, filename: str) -> pd.DataFrame:
    """Read a CSV, Excel (.xlsx) or JSON file into a DataFrame."""
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{ext or filename}'. Please upload one of: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    raw = bytes(file) if isinstance(file, (bytes, bytearray)) else file.read()
    if not raw:
        raise DataLoadError("The uploaded file is empty.")
    buffer = io.BytesIO(raw)

    try:
        if ext == ".csv":
            df = pd.read_csv(buffer)
        elif ext == ".xlsx":
            df = pd.read_excel(buffer, engine="openpyxl")
        else:
            df = _json_to_frame(json.loads(raw.decode("utf-8-sig")))
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no data.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Could not parse the JSON file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface any reader failure to the UI
        raise DataLoadError(f"Could not read the file: {exc}") from exc

    if df.empty:
        raise DataLoadError("The file was read but contains no rows.")
    return df


def _json_to_frame(payload: Any) -> pd.DataFrame:
    """Accept a list of records, {'data': [...]}, or a dict of columns."""
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.json_normalize(value)
        try:
            return pd.DataFrame(payload)
        except ValueError as exc:
            raise DataLoadError("JSON structure not recognised as tabular data.") from exc
    raise DataLoadError("JSON must contain a list of records or an object of columns.")


# --------------------------------------------------------------------------- #
# 2. Validation
# --------------------------------------------------------------------------- #
def count_unparsable(series: pd.Series, kind: str) -> int:
    """Count non-null entries that cannot be read as `kind` ('numeric' or 'date')."""
    non_null = series.dropna()
    if non_null.empty:
        return 0
    if kind == "numeric":
        parsed = pd.to_numeric(non_null, errors="coerce")
        return int(parsed.isna().sum() + np.isinf(parsed.dropna().astype(float)).sum())
    return int(parse_dates(non_null).isna().sum())


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return dimensions, schema, missing values, duplicates and type problems."""
    df = standardize_columns(df)
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    schema = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null": df.notna().sum().values,
        "Missing": missing.values,
        "Missing %": (missing.values / max(n_rows, 1) * 100).round(2),
        "Unique": df.nunique(dropna=True).values,
    })
    problems = []
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            bad = count_unparsable(df[col], "numeric")
            if bad:
                problems.append({"Column": col, "Issue": "Non-numeric / infinite values", "Count": bad})
    for col in DATE_COLUMNS:
        if col in df.columns:
            bad = count_unparsable(df[col], "date")
            if bad:
                problems.append({"Column": col, "Issue": "Unparsable dates", "Count": bad})
    return {
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "schema": schema,
        "total_missing": int(missing.sum()),
        "duplicates": int(df.duplicated(keep="first").sum()),
        "duplicate_rows": df[df.duplicated(keep=False)],
        "missing_required": [c for c in REQUIRED_COLUMNS if c not in df.columns],
        "problems": pd.DataFrame(problems, columns=["Column", "Issue", "Count"]),
    }


# --------------------------------------------------------------------------- #
# 3. Cleaning helpers
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and map case/underscore variants to canonical names."""
    known = REQUIRED_COLUMNS + NUMERIC_COLUMNS + DATE_COLUMNS + TEXT_COLUMNS
    canonical = {re.sub(r"[\W_]+", "", c).lower(): c for c in known}
    renamed = {}
    for col in df.columns:
        key = re.sub(r"[\W_]+", "", str(col)).lower()
        renamed[col] = canonical.get(key, str(col).strip())
    return df.rename(columns=renamed)


def clean_text(series: pd.Series) -> pd.Series:
    """Trim, collapse spaces and turn placeholder tokens into missing values."""
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.str.lower().isin(NULL_TOKENS))


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_categories(series: pd.Series, valid: list[str],
                         aliases: dict[str, str] | None = None) -> pd.Series:
    """Map case/spacing variants to canonical labels; unknown labels become missing."""
    lookup = {_key(v): v for v in valid}
    lookup.update(aliases or {})
    s = clean_text(series)
    mapped = [lookup.get(_key(v)) if pd.notna(v) else None for v in s]
    return pd.Series(mapped, index=series.index, dtype="string")


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats (ISO, dd/mm/yyyy, dd-Mon-yyyy ...) into datetime64."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("datetime64[ns]")
    s = clean_text(series)
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        pending = result.isna() & s.notna()
        if not pending.any():
            break
        parsed = pd.to_datetime(s.where(pending), format=fmt, errors="coerce")
        result = result.fillna(parsed.astype("datetime64[ns]"))
    return result


def fix_ambiguous_ship_dates(order: pd.Series, ship_raw: pd.Series, ship: pd.Series) -> pd.Series:
    """Re-read day/month-ambiguous ship dates as month-first when that gives a sane gap.

    '03-12-2025' is 3 Dec (dd-mm) by default, but if the order was on 8 Mar that is
    270 days later; read as 12 Mar (mm-dd) it is a normal 4-day shipment.
    """
    gap = (ship - order).dt.days
    suspicious = (gap > MAX_SHIPPING_DAYS) | (gap < 0)
    if not suspicious.any():
        return ship
    text = clean_text(ship_raw)
    for fmt in MONTH_FIRST_FORMATS:
        alt = pd.to_datetime(text.where(suspicious), format=fmt, errors="coerce").astype("datetime64[ns]")
        alt_gap = (alt - order).dt.days
        ok = suspicious & alt_gap.between(0, MAX_SHIPPING_DAYS)
        ship = ship.mask(ok, alt)
        suspicious = suspicious & ~ok
    return ship


def _words_to_numbers(s: pd.Series) -> pd.Series:
    mapped = [str(WORD_NUMBERS[v]) if pd.notna(v) and v in WORD_NUMBERS else v for v in s]
    return pd.Series(mapped, index=s.index, dtype="string")


def to_numeric_clean(series: pd.Series, allow_percent: bool = False) -> pd.Series:
    """Convert to float: strips currency/commas, understands 'five', drops +/-inf."""
    if pd.api.types.is_numeric_dtype(series):
        out = series.astype(float)
    else:
        s = clean_text(series).str.lower().str.replace(r"[$€£₹,]", "", regex=True)
        if allow_percent:  # '10%' / 'ten percent' -> 0.10
            pct = s.str.extract(r"^(\w+(?:\.\d+)?)\s*(?:%|percent)$")[0]
            pct_val = pd.to_numeric(_words_to_numbers(pct), errors="coerce") / 100
            s = s.where(pct.isna(), pct_val.astype("string"))
        out = pd.to_numeric(_words_to_numbers(s), errors="coerce")
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


def iqr_fences(series: pd.Series, k: float = OUTLIER_IQR_MULTIPLIER) -> tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _fill_from_mode(df: pd.DataFrame, target: str, source: str) -> pd.Series:
    """Fill `target` using the most common target value seen for each `source`."""
    pairs = df.dropna(subset=[target, source])
    if pairs.empty:
        return df[target]
    mode_map = pairs.groupby(source)[target].agg(lambda x: x.mode().iloc[0])
    return df[target].fillna(df[source].map(mode_map))


# --------------------------------------------------------------------------- #
# 4. Cleaning pipeline
# --------------------------------------------------------------------------- #
def clean_data(
    df: pd.DataFrame,
    money_strategy: str = "drop",
    other_numeric_strategy: str = "median",
    remove_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Parameters
    ----------
    money_strategy : 'drop' | 'median' - treatment of missing Sales / Profit.
    other_numeric_strategy : 'drop' | 'median' - treatment of missing Quantity / Discount.
    remove_outliers : treat extreme IQR outliers in Sales/Quantity/Profit as invalid.

    Returns
    -------
    (clean_df, log_df) - log_df lists each step and the rows/cells it touched.
    """
    log: list[dict[str, Any]] = []

    def record(step: str, detail: str, count: int) -> None:
        log.append({"Step": step, "What was done": detail, "Rows / cells affected": int(count)})

    df = standardize_columns(df.copy())
    start_rows = len(df)

    # a) empty rows and placeholder / test records
    empty = df.isna().all(axis=1)
    df = df[~empty].copy()
    record("Empty rows", "Removed rows with no data at all", empty.sum())

    for col in [c for c in TEXT_COLUMNS if c in df.columns]:
        df[col] = clean_text(df[col])
    placeholder = pd.Series(False, index=df.index)
    for col in [c for c in ("City", "State", "Product Name") if c in df.columns]:
        placeholder |= df[col].fillna("").str.contains(PLACEHOLDER_PATTERN, regex=True)
    df = df[~placeholder].copy()
    record("Test / placeholder records",
           "Removed rows such as 'Testville' / 'Test State' / 'Mystery Product'", placeholder.sum())

    # b) inconsistent categorical spelling
    fixed = 0
    for col, valid in VALID_VALUES.items():
        if col in df.columns:
            before = df[col].copy()
            df[col] = normalize_categories(df[col], valid, ALIASES.get(col))
            fixed += int((before.notna() & (before != df[col])).sum())
    record("Inconsistent text",
           "Standardised case/spelling (e.g. 'consumer'->'Consumer', 'SameDay'->'Same Day', "
           "'Office supply'->'Office Supplies'); unknown labels set to missing", fixed)

    # c) dates
    non_iso = 0
    for col in [c for c in DATE_COLUMNS if c in df.columns]:
        raw = df[col]
        if not pd.api.types.is_datetime64_any_dtype(raw):
            text = clean_text(raw).dropna()
            non_iso += int((~text.str.match(r"^\d{4}-\d{2}-\d{2}")).sum())
        df[col] = parse_dates(raw)
        if col == "Ship Date" and "Order Date" in df.columns:
            order_dates = parse_dates(df["Order Date"])
            df[col] = fix_ambiguous_ship_dates(order_dates, raw, df[col])
    record("Date formats", "Parsed mixed formats (dd/mm/yyyy, dd-Mon-yyyy, ISO ...) into one datetime type", non_iso)
    if {"Order Date", "Ship Date"} <= set(df.columns):
        bad_ship = df["Ship Date"] < df["Order Date"]
        df.loc[bad_ship, "Ship Date"] = pd.NaT
        record("Ship date before order date", "Set impossible ship dates to missing", bad_ship.sum())

    # d) numerics
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        before_missing = int(df[col].isna().sum())
        df[col] = to_numeric_clean(df[col], allow_percent=(col == "Discount"))
        record(f"Convert {col}", "Converted text to numbers (e.g. 'five'->5, 'ten percent'->0.10); count = "
               "unreadable entries set to missing",
               max(int(df[col].isna().sum()) - before_missing, 0))

    invalid_rules = {}
    if "Sales" in df:
        invalid_rules["Sales"] = df["Sales"] < 0
    if "Quantity" in df:
        invalid_rules["Quantity"] = (df["Quantity"] <= 0) | (df["Quantity"] % 1 != 0)
    if "Discount" in df:
        invalid_rules["Discount"] = (df["Discount"] < 0) | (df["Discount"] > 1)
    for col, mask in invalid_rules.items():
        df.loc[mask, col] = np.nan
        record(f"Invalid {col}", "Set impossible values to missing (negative sales, non-positive or "
               "fractional quantity, discount outside 0-1)", mask.sum())

    if remove_outliers:
        for col in [c for c in ("Sales", "Quantity", "Profit") if c in df.columns]:
            lo, hi = iqr_fences(df[col].dropna())
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, col] = np.nan
            record(f"Extreme outliers in {col}",
                   f"Values outside [{lo:,.0f}, {hi:,.0f}] ({OUTLIER_IQR_MULTIPLIER}xIQR fence) set to missing", mask.sum())

    # e) categoricals: infer from related columns, then label the rest
    if {"Category", "Sub-Category"} <= set(df.columns):
        n0 = int(df["Category"].isna().sum())
        df["Category"] = _fill_from_mode(df, "Category", "Sub-Category")
        record("Fill Category", "Inferred missing Category from Sub-Category", n0 - int(df["Category"].isna().sum()))
    if {"Region", "State"} <= set(df.columns):
        n0 = int(df["Region"].isna().sum())
        df["Region"] = _fill_from_mode(df, "Region", "State")
        record("Fill Region", "Inferred missing Region from State", n0 - int(df["Region"].isna().sum()))
    label_cols = [c for c in ("Ship Mode", "Segment", "Category", "Region", "Customer ID",
                              "Sub-Category", "City", "State", "Product Name") if c in df.columns]
    filled = 0
    for col in label_cols:
        filled += int(df[col].isna().sum())
        df[col] = df[col].fillna("Unknown")
    record("Label unknowns", "Remaining missing categorical values set to 'Unknown'", filled)

    # f) duplicates (after standardising so case/spacing variants are caught)
    dups = df.duplicated(keep="first")
    df = df[~dups].copy()
    record("Duplicate records", "Removed exact duplicate rows (first copy kept)", dups.sum())

    # g) missing keys, then missing numerics
    key_cols = [c for c in ("Order ID", "Order Date") if c in df.columns]
    missing_key = df[key_cols].isna().any(axis=1)
    df = df[~missing_key].copy()
    record("Missing key fields", "Dropped rows with no valid Order ID or Order Date", missing_key.sum())

    money = [c for c in MONEY_COLUMNS if c in df.columns]
    others = [c for c in ("Quantity", "Discount") if c in df.columns]
    for cols, strategy, label in ((money, money_strategy, "Sales/Profit"),
                                  (others, other_numeric_strategy, "Quantity/Discount")):
        if not cols:
            continue
        if strategy == "drop":
            mask = df[cols].isna().any(axis=1)
            df = df[~mask].copy()
            record(f"Missing {label}", "Dropped rows with missing values", mask.sum())
        else:
            n = 0
            for col in cols:
                n += int(df[col].isna().sum())
                if "Sub-Category" in df:
                    df[col] = df[col].fillna(df.groupby("Sub-Category")[col].transform("median"))
                df[col] = df[col].fillna(df[col].median())
                if col == "Quantity":
                    df[col] = df[col].round()
            record(f"Missing {label}", "Filled with median (by Sub-Category, else overall)", n)

    # h) derived columns
    df = df.reset_index(drop=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    if "Ship Date" in df.columns:
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit Margin %"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, np.nan)

    record("Result", f"{start_rows} raw rows -> {len(df)} clean rows", start_rows - len(df))
    return df, pd.DataFrame(log)


# --------------------------------------------------------------------------- #
# 5. Filtering, KPIs and statistics
# --------------------------------------------------------------------------- #
def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | None = None,
    selections: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Filter by an inclusive Order Date range and by column -> allowed-values selections."""
    out = df
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["Order Date"] >= start) & (out["Order Date"] <= end)]
    for col, chosen in (selections or {}).items():
        if col in out.columns and chosen is not None:
            out = out[out[col].isin(chosen)]
    return out


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Total Sales, Total Profit, Total Orders (unique Order ID), Quantity, Profit Margin %."""
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    return {
        "Total Sales": total_sales,
        "Total Profit": total_profit,
        "Total Orders": int(df["Order ID"].nunique()),
        "Total Quantity Sold": int(df["Quantity"].sum()),
        "Profit Margin (%)": (total_profit / total_sales * 100) if total_sales else 0.0,
    }


def numeric_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Count, mean, median, std, min/max, quartiles, IQR, skew, kurtosis and CV."""
    columns = columns or [c for c in ("Sales", "Quantity", "Discount", "Profit",
                                      "Profit Margin %", "Shipping Days") if c in df.columns]
    num = df[columns].apply(pd.to_numeric, errors="coerce")
    stats = num.describe().T
    stats["median"] = num.median()
    stats["IQR"] = stats["75%"] - stats["25%"]
    stats["skew"] = num.skew()
    stats["kurtosis"] = num.kurt()
    stats["CV %"] = (stats["std"] / stats["mean"].abs() * 100).replace([np.inf, -np.inf], np.nan)
    return stats[["count", "mean", "median", "std", "min", "25%", "75%",
                  "max", "IQR", "skew", "kurtosis", "CV %"]].round(2)


def categorical_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Unique count, most frequent value and its share for each categorical column."""
    columns = columns or [c for c in ("Segment", "Region", "Category", "Sub-Category", "Ship Mode")
                          if c in df.columns]
    rows = []
    for col in columns:
        counts = df[col].value_counts()
        rows.append({
            "Column": col,
            "Unique": int(df[col].nunique()),
            "Most common": counts.index[0] if len(counts) else None,
            "Share %": round(counts.iloc[0] / len(df) * 100, 1) if len(counts) else 0.0,
        })
    return pd.DataFrame(rows)


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Sales, Profit, Quantity, Orders and Profit Margin % per group."""
    grouped = df.groupby(by, observed=True).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"), Orders=("Order ID", "nunique"),
    ).reset_index()
    grouped["Profit Margin %"] = np.where(
        grouped["Sales"] != 0, grouped["Profit"] / grouped["Sales"] * 100, 0.0).round(2)
    return grouped.sort_values("Sales", ascending=False)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Sales", "Quantity", "Discount", "Profit") if c in df.columns]
    return df[cols].corr().round(2)
