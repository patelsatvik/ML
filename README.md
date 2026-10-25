# 📊 Sales Analytics Dashboard

Interactive Streamlit app for uploading, validating, cleaning and exploring sales data.

**Live app:** _paste your Streamlit Community Cloud URL here_
**Source:** _paste your GitHub repo URL here_

## Features
| Requirement | Where |
|---|---|
| Upload CSV / Excel / JSON | sidebar uploader (`data_processing.load_data`) |
| Validation: dimensions, column names, dtypes, missing values, duplicates | tab **1 Upload & Validation** |
| Cleaning: missing values, duplicates, inconsistent text, mixed date formats, invalid numbers, outliers | tab **2 Cleaning** (`clean_data`) with a step-by-step log |
| Statistical summaries (reusable functions) | tab **4 Statistics** (`numeric_summary`, `categorical_summary`, `group_summary`, `correlation_matrix`) |
| 5 filters (date range, Region, Category, Segment, Ship Mode) | sidebar |
| 5 live KPIs: Total Sales, Total Profit, Total Orders (unique Order ID), Total Quantity, Profit Margin % | tab **3 Dashboard** |
| 7 interactive Plotly charts | tabs 3 and 4 (`visuals.py`) |
| Error handling | friendly messages for bad file type, empty/corrupt file, missing columns, empty filter result; one failing chart never breaks the page |

## Project structure
```
app.py               # Streamlit UI (layout, filters, tabs)
data_processing.py   # load / validate / clean / KPI / statistics functions (no Streamlit)
visuals.py           # Plotly chart builders
data/                # sample dataset (used when nothing is uploaded)
tests/               # pytest unit + end-to-end UI tests
requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Test
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Cleaning decisions (documented in the app too)
* Missing **Sales / Profit** rows are dropped by default (money is not guessed); Quantity / Discount use the median. Both are switchable.
* Extreme outliers = beyond Q3 + 10×IQR (catches `9,999,999.99` sales and `10000` quantity without removing genuine large orders).
* Ambiguous dates such as `03-12-2025` are read day-first, then re-read month-first if the day-first reading would ship the order > 60 days later.
