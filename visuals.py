"""Plotly chart builders. Each function takes a DataFrame and returns a Figure."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_processing import group_summary

PALETTE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"


def _empty_figure(message: str = "No data for the selected filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=TEMPLATE, height=360)
    return fig


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin={"l": 10, "r": 10, "t": 50, "b": 10},
                      legend={"orientation": "h", "y": -0.2})
    return fig


def sales_profit_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly Sales (line) and Profit (line) over time."""
    if df.empty:
        return _empty_figure()
    monthly = df.groupby("Month", as_index=False)[["Sales", "Profit"]].sum().sort_values("Month")
    fig = go.Figure()
    fig.add_scatter(x=monthly["Month"], y=monthly["Sales"], name="Sales", mode="lines+markers",
                    line={"color": PALETTE[0]})
    fig.add_scatter(x=monthly["Month"], y=monthly["Profit"], name="Profit", mode="lines+markers",
                    line={"color": PALETTE[1]})
    fig.update_layout(title="Monthly Sales & Profit Trend", xaxis_title="Month", yaxis_title="Amount ($)",
                      hovermode="x unified")
    return _style(fig)


def category_performance(df: pd.DataFrame, by: str = "Category") -> go.Figure:
    """Grouped bars of Sales and Profit for the chosen dimension."""
    if df.empty:
        return _empty_figure()
    summary = group_summary(df, by)
    fig = px.bar(summary, x=by, y=["Sales", "Profit"], barmode="group",
                 color_discrete_sequence=PALETTE, title=f"Sales & Profit by {by}",
                 labels={"value": "Amount ($)", "variable": ""},
                 hover_data={"Orders": True, "Profit Margin %": True})
    return _style(fig)


def region_share(df: pd.DataFrame) -> go.Figure:
    """Donut chart of sales share by region."""
    if df.empty:
        return _empty_figure()
    summary = group_summary(df, "Region")
    fig = px.pie(summary, names="Region", values="Sales", hole=0.45,
                 color_discrete_sequence=PALETTE, title="Sales Share by Region")
    fig.update_traces(textinfo="percent+label")
    return _style(fig)


def subcategory_profit(df: pd.DataFrame) -> go.Figure:
    """Horizontal bars of profit by sub-category (negative values highlighted)."""
    if df.empty:
        return _empty_figure()
    summary = group_summary(df, "Sub-Category").sort_values("Profit")
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in summary["Profit"]]
    fig = go.Figure(go.Bar(x=summary["Profit"], y=summary["Sub-Category"], orientation="h",
                           marker_color=colors,
                           customdata=summary[["Sales", "Profit Margin %"]],
                           hovertemplate="%{y}<br>Profit: $%{x:,.0f}<br>Sales: $%{customdata[0]:,.0f}"
                                         "<br>Margin: %{customdata[1]}%<extra></extra>"))
    fig.update_layout(title="Profit by Sub-Category (red = loss)", xaxis_title="Profit ($)")
    return _style(fig, height=460)


def discount_vs_profit(df: pd.DataFrame) -> go.Figure:
    """Scatter of discount against profit, sized by sales."""
    if df.empty:
        return _empty_figure()
    fig = px.scatter(df, x="Discount", y="Profit", size="Sales", color="Category",
                     color_discrete_sequence=PALETTE, opacity=0.7, size_max=28,
                     hover_data=["Order ID", "Product Name", "Sales"],
                     title="Does Discounting Hurt Profit?")
    fig.update_xaxes(tickformat=".0%")
    return _style(fig)


def margin_heatmap(df: pd.DataFrame) -> go.Figure:
    """Region x Category heatmap of profit margin (%)."""
    if df.empty:
        return _empty_figure()
    pivot = df.groupby(["Region", "Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    pivot["Margin %"] = (pivot["Profit"] / pivot["Sales"] * 100).round(1)
    matrix = pivot.pivot(index="Region", columns="Category", values="Margin %")
    fig = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, title="Profit Margin (%) - Region x Category")
    return _style(fig)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    title="Correlation Matrix")
    return _style(fig, height=380)
