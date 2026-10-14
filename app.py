import hashlib
import os
import json
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pydantic import BaseModel, Field
from typing import List, Literal
from google import genai
from google.genai import types
from fpdf import FPDF

# ---------------------------------------------------------
# App Setup & Client Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Auto Data Analyst & BI Studio", layout="wide")
st.title("Auto Data Analyst & BI Studio")

# ---------------------------------------------------------
# Pydantic Schemas for Structured Output
# ---------------------------------------------------------
class KPICard(BaseModel):
    label: str = Field(description="Display label for the KPI card")
    column: str = Field(description="Column name from the dataset")
    aggregation: Literal["sum", "mean", "count", "nunique"] = Field(description="Math aggregation to apply")

class ChartSpec(BaseModel):
    type: Literal["bar", "line", "scatter", "box", "histogram"] = Field(description="Chart type")
    x: str = Field(description="Column name for x-axis")
    y: str = Field(default="", description="Column name for y-axis (leave empty for histogram)")
    aggregation: Literal["sum", "mean", "count"] = Field(default="sum", description="Aggregation method")
    title: str = Field(description="Clean, business-focused chart title")

class DashboardBlueprint(BaseModel):
    kpi_cards: List[KPICard] = Field(description="3 to 4 executive top-line KPI metrics")
    charts: List[ChartSpec] = Field(description="3 to 4 distinct Plotly charts tailored to the data structure")
    what_is_happening: List[str] = Field(description="3 sharp observations highlighting trends, skews, or patterns")
    recommended_actions: List[str] = Field(description="3 high-impact strategic actions for stakeholders")

# ---------------------------------------------------------
# 1. Automated Cleaning Engine
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_dataset(df: pd.DataFrame):
    df = df.copy()
    logs = []

    # Standardize column headers
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    # Drop completely empty columns
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        logs.append(f"Dropped completely empty columns: {all_null_cols}")

    # Remove duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logs.append(f"Removed {initial_len - len(df):,} exact duplicate rows.")

    # Datetime detection & numeric cleanup
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(50)
            cleaned_sample = sample.str.replace(r"[\$,€,£,%]", "", regex=True).str.replace(",", "")
            if pd.to_numeric(cleaned_sample, errors="coerce").notnull().mean() > 0.85:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[\$,€,£,%]", "", regex=True).str.replace(",", ""),
                    errors="coerce"
                )
                logs.append(f"Parsed numeric column '{col}' from string/currency format.")
                continue

            try:
                converted = pd.to_datetime(df[col], errors="raise")
                if converted.notnull().mean() > 0.8:
                    df[col] = converted
                    logs.append(f"Converted '{col}' to datetime.")
            except Exception:
                pass

    # Impute missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
            logs.append(f"Imputed missing numbers in '{c}' with median.")

    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna("Unknown")
            logs.append(f"Imputed missing text in '{c}' with 'Unknown'.")

    return df, logs

# ---------------------------------------------------------
# 2. Extract Data Profile for Gemini (Token-efficient)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_data_profile(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    stats_summary = {}
    for col in num_cols:
        stats_summary[col] = {
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2)
        }

    cat_summary = {}
    for col in cat_cols:
        cat_summary[col] = {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(3).to_dict()
        }

    top_corrs = []
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        corr_values = corr_matrix.to_numpy(copy=True)
        np.fill_diagonal(corr_values, np.nan)
        corr_matrix = pd.DataFrame(corr_values, index=corr_matrix.index, columns=corr_matrix.columns)
        unstacked = corr_matrix.unstack().dropna().drop_duplicates()
        for (col1, col2), val in unstacked.nlargest(3).items():
            top_corrs.append(f"{col1} & {col2} (r={val:.2f})")

    return {
        "row_count": len(df),
        "numeric_columns": stats_summary,
        "categorical_columns": cat_summary,
        "date_columns": date_cols,
        "strongest_correlations": top_corrs
    }

# ---------------------------------------------------------
# 3. Gemini Schema-Enforced Generation
# ---------------------------------------------------------
def generate_dashboard_with_gemini(profile: dict) -> DashboardBlueprint:
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in Streamlit secrets or as an environment variable.")

    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are an experienced business intelligence advisor writing for non-technical leaders.
    Use plain, direct language. Explain what changed and why it matters. Avoid technical jargon,
    unexplained acronyms, and unsupported causal claims. If a cause is uncertain, label it as
    something to investigate rather than a fact. Do not invent values or business context.

    Rules:
    1. Select 3-4 KPI metrics with suitable aggregations ('sum', 'mean', 'count', or 'nunique').
    2. Choose 3-4 diverse charts. Available types: 'bar', 'line', 'scatter', 'box', 'histogram'.
       - Only pick 'line' if a column from date_columns is available.
       - Use 'scatter' to cross-examine two correlated continuous numeric metrics.
       - Use 'bar' for categorical breakdown of sums or averages.
    3. Generate 3 concrete insights under 'what_is_happening'.
    4. Propose 3 high-leverage strategic actions under 'recommended_actions'.

    Dataset Profile:
    {json.dumps(profile, indent=2)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DashboardBlueprint,
            temperature=0.2
        )
    )

    # Convert JSON output back into the validated Pydantic model
    return DashboardBlueprint.model_validate_json(response.text)

# ---------------------------------------------------------
# 4. Dynamic Renderer
# ---------------------------------------------------------
def render_dynamic_dashboard(df: pd.DataFrame, spec: DashboardBlueprint):
    # Render KPI Cards
    st.subheader("📊 Executive Overview")
    if spec.kpi_cards:
        cols = st.columns(len(spec.kpi_cards))
        for i, kpi in enumerate(spec.kpi_cards):
            col_name = kpi.column
            agg = kpi.aggregation
            label = kpi.label

            if col_name in df.columns:
                if agg == "sum":
                    val = df[col_name].sum()
                elif agg == "mean":
                    val = df[col_name].mean()
                elif agg == "count":
                    val = df[col_name].count()
                else:
                    val = df[col_name].nunique()
                cols[i].metric(label, f"{val:,.2f}" if isinstance(val, (int, float)) else str(val))

    st.write("---")

    # Render Visualizations Dynamically
    st.subheader("📈 Tailored Analytics")
    charts = spec.charts
    for i in range(0, len(charts), 2):
        c1, c2 = st.columns(2)
        grid = [c1, c2]

        for j in range(2):
            if i + j < len(charts):
                chart = charts[i + j]
                ctype = chart.type
                x_col = chart.x
                y_col = chart.y
                title = chart.title
                agg = chart.aggregation

                with grid[j]:
                    try:
                        if x_col not in df.columns:
                            st.warning(f"Could not render {title}: column '{x_col}' is unavailable.")
                            continue
                        if ctype == "bar" and x_col in df.columns and y_col in df.columns:
                            grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
                            # Sort by metric and cap to top 15 to avoid clutter
                            grouped = grouped.sort_values(by=y_col, ascending=False).head(15)
                            fig = px.bar(grouped, x=x_col, y=y_col, title=title, template="plotly_white")
                        elif ctype == "line" and y_col in df.columns:
                            sorted_df = df.sort_values(x_col)
                            fig = px.line(sorted_df, x=x_col, y=y_col, title=title, template="plotly_white")
                        elif ctype == "scatter" and y_col in df.columns:
                            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=title, template="plotly_white")
                        elif ctype == "box" and y_col in df.columns:
                            fig = px.box(df, x=x_col, y=y_col, title=title, template="plotly_white")
                        elif ctype == "histogram":
                            fig = px.histogram(df, x=x_col, title=title, template="plotly_white")
                        else:
                            fig = px.bar(df.head(15), x=x_col, y=y_col if y_col in df.columns else None,
                                         title=title, template="plotly_white")

                        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                        chart_event = st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"blueprint_chart_{i + j}",
                            on_select="rerun",
                            selection_mode="points",
                        )
                        selected_values = list(dict.fromkeys(
                            point.get("x")
                            for point in chart_event.selection.points
                            if point.get("x") is not None
                        ))
                        if selected_values:
                            selected_values = [
                                value.item() if isinstance(value, np.generic) else value
                                for value in selected_values
                            ]
                            new_selection = {"column": x_col, "values": selected_values}
                            if st.session_state.get("chart_selection") != new_selection:
                                st.session_state["chart_selection"] = new_selection
                                st.rerun()
                    except Exception as err:
                        st.warning(f"Could not render {title}: {err}")

    st.write("---")

    # Strategic Narrative & Actions
    st.subheader("💡 Evidence & Execution Plan")
    col_findings, col_actions = st.columns(2)

    with col_findings:
        st.markdown("#### Measured facts for the active filters")
        for fact in build_measured_facts(df):
            st.info(fact)
        st.markdown("#### AI-generated interpretations")
        for finding in spec.what_is_happening:
            st.write(f"- {finding}")
        st.caption("Model interpretations are based on the full uploaded dataset; measured facts above reflect the active filters.")

    with col_actions:
        st.markdown("#### 🎯 Recommended Strategic Actions")
        for act in spec.recommended_actions:
            st.success(act)


def apply_dashboard_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df
    with st.sidebar:
        st.header("Dashboard filters")
        selection = st.session_state.get("chart_selection")
        if selection:
            st.caption(f"Chart filter: {selection['column']} ({len(selection['values'])} selected)")
            if st.button("Clear chart filter"):
                st.session_state.pop("chart_selection", None)
                st.rerun()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns
        for column in categorical_columns:
            values = df[column].dropna().unique().tolist()
            if 0 < len(values) <= 50:
                values = sorted(values, key=str)
                selected = st.multiselect(column.replace("_", " ").title(), values, default=values)
                if selected:
                    filtered_df = filtered_df[filtered_df[column].isin(selected)]

        date_columns = df.select_dtypes(include=["datetime"]).columns
        for column in date_columns:
            if df[column].notna().any():
                min_date = df[column].min().date()
                max_date = df[column].max().date()
                selected_range = st.date_input(
                    column.replace("_", " ").title(),
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"date_filter_{column}",
                )
                if isinstance(selected_range, tuple) and len(selected_range) == 2:
                    start_date, end_date = selected_range
                    filtered_df = filtered_df[
                        filtered_df[column].dt.date.between(start_date, end_date)
                    ]

        st.caption(f"{len(filtered_df):,} of {len(df):,} rows")
    return filtered_df


def render_chart_explorer(df: pd.DataFrame):
    if df.empty or len(df.columns) == 0:
        return

    with st.expander("Build a chart", expanded=False):
        chart_types = ["Bar", "Line", "Scatter", "Box", "Histogram"]
        chart_type = st.selectbox("Chart type", chart_types)
        x_col = st.selectbox("X axis", df.columns.tolist())
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        y_options = ["(row count)"] + numeric_columns
        y_col = st.selectbox("Y axis", y_options, disabled=chart_type == "Histogram")

        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, template="plotly_white")
        elif chart_type == "Bar" and y_col == "(row count)":
            grouped = df.groupby(x_col, dropna=False).size().reset_index(name="Rows")
            grouped = grouped.sort_values("Rows", ascending=False).head(30)
            fig = px.bar(grouped, x=x_col, y="Rows", template="plotly_white")
        elif y_col != "(row count)":
            if chart_type == "Bar":
                grouped = df.groupby(x_col, dropna=False)[y_col].sum().reset_index()
                grouped = grouped.sort_values(y_col, ascending=False).head(30)
                fig = px.bar(grouped, x=x_col, y=y_col, template="plotly_white")
            elif chart_type == "Line":
                fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, template="plotly_white")
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, template="plotly_white")
            else:
                fig = px.box(df, x=x_col, y=y_col, template="plotly_white")
        else:
            st.info("Choose a numeric Y axis for this chart type.")
            return

        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True, key="custom_chart")


def build_markdown_report(df: pd.DataFrame, spec: DashboardBlueprint, logs: list, quality: dict) -> str:
    def md(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    def number(value) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.3f}"

    report = [
        "# Data Analyst and Technical Report",
        "",
        "## Analysis scope",
        f"- Records in this analysis after active dashboard filters: {len(df):,}",
        f"- Fields in the cleaned dataset: {len(df.columns):,}",
        f"- Records in the uploaded file before duplicate removal: {quality['source_rows']:,}",
        f"- Records remaining after exact duplicate removal: {quality['cleaned_rows']:,}",
        "- Dashboard measures below use the active filters. Upload-level quality checks describe the full source file.",
        "",
        "## Executive KPIs",
        "| KPI | Field | Calculation | Value |",
        "| --- | --- | --- | ---: |",
    ]

    for kpi in spec.kpi_cards:
        if kpi.column not in df.columns:
            continue
        series = df[kpi.column]
        aggregations = {
            "sum": series.sum,
            "mean": series.mean,
            "count": series.count,
            "nunique": series.nunique,
        }
        value = aggregations[kpi.aggregation]()
        formatted_value = number(value) if isinstance(value, (int, float, np.number)) else md(value)
        report.append(f"| {md(kpi.label)} | `{md(kpi.column)}` | `{kpi.aggregation}` | {formatted_value} |")

    report.extend([
        "",
        "## Cleaned data schema and coverage",
        "| Field | Detected type | Records present | Missing after cleaning | Distinct values |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for column in df.columns:
        series = df[column]
        report.append(
            f"| `{md(column)}` | `{md(series.dtype)}` | {series.notna().sum():,} | "
            f"{series.isna().sum():,} | {series.nunique(dropna=True):,} |"
        )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report.extend([
        "",
        "## Numeric field profiles",
        "Values are calculated over the records included by the active filters.",
        "| Field | Count | Mean | Std. dev. | Min | 25th percentile | Median | 75th percentile | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for column in numeric_columns:
        values = df[column].dropna()
        quartiles = values.quantile([0.25, 0.5, 0.75]) if not values.empty else pd.Series(dtype=float)
        report.append(
            f"| `{md(column)}` | {len(values):,} | {number(values.mean()) if len(values) else 'n/a'} | "
            f"{number(values.std()) if len(values) > 1 else 'n/a'} | "
            f"{number(values.min()) if len(values) else 'n/a'} | "
            f"{number(quartiles.get(0.25, np.nan))} | {number(quartiles.get(0.5, np.nan))} | "
            f"{number(quartiles.get(0.75, np.nan))} | {number(values.max()) if len(values) else 'n/a'} |"
        )

    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    report.extend(["", "## Categorical field profiles"])
    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False).head(10)
        report.extend([f"### `{md(column)}`", "| Value | Records | Share |", "| --- | ---: | ---: |"])
        for value, count in counts.items():
            report.append(f"| {md(value)} | {count:,} | {count / max(len(df), 1) * 100:.1f}% |")
        if len(counts) == 0:
            report.append("| No values | 0 | n/a |")

    report.extend([
        "",
        "## Monthly and yearly comparisons",
        "The first detected date field and the first numeric KPI measure are used. If no numeric KPI is available, the report compares record counts.",
    ])
    date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
    primary_kpi = next(
        (kpi for kpi in spec.kpi_cards if kpi.column in numeric_columns and kpi.aggregation in {"sum", "mean", "count"}),
        None,
    )
    metric_col = primary_kpi.column if primary_kpi else (numeric_columns[0] if numeric_columns else "(row count)")
    aggregation = primary_kpi.aggregation if primary_kpi else "sum"
    if date_columns:
        date_column = date_columns[0]
        report.append(f"- Date field: `{md(date_column)}`")
        report.append(f"- Measure: `{md(primary_kpi.label if primary_kpi else metric_col)}` (`{aggregation}`)")
        for period_name in ("Month", "Year"):
            summary = build_period_summary(df, date_column, metric_col, period_name, aggregation)
            report.extend([
                "",
                f"### {period_name}ly summary" if period_name == "Month" else "### Yearly summary",
                "| Period | Measure | Change from prior period |",
                "| --- | ---: | ---: |",
            ])
            recent = summary.tail(36 if period_name == "Month" else 20)
            previous_value = None
            for row in recent.itertuples():
                value = float(row.Value)
                if previous_value is None:
                    change = "n/a"
                elif previous_value == 0:
                    change = "n/a (prior period was zero)"
                else:
                    change = f"{(value - previous_value) / abs(previous_value) * 100:+.1f}%"
                report.append(f"| {row.Period.strftime('%Y-%m' if period_name == 'Month' else '%Y')} | {number(value)} | {change} |")
                previous_value = value
            if summary.empty:
                report.append("| No dated records | n/a | n/a |")
    else:
        report.append("No date field was recognized; period-based comparisons could not be calculated.")

    report.extend(["", "## Categorical and numeric relationships"])
    if len(numeric_columns) >= 2 and len(df) >= 2:
        correlations = df[numeric_columns].corr().abs()
        pairs = []
        for left_index, left in enumerate(numeric_columns):
            for right in numeric_columns[left_index + 1:]:
                value = correlations.loc[left, right]
                if pd.notna(value):
                    pairs.append((left, right, float(value)))
        for left, right, value in sorted(pairs, key=lambda pair: pair[2], reverse=True)[:10]:
            report.append(f"- `{md(left)}` and `{md(right)}`: absolute correlation {value:.3f}.")
        if not pairs:
            report.append("No numeric relationships could be calculated from the selected records.")
    else:
        report.append("At least two numeric fields and two records are required to calculate numeric relationships.")
    report.append("Correlation describes how two measures move together; it does not establish that one caused the other.")

    report.extend(["", "## Measured facts for active filters"])
    report.extend(f"- {fact}" for fact in build_measured_facts(df))
    report.extend(["", "## AI-generated interpretations (full upload)"])
    report.extend(f"- {finding}" for finding in spec.what_is_happening)
    report.extend(["", "## Recommended actions"])
    report.extend(f"- {action}" for action in spec.recommended_actions)
    report.extend([
        "",
        "## Data quality checks (full upload, before cleaning)",
        f"- Exact duplicate records detected: {quality['duplicate_rows']:,}",
        f"- Missing cells detected: {quality['missing_cells']:,}",
        "- Unusual-value counts use a basic statistical range check. They flag records for review; they do not prove the values are errors.",
        "| Source field | Type after cleaning | Missing before cleaning | Missing share | Distinct source values | Unusual values flagged |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in quality["details"].iterrows():
        report.append(
            f"| `{md(row['Column'])}` | `{md(row['Detected type'])}` | {row['Missing before cleaning']:,} | "
            f"{row['Missing %']:.1f}% | {row['Unique values']:,} | {row['IQR outliers']:,} |"
        )

    report.extend([
        "",
        "## Dashboard chart specifications",
        "| Chart | Type | X field | Y field | Calculation |",
        "| --- | --- | --- | --- | --- |",
    ])
    for chart in spec.charts:
        report.append(
            f"| {md(chart.title)} | `{chart.type}` | `{md(chart.x)}` | "
            f"`{md(chart.y) if chart.y else 'record count'}` | `{chart.aggregation}` |"
        )

    report.extend(["", "## Cleaning and preparation log"])
    report.extend(f"- {entry}" for entry in logs or ["No cleaning changes were needed."])
    report.extend([
        "",
        "## Method and limitations",
        "- Column names were standardized; exact duplicate records were removed; numeric-looking text and date fields were converted when detected.",
        "- Missing numeric values were filled with the median, and missing text values were labeled `Unknown`; consult the quality table for the original missingness.",
        "- Dashboard measures, field profiles, correlations, and period comparisons use the active filtered records unless explicitly marked full upload.",
        "- AI-generated interpretations are based on the full upload summary and may not reflect the current filters. Validate them against the computed tables before relying on them.",
        "- This report describes patterns in the provided file. It cannot independently confirm operational causes, data lineage, or whether an unusual value is valid.",
    ])
    return "\n".join(report)


def build_data_quality_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    details = []
    for column in raw_df.columns:
        cleaned_name = (
            pd.Series([str(column)])
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", "_", regex=True)
            .iloc[0]
        )
        missing = int(raw_df[column].isna().sum())
        cleaned_series = cleaned_df[cleaned_name] if cleaned_name in cleaned_df else None
        outlier_count = 0
        if cleaned_series is not None and pd.api.types.is_numeric_dtype(cleaned_series):
            values = cleaned_series.dropna()
            if len(values) >= 4:
                first_quartile, third_quartile = values.quantile([0.25, 0.75])
                spread = third_quartile - first_quartile
                outlier_count = int(
                    ((values < first_quartile - 1.5 * spread) | (values > third_quartile + 1.5 * spread)).sum()
                )

        details.append({
            "Column": str(column),
            "Detected type": str(cleaned_series.dtype) if cleaned_series is not None else "empty / removed",
            "Missing before cleaning": missing,
            "Missing %": round(missing / max(len(raw_df), 1) * 100, 1),
            "Unique values": int(raw_df[column].nunique(dropna=True)),
            "IQR outliers": outlier_count,
        })

    return {
        "source_rows": len(raw_df),
        "cleaned_rows": len(cleaned_df),
        "duplicate_rows": int(raw_df.duplicated().sum()),
        "missing_cells": int(raw_df.isna().sum().sum()),
        "details": pd.DataFrame(details),
    }


def build_measured_facts(df: pd.DataFrame) -> list[str]:
    facts = []
    for column in df.select_dtypes(include=[np.number]).columns[:3]:
        values = df[column].dropna()
        if not values.empty:
            facts.append(
                f"{column}: median {values.median():,.2f}; range {values.min():,.2f} to {values.max():,.2f} ({len(values):,} rows)."
            )

    for column in df.select_dtypes(include=["object", "category"]).columns[:2]:
        counts = df[column].value_counts(dropna=False)
        if not counts.empty:
            top_value = counts.index[0]
            share = counts.iloc[0] / max(len(df), 1) * 100
            facts.append(f"{column}: '{top_value}' is most common ({counts.iloc[0]:,} rows, {share:.1f}%).")

    return facts or ["No measurable numeric or categorical patterns are available for this selection."]


def apply_chart_selection(df: pd.DataFrame) -> pd.DataFrame:
    selection = st.session_state.get("chart_selection")
    if not selection or selection["column"] not in df.columns:
        return df

    column = selection["column"]
    values = selection["values"]
    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        source = pd.to_datetime(series, errors="coerce", utc=True)
        selected = pd.to_datetime(values, errors="coerce", utc=True)
        return df[source.isin(selected)]
    if pd.api.types.is_numeric_dtype(series):
        selected = pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()
        return df[series.isin(selected)]
    return df[series.astype(str).isin([str(value) for value in values])]


def build_period_summary(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    period_name: str,
    aggregation: str,
) -> pd.DataFrame:
    frequency = {"Month": "M", "Quarter": "Q", "Year": "Y"}[period_name]
    periods = pd.to_datetime(df[date_col], errors="coerce").dt.to_period(frequency).dt.start_time
    comparison = pd.DataFrame({"Period": periods})
    comparison["Value"] = 1 if metric_col == "(row count)" else df[metric_col]
    operation = "sum" if metric_col == "(row count)" else aggregation.lower()
    return (
        comparison.dropna(subset=["Period"])
        .groupby("Period")["Value"]
        .agg(operation)
        .reset_index()
        .sort_values("Period")
    )


def render_period_comparison(df: pd.DataFrame):
    date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not date_columns or df.empty:
        return

    with st.expander("Compare performance over time", expanded=False):
        date_col = st.selectbox("Date field", date_columns, key="comparison_date")
        metric_options = ["(row count)"] + numeric_columns
        metric_col = st.selectbox("Metric", metric_options, key="comparison_metric")
        period_name = st.selectbox("Period", ["Month", "Quarter", "Year"], key="comparison_period")
        aggregation = st.selectbox(
            "Aggregation",
            ["Sum", "Mean", "Count"],
            key="comparison_aggregation",
            disabled=metric_col == "(row count)",
        )
        grouped = build_period_summary(df, date_col, metric_col, period_name, aggregation)

        if len(grouped) < 2:
            st.info("At least two time periods are needed for a comparison.")
            return

        previous_value = grouped.iloc[-2]["Value"]
        latest_value = grouped.iloc[-1]["Value"]
        delta = "n/a" if previous_value == 0 else f"{(latest_value - previous_value) / abs(previous_value) * 100:+.1f}% vs prior period"
        st.metric(f"Latest {period_name.lower()}", f"{latest_value:,.2f}", delta)
        fig = px.line(grouped, x="Period", y="Value", markers=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key="period_comparison_chart")


def build_stakeholder_report_sections(
    df: pd.DataFrame,
    spec: DashboardBlueprint,
    quality: dict,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[tuple[str, float]]]]]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    primary_kpi = next(
        (
            kpi for kpi in spec.kpi_cards
            if kpi.column in numeric_columns and kpi.aggregation in {"sum", "mean", "count"}
        ),
        None,
    )
    metric_col = primary_kpi.column if primary_kpi else (numeric_columns[0] if numeric_columns else "(row count)")
    aggregation = primary_kpi.aggregation if primary_kpi else "sum"
    metric_label = primary_kpi.label if primary_kpi else (metric_col if metric_col != "(row count)" else "record count")

    opening = [
        f"This report covers {quality['source_rows']:,} uploaded records. "
        f"After removing exact repeats, {quality['cleaned_rows']:,} records remain. "
        f"The figures below reflect the {len(df):,} records included by the current dashboard filters.",
    ]

    kpi_lines = []
    for kpi in spec.kpi_cards:
        if kpi.column not in df.columns:
            continue
        series = df[kpi.column]
        value = {
            "sum": series.sum,
            "mean": series.mean,
            "count": series.count,
            "nunique": series.nunique,
        }[kpi.aggregation]()
        kpi_lines.append(f"{kpi.label}: {value:,.2f}" if isinstance(value, (int, float, np.number)) else f"{kpi.label}: {value}")

    trend_sections = []
    date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
    if date_columns:
        date_col = date_columns[0]
        for period_name in ("Month", "Year"):
            grouped = build_period_summary(df, date_col, metric_col, period_name, aggregation)
            if grouped.empty:
                continue
            recent = grouped.tail(12 if period_name == "Month" else 6)
            values = [(row.Period.strftime("%b %Y" if period_name == "Month" else "%Y"), float(row.Value)) for row in recent.itertuples()]
            if len(grouped) >= 2:
                previous = float(grouped.iloc[-2]["Value"])
                latest = float(grouped.iloc[-1]["Value"])
                difference = latest - previous
                if previous == 0:
                    comparison = "The previous period was zero, so a percentage change would be misleading."
                else:
                    percent_change = difference / abs(previous) * 100
                    direction = "increased" if difference > 0 else "decreased" if difference < 0 else "was unchanged"
                    comparison = (
                        f"The latest {period_name.lower()} ({latest:,.2f}) {direction} by "
                        f"{abs(percent_change):.1f}% compared with the prior {period_name.lower()} "
                        f"({previous:,.2f}). This compares {metric_label.lower()} using {date_col.replace('_', ' ')}."
                    )
            else:
                comparison = f"Only one {period_name.lower()} is available, so there is no earlier period to compare."
            trend_sections.append((f"{period_name}-by-period trend", values))
            if period_name == "Month":
                opening.append(comparison)
            elif len(grouped) >= 2:
                opening.append(comparison)
    else:
        opening.append("No date field was recognized, so monthly and yearly comparisons are not available.")

    attention = []
    missing_fields = quality["details"].loc[quality["details"]["Missing before cleaning"] > 0]
    if not missing_fields.empty:
        top_missing = missing_fields.sort_values("Missing %", ascending=False).head(3)
        descriptions = ", ".join(
            f"{row['Column']} ({row['Missing %']:.1f}% missing)" for _, row in top_missing.iterrows()
        )
        attention.append(
            f"Some information was absent before cleaning: {descriptions}. Missing details can hide patterns "
            "or make comparisons less reliable. Check whether these fields were optional, skipped during entry, or omitted by an upstream export."
        )

    outlier_fields = quality["details"].loc[quality["details"]["IQR outliers"] > 0]
    if not outlier_fields.empty:
        top_outlier = outlier_fields.sort_values("IQR outliers", ascending=False).iloc[0]
        attention.append(
            f"{top_outlier['Column']} contains {int(top_outlier['IQR outliers']):,} unusually high or low values. "
            "These may be valid exceptional cases or entry errors; confirm them against the original records before acting."
        )

    if quality["duplicate_rows"]:
        attention.append(
            f"{quality['duplicate_rows']:,} exact repeat records were removed. Confirm that repeated entries are not legitimate separate events."
        )
    if not attention:
        attention.append("No major missing-value, repeat-record, or unusual-value warning was detected by these basic checks.")

    hypotheses = [
        "The file shows what changed, but it does not include enough context to prove why. For a meaningful rise or fall, compare staffing, pricing, availability, promotions, and reporting coverage for the same dates.",
        "If the change is unexpected, check whether the date range is complete and whether the same rules were used to record the earlier and later periods.",
    ]
    actions = list(spec.recommended_actions)
    if quality["duplicate_rows"] or not missing_fields.empty or not outlier_fields.empty:
        actions.insert(0, "Review the flagged records with the data owner and confirm corrections against the original source before changing business plans.")
    actions.extend([
        "Prevent repeat issues by requiring essential fields, checking records for duplicate IDs, and flagging unusually large or small values for review at entry time.",
        "Assign an owner to review the monthly report and investigate material changes before they reach leadership decisions.",
    ])

    sections = [
        ("Executive summary", opening),
        ("Key figures", kpi_lines or ["No executive measures were available for the selected records."]),
        (
            "Additional observations to verify",
            ["These observations are an AI-assisted reading of the uploaded information; confirm them against the source records before making decisions."]
            + list(spec.what_is_happening),
        ),
        ("What may need attention", attention),
        ("Possible explanations to investigate", hypotheses),
        ("Recommended next steps", list(dict.fromkeys(actions))),
    ]
    return sections, trend_sections


def build_pdf_report(df: pd.DataFrame, spec: DashboardBlueprint, logs: list, quality: dict) -> bytes:
    def safe_text(value) -> str:
        text = str(value).encode("latin-1", "replace").decode("latin-1")
        words = []
        for word in text.split():
            words.extend(word[index:index + 30] for index in range(0, len(word), 30))
        return " ".join(words)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    sections, trend_sections = build_stakeholder_report_sections(df, spec, quality)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Business Performance Briefing", ln=1)
    pdf.set_font("Arial", "", 10)

    def write_text(value, height: int):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, height, safe_text(value))

    write_text("Prepared from the uploaded records and the current dashboard filters.", 6)

    def section(title: str):
        pdf.ln(3)
        pdf.set_font("Arial", "B", 13)
        write_text(title, 8)
        pdf.set_font("Arial", "", 10)

    for title, lines in sections[:2]:
        section(title)
        for line in lines:
            write_text(f"- {line}", 6)

    for title, values in trend_sections:
        section(title)
        chart_x, chart_y, chart_width = 20, pdf.get_y() + 2, 170
        max_value = max((abs(value) for _, value in values), default=0) or 1
        for label, value in values:
            if pdf.get_y() > 255:
                pdf.add_page()
                chart_y = pdf.get_y()
            pdf.set_x(chart_x)
            pdf.cell(30, 6, safe_text(label))
            pdf.rect(chart_x + 32, pdf.get_y() + 1, chart_width - 32, 4)
            pdf.set_fill_color(45, 125, 105)
            pdf.rect(chart_x + 32, pdf.get_y() + 1, (chart_width - 32) * abs(value) / max_value, 4, "F")
            pdf.set_x(chart_x + chart_width - 30)
            pdf.cell(30, 6, safe_text(f"{value:,.2f}"), align="R", ln=1)
        pdf.ln(3)

    for title, lines in sections[2:]:
        section(title)
        for line in lines:
            write_text(f"- {line}", 6)

    section("How these figures were prepared")
    write_text("Dashboard measures use the active filters. Trend comparisons use the first recognized date field and the most relevant available measure. Unusual-value flags are prompts to review, not proof of an error. Possible explanations above are checks to make, not confirmed causes.", 6)
    pdf_output = pdf.output(dest="S")
    return bytes(pdf_output) if isinstance(pdf_output, (bytes, bytearray)) else pdf_output.encode("latin-1")

# ---------------------------------------------------------
# App Ingestion Pipeline
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload uncleaned CSV dataset", type=["csv"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    upload_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("active_upload_hash") != upload_hash:
        st.session_state["active_upload_hash"] = upload_hash
        st.session_state.pop("analysis_error", None)
        for state_key in ("cleaned_df", "clean_logs", "dashboard_blueprint", "data_quality"):
            st.session_state.pop(state_key, None)

        try:
            raw_df = pd.read_csv(io.BytesIO(file_bytes))
            cleaned_df, logs = clean_dataset(raw_df)
            quality = build_data_quality_report(raw_df, cleaned_df)
            if cleaned_df.empty or len(cleaned_df.columns) == 0:
                raise ValueError("The CSV has no usable rows or columns after cleaning.")

            with st.spinner("Gemini is analyzing the dataset and preparing your dashboard..."):
                profile = build_data_profile(cleaned_df)
                dashboard_blueprint = generate_dashboard_with_gemini(profile)

            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["clean_logs"] = logs
            st.session_state["dashboard_blueprint"] = dashboard_blueprint
            st.session_state["data_quality"] = quality
        except Exception as err:
            st.session_state["analysis_error"] = str(err)

    if "analysis_error" in st.session_state:
        st.error(f"Could not analyze this CSV: {st.session_state['analysis_error']}")
        if st.button("Retry analysis"):
            st.session_state.pop("active_upload_hash", None)
            st.rerun()
        st.stop()

    cleaned_df = st.session_state["cleaned_df"]
    filtered_df = apply_chart_selection(apply_dashboard_filters(cleaned_df))
    dashboard_tab, report_tab, data_tab, quality_tab = st.tabs(
        ["Dashboard", "Report & downloads", "Cleaned data", "Data quality"]
    )

    with dashboard_tab:
        if filtered_df.empty:
            st.warning("No rows match the current filters. Adjust the sidebar filters to continue.")
        else:
            render_dynamic_dashboard(filtered_df, st.session_state["dashboard_blueprint"])
            render_chart_explorer(filtered_df)
            render_period_comparison(filtered_df)
        with st.expander("View auto-cleaning audit log"):
            for entry in st.session_state["clean_logs"] or ["No cleaning changes were needed."]:
                st.write(f"- {entry}")

    with report_tab:
        report = build_markdown_report(
            filtered_df,
            st.session_state["dashboard_blueprint"],
            st.session_state["clean_logs"],
            st.session_state["data_quality"],
        )
        st.markdown("Measured facts and KPI values use the active filters. AI interpretations and data-quality metrics describe the full upload.")
        st.markdown(report)
        base_name = os.path.splitext(uploaded_file.name)[0]
        download_col, pdf_col, csv_col = st.columns(3)
        with download_col:
            st.download_button(
                "Download analysis report",
                data=report,
                file_name=f"{base_name}_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with pdf_col:
            st.download_button(
                "Download PDF report",
                data=build_pdf_report(
                    filtered_df,
                    st.session_state["dashboard_blueprint"],
                    st.session_state["clean_logs"],
                    st.session_state["data_quality"],
                ),
                file_name=f"{base_name}_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with csv_col:
            st.download_button(
                "Download filtered CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base_name}_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with data_tab:
        st.caption(f"Showing up to 1,000 of {len(filtered_df):,} filtered rows")
        st.dataframe(filtered_df.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "Download complete cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with quality_tab:
        quality = st.session_state["data_quality"]
        st.caption("Quality metrics describe the full uploaded file before cleaning, even when dashboard filters are active.")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Source rows", f"{quality['source_rows']:,}")
        metric_cols[1].metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
        metric_cols[2].metric("Missing cells", f"{quality['missing_cells']:,}")
        st.dataframe(quality["details"], use_container_width=True, hide_index=True)