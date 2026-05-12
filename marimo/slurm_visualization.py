import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import os
    from pathlib import Path
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime

    return Path, datetime, go, mo, os, pd, px


@app.cell
def _(mo):
    mo.md("""
    # ORCD Job Report
    """)
    return


@app.cell
def _(Path):
    DATA_DIR = Path("/orcd/data/orcd/022/util_viz/data/pi_partitions/pi_mghassem/")
    DATA_TYPE = "parquet"
    DATE_STRUCTURE = "month"
    return DATA_DIR, DATA_TYPE, DATE_STRUCTURE


@app.cell
def _(DATA_DIR, DATA_TYPE, DATE_STRUCTURE, os):
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No such directory: {DATA_DIR}")
    if DATA_TYPE not in ["parquet"]:
        raise ValueError(f"{DATA_TYPE} is not supported.")
    if DATE_STRUCTURE not in ["month"]:
        raise ValueError(f"{DATE_STRUCTURE} is not supported.")
    return


@app.cell
def _(DATA_DIR, DATA_TYPE, Path, mo, pd):
    files = list(Path(DATA_DIR).glob(f"*.{DATA_TYPE}"))
    file_names = [f.name for f in files]

    earliest_month = min(file_names)
    latest_month   = max(file_names)

    earliest_submit = pd.read_parquet(DATA_DIR / earliest_month, columns=["submit"]).submit.min()
    latest_submit   = pd.read_parquet(DATA_DIR / latest_month,   columns=["submit"]).submit.max()

    mo.output.append(mo.md("## Select Analysis Timeframe"))

    start_date = mo.ui.date(value=earliest_submit.date(), label="Start")
    end_date   = mo.ui.date(value=latest_submit.date(),   label="End")

    date_filter = mo.vstack([
        mo.md(f"**Date range in data (first submit, last submit):** `{earliest_submit.date()}` → `{latest_submit.date()}`"),
        start_date,
        end_date,
        mo.md("---"),
    ])
    mo.output.append(date_filter)

    button = mo.ui.run_button(label="Generate Report")
    mo.output.append(button)
    return button, end_date, file_names, start_date


@app.cell
def _(
    DATA_DIR,
    DATE_STRUCTURE,
    Path,
    button,
    datetime,
    end_date,
    file_names,
    mo,
    pd,
    start_date,
):
    mo.stop(
        not button.value,
        mo.md("Click \"Generate Report\" Button to get Report.")
    )

    def load_single_file(file_dir: Path, min_date: datetime, max_date: datetime):
        return pd.read_parquet(
            file_dir, engine="pyarrow",
            filters=[
                ("submit", ">=", pd.Timestamp(min_date)),
                ("submit", "<=", pd.Timestamp(max_date)),
            ],
        )

    def load_all_files(data_folder, data_filenames, min_date, max_date):
        min_ym = min_date.strftime("%Y%m")
        max_ym = max_date.strftime("%Y%m")
        dfs = []
        for file in data_filenames:
            if DATE_STRUCTURE == "month" and min_ym <= file[:6] <= max_ym:
                dfs.append(load_single_file(data_folder / file, min_date, max_date))
        return pd.concat(dfs, ignore_index=True)

    df = load_all_files(DATA_DIR, file_names, start_date.value, end_date.value)
    assert df.submit.min() >= pd.Timestamp(start_date.value)
    assert df.submit.max() <= pd.Timestamp(end_date.value)
    return (df,)


@app.cell
def _(df, pd):
    df_m = df.copy()
    df_m["alloctres_gpu"] = df_m["alloctres_gpu"].fillna(0)

    # Use existing cpu_hours if present; otherwise compute from ncpus * elapsed_seconds
    if "cpu_hours" not in df_m.columns:
        df_m["cpu_hours"] = df_m["ncpus"] * df_m["elapsed_seconds"] / 3600

    df_m["gpu_hours"] = df_m["alloctres_gpu"] * df_m["elapsed_seconds"] / 3600

    # timelimit_hours for efficiency (sacct stores minutes)
    if "timelimit" in df_m.columns:
        if pd.api.types.is_timedelta64_dtype(df_m["timelimit"]):
            df_m["timelimit_hours"] = df_m["timelimit"].dt.total_seconds() / 3600
        else:
            df_m["timelimit_hours"] = pd.to_numeric(df_m["timelimit"], errors="coerce") / 60
        df_m["cpu_hours_requested"] = df_m["ncpus"] * df_m["timelimit_hours"]
        df_m["gpu_hours_requested"] = df_m["alloctres_gpu"] * df_m["timelimit_hours"]
    else:
        df_m["cpu_hours_requested"] = df_m["cpu_hours"]
        df_m["gpu_hours_requested"] = df_m["gpu_hours"]
    return (df_m,)


@app.cell
def _(mo):
    granularity_ctrl = mo.ui.radio(
        options={"Daily": "D", "Weekly": "W", "Monthly": "MS"},
        value="Daily",
        label="Time Granularity",
        inline=True,
    )
    top_n_ctrl = mo.ui.slider(start=3, stop=20, value=10, step=1, label="Top N Users")
    top_metric_ctrl = mo.ui.dropdown(
        options=["Job Count", "CPU-Hours", "GPU-Hours"],
        value="Job Count",
        label="Rank users by",
    )
    mig_checkbox = mo.ui.checkbox(label="Bucket Multi-Instance GPUs (MIG)")
    return granularity_ctrl, mig_checkbox, top_metric_ctrl, top_n_ctrl


@app.cell
def _(df_m, mo):
    def _card(label, value, accent):
        return f"""
        <div style="flex:1;min-width:160px;background:#fff;border-radius:10px;
                    padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);
                    border:1px solid #e5e7eb;border-top:3px solid {accent};">
          <div style="font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
          <div style="font-size:30px;font-weight:800;color:#111827;">{value}</div>
        </div>"""

    kpi_cards = mo.Html(f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin:16px 0;">
      {_card("Total Jobs",  f"{len(df_m):,}",                 "#3b82f6")}
      {_card("CPU-Hours",   f"{df_m['cpu_hours'].sum():,.0f}", "#10b981")}
      {_card("GPU-Hours",   f"{df_m['gpu_hours'].sum():,.1f}", "#f59e0b")}
    </div>""")
    return (kpi_cards,)


@app.cell
def _(df_m, granularity_ctrl, mo, px):
    _freq  = granularity_ctrl.value
    _label = {"D": "Daily", "W": "Weekly", "MS": "Monthly"}[_freq]
    _ts    = df_m.set_index("submit").resample(_freq).size().reset_index(name="job_count")

    _fig = px.bar(
        _ts, x="submit", y="job_count",
        title=f"{_label} Job Count",
        labels={"submit": "Date", "job_count": "Jobs"},
        color_discrete_sequence=["#3b82f6"],
    )
    _fig.update_layout(template="plotly_white", bargap=0.25)
    job_count_chart = mo.ui.plotly(_fig)
    return (job_count_chart,)


@app.cell
def _(df_m, granularity_ctrl, mo, px):
    _freq  = granularity_ctrl.value
    _label = {"D": "Daily", "W": "Weekly", "MS": "Monthly"}[_freq]
    _ts    = df_m.set_index("start").resample(_freq)

    def _bar(col, color, title):
        _f = px.bar(
            _ts[col].sum().reset_index(),
            x="start", y=col, title=title,
            labels={"start": "Date", col: title},
            color_discrete_sequence=[color],
        )
        _f.update_layout(template="plotly_white", bargap=0.25)
        return mo.ui.plotly(_f)

    cpu_time_chart = _bar("cpu_hours", "#10b981", f"{_label} CPU-Hours")
    gpu_time_chart = _bar("gpu_hours", "#f59e0b", f"{_label} GPU-Hours")
    return cpu_time_chart, gpu_time_chart


@app.cell
def _(df_m, go, mo, top_metric_ctrl, top_n_ctrl):
    _n      = top_n_ctrl.value
    _metric = top_metric_ctrl.value
    _col    = {"Job Count": None, "CPU-Hours": "cpu_hours", "GPU-Hours": "gpu_hours"}[_metric]
    _stats = (
        df_m.groupby("user").size().reset_index(name="value")
        if _col is None
        else df_m.groupby("user")[_col].sum().reset_index(name="value")
    )
    _top = _stats.nlargest(_n, "value").sort_values("value")
    _fig = go.Figure(go.Bar(
        x=_top["value"], y=_top["user"], orientation="h",
        marker_color="#6366f1",
        text=_top["value"].apply(lambda v: f"{v:,.0f}"),
        textposition="outside",
    ))
    _fig.update_layout(
        title=f"Top {_n} Users by {_metric}",
        xaxis_title=_metric, yaxis_title=None,
        template="plotly_white",
        margin=dict(l=10, r=60, t=40, b=20),
        height=max(280, _n * 32),
    )
    top_users_chart = mo.ui.plotly(_fig)
    return (top_users_chart,)


@app.cell
def _(df_m, go, mo):
    _CAT_MAP = {
        "COMPLETED": "Completed", "FAILED": "Failed",
        "NODE_FAIL": "Failed",    "OUT_OF_MEMORY": "Failed",
        "TIMEOUT":   "Timed Out", "PREEMPTED": "Preempted",
    }
    _COLORS = {
        "Completed": "#22c55e", "Failed":    "#ef4444",
        "Timed Out": "#f97316", "Preempted": "#a855f7",
        "Cancelled": "#94a3b8", "Other":     "#64748b",
    }

    def _categorize(s):
        s = str(s).upper().strip()
        if s in _CAT_MAP:          return _CAT_MAP[s]
        if s.startswith("CANCEL"): return "Cancelled"
        return "Other"

    _h = df_m.copy()
    _h["category"] = _h["state"].apply(_categorize)
    _counts = (
        _h.groupby("category").size().reset_index(name="count")
        .assign(pct=lambda d: d["count"] / d["count"].sum() * 100)
        .sort_values("count", ascending=False)
    )

    _fig = go.Figure(go.Pie(
        labels=_counts["category"], values=_counts["count"], hole=0.55,
        marker_colors=[_COLORS.get(c, "#64748b") for c in _counts["category"]],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} jobs (%{percent})<extra></extra>",
    ))
    _fig.update_layout(
        showlegend=False,
        template="plotly_white", height=300,
        margin=dict(t=40, b=10, l=10, r=10),
    )

    _badges = "".join(
        f"""<div style="display:flex;align-items:center;gap:10px;padding:6px 0;
                        border-bottom:1px solid #f3f4f6;">
          <div style="width:12px;height:12px;border-radius:3px;flex-shrink:0;
                      background:{_COLORS.get(r['category'], '#64748b')};"></div>
          <span style="font-size:13px;font-weight:600;color:#374151;min-width:90px;">
            {r['category']}
          </span>
          <span style="font-size:13px;color:#6b7280;">{r['count']:,} jobs</span>
          <span style="font-size:13px;color:#9ca3af;margin-left:auto;">{r['pct']:.1f}%</span>
        </div>"""
        for _, r in _counts.iterrows()
    )

    health_section = mo.hstack([
        mo.ui.plotly(_fig),
        mo.Html(f"""
        <div style="min-width:280px;background:#fff;border-radius:10px;padding:16px 20px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);border:1px solid #e5e7eb;
                    align-self:center;">
          <div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:10px;">Job Outcomes</div>
          {_badges}
        </div>"""),
    ])
    return (health_section,)


@app.cell
def _(cpu_time_chart, df_m, gpu_time_chart, job_count_chart, mo, px):
    def prepare_gpu_df(df):
        gpu_df = df[df["alloctres_gpu"] > 0].copy()
        gpu_df = gpu_df[gpu_df["alloctres_gpu_type"] != "unspecified"]
        gpu_df["allocated_gpu"] = (
            gpu_df["alloctres_gpu_type"].astype(str).str.strip().str.lower()
        )
        gpu_df["allocated_gpu_MIG_bin"] = gpu_df["alloctres_gpu_type"].str.replace(
            r"(\w+)_(\d+g\.\d+gb)", r"\1_MIG", regex=True
        )
        return gpu_df

    def get_color_map(df, column):
        unique_labels = sorted(df[column].unique())
        palette = px.colors.qualitative.Plotly
        return {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    def gpu_time_series(gpu_df, column, color_map):
        _df = gpu_df.copy()
        _df["day"] = _df["submit"].dt.floor("D")
        gpu_time = _df.groupby(["day", column])["alloctres_gpu"].sum().reset_index()
        categories = sorted(gpu_time[column].dropna().unique(), reverse=True)

        fig = px.area(
            gpu_time, x="day", y="alloctres_gpu",
            title="Daily GPU Utilization by Type",
            color=column, color_discrete_map=color_map,
            category_orders={column: categories},
        )
        fig.update_layout(
            xaxis_title="Day", yaxis_title="Total GPUs Requested",
            template="plotly_white", hovermode="x unified",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>GPUs: %{y:,.0f}<extra></extra>",
            mode="lines", line=dict(width=0.5), stackgroup="one",
        )
        return mo.ui.plotly(fig)

    def gpu_pie_chart(gpu_df, column, color_map):
        gpu_usage = (
            gpu_df.groupby(column)["alloctres_gpu"].sum()
            .reset_index().sort_values("alloctres_gpu", ascending=False)
        )
        fig = px.pie(
            gpu_usage, values="alloctres_gpu", names=column,
            title="GPU Usage Distribution", hole=0.4,
            color=column, color_discrete_map=color_map,
        )
        fig.update_layout(template="plotly_white", margin=dict(t=50, b=20, l=20, r=20))
        fig.update_traces(
            textposition="auto", textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>GPUs: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
            marker=dict(line=dict(color="white", width=2)),
        )
        return mo.ui.plotly(fig)

    gpu_df = prepare_gpu_df(df_m)

    chart_tabs = mo.ui.tabs({
        "Job Count": job_count_chart,
        "CPU-Hours": cpu_time_chart,
        "GPU-Hours": gpu_time_chart,
    })
    return chart_tabs, get_color_map, gpu_df, gpu_pie_chart, gpu_time_series


@app.cell
def _(
    chart_tabs,
    cpu_time_chart,
    df,
    get_color_map,
    gpu_df,
    gpu_pie_chart,
    gpu_time_series,
    granularity_ctrl,
    health_section,
    kpi_cards,
    mig_checkbox,
    mo,
    top_metric_ctrl,
    top_n_ctrl,
    top_users_chart,
):
    _gpu_column = "allocated_gpu_MIG_bin" if mig_checkbox.value else "allocated_gpu"
    _color_map  = get_color_map(gpu_df, _gpu_column)

    overview_tab = mo.vstack([
        kpi_cards,
        mo.md("---"),
        mo.md("### Utilization"),
        granularity_ctrl,
        chart_tabs,
        mo.md("---"),
        mo.md("### Top Users"),
        mo.hstack([top_n_ctrl, top_metric_ctrl], wrap=True),
        top_users_chart,
        mo.md("---"),
        mo.md("### Job Health"),
        health_section,
    ])

    cpu_tab = mo.vstack([
        cpu_time_chart
    ])

    gpu_tab = mo.vstack([
        mig_checkbox,
        gpu_time_series(gpu_df, _gpu_column, _color_map),
        gpu_pie_chart(gpu_df, _gpu_column, _color_map),
    ])

    tabs = mo.ui.tabs({
        "Overview": overview_tab,
        "GPU Breakdown": gpu_tab,
        "Raw Data": df
    })
    return (tabs,)


@app.cell
def _(tabs):
    tabs
    return


if __name__ == "__main__":
    app.run()
