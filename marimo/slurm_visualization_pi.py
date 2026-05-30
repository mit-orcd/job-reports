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
    from dateutil.relativedelta import relativedelta
    import subprocess
    import re

    return Path, datetime, go, mo, os, pd, px, re, relativedelta, subprocess


@app.cell
def _(mo):
    mo.md("""
    # ORCD Job Report
    """)
    return


@app.cell
def _(Path):
    ### CONFIG ###
    DATA_DIR = Path("/orcd/data/orcd/022/util_viz/data/pi_partitions/pi_mghassem/")
    DATA_TYPE = "parquet" # currently only supports parquet
    DATE_STRUCTURE = "month" # currently only supports month
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
def _(DATA_DIR, DATA_TYPE, Path, mo, pd, relativedelta):
    ### User Timeframe Config ###

    # Get data files
    files = list(Path(DATA_DIR).glob(f"*.{DATA_TYPE}"))
    file_names = [f.name for f in files]

    # Compute date range of available files 
    earliest_month = min(file_names)
    latest_month   = max(file_names)
    earliest_submit = pd.read_parquet(DATA_DIR / earliest_month, columns=["submit"]).submit.min()
    latest_submit   = pd.read_parquet(DATA_DIR / latest_month,   columns=["submit"]).submit.max()

    # User-selected start and end date
    start_date = mo.ui.date(value=(latest_submit - relativedelta(months=1)).date(), label="Start")
    end_date   = mo.ui.date(value=latest_submit.date(), label="End")
    button = mo.ui.run_button(label="Generate Report")

    date_filter = mo.vstack([
        mo.md(f"**Date range in data (first submit, last submit):** `{earliest_submit.date()}` → `{latest_submit.date()}`"),
        start_date,
        end_date,
        mo.md("---"),
    ])


    mo.output.append(mo.md("## Select Analysis Timeframe"))
    mo.output.append(date_filter)
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
    ### Load Dataset using user config ###

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

    # Logic gate to prevent future code from running until button is pressed
    mo.stop(
        not button.value,
        mo.md("Click \"Generate Report\" Button to get Report.")
    )

    # Load all files within the user-specified start and end date
    df = load_all_files(DATA_DIR, file_names, start_date.value, end_date.value)

    # Confirm dataframe is within user-specified date range
    assert df.submit.min() >= pd.Timestamp(start_date.value)
    assert df.submit.max() <= pd.Timestamp(end_date.value)
    return (df,)


@app.cell
def _(df):
    ### Modify dataframe for job-level visualizations ###

    df_m = df.copy()
    df_m["alloctres_gpu"] = df_m["alloctres_gpu"].fillna(0)

    # Compute CPU and GPU hours
    df_m["cpu_hours"] = df_m["alloctres_cpu"] * df_m["elapsed_seconds"] / 3600
    df_m["gpu_hours"] = df_m["alloctres_gpu"] * df_m["elapsed_seconds"] / 3600
    return (df_m,)


@app.cell
def _(df, end_date, pd, re, start_date, subprocess):
    ### Get per-node CPU and GPU resource capacity information for node-level visualizations ###

    def load_node_specs():
        """Get per-node resource information""" 
        result = subprocess.run(
            ["sinfo", "-o", "%n %c %G", "--noheader"],
            capture_output=True, text=True
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            node, cpus, gres = parts[0], parts[1], parts[2]
            gpus = 0
            if gres != "(null)":
                try:
                    gpus = int(gres.split(":")[-1])
                except ValueError:
                    pass
            rows.append({
                "node": node.lower(),
                "cpu_capacity": int(cpus),
                "gpu_capacity": gpus
            })
        specs = pd.DataFrame(rows).drop_duplicates("node")
        return specs


    def expand_nodelist(nodelist_str):
        """
        Expands SLURM nodelist strings into individual node names.
        Handles:
          - plain comma-separated:  'node01,node02'
          - bracket ranges:         'node[01-03]'
          - mixed:                  'node[01-02],gpu[03-04]'
          - zero-padded:            'node[001-003]'
        """
        nodes = []
        # Split on commas that are NOT inside brackets
        parts = re.split(r",(?![^\[]*\])", nodelist_str)
        for part in parts:
            part = part.strip()
            bracket_match = re.match(r"^(.*?)\[(.+)\]$", part)
            if bracket_match:
                prefix = bracket_match.group(1)
                ranges = bracket_match.group(2).split(",")
                for r in ranges:
                    if "-" in r:
                        start, end = r.split("-")
                        width = len(start)  # preserve zero-padding
                        for i in range(int(start), int(end) + 1):
                            nodes.append(f"{prefix}{str(i).zfill(width)}")
                    else:
                        nodes.append(f"{prefix}{r}")
            else:
                nodes.append(part)
        return nodes


    def prepare_per_node_df(df):
        # Only jobs that actually ran on nodes
        _df = df[df["nodelist"].notna() & (df["nodelist"] != "") & (df["nodelist"] != "None assigned") & (df["nnodes"] > 0)].copy()

        # Expand nodelist into individual nodes
        _df["node"] = _df["nodelist"].apply(expand_nodelist)
        _df = _df.explode("node")
        _df["node"] = _df["node"].str.strip().str.lower()

        # Distribute CPU and GPU evenly across nodes
        _df["cpus_per_node"] = _df["alloctres_cpu"] / _df["nnodes"]
        _df["gpus_per_node"] = _df["alloctres_gpu"] / _df["nnodes"]

        return _df


    def compute_node_utilization(df, start, end):
        timeframe_seconds = (
            pd.Timestamp(end) - pd.Timestamp(start)
        ).total_seconds()

        node_specs = load_node_specs()
        per_node_df = prepare_per_node_df(df)

        merged = per_node_df.merge(node_specs, on="node", how="left")

        # Flag nodes missing from sinfo
        missing = merged[merged["cpu_capacity"].isna()]["node"].unique()
        if len(missing):
            print(f"Warning: {len(missing)} nodes not found in sinfo: {missing[:5]}")

        merged["gpu_seconds"] = merged["gpus_per_node"] * merged["elapsed_seconds"]
        merged["cpu_seconds"] = merged["cpus_per_node"] * merged["elapsed_seconds"]

        node_agg = (
            merged.groupby("node")
            .agg(
                gpu_seconds  =("gpu_seconds",   "sum"),
                cpu_seconds  =("cpu_seconds",   "sum"),
                gpu_capacity =("gpu_capacity",  "first"),
                cpu_capacity =("cpu_capacity",  "first"),
                job_count    =("node",          "count"),
            )
            .reset_index()
        )

        node_agg["gpu_util_pct"] = (
            node_agg["gpu_seconds"] /
            (node_agg["gpu_capacity"] * timeframe_seconds)
        ).where(node_agg["gpu_capacity"] > 0, 0)

        node_agg["cpu_util_pct"] = (
            node_agg["cpu_seconds"] /
            (node_agg["cpu_capacity"] * timeframe_seconds)
        ).where(node_agg["cpu_capacity"] > 0, 0)

        node_agg["timeframe_seconds"] = timeframe_seconds
        node_agg["gpu_avail"] = node_agg["gpu_capacity"] > 0
    
        return node_agg


    node_util_df = compute_node_utilization(df, start_date.value, end_date.value)
    return (node_util_df,)


@app.cell
def _(mo):
    ### Create interactive buttons for visualization customization ###

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
    ### Create usage summary cards ###

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
    ### Job Count Visualization ###

    _freq  = granularity_ctrl.value
    _label = {"D": "Daily", "W": "Weekly", "MS": "Monthly"}[_freq]
    _ts    = df_m.set_index("submit").resample(_freq).size().reset_index(name="job_count")

    _fig = px.bar(
        _ts, x="submit", y="job_count",
        title=f"{_label} Job Count",
        labels={"submit": "Date", "job_count": "Jobs"},
        color_discrete_sequence=["#3b82f6"],
    )

    _fig.update_traces(
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            "Jobs: <b>%{y:,}</b><br>"
            "<extra></extra>"
        )
    )

    _fig.update_layout(template="plotly_white", bargap=0.25)
    job_count_chart = mo.ui.plotly(_fig)
    return (job_count_chart,)


@app.cell
def _(df_m, granularity_ctrl, mo, px):
    ### CPU hours and GPU Hours Bar Chart ###

    _freq  = granularity_ctrl.value
    _label = {"D": "Daily", "W": "Weekly", "MS": "Monthly"}[_freq]
    _ts    = df_m.set_index("start").resample(_freq)

    def _bar(col, color, title):
        _unit = "GPU-Hours" if "gpu" in col else "CPU-Hours"
        _f = px.bar(
            _ts[col].sum().reset_index(),
            x="start", y=col, title=title,
            labels={"start": "Date", col: title},
            color_discrete_sequence=[color],
        )
        _f.update_traces(
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                f"{_unit}: <b>%{{y:,.1f}}</b><br>"
                "<extra></extra>"
            )
        )
        _f.update_layout(template="plotly_white", bargap=0.25)
        return mo.ui.plotly(_f)

    cpu_time_chart = _bar("cpu_hours", "#10b981", f"{_label} CPU-Hours")
    gpu_time_chart = _bar("gpu_hours", "#f59e0b", f"{_label} GPU-Hours")
    return cpu_time_chart, gpu_time_chart


@app.cell
def _(df_m, go, mo, top_metric_ctrl, top_n_ctrl):
    ### Top User Chart ###
    _n      = top_n_ctrl.value
    _metric = top_metric_ctrl.value
    _col    = {"Job Count": None, "CPU-Hours": "cpu_hours", "GPU-Hours": "gpu_hours"}[_metric]
    _stats = (
        df_m.groupby("user").size().reset_index(name="value")
        if _col is None
        else df_m.groupby("user")[_col].sum().reset_index(name="value")
    )
    _top = _stats.nlargest(_n, "value").sort_values("value")

    _fmt = ".0f" if _metric == "Job Count" else ",.1f"
    _hovertemplate = (
        "<b>%{y}</b><br>"
        f"{_metric}: <b>%{{x:{_fmt}}}</b><br>"
        "<extra></extra>"
    )

    _fig = go.Figure(go.Bar(
        x=_top["value"], y=_top["user"], orientation="h",
        marker_color="#6366f1",
        text=_top["value"].apply(lambda v: f"{v:,.0f}"),
        textposition="outside",
        hovertemplate=_hovertemplate,
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
    ### Job Outcomes ###
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
        if s in _CAT_MAP:          
            return _CAT_MAP[s]
        if s.startswith("CANCEL"): 
            return "Cancelled"
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
def _(df_m, mo, px):
    ### GPU Usage ###

    def prepare_gpu_df(df):
        """Format GPU jobs"""
        gpu_df = df[df["alloctres_gpu"] > 0].copy()
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
        """Creates a time-series plot of GPU Utilization by type"""
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
        """Creates a pie chart of the GPU Utilization by type"""
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
    return get_color_map, gpu_df, gpu_pie_chart, gpu_time_series


@app.cell
def _(mo, node_util_df, px):
    ### Per-node Utilization ###

    def node_util_bar(node_util_df, metric="gpu_util_pct", title="Mean GPU Utilization % by Node"):
        if "gpu" in metric.lower():
            node_util_df = node_util_df[node_util_df["gpu_avail"]]
        
        node_avg = (
            node_util_df.groupby("node")[metric]
            .mean()
            .reset_index()
            .sort_values(metric, ascending=True)
        )
        node_avg[metric] *= 100

        fig = px.bar(
            node_avg, x=metric, y="node",
            orientation="h",
            title=title,
            labels={metric: "Mean Utilization %", "node": "Node"},
            color=metric,
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(t=50, b=20, l=20, r=20),
            coloraxis_showscale=False,
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Utilization: %{x:.1f}%<extra></extra>",
        )
        return mo.ui.plotly(fig)


    gpu_node_bar    = node_util_bar(node_util_df, "gpu_util_pct", "Mean GPU Utilization % by Node")
    cpu_node_bar    = node_util_bar(node_util_df, "cpu_util_pct", "Mean CPU Utilization % by Node")
    return cpu_node_bar, gpu_node_bar


@app.cell
def _(
    cpu_node_bar,
    cpu_time_chart,
    gpu_node_bar,
    gpu_time_chart,
    job_count_chart,
    mo,
):
    chart_tabs = mo.ui.tabs({
        "Job Count": job_count_chart,
        "CPU-Hours": cpu_time_chart,
        "GPU-Hours": gpu_time_chart,
    })

    node_util_tabs = mo.ui.tabs({
        "Per-Node Utilization (GPU)": gpu_node_bar,
        "Per-Node Utilization (CPU)": cpu_node_bar,
    })

    return chart_tabs, node_util_tabs


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
    node_util_tabs,
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
        node_util_tabs,
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
