import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    # Import Libraries
    import pandas as pd
    import marimo as mo
    import os 
    import glob
    from pathlib import Path
    import plotly.express as px
    from datetime import datetime
    import re
    import hashlib
    import colorsys

    return Path, datetime, mo, os, pd, px


@app.cell
def _(mo):
    mo.md("""
    # ORCD Job Report
    """)
    return


@app.cell
def _(Path):
    ### CONFIG ###

    # DATA_DIR = Path("/orcd/data/orcd/022/util_viz/data/parsed_monthly") # folder containing parsed data
    DATA_DIR = Path("/orcd/data/orcd/022/util_viz/data/pi_partitions/pi_mghassem/") # folder containing parsed data
    DATA_TYPE = "parquet" # currently only supports parquet

    # How the data files are split up. If "month", files are in this format: {year}{month}*.parquet (e.g., 202512-sacct.parquet)
    DATE_STRUCTURE = "month"

    # GPU Visualization Color Palette
    gpu_colors = {
        "h100": "#1f77b4",  # blue
        "h200": "#ff7f0e",  # orange
        "l40s": "#2ca02c",  # green
        "a100": "#d62728",  # red
        "b200": "#9467bd",  # purple
    }
    return DATA_DIR, DATA_TYPE, DATE_STRUCTURE


@app.cell
def _(DATA_DIR, DATA_TYPE, DATE_STRUCTURE, os):
    # Check whether config is appropriate
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No such directory: {DATA_DIR}")

    supported_data_type = ["parquet"]
    supported_date_structure = ["month"]

    if DATA_TYPE not in supported_data_type:
        raise ValueError(f"{DATA_TYPE} is not supported.")
    if DATE_STRUCTURE not in supported_date_structure:
        raise ValueError(f"{DATE_STRUCTURE} is not supported.")
    return


@app.cell
def _(DATA_DIR, DATA_TYPE, DATE_STRUCTURE, Path, mo, pd):
    files = list(Path(DATA_DIR).glob(f"*.{DATA_TYPE}"))
    file_names = [f.name for f in files]

    # Extract the date of each data file
    if DATE_STRUCTURE == "month":
        # Uses structure: {year}{month}*.{data type} (e.g., 202505-sacct.parquet)
        earliest_month = min(file_names)
        latest_month = max(file_names)

        # Load earliest and latest date
        earliest_submit = pd.read_parquet(DATA_DIR / earliest_month, columns=["submit"]).submit.min()
        latest_submit = pd.read_parquet(DATA_DIR / latest_month, columns=["submit"]).submit.max()

        mo.output.append(mo.md("## Select Analysis Timeframe"))

        start_date = mo.ui.date(
            value=earliest_submit.date(),
            label="Start"
        )
        end_date = mo.ui.date(
            value=latest_submit.date(),
            label="End"
        )

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
def _(button, mo):
    mo.stop(
        not button.value,
    )
    mo.md("Generating...")
    return


@app.cell
def _(
    DATA_DIR,
    DATE_STRUCTURE,
    Path,
    datetime,
    end_date,
    file_names,
    pd,
    start_date,
):
    # get all appropriate folders

    def load_single_file(file_dir: Path, min_date: datetime, max_date: datetime):
        return pd.read_parquet(
            file_dir,
            engine='pyarrow',
            filters=[
                ('submit', '>=', pd.Timestamp(min_date)),
                ('submit', '<=', pd.Timestamp(max_date))
            ]
        )

    def load_all_files(data_folder: Path, data_filenames: list[str], min_date: datetime, max_date: datetime):
        min_formatted = min_date.strftime("%Y%m")
        max_formatted = max_date.strftime("%Y%m")

        dfs = []
        for file in data_filenames:
            if DATE_STRUCTURE == "month":
                file_date = file[:6]
                if min_formatted <= file_date <= max_formatted:
                    df = load_single_file(data_folder / file, min_date, max_date)
                    dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df

    df = load_all_files(DATA_DIR, file_names, start_date.value, end_date.value)
    # sanity check
    assert df.submit.min() >= pd.Timestamp(start_date.value)
    assert df.submit.max() <= pd.Timestamp(end_date.value)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, mo):
    import plotly.graph_objects as go

    user_counts = (
        df.groupby('user')
          .size()
          .reset_index(name='num_jobs')
          .sort_values(by='num_jobs', ascending=False)
    )

    # Top 5 users
    top_users = user_counts.head(5)
    top_user_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["User", "Number of Jobs"],
            align="left"
        ),
        cells=dict(
            values=[top_users['user'], top_users['num_jobs']],
            align="left"
        )
    )])

    top_user_fig.update_layout(title="Top 5 Users by Number of Jobs Submitted")

    mo.ui.plotly(top_user_fig)
    return


@app.cell
def _(df, end_date, mo, px, start_date):
    # Group by hour to see temporal trends

    duration = (end_date.value - start_date.value).days
    freq = "D" if duration > 14 else "h"
    fig_title = "Total CPU Used (Daily)" if freq == "D" else "Total CPU Used (Hourly)"

    hourly_cpu = (
        df.set_index("start")
        .resample(freq)["ncpus"]
        .sum()
        .reset_index()
    )

    fig_hourly = px.bar(
        hourly_cpu,
        x="start",
        y="ncpus",
        title=fig_title,
        labels={"start": "Time", "ncpus": "CPU Used"},
    )

    fig_hourly.update_layout(template="plotly_white")
    mo.ui.plotly(fig_hourly)
    return (freq,)


@app.cell
def _(df, freq, mo, px):
    # Group by hour to see temporal trends

    gpu_fig_title = "Total GPU Used (Daily)" if freq == "D" else "Total GPU Used (Hourly)"

    hourly_gpu = (
        df.set_index("start")
        .resample(freq)["alloctres_gpu"]
        .sum()
        .reset_index()
    )

    fig_hourly_gpu = px.bar(
        hourly_gpu,
        x="start",
        y="alloctres_gpu",
        title=gpu_fig_title,
        labels={"start": "Time", "alloctres_gpu": "GPU Used"},
    )

    fig_hourly_gpu.update_layout(template="plotly_white")
    mo.ui.plotly(fig_hourly_gpu)
    return


@app.cell
def _(df, mo, px):
    def get_consistent_color_map(df, column):
        unique_labels = sorted(df[column].unique())
        palette = px.colors.qualitative.Plotly 
        return {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    def gpu_piechart(gpu_df, color_map, order, target_column):
        # Aggregate GPU count by type
        gpu_usage = (
            gpu_df.groupby(target_column)["alloctres_gpu"]
            .sum()
            .reset_index()
            .sort_values("alloctres_gpu", ascending=False)
        )

        fig_gpu = px.pie(
            gpu_usage,
            values="alloctres_gpu",
            names=target_column,
            title="GPU Usage Distribution by Requested Type (Excluding Unspecified)",
            hole=0.4,
            color=target_column,
            color_discrete_map=color_map,
        )

        fig_gpu.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        return mo.ui.plotly(fig_gpu)

    def gpu_by_time(gpu_df, color_map, target_column):
        gpu_df["day"] = gpu_df["submit"].dt.floor("D")

        gpu_time = (
            gpu_df.groupby(["day", target_column])["alloctres_gpu"]
            .sum()
            .reset_index()
        )

        order = gpu_time.sort_values(target_column, ascending=False)[target_column].tolist()

        fig = px.area(
            gpu_time,
            x="day",
            y="alloctres_gpu",
            title="Daily GPU Utilization by Type",
            color=target_column,
            color_discrete_map=color_map,
            category_orders={target_column: order}
        )

        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Total GPUs Requested",
            margin=dict(t=50, b=20, l=20, r=20)
        )

        return mo.ui.plotly(fig), order

    gpu_df = df[df["alloctres_gpu"] > 0].copy()
    gpu_df = gpu_df[gpu_df.alloctres_gpu_type != "unspecified"]
    gpu_df["allocated_gpu"] = gpu_df['alloctres_gpu_type'].astype(str).str.strip().str.lower()
    gpu_df["allocated_gpu_MIG_bin"] = gpu_df['alloctres_gpu_type'].str.replace(
        r"(\w+)_(\d+g\.\d+gb)", 
        r"\1_MIG", 
        regex=True
    )
    return get_consistent_color_map, gpu_by_time, gpu_df, gpu_piechart


@app.cell
def _(mo):
    checkbox = mo.ui.checkbox(label="Bucket Multi-Instance GPUs (MIG)")
    return (checkbox,)


@app.cell
def _(
    checkbox,
    get_consistent_color_map,
    gpu_by_time,
    gpu_df,
    gpu_piechart,
    mo,
):
    mo.output.append(mo.hstack([checkbox, mo.md(f"Bucketting MIGs: {checkbox.value}")]))

    if len(gpu_df) == 0:
        mo.output.append(mo.md("No Specified GPUs Found"))

    if len(gpu_df) > 0:
        gpu_column = "allocated_gpu_MIG_bin" if checkbox.value else "allocated_gpu" 

        color_map = get_consistent_color_map(gpu_df, gpu_column)
        gpu_across_time, order = gpu_by_time(gpu_df, color_map, gpu_column)
        gpu_pie = gpu_piechart(gpu_df, color_map, order, gpu_column)

        mo.output.append(gpu_pie)
        mo.output.append(gpu_across_time)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
