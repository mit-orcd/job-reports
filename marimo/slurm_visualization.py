import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    # Import Libraries
    import pandas as pd
    import marimo as mo
    import numpy as np
    import os 
    import glob
    from pathlib import Path
    import plotly.express as px
    from datetime import datetime
    import re
    import hashlib
    import colorsys
    from dateutil.relativedelta import relativedelta

    return Path, datetime, mo, os, pd, px, relativedelta


@app.cell
def _(mo):
    mo.md("""
    # ORCD Job Report
    """)
    return


@app.cell
def _(Path):
    ### CONFIG ###

    DATA_DIR = Path("/orcd/data/orcd/022/util_viz/data/parsed_monthly") # folder containing parsed data
    DATA_TYPE = "parquet" # currently only supports parquet
    METADATA_FILE = DATA_DIR / "metadata.parquet"

    # How the data files are split up. If "month", files are in this format: {year}{month}*.parquet (e.g., 202512-sacct.parquet)
    DATE_STRUCTURE = "month"
    return DATA_DIR, DATA_TYPE, DATE_STRUCTURE, METADATA_FILE


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
def _(METADATA_FILE, mo, pd, relativedelta):
    # Load Meta data
    meta_data = pd.read_parquet(METADATA_FILE)

    # Filter Date
    earliest_submit = meta_data['earliest_submit'].min()
    latest_submit = meta_data['latest_submit'].max()

    start_date = mo.ui.date(
        value=latest_submit.date() - relativedelta(months=1) ,
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
    return end_date, meta_data, start_date


@app.cell
def _(end_date, meta_data, mo, pd, start_date):
    # Filter for Partitions
    filtered_meta = meta_data[
        (meta_data["latest_submit"] >= pd.Timestamp(start_date.value)) &
        (meta_data["earliest_submit"] <= pd.Timestamp(end_date.value))
    ]


    available_partitions = sorted(
        set("|".join(filtered_meta["partitions"]).split("|"))
    )
    available_partitions = [name for name in available_partitions if "," not in name]


    partition_search = mo.ui.multiselect(
        options=available_partitions,
        label=f"Partitions ({len(available_partitions)} found)",
    )

    controls = mo.vstack([
        mo.md("## Filters"),
        partition_search,
    ])

    mo.output.append(partition_search)
    return filtered_meta, partition_search


@app.cell
def _(mo, partition_search):
    selected_partitions = set(partition_search.value)

    mo.stop(
        not selected_partitions,
        mo.md("⚠️ No partitions selected.")
    )
    return (selected_partitions,)


@app.cell
def _(mo):
    button = mo.ui.run_button(label="Generate Report")
    mo.output.append(button)
    return (button,)


@app.cell
def _(
    Path,
    button,
    datetime,
    end_date,
    filtered_meta,
    mo,
    pd,
    selected_partitions,
    start_date,
):
    # get all appropriate folders
    from tqdm import tqdm

    def load_single_file(file_dir: Path, min_date: datetime, max_date: datetime, partitions: set[str]):
        return pd.read_parquet(
            file_dir,
            engine='pyarrow',
            filters=[
                ('submit', '>=', pd.Timestamp(min_date)),
                ('submit', '<=', pd.Timestamp(max_date)),
                ('partition', "in", partitions)
            ]
        )

    def load_all_files(data_filenames: list[str], min_date: datetime, max_date: datetime, partitions: set[str]):
        min_formatted = min_date.strftime("%Y%m")
        max_formatted = max_date.strftime("%Y%m")

        dfs = []
        for file in tqdm(data_filenames, desc="Loading parquet files"):
            df = load_single_file(Path(file), min_date, max_date, partitions)
            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)
        return final_df

    mo.stop(
        not button.value,
        mo.md("Click \"Generate Report\" Button to get Report.")
    )

    file_names = filtered_meta["data_path"].tolist()

    df = load_all_files(file_names, start_date.value, end_date.value, selected_partitions)

    # sanity check
    assert df.submit.min() >= pd.Timestamp(start_date.value)
    assert df.submit.max() <= pd.Timestamp(end_date.value)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(pd):
    # Load partition cpu
    partition_cpus = pd.read_csv("data/partition_cpus.csv")
    partition_gpus = pd.read_csv("data/partition_gpus.csv")
    return partition_cpus, partition_gpus


@app.cell
def _(
    df,
    end_date,
    partition_cpus,
    partition_gpus,
    pd,
    selected_partitions,
    start_date,
):
    hours_in_period = (pd.Timestamp(end_date.value) - pd.Timestamp(start_date.value)).total_seconds() / 3600

    df["cpu_hours"] = df["cpu_hours"].fillna(0)
    df["ncpus"] = df["ncpus"].fillna(0)
    df["alloctres_gpu"] = df["alloctres_gpu"].fillna(0)
    df["gpu_hours"] = df["alloctres_gpu"] * df["elapsed_seconds"] / 3600

    metrics = (
        df.groupby("partition", as_index=False)
        .agg(
            cpu_hours=("cpu_hours", "sum"),
            gpu_hours=("gpu_hours", "sum"),
        )
    )

    metrics = metrics.merge(partition_cpus, on="partition", how="left")
    metrics = metrics.merge(partition_gpus, on="partition", how="left")

    metrics["cpu_utilization"] = (
        metrics["cpu_hours"] / (metrics["total_cpus"] * hours_in_period) * 100
    )
    metrics["gpu_utilization"] = (
        metrics["gpu_hours"] / (metrics["total_gpus"] * hours_in_period) * 100
    )

    n_partitions = len(selected_partitions)
    return metrics, n_partitions


@app.cell
def _(df, end_date, metrics, mo, n_partitions, pd, px, start_date):
    ### CPU Utilization by Partition

    def cpu_visualizations():
        cpu_partition_util = px.bar(
            metrics,
            x="partition",
            y="cpu_utilization",
            title="CPU Utilization by Partition",
            labels={"cpu_utilization": "CPU Utilization", "partition": "Partition"},
        )

        cpu_partition_util.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Partition",
            yaxis_title="Utilization (%)",
            height=500,
            width=max(800, n_partitions * 40),  # ~40px per bar
        )

        cpu_partition_plot = mo.Html(f"""
        <div style="overflow-x: auto; width: 100%;">
            {mo.ui.plotly(cpu_partition_util).text}
        </div>
        """)

        # Temporal Trends
        duration = (end_date.value - start_date.value).days
        freq = "D" if duration > 14 else "h"
        fig_title = "Total CPU Used (Daily)" if freq == "D" else "Total CPU Used (Hourly)"

        hourly_cpu = (
            df.groupby(pd.Grouper(key="start", freq=freq))["ncpus"]
            .sum()
            .reset_index()
        )

        fig_hourly = px.bar(
            hourly_cpu,
            x="start",
            y="ncpus",
            title=fig_title,
            labels={"start": "Time", "cpu_hours": "CPU Hours"},
        )

        fig_hourly.update_layout(template="plotly_white")
        hourly_plot = mo.ui.plotly(fig_hourly)

        return mo.vstack([
            cpu_partition_plot,
            hourly_plot,
        ])

    return (cpu_visualizations,)


@app.cell
def _():

    # ### GPU Utilization by Partition
    # def gpu_utilization():
    #     gpu_metrics = metrics[
    #         (metrics["gpu_utilization"].notna()) & (metrics["gpu_utilization"] > 0)
    #     ]

    #     gpu_partition_util = px.bar(
    #         gpu_metrics,
    #         x="partition",
    #         y="gpu_utilization",
    #         title="GPU Utilization by Partition",
    #         labels={"gpu_utilization": "GPU Utilization", "partition": "Partition"},
    #     )

    #     gpu_partition_util.update_layout(
    #         xaxis_tickangle=-45,
    #         xaxis_title="Partition",
    #         yaxis_title="Utilization (%)",
    #         height=500,
    #         width=max(800, n_partitions * 40),  # ~40px per bar
    #     )

    #     return mo.Html(f"""
    #     <div style="overflow-x: auto; width: 100%;">
    #         {mo.ui.plotly(gpu_partition_util).text}
    #     </div>
    #     """)
    return


@app.cell
def _():
    # def get_consistent_color_map(df, column):
    #     unique_labels = sorted(df[column].unique())
    #     palette = px.colors.qualitative.Plotly 
    #     return {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    # def gpu_piechart(gpu_df, color_map, order, target_column):
    #     # Aggregate GPU count by type
    #     gpu_usage = (
    #         gpu_df.groupby(target_column)["alloctres_gpu"]
    #         .sum()
    #         .reset_index()
    #         .sort_values("alloctres_gpu", ascending=False)
    #     )

    #     fig_gpu = px.pie(
    #         gpu_usage,
    #         values="alloctres_gpu",
    #         names=target_column,
    #         title="GPU Usage Distribution by Requested Type (Excluding Unspecified)",
    #         hole=0.4,
    #         color=target_column,
    #         color_discrete_map=color_map,
    #     )

    #     fig_gpu.update_layout(margin=dict(t=50, b=20, l=20, r=20))
    #     return mo.ui.plotly(fig_gpu)

    # def gpu_by_time(gpu_df, color_map, target_column):
    #     gpu_df["day"] = gpu_df["submit"].dt.floor("D")

    #     gpu_time = (
    #         gpu_df.groupby(["day", target_column])["alloctres_gpu"]
    #         .sum()
    #         .reset_index()
    #     )

    #     order = gpu_time.sort_values(target_column, ascending=False)[target_column].tolist()

    #     fig = px.area(
    #         gpu_time,
    #         x="day",
    #         y="alloctres_gpu",
    #         title="Daily GPU Utilization by Type",
    #         color=target_column,
    #         color_discrete_map=color_map,
    #         category_orders={target_column: order}
    #     )

    #     fig.update_layout(
    #         xaxis_title="Day",
    #         yaxis_title="Total GPUs Requested",
    #         margin=dict(t=50, b=20, l=20, r=20)
    #     )

    #     return mo.ui.plotly(fig), order

    # gpu_df = df[df["alloctres_gpu"] > 0].copy()
    # gpu_df = gpu_df[gpu_df.alloctres_gpu_type != "unspecified"]
    # gpu_df["allocated_gpu"] = gpu_df['alloctres_gpu_type'].astype(str).str.strip().str.lower()
    # gpu_df["allocated_gpu_MIG_bin"] = gpu_df['alloctres_gpu_type'].str.replace(
    #     r"(\w+)_(\d+g\.\d+gb)", 
    #     r"\1_MIG", 
    #     regex=True
    # )
    return


@app.cell
def _():
    # checkbox = mo.ui.checkbox(label="Bucket Multi-Instance GPUs (MIG)")
    return


@app.cell
def _():
    # mo.output.append(mo.hstack([checkbox, mo.md(f"Bucketting MIGs: {checkbox.value}")]))

    # if len(gpu_df) == 0:
    #     mo.output.append(mo.md("No Specified GPUs Found"))

    # if len(gpu_df) > 0:
    #     gpu_column = "allocated_gpu_MIG_bin" if checkbox.value else "allocated_gpu" 

    #     color_map = get_consistent_color_map(gpu_df, gpu_column)
    #     gpu_across_time, order = gpu_by_time(gpu_df, color_map, gpu_column)
    #     gpu_pie = gpu_piechart(gpu_df, color_map, order, gpu_column)

    #     mo.output.append(gpu_pie)
    #     mo.output.append(gpu_across_time)
    return


@app.cell
def _(mo, px):
    def prepare_gpu_df(df):
        gpu_df = df[df["alloctres_gpu"] > 0].copy()
        gpu_df = gpu_df[gpu_df["alloctres_gpu_type"] != "unspecified"]

        gpu_df["allocated_gpu"] = (
            gpu_df["alloctres_gpu_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        gpu_df["allocated_gpu_MIG_bin"] = gpu_df["alloctres_gpu_type"].str.replace(
            r"(\w+)_(\d+g\.\d+gb)",
            r"\1_MIG",
            regex=True
        )

        return gpu_df


    checkbox = mo.ui.checkbox(label="Bucket Multi-Instance GPUs (MIG)")


    def gpu_partition_chart(metrics, n_partitions):
        gpu_metrics = metrics[
            (metrics["gpu_utilization"].notna()) &
            (metrics["gpu_utilization"] > 0)
        ]

        fig = px.bar(
            gpu_metrics,
            x="partition",
            y="gpu_utilization",
            title="GPU Utilization by Partition",
            labels={
                "gpu_utilization": "GPU Utilization",
                "partition": "Partition"
            },
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Partition",
            yaxis_title="Utilization (%)",
            height=500,
            width=max(800, n_partitions * 40),
        )

        return mo.Html(f"""
        <div style="overflow-x: auto; width: 100%;">
            {mo.ui.plotly(fig).text}
        </div>
        """)

    def get_color_map(df, column):
        unique_labels = sorted(df[column].unique())
        palette = px.colors.qualitative.Plotly

        return {
            label: palette[i % len(palette)]
            for i, label in enumerate(unique_labels)
        }

    return checkbox, get_color_map, gpu_partition_chart, prepare_gpu_df


@app.cell
def _(mo, px):
    def gpu_time_series(gpu_df, column, color_map):
        gpu_df = gpu_df.copy()
        gpu_df["day"] = gpu_df["submit"].dt.floor("D")

        gpu_time = (
            gpu_df.groupby(["day", column])["alloctres_gpu"]
            .sum()
            .reset_index()
        )

        fig = px.area(
            gpu_time,
            x="day",
            y="alloctres_gpu",
            title="Daily GPU Utilization by Type",
            color=column,
            color_discrete_map=color_map,
        )

        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Total GPUs Requested",
            margin=dict(t=50, b=20, l=20, r=20),
        )

        return mo.ui.plotly(fig)

    def gpu_pie_chart(gpu_df, column, color_map):
        gpu_usage = (
            gpu_df.groupby(column)["alloctres_gpu"]
            .sum()
            .reset_index()
            .sort_values("alloctres_gpu", ascending=False)
        )

        fig = px.pie(
            gpu_usage,
            values="alloctres_gpu",
            names=column,
            title="GPU Usage Distribution",
            hole=0.4,
            color=column,
            color_discrete_map=color_map,
        )

        fig.update_layout(margin=dict(t=50, b=20, l=20, r=20))

        return mo.ui.plotly(fig)

    return gpu_pie_chart, gpu_time_series


@app.cell
def _(
    checkbox,
    cpu_visualizations,
    df,
    get_color_map,
    gpu_partition_chart,
    gpu_pie_chart,
    gpu_time_series,
    metrics,
    mo,
    n_partitions,
    prepare_gpu_df,
):
    gpu_df = prepare_gpu_df(df)
    gpu_column = (
        "allocated_gpu_MIG_bin"
        if checkbox.value
        else "allocated_gpu"
    )
    color_map = get_color_map(gpu_df, gpu_column)

    gpu_visualizations = mo.vstack([
        checkbox,
        gpu_partition_chart(metrics, n_partitions),
        gpu_time_series(gpu_df, gpu_column, color_map),
        gpu_pie_chart(gpu_df, gpu_column, color_map),
    ])

    tabs = mo.ui.tabs({
        "CPU": cpu_visualizations(),
        "GPU": gpu_visualizations,
    })
    return (tabs,)


@app.cell
def _(tabs):
    tabs
    return


if __name__ == "__main__":
    app.run()
