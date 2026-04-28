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

    return Path, colorsys, datetime, hashlib, mo, os, pd, px


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
    return DATA_DIR, DATA_TYPE, DATE_STRUCTURE, gpu_colors


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
        earliest_submit = pd.read_parquet(DATA_DIR / earliest_month).submit.min()
        latest_submit = pd.read_parquet(DATA_DIR / latest_month).submit.max()

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

    return end_date, file_names, start_date


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
        df = pd.read_parquet(file_dir)
        mask = (df['submit'] >=  pd.Timestamp(min_date)) & (df['submit'] <= pd.Timestamp(max_date))
        return df[mask]

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
def _(pd):
    # Load partition cpu
    partition_cpus = pd.read_csv("../data/partition_cpus.csv")
    return (partition_cpus,)


@app.cell
def _(df, mo):
    # Get all the partitions
    partition_names = df["partition"].unique()
    partition_names = [name for name in partition_names if "," not in name] # remove multiple partitions

    partition_search = mo.ui.multiselect(
        options=sorted(partition_names),
        label=f"Partitions ({len(partition_names)} found)",
    )


    controls = mo.vstack([
        mo.md("## Filters"),
        partition_search,
    ])

    mo.output.append(partition_search)
    return (partition_search,)


@app.cell
def _(mo, partition_search):
    selected_partitions = partition_search.value

    mo.stop(
        not selected_partitions,
        mo.md("⚠️ No partitions selected.")
    )
    return


@app.cell
def _(df, end_date, mo, partition_cpus, pd, px, start_date):
    hours_in_period = (pd.Timestamp(end_date.value) - pd.Timestamp(start_date.value)).total_seconds() / 3600
    merged = df.merge(partition_cpus, on='partition', how='inner')
    utilization = merged.groupby('partition').apply(
        lambda g: g['cpu_hours'].sum() / (g['total_cpus'].iloc[0] * hours_in_period) * 100 
    ).reset_index()
    n_partitions = len(utilization)
    utilization.columns = ['partition', 'cpu_utilization']

    fig2 = px.bar(
        utilization,
        x="partition",
        y="cpu_utilization",
        title="CPU Utilization by Partition",
        labels={"cpu_utilization": "CPU Utilization", "partition": "Partition"},
    )

    fig2.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Partition",
        yaxis_title="Utilization (%)",
        height=500,
        width=max(800, n_partitions * 40),  # ~40px per bar
    )

    mo.Html(f"""
    <div style="overflow-x: auto; width: 100%;">
        {mo.ui.plotly(fig2).text}
    </div>
    """)
    return


@app.cell
def _(df, mo, px):
    # Group by hour to see temporal trends
    hourly_cpu = (
        df.set_index("start")
        .resample("h")["ncpus"]
        .sum()
        .reset_index()
    )

    fig_hourly = px.bar(
        hourly_cpu,
        x="start",
        y="ncpus",
        title="Total CPU Used (Hourly)",
        labels={"start": "Time", "cpu_hours": "CPU Hours"},
    )

    fig_hourly.update_layout(template="plotly_white")
    mo.ui.plotly(fig_hourly)
    return


@app.cell
def _(colorsys, df, fig, gpu_colors, hashlib, mo, px):
    def gpu_piechart(gpu_df, color_map, order):
        # Aggregate GPU count by type
        gpu_usage = (
            gpu_df.groupby("allocated_gpu")["alloctres_gpu"]
            .sum()
            .reset_index()
            .sort_values("alloctres_gpu", ascending=False)
        )
    
        fig_gpu = px.pie(
            gpu_usage,
            values="alloctres_gpu",
            names="allocated_gpu",
            title="GPU Usage Distribution by Requested Type (Excluding Unspecified)",
            hole=0.4,
            color="allocated_gpu",
            # color_discrete_map=color_map,
            # category_orders={
            #     "allocated_gpu": order
            # }
        )
    
        fig_gpu.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        print(fig.layout.template.layout.colorway)
        return mo.ui.plotly(fig_gpu)

    def gpu_by_time(gpu_df, color_map):
        gpu_df["day"] = gpu_df["submit"].dt.floor("D")
    
        gpu_time = (
            gpu_df.groupby(["day", "allocated_gpu"])["alloctres_gpu"]
            .sum()
            .reset_index()
        )

        order = gpu_time.sort_values("allocated_gpu", ascending=False)["allocated_gpu"].tolist()
    
        fig = px.area(
            gpu_time,
            x="day",
            y="alloctres_gpu",
            title="Daily GPU Utilization by Type",
            color="allocated_gpu",
            color_discrete_map=color_map,
            category_orders={"allocated_gpu": order}
        )
    
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Total GPUs Requested",
            margin=dict(t=50, b=20, l=20, r=20)
        )
    
        return mo.ui.plotly(fig), order

    gpu_df = df[df["alloctres_gpu"] > 0].copy()
    gpu_df = gpu_df[gpu_df.alloctres_gpu_type != "unspecified"]
    if len(gpu_df) > 0:
        gpu_df["allocated_gpu"] = gpu_df['alloctres_gpu_type'].str.split("_").str[0]
        gpu_df["allocated_gpu"] = gpu_df["allocated_gpu"].astype(str).str.strip().str.lower()
        unique_gpu = gpu_df.allocated_gpu.unique()

        all_gpu_colors = gpu_colors.copy()
        for gpu in unique_gpu:
            if gpu not in all_gpu_colors:
                # generate random color
                h = int(hashlib.md5(gpu.encode()).hexdigest(), 16)
                hue = (h % 360) / 360.0
                sat = 0.6
                val = 0.85
        
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                all_gpu_colors[gpu] = color

        filtered_color_map = {
            gpu: color
            for gpu, color in all_gpu_colors.items()
            if gpu in gpu_df["allocated_gpu"].unique()
        }

        gpu_across_time, order = gpu_by_time(gpu_df, filtered_color_map)
        mo.output.append(gpu_piechart(gpu_df, filtered_color_map, order))
        mo.output.append(gpu_across_time)


    
                    


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
