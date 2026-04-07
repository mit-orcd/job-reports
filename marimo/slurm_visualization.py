import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Import Libraries
    import pandas as pd
    import marimo as mo
    import os 
    import glob
    from pathlib import Path
    import plotly.express as px
    import re

    return Path, mo, os, pd, px


@app.cell
def _(mo):
    mo.md("""
    # ORCD Job Report
    """)
    return


@app.cell
def _(Path, mo):
    # Choose Data File
    files = list(Path("../data").glob("*.out"))
    file_names = [f.name for f in files]
    file_dropdown = mo.ui.dropdown(options=file_names, label="Choose Data File:")
    return file_dropdown, file_names, files


@app.cell
def _(file_dropdown, mo):
    mo.output.append(mo.md("## Select or Upload Data File"))
    file_selection = mo.hstack([file_dropdown, mo.md(f"File Chosen: {file_dropdown.value}")])
    mo.output.append(file_selection)
    return


@app.cell
def _(file_dropdown, mo, os):
    # Continue only if file is selected
    mo.stop(file_dropdown.value is None)

    # Check whether parsed data file is available for this file
    parsed_filename = f"{file_dropdown.value.rsplit('.',1)[0]}_parsed.parquet"
    if os.path.exists(f"../data/parsed_data/{parsed_filename}"):
        print(f"Found Parsed File: {parsed_filename}")
        use_parsed = mo.ui.radio(
            options=["Use existing parsed file", "Regenerate"],
            value="Use existing parsed file",
            label=f"**{parsed_filename}** already exists. What would you like to do?"
        )
    else:
        use_parsed = None
    return parsed_filename, use_parsed


@app.cell
def _(file_dropdown, mo, use_parsed):
    if use_parsed is not None:
        mo.output.append(mo.hstack([use_parsed, mo.md(f"Has value: {use_parsed.value}")]))
    else:
        mo.output.append(f"Did not find parsed file for {file_dropdown.value}")
    return


@app.cell
def _(pd):
    # File Processing Helper Functions
    def elapsed_to_seconds(s):
        if pd.isna(s): return pd.NA
        parts = str(s).split(":")
        try:
            return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        except:
            return pd.NA

    def parse_tres(tres_str):
        if not tres_str or tres_str.strip() == "":
            return {}
        result = {}
        for item in tres_str.split(","):
            key, _, value = item.partition("=")
            result[key] = value
        return result

    def extract_tres_fields(df, col):
        """Parse a TRES column and return a DataFrame of extracted fields."""
        parsed = df[col].fillna("").apply(parse_tres)

        out = pd.DataFrame()
        out[f"{col}_cpu"] = parsed.apply(lambda x: int(x.get("cpu", 0)))
        out[f"{col}_mem_mb"] = parsed.apply(lambda x: parse_mem(x.get("mem", "0")))
        out[f"{col}_nodes"] = parsed.apply(lambda x: int(x.get("node", 0)))
        out[f"{col}_gpu"] = parsed.apply(lambda x: int(x.get("gres/gpu", 0)))
        out[f"{col}_gpu_type"] = parsed.apply(lambda x: parse_gpu_type(x))

        return out

    def parse_mem(mem_str):
        """Convert mem string to MB as float."""
        if not mem_str or mem_str == "0":
            return 0.0
        if mem_str.endswith("M"):
            return float(mem_str[:-1])
        elif mem_str.endswith("G"):
            return float(mem_str[:-1]) * 1024
        elif mem_str.endswith("T"):
            return float(mem_str[:-1]) * 1024 * 1024
        return float(mem_str)

    def parse_gpu_type(parsed_dict):
        """Extract GPU type from keys like 'gres/gpu:a100'."""
        for key in parsed_dict:
            if key.startswith("gres/gpu:"):
                return key.split(":")[1]
        return None

    # def parse_nodelist(nodelist):
    #     """Parse SLURM nodelist like node[397-398] or node[1406,1414,1447-1452] into seperate rows"""
    #     if pd.isna(nodelist) or nodelist == '-':
    #         return [nodelist]

    #     match = re.match(r'(\D+)\[(.+)\]', nodelist)
    #     if not match:
    #         return [nodelist]  # single node like "node397"

    #     prefix, ranges = match.group(1), match.group(2)
    #     nodes = []

    #     for part in ranges.split(','):
    #         if '-' in part:
    #             start, end = part.split('-')
    #             nodes.extend(f"{prefix}{i}" for i in range(int(start), int(end) + 1))
    #         else:
    #             nodes.append(f"{prefix}{part}")

    #     return nodes
    return elapsed_to_seconds, extract_tres_fields


@app.cell
def _(
    elapsed_to_seconds,
    extract_tres_fields,
    file_dropdown,
    file_names,
    files,
    mo,
    parsed_filename,
    pd,
    use_parsed,
):
    if use_parsed is None or use_parsed.value == "Regenerate":
        print("Generating Parsed File...")
        # Load and Process .out file
        data_file = files[file_names.index(file_dropdown.value)]
        columns = [
            "jobid","jobidraw","cluster","partition","qos","account","group","gid",
            "user","uid","submit","eligible","start","end","elapsed","exitcode",
            "state","nnodes","ncpus","reqcpus","reqmem","reqtres","alloctres",
            "timelimit","nodelist","jobname"
        ]

        df = pd.read_csv(data_file, sep="|", header=None, names=columns) # Load .out file

        ### Update column values ###
        df.replace(["None", "Unknown", "N/A", ""], pd.NA, inplace=True) # Replace as Null 
        for col in ["submit", "eligible", "start", "end"]: # Turn time columns into datatime
            df[col] = pd.to_datetime(df[col], errors="coerce")

        ### Add useful columns ###
        df["elapsed_seconds"] = df["elapsed"].apply(elapsed_to_seconds)
        # Extract just the state without "CANCELLED by XXXX"
        df["state_clean"] = df["state"].str.extract(r"^(\w+)") 
        df["wait_time"] = df["start"] - df["eligible"]
        df["cpu_hours"] = df["ncpus"] * df["elapsed_seconds"] / 60 / 60

        ### Remove jobs that are canceled and start time is none
        df = df[~((df["state_clean"] == "CANCELLED") & (df["start"].isna()))]

        ### Parse Node List
        # df['node'] = df['nodelist'].apply(parse_nodelist)
        # df = df.explode('node').reset_index(drop=True)

        ### Parse out reqtres and alloctres ###
        req_fields = extract_tres_fields(df, "reqtres")
        alloc_fields = extract_tres_fields(df, "alloctres")
        df = pd.concat([df, req_fields, alloc_fields], axis=1)
        df['alloctres_gpu_type'] = df['alloctres_gpu_type'].fillna('unspecified')
        df['reqtres_gpu_type'] = df['reqtres_gpu_type'].fillna('unspecified')

        ### Add number of physical cores
        # Load physical cores per node
        # node_cores = pd.read_csv("../data/scontrol_nodes_parsed.csv")
        # df = df.merge(node_cores, left_on="node", right_on="NodeName")

        ### Save Data File in Parsed Folder ###
        df.to_parquet(f"../data/parsed_data/{parsed_filename}", index=False)
        print(f"Saved Data to {parsed_filename}")

    else:
        df = pd.read_parquet(f"../data/parsed_data/{parsed_filename}")
        mo.output.append(df)
        print(f"Successfully Loaded Data")
    return (df,)


@app.cell
def _(pd):
    # Load partition cpu
    partition_cpus = pd.read_csv("../data/partition_cpus.csv")
    return (partition_cpus,)


@app.cell
def _(df):
    # Get time frame and partition(s) to include
    earliest_submit_time = df["submit"].min()
    latest_end_time = df['end'].max()

    # Get all the partitions
    partition_names = df["partition"].unique()
    partition_names = [name for name in partition_names if "," not in name] # remove multiple partitions
    return earliest_submit_time, latest_end_time, partition_names


@app.cell
def _(earliest_submit_time, latest_end_time, mo, partition_names):
    # Time Range
    start_date = mo.ui.date(
        value=earliest_submit_time.date(),
        label="Start"
    )
    end_date = mo.ui.date(
        value=latest_end_time.date(),
        label="End"
    )

    partition_search = mo.ui.multiselect(
        options=sorted(partition_names),
        label=f"Partitions ({len(partition_names)} found)",
    )


    controls = mo.vstack([
        mo.md("## Filters"),
        mo.md(f"**Time range in data (first submit, last end):** `{earliest_submit_time}` → `{latest_end_time}`"),
        start_date,
        end_date,
        mo.md("---"),
        partition_search,
    ])

    controls
    return end_date, partition_search, start_date


@app.cell
def _(df, end_date, mo, partition_search, pd, start_date):
    selected_partitions = partition_search.value

    mo.stop(
        not selected_partitions,
        mo.md("⚠️ No partitions selected.")
    )

    filtered_df = df[
        (df["submit"] >= pd.Timestamp(start_date.value)) &
        (df["end"] <= pd.Timestamp(end_date.value)) &
        (df["partition"].isin(selected_partitions))
    ]
    return (filtered_df,)


@app.cell
def _():
    # filtered_df["wait_time_hours"] = filtered_df["wait_time"].dt.total_seconds() / 3600

    # avg_wait = (
    #     filtered_df.groupby("partition")["wait_time_hours"]
    #     .mean()
    #     .reset_index()
    #     .sort_values("partition")
    # )

    # n_partitions = len(avg_wait)

    # fig = px.bar(
    #     avg_wait,
    #     x="partition",
    #     y="wait_time_hours",
    #     title="Average Wait Time by Partition",
    #     labels={"wait_time_hours": "Avg Wait Time (hours)", "partition": "Partition"},
    # )

    # fig.update_layout(
    #     xaxis_tickangle=-45,
    #     xaxis_title="Partition",
    #     yaxis_title="Avg Wait Time (hours)",
    #     height=500,
    #     width=max(800, n_partitions * 40),  # ~40px per bar
    # )

    # mo.Html(f"""
    # <div style="overflow-x: auto; width: 100%;">
    #     {mo.ui.plotly(fig).text}
    # </div>
    # """)
    return


@app.cell
def _(end_date, filtered_df, mo, partition_cpus, pd, px, start_date):
    hours_in_period = (pd.Timestamp(end_date.value) - pd.Timestamp(start_date.value)).total_seconds() / 3600
    merged = filtered_df.merge(partition_cpus, on='partition', how='inner')
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
def _(filtered_df, mo, px):
    # Group by hour to see temporal trends
    hourly_cpu = (
        filtered_df.set_index("start")
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
def _(filtered_df):
    filtered_df
    return


@app.cell
def _(filtered_df, mo, px):
    # Create stacked area chart for 
    gpu_df = filtered_df[filtered_df["alloctres_gpu"] > 0].copy()
    gpu_df = gpu_df[gpu_df.reqtres_gpu_type != "unspecified"]

    # Aggregate GPU count by type
    gpu_usage = (
        gpu_df.groupby("reqtres_gpu_type")["alloctres_gpu"]
        .sum()
        .reset_index()
        .sort_values("alloctres_gpu", ascending=False)
    )

    fig_gpu = px.pie(
        gpu_usage,
        values="alloctres_gpu",
        names="reqtres_gpu_type",
        title="GPU Usage Distribution by Requested Type (Excluding Unspecified)",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_gpu.update_layout(margin=dict(t=50, b=20, l=20, r=20))
    mo.ui.plotly(fig_gpu)
    return (gpu_df,)


@app.cell
def _(gpu_df, mo, px):
    # Filter for jobs that actually requested GPUs
    gpu_df["day"] = gpu_df["submit"].dt.floor("D")

    gpu_time = (
        gpu_df.groupby(["day", "reqtres_gpu_type"])["alloctres_gpu"]
        .sum()
        .reset_index()
    )

    fig = px.area(
        gpu_time,
        x="day",
        y="alloctres_gpu",
        color="reqtres_gpu_type",
        title="Daily GPU Utilization by Type",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Total GPUs Requested",
        margin=dict(t=50, b=20, l=20, r=20)
    )

    mo.ui.plotly(fig)
    return


if __name__ == "__main__":
    app.run()
