import pandas as pd
from pathlib import Path
import os
import argparse

def elapsed_to_seconds(s):
    if pd.isna(s): return pd.NA
    parts = str(s).split(":")
    try:
        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    except Exception as e:
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
    out[f"{col}_gpu_type"] = parsed.apply(lambda x: parse_gpu_type(x) if isinstance(x, dict) else None)

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

def parse_file(data_file):
    columns = [
        "jobid","jobidraw","cluster","partition","qos","account","group","gid",
        "user","uid","submit","eligible","start","end","elapsed","exitcode",
        "state","nnodes","ncpus","reqcpus","reqmem","reqtres","alloctres",
        "timelimit","nodelist","jobname"
    ]

    # 59443519|59443519|eofe7|sched_any_quicktest|normal|mit_general|jaysonj|259823|
    # jaysonj|259823|2024-10-22T16:59:27|2024-10-22T16:59:27|None|2025-02-18T13:23:51|00:00:00|0:0|
    # CANCELLED by 0|1|0|1|1000M|billing=1,cpu=1,mem=1000M,node=1||10:00:00|None assigned|2DCV

    # 5841751_44|5841796|eofe7|mit_normal|normal|mit_general|newolfe|224670|
    # newolfe|224670|2025-11-19T15:40:50|2025-11-19T15:42:51|None|2026-04-07T09:22:19|00:00:00|0:0|
    # CANCELLED by 82831|1|0|1|4G|billing=1,cpu=1,mem=4G,node=1||12:00:00|None assigned||pe

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
    df["cpu_hours"] = df["ncpus"] * df["elapsed_seconds"] / 3600

    ### Remove jobs that are canceled and start time is none
    df = df[~((df["state_clean"] == "CANCELLED") & (df["start"].isna()))]

    ### Parse out reqtres and alloctres ###
    req_fields = extract_tres_fields(df, "reqtres")
    alloc_fields = extract_tres_fields(df, "alloctres")
    df = pd.concat([df, req_fields, alloc_fields], axis=1)
    df['alloctres_gpu_type'] = df['alloctres_gpu_type'].fillna('unspecified')
    df['reqtres_gpu_type'] = df['reqtres_gpu_type'].fillna('unspecified')

    return df

def parse_and_save(data_file, output_dir):
    # Parse the data file
    df = parse_file(data_file)
    

    # Seperate df into different year and month
    df["year_month"] = df["submit"].dt.strftime("%Y%m")

    # Save unique submit time into seperate files
    output_path = Path(output_dir)
    for year_month in df["year_month"].unique():
        print(f"Saving parsed file for {year_month}")
        file_path = output_path / f"{year_month}-sacct.parquet"
        final_df = df[df["year_month"] == year_month]

        if os.path.exists(file_path):
            print("Found existing parsed file. Updating...")
            existing_df = pd.read_parquet(file_path)
            updated_df = pd.concat([existing_df, final_df])
            final_df = updated_df.drop_duplicates()

        final_df.to_parquet(file_path)
        print(f"Successfully saved parsed file for {year_month}")
    
    print(f"Successfully saved parsed data into {len(df['year_month'].unique())} different data files in {output_dir}!")
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    parse_and_save(args.input, args.output)

    # Example usage: python generate_parsed.py --input=/orcd/data/orcd/022/util_viz/data/202301-sacct.out --output=/orcd/data/orcd/022/util_viz/data/temp

    
