import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def get_parquet_metadata(data_dir: Path):
    """Get metadata information for parquet file in data_dir

    Args:
        data_dir (str): Target parquet file path
    """
    df = pd.read_parquet(data_dir)

    # Get unique partitions
    partitions = "|".join(sorted(df.partition.unique()))
    

    # Get earliest and latest submit
    earliest_submit =  df.submit.min()
    latest_submit =  df.submit.max()

    return [str(data_dir), partitions, earliest_submit, latest_submit]

def get_metadata_summary(data_folder: Path):
    meta_data = []
    parquet_files = list(data_folder.glob("*.parquet"))
    for file in tqdm(parquet_files, desc="Parsing Parquet Files"):
        file_meta = get_parquet_metadata(Path(file))
        meta_data.append(file_meta)
    
    metadata_df = pd.DataFrame(meta_data, columns=['data_path', 'partitions', 'earliest_submit', 'latest_submit'])
    return metadata_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    df = get_metadata_summary(Path(args.input))
    df.to_parquet(args.output)

    # Example usage: python metadata_extracter.py --input=/orcd/data/orcd/022/util_viz/data/parsed_monthly --output=/orcd/data/orcd/022/util_viz/data/parsed_monthly/metadata.parquet







