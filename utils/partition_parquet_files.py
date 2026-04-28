import sys
import os
import glob
import pandas as pd

def main():
    if len(sys.argv) != 4:
        print("Usage: python partition_parquet_files.py <input_dir> <output_base_dir> <partition_name>")
        print("Example: python partition_parquet_files.py data/parsed_monthly data/pi_partitions pi_mghassem")
        sys.exit(1)

    source_dir = sys.argv[1]
    output_base_dir = sys.argv[2]
    partition_name = sys.argv[3]
    output_dir = os.path.join(output_base_dir, partition_name)

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing parquet files for partition: {partition_name}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}\n")

    parquet_files = sorted(glob.glob(os.path.join(source_dir, "*-sacct.parquet")))

    if not parquet_files:
        print(f"No parquet files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files to process\n")

    processed_count = 0
    skipped_count = 0

    for input_file in parquet_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        try:
            df = pd.read_parquet(input_file)
            filtered_df = df[df['partition'] == partition_name]

            if not filtered_df.empty:
                filtered_df.to_parquet(output_file, index=False)
                print(f"✓ {filename}: {len(filtered_df):,} rows written")
                processed_count += 1
            else:
                print(f"⊘ {filename}: no matching rows (skipped)")
                skipped_count += 1

        except Exception as e:
            print(f"✗ {filename}: error - {e}")
            skipped_count += 1

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Files processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
