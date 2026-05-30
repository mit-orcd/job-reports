#!/bin/bash


#SBATCH --job-name=marimo_data
#SBATCH --mem=16G
#SBATCH --time=02:00:00          
#SBATCH --partition=mit_normal


# Usage:
# ./parse_scheduler.sh <input_dir> <output_dir> [log_file] [error_file]
# ./parse_scheduler.sh /orcd/data/orcd/022/util_viz/data /orcd/data/orcd/022/util_viz/data/parsed_monthly /orcd/data/orcd/022/util_viz/data/parsed_monthly/processed_files.log  /orcd/data/orcd/022/util_viz/data/parsed_monthly/processing_error.log

module load miniforge
source ../.venv/bin/activate


INPUT_DIR="$1"
OUTPUT_DIR="$2"
LOG_FILE="${3:-processed_files.log}"
ERROR_FILE="${4:-processing_error.log}"

# Check args
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <input_dir> <output_dir> [log_file]"
  exit 1
fi

# Create output dir if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create log file if it doesn't exist
if [ ! -f "$LOG_FILE" ]; then
  touch "$LOG_FILE"
fi


if [ ! -f "$ERROR_FILE" ]; then
  touch "$ERROR_FILE"
fi

echo "Using log file: $LOG_FILE"
echo "Updating error file: $ERROR_FILE"


# Loop through files in input directory
for FILE in "$INPUT_DIR"/*.out; do
  # Skip if not a regular file
  [ -f "$FILE" ] || continue

  FILENAME=$(basename "$FILE")

  # Check if file already processed
  if grep -Fxq "$FILENAME" "$LOG_FILE"; then
    echo "Skipping already processed file: $FILENAME"
  else
    echo "Processing: $FILENAME"

    # Run parser (adjust args if needed)

    python data_parser.py --input "$FILE" --output "$OUTPUT_DIR" >> "$ERROR_FILE" 2>&1

    # Check if parsing succeeded
    if [ $? -eq 0 ]; then
      echo "$FILENAME" >> "$LOG_FILE"
      echo "Logged: $FILENAME"
    else
      echo "Error processing: $FILENAME" | tee -a "$ERROR_FILE"
    fi
  fi
done
