import re
import pandas as pd
import sys

def parse_scontrol(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as f:
        text = f.read()

    # Split into per-node blocks (separated by blank lines)
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]

    rows = []
    for block in blocks:
        node = {}
        for match in re.finditer(r'(\w+)=(\S+)', block):
            node[match.group(1)] = match.group(2)
        if "NodeName" in node:
            rows.append(node)

    df = pd.DataFrame(rows)

    # Keep and cast relevant columns
    cols = ["NodeName", "CoresPerSocket", "Sockets", "CPUTot"]
    df = df[cols].copy()
    for col in ["CoresPerSocket", "Sockets", "CPUTot"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physical cores = CoresPerSocket * Sockets
    df["PhysicalCores"] = df["CoresPerSocket"] * df["Sockets"]

    # Hyper-threaded if CPUTot > PhysicalCores
    df["is_hyperthreaded"] = df["CPUTot"] > df["PhysicalCores"]

    return df


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "scontrol_nodes.txt"
    df = parse_scontrol(filepath)

    print(df.to_string(index=False))
    print(f"\nTotal nodes:           {len(df)}")
    print(f"Hyper-threaded nodes:  {df['is_hyperthreaded'].sum()}")
    print(f"Normal nodes:          {(~df['is_hyperthreaded']).sum()}")

    out_csv = filepath.replace(".txt", "_parsed.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")