#!/usr/bin/env python3
"""
Generate factor.day.bin files for US market data from CSV files.

Factor is calculated as: factor = adjclose / close

Usage:
    python generate_us_factor_files.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import struct
from tqdm import tqdm

# Paths
CSV_DIR = Path.home() / ".qlib" / "sh_sp500_csv"
QLIB_DIR = Path.home() / ".qlib" / "sh_sp500_qlib" / "features"

def read_qlib_binary(file_path):
    """Read Qlib binary file and return list of float values"""
    with open(file_path, 'rb') as f:
        data = f.read()

    # Qlib binary format: continuous float32 values
    # First value is usually 0.0 (header), followed by actual data
    n_values = len(data) // 4
    values = []

    for i in range(n_values):
        offset = i * 4
        value = struct.unpack('<f', data[offset:offset+4])[0]
        values.append(value)

    return values

def write_qlib_binary(file_path, values):
    """Write values to Qlib binary format"""
    with open(file_path, 'wb') as f:
        for value in values:
            f.write(struct.pack('<f', value))

def generate_factor_for_symbol(symbol_lower):
    """Generate factor.day.bin for a single symbol"""
    symbol_upper = symbol_lower.upper()

    # Read CSV file
    csv_file = CSV_DIR / f"{symbol_upper}.csv"
    if not csv_file.exists():
        print(f"⚠️  CSV not found: {csv_file}")
        return False

    # Read existing close.day.bin to get the length
    close_file = QLIB_DIR / symbol_lower / "close.day.bin"
    if not close_file.exists():
        print(f"⚠️  close.day.bin not found: {close_file}")
        return False

    # Read close binary values
    close_values = read_qlib_binary(close_file)

    # Read CSV to get adjclose and close
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Build date to adjclose and close mapping from CSV
    date_to_adjclose = dict(zip(df['date'], df['adjclose']))
    date_to_close_csv = dict(zip(df['date'], df['close']))

    # Read calendar file
    calendar_file = Path.home() / ".qlib" / "sh_sp500_qlib" / "calendars" / "day.txt"
    if not calendar_file.exists():
        print(f"⚠️  Calendar not found: {calendar_file}")
        return False

    with open(calendar_file, 'r') as f:
        calendar_dates = [pd.to_datetime(line.strip()) for line in f if line.strip()]

    # Generate factor values (same length as close_values)
    factor_values = []

    for i in range(len(close_values)):
        # First value is header (0.0)
        if i == 0:
            factor_values.append(0.0)
            continue

        # Get corresponding date from calendar
        # i-1 because first value is header
        date_idx = i - 1
        if date_idx >= len(calendar_dates):
            # Pad with nan if beyond calendar
            factor_values.append(np.nan)
            continue

        date = calendar_dates[date_idx]

        # Get adjclose and close from CSV
        adjclose = date_to_adjclose.get(date)
        close_csv = date_to_close_csv.get(date)

        # Calculate factor = adjclose / close
        if adjclose is not None and close_csv is not None and close_csv != 0:
            if not np.isnan(adjclose) and not np.isnan(close_csv):
                factor_value = adjclose / close_csv
            else:
                factor_value = np.nan
        else:
            # If no CSV data, use nan
            factor_value = np.nan

        factor_values.append(factor_value)

    # Apply forward fill to factor values (like Yahoo collector does)
    # Convert to pandas Series, apply ffill, then back to list
    factor_series = pd.Series(factor_values)
    factor_series = factor_series.ffill()  # Forward fill NaN values
    factor_values = factor_series.tolist()

    # Write factor.day.bin
    factor_file = QLIB_DIR / symbol_lower / "factor.day.bin"
    factor_file.parent.mkdir(parents=True, exist_ok=True)
    write_qlib_binary(factor_file, factor_values)

    return True

def main():
    print("=" * 80)
    print("Generating factor.day.bin files for US market")
    print("=" * 80)
    print(f"CSV directory: {CSV_DIR}")
    print(f"Qlib directory: {QLIB_DIR}")
    print()

    # Get all symbol directories in Qlib features
    symbol_dirs = [d for d in QLIB_DIR.iterdir() if d.is_dir()]

    print(f"Found {len(symbol_dirs)} symbols in Qlib features directory")
    print()

    success_count = 0
    failed_count = 0

    for symbol_dir in tqdm(symbol_dirs, desc="Processing symbols"):
        symbol_lower = symbol_dir.name

        # Skip non-symbol directories (like benchmarks starting with ^)
        if symbol_lower.startswith('^'):
            continue

        try:
            if generate_factor_for_symbol(symbol_lower):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"❌ Error processing {symbol_lower}: {e}")
            failed_count += 1

    print()
    print("=" * 80)
    print(f"✅ Successfully generated factor files: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
