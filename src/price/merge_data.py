"""
Mandi Price & Weather Data Merger & Quality Reporter for KisanMitra AI.

Merges daily Agmarknet price series with Open-Meteo agro-weather observations
on (date, mandi), generates data/processed/mandi_weather_daily.csv,
and prints a comprehensive Data Quality Report.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR, AGMARKNET_RAW_DIR


def read_csv_records(file_path: Path) -> List[Dict[str, str]]:
    """Reads a CSV file into a list of row dictionaries."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def merge_price_and_weather(
    price_records: List[Dict[str, str]],
    weather_records: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Merges price records with weather records using (date, mandi) as key.
    """
    weather_map = {}
    for w in weather_records:
        key = (w["date"].strip(), w["mandi"].strip().lower())
        weather_map[key] = w

    merged = []
    for p in price_records:
        key = (p["date"].strip(), p["mandi"].strip().lower())
        w = weather_map.get(key, {})

        row = {
            "date": p.get("date", "").strip(),
            "mandi": p.get("mandi", "").strip(),
            "state": p.get("state", "").strip(),
            "district": p.get("district", "").strip(),
            "crop": p.get("crop", "").strip(),
            "variety": p.get("variety", "Standard").strip(),
            "modal_price_rs_qtl": p.get("modal_price_rs_qtl", "").strip(),
            "min_price_rs_qtl": p.get("min_price_rs_qtl", "").strip(),
            "max_price_rs_qtl": p.get("max_price_rs_qtl", "").strip(),
            "temp_mean_c": w.get("temp_mean_c", ""),
            "humidity_mean_pct": w.get("humidity_mean_pct", ""),
            "rainfall_mm": w.get("rainfall_mm", ""),
        }
        merged.append(row)

    # Sort deterministically by date, mandi, crop
    merged.sort(key=lambda x: (x["date"], x["mandi"], x["crop"]))
    return merged


def print_data_quality_report(records: List[Dict[str, Any]], title: str = "DATA QUALITY REPORT"):
    """
    Prints a detailed data quality assessment:
    - Date range
    - Row counts per mandi and crop
    - Missing values per column
    - 5 sample rows
    """
    print("\n" + "=" * 80)
    print(f" 📊 KISANMITRA AI — {title}")
    print("=" * 80)

    if not records:
        print("⚠️ No records found to report on.")
        print("=" * 80)
        return

    total_rows = len(records)
    columns = list(records[0].keys())

    # Date Range
    dates = [r["date"] for r in records if r["date"]]
    min_date = min(dates) if dates else "N/A"
    max_date = max(dates) if dates else "N/A"

    print(f"\n1. Overall Dataset Dimensions:")
    print(f"   • Total Rows:    {total_rows}")
    print(f"   • Total Columns: {len(columns)}")
    print(f"   • Date Range:    {min_date}  -->  {max_date}")

    # Row counts per Mandi
    mandi_counts = {}
    for r in records:
        m = r.get("mandi", "Unknown")
        mandi_counts[m] = mandi_counts.get(m, 0) + 1

    print(f"\n2. Row Distribution per APMC Mandi:")
    for mandi, cnt in sorted(mandi_counts.items()):
        print(f"   • {mandi:<15}: {cnt:>6} rows ({cnt / total_rows * 100:>5.1f}%)")

    # Row counts per Crop
    crop_counts = {}
    for r in records:
        c = r.get("crop", "Unknown")
        crop_counts[c] = crop_counts.get(c, 0) + 1

    print(f"\n3. Row Distribution per Crop:")
    for crop, cnt in sorted(crop_counts.items()):
        print(f"   • {crop:<15}: {cnt:>6} rows ({cnt / total_rows * 100:>5.1f}%)")

    # Missing values per column
    print(f"\n4. Missing Value Audit per Column:")
    print(f"   {'Column Name':<25} | {'Missing Count':<15} | {'Missing (%)':<12}")
    print("   " + "-" * 56)
    for col in columns:
        missing_count = sum(1 for r in records if str(r.get(col, "")).strip() in ("", "None", "nan", "NULL"))
        pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
        flag = "⚠️" if missing_count > 0 else "✓"
        print(f"   {col:<25} | {missing_count:>15} | {pct:>10.2f}%  {flag}")

    # 5 Sample rows
    print(f"\n5. First 5 Sample Rows:")
    print("   " + "-" * 76)
    sample_cols = ["date", "mandi", "crop", "modal_price_rs_qtl", "temp_mean_c", "humidity_mean_pct", "rainfall_mm"]
    header_str = " | ".join(f"{c:<12}" for c in sample_cols)
    print(f"   {header_str}")
    print("   " + "-" * 76)
    for r in records[:5]:
        row_str = " | ".join(f"{str(r.get(c, '')):<12}" for c in sample_cols)
        print(f"   {row_str}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Agmarknet mandi prices and Open-Meteo weather data"
    )
    parser.add_argument(
        "--price-csv",
        type=str,
        default=str(DATA_PROCESSED_DIR / "agmarknet_daily.csv"),
        help="Path to cleaned Agmarknet prices CSV",
    )
    parser.add_argument(
        "--weather-csv",
        type=str,
        default=str(DATA_PROCESSED_DIR / "weather_daily.csv"),
        help="Path to cleaned weather CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_PROCESSED_DIR / "mandi_weather_daily.csv"),
        help="Path where merged dataset will be saved",
    )
    args = parser.parse_args()

    price_path = Path(args.price_csv)
    weather_path = Path(args.weather_csv)
    output_path = Path(args.output)

    print("=" * 80)
    print("🔄 KISANMITRA AI — PRICE & WEATHER DATA MERGER")
    print("=" * 80)

    # 1. Check price data
    if not price_path.exists():
        print(f"\n❌ ERROR: Agmarknet price file not found at: {price_path}")
        print(
            "👉 Please download historical Agmarknet data or set DATA_GOV_IN_API_KEY,\n"
            f"   and run: python3 src/price/fetch_agmarknet.py"
        )
        sys.exit(1)

    # 2. Check weather data
    if not weather_path.exists():
        print(f"\nℹ️ Weather file not found at {weather_path}. Auto-fetching via Open-Meteo...")
        from src.price.fetch_weather import fetch_all_mandis_weather, save_weather_csv
        from datetime import date, timedelta
        end_d = (date.today() - timedelta(days=2)).isoformat()
        start_d = (date.today() - timedelta(days=365 * 2)).isoformat()
        w_records = fetch_all_mandis_weather(start_d, end_d)
        save_weather_csv(w_records, weather_path)

    # 3. Read both datasets
    price_records = read_csv_records(price_path)
    weather_records = read_csv_records(weather_path)

    print(f"\n• Read {len(price_records)} price records.")
    print(f">• Read {len(weather_records)} weather records.")

    # 4. Merge
    merged_records = merge_price_and_weather(price_records, weather_records)

    # 5. Save merged CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(merged_records[0].keys())
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for r in merged_records:
            f.write(",".join(str(r[c]) for c in fieldnames) + "\n")

    print(f"\n✓ Merged dataset saved to: {output_path.resolve()}")

    # 6. Print Quality Report
    print_data_quality_report(merged_records, "MANDI & WEATHER DATA QUALITY REPORT")


if __name__ == "__main__":
    main()
