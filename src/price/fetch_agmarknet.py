"""
Agmarknet Mandi Price Ingestion & Parser for KisanMitra AI.

Supports:
1. Automated querying via official data.gov.in API (if DATA_GOV_IN_API_KEY is in .env).
2. Direct ingestion and normalization of raw CSV/Excel files placed by user in data/raw/agmarknet/.
3. Clear actionable instructions when no key or raw files are detected (Strict Zero-Fake-Data policy).
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any
import urllib.parse
import urllib.request
import urllib.error

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    MANDI_CONFIGS,
    CROPS,
    AGMARKNET_RAW_DIR,
    DATA_PROCESSED_DIR,
)

# Optional loading of dotenv if installed
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "").strip()

# Official Data.gov.in Resource ID for Daily Mandi Prices
DATA_GOV_IN_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_IN_API_URL = f"https://api.data.gov.in/resource/{DATA_GOV_IN_RESOURCE_ID}"


# Column mapping dictionary for standardizing various Agmarknet export formats
COLUMN_ALIASES = {
    # Date
    "arrival_date": "date",
    "reported_date": "date",
    "date": "date",
    "arrival date": "date",
    "reported date": "date",
    # Market / Mandi
    "market": "mandi",
    "market_name": "mandi",
    "market name": "mandi",
    "mandi": "mandi",
    # State & District
    "state": "state",
    "state_name": "state",
    "state name": "state",
    "district": "district",
    "district_name": "district",
    "district name": "district",
    # Commodity / Crop
    "commodity": "crop",
    "crop": "crop",
    "commodity_name": "crop",
    "commodity name": "crop",
    # Variety
    "variety": "variety",
    # Modal Price
    "modal_price": "modal_price_rs_qtl",
    "modal_x0020_price": "modal_price_rs_qtl",
    "modal price": "modal_price_rs_qtl",
    "modal price (rs./quintal)": "modal_price_rs_qtl",
    "modal_price_(rs./quintal)": "modal_price_rs_qtl",
    # Min Price
    "min_price": "min_price_rs_qtl",
    "min_x0020_price": "min_price_rs_qtl",
    "min price": "min_price_rs_qtl",
    "min price (rs./quintal)": "min_price_rs_qtl",
    "min_price_(rs./quintal)": "min_price_rs_qtl",
    # Max Price
    "max_price": "max_price_rs_qtl",
    "max_x0020_price": "max_price_rs_qtl",
    "max price": "max_price_rs_qtl",
    "max price (rs./quintal)": "max_price_rs_qtl",
    "max_price_(rs./quintal)": "max_price_rs_qtl",
}


def normalize_date(raw_date_str: str) -> Optional[str]:
    """
    Standardizes varied date string formats (DD/MM/YYYY, YYYY-MM-DD, DD-Mon-YYYY) into YYYY-MM-DD.
    """
    if not raw_date_str:
        return None
    cleaned = str(raw_date_str).strip()
    formats_to_try = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%Y/%m/%d",
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def match_mandi(raw_mandi_str: str) -> Optional[str]:
    """
    Matches a raw mandi string from Agmarknet to our 3 configured mandis.
    """
    if not raw_mandi_str:
        return None
    cleaned = raw_mandi_str.strip().lower()
    for mandi_key, cfg in MANDI_CONFIGS.items():
        aliases = [a.lower() for a in cfg.get("market_alias", [mandi_key])]
        if any(alias in cleaned or cleaned in alias for alias in aliases):
            return mandi_key
    return None


def match_crop(raw_crop_str: str) -> Optional[str]:
    """
    Matches raw commodity string to Tomato or Potato.
    """
    if not raw_crop_str:
        return None
    cleaned = raw_crop_str.strip().lower()
    if "tomato" in cleaned:
        return "Tomato"
    if "potato" in cleaned:
        return "Potato"
    return None


def parse_price(val: Any) -> Optional[float]:
    """
    Safely converts price strings to positive float values.
    """
    if val is None:
        return None
    val_str = str(val).replace(",", "").strip()
    try:
        p = float(val_str)
        return p if p > 0 else None
    except ValueError:
        return None


def scan_local_agmarknet_files(data_dir: Path = AGMARKNET_RAW_DIR) -> List[Path]:
    """
    Finds all CSV and TSV files in the raw agmarknet directory.
    """
    if not data_dir.exists():
        return []
    files = [
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".csv", ".tsv", ".txt"] and p.name != ".gitkeep"
    ]
    return sorted(files)


def load_local_csv(file_path: Path) -> List[Dict]:
    """
    Loads and standardizes records from a local Agmarknet CSV file.
    """
    records = []
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        # Detect delimiter (comma or tab)
        sample = f.read(2048)
        f.seek(0)
        delimiter = "\t" if "\t" in sample and "," not in sample else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        raw_headers = reader.fieldnames or []

        # Build column index mapping
        col_map = {}
        for h in raw_headers:
            clean_h = h.strip().lower()
            if clean_h in COLUMN_ALIASES:
                col_map[h] = COLUMN_ALIASES[clean_h]

        for row in reader:
            std_row = {}
            for orig_key, std_key in col_map.items():
                std_row[std_key] = row.get(orig_key, "")

            # Filter and validate
            mandi = match_mandi(std_row.get("mandi", ""))
            crop = match_crop(std_row.get("crop", ""))
            norm_date = normalize_date(std_row.get("date", ""))
            modal_price = parse_price(std_row.get("modal_price_rs_qtl"))

            if mandi and crop and norm_date and modal_price:
                min_price = parse_price(std_row.get("min_price_rs_qtl")) or modal_price
                max_price = parse_price(std_row.get("max_price_rs_qtl")) or modal_price
                variety = std_row.get("variety", "Standard").strip() or "Standard"

                records.append(
                    {
                        "date": norm_date,
                        "mandi": mandi,
                        "state": MANDI_CONFIGS[mandi]["state"],
                        "district": MANDI_CONFIGS[mandi]["district"],
                        "crop": crop,
                        "variety": variety,
                        "modal_price_rs_qtl": modal_price,
                        "min_price_rs_qtl": min_price,
                        "max_price_rs_qtl": max_price,
                    }
                )

    return records


def fetch_from_data_gov_api(
    api_key: str,
    mandis: List[str] = list(MANDI_CONFIGS.keys()),
    crops: List[str] = CROPS,
    limit: int = 5000,
) -> List[Dict]:
    """
    Queries the official Data.gov.in API endpoint for Agmarknet mandi prices.
    """
    records = []
    headers = {"User-Agent": "KisanMitra-AI/1.0"}

    for crop in crops:
        for mandi in mandis:
            mandi_cfg = MANDI_CONFIGS[mandi]
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": limit,
                "filters[commodity]": crop,
                "filters[market]": mandi_cfg["market_alias"][0],
            }
            url = f"{DATA_GOV_IN_API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    rows = data.get("records", [])
                    for r in rows:
                        norm_date = normalize_date(r.get("arrival_date"))
                        modal_price = parse_price(r.get("modal_price"))
                        if norm_date and modal_price:
                            records.append(
                                {
                                    "date": norm_date,
                                    "mandi": mandi,
                                    "state": mandi_cfg["state"],
                                    "district": mandi_cfg["district"],
                                    "crop": crop,
                                    "variety": r.get("variety", "Standard"),
                                    "modal_price_rs_qtl": modal_price,
                                    "min_price_rs_qtl": parse_price(r.get("min_price")) or modal_price,
                                    "max_price_rs_qtl": parse_price(r.get("max_price")) or modal_price,
                                }
                            )
            except Exception as e:
                print(f"⚠️ Error fetching from data.gov.in API for {crop} at {mandi}: {e}")

    return records


def print_data_guidelines():
    """
    Prints clear, step-by-step instructions for obtaining real Agmarknet data.
    Strictly prevents fake or synthetic data generation.
    """
    print("\n" + "=" * 80)
    print(" 🌾 HOW TO OBTAIN REAL AGMARKNET MANDI DATA (ZERO-FAKE-DATA POLICY)")
    print("=" * 80)
    print("To train our price prediction model, we need real historical mandi records.")
    print("You can supply this in either of two ways:\n")

    print("METHOD 1: Manual CSV Download from Agmarknet Portal (Recommended & Instant)")
    print("  1. Visit the official portal: https://agmarknet.gov.in/")
    print("  2. Go to 'Price and Arrivals' -> 'Custom Query' or 'Yearly Analysis'.")
    print("  3. Select:")
    print("     • Commodities: Potato, Tomato")
    print("     • Markets: Azadpur (Delhi), Kolar (Karnataka), Agra (UP)")
    print("     • Date range: Past 2 to 3 years")
    print("  4. Click 'Submit' and export the result as CSV / Excel.")
    print(f"  5. Save the downloaded CSV file(s) inside:")
    print(f"     👉 {AGMARKNET_RAW_DIR.resolve()}/\n")

    print("METHOD 2: Official Data.gov.in API Key (Automated)")
    print("  1. Create a free account at: https://data.gov.in/")
    print("  2. Go to 'My Account' -> 'API Key' -> 'Generate API Key'.")
    print("  3. Open your local '.env' file and set:")
    print("     DATA_GOV_IN_API_KEY=your_actual_key_here")
    print("  4. Re-run: python3 src/price/fetch_agmarknet.py\n")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest and standardize Agmarknet mandi prices for Tomato & Potato"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(AGMARKNET_RAW_DIR),
        help="Directory containing downloaded Agmarknet CSVs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_PROCESSED_DIR / "agmarknet_daily.csv"),
        help="Path where standardized prices CSV will be saved",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📈 KISANMITRA AI — AGMARKNET MANDI PRICE INGESTION")
    print("=" * 80)

    all_records: List[Dict] = []

    # 1. Check for local raw CSV files
    local_files = scan_local_agmarknet_files(raw_dir)
    if local_files:
        print(f"\n📁 Found {len(local_files)} local Agmarknet file(s) in {raw_dir}:")
        for f in local_files:
            recs = load_local_csv(f)
            print(f"  • {f.name}: parsed {len(recs)} valid price records.")
            all_records.extend(recs)

    # 2. Check for data.gov.in API Key
    elif DATA_GOV_IN_API_KEY and DATA_GOV_IN_API_KEY != "your_data_gov_in_api_key_here":
        print("\n🔑 Data.gov.in API key found in .env. Fetching from API...")
        all_records = fetch_from_data_gov_api(DATA_GOV_IN_API_KEY)
        print(f"✓ Retrieved {len(all_records)} records from data.gov.in API.")

    # 3. Neither present - stop and explain exact steps
    else:
        print("\nℹ️  No local Agmarknet CSVs found in data/raw/agmarknet/ AND no API key in .env.")
        print_data_guidelines()
        sys.exit(0)

    if not all_records:
        print("\n⚠️ No valid records could be extracted. Please verify file format and columns.")
        sys.exit(1)

    # Deduplicate and sort by (date, mandi, crop)
    unique_map = {}
    for r in all_records:
        key = (r["date"], r["mandi"], r["crop"])
        unique_map[key] = r  # Keeps latest/unique

    sorted_records = sorted(
        unique_map.values(), key=lambda x: (x["date"], x["mandi"], x["crop"])
    )

    # Save standardized CSV
    fieldnames = [
        "date",
        "mandi",
        "state",
        "district",
        "crop",
        "variety",
        "modal_price_rs_qtl",
        "min_price_rs_qtl",
        "max_price_rs_qtl",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for r in sorted_records:
            f.write(
                f"{r['date']},{r['mandi']},{r['state']},{r['district']},"
                f"{r['crop']},{r['variety']},{r['modal_price_rs_qtl']},"
                f"{r['min_price_rs_qtl']},{r['max_price_rs_qtl']}\n"
            )

    print(f"\n✓ Successfully standardized {len(sorted_records)} mandi price records.")
    print(f"  Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
