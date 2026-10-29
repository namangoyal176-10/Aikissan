"""
Open-Meteo Agro-Weather Ingestion Module for KisanMitra AI.

Fetches daily agro-meteorological metrics (mean temperature, relative humidity,
and precipitation/rainfall) for target APMC Mandis using the public Open-Meteo Archive API.
No private API key required for non-commercial research use.
"""

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import urllib.parse
import urllib.request
import urllib.error

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import MANDI_CONFIGS, DATA_PROCESSED_DIR

# Open-Meteo Historical Weather Archive Endpoint
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_mandi_weather(
    mandi_key: str,
    start_date: str,
    end_date: str,
    timeout: int = 15,
) -> List[Dict]:
    """
    Queries Open-Meteo Archive API for a single mandi between start_date and end_date.
    Returns a list of dicts with daily weather readings.
    """
    if mandi_key not in MANDI_CONFIGS:
        raise ValueError(
            f"Unknown mandi: {mandi_key}. Available: {list(MANDI_CONFIGS.keys())}"
        )

    cfg = MANDI_CONFIGS[mandi_key]
    params = {
        "latitude": cfg["latitude"],
        "longitude": cfg["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
        ],
        "timezone": "Asia/Kolkata",
    }

    query_str = urllib.parse.urlencode(
        [
            ("latitude", params["latitude"]),
            ("longitude", params["longitude"]),
            ("start_date", params["start_date"]),
            ("end_date", params["end_date"]),
            ("daily", "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"),
            ("timezone", params["timezone"]),
        ]
    )
    url = f"{OPEN_METEO_ARCHIVE_URL}?{query_str}"

    headers = {"User-Agent": "KisanMitra-AI/1.0 (BTech Research Project)"}
    req = urllib.request.Request(url, headers=headers)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt == max_retries:
                print(f"❌ Failed to fetch weather for {mandi_key} after {max_retries} attempts: {e}")
                raise
            time.sleep(2 * attempt)

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    humidities = daily.get("relative_humidity_2m_mean", [])
    rainfalls = daily.get("precipitation_sum", [])

    records = []
    for d, t, h, r in zip(dates, temps, humidities, rainfalls):
        records.append(
            {
                "date": d,
                "mandi": cfg["mandi_name"],
                "state": cfg["state"],
                "district": cfg["district"],
                "latitude": cfg["latitude"],
                "longitude": cfg["longitude"],
                "temp_mean_c": t,
                "humidity_mean_pct": h,
                "rainfall_mm": r,
            }
        )

    return records


def fetch_all_mandis_weather(
    start_date: str,
    end_date: str,
    mandis: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Fetches weather records for all configured mandis.
    """
    target_mandis = mandis or list(MANDI_CONFIGS.keys())
    all_records = []

    print(f"\n🌤️ Fetching Open-Meteo weather data ({start_date} to {end_date})...")
    for mandi_key in target_mandis:
        print(f"  • Querying weather for {mandi_key} ({MANDI_CONFIGS[mandi_key]['state']})...")
        records = fetch_mandi_weather(mandi_key, start_date, end_date)
        all_records.extend(records)
        print(f"    ✓ Retrieved {len(records)} daily records.")
        # Brief courtesy pause to respect rate limits
        time.sleep(0.5)

    return all_records


def save_weather_csv(records: List[Dict], output_file: Path) -> Path:
    """
    Writes weather records to a clean CSV file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print("⚠️ No records to save.")
        return output_file

    fieldnames = [
        "date",
        "mandi",
        "state",
        "district",
        "latitude",
        "longitude",
        "temp_mean_c",
        "humidity_mean_pct",
        "rainfall_mm",
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for rec in records:
            row = [
                str(rec.get("date", "")),
                str(rec.get("mandi", "")),
                str(rec.get("state", "")),
                str(rec.get("district", "")),
                str(rec.get("latitude", "")),
                str(rec.get("longitude", "")),
                str(rec.get("temp_mean_c", "")),
                str(rec.get("humidity_mean_pct", "")),
                str(rec.get("rainfall_mm", "")),
            ]
            f.write(",".join(row) + "\n")

    return output_file


def main():
    default_end = (date.today() - timedelta(days=2)).isoformat()
    default_start = (date.today() - timedelta(days=365 * 2)).isoformat()

    parser = argparse.ArgumentParser(
        description="Fetch historical daily weather for mandis via Open-Meteo"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help="Start date YYYY-MM-DD (default: 2 years ago)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help="End date YYYY-MM-DD (default: 2 days ago)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_PROCESSED_DIR / "weather_daily.csv"),
        help="Path to save output CSV",
    )
    args = parser.parse_args()

    records = fetch_all_mandis_weather(args.start_date, args.end_date)
    out_path = save_weather_csv(records, Path(args.output))

    print(f"\n✓ Weather data successfully saved to: {out_path.resolve()}")
    print(f"  Total records: {len(records)} across {len(MANDI_CONFIGS)} mandis.")


if __name__ == "__main__":
    main()
