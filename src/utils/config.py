"""
Central Configuration for KisanMitra AI.

Defines target crops, mandi locations, PlantVillage classes, dataset split ratios,
and standard directory paths. Easy to extend with new mandis or crops.
"""

from pathlib import Path
import os
from typing import Dict, Any, List

# ------------------------------------------------------------------------------
# 1. Project Directory Paths
# ------------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED_DIR = Path(
    os.getenv("DATA_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed"))
)
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "reports")))

# Specific raw subdirectories
PLANTVILLAGE_RAW_DIR = DATA_RAW_DIR / "plantvillage"
AGMARKNET_RAW_DIR = DATA_RAW_DIR / "agmarknet"

# ------------------------------------------------------------------------------
# 2. Target Crops
# ------------------------------------------------------------------------------
CROPS: List[str] = ["Tomato", "Potato"]

# ------------------------------------------------------------------------------
# 3. Target Mandis Configuration
# ------------------------------------------------------------------------------
# Each mandi entry contains geographical coordinates (for Open-Meteo weather)
# and official state/district designations (for Agmarknet market queries).
MANDI_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Azadpur": {
        "mandi_name": "Azadpur",
        "state": "Delhi",
        "district": "North Delhi",
        "market_alias": ["Azadpur", "Azadpur Mandi", "Azadpur (Delhi)"],
        "latitude": 28.7161,
        "longitude": 77.1706,
        "crops": ["Tomato", "Potato"],
    },
    "Kolar": {
        "mandi_name": "Kolar",
        "state": "Karnataka",
        "district": "Kolar",
        "market_alias": ["Kolar", "Kolar APMC"],
        "latitude": 13.1367,
        "longitude": 78.1292,
        "crops": ["Tomato", "Potato"],
    },
    "Agra": {
        "mandi_name": "Agra",
        "state": "Uttar Pradesh",
        "district": "Agra",
        "market_alias": ["Agra", "Agra Mandi"],
        "latitude": 27.1767,
        "longitude": 78.0081,
        "crops": ["Tomato", "Potato"],
    },
}

# ------------------------------------------------------------------------------
# 4. PlantVillage Disease Classes (13 Target Classes)
# ------------------------------------------------------------------------------
POTATO_CLASSES: List[str] = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

TOMATO_CLASSES: List[str] = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

ALL_DISEASE_CLASSES: List[str] = sorted(POTATO_CLASSES + TOMATO_CLASSES)
CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(ALL_DISEASE_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# Common folder name variations in Kaggle downloads
CLASS_NAME_ALIASES: Dict[str, str] = {
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot": "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_Yellow_Leaf_curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy": "Tomato___healthy",
}

# ------------------------------------------------------------------------------
# 5. Dataset Splitting Configuration
# ------------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# 6. Supported Image Extensions
# ------------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
