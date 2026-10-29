"""
Unit Tests for Phase 2: Data Preparation & Ingestion.
Tests config parameters, PlantVillage folder verification,
Agmarknet normalization, Open-Meteo weather structures, and data merging.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from src.utils.config import (
    MANDI_CONFIGS,
    CROPS,
    ALL_DISEASE_CLASSES,
    POTATO_CLASSES,
    TOMATO_CLASSES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)
from src.disease.prepare_dataset import (
    verify_class_folders,
    stratified_split_dataset,
    save_split_manifests,
)
from src.price.fetch_agmarknet import (
    normalize_date,
    match_mandi,
    match_crop,
    parse_price,
    load_local_csv,
)
from src.price.merge_data import (
    merge_price_and_weather,
)


class TestConfigSanity(unittest.TestCase):
    """Verifies that the central project configuration is sound."""

    def test_mandis_configured(self):
        """Must have exactly 3 initial mandis configured with required fields."""
        self.assertEqual(len(MANDI_CONFIGS), 3)
        for mandi_name in ["Azadpur", "Kolar", "Agra"]:
            self.assertIn(mandi_name, MANDI_CONFIGS)
            cfg = MANDI_CONFIGS[mandi_name]
            self.assertIn("latitude", cfg)
            self.assertIn("longitude", cfg)
            self.assertIn("state", cfg)
            self.assertIn("district", cfg)

    def test_plantvillage_classes_count(self):
        """Must have exactly 13 disease classes: 3 Potato + 10 Tomato."""
        self.assertEqual(len(POTATO_CLASSES), 3)
        self.assertEqual(len(TOMATO_CLASSES), 10)
        self.assertEqual(len(ALL_DISEASE_CLASSES), 13)

    def test_split_ratios_sum_to_one(self):
        """Train (70) + Val (15) + Test (15) must sum to 1.0."""
        self.assertAlmostEqual(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0, places=5)


class TestDiseaseDataPreparation(unittest.TestCase):
    """Tests the dataset verification and stratified splitting logic."""

    def test_verify_class_folders_missing(self):
        """Checks that empty or partial directories properly report missing classes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create only 2 folders
            (tmp_path / "Potato___healthy").mkdir()
            (tmp_path / "Tomato___healthy").mkdir()

            matched, missing, extra = verify_class_folders(tmp_path)
            self.assertEqual(len(matched), 2)
            self.assertEqual(len(missing), 11)
            self.assertEqual(len(extra), 0)

    def test_stratified_split_ratios(self):
        """Checks that stratified splitting divides per-class images into 70/15/15."""
        # Mock 100 images for a class
        fake_images = [Path(f"/fake/img_{i}.jpg") for i in range(100)]
        mock_data = {ALL_DISEASE_CLASSES[0]: fake_images}

        splits = stratified_split_dataset(
            mock_data,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=RANDOM_SEED,
        )

        self.assertEqual(len(splits["train"]), 70)
        self.assertEqual(len(splits["val"]), 15)
        self.assertEqual(len(splits["test"]), 15)


class TestAgmarknetParsers(unittest.TestCase):
    """Tests date, mandi, crop, and price normalization helpers."""

    def test_normalize_date_formats(self):
        """Test multiple raw Indian date string formats."""
        self.assertEqual(normalize_date("15/08/2023"), "2023-08-15")
        self.assertEqual(normalize_date("2023-08-15"), "2023-08-15")
        self.assertEqual(normalize_date("15-08-2023"), "2023-08-15")
        self.assertEqual(normalize_date("15-Aug-2023"), "2023-08-15")
        self.assertIsNone(normalize_date("invalid-date"))

    def test_match_mandi(self):
        """Test fuzzy matching of mandi names."""
        self.assertEqual(match_mandi("Azadpur Mandi"), "Azadpur")
        self.assertEqual(match_mandi("Kolar APMC Market"), "Kolar")
        self.assertEqual(match_mandi("Agra"), "Agra")
        self.assertIsNone(match_mandi("Mumbai APMC"))

    def test_match_crop(self):
        """Test crop matching."""
        self.assertEqual(match_crop("Tomato Hybrid"), "Tomato")
        self.assertEqual(match_crop("Potato Jyoti"), "Potato")
        self.assertIsNone(match_crop("Wheat"))

    def test_parse_price(self):
        """Test price sanitization."""
        self.assertEqual(parse_price("1,850.50"), 1850.50)
        self.assertEqual(parse_price("2400"), 2400.0)
        self.assertIsNone(parse_price("0"))
        self.assertIsNone(parse_price("N/A"))


class TestMergePriceWeather(unittest.TestCase):
    """Tests merging price records with weather records."""

    def test_merge_alignment(self):
        """Price and weather records on same date and mandi should align."""
        price_records = [
            {
                "date": "2024-01-01",
                "mandi": "Azadpur",
                "state": "Delhi",
                "district": "North Delhi",
                "crop": "Tomato",
                "variety": "Hybrid",
                "modal_price_rs_qtl": "1800",
                "min_price_rs_qtl": "1600",
                "max_price_rs_qtl": "2000",
            }
        ]
        weather_records = [
            {
                "date": "2024-01-01",
                "mandi": "Azadpur",
                "temp_mean_c": "12.5",
                "humidity_mean_pct": "85",
                "rainfall_mm": "0.0",
            }
        ]

        merged = merge_price_and_weather(price_records, weather_records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["modal_price_rs_qtl"], "1800")
        self.assertEqual(merged[0]["temp_mean_c"], "12.5")
        self.assertEqual(merged[0]["rainfall_mm"], "0.0")


if __name__ == "__main__":
    unittest.main()
