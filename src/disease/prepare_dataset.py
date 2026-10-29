"""
PlantVillage Dataset Preparation Script for KisanMitra AI.

Verifies the 13 disease class folders (3 Potato, 10 Tomato), scans image counts,
detects class imbalance, performs a stratified 70/15/15 Train/Val/Test split,
and saves the split manifests to data/processed/ without duplicating image files.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    POTATO_CLASSES,
    TOMATO_CLASSES,
    ALL_DISEASE_CLASSES,
    CLASS_TO_IDX,
    CLASS_NAME_ALIASES,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    PLANTVILLAGE_RAW_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    IMAGE_EXTENSIONS,
)


def find_dataset_directory(candidate_dir: Optional[Path] = None) -> Path:
    """
    Locates the directory containing the PlantVillage crop class folders.
    Checks candidate_dir, then data/raw/plantvillage, then data/raw.
    """
    if candidate_dir and candidate_dir.exists():
        # Check if the candidate dir has any of our classes
        subfolders = {f.name for f in candidate_dir.iterdir() if f.is_dir()}
        if any(cls in subfolders for cls in ALL_DISEASE_CLASSES):
            return candidate_dir

    search_dirs = [
        PLANTVILLAGE_RAW_DIR,
        DATA_RAW_DIR,
        DATA_RAW_DIR / "PlantVillage",
        DATA_RAW_DIR / "plant_village",
        DATA_RAW_DIR / "color",  # Standard Kaggle folder name inside PlantVillage
    ]

    for directory in search_dirs:
        if directory.exists():
            subfolders = {f.name for f in directory.iterdir() if f.is_dir()}
            if any(cls in subfolders for cls in ALL_DISEASE_CLASSES):
                return directory

    # Fallback default
    return PLANTVILLAGE_RAW_DIR


def verify_class_folders(data_dir: Path) -> Tuple[Dict[str, Path], List[str], List[str]]:
    """
    Checks that all 13 required class folders exist in data_dir.
    Returns:
        matched_classes: Dict mapping canonical class name to folder Path.
        missing_classes: List of canonical class names not found.
        extra_folders: List of folder names found that are not in target classes.
    """
    if not data_dir.exists():
        return {}, list(ALL_DISEASE_CLASSES), []

    existing_subdirs = {d.name: d for d in data_dir.iterdir() if d.is_dir()}
    matched_classes: Dict[str, Path] = {}
    extra_folders: List[str] = []

    for folder_name, folder_path in existing_subdirs.items():
        canonical_name = CLASS_NAME_ALIASES.get(folder_name)
        if canonical_name and canonical_name in ALL_DISEASE_CLASSES:
            matched_classes[canonical_name] = folder_path
        else:
            extra_folders.append(folder_name)

    missing_classes = [cls for cls in ALL_DISEASE_CLASSES if cls not in matched_classes]
    return matched_classes, sorted(missing_classes), sorted(extra_folders)


def scan_images(matched_classes: Dict[str, Path]) -> Dict[str, List[Path]]:
    """
    Scans each class folder and collects valid image paths.
    """
    class_images: Dict[str, List[Path]] = {}
    for class_name, folder_path in matched_classes.items():
        images = [
            img_path
            for img_path in folder_path.iterdir()
            if img_path.is_file() and img_path.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}
        ]
        class_images[class_name] = sorted(images)
    return class_images


def report_class_distribution(class_images: Dict[str, List[Path]]) -> Dict[str, int]:
    """
    Prints a formatted distribution table and evaluates class imbalance.
    """
    counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    total_images = sum(counts.values())

    print("\n" + "=" * 80)
    print(" 🌿 PLANTVILLAGE DATASET INVENTORY (POTATO & TOMATO - 13 CLASSES)")
    print("=" * 80)
    print(f"{'Crop':<10} | {'Class Name':<45} | {'Count':<8} | {'Share (%)':<10}")
    print("-" * 80)

    for cls in ALL_DISEASE_CLASSES:
        crop = "Potato" if cls.startswith("Potato") else "Tomato"
        count = counts.get(cls, 0)
        pct = (count / total_images * 100) if total_images > 0 else 0.0
        print(f"{crop:<10} | {cls:<45} | {count:<8} | {pct:>8.2f}%")

    print("-" * 80)
    print(f"Total Images: {total_images}")
    print("=" * 80)

    if total_images == 0:
        return counts

    # Imbalance analysis
    min_count = min(counts.values())
    max_count = max(counts.values())
    ratio = max_count / min_count if min_count > 0 else float("inf")

    print(f"\n📊 Class Imbalance Analysis:")
    print(f"  • Min class count: {min_count}")
    print(f"  • Max class count: {max_count}")
    print(f"  • Imbalance Ratio (Max / Min): {ratio:.2f}")

    if ratio > 2.0:
        print(
            "\n⚠️  WARNING: Significant class imbalance detected (Ratio > 2.0).\n"
            "   Common causes: Tomato Yellow Leaf Curl Virus has substantially more samples\n"
            "   than bacterial spot or healthy classes in PlantVillage.\n"
            "   Strategy for Phase 3: We will implement Class-Weighted Cross-Entropy Loss\n"
            "   and Random Data Augmentation to prevent CNN prediction bias."
        )
    else:
        print("  ✓ Class distribution is reasonably balanced.")

    return counts


def stratified_split_dataset(
    class_images: Dict[str, List[Path]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[Tuple[Path, str, str, int]]]:
    """
    Performs a deterministic stratified Train/Val/Test split (70/15/15) per class.
    Returns:
        Dict with keys 'train', 'val', 'test', each containing tuples of
        (image_path, class_name, crop, class_idx).
    """
    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_name in ALL_DISEASE_CLASSES:
        images = class_images.get(class_name, [])
        if not images:
            continue

        crop = "Potato" if class_name.startswith("Potato") else "Tomato"
        class_idx = CLASS_TO_IDX[class_name]

        # Shuffle deterministically
        shuffled = images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining goes to test to ensure sum == n
        n_test = n - n_train - n_val

        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train : n_train + n_val]
        test_imgs = shuffled[n_train + n_val :]

        for img in train_imgs:
            splits["train"].append((img, class_name, crop, class_idx))
        for img in val_imgs:
            splits["val"].append((img, class_name, crop, class_idx))
        for img in test_imgs:
            splits["test"].append((img, class_name, crop, class_idx))

    return splits


def save_split_manifests(
    splits: Dict[str, List[Tuple[Path, str, str, int]]],
    output_dir: Path = DATA_PROCESSED_DIR,
) -> None:
    """
    Saves split records to CSVs and class mapping to JSON in output_dir.
    Does NOT copy images. Stores relative paths for portability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save master splits CSV
    master_csv_path = output_dir / "disease_splits.csv"
    with open(master_csv_path, "w", encoding="utf-8") as f:
        f.write("image_path,relative_path,crop,class_name,class_idx,split\n")
        for split_name, records in splits.items():
            for img_path, class_name, crop, class_idx in records:
                try:
                    rel_path = img_path.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel_path = img_path
                f.write(
                    f'"{img_path}","{rel_path}","{crop}","{class_name}",{class_idx},"{split_name}"\n'
                )

    # 2. Save individual split CSVs for direct consumption by PyTorch Dataset
    for split_name in ["train", "val", "test"]:
        split_csv_path = output_dir / f"disease_{split_name}.csv"
        records = splits.get(split_name, [])
        with open(split_csv_path, "w", encoding="utf-8") as f:
            f.write("image_path,relative_path,crop,class_name,class_idx\n")
            for img_path, class_name, crop, class_idx in records:
                try:
                    rel_path = img_path.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel_path = img_path
                f.write(
                    f'"{img_path}","{rel_path}","{crop}","{class_name}",{class_idx}\n'
                )

    # 3. Save class mapping dictionary
    mapping_json_path = output_dir / "class_to_idx.json"
    with open(mapping_json_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)

    # 4. Save distribution summary
    split_summary = {
        "total_train": len(splits["train"]),
        "total_val": len(splits["val"]),
        "total_test": len(splits["test"]),
        "total_images": sum(len(records) for records in splits.values()),
        "train_val_test_ratio": f"{TRAIN_RATIO * 100:.0f}/{VAL_RATIO * 100:.0f}/{TEST_RATIO * 100:.0f}",
        "random_seed": RANDOM_SEED,
        "classes": ALL_DISEASE_CLASSES,
    }
    summary_path = output_dir / "disease_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print("\n✓ Split manifests successfully generated (no images copied):")
    print(f"  • Master split list:       {master_csv_path}")
    print(f"  • Train manifest:          {output_dir / 'disease_train.csv'} ({len(splits['train'])} samples)")
    print(f"  • Validation manifest:     {output_dir / 'disease_val.csv'} ({len(splits['val'])} samples)")
    print(f"  • Test manifest:           {output_dir / 'disease_test.csv'} ({len(splits['test'])} samples)")
    print(f"  • Class mapping (0..12):   {mapping_json_path}")
    print(f"  • Split summary metadata:  {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PlantVillage Disease Dataset (13 classes for Potato and Tomato)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to folder containing the 13 PlantVillage class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_PROCESSED_DIR),
        help="Path where split CSVs and mapping JSON will be saved",
    )
    args = parser.parse_args()

    candidate_dir = Path(args.data_dir) if args.data_dir else None
    data_dir = find_dataset_directory(candidate_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("🌾 KISANMITRA AI — DISEASE DATASET VERIFIER & PREPARATION")
    print("=" * 80)
    print(f"Checking dataset directory: {data_dir.resolve()}")

    matched_classes, missing_classes, extra_folders = verify_class_folders(data_dir)

    # Report folder verification status
    print(f"\nTarget classes required: 13 (Potato: 3, Tomato: 10)")
    print(f"Matched classes found:   {len(matched_classes)} / 13")

    if extra_folders:
        print(f"\nℹ️  Extra folders detected ({len(extra_folders)}) — e.g. other crops in full PlantVillage:")
        for extra in extra_folders[:5]:
            print(f"   - {extra}")
        if len(extra_folders) > 5:
            print(f"   ... and {len(extra_folders) - 5} more (these will be safely ignored).")

    if missing_classes:
        print(f"\n❌ ERROR: Missing {len(missing_classes)} required class folder(s):")
        for missing in missing_classes:
            crop = "Potato" if missing.startswith("Potato") else "Tomato"
            print(f"   [MISSING] {crop}: {missing}")

        print(
            "\n👉 ACTION REQUIRED:\n"
            f"   Please download the PlantVillage dataset from Kaggle and ensure\n"
            f"   the 13 tomato and potato folders are placed inside:\n"
            f"   {PLANTVILLAGE_RAW_DIR.resolve()}\n"
            f"\n   See instructions in README.md or run with --data-dir <path>."
        )
        sys.exit(1)

    print("\n✓ All 13 target class folders are verified!")

    # Scan and count images
    class_images = scan_images(matched_classes)
    counts = report_class_distribution(class_images)

    total_images = sum(counts.values())
    if total_images == 0:
        print(
            f"\n⚠️  Found the 13 folders, but they contain 0 valid images (.jpg/.jpeg/.png).\n"
            f"   Please verify images are placed inside the folders."
        )
        sys.exit(1)

    # Perform stratified split
    print(f"\nSplitting dataset into Train ({TRAIN_RATIO*100:.0f}%), Val ({VAL_RATIO*100:.0f}%), Test ({TEST_RATIO*100:.0f}%)...")
    splits = stratified_split_dataset(class_images)

    # Save manifests
    save_split_manifests(splits, output_dir)
    print("\n🎉 Dataset preparation complete! Ready for Phase 3 (Disease Model Training).")


if __name__ == "__main__":
    main()
