"""
Model Evaluation, Benchmark Comparison & Confusion Matrix Analysis for KisanMitra AI.

Evaluates trained CNN checkpoints on the test split (data/processed/disease_test.csv).
Outputs:
1. Test Accuracy, Macro Precision, Macro Recall, Macro F1
2. Model Parameter Count, Model Size (MB), Inference Latency (ms/image)
3. Full 13x13 Confusion Matrix
4. Top 5 Most Confused Class Pairs with biological/pathological explanations
5. Automatic updating of reports/results.md for resume documentation
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    ALL_DISEASE_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.disease.dataset import PlantDiseaseDataset, get_eval_transforms
from src.disease.model import (
    get_disease_model,
    count_parameters,
    get_model_size_mb,
    measure_inference_latency,
)


# Agronomic knowledge base explaining why specific disease pairs get confused
PATHOLOGY_EXPLANATIONS = {
    ("Tomato___Early_blight", "Tomato___Target_Spot"): (
        "Both diseases are fungal and produce dark brown necrotic lesions with concentric 'target-board' "
        "rings and chlorotic yellow halos. Visually almost identical in early stages without a microscope."
    ),
    ("Tomato___Early_blight", "Tomato___Septoria_leaf_spot"): (
        "Both initiate on older foliage as circular brown spots with chlorotic margins. Septoria lesions "
        "have light grey centers with tiny fruiting bodies (pycnidia) that are hard for standard resolution to distinguish."
    ),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"): (
        "Spider mite feeding punctures cells causing diffuse yellow speckling (stippling). TYLCV causes marginal "
        "yellow chlorosis and leaf distortion. Both present as extensive yellowing of tomato leaf tissue."
    ),
    ("Potato___Early_blight", "Potato___Late_blight"): (
        "Both cause necrotic leaf lesions on potato. While Late Blight is water-soaked and spreads rapidly in humidity, "
        "under dry field conditions, the dried necrotic tissue looks remarkably similar to Early Blight lesions."
    ),
    ("Tomato___Leaf_Mold", "Tomato___Bacterial_spot"): (
        "Leaf Mold presents as pale chlorotic yellow patches on upper leaf surfaces. Bacterial spot causes "
        "water-soaked dark spots. Under direct sunlight glare, specular reflection on greasy bacterial spots mimics chlorotic patches."
    ),
    ("Tomato___healthy", "Tomato___Septoria_leaf_spot"): (
        "Early stage Septoria may only feature 1 or 2 tiny pinpoint spots on an otherwise green, healthy leaflet, "
        "leading the model to occasionally classify the leaf as overall healthy."
    ),
    ("Potato___healthy", "Potato___Early_blight"): (
        "Inciting Early Blight lesions start as micro-lesions (<2mm). If the lesion is at the extreme leaf margin or occluded, "
        "the model defaults to healthy."
    ),
}


def find_top_confused_pairs(
    conf_matrix: np.ndarray,
    top_k: int = 5,
) -> List[Dict]:
    """
    Finds off-diagonal elements in the confusion matrix with the highest error counts.
    Returns list of dicts with True Class, Predicted Class, Error Count, and Pathological Explanation.
    """
    errors = []
    num_classes = conf_matrix.shape[0]

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 0:
                true_cls = IDX_TO_CLASS[i]
                pred_cls = IDX_TO_CLASS[j]
                count = int(conf_matrix[i, j])

                # Lookup explanation (order agnostic)
                explanation = (
                    PATHOLOGY_EXPLANATIONS.get((true_cls, pred_cls))
                    or PATHOLOGY_EXPLANATIONS.get((pred_cls, true_cls))
                    or "Similar necrotic leaf discoloration and lesion texture between these two classes."
                )

                errors.append(
                    {
                        "true_class": true_cls,
                        "pred_class": pred_cls,
                        "count": count,
                        "explanation": explanation,
                    }
                )

    # Sort descending by error count
    errors.sort(key=lambda x: x["count"], reverse=True)
    return errors[:top_k]


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_csv: Path = DATA_PROCESSED_DIR / "disease_test.csv",
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict:
    """Evaluates a saved PyTorch checkpoint on the test set."""
    dev = torch.device(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split manifest not found: {test_csv}")

    chkpt = torch.load(checkpoint_path, map_location=dev)
    model_name = chkpt.get("model_name", "mobilenet_v3_small")
    state_dict = chkpt.get("model_state_dict", chkpt)

    num_classes = len(ALL_DISEASE_CLASSES)
    model = get_disease_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    test_dataset = PlantDiseaseDataset(test_csv, transform=get_eval_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_params, _ = count_parameters(model)
    size_mb = get_model_size_mb(model)
    latency_ms = measure_inference_latency(model, device=dev)

    all_preds = []
    all_targets = []

    print(f"\nEvaluating {model_name} on {len(test_dataset)} test samples...")
    start_eval = time.time()

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(dev)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    eval_duration = time.time() - start_eval

    # Metrics
    accuracy = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    conf_mat = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    top_confused = find_top_confused_pairs(conf_mat, top_k=5)

    return {
        "model_name": model_name,
        "checkpoint_file": checkpoint_path.name,
        "accuracy": round(accuracy * 100, 2),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "total_parameters": total_params,
        "size_mb": size_mb,
        "latency_ms": latency_ms,
        "num_test_samples": len(test_dataset),
        "eval_duration_sec": round(eval_duration, 2),
        "confusion_matrix": conf_mat.tolist(),
        "top_5_confused_pairs": top_confused,
    }


def print_evaluation_summary(results: Dict):
    """Prints a formatted evaluation report and the top 5 confused pairs."""
    print("\n" + "=" * 80)
    print(f" 📊 EVALUATION REPORT: {results['model_name'].upper()}")
    print("=" * 80)
    print(f"  • Test Accuracy:        {results['accuracy']}%")
    print(f"  • Macro F1-Score:       {results['macro_f1']}")
    print(f"  • Macro Precision:     {results['macro_precision']}")
    print(f"  • Macro Recall:        {results['macro_recall']}")
    print(f"  • Model Size:          {results['size_mb']} MB")
    print(f"  • Parameter Count:     {results['total_parameters']:,}")
    print(f"  • CPU Latency:         {results['latency_ms']} ms / image")
    print("=" * 80)

    print("\n🔍 TOP 5 MOST CONFUSED CLASS PAIRS:")
    print("-" * 80)
    for idx, item in enumerate(results["top_5_confused_pairs"], 1):
        print(f"{idx}. True: [{item['true_class']}]")
        print(f"   Predicted as: [{item['pred_class']}] ({item['count']} error instances)")
        print(f"   Biological Explanation: {item['explanation']}\n")
    print("-" * 80)


def update_results_markdown(results_list: List[Dict]):
    """Appends model benchmark comparisons to reports/results.md."""
    results_path = REPORTS_DIR / "results.md"
    if not results_path.exists():
        return

    content = results_path.read_text(encoding="utf-8")

    # Build comparison rows
    rows = []
    for r in results_list:
        rows.append(
            f"| **{r['model_name']}** | {r['total_parameters']:,} | {r['size_mb']} MB | "
            f"{r['accuracy']}% | {r['macro_f1']} | {r['latency_ms']} ms |"
        )
    table_content = "\n".join(rows)

    # Replace pending section
    target_marker = "| MobileNetV3-Small (Baseline) | ~2.5M | - | - | - | - | - | - |"
    replacement_table = (
        "| Architecture | Parameters | Size (MB) | Test Acc (%) | Macro F1 | Latency (CPU) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n" + table_content
    )

    if target_marker in content:
        # Replace the entire old table header + rows
        old_table_header = (
            "| Model Architecture | Parameters | Train Accuracy | Val Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Time (CPU) |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            "| MobileNetV3-Small (Baseline) | ~2.5M | - | - | - | - | - | - |\n"
            "| MobileNetV3-Large | ~5.4M | - | - | - | - | - | - |\n"
            "| EfficientNet-B0 | ~5.3M | - | - | - | - | - | - |"
        )
        updated_content = content.replace(old_table_header, replacement_table)
        results_path.write_text(updated_content, encoding="utf-8")
        print(f"✓ Updated benchmark table in {results_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate disease detection models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mobilenet_v3_small_best.pt",
        help="Path to trained checkpoint .pt file",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(DATA_PROCESSED_DIR / "disease_test.csv"),
        help="Path to test split manifest",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mps")
    args = parser.parse_args()

    chkpt_path = Path(args.checkpoint)
    test_csv = Path(args.test_csv)

    results = evaluate_checkpoint(chkpt_path, test_csv=test_csv, device=args.device)
    print_evaluation_summary(results)

    # Save evaluation json
    eval_json_path = REPORTS_DIR / f"{results['model_name']}_eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved detailed evaluation JSON to: {eval_json_path.resolve()}")


if __name__ == "__main__":
    main()
