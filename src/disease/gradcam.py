"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for KisanMitra AI.

Provides visual explanations for CNN disease predictions by highlighting
the precise pixel regions of the leaf that influenced the model's decision.
Compatible with MobileNetV3 and EfficientNet backbones.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ALL_DISEASE_CLASSES, IDX_TO_CLASS
from src.disease.dataset import get_eval_transforms
from src.disease.model import get_disease_model, get_target_layer_for_gradcam


class GradCAM:
    """
    Computes Grad-CAM heatmaps for any PyTorch convolutional neural network.
    Uses forward and backward hooks to capture activations and gradients at the target layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains gradients with respect to output feature maps
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a 2D normalized Grad-CAM heatmap for the given input image tensor.

        Args:
            input_tensor: Shape (1, 3, 224, 224)
            target_class: Target class index. If None, uses the model's predicted class.

        Returns:
            heatmap_norm: 2D numpy array of shape (224, 224) with values in [0, 1]
            pred_class: Predicted class index
            confidence: Prediction confidence probability (0.0 to 1.0)
        """
        self.model.zero_grad()

        # 1. Forward Pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # 2. Backward Pass for target class score
        score = logits[0, target_class]
        score.backward()

        # 3. Compute neuron importance weights via Global Average Pooling over spatial dimensions
        # self.gradients: (1, C, H, W) -> weights: (1, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Compute weighted combination of forward activation maps
        # cam: (1, 1, H, W)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 5. Apply ReLU to isolate features with a POSITIVE influence on the target class
        cam = F.relu(cam)

        # 6. Upsample heatmap to match input tensor spatial dimensions (224, 224)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam.squeeze().cpu().numpy()

        # 7. Normalize heatmap to [0.0, 1.0]
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max - heatmap_min > 1e-8:
            heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap_norm = np.zeros_like(heatmap)

        return heatmap_norm, pred_class, confidence


def apply_colormap_on_image(
    org_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap on top of a PIL RGB image.
    Uses pure NumPy + PIL without requiring OpenCV or matplotlib display GUI.
    """
    # Resize original image to 224x224 to align with heatmap
    resized_org = org_image.resize((224, 224), Image.Resampling.BILINEAR)
    org_np = np.array(resized_org, dtype=np.float32)

    # Color map mapping (Turbo-like color progression: Blue -> Cyan -> Yellow -> Red)
    # Heatmap values [0, 1] mapped to RGB channels
    h = heatmap.clip(0.0, 1.0)

    # Simple JET/Turbo-like RGB colormap:
    # R: high when h is near 1
    # G: high when h is near 0.5
    # B: high when h is near 0
    r = np.clip(1.5 - np.abs(h * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(h * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(h * 4.0 - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1) * 255.0

    # Alpha blending: blended = (1 - alpha) * original + alpha * colormap
    blended_np = (1.0 - alpha) * org_np + alpha * color_map
    blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_np)


def explain_leaf_image(
    model: nn.Module,
    model_name: str,
    image_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, float, Image.Image, np.ndarray]:
    """
    End-to-end explanation pipeline for an input leaf image:
    1. Preprocesses image
    2. Runs model inference
    3. Generates Grad-CAM saliency heatmap
    4. Overlays heatmap onto original image
    5. Saves output visualization if output_path is provided

    Returns:
        (predicted_class_name, confidence, blended_pil_image, raw_heatmap_numpy)
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    pil_img = Image.open(img_path).convert("RGB")
    eval_transform = get_eval_transforms()
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    model.to(device)
    target_layer = get_target_layer_for_gradcam(model, model_name)
    cam_engine = GradCAM(model, target_layer)

    heatmap_norm, pred_class_idx, confidence = cam_engine.generate_heatmap(input_tensor)
    pred_class_name = IDX_TO_CLASS.get(pred_class_idx, f"Class_{pred_class_idx}")

    blended_img = apply_colormap_on_image(pil_img, heatmap_norm)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        blended_img.save(out_p)

    return pred_class_name, confidence, blended_img, heatmap_norm


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for a leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mobilenet_v3_small_best.pt",
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mobilenet_v3_small",
        help="Architecture name (mobilenet_v3_small or efficientnet_b0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/gradcam_output.jpg",
        help="Path to save blended visualization",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda or mps")
    args = parser.parse_args()

    model = get_disease_model(args.model_name, num_classes=len(ALL_DISEASE_CLASSES), pretrained=False)
    chkpt_path = Path(args.model_path)
    if chkpt_path.exists():
        chkpt = torch.load(chkpt_path, map_location=args.device)
        state_dict = chkpt.get("model_state_dict", chkpt)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded weights from {chkpt_path}")
    else:
        print(f"⚠️ Checkpoint not found at {chkpt_path}, using initial weights.")

    pred_class, conf, _, _ = explain_leaf_image(
        model=model,
        model_name=args.model_name,
        image_path=args.image,
        device=args.device,
        output_path=args.output,
    )

    print("\n" + "=" * 60)
    print(" 🌿 GRAD-CAM EXPLANATION RESULT")
    print("=" * 60)
    print(f"Predicted Diagnosis: {pred_class}")
    print(f"Model Confidence:    {conf * 100:.2f}%")
    print(f"Visualization Saved: {Path(args.output).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
