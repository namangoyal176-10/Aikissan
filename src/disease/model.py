"""
CNN Model Architectures for Leaf Disease Classification.

Supports MobileNetV3-Small (lightweight mobile baseline) and EfficientNet-B0
with transfer learning pretrained on ImageNet, adapted for 13 PlantVillage classes.
Includes parameter counting, memory footprint calculation, and Grad-CAM target layer resolution.
"""

import time
from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models


SUPPORTED_MODELS = ["mobilenet_v3_small", "efficientnet_b0", "mobilenet_v3_large"]


def get_disease_model(
    model_name: str = "mobilenet_v3_small",
    num_classes: int = 13,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """
    Constructs a transfer learning CNN with updated classifier for plant disease classification.

    Args:
        model_name: 'mobilenet_v3_small', 'efficientnet_b0', or 'mobilenet_v3_large'
        num_classes: Number of disease categories (13)
        pretrained: If True, loads ImageNet pretrained feature extraction weights
        dropout_rate: Dropout probability in the classification head

    Returns:
        PyTorch nn.Module ready for training or inference.
    """
    model_key = model_name.lower().strip()

    if model_key == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

        # In MobileNetV3, classifier is Sequential(Linear(576, 1024), Hardswish(), Dropout, Linear(1024, 1000))
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout_rate)
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_key == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout_rate)
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_key == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # In EfficientNet, classifier is Sequential(Dropout, Linear(1280, 1000))
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout_rate)
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}. Supported: {SUPPORTED_MODELS}"
        )

    return model


def get_target_layer_for_gradcam(model: nn.Module, model_name: str) -> nn.Module:
    """
    Returns the final convolutional feature extraction layer used by Grad-CAM.
    For both MobileNetV3 and EfficientNet, this corresponds to model.features[-1].
    """
    model_key = model_name.lower().strip()
    if "mobilenet" in model_key or "efficientnet" in model_key:
        return model.features[-1]
    raise ValueError(f"Grad-CAM target layer undefined for model: {model_name}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculates the memory footprint of model parameters + buffers in Megabytes (MB).
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return round(size_mb, 2)


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: Union[str, torch.device] = "cpu",
    num_warmup: int = 10,
    num_runs: int = 50,
) -> float:
    """
    Measures average inference latency in milliseconds (ms) per single image on the given device.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size, device=device)

    # Warmup runs to stabilize caches
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    total_time = time.perf_counter() - start_time

    latency_ms = (total_time / num_runs) * 1000.0
    return round(latency_ms, 2)
