"""
Unit Tests for Phase 3: Disease Detection Models, Augmentation, and Grad-CAM.
Verifies model architecture shapes, parameter counting, Grad-CAM heatmap extraction,
and data augmentation pipelines using dummy tensors.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.utils.config import ALL_DISEASE_CLASSES
from src.disease.model import (
    get_disease_model,
    count_parameters,
    get_model_size_mb,
    get_target_layer_for_gradcam,
)
from src.disease.dataset import (
    get_training_transforms,
    get_eval_transforms,
    compute_class_weights,
)
from src.disease.gradcam import (
    GradCAM,
    apply_colormap_on_image,
)
from src.disease.evaluate import find_top_confused_pairs


class TestDiseaseModels(unittest.TestCase):
    """Tests CNN architecture instantiation and forward passes."""

    def test_mobilenet_v3_small_shape(self):
        """MobileNetV3-Small should accept (B, 3, 224, 224) and output (B, 13)."""
        model = get_disease_model(
            "mobilenet_v3_small",
            num_classes=13,
            pretrained=False,
        )
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        self.assertEqual(output.shape, (2, 13))
        total_p, trainable_p = count_parameters(model)
        self.assertGreater(total_p, 1_000_000)
        self.assertEqual(total_p, trainable_p)

        size_mb = get_model_size_mb(model)
        self.assertGreater(size_mb, 1.0)
        self.assertLess(size_mb, 50.0)

    def test_efficientnet_b0_shape(self):
        """EfficientNet-B0 should accept (B, 3, 224, 224) and output (B, 13)."""
        model = get_disease_model(
            "efficientnet_b0",
            num_classes=13,
            pretrained=False,
        )
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        self.assertEqual(output.shape, (2, 13))
        total_p, _ = count_parameters(model)
        self.assertGreater(total_p, 3_000_000)


class TestGradCAM(unittest.TestCase):
    """Tests Grad-CAM hook registration and heatmap generation."""

    def test_gradcam_heatmap_generation(self):
        """Grad-CAM should produce a (224, 224) normalized heatmap in [0, 1]."""
        model = get_disease_model("mobilenet_v3_small", num_classes=13, pretrained=False)
        target_layer = get_target_layer_for_gradcam(model, "mobilenet_v3_small")

        cam_engine = GradCAM(model, target_layer)
        dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

        heatmap, pred_class, confidence = cam_engine.generate_heatmap(dummy_tensor)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(0 <= pred_class < 13)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertTrue(0.0 <= heatmap.min())
        self.assertTrue(heatmap.max() <= 1.0)

    def test_colormap_overlay(self):
        """Colormap overlay should blend with PIL image and maintain 224x224 size."""
        dummy_pil = Image.new("RGB", (300, 300), color=(100, 150, 100))
        dummy_heatmap = np.ones((224, 224), dtype=np.float32) * 0.5

        blended = apply_colormap_on_image(dummy_pil, dummy_heatmap)
        self.assertEqual(blended.size, (224, 224))
        self.assertEqual(blended.mode, "RGB")


class TestAugmentationAndTransforms(unittest.TestCase):
    """Tests data augmentation pipelines and class weight computation."""

    def test_transforms_output_dimensions(self):
        """Transforms must output PyTorch tensors of shape (3, 224, 224)."""
        dummy_pil = Image.new("RGB", (300, 400), color=(50, 200, 50))

        train_transform = get_training_transforms()
        eval_transform = get_eval_transforms()

        train_tensor = train_transform(dummy_pil)
        eval_tensor = eval_transform(dummy_pil)

        self.assertEqual(train_tensor.shape, (3, 224, 224))
        self.assertEqual(eval_tensor.shape, (3, 224, 224))

    def test_fallback_class_weights(self):
        """When manifest does not exist, compute_class_weights returns ones."""
        weights = compute_class_weights("non_existent_manifest.csv", num_classes=13)
        self.assertEqual(len(weights), 13)
        self.assertTrue(torch.allclose(weights, torch.ones(13)))


class TestConfusedPairsAnalysis(unittest.TestCase):
    """Tests ranking and pathological explanation of confused class pairs."""

    def test_find_top_confused_pairs(self):
        """Should extract top off-diagonal confusion errors with explanations."""
        # 13x13 confusion matrix
        matrix = np.zeros((13, 13), dtype=int)
        # Create diagonal correct predictions
        for i in range(13):
            matrix[i, i] = 100

        # Inject specific error pairs
        matrix[1, 7] = 25  # Class 1 confused as Class 7 (25 errors)
        matrix[3, 0] = 15  # Class 3 confused as Class 0 (15 errors)

        confused = find_top_confused_pairs(matrix, top_k=2)
        self.assertEqual(len(confused), 2)
        self.assertEqual(confused[0]["count"], 25)
        self.assertEqual(confused[1]["count"], 15)
        self.assertIn("explanation", confused[0])


if __name__ == "__main__":
    unittest.main()
