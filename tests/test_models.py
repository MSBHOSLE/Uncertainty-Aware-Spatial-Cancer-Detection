"""
Unit Tests for Neural Architectures and Uncertainty Models
"""

import pytest
import torch
from src.models.unet_dropout import UncertaintyAttentionUNet, AttentionBlock
from src.models.dual_head_heteroscedastic import HeteroscedasticAttentionUNet, HeteroscedasticLoss
from src.models.ensemble import DeepEnsemble


def test_attention_unet_forward():
    model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3)
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    assert logits.shape == (2, 6, 256, 256), f"Unexpected logits shape: {logits.shape}"


def test_attention_unet_features():
    model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3)
    x = torch.randn(2, 3, 256, 256)
    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 6, 256, 256)
    assert features.shape == (2, 512), f"Unexpected features shape: {features.shape}"


def test_mc_dropout_mode():
    model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3)
    model.enable_mc_dropout()
    # Ensure at least one Dropout2d layer is in training mode
    dropout_found = False
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            dropout_found = True
            assert m.training is True
    assert dropout_found, "No Dropout2d layers found in model!"


def test_heteroscedastic_unet():
    model = HeteroscedasticAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3)
    x = torch.randn(2, 3, 256, 256)
    logits, log_var = model(x)
    assert logits.shape == (2, 6, 256, 256)
    assert log_var.shape == (2, 1, 256, 256)

    targets = torch.randint(0, 6, (2, 256, 256))
    loss_fn = HeteroscedasticLoss(num_samples=3)
    loss = loss_fn(logits, log_var, targets)
    assert loss.item() > 0.0


def test_deep_ensemble():
    ens = DeepEnsemble.create(num_models=2, in_channels=3, num_classes=6)
    x = torch.randn(2, 3, 256, 256)
    out = ens.predict_with_uncertainty(x)
    assert out["mean_probs"].shape == (2, 6, 256, 256)
    assert out["variance"].shape == (2, 6, 256, 256)
    assert out["disagreement"].shape == (2, 1, 256, 256)
