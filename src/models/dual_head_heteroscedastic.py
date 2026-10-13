"""
Dual-Head Heteroscedastic Uncertainty Network
Predicts both class mean logits and aleatoric log-variance (data noise/stain ambiguity)
using Kendall & Gal's heteroscedastic cross-entropy loss formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.unet_dropout import ConvBlock, AttentionBlock


class HeteroscedasticAttentionUNet(nn.Module):
    """
    Dual-head network outputting:
    1. Logits head: mean class predictions f(x)
    2. Variance head: spatial aleatoric uncertainty log(sigma^2)(x)
    """
    def __init__(self, in_channels=3, num_classes=6, dropout_p=0.3):
        super(HeteroscedasticAttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared Encoder & Bottleneck
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128, dropout_p=dropout_p)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(128, 256, dropout_p=dropout_p)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512, dropout_p=dropout_p)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = ConvBlock(512, 256, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = ConvBlock(256, 128, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att1 = AttentionBlock(32, 32, 16)
        self.dec1 = ConvBlock(64, 32)

        # Dual Heads
        self.logits_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.log_var_head = nn.Conv2d(32, 1, kernel_size=1)  # Spatial aleatoric uncertainty

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.att4(self.up4(b), e4), self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([self.att3(self.up3(d4), e3), self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([self.att2(self.up2(d3), e2), self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([self.att1(self.up1(d2), e1), self.up1(d2)], dim=1))

        logits = self.logits_head(d1)
        log_var = self.log_var_head(d1)  # Log variance of observation noise

        return logits, log_var


class HeteroscedasticLoss(nn.Module):
    """
    Monte Carlo Heteroscedastic Cross-Entropy Loss (Kendall & Gal, 2017).
    s_hat = logits + std * epsilon, where epsilon ~ N(0, I).
    Minimizing this balances predictive cross entropy against uncertainty regularization.
    """
    def __init__(self, num_samples=10):
        super(HeteroscedasticLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, logits, log_var, targets):
        """
        logits: (B, C, H, W)
        log_var: (B, 1, H, W)
        targets: (B, H, W)
        """
        std = torch.exp(0.5 * log_var)
        loss = 0.0

        for _ in range(self.num_samples):
            noise = torch.randn_like(logits)
            perturbed_logits = logits + std * noise
            loss += F.cross_entropy(perturbed_logits, targets)

        loss = loss / self.num_samples
        # Add slight regularization to prevent unbounded log_var
        var_reg = 0.01 * torch.mean(torch.exp(log_var))
        return loss + var_reg
