"""
Uncertainty-Aware Attention U-Net with Spatial Monte Carlo Dropout
Implements Attention Gates, Spatial Dropout2d, and Feature Extraction for OoD Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Attention Gate for medical segmentation skip connections.
    Suppresses irrelevant feature activations and highlights salient nuclei boundaries.
    """
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm, ReLU, and optional Spatial Dropout."""
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UncertaintyAttentionUNet(nn.Module):
    """
    Attention U-Net equipped with Spatial Dropout2d layers.
    Designed for pixel-wise cell segmentation and Monte Carlo Epistemic Uncertainty estimation.
    """
    def __init__(self, in_channels=3, num_classes=6, dropout_p=0.3):
        super(UncertaintyAttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32, dropout_p=0.0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(32, 64, dropout_p=0.0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(64, 128, dropout_p=dropout_p)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(128, 256, dropout_p=dropout_p)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (active dropout during MC sampling)
        self.bottleneck = ConvBlock(256, 512, dropout_p=dropout_p)

        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.dec4 = ConvBlock(512, 256, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.dec3 = ConvBlock(256, 128, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(f_g=64, f_l=64, f_int=32)
        self.dec2 = ConvBlock(128, 64, dropout_p=0.0)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(f_g=32, f_l=32, f_int=16)
        self.dec1 = ConvBlock(64, 32, dropout_p=0.0)

        # Output classification head
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, return_features=False):
        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder forward pass
        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)

        logits = self.classifier(d1)

        if return_features:
            # Global pooled bottleneck features for OoD detection
            pooled_features = F.adaptive_avg_pool2d(b, (1, 1)).flatten(1)
            return logits, pooled_features

        return logits

    def enable_mc_dropout(self):
        """
        Puts spatial dropout layers in training mode while keeping
        BatchNorm in evaluation mode (frozen running statistics).
        This is crucial for unbiased Monte Carlo epistemic uncertainty sampling.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
