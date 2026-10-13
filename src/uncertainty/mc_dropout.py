"""
Monte Carlo Spatial Dropout Uncertainty Estimator
Executes T stochastic forward passes with active Spatial Dropout2d.
Calculates mean predictions, predictive variance, mutual information, and entropy.
Stores individual stochastic passes for downstream TME spatial biomarker propagation.
"""

import torch
import torch.nn.functional as F


def run_mc_dropout_inference(model, image_tensor, num_samples=15, return_samples=True):
    """
    Executes T stochastic forward passes on image_tensor.
    
    Args:
        model: UncertaintyAttentionUNet
        image_tensor: (B, C, H, W)
        num_samples: T stochastic passes
        return_samples: if True, returns the individual probability samples [T, B, C, H, W]
    
    Returns:
        dict containing:
            - mean_prediction: (B, C, H, W) mean probabilities
            - predicted_classes: (B, H, W) argmax class indices
            - variance_map: (B, C, H, W) class-wise variance
            - foreground_variance: (B, 1, H, W) sum of variance across cell classes (1..5)
            - predictive_entropy: (B, 1, H, W) total uncertainty H(p_mean)
            - mutual_information: (B, 1, H, W) epistemic uncertainty I(y, w | x)
            - stochastic_samples: (T, B, C, H, W) (optional)
    """
    model.enable_mc_dropout()  # Keep Dropout active, BatchNorm in eval

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            samples.append(probs)

    # Stack along first dimension: [T, B, C, H, W]
    stacked_probs = torch.stack(samples, dim=0)

    # 1. Predictive Mean
    mean_probs = torch.mean(stacked_probs, dim=0)  # [B, C, H, W]
    pred_classes = torch.argmax(mean_probs, dim=1) # [B, H, W]

    # 2. Predictive Variance
    variance_map = torch.var(stacked_probs, dim=0) # [B, C, H, W]
    
    # Foreground variance (excluding background class 0)
    foreground_variance = torch.sum(variance_map[:, 1:, :, :], dim=1, keepdim=True)

    # 3. Predictive Entropy (Total Uncertainty)
    eps = 1e-8
    predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1, keepdim=True)

    # 4. Mutual Information (Epistemic / Model Doubt)
    # MI = H(mean_p) - mean(H(p_t))
    sample_entropies = -torch.sum(stacked_probs * torch.log(stacked_probs + eps), dim=2) # [T, B, H, W]
    mean_sample_entropy = torch.mean(sample_entropies, dim=0).unsqueeze(1) # [B, 1, H, W]
    mutual_information = torch.clamp(predictive_entropy - mean_sample_entropy, min=0.0)

    output = {
        "mean_prediction": mean_probs,
        "predicted_classes": pred_classes,
        "variance_map": variance_map,
        "foreground_variance": foreground_variance,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
    }

    if return_samples:
        output["stochastic_samples"] = stacked_probs

    return output
