"""
Feature-Based and Energy-Based Out-of-Distribution (OoD) Detectors
Implements Mahalanobis distance on encoder bottleneck representations and
Energy scores to serve as rigorous baselines against uncertainty-based OoD detection.
"""

import numpy as np
import torch
import torch.nn.functional as F


class FeatureMahalanobisOOD:
    """
    Computes Mahalanobis distance in latent bottleneck feature space
    fitted on In-Distribution (ID) training representations.
    """
    def __init__(self):
        self.mean_vec = None
        self.inv_cov = None

    def fit(self, id_features):
        """
        Fits mean vector and covariance matrix on ID features (N, D).
        """
        if isinstance(id_features, torch.Tensor):
            id_features = id_features.detach().cpu().numpy()
        
        self.mean_vec = np.mean(id_features, axis=0)
        cov = np.cov(id_features, rowvar=False)
        # Add slight ridge regularization for numerical stability
        cov += 1e-4 * np.eye(cov.shape[0])
        self.inv_cov = np.linalg.pinv(cov)

    def score(self, test_features):
        """
        Computes Mahalanobis distance for test features (M, D).
        Higher distance indicates Out-of-Distribution shift.
        """
        if isinstance(test_features, torch.Tensor):
            test_features = test_features.detach().cpu().numpy()

        diff = test_features - self.mean_vec[None, :]
        # (diff @ inv_cov * diff).sum(axis=1)
        distances = np.sum(np.dot(diff, self.inv_cov) * diff, axis=1)
        return distances.astype(np.float32)


def compute_energy_score(logits, temperature=1.0):
    """
    Computes negative free energy score:
    E(x) = -T * logsumexp(logits / T).
    Higher score indicates Out-of-Distribution shift.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Average logits over spatial dimensions if 4D
    if logits.ndim == 4:
        logits = torch.mean(logits, dim=(2, 3))

    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    # Return negative energy so higher means more OOD
    return (-energy).detach().cpu().numpy()
