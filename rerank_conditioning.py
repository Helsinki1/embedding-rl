"""A training-fitted, invertible change of coordinates for frozen features."""
import torch


def fit_residualization(features, neural, minimum_scale=.001):
    """Fit using training queries only; no labels or validation tensors enter."""
    f = features.double().reshape(-1, features.shape[-1])
    centered = neural.double() - neural.double().mean(1, keepdim=True)
    h = centered.reshape(-1, neural.shape[-1])
    projection = torch.linalg.lstsq(f, h, driver="gelsd").solution.clone()
    residual = h - f @ projection
    raw_scale = residual.square().mean(0).sqrt()
    return {"projection": projection, "scale": raw_scale.clamp_min(minimum_scale), "raw_scale": raw_scale}


def residualize(features, neural, fitted):
    centered = neural.double() - neural.double().mean(1, keepdim=True)
    return (centered - features.double() @ fitted["projection"]) / fitted["scale"]


def original_coordinates(theta, delta, fitted):
    """Equivalent original-coordinate scores, up to a per-query constant."""
    original_delta = delta.double() / fitted["scale"]
    return theta.double() - fitted["projection"] @ original_delta, original_delta
