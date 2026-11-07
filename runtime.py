"""Explicit startup boundaries; no file access or scientific imports at module import."""

from functools import partial
from pathlib import Path
from typing import Any, Callable

from config import ConfigurationError


def load_public_key(path: str, *, skip_auth: bool) -> str | None:
    if skip_auth:
        return None
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

    try:
        if Path(path).suffix.lower() != ".pem":
            raise ValueError("Expected PEM file")
        material = Path(path).read_text(encoding="utf-8")
        key = serialization.load_pem_public_key(material.encode("utf-8"))
        if not isinstance(key, RSAPublicKey):
            raise ValueError("Expected RSA public key")
    except (OSError, ValueError, TypeError, UnicodeError):
        raise ConfigurationError("JWT_PUBLIC_KEY_PATH must reference a readable RSA public PEM file") from None
    return material


def load_inference_runtime(*, package_dir: str | Path | None = None) -> tuple[Any, str, Callable[..., Any]]:
    import torch
    from api.inference import load_thresholds, predict
    from mercurion.model import MercurionMLP

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, TypeError):
        pass
    if package_dir is not None:
        from importlib.metadata import version
        from mercurion.model_package import ModelPackageError, validate_package
        manifest, package_thresholds = validate_package(package_dir)
        for name, expected in manifest["libraries"].items():
            if version(name).split("+")[0] != expected.split("+")[0]:
                raise ModelPackageError(f"Incompatible inference library: {name}")
        thresholds = package_thresholds["per_label"]
        weights = Path(package_dir) / "weights.pt"
    else:
        # Explicit legacy compatibility, not evidence of coherent model provenance.
        thresholds = load_thresholds()
        weights = Path("outputs/models/best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MercurionMLP().to(device)
    model.load_state_dict(torch.load(
        weights, map_location=device, weights_only=True
    ))
    model.eval()
    return model, device, partial(predict, thresholds=thresholds)
