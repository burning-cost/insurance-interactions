"""insurance-interactions: Automated GLM interaction detection for UK personal lines.

Uses CANN (Combined Actuarial Neural Network) and NID (Neural Interaction Detection)
to identify missing interaction terms in Poisson and Gamma GLMs.

Primary interface:
    InteractionDetector  - orchestrates the full CANN -> NID -> GLM test pipeline
    CANN                 - the neural network skip-connection model
    CANNConfig           - training hyperparameters
    DetectorConfig       - full pipeline configuration

GLM integration:
    test_interactions           - LR-test each candidate pair individually
    build_glm_with_interactions - refit GLM with approved interactions

References:
    Schelldorfer & Wuethrich (2019) - CANN architecture
    Tsang, Cheng & Liu (2018) - NID weight-matrix scoring
    Lindstrom & Palmquist (2023) - CANN+NID applied to insurance GLMs

Note on torch dependency:
    CANN, CANNConfig, DetectorConfig, and InteractionDetector require torch.
    The other symbols (compute_nid_scores, nid_to_dataframe, test_interactions,
    build_glm_with_interactions) do not require torch and are always available.
    Install with: pip install insurance-interactions[torch]
"""

from __future__ import annotations

# Always-available exports (no torch dependency)
from .glm_builder import build_glm_with_interactions, test_interactions
from .nid import compute_nid_scores, nid_to_dataframe

# Torch-dependent exports — loaded eagerly when torch is present,
# accessible via __getattr__ with a helpful error when it is not.
_TORCH_NAMES = ("CANN", "CANNConfig", "DetectorConfig", "InteractionDetector")

try:
    from .cann import CANN, CANNConfig
    from .selector import DetectorConfig, InteractionDetector
    _torch_available = True
except ImportError:
    _torch_available = False


def __getattr__(name: str):
    if name in _TORCH_NAMES:
        if _torch_available:
            # Should not reach here if torch loaded successfully above,
            # but handle re-import edge cases defensively.
            if name == "CANN":
                from .cann import CANN
                return CANN
            if name == "CANNConfig":
                from .cann import CANNConfig
                return CANNConfig
            if name == "DetectorConfig":
                from .selector import DetectorConfig
                return DetectorConfig
            if name == "InteractionDetector":
                from .selector import InteractionDetector
                return InteractionDetector
        raise ImportError(
            f"insurance_interactions.{name} requires torch. "
            "Install with: pip install insurance-interactions[torch]"
        )
    raise AttributeError(f"module 'insurance_interactions' has no attribute {name!r}")


__all__ = [
    "CANN",
    "CANNConfig",
    "DetectorConfig",
    "InteractionDetector",
    "compute_nid_scores",
    "nid_to_dataframe",
    "test_interactions",
    "build_glm_with_interactions",
]

__version__ = "0.1.5"
