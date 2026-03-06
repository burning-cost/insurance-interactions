"""insurance-interactions: Automated GLM interaction detection for UK personal lines.

Uses CANN (Combined Actuarial Neural Network) and NID (Neural Interaction Detection)
to identify missing interaction terms in Poisson and Gamma GLMs.

Primary interface:
    InteractionDetector  - orchestrates the full CANN → NID → GLM test pipeline
    CANN                 - the neural network skip-connection model
    CANNConfig           - training hyperparameters
    DetectorConfig       - full pipeline configuration

GLM integration:
    test_interactions           - LR-test each candidate pair individually
    build_glm_with_interactions - refit GLM with approved interactions

References:
    Schelldorfer & Wüthrich (2019) - CANN architecture
    Tsang, Cheng & Liu (2018) - NID weight-matrix scoring
    Lindström & Palmquist (2023) - CANN+NID applied to insurance GLMs
"""

from .cann import CANN, CANNConfig
from .glm_builder import build_glm_with_interactions, test_interactions
from .nid import compute_nid_scores, nid_to_dataframe
from .selector import DetectorConfig, InteractionDetector

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

__version__ = "0.1.0"
