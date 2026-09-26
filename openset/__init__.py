"""
AHRAS Open-Set & Unknown Attack Generalization Package
------------------------------------------------------
Enables classification and detection of unseen zero-day cyber attacks
via OpenMax Weibull tail modeling and Helmholtz Free Energy scoring.
"""

from openset.open_max import OpenMaxEngine, OpenMaxOutput
from openset.energy_detector import EnergyBasedOODDetector, EnergyDetectionOutput
from openset.unknown_classifier import OpenSetClassifier, OpenSetVerdict

__all__ = [
    "OpenMaxEngine",
    "OpenMaxOutput",
    "EnergyBasedOODDetector",
    "EnergyDetectionOutput",
    "OpenSetClassifier",
    "OpenSetVerdict",
]
