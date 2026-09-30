from __future__ import annotations
import os

# Ensure test suite runs in DEV mode by default
os.environ.setdefault("AHRAS_ENV", "DEV")
os.environ.setdefault("AHRAS_DEV_MODE", "true")
