from __future__ import annotations
"""
AHRAS Active Deception Package
------------------------------
Exporting public API for dynamic honeypots and honey-token bait lure management.
"""

from deception.honeypot_manager import (
    DeceptionLure,
    DeceptionManager,
    get_deception_manager,
)

__all__ = [
    "DeceptionLure",
    "DeceptionManager",
    "get_deception_manager",
]
