from __future__ import annotations
"""
AHRAS Explainable AI & Threat Narration Package
----------------------------------------------
Exporting public API for natural language SOC incident narration and feature attribution.
"""

from xai.llm_narrator import (
    LLMNarrative,
    LLMThreatNarrator,
    get_llm_narrator,
)

__all__ = [
    "LLMNarrative",
    "LLMThreatNarrator",
    "get_llm_narrator",
]
