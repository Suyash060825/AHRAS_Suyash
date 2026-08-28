from __future__ import annotations
"""
AHRAS Federated Learning Package
-------------------------------
Exporting public API for privacy-preserving multi-tenant model aggregation.
"""

from federated.fed_learning import (
    ModelUpdate,
    FederatedIDSServer,
    get_federated_server,
)

__all__ = [
    "ModelUpdate",
    "FederatedIDSServer",
    "get_federated_server",
]
