from __future__ import annotations
"""
AHRAS Federated Learning Hardening & Byzantine Robustness Test Suite
---------------------------------------------------------------------
Verifies gradient norm clipping, NaN/Inf rejection, and FedAvg aggregation.
"""

import unittest
import numpy as np
from federated.fed_learning import FederatedIDSServer, ModelUpdate


class TestFederatedHardening(unittest.TestCase):
    def setUp(self):
        self.server = FederatedIDSServer(min_clients=2, byzantine_clip_norm=5.0)

    def test_valid_fedavg_aggregation(self):
        up1 = ModelUpdate(
            client_id="Bank_A",
            num_samples=100,
            weights={"layer1": np.array([1.0, 2.0, 3.0]), "layer2": np.array([0.5, 0.5])},
            local_loss=0.05,
            timestamp=100.0,
        )
        up2 = ModelUpdate(
            client_id="Hospital_B",
            num_samples=100,
            weights={"layer1": np.array([3.0, 2.0, 1.0]), "layer2": np.array([1.5, 1.5])},
            local_loss=0.04,
            timestamp=101.0,
        )

        self.server.receive_update(up1)
        can_agg = self.server.receive_update(up2)
        self.assertTrue(can_agg)

        global_w = self.server.aggregate_round()
        np.testing.assert_allclose(global_w["layer1"], np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(global_w["layer2"], np.array([1.0, 1.0]))

    def test_poisoned_gradient_norm_rejection(self):
        bad_update = ModelUpdate(
            client_id="Malicious_Client",
            num_samples=50,
            weights={"layer1": np.array([1000.0, 5000.0, 9999.0])}, # Massive gradient explosion
            local_loss=10.0,
            timestamp=102.0,
        )
        accepted = self.server.receive_update(bad_update)
        self.assertFalse(accepted)

    def test_nan_update_rejection(self):
        nan_update = ModelUpdate(
            client_id="Corrupt_Client",
            num_samples=50,
            weights={"layer1": np.array([np.nan, 1.0, 2.0])},
            local_loss=0.0,
            timestamp=103.0,
        )
        accepted = self.server.receive_update(nan_update)
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
