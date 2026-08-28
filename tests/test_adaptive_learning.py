from __future__ import annotations
import unittest
from adaptive_learning.weight_learner import AdaptiveWeightLearner, FeedbackSample


class TestAdaptiveLearning(unittest.TestCase):
    def setUp(self):
        self.learner = AdaptiveWeightLearner(lr=0.05)

    def test_01_initial_weights_normalized(self):
        w = self.learner.get_weights()
        self.assertAlmostEqual(sum(w.values()), 1.0, places=3)

    def test_02_record_feedback_updates_weights(self):
        sample = FeedbackSample(
            src_ip="10.0.0.1",
            label=1,
            components={"signature": 0.9, "anomaly": 0.8, "density": 0.5, "drift_rate": 0.2},
            predicted_risk=0.5,
        )
        updated = self.learner.record_feedback(sample)
        self.assertAlmostEqual(sum(updated.values()), 1.0, places=3)
        self.assertGreater(updated["signature"], 0.0)


if __name__ == "__main__":
    unittest.main()
