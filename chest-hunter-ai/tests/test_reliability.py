"""Phase 6 reliability estimation for the learned policy (calibration, opt-in gate)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chest_hunter.learning.reliability import (
    ReliabilityCalibrator,
    auc,
    brier,
    expected_calibration_error,
)


class Metrics(unittest.TestCase):
    def test_perfectly_calibrated_scores_have_zero_ece(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20000)
        y = rng.uniform(0, 1, 20000) < p
        self.assertLess(expected_calibration_error(p, y), 0.02)

    def test_overconfident_scores_have_a_large_ece(self) -> None:
        p = np.full(1000, 0.95)
        y = np.r_[np.ones(500), np.zeros(500)]
        self.assertAlmostEqual(expected_calibration_error(p, y), 0.45, places=6)
        self.assertAlmostEqual(brier(p, y), 0.5 * 0.05 ** 2 + 0.5 * 0.95 ** 2, places=9)

    def test_auc(self) -> None:
        self.assertEqual(auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])), 1.0)
        self.assertEqual(auc(np.array([0.5, 0.5]), np.array([0, 1])), 0.5)
        self.assertIsNone(auc(np.array([0.1, 0.2]), np.array([1, 1])))


class Calibrator(unittest.TestCase):
    def test_binning_corrects_a_miscalibrated_score(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < 0.5 + 0.3 * (p - 0.5)).astype(float)   # flatter truth
        cal = ReliabilityCalibrator.fit(p[:2500], y[:2500])
        test_p, test_y = p[2500:], y[2500:]
        self.assertLess(expected_calibration_error(cal.predict(test_p), test_y),
                        expected_calibration_error(test_p, test_y))

    def test_round_trip(self) -> None:
        cal = ReliabilityCalibrator.fit(np.linspace(0, 1, 100), np.linspace(0, 1, 100) > 0.5)
        path = Path(tempfile.mkdtemp()) / "reliability.json"
        cal.save(path)
        again = ReliabilityCalibrator.load(path)
        self.assertEqual(again, cal)
        self.assertTrue(np.all((again.predict(np.linspace(0, 1, 7)) >= 0)
                               & (again.predict(np.linspace(0, 1, 7)) <= 1)))

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityCalibrator.fit(np.array([]), np.array([]))


class OptInGate(unittest.TestCase):
    def _selector(self, reliability: object = None) -> object:
        from chest_hunter.learning.policy_selector import PolicySelector, SafetyLimits

        sel = PolicySelector.__new__(PolicySelector)
        sel.limits = SafetyLimits()
        sel.reliability = reliability
        sel.reliability_threshold = 0.6
        return sel

    def _out(self, p: float) -> object:
        return SimpleNamespace(phase_probability=lambda *_names: p)

    def test_without_a_calibrator_the_phase5_rule_applies(self) -> None:
        sel = self._selector()
        self.assertTrue(sel._confident(self._out(0.61)))          # type: ignore[attr-defined]
        self.assertFalse(sel._confident(self._out(0.59)))         # type: ignore[attr-defined]

    def test_a_calibrator_replaces_the_raw_score(self) -> None:
        low = ReliabilityCalibrator((0.0, 0.5, 1.0), (0.2, 0.4), (10, 10), 0.25)
        sel = self._selector(low)
        self.assertFalse(sel._confident(self._out(0.95)))         # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
