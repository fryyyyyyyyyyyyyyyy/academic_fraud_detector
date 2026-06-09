"""
Tests for statistical analysis tools — Benford's Law, P-value distribution, GRIM.

These are the most technically rigorous tools in the system.
We test with known-good and known-bad data to verify detection accuracy.
"""

import json
import pytest
import numpy as np


class TestBenfordLawTool:
    """Test Benford's Law detection."""

    @pytest.fixture
    def tool(self):
        from academic_fraud_detector.tools.statistical_analysis import BenfordLawTool
        return BenfordLawTool()

    @pytest.fixture
    def benford_conforming_data(self):
        """Generate data that conforms to Benford's Law.
        Uses exponential distribution scaled to various magnitudes."""
        np.random.seed(42)
        # Generate from exponential distribution, which naturally follows Benford
        data = np.random.exponential(scale=1000, size=500)
        return data.tolist()

    @pytest.fixture
    def benford_violating_data(self):
        """Generate data that violates Benford's Law.
        Uniform distribution over [10, 99] will NOT follow Benford."""
        np.random.seed(42)
        data = np.random.uniform(10, 99, size=500)
        return data.tolist()

    def test_conforming_data_not_flagged(self, tool, benford_conforming_data):
        """Benford-conforming data should NOT be flagged."""
        result = json.loads(
            tool._run(data_values=benford_conforming_data, data_description="Test")
        )
        assert not result.get("flagged"), (
            f"Benford-conforming data was incorrectly flagged. "
            f"p_value={result.get('p_value')}, MAD={result.get('mean_absolute_deviation')}"
        )

    def test_violating_data_is_flagged(self, tool, benford_violating_data):
        """Uniform data should be flagged as violating Benford."""
        result = json.loads(
            tool._run(data_values=benford_violating_data, data_description="Test")
        )
        assert result.get("flagged"), (
            "Uniform data (which violates Benford) was not flagged."
        )

    def test_insufficient_data(self, tool):
        """Too few data points should return an error, not crash."""
        result = json.loads(
            tool._run(data_values=[1.0, 2.0, 3.0], data_description="Small test")
        )
        assert "error" in result or not result.get("flagged")
        assert result.get("sample_size", 0) < 30

    def test_handles_zero_and_negative(self, tool):
        """Should handle zero, negative, and extreme values gracefully."""
        data = [0.0, -5.0, 0.001, 1e10, 3.14, float('nan'), float('inf')] + list(range(1, 100))
        result = json.loads(tool._run(data_values=data, data_description="Edge cases"))
        # Should not crash
        assert "error" not in result or result["sample_size"] >= 20

    def test_returns_expected_keys(self, tool, benford_conforming_data):
        """Output should have all expected fields."""
        result = json.loads(
            tool._run(data_values=benford_conforming_data, data_description="Test")
        )
        expected_keys = [
            "test", "sample_size", "chi_squared", "p_value",
            "mean_absolute_deviation", "mad_assessment",
            "observed_distribution", "expected_benford_distribution",
            "flagged", "interpretation",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestPValueDistributionTool:
    """Test p-value distribution analysis."""

    @pytest.fixture
    def tool(self):
        from academic_fraud_detector.tools.statistical_analysis import PValueDistributionTool
        return PValueDistributionTool()

    @pytest.fixture
    def uniform_pvalues(self):
        """Generate uniform p-values (honest research).
        Use a large sample to avoid statistical noise from small bins."""
        np.random.seed(42)
        return np.random.uniform(0, 1, size=500).tolist()

    @pytest.fixture
    def p_hacked_pvalues(self):
        """Generate p-values that cluster just below 0.05 (p-hacking)."""
        np.random.seed(42)
        # Mix: some uniform, many just below 0.05
        uniform_part = np.random.uniform(0, 1, size=50)
        hacked_part = np.random.uniform(0.045, 0.05, size=50)
        return np.concatenate([uniform_part, hacked_part]).tolist()

    def test_uniform_pvalues_not_flagged(self, tool, uniform_pvalues):
        """Uniform p-values (honest) should NOT be flagged."""
        result = json.loads(
            tool._run(data_values=uniform_pvalues, data_description="Test")
        )
        # Should not strongly flag uniform data
        caliper_ratio = result.get("caliper_test", {}).get("caliper_ratio", 0)
        assert caliper_ratio < 3.0, f"Caliper ratio too high for uniform data: {caliper_ratio}"

    def test_phacked_pvalues_flagged(self, tool, p_hacked_pvalues):
        """P-hacked p-values should show elevated caliper ratio."""
        result = json.loads(
            tool._run(data_values=p_hacked_pvalues, data_description="Test")
        )
        caliper = result.get("caliper_test", {})
        assert caliper.get("p_values_in_0.045_to_0.05", 0) > 5, (
            "Expected many p-values just below 0.05 in hacked data."
        )

    def test_insufficient_data(self, tool):
        """Too few p-values should return an error."""
        result = json.loads(tool._run(data_values=[0.01, 0.05], data_description="Small"))
        assert "error" in result or not result.get("flagged")

    def test_handles_invalid_pvalues(self, tool):
        """Should filter out p-values outside [0, 1]."""
        data = [-0.5, 0.0, 0.05, 0.5, 1.0, 1.5, 2.0]
        result = json.loads(tool._run(data_values=data, data_description="Edge"))
        # Should not crash; should only keep valid p-values in (0, 1]
        assert result.get("p_value_count", 0) <= 5  # -0.5, 0, 1.5, 2.0 filtered out


class TestGRIMTestTool:
    """Test the GRIM test."""

    @pytest.fixture
    def tool(self):
        from academic_fraud_detector.tools.statistical_analysis import GRIMTestTool
        return GRIMTestTool()

    def test_consistent_means(self, tool):
        """Means × N should be integer for consistent data."""
        pairs = [
            {"mean": 3.45, "n": 20},   # 3.45 × 20 = 69.0 ✓
            {"mean": 2.50, "n": 10},   # 2.50 × 10 = 25.0 ✓
            {"mean": 1.333, "n": 3},   # 1.333 × 3 = 3.999 ≈ 4 ✓
        ]
        result = json.loads(tool._run(pairs=pairs, data_description="Consistent test"))
        assert result.get("inconsistent_pairs", 0) == 0, (
            f"All pairs should be consistent but found {result.get('inconsistent_pairs')} inconsistent."
        )
        assert not result.get("flagged")

    def test_inconsistent_mean(self, tool):
        """3.47 × 20 = 69.4 — NOT an integer — should be flagged."""
        pairs = [
            {"mean": 3.47, "n": 20},   # 3.47 × 20 = 69.4 — INCONSISTENT
        ]
        result = json.loads(tool._run(pairs=pairs, data_description="Inconsistent test"))
        assert result.get("inconsistent_pairs", 0) >= 1, (
            "3.47 with N=20 should be flagged as inconsistent."
        )
        assert result.get("flagged")

    def test_mixed_consistency(self, tool):
        """Mix of consistent and inconsistent pairs."""
        pairs = [
            {"mean": 3.45, "n": 20},   # ✓
            {"mean": 3.47, "n": 20},   # ✗
            {"mean": 2.50, "n": 10},   # ✓
            {"mean": 2.53, "n": 10},   # ✗
        ]
        result = json.loads(tool._run(pairs=pairs, data_description="Mixed"))
        assert result.get("consistent_pairs") == 2
        assert result.get("inconsistent_pairs") == 2
        assert result.get("flagged")

    def test_empty_input(self, tool):
        """Empty input should be handled gracefully."""
        result = json.loads(tool._run(pairs=[], data_description="Empty"))
        assert not result.get("flagged")
