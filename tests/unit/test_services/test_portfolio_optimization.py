"""Unit tests for portfolio optimization service."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from app.models.domain.portfolio import (
    Constraint,
    OptimizationResult,
    BlackLittermanInputs,
)
from app.services.portfolio_optimization import PortfolioOptimizationService
from app.services.pypfopt_adapter import PyPortfolioOptAdapter
from app.services.cvxpy_optimizer import CVXPYOptimizer


class TestConstraint:
    """Test Constraint domain model."""

    def test_weight_bounds_constraint_valid(self) -> None:
        """Test valid weight bounds constraint."""
        constraint = Constraint(
            type="weight_bounds", params={"lower": 0.0, "upper": 0.5}
        )
        assert constraint.type == "weight_bounds"
        assert constraint.params["lower"] == 0.0
        assert constraint.params["upper"] == 0.5

    def test_weight_bounds_constraint_invalid_lower(self) -> None:
        """Test weight bounds constraint with negative lower bound."""
        with pytest.raises(ValueError, match="Lower bound must be non-negative"):
            Constraint(type="weight_bounds", params={"lower": -0.1, "upper": 0.5})

    def test_weight_bounds_constraint_invalid_upper(self) -> None:
        """Test weight bounds constraint with upper bound > 1."""
        with pytest.raises(ValueError, match="Upper bound must not exceed 1.0"):
            Constraint(type="weight_bounds", params={"lower": 0.0, "upper": 1.5})

    def test_leverage_constraint_valid(self) -> None:
        """Test valid leverage constraint."""
        constraint = Constraint(type="leverage", params={"max": 1.0})
        assert constraint.type == "leverage"
        assert constraint.params["max"] == 1.0

    def test_variance_constraint_valid(self) -> None:
        """Test valid variance constraint."""
        constraint = Constraint(type="variance", params={"max": 0.02})
        assert constraint.type == "variance"
        assert constraint.params["max"] == 0.02


class TestOptimizationResult:
    """Test OptimizationResult domain model."""

    def test_optimization_result_valid(self) -> None:
        """Test valid optimization result."""
        result = OptimizationResult(
            weights={"AAPL": 0.5, "GOOGL": 0.5},
            expected_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            method="max_sharpe",
            metadata={},
            optimized_at=datetime.now(),
        )
        assert result.weights == {"AAPL": 0.5, "GOOGL": 0.5}
        assert result.expected_return == 0.12

    def test_optimization_result_weights_sum_validation(self) -> None:
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            OptimizationResult(
                weights={"AAPL": 0.3, "GOOGL": 0.5},  # Sum = 0.8
                expected_return=0.12,
                volatility=0.15,
                sharpe_ratio=0.8,
                method="max_sharpe",
                metadata={},
                optimized_at=datetime.now(),
            )

    def test_optimization_result_negative_weight_validation(self) -> None:
        """Test that weights must be non-negative."""
        with pytest.raises(ValueError, match="must be non-negative"):
            OptimizationResult(
                weights={"AAPL": 1.2, "GOOGL": -0.2},  # Negative weight
                expected_return=0.12,
                volatility=0.15,
                sharpe_ratio=0.8,
                method="max_sharpe",
                metadata={},
                optimized_at=datetime.now(),
            )


class TestPyPortfolioOptAdapter:
    """Test PyPortfolioOpt adapter."""

    @pytest.fixture
    def sample_returns(self) -> pd.DataFrame:
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.018, 252),
                "MSFT": np.random.normal(0.0009, 0.019, 252),
            },
            index=dates,
        )
        return returns

    def test_max_sharpe_portfolio(self, sample_returns: pd.DataFrame) -> None:
        """Test maximum Sharpe ratio portfolio calculation."""
        adapter = PyPortfolioOptAdapter(risk_free_rate=0.02)
        result = adapter.max_sharpe_portfolio(sample_returns)

        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

        # Weights should sum to approximately 1
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_min_volatility_portfolio(self, sample_returns: pd.DataFrame) -> None:
        """Test minimum volatility portfolio calculation."""
        adapter = PyPortfolioOptAdapter(risk_free_rate=0.02)
        result = adapter.min_volatility_portfolio(sample_returns)

        assert "weights" in result
        assert "volatility" in result

        # Weights should sum to approximately 1
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_risk_parity_portfolio(self, sample_returns: pd.DataFrame) -> None:
        """Test risk parity portfolio calculation."""
        adapter = PyPortfolioOptAdapter(risk_free_rate=0.02)
        weights = adapter.risk_parity_portfolio(sample_returns)

        assert len(weights) > 0

        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestCVXPYOptimizer:
    """Test CVXPY optimizer."""

    @pytest.fixture
    def sample_data(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Create sample optimization data."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        expected_returns = np.array([0.12, 0.10, 0.11])
        cov_matrix = np.array(
            [[0.04, 0.01, 0.015], [0.01, 0.03, 0.012], [0.015, 0.012, 0.035]]
        )
        return symbols, expected_returns, cov_matrix

    def test_maximize_return(
        self, sample_data: tuple[list[str], np.ndarray, np.ndarray]
    ) -> None:
        """Test maximize return objective."""
        symbols, expected_returns, cov_matrix = sample_data
        optimizer = CVXPYOptimizer()

        weights = optimizer.solve(
            symbols,
            expected_returns,
            cov_matrix,
            constraints=[],
            objective="maximize_return",
        )

        assert len(weights) > 0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_minimize_variance(
        self, sample_data: tuple[list[str], np.ndarray, np.ndarray]
    ) -> None:
        """Test minimize variance objective."""
        symbols, expected_returns, cov_matrix = sample_data
        optimizer = CVXPYOptimizer()

        weights = optimizer.solve(
            symbols,
            expected_returns,
            cov_matrix,
            constraints=[],
            objective="minimize_variance",
        )

        assert len(weights) > 0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_with_weight_bounds_constraint(
        self, sample_data: tuple[list[str], np.ndarray, np.ndarray]
    ) -> None:
        """Test optimization with weight bounds constraint."""
        symbols, expected_returns, cov_matrix = sample_data
        optimizer = CVXPYOptimizer()

        constraint = Constraint(type="weight_bounds", params={"lower": 0.1, "upper": 0.5})

        weights = optimizer.solve(
            symbols,
            expected_returns,
            cov_matrix,
            constraints=[constraint],
            objective="maximize_return",
        )

        # All weights should be within bounds
        for weight in weights.values():
            assert 0.1 <= weight <= 0.5 + 0.01  # Small tolerance


class TestBlackLittermanInputs:
    """Test Black-Litterman inputs validation."""

    def test_valid_inputs(self) -> None:
        """Test valid Black-Litterman inputs."""
        inputs = BlackLittermanInputs(
            views={"AAPL": 0.15, "GOOGL": 0.12},
            view_confidences={"AAPL": 0.8, "GOOGL": 0.7},
        )
        assert inputs.views == {"AAPL": 0.15, "GOOGL": 0.12}
        assert inputs.risk_aversion == 2.5  # Default value

    def test_invalid_confidence(self) -> None:
        """Test that confidence must be in [0, 1]."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            BlackLittermanInputs(
                views={"AAPL": 0.15},
                view_confidences={"AAPL": 1.5},  # Invalid
            )

    def test_mismatched_symbols(self) -> None:
        """Test that views and confidences must have same symbols."""
        with pytest.raises(ValueError, match="must have the same symbols"):
            BlackLittermanInputs(
                views={"AAPL": 0.15, "GOOGL": 0.12},
                view_confidences={"AAPL": 0.8},  # Missing GOOGL
            )
