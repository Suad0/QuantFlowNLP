"""Portfolio optimization service.

This module provides the main service for portfolio optimization,
orchestrating PyPortfolioOpt and CVXPY optimizers with data fetching
and result formatting.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import PortfolioOptimizationError
from app.models.domain.portfolio import (
    BlackLittermanInputs,
    Constraint,
    FrontierPoint,
    OptimizationResult,
)
from app.repositories.base import OHLCVRepository
from app.services.cvxpy_optimizer import CVXPYOptimizer
from app.services.pypfopt_adapter import PyPortfolioOptAdapter

logger = logging.getLogger(__name__)


class PortfolioOptimizationService:
    """Service for portfolio optimization.

    Orchestrates various optimization strategies including mean-variance,
    Black-Litterman, risk parity, and constrained optimization.
    """

    def __init__(
        self,
        ohlcv_repository: OHLCVRepository,
        risk_free_rate: float | None = None,
    ):
        """Initialize the service.

        Args:
            ohlcv_repository: Repository for fetching OHLCV data
            risk_free_rate: Risk-free rate (uses config default if None)
        """
        self.ohlcv_repository = ohlcv_repository
        self.risk_free_rate = (
            risk_free_rate if risk_free_rate is not None else settings.portfolio_risk_free_rate
        )
        self.pypfopt_adapter = PyPortfolioOptAdapter(risk_free_rate=self.risk_free_rate)
        self.cvxpy_optimizer = CVXPYOptimizer()

    async def optimize_portfolio(
        self,
        symbols: list[str],
        expected_returns: dict[str, float] | None = None,
        method: str = "max_sharpe",
        risk_free_rate: float | None = None,
        lookback_days: int = 252,
    ) -> OptimizationResult:
        """Optimize portfolio using specified method.

        Args:
            symbols: List of asset symbols
            expected_returns: Optional expected returns (uses historical if None)
            method: Optimization method (max_sharpe, min_volatility, risk_parity)
            risk_free_rate: Optional risk-free rate
            lookback_days: Number of days of historical data to use

        Returns:
            OptimizationResult with weights and metrics

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            logger.info(f"Starting portfolio optimization for {len(symbols)} assets using {method}")

            # Fetch historical returns
            returns_df = await self._fetch_historical_returns(symbols, lookback_days)

            # Validate we have enough data
            if len(returns_df) < 30:
                raise PortfolioOptimizationError(
                    f"Insufficient historical data: {len(returns_df)} days. Need at least 30 days."
                )

            # Use provided risk-free rate or instance default
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

            # Perform optimization based on method
            if method == "max_sharpe":
                result = self.pypfopt_adapter.max_sharpe_portfolio(returns_df, rf_rate)

            elif method == "min_volatility":
                result = self.pypfopt_adapter.min_volatility_portfolio(returns_df, rf_rate)

            elif method == "risk_parity":
                weights = self.pypfopt_adapter.risk_parity_portfolio(returns_df)
                # Calculate metrics for risk parity
                result = self._calculate_portfolio_metrics(weights, returns_df, rf_rate)

            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Create OptimizationResult
            optimization_result = OptimizationResult(
                weights=result["weights"],
                expected_return=result["expected_return"],
                volatility=result["volatility"],
                sharpe_ratio=result["sharpe_ratio"],
                method=method,
                metadata={
                    "lookback_days": lookback_days,
                    "risk_free_rate": rf_rate,
                    "num_assets": len(symbols),
                    "data_points": len(returns_df),
                },
                optimized_at=datetime.now(),
            )

            logger.info(
                f"Optimization complete: {method}, "
                f"return={result['expected_return']:.4f}, "
                f"volatility={result['volatility']:.4f}, "
                f"sharpe={result['sharpe_ratio']:.4f}"
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise PortfolioOptimizationError(f"Portfolio optimization failed: {e}") from e

    async def efficient_frontier(
        self,
        symbols: list[str],
        num_points: int = 100,
        risk_free_rate: float | None = None,
        lookback_days: int = 252,
    ) -> list[FrontierPoint]:
        """Calculate efficient frontier.

        Args:
            symbols: List of asset symbols
            num_points: Number of points on the frontier
            risk_free_rate: Optional risk-free rate
            lookback_days: Number of days of historical data

        Returns:
            List of FrontierPoint objects

        Raises:
            PortfolioOptimizationError: If calculation fails
        """
        try:
            logger.info(f"Calculating efficient frontier with {num_points} points")

            # Fetch historical returns
            returns_df = await self._fetch_historical_returns(symbols, lookback_days)

            # Use provided risk-free rate or instance default
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

            # Get min and max return portfolios
            min_vol_result = self.pypfopt_adapter.min_volatility_portfolio(returns_df, rf_rate)
            max_sharpe_result = self.pypfopt_adapter.max_sharpe_portfolio(returns_df, rf_rate)

            min_return = min_vol_result["expected_return"]
            max_return = max_sharpe_result["expected_return"]

            # Generate target returns
            target_returns = np.linspace(min_return, max_return, num_points)

            # Calculate portfolio for each target return
            frontier_points = []

            for target_return in target_returns:
                try:
                    result = self.pypfopt_adapter.efficient_return_portfolio(
                        returns_df, target_return, rf_rate
                    )

                    frontier_point = FrontierPoint(
                        expected_return=result["expected_return"],
                        volatility=result["volatility"],
                        sharpe_ratio=result["sharpe_ratio"],
                        weights=result["weights"],
                    )

                    frontier_points.append(frontier_point)

                except Exception as e:
                    logger.warning(
                        f"Failed to calculate frontier point at return {target_return}: {e}"
                    )
                    continue

            logger.info(f"Calculated {len(frontier_points)} points on efficient frontier")

            return frontier_points

        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {e}")
            raise PortfolioOptimizationError(f"Efficient frontier calculation failed: {e}") from e

    async def risk_parity(
        self,
        symbols: list[str],
        lookback_days: int = 252,
    ) -> OptimizationResult:
        """Calculate risk parity portfolio.

        Args:
            symbols: List of asset symbols
            lookback_days: Number of days of historical data

        Returns:
            OptimizationResult with risk parity weights

        Raises:
            PortfolioOptimizationError: If calculation fails
        """
        try:
            logger.info(f"Calculating risk parity portfolio for {len(symbols)} assets")

            # Fetch historical returns
            returns_df = await self._fetch_historical_returns(symbols, lookback_days)

            # Calculate risk parity weights
            weights = self.pypfopt_adapter.risk_parity_portfolio(returns_df)

            # Calculate metrics
            result = self._calculate_portfolio_metrics(weights, returns_df, self.risk_free_rate)

            # Create OptimizationResult
            optimization_result = OptimizationResult(
                weights=weights,
                expected_return=result["expected_return"],
                volatility=result["volatility"],
                sharpe_ratio=result["sharpe_ratio"],
                method="risk_parity",
                metadata={
                    "lookback_days": lookback_days,
                    "num_assets": len(symbols),
                    "data_points": len(returns_df),
                },
                optimized_at=datetime.now(),
            )

            logger.info(
                f"Risk parity portfolio: "
                f"return={result['expected_return']:.4f}, "
                f"volatility={result['volatility']:.4f}"
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Risk parity calculation failed: {e}")
            raise PortfolioOptimizationError(f"Risk parity calculation failed: {e}") from e

    async def black_litterman(
        self,
        symbols: list[str],
        bl_inputs: BlackLittermanInputs,
        risk_free_rate: float | None = None,
        lookback_days: int = 252,
    ) -> OptimizationResult:
        """Perform Black-Litterman optimization.

        Args:
            symbols: List of asset symbols
            bl_inputs: Black-Litterman inputs (views, confidences, etc.)
            risk_free_rate: Optional risk-free rate
            lookback_days: Number of days of historical data

        Returns:
            OptimizationResult with Black-Litterman weights

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            logger.info(f"Performing Black-Litterman optimization for {len(symbols)} assets")

            # Fetch historical returns
            returns_df = await self._fetch_historical_returns(symbols, lookback_days)

            # Use provided risk-free rate or instance default
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

            # Perform Black-Litterman optimization
            result = self.pypfopt_adapter.black_litterman_optimization(
                returns_df,
                bl_inputs.views,
                bl_inputs.view_confidences,
                bl_inputs.market_caps,
                bl_inputs.risk_aversion,
                bl_inputs.tau,
                rf_rate,
            )

            # Create OptimizationResult
            optimization_result = OptimizationResult(
                weights=result["weights"],
                expected_return=result["expected_return"],
                volatility=result["volatility"],
                sharpe_ratio=result["sharpe_ratio"],
                method="black_litterman",
                metadata={
                    "lookback_days": lookback_days,
                    "risk_free_rate": rf_rate,
                    "num_assets": len(symbols),
                    "data_points": len(returns_df),
                    "risk_aversion": bl_inputs.risk_aversion,
                    "tau": bl_inputs.tau,
                    "posterior_returns": result.get("posterior_returns", {}),
                },
                optimized_at=datetime.now(),
            )

            logger.info(
                f"Black-Litterman optimization complete: "
                f"return={result['expected_return']:.4f}, "
                f"volatility={result['volatility']:.4f}, "
                f"sharpe={result['sharpe_ratio']:.4f}"
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            raise PortfolioOptimizationError(f"Black-Litterman optimization failed: {e}") from e

    async def constrained_optimization(
        self,
        symbols: list[str],
        expected_returns: dict[str, float],
        covariance_matrix: list[list[float]],
        constraints: list[Constraint],
        objective: str = "maximize_return",
        risk_aversion: float = 1.0,
    ) -> OptimizationResult:
        """Perform constrained optimization using CVXPY.

        Args:
            symbols: List of asset symbols
            expected_returns: Expected returns for each symbol
            covariance_matrix: Covariance matrix as 2D list
            constraints: List of optimization constraints
            objective: Optimization objective
            risk_aversion: Risk aversion parameter

        Returns:
            OptimizationResult with optimal weights

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            logger.info(
                f"Performing constrained optimization for {len(symbols)} assets "
                f"with {len(constraints)} constraints"
            )

            # Convert inputs to numpy arrays
            returns_array = np.array([expected_returns[symbol] for symbol in symbols])
            cov_array = np.array(covariance_matrix)

            # Validate covariance matrix
            if cov_array.shape != (len(symbols), len(symbols)):
                raise ValueError(
                    f"Covariance matrix shape {cov_array.shape} "
                    f"does not match number of symbols {len(symbols)}"
                )

            # Solve optimization problem
            weights = self.cvxpy_optimizer.solve(
                symbols,
                returns_array,
                cov_array,
                constraints,
                objective,  # type: ignore[arg-type]
                risk_aversion,
            )

            # Calculate metrics
            weights_array = np.array([weights.get(symbol, 0.0) for symbol in symbols])
            portfolio_return = float(returns_array @ weights_array)
            portfolio_variance = float(weights_array @ cov_array @ weights_array)
            portfolio_volatility = float(np.sqrt(portfolio_variance))

            # Calculate Sharpe ratio
            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else 0.0
            )

            # Create OptimizationResult
            optimization_result = OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                method=f"cvxpy_{objective}",
                metadata={
                    "objective": objective,
                    "risk_aversion": risk_aversion,
                    "num_constraints": len(constraints),
                    "num_assets": len(symbols),
                },
                optimized_at=datetime.now(),
            )

            logger.info(
                f"Constrained optimization complete: {objective}, "
                f"return={portfolio_return:.4f}, "
                f"volatility={portfolio_volatility:.4f}, "
                f"sharpe={sharpe_ratio:.4f}"
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Constrained optimization failed: {e}")
            raise PortfolioOptimizationError(f"Constrained optimization failed: {e}") from e

    async def _fetch_historical_returns(
        self, symbols: list[str], lookback_days: int
    ) -> pd.DataFrame:
        """Fetch historical returns for symbols.

        Args:
            symbols: List of asset symbols
            lookback_days: Number of days to look back

        Returns:
            DataFrame with returns for each symbol

        Raises:
            PortfolioOptimizationError: If data fetching fails
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)

            # Fetch OHLCV data for each symbol
            returns_dict = {}

            for symbol in symbols:
                ohlcv_data = await self.ohlcv_repository.get_range(symbol, start_time, end_time)

                if not ohlcv_data:
                    raise PortfolioOptimizationError(
                        f"No historical data found for {symbol} "
                        f"in range {start_time} to {end_time}"
                    )

                # Extract close prices
                prices = pd.Series(
                    [data.close for data in ohlcv_data],
                    index=[data.timestamp for data in ohlcv_data],
                )

                # Calculate returns
                returns = prices.pct_change().dropna()
                returns_dict[symbol] = returns

            # Create DataFrame
            returns_df = pd.DataFrame(returns_dict)

            # Drop rows with any NaN values
            returns_df = returns_df.dropna()

            logger.info(f"Fetched {len(returns_df)} days of returns for {len(symbols)} symbols")

            return returns_df

        except Exception as e:
            logger.error(f"Failed to fetch historical returns: {e}")
            raise PortfolioOptimizationError(f"Failed to fetch historical returns: {e}") from e

    def _calculate_portfolio_metrics(
        self, weights: dict[str, float], returns_df: pd.DataFrame, risk_free_rate: float
    ) -> dict[str, Any]:
        """Calculate portfolio metrics from weights and returns.

        Args:
            weights: Portfolio weights
            returns_df: Historical returns DataFrame
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with expected_return, volatility, and sharpe_ratio
        """
        # Convert weights to array aligned with returns_df columns
        weights_array = np.array([weights.get(col, 0.0) for col in returns_df.columns])

        # Calculate expected return (annualized)
        mean_returns = returns_df.mean()
        expected_return = float(mean_returns @ weights_array * 252)

        # Calculate volatility (annualized)
        cov_matrix = returns_df.cov()
        portfolio_variance = weights_array @ cov_matrix @ weights_array
        volatility = float(np.sqrt(portfolio_variance * 252))

        # Calculate Sharpe ratio
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0

        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "weights": weights,
        }
