"""PyPortfolioOpt adapter for portfolio optimization.

This module provides a wrapper around the PyPortfolioOpt library,
implementing various portfolio optimization strategies including
mean-variance optimization, Black-Litterman, and risk parity.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pypfopt import BlackLittermanModel, EfficientFrontier, expected_returns
from pypfopt.risk_models import CovarianceShrinkage

from app.core.exceptions import PortfolioOptimizationError

logger = logging.getLogger(__name__)


class PyPortfolioOptAdapter:
    """Adapter for PyPortfolioOpt library.

    Provides a clean interface to PyPortfolioOpt functionality with
    error handling and validation.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize the adapter.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_efficient_frontier(
        self, returns: pd.DataFrame, risk_free_rate: float | None = None
    ) -> EfficientFrontier:
        """Calculate efficient frontier from historical returns.

        Args:
            returns: DataFrame of historical returns with symbols as columns
            risk_free_rate: Optional risk-free rate (uses instance default if None)

        Returns:
            EfficientFrontier object

        Raises:
            PortfolioOptimizationError: If calculation fails
        """
        try:
            # Calculate expected returns using mean historical returns
            mu = expected_returns.mean_historical_return(returns)

            # Calculate covariance matrix with shrinkage
            S = CovarianceShrinkage(returns).ledoit_wolf()

            # Create efficient frontier
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

            logger.info(
                f"Created efficient frontier for {len(returns.columns)} assets "
                f"with risk-free rate {rf_rate}"
            )

            return ef

        except Exception as e:
            logger.error(f"Failed to calculate efficient frontier: {e}")
            raise PortfolioOptimizationError(f"Failed to calculate efficient frontier: {e}") from e

    def max_sharpe_portfolio(
        self, returns: pd.DataFrame, risk_free_rate: float | None = None
    ) -> dict[str, Any]:
        """Calculate maximum Sharpe ratio portfolio.

        Args:
            returns: DataFrame of historical returns
            risk_free_rate: Optional risk-free rate

        Returns:
            Dictionary with weights and performance metrics

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            ef = self.calculate_efficient_frontier(returns, risk_free_rate)

            # Maximize Sharpe ratio
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            ef.max_sharpe(risk_free_rate=rf_rate)
            cleaned_weights = ef.clean_weights()

            # Get performance metrics
            perf = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)

            result = {
                "weights": cleaned_weights,
                "expected_return": perf[0],
                "volatility": perf[1],
                "sharpe_ratio": perf[2],
            }

            logger.info(
                f"Max Sharpe portfolio: return={perf[0]:.4f}, "
                f"volatility={perf[1]:.4f}, sharpe={perf[2]:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to calculate max Sharpe portfolio: {e}")
            raise PortfolioOptimizationError(
                f"Failed to calculate max Sharpe portfolio: {e}"
            ) from e

    def min_volatility_portfolio(
        self, returns: pd.DataFrame, risk_free_rate: float | None = None
    ) -> dict[str, Any]:
        """Calculate minimum volatility portfolio.

        Args:
            returns: DataFrame of historical returns
            risk_free_rate: Optional risk-free rate

        Returns:
            Dictionary with weights and performance metrics

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            ef = self.calculate_efficient_frontier(returns, risk_free_rate)

            # Minimize volatility
            ef.min_volatility()
            cleaned_weights = ef.clean_weights()

            # Get performance metrics
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            perf = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)

            result = {
                "weights": cleaned_weights,
                "expected_return": perf[0],
                "volatility": perf[1],
                "sharpe_ratio": perf[2],
            }

            logger.info(
                f"Min volatility portfolio: return={perf[0]:.4f}, "
                f"volatility={perf[1]:.4f}, sharpe={perf[2]:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to calculate min volatility portfolio: {e}")
            raise PortfolioOptimizationError(
                f"Failed to calculate min volatility portfolio: {e}"
            ) from e

    def efficient_return_portfolio(
        self, returns: pd.DataFrame, target_return: float, risk_free_rate: float | None = None
    ) -> dict[str, Any]:
        """Calculate portfolio with target return and minimum volatility.

        Args:
            returns: DataFrame of historical returns
            target_return: Target portfolio return
            risk_free_rate: Optional risk-free rate

        Returns:
            Dictionary with weights and performance metrics

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            ef = self.calculate_efficient_frontier(returns, risk_free_rate)

            # Efficient return
            ef.efficient_return(target_return)
            cleaned_weights = ef.clean_weights()

            # Get performance metrics
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            perf = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)

            result = {
                "weights": cleaned_weights,
                "expected_return": perf[0],
                "volatility": perf[1],
                "sharpe_ratio": perf[2],
            }

            logger.info(
                f"Efficient return portfolio (target={target_return:.4f}): "
                f"return={perf[0]:.4f}, volatility={perf[1]:.4f}, sharpe={perf[2]:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to calculate efficient return portfolio: {e}")
            raise PortfolioOptimizationError(
                f"Failed to calculate efficient return portfolio: {e}"
            ) from e

    def black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        views: dict[str, float],
        view_confidences: dict[str, float],
        market_caps: dict[str, float] | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float | None = None,
    ) -> dict[str, Any]:
        """Perform Black-Litterman optimization.

        Args:
            returns: DataFrame of historical returns
            views: Dictionary of investor views on expected returns
            view_confidences: Dictionary of confidence levels for each view
            market_caps: Optional market capitalizations for equilibrium returns
            risk_aversion: Risk aversion parameter
            tau: Uncertainty scaling factor
            risk_free_rate: Optional risk-free rate

        Returns:
            Dictionary with weights and performance metrics

        Raises:
            PortfolioOptimizationError: If optimization fails
        """
        try:
            # Calculate covariance matrix
            S = CovarianceShrinkage(returns).ledoit_wolf()

            # Prepare market caps if provided
            if market_caps is not None:
                market_caps_series = pd.Series(market_caps)
            else:
                # Use equal weights if no market caps provided
                market_caps_series = None

            # Create Black-Litterman model
            bl = BlackLittermanModel(
                cov_matrix=S,
                pi=market_caps_series,
                absolute_views=views,
                omega="idzorek",  # Use Idzorek method for omega
                view_confidences=view_confidences,
                risk_aversion=risk_aversion,
                tau=tau,
            )

            # Get posterior estimates
            ret_bl = bl.bl_returns()
            S_bl = bl.bl_cov()

            # Optimize using posterior estimates
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=rf_rate)
            cleaned_weights = ef.clean_weights()

            # Get performance metrics
            perf = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)

            result = {
                "weights": cleaned_weights,
                "expected_return": perf[0],
                "volatility": perf[1],
                "sharpe_ratio": perf[2],
                "posterior_returns": ret_bl.to_dict(),
            }

            logger.info(
                f"Black-Litterman portfolio: return={perf[0]:.4f}, "
                f"volatility={perf[1]:.4f}, sharpe={perf[2]:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to perform Black-Litterman optimization: {e}")
            raise PortfolioOptimizationError(
                f"Failed to perform Black-Litterman optimization: {e}"
            ) from e

    def risk_parity_portfolio(self, returns: pd.DataFrame) -> dict[str, float]:
        """Calculate risk parity portfolio weights.

        Risk parity allocates capital so that each asset contributes
        equally to portfolio risk.

        Args:
            returns: DataFrame of historical returns

        Returns:
            Dictionary of portfolio weights

        Raises:
            PortfolioOptimizationError: If calculation fails
        """
        try:
            # Calculate covariance matrix
            S = CovarianceShrinkage(returns).ledoit_wolf()
            cov_matrix = S.values

            # Number of assets
            n = len(returns.columns)

            # Start with equal weights
            weights = np.ones(n) / n

            # Iterative algorithm to find risk parity weights
            max_iterations = 1000
            tolerance = 1e-6

            for iteration in range(max_iterations):
                # Calculate portfolio variance
                portfolio_var = weights @ cov_matrix @ weights

                # Calculate marginal risk contributions
                marginal_risk = cov_matrix @ weights

                # Calculate risk contributions
                risk_contributions = weights * marginal_risk

                # Target risk contribution (equal for all assets)
                target_risk = portfolio_var / n

                # Update weights
                weights_new = weights * (target_risk / risk_contributions)

                # Normalize weights
                weights_new = weights_new / weights_new.sum()

                # Check convergence
                if np.max(np.abs(weights_new - weights)) < tolerance:
                    weights = weights_new
                    logger.info(f"Risk parity converged in {iteration + 1} iterations")
                    break

                weights = weights_new
            else:
                logger.warning(f"Risk parity did not converge after {max_iterations} iterations")

            # Create result dictionary
            result = {
                symbol: float(weight)
                for symbol, weight in zip(returns.columns, weights, strict=True)
            }

            # Clean very small weights
            result = {k: v for k, v in result.items() if v > 1e-4}

            # Renormalize after cleaning
            total = sum(result.values())
            result = {k: v / total for k, v in result.items()}

            logger.info(f"Risk parity portfolio calculated for {len(result)} assets")

            return result

        except Exception as e:
            logger.error(f"Failed to calculate risk parity portfolio: {e}")
            raise PortfolioOptimizationError(
                f"Failed to calculate risk parity portfolio: {e}"
            ) from e
