"""CVXPY-based constrained portfolio optimization.

This module provides convex optimization capabilities using CVXPY,
allowing for flexible constraint specification and multiple objective functions.
"""

import logging
from typing import Literal

import cvxpy as cp
import numpy as np

from app.core.exceptions import PortfolioOptimizationError
from app.models.domain.portfolio import Constraint

logger = logging.getLogger(__name__)


class CVXPYOptimizer:
    """Convex optimizer using CVXPY.

    Provides flexible portfolio optimization with custom constraints
    and objective functions.
    """

    def __init__(self) -> None:
        """Initialize the optimizer."""
        pass

    def solve(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: list[Constraint],
        objective: Literal["maximize_return", "minimize_variance", "maximize_utility"],
        risk_aversion: float = 1.0,
    ) -> dict[str, float]:
        """Solve constrained portfolio optimization problem.

        Args:
            symbols: List of asset symbols
            expected_returns: Array of expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
            constraints: List of optimization constraints
            objective: Optimization objective function
            risk_aversion: Risk aversion parameter for utility maximization

        Returns:
            Dictionary mapping symbols to optimal weights

        Raises:
            PortfolioOptimizationError: If optimization fails or is infeasible
        """
        try:
            n = len(symbols)

            # Validate inputs
            if len(expected_returns) != n:
                raise ValueError(
                    f"Expected returns length {len(expected_returns)} "
                    f"does not match number of symbols {n}"
                )

            if cov_matrix.shape != (n, n):
                raise ValueError(
                    f"Covariance matrix shape {cov_matrix.shape} "
                    f"does not match number of symbols ({n}, {n})"
                )

            # Define optimization variable
            w = cp.Variable(n)

            # Define objective function
            if objective == "maximize_return":
                # Maximize expected return
                obj = cp.Maximize(expected_returns @ w)

            elif objective == "minimize_variance":
                # Minimize portfolio variance
                obj = cp.Minimize(cp.quad_form(w, cov_matrix))  # type: ignore[attr-defined]

            elif objective == "maximize_utility":
                # Maximize utility: return - risk_aversion * variance
                portfolio_return = expected_returns @ w
                portfolio_variance = cp.quad_form(w, cov_matrix)  # type: ignore[attr-defined]
                utility = portfolio_return - risk_aversion * portfolio_variance
                obj = cp.Maximize(utility)

            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Build constraint list
            cvxpy_constraints = []

            # Always add: weights sum to 1
            cvxpy_constraints.append(cp.sum(w) == 1)  # type: ignore[attr-defined]

            # Always add: weights are non-negative (long-only)
            cvxpy_constraints.append(w >= 0)

            # Add custom constraints
            for constraint in constraints:
                if constraint.type == "weight_bounds":
                    lower = constraint.params["lower"]
                    upper = constraint.params["upper"]
                    cvxpy_constraints.append(w >= lower)
                    cvxpy_constraints.append(w <= upper)

                elif constraint.type == "leverage":
                    max_leverage = constraint.params["max"]
                    # Sum of absolute weights <= max_leverage
                    # For long-only, this is just sum(w) <= max_leverage
                    # But we already have sum(w) == 1, so this is automatically satisfied
                    # if max_leverage >= 1
                    if max_leverage < 1.0:
                        raise ValueError(
                            f"Maximum leverage {max_leverage} is less than 1.0, "
                            "which is incompatible with sum(w) == 1 constraint"
                        )

                elif constraint.type == "variance":
                    max_variance = constraint.params["max"]
                    cvxpy_constraints.append(cp.quad_form(w, cov_matrix) <= max_variance)  # type: ignore[attr-defined]

                elif constraint.type == "sector_exposure":
                    # This would require sector mapping information
                    # For now, we'll skip this as it requires additional data
                    logger.warning(
                        "Sector exposure constraints require sector mapping data, skipping"
                    )

            # Solve the problem
            problem = cp.Problem(obj, cvxpy_constraints)

            try:
                problem.solve(solver=cp.ECOS, verbose=False)  # type: ignore[no-untyped-call]
            except cp.SolverError:
                # Try alternative solver
                logger.warning("ECOS solver failed, trying SCS")
                problem.solve(solver=cp.SCS, verbose=False)  # type: ignore[no-untyped-call]

            # Check if solution was found
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # Identify which constraints might be infeasible
                infeasible_constraints = self._diagnose_infeasibility(
                    symbols, expected_returns, cov_matrix, constraints
                )

                raise PortfolioOptimizationError(
                    f"Optimization problem is {problem.status}. "
                    f"Possibly infeasible constraints: {infeasible_constraints}"
                )

            # Extract solution
            weights_array = w.value

            if weights_array is None:
                raise PortfolioOptimizationError("Solver returned None for weights")

            # Create result dictionary
            result = {
                symbol: float(weight) for symbol, weight in zip(symbols, weights_array, strict=True)
            }

            # Clean very small weights
            result = {k: v for k, v in result.items() if abs(v) > 1e-6}

            # Renormalize to ensure sum is exactly 1.0
            total = sum(result.values())
            if abs(total - 1.0) > 1e-4:
                logger.warning(f"Weights sum to {total}, renormalizing")
                result = {k: v / total for k, v in result.items()}

            # Calculate final metrics
            final_return = float(expected_returns @ weights_array)
            final_variance = float(weights_array @ cov_matrix @ weights_array)

            logger.info(
                f"CVXPY optimization successful: {objective}, "
                f"return={final_return:.4f}, variance={final_variance:.6f}, "
                f"status={problem.status}"
            )

            return result

        except cp.SolverError as e:
            logger.error(f"CVXPY solver error: {e}")
            raise PortfolioOptimizationError(f"Solver error: {e}") from e

        except Exception as e:
            logger.error(f"CVXPY optimization failed: {e}")
            raise PortfolioOptimizationError(f"Optimization failed: {e}") from e

    def _diagnose_infeasibility(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: list[Constraint],
    ) -> list[str]:
        """Diagnose which constraints might be causing infeasibility.

        Args:
            symbols: List of asset symbols
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            constraints: List of constraints

        Returns:
            List of potentially infeasible constraint descriptions
        """
        infeasible = []

        n = len(symbols)
        w = cp.Variable(n)

        # Test each constraint individually
        for i, constraint in enumerate(constraints):
            try:
                # Basic constraints
                basic_constraints = [cp.sum(w) == 1, w >= 0]  # type: ignore[attr-defined]

                # Add the constraint being tested
                if constraint.type == "weight_bounds":
                    lower = constraint.params["lower"]
                    upper = constraint.params["upper"]
                    basic_constraints.append(w >= lower)
                    basic_constraints.append(w <= upper)

                    # Check if bounds are compatible with sum(w) == 1
                    if lower * n > 1.0:
                        infeasible.append(
                            f"Constraint {i}: weight_bounds lower={lower} "
                            f"incompatible with {n} assets"
                        )
                        continue

                    if upper * n < 1.0:
                        infeasible.append(
                            f"Constraint {i}: weight_bounds upper={upper} "
                            f"incompatible with {n} assets"
                        )
                        continue

                elif constraint.type == "variance":
                    max_variance = constraint.params["max"]
                    basic_constraints.append(cp.quad_form(w, cov_matrix) <= max_variance)  # type: ignore[attr-defined]

                    # Check if minimum possible variance exceeds max_variance
                    # Minimum variance is achieved with min volatility portfolio
                    min_var_problem = cp.Problem(
                        cp.Minimize(cp.quad_form(w, cov_matrix)),  # type: ignore[attr-defined]
                        [cp.sum(w) == 1, w >= 0],  # type: ignore[attr-defined]
                    )
                    min_var_problem.solve(verbose=False)  # type: ignore[no-untyped-call]

                    if min_var_problem.status == "optimal":
                        min_variance = min_var_problem.value
                        if min_variance > max_variance:
                            infeasible.append(
                                f"Constraint {i}: variance max={max_variance} "
                                f"is less than minimum achievable variance {min_variance:.6f}"
                            )
                            continue

                # Test if constraints are feasible
                test_problem = cp.Problem(cp.Minimize(0), basic_constraints)
                test_problem.solve(verbose=False)  # type: ignore[no-untyped-call]

                if test_problem.status not in ["optimal", "optimal_inaccurate"]:
                    infeasible.append(f"Constraint {i}: {constraint.type} appears infeasible")

            except Exception as e:
                infeasible.append(f"Constraint {i}: {constraint.type} - error testing: {e}")

        return infeasible

    def calculate_efficient_frontier_cvxpy(
        self,
        symbols: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: list[Constraint],
        num_points: int = 100,
    ) -> list[dict[str, float]]:
        """Calculate efficient frontier using CVXPY.

        Args:
            symbols: List of asset symbols
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            constraints: List of constraints
            num_points: Number of points on the frontier

        Returns:
            List of dictionaries with weights for each frontier point

        Raises:
            PortfolioOptimizationError: If calculation fails
        """
        try:
            # Find minimum and maximum achievable returns
            min_return_weights = self.solve(
                symbols,
                expected_returns,
                cov_matrix,
                constraints,
                objective="minimize_variance",
            )
            min_return = sum(
                expected_returns[i] * min_return_weights.get(symbols[i], 0)
                for i in range(len(symbols))
            )

            max_return_weights = self.solve(
                symbols,
                expected_returns,
                cov_matrix,
                constraints,
                objective="maximize_return",
            )
            max_return = sum(
                expected_returns[i] * max_return_weights.get(symbols[i], 0)
                for i in range(len(symbols))
            )

            # Generate target returns
            target_returns = np.linspace(min_return, max_return, num_points)

            # Calculate optimal portfolio for each target return
            frontier_portfolios = []

            for target_return in target_returns:
                try:
                    # Solve for minimum variance at target return
                    n = len(symbols)
                    w = cp.Variable(n)

                    # Objective: minimize variance
                    obj = cp.Minimize(cp.quad_form(w, cov_matrix))  # type: ignore[attr-defined]

                    # Constraints
                    cvxpy_constraints = [
                        cp.sum(w) == 1,  # type: ignore[attr-defined]
                        w >= 0,
                        expected_returns @ w >= target_return,  # Target return constraint
                    ]

                    # Add custom constraints
                    for constraint in constraints:
                        if constraint.type == "weight_bounds":
                            lower = constraint.params["lower"]
                            upper = constraint.params["upper"]
                            cvxpy_constraints.append(w >= lower)
                            cvxpy_constraints.append(w <= upper)

                    # Solve
                    problem = cp.Problem(obj, cvxpy_constraints)
                    problem.solve(verbose=False)  # type: ignore[no-untyped-call]

                    if problem.status in ["optimal", "optimal_inaccurate"]:
                        weights_array = w.value
                        if weights_array is not None:
                            result = {
                                symbol: float(weight)
                                for symbol, weight in zip(symbols, weights_array, strict=True)
                            }
                            frontier_portfolios.append(result)

                except Exception as e:
                    logger.warning(
                        f"Failed to calculate frontier point at return {target_return}: {e}"
                    )
                    continue

            logger.info(f"Calculated {len(frontier_portfolios)} points on efficient frontier")

            return frontier_portfolios

        except Exception as e:
            logger.error(f"Failed to calculate efficient frontier: {e}")
            raise PortfolioOptimizationError(f"Failed to calculate efficient frontier: {e}") from e
