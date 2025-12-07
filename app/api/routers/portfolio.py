"""Portfolio optimization API router.

This module provides REST endpoints for portfolio optimization using
various algorithms including PyPortfolioOpt and CVXPY.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import PortfolioServiceDep
from app.core.exceptions import PortfolioOptimizationError
from app.models.api.portfolio import (
    BlackLittermanRequest,
    ConstrainedOptimizationRequest,
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    FrontierPointResponse,
    OptimizationResultResponse,
    OptimizePortfolioRequest,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/portfolio/optimize",
    response_model=OptimizationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize portfolio allocation",
    description=(
        "Optimize portfolio weights using various methods including max Sharpe ratio, "
        "minimum volatility, risk parity, and Black-Litterman. Returns optimal weights "
        "and performance metrics."
    ),
)
async def optimize_portfolio(
    request: OptimizePortfolioRequest,
    portfolio_service: PortfolioServiceDep,
) -> OptimizationResultResponse:
    """Optimize portfolio allocation.
    
    Supports multiple optimization methods:
    - max_sharpe: Maximize Sharpe ratio
    - min_volatility: Minimize portfolio volatility
    - risk_parity: Equal risk contribution
    - black_litterman: Incorporate investor views
    
    Args:
        request: Portfolio optimization request
        portfolio_service: Portfolio service (injected)
    
    Returns:
        Optimal weights and performance metrics
    
    Raises:
        HTTPException: If optimization fails
    """
    logger.info(
        f"Optimizing portfolio with method: {request.method}",
        extra={
            "symbols": request.symbols,
            "method": request.method,
            "has_expected_returns": request.expected_returns is not None,
        },
    )
    
    try:
        # Perform optimization
        result = await portfolio_service.optimize_portfolio(
            symbols=request.symbols,
            expected_returns=request.expected_returns,
            method=request.method,
            risk_free_rate=request.risk_free_rate,
        )
        
        logger.info(
            f"Portfolio optimization completed: {request.method}",
            extra={
                "sharpe_ratio": result.sharpe_ratio,
                "expected_return": result.expected_return,
                "volatility": result.volatility,
            },
        )
        
        # Convert to API response model
        return OptimizationResultResponse(
            weights=result.weights,
            expected_return=result.expected_return,
            volatility=result.volatility,
            sharpe_ratio=result.sharpe_ratio,
            method=result.method,
            metadata=result.metadata,
            optimized_at=datetime.utcnow(),
        )
        
    except PortfolioOptimizationError as e:
        logger.error(f"Portfolio optimization failed: {e}")
        
        # Check for specific error types
        if "data" in str(e).lower() or "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insufficient data for optimization: {str(e)}",
            ) from e
        elif "infeasible" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Optimization problem is infeasible: {str(e)}",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization failed: {str(e)}",
            ) from e
            
    except ValueError as e:
        logger.error(f"Invalid input for portfolio optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(
            f"Unexpected error during portfolio optimization: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.post(
    "/portfolio/solve",
    response_model=OptimizationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Solve constrained optimization problem",
    description=(
        "Solve a constrained portfolio optimization problem using CVXPY. "
        "Supports custom constraints and multiple objective functions."
    ),
)
async def solve_constrained_optimization(
    request: ConstrainedOptimizationRequest,
    portfolio_service: PortfolioServiceDep,
) -> OptimizationResultResponse:
    """Solve constrained portfolio optimization.
    
    Uses CVXPY to solve convex optimization problems with custom constraints.
    Supports objectives:
    - maximize_return: Maximize expected return
    - minimize_variance: Minimize portfolio variance
    - maximize_utility: Maximize risk-adjusted utility
    
    Args:
        request: Constrained optimization request
        portfolio_service: Portfolio service (injected)
    
    Returns:
        Optimal weights and performance metrics
    
    Raises:
        HTTPException: If optimization fails or is infeasible
    """
    logger.info(
        f"Solving constrained optimization with objective: {request.objective}",
        extra={
            "symbols": request.symbols,
            "objective": request.objective,
            "num_constraints": len(request.constraints),
        },
    )
    
    try:
        # Convert list to numpy array for covariance matrix
        import numpy as np
        cov_matrix = np.array(request.covariance_matrix)
        
        # Perform constrained optimization
        result = await portfolio_service.constrained_optimization(
            symbols=request.symbols,
            expected_returns=request.expected_returns,
            covariance_matrix=cov_matrix,
            constraints=request.constraints,
            objective=request.objective,
            risk_aversion=request.risk_aversion,
        )
        
        logger.info(
            f"Constrained optimization completed: {request.objective}",
            extra={
                "sharpe_ratio": result.sharpe_ratio,
                "expected_return": result.expected_return,
                "volatility": result.volatility,
            },
        )
        
        # Convert to API response model
        return OptimizationResultResponse(
            weights=result.weights,
            expected_return=result.expected_return,
            volatility=result.volatility,
            sharpe_ratio=result.sharpe_ratio,
            method=f"cvxpy_{request.objective}",
            metadata=result.metadata,
            optimized_at=datetime.utcnow(),
        )
        
    except PortfolioOptimizationError as e:
        logger.error(f"Constrained optimization failed: {e}")
        
        if "infeasible" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Optimization problem is infeasible. Check constraints: {str(e)}",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization failed: {str(e)}",
            ) from e
            
    except ValueError as e:
        logger.error(f"Invalid input for constrained optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(
            f"Unexpected error during constrained optimization: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/portfolio/efficient-frontier",
    response_model=EfficientFrontierResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate efficient frontier",
    description=(
        "Calculate the efficient frontier for a set of assets. "
        "Returns multiple portfolio allocations along the frontier."
    ),
)
async def get_efficient_frontier(
    request: EfficientFrontierRequest,
    portfolio_service: PortfolioServiceDep,
) -> EfficientFrontierResponse:
    """Calculate efficient frontier.
    
    Computes multiple portfolio allocations along the efficient frontier,
    showing the trade-off between risk and return.
    
    Args:
        request: Efficient frontier request
        portfolio_service: Portfolio service (injected)
    
    Returns:
        List of frontier points with weights and metrics
    
    Raises:
        HTTPException: If calculation fails
    """
    logger.info(
        f"Calculating efficient frontier for {len(request.symbols)} symbols",
        extra={
            "symbols": request.symbols,
            "num_points": request.num_points,
        },
    )
    
    try:
        # Calculate efficient frontier
        frontier_points = await portfolio_service.efficient_frontier(
            symbols=request.symbols,
            num_points=request.num_points,
            risk_free_rate=request.risk_free_rate,
        )
        
        logger.info(
            f"Efficient frontier calculated with {len(frontier_points)} points"
        )
        
        # Convert to API response models
        frontier_responses = [
            FrontierPointResponse(
                expected_return=point.expected_return,
                volatility=point.volatility,
                sharpe_ratio=point.sharpe_ratio,
                weights=point.weights,
            )
            for point in frontier_points
        ]
        
        return EfficientFrontierResponse(
            frontier_points=frontier_responses,
            symbols=request.symbols,
            risk_free_rate=request.risk_free_rate or 0.02,
            generated_at=datetime.utcnow(),
        )
        
    except PortfolioOptimizationError as e:
        logger.error(f"Efficient frontier calculation failed: {e}")
        
        if "data" in str(e).lower() or "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insufficient data for frontier calculation: {str(e)}",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Frontier calculation failed: {str(e)}",
            ) from e
            
    except ValueError as e:
        logger.error(f"Invalid input for efficient frontier: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(
            f"Unexpected error calculating efficient frontier: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e
