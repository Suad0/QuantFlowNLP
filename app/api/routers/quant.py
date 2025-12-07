"""Quantitative finance API router.

This module provides REST endpoints for quantitative finance utilities
including option pricing, bond pricing, and yield curve construction.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import QuantServiceDep
from app.core.exceptions import TradingSystemError
from app.models.api.quant import (
    BondPricingRequest,
    BondPricingResponse,
    GreeksResponse,
    OptionPricingRequest,
    OptionPricingResponse,
    YieldCurveRequest,
    YieldCurveResponse,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/quant/pricing/option",
    response_model=OptionPricingResponse,
    status_code=status.HTTP_200_OK,
    summary="Price options using Black-Scholes",
    description=(
        "Calculate option prices and Greeks using the Black-Scholes model. "
        "Supports both European call and put options."
    ),
)
async def price_option(
    request: OptionPricingRequest,
    quant_service: QuantServiceDep,
) -> OptionPricingResponse:
    """Price an option using Black-Scholes model.
    
    Calculates the theoretical price of a European option and its Greeks
    (delta, gamma, vega, theta, rho) using the Black-Scholes formula.
    
    Args:
        request: Option pricing parameters
        quant_service: Quantitative finance service (injected)
    
    Returns:
        Option price and Greeks
    
    Raises:
        HTTPException: If pricing fails
    """
    logger.info(
        f"Pricing {request.option_type} option",
        extra={
            "spot": request.spot,
            "strike": request.strike,
            "time_to_maturity": request.time_to_maturity,
            "volatility": request.volatility,
        },
    )
    
    try:
        # Convert request to OptionParams
        from app.models.domain.quant import OptionParams
        option_params = OptionParams(
            spot=request.spot,
            strike=request.strike,
            time_to_maturity=request.time_to_maturity,
            risk_free_rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=request.option_type,
        )
        
        # Price option
        result = await quant_service.price_option(option_params)
        
        logger.info(
            f"Option priced successfully: {result.price:.4f}",
            extra={
                "price": result.price,
                "delta": result.greeks.delta,
            },
        )
        
        # Convert to API response model
        return OptionPricingResponse(
            price=result.price,
            greeks=GreeksResponse(
                delta=result.greeks.delta,
                gamma=result.greeks.gamma,
                vega=result.greeks.vega,
                theta=result.greeks.theta,
                rho=result.greeks.rho,
            ),
            method=result.method,
            params=request,
        )
        
    except ValueError as e:
        logger.error(f"Invalid input for option pricing: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except TradingSystemError as e:
        logger.error(f"Option pricing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pricing failed: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(f"Unexpected error pricing option: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.post(
    "/quant/pricing/bond",
    response_model=BondPricingResponse,
    status_code=status.HTTP_200_OK,
    summary="Price bonds or calculate yield",
    description=(
        "Calculate bond price from yield or yield-to-maturity from price. "
        "Provide either yield_rate (for price calculation) or price (for YTM calculation)."
    ),
)
async def price_bond(
    request: BondPricingRequest,
    quant_service: QuantServiceDep,
) -> BondPricingResponse:
    """Price a bond or calculate yield-to-maturity.
    
    Calculates either:
    - Bond price from yield (if yield_rate provided)
    - Yield-to-maturity from price (if price provided)
    
    Args:
        request: Bond pricing parameters
        quant_service: Quantitative finance service (injected)
    
    Returns:
        Bond price and/or yield-to-maturity
    
    Raises:
        HTTPException: If pricing fails
    """
    logger.info(
        "Pricing bond",
        extra={
            "face_value": request.face_value,
            "coupon_rate": request.coupon_rate,
            "maturity": request.maturity,
            "has_yield": request.yield_rate is not None,
            "has_price": request.price is not None,
        },
    )
    
    try:
        # Validate input
        if request.yield_rate is None and request.price is None:
            raise ValueError("Must provide either yield_rate or price")
        
        if request.yield_rate is not None and request.price is not None:
            raise ValueError("Provide only one of yield_rate or price, not both")
        
        # Convert request to BondParams
        from app.models.domain.quant import BondParams
        bond_params = BondParams(
            face_value=request.face_value,
            coupon_rate=request.coupon_rate,
            maturity=request.maturity,
            frequency=request.frequency,
            yield_rate=request.yield_rate,
            price=request.price,
        )
        
        # Price bond
        result = await quant_service.price_bond(bond_params)
        
        logger.info(
            "Bond priced successfully",
            extra={
                "price": result.price,
                "yield_to_maturity": result.yield_to_maturity,
            },
        )
        
        # Convert to API response model
        return BondPricingResponse(
            price=result.price,
            yield_to_maturity=result.yield_to_maturity,
            duration=result.duration,
            convexity=result.convexity,
            method=result.method,
        )
        
    except ValueError as e:
        logger.error(f"Invalid input for bond pricing: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except TradingSystemError as e:
        logger.error(f"Bond pricing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pricing failed: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(f"Unexpected error pricing bond: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.post(
    "/quant/yield-curve",
    response_model=YieldCurveResponse,
    status_code=status.HTTP_200_OK,
    summary="Bootstrap yield curve",
    description=(
        "Bootstrap a yield curve from market rates using various instruments "
        "(deposits, swaps, bonds). Returns interpolated rates for standard maturities."
    ),
)
async def bootstrap_yield_curve(
    request: YieldCurveRequest,
    quant_service: QuantServiceDep,
) -> YieldCurveResponse:
    """Bootstrap yield curve from market rates.
    
    Constructs a yield curve by bootstrapping from market rates of various
    instruments. Supports linear and other interpolation methods.
    
    Args:
        request: Yield curve construction parameters
        quant_service: Quantitative finance service (injected)
    
    Returns:
        Yield curve with interpolated rates
    
    Raises:
        HTTPException: If curve construction fails
    """
    logger.info(
        f"Bootstrapping yield curve with {len(request.market_rates)} rates",
        extra={
            "num_rates": len(request.market_rates),
            "interpolation": request.interpolation_method,
        },
    )
    
    try:
        # Convert market rates to domain objects
        from app.models.domain.quant import MarketRate as DomainMarketRate
        domain_rates = [
            DomainMarketRate(
                maturity=rate.maturity,
                rate=rate.rate,
                instrument_type=rate.instrument_type,
            )
            for rate in request.market_rates
        ]
        
        # Bootstrap yield curve
        result = await quant_service.bootstrap_yield_curve(
            market_rates=domain_rates,
            reference_date=request.reference_date,
            interpolation_method=request.interpolation_method,
        )
        
        logger.info(
            f"Yield curve bootstrapped with {len(result.maturities)} points"
        )
        
        # Convert to API response model
        return YieldCurveResponse(
            maturities=result.maturities,
            rates=result.rates,
            interpolation_method=result.interpolation_method,
            reference_date=result.reference_date,
        )
        
    except ValueError as e:
        logger.error(f"Invalid input for yield curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        ) from e
        
    except TradingSystemError as e:
        logger.error(f"Yield curve construction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Curve construction failed: {str(e)}",
        ) from e
        
    except Exception as e:
        logger.error(
            f"Unexpected error constructing yield curve: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e
