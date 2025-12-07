"""Quantitative finance service."""

from datetime import datetime

from app.models.domain.quant import (
    BondParams,
    BondPricing,
    Greeks,
    MarketRate,
    OptionParams,
    OptionPricing,
    YieldCurve,
)
from app.services.black_scholes import BlackScholesCalculator
from app.services.bond_pricer import BondPricer
from app.services.yield_curve_builder import YieldCurveBuilder


class QuantitativeFinanceService:
    """
    Quantitative finance service providing pricing and analytics.

    Provides methods for:
    - Option pricing using Black-Scholes
    - Bond pricing and yield calculations
    - Yield curve bootstrapping
    - Greeks calculation
    """

    def __init__(self):
        """Initialize quantitative finance service."""
        self.bs_calculator = BlackScholesCalculator()
        self.bond_pricer = BondPricer()
        self.yield_curve_builder = YieldCurveBuilder()

    async def price_option(self, option_params: OptionParams) -> OptionPricing:
        """
        Price an option using Black-Scholes model.

        Args:
            option_params: Option parameters

        Returns:
            OptionPricing with price and Greeks

        Raises:
            ValueError: If parameters are invalid
        """
        # Calculate price
        price = self.bs_calculator.price(
            spot=option_params.spot,
            strike=option_params.strike,
            time_to_maturity=option_params.time_to_maturity,
            risk_free_rate=option_params.risk_free_rate,
            volatility=option_params.volatility,
            option_type=option_params.option_type,
        )

        # Calculate Greeks
        greeks = self.bs_calculator.greeks(
            spot=option_params.spot,
            strike=option_params.strike,
            time_to_maturity=option_params.time_to_maturity,
            risk_free_rate=option_params.risk_free_rate,
            volatility=option_params.volatility,
            option_type=option_params.option_type,
        )

        return OptionPricing(
            price=price,
            greeks=greeks,
            method="Black-Scholes",
            params=option_params,
        )

    async def calculate_greeks(self, option_params: OptionParams) -> Greeks:
        """
        Calculate option Greeks.

        Args:
            option_params: Option parameters

        Returns:
            Greeks object

        Raises:
            ValueError: If parameters are invalid
        """
        return self.bs_calculator.greeks(
            spot=option_params.spot,
            strike=option_params.strike,
            time_to_maturity=option_params.time_to_maturity,
            risk_free_rate=option_params.risk_free_rate,
            volatility=option_params.volatility,
            option_type=option_params.option_type,
        )

    async def price_bond(
        self,
        bond_params: BondParams,
        settlement_date: datetime | None = None,
    ) -> BondPricing:
        """
        Price a bond or calculate yield to maturity.

        If yield_rate is provided, calculates price.
        If price is provided, calculates yield to maturity.

        Args:
            bond_params: Bond parameters
            settlement_date: Settlement date (defaults to today)

        Returns:
            BondPricing with price and/or yield

        Raises:
            ValueError: If parameters are invalid or both/neither price and yield provided
        """
        if bond_params.yield_rate is not None and bond_params.price is not None:
            raise ValueError("Cannot provide both yield_rate and price")

        if bond_params.yield_rate is None and bond_params.price is None:
            raise ValueError("Must provide either yield_rate or price")

        result = BondPricing(method="QuantLib")

        if bond_params.yield_rate is not None:
            # Calculate price from yield
            price = self.bond_pricer.price_from_yield(
                face_value=bond_params.face_value,
                coupon_rate=bond_params.coupon_rate,
                yield_rate=bond_params.yield_rate,
                maturity=bond_params.maturity,
                frequency=bond_params.frequency,
                settlement_date=settlement_date,
            )
            result.price = price

            # Calculate duration and convexity
            duration, convexity = self.bond_pricer.calculate_duration_convexity(
                price=price,
                face_value=bond_params.face_value,
                coupon_rate=bond_params.coupon_rate,
                maturity=bond_params.maturity,
                frequency=bond_params.frequency,
                yield_rate=bond_params.yield_rate,
                settlement_date=settlement_date,
            )
            result.duration = duration
            result.convexity = convexity

        else:
            # Calculate yield from price
            ytm = self.bond_pricer.yield_to_maturity(
                price=bond_params.price,
                face_value=bond_params.face_value,
                coupon_rate=bond_params.coupon_rate,
                maturity=bond_params.maturity,
                frequency=bond_params.frequency,
                settlement_date=settlement_date,
            )
            result.yield_to_maturity = ytm
            result.price = bond_params.price

            # Calculate duration and convexity
            duration, convexity = self.bond_pricer.calculate_duration_convexity(
                price=bond_params.price,
                face_value=bond_params.face_value,
                coupon_rate=bond_params.coupon_rate,
                maturity=bond_params.maturity,
                frequency=bond_params.frequency,
                yield_rate=ytm,
                settlement_date=settlement_date,
            )
            result.duration = duration
            result.convexity = convexity

        return result

    async def bootstrap_yield_curve(
        self,
        market_rates: list[MarketRate],
        reference_date: datetime | None = None,
        interpolation_method: str = "linear",
    ) -> YieldCurve:
        """
        Bootstrap yield curve from market rates.

        Args:
            market_rates: List of market rates
            reference_date: Reference date (defaults to today)
            interpolation_method: Interpolation method ("linear", "loglinear", "cubic")

        Returns:
            YieldCurve object

        Raises:
            ValueError: If parameters are invalid
        """
        return self.yield_curve_builder.bootstrap_yield_curve(
            market_rates=market_rates,
            reference_date=reference_date,
            interpolation_method=interpolation_method,
        )

    async def interpolate_yield_curve_rate(
        self,
        yield_curve: YieldCurve,
        maturity: float,
    ) -> float:
        """
        Interpolate rate from yield curve at specific maturity.

        Args:
            yield_curve: YieldCurve object
            maturity: Maturity in years

        Returns:
            Interpolated rate

        Raises:
            ValueError: If maturity is outside curve range
        """
        return self.yield_curve_builder.interpolate_rate(
            yield_curve=yield_curve,
            maturity=maturity,
        )
