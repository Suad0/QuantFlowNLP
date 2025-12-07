"""Unit tests for quantitative finance service."""

import math
from datetime import datetime, timedelta

import pytest

from app.models.domain.quant import (
    BondParams,
    Greeks,
    MarketRate,
    OptionParams,
)
from app.services.black_scholes import BlackScholesCalculator
from app.services.quantitative_finance import QuantitativeFinanceService


class TestBlackScholesCalculator:
    """Test Black-Scholes calculator."""

    def test_call_option_price_at_the_money(self) -> None:
        """Test call option price when spot equals strike."""
        calculator = BlackScholesCalculator()
        price = calculator.price(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        # At-the-money call should have positive value
        assert price > 0
        # Approximate expected value
        assert 8 < price < 12

    def test_put_option_price_at_the_money(self) -> None:
        """Test put option price when spot equals strike."""
        calculator = BlackScholesCalculator()
        price = calculator.price(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        # At-the-money put should have positive value
        assert price > 0
        # Approximate expected value
        assert 5 < price < 9

    def test_call_option_in_the_money(self) -> None:
        """Test call option price when spot > strike."""
        calculator = BlackScholesCalculator()
        price = calculator.price(
            spot=110.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        # In-the-money call should be worth at least intrinsic value
        intrinsic_value = 110.0 - 100.0
        assert price >= intrinsic_value

    def test_put_option_in_the_money(self) -> None:
        """Test put option price when spot < strike."""
        calculator = BlackScholesCalculator()
        price = calculator.price(
            spot=90.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        # In-the-money put should be worth at least intrinsic value
        intrinsic_value = 100.0 - 90.0
        assert price >= intrinsic_value

    def test_option_price_invalid_spot(self) -> None:
        """Test option pricing with invalid spot price."""
        calculator = BlackScholesCalculator()
        with pytest.raises(ValueError, match="Spot price must be positive"):
            calculator.price(
                spot=-100.0,
                strike=100.0,
                time_to_maturity=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
            )

    def test_option_price_invalid_strike(self) -> None:
        """Test option pricing with invalid strike price."""
        calculator = BlackScholesCalculator()
        with pytest.raises(ValueError, match="Strike price must be positive"):
            calculator.price(
                spot=100.0,
                strike=0.0,
                time_to_maturity=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
            )

    def test_option_price_invalid_time(self) -> None:
        """Test option pricing with invalid time to maturity."""
        calculator = BlackScholesCalculator()
        with pytest.raises(ValueError, match="Time to maturity must be positive"):
            calculator.price(
                spot=100.0,
                strike=100.0,
                time_to_maturity=0.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
            )

    def test_option_price_invalid_volatility(self) -> None:
        """Test option pricing with invalid volatility."""
        calculator = BlackScholesCalculator()
        with pytest.raises(ValueError, match="Volatility must be positive"):
            calculator.price(
                spot=100.0,
                strike=100.0,
                time_to_maturity=1.0,
                risk_free_rate=0.05,
                volatility=0.0,
                option_type="call",
            )

    def test_greeks_call_delta_range(self) -> None:
        """Test that call delta is in valid range [0, 1]."""
        calculator = BlackScholesCalculator()
        greeks = calculator.greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        assert 0 <= greeks.delta <= 1

    def test_greeks_put_delta_range(self) -> None:
        """Test that put delta is in valid range [-1, 0]."""
        calculator = BlackScholesCalculator()
        greeks = calculator.greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        assert -1 <= greeks.delta <= 0

    def test_greeks_gamma_positive(self) -> None:
        """Test that gamma is always positive."""
        calculator = BlackScholesCalculator()
        greeks = calculator.greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        assert greeks.gamma > 0

    def test_greeks_vega_positive(self) -> None:
        """Test that vega is always positive."""
        calculator = BlackScholesCalculator()
        greeks = calculator.greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        assert greeks.vega > 0

    def test_greeks_call_rho_positive(self) -> None:
        """Test that call rho is positive."""
        calculator = BlackScholesCalculator()
        greeks = calculator.greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        assert greeks.rho > 0

    def test_greeks_put_rho_negative(self) -> None:
        """Test that put rho is negative."""
        calculator = BlackScholesCalculator()
        greeks = calculator.greeks(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        assert greeks.rho < 0


@pytest.mark.asyncio
class TestQuantitativeFinanceService:
    """Test quantitative finance service."""

    async def test_price_option_call(self) -> None:
        """Test option pricing for call option."""
        service = QuantitativeFinanceService()
        option_params = OptionParams(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )

        result = await service.price_option(option_params)

        assert result.price > 0
        assert result.method == "Black-Scholes"
        assert result.params == option_params
        assert isinstance(result.greeks, Greeks)

    async def test_price_option_put(self) -> None:
        """Test option pricing for put option."""
        service = QuantitativeFinanceService()
        option_params = OptionParams(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )

        result = await service.price_option(option_params)

        assert result.price > 0
        assert result.method == "Black-Scholes"
        assert result.params == option_params
        assert isinstance(result.greeks, Greeks)

    async def test_calculate_greeks(self) -> None:
        """Test Greeks calculation."""
        service = QuantitativeFinanceService()
        option_params = OptionParams(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )

        greeks = await service.calculate_greeks(option_params)

        assert isinstance(greeks, Greeks)
        assert 0 <= greeks.delta <= 1
        assert greeks.gamma > 0
        assert greeks.vega > 0
        assert greeks.rho > 0

    @pytest.mark.skipif(
        True, reason="QuantLib not available in test environment"
    )
    async def test_price_bond_from_yield(self) -> None:
        """Test bond pricing from yield."""
        service = QuantitativeFinanceService()
        maturity = datetime.now() + timedelta(days=365 * 5)
        bond_params = BondParams(
            face_value=1000.0,
            coupon_rate=0.05,
            maturity=maturity,
            frequency=2,
            yield_rate=0.04,
        )

        result = await service.price_bond(bond_params)

        assert result.price is not None
        assert result.price > 0
        assert result.method == "QuantLib"

    @pytest.mark.skipif(
        True, reason="QuantLib not available in test environment"
    )
    async def test_price_bond_from_price(self) -> None:
        """Test yield calculation from bond price."""
        service = QuantitativeFinanceService()
        maturity = datetime.now() + timedelta(days=365 * 5)
        bond_params = BondParams(
            face_value=1000.0,
            coupon_rate=0.05,
            maturity=maturity,
            frequency=2,
            price=1050.0,
        )

        result = await service.price_bond(bond_params)

        assert result.yield_to_maturity is not None
        assert result.yield_to_maturity > 0
        assert result.method == "QuantLib"

    async def test_price_bond_invalid_params(self) -> None:
        """Test bond pricing with invalid parameters."""
        service = QuantitativeFinanceService()
        maturity = datetime.now() + timedelta(days=365 * 5)
        bond_params = BondParams(
            face_value=1000.0,
            coupon_rate=0.05,
            maturity=maturity,
            frequency=2,
            # Neither yield nor price provided
        )

        with pytest.raises(ValueError, match="Must provide either yield_rate or price"):
            await service.price_bond(bond_params)

    async def test_price_bond_both_params(self) -> None:
        """Test bond pricing with both yield and price provided."""
        service = QuantitativeFinanceService()
        maturity = datetime.now() + timedelta(days=365 * 5)
        bond_params = BondParams(
            face_value=1000.0,
            coupon_rate=0.05,
            maturity=maturity,
            frequency=2,
            yield_rate=0.04,
            price=1050.0,
        )

        with pytest.raises(ValueError, match="Cannot provide both yield_rate and price"):
            await service.price_bond(bond_params)

    @pytest.mark.skipif(
        True, reason="QuantLib not available in test environment"
    )
    async def test_bootstrap_yield_curve(self) -> None:
        """Test yield curve bootstrapping."""
        service = QuantitativeFinanceService()
        market_rates = [
            MarketRate(maturity=0.25, rate=0.02, instrument_type="deposit"),
            MarketRate(maturity=0.5, rate=0.025, instrument_type="deposit"),
            MarketRate(maturity=1.0, rate=0.03, instrument_type="deposit"),
            MarketRate(maturity=2.0, rate=0.035, instrument_type="swap"),
            MarketRate(maturity=5.0, rate=0.04, instrument_type="swap"),
        ]

        result = await service.bootstrap_yield_curve(market_rates)

        assert len(result.maturities) > 0
        assert len(result.rates) > 0
        assert len(result.maturities) == len(result.rates)
        assert result.interpolation_method == "linear"

    async def test_bootstrap_yield_curve_empty_rates(self) -> None:
        """Test yield curve bootstrapping with empty rates."""
        service = QuantitativeFinanceService()
        market_rates = []

        with pytest.raises(ValueError, match="Market rates list cannot be empty"):
            await service.bootstrap_yield_curve(market_rates)

    async def test_bootstrap_yield_curve_insufficient_rates(self) -> None:
        """Test yield curve bootstrapping with insufficient rates."""
        service = QuantitativeFinanceService()
        market_rates = [
            MarketRate(maturity=1.0, rate=0.03, instrument_type="deposit"),
        ]

        with pytest.raises(
            ValueError, match="At least 2 market rates are required for bootstrapping"
        ):
            await service.bootstrap_yield_curve(market_rates)
