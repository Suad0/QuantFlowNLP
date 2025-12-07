"""Integration tests for quantitative finance service."""

from datetime import datetime, timedelta

import pytest

from app.models.domain.quant import BondParams, MarketRate, OptionParams
from app.services.quantitative_finance import QuantitativeFinanceService


@pytest.mark.asyncio
class TestQuantitativeFinanceIntegration:
    """Integration tests for quantitative finance service."""

    async def test_option_pricing_workflow(self) -> None:
        """Test complete option pricing workflow."""
        service = QuantitativeFinanceService()

        # Create option parameters
        option_params = OptionParams(
            spot=100.0,
            strike=105.0,
            time_to_maturity=0.5,
            risk_free_rate=0.03,
            volatility=0.25,
            option_type="call",
        )

        # Price the option
        result = await service.price_option(option_params)

        # Verify results
        assert result.price > 0
        assert result.method == "Black-Scholes"
        assert result.greeks.delta > 0
        assert result.greeks.gamma > 0
        assert result.greeks.vega > 0
        assert result.greeks.theta < 0  # Call theta is typically negative
        assert result.greeks.rho > 0

    async def test_greeks_calculation_workflow(self) -> None:
        """Test Greeks calculation workflow."""
        service = QuantitativeFinanceService()

        # Create option parameters for put
        option_params = OptionParams(
            spot=100.0,
            strike=95.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )

        # Calculate Greeks
        greeks = await service.calculate_greeks(option_params)

        # Verify Greeks for put option
        assert -1 <= greeks.delta <= 0  # Put delta is negative
        assert greeks.gamma > 0  # Gamma is always positive
        assert greeks.vega > 0  # Vega is always positive
        assert greeks.rho < 0  # Put rho is negative

    async def test_call_put_parity(self) -> None:
        """Test call-put parity relationship."""
        service = QuantitativeFinanceService()

        # Same parameters for call and put
        spot = 100.0
        strike = 100.0
        time_to_maturity = 1.0
        risk_free_rate = 0.05
        volatility = 0.2

        call_params = OptionParams(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type="call",
        )

        put_params = OptionParams(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type="put",
        )

        # Price both options
        call_result = await service.price_option(call_params)
        put_result = await service.price_option(put_params)

        # Verify call-put parity: C - P = S - K * e^(-r*T)
        import math

        expected_diff = spot - strike * math.exp(-risk_free_rate * time_to_maturity)
        actual_diff = call_result.price - put_result.price

        # Allow small tolerance for floating point arithmetic
        assert abs(actual_diff - expected_diff) < 0.01

    @pytest.mark.skipif(True, reason="QuantLib not available in test environment")
    async def test_bond_pricing_workflow(self) -> None:
        """Test complete bond pricing workflow."""
        service = QuantitativeFinanceService()

        # Create bond parameters
        maturity = datetime.now() + timedelta(days=365 * 10)
        bond_params = BondParams(
            face_value=1000.0,
            coupon_rate=0.06,
            maturity=maturity,
            frequency=2,
            yield_rate=0.05,
        )

        # Price the bond
        result = await service.price_bond(bond_params)

        # Verify results
        assert result.price is not None
        assert result.price > 0
        # Bond with coupon rate > yield should trade at premium
        assert result.price > 1000.0
        assert result.duration is not None
        assert result.convexity is not None

    @pytest.mark.skipif(True, reason="QuantLib not available in test environment")
    async def test_yield_curve_workflow(self) -> None:
        """Test complete yield curve bootstrapping workflow."""
        service = QuantitativeFinanceService()

        # Create market rates
        market_rates = [
            MarketRate(maturity=0.25, rate=0.015, instrument_type="deposit"),
            MarketRate(maturity=0.5, rate=0.02, instrument_type="deposit"),
            MarketRate(maturity=1.0, rate=0.025, instrument_type="deposit"),
            MarketRate(maturity=2.0, rate=0.03, instrument_type="swap"),
            MarketRate(maturity=3.0, rate=0.035, instrument_type="swap"),
            MarketRate(maturity=5.0, rate=0.04, instrument_type="swap"),
            MarketRate(maturity=10.0, rate=0.045, instrument_type="swap"),
        ]

        # Bootstrap yield curve
        result = await service.bootstrap_yield_curve(
            market_rates=market_rates, interpolation_method="linear"
        )

        # Verify results
        assert len(result.maturities) > 0
        assert len(result.rates) > 0
        assert len(result.maturities) == len(result.rates)
        assert result.interpolation_method == "linear"

        # Verify rates are increasing (normal yield curve)
        for i in range(len(result.rates) - 1):
            # Allow for small fluctuations due to interpolation
            assert result.rates[i] <= result.rates[i + 1] + 0.01

    async def test_multiple_option_pricing(self) -> None:
        """Test pricing multiple options with different parameters."""
        service = QuantitativeFinanceService()

        # Create multiple option scenarios
        scenarios = [
            # Deep in-the-money call
            OptionParams(
                spot=120.0,
                strike=100.0,
                time_to_maturity=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
            ),
            # At-the-money call
            OptionParams(
                spot=100.0,
                strike=100.0,
                time_to_maturity=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
            ),
            # Out-of-the-money call
            OptionParams(
                spot=80.0,
                strike=100.0,
                time_to_maturity=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="call",
            ),
        ]

        results = []
        for params in scenarios:
            result = await service.price_option(params)
            results.append(result)

        # Verify pricing relationships
        # Deep ITM should be most expensive
        assert results[0].price > results[1].price
        # ATM should be more expensive than OTM
        assert results[1].price > results[2].price

        # Verify delta relationships
        # Deep ITM call should have delta close to 1
        assert results[0].greeks.delta > 0.8
        # ATM call should have delta around 0.5-0.7 (depends on risk-free rate and volatility)
        assert 0.4 < results[1].greeks.delta < 0.7
        # OTM call should have delta less than ATM
        assert results[2].greeks.delta < results[1].greeks.delta
