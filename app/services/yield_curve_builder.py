"""Yield curve bootstrapping using QuantLib."""

from datetime import datetime

try:
    import QuantLib as ql

    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

from app.models.domain.quant import MarketRate, YieldCurve


class YieldCurveBuilder:
    """
    Yield curve bootstrapping using QuantLib.

    Constructs yield curves from market rates using various interpolation methods.
    """

    def __init__(self):
        """Initialize yield curve builder."""
        if not QUANTLIB_AVAILABLE:
            raise ImportError(
                "QuantLib is not installed. Install it with: pip install QuantLib-Python"
            )

    def bootstrap_yield_curve(
        self,
        market_rates: list[MarketRate],
        reference_date: datetime | None = None,
        interpolation_method: str = "linear",
    ) -> YieldCurve:
        """
        Bootstrap yield curve from market rates.

        Args:
            market_rates: List of market rates with maturities
            reference_date: Reference date for the curve (defaults to today)
            interpolation_method: Interpolation method ("linear", "loglinear", "cubic")

        Returns:
            YieldCurve object with interpolated rates

        Raises:
            ValueError: If parameters are invalid
        """
        if not market_rates:
            raise ValueError("Market rates list cannot be empty")

        if len(market_rates) < 2:
            raise ValueError("At least 2 market rates are required for bootstrapping")

        # Set reference date
        if reference_date is None:
            reference_date = datetime.now()

        reference_ql = self._to_ql_date(reference_date)
        ql.Settings.instance().evaluationDate = reference_ql

        # Sort market rates by maturity
        sorted_rates = sorted(market_rates, key=lambda x: x.maturity)

        # Create helpers based on instrument type
        helpers = []
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        day_count = ql.ActualActual(ql.ActualActual.Bond)

        for rate in sorted_rates:
            if rate.instrument_type == "deposit":
                # Deposit rate helper
                maturity_period = self._maturity_to_period(rate.maturity)
                helper = ql.DepositRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(rate.rate)),
                    maturity_period,
                    2,  # settlement days
                    calendar,
                    ql.ModifiedFollowing,
                    False,  # end of month
                    day_count,
                )
                helpers.append(helper)

            elif rate.instrument_type == "swap":
                # Swap rate helper
                maturity_period = self._maturity_to_period(rate.maturity)
                helper = ql.SwapRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(rate.rate)),
                    maturity_period,
                    calendar,
                    ql.Semiannual,  # fixed leg frequency
                    ql.ModifiedFollowing,
                    day_count,
                    ql.Euribor6M(),  # floating leg index
                )
                helpers.append(helper)

            elif rate.instrument_type == "bond":
                # For bonds, we use a simple zero rate
                # In practice, you'd use BondHelper with actual bond details
                maturity_period = self._maturity_to_period(rate.maturity)
                helper = ql.DepositRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(rate.rate)),
                    maturity_period,
                    2,
                    calendar,
                    ql.ModifiedFollowing,
                    False,
                    day_count,
                )
                helpers.append(helper)

        # Create yield curve based on interpolation method
        if interpolation_method == "linear":
            curve = ql.PiecewiseLinearZero(reference_ql, helpers, day_count)
        elif interpolation_method == "loglinear":
            curve = ql.PiecewiseLogLinearDiscount(reference_ql, helpers, day_count)
        elif interpolation_method == "cubic":
            curve = ql.PiecewiseCubicZero(reference_ql, helpers, day_count)
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")

        # Extract rates at various maturities
        maturities = []
        rates = []

        # Generate points from 0.25 years to max maturity
        max_maturity = max(rate.maturity for rate in sorted_rates)
        step = 0.25  # quarterly points

        current_maturity = step
        while current_maturity <= max_maturity:
            maturities.append(current_maturity)
            maturity_date = reference_ql + ql.Period(int(current_maturity * 12), ql.Months)
            zero_rate = curve.zeroRate(maturity_date, day_count, ql.Continuous).rate()
            rates.append(zero_rate)
            current_maturity += step

        return YieldCurve(
            maturities=maturities,
            rates=rates,
            interpolation_method=interpolation_method,
            reference_date=reference_date,
        )

    def interpolate_rate(
        self,
        yield_curve: YieldCurve,
        maturity: float,
    ) -> float:
        """
        Interpolate rate at a specific maturity.

        Args:
            yield_curve: YieldCurve object
            maturity: Maturity in years

        Returns:
            Interpolated rate

        Raises:
            ValueError: If maturity is outside curve range
        """
        if maturity < min(yield_curve.maturities):
            raise ValueError(f"Maturity {maturity} is below minimum curve maturity")

        if maturity > max(yield_curve.maturities):
            raise ValueError(f"Maturity {maturity} is above maximum curve maturity")

        # Simple linear interpolation
        # Find surrounding points
        for i in range(len(yield_curve.maturities) - 1):
            if yield_curve.maturities[i] <= maturity <= yield_curve.maturities[i + 1]:
                # Linear interpolation
                t1, r1 = yield_curve.maturities[i], yield_curve.rates[i]
                t2, r2 = yield_curve.maturities[i + 1], yield_curve.rates[i + 1]

                weight = (maturity - t1) / (t2 - t1)
                return r1 + weight * (r2 - r1)

        # If exact match
        for i, mat in enumerate(yield_curve.maturities):
            if abs(mat - maturity) < 1e-6:
                return yield_curve.rates[i]

        raise ValueError(f"Could not interpolate rate at maturity {maturity}")

    def _to_ql_date(self, dt: datetime) -> "ql.Date":
        """Convert Python datetime to QuantLib Date."""
        return ql.Date(dt.day, dt.month, dt.year)

    def _maturity_to_period(self, maturity_years: float) -> "ql.Period":
        """Convert maturity in years to QuantLib Period."""
        if maturity_years < 1:
            # Use months for maturities less than 1 year
            months = int(maturity_years * 12)
            return ql.Period(months, ql.Months)
        else:
            # Use years for longer maturities
            years = int(maturity_years)
            return ql.Period(years, ql.Years)
