"""Bond pricing calculator using QuantLib."""

from datetime import datetime

try:
    import QuantLib as ql

    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

from app.models.domain.quant import BondParams, BondPricing


class BondPricer:
    """
    Bond pricing calculator using QuantLib.

    Provides methods for calculating bond prices, yield to maturity,
    duration, and convexity.
    """

    def __init__(self):
        """Initialize bond pricer."""
        if not QUANTLIB_AVAILABLE:
            raise ImportError(
                "QuantLib is not installed. Install it with: pip install QuantLib-Python"
            )

    def price_from_yield(
        self,
        face_value: float,
        coupon_rate: float,
        yield_rate: float,
        maturity: datetime,
        frequency: int,
        settlement_date: datetime | None = None,
    ) -> float:
        """
        Calculate bond price from yield.

        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            yield_rate: Yield to maturity (e.g., 0.04 for 4%)
            maturity: Maturity date
            frequency: Number of coupon payments per year
            settlement_date: Settlement date (defaults to today)

        Returns:
            Bond price

        Raises:
            ValueError: If parameters are invalid
        """
        if face_value <= 0:
            raise ValueError("Face value must be positive")
        if coupon_rate < 0:
            raise ValueError("Coupon rate must be non-negative")
        if frequency <= 0:
            raise ValueError("Frequency must be positive")

        # Set settlement date
        if settlement_date is None:
            settlement_date = datetime.now()

        # Convert to QuantLib dates
        settlement_ql = self._to_ql_date(settlement_date)
        maturity_ql = self._to_ql_date(maturity)

        # Set evaluation date
        ql.Settings.instance().evaluationDate = settlement_ql

        # Create schedule
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        business_convention = ql.Unadjusted
        date_generation = ql.DateGeneration.Backward
        month_end = False

        # Determine frequency
        if frequency == 1:
            period = ql.Period(ql.Annual)
            freq = ql.Annual
        elif frequency == 2:
            period = ql.Period(ql.Semiannual)
            freq = ql.Semiannual
        elif frequency == 4:
            period = ql.Period(ql.Quarterly)
            freq = ql.Quarterly
        elif frequency == 12:
            period = ql.Period(ql.Monthly)
            freq = ql.Monthly
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        schedule = ql.Schedule(
            settlement_ql,
            maturity_ql,
            period,
            calendar,
            business_convention,
            business_convention,
            date_generation,
            month_end,
        )

        # Create fixed rate bond
        day_count = ql.ActualActual(ql.ActualActual.Bond)
        coupons = [coupon_rate]

        bond = ql.FixedRateBond(
            0,  # settlement days
            face_value,
            schedule,
            coupons,
            day_count,
        )

        # Create yield curve
        flat_forward = ql.FlatForward(settlement_ql, yield_rate, day_count, ql.Compounded, freq)
        yield_curve_handle = ql.YieldTermStructureHandle(flat_forward)

        # Create pricing engine
        bond_engine = ql.DiscountingBondEngine(yield_curve_handle)
        bond.setPricingEngine(bond_engine)

        # Return clean price
        return bond.cleanPrice()

    def yield_to_maturity(
        self,
        price: float,
        face_value: float,
        coupon_rate: float,
        maturity: datetime,
        frequency: int,
        settlement_date: datetime | None = None,
    ) -> float:
        """
        Calculate yield to maturity from bond price.

        Args:
            price: Bond price
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            maturity: Maturity date
            frequency: Number of coupon payments per year
            settlement_date: Settlement date (defaults to today)

        Returns:
            Yield to maturity

        Raises:
            ValueError: If parameters are invalid
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        if face_value <= 0:
            raise ValueError("Face value must be positive")
        if coupon_rate < 0:
            raise ValueError("Coupon rate must be non-negative")
        if frequency <= 0:
            raise ValueError("Frequency must be positive")

        # Set settlement date
        if settlement_date is None:
            settlement_date = datetime.now()

        # Convert to QuantLib dates
        settlement_ql = self._to_ql_date(settlement_date)
        maturity_ql = self._to_ql_date(maturity)

        # Set evaluation date
        ql.Settings.instance().evaluationDate = settlement_ql

        # Create schedule
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        business_convention = ql.Unadjusted
        date_generation = ql.DateGeneration.Backward
        month_end = False

        # Determine frequency
        if frequency == 1:
            period = ql.Period(ql.Annual)
            freq = ql.Annual
        elif frequency == 2:
            period = ql.Period(ql.Semiannual)
            freq = ql.Semiannual
        elif frequency == 4:
            period = ql.Period(ql.Quarterly)
            freq = ql.Quarterly
        elif frequency == 12:
            period = ql.Period(ql.Monthly)
            freq = ql.Monthly
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        schedule = ql.Schedule(
            settlement_ql,
            maturity_ql,
            period,
            calendar,
            business_convention,
            business_convention,
            date_generation,
            month_end,
        )

        # Create fixed rate bond
        day_count = ql.ActualActual(ql.ActualActual.Bond)
        coupons = [coupon_rate]

        bond = ql.FixedRateBond(
            0,  # settlement days
            face_value,
            schedule,
            coupons,
            day_count,
        )

        # Calculate yield from clean price
        # bondYield expects accuracy and max iterations as optional parameters
        ytm = bond.bondYield(day_count, ql.Compounded, freq, price)

        return ytm

    def calculate_duration_convexity(
        self,
        price: float,
        face_value: float,
        coupon_rate: float,
        maturity: datetime,
        frequency: int,
        yield_rate: float,
        settlement_date: datetime | None = None,
    ) -> tuple[float, float]:
        """
        Calculate Macaulay duration and convexity.

        Args:
            price: Bond price
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate
            maturity: Maturity date
            frequency: Number of coupon payments per year
            yield_rate: Yield to maturity
            settlement_date: Settlement date (defaults to today)

        Returns:
            Tuple of (duration, convexity)
        """
        if settlement_date is None:
            settlement_date = datetime.now()

        # Convert to QuantLib dates
        settlement_ql = self._to_ql_date(settlement_date)
        maturity_ql = self._to_ql_date(maturity)

        # Set evaluation date
        ql.Settings.instance().evaluationDate = settlement_ql

        # Create schedule
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        business_convention = ql.Unadjusted
        date_generation = ql.DateGeneration.Backward
        month_end = False

        # Determine frequency
        if frequency == 1:
            period = ql.Period(ql.Annual)
            freq = ql.Annual
        elif frequency == 2:
            period = ql.Period(ql.Semiannual)
            freq = ql.Semiannual
        elif frequency == 4:
            period = ql.Period(ql.Quarterly)
            freq = ql.Quarterly
        elif frequency == 12:
            period = ql.Period(ql.Monthly)
            freq = ql.Monthly
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        schedule = ql.Schedule(
            settlement_ql,
            maturity_ql,
            period,
            calendar,
            business_convention,
            business_convention,
            date_generation,
            month_end,
        )

        # Create fixed rate bond
        day_count = ql.ActualActual(ql.ActualActual.Bond)
        coupons = [coupon_rate]

        bond = ql.FixedRateBond(
            0,  # settlement days
            face_value,
            schedule,
            coupons,
            day_count,
        )

        # Create yield curve
        flat_forward = ql.FlatForward(settlement_ql, yield_rate, day_count, ql.Compounded, freq)
        yield_curve_handle = ql.YieldTermStructureHandle(flat_forward)

        # Create pricing engine
        bond_engine = ql.DiscountingBondEngine(yield_curve_handle)
        bond.setPricingEngine(bond_engine)

        # Calculate duration and convexity
        ytm = bond.bondYield(day_count, ql.Compounded, freq, price)
        duration = ql.BondFunctions.duration(bond, ytm, day_count, ql.Compounded, freq)
        convexity = ql.BondFunctions.convexity(bond, ytm, day_count, ql.Compounded, freq)

        return duration, convexity

    def _to_ql_date(self, dt: datetime) -> "ql.Date":
        """Convert Python datetime to QuantLib Date."""
        return ql.Date(dt.day, dt.month, dt.year)
