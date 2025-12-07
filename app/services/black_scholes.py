"""Black-Scholes option pricing calculator."""

import math
from typing import Literal

from scipy.stats import norm

from app.models.domain.quant import Greeks, OptionParams


class BlackScholesCalculator:
    """
    Black-Scholes option pricing calculator.

    Implements the Black-Scholes-Merton model for European option pricing
    and Greeks calculation.
    """

    def price(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: Literal["call", "put"],
    ) -> float:
        """
        Calculate option price using Black-Scholes formula.

        Args:
            spot: Current spot price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate (annualized)
            volatility: Volatility of the underlying asset (annualized)
            option_type: Type of option ("call" or "put")

        Returns:
            Option price

        Raises:
            ValueError: If parameters are invalid
        """
        if spot <= 0:
            raise ValueError("Spot price must be positive")
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        if time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        d1 = self._calculate_d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
        d2 = self._calculate_d2(d1, volatility, time_to_maturity)

        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * math.exp(
                -risk_free_rate * time_to_maturity
            ) * norm.cdf(d2)
        elif option_type == "put":
            price = strike * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(
                -d2
            ) - spot * norm.cdf(-d1)
        else:
            raise ValueError(f"Invalid option type: {option_type}")

        return price

    def greeks(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: Literal["call", "put"],
    ) -> Greeks:
        """
        Calculate option Greeks.

        Args:
            spot: Current spot price of the underlying asset
            strike: Strike price of the option
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate (annualized)
            volatility: Volatility of the underlying asset (annualized)
            option_type: Type of option ("call" or "put")

        Returns:
            Greeks object containing delta, gamma, vega, theta, and rho

        Raises:
            ValueError: If parameters are invalid
        """
        if spot <= 0:
            raise ValueError("Spot price must be positive")
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        if time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        d1 = self._calculate_d1(spot, strike, time_to_maturity, risk_free_rate, volatility)
        d2 = self._calculate_d2(d1, volatility, time_to_maturity)

        # Delta
        if option_type == "call":
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time_to_maturity))

        # Vega (same for calls and puts)
        vega = spot * norm.pdf(d1) * math.sqrt(time_to_maturity) / 100  # divided by 100 for 1% change

        # Theta
        term1 = -(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_maturity))
        if option_type == "call":
            term2 = risk_free_rate * strike * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
            theta = (term1 - term2) / 365  # per day
        else:  # put
            term2 = risk_free_rate * strike * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
            theta = (term1 + term2) / 365  # per day

        # Rho
        if option_type == "call":
            rho = (
                strike
                * time_to_maturity
                * math.exp(-risk_free_rate * time_to_maturity)
                * norm.cdf(d2)
                / 100  # divided by 100 for 1% change
            )
        else:  # put
            rho = (
                -strike
                * time_to_maturity
                * math.exp(-risk_free_rate * time_to_maturity)
                * norm.cdf(-d2)
                / 100  # divided by 100 for 1% change
            )

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    def _calculate_d1(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
    ) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        return (
            math.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))

    def _calculate_d2(self, d1: float, volatility: float, time_to_maturity: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return d1 - volatility * math.sqrt(time_to_maturity)
