import math


def normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function using erf.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_d1(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
) -> float | None:
    """
    Computes d1 for Black-Scholes.
    Returns None if inputs are invalid.
    """
    if (
        spot is None
        or strike is None
        or time_to_expiry_years is None
        or volatility is None
        or spot <= 0
        or strike <= 0
        or time_to_expiry_years <= 0
        or volatility <= 0
    ):
        return None

    try:
        numerator = (
            math.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry_years
        )
        denominator = volatility * math.sqrt(time_to_expiry_years)
        if denominator <= 0:
            return None
        return numerator / denominator
    except Exception:
        return None


def black_scholes_put_delta(
    spot: float,
    strike: float,
    dte: int,
    implied_volatility: float,
    risk_free_rate: float = 0.045,
) -> float | None:
    """
    Estimate European put delta using Black-Scholes.

    Parameters
    ----------
    spot : float
        Underlying stock price.
    strike : float
        Option strike price.
    dte : int
        Days to expiration.
    implied_volatility : float
        Implied volatility as a decimal (e.g. 0.25 for 25%).
    risk_free_rate : float
        Annualized risk-free rate as decimal.

    Returns
    -------
    float | None
        Estimated put delta, usually between -1 and 0.
    """
    if dte is None or dte <= 0:
        return None

    t = dte / 365.0
    d1 = black_scholes_d1(
        spot=spot,
        strike=strike,
        time_to_expiry_years=t,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
    )
    if d1 is None:
        return None

    # European put delta = N(d1) - 1
    return normal_cdf(d1) - 1.0
