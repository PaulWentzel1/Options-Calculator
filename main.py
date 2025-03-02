import math
from statistics import NormalDist


def bs_call_stdlib(s: float, k: float, r: float, sigma: float, t: float) -> float:
    """
    Calculates the price of a european call option using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration

    Returns:
        float: Price of the call option
    """

    d1 = ( math.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    
    call_price = NormalDist().cdf(d1) * s - NormalDist().cdf(d2) * k * ( math.exp(-r*t) ) 
    
    return call_price


def bs_put_stdlib(s: float, k: float, r: float, sigma: float, t: float) -> float:
    """
    Calculates the price of a european put option using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration

    Returns:
        float: Price of the put option
    """

    d1 = ( math.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    
    put_price = NormalDist().cdf(-d2) * k * (math.exp(-r*t)) - NormalDist().cdf(-d1) * s
    
    return put_price


if __name__ == "__main__":
    stock_price = 5
    strike_price = 10
    time_to_maturity = 3.5
    risk_free_rate = 0.05
    volatility = 0.20

    print(f"Call Option price: {bs_call_stdlib(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):.4f}")
    print(f"Put Option price: {bs_put_stdlib(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):.4f}")
