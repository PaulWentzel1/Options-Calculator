import math
from statistics import NormalDist

import numpy as np
from scipy.stats import norm


# Functions for the regular Black-Scholes-Formula (No dividends)

def d1_d2_stdlib(s: float, k: float, r: float, sigma: float, t: float) -> tuple[float, float]:
    """
    Calculates d1 and d2 needed for the Black-Scholes Formula, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration

    Returns:
        tuple[float, float]: d1 and d2
    """

    d1 = ( math.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    return d1, d2

def d1_d2_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array) -> tuple[np.array, np.array]:
    """
    Calculates d1 and d2 needed for the Black-Scholes Formula, using numpy & scipy, for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration

    Returns:
        tuple[np.array, np.array]: d1 and d2
    """

    d1 = ( np.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    return d1, d2


def bs_call_stdlib(s: float, k: float, r: float, sigma: float, t: float) -> float:
    """
    Calculates the price of a european call option paying no dividends using the Python standard library

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


def bs_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array) -> np.array:
    """
    Calculates the price of a european call option paying no divideds using numpy & scipy, for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration

    Returns:
        np.array: Price of the call option
    """

    d1 = ( np.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    call_price = norm.cdf(d1) * s - norm.cdf(d2) * k * ( np.exp(-r*t) ) 
    
    return call_price


def bs_put_stdlib(s: float, k: float, r: float, sigma: float, t: float) -> float:
    """
    Calculates the price of a european put option  paying no dividends using the Python standard library

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

def bs_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array) -> np.array:
    """
    Calculates the price of a european put option paying no dividenss using numpy & scipy, for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration

    Returns:
        np.array: Price of the put option
    """

    d1 = ( np.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    
    put_price = norm.cdf(-d2) * k * (np.exp(-r*t)) - norm.cdf(-d1) * s
    
    return put_price




if __name__ == "__main__":
    stock_price = 5
    strike_price = 10
    time_to_maturity = 3.5
    risk_free_rate = 0.05
    volatility = 0.20

    print(f"Call Option price: {bs_call_stdlib(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):.4f}")
    print(f"Put Option price: {bs_put_stdlib(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):.4f}")