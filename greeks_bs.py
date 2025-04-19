import math
from statistics import NormalDist

import numpy as np
from scipy.stats import norm

# Functions for Greeks on european options paying a no dividend (Black-Scholes)

# d1 = ( math.log( s/k ) + ( r + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * math.sqrt(t))
# d2 = d1 - sigma * math.sqrt(t)

def delta_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, d1: float = None) -> float:
    """
    Calculates the delta of a european call option, whose underlying pays no dividend, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The delta of the option
    """

    if d1 == None:
        d1 = (math.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * math.sqrt(t))

    delta = NormalDist().cdf(d1)

    return delta


def delta_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, d1: np.array = None) -> np.array:
    """
    Calculates the delta of a european call option, whose underlying pays no dividend, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        d1 (_np.array_): The precalculated value for d1
    Returns:
        np.array: The delta of the option
    """

    if d1 == None:
        d1 = (np.log(s / k) + (r  + (sigma ** 2 / 2)) * t) / (sigma * np.sqrt(t))
    
    delta = norm.cdf(d1)

    return delta


def delta_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, d1: float = None) -> float:
    """
    Calculates the delta of a european put option, whose underlying pays no dividend, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The delta of the option
    """

    if d1 == None:
        d1 = (math.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * math.sqrt(t))
    
    delta = NormalDist().cdf(d1) - 1

    return delta


def delta_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, d1: np.array = None) -> np.array:
    """
    Calculates the delta of a european put option, whose underlying pays no dividend, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        d1 (_np.array_): The precalculated value for d1

    Returns:
        np.array: The delta of the option
    """

    if d1 == None:
        d1 = (np.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * np.sqrt(t))

    delta = norm.cdf(d1) - 1

    return delta


def delta_stdlib(s: float, k: float, r: float, sigma: float, t: float, flag: str, d1: float = None):
    """
    Calculates the delta of a european option (call or put), whose underlying pays no dividend, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        flag (_str_): Determines the type of the option
        d1 (_float_): The recalculated value for d1

    Returns:
        float: The delta of the option
    """

    if flag.lower() == "c":
        return delta_call_stdlib(s, k, r, sigma, t, d1)
    elif flag.lower() == "p":
        return delta_put_stdlib(s, k, r, sigma, t, d1)
    else:
        raise ValueError("Invalid option type")


def delta_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, flag: str, d1: np.array = None) -> np.array:
    """
    Calculates the delta of a european option (call or put), whose underlying pays no dividend, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1
        
    Returns:
        np.array: The delta of the option
    """

    if flag.lower() == "c":
        return	delta_call_vector(s, k, r, sigma, t, d1)
    elif flag.lower() == "p":
        return	delta_put_vector(s, k, r, sigma, t, d1)
    else:
        raise ValueError("Invalid option type")