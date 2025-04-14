import math
from statistics import NormalDist

import numpy as np
from scipy.stats import norm

def d1_d2_stdlib(s: float, k: float, r: float, sigma: float, t: float) -> tuple[float, float]:
    """
    Calculates d1 and d2 for a european option (call or put), whose underlying pays no dividends, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration

    Returns:
        tuple[float, float]: d1 and d2
    """

    d1 = ( math.log(s / k) + (r + (sigma ** 2 / 2 )) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    return d1, d2


def d1_d2_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array) -> tuple[np.array, np.array]:
    """
    Calculates d1 and d2 for a european option (call or put), whose underlying pays no dividends, for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration

    Returns:
        tuple[np.array, np.array]: d1 and d2
    """

    d1 = (np.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    return d1, d2


def black_scholes_stdlib(s: float, k: float, r: float, sigma: float, t: float, flag: str, d1: float = None, d2: float = None) -> float:
    """
    Calculates the price of a european option (call or put), whose underlying pays no dividends, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        flag (_str_): Determines the type of the option
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2

    Returns:
        float: The price of the option
    """
    
    if flag.lower() == "c":
        return bs_call_stdlib(s, k, r, sigma, t, d1, d2)
    elif flag.lower() == "p":
        return bs_put_stdlib(s, k, r, sigma, t, d1, d2)
    else:
        raise ValueError("Invalid option type")   
	

def black_scholes_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, flag: str, d1: np.array = None, d2: np.array = None) -> np.array:
    """
    Calculates the price of a european option (call or put), whose underlying pays no dividends, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        
    Returns:
        np.array: The price of the option
    """

    if flag.lower() == "c":
        return bs_call_vector(s, k, r, sigma, t, d1, d2)
    elif flag.lower() == "p":
        return bs_put_vector(s, k, r, sigma, t, d1, d2)
    else:
        raise ValueError("Invalid option type")



def bs_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, d1: float = None, d2: float = None) -> float:
    """
    Calculates the price of a european call option whose underlying pays no dividends, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2

    Returns:
        float: Price of the call option
    """

    if d1 == None:
        d1 = (math.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * math.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * math.sqrt(t)
    
    call_price = NormalDist().cdf(d1) * s - NormalDist().cdf(d2) * k * (math.exp(-r * t)) 
    
    return call_price


def bs_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, d1: np.array = None, d2: np.array = None) -> np.array:
    """
    Calculates the price of a european call option whose underlying pays no dividends, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2

    Returns:
        np.array: Price of the call option
    """
	
    if d1 == None:
        d1 = (np.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * np.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)
           
    call_price = norm.cdf(d1) * s - norm.cdf(d2) * k * (np.exp(-r * t)) 
    
    return call_price


def bs_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, d1: float = None, d2: float = None) -> float:
    """
    Calculates the price of a european put option whose underlying pays no dividends, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2

    Returns:
        float: Price of the put option
    """

    if d1 == None:
        d1 = (math.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * math.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * math.sqrt(t)
    
    put_price = NormalDist().cdf(-d2) * k * (math.exp(-r * t)) - NormalDist().cdf(-d1) * s
    
    return put_price


def bs_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, d1: np.array = None, d2: np.array = None) -> np.array:
    """
    Calculates the price of a european put option whose underlying pays no dividends, using numpy & scipy, for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2    

    Returns:
        np.array: Price of the put option
    """
	
    if d1 == None:
        d1 = (np.log(s / k) + (r + (sigma ** 2 / 2)) * t) / (sigma * np.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)    
    
    put_price = norm.cdf(-d2) * k * (np.exp(-r * t)) - norm.cdf(-d1) * s
    
    return put_price