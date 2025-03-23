import math
from statistics import NormalDist

import numpy as np
from scipy.stats import norm

# Functions for Greeks on european options paying a Continuous dividend yield (Black-Scholes-Merton)

def delta_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None) -> float:
    """
    Calculates the delta of a european call option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The delta of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))

    delta = math.exp(-q*t) * NormalDist().cdf(d1)

    return delta


def delta_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None) -> np.array:
    """
    Calculates the delta of a european call option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
    Returns:
        np.array: The delta of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t )  /  (sigma * np.sqrt(t))
    
    delta = np.exp(-q*t) * norm.cdf(d1)

    return delta


def delta_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None) -> float:
    """
    Calculates the delta of a european put option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The delta of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    
    delta = -math.exp(-q*t) * NormalDist().cdf(-d1)

    return delta


def delta_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None) -> np.array:
    """
    Calculates the delta of a european put option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1

    Returns:
        np.array: The delta of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))

    delta = -np.exp(-q*t) * norm.cdf(-d1)

    return delta


def delta_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, flag: str, d1: float = None):
    """
    Calculates the delta of a european option (call or put), whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The delta of the option
    """

    if flag.lower() == "c":
        return delta_call_stdlib(s, k, r, sigma, t, q, d1)
    elif flag.lower() == "p":
        return delta_put_stdlib(s, k, r, sigma, t, q, d1)
    else:
        raise ValueError("Invalid option type")


def delta_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, flag: str, d1: np.array = None) -> np.array:
    """
    Calculates the delta of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1
        
    Returns:
        np.array: The delta of the option
    """

    if flag.lower() == "c":
        return	delta_call_vector(s, k, r, sigma, t, q, d1)
    elif flag.lower() == "p":
        return	delta_put_vector(s, k, r, sigma, t, q, d1)
    else:
        raise ValueError("Invalid option type")


def vega_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None):
    """
    Calculates the vega of a european option (call or put), whose underlying pays a continuous dividend yield q, using the Python standard library
    Divide by 100 to get the vega as option price change for 1%
    
    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The vega of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))

    vega = s * math.exp(-q*t) * math.sqrt(t) * NormalDist().pdf(d1)

    return vega


def vega_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None):
    """
    Calculates the vega of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations
    Divide by 100 to get the vega as option price change for 1%
    
    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1

    Returns:
        _np.array_: The vega of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))

    vega = s * np.exp(-q*t) * np.sqrt(t) * norm.pdf(d1)

    return vega


def theta_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None, trading_days: int = None):
    """
    Calculates the theta of a european call option, whose underlying pays a continuous dividend yield q, using the Python standard library
    Returns the yearly Theta, divide by number of days to get daily theta

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        trading_days (_int_): The number of trading days in a year
        
    Returns:
        float: The theta of the option
    """
    if t <= 0:
        raise ValueError("Time to expiration (t) must be positive.")

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * math.sqrt(t)

    theta = -math.exp(-q * t) * ( (s * NormalDist().pdf(d1) * sigma) / (2 * math.sqrt(t)) ) - r * k * math.exp(-r * t) * NormalDist().cdf(d2) + q * s * math.exp(-q*t) * NormalDist().cdf(d1)

    if trading_days != None:
        theta /= float(trading_days)
    
    return theta


def theta_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None, d2: np.array = None, trading_days: int = None):
    """
    Calculates the theta of a european call option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations
    Returns the yearly Theta, divide by number of days to get daily theta

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2
        trading_days (_int_): The number of trading days in a year
        
    Returns:
        np.array: The theta of the option
    """
    if t <= 0:
        raise ValueError("Time to expiration (t) must be positive.")

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    theta = -np.exp(-q * t) * ( (s * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) ) - r * k * np.exp(-r * t) * norm.cdf(d2) + q * s * np.exp(-q*t) * norm.cdf(d1)

    if trading_days != None:
        theta /= np.array(trading_days)
    
    return theta


def theta_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None, trading_days: int = None):
    """
    Calculates the theta of a european put option, whose underlying pays a continuous dividend yield q, using the Python standard library
    Returns the yearly Theta, divide by number of days to get daily theta

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        trading_days (_int_): The number of trading days in a year
        
    Returns:
        float: The theta of the option
    """
    if t <= 0:
        raise ValueError("Time to expiration (t) must be positive.")

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * math.sqrt(t)

    theta = -math.exp(-q * t) * ( (s * NormalDist().pdf(d1) * sigma) / (2 * math.sqrt(t)) ) + r * k * math.exp(-r * t) * NormalDist().cdf(-d2) - q * s * math.exp(-q*t) * NormalDist().cdf(-d1)

    if trading_days != None:
        theta /= float(trading_days)
    
    return theta


def theta_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None, d2: np.array = None, trading_days: int = None):
    """
    Calculates the theta of a european put option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations
    Returns the yearly Theta, divide by number of days to get daily theta

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2
        trading_days (_int_): The number of trading days in a year
        
    Returns:
        np.array: The theta of the option
    """
    if t <= 0:
        raise ValueError("Time to expiration (t) must be positive.")

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    theta = -np.exp(-q * t) * ( (s * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) ) + r * k * np.exp(-r * t) * norm.cdf(-d2) - q * s * np.exp(-q*t) * norm.cdf(-d1)

    if trading_days != None:
        theta /= np.array(trading_days)
    
    return theta


def theta_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, flag: str, d1: float = None, d2: float = None, trading_days: int = None) -> float:
    """
    Calculates the theta of a european option (call or put), whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        trading_days (_int_): The number of trading days in a year

    Returns:
        float: The theta of the option
    """

    if flag.lower() == "c":
        return theta_call_stdlib(s, k, r, sigma, t, q, d1, d2, trading_days)
    elif flag.lower() == "p":
        return theta_put_stdlib(s, k, r, sigma, t, q, d1, d2, trading_days)
    else:
        raise ValueError("Invalid option type")


def theta_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, flag: str, d1: np.array = None, d2: np.array = None, trading_days: int = None) -> np.array:
    """
    Calculates the theta of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2
        trading_days (_int_): The number of trading days in a year

    Returns:
        np.array: The theta of the option
    """

    if flag.lower() == "c":
        return	delta_call_vector(s, k, r, sigma, t, q, d1, d2, trading_days)
    elif flag.lower() == "p":
        return	delta_put_vector(s, k, r, sigma, t, q, d1, d2, trading_days)
    else:
        raise ValueError("Invalid option type")
    

def rho_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None):
    """
    Calculates the rho of a european call option, whose underlying pays a continuous dividend yield q, using the Python standard library
    Divide by 100 to get the rho as option price change for 1%

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        
    Returns:
        float: The theta of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * math.sqrt(t)

    rho = k * t * math.exp(-r*t) * NormalDist().cdf(d2)

    return rho


def rho_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None, d2: np.array = None):
    """
    Calculates the rho of a european call option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations
    Divide by 100 to get the rho as option price change for 1%

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2
        
    Returns:
        np.array: The rho of the option
    """
    if t <= 0:
        raise ValueError("Time to expiration (t) must be positive.")

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    rho = k * t * np.exp(-r*t) * norm.cdf(d2)

    return rho


def rho_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None):
    """
    Calculates the rho of a european put option, whose underlying pays a continuous dividend yield q, using the Python standard library
    Divide by 100 to get the rho as option price change for 1%

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        
    Returns:
        float: The rho of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * math.sqrt(t)

    rho = -k * t * math.exp(-r*t) * NormalDist().cdf(-d2)

    return rho


def rho_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None, d2: np.array = None):
    """
    Calculates the rho of a european put option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations
    Divide by 100 to get the rho as option price change for 1%

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2
        
    Returns:
        np.array: The rho of the option
    """
    if t <= 0:
        raise ValueError("Time to expiration (t) must be positive.")

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))
    
    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    rho = -k * t * np.exp(-r*t) * norm.cdf(-d2)

    return rho


def rho_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, flag: str, d1: float = None, d2: float = None) -> float:
    """
    Calculates the rho of a european option (call or put), whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2

    Returns:
        float: The rho of the option
    """

    if flag.lower() == "c":
        return rho_call_stdlib(s, k, r, sigma, t, q, d1, d2)
    elif flag.lower() == "p":
        return rho_put_stdlib(s, k, r, sigma, t, q, d1, d2)
    else:
        raise ValueError("Invalid option type")


def rho_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, flag: str, d1: np.array = None, d2: np.array = None) -> np.array:
    """
    Calculates the rho of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2

    Returns:
        np.array: The rho of the option
    """

    if flag.lower() == "c":
        return	rho_call_vector(s, k, r, sigma, t, q, d1, d2)
    elif flag.lower() == "p":
        return	rho_put_vector(s, k, r, sigma, t, q, d1, d2)
    else:
        raise ValueError("Invalid option type")

def epsilon_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None):
    """
    Calculates the epsilon of a european call option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        
    Returns:
        float: The epsilon of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    

    rho = -s * t * math.exp(-q*t) * NormalDist().cdf(d1)

    return rho


def epsilon_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None):
    """
    Calculates the epsilon of a european call option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
        
    Returns:
        np.array: The epsilon of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))
    
    rho = -s * t * np.exp(-q*t) * norm.cdf(d1)

    return rho


def epsilon_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None):
    """
    Calculates the epsilon of a european put option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        
    Returns:
        float: The epsilon of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))
    
    rho = s * t * math.exp(-q*t) * NormalDist().cdf(-d1)

    return rho


def epsilon_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None):
    """
    Calculates the epsilon of a european put option, whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        d1 (_np.array_): The precalculated value for d1
        
    Returns:
        np.array: The epsilon of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))
    
    rho = s * t * np.exp(-q*t) * norm.cdf(-d1)

    return rho


def epsilon_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, flag: str, d1: float = None) -> float:
    """
    Calculates the epsilon of a european option (call or put), whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_float_): The precalculated value for d1

    Returns:
        float: The epsilon of the option
    """

    if flag.lower() == "c":
        return epsilon_call_stdlib(s, k, r, sigma, t, q, d1, d2)
    elif flag.lower() == "p":
        return epsilon_put_stdlib(s, k, r, sigma, t, q, d1, d2)
    else:
        raise ValueError("Invalid option type")


def epsilon_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, flag: str, d1: np.array = None) -> np.array:
    """
    Calculates the epsilon of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1

    Returns:
        np.array: The epsilon of the option
    """

    if flag.lower() == "c":
        return	epsilon_call_vector(s, k, r, sigma, t, q, d1, d2)
    elif flag.lower() == "p":
        return	epsilon_put_vector(s, k, r, sigma, t, q, d1, d2)
    else:
        raise ValueError("Invalid option type")

def gamma_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None):
    """
    Calculates the gamma of a european (Call or Put) option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        
    Returns:
        float: The gamma of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))

    gamma = math.exp(-q*t) * ( NormalDist().pdf(d1) / (s * sigma * math.sqrt(t)) )

    return gamma


def gamma_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, flag: str, d1: np.array = None) -> np.array:
    """
    Calculates the gamma of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1

    Returns:
        np.array: The gamma of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))

    gamma = np.exp(-q*t) * ( norm.pdf(d1) / (s * sigma * np.sqrt(t)) )

    return gamma


def vanna_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None):
    """
    Calculates the vanna of a european (Call or Put) option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        
    Returns:
        float: The vanna of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    vanna = -math.exp(-q*t) * NormalDist().pdf(d1) * (d2 / sigma)

    return vanna


def vanna_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, flag: str, d1: np.array = None d2: np.array = None) -> np.array:
    """
    Calculates the vanna of a european option (call or put), whose underlying pays a continuous dividend yield q, using numpy & scipy for speed & vectorized operations

    Args:
        s (_np.array_): Current price of the underlying
        k (_np.array_): Strike price
        r (_np.array_): Risk-free rate (Annual)
        sigma (_np.array_): Volatility of the underlying (Annual)
        t (_np.array_): Time to expiration
        q (_np.array_): Continuous dividend yield
        flag (_str_): Determines the type of the option
        d1 (_np.array_): The precalculated value for d1
        d2 (_np.array_): The precalculated value for d2

    Returns:
        np.array: The vanna of the option
    """

    if d1 == None:
        d1 = ( np.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * np.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    vanna = -np.exp(-q*t) * norm.pdf(d1) * (d2 / sigma)
    
    return vanna


def charm_call_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None):
    """
    Calculates the charm of a european call option, whose underlying pays a continuous dividend yield q, using the Python standard library

    Args:
        s (_float_): Current price of the underlying
        k (_float_): Strike price
        r (_float_): Risk-free rate (Annual)
        sigma (_float_): Volatility of the underlying (Annual)
        t (_float_): Time to expiration
        q (_float_): Continuous dividend yield
        d1 (_float_): The precalculated value for d1
        d2 (_float_): The precalculated value for d2
        
    Returns:
        float: The charm of the option
    """

    if d1 == None:
        d1 = ( math.log( s/k ) + ( r - q + ( sigma ** 2 / 2 ) ) * t ) / (sigma * math.sqrt(t))

    if d2 == None:
        d2 = d1 - sigma * np.sqrt(t)

    charm = q * math.exp(-q*t) * NormalDist().cdf(d1) - math.exp(-q*t) * NormalDist().cdf(d1) * ( ( 2*(r-q)* t - d2*sigma * math.sqrt(t) ) / (2 * t * sigma * math.sqrt(t)) )

    return charm


def charm_call_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None, d2: np.array = None):
def charm_put_stdlib(s: float, k: float, r: float, sigma: float, t: float, q: float, d1: float = None, d2: float = None):
def charm_put_vector(s: np.array, k: np.array, r: np.array, sigma: np.array, t: np.array, q: np.array, d1: np.array = None, d2: np.array = None):