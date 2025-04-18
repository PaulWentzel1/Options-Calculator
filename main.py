from black_scholes import black_scholes_stdlib


if __name__ == "__main__":
    stock_price = 5
    strike_price = 10
    time_to_maturity = 3.5
    risk_free_rate = 0.05
    volatility = 0.20

    print(f"Call Option price: {black_scholes_stdlib(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity, 'C'):.4f}")
    print(f"Put Option price: {black_scholes_stdlib(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity, 'P'):.4f}")


