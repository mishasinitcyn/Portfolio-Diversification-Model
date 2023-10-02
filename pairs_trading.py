import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fetch data
def fetch_data(stock1, stock2, start_date, end_date):
    data = yf.download([stock1, stock2], start=start_date, end=end_date)
    return data['Adj Close']

# Calculate returns
def calculate_returns(data):
    return data.pct_change().dropna()

# Calculate covariance and hedge ratio
def calculate_cov_and_hedge_ratio(returns):
    cov_matrix = returns.cov()
    covariance = cov_matrix.iloc[0, 1]
    variance = returns.iloc[:, 0].var()
    hedge_ratio = covariance / variance
    return covariance, hedge_ratio

# Calculate spread
def calculate_spread(data, hedge_ratio):
    spread = data.iloc[:, 0] - hedge_ratio * data.iloc[:, 1]
    return spread

# Analyze Spread
def analyze_spread(spread):
    plt.figure(figsize=(12, 6))
    plt.plot(spread, label='Spread')
    plt.axhline(spread.mean(), color='red', linestyle='--', label='Mean Spread')
    plt.title('Spread Over Time')
    plt.legend()
    plt.show()

# Cointegration Testing using Johansen Test
def test_cointegration(data):
    johansen_test = sm.tsa.vector_ar.vecm.coint_johansen(data, det_order=0, k_ar_diff=1)
    is_cointegrated = johansen_test.lr1[0] > johansen_test.cvt[0][1]  # Using 5% significance level
    return is_cointegrated

# Generate Trading Signals
def generate_signals(spread):
    std_dev = spread.std()
    signals = []

    for value in spread:
        if value > spread.mean() + std_dev:
            signals.append('Short Stock1 / Long Stock2')
        elif value < spread.mean() - std_dev:
            signals.append('Long Stock1 / Short Stock2')
        else:
            signals.append('No Trade')

    return signals

# Backtesting function
def backtest(data, signals, hedge_ratio, initial_capital=1000000):
    capital = initial_capital
    position = 0  # Represents the amount of stock1 we hold (negative for short position)
    capital_history = [capital]
    
    for i in range(1, len(signals)):
        if signals[i] == 'Short Stock1 / Long Stock2':
            position = -capital // data.iloc[i, 0]  # Short stock1
            capital += position * data.iloc[i, 0]  # Adjust capital
            capital -= position * hedge_ratio * data.iloc[i, 1]  # Adjust capital for long stock2
        elif signals[i] == 'Long Stock1 / Short Stock2':
            position = capital // data.iloc[i, 0]  # Long stock1
            capital -= position * data.iloc[i, 0]  # Adjust capital
            capital += position * hedge_ratio * data.iloc[i, 1]  # Adjust capital for short stock2
        elif signals[i] == 'No Trade' and position != 0:
            # Closing positions
            capital -= position * data.iloc[i, 0]  # Adjust capital for stock1
            capital += position * hedge_ratio * data.iloc[i, 1]  # Adjust capital for stock2
            position = 0
        print(signals[i])
        capital_history.append(capital)
    
    return capital_history

# Sample execution
stock1 = 'MSFT'
stock2 = 'GOOGL'
start_date = '2020-01-01'
end_date = '2021-01-01'

data = fetch_data(stock1, stock2, start_date, end_date)
returns = calculate_returns(data)
covariance, hedge_ratio = calculate_cov_and_hedge_ratio(returns)
spread = calculate_spread(data, hedge_ratio)

signals = generate_signals(spread)

# Backtesting
capital_history = backtest(data, signals, hedge_ratio)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(capital_history, label='Portfolio Value')
plt.axhline(y=100000, color='red', linestyle='--', label='Initial Capital')
plt.title('Backtest Results')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()