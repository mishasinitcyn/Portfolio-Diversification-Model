import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_data(ticker, period): 
    tickerData = yf.Ticker(ticker)
    tickerDf = tickerData.history(period=period, interval='1d')
    #tickerDf = pd.read_csv('MSFT.csv', index_col=0, parse_dates=True)
    return tickerDf

def calculate_ucb(prices):
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate mean and variance of returns
    mean_return = daily_returns.mean()
    variance_return = daily_returns.var()
    
    # Number of observations
    n = len(prices)
    
    # Calculate UCB
    ucb = mean_return + np.sqrt((2 * np.log(n) / n) * variance_return)
    return ucb

def calculate_ucb_tuned(prices):
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate mean and variance of returns
    mean_return = daily_returns.mean()
    variance_return = daily_returns.var()
    
    # Total number of observations
    t = len(prices)

    # Number of observations for this stock (same as t in this context)
    n = t

    # Calculate the exploration term (inside the square root)
    exploration_term = min(1/4, variance_return + np.sqrt((2 * np.log(t) / n)))
    
    # Calculate UCB using the tuned formula
    ucb = mean_return + np.sqrt((np.log(t) / n) * exploration_term)
    
    return ucb

def calculate_rl_ucb(prices, delta):
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate mean and variance of returns up to time t
    mean_return = daily_returns.mean()
    variance = daily_returns.var()

    # Number of observations up to time t
    n = len(prices)
    
    # Calculate UCB using the RL formula with delta
    ucb = mean_return + np.sqrt((2 * np.log(1/delta) / n) * variance)
    
    return ucb, mean_return, variance

MSFT_stock_data = pd.read_csv('MSFT.csv', index_col=0, parse_dates=True) #get_stock_data('MSFT', '1wk')
MSFT_closing_prices = pd.Series(MSFT_stock_data['Close'])
print(calculate_rl_ucb(MSFT_closing_prices, 0.05))

TSLA_stock_data = pd.read_csv('TSLA.csv', index_col=0, parse_dates=True) #get_stock_data('TSLA', '1wk')
TSLA_closing_prices = pd.Series(TSLA_stock_data['Close'])
print(calculate_rl_ucb(TSLA_closing_prices, 0.05))

GOOGL_stock_data = pd.read_csv('GOOGL.csv', index_col=0, parse_dates=True) #get_stock_data('GOOGL', '1wk')
GOOGL_closing_prices = pd.Series(GOOGL_stock_data['Close'])
print(calculate_rl_ucb(GOOGL_closing_prices, 0.05))