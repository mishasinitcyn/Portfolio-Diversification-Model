from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import yfinance as yf
import pandas as pd
from urllib.parse import urlparse, parse_qs 
import numpy as np
DELTA = 0.05

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def get_stock_data(self, ticker, period): 
        tickerData = yf.Ticker(ticker)
        tickerDf = tickerData.history(period=period, interval='1d')
        #tickerDf = pd.read_csv('MSFT.csv', index_col=0, parse_dates=True)
        return tickerDf # Convert DataFrame to dictionary for JSON serialization

    def calculate_ucb(self, prices):
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

    def calculate_ucb_tuned(self, prices):
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

    def calculate_rl_ucb(self, prices, delta):
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

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        ticker = query_components.get('ticker')[0]
        period = query_components.get('period', ['1d'])[0]

        print('GET:', ticker, period)
        stock_data = self.get_stock_data(ticker, period)
        closing_prices = pd.Series(stock_data['Close'])
        print(closing_prices)
        print(DELTA)
        ucb_tuple = self.calculate_rl_ucb(closing_prices, DELTA)

        response_data = {
            'ucb_tuple': ucb_tuple,
            'ticker': ticker
        }
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Access-Control-Allow-Headers', 'content-type')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))


if __name__ == '__main__':
    httpd = HTTPServer(('localhost', 5000), SimpleHTTPRequestHandler)
    print("Serving on port 5000")
    httpd.serve_forever()
