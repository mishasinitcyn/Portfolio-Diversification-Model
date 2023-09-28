from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import yfinance as yf
import pandas as pd
from urllib.parse import urlparse, parse_qs 
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
from keras import backend as K
print(K.backend())

np.random.seed(7)
tf.random.set_seed(7)
DELTA = 0.05

def reshape_series(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def forecast_closing_price(df):
    values = df[['High', 'Low', 'Volume', 'Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    reframed = reshape_series(scaled, 1, 1)
    reframed.drop(reframed.columns[[6,7]], axis=1, inplace=True)

    # Split data
    values = reframed.values
    train = values[:int(0.8*len(values)), :]
    test = values[int(0.8*len(values)):, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape data [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # Instantiate and train model
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # Forecast
    yhat = model.predict(test_X)
    test_X_reshaped = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # Extract ('High', 'Low', 'Volume')
    relevant_test_X = test_X_reshaped[:, 0:3]
    inv_yhat = np.column_stack((relevant_test_X, yhat))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    predicted_close_prices = inv_yhat[:, -1]

    return predicted_close_prices

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def get_stock_data(self, ticker, period): 
        tickerData = yf.Ticker(ticker)
        tickerDf = tickerData.history(period=period, interval='1d')
        #tickerDf = pd.read_csv('MSFT.csv', index_col=0, parse_dates=True)
        return tickerDf # Convert DataFrame to dictionary for JSON serialization

    def calculate_ucb_exploration_term(self, prices):
        daily_returns = prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        variance_return = daily_returns.var()
        t = len(prices)
        n = t

        exploration_term = min(1/4, variance_return + np.sqrt((2 * np.log(t) / n)))

        ucb = mean_return + np.sqrt((np.log(t) / n) * exploration_term)
        
        return ucb

    def calculate_rl_ucb(self, prices, delta):
        daily_returns = prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        variance = daily_returns.var()
        n = len(prices)

        ucb = mean_return + np.sqrt((2 * np.log(1/delta) / n) * variance)
        return ucb, mean_return, variance

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_components = parse_qs(parsed_url.query)
        
        if path == '/stock_data':
            ticker = query_components.get('ticker')[0]
            period = query_components.get('period', ['1d'])[0]

            print('GET:', ticker, period)
            stock_data = self.get_stock_data(ticker, period)
            closing_prices = pd.Series(stock_data['Close'])
            ucb_tuple = self.calculate_rl_ucb(closing_prices, DELTA)

            response_data = {
                'ucb_tuple': ucb_tuple,
                'ticker': ticker
            }
            print("SENDING response_data:", response_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET')
            self.send_header('Access-Control-Allow-Headers', 'content-type')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        elif path == '/forecast':
            ticker = query_components.get('ticker')[0]
            period = query_components.get('period', ['1d'])[0]
            steps = 3

            print('GET forecast:', ticker, period)
            stock_data = self.get_stock_data(ticker, period)
            closing_prices = pd.Series(stock_data['Close'])
            forecast_data = forecast_closing_price(stock_data)  
            forecast_series = pd.Series(forecast_data)
            closing_prices = closing_prices.append(forecast_series, ignore_index=True)

            ucb_tuple = self.calculate_rl_ucb(closing_prices, DELTA)

            response_data = {
                'ucb_tuple': ucb_tuple,
                'ticker': ticker
            }
            print("SENDING response_data:", response_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET')
            self.send_header('Access-Control-Allow-Headers', 'content-type')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    httpd = HTTPServer(('localhost', 5000), SimpleHTTPRequestHandler)
    print("Serving on port 5000")
    httpd.serve_forever()
