"""
This file contains the server code for the RL agent. It is responsible 
or connecting to the Yahoo Finance API to retrieve stock data, and 
for calculating the UCB for the RL agent. It also contains the LSTM
code for forecasting closing prices.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import urlparse, parse_qs
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print(K.backend())

np.random.seed(7)
tf.random.set_seed(7)
DELTA = 0.05


def reshape_series(data, n_in=1, n_out=1, dropnan=True):
    """
    Reshape time series data for time series forecasting
    """
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    data_frame = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(data_frame.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(data_frame.shift(-i))
        if i == 0:
            names += [f"var{j + 1}(t)" for j in range(n_vars)]
        else:
            names += [f"var{j +1 }(t+{i})" for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def preprocess_data(data_frame):
    """
    Preprocess the input data, scaling and reshaping as necessary
    """
    values = data_frame[["High", "Low", "Volume", "Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = reshape_series(scaled, 1, 1)
    reframed.drop(reframed.columns[[6, 7]], axis=1, inplace=True)
    return reframed, scaler


def split_data(reframed):
    """
    Split the reframed data into training and testing sets.
    """
    values = reframed.values
    train = values[: int(0.8 * len(values)), :]
    test = values[int(0.8 * len(values)):, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    return train_x, train_y, test_x, test_y


def reshape_for_lstm(train_x, test_x):
    """
    Reshape the input data to the required shape for LSTM input.
    """
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    return train_x, test_x


def build_and_train_model(train_x, train_y, test_x, test_y):
    """
    Build, compile, and train an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam")
    model.fit(
        train_x,
        train_y,
        epochs=50,
        batch_size=72,
        validation_data=(test_x, test_y),
        verbose=2,
        shuffle=False,
    )
    return model


def forecast_and_inverse_transform(test_x, model, scaler):
    """
    Uses the trained LSTM model to forecast and then inverse scales the predictions.
    """
    yhat = model.predict(test_x)
    test_x_reshaped = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    relevant_test_x = test_x_reshaped[:, 0:3]
    inv_yhat = np.column_stack((relevant_test_x, yhat))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    predicted_close_prices = inv_yhat[:, -1]
    return predicted_close_prices


def forecast_closing_price(data_frame):
    """
    Forecast closing price using LSTM
    """
    reframed, scaler = preprocess_data(data_frame)
    train_x, train_y, test_x, test_y = split_data(reframed)
    train_x, test_x = reshape_for_lstm(train_x, test_x)
    model = build_and_train_model(train_x, train_y, test_x, test_y)
    predicted_close_prices = forecast_and_inverse_transform(
        test_x, model, scaler)
    return predicted_close_prices


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Simple HTTP request handler with CORS support and stock data retrieval
    """

    def get_stock_data(self, ticker, period):
        """
        Get stock data from Yahoo Finance API, return as dictionary
        """
        ticker_data = yf.Ticker(ticker)
        ticker_df = ticker_data.history(period=period, interval="1d")
        # tickerDf = pd.read_csv('MSFT.csv', index_col=0, parse_dates=True)
        return ticker_df  # Convert DataFrame to dictionary for JSON serialization

    def calculate_ucb_exploration_term(self, prices):
        """
        Calculate exploration term for UCB
        """
        daily_returns = prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        variance_return = daily_returns.var()
        t = len(prices)
        n = t

        exploration_term = min(1 / 4, variance_return +
                               np.sqrt((2 * np.log(t) / n)))

        ucb = mean_return + np.sqrt((np.log(t) / n) * exploration_term)

        return ucb

    def calculate_rl_ucb(self, prices, delta):
        """
        Calculate UCB for RL agent and return tuple of UCB, mean return, and variance
        """
        daily_returns = prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        variance = daily_returns.var()
        n = len(prices)

        ucb = mean_return + np.sqrt((2 * np.log(1 / delta) / n) * variance)
        return ucb, mean_return, variance

    def do_options(self):
        """
        Respond to CORS preflight request
        """
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "X-Requested-With, Content-type"
        )
        self.end_headers()

    def do_GET(self): # pylint: disable=invalid-name
        """
        Respond to GET request with stock data and UCB
        """
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_components = parse_qs(parsed_url.query)
        if path == "/stock_data":
            ticker = query_components.get("ticker")[0]
            period = query_components.get("period", ["1d"])[0]

            print("GET:", ticker, period)
            stock_data = self.get_stock_data(ticker, period)
            closing_prices = pd.Series(stock_data["Close"])
            ucb_tuple = self.calculate_rl_ucb(closing_prices, DELTA)

            response_data = {"ucb_tuple": ucb_tuple, "ticker": ticker}
            print("SENDING response_data:", response_data)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        elif path == "/forecast":
            ticker = query_components.get("ticker")[0]
            period = query_components.get("period", ["1d"])[0]
            # steps = 3

            print("GET forecast:", ticker, period)
            stock_data = self.get_stock_data(ticker, period)
            closing_prices = pd.Series(stock_data['Close'])
            forecast_data = forecast_closing_price(stock_data)
            forecast_series = pd.Series(forecast_data)
            closing_prices = closing_prices.append(
                forecast_series, ignore_index=True)
            ucb_tuple = self.calculate_rl_ucb(closing_prices, DELTA)

            response_data = {"ucb_tuple": ucb_tuple, "ticker": ticker}
            print("SENDING response_data:", response_data)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    httpd = HTTPServer(("localhost", 8080), SimpleHTTPRequestHandler)
    print("Serving on port 8080")
    httpd.serve_forever()
