# Portfolio Diversification Model

## Overview

This project aims to implement Reinforcement Learning algorithms for financial data analysis and forecasting. **Time series** stock data is fetched from Yahoo Finance API and analyzed using the 
**Upper Confidence Bound** (UCB) algorithm. The confidence interval is displayed in the Angular frontend. **Forecasting** functionality is in development, using **Vector Auto Regression** (VAR) and Keras 
**Long Short-Term Memory** (LSTM) models.

## Features

- **Upper Confidence Bound (UCB)**: Reinforcement Learning algorithm to analyze financial data and create recommendations based on risk tolerance and historical performance.
- **Angular Frontend**: Interactive user interface to search for stocks and display the confidence bounds.
- **VAR Forecasting**: Utilizes the Vector Auto Regression model to forecast financial trends.
- **Keras LSTM Forecasting**: Incorporates the Long Short-Term Memory model from Keras for time series forecasting.

## Installation

### Backend

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the backend using `python server.py`.

### Frontend

1. cd into "UCB-frontend".
2. Install the required packages using `npm install`.
3. Run the frontend using `ng serve`.
4. The frontend will be running on `http://localhost:4200`.


## Demo

A demo of the application is yet to be deployed.
![image](https://github.com/mishasinitcyn/Portfolio-Diversification-Model/assets/45673816/d475a7a3-4dfa-4723-94a4-a4c2c3faeeee)
