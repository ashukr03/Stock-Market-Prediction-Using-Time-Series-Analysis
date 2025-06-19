Creating a Model to Predict the Stock Using Time Series
# Stock Market Trend Prediction using LSTM

This project predicts the next day's stock closing price using a trained **Long Short-Term Memory (LSTM)** model. It uses **historical stock data** from Yahoo Finance and offers an interactive UI built with **Streamlit**.

##  Features

- Input any **stock ticker** (e.g., `GOOGL`, `AAPL`, `TSLA`)  
- Visualizes historical trends with:
  - Basic time series plots
  - 100-day and 200-day moving averages  
- Predicts the **next trading day's closing price** using an LSTM model
- Uses real-time data via `yfinance`
- Simple interface with **Streamlit**

## Tech Stack

| Component         | Technology         |
|------------------|--------------------|
| Backend Model     | LSTM (Keras + TensorFlow) |
| UI                | Streamlit          |
| Data Source       | yFinance           |
| Plotting          | Matplotlib         |
| Scaling           | MinMaxScaler (sklearn) |

## Project Structure

- app.py # Streamlit app (main UI + prediction logic)
- keras_model.h5 # Trained LSTM model
- LSTM_Model.ipynb # Jupyter Notebook used to build and train the model
- README.md # Project documentation (this file)


