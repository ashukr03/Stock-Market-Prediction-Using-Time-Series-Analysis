import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from keras.models import load_model
import streamlit as st
st.cache_data.clear()



start = '2010-01-01'
end = '2025-06-19'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'GOOGL')

# Fetching stock data using yfinance
df = yf.download(user_input, start=start, end=end)

# Describing the data
st.subheader('Data from 2010-2025')
st.write(df.describe())


# Visualizing the closing price history
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting the data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Loading the model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Debug: show last date
st.write(f"Last date in dataset: {df.index[-1]}")

st.subheader("Tomorrow's Predicted Price")

# Get last 100 days from the full Close price column
last_100_days_full = df['Close'].tail(100)

# Create a new scaler for this prediction only
from sklearn.preprocessing import MinMaxScaler
temp_scaler = MinMaxScaler(feature_range=(0, 1))
final_input = temp_scaler.fit_transform(last_100_days_full.values.reshape(-1, 1))

# Reshape to LSTM input shape
final_input = final_input.reshape(1, -1, 1)

# Predict the next price
predicted_price = model.predict(final_input)
predicted_price = temp_scaler.inverse_transform(predicted_price)
predicted_price = predicted_price[0][0]

# Get last and next business date
last_date = df.index[-1]
from pandas.tseries.offsets import BDay
next_date = last_date + BDay(1)  # business day = skips weekends

# Show the predicted price
st.success(f"Predicted Closing Price for {next_date.date()}: {predicted_price:.2f}")



