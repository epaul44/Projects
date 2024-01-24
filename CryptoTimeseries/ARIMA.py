import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('Datasets/dc.csv')
df = df.rename(columns={'Unnamed: 0': 'Time'})
df['Time'] = pd.to_datetime(df['Time'])
df = df.iloc[::-1].set_index('Time')

train = df.iloc[:-200, 7]
test = df.iloc[-200:, 7]

model = ARIMA(train, order=(2, 1, 0))
results = model.fit()

# Make predictions for the test set
forecast = results.forecast(steps=200)

mae = mean_absolute_error(test, forecast)

# root mean square error
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

# mean absolute percentage error
mape = (forecast - test).abs().div(test).mean()

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# checking the model
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.show()
