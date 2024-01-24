import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

df = pd.read_csv('Datasets/dc.csv')
df = df.rename(columns={'Unnamed: 0': 'Time'})
df['Time'] = pd.to_datetime(df['Time'])
df = df.iloc[::-1].set_index('Time')
df_p = df.reset_index()[["Time", "close_USD"]].rename(
    columns={"Time": "ds", "close_USD": "y"}
)

train = df.iloc[:-200, 7]
test = df.iloc[-200:, 7]

model = Prophet()

# Fit the model
model.fit(df_p)

# create date to predict
future_dates = model.make_future_dataframe(periods=365)

# Make predictions
predictions = model.predict(future_dates)

print(predictions.head())

model.plot(predictions)
plt.show()

model.plot_components(predictions)
plt.show()

df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='365 days')

# Calculate evaluation metrics
res = performance_metrics(df_cv)

print(res)

# choose between 'mse', 'rmse', 'mae', 'mape', 'coverage'
plot_cross_validation_metric(df_cv, metric='coverage')
plt.show()
