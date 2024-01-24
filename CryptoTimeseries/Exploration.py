import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/dc.csv')

df = df.rename(columns={'Unnamed: 0': 'Time'})
df['Time'] = pd.to_datetime(df['Time'])
df = df.iloc[::-1].set_index('Time')

plt.plot(df['close_USD'])
plt.show()
