import pandas as pd

#Also, we want to avoid any NaN values, so we do:
df = pd.read_csv('data.csv', sep = ",").fillna(0)

print(df.info())

print(df.head())