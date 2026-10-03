import pandas as pd

df = pd.read_parquet("data/software-data.parquet")
df_filled = df.ffill()
print(df_filled)