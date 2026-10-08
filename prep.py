import numpy as np
import pandas as pd

df = pd.read_csv("data/IoT_Network_Intrusion_Dataset.csv")
print("Loaded:", df.shape)

df = df.replace([np.inf, -np.inf], np.nan)
print("NaN before fill:", int(df.isnull().sum().sum()))

num_cols = df.select_dtypes(include="number").columns
obj_cols = df.select_dtypes(exclude="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df[obj_cols] = df[obj_cols].fillna("")
print("NaN after fill:", int(df.isnull().sum().sum()))

print("Duplicate rows:", int(df.duplicated().sum()))
print(df["Label"].value_counts())
print(df["Cat"].value_counts())