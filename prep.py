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

df["Fwd_Bwd_Pkt_Ratio"] = df["Tot_Fwd_Pkts"] / (df["Tot_Bwd_Pkts"] + 1)
df["Fwd_Bwd_Byts_Ratio"] = df["TotLen_Fwd_Pkts"] / (df["TotLen_Bwd_Pkts"] + 1)
df["Total_Pkts"] = df["Tot_Fwd_Pkts"] + df["Tot_Bwd_Pkts"]
df["Total_Byts"] = df["TotLen_Fwd_Pkts"] + df["TotLen_Bwd_Pkts"]
df["Avg_Byts_Per_Pkt"] = df["Total_Byts"] / (df["Total_Pkts"] + 1)
df["Total_Flag_Cnt"] = df[["FIN_Flag_Cnt", "SYN_Flag_Cnt", "RST_Flag_Cnt", "PSH_Flag_Cnt", "ACK_Flag_Cnt", "URG_Flag_Cnt", "ECE_Flag_Cnt"]].sum(axis=1)
df["Total_Hdr_Len"] = df["Fwd_Header_Len"] + df["Bwd_Header_Len"]
print("After adding 7 columns:", df.shape)

df = df.drop_duplicates()
print("After removing duplicate rows:", df.shape)

df.to_csv("data/processed.csv", index=False)
print("Saved data/processed.csv")