import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("data/processed.csv")
y = (df["Label"] == "Anomaly").astype(int)

X = df.drop(columns=["Label", "Cat", "Sub_Cat"]).select_dtypes(include="number")
print("Numeric features:", X.shape[1])

src = X.apply(lambda c: abs(c.corr(y, method="spearman"))).fillna(0)
src_keep = src[src > 0.2].index.tolist()
print("Spearman > 0.2:", len(src_keep))

sample = X.sample(100000, random_state=42)
mi = pd.Series(mutual_info_classif(sample, y.loc[sample.index], random_state=42), index=X.columns)
mi_keep = mi[mi > 0.1].index.tolist()
print("MI > 0.1:", len(mi_keep))

pd.DataFrame({"spearman": src, "mi": mi}).to_csv("data/scores.csv")
print("Saved data/scores.csv")