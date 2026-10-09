import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score

TARGET = sys.argv[1] if len(sys.argv) > 1 else "Cat"

FEATURES = [c for c in pd.read_csv("data/processed.csv", nrows=1).select_dtypes(include="number").columns if c != "Src_Port"]
df = pd.read_csv("data/processed.csv")
df = df[df["Cat"] != "Normal"]
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[FEATURES].values.astype(np.float32)
n_classes = len(le.classes_)
print("Target:", TARGET, "| classes:", list(le.classes_))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

model = HistGradientBoostingClassifier(max_iter=300, random_state=42)
model.fit(X_tr, y_tr)
probs = model.predict_proba(X_te)
pred = probs.argmax(1)

p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_te, probs[:, 1]) if n_classes == 2 else roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
print("Accuracy :", round(accuracy_score(y_te, pred), 4))
print("Precision:", round(p, 4))
print("Recall   :", round(r, 4))
print("F1       :", round(f, 4))
print("AUC      :", round(auc, 4))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))

