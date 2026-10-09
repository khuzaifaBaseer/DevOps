import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

TARGET = sys.argv[1] if len(sys.argv) > 1 else "Cat"

FEATURES = ["Protocol", "Tot_Bwd_Pkts", "Flow_Pkts/s", "Flow_IAT_Min", "Fwd_IAT_Mean",
            "Fwd_Pkts/s", "SYN_Flag_Cnt", "RST_Flag_Cnt", "URG_Flag_Cnt", "ECE_Flag_Cnt",
            "Fwd_Seg_Size_Avg", "Subflow_Fwd_Pkts", "Subflow_Bwd_Pkts", "Subflow_Bwd_Byts",
            "Fwd_Act_Data_Pkts"]

df = pd.read_csv("data/processed.csv")
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[FEATURES].values.astype(np.float32)
n_classes = len(le.classes_)
print("Target:", TARGET, "| classes:", list(le.classes_))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32).reshape(-1, 1, len(FEATURES))
X_te = scaler.transform(X_te).astype(np.float32).reshape(-1, 1, len(FEATURES))
y_tr_oh = to_categorical(y_tr, n_classes)
y_te_oh = to_categorical(y_te, n_classes)

model = keras.Sequential([
    layers.Input(shape=(1, len(FEATURES))),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr_oh, epochs=20, batch_size=64, validation_data=(X_te, y_te_oh))

probs = model.predict(X_te, batch_size=4096)
pred = probs.argmax(1)

p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_te, probs[:, 1]) if n_classes == 2 else roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
print("Accuracy :", round(accuracy_score(y_te, pred), 4))
print("Precision:", round(p, 4))
print("Recall   :", round(r, 4))
print("F1       :", round(f, 4))
print("AUC      :", round(auc, 4))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))