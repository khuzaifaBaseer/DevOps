import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score

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

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xt = torch.tensor(X_tr).unsqueeze(1).to(dev)
yt = torch.tensor(y_tr, dtype=torch.long).to(dev)
Xe = torch.tensor(X_te).unsqueeze(1).to(dev)

class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), 64, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1]))

model = LSTMNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    perm = torch.randperm(len(Xt), device=dev)
    total = 0.0
    for i in range(0, len(Xt), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    print(f"Epoch {epoch + 1}/20  loss {total / len(Xt):.4f}")

model.eval()
with torch.no_grad():
    probs = torch.cat([torch.softmax(model(Xe[i:i + 4096]), 1) for i in range(0, len(Xe), 4096)]).cpu().numpy()
pred = probs.argmax(1)

p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_te, probs[:, 1]) if n_classes == 2 else roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
print("Accuracy :", round(accuracy_score(y_te, pred), 4))
print("Precision:", round(p, 4))
print("Recall   :", round(r, 4))
print("F1       :", round(f, 4))
print("AUC      :", round(auc, 4))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))