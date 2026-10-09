import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})
os.makedirs("figures", exist_ok=True)

# ---- Figure 1: accuracy and F1 of all experiments ----
ids = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"]
acc = [79.60, 79.60, 97.09, 83.61, 97.80, 85.92, 86.63, 86.66, 88.35, 88.89, 86.19, 88.18]
f1 = [72.96, 72.97, 97.02, 82.93, 97.79, 86.28, 86.73, 86.80, 88.50, 88.84, 87.63, 88.05]
x = np.arange(len(ids))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(x - w / 2, acc, w, color="0.25", edgecolor="black", label="Accuracy")
ax.bar(x + w / 2, f1, w, color="0.8", edgecolor="black", label="F1 score")
ax.axhline(89.54, color="black", linestyle="--", linewidth=1, label="Paper accuracy (89.54%)")
ax.set_ylim(60, 100)
ax.set_xticks(x)
ax.set_xticklabels(ids)
ax.set_xlabel("Experiment")
ax.set_ylabel("Score (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig1_all_experiments.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: Random Forest feature importance (all numeric columns) ----
names = ["Src_Port", "Flow_Duration", "Dst_Port", "Init_Bwd_Win_Byts", "Flow_Pkts/s",
         "Flow_IAT_Mean", "Flow_IAT_Max", "Idle_Max", "Bwd_IAT_Tot", "Bwd_IAT_Max"]
imp = [21.5, 7.8, 7.5, 7.1, 3.9, 3.7, 3.6, 3.2, 2.8, 2.7]
fig, ax = plt.subplots(figsize=(8, 4.5))
cols = ["black"] + ["0.6"] * 9
ax.barh(names[::-1], imp[::-1], color=cols[::-1], edgecolor="black")
ax.set_xlabel("Feature importance (%)")
fig.tight_layout()
fig.savefig("figures/fig2_feature_importance.png", dpi=300)
plt.close(fig)

# ---- Figure 3: recall per class ----
classes = ["DoS", "MITM", "Mirai", "Normal", "Scan"]
lstm = [99.6, 15.1, 97.5, 79.2, 0.04]
gb = [99.9, 36.1, 90.4, 97.0, 88.2]
x = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w / 2, lstm, w, color="0.8", edgecolor="black", hatch="//", label="LSTM (E2)")
ax.bar(x + w / 2, gb, w, color="0.25", edgecolor="black", label="Gradient boosting (E10)")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Recall (%)")
ax.set_ylim(0, 110)
ax.legend(loc="upper center", ncol=2, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig3_recall_per_class.png", dpi=300)
plt.close(fig)

# ---- Figure 4: confusion matrix of gradient boosting (E10) ----
cm = np.array([[11872, 3, 1, 2, 0],
               [0, 1866, 1732, 24, 1550],
               [1, 564, 50844, 11, 4801],
               [0, 17, 141, 7487, 75],
               [0, 174, 1157, 5, 10013]])
norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
fig.tight_layout()
fig.savefig("figures/fig4_confusion_matrix.png", dpi=300)
plt.close(fig)
print("saved 4 figures in figures/")