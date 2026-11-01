"""Test the saved WIDS model on Deauth chunks it never saw during training.

Put this file in C:\\wids next to wids_tuned.py and wids_model.joblib, then run:
    python wids_holdout.py
It reuses the loading and feature code from wids_tuned.py, so keep CONFIG there
pointing at your 12 training chunks (those are excluded here automatically).
"""
import glob
import os
import re

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

import wids_tuned as w

FOLDER = r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth"


def chunk_no(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def norm(p):
    return os.path.normcase(os.path.abspath(p))


used = {norm(p) for p in w.CONFIG["files"]}
all_files = sorted(glob.glob(os.path.join(FOLDER, "*.csv")), key=chunk_no)
held = [p for p in all_files if norm(p) not in used]
print(f"{len(held)} unseen chunks: " + ", ".join(os.path.basename(p) for p in held))

w.CONFIG["files"] = held
feats = w.build_windows(w.load())
feats["chunk"] = (feats["win"] // 10**7).map(lambda i: os.path.basename(held[int(i)]))

saved = joblib.load("wids_model.joblib")
model, cols, thr = saved["model"], saved["features"], saved["threshold"]
y = feats["y"]
feats["p_attack"] = model.predict_proba(feats[cols])[:, 1]
feats["pred"] = (feats["p_attack"] >= thr).astype(int)

print(f"\n{len(feats)} windows | normal={int((y == 0).sum())} attack={int((y == 1).sum())}")
print(f"Accuracy {accuracy_score(y, feats['pred']):.4f} | "
      f"precision {precision_score(y, feats['pred'], zero_division=0):.4f} | "
      f"recall {recall_score(y, feats['pred'], zero_division=0):.4f} | "
      f"F1 {f1_score(y, feats['pred'], zero_division=0):.4f}")
print(classification_report(y, feats["pred"], labels=[0, 1],
                            target_names=["normal", "attack"], zero_division=0))
cm = confusion_matrix(y, feats["pred"], labels=[0, 1])
print("confusion matrix [[TN FP] [FN TP]]:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot(cmap="Blues")
plt.title("Unseen chunks (held-out)")
plt.savefig("confusion_holdout.png", dpi=150, bbox_inches="tight")
plt.close()

feats["FP"] = ((feats.pred == 1) & (y == 0)).astype(int)
feats["FN"] = ((feats.pred == 0) & (y == 1)).astype(int)
per_chunk = feats.groupby("chunk").agg(windows=("y", "size"), attack_windows=("y", "sum"),
                                       alerts=("pred", "sum"), false_alarms=("FP", "sum"),
                                       missed=("FN", "sum"))
per_chunk = per_chunk.loc[sorted(per_chunk.index, key=chunk_no)]
print("\nPer chunk:")
print(per_chunk.to_string())
per_chunk.to_csv("holdout_per_chunk.csv")

fp = feats[feats.FP == 1]
if len(fp):
    print("\nFalse alarms by sender MAC:")
    print(fp["src"].value_counts().to_string())
fn = feats[feats.FN == 1]
if len(fn):
    print("\nMissed attack windows (what they looked like):")
    print(fn[["chunk", "src", "frames_per_s", "deauth_per_s"]].describe().round(2).to_string())
