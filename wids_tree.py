"""Draw the Decision Tree the WIDS uses, in a size that fits on a poster.

Run from C:\\wids (needs wids_tuned.py and the windows_1.0.pkl cache from an earlier run):
    python wids_tree.py
Outputs: decision_tree.png and decision_tree_rules.txt
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import wids_tuned as w

MAX_DEPTH = 3          # small enough to read on a poster; raise to 4 for more detail

feats = w.get_windows()
y = feats["y"]
groups = feats["win"] // 60                      # same split as the main script
tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=42)
              .split(feats, y, groups))
cols = w.ALL_FEATURES

tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=5,
                              class_weight="balanced", random_state=42)
tree.fit(feats.loc[tr, cols], y.iloc[tr])
pred = tree.predict(feats.loc[te, cols])

print(f"Depth-{MAX_DEPTH} tree on the held-out test windows:")
print(f"  accuracy  {accuracy_score(y.iloc[te], pred):.4f}")
print(f"  precision {precision_score(y.iloc[te], pred, zero_division=0):.4f}")
print(f"  recall    {recall_score(y.iloc[te], pred, zero_division=0):.4f}")
print(f"  F1        {f1_score(y.iloc[te], pred, zero_division=0):.4f}")
print("  confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y.iloc[te], pred, labels=[0, 1]))

rules = export_text(tree, feature_names=cols)
print("\nRules the tree learned:\n" + rules)
open("decision_tree_rules.txt", "w", encoding="utf-8").write(rules)

fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(tree, feature_names=cols, class_names=["normal", "attack"],
          filled=True, rounded=True, fontsize=11, proportion=False, ax=ax)
plt.savefig("decision_tree.png", dpi=200, bbox_inches="tight")
print("Saved decision_tree.png and decision_tree_rules.txt")
