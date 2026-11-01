"""Tuned WIDS prototype: deauth flood detection with scikit-learn.

Setup:   pip install pandas scikit-learn matplotlib joblib
Run:     python wids_tuned.py
Before running, edit CONFIG. Check your column names with:
    python -c "import pandas as pd; print(pd.read_csv('file.csv', nrows=2).columns.tolist())"
If you change files, window size or row limits, delete the windows_*.pkl cache.
"""
import os
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV,
                                     StratifiedGroupKFold, cross_val_score)
from sklearn.tree import DecisionTreeClassifier

CONFIG = {
    "files": [                               # training chunks (from scan_labels.py)
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_0.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_4.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_8.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_12.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_16.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_20.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_22.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_23.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_27.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_29.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_30.csv",
        r"C:\Users\otiat\Downloads\archive\CSV\1.Deauth\Deauth_31.csv",
    ],
    "max_rows_per_file": 500_000,
    "start_row": 0,                          # skip this many rows at the top
    "col_time": "frame.time_epoch",
    "col_subtype": "wlan.fc.subtype",        # 12 = deauth, 10 = disassoc (management frames)
    "col_type": "wlan.fc.type",              # 0 = management; merged with subtype in load()
    "col_signal": "radiotap.dbm_antsignal",
    "col_src": "wlan.ta",                    # transmitter address
    "col_dst": "wlan.da",
    "col_label": "Label",
    "attack_keyword": "deauth",              # label text that means attack
    "window_s": 1.0,                         # try 0.5 / 1 / 2 and compare
    "threshold": 0.5,                        # alert if P(attack) >= this
    "tune": False,                           # True = quick RandomForest search
    "wips": True,                            # log offending MACs
    "never_block": [],                       # MACs to never block, e.g. your own access point
    "use_cache": True,
}
ALL_FEATURES = ["frames_per_s", "deauth_per_s", "deauth_ratio", "sig_mean",
                "sig_std", "n_dst", "n_subtypes", "bcast_ratio",
                "iat_mean", "iat_min"]
NO_DEAUTH_FEATURES = [f for f in ALL_FEATURES
                      if f not in ("deauth_per_s", "deauth_ratio")]


def to_num(v):
    """Handles hex strings ('0x000c') and multi-value fields ('-40,-41')."""
    try:
        v = str(v).split(",")[0].strip()
        return int(v, 16) if v.lower().startswith("0x") else float(v)
    except ValueError:
        return np.nan


def load():
    c, parts = CONFIG, []
    cols = [c["col_time"], c["col_subtype"], c["col_type"], c["col_signal"],
            c["col_src"], c["col_dst"], c["col_label"]]
    skip = range(1, c["start_row"] + 1) if c["start_row"] else None
    for fid, path in enumerate(c["files"]):
        raw = pd.read_csv(path, usecols=cols, nrows=c["max_rows_per_file"],
                          skiprows=skip, low_memory=False)
        df = pd.DataFrame({
            "t": raw[c["col_time"]].map(to_num),
            "sub": raw[c["col_subtype"]].map(to_num) + 16 * raw[c["col_type"]].map(to_num),
            "sig": raw[c["col_signal"]].map(to_num).astype("float32"),
            "src": raw[c["col_src"]].astype(str),
            "dst": raw[c["col_dst"]].astype(str),
            "y": raw[c["col_label"]].astype(str).str.lower()
                    .str.contains(c["attack_keyword"]).astype("int8"),
        }).dropna(subset=["t", "sub"])
        df = df[df["src"] != "nan"]
        # offset per file so windows from different files never mix
        df["win"] = ((df["t"] - df["t"].min()) / c["window_s"]).astype(int) + fid * 10**7
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_windows(df):
    W = CONFIG["window_s"]
    df["is_deauth"] = df["sub"].isin([12, 10]).astype("int8")
    df["is_bcast"] = (df["dst"].str.lower() == "ff:ff:ff:ff:ff:ff").astype("int8")
    df = df.sort_values(["src", "t"])
    df["gap"] = df.groupby("src")["t"].diff()            # inter-arrival time per MAC
    f = df.groupby(["win", "src"]).agg(
        frames=("t", "size"), deauth=("is_deauth", "sum"),
        sig_mean=("sig", "mean"), sig_std=("sig", "std"),
        n_dst=("dst", "nunique"), n_subtypes=("sub", "nunique"),
        bcast_ratio=("is_bcast", "mean"),
        iat_mean=("gap", "mean"), iat_min=("gap", "min"),
        y=("y", "max"), t=("t", "min")).reset_index()
    f["frames_per_s"] = f["frames"] / W
    f["deauth_per_s"] = f["deauth"] / W
    f["deauth_ratio"] = f["deauth"] / f["frames"]
    f[["iat_mean", "iat_min"]] = f[["iat_mean", "iat_min"]].fillna(W)
    return f.fillna(0)


def get_windows():
    cache = f"windows_{CONFIG['window_s']}.pkl"
    if CONFIG["use_cache"] and os.path.exists(cache):
        print(f"Loading cached windows from {cache}")
        return pd.read_pickle(cache)
    feats = build_windows(load())
    feats.to_pickle(cache)
    return feats


def blockable(mac):
    """False for broadcast/multicast/invalid addresses, which are not a single device."""
    try:
        mac = str(mac).strip().lower()
        if mac in ("", "nan", "ff:ff:ff:ff:ff:ff"):
            return False
        return (int(mac.split(":")[0], 16) & 1) == 0   # low bit set = group address
    except ValueError:
        return False


def make_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=42),
    }
    try:
        models["Hist Gradient Boosting"] = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42)
    except TypeError:   # older scikit-learn: pip install -U scikit-learn
        models["Hist Gradient Boosting"] = HistGradientBoostingClassifier(random_state=42)
    return models


def scores(y, pred):
    return (accuracy_score(y, pred), precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0), f1_score(y, pred, zero_division=0))


def main():
    feats = get_windows()
    print(f"{len(feats)} windows | normal={int((feats.y == 0).sum())} "
          f"attack={int((feats.y == 1).sum())}")
    if feats["y"].nunique() < 2:
        raise SystemExit("Only one class found: check attack_keyword / label column.")

    y_all = feats["y"]
    groups = feats["win"] // 60          # 60-second blocks: avoids leakage between neighbours
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=42)
                  .split(feats, y_all, groups))
    X_tr, y_tr = feats.loc[tr, ALL_FEATURES], y_all.iloc[tr]
    X_te, y_te = feats.loc[te, ALL_FEATURES], y_all.iloc[te]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(cv.split(feats[ALL_FEATURES], y_all, groups))

    # ---- Compare classifiers -------------------------------------------------
    results, fitted = [], {}
    for name, m in make_models().items():
        if CONFIG["tune"] and name == "Random Forest":
            search = RandomizedSearchCV(
                m, {"n_estimators": [100, 150, 250], "max_depth": [None, 8, 16],
                    "min_samples_leaf": [1, 2, 5]},
                n_iter=8, scoring="f1", cv=cv_splits[:3], n_jobs=-1, random_state=42)
            search.fit(feats[ALL_FEATURES], y_all)
            m = search.best_estimator_
            print("Tuned Random Forest params:", search.best_params_)
        cv_f1 = cross_val_score(m, feats[ALL_FEATURES], y_all, groups=groups,
                                cv=cv_splits, scoring="f1", n_jobs=1)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        pred = m.predict(X_te)
        acc, prec, rec, f1 = scores(y_te, pred)
        results.append({"model": name, "accuracy": acc, "precision": prec,
                        "recall": rec, "f1": f1,
                        "cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
                        "train_s": train_s})
        fitted[name] = m
        print(f"\n=== {name} ===")
        print(classification_report(y_te, pred, target_names=["normal", "attack"],
                                    zero_division=0))
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        print("confusion matrix [[TN FP] [FN TP]]:\n", cm)
        ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot(cmap="Blues")
        plt.title(f"{name} (acc {acc:.3f})")
        plt.savefig(f"confusion_{name.replace(' ', '_').lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    table = pd.DataFrame(results).round(4)
    print("\n=== Model comparison (put this on the poster) ===")
    print(table.to_string(index=False))
    table.to_csv("model_comparison.csv", index=False)

    best_name = table.sort_values(["f1", "recall"], ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nBest model by F1: {best_name}")

    # ---- Feature importance (works for every model type) ---------------------
    n = min(5000, len(X_te))
    idx = np.random.RandomState(42).choice(len(X_te), n, replace=False)
    pi = permutation_importance(best, X_te.iloc[idx], y_te.iloc[idx],
                                n_repeats=3, scoring="f1", random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
    print("\nPermutation importance (drop in F1 when feature is shuffled):")
    print(imp.round(4).to_string())

    # ---- Ablation: does it still work without the deauth-count features? -----
    rf = make_models()["Random Forest"]
    rf.fit(X_tr[NO_DEAUTH_FEATURES], y_tr)
    a, p, r, f = scores(y_te, rf.predict(X_te[NO_DEAUTH_FEATURES]))
    print(f"\nAblation (Random Forest, no deauth_per_s/deauth_ratio): "
          f"acc={a:.4f} precision={p:.4f} recall={r:.4f} f1={f:.4f}")

    # ---- Threshold trade-off ---------------------------------------------------
    proba = best.predict_proba(X_te)[:, 1]
    print("\nAlert threshold trade-off:")
    for th in (0.3, 0.5, 0.7, 0.9):
        _, p, r, f = scores(y_te, (proba >= th).astype(int))
        print(f"  threshold {th}: precision={p:.3f} recall={r:.3f} f1={f:.3f}")

    joblib.dump({"model": best, "features": ALL_FEATURES,
                 "threshold": CONFIG["threshold"]}, "wids_model.joblib")

    # ---- Replay test windows in time order as "live" traffic -------------------
    test = feats.loc[te].copy()
    test["p_attack"] = proba
    test = test.sort_values("t")
    alerts = test[test["p_attack"] >= CONFIG["threshold"]]
    blocked, skipped = set(), set()
    never = {m.lower() for m in CONFIG["never_block"]}
    with open("blocklist.log", "a") as log:
        for r in alerts.itertuples():
            note = ""
            if not blockable(r.src):
                note = "  [not a single device address: not blocked]"
            elif r.src.lower() in never:
                note = "  [on never_block list: not blocked]"
            print(f"[ALERT] possible deauth flood from {r.src} "
                  f"(P={r.p_attack:.2f}, {r.frames_per_s:.0f} frames/s, "
                  f"{r.deauth_per_s:.0f} deauth/s){note}")
            if note:
                skipped.add(r.src)
            elif CONFIG["wips"] and r.src not in blocked:
                blocked.add(r.src)
                log.write(f"{pd.Timestamp.now().isoformat()} BLOCK {r.src} "
                          f"P={r.p_attack:.2f} {r.deauth_per_s:.0f} deauth/s\n")
    print(f"\n{len(alerts)} alerts | {len(blocked)} MACs logged to blocklist.log | "
          f"{len(skipped)} address(es) skipped: {sorted(skipped)}")


if __name__ == "__main__":
    main()