"""Report Normal/attack label counts for every CSV chunk in an AWID3 folder,
then print a balanced file list you can paste into CONFIG["files"] in wids_tuned.py.

Usage (PowerShell):
    python scan_labels.py "C:\\Users\\otiat\\Downloads\\archive\\CSV\\1.Deauth"
    python scan_labels.py "<folder>" Label      # optional: label column name
"""
import os
import re
import sys

import numpy as np
import pandas as pd

folder = sys.argv[1]
label_col = sys.argv[2] if len(sys.argv) > 2 else "Label"
N_ATTACK_FILES = 6     # how many attack-containing chunks to keep
N_NORMAL_FILES = 6     # how many normal-only chunks to add


def chunk_number(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


paths = sorted((os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith((".csv", ".xls"))), key=chunk_number)
if not paths:
    raise SystemExit(f"No .csv/.xls files found in {folder}")

rows = []
for p in paths:
    try:
        s = pd.read_csv(p, usecols=[label_col], low_memory=False)[label_col].astype(str)
    except ValueError:
        raise SystemExit(f"Column '{label_col}' not found in {os.path.basename(p)}. "
                         f"Pass the right name as the 2nd argument.")
    counts = s.value_counts()
    attack = {k: int(v) for k, v in counts.items() if k.strip().lower() != "normal"}
    rows.append({"path": p, "file": os.path.basename(p), "total": len(s),
                 "normal": int(counts.get("Normal", 0)),
                 "attack": sum(attack.values()), "labels": ", ".join(attack) or "-"})
    print(f"{rows[-1]['file']:<28} total={len(s):>6} normal={rows[-1]['normal']:>6} "
          f"attack={rows[-1]['attack']:>6} {rows[-1]['labels']}")

df = pd.DataFrame(rows)
print(f"\nTOTAL: {int(df.total.sum())} frames | normal={int(df.normal.sum())} "
      f"attack={int(df.attack.sum())} "
      f"({100 * df.attack.sum() / max(df.total.sum(), 1):.2f}% attack)")
print("Attack label(s) found:", ", ".join(sorted(set(", ".join(df.labels).split(", ")) - {"-"})) or "none")

atk = df[df.attack > 0].sort_values("attack", ascending=False).head(N_ATTACK_FILES)
nrm = df[df.attack == 0]
if len(nrm):
    idx = np.linspace(0, len(nrm) - 1, min(N_NORMAL_FILES, len(nrm))).astype(int)
    nrm = nrm.iloc[sorted(set(idx))]
chosen = pd.concat([atk, nrm]).sort_values("file", key=lambda c: c.map(chunk_number))

print("\n--- Paste this into CONFIG in wids_tuned.py ---")
print('"files": [')
for p in chosen.path:
    print(f'    r"{p}",')
print("],")
print(f'"col_label": "{label_col}",')
print("# set \"attack_keyword\" to a word from the attack label(s) above, e.g. \"deauth\"")
