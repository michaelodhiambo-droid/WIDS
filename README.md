# WIDS: Detecting Wi-Fi Deauthentication Floods with Machine Learning

A prototype wireless intrusion detection system (WIDS) with an optional prevention step (WIPS). It reads recorded 802.11 traffic, groups it into one-second windows per sender, computes simple features, and uses scikit-learn classifiers (Decision Tree, Random Forest, Histogram Gradient Boosting) to label each window as normal or attack. It raises an alert for each detected deauthentication flood and logs the offending sender address.

**Author:** Michael Odhiambo in collab with claude ai

## Data

- **Dataset:** AWID3 (Deauth capture), obtained from Kaggle. The dataset is **not included** in this repository because of its size and licence. Download it separately.
- The Deauth folder has 33 CSV chunks (about 50,000 frames each, 1.63 million frames in total, 2.39% attack). Every frame has a `Label` column with `Normal` or `Deauth`.
- **Training:** 12 chunks (0, 4, 8, 12, 16, 20, 22, 23, 27, 29, 30, 31), giving 3,074 one-second windows (2,805 normal, 269 attack).
- **Unseen test:** the other 21 chunks, giving 3,606 windows (3,472 normal, 134 attack).
- No traffic was captured and no attacks were generated for this project. Everything uses the public dataset. Detection runs by **replaying recorded traffic**, because the author's laptop Wi-Fi card does not support monitor mode.

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## How to run

Run from the repository folder. Edit the paths marked below to point at your copy of the dataset.

1. **Choose training files** (optional). Scans the label counts in each chunk and prints a balanced file list:
   ```
   python scan_labels.py "C:\path\to\archive\CSV\1.Deauth"
   ```
2. **Train, compare and replay.** Open `wids_tuned.py` and set `CONFIG["files"]` to your chunk paths. Column names are set for the AWID3 CSVs (`wlan.fc.subtype`, `wlan.fc.type`, and so on). Then:
   ```
   python wids_tuned.py
   ```
   Trains the three models, prints accuracy, precision, recall, F1 and confusion matrices, runs grouped cross-validation, feature importance, an ablation test and a threshold table, then replays the test windows as live traffic with alerts and a block list.
3. **Test on unseen chunks.** Open `wids_holdout.py`, set `FOLDER` to the Deauth folder, then:
   ```
   python wids_holdout.py
   ```
   Takes a few minutes. Needs `wids_model.joblib`, which step 2 creates.
4. **Draw the Decision Tree** for the poster:
   ```
   python wids_tree.py
   ```

## Files

| File | Purpose |
|---|---|
| `scan_labels.py` | Label counts per dataset chunk and a suggested training file list |
| `wids_tuned.py` | Main pipeline: features, three models, evaluation, alerts, block list |
| `wids_holdout.py` | Evaluates the saved model on chunks not used for training |
| `wids_tree.py` | Trains a small Decision Tree and saves a diagram and the rules |
| `model_comparison.csv` | Results table from the held-out test |
| `confusion_*.png` | Confusion matrices (one per model, plus `confusion_holdout.png`) |
| `holdout_per_chunk.csv` | Alerts, false alarms and misses for each unseen chunk |
| `decision_tree.png`, `decision_tree_rules.txt` | The tree and its rules in plain text |
| `blocklist.log` | Addresses flagged by the WIPS step |

## Features (per sender, per one-second window)

Frames per second, deauthentication and disassociation frames per second, share of frames that are deauthentication, mean and standard deviation of signal strength, number of distinct destinations, number of distinct frame types, share of broadcast frames, and mean and minimum time between frames.

## Results

Held-out test (872 windows, 74 attack):

| Model | Accuracy | Precision | Recall | F1 | Cross-val F1 |
|---|---|---|---|---|---|
| Decision Tree | 99.89% | 98.7% | 100% | 0.993 | 0.991 ± 0.005 |
| Random Forest | 99.89% | 98.7% | 100% | 0.993 | 0.991 ± 0.005 |
| Hist Gradient Boosting | 99.66% | 98.6% | 97.3% | 0.980 | 0.994 ± 0.005 |

Unseen chunks (3,606 windows, 134 attack): recall 100%, precision 95.7%, F1 0.978, with 6 false alarms and no false alarms in 2,616 attack-free windows. With the two deauthentication-count features removed, the Random Forest still reached F1 0.973.

## Prevention (WIPS) step

Detected sender addresses are written to `blocklist.log`. Broadcast and group addresses are never logged as blocked, and `CONFIG["never_block"]` can hold addresses of trusted devices such as the access point. The system logs only and does not block traffic.

## Limitations

- Replay of recorded traffic, not live capture.
- All files come from one capture session, so results do not show transfer to other networks.
- The learned cutoff is low (about 1.5 deauthentication frames per second), which may cause false alarms on busy networks where devices roam.
- Many attack frames carry broadcast or forged sender addresses, so blocking by address is weak.

## Possible improvements

Live capture with a monitor-mode adapter on an authorised test network, testing on a different capture, adding the other AWID3 attack types, features that separate roaming from attacks, real blocking, and protected management frames (802.11w).

## Reference

Chatzoglou, Kambourakis and Kolias, AWID3 dataset paper (IEEE Access, 2021). Check the exact citation on the AWID3 page.
