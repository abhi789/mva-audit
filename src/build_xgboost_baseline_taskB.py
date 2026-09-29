"""
XGBoost baseline on the ICU task.

Mirrors build_mlp_baseline.py:
  - same 11 features (MAP, SpO2, HR, RR, temp, lactate, FiO2, fluid_bolus_3h,
    n_vasopressors, NEWS2, NEWS2_6h_delta)
  - same 5-fold stratified CV and seed (20260517)
  - same 6-class reference labels

Default XGBoost hyperparameters, matching the MLP's use of defaults.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

BASE = Path(__file__).resolve().parents[1]  # repository root
SC_DIR = BASE / "icu_data_v1" / "scenarios"
SEED = 20260517

ICU_ACTIONS = ["continue_monitoring", "recheck_vitals", "adjust_positioning_or_oxygen",
               "fluid_or_vasopressor_adjustment", "escalate_to_attending", "transfer_higher_acuity"]


def extract_features_label(window):
    vs = window["vital_snapshot"]
    feats = [
        vs.get("map_mmhg") if vs.get("map_mmhg") is not None else np.nan,
        vs.get("spo2_pct") if vs.get("spo2_pct") is not None else np.nan,
        vs.get("hr_bpm") if vs.get("hr_bpm") is not None else np.nan,
        vs.get("rr_per_min") if vs.get("rr_per_min") is not None else np.nan,
        vs.get("temp_celsius") if vs.get("temp_celsius") is not None else np.nan,
        vs.get("lactate_mmol_per_l") if vs.get("lactate_mmol_per_l") is not None else np.nan,
        vs.get("fio2_frac") or 0.21,
        vs.get("fluid_bolus_ml_past_3h") or 0,
        len(vs.get("vasopressors_active", [])),
        vs.get("news2_score") or 0,
        vs.get("news2_6h_delta") or 0,
    ]
    return feats, window["ground_truth"]["action"]


def load_data():
    X, y, ids = [], [], []
    for p in sorted(SC_DIR.glob("*.json")):
        j = json.load(open(p))
        w = j["window_stream"][0]
        feats, label = extract_features_label(w)
        X.append(feats); y.append(label); ids.append(j["scenario_id"])
    return np.array(X, dtype=float), np.array(y), ids


def main():
    X, y, ids = load_data()
    print(f"Loaded {len(X)} scenarios, {len(set(y))} classes, feature shape {X.shape}")
    print(f"  class dist: {Counter(y)}")
    le = LabelEncoder().fit(ICU_ACTIONS)
    y_idx = le.transform(y)

    # Impute NaN (XGBoost handles it natively but the MLP imputed → match protocol)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros(len(y), dtype=int)
    for fold, (tr, te) in enumerate(skf.split(X, y_idx)):
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=6,
            random_state=SEED, tree_method="hist", n_jobs=4,
            verbosity=0,
        )
        clf.fit(X[tr], y_idx[tr])
        preds[te] = clf.predict(X[te])
        fold_acc = (preds[te] == y_idx[te]).mean()
        print(f"  fold {fold}: acc {fold_acc:.3f}")

    overall_acc = (preds == y_idx).mean()
    print(f"\nXGBoost overall 5-fold CV accuracy: {overall_acc:.4f} ({(preds==y_idx).sum()}/{len(y)})")
    # Per-class recall
    pred_labels = le.inverse_transform(preds)
    print("\nPer-class recall:")
    for cls in ICU_ACTIONS:
        mask = (y == cls)
        if mask.sum() > 0:
            rec = (pred_labels[mask] == cls).mean()
            print(f"  {cls:<35s} n={mask.sum():3d}  recall={rec:.3f}")

    # k_used, H, dom
    pc = Counter(pred_labels)
    n_total = sum(pc.values())
    import math
    H = -sum((v/n_total) * math.log2(v/n_total) for v in pc.values() if v > 0)
    print(f"\nk_used={len(pc)}, H={H:.2f} bits, dom={max(pc.values())/n_total:.2f}")

    out = {
        "model": "XGBoost (defaults, n_est=200, max_depth=6)",
        "n_scenarios": len(X),
        "accuracy": float(overall_acc),
        "k_used": len(pc),
        "H_bits": H,
        "dominant_share": max(pc.values())/n_total,
        "per_class_recall": {
            cls: float(((y == cls) & (pred_labels == cls)).sum() / max((y == cls).sum(), 1))
            for cls in ICU_ACTIONS
        },
        "predictions": {ids[i]: pred_labels[i] for i in range(len(ids))},
    }
    json.dump(out, open(BASE / "xgboost_baseline_taskB_summary.json", "w"), indent=2)
    print("\nSaved xgboost_baseline_taskB_summary.json")


if __name__ == "__main__":
    main()
