"""
Learned-MLP baseline on Task B (ICU). 3-layer MLP on the upstream-signal
features → gt_action_v1 (6 classes). 5-fold stratified CV gives pooled
predictions on all 150 main scenarios, directly comparable to per-LLM
accuracy in Table I.

Purpose: pre-empt the "your 2-line classifier rule is hand-built; an LLM
might do better than a more sophisticated baseline" attack. If the MLP
also beats every LLM, the contribution generalizes from "rule beats LLM"
to "any simple supervised model on the same inputs beats every LLM."
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import Counter

BASE = Path(__file__).resolve().parents[1]  # repository root
SC_DIR = BASE / "icu_data_v1" / "scenarios"
SEED = 20260517

ICU_ACTIONS = ["continue_monitoring", "recheck_vitals", "adjust_positioning_or_oxygen",
               "fluid_or_vasopressor_adjustment", "escalate_to_attending", "transfer_higher_acuity"]


def extract_features_label(window):
    """Return feature vector + GT action from one scenario window."""
    vs = window["vital_snapshot"]
    feats = [
        vs.get("map_mmhg") or np.nan,
        vs.get("spo2_pct") or np.nan,
        vs.get("hr_bpm") or np.nan,
        vs.get("rr_per_min") or np.nan,
        vs.get("temp_celsius") or np.nan,
        vs.get("lactate_mmol_per_l") if vs.get("lactate_mmol_per_l") is not None else np.nan,
        vs.get("fio2_frac") or 0.21,
        vs.get("fluid_bolus_ml_past_3h") or 0,
        len(vs.get("vasopressors_active", [])),
        vs.get("news2_score") or 0,
        vs.get("news2_6h_delta") or 0,
    ]
    label = window["ground_truth"]["action"]
    return feats, label


def load_data():
    X, y, ids = [], [], []
    for p in sorted(SC_DIR.glob("*.json")):
        j = json.load(open(p))
        w = j["window_stream"][0]
        feats, label = extract_features_label(w)
        X.append(feats); y.append(label); ids.append(j["scenario_id"])
    X = np.array(X, dtype=float)
    y = np.array(y)
    return X, y, ids


def main():
    X, y, ids = load_data()
    n = len(X)
    print(f"Loaded {n} scenarios. Feature dim: {X.shape[1]}")
    print(f"GT distribution: {dict(Counter(y))}")

    # 5-fold stratified CV; we won't tune hyperparameters here (default MLP
    # to keep it strictly a baseline, not an engineered model).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = np.empty(n, dtype=object)
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16),
                                  max_iter=2000, random_state=SEED,
                                  early_stopping=False)),
        ])
        pipe.fit(X[tr], y[tr])
        predictions[te] = pipe.predict(X[te])
        acc_fold = (predictions[te] == y[te]).mean()
        print(f"  fold {fold_i}: n_test={len(te)}, acc={acc_fold:.3f}")

    overall_acc = (predictions == y).mean()
    print(f"\nMLP overall accuracy (pooled across 5 folds): {overall_acc:.3f}")
    print(f"  Correct: {int((predictions == y).sum())} / {n}")

    # Per-class recall
    print(f"\nPer-class recall:")
    for c in ICU_ACTIONS:
        mask = y == c
        n_c = mask.sum()
        if n_c == 0:
            print(f"  {c}: (no GT)")
            continue
        recall = (predictions[mask] == c).mean()
        print(f"  R({c}) = {(predictions[mask] == c).sum()}/{n_c} = {recall:.3f}")

    # Save scenario-level predictions for downstream comparison
    out = {ids[i]: {"gt": y[i], "mlp_pred": predictions[i]} for i in range(n)}
    with open(BASE / "mlp_baseline_predictions.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved mlp_baseline_predictions.json")

    # Also save aggregate summary
    summary = {
        "n_scenarios": n,
        "n_features": X.shape[1],
        "feature_names": ["map_mmhg", "spo2_pct", "hr_bpm", "rr_per_min", "temp_celsius",
                          "lactate_mmol_per_l", "fio2_frac", "fluid_bolus_ml_past_3h",
                          "n_vasopressors_active", "news2_score", "news2_6h_delta"],
        "mlp_architecture": "(32, 16) hidden layers; ReLU; Adam; max_iter=2000",
        "cv": "5-fold stratified",
        "seed": SEED,
        "overall_accuracy": float(overall_acc),
        "per_class_recall": {c: float((predictions[y==c] == c).mean()) if (y==c).sum() else None
                              for c in ICU_ACTIONS},
        "gt_distribution": {c: int((y==c).sum()) for c in ICU_ACTIONS},
        "predicted_distribution": {c: int((predictions==c).sum()) for c in ICU_ACTIONS},
    }
    with open(BASE / "mlp_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved mlp_baseline_summary.json")


if __name__ == "__main__":
    main()
