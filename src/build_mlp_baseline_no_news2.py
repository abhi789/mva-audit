"""
Learned-MLP baseline on the ICU task WITHOUT NEWS2 features.

The NEWS2 rule, the reference mapping and the 11-feature MLP all use NEWS2.
This variant removes news2_score and news2_6h_delta from the feature set to
bound NEWS2's contribution.

Feature set: 9 features = {map_mmhg, spo2_pct, hr_bpm, rr_per_min,
temp_celsius, lactate_mmol_per_l, fio2_frac, fluid_bolus_ml_past_3h,
n_vasopressors_active}.

Architecture, CV, seed and scoring are identical to build_mlp_baseline.py.
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
    """Return feature vector + GT action; NEWS2 deliberately excluded."""
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


def vital_only_rule(window):
    """Vital-only escalate-if rule (NEWS2-free analog of NEWS2 >= 5 rule).

    Approximates "concerning vital pattern" without using NEWS2 directly:
    escalate-equivalent (recheck) if any of:
      MAP < 65, SpO2 < 92, HR > 130 or < 50, RR > 24 or < 8, lactate >= 2.
    else continue_monitoring.

    This matches the 2-line classifier-rule structure on Task B but with
    NEWS2 replaced by its component thresholds, recovering only the
    vitals subset RCP 2017 weights without using the NEWS2 score itself.
    """
    vs = window["vital_snapshot"]
    map_ = vs.get("map_mmhg") or 80
    spo2 = vs.get("spo2_pct") or 98
    hr = vs.get("hr_bpm") or 80
    rr = vs.get("rr_per_min") or 16
    lac = vs.get("lactate_mmol_per_l")
    if (map_ < 65 or spo2 < 92 or hr > 130 or hr < 50
            or rr > 24 or rr < 8 or (lac is not None and lac >= 2.0)):
        return "recheck_vitals"
    return "continue_monitoring"


def main():
    X, y, ids = load_data()
    n = len(X)
    print(f"Loaded {n} scenarios. Feature dim: {X.shape[1]} (NEWS2 excluded)")
    print(f"GT distribution: {dict(Counter(y))}")

    # 5-fold stratified CV, identical hyperparameters to build_mlp_baseline.py
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
    print(f"\nNo-NEWS2 MLP overall accuracy: {overall_acc:.3f}")
    print(f"  Correct: {int((predictions == y).sum())} / {n}")

    # Per-class recall
    print(f"\nPer-class recall:")
    per_class_recall = {}
    for c in ICU_ACTIONS:
        mask = y == c
        n_c = mask.sum()
        if n_c == 0:
            per_class_recall[c] = None
            continue
        recall = (predictions[mask] == c).mean()
        per_class_recall[c] = float(recall)
        print(f"  R({c}) = {(predictions[mask] == c).sum()}/{n_c} = {recall:.3f}")

    # Also evaluate the NEWS2-free vital-only rule
    rule_preds = []
    for p in sorted(SC_DIR.glob("*.json")):
        j = json.load(open(p))
        w = j["window_stream"][0]
        rule_preds.append(vital_only_rule(w))
    rule_preds = np.array(rule_preds)
    rule_acc = (rule_preds == y).mean()
    print(f"\nNo-NEWS2 vital-only rule accuracy: {rule_acc:.3f}")

    summary = {
        "n_scenarios": n,
        "n_features": X.shape[1],
        "feature_names": ["map_mmhg", "spo2_pct", "hr_bpm", "rr_per_min", "temp_celsius",
                          "lactate_mmol_per_l", "fio2_frac", "fluid_bolus_ml_past_3h",
                          "n_vasopressors_active"],
        "news2_features_removed": ["news2_score", "news2_6h_delta"],
        "mlp_architecture": "(32, 16) hidden layers; ReLU; Adam; max_iter=2000",
        "cv": "5-fold stratified",
        "seed": SEED,
        "mlp_overall_accuracy_no_news2": float(overall_acc),
        "mlp_per_class_recall_no_news2": per_class_recall,
        "vital_only_rule_accuracy_no_news2": float(rule_acc),
        "gt_distribution": {c: int((y==c).sum()) for c in ICU_ACTIONS},
        "mlp_predicted_distribution": {c: int((predictions==c).sum()) for c in ICU_ACTIONS},
        "rule_predicted_distribution": {c: int((rule_preds==c).sum()) for c in ICU_ACTIONS},
    }
    with open(BASE / "mlp_baseline_no_news2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved mlp_baseline_no_news2_summary.json")

    # Also save per-scenario predictions for downstream paired McNemar
    out = {ids[i]: {"gt": y[i], "mlp_pred_no_news2": predictions[i],
                    "rule_pred_no_news2": rule_preds[i]} for i in range(n)}
    with open(BASE / "mlp_baseline_no_news2_predictions.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
