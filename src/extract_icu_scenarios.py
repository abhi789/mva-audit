"""
Extract ICU scenarios from eicu.db, apply the reference rules (gt_icu_v1),
and write scenario JSONs in the format run_agent_prompt9_icu.py expects.

Sampling protocol:
  - 150 scenarios total, 1 window each (= 1 cycle), stratified by reference action.
  - Inclusion: adults age >= 18, LOS >= 6h, admissionweight present.
  - Sample one 1-hour observation window per included stay.
  - Held-out subset reserved from a separate stratified draw.
  - Seed: 20260517.

Reference rules (paper Appendix A):
  - 6-action priority hierarchy, first-match-wins.
  - NEWS2 (not qSOFA) is the early-warning trigger.
  - Lactate window: 2h primary, 6h fallback.
  - Refractory shock: NE/EPI >= 0.25 ug/kg/min >= 4h + vasopressin + MAP<65.
"""
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # repository root
DB = BASE / "raw_data" / "eicu.db"
OUT_DIR = BASE / "icu_data_v1"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "scenarios").mkdir(exist_ok=True)

PREREG_SEED = 20260517
TARGET_TOTAL = 150
HELD_OUT = 20
STRATIFY_TARGETS = {
    "continue_monitoring":              50,
    "recheck_vitals":                   25,
    "adjust_positioning_or_oxygen":     25,
    "fluid_or_vasopressor_adjustment":  20,
    "escalate_to_attending":            20,
    "transfer_higher_acuity":           10,
}
# Held-out is a second, separate stratified draw matching the same proportions.
HELDOUT_TARGETS = {k: max(1, v * HELD_OUT // TARGET_TOTAL) for k, v in STRATIFY_TARGETS.items()}

# ---------------------------------------------------------------------------
# Drug name parsing for infusionDrug.drugname like "Norepinephrine (mcg/min)"
# ---------------------------------------------------------------------------
NOREPI_RE = re.compile(r"norepinephrine", re.I)
EPI_RE    = re.compile(r"\b(epinephrine|adrenaline)\b", re.I)
VASOPRESSIN_RE = re.compile(r"vasopressin", re.I)
UNIT_RE = re.compile(r"\(([^)]+)\)")


def parse_pressor_dose_ugkgmin(drugname, rate, patient_weight_kg):
    """Convert an infusionDrug row to mcg/kg/min if it's a vasopressor;
    return None if not interpretable. Conservative: drop ml/hr rows where
    we don't know concentration."""
    if rate is None or patient_weight_kg in (None, 0):
        return None
    unit_match = UNIT_RE.search(drugname or "")
    unit = unit_match.group(1).lower() if unit_match else ""
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return None
    if "mcg/kg/min" in unit or "ug/kg/min" in unit:
        return rate
    if "mcg/min" in unit or "ug/min" in unit:
        return rate / patient_weight_kg
    # ml/hr without concentration -> unsafe to convert
    return None


def get_active_vasopressors(conn, stayid, offset_min, patient_weight_kg, lookback_min=60):
    """List of {drug, dose_ugkgmin, duration_hours} active in the window
    [offset_min - lookback_min, offset_min].
    Currently: we approximate 'duration on this drug' by the earliest
    infusion of the same drug class within the 24h prior at non-zero rate."""
    rows = conn.execute(
        "SELECT drugname, drugrate, infusionoffset "
        "FROM infusionDrug WHERE patientunitstayid = ? "
        "AND infusionoffset >= ? AND infusionoffset <= ?",
        (stayid, offset_min - 24 * 60, offset_min),
    ).fetchall()
    by_drug = defaultdict(list)
    for name, rate, off in rows:
        if not name:
            continue
        if NOREPI_RE.search(name):
            drug = "norepinephrine"
        elif EPI_RE.search(name):
            drug = "epinephrine"
        elif VASOPRESSIN_RE.search(name):
            drug = "vasopressin"
        else:
            continue
        if drug == "vasopressin":
            try:
                dose = float(rate) if rate not in (None, "") else None
            except (TypeError, ValueError):
                dose = None  # handles "OFF" and other non-numeric placeholders
        else:
            dose = parse_pressor_dose_ugkgmin(name, rate, patient_weight_kg)
        by_drug[drug].append((off, dose))
    actives = []
    for drug, points in by_drug.items():
        recent = [(o, d) for o, d in points if o >= offset_min - lookback_min and d not in (None, 0)]
        if not recent:
            continue
        # earliest non-zero infusion in past 24h => duration proxy
        all_nonzero = [(o, d) for o, d in points if d not in (None, 0)]
        earliest = min(o for o, _ in all_nonzero)
        max_dose = max(d for _, d in recent)
        duration_hours = (offset_min - earliest) / 60.0
        actives.append({
            "drug": drug,
            "dose_ugkgmin" if drug != "vasopressin" else "dose_units_min": max_dose,
            "duration_hours": round(duration_hours, 1),
        })
    return actives


# ---------------------------------------------------------------------------
# Vitals & labs extraction
# ---------------------------------------------------------------------------
def get_vitals_at_window(conn, stayid, offset_min, lookback_min=60):
    """End-of-window vitals: median of last `lookback_min` of vitalPeriodic
    measurements, plus 1h and 6h trends on MAP/SpO2/HR."""
    rows = conn.execute(
        "SELECT observationoffset, temperature, sao2, heartrate, respiration, "
        "       systemicmean, systemicsystolic "
        "FROM vitalPeriodic WHERE patientunitstayid = ? "
        "AND observationoffset >= ? AND observationoffset <= ? "
        "ORDER BY observationoffset",
        (stayid, offset_min - 6 * 60, offset_min),
    ).fetchall()
    if not rows:
        return None
    # End-of-window snapshot: last 30 min of measurements, median
    eow = [r for r in rows if r[0] >= offset_min - 30]
    if not eow:
        return None

    def med(idx, subset):
        vals = [r[idx] for r in subset if r[idx] not in (None, "")]
        if not vals:
            return None
        vals = sorted(float(v) for v in vals)
        return vals[len(vals) // 2]

    snapshot = {
        "temperature": med(1, eow),
        "sao2": med(2, eow),
        "heartrate": med(3, eow),
        "respiration": med(4, eow),
        "systemicmean": med(5, eow),
        "systemicsystolic": med(6, eow),
    }
    if snapshot["systemicmean"] is None or snapshot["sao2"] is None or snapshot["heartrate"] is None:
        return None  # require core triad

    # Trends: median of [-1h to now] vs [-1h to -2h]
    def median_in(window_start_off, window_end_off, idx):
        subset = [r for r in rows if window_start_off <= r[0] <= window_end_off]
        return med(idx, subset)

    map_now = snapshot["systemicmean"]
    map_1h_ago = median_in(offset_min - 120, offset_min - 60, 5)
    map_6h_ago = median_in(offset_min - 6 * 60, offset_min - 5 * 60, 5)
    snapshot["map_1h_trend"] = (map_now - map_1h_ago) if map_1h_ago is not None else 0
    snapshot["map_6h_trend"] = (map_now - map_6h_ago) if map_6h_ago is not None else 0

    spo2_1h_ago = median_in(offset_min - 120, offset_min - 60, 2)
    snapshot["spo2_1h_trend"] = (snapshot["sao2"] - spo2_1h_ago) if spo2_1h_ago is not None else 0
    hr_1h_ago = median_in(offset_min - 120, offset_min - 60, 3)
    snapshot["hr_1h_trend"] = (snapshot["heartrate"] - hr_1h_ago) if hr_1h_ago is not None else 0
    return snapshot


def get_recent_lab(conn, stayid, labname, offset_min, primary_h=2, fallback_h=6):
    """Most-recent lab value within primary window, fallback to longer window."""
    row = conn.execute(
        "SELECT labresult, labresultoffset FROM lab "
        "WHERE patientunitstayid = ? AND labname = ? "
        "AND labresultoffset <= ? AND labresultoffset >= ? "
        "ORDER BY labresultoffset DESC LIMIT 1",
        (stayid, labname, offset_min, offset_min - primary_h * 60),
    ).fetchone()
    window_h = primary_h
    if row is None:
        row = conn.execute(
            "SELECT labresult, labresultoffset FROM lab "
            "WHERE patientunitstayid = ? AND labname = ? "
            "AND labresultoffset <= ? AND labresultoffset >= ? "
            "ORDER BY labresultoffset DESC LIMIT 1",
            (stayid, labname, offset_min, offset_min - fallback_h * 60),
        ).fetchone()
        window_h = fallback_h
    if row is None or row[0] in (None, ""):
        return None, None
    try:
        return float(row[0]), (offset_min - row[1]) / 60.0
    except (TypeError, ValueError):
        return None, None


def get_fluid_bolus_ml(conn, stayid, offset_min, lookback_min):
    """Approximate crystalloid bolus volume in past `lookback_min`. eICU
    doesn't have a dedicated bolus table; we approximate via infusionDrug
    rows whose drug name contains common crystalloid names."""
    rows = conn.execute(
        "SELECT drugname, volumeoffluid FROM infusionDrug "
        "WHERE patientunitstayid = ? "
        "AND infusionoffset >= ? AND infusionoffset <= ?",
        (stayid, offset_min - lookback_min, offset_min),
    ).fetchall()
    total = 0.0
    crystalloid_re = re.compile(r"NS|normal saline|lactated ringer|LR|crystalloid|plasmalyte", re.I)
    for name, vol in rows:
        if name and crystalloid_re.search(name) and vol not in (None, "", 0):
            try:
                total += float(vol)
            except (TypeError, ValueError):
                pass
    return total


# ---------------------------------------------------------------------------
# NEWS2 score (RCP 2017)
# ---------------------------------------------------------------------------
def compute_news2(snapshot, on_supp_o2, gcs_total):
    rr = snapshot.get("respiration") or 16
    spo2 = snapshot.get("sao2") or 98
    sbp = snapshot.get("systemicsystolic") or 120
    hr = snapshot.get("heartrate") or 80
    temp = snapshot.get("temperature") or 36.8
    rr_score = (3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20
                else 2 if rr <= 24 else 3)
    spo2_score = (3 if spo2 <= 91 else 2 if spo2 <= 93
                  else 1 if spo2 <= 95 else 0)
    supp_o2_score = 2 if on_supp_o2 else 0
    sbp_score = (3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110
                 else 0 if sbp <= 219 else 3)
    hr_score = (3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90
                else 1 if hr <= 110 else 2 if hr <= 130 else 3)
    loc_score = 3 if gcs_total < 15 else 0
    temp_score = (3 if temp <= 35.0 else 1 if temp <= 36.0
                  else 0 if temp <= 38.0 else 1 if temp <= 39.0 else 2)
    return rr_score + spo2_score + supp_o2_score + sbp_score + hr_score + loc_score + temp_score


# ---------------------------------------------------------------------------
# gt_icu_v1 reference rules (paper Appendix A)
# ---------------------------------------------------------------------------
def apply_gt_icu_v1(snap, vasopressors, lactate, lactate_age_h,
                    fluid_3h_ml, fluid_1h_ml, weight_kg, news2_score, news2_6h_delta,
                    gcs_total, fio2_frac):
    map_mmhg = snap["systemicmean"]
    spo2 = snap["sao2"]
    map_1h_trend = snap.get("map_1h_trend", 0)
    spo2_1h_trend = snap.get("spo2_1h_trend", 0)
    hr_1h_trend = snap.get("hr_1h_trend", 0)

    # R1 transfer_higher_acuity
    ne_or_epi_max_4h = 0.0
    on_vasopressin = False
    for v in vasopressors:
        if v["drug"] in ("norepinephrine", "epinephrine") and v["duration_hours"] >= 4:
            d = v.get("dose_ugkgmin") or 0
            if d > ne_or_epi_max_4h:
                ne_or_epi_max_4h = d
        if v["drug"] == "vasopressin":
            on_vasopressin = True
    if ne_or_epi_max_4h >= 0.25 and on_vasopressin and map_mmhg < 65:
        return ("transfer_higher_acuity", "R1_refractory_shock")

    # R2 escalate_to_attending
    fluid_adequate = (fluid_3h_ml / max(1, weight_kg)) >= 30
    on_any_vasopressor = len(vasopressors) > 0
    if map_mmhg < 65 and map_1h_trend <= 0 and fluid_adequate:
        return ("escalate_to_attending", "R2a_map_sustained_post_resus")
    if lactate is not None and lactate >= 2.0 and on_any_vasopressor and fluid_adequate:
        return ("escalate_to_attending", "R2b_sepsis3_septic_shock")
    if news2_score >= 7:
        return ("escalate_to_attending", "R2c_news2_high")
    if news2_score >= 5 and (news2_6h_delta >= 1 or map_1h_trend <= -5
                             or spo2_1h_trend <= -2 or hr_1h_trend >= 10):
        return ("escalate_to_attending", "R2d_news2_med_worsening")

    # R3 fluid_or_vasopressor_adjustment
    if 65 <= map_mmhg <= 70 and map_1h_trend <= -5 and fluid_1h_ml == 0:
        return ("fluid_or_vasopressor_adjustment", "R3a_map_borderline_falling")
    ne_now = 0.0
    for v in vasopressors:
        if v["drug"] == "norepinephrine" and (v.get("dose_ugkgmin") or 0) > ne_now:
            ne_now = v["dose_ugkgmin"]
    if 0 < ne_now < 0.25 and map_mmhg < 65:
        return ("fluid_or_vasopressor_adjustment", "R3b_pressor_titration")

    # R4 adjust_positioning_or_oxygen
    if spo2 < 92:
        return ("adjust_positioning_or_oxygen", "R4a_hypoxia")
    sf_ratio = spo2 / (fio2_frac * 100.0) if fio2_frac else None
    if sf_ratio is not None and spo2 <= 97 and sf_ratio <= 3.15:
        return ("adjust_positioning_or_oxygen", "R4b_sf_ratio")
    # 4c (sudden drop) needs higher-res data than this implementation provides;
    # we approximate via the 1h trend: drop >= 4 pp
    if spo2_1h_trend <= -4:
        return ("adjust_positioning_or_oxygen", "R4c_spo2_drop")

    # R5 recheck_vitals
    # Single-outlier proxy: window's median MAP normal but min was concerning;
    # the lab-window or trend was ambiguous. Operationalized minimally as:
    #   MAP exactly 65 +/- 1, or SpO2 exactly 92 +/- 1.
    if abs(map_mmhg - 65) <= 1 or abs(spo2 - 92) <= 1:
        return ("recheck_vitals", "R5_at_threshold")

    # R6 default
    return ("continue_monitoring", "R6_default")


# ---------------------------------------------------------------------------
# Deterioration prediction (NEWS2-binary classifier-analog)
# ---------------------------------------------------------------------------
def compute_deterioration_signal(news2_score, news2_6h_delta):
    """Binary deterioration label + NEWS2-derived confidence proxy + trend.
    Mirrors the wearable's binary fatigue_prediction structure. Confidence
    is NOT a calibrated probability; documented as such in the pre-reg."""
    NEWS2_MAX_OBSERVED = 14.0
    if news2_score >= 5:
        level = "D"
        conf = min(1.0, 0.5 + (news2_score - 5) / NEWS2_MAX_OBSERVED)
    else:
        level = "S"
        conf = min(1.0, 0.5 + (5 - news2_score) / NEWS2_MAX_OBSERVED)
    if news2_6h_delta >= 2:
        trend = "increasing"
    elif news2_6h_delta <= -2:
        trend = "decreasing"
    else:
        trend = "stable"
    return {"predicted_level": level, "confidence": round(conf, 2), "trend": trend}


# ---------------------------------------------------------------------------
# Eligibility & sampling
# ---------------------------------------------------------------------------
def get_eligible_stays(conn):
    rows = conn.execute(
        "SELECT patientunitstayid, age, admissionweight, "
        "       unittype, unitdischargeoffset, apacheadmissiondx, gender "
        "FROM patient "
        "WHERE admissionweight IS NOT NULL AND admissionweight > 0 "
        "AND unitdischargeoffset >= 360 "
        "AND age NOT IN ('', '> 89') "  # exclude blank-age + >89 (eICU's de-id top-coding)
    ).fetchall()
    out = []
    for stayid, age, weight, unit, los_off, dx, sex in rows:
        try:
            a = int(age)
            if a < 18:
                continue
            out.append({
                "patientunitstayid": stayid,
                "age": a,
                "weight_kg": float(weight),
                "unittype": unit,
                "los_minutes": los_off,
                "apache_dx": dx,
                "sex": sex,
            })
        except (TypeError, ValueError):
            continue
    return out


def pick_random_window(stay, rng):
    """Random 1h window that has at least 6h of prior ICU time for trend
    computation. Window end offset in [360, los_minutes]."""
    earliest = 360  # need 6h prior of vitalPeriodic
    latest = stay["los_minutes"]
    if latest <= earliest:
        return None
    return rng.randint(earliest, latest)


def extract_one_window(conn, stay, offset_min):
    snap = get_vitals_at_window(conn, stay["patientunitstayid"], offset_min)
    if snap is None:
        return None

    # Vasopressor + fluid context
    vasopressors = get_active_vasopressors(
        conn, stay["patientunitstayid"], offset_min, stay["weight_kg"])
    fluid_3h_ml = get_fluid_bolus_ml(
        conn, stay["patientunitstayid"], offset_min, 180)
    fluid_1h_ml = get_fluid_bolus_ml(
        conn, stay["patientunitstayid"], offset_min, 60)

    # Lactate (and FiO2 proxy)
    lactate, lactate_age_h = get_recent_lab(
        conn, stay["patientunitstayid"], "lactate", offset_min)
    fio2_val, _ = get_recent_lab(
        conn, stay["patientunitstayid"], "FiO2", offset_min)
    # FiO2 in eICU is reported as percent (e.g., 40) or fraction; normalize.
    if fio2_val is None:
        fio2_frac = 0.21  # room air default
    elif fio2_val > 1.0:
        fio2_frac = fio2_val / 100.0
    else:
        fio2_frac = fio2_val
    on_supp_o2 = fio2_frac > 0.21

    gcs_total = 15  # placeholder; eICU's nurseCharting has GCS but we skipped it
    news2 = compute_news2(snap, on_supp_o2, gcs_total)

    # Approximate NEWS2 6h delta: re-evaluate at offset_min - 360
    snap_6h = get_vitals_at_window(
        conn, stay["patientunitstayid"], offset_min - 360)
    if snap_6h is not None:
        news2_6h_ago = compute_news2(snap_6h, on_supp_o2, gcs_total)
        news2_6h_delta = news2 - news2_6h_ago
    else:
        news2_6h_delta = 0

    gt_action, gt_rule = apply_gt_icu_v1(
        snap, vasopressors, lactate, lactate_age_h,
        fluid_3h_ml, fluid_1h_ml, stay["weight_kg"],
        news2, news2_6h_delta, gcs_total, fio2_frac,
    )

    deterioration = compute_deterioration_signal(news2, news2_6h_delta)

    window = {
        "window_number": 1,
        "elapsed_seconds": int(offset_min * 60),
        "session_context": {
            "age": stay["age"],
            "sex": stay["sex"],
            "care_unit_type": stay["unittype"],
            "apache_admission_dx": stay["apache_dx"],
        },
        "deterioration_prediction": deterioration,
        "vital_snapshot": {
            "map_mmhg": snap["systemicmean"],
            "spo2_pct": snap["sao2"],
            "fio2_frac": round(fio2_frac, 2),
            "hr_bpm": snap["heartrate"],
            "rr_per_min": snap["respiration"],
            "temp_celsius": snap["temperature"],
            "lactate_mmol_per_l": lactate,
            "lactate_age_hours": round(lactate_age_h, 1) if lactate_age_h is not None else None,
            "vasopressors_active": vasopressors,
            "fluid_bolus_ml_past_3h": round(fluid_3h_ml, 0),
            "news2_score": news2,
            "news2_6h_delta": news2_6h_delta,
        },
        "ground_truth": {"action": gt_action, "rule": gt_rule},
    }
    return window


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def sample_to_targets(pool_by_class, targets, rng):
    """Stratified sample from pool_by_class to hit targets (best-effort)."""
    out = []
    for cls, target in targets.items():
        pool = pool_by_class.get(cls, [])
        rng.shuffle(pool)
        take = pool[:target]
        if len(take) < target:
            print(f"  WARN: class {cls!r} only had {len(pool)} candidates "
                  f"(target {target})")
        out.extend(take)
    return out


def main(dry_run_n=None):
    rng = random.Random(PREREG_SEED)
    conn = sqlite3.connect(DB)

    print("Loading eligible stays ...")
    stays = get_eligible_stays(conn)
    print(f"  {len(stays):,} eligible stays (adult, weight present, LOS>=6h)")

    pool_target = TARGET_TOTAL + HELD_OUT
    rng.shuffle(stays)
    if dry_run_n is not None:
        candidate_stays = stays[:dry_run_n]
        print(f"  scanning {len(candidate_stays):,} candidate stays (dry-run)")
    else:
        candidate_stays = stays
        print(f"  scanning up to {len(candidate_stays):,} candidate stays "
              f"with early-stop on rare-class saturation")

    # Per-class saturation target = 2x main+heldout so we have diversity for sampling
    saturation = {cls: 2 * (STRATIFY_TARGETS[cls] + HELDOUT_TARGETS[cls])
                  for cls in STRATIFY_TARGETS}

    pool_by_class = defaultdict(list)
    n_processed = 0
    n_kept = 0
    for stay in candidate_stays:
        # Early stop: every class has reached its saturation target
        if (dry_run_n is None
                and all(len(pool_by_class.get(cls, [])) >= saturation[cls]
                        for cls in STRATIFY_TARGETS)):
            print(f"    EARLY STOP at processed={n_processed:,}: all classes saturated")
            break
        off = pick_random_window(stay, rng)
        if off is None:
            continue
        n_processed += 1
        if n_processed % 500 == 0:
            counts = {cls: len(v) for cls, v in pool_by_class.items()}
            print(f"    processed {n_processed:,}  kept {n_kept:,}  "
                  f"class counts: {counts}", flush=True)
        win = extract_one_window(conn, stay, off)
        if win is None:
            continue
        cls = win["ground_truth"]["action"]
        # If this class is already saturated, skip storing more to save memory
        if len(pool_by_class[cls]) >= saturation[cls]:
            continue
        pool_by_class[cls].append({
            "scenario_id": f"icu_{stay['patientunitstayid']}_{off:05d}",
            "patientunitstayid": stay["patientunitstayid"],
            "care_unit_type": stay["unittype"],
            "n_windows": 1,
            "total_duration_seconds": 3600,
            "tags": [],
            "model_reliability": {
                "news2_proxy": {"calibrated": False,
                                "notes": "NEWS2-magnitude confidence proxy (uncalibrated)"},
            },
            "window_stream": [win],
        })
        n_kept += 1

    print("\nPool by class (after extraction):")
    for cls in STRATIFY_TARGETS:
        print(f"  {cls:36s} {len(pool_by_class.get(cls, [])):>4} candidates "
              f"(target {STRATIFY_TARGETS[cls]} + heldout {HELDOUT_TARGETS[cls]})")

    if dry_run_n is not None:
        print(f"\nDry-run only ({dry_run_n} stays). No files written.")
        conn.close()
        return

    main_sample = sample_to_targets(pool_by_class, STRATIFY_TARGETS, rng)
    # remove main from pools, then sample held-out
    used_ids = {s["scenario_id"] for s in main_sample}
    pool_remaining = defaultdict(list)
    for cls, items in pool_by_class.items():
        pool_remaining[cls] = [s for s in items if s["scenario_id"] not in used_ids]
    heldout_sample = sample_to_targets(pool_remaining, HELDOUT_TARGETS, rng)

    print(f"\nFinal main sample: {len(main_sample)}, held-out: {len(heldout_sample)}")
    write_dir = OUT_DIR / "scenarios"
    heldout_dir = OUT_DIR / "scenarios_heldout"
    heldout_dir.mkdir(exist_ok=True)
    for s in main_sample:
        with open(write_dir / f"{s['scenario_id']}.json", "w") as f:
            json.dump(s, f, indent=2)
    for s in heldout_sample:
        with open(heldout_dir / f"{s['scenario_id']}.json", "w") as f:
            json.dump(s, f, indent=2)

    # Index file
    with open(OUT_DIR / "scenarios_index.json", "w") as f:
        json.dump({
            "prereg_seed": PREREG_SEED,
            "stratify_targets": STRATIFY_TARGETS,
            "heldout_targets": HELDOUT_TARGETS,
            "main_scenarios": [s["scenario_id"] for s in main_sample],
            "heldout_scenarios": [s["scenario_id"] for s in heldout_sample],
            "main_class_counts": dict(Counter(
                s["window_stream"][0]["ground_truth"]["action"] for s in main_sample)),
            "heldout_class_counts": dict(Counter(
                s["window_stream"][0]["ground_truth"]["action"] for s in heldout_sample)),
        }, f, indent=2)

    print(f"\nWrote {len(main_sample)} scenarios to {write_dir}")
    print(f"Wrote {len(heldout_sample)} heldout scenarios to {heldout_dir}")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--dry"):
        n = int(sys.argv[1].split("=")[1]) if "=" in sys.argv[1] else 50
        main(dry_run_n=n)
    else:
        main()
