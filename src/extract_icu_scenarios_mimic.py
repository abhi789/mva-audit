"""
MIMIC-IV v3.1 → scenario extraction parallel to extract_icu_scenarios.py (eICU).
Produces identical scenario JSON schema so run_agent_prompt9_icu.py runs without
modification on either eICU or MIMIC scenarios.

Schema-level differences from eICU we account for:
  - MIMIC uses long-format chartevents (one row per measurement) keyed by itemid;
    eICU has vitalPeriodic with column-per-vital. We pivot.
  - MIMIC vasopressor doses are stored in inputevents.rate with rateuom
    ('mcg/kg/min', 'units/min', etc.) directly — no string-parsing needed.
  - MIMIC offsets are absolute datetimes (charttime); we convert to
    minutes-since-intime to match eICU's *offset semantics.
  - MIMIC patients table is in hosp/, icustays in icu/. We need both.

GT mapping is BIT-IDENTICAL to eICU's: we import apply_gt_icu_v1, compute_news2,
and compute_deterioration_signal directly from extract_icu_scenarios.py.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Reuse the GT-application functions from the eICU extractor so the rule
# evaluation is provably identical across databases.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_icu_scenarios import (
    apply_gt_icu_v1,
    compute_news2,
    compute_deterioration_signal,
    PREREG_SEED,
    TARGET_TOTAL,
    HELD_OUT,
    STRATIFY_TARGETS,
    HELDOUT_TARGETS,
    sample_to_targets,
)

BASE = Path(__file__).resolve().parents[1]  # repository root
MIMIC = Path.home() / "mimic-iv"  # user downloaded to ~/mimic-iv/
OUT_DIR = BASE / "icu_data_mimic_v1"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "scenarios").mkdir(exist_ok=True)


# MIMIC-IV chartevents itemids for the vitals we need
ITEMID = {
    "hr":       220045,  # Heart Rate
    "map_abp":  220052,  # Arterial Blood Pressure mean (a-line)
    "map_nbp":  220181,  # Non Invasive Blood Pressure mean (cuff)
    "sbp_abp":  220050,  # Arterial BP Systolic
    "sbp_nbp":  220179,  # Non Invasive BP Systolic
    "rr":       220210,  # Respiratory Rate
    "spo2":     220277,  # O2 saturation pulseoxymetry
    "temp_c":   223762,  # Temperature Celsius
    "temp_f":   223761,  # Temperature Fahrenheit (we'll convert)
    "fio2":     223835,  # Inspired O2 Fraction
    "weight":   224639,  # Daily Weight (kg)
    "gcs_total": 226755, # GCS Total
}
VITAL_ITEMIDS = set(ITEMID.values())

# MIMIC-IV labevents itemid for lactate
LACTATE_ITEMID = 50813

# MIMIC-IV inputevents itemids for vasopressors
PRESSOR_ITEMIDS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    222315: "vasopressin",
}
# Common crystalloid itemids in MIMIC inputevents
CRYSTALLOID_ITEMIDS = {
    225158,  # NaCl 0.9% (saline)
    225828,  # LR (Lactated Ringers)
    225943,  # Plasmalyte
    225823,  # D5LR
    225825,  # D5NS
    225827,  # D5W (counts as fluid)
}


def load_icustays():
    """ICU stays with adult age, LOS>=6h (matches eICU eligibility)."""
    print("Loading icustays + patients + admissions...")
    icu = pd.read_csv(MIMIC / "icu" / "icustays.csv.gz",
                     parse_dates=["intime", "outtime"])
    pats = pd.read_csv(MIMIC / "hosp" / "patients.csv.gz")
    adms = pd.read_csv(MIMIC / "hosp" / "admissions.csv.gz",
                     usecols=["hadm_id", "admission_type", "admission_location"])
    # ICU stays >= 6h
    icu["los_minutes"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 60.0
    icu = icu[icu["los_minutes"] >= 360].copy()
    # Merge in age (MIMIC has anchor_age per subject)
    icu = icu.merge(pats[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left")
    icu = icu[icu["anchor_age"] >= 18].copy()
    # Merge admission_type as a proxy for eICU's apacheadmissiondx
    icu = icu.merge(adms, on="hadm_id", how="left")
    print(f"  {len(icu):,} adult ICU stays, LOS>=6h")
    return icu


def load_d_items():
    return pd.read_csv(MIMIC / "icu" / "d_items.csv.gz", usecols=["itemid", "label"])


def load_filtered_chartevents(target_stay_ids):
    """Stream chartevents in chunks, retain only target stay_ids + vital itemids.
    Returns DataFrame with columns: stay_id, charttime, itemid, valuenum."""
    print(f"  Filtering chartevents.csv.gz for {len(target_stay_ids):,} stays × "
          f"{len(VITAL_ITEMIDS)} vital itemids...")
    target_stay_ids = set(int(x) for x in target_stay_ids)
    chunks = pd.read_csv(
        MIMIC / "icu" / "chartevents.csv.gz",
        chunksize=2_000_000,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    kept = []
    n_total = 0
    for chunk in chunks:
        n_total += len(chunk)
        mask = (chunk["stay_id"].isin(target_stay_ids)
                & chunk["itemid"].isin(VITAL_ITEMIDS)
                & chunk["valuenum"].notna())
        if mask.any():
            kept.append(chunk[mask])
        if n_total % 20_000_000 == 0:
            print(f"    scanned {n_total:,} rows, kept {sum(len(k) for k in kept):,}")
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    print(f"    scanned {n_total:,} rows total, kept {len(df):,} vital rows")
    return df


def load_filtered_labevents(target_stay_ids):
    """Filter labevents for lactate measurements in target stays."""
    print(f"  Filtering labevents.csv.gz for lactate (itemid {LACTATE_ITEMID})...")
    chunks = pd.read_csv(
        MIMIC / "hosp" / "labevents.csv.gz",
        chunksize=2_000_000,
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    target_subject_ids = set(int(x) for x in target_stay_ids["subject_id"])
    kept = []
    n_total = 0
    for chunk in chunks:
        n_total += len(chunk)
        mask = (chunk["subject_id"].isin(target_subject_ids)
                & (chunk["itemid"] == LACTATE_ITEMID)
                & chunk["valuenum"].notna())
        if mask.any():
            kept.append(chunk[mask])
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    print(f"    kept {len(df):,} lactate rows")
    return df


def load_filtered_inputevents(target_stay_ids):
    """Vasopressors + crystalloid boluses."""
    print(f"  Filtering inputevents.csv.gz for vasopressors + crystalloid...")
    relevant_items = set(PRESSOR_ITEMIDS) | CRYSTALLOID_ITEMIDS
    chunks = pd.read_csv(
        MIMIC / "icu" / "inputevents.csv.gz",
        chunksize=2_000_000,
        usecols=["stay_id", "starttime", "endtime", "itemid", "amount", "amountuom",
                 "rate", "rateuom", "patientweight"],
        parse_dates=["starttime", "endtime"],
    )
    target = set(int(x) for x in target_stay_ids)
    kept = []
    for chunk in chunks:
        mask = chunk["stay_id"].isin(target) & chunk["itemid"].isin(relevant_items)
        if mask.any():
            kept.append(chunk[mask])
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    print(f"    kept {len(df):,} pressor+crystalloid rows")
    return df


def vitals_at_window(stay_chartevents, intime, offset_min, lookback_min=60):
    """End-of-window vitals snapshot.
    stay_chartevents: subset of chartevents for this stay (pre-pivoted by itemid).
    intime: pd.Timestamp = ICU admit time.
    offset_min: minutes since intime for window end.
    Returns dict matching eICU's `snapshot` schema OR None if too sparse."""
    end_time = intime + pd.Timedelta(minutes=offset_min)
    eow_start = end_time - pd.Timedelta(minutes=30)
    eow = stay_chartevents[(stay_chartevents["charttime"] >= eow_start)
                            & (stay_chartevents["charttime"] <= end_time)]
    if eow.empty:
        return None

    def med(itemid_list):
        sub = eow[eow["itemid"].isin(itemid_list)]
        if sub.empty:
            return None
        return float(sub["valuenum"].median())

    # Pick whichever MAP/SBP we have (prefer arterial, fall back to cuff)
    snapshot = {
        "temperature": _temp_celsius(med([ITEMID["temp_c"]]), med([ITEMID["temp_f"]])),
        "sao2": med([ITEMID["spo2"]]),
        "heartrate": med([ITEMID["hr"]]),
        "respiration": med([ITEMID["rr"]]),
        "systemicmean": med([ITEMID["map_abp"]]) or med([ITEMID["map_nbp"]]),
        "systemicsystolic": med([ITEMID["sbp_abp"]]) or med([ITEMID["sbp_nbp"]]),
    }
    if (snapshot["systemicmean"] is None or snapshot["sao2"] is None
            or snapshot["heartrate"] is None):
        return None

    # Trends: median in window [-2h, -1h] vs current end
    def median_in_window(start_min, end_min, itemid_list):
        ws = intime + pd.Timedelta(minutes=start_min)
        we = intime + pd.Timedelta(minutes=end_min)
        sub = stay_chartevents[(stay_chartevents["charttime"] >= ws)
                                & (stay_chartevents["charttime"] <= we)
                                & stay_chartevents["itemid"].isin(itemid_list)]
        if sub.empty:
            return None
        return float(sub["valuenum"].median())

    map_now = snapshot["systemicmean"]
    map_1h = median_in_window(offset_min - 120, offset_min - 60,
                               [ITEMID["map_abp"], ITEMID["map_nbp"]])
    map_6h = median_in_window(offset_min - 360, offset_min - 300,
                               [ITEMID["map_abp"], ITEMID["map_nbp"]])
    snapshot["map_1h_trend"] = (map_now - map_1h) if map_1h is not None else 0
    snapshot["map_6h_trend"] = (map_now - map_6h) if map_6h is not None else 0
    spo2_1h = median_in_window(offset_min - 120, offset_min - 60, [ITEMID["spo2"]])
    snapshot["spo2_1h_trend"] = (snapshot["sao2"] - spo2_1h) if spo2_1h is not None else 0
    hr_1h = median_in_window(offset_min - 120, offset_min - 60, [ITEMID["hr"]])
    snapshot["hr_1h_trend"] = (snapshot["heartrate"] - hr_1h) if hr_1h is not None else 0
    return snapshot


def _temp_celsius(temp_c, temp_f):
    if temp_c is not None:
        return temp_c
    if temp_f is not None:
        return (temp_f - 32.0) * 5.0 / 9.0
    return None


def recent_lactate(stay_subject_id, stay_intime, offset_min, lab_df,
                   primary_h=2, fallback_h=6):
    """Most-recent lactate within primary or fallback window."""
    target_time = stay_intime + pd.Timedelta(minutes=offset_min)
    sub = lab_df[lab_df["subject_id"] == stay_subject_id]
    if sub.empty:
        return None, None
    for h in (primary_h, fallback_h):
        win_start = target_time - pd.Timedelta(hours=h)
        win = sub[(sub["charttime"] >= win_start) & (sub["charttime"] <= target_time)]
        if not win.empty:
            row = win.sort_values("charttime").iloc[-1]
            age_h = (target_time - row["charttime"]).total_seconds() / 3600.0
            return float(row["valuenum"]), age_h
    return None, None


def active_vasopressors(stay_id, stay_intime, offset_min, weight_kg, input_df,
                        lookback_min=60):
    """Vasopressors active in [offset-lookback, offset]. Same return schema as
    eICU's get_active_vasopressors."""
    target_time = stay_intime + pd.Timedelta(minutes=offset_min)
    window_start = target_time - pd.Timedelta(minutes=lookback_min)
    sub = input_df[(input_df["stay_id"] == stay_id)
                    & input_df["itemid"].isin(PRESSOR_ITEMIDS)
                    & (input_df["endtime"] >= window_start)
                    & (input_df["starttime"] <= target_time)]
    if sub.empty:
        return []
    actives = []
    for itemid, drug in PRESSOR_ITEMIDS.items():
        drug_sub = sub[sub["itemid"] == itemid]
        if drug_sub.empty:
            continue
        # Dose: prefer rate column (MIMIC stores mcg/kg/min directly for vasoactives)
        max_dose = None
        for _, row in drug_sub.iterrows():
            rate = row.get("rate")
            rate_uom = (row.get("rateuom") or "").lower()
            if pd.isna(rate) or rate == 0:
                continue
            if drug == "vasopressin":
                # Vasopressin units/min in MIMIC
                if max_dose is None or rate > max_dose:
                    max_dose = float(rate)
            else:
                # Norepinephrine/Epinephrine: ensure mcg/kg/min
                if "mcg/kg/min" in rate_uom:
                    dose = float(rate)
                elif "mcg/min" in rate_uom and weight_kg:
                    dose = float(rate) / weight_kg
                else:
                    continue
                if max_dose is None or dose > max_dose:
                    max_dose = dose
        if max_dose is None:
            continue
        # Duration: earliest start of this drug in past 24h
        all_drug = input_df[(input_df["stay_id"] == stay_id)
                              & (input_df["itemid"] == itemid)
                              & (input_df["starttime"] >= target_time - pd.Timedelta(hours=24))]
        if all_drug.empty:
            duration_h = 0.0
        else:
            earliest = all_drug["starttime"].min()
            duration_h = (target_time - earliest).total_seconds() / 3600.0
        active = {"drug": drug, "duration_hours": round(duration_h, 1)}
        if drug == "vasopressin":
            active["dose_units_min"] = max_dose
        else:
            active["dose_ugkgmin"] = max_dose
        actives.append(active)
    return actives


def fluid_bolus_ml(stay_id, stay_intime, offset_min, lookback_min, input_df):
    """Sum crystalloid bolus volume in past `lookback_min`."""
    target_time = stay_intime + pd.Timedelta(minutes=offset_min)
    window_start = target_time - pd.Timedelta(minutes=lookback_min)
    sub = input_df[(input_df["stay_id"] == stay_id)
                    & input_df["itemid"].isin(CRYSTALLOID_ITEMIDS)
                    & (input_df["starttime"] >= window_start)
                    & (input_df["starttime"] <= target_time)]
    if sub.empty:
        return 0.0
    # MIMIC inputevents.amount is in mL for fluids when amountuom='ml'
    total = 0.0
    for _, row in sub.iterrows():
        if (row.get("amountuom") or "").lower() == "ml":
            try:
                total += float(row["amount"])
            except (TypeError, ValueError):
                pass
    return total


def get_weight(stay_id, stay_chartevents, anchor_default=70.0):
    """Daily weight from chartevents; fallback to 70 kg if not recorded."""
    sub = stay_chartevents[stay_chartevents["itemid"] == ITEMID["weight"]]
    if sub.empty:
        return anchor_default
    return float(sub["valuenum"].median())


def pick_random_window(stay_los_minutes, rng):
    earliest = 360
    latest = int(stay_los_minutes)
    if latest <= earliest:
        return None
    return rng.randint(earliest, latest)


def extract_one_window(stay_row, stay_chartevents, lab_df, input_df, offset_min):
    intime = stay_row["intime"]
    snap = vitals_at_window(stay_chartevents, intime, offset_min)
    if snap is None:
        return None

    weight_kg = get_weight(stay_row["stay_id"], stay_chartevents)
    vasopressors = active_vasopressors(
        stay_row["stay_id"], intime, offset_min, weight_kg, input_df)
    fluid_3h = fluid_bolus_ml(stay_row["stay_id"], intime, offset_min, 180, input_df)
    fluid_1h = fluid_bolus_ml(stay_row["stay_id"], intime, offset_min, 60, input_df)

    lactate, lactate_age = recent_lactate(
        stay_row["subject_id"], intime, offset_min, lab_df)

    # FiO2: median in EOW from chartevents (item 223835), as fraction
    end_time = intime + pd.Timedelta(minutes=offset_min)
    fio2_sub = stay_chartevents[(stay_chartevents["itemid"] == ITEMID["fio2"])
                                 & (stay_chartevents["charttime"] >= end_time - pd.Timedelta(minutes=120))
                                 & (stay_chartevents["charttime"] <= end_time)]
    if fio2_sub.empty:
        fio2_frac = 0.21
    else:
        v = float(fio2_sub["valuenum"].median())
        fio2_frac = v / 100.0 if v > 1.0 else v
    on_supp_o2 = fio2_frac > 0.21

    gcs_total = 15
    gcs_sub = stay_chartevents[(stay_chartevents["itemid"] == ITEMID["gcs_total"])
                                 & (stay_chartevents["charttime"] <= end_time)]
    if not gcs_sub.empty:
        gcs_total = int(gcs_sub.sort_values("charttime").iloc[-1]["valuenum"])

    news2 = compute_news2(snap, on_supp_o2, gcs_total)

    snap_6h = vitals_at_window(stay_chartevents, intime, offset_min - 360)
    if snap_6h is not None:
        news2_6h_ago = compute_news2(snap_6h, on_supp_o2, gcs_total)
        news2_6h_delta = news2 - news2_6h_ago
    else:
        news2_6h_delta = 0

    gt_action, gt_rule = apply_gt_icu_v1(
        snap, vasopressors, lactate, lactate_age,
        fluid_3h, fluid_1h, weight_kg,
        news2, news2_6h_delta, gcs_total, fio2_frac,
    )

    deterioration = compute_deterioration_signal(news2, news2_6h_delta)

    window = {
        "window_number": 1,
        "elapsed_seconds": int(offset_min * 60),
        "session_context": {
            "age": int(stay_row["anchor_age"]),
            "sex": stay_row["gender"],
            "care_unit_type": stay_row.get("first_careunit") or "MICU",
            "apache_admission_dx": (stay_row.get("admission_type") or "") + " / "
                                    + (stay_row.get("admission_location") or ""),
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
            "lactate_age_hours": round(lactate_age, 1) if lactate_age is not None else None,
            "vasopressors_active": vasopressors,
            "fluid_bolus_ml_past_3h": round(fluid_3h, 0),
            "news2_score": news2,
            "news2_6h_delta": news2_6h_delta,
        },
        "ground_truth": {"action": gt_action, "rule": gt_rule},
    }
    return window


def main(dry_run_n=None):
    rng = random.Random(PREREG_SEED)
    stays = load_icustays()
    rng_for_shuffle = random.Random(PREREG_SEED)
    stay_order = list(range(len(stays)))
    rng_for_shuffle.shuffle(stay_order)
    stays = stays.iloc[stay_order].reset_index(drop=True)

    pool_target = TARGET_TOTAL + HELD_OUT
    if dry_run_n is not None:
        stays = stays.head(dry_run_n)
        print(f"  Dry-run on {len(stays):,} candidate stays")

    # Eagerly cap candidate stays for speed — we don't need all 60k stays
    # to fill 150 + 20 with stratification. ~3000 stays is usually enough.
    candidate_cap = 3000 if dry_run_n is None else dry_run_n
    candidate_stays = stays.head(candidate_cap).copy()
    print(f"  Pre-loading data for top {len(candidate_stays):,} candidate stays...")

    target_stay_ids = candidate_stays["stay_id"].tolist()
    chartevents = load_filtered_chartevents(target_stay_ids)
    lab_df = load_filtered_labevents(candidate_stays)
    input_df = load_filtered_inputevents(target_stay_ids)

    # Group chartevents by stay_id for fast per-stay slicing
    by_stay = {sid: g for sid, g in chartevents.groupby("stay_id", sort=False)}

    saturation = {cls: 2 * (STRATIFY_TARGETS[cls] + HELDOUT_TARGETS[cls])
                  for cls in STRATIFY_TARGETS}
    pool_by_class = defaultdict(list)
    n_processed = 0
    n_kept = 0
    for _, stay_row in candidate_stays.iterrows():
        if all(len(pool_by_class.get(cls, [])) >= saturation[cls]
               for cls in STRATIFY_TARGETS):
            print(f"    EARLY STOP at processed={n_processed:,}: all classes saturated")
            break
        if stay_row["stay_id"] not in by_stay:
            continue
        off = pick_random_window(stay_row["los_minutes"], rng)
        if off is None:
            continue
        n_processed += 1
        if n_processed % 100 == 0:
            counts = {cls: len(v) for cls, v in pool_by_class.items()}
            print(f"    processed {n_processed:,}  kept {n_kept:,}  counts={counts}")
        try:
            window = extract_one_window(stay_row, by_stay[stay_row["stay_id"]],
                                          lab_df, input_df, off)
        except Exception as e:
            print(f"    [warn] stay {stay_row['stay_id']}: {e}")
            continue
        if window is None:
            continue
        n_kept += 1
        gt = window["ground_truth"]["action"]
        pool_by_class[gt].append((stay_row, window, off))

    print()
    print(f"Final pool: {n_processed:,} processed, {n_kept:,} kept")
    for cls, lst in pool_by_class.items():
        print(f"  {cls}: {len(lst)}")

    print()
    print(f"Stratified-sampling to {TARGET_TOTAL} main + {HELD_OUT} held-out...")
    pool_copies = {k: list(v) for k, v in pool_by_class.items()}
    main_picks = sample_to_targets(pool_copies, STRATIFY_TARGETS, rng)
    # held-out is a second stratified draw from REMAINING pool — mirror eICU pattern;
    # explicitly EXCLUDE the main_picks before the second draw (prior bug: pool_copies
    # was reused unchanged, allowing duplicates across main/heldout).
    used_ids = {(stay_row["stay_id"], off) for (stay_row, _w, off) in main_picks}
    pool_remaining = {cls: [(r, w, o) for (r, w, o) in v
                              if (r["stay_id"], o) not in used_ids]
                       for cls, v in pool_by_class.items()}
    heldout_picks = sample_to_targets(pool_remaining, HELDOUT_TARGETS, rng)

    def write_scenario(stay_row, window, offset_min, scenario_id):
        scenario = {
            "scenario_id": scenario_id,
            "patientunitstayid": int(stay_row["stay_id"]),
            "care_unit_type": stay_row.get("first_careunit") or "MICU",
            "n_windows": 1,
            "total_duration_seconds": 3600,
            "tags": ["mimic-iv-v3.1"],
            "model_reliability": {
                "news2_proxy": {
                    "calibrated": False,
                    "notes": "NEWS2-magnitude confidence proxy (uncalibrated)"
                }
            },
            "window_stream": [window],
        }
        return scenario

    print(f"  Writing {len(main_picks)} main scenarios...")
    main_dir = OUT_DIR / "scenarios"
    main_dir.mkdir(exist_ok=True)
    for stay_row, window, off in main_picks:
        sid = f"mimic_{int(stay_row['stay_id'])}_{off:05d}"
        scen = write_scenario(stay_row, window, off, sid)
        (main_dir / f"{sid}.json").write_text(json.dumps(scen, indent=2, default=str))

    heldout_dir = OUT_DIR / "scenarios_heldout"
    heldout_dir.mkdir(exist_ok=True)
    print(f"  Writing {len(heldout_picks)} held-out scenarios...")
    for stay_row, window, off in heldout_picks:
        sid = f"mimic_held_{int(stay_row['stay_id'])}_{off:05d}"
        scen = write_scenario(stay_row, window, off, sid)
        (heldout_dir / f"{sid}.json").write_text(json.dumps(scen, indent=2, default=str))

    print()
    print(f"Done. Main: {len(main_picks)} scenarios in {main_dir}")
    print(f"      Held-out: {len(heldout_picks)} scenarios in {heldout_dir}")


if __name__ == "__main__":
    dry = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--dry"):
        dry = int(sys.argv[1].split("=")[1]) if "=" in sys.argv[1] else 100
    main(dry_run_n=dry)
