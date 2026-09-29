"""
Load 6 selected eICU-CRD tables into SQLite for fast querying.
Filters `lab` to a labname allowlist on the way in (it's 2.2 GB otherwise).

Usage:
    python load_eicu_to_sqlite.py

Output: raw_data/eicu.db (~1-2 GB)
"""
import sqlite3
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # repository root
CSV_DIR = BASE / "raw_data" / "eicu-collaborative-research-database-2.0"
DB_PATH = BASE / "raw_data" / "eicu.db"

LAB_ALLOWLIST = {
    "lactate", "creatinine", "Hgb", "WBC x 1000", "BUN", "sodium",
    "potassium", "glucose", "bicarbonate", "platelets x 1000",
    "PaO2", "FiO2", "pH",
}

# Tables to load as-is (chunked for memory safety)
SIMPLE_TABLES = [
    ("patient", "patient.csv", None),
    ("apachePatientResult", "apachePatientResult.csv", None),
    ("infusionDrug", "infusionDrug.csv", None),
    ("vitalAperiodic", "vitalAperiodic.csv", None),
    ("medication", "medication.csv", None),
]


def load_table(conn, table_name, csv_path, filter_fn=None, chunksize=200_000):
    print(f"  loading {table_name} from {csv_path.name} ...", flush=True)
    t0 = time.time()
    total_rows = 0
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if filter_fn is not None:
            chunk = filter_fn(chunk)
        if len(chunk) == 0:
            continue
        chunk.to_sql(
            table_name, conn,
            if_exists="replace" if first else "append",
            index=False,
        )
        total_rows += len(chunk)
        first = False
    print(f"    -> {total_rows:,} rows in {time.time()-t0:.1f}s", flush=True)


def filter_lab(chunk):
    return chunk[chunk["labname"].isin(LAB_ALLOWLIST)]


def create_indexes(conn):
    """Indexes on patientunitstayid for join performance."""
    cur = conn.cursor()
    for tbl in ["apachePatientResult", "infusionDrug", "vitalAperiodic",
                "medication", "lab"]:
        idx_name = f"idx_{tbl}_stayid"
        try:
            print(f"  index {idx_name}", flush=True)
            cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} "
                        f"ON {tbl}(patientunitstayid)")
        except sqlite3.OperationalError as e:
            print(f"    skip: {e}")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_patient_stayid "
                "ON patient(patientunitstayid)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lab_name "
                "ON lab(labname)")
    conn.commit()


def main():
    t0 = time.time()
    print(f"Writing to {DB_PATH}")
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)

    for table, fname, ffn in SIMPLE_TABLES:
        load_table(conn, table, CSV_DIR / fname, filter_fn=ffn)

    # lab gets filtered on the way in
    load_table(conn, "lab", CSV_DIR / "lab.csv", filter_fn=filter_lab)

    print("Creating indexes ...", flush=True)
    create_indexes(conn)

    # Sanity counts
    print("\nRow counts:")
    cur = conn.cursor()
    for tbl in ["patient", "apachePatientResult", "infusionDrug",
                "vitalAperiodic", "medication", "lab"]:
        n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:24s} {n:>12,}")

    # File size
    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"\nDB size: {size_mb:.1f} MB")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
