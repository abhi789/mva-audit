"""
Stream-load vitalPeriodic.csv.gz into the existing eicu.db with column
projection (keeps only the 9 columns we need for NEWS2 + gt_icu_v1 triggers).

This avoids writing a 30 GB decompressed CSV to disk; pandas reads gzip
chunks directly.

Run AFTER load_eicu_to_sqlite.py.
"""
import sqlite3
import time
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # repository root
GZ = BASE / "raw_data" / "eicu-collaborative-research-database-2.0" / "vitalPeriodic.csv.gz"
DB = BASE / "raw_data" / "eicu.db"

KEEP_COLS = [
    "vitalperiodicid", "patientunitstayid", "observationoffset",
    "temperature", "sao2", "heartrate", "respiration", "systemicmean",
    "systemicsystolic",
]


def main():
    t0 = time.time()
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS vitalPeriodic")
    conn.commit()

    total_rows = 0
    first = True
    chunksize = 500_000
    print(f"Streaming {GZ.name} with chunksize={chunksize}", flush=True)
    for i, chunk in enumerate(pd.read_csv(
            GZ, chunksize=chunksize, usecols=KEEP_COLS,
            compression="gzip", low_memory=False)):
        chunk.to_sql(
            "vitalPeriodic", conn,
            if_exists="replace" if first else "append",
            index=False,
        )
        total_rows += len(chunk)
        first = False
        if i % 20 == 0:
            elapsed = time.time() - t0
            rate = total_rows / max(1, elapsed)
            print(f"  chunk {i:4d}  total {total_rows:>12,}  "
                  f"rate {rate:,.0f} rows/s  elapsed {elapsed:.0f}s",
                  flush=True)

    print(f"\nIndexing vitalPeriodic by patientunitstayid + observationoffset ...", flush=True)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_vitalPeriodic_stay_off "
                "ON vitalPeriodic(patientunitstayid, observationoffset)")
    conn.commit()

    n = cur.execute("SELECT COUNT(*) FROM vitalPeriodic").fetchone()[0]
    db_size_mb = DB.stat().st_size / 1e6
    print(f"\nFinal: {n:,} rows; DB size {db_size_mb:.0f} MB; "
          f"total elapsed {time.time()-t0:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
