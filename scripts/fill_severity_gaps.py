#!/usr/bin/env python3
"""Fill severe-band gaps for MoCA/MMSE/Barthel/NDDI-E + tag all assessments synthetic/real."""
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clinical_db as cdb

severe_extra = {
  "EPAT021": {"MOCA": [1, 1, 1, 1, 1, 0, 0], "MMSE": [1, 1, 1, 1, 0, 1, 0], "BARTHEL": [0, 0, 0, 5, 0, 0, 0, 5, 0, 0], "PHQ9": [3, 3, 3, 3, 2, 2, 2, 2, 1], "GAD7": [3, 3, 3, 3, 2, 2, 3]},
  "EPAT022": {"MOCA": [2, 1, 1, 1, 0, 0, 0], "MMSE": [2, 1, 0, 1, 0, 1, 0], "BARTHEL": [5, 0, 0, 0, 5, 0, 0, 5, 0, 0], "NDDIE": [3, 3, 3, 3, 3, 3]},
}
for pid, prof in severe_extra.items():
    cdb.upsert_patient(pid, name=f"Epilepsy Patient {pid[-3:]}", age=65, gender="Male", disease="epilepsy", department="Patient Registry")
    for inst, vals in prof.items():
        cdb.save_assessment(pid, inst, {f"item{i+1}": v for i, v in enumerate(vals)}, examiner="SYNTHETIC-high")
print("added 2 severe-profile patients")

c = sqlite3.connect(str(Path(__file__).resolve().parent.parent / "data" / "clinical.db"))
c.execute("UPDATE assessments SET examiner='SYNTHETIC' WHERE examiner LIKE 'clinical%' OR examiner='UI' OR examiner IS NULL OR examiner=''")
c.execute("UPDATE assessments SET examiner='REAL' WHERE patient_id='CASE001'")
c.commit()
print("tagged: synthetic -> 'SYNTHETIC', CASE001 -> 'REAL'")

print("\nPer-instrument severity coverage:")
print("  %-10s %7s %6s %9s %7s" % ("instrument", "normal", "mild", "moderate", "severe"))
for inst in ["MOCA", "PHQ9", "GAD7", "QOLIE31", "MMSE", "BARTHEL", "EPWORTH", "NDDIE"]:
    rows = c.execute("SELECT level,count(*) FROM assessments WHERE instrument=? GROUP BY level", (inst,)).fetchall()
    dd = {r[0]: r[1] for r in rows}
    if dd:
        print("  %-10s %7d %6d %9d %7d" % (inst, dd.get("normal", 0), dd.get("mild", 0), dd.get("moderate", 0), dd.get("severe", 0)))
print("\ntag distribution:", dict(c.execute("SELECT examiner,count(*) FROM assessments GROUP BY examiner").fetchall()))
