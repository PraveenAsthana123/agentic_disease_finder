"""Seed surgical_outcomes table in clinical.db with realistic epilepsy surgery data.

Generates ~28 surgical outcome records for ~22 patients using clinically
realistic distributions:
- Engel I (seizure free) ~65%, Engel II ~20%, Engel III ~10%, Engel IV ~5%
- Temporal lobectomy ~50%, lesionectomy ~15%, LITT ~10%, VNS ~10%, others
- Complications in ~15-20% of cases
- Follow-up 6-60 months
"""

import pathlib
import random
import sqlite3

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"

random.seed(42)


def seed():
    con = sqlite3.connect(str(DB))
    c = con.cursor()

    # Get existing patient_ids
    c.execute("SELECT patient_id FROM patients")
    all_patients = [r[0] for r in c.fetchall()]
    if not all_patients:
        print("No patients found in patients table — seed patients first.")
        return

    # Pick ~22 patients for surgery records
    surgery_patients = random.sample(all_patients, min(22, len(all_patients)))

    # Idempotent: drop and recreate
    c.execute("DROP TABLE IF EXISTS surgical_outcomes")
    c.execute("""
        CREATE TABLE surgical_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            surgery_type TEXT,
            surgery_date TEXT,
            hemisphere TEXT,
            pathology TEXT,
            engel_class TEXT,
            ilae_outcome INTEGER,
            follow_up_months INTEGER,
            seizure_free INTEGER,
            aed_reduction INTEGER,
            complications TEXT,
            complication_severity TEXT,
            pre_surgery_frequency REAL,
            post_surgery_frequency REAL,
            notes TEXT
        )
    """)

    # --- Distributions ---
    surgery_types = [
        ("temporal lobectomy", 14),
        ("lesionectomy", 4),
        ("LITT", 3),
        ("VNS implant", 3),
        ("RNS implant", 1),
        ("hemispherectomy", 1),
        ("corpus callosotomy", 1),
        ("multiple subpial transections", 1),
    ]
    type_pool = []
    for stype, weight in surgery_types:
        type_pool.extend([stype] * weight)

    pathology_by_type = {
        "temporal lobectomy": ["mesial temporal sclerosis", "mesial temporal sclerosis",
                               "mesial temporal sclerosis", "focal cortical dysplasia",
                               "low-grade tumor", "cavernoma"],
        "lesionectomy": ["focal cortical dysplasia", "cavernoma", "tumor",
                         "arteriovenous malformation"],
        "LITT": ["mesial temporal sclerosis", "hypothalamic hamartoma",
                 "focal cortical dysplasia"],
        "VNS implant": ["multifocal epilepsy", "generalized epilepsy",
                        "Lennox-Gastaut syndrome"],
        "RNS implant": ["bilateral mesial temporal sclerosis",
                        "eloquent cortex focal epilepsy"],
        "hemispherectomy": ["Rasmussen encephalitis", "hemispheric cortical dysplasia",
                           "Sturge-Weber syndrome"],
        "corpus callosotomy": ["Lennox-Gastaut syndrome", "generalized epilepsy",
                               "drop attacks"],
        "multiple subpial transections": ["eloquent cortex focal epilepsy",
                                          "Landau-Kleffner syndrome"],
    }

    hemisphere_options = ["left", "right", "bilateral"]

    engel_pool = (
        ["IA"] * 12 + ["IB"] * 3 + ["IC"] * 2 + ["ID"] * 1 +    # ~65% Engel I
        ["IIA"] * 3 + ["IIB"] * 2 + ["IID"] * 1 +                 # ~20% Engel II
        ["IIIA"] * 2 + ["IIIB"] * 1 +                             # ~10% Engel III
        ["IVA"] * 1 + ["IVB"] * 1                                 # ~5% Engel IV
    )

    engel_to_ilae = {
        "IA": 1, "IB": 2, "IC": 2, "ID": 2,
        "IIA": 3, "IIB": 3, "IID": 4,
        "IIIA": 4, "IIIB": 5,
        "IVA": 5, "IVB": 6,
    }

    complication_options = [
        ("infection", "minor"), ("infection", "major"),
        ("hemorrhage", "major"), ("visual field deficit", "minor"),
        ("memory decline", "minor"), ("memory decline", "major"),
        ("hemiparesis", "major"), ("aphasia", "major"),
        ("none", None),
    ]

    records = []
    # ~22 patients, some get 2 surgeries => ~28 records
    patients_with_two = random.sample(surgery_patients, min(6, len(surgery_patients)))

    for pid in surgery_patients:
        n_surgeries = 2 if pid in patients_with_two else 1
        for s_idx in range(n_surgeries):
            stype = random.choice(type_pool)
            pathology = random.choice(pathology_by_type[stype])

            # Hemisphere — bilateral for certain types
            if stype in ("corpus callosotomy", "VNS implant"):
                hemisphere = "bilateral"
            elif stype == "hemispherectomy":
                hemisphere = random.choice(["left", "right"])
            else:
                hemisphere = random.choice(["left", "right"])

            engel = random.choice(engel_pool)
            # Worse outcomes for device implants and repeat surgeries
            if stype in ("VNS implant", "RNS implant") and random.random() < 0.5:
                engel = random.choice(["IIA", "IIB", "IIIA", "IIIB"])
            if s_idx == 1 and random.random() < 0.4:
                engel = random.choice(["IIA", "IIB", "IIIA"])

            ilae = engel_to_ilae.get(engel, 3)
            seizure_free = 1 if engel.startswith("IA") else 0

            # Follow-up: 6-60 months
            follow_up = random.randint(6, 60)

            # AED reduction — more likely if seizure free
            if seizure_free:
                aed_reduction = 1 if random.random() < 0.7 else 0
            else:
                aed_reduction = 1 if random.random() < 0.15 else 0

            # Pre-surgery frequency: 2-30/month
            pre_freq = round(random.uniform(2.0, 30.0), 1)
            # Post-surgery frequency based on Engel
            if engel.startswith("IA"):
                post_freq = 0.0
            elif engel.startswith("I"):
                post_freq = round(random.uniform(0.0, 1.0), 1)
            elif engel.startswith("II"):
                post_freq = round(pre_freq * random.uniform(0.05, 0.25), 1)
            elif engel.startswith("III"):
                post_freq = round(pre_freq * random.uniform(0.25, 0.60), 1)
            else:
                post_freq = round(pre_freq * random.uniform(0.60, 1.0), 1)

            # Complications: ~18% chance
            if random.random() < 0.18:
                comp, sev = random.choice(complication_options[:-1])  # exclude "none"
            else:
                comp = None
                sev = None

            # Surgery date: 2022-01 to 2025-12
            year = random.randint(2022, 2025)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            surgery_date = f"{year}-{month:02d}-{day:02d}"

            # Second surgery later
            if s_idx == 1:
                year2 = min(year + random.randint(1, 2), 2025)
                surgery_date = f"{year2}-{month:02d}-{day:02d}"

            notes_options = [
                None,
                "Intraoperative ECoG confirmed seizure focus.",
                "Wada test confirmed language lateralization.",
                "MRI-guided stereotactic approach.",
                "Stereo-EEG confirmed bilateral independent foci.",
                "Phase II monitoring with subdural grids.",
                "Neuropsychological testing stable at follow-up.",
                "PET hypometabolism concordant with MRI lesion.",
                "MEG dipole cluster at resection margin.",
            ]
            notes = random.choice(notes_options)

            records.append((
                pid, stype, surgery_date, hemisphere, pathology,
                engel, ilae, follow_up, seizure_free, aed_reduction,
                comp, sev, pre_freq, post_freq, notes
            ))

    c.executemany("""
        INSERT INTO surgical_outcomes
            (patient_id, surgery_type, surgery_date, hemisphere, pathology,
             engel_class, ilae_outcome, follow_up_months, seizure_free,
             aed_reduction, complications, complication_severity,
             pre_surgery_frequency, post_surgery_frequency, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    con.commit()
    print(f"Seeded {len(records)} surgical_outcomes for {len(surgery_patients)} patients.")

    # Verify
    c.execute("SELECT COUNT(*) FROM surgical_outcomes")
    print(f"Table row count: {c.fetchone()[0]}")
    c.execute("SELECT engel_class, COUNT(*) FROM surgical_outcomes GROUP BY engel_class ORDER BY engel_class")
    print("Engel distribution:", dict(c.fetchall()))
    c.execute("SELECT surgery_type, COUNT(*) FROM surgical_outcomes GROUP BY surgery_type ORDER BY COUNT(*) DESC")
    print("Surgery types:", dict(c.fetchall()))

    con.close()


if __name__ == "__main__":
    seed()
