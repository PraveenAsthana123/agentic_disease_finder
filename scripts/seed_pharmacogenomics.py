#!/usr/bin/env python3
"""Seed pharmacogenomics table in clinical.db with real PGx data for epilepsy AEDs.

Sources: CPIC guidelines, PharmGKB, FDA pharmacogenomics labels.
Genes: HLA-B, HLA-A, CYP2C9, CYP2C19, UGT1A4, SCN1A, ABCB1.
"""
import sqlite3, random, pathlib

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"

def seed():
    conn = sqlite3.connect(str(DB))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pharmacogenomics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        gene TEXT NOT NULL,
        variant TEXT NOT NULL,
        allele_function TEXT,
        metabolizer_status TEXT,
        clinical_significance TEXT,
        affected_drugs TEXT,
        recommendation TEXT,
        evidence_level TEXT,
        source TEXT DEFAULT 'CPIC',
        test_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    patients = [r[0] for r in c.execute("SELECT DISTINCT patient_id FROM patients").fetchall()]
    if not patients:
        patients = [f"P{i:03d}" for i in range(1, 41)]

    pgx_entries = [
        {"gene": "HLA-B", "variant": "*15:02", "allele_function": "Risk allele",
         "metabolizer_status": "Carrier", "clinical_significance": "High — SJS/TEN risk",
         "affected_drugs": "Carbamazepine, Oxcarbazepine, Phenytoin",
         "recommendation": "Avoid carbamazepine/oxcarbazepine; use alternative AED (levetiracetam, valproate)",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "HLA-B", "variant": "*15:02", "allele_function": "Non-carrier",
         "metabolizer_status": "Non-carrier", "clinical_significance": "Normal risk",
         "affected_drugs": "Carbamazepine, Oxcarbazepine",
         "recommendation": "Standard dosing; monitor for rash in first 8 weeks",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "HLA-A", "variant": "*31:01", "allele_function": "Risk allele",
         "metabolizer_status": "Carrier", "clinical_significance": "Moderate — DRESS/MPE risk",
         "affected_drugs": "Carbamazepine",
         "recommendation": "Consider alternative AED unless benefits clearly outweigh risks",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "HLA-A", "variant": "*31:01", "allele_function": "Non-carrier",
         "metabolizer_status": "Non-carrier", "clinical_significance": "Normal risk",
         "affected_drugs": "Carbamazepine",
         "recommendation": "Standard prescribing",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "CYP2C9", "variant": "*1/*1", "allele_function": "Normal function",
         "metabolizer_status": "Normal Metabolizer", "clinical_significance": "Standard metabolism",
         "affected_drugs": "Phenytoin, Valproate",
         "recommendation": "Standard dosing (300 mg/day phenytoin)",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "CYP2C9", "variant": "*1/*2", "allele_function": "Decreased function",
         "metabolizer_status": "Intermediate Metabolizer", "clinical_significance": "Reduced clearance",
         "affected_drugs": "Phenytoin, Valproate",
         "recommendation": "Reduce phenytoin dose by 25%; monitor levels closely",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "CYP2C9", "variant": "*2/*3", "allele_function": "Decreased function",
         "metabolizer_status": "Poor Metabolizer", "clinical_significance": "Significantly reduced clearance",
         "affected_drugs": "Phenytoin, Valproate",
         "recommendation": "Reduce phenytoin dose by 50%; consider alternative AED",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "CYP2C9", "variant": "*3/*3", "allele_function": "No function",
         "metabolizer_status": "Poor Metabolizer", "clinical_significance": "Markedly reduced clearance — toxicity risk",
         "affected_drugs": "Phenytoin",
         "recommendation": "Avoid phenytoin or reduce dose by 50%+; TDM mandatory",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "CYP2C19", "variant": "*1/*1", "allele_function": "Normal function",
         "metabolizer_status": "Normal Metabolizer", "clinical_significance": "Standard metabolism",
         "affected_drugs": "Clobazam, Brivaracetam, Lacosamide",
         "recommendation": "Standard dosing",
         "evidence_level": "2A", "source": "CPIC"},
        {"gene": "CYP2C19", "variant": "*1/*2", "allele_function": "Decreased function",
         "metabolizer_status": "Intermediate Metabolizer", "clinical_significance": "Reduced N-desmethylclobazam clearance",
         "affected_drugs": "Clobazam",
         "recommendation": "Start at 50% of standard clobazam dose; titrate by levels",
         "evidence_level": "2A", "source": "CPIC"},
        {"gene": "CYP2C19", "variant": "*2/*2", "allele_function": "No function",
         "metabolizer_status": "Poor Metabolizer", "clinical_significance": "N-desmethylclobazam accumulation — sedation risk",
         "affected_drugs": "Clobazam",
         "recommendation": "Start at 25% of standard dose; consider alternative (levetiracetam)",
         "evidence_level": "1A", "source": "CPIC"},
        {"gene": "CYP2C19", "variant": "*1/*17", "allele_function": "Increased function",
         "metabolizer_status": "Rapid Metabolizer", "clinical_significance": "Increased clearance — subtherapeutic risk",
         "affected_drugs": "Clobazam",
         "recommendation": "May need higher dose; TDM recommended",
         "evidence_level": "2B", "source": "PharmGKB"},
        {"gene": "UGT1A4", "variant": "*1/*1", "allele_function": "Normal function",
         "metabolizer_status": "Normal Metabolizer", "clinical_significance": "Standard glucuronidation",
         "affected_drugs": "Lamotrigine",
         "recommendation": "Standard dosing; reduce if co-administered with valproate",
         "evidence_level": "3", "source": "PharmGKB"},
        {"gene": "UGT1A4", "variant": "*3/*3", "allele_function": "Decreased function",
         "metabolizer_status": "Poor Metabolizer", "clinical_significance": "Reduced lamotrigine clearance",
         "affected_drugs": "Lamotrigine",
         "recommendation": "Lower starting dose; titrate slowly; monitor for rash",
         "evidence_level": "3", "source": "PharmGKB"},
        {"gene": "SCN1A", "variant": "IVS5-91 G>A (rs3812718)", "allele_function": "Altered splicing",
         "metabolizer_status": "AA genotype", "clinical_significance": "Higher dose needed for seizure freedom",
         "affected_drugs": "Carbamazepine, Phenytoin",
         "recommendation": "May require higher dose; consider max tolerated dose earlier",
         "evidence_level": "2B", "source": "PharmGKB"},
        {"gene": "SCN1A", "variant": "IVS5-91 G>A (rs3812718)", "allele_function": "Normal splicing",
         "metabolizer_status": "GG genotype", "clinical_significance": "Better response to sodium-channel blockers",
         "affected_drugs": "Carbamazepine, Phenytoin",
         "recommendation": "Standard dosing expected to achieve seizure freedom",
         "evidence_level": "2B", "source": "PharmGKB"},
        {"gene": "ABCB1", "variant": "C3435T (rs1045642)", "allele_function": "Reduced efflux",
         "metabolizer_status": "TT genotype", "clinical_significance": "Higher brain AED concentration",
         "affected_drugs": "Phenytoin, Carbamazepine, Lamotrigine",
         "recommendation": "Monitor for CNS side effects; may achieve seizure control at lower dose",
         "evidence_level": "3", "source": "PharmGKB"},
        {"gene": "ABCB1", "variant": "C3435T (rs1045642)", "allele_function": "Normal efflux",
         "metabolizer_status": "CC genotype", "clinical_significance": "Possible drug resistance mechanism",
         "affected_drugs": "Phenytoin, Carbamazepine, Lamotrigine",
         "recommendation": "Standard dosing; consider polytherapy if refractory",
         "evidence_level": "3", "source": "PharmGKB"},
    ]

    random.seed(42)
    months = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    inserted = 0
    for pid in patients:
        n_tests = random.randint(3, 6)
        test_genes = random.sample(pgx_entries, min(n_tests, len(pgx_entries)))
        test_date = random.choice(months) + "-" + str(random.randint(1, 28)).zfill(2)
        for entry in test_genes:
            c.execute("""INSERT INTO pharmacogenomics
                (patient_id, gene, variant, allele_function, metabolizer_status,
                 clinical_significance, affected_drugs, recommendation, evidence_level, source, test_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, entry["gene"], entry["variant"], entry["allele_function"],
                 entry["metabolizer_status"], entry["clinical_significance"],
                 entry["affected_drugs"], entry["recommendation"],
                 entry["evidence_level"], entry["source"], test_date))
            inserted += 1

    conn.commit()
    conn.close()
    print(f"Created pharmacogenomics table: {inserted} rows for {len(patients)} patients")

if __name__ == "__main__":
    seed()
