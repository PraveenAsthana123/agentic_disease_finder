"""Genetic Epilepsy Syndromes Dashboard — backend data module.

Real data sources:
  - clinical.db::seizure_metadata  (71 rows; 17 genetic-etiology cases)
  - clinical.db::pharmacogenomics  (172 rows; SCN1A gene × drug-interaction)
  - clinical.db::patient_demographics (for age/sex context)

Covers: SCN1A (Dravet-spectrum / JME), KCNQ2 (neonatal/childhood), familial.
Three functions: overview(), breakdown(), definitions().
"""

import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def _genetic_rows():
    """Return all genetic-etiology seizure_metadata rows."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata "
        "WHERE fields_json LIKE '%Genetic%'"
    )
    rows = []
    for r in cur.fetchall():
        f = json.loads(r["fields_json"])
        f["patient_id"] = r["patient_id"]
        rows.append(f)
    conn.close()
    return rows


def overview():
    rows = _genetic_rows()
    n = len(rows)

    # Gene distribution
    gene_counts = {}
    for r in rows:
        raw = r.get("etiology", "Unknown")
        # "Genetic — SCN1A" → "SCN1A"; "Genetic — familial" → "Familial"
        gene_raw = raw.replace("Genetic — ", "").replace("Genetic", "Unknown")
        # Preserve uppercase gene symbols; only title-case plain words
        if gene_raw.upper() == gene_raw or gene_raw in ("SCN1A", "KCNQ2"):
            gene = gene_raw
        else:
            gene = gene_raw.title()
        gene_counts[gene] = gene_counts.get(gene, 0) + 1

    # Drug responsiveness
    resp_counts = {}
    for r in rows:
        resp = r.get("drug_responsiveness", "Unknown")
        # Simplify
        if "Drug-resistant" in resp:
            key = "Drug-Resistant"
        elif "Drug-responsive" in resp:
            key = "Drug-Responsive"
        elif "Partial" in resp:
            key = "Partial Response"
        else:
            key = "Newly Diagnosed"
        resp_counts[key] = resp_counts.get(key, 0) + 1

    drug_resistant = resp_counts.get("Drug-Resistant", 0)
    drug_responsive = resp_counts.get("Drug-Responsive", 0)

    # Syndrome distribution
    syndrome_counts = {}
    for r in rows:
        s = r.get("syndrome", "Unclassified")
        syndrome_counts[s] = syndrome_counts.get(s, 0) + 1

    # Age at onset stats
    ages = [r.get("age_at_onset") for r in rows if r.get("age_at_onset") is not None]
    pediatric_onset = sum(1 for a in ages if a <= 18)
    avg_age_onset = round(sum(ages) / len(ages), 1) if ages else 0

    # SCN1A pharmacogenomics cross-link
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) as n FROM pharmacogenomics WHERE gene = 'SCN1A'"
    )
    scn1a_pgx = cur.fetchone()["n"]
    conn.close()

    return {
        "available": True,
        "kpis": {
            "total_genetic_cases": n,
            "distinct_genes": len(gene_counts),
            "drug_resistant_count": drug_resistant,
            "drug_responsive_count": drug_responsive,
            "pediatric_onset_count": pediatric_onset,
            "avg_age_at_onset": avg_age_onset,
            "scn1a_pgx_records": scn1a_pgx,
        },
        "gene_distribution": [
            {"gene": k, "count": v, "pct": round(v / n * 100, 1)}
            for k, v in sorted(gene_counts.items(), key=lambda x: -x[1])
        ],
        "drug_response_distribution": [
            {"response": k, "count": v}
            for k, v in sorted(resp_counts.items(), key=lambda x: -x[1])
        ],
        "syndrome_distribution": [
            {"syndrome": k, "count": v}
            for k, v in sorted(syndrome_counts.items(), key=lambda x: -x[1])
        ],
    }


def breakdown():
    rows = _genetic_rows()

    # Per-patient table
    per_patient = []
    for r in rows:
        raw_etio = r.get("etiology", "")
        gene_raw2 = raw_etio.replace("Genetic — ", "").replace("Genetic", "Unknown")
        gene = gene_raw2 if gene_raw2.upper() == gene_raw2 else gene_raw2.title()
        resp = r.get("drug_responsiveness", "")
        if "Drug-resistant" in resp:
            resp_label = "Drug-Resistant"
            resp_class = "danger"
        elif "Drug-responsive" in resp:
            resp_label = "Drug-Responsive"
            resp_class = "success"
        elif "Partial" in resp:
            resp_label = "Partial"
            resp_class = "warning"
        else:
            resp_label = "Pending"
            resp_class = "secondary"

        per_patient.append({
            "patient_id": r["patient_id"],
            "gene": gene,
            "syndrome": r.get("syndrome", "Unclassified"),
            "onset_zone": r.get("onset_zone", "—"),
            "lateralization": r.get("lateralization", "—"),
            "age_at_onset": r.get("age_at_onset"),
            "drug_response": resp_label,
            "drug_response_class": resp_class,
            "seizure_frequency": r.get("current_seizure_frequency", "—"),
            "surgery_candidacy": r.get("surgery_candidacy", "—"),
        })

    # Gene × drug-response matrix
    matrix = {}
    for p in per_patient:
        gene = p["gene"]
        resp = p["drug_response"]
        if gene not in matrix:
            matrix[gene] = {}
        matrix[gene][resp] = matrix[gene].get(resp, 0) + 1

    matrix_rows = [
        {"gene": gene, **counts}
        for gene, counts in sorted(matrix.items())
    ]

    # SCN1A pharmacogenomics records
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT patient_id, variant, allele_function, metabolizer_status, "
        "clinical_significance, affected_drugs "
        "FROM pharmacogenomics WHERE gene = 'SCN1A' LIMIT 20"
    )
    scn1a_pgx = [
        {
            "patient_id": r["patient_id"],
            "variant": r["variant"],
            "allele_function": r["allele_function"],
            "metabolizer_status": r["metabolizer_status"],
            "clinical_significance": r["clinical_significance"],
            "affected_drugs": r["affected_drugs"],
        }
        for r in cur.fetchall()
    ]
    conn.close()

    # Onset-zone breakdown for genetic cases
    onset_counts = {}
    for r in rows:
        oz = r.get("onset_zone", "Unknown")
        onset_counts[oz] = onset_counts.get(oz, 0) + 1

    onset_distribution = [
        {"onset_zone": k, "count": v}
        for k, v in sorted(onset_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "per_patient": sorted(per_patient, key=lambda x: x["patient_id"]),
        "gene_response_matrix": matrix_rows,
        "scn1a_pgx_records": scn1a_pgx,
        "onset_distribution": onset_distribution,
    }


def definitions():
    return {
        "dashboard": "Genetic Epilepsy Syndromes Dashboard",
        "description": (
            "Profiles 17 patients with genetically-confirmed or familially-linked epilepsy "
            "etiologies. Covers SCN1A channelopathies (Dravet spectrum / JME), "
            "KCNQ2-related epilepsies (neonatal/childhood onset), and familial epilepsy "
            "syndromes. Cross-references pharmacogenomics SCN1A data to highlight "
            "drug-gene interaction risks."
        ),
        "genes": [
            {
                "gene": "SCN1A",
                "full_name": "Sodium Voltage-Gated Channel Alpha Subunit 1",
                "associated_syndromes": [
                    "Dravet Syndrome",
                    "Genetic Epilepsy with Febrile Seizures Plus (GEFS+)",
                    "Juvenile Myoclonic Epilepsy (JME)",
                    "Lennox-Gastaut Syndrome",
                ],
                "inheritance": "Autosomal dominant (de novo or inherited)",
                "clinical_note": (
                    "Loss-of-function SCN1A variants impair inhibitory interneuron firing. "
                    "Sodium-channel blockers (phenytoin, carbamazepine, lamotrigine) may "
                    "WORSEN seizures in Dravet syndrome."
                ),
            },
            {
                "gene": "KCNQ2",
                "full_name": "Potassium Voltage-Gated Channel Subfamily Q Member 2",
                "associated_syndromes": [
                    "Benign Familial Neonatal Epilepsy (BFNE)",
                    "KCNQ2 Encephalopathy",
                    "Childhood Absence Epilepsy",
                ],
                "inheritance": "Autosomal dominant",
                "clinical_note": (
                    "KCNQ2 variants cause neonatal seizures. Benign forms remit by 12 months; "
                    "encephalopathic forms require aggressive treatment. Potassium channel "
                    "openers (ezogabine/retigabine) historically targeted this pathway."
                ),
            },
            {
                "gene": "Familial",
                "full_name": "Familial Epilepsy (multi-gene / undetermined variant)",
                "associated_syndromes": [
                    "Childhood Absence Epilepsy (CAE)",
                    "Juvenile Myoclonic Epilepsy (JME)",
                    "Juvenile Absence Epilepsy (JAE)",
                    "GTCS Alone",
                    "Focal Epilepsy",
                ],
                "inheritance": "Complex/polygenic or unidentified monogenic",
                "clinical_note": (
                    "Positive family history without confirmed single-gene variant. "
                    "Whole-exome sequencing (WES) recommended to identify actionable variants."
                ),
            },
        ],
        "drug_response_tiers": [
            {
                "tier": "Drug-Responsive",
                "definition": "Seizure-free or >75% reduction on ≤2 AEDs",
                "typical_management": "Maintain current AED regimen; annual review",
            },
            {
                "tier": "Partial Response",
                "definition": "≥50% seizure reduction but not seizure-free",
                "typical_management": "Optimise AED dose; consider add-on therapy",
            },
            {
                "tier": "Drug-Resistant",
                "definition": "Failed ≥2 appropriate AEDs at adequate dose/duration (ILAE 2010)",
                "typical_management": "Pre-surgical evaluation; dietary therapies; neuromodulation",
            },
            {
                "tier": "Newly Diagnosed",
                "definition": "Treatment initiated <12 months ago; response not yet classifiable",
                "typical_management": "First-line AED; monthly follow-up; genetic counselling",
            },
        ],
        "syndromes": [
            {"name": "Childhood Absence Epilepsy (CAE)", "onset": "4–10 years", "prognosis": "Remission in ~70% by adolescence"},
            {"name": "Juvenile Myoclonic Epilepsy (JME)", "onset": "12–18 years", "prognosis": "Lifelong AED requirement in ~90%"},
            {"name": "Juvenile Absence Epilepsy (JAE)", "onset": "10–17 years", "prognosis": "Remission ~50%; GTCS risk persists"},
            {"name": "Lennox-Gastaut Syndrome (LGS)", "onset": "1–8 years", "prognosis": "Severe; <10% seizure-free long-term"},
            {"name": "GTCS Alone", "onset": "Any", "prognosis": "Variable; generally good with valproate"},
        ],
        "references": [
            "Scheffer IE et al. ILAE classification of the epilepsies. Epilepsia 2017;58:512–521.",
            "Dravet C. Dravet syndrome history. Dev Med Child Neurol 2011;53:1–6.",
            "Weckhuysen S et al. KCNQ2 encephalopathy. Ann Neurol 2012;71:15–25.",
            "Brunklaus A et al. SCN1A mutations. Brain 2012;135:2304–2317.",
            "Kearney JA. Genetic epilepsies. Neurol Clin 2021;39:655–668.",
        ],
        "metrics": [
            {"metric": "Total Genetic Cases", "definition": "Patients with confirmed genetic or familial etiology in seizure_metadata"},
            {"metric": "Drug-Resistant Count", "definition": "Patients meeting ILAE 2010 drug-resistance definition (failed ≥2 AEDs)"},
            {"metric": "Pediatric Onset", "definition": "Cases with age_at_onset ≤18 years"},
            {"metric": "SCN1A PGx Records", "definition": "SCN1A pharmacogenomics entries cross-referencing drug-gene interaction risk"},
        ],
    }
