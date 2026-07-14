"""Pharmacogenomics Dashboard — PGx-guided AED prescribing analytics from clinical.db.

Provides pharmacogenomic profiling for anti-epileptic drug (AED) selection,
including HLA allele screening, CYP enzyme metabolizer status, drug-gene
interaction alerts, and CPIC-guideline adherence.

Clinically this matters because:
- HLA-B*15:02 screening before carbamazepine prevents Stevens-Johnson
  syndrome / toxic epidermal necrolysis — FDA boxed warning (2007).
- CYP2C9 poor metabolizers accumulate phenytoin → dose-dependent toxicity
  (ataxia, nystagmus, cardiac arrhythmia) — CPIC guideline (2020).
- CYP2C19 poor metabolizers accumulate N-desmethylclobazam → excessive
  sedation in Dravet syndrome — FDA label update (2018).

Sources:
- pharmacogenomics table  (~172 rows, 40 patients, 7 genes)
- medications table       (~9 rows)
- patients table          (~40 patients)
"""

import pathlib
import sqlite3
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def overview():
    con = _conn()
    c = con.cursor()

    total_tests = _safe(c, "SELECT COUNT(*) FROM pharmacogenomics")
    total_patients = _safe(c, "SELECT COUNT(DISTINCT patient_id) FROM pharmacogenomics")
    total_genes = _safe(c, "SELECT COUNT(DISTINCT gene) FROM pharmacogenomics")
    actionable = _safe(c, """SELECT COUNT(*) FROM pharmacogenomics
        WHERE clinical_significance NOT LIKE '%Normal%' AND clinical_significance NOT LIKE '%Standard%'""")
    actionable_pct = round(actionable / total_tests * 100, 1) if total_tests else 0

    # Gene distribution
    c.execute("SELECT gene, COUNT(*) as cnt FROM pharmacogenomics GROUP BY gene ORDER BY cnt DESC")
    gene_dist = [{"gene": r["gene"], "count": r["cnt"]} for r in c.fetchall()]

    # Metabolizer status distribution
    c.execute("""SELECT metabolizer_status, COUNT(*) as cnt FROM pharmacogenomics
        GROUP BY metabolizer_status ORDER BY cnt DESC""")
    metabolizer_dist = [{"status": r["metabolizer_status"], "count": r["cnt"]} for r in c.fetchall()]

    # Evidence level distribution
    c.execute("""SELECT evidence_level, COUNT(*) as cnt FROM pharmacogenomics
        GROUP BY evidence_level ORDER BY evidence_level""")
    evidence_dist = [{"level": r["evidence_level"], "count": r["cnt"]} for r in c.fetchall()]

    # High-risk alerts (HLA carriers, poor metabolizers)
    c.execute("""SELECT patient_id, gene, variant, clinical_significance, affected_drugs, recommendation
        FROM pharmacogenomics
        WHERE clinical_significance LIKE '%High%' OR clinical_significance LIKE '%Markedly%'
           OR clinical_significance LIKE '%SJS%' OR clinical_significance LIKE '%toxicity%'
        ORDER BY patient_id""")
    alerts = [dict(r) for r in c.fetchall()]

    # Drug-gene interaction summary
    c.execute("""SELECT affected_drugs, COUNT(DISTINCT patient_id) as patients,
        SUM(CASE WHEN clinical_significance NOT LIKE '%Normal%'
             AND clinical_significance NOT LIKE '%Standard%' THEN 1 ELSE 0 END) as actionable
        FROM pharmacogenomics GROUP BY affected_drugs ORDER BY actionable DESC""")
    drug_interactions = [{"drugs": r["affected_drugs"], "patients": r["patients"],
                          "actionable": r["actionable"]} for r in c.fetchall()]

    # Source distribution
    c.execute("SELECT source, COUNT(*) as cnt FROM pharmacogenomics GROUP BY source ORDER BY cnt DESC")
    source_dist = [{"source": r["source"], "count": r["cnt"]} for r in c.fetchall()]

    con.close()
    return {
        "total_tests": total_tests,
        "total_patients": total_patients,
        "total_genes": total_genes,
        "actionable_results": actionable,
        "actionable_pct": actionable_pct,
        "gene_distribution": gene_dist,
        "metabolizer_distribution": metabolizer_dist,
        "evidence_distribution": evidence_dist,
        "high_risk_alerts": alerts,
        "drug_gene_interactions": drug_interactions,
        "source_distribution": source_dist,
    }


def breakdown():
    con = _conn()
    c = con.cursor()

    # Per-patient PGx profile
    c.execute("""SELECT patient_id,
        COUNT(*) as total_tests,
        SUM(CASE WHEN clinical_significance NOT LIKE '%Normal%'
             AND clinical_significance NOT LIKE '%Standard%' THEN 1 ELSE 0 END) as actionable,
        GROUP_CONCAT(DISTINCT gene) as genes_tested,
        test_date
        FROM pharmacogenomics
        GROUP BY patient_id ORDER BY actionable DESC""")
    patient_profiles = [dict(r) for r in c.fetchall()]

    # Gene-variant matrix
    c.execute("""SELECT gene, variant, metabolizer_status, COUNT(DISTINCT patient_id) as patients,
        clinical_significance, recommendation
        FROM pharmacogenomics
        GROUP BY gene, variant, metabolizer_status
        ORDER BY gene, variant""")
    gene_variant_matrix = [dict(r) for r in c.fetchall()]

    # HLA screening summary
    c.execute("""SELECT gene, variant, metabolizer_status, COUNT(DISTINCT patient_id) as patients
        FROM pharmacogenomics WHERE gene LIKE 'HLA%'
        GROUP BY gene, variant, metabolizer_status ORDER BY gene, variant""")
    hla_screening = [dict(r) for r in c.fetchall()]

    # CYP enzyme panel
    c.execute("""SELECT gene, variant, metabolizer_status, COUNT(DISTINCT patient_id) as patients,
        affected_drugs
        FROM pharmacogenomics WHERE gene LIKE 'CYP%'
        GROUP BY gene, variant, metabolizer_status ORDER BY gene, variant""")
    cyp_panel = [dict(r) for r in c.fetchall()]

    # Medication-PGx cross-reference (which patients on AEDs have relevant PGx results)
    c.execute("""SELECT p.patient_id, p.gene, p.variant, p.metabolizer_status,
        p.affected_drugs, p.recommendation, m.fields_json
        FROM pharmacogenomics p
        INNER JOIN medications m ON p.patient_id = m.patient_id
        ORDER BY p.patient_id""")
    import json as _json
    med_pgx_cross = []
    for r in c.fetchall():
        row = dict(r)
        try:
            fields = _json.loads(row.pop("fields_json", "{}"))
            row["drug_name"] = fields.get("drug_name", "Unknown")
        except Exception:
            row["drug_name"] = "Unknown"
        med_pgx_cross.append(row)

    # Actionable by evidence level
    c.execute("""SELECT evidence_level,
        COUNT(*) as total,
        SUM(CASE WHEN clinical_significance NOT LIKE '%Normal%'
             AND clinical_significance NOT LIKE '%Standard%' THEN 1 ELSE 0 END) as actionable
        FROM pharmacogenomics GROUP BY evidence_level ORDER BY evidence_level""")
    evidence_actionable = [dict(r) for r in c.fetchall()]

    # Monthly testing trend
    c.execute("""SELECT SUBSTR(test_date, 1, 7) as month, COUNT(*) as tests,
        COUNT(DISTINCT patient_id) as patients
        FROM pharmacogenomics GROUP BY month ORDER BY month""")
    testing_trend = [dict(r) for r in c.fetchall()]

    con.close()
    return {
        "patient_profiles": patient_profiles,
        "gene_variant_matrix": gene_variant_matrix,
        "hla_screening": hla_screening,
        "cyp_panel": cyp_panel,
        "med_pgx_crossref": med_pgx_cross,
        "evidence_actionable": evidence_actionable,
        "testing_trend": testing_trend,
    }


def definitions():
    return {
        "title": "Pharmacogenomics Dashboard — Definitions",
        "terms": [
            {"term": "Pharmacogenomics (PGx)",
             "definition": "Study of how genetic variation affects drug response. In epilepsy, PGx guides AED selection to maximize efficacy and minimize adverse drug reactions."},
            {"term": "HLA-B*15:02",
             "definition": "HLA allele associated with carbamazepine-induced Stevens-Johnson syndrome (SJS) and toxic epidermal necrolysis (TEN). FDA boxed warning mandates screening before prescribing carbamazepine in at-risk populations (South/Southeast Asian ancestry)."},
            {"term": "HLA-A*31:01",
             "definition": "HLA allele associated with carbamazepine-induced drug reaction with eosinophilia and systemic symptoms (DRESS) and maculopapular exanthema (MPE) across all ethnic groups. CPIC Level 1A evidence."},
            {"term": "CYP2C9",
             "definition": "Cytochrome P450 enzyme responsible for phenytoin hydroxylation. Poor metabolizers (*2/*3, *3/*3) have 50-75% reduced clearance → dose-dependent toxicity risk."},
            {"term": "CYP2C19",
             "definition": "Cytochrome P450 enzyme that converts clobazam to active metabolite N-desmethylclobazam. Poor metabolizers (*2/*2) accumulate the active metabolite → excessive sedation, particularly in Dravet syndrome."},
            {"term": "UGT1A4",
             "definition": "UDP-glucuronosyltransferase that glucuronidates lamotrigine. Decreased-function variants reduce clearance; combined with valproate inhibition → risk of lamotrigine toxicity (rash, SJS)."},
            {"term": "SCN1A",
             "definition": "Sodium channel gene. The IVS5-91 G>A variant (rs3812718) affects alternative splicing and modifies response to sodium-channel blocking AEDs (carbamazepine, phenytoin). AA genotype associated with higher dose requirements."},
            {"term": "ABCB1 (MDR1/P-glycoprotein)",
             "definition": "Drug efflux transporter at the blood-brain barrier. C3435T (rs1045642) TT genotype reduces P-gp expression → higher brain AED concentration. CC genotype may contribute to pharmacoresistant epilepsy."},
            {"term": "CPIC",
             "definition": "Clinical Pharmacogenetics Implementation Consortium. Publishes peer-reviewed, evidence-based guidelines for gene-drug pairs. Level 1A = strong evidence + strong recommendation."},
            {"term": "Metabolizer Status",
             "definition": "Predicted enzyme activity based on diplotype: Poor (PM) = no/minimal function; Intermediate (IM) = reduced; Normal (NM/EM) = typical; Rapid/Ultrarapid (RM/UM) = increased."},
            {"term": "Actionable Result",
             "definition": "A PGx finding that changes prescribing: avoid drug, adjust dose, increase monitoring, or switch to alternative. Non-actionable = standard prescribing."},
            {"term": "Evidence Level",
             "definition": "CPIC classification: 1A (strong evidence + strong recommendation), 2A (moderate evidence), 2B (weak evidence), 3 (annotation only — informative but not yet guideline-level)."},
        ],
        "data_sources": [
            "pharmacogenomics table (clinical.db) — 172 rows, 40 patients, 7 genes",
            "medications table (clinical.db) — AED prescriptions for cross-reference",
            "CPIC guidelines (cpicpgx.org) — gene-drug pair recommendations",
            "PharmGKB (pharmgkb.org) — variant annotations and clinical annotations",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== Breakdown (keys) ===")
    bd = breakdown()
    for k, v in bd.items():
        print(f"  {k}: {len(v) if isinstance(v, list) else v}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"  {len(d['terms'])} terms, {len(d['data_sources'])} sources")
