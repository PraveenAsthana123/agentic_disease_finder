"""Pharmacogenomics Dashboard — pharmacogenomic test analytics from clinical.db.

Tracks patient pharmacogenomic profiles including gene variants, metabolizer
status, clinical significance, drug interactions, and evidence-based
recommendations for epilepsy pharmacotherapy.

Sources:
- pharmacogenomics table (patient_id, gene, variant, allele_function,
  metabolizer_status, clinical_significance, affected_drugs, recommendation,
  evidence_level, source, test_date, created_at)
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# ──────────────────────────────────────────────────────────────
#  /api/pharmacogenomics/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level pharmacogenomics metrics."""
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM pharmacogenomics")
    total_tests = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM pharmacogenomics")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT gene) FROM pharmacogenomics")
    unique_genes = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM pharmacogenomics "
        "WHERE clinical_significance LIKE '%High%'"
    )
    high_risk_count = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM pharmacogenomics WHERE evidence_level = '1A'"
    )
    actionable_results = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM pharmacogenomics "
        "WHERE metabolizer_status LIKE '%Poor%'"
    )
    poor_metabolizer_count = cur.fetchone()[0]

    kpis = {
        "total_tests": total_tests,
        "total_patients": total_patients,
        "unique_genes": unique_genes,
        "high_risk_count": high_risk_count,
        "actionable_results": actionable_results,
        "poor_metabolizer_count": poor_metabolizer_count,
    }

    # Gene distribution
    cur.execute(
        "SELECT gene, COUNT(*) AS cnt FROM pharmacogenomics "
        "GROUP BY gene ORDER BY cnt DESC"
    )
    gene_rows = cur.fetchall()
    gene_distribution = [
        {
            "gene": r[0],
            "count": r[1],
            "pct": round(r[1] / total_tests * 100, 1) if total_tests else 0.0,
        }
        for r in gene_rows
    ]

    # Metabolizer distribution
    cur.execute(
        "SELECT metabolizer_status, COUNT(*) AS cnt FROM pharmacogenomics "
        "GROUP BY metabolizer_status ORDER BY cnt DESC"
    )
    metabolizer_distribution = [
        {"metabolizer_status": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    # Evidence level distribution
    cur.execute(
        "SELECT evidence_level, COUNT(*) AS cnt FROM pharmacogenomics "
        "GROUP BY evidence_level ORDER BY cnt DESC"
    )
    evidence_level_distribution = [
        {"evidence_level": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    # Source distribution
    cur.execute(
        "SELECT source, COUNT(*) AS cnt FROM pharmacogenomics "
        "GROUP BY source ORDER BY cnt DESC"
    )
    source_distribution = [
        {"source": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    # High risk by gene
    cur.execute(
        "SELECT gene, COUNT(*) AS cnt FROM pharmacogenomics "
        "WHERE clinical_significance LIKE '%High%' "
        "GROUP BY gene ORDER BY cnt DESC"
    )
    high_risk_by_gene = [
        {"gene": r[0], "count": r[1]} for r in cur.fetchall()
    ]

    conn.close()
    return {
        "available": True,
        "kpis": kpis,
        "gene_distribution": gene_distribution,
        "metabolizer_distribution": metabolizer_distribution,
        "evidence_level_distribution": evidence_level_distribution,
        "source_distribution": source_distribution,
        "high_risk_by_gene": high_risk_by_gene,
    }


# ──────────────────────────────────────────────────────────────
#  /api/pharmacogenomics/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient detail, high-risk results, poor metabolizers, recent tests."""
    conn = _conn()
    cur = conn.cursor()

    # High-risk results
    cur.execute(
        "SELECT patient_id, gene, variant, clinical_significance, "
        "affected_drugs, recommendation, evidence_level "
        "FROM pharmacogenomics "
        "WHERE clinical_significance LIKE '%High%' "
        "ORDER BY patient_id"
    )
    high_risk_results = _dict_rows(cur)

    # Poor metabolizers
    cur.execute(
        "SELECT patient_id, gene, variant, metabolizer_status, "
        "affected_drugs, recommendation "
        "FROM pharmacogenomics "
        "WHERE metabolizer_status LIKE '%Poor%' "
        "ORDER BY patient_id"
    )
    poor_metabolizers = _dict_rows(cur)

    # Per-patient summary
    cur.execute(
        "SELECT patient_id, "
        "COUNT(*) AS total_tests, "
        "COUNT(DISTINCT gene) AS genes_tested, "
        "SUM(CASE WHEN clinical_significance LIKE '%High%' THEN 1 ELSE 0 END) AS high_risk_count, "
        "SUM(CASE WHEN evidence_level = '1A' THEN 1 ELSE 0 END) AS actionable_count, "
        "GROUP_CONCAT(DISTINCT source) AS sources "
        "FROM pharmacogenomics "
        "GROUP BY patient_id ORDER BY patient_id"
    )
    per_patient = _dict_rows(cur)

    # Recent tests (last 30)
    cur.execute(
        "SELECT id, patient_id, gene, variant, allele_function, "
        "metabolizer_status, clinical_significance, affected_drugs, "
        "recommendation, evidence_level, source, test_date, created_at "
        "FROM pharmacogenomics ORDER BY test_date DESC LIMIT 30"
    )
    recent_tests = _dict_rows(cur)

    conn.close()
    return {
        "high_risk_results": high_risk_results,
        "poor_metabolizers": poor_metabolizers,
        "per_patient": per_patient,
        "recent_tests": recent_tests,
    }


# ──────────────────────────────────────────────────────────────
#  /api/pharmacogenomics/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Gene descriptions, metabolizer categories, evidence levels, clinical notes, glossary."""
    return {
        "gene_descriptions": {
            "HLA-B": "Human Leukocyte Antigen B — immune system gene critical for drug "
                     "hypersensitivity screening. The HLA-B*15:02 allele is strongly associated "
                     "with carbamazepine-induced Stevens-Johnson syndrome (SJS) and toxic "
                     "epidermal necrolysis (TEN), particularly in Southeast Asian populations. "
                     "Pre-treatment screening is FDA-mandated before prescribing carbamazepine.",
            "CYP2C19": "Cytochrome P450 2C19 — a major hepatic drug-metabolizing enzyme involved "
                       "in the metabolism of several anti-epileptic drugs including clobazam, "
                       "diazepam, and phenytoin (minor pathway). Poor metabolizers accumulate "
                       "active metabolites (e.g., N-desmethylclobazam), increasing sedation risk; "
                       "ultra-rapid metabolizers may have subtherapeutic levels.",
            "CYP2C9": "Cytochrome P450 2C9 — the primary enzyme responsible for phenytoin "
                      "metabolism. CYP2C9*2 and *3 variants reduce enzymatic activity, leading "
                      "to elevated phenytoin levels, increased toxicity risk (ataxia, nystagmus), "
                      "and the need for lower maintenance doses. CPIC guidelines recommend "
                      "25-50% dose reduction for poor metabolizers.",
            "ABCB1": "ATP Binding Cassette Subfamily B Member 1 (P-glycoprotein/MDR1) — an "
                     "efflux transporter expressed at the blood-brain barrier that actively "
                     "pumps out several AEDs (phenytoin, carbamazepine, lamotrigine). The "
                     "C3435T polymorphism (rs1045642) may alter drug penetration into the "
                     "brain and contribute to pharmacoresistant epilepsy.",
            "HLA-A": "Human Leukocyte Antigen A — immune gene associated with drug "
                     "hypersensitivity reactions. The HLA-A*31:01 allele is associated with "
                     "carbamazepine-induced hypersensitivity syndrome (including DRESS, SJS, "
                     "and maculopapular exanthema) across multiple ethnic groups. Screening "
                     "is recommended in some guidelines before carbamazepine initiation.",
            "UGT1A4": "UDP-Glucuronosyltransferase 1A4 — a Phase II conjugation enzyme that "
                      "glucuronidates lamotrigine and is the primary clearance pathway for "
                      "this drug. Genetic variants (e.g., L48V) can alter lamotrigine "
                      "metabolism, affecting steady-state concentrations and requiring dose "
                      "adjustments to maintain therapeutic levels.",
            "SCN1A": "Sodium Voltage-Gated Channel Alpha Subunit 1 — encodes the Nav1.1 "
                     "sodium channel, the direct pharmacological target of several AEDs "
                     "(carbamazepine, phenytoin, lamotrigine). SCN1A variants (e.g., IVS5-91 "
                     "G>A) can influence drug responsiveness and are associated with Dravet "
                     "syndrome, where sodium channel blockers may paradoxically worsen seizures.",
        },
        "metabolizer_categories": {
            "Poor metabolizer": "Significantly reduced or absent enzyme activity. Leads to drug "
                                "accumulation, higher plasma levels, and increased risk of "
                                "adverse effects. Requires substantial dose reduction or "
                                "alternative drug selection.",
            "Intermediate metabolizer": "Reduced enzyme activity compared to normal. May require "
                                        "modest dose adjustments. Monitor for dose-dependent "
                                        "side effects.",
            "Normal metabolizer": "Typical enzyme activity (formerly called 'extensive "
                                  "metabolizer'). Standard dosing guidelines apply.",
            "Rapid metabolizer": "Increased enzyme activity leading to faster drug clearance. "
                                 "May require higher doses to achieve therapeutic levels.",
            "Ultra-rapid metabolizer": "Markedly increased enzyme activity, often due to gene "
                                       "duplication. Risk of subtherapeutic drug levels at "
                                       "standard doses. May need significantly higher doses or "
                                       "alternative agents.",
            "Carrier": "Carries one or more risk alleles for a pharmacogenomic marker (e.g., "
                       "HLA-B*15:02). Indicates susceptibility to a specific adverse drug "
                       "reaction rather than altered metabolism.",
            "Non-carrier": "Does not carry the risk allele. Standard prescribing applies for "
                           "the associated drug-gene interaction.",
            "CC genotype": "Homozygous wild-type for the ABCB1 C3435T polymorphism. Associated "
                           "with normal P-glycoprotein expression and standard drug efflux "
                           "activity at the blood-brain barrier.",
            "CT genotype": "Heterozygous for the ABCB1 C3435T polymorphism. Intermediate "
                           "P-glycoprotein expression. May have mildly altered AED penetration "
                           "into the CNS.",
            "TT genotype": "Homozygous variant for the ABCB1 C3435T polymorphism. Reduced "
                           "P-glycoprotein expression, potentially allowing greater AED "
                           "penetration into the brain but also altered drug disposition.",
        },
        "evidence_levels": {
            "1A": "Strong evidence — the gene-drug interaction has been validated in well-powered "
                  "studies with replication, and clinical guidelines (e.g., CPIC, DPWG) provide "
                  "prescribing recommendations. Action is required.",
            "1B": "Strong evidence — clinical evidence supports the gene-drug association, but "
                  "formal prescribing guidelines may be limited or emerging. Action is "
                  "recommended.",
            "2A": "Moderate evidence — the gene-drug interaction is supported by moderate clinical "
                  "evidence, and the gene is a known pharmacogene for the drug class. Consider "
                  "action based on clinical context.",
            "3": "Low or emerging evidence — the association is based on limited studies, small "
                 "sample sizes, or mechanistic plausibility. No formal prescribing guideline "
                 "changes recommended; monitoring and further study advised.",
        },
        "clinical_notes": [
            "HLA-B*15:02 screening is FDA-mandated before carbamazepine initiation in patients "
            "with Southeast Asian ancestry. Consider screening in all populations per CPIC 2024.",
            "CYP2C9 poor metabolizers require 25-50% lower phenytoin maintenance doses to avoid "
            "toxicity (ataxia, nystagmus, cardiac arrhythmia).",
            "CYP2C19 poor metabolizers on clobazam accumulate N-desmethylclobazam; reduce dose "
            "by 50% and monitor for excessive sedation.",
            "ABCB1 genotyping is investigational for predicting drug resistance but may inform "
            "polytherapy decisions in refractory epilepsy.",
            "SCN1A variants should prompt genetic counselling; sodium channel blockers "
            "(carbamazepine, phenytoin, lamotrigine) may worsen seizures in Dravet syndrome.",
            "Pharmacogenomic results are lifelong and should be documented in the patient's "
            "permanent medical record and shared across care providers.",
            "Multi-gene panel testing is more cost-effective than sequential single-gene tests "
            "and provides a comprehensive pharmacogenomic profile for AED selection.",
            "Evidence levels follow CPIC classification: 1A/1B = strong, 2A = moderate, "
            "3 = emerging. Only 1A/1B results warrant immediate prescribing changes.",
        ],
        "glossary": {
            "Pharmacogenomics": "The study of how genetic variation affects individual responses "
                                "to drugs, including efficacy, dosing, and adverse reactions.",
            "CPIC": "Clinical Pharmacogenetics Implementation Consortium — a body that develops "
                    "evidence-based, peer-reviewed clinical practice guidelines for "
                    "pharmacogenetics.",
            "PharmGKB": "Pharmacogenomics Knowledge Base — a comprehensive resource that curates "
                        "pharmacogenomic relationships including gene-drug associations, variant "
                        "annotations, and clinical guidelines.",
            "Metabolizer Status": "A phenotype classification (poor, intermediate, normal, rapid, "
                                  "ultra-rapid) describing an individual's capacity to metabolize "
                                  "a drug via a specific enzyme.",
            "Allele Function": "The functional impact of a specific genetic variant on gene "
                               "product activity (e.g., no function, decreased function, "
                               "normal function, increased function).",
            "SJS/TEN": "Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis — severe, "
                       "life-threatening cutaneous adverse drug reactions characterized by "
                       "widespread skin detachment. HLA-B*15:02 carriers have greatly elevated "
                       "risk with carbamazepine.",
            "P-glycoprotein": "An efflux transporter (encoded by ABCB1) at the blood-brain "
                              "barrier that pumps drugs out of the CNS. Variants may affect "
                              "AED brain penetration and treatment response.",
            "CYP450": "Cytochrome P450 — a superfamily of hepatic enzymes responsible for "
                      "Phase I oxidative metabolism of most drugs, including many AEDs.",
            "DRESS": "Drug Reaction with Eosinophilia and Systemic Symptoms — a severe "
                     "hypersensitivity reaction associated with HLA variants and certain "
                     "AEDs (carbamazepine, phenytoin, lamotrigine).",
            "Therapeutic Drug Monitoring": "Measurement of drug plasma concentrations to guide "
                                           "dosing. Particularly important in epilepsy where "
                                           "pharmacogenomic variants alter expected drug levels.",
        },
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2))
    print("\n=== Breakdown (summary) ===")
    b = breakdown()
    print(f"Patients: {len(b['per_patient'])}, High-risk results: {len(b['high_risk_results'])}")
    print(f"Poor metabolizers: {len(b['poor_metabolizers'])}, Recent tests: {len(b['recent_tests'])}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"Genes described: {len(d['gene_descriptions'])}, "
          f"Metabolizer categories: {len(d['metabolizer_categories'])}")
    print(f"Evidence levels: {len(d['evidence_levels'])}, "
          f"Clinical notes: {len(d['clinical_notes'])}, Glossary: {len(d['glossary'])}")
