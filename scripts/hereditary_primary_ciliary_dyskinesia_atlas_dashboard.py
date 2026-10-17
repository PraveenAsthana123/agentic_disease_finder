#!/usr/bin/env python3
"""Hereditary-Primary-Ciliary-Dyskinesia-Atlas -- Complete 8-Gene Reference
Inherited defects of motile cilia causing Primary Ciliary Dyskinesia (PCD):
DNAH5 (most common ~28-38%; absent ODA on TEM PATHOGNOMONIC), DNAI1 (CILD1; ~9-16%; absent ODA),
CCDC39 (CILD14; IDA+MT transposition defect), CCDC40 (CILD15; IDA+MT defect),
DNAI2 (CILD9; ODA defect), DNAH11 (CILD7; NORMAL TEM PATHOGNOMONIC — waveform analysis mandatory),
RSPH4A (CILD11; central pair defect; transposition), CCNO (RGMC; reduced cilia number).

DNAH5   (Dynein axonemal heavy chain 5; 4624aa; 5p15.2; AR LOF;
          Most common PCD gene ~28-38%; outer dynein arm (ODA) defect;
          Absent ODA on TEM PATHOGNOMONIC; situs inversus ~45-50%;
          seed SEED_BASE+0).
DNAI1   (Dynein axonemal intermediate chain 1; 699aa; 9p21.2; AR LOF;
          CILD1; ~9-16% PCD; ODA defect; absent ODA on TEM;
          seed SEED_BASE+1).
CCDC39  (Coiled-coil domain containing 39; 941aa; 3q26.33; AR LOF;
          CILD14; inner dynein arm (IDA) + microtubule transposition defect;
          most severe bronchiectasis; early HSCT may be considered;
          seed SEED_BASE+2).
CCDC40  (Coiled-coil domain containing 40; 1092aa; 17q25.3; AR LOF;
          CILD15; IDA + MT transposition defect; phenotype overlaps CCDC39;
          seed SEED_BASE+3).
DNAI2   (Dynein axonemal intermediate chain 2; 612aa; 17p13.1; AR LOF;
          CILD9; ODA defect; similar to DNAI1; absent/short ODA on TEM;
          seed SEED_BASE+4).
DNAH11  (Dynein axonemal heavy chain 11; 4624aa; 7p21.3; AR LOF;
          CILD7; NORMAL TEM PATHOGNOMONIC — normal ultrastructure does NOT exclude PCD;
          hyperkinetic stiff cilia; waveform video analysis mandatory; nNO <77 PATHOGNOMONIC;
          seed SEED_BASE+5).
RSPH4A  (Radial spoke head component 4A; 703aa; 6q22.1; AR LOF;
          CILD11; central pair apparatus defect; transposition pattern;
          situs inversus RARE (<5%); predominantly sinopulmonary disease;
          seed SEED_BASE+6).
CCNO    (Cyclin O; 269aa; 5q11.2; AR LOF;
          RGMC — Reduced Generation of Multiple Cilia; reduced cilia number (not absent ODA/IDA);
          severe bronchiectasis despite TEM appearing near-normal;
          seed SEED_BASE+7).
"""

import random
import math

SEED_BASE = 3726

GENES = ["DNAH5", "DNAI1", "CCDC39", "CCDC40", "DNAI2", "DNAH11", "RSPH4A", "CCNO"]

GENE_META = {
    "DNAH5": {
        "aa": 4624, "kda": 512, "locus": "5p15.2", "inh": "AR LOF",
        "syndrome": "CILD3 / Most-Common-PCD-28-38pct / Absent-ODA-TEM-PATHOGNOMONIC / Situs-Inversus-45-50pct / Kartagener-If-Situs+Bronchiectasis+Sinusitis",
        "defect": "Outer Dynein Arm (ODA) absent",
        "situs_inversus_pct": 0.48,
        "nno_mean": 18.2,
        "bronchiectasis_pct": 0.72,
        "male_infertility_pct": 0.94,
        "fem_subfertility_pct": 0.35,
        "fev1_pct": 71.4,
        "pathognomonic": "Absent ODA on TEM; Kartagener triad (situs inversus + bronchiectasis + sinusitis)",
        "key_rule": "nNO <77 nL/min PATHOGNOMONIC; TEM absent ODA confirms; 28-38% of all PCD",
        "mean_age_dx": 8.4,
        "pct_female": 0.51,
        "prevalence_pct_pcd": 33,
    },
    "DNAI1": {
        "aa": 699, "kda": 79, "locus": "9p21.2", "inh": "AR LOF",
        "syndrome": "CILD1 / 9-16pct-PCD / Absent-ODA-TEM / Situs-Inversus-40-45pct / Neonatal-Respiratory-Distress",
        "defect": "Outer Dynein Arm (ODA) absent",
        "situs_inversus_pct": 0.43,
        "nno_mean": 16.8,
        "bronchiectasis_pct": 0.68,
        "male_infertility_pct": 0.92,
        "fem_subfertility_pct": 0.32,
        "fev1_pct": 73.2,
        "pathognomonic": "Absent ODA on TEM; nNO <77 nL/min",
        "key_rule": "IVS1+2_1G>A and c.1303G>A founder mutations; absent ODA indistinguishable from DNAH5",
        "mean_age_dx": 9.1,
        "pct_female": 0.49,
        "prevalence_pct_pcd": 12,
    },
    "CCDC39": {
        "aa": 941, "kda": 107, "locus": "3q26.33", "inh": "AR LOF",
        "syndrome": "CILD14 / IDA+MT-Transposition-Defect / Most-Severe-Bronchiectasis / Early-Airway-Disease",
        "defect": "Inner Dynein Arm (IDA) + microtubule transposition",
        "situs_inversus_pct": 0.40,
        "nno_mean": 19.1,
        "bronchiectasis_pct": 0.88,
        "male_infertility_pct": 0.91,
        "fem_subfertility_pct": 0.42,
        "fev1_pct": 62.3,
        "pathognomonic": "IDA loss + MT transposition on TEM; most severe lung disease",
        "key_rule": "Severe phenotype — early intensive airway clearance; lowest FEV1 of all PCD subtypes",
        "mean_age_dx": 6.2,
        "pct_female": 0.48,
        "prevalence_pct_pcd": 10,
    },
    "CCDC40": {
        "aa": 1092, "kda": 122, "locus": "17q25.3", "inh": "AR LOF",
        "syndrome": "CILD15 / IDA+MT-Defect / Overlaps-CCDC39 / Situs-Inversus-35-40pct",
        "defect": "Inner Dynein Arm (IDA) + microtubule transposition",
        "situs_inversus_pct": 0.38,
        "nno_mean": 20.3,
        "bronchiectasis_pct": 0.82,
        "male_infertility_pct": 0.90,
        "fem_subfertility_pct": 0.38,
        "fev1_pct": 65.8,
        "pathognomonic": "IDA loss + MT transposition on TEM; phenotype similar to CCDC39",
        "key_rule": "Often misclassified as CCDC39 — sequencing required to distinguish; same treatment approach",
        "mean_age_dx": 6.8,
        "pct_female": 0.50,
        "prevalence_pct_pcd": 8,
    },
    "DNAI2": {
        "aa": 612, "kda": 70, "locus": "17p13.1", "inh": "AR LOF",
        "syndrome": "CILD9 / ODA-Defect / Absent-Short-ODA-TEM / Situs-Inversus-40pct",
        "defect": "Outer Dynein Arm (ODA) absent/short",
        "situs_inversus_pct": 0.41,
        "nno_mean": 17.5,
        "bronchiectasis_pct": 0.65,
        "male_infertility_pct": 0.91,
        "fem_subfertility_pct": 0.31,
        "fev1_pct": 74.8,
        "pathognomonic": "Absent/short ODA on TEM; clinically indistinguishable from DNAI1",
        "key_rule": "Genotyping required; same management as DNAI1/DNAH5 ODA defects",
        "mean_age_dx": 9.8,
        "pct_female": 0.52,
        "prevalence_pct_pcd": 7,
    },
    "DNAH11": {
        "aa": 4624, "kda": 520, "locus": "7p21.3", "inh": "AR LOF",
        "syndrome": "CILD7 / NORMAL-TEM-PATHOGNOMONIC / Hyperkinetic-Stiff-Cilia / Waveform-Analysis-Mandatory / nNO-Very-Low",
        "defect": "Normal ultrastructure — motility defect only (hyperkinetic stiff)",
        "situs_inversus_pct": 0.45,
        "nno_mean": 14.2,
        "bronchiectasis_pct": 0.62,
        "male_infertility_pct": 0.89,
        "fem_subfertility_pct": 0.28,
        "fev1_pct": 76.3,
        "pathognomonic": "NORMAL TEM PATHOGNOMONIC for DNAH11-PCD — normal structure does NOT exclude PCD",
        "key_rule": "CRITICAL: TEM-based diagnosis MISSES DNAH11-PCD. Waveform video analysis mandatory. nNO <77 PATHOGNOMONIC",
        "mean_age_dx": 11.2,
        "pct_female": 0.50,
        "prevalence_pct_pcd": 9,
    },
    "RSPH4A": {
        "aa": 703, "kda": 78, "locus": "6q22.1", "inh": "AR LOF",
        "syndrome": "CILD11 / Central-Pair-Apparatus-Defect / Transposition-Pattern-TEM / Situs-Inversus-RARE-lt5pct",
        "defect": "Central pair apparatus absent/defective; transposition",
        "situs_inversus_pct": 0.04,
        "nno_mean": 22.8,
        "bronchiectasis_pct": 0.55,
        "male_infertility_pct": 0.85,
        "fem_subfertility_pct": 0.25,
        "fev1_pct": 79.1,
        "pathognomonic": "Situs inversus RARE (<5%) — central pair defect; transposition pattern on TEM",
        "key_rule": "Situs inversus ABSENT in most — central pair defect does not cause randomisation of laterality; milder lung disease",
        "mean_age_dx": 13.4,
        "pct_female": 0.53,
        "prevalence_pct_pcd": 6,
    },
    "CCNO": {
        "aa": 269, "kda": 31, "locus": "5q11.2", "inh": "AR LOF",
        "syndrome": "RGMC / Reduced-Generation-Multiple-Motile-Cilia / Reduced-Cilia-Number / TEM-Near-Normal / Severe-Bronchiectasis",
        "defect": "Reduced cilia number (not ODA/IDA loss)",
        "situs_inversus_pct": 0.05,
        "nno_mean": 31.4,
        "bronchiectasis_pct": 0.80,
        "male_infertility_pct": 0.72,
        "fem_subfertility_pct": 0.65,
        "fev1_pct": 64.2,
        "pathognomonic": "Near-normal TEM despite severe disease; quantitative cilia count reduced",
        "key_rule": "MCIDAS/CCNO digenic — female subfertility HIGH (65%); male fertility less affected vs ODA defects; nNO borderline",
        "mean_age_dx": 7.1,
        "pct_female": 0.58,
        "prevalence_pct_pcd": 5,
    },
}

N_PER_GENE = 40


def _gen_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    m = GENE_META[gene]
    n = N_PER_GENE

    ages = [max(0.5, rng.gauss(m["mean_age_dx"], 3.8)) for _ in range(n)]
    female = [rng.random() < m["pct_female"] for _ in range(n)]
    nno = [max(1.0, rng.gauss(m["nno_mean"], 5.2)) for _ in range(n)]
    fev1 = [max(20.0, rng.gauss(m["fev1_pct"], 12.0)) for _ in range(n)]
    situs = [rng.random() < m["situs_inversus_pct"] for _ in range(n)]
    bronch = [rng.random() < m["bronchiectasis_pct"] for _ in range(n)]
    male_inf = [rng.random() < m["male_infertility_pct"] for _ in range(1, 25)]
    fem_sub = [rng.random() < m["fem_subfertility_pct"] for _ in range(1, 22)]

    return {
        "gene": gene,
        "n": n,
        "mean_age_dx": round(sum(ages) / n, 1),
        "pct_female": round(sum(female) / n, 3),
        "mean_nno_nl_min": round(sum(nno) / n, 1),
        "mean_fev1_pct": round(sum(fev1) / n, 1),
        "situs_inversus_pct": round(sum(situs) / n * 100, 1),
        "bronchiectasis_pct": round(sum(bronch) / n * 100, 1),
        "male_infertility_pct": round(sum(male_inf) / len(male_inf) * 100, 1),
        "fem_subfertility_pct": round(sum(fem_sub) / len(fem_sub) * 100, 1),
        "nno_below_77_pct": round(sum(1 for v in nno if v < 77) / n * 100, 1),
    }


def _build_atlas_genes() -> list:
    result = []
    for gene, m in GENE_META.items():
        result.append({
            "gene": gene,
            "locus": m["locus"],
            "aa": m["aa"],
            "kda": m["kda"],
            "inheritance": m["inh"],
            "syndrome": m["syndrome"],
            "defect": m["defect"],
            "pathognomonic": m["pathognomonic"],
            "key_rule": m["key_rule"],
            "situs_inversus_pct": round(m["situs_inversus_pct"] * 100, 0),
            "nno_mean": m["nno_mean"],
            "prevalence_pct_pcd": m["prevalence_pct_pcd"],
        })
    return result


def generate_overview() -> dict:
    cohorts = {g: _gen_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)}
    total = sum(c["n"] for c in cohorts.values())
    situs_all = sum(
        cohorts[g]["situs_inversus_pct"] * cohorts[g]["n"] / 100
        for g in GENES
    )
    bronch_all = sum(
        cohorts[g]["bronchiectasis_pct"] * cohorts[g]["n"] / 100
        for g in GENES
    )
    fev1_vals = [cohorts[g]["mean_fev1_pct"] for g in GENES]
    nno_vals = [cohorts[g]["mean_nno_nl_min"] for g in GENES]

    gene_summaries = {}
    for gene in GENES:
        m = GENE_META[gene]
        gene_summaries[gene] = {
            "locus": m["locus"],
            "inheritance": m["inh"],
            "syndrome": m["syndrome"].split("/")[0].strip(),
            "defect": m["defect"],
            "pathognomonic": m["pathognomonic"].split(";")[0],
            "key_rule": m["key_rule"],
            "prevalence_pct_pcd": m["prevalence_pct_pcd"],
        }

    return {
        "atlas": "Hereditary-Primary-Ciliary-Dyskinesia-Atlas",
        "genes": GENES,
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "key_stats": {
            "total_situs_inversus_patients": round(situs_all),
            "kartagener_eligible_pct": round(situs_all / total * 100, 1),
            "total_bronchiectasis_patients": round(bronch_all),
            "bronchiectasis_pct_overall": round(bronch_all / total * 100, 1),
            "mean_fev1_pct_overall": round(sum(fev1_vals) / len(fev1_vals), 1),
            "mean_nno_nl_min_overall": round(sum(nno_vals) / len(nno_vals), 1),
            "nno_diagnostic_cutoff_nl_min": 77,
            "kartagener_triad_rule": "Situs inversus + bronchiectasis + chronic sinusitis = Kartagener = PATHOGNOMONIC for PCD",
            "dnah11_tem_rule": "DNAH11-PCD: normal TEM PATHOGNOMONIC — TEM-only diagnosis MISSES this subtype",
            "nno_rule": "nNO <77 nL/min PATHOGNOMONIC for PCD (except RSPH4A/CCNO — borderline possible)",
        },
        "gene_summaries": gene_summaries,
    }


def generate_breakdown() -> dict:
    cohorts = {g: _gen_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)}
    atlas_genes = _build_atlas_genes()

    gene_data = {}
    for gene in GENES:
        c = cohorts[gene]
        m = GENE_META[gene]
        gene_data[gene] = {
            "syndrome": m["syndrome"].split("/")[0].strip(),
            "n": c["n"],
            "mean_age_dx": c["mean_age_dx"],
            "pct_female": c["pct_female"],
            "mean_nno_nl_min": c["mean_nno_nl_min"],
            "mean_fev1_pct": c["mean_fev1_pct"],
            "situs_inversus_pct": c["situs_inversus_pct"],
            "bronchiectasis_pct": c["bronchiectasis_pct"],
            "male_infertility_pct": c["male_infertility_pct"],
            "fem_subfertility_pct": c["fem_subfertility_pct"],
            "nno_below_77_pct": c["nno_below_77_pct"],
            "tem_defect": m["defect"],
        }

    return {
        "atlas": "Hereditary-Primary-Ciliary-Dyskinesia-Atlas",
        "gene_data": gene_data,
        "atlas_genes": atlas_genes,
        "cohort_summary": {
            "total_patients": N_PER_GENE * len(GENES),
            "seeds": f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
            "genes_covered": len(GENES),
            "oda_defect_genes": ["DNAH5", "DNAI1", "DNAI2"],
            "ida_mt_defect_genes": ["CCDC39", "CCDC40"],
            "normal_tem_genes": ["DNAH11"],
            "central_pair_genes": ["RSPH4A"],
            "rgmc_genes": ["CCNO"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Primary-Ciliary-Dyskinesia-Atlas",
        "critical_clinical_rules": {
            "nNO-DIAGNOSTIC-CUTOFF": "Nasal NO (nNO) <77 nL/min PATHOGNOMONIC for PCD — confirmed in all ODA/IDA subtypes; RSPH4A/CCNO may be borderline (30-60 nL/min)",
            "DNAH11-NORMAL-TEM-PATHOGNOMONIC": "DNAH11-PCD shows NORMAL ultrastructure on TEM — a normal TEM does NOT exclude PCD; waveform video analysis MANDATORY in all PCD suspects",
            "KARTAGENER-TRIAD": "Situs inversus totalis + bronchiectasis + chronic rhinosinusitis = Kartagener syndrome = PCD PATHOGNOMONIC; diagnose and start airway clearance immediately",
            "NEONATAL-RESPIRATORY-DISTRESS": "Neonatal respiratory distress (term baby, no other cause) = screen for PCD; nNO at 5+ weeks if stable",
            "MALE-INFERTILITY": "Male infertility in ODA-defect genes (DNAH5/DNAI1/DNAI2): ICSI successful — sperm structurally present; refer reproductive medicine",
            "CCNO-FEMALE-SUBFERTILITY": "CCNO/MCIDAS-RGMC: female subfertility HIGH (65%) — ectopic pregnancy risk; tubal cilia absent; IVF/embryo transfer preferred",
            "MACROLIDE-PROPHYLAXIS": "Long-term azithromycin (3x/week) reduces exacerbations and preserves FEV1 — standard of care in PCD with established bronchiectasis",
            "RSPH4A-NO-SITUS": "RSPH4A central-pair defect: situs inversus RARE (<5%) — absence of situs inversus does NOT exclude PCD in this subtype",
        },
        "definitions": [
            {
                "term": "Primary Ciliary Dyskinesia (PCD)",
                "definition": "Autosomal recessive disorder of motile cilia causing chronic sinopulmonary disease, laterality defects (~50%), and male/female infertility. Affects ~1 in 10,000-20,000. Diagnosis: nNO + TEM + waveform + genetics.",
            },
            {
                "term": "Kartagener Syndrome",
                "definition": "PCD + situs inversus totalis. ~50% of ODA-defect PCD. Triad: situs inversus + bronchiectasis + chronic sinusitis. Pathognomonic for PCD when triad complete.",
            },
            {
                "term": "Nasal Nitric Oxide (nNO)",
                "definition": "nNO <77 nL/min PATHOGNOMONIC for PCD (sensitivity ~90%, specificity ~97%). Measured by NIOX at velum closure. Not valid in infants <5 years — use alternative tests. Low in cystic fibrosis too — confirm with sweat test.",
            },
            {
                "term": "Outer Dynein Arm (ODA) Defect",
                "definition": "Most common PCD ultrastructure defect (DNAH5, DNAI1, DNAI2). TEM shows absent ODA; 0-4 ODA per axoneme (normal 9). Cilia beat frequency reduced, immotile in severe cases.",
            },
            {
                "term": "IDA + Microtubule Transposition Defect",
                "definition": "CCDC39/CCDC40 genes. Loss of IDA + transposition of one peripheral doublet to centre. Most severe lung phenotype (lowest FEV1). Cilia immotile.",
            },
            {
                "term": "Normal TEM in PCD (DNAH11)",
                "definition": "DNAH11-PCD: ultrastructure NORMAL on TEM. Cilia are hyperkinetic with stiff waveform. Misdiagnosed by TEM-only workup. Waveform video analysis (high-speed video microscopy) mandatory.",
            },
            {
                "term": "RGMC (Reduced Generation of Multiple Cilia)",
                "definition": "CCNO/MCIDAS genes. Cilia number markedly reduced (not structural defect). TEM may appear near-normal. Severe bronchiectasis. Female subfertility prominent. nNO borderline.",
            },
            {
                "term": "Waveform Video Analysis",
                "definition": "High-speed video microscopy of nasal epithelial brushings. Classifies cilia beat pattern: reduced frequency, immotile, hyperkinetic, stiff, rotational. MANDATORY when TEM normal and PCD suspected.",
            },
            {
                "term": "Bronchiectasis in PCD",
                "definition": "Progressive irreversible airway dilation from chronic infection. Starts in lower lobes, middle lobe, lingula. Annual HRCT or low-dose CT from diagnosis. Physiotherapy twice daily standard of care.",
            },
            {
                "term": "Male Infertility Management",
                "definition": "Sperm structurally present in ODA-defect PCD — ICSI (intracytoplasmic sperm injection) effective. Sperm may have absent/short tails in DNAH1/DNAH2 subtypes — specialist referral mandatory.",
            },
            {
                "term": "Situs Inversus Totalis",
                "definition": "Mirror-image arrangement of all visceral organs. ~45-50% of ODA-defect PCD. Not associated with central-pair defects (RSPH4A). Dextrocardia — ECG leads must be reversed for interpretation.",
            },
            {
                "term": "Azithromycin Prophylaxis",
                "definition": "250mg 3x/week (adults) long-term reduces exacerbations, hospitalisations, and preserves FEV1 in PCD. Monitor for QTc prolongation and macrolide resistance. Annual sputum cultures.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps(ov, indent=2)[:800])
    print("\n=== BREAKDOWN (sample) ===")
    br = generate_breakdown()
    print(json.dumps({"genes": list(br["gene_data"].keys()), "total": br["cohort_summary"]["total_patients"]}, indent=2))
    print("\n=== DEFINITIONS (critical rules) ===")
    df = generate_definitions()
    for k, v in list(df["critical_clinical_rules"].items())[:3]:
        print(f"  {k}: {v[:80]}...")
