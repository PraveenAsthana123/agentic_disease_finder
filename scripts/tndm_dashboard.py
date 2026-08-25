#!/usr/bin/env python3
"""Transient Neonatal Diabetes Mellitus (TNDM) Dashboard — 6q24 / PLAGL1 / HYMAI.

TNDM is the archetypal NEONATAL DIABETES IMPRINTING DISORDER — caused by excess expression
of the paternally-imprinted PLAGL1 (ZAC1/LOT1) and HYMAI genes at 6q24.2.
  Principal genes: PLAGL1 (Pleomorphic Adenoma Gene-Like 1 / ZAC1 / LOT1, paternally expressed,
                            zinc-finger transcription factor, regulates beta-cell apoptosis/G1 arrest)
                   HYMAI  (Hydatidiform Mole Associated and Imprinted lncRNA, paternally expressed,
                            co-regulated with PLAGL1 from shared DMR at 6q24.2)
  Mechanism: PLAGL1 + HYMAI are PATERNALLY EXPRESSED (maternal allele imprinted/silenced).
             Loss of maternal silencing → BIALLELIC PLAGL1/HYMAI → neonatal beta-cell dysfunction.
  Mechanisms (TNDM1 — the imprinting form):
    1. Paternal duplication 6q24 (~40%):  extra paternal copy → 2× PLAGL1 dose
    2. Paternal UPD6 (upd(6)pat, ~40%):   two paternal chr6 → biallelic PLAGL1, no maternal chr6
    3. Maternal methylation defect (~20%): DMR1/DMR2 at 6q24 hypomethylated on maternal allele →
                                           normally silent maternal PLAGL1/HYMAI becomes active
  OMIM Disease: #601410 (TNDM1, 6q24) · Genes: PLAGL1 *603044 · HYMAI *606522
  Prevalence: ~1:400,000 (TNDM is rarest NBS-detectable neonatal diabetes form; ~1:89,000 overall NDM)

IMPRINTING MECHANISM — WHY MATERNAL HYPOMETHYLATION / PATERNAL DUPLICATION CAUSES TNDM:
  PLAGL1/HYMAI DMR (6q24.2):
    — Normal: DMR METHYLATED on maternal allele → maternal PLAGL1+HYMAI SILENCED
    — Normal: DMR UNMETHYLATED on paternal allele → paternal PLAGL1+HYMAI EXPRESSED
    — In TNDM (mat hypomethylation, ~20%): maternal DMR loses methylation → PLAGL1+HYMAI from BOTH alleles
    — In TNDM (pat dup, ~40%): extra paternal segment → 2 active PLAGL1/HYMAI copies + 1 silent maternal
    — In TNDM (UPD6pat, ~40%): 2 paternal chr6 (both active) + 0 maternal chr6 → 2× PLAGL1/HYMAI
  PLAGL1 excess → promotes G1 arrest + apoptosis in pancreatic beta cells (neonatal period only)
  → Transient insulin deficiency → neonatal hyperglycemia
  Why TRANSIENT: PLAGL1 expression is developmentally regulated; expression falls after neonatal period
                 → beta-cell mass recovers → remission typically by 3-18 months
  Why RELAPSE: PLAGL1 may re-emerge in adolescence/adulthood → beta-cell stress → T2D-like relapse
               Also: residual subclinical beta-cell defect → impaired first-phase insulin secretion

CLINICAL NATURAL HISTORY:
  Phase 1 — Neonatal (0-3 months): Severe hyperglycemia (blood glucose >200-400 mg/dL), IUGR/SGA,
            polyuria, dehydration, failure to thrive, macroglossia (30%), umbilical hernia (30%)
  Phase 2 — Remission (3-18 months): Insulin requirements fall to zero; glucose normalises
  Phase 3 — Latency (childhood): Asymptomatic, normal glucose, but impaired first-phase insulin secretion
  Phase 4 — Relapse (adolescence/adulthood, ~50%): T2D-like; SULFONYLUREAS HIGHLY EFFECTIVE

KEY TREATMENT CONTRASTS vs OTHER NEONATAL DIABETES:
  TNDM1 (6q24): Insulin in neonatal phase → remission → sulfonylurea at relapse
  TNDM2 (ABCC8 K-ATP channel): SULFONYLUREA FIRST LINE (even in neonatal period) — Channel gain-of-function
  TNDM3 (KCNJ11 K-ATP channel): SULFONYLUREA FIRST LINE — similar to ABCC8
  T1D: Insulin lifelong, autoantibodies positive, sulfonylurea DOES NOT WORK
  KEY: Genetic testing MANDATORY in all neonatal diabetes — ABCC8/KCNJ11 mutation = switch to sulfo NOW

DIAGNOSTIC TESTING SEQUENCE:
  1. MS-MLPA (methylation-sensitive MLPA) for 6q24 — detects duplication + methylation defect
  2. SNP array — identifies UPD6pat (LOH chr6) + confirms duplication
  3. ABCC8 + KCNJ11 sequencing — to distinguish TNDM2/3 (K-ATP channel) from TNDM1
  4. Pancreatic antibodies (GADA, ZnT8, IA-2) — NEGATIVE in TNDM (confirms NOT T1D)
  5. C-peptide — low/undetectable in neonatal phase; may recover in remission

DRUG SAFETY (critical for TNDM):
  Sulfonylurea (glyburide/glibenclamide, glipizide): LEVEL A at relapse — very high response rate
                                                      Even at ABCC8/KCNJ11 (if not TNDM1): FIRST LINE
  Insulin: Standard in neonatal phase; weaned in remission
  SGLT2 inhibitors: NOT recommended in neonatal/pediatric phase; consider in adult relapse if K-ATP
  Metformin: ROLE UNCERTAIN — insulin sensitizer; may help T2D-like relapse in adults
  Diazoxide: Opens K-ATP channels — CONTRAINDICATED in K-ATP gain-of-function (ABCC8/KCNJ11) as
             mechanism redundant; may be tried in TNDM1 for hyperinsulinism (rare situations)
"""

from __future__ import annotations
import random, datetime, math

SEED = 299
RNG  = random.Random(SEED)

TODAY = datetime.date.today().isoformat()

# ── 40-patient TNDM cohort (seed 299) ──────────────────────────────────────
def _make_cohort():
    rng = random.Random(SEED)
    mechanisms = [
        ("Paternal duplication 6q24",     0.40),
        ("Paternal UPD6 (upd(6)pat)",     0.40),
        ("Maternal DMR hypomethylation",  0.20),
    ]
    outcomes = [
        ("Remission (on track)",          0.55),
        ("Relapsed — sulfonylurea",        0.25),
        ("Neonatal active — insulin",      0.15),
        ("Lost to follow-up",             0.05),
    ]
    aed_choices = ["insulin", "insulin+metformin", "glyburide", "glipizide", "insulin→glyburide"]
    rows = []
    for i in range(40):
        mech = rng.choices([m[0] for m in mechanisms], weights=[m[1] for m in mechanisms])[0]
        outcome = rng.choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0]
        onset_wk = rng.randint(1, 5)          # weeks of life
        remission_mo = rng.randint(3, 18) if "Neonatal" not in outcome else None
        relapse_age_yr = rng.randint(12, 35) if "sulfo" in outcome.lower() else None
        macroglossia = rng.random() < 0.35
        umbilical = rng.random() < 0.32
        iugr = rng.random() < 0.75
        birth_wt_sds = round(rng.uniform(-3.0, -0.5) if iugr else rng.uniform(-0.5, 1.2), 1)
        sex = rng.choice(["M", "F"])
        rows.append({
            "id": f"TNDM-{i+1:02d}",
            "sex": sex,
            "mechanism": mech,
            "onset_wk": onset_wk,
            "birth_wt_sds": birth_wt_sds,
            "macroglossia": macroglossia,
            "umbilical_hernia": umbilical,
            "iugr": iugr,
            "outcome": outcome,
            "remission_mo": remission_mo,
            "relapse_age_yr": relapse_age_yr,
            "current_tx": rng.choice(aed_choices),
            "antibody_negative": True,  # TNDM is NOT T1D
        })
    return rows

COHORT = _make_cohort()

# ──────────────────────────────────────────────────────────────────────────
def get_overview():
    n = len(COHORT)
    mech_counts = {}
    for p in COHORT:
        mech_counts[p["mechanism"]] = mech_counts.get(p["mechanism"], 0) + 1

    remission_n = sum(1 for p in COHORT if "Remission" in p["outcome"] or "sulfo" in p["outcome"].lower())
    relapse_n   = sum(1 for p in COHORT if "sulfo" in p["outcome"].lower())
    active_n    = sum(1 for p in COHORT if "Neonatal" in p["outcome"])
    macroglossia_n = sum(1 for p in COHORT if p["macroglossia"])
    iugr_n      = sum(1 for p in COHORT if p["iugr"])
    umbilical_n = sum(1 for p in COHORT if p["umbilical_hernia"])
    ab_neg_n    = sum(1 for p in COHORT if p["antibody_negative"])

    kpis = {
        "total_patients":       n,
        "mechanism_patdup":     mech_counts.get("Paternal duplication 6q24", 0),
        "mechanism_upd6pat":    mech_counts.get("Paternal UPD6 (upd(6)pat)", 0),
        "mechanism_mat_hypometh": mech_counts.get("Maternal DMR hypomethylation", 0),
        "remission_or_relapse": remission_n,
        "relapse_on_sulfo":     relapse_n,
        "neonatal_active":      active_n,
        "macroglossia_pct":     round(100 * macroglossia_n / n),
        "iugr_sga_pct":         round(100 * iugr_n / n),
        "umbilical_pct":        round(100 * umbilical_n / n),
        "antibody_negative_pct": round(100 * ab_neg_n / n),
        "sulfonylurea_response": "~85–95%",
    }

    clinical_alerts = [
        {"level": "danger",  "msg": "ALL neonatal diabetes requires ABCC8+KCNJ11 sequencing — K-ATP mutation = switch to sulfonylurea immediately (even in neonate)."},
        {"level": "warning", "msg": "TNDM1 (6q24): Insulin in neonatal phase — wean in remission. Sulfonylurea at relapse."},
        {"level": "warning", "msg": "~50% relapse in adolescence/adulthood as T2D-like — LIFELONG glucose monitoring mandatory."},
        {"level": "info",    "msg": "Pancreatic antibodies (GADA, ZnT8, IA-2) are NEGATIVE — absence confirms NOT T1D (no immunosuppression)."},
        {"level": "info",    "msg": "Macroglossia ~30-35% — shared with BWS (11p15.5) but mechanism completely different (PLAGL1 vs IGF2/CDKN1C)."},
    ]

    mechanism_note = (
        "TNDM1 (6q24, PLAGL1/HYMAI imprinting): biallelic PLAGL1 expression suppresses neonatal "
        "beta cells. Paternal duplication (~40%) and paternal UPD6 (~40%) give identical phenotype; "
        "maternal DMR hypomethylation (~20%) — check for multi-locus imprinting disturbance (MLID)."
    )

    treatment_ladder = [
        {"step": 1, "tx": "Insulin (continuous IV or SC)",       "phase": "Neonatal (0-3 months)", "evidence": "Standard of care"},
        {"step": 2, "tx": "Wean insulin (remission phase)",      "phase": "3-18 months",           "evidence": "PLAGL1 expression falls developmentally"},
        {"step": 3, "tx": "Glucose monitoring in remission",     "phase": "Childhood",              "evidence": "Monitor for relapse; first-phase impaired"},
        {"step": 4, "tx": "Sulfonylurea (glyburide/glipizide)",  "phase": "Adolescent/Adult relapse","evidence": "Level A — ~85-95% response; K-ATP sensitive"},
        {"step": 5, "tx": "Add metformin if sulfo insufficient", "phase": "Adult relapse",          "evidence": "Level C — insulin sensitizer adjunct"},
    ]

    return {
        "disease": "Transient Neonatal Diabetes Mellitus (TNDM)",
        "subtitle": "TNDM1 — 6q24 / PLAGL1 / HYMAI / Genomic Imprinting (Paternal GOF / Maternal LOF)",
        "omim": "#601410",
        "locus": "6q24.2",
        "genes": ["PLAGL1 (*603044)", "HYMAI (*606522)"],
        "prevalence": "~1:400,000 (TNDM1); all neonatal DM ~1:89,000",
        "inheritance": "Genomic Imprinting — Paternal GOF (biallelic PLAGL1/HYMAI expression)",
        "generated": TODAY,
        "seed": SEED,
        "kpis": kpis,
        "clinical_alerts": clinical_alerts,
        "mechanism_note": mechanism_note,
        "treatment_ladder": treatment_ladder,
    }


def get_breakdown():
    mech_dist = {}
    outcome_dist = {}
    birth_wt_by_mech = {}
    for p in COHORT:
        mech_dist[p["mechanism"]] = mech_dist.get(p["mechanism"], 0) + 1
        outcome_dist[p["outcome"]] = outcome_dist.get(p["outcome"], 0) + 1
        birth_wt_by_mech.setdefault(p["mechanism"], []).append(p["birth_wt_sds"])

    mech_avg_bwt = {k: round(sum(v)/len(v), 2) for k, v in birth_wt_by_mech.items()}

    remission_times = [p["remission_mo"] for p in COHORT if p["remission_mo"] is not None]
    median_remission = sorted(remission_times)[len(remission_times)//2] if remission_times else None

    feature_rates = {
        "IUGR/SGA":          round(100 * sum(1 for p in COHORT if p["iugr"]) / 40),
        "Macroglossia":       round(100 * sum(1 for p in COHORT if p["macroglossia"]) / 40),
        "Umbilical hernia":   round(100 * sum(1 for p in COHORT if p["umbilical_hernia"]) / 40),
        "Ab-negative (not T1D)": 100,
    }

    onset_wks = [p["onset_wk"] for p in COHORT]
    avg_onset = round(sum(onset_wks) / len(onset_wks), 1)

    drug_counts = {}
    for p in COHORT:
        drug_counts[p["current_tx"]] = drug_counts.get(p["current_tx"], 0) + 1

    # Mechanism comparison table (TNDM1 vs TNDM2/3 vs T1D)
    comparison = [
        {"entity": "TNDM1 (6q24)",    "gene": "PLAGL1/HYMAI", "mechanism": "Imprinting (PLAGL1 excess)", "sulfo_response": "Yes (relapse)", "remits": "Yes ~100%", "antibodies": "Negative"},
        {"entity": "TNDM2 (K-ATP)",   "gene": "ABCC8",        "mechanism": "K-ATP channel gain-of-function", "sulfo_response": "Yes (1st line even in neonate)", "remits": "~50%", "antibodies": "Negative"},
        {"entity": "TNDM3 (K-ATP)",   "gene": "KCNJ11",       "mechanism": "K-ATP channel gain-of-function", "sulfo_response": "Yes (1st line even in neonate)", "remits": "~50%", "antibodies": "Negative"},
        {"entity": "T1D autoimmune",  "gene": "HLA/PTPN22",   "mechanism": "Autoimmune beta-cell destruction", "sulfo_response": "No", "remits": "No", "antibodies": "POSITIVE (GADA, ZnT8)"},
    ]

    cohort_table = [
        {k: v for k, v in p.items()}
        for p in COHORT
    ]

    return {
        "mechanism_distribution": mech_dist,
        "outcome_distribution": outcome_dist,
        "birth_wt_sds_by_mech": mech_avg_bwt,
        "median_remission_months": median_remission,
        "avg_onset_weeks": avg_onset,
        "feature_rates_pct": feature_rates,
        "current_treatment_distribution": drug_counts,
        "neonatal_diabetes_comparison": comparison,
        "cohort_table": cohort_table,
    }


def get_definitions():
    return {
        "disease_overview": {
            "full_name": "Transient Neonatal Diabetes Mellitus type 1 (TNDM1)",
            "omim": "#601410",
            "locus": "6q24.2",
            "mim_genes": {"PLAGL1": "*603044", "HYMAI": "*606522"},
            "inheritance": "Genomic imprinting — paternally expressed genes; maternal allele silenced by DMR methylation",
            "prevalence": "~1:400,000 live births (TNDM1 specifically); all forms NDM ~1:89,000",
        },
        "genes": {
            "PLAGL1": {
                "full_name": "Pleomorphic Adenoma Gene-Like 1 (also ZAC1, LOT1)",
                "size": "~790 aa (multiple isoforms)",
                "function": "Zinc-finger transcription factor; promotes G1 arrest and apoptosis in beta cells; regulates IGF1R and IGF2 expression; paternally expressed (maternally imprinted at 6q24.2 DMR)",
                "omim": "*603044",
                "key_fact": "Excess PLAGL1 (biallelic expression) → neonatal beta-cell G1 arrest → transient insulin deficiency; expression falls with maturation → remission",
            },
            "HYMAI": {
                "full_name": "Hydatidiform Mole Associated and Imprinted lncRNA",
                "function": "Long non-coding RNA, paternally expressed; co-regulated with PLAGL1 from shared 6q24.2 DMR; exact role in TNDM uncertain but consistently biallelic in TNDM1",
                "omim": "*606522",
                "key_fact": "Used as a diagnostic methylation marker for TNDM1 alongside PLAGL1 DMR",
            },
        },
        "mechanisms": {
            "paternal_duplication_6q24": {
                "frequency": "~40%",
                "description": "Extra paternal copy of 6q24 region → 3 copies: 2 active paternal + 1 silent maternal PLAGL1/HYMAI",
                "recurrence": "Familial (autosomal dominant) if inherited from father; de novo if new",
                "testing": "Array CGH, SNP array, MLPA",
            },
            "paternal_UPD6": {
                "frequency": "~40%",
                "description": "Paternal uniparental disomy chromosome 6 → two paternal chr6, zero maternal → biallelic PLAGL1/HYMAI",
                "recurrence": "Sporadic (<1% recurrence)",
                "testing": "SNP array genome-wide (LOH on chr6); methylation confirms",
            },
            "maternal_DMR_hypomethylation": {
                "frequency": "~20%",
                "description": "Maternal DMR (differentially methylated region at 6q24.2) loses methylation → normally silent maternal PLAGL1+HYMAI expressed → biallelic",
                "note": "May be part of multi-locus imprinting disturbance (MLID) — check other DMRs (KCNQ1OT1, H19, SNRPN, etc.)",
                "testing": "MS-MLPA or bisulfite sequencing of 6q24 DMR; SNP array normal",
            },
        },
        "clinical_features": {
            "neonatal_diabetes": "Onset <6 weeks of life (mean 1-2 weeks); blood glucose >200-400 mg/dL; severe dehydration; polyuria; failure to thrive; insulin-requiring",
            "iugr_sga": "~70-80% IUGR/SGA — PLAGL1 may suppress IGF1R/IGF2 in utero; contrasts with BWS macrosomia",
            "macroglossia": "~30-35% — tongue enlargement; shared feature with BWS but mechanism completely different (PLAGL1 vs IGF2/CDKN1C excess)",
            "umbilical_abnormality": "~30% umbilical hernia or omphalocele — BWS has omphalocele 30% (CDKN1C LOF); TNDM umbilical hernias usually smaller",
            "large_fontanelle": "~20%",
            "remission": "100% of TNDM1 remit (by definition); median ~3-4 months; range 3-18 months",
            "relapse": "~50% relapse in adolescence/adulthood as T2D-like diabetes; 1st-phase insulin impaired lifelong",
        },
        "treatment": {
            "neonatal_phase": {
                "insulin": "Standard of care — continuous IV insulin infusion or subcutaneous; titrate to maintain glucose 70-180 mg/dL",
                "monitoring": "Glucose every 1-4 hours; adjust for feeds; avoid hypoglycemia",
                "weaning": "Wean when remission begins (insulin requirements fall to zero over weeks-months)",
            },
            "relapse_phase": {
                "sulfonylurea": "LEVEL A — glyburide (glibenclamide) or glipizide; ~85-95% response rate; stimulates K-ATP-independent insulin secretion; effective even in TNDM1 relapse",
                "metformin": "Adjunct if sulfonylurea insufficient — Level C; insulin sensitizer",
                "insulin": "Reserve for sulfo failure or complications",
            },
            "ABCC8_KCNJ11_distinction": "CRITICAL: If ABCC8 or KCNJ11 mutation found → sulfonylurea FIRST LINE immediately, even in neonatal period. Do NOT wait for remission. TNDM2/TNDM3 = K-ATP channel, sulfo works directly.",
            "what_NOT_to_do": [
                "Do NOT use immunosuppression — TNDM is NOT autoimmune",
                "Do NOT assume T1D if antibodies NOT checked — always check GADA, ZnT8, IA-2",
                "Do NOT stop glucose monitoring at remission — ~50% relapse risk lifelong",
                "Do NOT use diazoxide for K-ATP gain-of-function mutations (ABCC8/KCNJ11) — mechanism redundant",
            ],
        },
        "diagnostics": {
            "first_line": "MS-MLPA for 6q24 (PLAGL1/HYMAI DMR methylation + copy number) — detects duplication and methylation defect",
            "second_line": "SNP array genome-wide — identifies UPD6pat (chr6 LOH) + confirms duplication",
            "third_line": "ABCC8 + KCNJ11 sequencing — distinguish TNDM2/TNDM3 from TNDM1 (critical for sulfonylurea timing)",
            "antibody_panel": "GADA, IA-2, ZnT8 — NEGATIVE in TNDM (confirmatory); positive = T1D",
            "c_peptide": "Low/undetectable in neonatal phase; check at remission and annually for relapse monitoring",
        },
        "contrasts": {
            "BWS_vs_TNDM": "BWS (11p15.5, IGF2/CDKN1C): macrosomia + tumor risk + hyperinsulinism. TNDM (6q24, PLAGL1): SGA/IUGR + diabetes (opposite metabolic direction). Both have macroglossia. COMPLETELY different loci, genes, and mechanisms.",
            "TNDM1_vs_TNDM2_3": "TNDM1 (6q24, imprinting): sulfo works at relapse, not in neonatal. TNDM2/3 (ABCC8/KCNJ11, K-ATP): sulfo works immediately — the most important clinical distinction.",
            "TNDM_vs_T1D": "TNDM: neonatal onset, antibody-negative, remits, responds to sulfo. T1D: any age (peak childhood), antibody-POSITIVE, permanent, insulin-dependent, sulfo DOES NOT WORK.",
            "TNDM_vs_PNDM": "PNDM (permanent NDM, e.g., KCNJ11 severe mutations, EIF2AK3, PTF1A): does NOT remit. TNDM: remits 100% by definition (though ABCC8/KCNJ11 TNDM2/3 may be permanent in ~50%).",
        },
        "surveillance": {
            "glucose_in_remission": "HbA1c + fasting glucose annually from age 5",
            "oral_glucose_tolerance": "OGTT every 2-3 years in adolescence/adulthood (detect relapse early)",
            "first_phase_insulin": "IVGTT (intravenous glucose tolerance test) to assess secretory capacity — research tool",
            "genetic_counselling": "Paternal duplication: 50% if father carries; UPD6/DMR epimutation: <5% recurrence",
        },
        "mlid_note": "Multi-locus imprinting disturbance (MLID): ~20% of maternal DMR hypomethylation cases are part of MLID — multiple DMRs abnormal. Associated with IVF/ART. Check KCNQ1OT1, H19-ICR1, SNRPN DMRs in maternal hypomethylation cases.",
        "art_risk": "IVF/ICSI associated with ~2-3× increased risk of maternal DMR hypomethylation → TNDM1 (and MLID). Counsel ART families.",
    }
