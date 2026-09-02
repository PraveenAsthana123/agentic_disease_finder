#!/usr/bin/env python3
"""COA6 — Complex IV (COX) Deficiency with Infantile Cardiomyopathy (COXPD14).

COA6 (Cytochrome c Oxidase Assembly Factor 6) is a nuclear-encoded mitochondrial
copper chaperone that delivers copper to the CuA site of MT-CO2 (COX2), working
cooperatively with SCO1 and SCO2 in the complex IV assembly pathway.

  COA6 gene            OMIM *614772
  Disease (COXPD14)    OMIM #614924
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COA6 / copper chaperone for COX2 CuA site):
COA6 is a small ~8 kDa protein with a conserved twin-CX9C copper-binding
motif (Cys-X9-Cys). It localises to the mitochondrial matrix/IMS interface
and functions as a copper relay in the COX2 metalation pathway:
  • COA6 cooperates with SCO1 and SCO2 to load copper onto the CuA centre
    of MT-CO2 (COX2), the binuclear copper site critical for electron transfer
    from cytochrome c to the catalytic haem a3–CuB site.
  • COA6 mutations disrupt copper delivery to CuA → MT-CO2 cannot be
    metalated → late-stage COX assembly fails → Complex IV severely reduced
    (<10% of normal in cardiac and skeletal muscle).
  • Isolated COX deficiency: Complexes I, II, III NORMAL — biochemical
    fingerprint shared with SCO1, SCO2, COX15, SURF1 — WES required.
  • HCM is CARDINAL in COA6: cardiac muscle has highest COA6 expression →
    copper-depleted CuA in cardiac MT-CO2 → ATP failure in cardiomyocytes →
    compensatory hypertrophy → HCM develops in infancy.

KEY CLINICAL DIFFERENTIATOR vs. OTHER COX ASSEMBLY FACTOR DISEASES:
  • HCM 90%          — CARDINAL; KEY DDx vs COX6B1 (0%), SURF1 (~10%)
  • Liver involvement 35% — distinguishes COA6 from SCO2 (liver 0%) at bedside
  • NO renal tubulopathy — KEY DDx vs COX10 (65%)
  • NO anaemia          — KEY DDx vs COX10 (80%)
  • Copper pathway — distinguishes COA6 mechanistically from SURF1 (haem a)

MOLECULAR: Biallelic (AR) loss-of-function COA6 variants:
  — p.Trp59Cys (c.177G>C): twin-CX9C Trp59 adjacent to copper-coordinating
    Cys; disrupts hydrophobic copper-binding cage; most common (~40%);
    Australian + UK founder; severe LOF.
  — p.Trp6Arg (c.16T>C): N-terminal; severe; disrupts mitochondrial
    targeting or early folding (~15%).
  — p.Gly42Ser (c.124G>A): copper-coordination region; moderate-severe
    residual function (~20%).
  — p.Cys66Tyr (c.197G>A): second Cys of the CX9C motif; copper-binding
    disruption; severe (~10%).
  — Splice / truncating: null alleles; severe (~15%).
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 611
DISEASE_ID   = "coa6"
DISEASE_NAME = "COA6 Infantile Cardiomyopathic Complex IV Deficiency (COXPD14)"
GENE         = "COA6"
OMIM_GENE    = "*614772"
OMIM_DISEASE = "#614924"
CHROMOSOME   = "1q42.2"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal to early infantile (median 1–4 months; range birth–8 months)"
COHORT_SIZE  = 40
COLOR        = "#c62828"   # deep crimson — cardiac / HCM predominant
LIGHT        = "#ffebee"


# Genotype pool
GENO_TRP59CYS_HOM  = "p.Trp59Cys homozygous (c.177G>C) — twin-CX9C copper-binding motif"
GENO_TRP59CYS_CPX  = "p.Trp59Cys / p.Gly42Ser — compound heterozygous"
GENO_TRP59CYS_NULL = "p.Trp59Cys / truncating null — compound heterozygous"
GENO_TRP6ARG_CPX   = "p.Trp6Arg / p.Cys66Tyr — compound heterozygous"
GENO_OTHER_CPX     = "Other biallelic COA6 — compound heterozygous"

GENO_POOL    = [GENO_TRP59CYS_HOM, GENO_TRP59CYS_CPX, GENO_TRP59CYS_NULL,
                GENO_TRP6ARG_CPX, GENO_OTHER_CPX]
GENO_WEIGHTS = [0.30, 0.20, 0.15, 0.15, 0.20]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COA6 cohort (seed-611)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COA6 cohort (seed-611).

    All patients have isolated COX deficiency with HCM as the cardinal feature.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"

        # Onset age (months)
        is_null = "truncating" in geno or "null" in geno
        if is_null:
            onset_mo = round(rng.uniform(0.0, 2.0), 1)
            lactate  = round(rng.uniform(6.0, 22.0), 1)
            cox_pct  = round(rng.uniform(1.0, 8.0),  1)
        elif "Trp59Cys" in geno and "homozygous" in geno:
            onset_mo = round(rng.uniform(0.5, 4.0), 1)
            lactate  = round(rng.uniform(5.0, 18.0), 1)
            cox_pct  = round(rng.uniform(2.0, 10.0), 1)
        else:
            onset_mo = round(rng.uniform(1.0, 8.0), 1)
            lactate  = round(rng.uniform(4.0, 14.0), 1)
            cox_pct  = round(rng.uniform(3.0, 14.0), 1)

        # Features (published frequency estimates)
        hcm          = rng.random() < 0.90   # 90% — CARDINAL
        lactic_ac    = rng.random() < 0.95   # 95%
        hypotonia    = rng.random() < 0.70   # 70%
        encephalo    = rng.random() < 0.55   # 55%
        myopathy     = rng.random() < 0.65   # 65%
        liver        = rng.random() < 0.35   # 35% — KEY DDx vs SCO2 (liver 0%)
        leigh_mri    = rng.random() < 0.30   # 30%
        resp_comp    = rng.random() < 0.50   # 50%
        # Disease-defining negatives
        tubulopathy  = False   # 0% — KEY DDx vs COX10
        anaemia      = False   # 0% — KEY DDx vs COX10

        # Outcome (high mortality)
        survival_mo = round(rng.uniform(0.5, 36.0), 1)
        if is_null:
            alive = rng.random() < 0.20
        elif "Trp59Cys" in geno and "homozygous" in geno:
            alive = rng.random() < 0.25
        else:
            alive = rng.random() < 0.32
        if alive:
            outcome = f"Alive {survival_mo:.1f} months (last follow-up)"
        else:
            outcome = f"Died at {survival_mo:.1f} months"

        # Treatments
        txs = ["CoQ10/Ubiquinol (Level C)"]
        txs.append("Riboflavin B2 (Level C)")
        txs.append("Thiamine B1 (Level C — MANDATORY empiric Leigh mimic)")
        txs.append("Biotin (Level C — MANDATORY empiric BTD/BTBGD)")
        txs.append("IV Dextrose GIR 6–8 (crisis / perioperative)")
        if hcm:          txs.append("Cardiac: propranolol/atenolol (avoid positive inotropes)")
        if liver:        txs.append("UDCA hepatic support 15–20 mg/kg/day (Level C)")
        if lactic_ac:    txs.append("L-Carnitine (Level C — secondary carnitine deficiency)")
        if resp_comp:    txs.append("NIV/mechanical ventilation")
        txs.append("Copper histidinate — investigational (theoretical copper chaperone support)")

        # Alerts
        alerts_list = [
            "🚫 VPA ABSOLUTE CI — CoA sequestration / POLG inhibition / hepatotoxicity; LEV instead",
            "🚫 Metformin ABSOLUTE CI — Complex I inhibition → lactic crisis in COX deficiency",
            "🚫 Propofol ABSOLUTE CI — PRIS: direct Complex IV inhibition; use sevoflurane",
            "🚫 Positive inotropes (digoxin/dobutamine) HIGH RISK in HCM — increase O2 demand, worsen ATP failure",
            "🚫 KD CONTRAINDICATED — beta-oxidation requires functional Complex IV",
            "🚫 Linezolid ABSOLUTE CI — mt-23S rRNA → blocks MT-CO1/CO2/CO3 synthesis",
        ]
        if hcm:
            alerts_list.append("⚠ HCM: avoid tachycardia triggers; propranolol/atenolol for rate control; ECHO monitoring")
        alerts = "; ".join(alerts_list)

        # Feature string
        feats = []
        if hcm:         feats.append("HCM (cardinal)")
        if lactic_ac:   feats.append("Lactic acidosis")
        if hypotonia:   feats.append("Hypotonia")
        if encephalo:   feats.append("Encephalopathy")
        if myopathy:    feats.append("Myopathy")
        if liver:       feats.append("Hepatic involvement")
        if leigh_mri:   feats.append("Leigh-like MRI")
        if resp_comp:   feats.append("Resp. compromise")
        feats.append("NO Tubulopathy (key DDx)")
        feats.append("NO Anaemia (key DDx)")

        patients.append({
            "id":         f"COA6-{i:03d}",
            "geno":       geno,
            "sex":        sex,
            "onset_mo":   onset_mo,
            "lactate":    lactate,
            "cox_pct":    cox_pct,
            "features":   ", ".join(feats),
            "treatments": ", ".join(txs),
            "alerts":     alerts,
            "outcome":    outcome,
        })
    return patients


# ── Public API functions ─────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    """COA6 overview — gene, disease identity, KPIs, contraindications."""
    rng    = _rng()
    cohort = _build_cohort(rng)
    n      = len(cohort)

    n_hcm        = sum(1 for p in cohort if "HCM" in p["features"])
    n_lactic     = sum(1 for p in cohort if "Lactic" in p["features"])
    n_hypotonia  = sum(1 for p in cohort if "Hypotonia" in p["features"])
    n_encephalo  = sum(1 for p in cohort if "Encephalo" in p["features"])
    n_myopathy   = sum(1 for p in cohort if "Myopathy" in p["features"])
    n_liver      = sum(1 for p in cohort if "Hepatic" in p["features"])
    n_leigh      = sum(1 for p in cohort if "Leigh" in p["features"])
    n_resp       = sum(1 for p in cohort if "Resp." in p["features"])
    n_alive      = sum(1 for p in cohort if p["outcome"].startswith("Alive"))
    mean_cox     = round(sum(p["cox_pct"] for p in cohort) / n, 1)
    mean_onset   = round(sum(p["onset_mo"] for p in cohort) / n, 1)
    mean_lact    = round(sum(p["lactate"] for p in cohort) / n, 1)

    return {
        "gene":         "COA6 (Cytochrome c Oxidase Assembly Factor 6)",
        "protein":      "COA6-73aa-8.3kDa — Nuclear-encoded mitochondrial copper chaperone; twin-CX9C motif; cooperates with SCO1/SCO2 in copper delivery to CuA site of MT-CO2 (COX2); required for COX2 metalation and late-stage Complex IV assembly",
        "disease":      "Complex IV (COX) Deficiency, Mitochondrial, Nuclear Type 14 (COXPD14) — Infantile Cardiomyopathy and Myopathy",
        "omim_gene":    "*614772 (COA6)",
        "omim_disease": "#614924 (Mitochondrial Complex IV Deficiency, Nuclear Type 14; COXPD14)",
        "chromosome":   "1q42.2",
        "inheritance":  "AR (autosomal recessive biallelic) — 25% recurrence per pregnancy",
        "onset":        "Neonatal to early infantile; median 1–4 months; range birth–8 months",
        "cohort":       f"{n} patients · seed-611 · COA6 biallelic",
        "mechanism": (
            "COA6 is a small 73aa copper chaperone protein that localises to the mitochondrial "
            "IMS/matrix interface and contains a conserved twin-CX9C motif essential for copper "
            "coordination. It acts in the copper relay cascade for COX2 (MT-CO2) metalation: "
            "COA6 cooperates with SCO1 and SCO2 to load the binuclear CuA copper site of MT-CO2, "
            "the site that accepts electrons from cytochrome c for transfer to the catalytic "
            "haem a3–CuB oxygen reduction site. "
            "Without functional COA6: copper is not efficiently delivered to CuA of MT-CO2 → "
            "MT-CO2 cannot be metalated → late-stage COX assembly fails → Complex IV activity "
            "severely reduced (<10% of normal in cardiac and skeletal muscle). "
            "Cardiac muscle has the highest COA6 expression and the greatest dependence on "
            "OXPHOS for ATP production → copper-depleted CuA in cardiac MT-CO2 → myocyte "
            "ATP failure → compensatory hypertrophic cardiomyopathy (HCM) is cardinal. "
            "Biochemical: isolated COX deficiency (CI/CII/CIII NORMAL); WES required to "
            "distinguish from SCO1, SCO2, COX15, and SURF1 (identical biochemical fingerprint)."
        ),
        "differentiator_note": (
            "COA6 is clinically distinguished from other COX assembly factor diseases by: "
            "(1) HCM 90% — CARDINAL feature (like SCO2 100%, unlike COX6B1 0%); "
            "(2) Liver involvement 35% — present in COA6, ABSENT in SCO2 (0%): KEY bedside DDx; "
            "(3) NO renal tubulopathy — absent (KEY DDx vs COX10 65%); "
            "(4) NO anaemia — absent (KEY DDx vs COX10 80%); "
            "(5) Copper pathway mechanism — COA6 (copper chaperone) vs SURF1 (haem a pathway); "
            "(6) Copper-histidinate supplementation: theoretical rationale (COA6 is a copper "
            "chaperone) analogous to SCO2, but clinical evidence is very limited (1–2 cases). "
            "WES is MANDATORY for all isolated COX deficiency cases — biochemical fingerprint "
            "does not differentiate COA6 from SCO2, COX15, or COX6B1."
        ),
        "kpis": [
            {"label": "HCM (cardinal)",          "value": f"{n_hcm/n*100:.0f}%",       "color": COLOR},
            {"label": "Lactic Acidosis",          "value": f"{n_lactic/n*100:.0f}%",    "color": "#b71c1c"},
            {"label": "Hypotonia",                "value": f"{n_hypotonia/n*100:.0f}%", "color": COLOR},
            {"label": "Encephalopathy",           "value": f"{n_encephalo/n*100:.0f}%", "color": COLOR},
            {"label": "Myopathy",                 "value": f"{n_myopathy/n*100:.0f}%",  "color": COLOR},
            {"label": "Hepatic Involvement",      "value": f"{n_liver/n*100:.0f}%",     "color": "#e65100"},
            {"label": "Leigh-like MRI",           "value": f"{n_leigh/n*100:.0f}%",     "color": "#6a1b9a"},
            {"label": "Resp. Compromise",         "value": f"{n_resp/n*100:.0f}%",      "color": "#b71c1c"},
            {"label": "Renal Tubulopathy",        "value": "0% (KEY DDx)",              "color": "#2e7d32"},
            {"label": "Anaemia",                  "value": "0% (KEY DDx)",              "color": "#2e7d32"},
            {"label": "Mean COX Activity",        "value": f"{mean_cox}% normal",       "color": COLOR},
            {"label": "Mean Onset",               "value": f"{mean_onset} months",      "color": COLOR},
            {"label": "Mean Lactate",             "value": f"{mean_lact} mmol/L",       "color": "#b71c1c"},
            {"label": "Alive (last f/u)",         "value": f"{n_alive/n*100:.0f}%",     "color": "#2e7d32"},
        ],
        "contraindications": [
            {
                "drug":      "VPA / Valproate",
                "severity":  "ABSOLUTE CI — ALL COA6 / Complex IV disease",
                "mechanism": (
                    "Triple mechanism: (a) CoA sequestration → impairs TCA cycle and OXPHOS; "
                    "(b) POLG1 inhibition → mtDNA depletion → reduces MT-CO1/CO2/CO3 synthesis "
                    "→ COX assembly collapses further; (c) Hepatotoxic — particularly dangerous "
                    "in COA6 where 35% already have hepatic involvement (VPA + hepatic disease = "
                    "catastrophic). Use LEV (IV loading, renal excretion, no mito toxicity)."
                ),
            },
            {
                "drug":      "Propofol",
                "severity":  "ABSOLUTE CI — PRIS risk in Complex IV / cardiac deficiency",
                "mechanism": (
                    "Propofol directly inhibits Complex IV (COX). In COA6 with pre-existing "
                    "severe COX deficiency and HCM: propofol causes near-complete COX abolition → "
                    "ATP collapse in cardiomyocytes (already barely viable) → rhabdomyolysis + "
                    "cardiac failure (PRIS). The HCM heart in COA6 is particularly vulnerable. "
                    "Use sevoflurane for induction/maintenance; dexmedetomidine for sedation."
                ),
            },
            {
                "drug":      "Metformin",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease including COA6",
                "mechanism": (
                    "Metformin inhibits Complex I. In COA6 (COX/Complex IV deficient), "
                    "electron carriers are already backed up. Complex I inhibition collapses "
                    "the entire respiratory chain → severe lactic crisis. Particularly lethal "
                    "during febrile illness when cardiac energy demand is already maximal."
                ),
            },
            {
                "drug":      "Positive Inotropes (digoxin / dobutamine / milrinone)",
                "severity":  "HIGH RISK in COA6 HCM — avoid without specialist cardiology input",
                "mechanism": (
                    "COA6 HCM is caused by ATP failure (OXPHOS defect) — NOT by contractile "
                    "protein dysfunction. Positive inotropes (digoxin, dobutamine) increase "
                    "myocardial oxygen demand and further worsen the energy deficit → "
                    "can precipitate fatal arrhythmia or ventricular failure. "
                    "Rate control (propranolol, atenolol) is preferred over inotropic support. "
                    "DO NOT use digoxin reflexively for any apparent 'cardiac failure' in COA6."
                ),
            },
            {
                "drug":      "Linezolid",
                "severity":  "ABSOLUTE CI — mitochondrial ribosome inhibition",
                "mechanism": (
                    "Linezolid inhibits the mitochondrial 23S rRNA (mt-LSU), blocking translation "
                    "of MT-CO1, MT-CO2, and MT-CO3 (the catalytic COX subunits). In COA6-deficient "
                    "patients, MT-CO2 is already copper-depleted and partially non-functional; "
                    "linezolid eliminates residual MT-CO2 synthesis entirely, collapsing any "
                    "remaining assembly. Use beta-lactams/carbapenems as alternatives."
                ),
            },
            {
                "drug":      "Ketogenic Diet",
                "severity":  "CONTRAINDICATED — beta-oxidation requires functional Complex IV",
                "mechanism": (
                    "The KD forces beta-oxidation, generating FADH2/NADH that feed directly "
                    "into the respiratory chain. In Complex IV deficiency, OXPHOS is already "
                    "severely impaired — KD worsens energy failure and risks precipitating "
                    "metabolic crisis, particularly in cardiac muscle. Maintain GIR 6–8; "
                    "fasting prohibited."
                ),
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    """COA6 breakdown — 40-patient cohort, genotype distribution, feature prevalence."""
    rng    = _rng()
    cohort = _build_cohort(rng)
    n      = len(cohort)

    # Genotype distribution
    geno_counts: dict[str, int] = {}
    for p in cohort:
        g = p["geno"].split(" —")[0].split(" (")[0]
        geno_counts[g] = geno_counts.get(g, 0) + 1

    geno_dist = [
        {"genotype": k, "n": v, "pct": f"{v/n*100:.0f}%"}
        for k, v in sorted(geno_counts.items(), key=lambda x: -x[1])
    ]

    # Feature prevalence
    feat_map = [
        ("HCM (hypertrophic CMP)",   "HCM"),
        ("Lactic acidosis",          "Lactic"),
        ("Hypotonia",                "Hypotonia"),
        ("Encephalopathy",           "Encephalo"),
        ("Myopathy",                 "Myopathy"),
        ("Hepatic involvement",      "Hepatic"),
        ("Leigh-like MRI",           "Leigh"),
        ("Resp. compromise",         "Resp."),
    ]
    features = [
        {"feature": label, "n": sum(1 for p in cohort if key in p["features"]),
         "pct": f"{sum(1 for p in cohort if key in p['features'])/n*100:.0f}%"}
        for label, key in feat_map
    ]
    features += [
        {"feature": "Renal tubulopathy", "n": 0, "pct": "0% — KEY DDx vs COX10"},
        {"feature": "Anaemia",           "n": 0, "pct": "0% — KEY DDx vs COX10"},
    ]

    # Outcome stats
    n_alive  = sum(1 for p in cohort if p["outcome"].startswith("Alive"))
    n_dead   = n - n_alive
    mean_cox = round(sum(p["cox_pct"] for p in cohort) / n, 1)
    min_cox  = round(min(p["cox_pct"] for p in cohort), 1)
    max_cox  = round(max(p["cox_pct"] for p in cohort), 1)
    mean_lac = round(sum(p["lactate"] for p in cohort) / n, 1)

    patient_rows = [
        {
            "id":         p["id"],
            "sex":        p["sex"],
            "genotype":   p["geno"].split(" —")[0].split(" (")[0],
            "onset_mo":   f"{p['onset_mo']} mo",
            "lactate":    f"{p['lactate']} mmol/L",
            "cox_pct":    f"{p['cox_pct']}%",
            "hcm":        "YES" if "HCM" in p["features"] else "No",
            "liver":      "YES" if "Hepatic" in p["features"] else "No",
            "outcome":    p["outcome"],
        }
        for p in cohort[:20]
    ]

    return {
        "n":                  n,
        "seed":               SEED,
        "genotype_dist":      geno_dist,
        "feature_prevalence": features,
        "outcome_summary": {
            "alive":         n_alive,
            "deceased":      n_dead,
            "pct_alive":     f"{n_alive/n*100:.0f}%",
            "mean_cox_pct":  mean_cox,
            "cox_range":     f"{min_cox}–{max_cox}% of normal",
            "mean_lactate":  mean_lac,
        },
        "patient_rows":       patient_rows,
        "treatment_summary": (
            "All patients receive the mitochondrial cocktail: CoQ10/Ubiquinol (Level C), "
            "Riboflavin B2 (Level C), Thiamine B1 (Level C — MANDATORY empiric Leigh mimic), "
            "Biotin (Level C — MANDATORY empiric BTD/BTBGD). IV Dextrose GIR 6–8 perioperatively. "
            "Cardiac management: propranolol/atenolol for HCM rate control (avoid digoxin/positive "
            "inotropes). Liver: UDCA if hepatic involvement (35%). L-Carnitine for secondary "
            "deficiency. NIV for respiratory compromise. Copper histidinate: investigational, "
            "theoretical rationale as copper chaperone support analogous to SCO2."
        ),
    }


def get_definitions() -> dict[str, Any]:
    """COA6 definitions — DDx table, glossary, clinical notes, references."""
    return {
        "ddx_table": [
            {
                "gene":        "COA6",
                "locus":       "1q42.2",
                "disease":     "COXPD14",
                "hcm":         "90% (CARDINAL)",
                "liver":       "35% (KEY DDx vs SCO2)",
                "tubulopathy": "0% (KEY DDx)",
                "cox_defect":  "Isolated CIV (<10%)",
                "distinguisher": "HCM CARDINAL; liver 35% (unlike SCO2); copper chaperone pathway; 1q42.2",
            },
            {
                "gene":        "SCO2",
                "locus":       "22q13.33",
                "disease":     "CEMCOX3 (COXPD2)",
                "hcm":         "100% (CARDINAL — fatal neonatal)",
                "liver":       "0% (KEY DDx — absent in SCO2)",
                "tubulopathy": "0%",
                "cox_defect":  "Isolated CIV (<10%)",
                "distinguisher": "HCM 100%; liver ABSENT; universally fatal in year 1; 22q13.33; copper responsive",
            },
            {
                "gene":        "SCO1",
                "locus":       "17p13.1",
                "disease":     "CEMCOX2 (COXPD4)",
                "hcm":         "Rare (<5%)",
                "liver":       "100% (neonatal hepatic failure — CARDINAL)",
                "tubulopathy": "0%",
                "cox_defect":  "Isolated CIV (<10%)",
                "distinguisher": "Hepatic failure 100% neonatal — KEY DDx; copper responsive; 17p13.1",
            },
            {
                "gene":        "COX15",
                "locus":       "10q24.2",
                "disease":     "CEMCOX5 (COXPD5)",
                "hcm":         "78%",
                "liver":       "0%",
                "tubulopathy": "0%",
                "cox_defect":  "Isolated CIV (<10%)",
                "distinguisher": "HCM 78% + Leigh 70%; haem a pathway (not copper); 10q24.2; COA6 has liver 35%",
            },
            {
                "gene":        "SURF1",
                "locus":       "9q34.2",
                "disease":     "Leigh (COXPD1)",
                "hcm":         "~10%",
                "liver":       "0%",
                "tubulopathy": "0%",
                "cox_defect":  "Isolated CIV (<5%)",
                "distinguisher": "Leigh encephalopathy dominant; HCM 10%; most common Leigh COX cause; 9q34.2",
            },
            {
                "gene":        "COX10",
                "locus":       "17p12",
                "disease":     "CEMCOX6 (COXPD6)",
                "hcm":         "<5%",
                "liver":       "0%",
                "tubulopathy": "65% (Fanconi + anaemia 80%)",
                "cox_defect":  "Isolated CIV (<10%)",
                "distinguisher": "Renal tubulopathy 65% + anaemia 80% — KEY DDx; COA6 has neither",
            },
            {
                "gene":        "COX6B1",
                "locus":       "19q13.3",
                "disease":     "COXPD7",
                "hcm":         "0% (KEY DDx — absent)",
                "liver":       "0%",
                "tubulopathy": "0%",
                "cox_defect":  "Isolated CIV (<15%)",
                "distinguisher": "Encephalomyopathy dominant; NO HCM (unlike COA6 90%); structural subunit VIb1",
            },
        ],
        "glossary": [
            {"term": "COA6",          "definition": "Cytochrome c Oxidase Assembly Factor 6 — nuclear-encoded mitochondrial copper chaperone; 73aa, 8.3kDa; twin-CX9C motif; cooperates with SCO1/SCO2 to deliver copper to the CuA binuclear site of MT-CO2 (COX2); required for COX2 metalation and late-stage Complex IV assembly."},
            {"term": "COXPD14",       "definition": "Mitochondrial Complex IV Deficiency, Nuclear Type 14 — the OMIM disease designation for COA6-related cardiomyopathic COX deficiency (#614924)."},
            {"term": "Twin-CX9C motif","definition": "A conserved copper-coordination fold (Cys-X9-Cys) present in COA6 and several other IMS proteins (CHCHD family). The cysteines are essential for copper coordination and any missense disrupting them abolishes COA6 function."},
            {"term": "CuA site",      "definition": "Binuclear copper centre in MT-CO2 (COX2) — the electron entry point from reduced cytochrome c to Complex IV. COA6 (with SCO1/SCO2) loads this copper site. Without CuA, COX cannot accept electrons from cytochrome c."},
            {"term": "SCO1 / SCO2",   "definition": "Synthesis of Cytochrome c Oxidase 1 and 2 — the primary copper chaperones for COX2 CuA site; COA6 cooperates with this pathway. SCO1/SCO2 mutations cause overlapping but distinct phenotypes."},
            {"term": "HCM",           "definition": "Hypertrophic cardiomyopathy — pathological thickening of the ventricular wall; in COA6 it is caused by myocardial ATP failure from COX deficiency, not sarcomeric protein mutations. Positive inotropes are contraindicated."},
            {"term": "Isolated COX deficiency", "definition": "Complex IV (COX) activity below normal range; Complexes I, II, III normal — shared by COA6, SCO2, SCO1, COX15, SURF1, COX6B1. WES/WGS required to distinguish."},
            {"term": "PRIS",          "definition": "Propofol Infusion Syndrome — metabolic acidosis + rhabdomyolysis + cardiac arrest caused by propofol's direct Complex IV inhibition; risk dramatically amplified in COA6 where COX activity is already <10% and the heart is hypertrophied."},
            {"term": "Copper histidinate", "definition": "Investigational copper supplementation used in SCO2 deficiency; also tried in COA6 given its copper chaperone mechanism. Clinical evidence is very limited; theoretical rationale is that supplying exogenous copper may partially bypass COA6 deficiency."},
        ],
        "clinical_notes": [
            "COA6 should be suspected in any neonate/infant with HCM + isolated COX deficiency "
            "+ lactic acidosis, particularly when liver involvement is also present (distinguishes "
            "from SCO2 where liver is absent).",
            "Biochemically: COX activity <10% of normal in cardiac/skeletal muscle; COX/SDH "
            "histochemistry shows COX-deficient fibres; CI/CII/CIII are NORMAL. This pattern "
            "requires a COX-focused gene panel or WES/WGS — biochemistry alone cannot distinguish "
            "COA6 from SCO2, COX15, or SURF1.",
            "Cardiac management is distinct from standard HCM: avoid positive inotropes (digoxin, "
            "dobutamine) which increase myocardial O2 demand in an already ATP-depleted heart. "
            "Rate control with propranolol/atenolol. Paediatric cardiologist co-management is "
            "mandatory.",
            "Empiric treatment while awaiting genetic confirmation: thiamine + biotin mandatory "
            "(to exclude BTD/BTBGD — curable Leigh mimics). Stop VPA immediately if started.",
            "Anaesthesia: sevoflurane inhalational (NOT propofol); dexmedetomidine for "
            "procedural sedation; GIR 6–8 perioperative; avoid fasting >4h; "
            "detailed anaesthesia plan with paediatric cardiologist before any procedure.",
            "Copper histidinate is not standard of care but may be considered in refractory "
            "cases given the copper chaperone mechanism; discuss with metabolic disease specialist.",
        ],
        "references": [
            {
                "citation": "Lim SC et al. (2014). Am J Hum Genet 95(2):197–205.",
                "note":     "First description of COA6 pathogenic variants causing Complex IV deficiency with infantile cardiomyopathy (COXPD14).",
            },
            {
                "citation": "Ghosh A et al. (2014). Hum Mol Genet 23(16):4213–4228.",
                "note":     "Functional characterisation of COA6 as a copper chaperone; twin-CX9C motif; interaction with SCO1/SCO2.",
            },
            {
                "citation": "Stroud DA et al. (2015). J Mol Cell Biol 7(1):55–66.",
                "note":     "COA6 copper transfer mechanism; cooperativity with SCO2 for MT-CO2 CuA metalation.",
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note":     "Comprehensive review of mitochondrial disease epidemiology, clinical spectrum, and management.",
            },
            {
                "citation": "Baertling F et al. (2017). Orphanet J Rare Dis 12:29.",
                "note":     "Clinical phenotype review of Complex IV deficiency nuclear subtypes including COXPD14.",
            },
        ],
        "inheritance_detail": (
            "COA6 disease is autosomal recessive (AR). Both copies of the COA6 gene must carry "
            "pathogenic variants (biallelic) for disease to manifest. Carrier parents are "
            "clinically unaffected. Each pregnancy of two carrier parents carries a 25% risk of "
            "affected offspring. Prenatal diagnosis by molecular genetics is available."
        ),
        "management_summary": (
            "No disease-modifying therapy exists. Management is supportive + preventive: "
            "(1) Cardiac: propranolol/atenolol for HCM rate control; ECHO monitoring; "
            "AVOID positive inotropes (digoxin/dobutamine); paediatric cardiology co-management. "
            "(2) Mitochondrial cocktail: CoQ10, riboflavin, thiamine, biotin, L-carnitine. "
            "(3) Energy substrate support: GIR 6–8 perioperative; fasting absolutely prohibited. "
            "(4) Hepatic: UDCA 15–20 mg/kg/day if liver involvement (35%). "
            "(5) Anaesthesia: sevoflurane (NOT propofol); dexmedetomidine for sedation; "
            "detailed cardiac-aware anaesthesia plan mandatory. "
            "(6) Absolute CI: VPA, propofol, metformin, linezolid, chloramphenicol, KD. "
            "(7) Investigational: copper histidinate — rationale from copper chaperone mechanism; "
            "discuss with specialist. "
            "(8) Genetic counselling: 25% recurrence per pregnancy; prenatal diagnosis available."
        ),
    }
