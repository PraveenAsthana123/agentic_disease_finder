#!/usr/bin/env python3
"""COX6B1 — Encephalomyopathic Complex IV (COX) Deficiency, Nuclear Type 7 (COXPD7).

COX6B1 (Cytochrome c Oxidase Subunit 6B1) is a nuclear-encoded structural subunit
of cytochrome c oxidase (Complex IV), the terminal electron acceptor of the
mitochondrial respiratory chain.

  COX6B1 gene           OMIM *124977
  Disease (COXPD7)      OMIM #607266
  Complex IV (COX)      deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX6B1 / Complex IV structural subunit / assembly):
COX6B1 (Subunit VIb1, constitutive isoform) is a small ~10 kDa protein localised
to the IMS-exposed surface of the holoenzyme. It forms a critical dimer contact
between two COX monomers in the COX homodimer (supercomplex organisation) and
directly stabilises the incorporation of COX2 (MT-CO2) into the holocomplex.
  • COX6B1 interacts with COX2 (MT-CO2) and helps tether the COX homodimer,
    stabilising the supercomplex III2–IV2 configuration within the cristae membrane.
  • Loss of COX6B1: COX2 incorporation fails → Complex IV assembly is blocked at
    a late stage → COX holoenzyme cannot form → functional Complex IV severely
    reduced (<15% of normal in muscle, <20% in fibroblasts).
  • Isolated COX deficiency: Complexes I, II, III NORMAL — same biochemical
    fingerprint as SURF1, SCO2, SCO1, COX10, COX15 — WES required to distinguish.
  • Unlike SCO2 (HCM 100%) and COX15 (HCM 78%): COX6B1 disease has NO HCM
    — neurological features dominate, resembling SURF1 but with distinct onset/DDx.

KEY CLINICAL DIFFERENTIATOR vs. OTHER COX ASSEMBLY FACTOR DISEASES:
  • NO HCM         — KEY DDx vs. SCO2 (100%) and COX15 (78%)
  • NO Hepatopathy — KEY DDx vs. SCO1 (100% neonatal hepatic failure)
  • NO Tubulopathy — KEY DDx vs. COX10 (65% renal Fanconi tubulopathy / anaemia)
  • ISOLATED COX   — KEY DDx vs. LRPPRC (combined CI + CIV deficiency)
  • CHILDHOOD ONSET onset can be slightly later than SURF1 — median 4–18 months

MOLECULAR: Biallelic (AR) loss-of-function COX6B1 variants:
  — p.Trp38Ser (c.113G>C): Turkish founder; structural core disruption; most common
    reported allele (~35% of all alleles); severe LOF; disrupts hydrophobic core of
    the 4-helix bundle of subunit VIb1.
  — p.Arg11Pro (c.32G>C): N-terminal dimer interface; proline insertion breaks the
    alpha-helix required for homodimer contact; severe LOF (~20%).
  — p.Ser72Tyr (c.215C>A): surface charge disruption at COX2 interaction surface;
    moderate-severe, partial residual activity.
  — Truncating variants (nonsense/frameshift): severe, null alleles (~20%).
  — p.Ala75Val (c.224C>T): hydrophobic core packing; intermediate (~10%).
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 606
DISEASE_ID   = "cox6b1"
DISEASE_NAME = "COX6B1 Encephalomyopathic Complex IV Deficiency (COXPD7)"
GENE         = "COX6B1"
OMIM_GENE    = "*124977"
OMIM_DISEASE = "#607266"
CHROMOSOME   = "19q13.3"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile (median 4–18 months; range birth–3 years)"
COHORT_SIZE  = 40
COLOR        = "#1565c0"   # deep blue — Complex IV / COX / encephalomyopathic
LIGHT        = "#e3f2fd"


# Genotype pool (published + modelled variants)
GENO_TRP38SER_HOM  = "p.Trp38Ser homozygous (c.113G>C) — Turkish founder"
GENO_TRP38SER_CPX  = "p.Trp38Ser / p.Arg11Pro — compound heterozygous"
GENO_TRP38SER_NULL = "p.Trp38Ser / truncating — compound heterozygous"
GENO_SER72TYR_CPX  = "p.Ser72Tyr / truncating — compound heterozygous"
GENO_OTHER_CPX     = "Other biallelic missense — compound heterozygous"

GENO_POOL    = [GENO_TRP38SER_HOM, GENO_TRP38SER_CPX, GENO_TRP38SER_NULL,
                GENO_SER72TYR_CPX, GENO_OTHER_CPX]
GENO_WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.20]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX6B1 cohort (seed-606)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COX6B1 cohort (seed-606).

    All patients have isolated COX deficiency.
    Additional features stochastic per published/inferred frequencies.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"

        # Onset age (months of life)
        onset_mo = round(rng.uniform(1.0, 24.0), 1)

        # Lactate (mmol/L)
        is_null  = "truncating" in geno or "null" in geno
        if is_null:
            lactate  = round(rng.uniform(4.5, 16.0), 1)
            cox_pct  = round(rng.uniform(2.0, 10.0), 1)
        elif "Trp38Ser" in geno and "homozygous" in geno:
            lactate  = round(rng.uniform(4.0, 14.0), 1)
            cox_pct  = round(rng.uniform(3.0, 12.0), 1)
        else:
            lactate  = round(rng.uniform(3.0, 12.0), 1)
            cox_pct  = round(rng.uniform(5.0, 20.0), 1)

        # Features
        encephalo     = rng.random() < 0.90   # 90%
        lactic_ac     = rng.random() < 0.90   # 90%
        hypotonia     = rng.random() < 0.85   # 85%
        leigh_mri     = rng.random() < 0.70   # 70%
        myopathy      = rng.random() < 0.80   # 80%
        growth_fail   = rng.random() < 0.75   # 75%
        seizures      = rng.random() < 0.45   # 45%
        psychomot_reg = rng.random() < 0.85   # 85%
        resp_comp     = rng.random() < 0.60   # 60%
        optic_atrph   = rng.random() < 0.20   # 20%
        # Disease-defining negatives
        hcm           = False    # 0% — KEY DDx vs SCO2/COX15
        hepatopathy   = False    # 0% — KEY DDx vs SCO1
        tubulopathy   = False    # 0% — KEY DDx vs COX10

        # Outcome
        survival_mo = round(rng.uniform(4.0, 72.0), 1)
        if is_null:
            alive = rng.random() < 0.25
        else:
            alive = rng.random() < 0.40
        if alive:
            outcome = f"Alive {survival_mo:.1f} months (last follow-up)"
        else:
            outcome = f"Died at {survival_mo:.1f} months"

        # Treatments
        txs = ["CoQ10/Ubiquinol (Level C)"]
        txs.append("Riboflavin B2 (Level C)")
        txs.append("Thiamine B1 (Level C — MANDATORY empiric Leigh)")
        txs.append("Biotin (Level C — MANDATORY empiric BTD treatable mimic)")
        txs.append("IV Dextrose GIR 6–8 (crisis / perioperative / fasting)")
        if lactic_ac:  txs.append("L-Carnitine (Level C — secondary deficiency)")
        if seizures:   txs.append("LEV (preferred AED — renal excretion, no mito toxicity)")
        if resp_comp:  txs.append("NIV / mechanical ventilation (respiratory support)")
        if growth_fail: txs.append("Nutritional support / gastrostomy")

        # Alerts
        alerts_list = [
            "🚫 VPA ABSOLUTE CI — CoA sequestration / POLG inhibition / hepatotoxicity",
            "🚫 Metformin ABSOLUTE CI — Complex I inhibition → lactic crisis",
            "🚫 Linezolid ABSOLUTE CI — 23S rRNA inhibition → eliminates residual MT-CO1/CO2 synthesis",
            "🚫 Propofol ABSOLUTE CI — PRIS: Complex IV direct inhibition → catastrophic in COX6B1",
            "🚫 KD CONTRAINDICATED — beta-oxidation requires functional Complex IV",
            "🚫 Fasting DANGEROUS — glucose mandatory; GIR 6–8 perioperative",
        ]
        if seizures:
            alerts_list.append("⚠ Seizures: LEV ONLY — NO VPA (ABSOLUTE CI), NO phenobarbital")
        alerts = "; ".join(alerts_list)

        # Feature string
        feats = []
        if psychomot_reg: feats.append("Psychomotor regression")
        if encephalo:     feats.append("Encephalomyopathy")
        if hypotonia:     feats.append("Hypotonia")
        if lactic_ac:     feats.append("Lactic acidosis")
        if leigh_mri:     feats.append("Leigh-like MRI")
        if myopathy:      feats.append("Myopathy")
        if growth_fail:   feats.append("Growth failure")
        if seizures:      feats.append("Seizures")
        if resp_comp:     feats.append("Resp. compromise")
        if optic_atrph:   feats.append("Optic atrophy")
        feats.append("NO HCM (key DDx)")
        feats.append("NO Hepatopathy (key DDx)")
        feats.append("NO Tubulopathy (key DDx)")

        patients.append({
            "id":         f"COX6B1-{i:03d}",
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
    """COX6B1 overview — gene, disease identity, KPIs, contraindications."""
    rng    = _rng()
    cohort = _build_cohort(rng)
    n      = len(cohort)

    n_encephalo  = sum(1 for p in cohort if "Encephalo" in p["features"])
    n_lactic     = sum(1 for p in cohort if "Lactic" in p["features"])
    n_hypotonia  = sum(1 for p in cohort if "Hypotonia" in p["features"])
    n_leigh      = sum(1 for p in cohort if "Leigh" in p["features"])
    n_myopathy   = sum(1 for p in cohort if "Myopathy" in p["features"])
    n_growth     = sum(1 for p in cohort if "Growth" in p["features"])
    n_seizures   = sum(1 for p in cohort if "Seizures" in p["features"])
    n_pmr        = sum(1 for p in cohort if "Psychomotor" in p["features"])
    n_resp       = sum(1 for p in cohort if "Resp." in p["features"])
    n_alive      = sum(1 for p in cohort if p["outcome"].startswith("Alive"))
    mean_cox     = round(sum(p["cox_pct"] for p in cohort) / n, 1)
    mean_onset   = round(sum(p["onset_mo"] for p in cohort) / n, 1)
    mean_lact    = round(sum(p["lactate"] for p in cohort) / n, 1)

    return {
        "gene":         "COX6B1 (Cytochrome c Oxidase Subunit 6B1)",
        "protein":      "COX6B1-109aa-10.2kDa — Nuclear-encoded structural subunit VIb (constitutive isoform); IMS-exposed; stabilises COX homodimer by bridging COX2 (MT-CO2) at the dimer interface; required for late-stage Complex IV assembly",
        "disease":      "Encephalomyopathic Complex IV (COX) Deficiency, Nuclear Type 7 (COXPD7)",
        "omim_gene":    "*124977 (COX6B1)",
        "omim_disease": "#607266 (Mitochondrial Complex IV Deficiency, Nuclear Type 7; COXPD7)",
        "chromosome":   "19q13.3",
        "inheritance":  "AR (autosomal recessive biallelic) — 25% recurrence per pregnancy",
        "onset":        "Infantile to early childhood; median 4–18 months; range birth–3 years",
        "cohort":       f"{n} patients · seed-606 · COX6B1 biallelic",
        "mechanism": (
            "COX6B1 (Subunit VIb1, the constitutively expressed isoform) is a nuclear-encoded, "
            "~10 kDa protein that occupies the IMS-exposed peripheral surface of the COX holoenzyme. "
            "Its primary structural role is stabilising the COX homodimer configuration that forms "
            "the functional supercomplex (CIII₂–CIV₂) within cristae membranes. "
            "COX6B1 directly contacts COX2 (MT-CO2), the copper-containing catalytic subunit, "
            "and is required for its stable incorporation into the late-stage assembly intermediate. "
            "Without functional COX6B1: COX2 integration fails → Complex IV assembly arrests → "
            "COX holoenzyme cannot form → functional cytochrome c oxidase severely reduced "
            "(<15% of normal in skeletal muscle). "
            "Consequence: neurons and muscle — which rely on OXPHOS for the majority of their ATP "
            "— fail to sustain energy production → encephalomyopathy, psychomotor regression, "
            "Leigh-like neurodegeneration. "
            "Isolated COX deficiency: Complexes I, II, III are NORMAL (identical biochemical "
            "fingerprint to SURF1, SCO2, SCO1, COX10, COX15 — WES/WGS required to distinguish)."
        ),
        "differentiator_note": (
            "COX6B1 is clinically distinguished from other COX assembly factor diseases by: "
            "(1) NO HCM — absent in 100% of patients (KEY DDx vs SCO2-100% and COX15-78%); "
            "(2) NO hepatopathy — absent (KEY DDx vs SCO1-100% neonatal hepatic failure); "
            "(3) NO renal tubulopathy — absent (KEY DDx vs COX10-65% Fanconi syndrome); "
            "(4) ISOLATED COX deficiency — no combined CI+CIV (KEY DDx vs LRPPRC); "
            "(5) MILD-MODERATE anaemia absent — absent (KEY DDx vs COX10 anaemia 80%). "
            "The dominant phenotype is encephalomyopathy with Leigh-like MRI, "
            "resembling SURF1 but differentiated by locus (19q13.3 vs 9q34.2) on WES/WGS."
        ),
        "kpis": [
            {"label": "Psychomotor Regression",  "value": f"{n_pmr/n*100:.0f}%",        "color": COLOR},
            {"label": "Encephalomyopathy",        "value": f"{n_encephalo/n*100:.0f}%",  "color": COLOR},
            {"label": "Lactic Acidosis",          "value": f"{n_lactic/n*100:.0f}%",     "color": "#b71c1c"},
            {"label": "Hypotonia",                "value": f"{n_hypotonia/n*100:.0f}%",  "color": COLOR},
            {"label": "Leigh-like MRI",           "value": f"{n_leigh/n*100:.0f}%",      "color": COLOR},
            {"label": "Myopathy",                 "value": f"{n_myopathy/n*100:.0f}%",   "color": COLOR},
            {"label": "Growth Failure",           "value": f"{n_growth/n*100:.0f}%",     "color": COLOR},
            {"label": "Seizures",                 "value": f"{n_seizures/n*100:.0f}%",   "color": "#6a1b9a"},
            {"label": "Resp. Compromise",         "value": f"{n_resp/n*100:.0f}%",       "color": "#b71c1c"},
            {"label": "HCM",                      "value": "0% (KEY DDx)",               "color": "#2e7d32"},
            {"label": "Hepatopathy",              "value": "0% (KEY DDx)",               "color": "#2e7d32"},
            {"label": "Mean COX Activity",        "value": f"{mean_cox}% normal",        "color": COLOR},
            {"label": "Mean Onset",               "value": f"{mean_onset} months",       "color": COLOR},
            {"label": "Mean Lactate",             "value": f"{mean_lact} mmol/L",        "color": "#b71c1c"},
            {"label": "Alive (last f/u)",         "value": f"{n_alive/n*100:.0f}%",      "color": "#2e7d32"},
        ],
        "contraindications": [
            {
                "drug":      "VPA / Valproate",
                "severity":  "ABSOLUTE CI — ALL COX6B1 / Complex IV disease",
                "mechanism": (
                    "Triple mechanism: (a) CoA sequestration → depletes free CoA → impairs TCA "
                    "cycle, worsening OXPHOS failure in neurons and skeletal muscle; "
                    "(b) POLG1 inhibition → mtDNA depletion → reduces residual MT-CO1/CO2 "
                    "synthesis, collapsing any remaining COX assembly; (c) Idiosyncratic VPA "
                    "hepatotoxicity — even without baseline hepatopathy (COX6B1 has no liver "
                    "disease at baseline, but VPA can still trigger hepatic failure). "
                    "NEVER use VPA. Use LEV (IV loading, renal excretion, no mito toxicity)."
                ),
            },
            {
                "drug":      "Metformin",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease including COX6B1",
                "mechanism": (
                    "Metformin inhibits Complex I. In COX6B1 (COX/Complex IV deficient), "
                    "upstream electron carriers are already backed up. Adding Complex I inhibition "
                    "collapses the entire respiratory chain → lactic crisis (L:P ratio >30). "
                    "Particularly dangerous during intercurrent illness when energy demand surges."
                ),
            },
            {
                "drug":      "Propofol",
                "severity":  "ABSOLUTE CI — PRIS risk in Complex IV deficiency",
                "mechanism": (
                    "Propofol inhibits Complex IV (COX) directly. In COX6B1 with pre-existing "
                    "severe COX deficiency: propofol causes near-complete COX abolition → "
                    "ATP collapse in neurons and myocytes → metabolic acidosis, rhabdomyolysis, "
                    "cardiac arrest (PRIS triad). Even in the absence of HCM (unlike SCO2), "
                    "the residual COX activity in neurons and muscle is critically low and "
                    "propofol abolishes it. Use sevoflurane (inhalational) for induction/maintenance; "
                    "dexmedetomidine for sedation."
                ),
            },
            {
                "drug":      "Linezolid",
                "severity":  "ABSOLUTE CI — mitochondrial ribosome inhibition",
                "mechanism": (
                    "Linezolid inhibits the mitochondrial 23S rRNA (mt-LSU), blocking translation "
                    "of all 13 mitochondrially encoded OXPHOS subunits, including MT-CO1, MT-CO2, "
                    "and MT-CO3 (the 3 catalytic COX subunits). In COX6B1-deficient patients who "
                    "depend on any residual Complex IV assembly: eliminating MT-CO1/CO2 synthesis "
                    "is catastrophic. Residual COX activity collapses to near zero. "
                    "Use alternative antibiotics (aminoglycosides with hearing monitoring, "
                    "or beta-lactams/carbapenems as appropriate)."
                ),
            },
            {
                "drug":      "Ketogenic Diet",
                "severity":  "CONTRAINDICATED — beta-oxidation requires functional Complex IV",
                "mechanism": (
                    "The ketogenic diet (KD) forces beta-oxidation of fatty acids as the primary "
                    "fuel, generating acetyl-CoA and FADH2/NADH that feed directly into the "
                    "respiratory chain. In Complex IV deficiency, OXPHOS is already severely "
                    "impaired — forcing greater reliance on OXPHOS via KD worsens energy failure "
                    "and may precipitate metabolic crisis. KD is absolutely contraindicated. "
                    "Maintain adequate carbohydrate intake; GIR 6–8 perioperatively."
                ),
            },
            {
                "drug":      "Chloramphenicol",
                "severity":  "ABSOLUTE CI — mitochondrial ribosome inhibition",
                "mechanism": (
                    "Chloramphenicol inhibits the mitochondrial 70S ribosome (same target as "
                    "linezolid), blocking all 13 mt-encoded OXPHOS subunit translations. "
                    "In COX6B1, this eliminates MT-CO1/CO2/CO3 — the catalytic core of Complex IV — "
                    "abolishing any residual COX assembly. Absolute prohibition."
                ),
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    """COX6B1 breakdown — 40-patient cohort table, genotype distribution, feature prevalence."""
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
        ("Psychomotor regression", "Psychomotor"),
        ("Encephalomyopathy",      "Encephalo"),
        ("Lactic acidosis",        "Lactic"),
        ("Hypotonia",              "Hypotonia"),
        ("Leigh-like MRI",         "Leigh"),
        ("Myopathy",               "Myopathy"),
        ("Growth failure",         "Growth"),
        ("Seizures",               "Seizures"),
        ("Resp. compromise",       "Resp."),
        ("Optic atrophy",          "Optic"),
    ]
    features = [
        {"feature": label, "n": sum(1 for p in cohort if key in p["features"]),
         "pct": f"{sum(1 for p in cohort if key in p['features'])/n*100:.0f}%"}
        for label, key in feat_map
    ]
    # Add "absent" features (key DDx)
    features += [
        {"feature": "HCM (hypertrophic CMP)",        "n": 0, "pct": "0% — KEY DDx vs SCO2/COX15"},
        {"feature": "Hepatopathy",                    "n": 0, "pct": "0% — KEY DDx vs SCO1"},
        {"feature": "Renal tubulopathy",              "n": 0, "pct": "0% — KEY DDx vs COX10"},
    ]

    # Outcome stats
    n_alive  = sum(1 for p in cohort if p["outcome"].startswith("Alive"))
    n_dead   = n - n_alive
    mean_cox = round(sum(p["cox_pct"] for p in cohort) / n, 1)
    min_cox  = round(min(p["cox_pct"] for p in cohort), 1)
    max_cox  = round(max(p["cox_pct"] for p in cohort), 1)
    mean_lac = round(sum(p["lactate"] for p in cohort) / n, 1)

    # Patient table (first 20 shown)
    patient_rows = [
        {
            "id":         p["id"],
            "sex":        p["sex"],
            "genotype":   p["geno"].split(" —")[0].split(" (")[0],
            "onset_mo":   f"{p['onset_mo']} mo",
            "lactate":    f"{p['lactate']} mmol/L",
            "cox_pct":    f"{p['cox_pct']}%",
            "features":   p["features"],
            "outcome":    p["outcome"],
        }
        for p in cohort[:20]
    ]

    return {
        "n":                 n,
        "seed":              SEED,
        "genotype_dist":     geno_dist,
        "feature_prevalence": features,
        "outcome_summary": {
            "alive":          n_alive,
            "deceased":       n_dead,
            "pct_alive":      f"{n_alive/n*100:.0f}%",
            "mean_cox_pct":   mean_cox,
            "cox_range":      f"{min_cox}–{max_cox}% of normal",
            "mean_lactate":   mean_lac,
        },
        "patient_rows":      patient_rows,
        "treatment_summary": (
            "All patients receive the mitochondrial 'cocktail': CoQ10/Ubiquinol (Level C), "
            "Riboflavin B2 (Level C), Thiamine B1 (Level C — MANDATORY empiric Leigh mimic), "
            "Biotin (Level C — MANDATORY empiric BTD treatable mimic). "
            "IV Dextrose GIR 6–8 perioperatively / fasting is prohibited. "
            "L-Carnitine for secondary deficiency. LEV first-line for seizures. "
            "NIV/ventilation for respiratory compromise. Nutritional support/PEG for "
            "growth failure. Aggressive fever management to prevent crisis."
        ),
    }


def get_definitions() -> dict[str, Any]:
    """COX6B1 definitions — DDx table, glossary, clinical notes, references."""
    return {
        "ddx_table": [
            {
                "gene":       "COX6B1",
                "locus":      "19q13.3",
                "disease":    "COXPD7",
                "hcm":        "0% (KEY DDx)",
                "hepatopathy":"0% (KEY DDx)",
                "tubulopathy":"0% (KEY DDx)",
                "cox_defect": "Isolated CIV (<15%)",
                "distinguisher": "Encephalomyopathy dominant; NO cardiac/hepatic/renal; WES needed vs SURF1",
            },
            {
                "gene":       "SURF1",
                "locus":      "9q34.2",
                "disease":    "Leigh (COXPD1)",
                "hcm":        "~10%",
                "hepatopathy":"0%",
                "tubulopathy":"0%",
                "cox_defect": "Isolated CIV (<5%)",
                "distinguisher": "Most common COX/Leigh (~10–15% all Leigh); identical biochemical → WES",
            },
            {
                "gene":       "SCO2",
                "locus":      "22q13.33",
                "disease":    "CEMCOX3 (COXPD2)",
                "hcm":        "100% (CARDINAL)",
                "hepatopathy":"0%",
                "tubulopathy":"0%",
                "cox_defect": "Isolated CIV (<15%)",
                "distinguisher": "HCM 100% — if neonatal HCM present, SCO2 first DDx; fatal in year 1",
            },
            {
                "gene":       "SCO1",
                "locus":      "17p13.1",
                "disease":    "CEMCOX2 (COXPD4)",
                "hcm":        "Rare",
                "hepatopathy":"100% (CARDINAL — neonatal)",
                "tubulopathy":"0%",
                "cox_defect": "Isolated CIV (<10%)",
                "distinguisher": "Hepatopathy 100% neonatal — if neonatal hepatic failure present, SCO1 first DDx",
            },
            {
                "gene":       "COX10",
                "locus":      "17p12",
                "disease":    "CEMCOX6 (COXPD6)",
                "hcm":        "<5%",
                "hepatopathy":"0%",
                "tubulopathy":"65% (Fanconi + anaemia)",
                "cox_defect": "Isolated CIV (<10%)",
                "distinguisher": "Renal tubulopathy 65% + anaemia 80% — KEY DDx; COX6B1 has neither",
            },
            {
                "gene":       "COX15",
                "locus":      "10q24.2",
                "disease":    "CEMCOX5 (COXPD5)",
                "hcm":        "78%",
                "hepatopathy":"0%",
                "tubulopathy":"0%",
                "cox_defect": "Isolated CIV (<10%)",
                "distinguisher": "HCM 78% — if HCM present, COX15 before COX6B1; COX6B1 has NO HCM",
            },
            {
                "gene":       "TACO1",
                "locus":      "17q23.3",
                "disease":    "Leigh (COXPD9)",
                "hcm":        "0%",
                "hepatopathy":"0%",
                "tubulopathy":"0%",
                "cox_defect": "Isolated CIV",
                "distinguisher": "Childhood onset; dysarthria dominant; Inuit founder; later onset than COX6B1",
            },
            {
                "gene":       "LRPPRC",
                "locus":      "2p21",
                "disease":    "LSFC (Leigh Syndrome French-Canadian)",
                "hcm":        "0%",
                "hepatopathy":"Mild-moderate 45%",
                "tubulopathy":"0%",
                "cox_defect": "Combined CI + CIV (KEY DDx — not isolated)",
                "distinguisher": "COMBINED CI+CIV deficiency; episodic metabolic crises; French-Canadian founder",
            },
        ],
        "glossary": [
            {"term": "COX6B1",       "definition": "Cytochrome c Oxidase Subunit 6B1 — nuclear-encoded structural subunit VIb (constitutive isoform); 109 aa, 10.2 kDa; IMS-exposed; required for COX homodimer stabilisation and late-stage Complex IV assembly at the COX2 (MT-CO2) interface."},
            {"term": "COXPD7",       "definition": "Mitochondrial Complex IV Deficiency, Nuclear Type 7 — the OMIM disease designation for COX6B1-related encephalomyopathic COX deficiency."},
            {"term": "Complex IV",   "definition": "Cytochrome c oxidase — the terminal enzyme of the mitochondrial respiratory chain; transfers electrons from reduced cytochrome c to molecular oxygen; drives proton pumping for ATP synthesis."},
            {"term": "Subunit VIb",  "definition": "COX6B1 is the VIb1 (constitutive) isoform of the VIb subunit family. A VIb2 (heart/skeletal-muscle-specific) isoform exists but is not disease-causing for COXPD7."},
            {"term": "COX homodimer","definition": "Two COX monomers associate into a homodimer within the cristae membrane. COX6B1 bridges the dimer interface at the COX2 surface, stabilising the homodimeric CIII₂–CIV₂ supercomplex."},
            {"term": "Leigh syndrome","definition": "Subacute necrotising encephalomyopathy; bilateral symmetric signal change in basal ganglia and brainstem on MRI; elevated CSF/blood lactate; caused by many OXPHOS defects including CIV deficiency."},
            {"term": "Isolated COX deficiency", "definition": "Complex IV (COX) activity below normal range; Complexes I, II, III normal — the biochemical fingerprint shared by SURF1, SCO2, SCO1, COX10, COX15, TACO1, and COX6B1; WES/WGS required to distinguish."},
            {"term": "PRIS",         "definition": "Propofol Infusion Syndrome — metabolic acidosis, rhabdomyolysis, cardiac arrhythmia caused by propofol's direct inhibition of Complex IV (COX); risk dramatically amplified in pre-existing COX deficiency."},
            {"term": "L:P ratio",    "definition": "Lactate:Pyruvate ratio — >20 indicates cytoplasmic NADH accumulation consistent with OXPHOS failure; L:P >30 is critical."},
            {"term": "GIR",          "definition": "Glucose Infusion Rate — mg/kg/min IV dextrose to prevent catabolic fasting state in mitochondrial disease; target GIR 6–8 perioperatively."},
        ],
        "clinical_notes": [
            "COX6B1 should be suspected in any child with isolated COX deficiency, Leigh-like MRI, "
            "encephalomyopathy, and absence of HCM, hepatopathy, and renal tubulopathy.",
            "The Turkish founder variant p.Trp38Ser (c.113G>C) accounts for ~35% of alleles in "
            "published cases — homozygosity is common in consanguineous Turkish families.",
            "Biochemically: COX activity <15% of normal in muscle biopsy; COX staining (COX/SDH "
            "histochemistry) shows COX-deficient fibres; CI/CII/CIII are NORMAL — this pattern "
            "mandates a COX-focused gene panel or WES/WGS.",
            "Empiric treatment while awaiting genetic confirmation: thiamine + biotin mandatory "
            "(to exclude treatable mimics: biotinidase deficiency, BTBGD); stop VPA immediately "
            "if started for seizures.",
            "Anaesthesia protocol: sevoflurane inhalational (NOT propofol); dexmedetomidine for "
            "sedation; avoid all mito-toxic agents; GIR 6–8 perioperative; avoid fasting >4 h.",
        ],
        "references": [
            {
                "citation": "Massa V et al. (2008). Am J Hum Genet 82:1281–1289.",
                "note":     "First description of COX6B1 pathogenic variants causing isolated Complex IV deficiency with encephalomyopathy.",
            },
            {
                "citation": "Invernizzi F et al. (2012). Mol Genet Metab 106:178–183.",
                "note":     "Characterisation of additional COX6B1 variants; phenotype expansion.",
            },
            {
                "citation": "Rak M et al. (2016). Biochim Biophys Acta 1857:1158–1164.",
                "note":     "Structural role of COX6B1 in COX homodimer formation and supercomplex assembly.",
            },
            {
                "citation": "Zong S et al. (2018). Science 362:eaat9178.",
                "note":     "Cryo-EM structure of mammalian Complex IV; COX6B1 position at IMS-exposed dimer interface.",
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note":     "Comprehensive review of mitochondrial disease epidemiology, clinical spectrum, and management.",
            },
        ],
        "inheritance_detail": (
            "COX6B1 disease is autosomal recessive (AR). Both copies of the COX6B1 gene must carry "
            "pathogenic variants (biallelic) for disease to manifest. Carrier parents (one normal + "
            "one pathogenic allele) are clinically unaffected. Each pregnancy of two carrier parents "
            "carries a 25% risk of affected offspring."
        ),
        "management_summary": (
            "No disease-modifying therapy exists. Management is supportive + preventive: "
            "(1) Mitochondrial cocktail: CoQ10, riboflavin, thiamine, biotin, L-carnitine. "
            "(2) Energy substrate support: GIR 6–8 perioperative; fasting prohibited. "
            "(3) Seizures: LEV first-line (renal excretion, no mito toxicity); VPA ABSOLUTELY CONTRAINDICATED. "
            "(4) Respiratory: NIV/mechanical ventilation for respiratory compromise. "
            "(5) Nutrition: PEG/gastrostomy for growth failure + dysphagia. "
            "(6) Anaesthesia: sevoflurane or dexmedetomidine; avoid propofol, linezolid, metformin, VPA. "
            "(7) Fever management: aggressive cooling + GIR augmentation to prevent metabolic crisis. "
            "(8) Genetic counselling: 25% recurrence per pregnancy; prenatal diagnosis available."
        ),
    }
