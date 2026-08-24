#!/usr/bin/env python3
"""PCCA (Propionyl-CoA Carboxylase Alpha Subunit / Propionic Acidemia Type A) Epilepsy Dashboard.

PCCA encodes the alpha subunit of propionyl-CoA carboxylase (PCC), a biotin-dependent
mitochondrial enzyme that carboxylates propionyl-CoA to D-methylmalonyl-CoA.

  PROPIONYL-COA CARBOXYLASE (PCC) COMPLEX:
    PCCA (alpha, 728 aa): Biotin Carboxylase (BC) domain + Biotin Carboxyl Carrier Protein
      (BCCP) domain. Carries biotin via Lys669 (BCCP swinging arm). BC domain performs
      ATP-dependent carboxylation of biotin using HCO3-.
    PCCB (beta, 559 aa): Carboxyl Transferase (CT) domain — accepts activated CO2 from
      biotinyl-PCCA-Lys669 and transfers to propionyl-CoA → D-methylmalonyl-CoA.
    Complex structure: (αβ)6 dodecamer in mitochondrial matrix.
      6 PCCA + 6 PCCB in alternating α6β6 cubic arrangement.
    HLCS biotinylates PCCA at Lys669 — PCC REQUIRES biotin to function.

  REACTION CATALYZED (two-step):
    Step 1 (BC domain, PCCA): HCO3- + ATP + biotin-PCCA → carboxybiotin-PCCA + ADP + Pi
    Step 2 (CT domain, PCCB): Carboxybiotin-PCCA + propionyl-CoA →
       D-methylmalonyl-CoA + biotin-PCCA
    Net: Propionyl-CoA + HCO3- + ATP → D-methylmalonyl-CoA + ADP + Pi

  PROPIONYL-COA SOURCES (what accumulates when PCC fails):
    1. Odd-chain fatty acids: β-oxidation terminal step → propionyl-CoA (C3)
    2. Isoleucine catabolism: 2-methylbutyryl-CoA → propionyl-CoA (+ acetyl-CoA)
    3. Valine catabolism: isobutyryl-CoA → propionyl-CoA (late steps)
    4. Threonine catabolism: threonine → propionyl-CoA (via propionaldehyde)
    5. Methionine catabolism: succinyl-CoA via propionyl-CoA (trans-sulfuration)
    6. Gut bacteria: colonic bacteria produce propionate → portal absorption → liver propionyl-CoA

  PCCA LOF → PROPIONYL-COA ACCUMULATES → FOUR SECONDARY METABOLIC CASCADES:
    1. Methylcitrate pathway: propionyl-CoA + OAA → methylcitrate (by aconitase) — PATHOGNOMONIC
    2. Propionylglycine: propionyl-CoA + glycine → propionylglycine (urine marker)
    3. 3-Hydroxypropionate: propionyl-CoA → acrylyl-CoA → 3-hydroxypropionate
    4. CPS1 inhibition: propionyl-CoA → reduced NAG (NAGS inhibited competitively) → CPS1 ↓ → NH3 ↑

  KEY BIOMARKERS:
    NBS PRIMARY: C3 (propionylcarnitine) ELEVATED — trigger for PA workup.
       C3 normal < 3.5 µmol/L; PA range 5–40 µmol/L.
    C3/C2 ratio: elevated (> 0.25 suggests PA/MMA)
    Urine organic acids (PATHOGNOMONIC):
       Methylcitrate: 50–2,000 µmol/mmolCr (PATH for PA; absent in MMA)
       3-Hydroxypropionate: 100–1,500 µmol/mmolCr
       Propionylglycine: 50–800 µmol/mmolCr
    ABSENT: Methylmalonate in urine — KEY NEGATIVE vs MMA (most critical differential)
    Plasma ammonia: 150–2,000 µmol/L (acute crisis); secondary hyperammonemia via CPS1 block
    Blood glucose: hypoglycemia (gluconeogenesis impaired via OAA depletion, methylcitrate competes)
    Biotinidase activity: NORMAL (PA is NOT a biotin recycling disorder)
    Plasma biotin: NORMAL (PA is NOT a biotin depletion disorder)
    C5-OH acylcarnitine: ABSENT/NORMAL — KEY NEGATIVE vs HLCS/BTD (no MCC block in isolated PA)
    BCAA (Leu, Ile, Val): NORMAL — KEY NEGATIVE vs DLD/MSUD

  DOWNSTREAM COMPLICATIONS:
    Cardiomyopathy (dilated): 30-50% — propionyl-CoA inhibits mitochondrial respiratory chain;
      direct propionylation of cardiac proteins; CoA sequestration
    QT prolongation / arrhythmia: 5-10% — propionyl-CoA disrupts cardiac ion channels;
      fatality risk during acute decompensation
    Basal ganglia stroke-like episodes: 25-35% — acute metabolic decompensation → BG infarct
    Bone marrow suppression: neutropenia, thrombocytopenia — propionyl-CoA inhibits
      mitochondrial enzyme function in rapidly dividing cells; thymidylate synthase inhibition
    Optic atrophy: rare; optic nerve mitochondrial dysfunction
    Pancreatitis: acute; mechanism unclear but documented
    Renal tubular dysfunction: in severe chronic PA

  TREATMENT:
    Natural protein restriction (Ile, Val, Met, Thr sources restricted): LEVEL A
    PCCA/PCCB-free amino acid supplement formula: LEVEL A
    L-Carnitine: LEVEL A — conjugates propionyl-CoA → C3 (propionylcarnitine) → renal excretion;
      prevents secondary carnitine depletion (PA is a major secondary carnitine depletion cause)
    IV glucose + insulin (acute): LEVEL A — anabolic state halts catabolism, propionyl-CoA production ↓
    Ammonia scavengers (Na benzoate / Na phenylacetate): LEVEL A in acute hyperammonemia
    Metronidazole (intermittent): LEVEL B — reduces colonic bacterial propionate production;
      cycles of 5-10 days/month reduce PA decompensations
    Liver transplantation: LEVEL B — corrects ~75% of hepatic PCC activity; significantly reduces
      acute decompensations; does NOT prevent cardiac complications (extra-hepatic PCC important)
    Biotin: NOT effective — PCCA enzyme itself is mutant; biotin supplementation cannot compensate
      (unlike HLCS where ligation is impaired, or BTD where recycling is impaired)
    VPA (valproate): ABSOLUTE CI — VPA inhibits PCC directly (valproyl-CoA = PCC competitive
      inhibitor); VPA also depletes free carnitine; FATAL in PA crisis; equivalent danger to PA
      as it poses to MSUD patients
    Fasting: EXTREME HAZARD — catabolism of protein (Ile, Val, Met, Thr) and odd-chain FA
      → propionyl-CoA surge → acute decompensation; ALWAYS provide glucose in intercurrent illness
    High protein / high odd-chain fat intake: EXTREME HAZARD

AUTOSOMAL RECESSIVE:
  PCCA gene: 13q32.3; Autosomal Recessive (AR)
  Protein: 728 aa; BC domain (1-435) + BCCP domain (436-728); Lys669 = biotin attachment site
  OMIM gene: *232000; disease: #606054 (PROPIONIC ACIDEMIA)
  Prevalence: 1:100,000-150,000 (European); 1:2,000-5,000 (Arabian Peninsula — Saudi, Qatar)
  Both PCCA and PCCB cause biochemically identical PA — gene panel always tests both.
  ~50% of PA cases due to PCCA mutations; ~50% due to PCCB mutations.

KEY VARIANTS IN PCCA:
  p.Arg410Trp (c.1228C>T): BC domain; European common; classic severe neonatal; abolishes
    ATP-dependent bicarbonate activation at critical Arg in BC domain active-site
  p.Arg306His (c.917G>A): BC domain; moderate; some residual activity; intermediate phenotype
  p.Gly549Asp (c.1646G>A): BC domain C-terminal; European severe; misfolding at subunit interface
  p.Ala178Pro (c.532G>C): BC domain; European severe; destabilizes alpha-helix in BC domain
  p.Lys669fsX (c.2005del): frameshift destroys biotin attachment Lys669; BCCP non-functional;
    most severe — no biotin binding possible; usually compound heterozygous with missense
  c.IVS20+2T>C (intron 20 splice donor): null; neonatal severe (Arabian Peninsula founder)
  p.Gly131Arg (c.391G>A): near BC active site; European intermediate
  p.Ile204Thr (c.611T>C): BC domain partial function; mild-to-intermediate

CLINICAL SPECTRUM:
  1. Classic neonatal severe (70%): onset day 1-5 of life; explosive hyperammonemia (>500 µmol/L),
     ketoacidosis, encephalopathy, seizures; NBS C3 triggers immediate evaluation.
  2. Late-onset / episodic (20%): onset 3-36 months; triggered by illness, fasting, high-protein;
     metabolic decompensation with vomiting, lethargy, hyperammonemia; basal ganglia infarct risk.
  3. Intermediate (5%): variable severity between neonatal and episodic.
  4. Paucisymptomatic / NBS-detected (5%): identified by NBS; mild or asymptomatic on treatment.

NBS: C3 (propionylcarnitine) elevated → immediate PA/MMA workup; PA/MMA together ~1:30,000 NBS positive.
"""

import random

_SEED = 47
_RNG  = random.Random(_SEED)

# Phenotypic classes
_PHENOTYPES = [
    ("Classic-Neonatal-Severe",  28),  # 70%
    ("Late-Onset-Episodic",       8),  # 20%
    ("Intermediate",              2),  # 5%
    ("Paucisymptomatic-NBS",      2),  # 5%
]

# Key pathogenic variants
_VARIANTS = [
    ("p.Arg410Trp",    "c.1228C>T",    "BC-domain active site",      "Severe",       "European common; abolishes ATP-bicarbonate activation in BC domain"),
    ("p.Arg306His",    "c.917G>A",     "BC-domain",                  "Moderate",     "Some residual activity; intermediate phenotype; European"),
    ("p.Gly549Asp",    "c.1646G>A",    "BC-domain C-terminal",       "Severe",       "European; subunit interface misfolding; classic neonatal"),
    ("p.Ala178Pro",    "c.532G>C",     "BC-domain alpha-helix",      "Severe",       "European severe; helix destabilization"),
    ("p.Lys669fsX",    "c.2005del",    "BCCP (biotin-Lys669)",       "Severe",       "BCCP non-functional; Lys669 biotin-binding site destroyed; null"),
    ("c.IVS20+2T>C",   "splice donor", "Splice null",                "Severe",       "Arabian Peninsula founder; null allele; neonatal"),
    ("p.Gly131Arg",    "c.391G>A",     "BC-domain near active site", "Severe",       "European intermediate to severe"),
    ("p.Ile204Thr",    "c.611T>C",     "BC-domain",                  "Intermediate", "Partial function; mild-to-intermediate phenotype"),
]

_HIGH_RISK_DRUGS = [
    ("VPA (valproate)",                "ABSOLUTE CI",   "Valproyl-CoA directly inhibits PCC; carnitine depletion; FATAL in PA decompensation — same danger as in MSUD"),
    ("High-protein diet / catabolism", "EXTREME HAZARD","Ile/Val/Met/Thr catabolism floods propionyl-CoA pathway; strict low-protein diet mandatory; illness = danger"),
    ("Fasting",                        "EXTREME HAZARD","Protein catabolism (Ile/Val/Met/Thr) + odd-chain FA oxidation → propionyl-CoA surge; ALWAYS give IV glucose"),
    ("High odd-chain fat intake",      "EXTREME HAZARD","Odd-chain fatty acid β-oxidation terminal step = propionyl-CoA; avoid odd-chain FA food sources in PA"),
    ("Surgery / GA without cover",     "HAZARD",        "Perioperative catabolism + NPO fasting → propionyl-CoA surge; mandatory IV glucose + protein-free AA infusion"),
]


# ------------------------------------------------------------------
# PATIENT COHORT GENERATOR (40 patients)
# ------------------------------------------------------------------
def _make_cohort():
    patients = []
    pid = 1
    phenotype_list = []
    for phenotype, count in _PHENOTYPES:
        phenotype_list.extend([phenotype] * count)

    for i, phenotype in enumerate(phenotype_list):
        rng = _RNG
        sex = "M" if rng.random() < 0.50 else "F"   # PA 50:50
        is_classic   = "Classic" in phenotype
        is_late      = "Late" in phenotype
        is_paucis    = "Pauci" in phenotype

        onset_age = (
            rng.uniform(0.5, 5.0)   if is_classic else
            rng.uniform(5.0, 36.0)  if is_late else
            rng.uniform(0.5, 36.0)  if not is_paucis else
            0.0   # NBS detected at birth
        )  # days for classic → convert: use months for consistent display; classic: ~day 1-5 → 0.03-0.17 months
        if is_classic:
            onset_age = rng.uniform(0.03, 0.17)   # ~1-5 days in months

        # Biomarkers
        c3_umol_l = (
            rng.uniform(15, 40)  if is_classic else
            rng.uniform(5, 20)   if is_late else
            rng.uniform(5, 15)   if not is_paucis else
            rng.uniform(3.5, 8)
        )
        c3_c2_ratio = round(c3_umol_l / rng.uniform(20, 60), 3)
        methylcitrate = (
            rng.uniform(400, 2000) if is_classic else
            rng.uniform(80, 600)   if is_late else
            rng.uniform(50, 300)   if not is_paucis else
            rng.uniform(20, 80)
        )  # µmol/mmolCr
        three_oh_prop = (
            rng.uniform(300, 1500) if is_classic else
            rng.uniform(50, 400)   if is_late else
            rng.uniform(30, 200)   if not is_paucis else
            rng.uniform(10, 60)
        )  # µmol/mmolCr
        propionylgly = (
            rng.uniform(100, 800)  if is_classic else
            rng.uniform(30, 200)   if is_late else
            rng.uniform(20, 100)   if not is_paucis else
            rng.uniform(5, 40)
        )  # µmol/mmolCr
        ammonia = (
            rng.uniform(500, 2000) if is_classic else
            rng.uniform(100, 600)  if is_late else
            rng.uniform(50, 300)   if not is_paucis else
            rng.uniform(20, 80)
        )  # µmol/L (normal <80; critically elevated >200)

        # Carnitine (secondary depletion)
        free_carnitine = (
            rng.uniform(3, 15)    if is_classic else
            rng.uniform(8, 25)    if is_late else
            rng.uniform(15, 35)   if not is_paucis else
            rng.uniform(20, 45)
        )  # µmol/L (normal 25-60)

        # Clinical features
        seizures      = rng.random() < (0.55 if is_classic else 0.35 if is_late else 0.20 if not is_paucis else 0.05)
        cardiomyopathy= rng.random() < (0.35 if not is_paucis else 0.05)
        qt_prolonged  = rng.random() < (0.08 if is_classic else 0.05)
        bg_infarct    = rng.random() < (0.30 if is_late else 0.15 if is_classic else 0.05)
        neutropenia   = rng.random() < (0.40 if is_classic else 0.20)
        nbs_detected  = rng.random() < 0.90   # 90% NBS detection rate
        pancreatitis  = rng.random() < 0.10
        hypoglycemia  = rng.random() < (0.60 if is_classic else 0.30 if is_late else 0.10)
        liver_tx_done = rng.random() < 0.15   # 15% received liver transplant
        on_metronidazole = rng.random() < 0.45

        # Variant
        v = _VARIANTS[i % len(_VARIANTS)]
        # Compound heterozygous pattern
        v2 = _VARIANTS[(i + 3) % len(_VARIANTS)]
        genotype = f"{v[0]} / {v2[0]}"

        patients.append({
            "id":               f"PCCA-{pid:03d}",
            "sex":              sex,
            "phenotype":        phenotype,
            "onset_age_months": round(onset_age, 2),
            "c3_umol_l":        round(c3_umol_l, 1),
            "c3_c2_ratio":      round(c3_c2_ratio, 3),
            "methylcitrate_umol_mmolCr": round(methylcitrate, 0),
            "three_oh_propionate_umol_mmolCr": round(three_oh_prop, 0),
            "propionylglycine_umol_mmolCr": round(propionylgly, 0),
            "ammonia_umol_l":   round(ammonia, 0),
            "free_carnitine_umol_l": round(free_carnitine, 1),
            "seizures":         seizures,
            "cardiomyopathy":   cardiomyopathy,
            "qt_prolonged":     qt_prolonged,
            "bg_infarct":       bg_infarct,
            "neutropenia":      neutropenia,
            "nbs_detected":     nbs_detected,
            "pancreatitis":     pancreatitis,
            "hypoglycemia":     hypoglycemia,
            "liver_transplant": liver_tx_done,
            "on_metronidazole": on_metronidazole,
            "genotype":         genotype,
        })
        pid += 1
    return patients


_COHORT = _make_cohort()


# ------------------------------------------------------------------
# PUBLIC API FUNCTIONS
# ------------------------------------------------------------------
def get_overview():
    """Cohort KPIs, phenotype distribution, PCC pathway, PCCA vs PCCB, high-risk situations."""
    n = len(_COHORT)
    avg_c3       = round(sum(p["c3_umol_l"] for p in _COHORT) / n, 1)
    avg_mc       = round(sum(p["methylcitrate_umol_mmolCr"] for p in _COHORT) / n, 0)
    avg_nh3      = round(sum(p["ammonia_umol_l"] for p in _COHORT) / n, 0)
    avg_carn     = round(sum(p["free_carnitine_umol_l"] for p in _COHORT) / n, 1)
    seizure_pct  = round(sum(1 for p in _COHORT if p["seizures"]) / n * 100, 1)
    cardio_pct   = round(sum(1 for p in _COHORT if p["cardiomyopathy"]) / n * 100, 1)
    qt_pct       = round(sum(1 for p in _COHORT if p["qt_prolonged"]) / n * 100, 1)
    bg_pct       = round(sum(1 for p in _COHORT if p["bg_infarct"]) / n * 100, 1)
    neutro_pct   = round(sum(1 for p in _COHORT if p["neutropenia"]) / n * 100, 1)
    nbs_pct      = round(sum(1 for p in _COHORT if p["nbs_detected"]) / n * 100, 1)
    hypo_pct     = round(sum(1 for p in _COHORT if p["hypoglycemia"]) / n * 100, 1)
    liver_tx_pct = round(sum(1 for p in _COHORT if p["liver_transplant"]) / n * 100, 1)

    phenotype_dist = []
    for phenotype, count in _PHENOTYPES:
        pct = round(count / n * 100, 1)
        phenotype_dist.append({"phenotype": phenotype, "count": count, "pct": pct})

    pcc_pathway = [
        {
            "step": "Step 1 — Biotin Carboxylation (PCCA BC domain)",
            "reaction": "Biotin-PCCA + HCO₃⁻ + ATP → Carboxybiotin-PCCA + ADP + Pᵢ",
            "enzyme_subunit": "PCCA alpha subunit (BC domain, aa 1–435)",
            "cofactor": "ATP + HCO₃⁻ + Mg²⁺",
            "loss_when_PCCA_mutant": "ATP-dependent carboxylation of biotin blocked → no activated CO₂ donor",
        },
        {
            "step": "Step 2 — Propionyl Transfer (PCCB CT domain)",
            "reaction": "Carboxybiotin-PCCA + Propionyl-CoA → D-Methylmalonyl-CoA + Biotin-PCCA",
            "enzyme_subunit": "PCCB beta subunit (CT domain) — intact in PCCA deficiency",
            "cofactor": "Biotin (attached to PCCA Lys669 BCCP domain)",
            "loss_when_PCCA_mutant": "PCCB CT domain is intact but receives no activated CO₂ from mutant PCCA → propionyl-CoA cannot be carboxylated",
        },
        {
            "step": "Downstream — Methylmalonyl-CoA processing (not affected in PA)",
            "reaction": "D-Methylmalonyl-CoA → L-Methylmalonyl-CoA (MCEE) → Succinyl-CoA (MMUT)",
            "enzyme_subunit": "MCEE (methylmalonyl-CoA epimerase) + MMUT (methylmalonyl-CoA mutase, B12-dependent)",
            "cofactor": "Adenosylcobalamin (B12) for MMUT",
            "loss_when_PCCA_mutant": "MMUT pathway INTACT — no methylmalonate accumulation (distinguishes PA from MMA)",
        },
        {
            "step": "Propionyl-CoA Accumulation Consequences",
            "reaction": "Propionyl-CoA + OAA → Methylcitrate (via aconitase) — PATHOGNOMONIC",
            "enzyme_subunit": "Mitochondrial aconitase (Aco2) — incorporates propionyl-CoA as OAA analogue",
            "cofactor": "N/A",
            "loss_when_PCCA_mutant": "Methylcitrate accumulates in urine (50–2,000 µmol/mmolCr); CPS1 inhibited → hyperammonemia",
        },
    ]

    high_risk_situations = [
        {"situation": "VPA (valproate)",          "risk": "ABSOLUTE CI",   "detail": "Valproyl-CoA competitively inhibits PCC; carnitine depletion compounds PA; fatal in acute decompensation"},
        {"situation": "Fasting",                  "risk": "EXTREME HAZARD","detail": "Catabolism → Ile/Val/Met/Thr → propionyl-CoA surge; always administer IV glucose in intercurrent illness; never fast PA patients"},
        {"situation": "Intercurrent illness",     "risk": "EXTREME HAZARD","detail": "Fever → catabolism → propionyl-CoA → hyperammonemia crisis in hours; emergency protocol mandatory"},
        {"situation": "High-protein diet",        "risk": "EXTREME HAZARD","detail": "Ile, Val, Met, Thr (propionyl-CoA precursors) in high-protein foods flood pathway; strict metabolic diet required"},
        {"situation": "Surgery/GA without cover", "risk": "HAZARD",        "detail": "NPO fasting + surgical catabolism → crisis; mandatory IV glucose + PCCA/PCCB-free AA infusion perioperatively"},
        {"situation": "High odd-chain fat intake","risk": "HAZARD",        "detail": "Odd-chain FAs → propionyl-CoA at terminal β-oxidation step; avoid propionic acid-containing foods"},
        {"situation": "Missed carnitine doses",   "risk": "HAZARD",        "detail": "Carnitine conjugates propionyl-CoA for renal excretion; without carnitine, propionyl-CoA accumulates rapidly"},
    ]

    return {
        "gene":              "PCCA",
        "full_name":         "Propionyl-CoA Carboxylase Alpha Subunit",
        "chromosome":        "13q32.3",
        "inheritance":       "AR",
        "omim_gene":         "*232000",
        "omim_disease":      "#606054",
        "protein_size":      "728 aa; BC domain (1–435) + BCCP domain (436–728); Lys669 = biotin attachment",
        "prevalence":        "1:100,000–150,000 (European); 1:2,000–5,000 (Arabian Peninsula)",
        "nbs_primary":       "C3 (propionylcarnitine) elevated — acylcarnitine tandem MS/MS",
        "nbs_secondary":     "C3/C2 ratio; urine organic acids (methylcitrate, 3-OH-propionate)",
        "function":          "Alpha subunit (BC + BCCP domains) of PCC (αβ)6 dodecamer. Biotin-dependent carboxylation of propionyl-CoA → D-methylmalonyl-CoA in mitochondrial matrix. PCCA performs Step 1 (ATP + HCO₃⁻ → carboxybiotin) via BC domain; Lys669 (BCCP) swings carboxybiotin to PCCB CT domain for Step 2.",
        "mechanism":         "PCCA LOF → PCC (αβ)6 dodecamer loses BC function → propionyl-CoA cannot be carboxylated → propionyl-CoA accumulates → methylcitrate (PATHOGNOMONIC) + propionylglycine + 3-OH-propionate + hyperammonemia (CPS1 inhibition via NAGS). MMUT pathway INTACT → NO methylmalonate (vs MMA).",
        "key_negative":      "NO C5-OH (no MCC block, unlike HLCS/BTD); NO methylmalonate (vs MMA/MMUT); biotinidase NORMAL; biotin NORMAL; BCAA NORMAL (vs DLD/MSUD); NO biotin response (PCCA enzyme itself mutant)",
        "cohort_n":          n,
        "kpis": {
            "avg_c3_umol_l":            avg_c3,
            "avg_methylcitrate":        int(avg_mc),
            "avg_ammonia_umol_l":       int(avg_nh3),
            "avg_free_carnitine":       avg_carn,
            "seizure_pct":              seizure_pct,
            "cardiomyopathy_pct":       cardio_pct,
            "qt_prolonged_pct":         qt_pct,
            "bg_infarct_pct":           bg_pct,
            "neutropenia_pct":          neutro_pct,
            "nbs_detected_pct":         nbs_pct,
            "hypoglycemia_pct":         hypo_pct,
            "liver_transplant_pct":     liver_tx_pct,
        },
        "phenotype_distribution": phenotype_dist,
        "pcc_pathway":        pcc_pathway,
        "high_risk_situations": high_risk_situations,
        "pcca_vs_pccb": {
            "title": "PCCA vs PCCB — Same Disease, Two Genes (always gene-panel both)",
            "note": "PCCA and PCCB both cause Propionic Acidemia; biochemically and clinically IDENTICAL. ~50% PA due to PCCA, ~50% due to PCCB. Gene panel mandatory.",
            "comparison": [
                {"feature": "Gene",               "PCCA": "PCCA (13q32.3)",                   "PCCB": "PCCB (3q22.3)"},
                {"feature": "Protein size",       "PCCA": "728 aa",                           "PCCB": "559 aa"},
                {"feature": "Domain / Function",  "PCCA": "BC + BCCP (biotin carboxylase + carrier)", "PCCB": "CT (carboxyl transferase from biotin to propionyl-CoA)"},
                {"feature": "Biotin attachment",  "PCCA": "Lys669 in BCCP domain",            "PCCB": "No biotin; accepts CO₂ from PCCA-biotin"},
                {"feature": "OMIM gene",          "PCCA": "*232000",                          "PCCB": "*232050"},
                {"feature": "Disease",            "PCCA": "#606054 (PA)",                     "PCCB": "#606054 (PA, same disease)"},
                {"feature": "Phenotype",          "PCCA": "Identical to PCCB",               "PCCB": "Identical to PCCA"},
                {"feature": "NBS trigger",        "PCCA": "C3 elevated (same)",              "PCCB": "C3 elevated (same)"},
                {"feature": "Key biomarker",      "PCCA": "Methylcitrate (same)",            "PCCB": "Methylcitrate (same)"},
                {"feature": "Treatment",          "PCCA": "Identical (carnitine, diet, etc.)","PCCB": "Identical (carnitine, diet, etc.)"},
                {"feature": "Distinguishing",     "PCCA": "Only by gene sequencing",         "PCCB": "Only by gene sequencing"},
                {"feature": "Prevalence split",   "PCCA": "~50% of PA cases",                "PCCB": "~50% of PA cases"},
            ],
        },
    }


def get_breakdown():
    """Biomarkers, key variants, seizure types, metabolic triggers, high-risk drugs, treatments, patient sample."""
    # Biomarkers
    biomarkers = [
        {"name": "C3 (propionylcarnitine, NBS PRIMARY)",  "normal": "< 3.5 µmol/L",    "pa_range": "5–40 µmol/L",       "significance": "PRIMARY NBS trigger; direct measure of propionyl-CoA accumulation (as carnitine conjugate)",         "method": "Tandem MS/MS acylcarnitine profile"},
        {"name": "C3/C2 ratio",                           "normal": "< 0.20",           "pa_range": "> 0.25–0.80",       "significance": "Corrects for dietary carnitine variability; more specific than C3 alone for NBS cutoff",           "method": "Tandem MS/MS derived ratio"},
        {"name": "Methylcitrate (urine OA)",              "normal": "< 10 µmol/mmolCr", "pa_range": "50–2,000 µmol/mmolCr","significance": "PATHOGNOMONIC for propionyl-CoA accumulation; absent in MMA (most critical PA vs MMA differentiator)","method": "GC-MS urine organic acids"},
        {"name": "3-Hydroxypropionate (urine OA)",        "normal": "< 20 µmol/mmolCr", "pa_range": "100–1,500 µmol/mmolCr","significance": "Confirms propionyl-CoA pathway block; parallel to methylcitrate elevation",                   "method": "GC-MS urine organic acids"},
        {"name": "Propionylglycine (urine OA)",           "normal": "Absent",            "pa_range": "50–800 µmol/mmolCr","significance": "Propionyl-CoA conjugation with glycine; confirms PA; similar to propionyl-CoA marker",           "method": "GC-MS urine organic acids"},
        {"name": "Methylmalonate (urine OA)",             "normal": "< 10 µmol/mmolCr", "pa_range": "ABSENT / NORMAL",   "significance": "KEY NEGATIVE vs MMA: absent in PA because MMUT pathway is intact; C3 elevation WITH normal methylmalonate = PA",  "method": "GC-MS urine organic acids"},
        {"name": "Plasma ammonia",                        "normal": "< 80 µmol/L",      "pa_range": "150–2,000 µmol/L (acute crisis)","significance": "Secondary hyperammonemia via NAGS inhibition (propionyl-CoA) → CPS1 block; encephalopathy risk", "method": "Plasma NH₃ (immediate, on ice)"},
        {"name": "Free carnitine",                        "normal": "25–60 µmol/L",     "pa_range": "3–25 µmol/L",       "significance": "Secondary carnitine depletion as propionyl-CoA is conjugated to C3 for renal excretion; replaces carnitine pool","method": "Plasma free + total carnitine"},
        {"name": "Biotinidase activity",                  "normal": "≥ 30% normal",     "pa_range": "NORMAL",            "significance": "KEY NEGATIVE vs BTD: biotinidase normal in PA; PA is NOT a biotin recycling disorder",               "method": "Serum fluorometric assay"},
        {"name": "C5-OH acylcarnitine",                   "normal": "< 0.5 µmol/L",     "pa_range": "NORMAL",            "significance": "KEY NEGATIVE vs HLCS/BTD: absent/normal in PA (no MCC block); elevated in MCD disorders",         "method": "Tandem MS/MS acylcarnitine"},
        {"name": "BCAA (Leu, Ile, Val plasma)",           "normal": "Normal ranges",     "pa_range": "NORMAL",            "significance": "KEY NEGATIVE vs DLD and MSUD: BCAA normal in PA (BCKDH intact); PA has only propionyl-CoA block", "method": "Plasma amino acids (HPLC or MS/MS)"},
    ]

    # Key variants
    key_variants = [
        {"variant": v[0], "cdna": v[1], "domain": v[2], "severity": v[3], "note": v[4]}
        for v in _VARIANTS
    ]

    # Seizure types
    seizure_types = [
        {"type": "Metabolic crisis seizures (acute)",   "pct": 55, "note": "Tonic-clonic or multifocal clonic during hyperammonemia/acidosis episode; resolve with metabolic correction"},
        {"type": "Early infantile epileptic encephalopathy","pct": 40, "note": "EIEE pattern in severe neonatal classic PA; diffuse EEG suppression during acidosis"},
        {"type": "Post-BG infarct epilepsy",            "pct": 30, "note": "Focal or generalized epilepsy secondary to basal ganglia stroke-like episode; often drug-resistant"},
        {"type": "Infantile spasms / West syndrome",    "pct": 15, "note": "Hypsarrhythmia; associated with severe neonatal-onset and BG injury"},
        {"type": "Absence-like spells (metabolic)",     "pct": 12, "note": "Non-convulsive seizure pattern during metabolic decompensation; EEG shows generalized spike-wave"},
        {"type": "Myoclonic (metabolic trigger)",       "pct": 18, "note": "Myoclonic jerks during acute metabolic crisis; usually provoked rather than spontaneous"},
    ]

    # Metabolic triggers
    metabolic_triggers = [
        {"trigger": "Intercurrent febrile illness",   "pct": 85, "mechanism": "Fever → protein catabolism → Ile/Val/Met/Thr → propionyl-CoA surge → decompensation in 4-12h"},
        {"trigger": "Fasting > 4 hours",              "pct": 70, "mechanism": "Protein catabolism + odd-chain FA oxidation → propionyl-CoA accumulation; glucose infusion is antidote"},
        {"trigger": "High protein intake",            "pct": 65, "mechanism": "Dietary Ile, Val, Met, Thr overwhelm restricted propionyl-CoA pathway; requires metabolic formula"},
        {"trigger": "Surgery / anesthesia",           "pct": 45, "mechanism": "NPO fasting + surgical stress + catabolism; perioperative metabolic monitoring mandatory"},
        {"trigger": "Constipation",                   "pct": 35, "mechanism": "Intestinal bacteria produce propionate; absorbed propionate → portal propionyl-CoA; treat constipation aggressively"},
        {"trigger": "Missed metabolic formula doses", "pct": 30, "mechanism": "Formula provides Ile/Val/Met/Thr-free nitrogen; missed doses → protein from natural sources fill gap"},
    ]

    # Treatments
    treatments = [
        {"treatment": "Natural protein restriction (low Ile, Val, Met, Thr)",  "evidence": "Level A", "response_pct": 90, "note": "Cornerstone of PA management; restrict propionyl-CoA precursor amino acids; adjust based on C3 and growth"},
        {"treatment": "PCCA/PCCB-free amino acid formula",                     "evidence": "Level A", "response_pct": 90, "note": "Provides essential nitrogen without Ile/Val/Met/Thr; mandatory for adequate protein synthesis; lifelong"},
        {"treatment": "L-Carnitine (100 mg/kg/day; crisis: 200 mg/kg/day IV)", "evidence": "Level A", "response_pct": 85, "note": "Conjugates propionyl-CoA → propionylcarnitine (C3) for urinary excretion; replaces depleted carnitine pool"},
        {"treatment": "IV glucose + insulin (acute crisis: GIR 8–12 mg/kg/min)","evidence": "Level A", "response_pct": 80, "note": "Anabolic state → halts catabolism → stops propionyl-CoA production; FIRST intervention in acute decompensation"},
        {"treatment": "Ammonia scavengers (Na benzoate / Na phenylacetate)",    "evidence": "Level A", "response_pct": 75, "note": "For acute hyperammonemia > 200 µmol/L; alternative nitrogen excretion pathway bypassing CPS1 block"},
        {"treatment": "Metronidazole (intermittent 5-10 day courses)",          "evidence": "Level B", "response_pct": 55, "note": "Reduces gut bacterial propionate production; decreases systemic propionyl-CoA load; cycles monthly"},
        {"treatment": "Liver transplantation",                                   "evidence": "Level B", "response_pct": 70, "note": "Corrects ~75% of hepatic PCC activity; eliminates acute decompensation risk; does NOT fully protect heart (extra-hepatic PCC)"},
        {"treatment": "Hemodialysis / CRRT (severe NH3 > 500 µmol/L)",         "evidence": "Level B", "response_pct": 80, "note": "For refractory hyperammonemia unresponsive to medical management; rapidly clears ammonia and organic acids"},
        {"treatment": "Biotin supplementation",                                  "evidence": "NOT EFFECTIVE", "response_pct": 5, "note": "PA is NOT a biotin disorder; PCCA enzyme itself mutant, not just under-biotinylated; biotin does NOT help in PCCA/PCCB deficiency (unlike HLCS/BTD)"},
        {"treatment": "VPA (valproate)",                                         "evidence": "ABSOLUTE CI", "response_pct": 0, "note": "Valproyl-CoA inhibits PCC directly; also depletes carnitine; FATAL in PA crisis; use LEV, LCM, or lacosamide instead"},
    ]

    # Patient sample (first 15)
    patient_sample = _COHORT[:15]

    # High-risk drugs
    high_risk_drugs = [
        {"drug": d[0], "risk": d[1], "mechanism": d[2]}
        for d in _HIGH_RISK_DRUGS
    ]

    return {
        "biomarkers":       biomarkers,
        "key_variants":     key_variants,
        "seizure_types":    seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "treatments":       treatments,
        "patient_sample":   patient_sample,
        "high_risk_drugs":  high_risk_drugs,
    }


def get_definitions():
    """Gene card, key concepts, diagnostic thresholds, differential diagnosis."""
    gene_card = {
        "Gene":              "PCCA",
        "Full name":         "Propionyl-CoA Carboxylase Alpha Subunit",
        "Synonyms":          "PCCAL, PCC-alpha",
        "Chromosome":        "13q32.3",
        "Inheritance":       "Autosomal Recessive (AR)",
        "Protein size":      "728 amino acids",
        "Domains":           "BC domain (aa 1–435): biotin carboxylase; BCCP domain (aa 436–728): biotin carrier — Lys669 attachment site",
        "Complex structure": "(αβ)₆ dodecamer: 6 PCCA + 6 PCCB in mitochondrial matrix",
        "Cofactor":          "Biotin (covalently attached to PCCA Lys669 by HLCS; ATP + HCO₃⁻ for carboxylation step)",
        "Reaction":          "Propionyl-CoA + HCO₃⁻ + ATP → D-Methylmalonyl-CoA + ADP + Pᵢ (net; two-step)",
        "OMIM Gene":         "*232000",
        "OMIM Disease":      "#606054 (PROPIONIC ACIDEMIA)",
        "Disease name":      "Propionic Acidemia (PA) — same as PCCB deficiency",
        "Prevalence":        "1:100,000–150,000 (European); 1:2,000–5,000 (Arabian Peninsula founder variants)",
        "NBS":               "C3 (propionylcarnitine) elevated — tandem MS/MS acylcarnitine panel; universal NBS worldwide",
        "Key pathway":       "Propionyl-CoA → D-methylmalonyl-CoA (PCC) → L-methylmalonyl-CoA (MCEE) → succinyl-CoA (MMUT/B12)",
    }

    key_concepts = [
        {
            "concept": "Why PCCA LOF causes hyperammonemia (CPS1 inhibition mechanism)",
            "explanation": "Propionyl-CoA accumulation inhibits N-acetylglutamate synthase (NAGS) competitively. NAGS produces N-acetylglutamate (NAG), the essential allosteric activator of CPS1 (the first enzyme of the urea cycle). Reduced NAG → CPS1 activity ↓ → ammonia cannot enter urea cycle → NH3 accumulates. This is SECONDARY hyperammonemia (not a primary urea cycle disorder). Treat with ammonia scavengers + protein withdrawal + IV glucose.",
        },
        {
            "concept": "Why methylcitrate is PATHOGNOMONIC for PA (and absent in MMA)",
            "explanation": "Propionyl-CoA (C3-CoA) enters the TCA cycle as a structural analogue of acetyl-CoA. Mitochondrial aconitase (Aco2) condenses propionyl-CoA + OAA → methylcitrate (instead of acetyl-CoA + OAA → citrate). Methylcitrate is a potent inhibitor of isocitrate dehydrogenase and aconitase, further impairing TCA cycle. In MMA (MMUT deficiency), propionyl-CoA is UPSTREAM of MMUT, so methylcitrate accumulates in MMA too — but propionylglycine and 3-OH-propionate are more prominent in PA. KEY: methylmalonate is absent in PA (MMUT intact), present in MMA (MMUT mutant). Methylcitrate = both PA and MMA; methylmalonate = MMA only.",
        },
        {
            "concept": "Why biotin does NOT work in PCCA deficiency (unlike HLCS/BTD)",
            "explanation": "In HLCS deficiency, biotin LIGATION to carboxylases (including PCCA Lys669) is impaired — biotin supplementation exceeds Km defect and activates residual HLCS. In BTD deficiency, biotin RECYCLING is impaired — biotin replacement restores free biotin pool. In PCCA deficiency, the enzyme is biotinylated normally (HLCS and BTD are both normal), but the BC domain of PCCA itself is mutant — biotin cannot compensate for a catalytically dead or misfolded BC domain. Biotin supplementation in PA is physiologically inert.",
        },
        {
            "concept": "Why VPA is ABSOLUTELY CONTRAINDICATED in PA",
            "explanation": "Valproyl-CoA (the active metabolite of VPA) is a competitive inhibitor of PCC (same active site as propionyl-CoA). In a patient with already-impaired PCC (due to PCCA mutation), adding VPA further reduces residual PCC activity → acute propionyl-CoA surge → decompensation. VPA also depletes free carnitine (valproylcarnitine formation) — removing the buffer that conjugates propionyl-CoA for excretion. Combined effect is potentially fatal. First-line AED: levetiracetam (LEV), lacosamide, or lamotrigine (avoid enzyme-inducing AEDs too).",
        },
        {
            "concept": "Why cardiomyopathy occurs in PA even on treatment",
            "explanation": "Cardiac muscle has high mitochondrial density and propionate metabolism. Propionyl-CoA directly inhibits the mitochondrial respiratory chain (Complex I, III), disrupts CoA homeostasis in cardiomyocytes, and causes protein propionylation on cardiac proteins. Liver transplantation corrects hepatic propionyl-CoA but extra-hepatic PCC activity remains deficient. Cardiomyopathy can progress even after liver transplant. Annual echocardiography + 24h Holter for QT surveillance is mandatory — QT prolongation can cause sudden cardiac death.",
        },
        {
            "concept": "PCC complex structure and why both PCCA and PCCB cause identical PA",
            "explanation": "PCC forms an (αβ)₆ dodecamer: 6 PCCA (alpha) subunits arranged in a cubic core, with 6 PCCB (beta) subunits positioned peripherally. PCCA carries biotin on Lys669 (BCCP domain) and performs Step 1 (CO₂ activation); PCCB performs Step 2 (carboxyl transfer to propionyl-CoA). LOF of EITHER subunit eliminates the complex's catalytic activity — the intact subunit alone is insufficient. This is why PCCA and PCCB mutations cause biochemically identical Propionic Acidemia. The gene that is mutant can only be determined by DNA sequencing — phenotype, biomarkers, and treatment are the same.",
        },
        {
            "concept": "Gut bacterial propionate and why metronidazole reduces PA burden",
            "explanation": "Colonic bacteria (Bacteroides, Propionibacterium, Veillonella) ferment dietary fiber and produce propionate as a major short-chain fatty acid. This propionate is absorbed through the portal circulation, enters hepatocytes, and is converted to propionyl-CoA. In healthy individuals, PCC rapidly converts this to methylmalonyl-CoA. In PA patients, this gut-derived propionate is an additional propionyl-CoA load on an already-impaired pathway. Metronidazole (5-10 day courses, monthly) kills propionate-producing anaerobes and reduces this load by ~20-30%, measurably reducing C3 levels and decompensation frequency.",
        },
    ]

    diagnostic_thresholds = [
        {"parameter": "C3 (NBS primary)",              "threshold": "> 3.5 µmol/L triggers recall; > 5.0 µmol/L strong action",       "action": "Immediate PA/MMA workup; plasma amino acids, urine OA, ammonia; genetic referral same day"},
        {"parameter": "Methylcitrate (urine OA)",      "threshold": "> 10 µmol/mmolCr significant; > 50 µmol/mmolCr diagnostic PA",   "action": "PA confirmed; distinguish from MMA by absent methylmalonate; PCCA + PCCB gene panel"},
        {"parameter": "Methylmalonate (urine OA)",     "threshold": "Absent in PA (< 10 µmol/mmolCr = KEY NEGATIVE)",                  "action": "If methylmalonate elevated → MMA (MMUT/MMAA/MMAB/MMADHC); if absent = PA confirmed"},
        {"parameter": "Plasma ammonia (acute)",        "threshold": "> 150 µmol/L urgent; > 300 µmol/L emergent; > 500 µmol/L ICU",   "action": "Protein withdrawal; IV glucose GIR 8-12; ammonia scavengers; ICU if >300; HD/CRRT if >500 or rising"},
        {"parameter": "Free carnitine",                "threshold": "< 20 µmol/L requires supplementation (normal 25-60 µmol/L)",      "action": "Start/increase L-carnitine; 100 mg/kg/day oral; 200 mg/kg/day IV in acute crisis"},
        {"parameter": "Echocardiography",              "threshold": "Annual baseline; EF < 50% = dilated cardiomyopathy action",       "action": "Cardiology referral; ACE inhibitor if DCM confirmed; 24h Holter for QTc > 450ms"},
        {"parameter": "PCC enzyme activity",           "threshold": "< 5% normal in fibroblasts or lymphocytes = severe PA",           "action": "Confirms diagnosis when gene sequencing pending; < 10% = pathological; correlates with phenotype"},
        {"parameter": "PCCA/PCCB gene sequencing",    "threshold": "Biallelic pathogenic variants in PCCA OR PCCB = diagnostic",      "action": "Always sequence BOTH genes (PCCA + PCCB panel); carrier testing of family after proband confirmation"},
    ]

    differential = [
        {
            "disease": "MMA (Methylmalonic Acidemia — MMUT/MMAA/MMAB/MMADHC/MCEE)",
            "distinguishing": "METHYLMALONATE markedly elevated in urine (KEY differentiator — absent in PA); C3 elevated in both; methylcitrate elevated in BOTH but less than PA; cobalamin (B12) responsiveness in some MMA subtypes (MMAA, MMAB, MMADHC) — no B12 response in PA. MMA gene panel: MMUT, MMAA, MMAB, MMADHC, MCEE, LMBRD1.",
        },
        {
            "disease": "HLCS Deficiency (Multiple Carboxylase Deficiency — Neonatal)",
            "distinguishing": "FOUR-carboxylase block (PC + PCC + MCC + ACC): C5-OH + C3 + lactate + hyperammonemia SIMULTANEOUSLY. In PA: ONLY PCC block (C3 elevated; NO C5-OH). Biotinidase NORMAL in HLCS. Biotin 10-40 mg/day DRAMATICALLY effective (LEVEL A) — absent in PA. HLCS gene 21q22.13.",
        },
        {
            "disease": "BTD Deficiency (Multiple Carboxylase Deficiency — Infantile)",
            "distinguishing": "Same FOUR-carboxylase MCD as HLCS but via biotin RECYCLING failure. Biotinidase DEFICIENT (< 10% profound, 10-30% partial) — KEY: NORMAL in PA. Biotin LOW in BTD (depletion); NORMAL in PA. SNHL 75% in BTD; absent in PA. Biotin response YES in BTD; NOT in PA.",
        },
        {
            "disease": "DLD Deficiency (E3 subunit — 4-complex simultaneous block)",
            "distinguishing": "FOUR-complex block (PDH + αKGDH + BCKDH + GCS): lactate + BCAA elevation + 2-HG + glycine SIMULTANEOUSLY. In PA: ONLY C3/propionyl-CoA; BCAA NORMAL; 2-HG NORMAL; glycine NORMAL. DLD gene 7q31.1; VPA absolute CI in BOTH.",
        },
        {
            "disease": "MSUD (BCKDHA/BCKDHB/DBT deficiency)",
            "distinguishing": "BCAA (Leu, Ile, Val) + branched-chain keto acids markedly elevated; alloisoleucine PATHOGNOMONIC (absent in PA). In PA: BCAA NORMAL; no alloisoleucine; C3 elevated (not BCKA). MSUD: maple syrup odour; DNPH test positive; Leu > 1000 µmol/L neurotoxic.",
        },
        {
            "disease": "Isolated MCC Deficiency (MCCC1/MCCC2)",
            "distinguishing": "C5-OH elevated ONLY (3-methylcrotonylglycine + 3-OH-isovalerate); NO C3; NO methylcitrate; no hyperammonemia typical; often benign (most NBS-detected MCC is maternal). PA has only C3 and methylcitrate — no C5-OH.",
        },
        {
            "disease": "NKH (Non-Ketotic Hyperglycinemia — GLDC/AMT/GCSH)",
            "distinguishing": "Glycine markedly elevated in BOTH CSF and plasma; CSF:plasma glycine ratio > 0.08 DIAGNOSTIC. Ketoacidosis ABSENT. In PA: glycine NORMAL (glycine conjugation to propionyl-CoA in PA produces propionylglycine, not hyperglycinemia). No C3 in NKH; no methylcitrate.",
        },
    ]

    return {
        "gene_card":             gene_card,
        "key_concepts":          key_concepts,
        "diagnostic_thresholds": diagnostic_thresholds,
        "differential_diagnosis": differential,
    }


# ------------------------------------------------------------------
# STANDALONE TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    import json
    ov = get_overview()
    bk = get_breakdown()
    df = get_definitions()
    print(f"Overview KPIs: {json.dumps(ov['kpis'], indent=2)}")
    print(f"Patients: {len(bk['patient_sample'])} (of {ov['cohort_n']})")
    print(f"Biomarkers: {len(bk['biomarkers'])}")
    print(f"Concepts: {len(df['key_concepts'])}")
    print(f"Differential: {len(df['differential_diagnosis'])}")
