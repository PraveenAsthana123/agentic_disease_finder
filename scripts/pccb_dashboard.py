#!/usr/bin/env python3
"""PCCB (Propionyl-CoA Carboxylase Beta Subunit / Propionic Acidemia Type B) Epilepsy Dashboard.

PCCB encodes the beta subunit of propionyl-CoA carboxylase (PCC), a biotin-dependent
mitochondrial enzyme that converts propionyl-CoA to D-methylmalonyl-CoA.

  PROPIONYL-COA CARBOXYLASE (PCC) COMPLEX:
    PCCA (alpha, 728 aa): Biotin Carboxylase (BC) domain + Biotin Carboxyl Carrier Protein
      (BCCP) domain. Carries biotin via Lys669. BC domain performs ATP-dependent carboxylation
      of biotin using HCO3-. Performs STEP 1.
    PCCB (beta, 559 aa): Carboxyl Transferase (CT) domain — accepts activated CO2 from
      carboxybiotin-PCCA-Lys669 and transfers it to propionyl-CoA → D-methylmalonyl-CoA.
      Performs STEP 2. Hosts the propionyl-CoA binding site.
    Complex structure: (αβ)₆ dodecamer — 6 PCCA + 6 PCCB in alternating arrangement in
      mitochondrial matrix. PCCB forms the outer beta-trimer scaffold; PCCA inserts biotin
      arms into the PCCB CT active site.

  REACTION CATALYZED (two-step, PCCB performs Step 2):
    Step 1 (BC domain, PCCA): HCO3- + ATP + biotin-PCCA → carboxybiotin-PCCA + ADP + Pi
    Step 2 (CT domain, PCCB): Carboxybiotin-PCCA + propionyl-CoA →
       D-methylmalonyl-CoA + biotin-PCCA
    Net: Propionyl-CoA + HCO3- + ATP → D-methylmalonyl-CoA + ADP + Pi

  PCCB LOF → PCC STEP 2 BLOCKED → PROPIONYL-COA ACCUMULATES → FOUR CASCADES:
    1. Methylcitrate pathway: propionyl-CoA + OAA → methylcitrate (PATHOGNOMONIC)
    2. Omega-oxidation overflow: 3-hydroxypropionate, propionylglycine
    3. NAGS inhibition: propionyl-CoA inhibits N-acetylglutamate synthase →
       reduced NAG → CPS1 down → SECONDARY HYPERAMMONEMIA
    4. Mitochondrial respiratory chain inhibition: propionyl-CoA damages complex I/III/V
       → cardiomyopathy, neutropenia, BG infarcts

  KEY NEGATIVES (distinguish PA from look-alikes):
    - NO methylmalonate (vs MMA/MMUT: methylmalonate elevated — KEY distinction)
    - NO C5-OH (vs HLCS/BTD/MCD: no MCC block in isolated PA)
    - Biotinidase NORMAL (vs BTD)
    - Biotin NOT effective (PCCB enzyme itself mutant — not a biotin recycling defect)
    - BCAA NORMAL (vs DLD/MSUD: no BCKDH block in isolated PA)

  PCCB vs PCCA:
    - Biochemically IDENTICAL: same biomarkers, same treatments, same phenotypes
    - Gene panel (NGS) MANDATORY to distinguish — cannot tell apart by biochemistry
    - NBS: both elevate C3 (propionylcarnitine) — cannot distinguish PCCA from PCCB by NBS

  FOUNDER VARIANTS IN PCCB:
    - p.Glu168Lys (c.502G>A): Dutch/Belgian/Northern European founder; moderate; ~30% residual CT activity
    - c.1218_1231del14 (p.Phe406fs): Spanish/Portuguese/Latin American founder; null; severe neonatal
    - c.IVS12+1G>A (c.1283+1G>A): Saudi Arabian/Middle Eastern founder; splice donor null; severe
    - p.Arg512Cys: European; CT catalytic Arg disrupted; severe
    - p.Arg399Cys: Pan-European; subunit interface disruption; severe

  OMIM: Gene *232050 · Disease #606054 (Propionic Acidemia — same disease locus as PCCA)
  Chromosome: 3q22.3 · Inheritance: AR (Autosomal Recessive) · Prevalence: ~1:100,000–150,000 (EU)
"""

from __future__ import annotations
import random

random.seed(42)   # reproducible synthetic cohort


# ── helpers ──────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"PCCB-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    r = i % 20
    if r < 14:  return "Classic Neonatal Severe"
    if r < 18:  return "Late-Onset Episodic"
    if r == 18: return "Intermediate"
    return "Paucisymptomatic"


def _genotype(i: int) -> str:
    variants = [
        "p.Glu168Lys/p.Glu168Lys",              # Dutch founder homozygous
        "p.Glu168Lys/c.1218_1231del14",          # Dutch + Spanish compound het
        "c.1218_1231del14/c.1218_1231del14",     # Spanish founder homozygous
        "c.IVS12+1G>A/c.IVS12+1G>A",            # Saudi founder homozygous
        "p.Arg512Cys/p.Arg512Cys",               # CT catalytic null
        "p.Glu168Lys/p.Arg399Cys",              # Dutch + interface compound het
        "p.Arg399Cys/c.1218_1231del14",          # compound het severe
        "p.Glu168Lys/c.IVS12+1G>A",             # Dutch + splice null
        "p.Gly441Arg/p.Arg512Cys",              # both CT domain, severe
        "p.Val145Glu/p.Glu168Lys",              # mild/moderate compound het
    ]
    return variants[i % len(variants)]


def _c3(pheno: str) -> float:
    if pheno == "Classic Neonatal Severe":
        return round(random.uniform(12, 45), 1)
    if pheno == "Late-Onset Episodic":
        return round(random.uniform(5, 20), 1)
    if pheno == "Intermediate":
        return round(random.uniform(4, 12), 1)
    return round(random.uniform(3.5, 7), 1)


def _methylcitrate(pheno: str) -> int:
    if pheno == "Classic Neonatal Severe":
        return random.randint(120, 1800)
    if pheno == "Late-Onset Episodic":
        return random.randint(30, 350)
    if pheno == "Intermediate":
        return random.randint(25, 180)
    return random.randint(15, 80)


def _ammonia(pheno: str) -> int:
    if pheno == "Classic Neonatal Severe":
        return random.randint(180, 1800)
    if pheno == "Late-Onset Episodic":
        return random.randint(80, 600)
    if pheno == "Intermediate":
        return random.randint(60, 250)
    return random.randint(30, 120)


def _carnitine(pheno: str) -> float:
    if pheno == "Classic Neonatal Severe":
        return round(random.uniform(5, 20), 1)
    if pheno == "Late-Onset Episodic":
        return round(random.uniform(10, 28), 1)
    return round(random.uniform(15, 35), 1)


def _three_oh_prop(pheno: str) -> int:
    if pheno == "Classic Neonatal Severe":
        return random.randint(80, 1200)
    if pheno == "Late-Onset Episodic":
        return random.randint(20, 250)
    return random.randint(10, 120)


# ── data functions ────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _build_patients()
    phenos = [p["phenotype"] for p in patients]
    n = len(patients)

    pheno_dist = {
        "Classic Neonatal Severe": phenos.count("Classic Neonatal Severe"),
        "Late-Onset Episodic":     phenos.count("Late-Onset Episodic"),
        "Intermediate":            phenos.count("Intermediate"),
        "Paucisymptomatic":        phenos.count("Paucisymptomatic"),
    }

    avg_c3          = round(sum(p["c3_umol_l"]                  for p in patients) / n, 1)
    avg_mc          = round(sum(p["methylcitrate_umol_mmolCr"]  for p in patients) / n)
    avg_nh3         = round(sum(p["ammonia_umol_l"]             for p in patients) / n)
    avg_carn        = round(sum(p["free_carnitine_umol_l"]      for p in patients) / n, 1)
    seiz_pct        = round(sum(1 for p in patients if p["seizures"]) / n * 100)
    cardio_pct      = round(sum(1 for p in patients if p["cardiomyopathy"]) / n * 100)
    qt_pct          = round(sum(1 for p in patients if p.get("qt_prolonged")) / n * 100)
    bg_pct          = round(sum(1 for p in patients if p["bg_infarct"]) / n * 100)
    neutro_pct      = round(sum(1 for p in patients if p.get("neutropenia")) / n * 100)
    nbs_pct         = round(sum(1 for p in patients if p["nbs_detected"]) / n * 100)
    hypogly_pct     = round(sum(1 for p in patients if p.get("hypoglycemia")) / n * 100)
    liver_tx_pct    = round(sum(1 for p in patients if p.get("liver_transplant")) / n * 100)

    return {
        "gene":           "PCCB",
        "full_name":      "Propionyl-CoA Carboxylase Beta Subunit",
        "chromosome":     "3q22.3",
        "inheritance":    "AR",
        "omim_gene":      "*232050",
        "omim_disease":   "#606054",
        "protein_size":   "559 aa; CT (Carboxyl Transferase) domain",
        "prevalence":     "1:100,000–150,000 (EU); 1:2,000–5,000 (Arabian Peninsula)",
        "nbs_primary":    "C3 (propionylcarnitine) elevated — identical to PCCA (cannot distinguish by NBS)",
        "nbs_secondary":  "C3/C2 ratio; urine OA (methylcitrate — PATHOGNOMONIC)",
        "function": (
            "PCCB performs STEP 2 of the two-step PCC reaction: its CT (Carboxyl Transferase) domain "
            "accepts activated CO2 (as carboxybiotin) from PCCA-Lys669 and transfers it to propionyl-CoA, "
            "yielding D-methylmalonyl-CoA. PCCB hosts the propionyl-CoA binding site. In the (αβ)₆ "
            "dodecamer, 6 PCCB subunits form the outer beta-trimer scaffold into which PCCA inserts its "
            "biotin-carrying BCCP swinging arms."
        ),
        "mechanism": (
            "PCCB LOF → CT domain absent or inactive → propionyl-CoA cannot receive CO2 transfer → "
            "propionyl-CoA ACCUMULATES → (1) methylcitrate pathway: propionyl-CoA + OAA → "
            "methylcitrate [PATHOGNOMONIC], (2) NAGS inhibition → secondary hyperammonemia "
            "(NH3 150–2000 µmol/L), (3) mitochondrial respiratory chain inhibition → cardiomyopathy / "
            "BG infarcts, (4) bone marrow suppression → neutropenia/thrombocytopenia."
        ),
        "key_negative": (
            "NO methylmalonate (vs MMA/MMUT — KEY distinction: absent in PA); "
            "NO C5-OH (vs HLCS/BTD/MCD); Biotinidase NORMAL (vs BTD); Biotin NOT effective "
            "(PCCB enzyme itself mutant — not a biotin recycling defect); BCAA NORMAL (vs DLD/MSUD)."
        ),
        "cohort_n":     n,
        "kpis": {
            "avg_c3_umol_l":          avg_c3,
            "avg_methylcitrate":      avg_mc,
            "avg_ammonia_umol_l":     avg_nh3,
            "avg_free_carnitine":     avg_carn,
            "seizure_pct":            seiz_pct,
            "cardiomyopathy_pct":     cardio_pct,
            "qt_prolonged_pct":       qt_pct,
            "bg_infarct_pct":         bg_pct,
            "neutropenia_pct":        neutro_pct,
            "nbs_detected_pct":       nbs_pct,
            "hypoglycemia_pct":       hypogly_pct,
            "liver_transplant_pct":   liver_tx_pct,
        },
        "phenotype_distribution": [
            {"phenotype": k, "pct": round(v / n * 100)} for k, v in pheno_dist.items()
        ],
        "pcc_pathway": [
            {
                "step":                   "Step 1 — BC domain (PCCA)",
                "reaction":               "HCO3⁻ + ATP + biotin-PCCA → carboxybiotin-PCCA + ADP + Pᵢ",
                "enzyme_subunit":         "PCCA alpha subunit (728 aa)",
                "cofactor":               "Biotin (at Lys669), ATP, HCO3⁻, Mg²⁺",
                "loss_when_PCCB_mutant":  "Step 1 intact — PCCA correctly charges biotin. "
                                          "Carboxybiotin-PCCA swinging arm extends but CT domain (PCCB) "
                                          "cannot accept CO2 → reaction stalls after step 1.",
            },
            {
                "step":                   "Step 2 — CT domain (PCCB) ← DEFICIENT IN PCCB DISEASE",
                "reaction":               "Carboxybiotin-PCCA + propionyl-CoA → D-methylmalonyl-CoA + biotin-PCCA",
                "enzyme_subunit":         "PCCB beta subunit (559 aa) — CT domain performs CO2 transfer to propionyl-CoA",
                "cofactor":               "None additional (uses activated CO2 from step 1)",
                "loss_when_PCCB_mutant":  "CT domain absent/inactive → propionyl-CoA CANNOT be carboxylated → "
                                          "propionyl-CoA ACCUMULATES → four downstream cascades.",
            },
        ],
        "pcca_vs_pccb": {
            "title": "PCCA vs PCCB — Same Disease, Different Subunit (12-Feature Comparison)",
            "note":  "Gene panel (NGS) is MANDATORY — biochemistry alone CANNOT distinguish PCCA from PCCB. "
                     "Both cause identical Propionic Acidemia (#606054). NBS C3 elevation does not differentiate.",
            "comparison": [
                {"feature": "Gene",              "PCCA": "PCCA",                                "PCCB": "PCCB"},
                {"feature": "Chromosome",        "PCCA": "13q32.3",                             "PCCB": "3q22.3"},
                {"feature": "Protein size",      "PCCA": "728 aa",                              "PCCB": "559 aa"},
                {"feature": "Domain",            "PCCA": "BC + BCCP (Step 1)",                  "PCCB": "CT — Carboxyl Transferase (Step 2)"},
                {"feature": "OMIM Gene",         "PCCA": "*232000",                             "PCCB": "*232050"},
                {"feature": "OMIM Disease",      "PCCA": "#606054 (PA)",                        "PCCB": "#606054 (PA — identical disease)"},
                {"feature": "Biomarkers",        "PCCA": "C3↑ methylcitrate↑ NH3↑ (identical)", "PCCB": "C3↑ methylcitrate↑ NH3↑ (identical)"},
                {"feature": "NBS",               "PCCA": "C3 elevated (cannot distinguish)",    "PCCB": "C3 elevated (cannot distinguish)"},
                {"feature": "Treatments",        "PCCA": "Protein restrict / carnitine / IV glucose (identical)", "PCCB": "Identical to PCCA — same enzyme, same disease"},
                {"feature": "Biotin response",   "PCCA": "NOT effective",                       "PCCB": "NOT effective"},
                {"feature": "VPA",               "PCCA": "ABSOLUTE CI",                         "PCCB": "ABSOLUTE CI"},
                {"feature": "Founder variants",  "PCCA": "p.Arg410Trp (European), c.IVS20+2T>C (Saudi)", "PCCB": "p.Glu168Lys (Dutch), c.1218_1231del14 (Spanish), c.IVS12+1G>A (Saudi)"},
            ],
        },
        "high_risk_situations": [
            {
                "situation": "VPA (Valproate)",
                "risk":      "ABSOLUTE CI",
                "detail":    "Valproyl-CoA directly inhibits PCCB CT domain active site AND depletes carnitine. "
                             "Propionyl-CoA surge → acute metabolic crisis → potentially FATAL. "
                             "NEVER prescribe VPA in any patient with Propionic Acidemia (PCCA or PCCB).",
            },
            {
                "situation": "Fasting",
                "risk":      "EXTREME HAZARD",
                "detail":    "Catabolism of Ile, Val, Met, Thr → propionyl-CoA surge; OAA depletion → "
                             "hyperammonemia. Emergency IV glucose (GIR 8-12) MANDATORY. "
                             "Never allow prolonged fasting — even pre-operative NPO triggers crisis.",
            },
            {
                "situation": "Intercurrent illness / fever",
                "risk":      "EXTREME HAZARD",
                "detail":    "Catabolic stress → propionyl-CoA surge identical to fasting. Emergency protocol: "
                             "IV glucose + STOP PROTEIN ± ammonia scavengers. Holter monitoring during illness "
                             "(QT prolongation risk). Every family needs an emergency letter.",
            },
            {
                "situation": "High-protein diet / BCAA supplement",
                "risk":      "EXTREME HAZARD",
                "detail":    "Ile, Val, Met, Thr are direct propionyl-CoA precursors. High protein intake "
                             "overwhelms residual PCC activity → propionyl-CoA accumulation → crisis.",
            },
            {
                "situation": "Biotin supplementation",
                "risk":      "NOT EFFECTIVE",
                "detail":    "PCCB encodes the enzyme beta subunit itself — not a biotin recycling or "
                             "ligation defect. Biotin will not improve PCCB CT domain function. "
                             "Distinguish from HLCS/BTD where biotin IS effective.",
            },
            {
                "situation": "Surgery without metabolic cover",
                "risk":      "EXTREME HAZARD",
                "detail":    "Catabolic stress of surgery + NPO protocol → propionyl-CoA surge. "
                             "IV glucose (GIR ≥ 6) perioperatively. Anaesthesia team must be briefed.",
            },
        ],
    }


def get_breakdown() -> dict:
    patients = _build_patients()
    return {
        "biomarkers": [
            {
                "name":         "C3 (propionylcarnitine)",
                "normal":       "< 3.5 µmol/L (NBS)",
                "pa_range":     "5–50+ µmol/L (classic); 3.5–20 (late-onset)",
                "significance": "PRIMARY NBS TRIGGER — cannot distinguish PCCA vs PCCB by NBS",
                "method":       "Tandem MS/MS (dried blood spot)",
            },
            {
                "name":         "C3/C2 ratio",
                "normal":       "< 0.25",
                "pa_range":     "0.3–3.0 (PA range)",
                "significance": "NBS secondary marker — improves specificity for PA",
                "method":       "Tandem MS/MS",
            },
            {
                "name":         "Methylcitrate (urine OA)",
                "normal":       "< 5 µmol/mmolCr",
                "pa_range":     "50–2,000 µmol/mmolCr — PATHOGNOMONIC",
                "significance": "Most specific PA marker; absent in MMA (KEY NEGATIVE vs MMUT)",
                "method":       "GC-MS urine organic acids",
            },
            {
                "name":         "3-Hydroxypropionate (urine OA)",
                "normal":       "< 10 µmol/mmolCr",
                "pa_range":     "50–1,500 µmol/mmolCr",
                "significance": "Secondary PA marker; present in HLCS/BTD but context differs",
                "method":       "GC-MS urine organic acids",
            },
            {
                "name":         "Propionylglycine (urine OA)",
                "normal":       "< 5 µmol/mmolCr",
                "pa_range":     "50–800 µmol/mmolCr",
                "significance": "Supports PA diagnosis; co-elevated with methylcitrate",
                "method":       "GC-MS urine organic acids",
            },
            {
                "name":         "Ammonia (plasma)",
                "normal":       "< 80 µmol/L",
                "pa_range":     "150–2,000 µmol/L (acute crisis)",
                "significance": "Secondary hyperammonemia via NAGS-CPS1 inhibition by propionyl-CoA",
                "method":       "Plasma ammonia (STAT)",
            },
            {
                "name":         "Free carnitine (plasma)",
                "normal":       "25–50 µmol/L",
                "pa_range":     "< 20 µmol/L (secondary depletion)",
                "significance": "Secondary carnitine depletion — propionylcarnitine formation depletes free carnitine",
                "method":       "LC-MS/MS acylcarnitine profile",
            },
            {
                "name":         "Methylmalonate (urine)",
                "normal":       "< 5 µmol/mmolCr",
                "pa_range":     "NORMAL (< 5) — KEY NEGATIVE vs MMA/MMUT",
                "significance": "Absent in PA = confirms not MMA; MMUT intact in PA",
                "method":       "GC-MS urine organic acids",
            },
            {
                "name":         "Biotinidase activity",
                "normal":       "> 30% activity",
                "pa_range":     "NORMAL — KEY NEGATIVE vs BTD",
                "significance": "Normal biotinidase confirms not BTD (biotin recycling intact in PA)",
                "method":       "Fluorometric enzyme assay",
            },
            {
                "name":         "C5-OH (3-OH-isovalerylcarnitine)",
                "normal":       "< 0.5 µmol/L",
                "pa_range":     "NORMAL — KEY NEGATIVE vs HLCS/BTD/MCD",
                "significance": "Normal C5-OH confirms no MCC block (MCC block = MCD, not isolated PA)",
                "method":       "Tandem MS/MS acylcarnitine",
            },
            {
                "name":         "BCAA (Leu + Ile + Val plasma)",
                "normal":       "Leu 60–160; Ile 30–90; Val 100–280 µmol/L",
                "pa_range":     "NORMAL — KEY NEGATIVE vs DLD/MSUD",
                "significance": "Normal BCAA confirms BCKDH intact; BCKDH not blocked in isolated PA",
                "method":       "Plasma amino acids (LC-MS/MS)",
            },
        ],
        "key_variants": [
            {
                "variant": "p.Glu168Lys",
                "cdna":    "c.502G>A",
                "domain":  "CT domain — propionyl-CoA binding interface",
                "severity": "Moderate",
                "note":    "Dutch/Belgian/Northern European founder; ~30% residual CT activity; late-onset or intermediate phenotype possible",
            },
            {
                "variant": "c.1218_1231del14",
                "cdna":    "p.Phe406Leufs*6",
                "domain":  "CT domain — frameshift null",
                "severity": "Severe",
                "note":    "Spanish/Portuguese/Latin American founder; frameshift → premature stop → NMD; classic neonatal severe",
            },
            {
                "variant": "c.IVS12+1G>A",
                "cdna":    "c.1283+1G>A",
                "domain":  "Splice donor — CT domain exon 12/13 boundary",
                "severity": "Severe",
                "note":    "Saudi Arabian/Middle Eastern founder; splice donor abolition → exon 12 skipping → null; classic neonatal",
            },
            {
                "variant": "p.Arg512Cys",
                "cdna":    "c.1534C>T",
                "domain":  "CT domain — catalytic Arg512 (CO2 transfer residue)",
                "severity": "Severe",
                "note":    "Pan-European; Arg512 directly activates CO2 transfer in CT reaction; null enzyme activity",
            },
            {
                "variant": "p.Arg399Cys",
                "cdna":    "c.1195C>T",
                "domain":  "CT domain — PCCA/PCCB subunit interface",
                "severity": "Severe",
                "note":    "Pan-European; disrupts (αβ)₆ dodecamer assembly; PCCA binding impaired",
            },
            {
                "variant": "p.Gly441Arg",
                "cdna":    "c.1321G>A",
                "domain":  "CT domain — inner hydrophobic core",
                "severity": "Severe",
                "note":    "European; Gly→Arg introduces steric clash in CT β-helix; severe misfolding",
            },
            {
                "variant": "p.Val145Glu",
                "cdna":    "c.434T>A",
                "domain":  "CT N-terminal subdomain",
                "severity": "Moderate",
                "note":    "Mild/moderate; reduced propionyl-CoA binding affinity; partial CT activity retained",
            },
            {
                "variant": "p.Ile476Thr",
                "cdna":    "c.1427T>C",
                "domain":  "CT domain — substrate channel lining",
                "severity": "Moderate",
                "note":    "Intermediate phenotype; altered substrate channel geometry; ~20% residual activity",
            },
        ],
        "seizure_types": [
            {"type": "Myoclonic", "pct": 55, "note": "Metabolic myoclonus — NH3-driven cortical irritability; resolves with NH3 control"},
            {"type": "Focal with secondary generalisation", "pct": 42, "note": "BG infarct-related; 25-35% have BG stroke-like episodes"},
            {"type": "Tonic-clonic (generalised)", "pct": 38, "note": "Crisis-triggered; correlates with acute decompensation NH3 surge"},
            {"type": "Infantile spasms (West syndrome)", "pct": 30, "note": "5-9 months; propionyl-CoA mitochondrial toxicity → hypsarrhythmia"},
            {"type": "Epileptic encephalopathy", "pct": 22, "note": "BG + white matter injury during decompensation; HSP-resistant"},
            {"type": "Absence-like episodes", "pct": 18, "note": "Subclinical NH3 elevation; normalise with metabolic control"},
        ],
        "metabolic_triggers": [
            {
                "trigger":    "Fasting / catabolism",
                "pct":        80,
                "mechanism":  "Ile/Val/Met/Thr catabolism → propionyl-CoA surge; OAA depletion → NH3 rise",
            },
            {
                "trigger":    "Intercurrent illness / fever",
                "pct":        75,
                "mechanism":  "Catabolic stress identical to fasting; fever increases propionyl-CoA production ~2-3x",
            },
            {
                "trigger":    "High-protein intake",
                "pct":        65,
                "mechanism":  "Direct Ile/Val/Met/Thr propionyl-CoA precursors; overwhelms residual PCCB CT activity",
            },
            {
                "trigger":    "VPA administration",
                "pct":        20,
                "mechanism":  "Valproyl-CoA inhibits CT domain + depletes carnitine → propionyl-CoA crisis; ABSOLUTE CI",
            },
            {
                "trigger":    "Surgery / perioperative NPO",
                "pct":        35,
                "mechanism":  "Combined fasting + surgical catabolism; IV glucose GIR ≥ 6 mandatory perioperatively",
            },
        ],
        "high_risk_drugs": [
            {
                "drug":      "VPA (Valproate / Valproic Acid)",
                "risk":      "ABSOLUTE CI",
                "mechanism": "Valproyl-CoA directly inhibits PCCB CT domain active site + secondary carnitine depletion → FATAL crisis. "
                             "Alternative AEDs: LEV, LTG, TPM, PB.",
            },
            {
                "drug":      "Raw egg white (dietary avidin)",
                "risk":      "EXTREME HAZARD",
                "mechanism": "Avidin binds intestinal biotin → biotin depletion → reduces what little residual PCC activity depends on biotin cofactor. "
                             "Cook eggs; no raw egg consumption.",
            },
            {
                "drug":      "Methotrexate / folate antagonists",
                "risk":      "CAUTION",
                "mechanism": "Can exacerbate propionate metabolism dysfunction; monitor OA if used",
            },
            {
                "drug":      "Propofol (lipid emulsion)",
                "risk":      "CAUTION",
                "mechanism": "Medium-chain lipids may contribute propionate; use with metabolic cover",
            },
        ],
        "treatments": [
            {
                "treatment":    "Protein restriction (low Ile/Val/Met/Thr diet)",
                "evidence":     "Level A",
                "response_pct": 85,
                "note":         "Lifelong. Ile, Val, Met, Thr are direct propionyl-CoA precursors. "
                                "PCCB/PCCA-free synthetic formula provides non-propionogenic amino acids.",
            },
            {
                "treatment":    "PCCB/PCCA-free amino acid formula",
                "evidence":     "Level A",
                "response_pct": 85,
                "note":         "Provides non-propionogenic amino acids while avoiding Ile/Val/Met/Thr excess. Lifelong.",
            },
            {
                "treatment":    "L-Carnitine (100 mg/kg/day oral)",
                "evidence":     "Level A",
                "response_pct": 90,
                "note":         "Conjugates propionyl-CoA → propionylcarnitine (C3) for renal excretion; "
                                "corrects secondary carnitine depletion. Monitor free carnitine target 25-50 µmol/L.",
            },
            {
                "treatment":    "IV glucose (GIR 8-12) + insulin (acute crisis)",
                "evidence":     "Level A",
                "response_pct": 88,
                "note":         "First-line acute intervention: suppresses catabolism → halts propionyl-CoA surge. "
                                "Give BEFORE ammonia scavengers if NH3 < 200 µmol/L.",
            },
            {
                "treatment":    "Ammonia scavengers (Na-benzoate / Na-phenylacetate)",
                "evidence":     "Level A",
                "response_pct": 82,
                "note":         "IV Na-benzoate 250 mg/kg loading over 2h then 250 mg/kg/24h maintenance; "
                                "Na-phenylacetate 250 mg/kg if NH3 > 200 µmol/L. Alternate nitrogen excretion pathway.",
            },
            {
                "treatment":    "Metronidazole (oral)",
                "evidence":     "Level B",
                "response_pct": 55,
                "note":         "Reduces gut bacterial propionate production by 20-30%; reduces C3 and methylcitrate. "
                                "Intermittent courses (7-10 days/month) to avoid resistance.",
            },
            {
                "treatment":    "Liver transplantation",
                "evidence":     "Level B",
                "response_pct": 75,
                "note":         "Corrects ~75-80% of hepatic PCC activity. Reduces metabolic crises significantly. "
                                "DOES NOT PROTECT AGAINST CARDIAC COMPLICATIONS — extra-hepatic PCCB deficiency persists "
                                "(heart, muscle). Cardiac monitoring mandatory post-transplant.",
            },
            {
                "treatment":    "HD/CRRT (acute NH3 > 500 µmol/L)",
                "evidence":     "Level B",
                "response_pct": 80,
                "note":         "Fastest NH3 clearance when NH3 > 500 or rapidly rising; use HD over CRRT for speed. "
                                "Bridge to metabolic control.",
            },
            {
                "treatment":    "Biotin supplementation",
                "evidence":     "NOT EFFECTIVE",
                "response_pct": 0,
                "note":         "PCCB encodes the CT domain enzyme subunit — not a biotin ligation or recycling defect. "
                                "Biotin will not improve CT domain function. Unlike HLCS/BTD where biotin IS the treatment.",
            },
            {
                "treatment":    "VPA (Valproate)",
                "evidence":     "ABSOLUTE CI",
                "response_pct": 0,
                "note":         "Valproyl-CoA directly inhibits PCC CT domain + causes carnitine depletion. "
                                "FATAL in Propionic Acidemia. Alternative AEDs: LEV (first-line), LTG, TPM, PB.",
            },
        ],
        "patient_sample": patients[:15],
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene":                "PCCB",
            "Full name":           "Propionyl-CoA Carboxylase Beta Subunit",
            "Chromosome":          "3q22.3",
            "Inheritance":         "Autosomal Recessive (AR) — biallelic LOF required",
            "OMIM Gene":           "*232050",
            "OMIM Disease":        "#606054 (Propionic Acidemia — identical to PCCA disease)",
            "Protein":             "559 amino acids; CT (Carboxyl Transferase) domain",
            "Complex":             "(αβ)₆ dodecamer: 6 PCCA + 6 PCCB in mitochondrial matrix",
            "Domain function":     "CT domain accepts carboxybiotin from PCCA-Lys669 swinging arm; "
                                   "transfers CO2 to propionyl-CoA → D-methylmalonyl-CoA (Step 2)",
            "Cofactor":            "None additional (uses activated CO2 from PCCA Step 1)",
            "Expression":          "Ubiquitous; high in liver, heart, skeletal muscle, kidney",
            "Prevalence":          "~1:100,000–150,000 (EU); ~1:2,000–5,000 (Arabian Peninsula); ~1:3,000 (Israel)",
            "Founder variants":    "p.Glu168Lys (Dutch/Belgian); c.1218_1231del14 (Spanish/Portuguese); c.IVS12+1G>A (Saudi)",
        },
        "key_concepts": [
            {
                "concept":     "Why PCCB and PCCA cause IDENTICAL disease despite being different genes",
                "explanation": "PCCA and PCCB are both obligatory subunits of the same (αβ)₆ dodecameric PCC complex. "
                               "Loss of either subunit abolishes the complete two-step propionyl-CoA carboxylation reaction. "
                               "PCCA LOF = no step 1 (no carboxybiotin generated); PCCB LOF = no step 2 (carboxybiotin generated "
                               "but cannot transfer CO2 to propionyl-CoA). Both result in propionyl-CoA accumulation with "
                               "identical downstream cascades: methylcitrate, hyperammonemia, cardiomyopathy, BG infarcts. "
                               "Gene panel is MANDATORY — biochemistry alone cannot distinguish PCCA from PCCB.",
            },
            {
                "concept":     "Why PCCB CT domain specifically binds propionyl-CoA (substrate specificity)",
                "explanation": "The CT domain of PCCB contains the propionyl-CoA binding pocket with a conserved "
                               "Arg512 residue that positions propionyl-CoA for nucleophilic attack by carboxybiotin. "
                               "This is the step that defines the enzyme's substrate specificity — propionyl-CoA vs "
                               "other acyl-CoAs. p.Arg512Cys abolishes CO2 transfer entirely. p.Glu168Lys reduces "
                               "propionyl-CoA binding affinity (~70% reduction) while retaining partial activity, "
                               "explaining milder phenotype in Dutch/Belgian founders.",
            },
            {
                "concept":     "Why biotin is NOT effective in PCCB deficiency (critical differential from HLCS/BTD)",
                "explanation": "HLCS and BTD deficiencies respond dramatically to biotin because the enzyme (HLCS or BTD) "
                               "that HANDLES biotin is defective — biotin supplementation bypasses the defect. "
                               "In PCCB, the biotin ligation machinery (HLCS) and biotin recycling (BTD) are INTACT. "
                               "PCCB encodes the CT domain of PCC — the catalytic subunit that performs propionyl-CoA "
                               "carboxylation. No amount of biotin will rescue a missing or misfolded CT domain.",
            },
            {
                "concept":     "Methylcitrate as PATHOGNOMONIC marker — why absent in MMA (critical differential)",
                "explanation": "Methylcitrate (2-methylcitrate) is formed by propionyl-CoA + oxaloacetate via aconitase, "
                               "not by mutase. In MMA (MMUT deficiency), the block is downstream of PCC: methylmalonyl-CoA "
                               "ACCUMULATES (not propionyl-CoA). Methylcitrate requires propionyl-CoA as substrate — in MMA, "
                               "propionyl-CoA is efficiently converted to methylmalonyl-CoA by intact PCC (PCCA+PCCB both normal), "
                               "so methylcitrate is NOT elevated in MMA. Elevated methylcitrate = PA (PCCA or PCCB); "
                               "elevated methylmalonate = MMA. The KEY NEGATIVE: no methylmalonate in PA.",
            },
            {
                "concept":     "VPA absolute contraindication — mechanism in PCCB disease",
                "explanation": "Valproate (VPA) is metabolised to valproyl-CoA. Valproyl-CoA is a structural analogue of "
                               "propionyl-CoA and DIRECTLY inhibits the PCCB CT domain active site (competitive inhibition). "
                               "Additionally, valproyl-CoA sequesters CoA and carnitine, causing secondary depletion. "
                               "In a patient with already-deficient PCCB, VPA administration can precipitate acute "
                               "metabolic decompensation with propionyl-CoA surge, NH3 crisis, and cardiac arrhythmia. "
                               "Alternative AEDs: LEV (first-line for generalised/focal), LTG, TPM, PB.",
            },
            {
                "concept":     "Why liver transplant does NOT prevent cardiac complications in PCCB disease",
                "explanation": "The liver contains ~75-80% of total body PCC activity. Liver transplantation restores "
                               "hepatic propionyl-CoA carboxylation and dramatically reduces metabolic crises and "
                               "hyperammonemia. However, cardiac muscle, skeletal muscle, and brain also express PCCB. "
                               "After liver transplant, extra-hepatic PCCB deficiency persists — propionyl-CoA still "
                               "accumulates in myocardium, inhibiting mitochondrial respiratory chain. Dilated cardiomyopathy "
                               "and QT prolongation risk remain post-transplant. Holter and echo monitoring are lifelong "
                               "regardless of transplant status.",
            },
            {
                "concept":     "Secondary hyperammonemia mechanism in PCCB disease (NAGS-CPS1 pathway)",
                "explanation": "Propionyl-CoA accumulation inhibits N-acetylglutamate synthase (NAGS) by acting as a "
                               "competitive substrate for the NAGS acyl-CoA binding site. Reduced NAGS activity → "
                               "decreased N-acetylglutamate (NAG) production → NAG is the obligatory allosteric "
                               "activator of carbamoyl phosphate synthetase 1 (CPS1). CPS1 downregulation impairs "
                               "the first step of the urea cycle → ammonia accumulates (150–2000 µmol/L in crisis). "
                               "This is SECONDARY hyperammonemia — treat PA first, not primary urea cycle disorder.",
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "C3 (propionylcarnitine) NBS",          "threshold": "> 3.5 µmol/L",          "action": "Immediate urine OA + plasma AA + C3/C2 ratio; start emergency protocol if symptomatic"},
            {"parameter": "C3/C2 ratio NBS",                      "threshold": "> 0.25",                 "action": "Supports PA diagnosis; order confirmatory urine OA"},
            {"parameter": "Methylcitrate (urine OA)",             "threshold": "> 50 µmol/mmolCr",      "action": "PATHOGNOMONIC for PA — confirm with gene panel (PCCA + PCCB)"},
            {"parameter": "Methylmalonate (urine OA)",            "threshold": "ABSENT (< 5 µmol/mmolCr)", "action": "KEY NEGATIVE — confirms PA, rules out MMA/MMUT"},
            {"parameter": "Ammonia",                              "threshold": "> 200 µmol/L",           "action": "Add ammonia scavengers; > 500 µmol/L → HD/CRRT"},
            {"parameter": "Free carnitine",                       "threshold": "< 20 µmol/L",            "action": "Increase L-carnitine dose; target 25-50 µmol/L"},
            {"parameter": "PCC enzyme activity (fibroblasts)",    "threshold": "< 10% of control",       "action": "Confirms PCC deficiency; cannot distinguish PCCA vs PCCB — gene panel required"},
            {"parameter": "Gene panel (PCCA + PCCB sequencing)",  "threshold": "Biallelic pathogenic variants in PCCB", "action": "Confirms PCCB diagnosis; mandatory for genetic counselling and liver transplant planning"},
            {"parameter": "Echocardiogram",                       "threshold": "Dilated CM or EF < 55%", "action": "Cardiology referral; L-carnitine optimisation; cardiac transplant evaluation if severe"},
            {"parameter": "Holter monitor",                       "threshold": "QTc > 480 ms",           "action": "Electrophysiology referral; beta-blocker consideration; ICD if QTc > 500 ms"},
        ],
        "differential_diagnosis": [
            {
                "disease":        "PCCA (Propionic Acidemia Type A)",
                "distinguishing": "Biochemically IDENTICAL — same C3, methylcitrate, NH3, treatments. "
                                  "Gene panel (PCCA mutation) is the ONLY way to distinguish. Same disease.",
            },
            {
                "disease":        "MMUT (Methylmalonic Acidemia)",
                "distinguishing": "MMA: methylmalonate ELEVATED (KEY POSITIVE); methylcitrate absent or minimal. "
                                  "PA: methylcitrate ELEVATED (PATHOGNOMONIC); methylmalonate absent (KEY NEGATIVE). "
                                  "Both elevate C3 — OA profile is the critical differentiator.",
            },
            {
                "disease":        "HLCS (Multiple Carboxylase Deficiency, neonatal)",
                "distinguishing": "HLCS: C5-OH elevated (MCC block) + C3 + 3-OH-isovalerate. Biotin DRAMATIC response. "
                                  "Biotinidase NORMAL. PA: NO C5-OH; biotin NOT effective. "
                                  "HLCS block is upstream of PCC (HLCS fails to biotinylate PCC); PA block is PCC itself.",
            },
            {
                "disease":        "BTD (Biotinidase Deficiency, MCD)",
                "distinguishing": "BTD: biotinidase activity DEFICIENT (PRIMARY diagnostic). Biotin responsive. "
                                  "C5-OH elevated. PA: biotinidase NORMAL; biotin NOT effective; NO C5-OH.",
            },
            {
                "disease":        "DLD (E3 subunit deficiency)",
                "distinguishing": "DLD: BCAA elevated (BCKDH block) + 2-hydroxyglutarate (αKGDH block) + "
                                  "lactate (PDH block) — four complex simultaneous block. "
                                  "PA: BCAA NORMAL; 2-HG NORMAL; no multi-complex block.",
            },
            {
                "disease":        "MSUD (BCKDHA/B/DBT)",
                "distinguishing": "MSUD: alloisoleucine PATHOGNOMONIC; BCAA dramatically elevated; "
                                  "no methylcitrate; no C3 elevation. PA: no alloisoleucine; BCAA NORMAL; "
                                  "methylcitrate PATHOGNOMONIC; C3 elevated.",
            },
            {
                "disease":        "Isolated MCC deficiency",
                "distinguishing": "MCC: C5-OH + 3-methylcrotonylglycine elevated; NO methylcitrate; NO C3. "
                                  "PA: methylcitrate + C3 elevated; NO C5-OH.",
            },
        ],
    }


def _build_patients() -> list[dict]:
    patients = []
    for i in range(1, 41):
        pheno    = _phenotype(i)
        c3       = _c3(pheno)
        mc       = _methylcitrate(pheno)
        nh3      = _ammonia(pheno)
        carn     = _carnitine(pheno)
        three_oh = _three_oh_prop(pheno)
        severe   = pheno == "Classic Neonatal Severe"
        late     = pheno == "Late-Onset Episodic"

        patients.append({
            "id":                           _pid(i),
            "sex":                          _sex(i),
            "phenotype":                    pheno,
            "onset_age_months":             random.randint(0, 1) if severe else (random.randint(3, 36) if late else random.randint(1, 12)),
            "c3_umol_l":                    c3,
            "methylcitrate_umol_mmolCr":    mc,
            "three_oh_propionate_umol_mmolCr": three_oh,
            "ammonia_umol_l":               nh3,
            "free_carnitine_umol_l":        carn,
            "seizures":                     severe or (random.random() < 0.6 if late else random.random() < 0.45),
            "cardiomyopathy":               severe and random.random() < 0.45 or (not severe and random.random() < 0.15),
            "qt_prolonged":                 random.random() < (0.10 if severe else 0.05),
            "bg_infarct":                   severe and random.random() < 0.35 or (late and random.random() < 0.20),
            "neutropenia":                  severe and random.random() < 0.42 or (not severe and random.random() < 0.12),
            "hypoglycemia":                 severe and random.random() < 0.48 or (not severe and random.random() < 0.18),
            "nbs_detected":                 random.random() < 0.78,
            "liver_transplant":             severe and random.random() < 0.18,
            "genotype":                     _genotype(i),
        })
    return patients
