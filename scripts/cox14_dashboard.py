#!/usr/bin/env python3
"""COX14 — Complex IV (COX) Deficiency Nuclear Type 6 (COXPD6).

COX14 (also known as C12orf62) is a nuclear-encoded mitochondrial inner
membrane protein that is essential for the early co-translational assembly
of MT-CO1 (COX1), the catalytic core subunit of Complex IV.

  COX14 gene           OMIM *614478
  Disease (COXPD6)     OMIM #614749
  Complex IV (COX)     deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX14 / early MT-CO1 co-translational assembly factor):
COX14 is a 66-amino acid protein with a single N-terminal transmembrane helix
anchored in the inner mitochondrial membrane (IMM), with its C-terminus facing
the intermembrane space (IMS). It functions as a critical component of the
MITRAC (Mitochondrial Translation Regulation Assembly intermediate of
Cytochrome c oxidase) complex:

  • COX14, together with COA3 (MITRAC12), binds to newly synthesised
    MT-CO1 (COX1) immediately after its release from the mitoribosome.
  • This early MITRAC complex stabilises the nascent MT-CO1 polypeptide,
    preventing its premature degradation by the IMM quality-control
    proteases (YME1L / m-AAA / AFG3L2).
  • Without COX14 stabilisation, MT-CO1 is rapidly degraded → CIV
    assembly fails at its earliest step → Complex IV is severely depleted
    (<5% of normal) across all tissues.
  • COX20 (COXPD8) plays an analogous role in the early assembly of
    MT-CO2 — a closely related but distinct MITRAC-class assembly factor.

KEY CLINICAL DIFFERENTIATOR vs. OTHER COX ASSEMBLY FACTOR DISEASES:
  • NO HCM           — KEY DDx vs SCO2 (100%), COX15 (78%), COA6 (90%)
  • NO hepatopathy   — KEY DDx vs SCO1 (100% neonatal hepatic failure)
  • NO tubulopathy   — KEY DDx vs COX10 (65% Fanconi syndrome)
  • NO anaemia       — KEY DDx vs COX10 (80%)
  • Severe Leigh / Leigh-like MRI (~80%) — similar to SURF1 but COX14 far rarer
  • Isolated COX deficiency <5% — one of the lowest residual activities
  • Psychomotor regression 100% — cardinal feature shared with SURF1/COX6B1
  • MITRAC early-assembly pathway — mechanistically distinct from haem
    pathway (COX10/COX15) and copper pathway (SCO1/SCO2/COA6)

MOLECULAR: Biallelic (AR) loss-of-function COX14 variants:
  — p.Arg15His (c.44G>A): TM helix Arg15; disrupts IMM insertion topology;
    first reported in Canadian patient (Weraarpachai 2012); most documented
    allele (~30%); severe LOF with <5% COX.
  — p.Val18Leu (c.52G>C): TM helix hydrophobic core packing; severe;
    destabilises IMM-anchoring of COX14; ~15%.
  — p.Gly47Arg (c.139G>C): C-terminal IMS-facing domain; disrupts
    COA3/MITRAC protein-protein interface; moderate-severe; ~20%.
  — p.Phe55Leu (c.165C>G): IMS domain; hydrophobic contact; ~10%.
  — Splice / null (truncating): severe; complete LOF; ~25%.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 617
DISEASE_ID   = "cox14"
DISEASE_NAME = "COX14 MITRAC-Assembly Complex IV Deficiency (COXPD6)"
GENE         = "COX14"
OMIM_GENE    = "*614478"
OMIM_DISEASE = "#614749"
CHROMOSOME   = "12q24.31"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal to early infantile (1–6 months; range birth–10 months)"
COHORT_SIZE  = 40
COLOR        = "#4527a0"   # deep purple — rare early-assembly, no cardinal organ
LIGHT        = "#ede7f6"


# ── Genotype pool ─────────────────────────────────────────────────────────────
GENO_ARG15HIS_HOM   = "p.Arg15His homozygous (c.44G>A) — TM helix IMM insertion"
GENO_ARG15HIS_CPX   = "p.Arg15His / p.Gly47Arg — compound heterozygous"
GENO_ARG15HIS_NULL  = "p.Arg15His / splice null — compound heterozygous"
GENO_GLY47ARG_CPX   = "p.Gly47Arg / p.Val18Leu — compound heterozygous"
GENO_NULL_CPX       = "Biallelic splice/truncating null — compound heterozygous"

GENO_POOL    = [GENO_ARG15HIS_HOM, GENO_ARG15HIS_CPX, GENO_ARG15HIS_NULL,
                GENO_GLY47ARG_CPX, GENO_NULL_CPX]
GENO_WEIGHTS = [0.30, 0.20, 0.15, 0.25, 0.10]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX14 cohort (seed-617)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COX14 cohort (seed-617).

    All patients have isolated COX deficiency; encephalopathy is cardinal.
    NO HCM, NO hepatopathy, NO renal tubulopathy.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno     = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex      = "F" if rng.random() < 0.50 else "M"
        is_null  = "null" in geno or "splice" in geno.lower() or "truncating" in geno

        onset_mo = round(rng.uniform(0.0, 2.5) if is_null else rng.uniform(1.0, 6.0), 1)
        lactate  = round(rng.uniform(5.0, 20.0) if is_null else rng.uniform(3.5, 14.0), 1)
        cox_pct  = round(rng.uniform(1.0, 4.0)  if is_null else rng.uniform(2.0, 8.0),  1)

        # Clinical features — encephalopathy cardinal; NO HCM / hepatopathy / tubulopathy
        has_leigh         = rng.random() < (0.90 if is_null else 0.75)
        has_regression    = True
        has_hypotonia     = rng.random() < 0.90
        has_seizures      = rng.random() < 0.45
        has_resp          = rng.random() < (0.75 if is_null else 0.60)
        has_growth_fail   = rng.random() < 0.70
        has_myopathy      = rng.random() < 0.65
        has_nystagmus     = rng.random() < 0.25
        has_optic_atr     = rng.random() < 0.30

        # Outcome
        if is_null:
            survived = rng.random() < 0.30
        else:
            survived = rng.random() < 0.55

        patients.append({
            "id":              f"COX14-{i:03d}",
            "sex":             sex,
            "genotype":        geno,
            "onset_mo":        onset_mo,
            "lactate_mM":      lactate,
            "cox_pct":         cox_pct,
            "leigh_mri":       has_leigh,
            "regression":      has_regression,
            "hypotonia":       has_hypotonia,
            "seizures":        has_seizures,
            "resp_compromise": has_resp,
            "growth_failure":  has_growth_fail,
            "myopathy":        has_myopathy,
            "nystagmus":       has_nystagmus,
            "optic_atrophy":   has_optic_atr,
            "hcm":             False,    # NEVER — KEY DDx
            "hepatopathy":     False,    # NEVER — KEY DDx
            "tubulopathy":     False,    # NEVER — KEY DDx
            "survived_5yr":    survived,
        })
    return patients


# ── Feature extraction ────────────────────────────────────────────────────────
def _cohort_features(cohort: list[dict]) -> dict:
    n = len(cohort)
    pct = lambda key: round(sum(1 for p in cohort if p[key]) / n * 100)
    avg = lambda key: round(sum(p[key] for p in cohort) / n, 1)

    return {
        "total":             n,
        "pct_female":        pct("sex") if False else round(sum(1 for p in cohort if p["sex"] == "F") / n * 100),
        "avg_onset_mo":      avg("onset_mo"),
        "avg_lactate":       avg("lactate_mM"),
        "avg_cox_pct":       avg("cox_pct"),
        "pct_leigh":         pct("leigh_mri"),
        "pct_regression":    pct("regression"),
        "pct_hypotonia":     pct("hypotonia"),
        "pct_seizures":      pct("seizures"),
        "pct_resp":          pct("resp_compromise"),
        "pct_growth_fail":   pct("growth_failure"),
        "pct_myopathy":      pct("myopathy"),
        "pct_nystagmus":     pct("nystagmus"),
        "pct_optic_atr":     pct("optic_atrophy"),
        "pct_hcm":           0,   # KEY DDx — always 0
        "pct_hepatopathy":   0,   # KEY DDx — always 0
        "pct_tubulopathy":   0,   # KEY DDx — always 0
        "pct_survived_5yr":  pct("survived_5yr"),
        "pct_deceased_5yr":  100 - pct("survived_5yr"),
    }


# ── Public API functions ──────────────────────────────────────────────────────
def get_overview() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)
    feat   = _cohort_features(cohort)

    return {
        # Gene / disease identity
        "gene":        GENE,
        "alias":       "C12orf62",
        "protein":     "COX14 — 66 aa, ~7.5 kDa, single N-terminal TM helix, IMS-facing C-terminus",
        "disease":     DISEASE_NAME,
        "omim_gene":   OMIM_GENE,
        "omim_disease":OMIM_DISEASE,
        "chromosome":  CHROMOSOME,
        "inheritance": INHERITANCE,
        "onset":       ONSET,
        "cohort":      f"{COHORT_SIZE} patients (seed-{SEED})",

        # KPI cards
        "kpis": [
            {"label": "COXPD type",         "value": "COXPD6"},
            {"label": "Protein size",        "value": "66 aa / 7.5 kDa"},
            {"label": "Assembly role",       "value": "MITRAC early MT-CO1"},
            {"label": "COX activity",        "value": f"<5% (avg {feat['avg_cox_pct']}%)"},
            {"label": "Onset",               "value": f"Avg {feat['avg_onset_mo']} mo"},
            {"label": "Lactate (avg)",       "value": f"{feat['avg_lactate']} mM"},
            {"label": "Leigh MRI",           "value": f"{feat['pct_leigh']}%"},
            {"label": "Regression",          "value": "100%"},
            {"label": "HCM",                 "value": "0% ⚑ KEY DDx"},
            {"label": "Hepatopathy",         "value": "0% ⚑ KEY DDx"},
            {"label": "Tubulopathy",         "value": "0% ⚑ KEY DDx"},
            {"label": "5-yr Survival",       "value": f"{feat['pct_survived_5yr']}%"},
        ],

        # Summary paragraph
        "summary": (
            f"COX14 (C12orf62) is a nuclear-encoded, 66-amino acid IMM-anchored protein "
            f"that forms part of the MITRAC early-assembly complex for MT-CO1 (COX1). "
            f"Loss-of-function variants (AR biallelic) cause Mitochondrial Complex IV "
            f"Deficiency, Nuclear Type 6 (COXPD6; OMIM #614749). COX14 deficiency is "
            f"extremely rare (<15 cases published worldwide) and presents with severe "
            f"isolated COX deficiency (<5% of normal), Leigh/Leigh-like encephalopathy, "
            f"lactic acidosis, and psychomotor regression. The cardinal biochemical "
            f"fingerprint — isolated COX deficiency with CI/CII/CIII normal — is shared "
            f"with SURF1, COX6B1, SCO2, SCO1, COX10, COX15, and COA6, making WES/WGS "
            f"mandatory. Key bedside differentiators: NO HCM (unlike SCO2 100%, COX15 "
            f"78%, COA6 90%), NO hepatopathy (unlike SCO1 100%), NO renal tubulopathy "
            f"(unlike COX10 65%). In this {COHORT_SIZE}-patient cohort (seed-{SEED}): "
            f"Leigh MRI {feat['pct_leigh']}%, hypotonia {feat['pct_hypotonia']}%, "
            f"respiratory compromise {feat['pct_resp']}%, seizures {feat['pct_seizures']}%, "
            f"5-year survival {feat['pct_survived_5yr']}%."
        ),

        # Clinical feature bars
        "feature_bars": [
            {"label": "Psychomotor regression", "value": 100},
            {"label": "Lactic acidosis",         "value": 95},
            {"label": "Hypotonia",               "value": feat["pct_hypotonia"]},
            {"label": "Leigh/Leigh-like MRI",    "value": feat["pct_leigh"]},
            {"label": "Respiratory compromise",   "value": feat["pct_resp"]},
            {"label": "Growth failure",           "value": feat["pct_growth_fail"]},
            {"label": "Myopathy",                "value": feat["pct_myopathy"]},
            {"label": "Seizures",                "value": feat["pct_seizures"]},
            {"label": "Optic atrophy",           "value": feat["pct_optic_atr"]},
            {"label": "Nystagmus",               "value": feat["pct_nystagmus"]},
            {"label": "HCM",                     "value": 0},
            {"label": "Hepatopathy",             "value": 0},
            {"label": "Renal tubulopathy",       "value": 0},
        ],

        # Absolute contraindications
        "absolute_ci": [
            {"drug": "VPA (valproate)",        "reason": "CoA sequestration + POLG inhibition — worsens COX depletion; hepatotoxicity risk elevated"},
            {"drug": "Metformin",              "reason": "Complex I inhibition → uncontrolled lactic acidosis; fatal in OXPHOS disease"},
            {"drug": "Propofol",              "reason": "PRIS — direct Complex IV inhibition; residual COX <5% → cardiac arrest risk"},
            {"drug": "Linezolid",             "reason": "mt-23S rRNA translation block → MT-CO1/CO2/CO3 synthesis abolished → residual COX eliminated"},
            {"drug": "Chloramphenicol",       "reason": "mt-ribosome block (same mechanism as linezolid); synergistic COX destruction"},
            {"drug": "Ketogenic diet (KD)",   "reason": "Beta-oxidation obligatorily requires functional Complex IV; KD risks fatal metabolic decompensation"},
        ],

        # Pathway context
        "pathway": {
            "name":   "MITRAC Early MT-CO1 Co-translational Assembly Pathway",
            "steps":  [
                "Mitoribosome synthesises MT-CO1 (COX1) polypeptide",
                "COX14 + COA3 (MITRAC12) bind nascent MT-CO1 immediately post-translation",
                "MITRAC complex stabilises MT-CO1; prevents YME1L/AFG3L2 degradation",
                "MT-CO1 module matures: COX4, COX5A, COX6A, COX6C, COX7A, COX7C added",
                "Haem a + CuB metalation (SURF1, COX10, COX15, COX11 pathway)",
                "MT-CO2 module joins (SCO1/SCO2/COA6 copper pathway for CuA site)",
                "MT-CO3 module and remaining subunits assembled → holocomplex CIV",
                "CIV dimerises; CIV homodimer integrates into respiratory supercomplex",
            ],
            "cox14_step": 2,
            "footnote": "COX14 acts at Step 2 — the earliest possible assembly checkpoint for MT-CO1.",
        },

        # Rapid diagnosis checklist
        "diagnosis_checklist": [
            "Isolated COX deficiency <5% (muscle/fibroblasts) — CI/CII/CIII normal",
            "Leigh / Leigh-like bilateral symmetric MRI (putamen + brainstem T2 bright)",
            "Psychomotor regression (100%) + severe lactic acidosis (lactate >3.5 mM)",
            "Hypotonia + respiratory compromise in neonate/infant",
            "NO HCM on ECHO — rules out SCO2, COX15, COA6 as primary diagnosis",
            "NO hepatopathy / elevated LFTs — rules out SCO1 as primary diagnosis",
            "NO renal Fanconi / aminoaciduria / phosphaturia — rules out COX10",
            "WES/WGS + COX14 (C12orf62) gene: biallelic variants → confirm COXPD6",
        ],
    }


def get_breakdown() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)

    # Genotype distribution
    geno_counts: dict[str, int] = {}
    for p in cohort:
        geno_counts[p["genotype"]] = geno_counts.get(p["genotype"], 0) + 1

    geno_dist = sorted(
        [{"genotype": g, "count": c, "pct": round(c / len(cohort) * 100)}
         for g, c in geno_counts.items()],
        key=lambda x: -x["count"],
    )

    # COX % by genotype
    geno_cox: dict[str, list] = {}
    for p in cohort:
        geno_cox.setdefault(p["genotype"], []).append(p["cox_pct"])
    geno_avg_cox = {g: round(sum(v) / len(v), 1) for g, v in geno_cox.items()}

    # Per-patient table (first 20 for display)
    patient_table = [
        {
            "id":          p["id"],
            "sex":         p["sex"],
            "genotype":    p["genotype"][:55] + "…" if len(p["genotype"]) > 55 else p["genotype"],
            "onset_mo":    p["onset_mo"],
            "lactate":     p["lactate_mM"],
            "cox_pct":     p["cox_pct"],
            "leigh":       "Yes" if p["leigh_mri"]       else "No",
            "hypotonia":   "Yes" if p["hypotonia"]        else "No",
            "seizures":    "Yes" if p["seizures"]         else "No",
            "resp":        "Yes" if p["resp_compromise"]  else "No",
            "hcm":         "No ⚑",
            "hepatopathy": "No ⚑",
            "tubulopathy": "No ⚑",
            "survived":    "Yes" if p["survived_5yr"]     else "No",
        }
        for p in cohort[:20]
    ]

    # DDx comparison table
    ddx_table = [
        {
            "gene":         "COX14 (this disease)",
            "locus":        "12q24.31",
            "disease":      "COXPD6",
            "hcm":          "0% (KEY DDx)",
            "hepatopathy":  "0% (KEY DDx)",
            "tubulopathy":  "0% (KEY DDx)",
            "leigh":        "~80%",
            "cox_defect":   "Isolated CIV (<5%) — lowest residual",
            "distinguisher":"MITRAC early MT-CO1 assembly; no cardinal organ; extremely rare",
        },
        {
            "gene":         "SURF1",
            "locus":        "9q34.2",
            "disease":      "Leigh (COXPD1)",
            "hcm":          "~10%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "95%",
            "cox_defect":   "Isolated CIV (<5%)",
            "distinguisher":"Most common Leigh COX cause; European founder c.845_846delCT; haem a3/CuB assembly step",
        },
        {
            "gene":         "COX6B1",
            "locus":        "19q13.3",
            "disease":      "COXPD7",
            "hcm":          "0% (KEY DDx)",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "70%",
            "cox_defect":   "Isolated CIV (<15%)",
            "distinguisher":"Structural subunit VIb1 (survives in assembled CIV); Turkish founder p.Trp38Ser; slightly higher residual",
        },
        {
            "gene":         "SCO2",
            "locus":        "22q13.33",
            "disease":      "COXPD2",
            "hcm":          "100% CARDINAL",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "55%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"HCM 100% — KEY DDx vs COX14 0% HCM; copper chaperone for MT-CO2 CuA",
        },
        {
            "gene":         "SCO1",
            "locus":        "17p13.2",
            "disease":      "COXPD3",
            "hcm":          "<5%",
            "hepatopathy":  "100% CARDINAL",
            "tubulopathy":  "0%",
            "leigh":        "45%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"Neonatal hepatic failure 100% — KEY DDx vs COX14 0% hepatopathy; copper pathway",
        },
        {
            "gene":         "COX10",
            "locus":        "17p12",
            "disease":      "COXPD4",
            "hcm":          "<5%",
            "hepatopathy":  "0%",
            "tubulopathy":  "65% Fanconi + anaemia 80%",
            "leigh":        "88%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"Renal tubulopathy 65% + anaemia 80% — KEY DDx vs COX14 0%; haem a (step 1)",
        },
        {
            "gene":         "COX15",
            "locus":        "10q24.2",
            "disease":      "COXPD5",
            "hcm":          "78%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "82%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"HCM 78% — KEY DDx vs COX14 0% HCM; haem a (step 2)",
        },
        {
            "gene":         "COA6",
            "locus":        "1q42.2",
            "disease":      "COXPD14",
            "hcm":          "90% CARDINAL",
            "hepatopathy":  "35%",
            "tubulopathy":  "0%",
            "leigh":        "30%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"HCM 90% + liver 35% — KEY DDx vs COX14 0%/0%; copper chaperone twin-CX9C",
        },
        {
            "gene":         "COX20",
            "locus":        "1p31.3",
            "disease":      "COXPD8",
            "hcm":          "<5%",
            "hepatopathy":  "0%",
            "tubulopathy":  "0%",
            "leigh":        "70%",
            "cox_defect":   "Isolated CIV (<10%)",
            "distinguisher":"MITRAC partner; COX20 = early MT-CO2 assembly vs COX14 = early MT-CO1 assembly; WES mandatory",
        },
    ]

    return {
        "cohort_size": COHORT_SIZE,
        "seed": SEED,
        "genotype_distribution": geno_dist,
        "genotype_avg_cox_pct": [
            {"genotype": g[:50] + "…" if len(g) > 50 else g, "avg_cox_pct": v}
            for g, v in sorted(geno_avg_cox.items(), key=lambda x: x[1])
        ],
        "patient_table": patient_table,
        "ddx_table": ddx_table,
        "absolute_ci_drugs": [
            {"drug": "VPA",          "mechanism": "CoA sequestration + POLG inhibition → worsens COX; hepatotoxicity"},
            {"drug": "Metformin",    "mechanism": "Complex I inhibitor → lactic acidosis, fatal in OXPHOS disease"},
            {"drug": "Propofol",     "mechanism": "PRIS: direct CIV inhibition; <5% residual COX → cardiac arrest"},
            {"drug": "Linezolid",    "mechanism": "mt-23S rRNA: blocks MT-CO1/CO2/CO3 synthesis → abolishes residual CIV"},
            {"drug": "Chloramphenicol", "mechanism": "mt-ribosome block (same mechanism as linezolid)"},
            {"drug": "Ketogenic diet",  "mechanism": "FAO requires CIV; KD risks fatal metabolic decompensation"},
        ],
        "treatment_ladder": [
            {"agent": "CoQ10 (ubiquinol)",   "dose": "10–30 mg/kg/day",      "level": "C", "note": "Electron shuttle bypass"},
            {"agent": "Riboflavin (B2)",     "dose": "100–400 mg/day",       "level": "C", "note": "CI/CIII cofactor support"},
            {"agent": "Thiamine (B1)",       "dose": "5–10 mg/kg/day",       "level": "C", "note": "MANDATORY empiric — exclude SLC19A3/BTD"},
            {"agent": "Biotin",              "dose": "5–20 mg/day",          "level": "C", "note": "MANDATORY empiric — exclude biotinidase deficiency"},
            {"agent": "L-carnitine",         "dose": "50–100 mg/kg/day",     "level": "C", "note": "Secondary carnitine deficiency prevention"},
            {"agent": "LEV (levetiracetam)", "dose": "20–60 mg/kg/day",      "level": "C", "note": "Preferred AED — no mito toxicity, renal excretion"},
            {"agent": "GIR 6–8 periop.",     "dose": "Glucose infusion rate", "level": "C", "note": "Never fast >4h; prevent catabolism and lactate surge"},
            {"agent": "NIV / BiPAP",         "dose": "Per respiratory team",  "level": "C", "note": "Respiratory compromise 60–75%"},
            {"agent": "Sevoflurane (NOT propofol)", "dose": "Inhaled anaesthetic", "level": "C", "note": "PRIS risk with propofol + <5% CIV is catastrophic"},
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_id": DISEASE_ID,
        "glossary": [
            {"term": "COX14",
             "definition": "Cytochrome c Oxidase Assembly Factor 14 — nuclear-encoded, 66 aa, ~7.5 kDa; single N-terminal TM helix anchored in IMM; C-terminus faces IMS; originally annotated C12orf62; component of the MITRAC early MT-CO1 co-translational assembly complex."},
            {"term": "C12orf62",
             "definition": "Original Open Reading Frame designation for COX14 on chromosome 12q24.31; the two names (COX14 / C12orf62) refer to the identical gene."},
            {"term": "COXPD6",
             "definition": "Mitochondrial Complex IV Deficiency, Nuclear Type 6 — the OMIM disease designation (#614749) for COX14-related COX deficiency. One of the rarest COXPD subtypes; <15 cases published as of 2024."},
            {"term": "MITRAC complex",
             "definition": "Mitochondrial Translation Regulation Assembly Intermediate of Cytochrome c oxidase — a multi-subunit complex including COX14, COA3 (MITRAC12), and COX20 that binds to newly synthesised mitoribosomal MT-CO1 and MT-CO2 immediately after translation. MITRAC prevents premature degradation of these nascent polypeptides and coordinates their folding and early module assembly."},
            {"term": "MT-CO1 (COX1)",
             "definition": "Mitochondrial-encoded COX subunit 1 (cytochrome c oxidase subunit I) — the catalytic core of Complex IV; contains the haem a, haem a3, and CuB metal centres that carry out oxygen reduction. COX14 is required for its early co-translational stabilisation."},
            {"term": "COA3 (MITRAC12)",
             "definition": "Cytochrome c Oxidase Assembly Factor 3 — the primary MITRAC binding partner of COX14; together they form a protective scaffold around newly synthesised MT-CO1. COA3 mutations cause COXPD10 (OMIM #614702)."},
            {"term": "COX20",
             "definition": "Cytochrome c Oxidase Assembly Factor 20 — an analogous early assembly factor for MT-CO2 (COX2); mutations cause COXPD8. COX14 and COX20 are both MITRAC-class factors but operate on different mtDNA-encoded COX subunits."},
            {"term": "Isolated COX deficiency",
             "definition": "Complex IV (COX) enzyme activity below the reference range while Complexes I, II, and III remain normal. The biochemical 'fingerprint' for COX14/SURF1/SCO1/SCO2/COX10/COX15/COA6/COX6B1 — requires WES/WGS to distinguish."},
            {"term": "MITRAC early MT-CO1 pathway",
             "definition": "The co-translational quality-control checkpoint (Step 2 of CIV assembly) where COX14 + COA3 bind to nascent MT-CO1 immediately post-ribosomal release. Without COX14, MT-CO1 is degraded by IMM quality-control proteases (YME1L, m-AAA/AFG3L2), blocking all downstream CIV assembly."},
            {"term": "YME1L / m-AAA / AFG3L2",
             "definition": "IMM quality-control proteases that degrade misfolded or unprotected IMM proteins. COX14 normally shields nascent MT-CO1 from these proteases; loss of COX14 exposes MT-CO1 to irreversible degradation."},
            {"term": "PRIS",
             "definition": "Propofol Infusion Syndrome — potentially fatal metabolic acidosis + rhabdomyolysis + cardiac arrest from propofol's direct Complex IV inhibition. With COX14 deficiency and <5% residual COX, even brief propofol exposure is catastrophic; sevoflurane must be used instead."},
            {"term": "TM helix (N-terminal)",
             "definition": "Transmembrane helix at the N-terminus of COX14 that anchors the protein in the IMM. Variants affecting the TM helix (e.g., p.Arg15His, p.Val18Leu) disrupt IMM insertion topology, mislocalise COX14, and abolish its MITRAC function."},
            {"term": "WES/WGS",
             "definition": "Whole Exome Sequencing / Whole Genome Sequencing — the mandatory diagnostic tool for COX14 and all rare COXPD subtypes. Biochemistry alone cannot distinguish COX14 from SURF1, COX6B1, SCO2, SCO1, COX10, COX15, or COA6 because all share isolated COX deficiency."},
            {"term": "Leigh syndrome",
             "definition": "Progressive necrotising encephalopathy characterised by bilateral symmetric T2 hyperintensities in the basal ganglia (putamen, caudate) and brainstem on MRI, with lactic acidosis and psychomotor regression. COX14 deficiency causes Leigh or Leigh-like MRI in ~80% of cases."},
        ],
        "clinical_notes": [
            "COX14 deficiency should be suspected in any neonate or infant with Leigh-like "
            "encephalopathy + isolated COX deficiency when: (a) NO HCM on ECHO, (b) NO "
            "hepatopathy on LFTs, (c) NO renal Fanconi on urine amino acids / phosphate. "
            "All three negative findings together make COX14/COX6B1/SURF1 the priority "
            "WES targets among COX assembly factor diseases.",

            "Biochemical workup: muscle + fibroblast respiratory chain enzyme assay — "
            "COX activity <5% of normal with CI/CII/CIII normal is the cardinal finding. "
            "Confirm by BN-PAGE + anti-COX1 immunoblot: COX holoenzyme absent; compare "
            "COX14 (C12orf62) protein band — absent or truncated in COXPD6.",

            "Empiric treatment while awaiting molecular result: thiamine 5–10 mg/kg/day "
            "MANDATORY (to exclude BTBGD/SLC19A3 Leigh mimic — curable) + biotin "
            "MANDATORY (BTD) + CoQ10 + riboflavin. Stop VPA immediately if inadvertently "
            "started. Avoid propofol categorically for any procedure.",

            "Anaesthesia protocol: sevoflurane inhalational (NOT propofol, NOT ketamine-only); "
            "dexmedetomidine is preferred for procedural sedation; GIR 6–8 mg/kg/min "
            "perioperatively; fasting >4h absolutely prohibited; lactic acidosis monitoring "
            "intra- and post-procedure.",

            "Prognosis: very poor with null/severe biallelic alleles (5-year survival ~30%). "
            "Milder missense compound heterozygotes may survive longer with supportive care. "
            "No disease-modifying therapy exists; gene therapy and pharmacological chaperones "
            "for MITRAC-class assembly factors are under early investigation.",

            "Genetic counselling: 25% recurrence risk per pregnancy for carrier parents. "
            "Prenatal diagnosis by molecular genetics available once proband variants "
            "confirmed. Newborn sibling screening with early empiric mitochondrial cocktail "
            "is recommended while awaiting genetic results.",
        ],
        "references": [
            {
                "citation": "Weraarpachai W et al. (2012). PLoS Genet 8(10):e1002951.",
                "note":     "First description of COX14 (C12orf62) pathogenic variants causing Complex IV deficiency with Leigh-like encephalopathy (COXPD6). Identified p.Arg15His as the founding variant.",
            },
            {
                "citation": "Mick DU et al. (2012). Cell Metab 16(4):449–460.",
                "note":     "Characterisation of the MITRAC complex; established COX14 and COA3 as early co-translational MT-CO1 assembly factors. Showed COX14 prevents MT-CO1 degradation by IMM proteases.",
            },
            {
                "citation": "Richter-Dennerlein R et al. (2016). Cell 167(4):1067–1080.e18.",
                "note":     "High-resolution analysis of MITRAC–ribosome interface; structural context for COX14 TM helix function and its role in MT-CO1 co-translational stabilisation.",
            },
            {
                "citation": "Dennerlein S & Rehling P (2015). FEBS Lett 589(14):1909–1919.",
                "note":     "Review of mt-protein synthesis and Complex IV assembly; COX14 pathway context relative to other COXPD assembly factors.",
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note":     "Comprehensive review of mitochondrial disease epidemiology, clinical spectrum, and management — framework for all COXPD subtypes.",
            },
        ],
        "inheritance_detail": (
            "COX14 (COXPD6) is autosomal recessive (AR). Both copies of the COX14 gene must "
            "carry pathogenic loss-of-function variants (biallelic) for disease to occur. "
            "Heterozygous carriers are clinically unaffected. Each pregnancy of two carrier "
            "parents carries a 25% risk of an affected child. Prenatal molecular diagnosis "
            "is available once biallelic variants are confirmed in the proband."
        ),
        "management_summary": (
            "No disease-modifying therapy exists for COXPD6. Management is supportive: "
            "(1) Mitochondrial cocktail: CoQ10, riboflavin, thiamine (MANDATORY empiric), "
            "biotin (MANDATORY empiric), L-carnitine. "
            "(2) Energy substrate: GIR 6–8 mg/kg/min perioperative; fasting >4h prohibited. "
            "(3) Seizures: levetiracetam preferred (no mito toxicity). "
            "(4) Respiratory: NIV/BiPAP for central/peripheral respiratory compromise. "
            "(5) Anaesthesia: sevoflurane inhalational ONLY — propofol absolutely contraindicated. "
            "(6) Absolute CI: VPA, propofol, metformin, linezolid, chloramphenicol, KD. "
            "(7) Multidisciplinary: neurology + metabolics + respiratory + nutrition mandatory. "
            "(8) Genetic counselling: 25% recurrence; prenatal diagnosis available."
        ),
    }
