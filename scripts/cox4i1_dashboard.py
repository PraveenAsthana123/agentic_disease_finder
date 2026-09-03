#!/usr/bin/env python3
"""COX4I1 — Exocrine Pancreatic Insufficiency / Dyserythropoietic Anemia / Calvarial Hyperostosis
Complex IV Deficiency Nuclear Type 12 (COXPD12).

COX4I1 (Cytochrome c Oxidase Subunit 4, Isoform 1) is a nuclear-encoded structural subunit
of cytochrome c oxidase (Complex IV), the terminal electron acceptor of the mitochondrial
respiratory chain.

  COX4I1 gene         OMIM *123864
  Disease (COXPD12)   OMIM #616501
  Complex IV (COX)    deficiency — isolated; Complexes I, II, III normal

PATHOPHYSIOLOGY (COX4I1 / Complex IV structural subunit):
COX4I1 encodes Subunit IV (historically Subunit 4) of Complex IV. The precursor protein
is 169 amino acids; after cleavage of the mitochondrial targeting sequence (MTS, ~22 aa),
the mature protein is ~147 aa (~17.2 kDa). COX4I1 is a MATRIX-FACING peripheral subunit:
it has no transmembrane helix and contacts the matrix face of the MT-CO1/MT-CO2 core.

  • COX4I1 is the ubiquitous isoform expressed in all tissues (normoxia).
    COX4I2 is the lung/hypoxia-specific isoform, induced by HIF-1α under low O₂.
  • COX4I1 assembles EARLY in CIV biogenesis — it joins the nascent MT-CO1 module
    (MITRAC intermediate) before heme a/a3 and copper metalation. Loss of COX4I1
    stalls early CIV biogenesis and severely reduces assembled CIV.
  • ATP ALLOSTERIC REGULATION: COX4I1 contains a regulatory ATP binding site on its
    matrix-facing surface. When ATP:ADP ratio is high, ATP binds COX4I1 and inhibits
    CIV activity (feedback regulation of respiration). This is UNIQUE to COX4I1 among
    CIV subunits and explains the multi-organ phenotype on ATP energy failure.
  • Isolated COX deficiency: ~15-30% residual CIV in fibroblasts / tissues.
    Complexes I, II, III remain NORMAL — this is the biochemical fingerprint.

KEY CLINICAL DIFFERENTIATOR vs. OTHER COX DEFICIENCY DISEASES (CRITICAL):
  • EXOCRINE PANCREATIC INSUFFICIENCY (EPl): PATHOGNOMONIC — not seen in ANY other COXPD.
    Steatorrhoea, fat-soluble vitamin deficiency (ADEK), malabsorption. PERT is mandatory.
  • DYSERYTHROPOIETIC ANAEMIA: PATHOGNOMONIC pattern — dysplastic erythroid precursors on
    bone marrow aspirate; different from normochromic anaemia in COX10 (Fanconi).
  • CALVARIAL HYPEROSTOSIS: UNIQUE radiographic finding — skull thickening on CT/X-ray;
    not seen in other mitochondrial CIV diseases.
  • NO HCM — KEY DDx vs. SCO2 (100%), COA5 (88%), COA6 (90%), COX15 (78%)
  • NO Leigh MRI — DOMINANT pattern absent: ~25% vs. SURF1 95%, COX14 80%, COX8A 85%
  • NO pure renal tubulopathy (Fanconi) — KEY DDx vs. COX10 (65% Fanconi + anaemia 80%)
  • NO ataxia — KEY DDx vs. COX20 (100% childhood cerebellar ataxia)
  • SEIZURES MODERATE (~38%) — NOT cardinal as in COX8A (85%) or COX6B1 (45%)

MOLECULAR: Biallelic (AR) loss-of-function COX4I1 variants:
  — Homozygous COX4I1 deletion (chr16q22 contiguous gene deletion):
    Original Israeli Bedouin family; complete LOF; PINDAC founder; ~35%.
  — p.Arg78Stop (c.232C>T): Nonsense truncation; matrix-domain loss; severe; ~20%.
  — p.Glu36Lys (c.106G>A): Near MTS cleavage site; impaired import/folding; severe; ~18%.
  — p.Gly108Arg (c.322G>C): Matrix-domain β-sheet core disruption; severe LOF; ~15%.
  — Biallelic splice/compound heterozygous null: complete LOF; ~12%.

ACRONYM: PINDAC — Pancreatic Insufficiency aNd Dyserythropoietic Anemia and Calvarial
hyperostosis — original phenotypic designation for COX4I1 deficiency.
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 641
DISEASE_ID   = "cox4i1"
DISEASE_NAME = "COX4I1 PINDAC — Exocrine Pancreatic Insufficiency / Dyserythropoietic Anaemia / Calvarial Hyperostosis (COXPD12)"
GENE         = "COX4I1"
OMIM_GENE    = "*123864"
OMIM_DISEASE = "#616501"
CHROMOSOME   = "16q22.1"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal to early infantile (prenatal IUGR common; postnatal onset birth–6 months)"
COHORT_SIZE  = 40
COLOR        = "#4a148c"   # deep purple — matrix-domain subunit / multi-organ phenotype
LIGHT        = "#f3e5f5"


# ── Genotype pool ─────────────────────────────────────────────────────────────
GENO_DEL_HOM  = "Homozygous COX4I1 deletion (16q22 contiguous) — Bedouin founder PINDAC"
GENO_R78X_HOM = "p.Arg78Stop homozygous (c.232C>T) — matrix-domain truncating"
GENO_E36K_CPX = "p.Glu36Lys / p.Gly108Arg — compound heterozygous"
GENO_G108R_CPX = "p.Gly108Arg / p.Arg78Stop — compound heterozygous"
GENO_NULL_CPX  = "Biallelic splice/compound null — complete LOF"

GENO_POOL    = [GENO_DEL_HOM, GENO_R78X_HOM, GENO_E36K_CPX, GENO_G108R_CPX, GENO_NULL_CPX]
GENO_WEIGHTS = [35, 20, 18, 15, 12]   # %


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient COX4I1 cohort (seed-641)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient COX4I1/COXPD12 cohort (seed-641).

    COX4I1 deficiency — isolated COX deficiency (15-30% residual).
    PINDAC phenotype: exocrine pancreatic insufficiency + dyserythropoietic anaemia
    + calvarial hyperostosis. NOT classic Leigh syndrome.
    NO HCM, NO Fanconi tubulopathy, NO cerebellar ataxia.
    Seizures moderate (~38%) — NOT cardinal as in COX8A.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"
        is_null = "null" in geno or "Stop" in geno or "deletion" in geno.lower()

        # Neonatal / early infantile onset; prenatal IUGR common
        onset_mo   = round(rng.uniform(0.0, 1.5) if is_null else rng.uniform(0.5, 6.0), 1)
        lactate    = round(rng.uniform(4.5, 10.0) if is_null else rng.uniform(2.5, 7.5), 1)
        cox_pct    = round(rng.uniform(8.0, 18.0) if is_null else rng.uniform(15.0, 30.0), 1)

        # PINDAC-specific features
        has_pancreatic_insuff  = rng.random() < (1.0  if is_null else 0.88)   # PATHOGNOMONIC
        has_dyserythropoietic  = rng.random() < (0.92 if is_null else 0.82)   # PATHOGNOMONIC
        has_calvarial_hyper    = rng.random() < (0.85 if is_null else 0.72)   # HALLMARK

        # General OXPHOS features
        has_lactic             = rng.random() < 0.90
        has_hypotonia          = rng.random() < 0.80
        has_regression         = rng.random() < 0.88
        has_dev_delay          = rng.random() < (1.0  if is_null else 0.88)
        has_hepatic            = rng.random() < 0.75   # elevated transaminases
        has_iugr               = rng.random() < 0.68
        has_growth_fail        = rng.random() < 0.82
        has_feeding            = rng.random() < 0.72
        has_respiratory        = rng.random() < 0.45
        has_seizures           = rng.random() < (0.48 if is_null else 0.35)   # moderate — NOT cardinal
        has_leigh_mri          = rng.random() < (0.35 if is_null else 0.20)   # NOT dominant

        # KEY DDx negatives — absent in COX4I1
        has_hcm            = False   # CRITICAL NEGATIVE — KEY DDx SCO2/COA5/COA6
        has_renal_tubular  = False   # CRITICAL NEGATIVE — KEY DDx COX10 Fanconi
        has_ataxia         = False   # CRITICAL NEGATIVE — KEY DDx COX20

        # Outcome — severe malabsorption + OXPHOS failure; variable
        if is_null:
            survived = rng.random() < 0.48   # significant early mortality
        else:
            survived = rng.random() < 0.65

        patients.append({
            "id":                f"COX4I1-{i:03d}",
            "sex":               sex,
            "genotype":          geno,
            "onset_mo":          onset_mo,
            "lactate_mM":        lactate,
            "cox_pct":           cox_pct,
            "pancreatic_insuff": has_pancreatic_insuff,
            "dyserythropoietic": has_dyserythropoietic,
            "calvarial_hyper":   has_calvarial_hyper,
            "lactic_acidosis":   has_lactic,
            "hypotonia":         has_hypotonia,
            "regression":        has_regression,
            "dev_delay":         has_dev_delay,
            "hepatic":           has_hepatic,
            "iugr":              has_iugr,
            "growth_failure":    has_growth_fail,
            "feeding_difficulty": has_feeding,
            "respiratory":       has_respiratory,
            "seizures":          has_seizures,
            "leigh_mri":         has_leigh_mri,
            "hcm":               has_hcm,
            "renal_tubular":     has_renal_tubular,
            "ataxia":            has_ataxia,
            "survived_1yr":      survived,
        })
    return patients


# ── Public API ───────────────────────────────────────────────────────────────
def get_overview() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)

    total         = len(cohort)
    avg_lat       = round(sum(p["lactate_mM"] for p in cohort) / total, 1)
    avg_cox       = round(sum(p["cox_pct"]    for p in cohort) / total, 1)
    pct_pancreatic= round(sum(1 for p in cohort if p["pancreatic_insuff"]) / total * 100)
    pct_dyseryth  = round(sum(1 for p in cohort if p["dyserythropoietic"]) / total * 100)
    pct_calvarial = round(sum(1 for p in cohort if p["calvarial_hyper"])   / total * 100)
    pct_seizures  = round(sum(1 for p in cohort if p["seizures"])          / total * 100)
    pct_leigh     = round(sum(1 for p in cohort if p["leigh_mri"])         / total * 100)
    pct_surv      = round(sum(1 for p in cohort if p["survived_1yr"])      / total * 100)
    pct_hypo      = round(sum(1 for p in cohort if p["hypotonia"])         / total * 100)
    pct_hepatic   = round(sum(1 for p in cohort if p["hepatic"])           / total * 100)

    return {
        "gene": GENE,
        "alias": "COX4 · COX IV-1 · Subunit IV (ubiquitous/normoxia isoform) · PINDAC gene",
        "protein": "169 aa precursor (~22 aa MTS + 147 aa mature) · ~17.2 kDa · NO transmembrane helix · matrix-facing peripheral subunit · ATP allosteric site",
        "disease": DISEASE_NAME,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "onset": ONSET,
        "cohort_size": total,
        "avg_lactate_mM": avg_lat,
        "avg_cox_residual_pct": avg_cox,
        "biochemical_fingerprint": (
            "Isolated COX deficiency (15–30% residual in fibroblasts/muscle/liver); "
            "Complexes I, II, III NORMAL. BN-PAGE: severely reduced/absent assembled CIV. "
            "Immunoblot: COX4I1 absent, secondary reduction of MT-CO1/CO2 core. "
            "Biochemically IDENTICAL to other isolated COXPD — WES/WGS mandatory. "
            "ATP allosteric regulation lost (COX4I1 ATP-binding site absent)."
        ),
        "cardinal_feature": (
            "PINDAC triad — Exocrine Pancreatic Insufficiency + Dyserythropoietic Anaemia "
            "+ Calvarial Hyperostosis: PATHOGNOMONIC for COX4I1 deficiency. "
            "No other COXPD produces this multi-system combination. "
            "NOT classic Leigh syndrome. NO HCM. NO renal Fanconi. NO cerebellar ataxia."
        ),
        "key_ddx_negatives": [
            "NO HCM (DDx SCO2 100% / COA5 88% / COA6 90% / COX15 78% — matrix subunit, no cardiac-specific isoform loss)",
            "NO Leigh MRI dominant (DDx SURF1 95% / COX14 80% / COX8A 85% — only ~20-25% in COX4I1)",
            "NO renal Fanconi tubulopathy (DDx COX10 65% Fanconi + anaemia 80% — different anaemia mechanism)",
            "NO cerebellar ataxia (DDx COX20 100% CARDINAL childhood cerebellar ataxia)",
            "Dyserythropoietic anaemia is DISTINCT from COX10 anaemia (normochromic vs dysplastic erythropoiesis)",
        ],
        "kpis": {
            "pancreatic_insuff_pct": pct_pancreatic,
            "dyserythropoietic_pct": pct_dyseryth,
            "calvarial_hyper_pct":   pct_calvarial,
            "seizures_pct":          pct_seizures,
            "leigh_mri_pct":         pct_leigh,
            "survived_1yr_pct":      pct_surv,
            "hypotonia_pct":         pct_hypo,
            "hepatic_pct":           pct_hepatic,
        },
        "key_contrasts": {
            "COX4I1_vs_COX10":   "Both have anaemia — CRITICAL DDx: COX4I1=dyserythropoietic (bone marrow dysplasia) vs COX10=normochromic (type IV, Fanconi-associated). COX4I1: pancreatic insufficiency (ABSENT in COX10). COX10: Fanconi 65% (ABSENT in COX4I1). Different anaemia mechanisms.",
            "COX4I1_vs_COX8A":   "COX4I1: PINDAC triad — pancreatic/haematological/skeletal. COX8A: brain-dominant epileptic encephalopathy (~85% seizures). COX4I1 seizures moderate (~38%). No Leigh dominant in COX4I1 vs Leigh 85% in COX8A.",
            "COX4I1_vs_SURF1":   "SURF1: Leigh syndrome dominant (95%), haem a3/CuB insertion. COX4I1: NO Leigh dominant (25%), structural subunit, PINDAC triad. SURF1: no pancreatic/haematological features.",
            "COX4I1_vs_SCO2":    "SCO2: HCM 100% — OPPOSITE to COX4I1 (NO HCM). COX4I1: PINDAC exocrine/haematological/skeletal vs SCO2 cardiac/neurological dominant.",
            "COX4I1_isoforms":   "COX4I1 (ubiquitous/normoxia) vs COX4I2 (lung/hypoxia — HIF-1α induced). Loss of COX4I1 cannot be compensated by COX4I2 in non-lung tissues. ATP allosteric regulation site on COX4I1 — unique among CIV subunits (feedback inhibition when ATP:ADP high).",
        },
        "assembly_pathway": "CIV structural subunit (matrix-facing) — assembles EARLY with MT-CO1 MITRAC intermediate before heme a/a3 insertion; no transmembrane anchor; contains ATP allosteric feedback site; coordinates matrix-domain stability and ATP:ADP ratio regulation",
        "color": COLOR,
        "light": LIGHT,
    }


def get_breakdown() -> dict:
    rng    = _rng()
    cohort = _build_cohort(rng)
    total  = len(cohort)

    # Genotype distribution
    geno_dist: dict[str, int] = {}
    for p in cohort:
        geno_dist[p["genotype"]] = geno_dist.get(p["genotype"], 0) + 1
    geno_rows = [
        {"genotype": g, "n": n, "pct": round(n / total * 100)}
        for g, n in sorted(geno_dist.items(), key=lambda x: -x[1])
    ]

    # Feature prevalence
    features = [
        ("Exocrine Pancreatic Insufficiency — PATHOGNOMONIC",  "pancreatic_insuff", "Steatorrhoea, fat-soluble vitamin (ADEK) deficiency, malabsorption; PERT is first-line management"),
        ("Dyserythropoietic Anaemia — PATHOGNOMONIC",          "dyserythropoietic", "Dysplastic erythroid precursors on BM aspirate; different mechanism from COX10 normochromic anaemia"),
        ("Calvarial Hyperostosis — UNIQUE",                     "calvarial_hyper",   "Skull thickening on CT/X-ray; pathognomonic radiographic finding; not seen in any other COXPD"),
        ("Developmental Delay / Cognitive Impairment",          "dev_delay",         "Global developmental delay; IQ impaired proportional to severity of COX deficiency"),
        ("Lactic Acidosis",                                     "lactic_acidosis",   "Elevated plasma + CSF lactate; reflects OXPHOS block; severity correlates with genotype"),
        ("Growth Failure",                                      "growth_failure",    "Postnatal growth retardation; compounded by malabsorption from pancreatic insufficiency"),
        ("Hepatic Involvement (elevated transaminases)",        "hepatic",           "Elevated AST/ALT; hepatic COX4I1 loss; usually not fulminant hepatic failure (DDx SCO1)"),
        ("Hypotonia",                                           "hypotonia",         "Axial and peripheral; OXPHOS failure in muscle"),
        ("Psychomotor Regression",                              "regression",        "Loss of milestones from ongoing OXPHOS failure; progressive"),
        ("IUGR (intrauterine growth restriction)",              "iugr",              "Prenatal onset; placenta requires high OXPHOS; common in severe biallelic LOF"),
        ("Feeding Difficulties",                                "feeding_difficulty", "Hypotonia + malabsorption compound feeding; NG/G-tube often required"),
        ("Seizures (moderate — NOT cardinal)",                  "seizures",          "~38% — moderate frequency; NOT the dominant feature (contrast: COX8A ~85% CARDINAL)"),
        ("Leigh-like MRI (minority)",                           "leigh_mri",         "~20-25% — NOT the dominant pattern; distinguishes COX4I1 from SURF1/COX14/COX8A"),
        ("Respiratory Compromise",                              "respiratory",       "Less frequent than other COXPD; when present, may require NIV"),
        ("NO HCM (KEY DDx)",                                    "hcm",              "Absent — critical DDx vs SCO2 (100%) / COA5 (88%) / COA6 (90%) / COX15 (78%)"),
        ("NO Renal Fanconi Tubulopathy (KEY DDx)",              "renal_tubular",     "Absent — critical DDx vs COX10 (Fanconi 65%); COX4I1 anaemia is dyserythropoietic, not Fanconi-linked"),
        ("NO Cerebellar Ataxia (KEY DDx)",                      "ataxia",            "Absent — critical DDx vs COX20 (100% childhood ataxia CARDINAL)"),
    ]
    feature_rows = []
    for label, key, note in features:
        count = sum(1 for p in cohort if p.get(key))
        feature_rows.append({
            "feature": label, "n": count,
            "pct": round(count / total * 100), "note": note
        })

    # Outcome by genotype class
    null_pts = [p for p in cohort if "Stop" in p["genotype"] or "deletion" in p["genotype"].lower() or "null" in p["genotype"]]
    miss_pts = [p for p in cohort if p not in null_pts]
    out_null = round(sum(1 for p in null_pts if p["survived_1yr"]) / max(len(null_pts), 1) * 100)
    out_miss = round(sum(1 for p in miss_pts if p["survived_1yr"]) / max(len(miss_pts), 1) * 100)

    # Pancreatic vs COX activity
    low_cox  = [p for p in cohort if p["cox_pct"] <= 18]
    high_cox = [p for p in cohort if p["cox_pct"] > 18]
    panc_low = round(sum(1 for p in low_cox  if p["pancreatic_insuff"]) / max(len(low_cox),  1) * 100)
    panc_hi  = round(sum(1 for p in high_cox if p["pancreatic_insuff"]) / max(len(high_cox), 1) * 100)

    # Treatment contraindications
    cis = [
        {"drug": "Valproic acid (VPA)",       "severity": "ABSOLUTE CI",
         "reason": "CoA sequestration (inhibits mito β-oxidation) + POLG inhibition + hepatotoxicity; all AEDs for COX4I1 seizures must avoid VPA — use LEV"},
        {"drug": "Metformin",                 "severity": "ABSOLUTE CI",
         "reason": "Complex I inhibition → lactic crisis; especially dangerous with any OXPHOS defect"},
        {"drug": "Propofol",                  "severity": "ABSOLUTE CI",
         "reason": "PRIS: direct CIV inhibition; with 15-30% residual COX → metabolic collapse; use sevoflurane only"},
        {"drug": "Linezolid",                 "severity": "ABSOLUTE CI",
         "reason": "mt 23S rRNA inhibition → blocks MT-CO1/CO2/CO3 translation → eliminates residual COX"},
        {"drug": "Chloramphenicol",           "severity": "ABSOLUTE CI",
         "reason": "Mitoribosome block (same mechanism as linezolid)"},
        {"drug": "Ketogenic diet (KD)",       "severity": "CONTRAINDICATED",
         "reason": "β-oxidation requires CIV flux (FADH₂→CIII→CIV); KD with COX deficiency → metabolic crisis; also contraindicated because malabsorption from EPI makes KD adherence dangerous"},
        {"drug": "High-fat diet (unsupported)", "severity": "CAUTION",
         "reason": "Exocrine pancreatic insufficiency causes fat malabsorption; dietary fat must be given with PERT — unmanaged high fat diet worsens malabsorption"},
        {"drug": "Fasting > 4 hours",         "severity": "HIGH RISK",
         "reason": "OXPHOS failure + hypoglycaemia risk; maintain GIR 6-8 mg/kg/min perioperative; never fast"},
    ]

    # Recommended treatments (COX4I1-specific)
    txs = [
        {"tx": "Pancreatic Enzyme Replacement Therapy (PERT)",  "level": "MANDATORY — first-line",
         "note": "Creon/Pancreaze with every meal and snack; titrate to stool fat normalisation; PERT is the most impactful intervention for quality of life in COX4I1"},
        {"tx": "Fat-Soluble Vitamins (ADEK)",                   "level": "MANDATORY",
         "note": "Vitamin A, D, E, K supplementation due to fat malabsorption from EPI; monitor levels quarterly"},
        {"tx": "Levetiracetam (LEV)",                           "level": "Level B preferred AED",
         "note": "First-line for seizures (~38%): renal excretion, no CYP, no mito toxicity; safe in COXPD12"},
        {"tx": "Folic acid / Iron supplementation",             "level": "Level C",
         "note": "For dyserythropoietic anaemia: folic acid supports erythroid maturation; iron for deficiency component from malabsorption"},
        {"tx": "Blood transfusion (if severe anaemia)",         "level": "Practical standard",
         "note": "For symptomatic dyserythropoietic anaemia; PRBC transfusion as clinically indicated"},
        {"tx": "Nutritional support (NG/G-tube)",               "level": "Practical standard",
         "note": "For feeding difficulties + malabsorption; continuous enteral nutrition maintains caloric intake"},
        {"tx": "CoQ10 (ubiquinol)",                             "level": "Level C",
         "note": "Mitochondrial cocktail; supports residual respiratory chain electron flow"},
        {"tx": "Thiamine (B1) — empiric",                       "level": "Level C MANDATORY",
         "note": "Empiric pending WES — excludes treatable SLC19A3 Leigh mimic; also important for energy metabolism"},
        {"tx": "Biotin — empiric",                              "level": "Level C MANDATORY",
         "note": "Empiric pending WES — excludes treatable BTD Leigh mimic"},
        {"tx": "Riboflavin (B2)",                               "level": "Level C",
         "note": "Flavoprotein support adjacent to ETC"},
        {"tx": "L-Carnitine",                                   "level": "Level C",
         "note": "Secondary carnitine deficiency from fat malabsorption + OXPHOS disease"},
        {"tx": "Sevoflurane (NOT propofol)",                    "level": "Mandatory — anaesthesia",
         "note": "Only acceptable general anaesthetic; propofol ABSOLUTE CI (PRIS)"},
        {"tx": "GIR 6-8 mg/kg/min perioperative",              "level": "Mandatory",
         "note": "Never fast >4h; steady glucose prevents metabolic crisis; continuous monitoring"},
    ]

    # Per-patient table
    patient_rows = []
    for p in cohort:
        patient_rows.append({
            "id":              p["id"],
            "sex":             p["sex"],
            "genotype":        p["genotype"],
            "onset_mo":        p["onset_mo"],
            "lactate_mM":      p["lactate_mM"],
            "cox_pct":         p["cox_pct"],
            "pancreatic":      "Yes" if p["pancreatic_insuff"] else "No",
            "dyseryth":        "Yes" if p["dyserythropoietic"]  else "No",
            "calvarial":       "Yes" if p["calvarial_hyper"]    else "No",
            "seizures":        "Yes" if p["seizures"]           else "No",
            "survived_1yr":    "Yes" if p["survived_1yr"]       else "No",
        })

    return {
        "gene_id":       DISEASE_ID,
        "genotype_dist": geno_rows,
        "feature_prev":  feature_rows,
        "outcome": {
            "null_allele_1yr_survival_pct":    out_null,
            "missense_allele_1yr_survival_pct": out_miss,
            "note": "Truncating/deletion alleles: severe malabsorption + OXPHOS failure drive early mortality (~48% survival at 1yr). Missense alleles: milder (~35-40% residual CIV) with better early survival.",
        },
        "pancreatic_vs_cox_activity": {
            "pancreatic_pct_when_cox_at_or_below_18pct": panc_low,
            "pancreatic_pct_when_cox_above_18pct":       panc_hi,
            "note": "Exocrine pancreatic insufficiency correlates with depth of COX deficiency — severe EPI in complete LOF alleles. Pancreas is highly ATP-demanding (exocrine secretion requires mitochondrial ATP).",
        },
        "contraindications": cis,
        "treatments":        txs,
        "patient_table":     patient_rows,
        "ddx_matrix": [
            {"disease": "COX10 (COXPD3)",   "shared": "Isolated COX deficiency, AR, anaemia",            "distinguishing": "COX4I1: dyserythropoietic anaemia + pancreatic insufficiency + calvarial hyperostosis. COX10: Fanconi renal tubulopathy 65% + normochromic anaemia (type IV). PERT is COX4I1-specific; renal replacement COX10-specific."},
            {"disease": "SURF1 (COXPD1)",   "shared": "Isolated COX deficiency, AR",                    "distinguishing": "SURF1: Leigh 95% CARDINAL, haem a3/CuB insertion; NO pancreatic/haematological/skeletal features. COX4I1: PINDAC triad, Leigh minority (~20%)."},
            {"disease": "COX8A (COXPD15)",  "shared": "Isolated COX deficiency, AR, structural subunit", "distinguishing": "COX8A: brain-dominant epileptic encephalopathy (seizures 85% CARDINAL). COX4I1: PINDAC triad (pancreatic/haematological/skeletal CARDINAL), seizures moderate (38%)."},
            {"disease": "COX6B1 (COXPD7)",  "shared": "Isolated COX deficiency, AR, structural subunit, NO HCM", "distinguishing": "COX6B1: encephalomyopathy + myopathy dominant; NO pancreatic/haematological/skeletal features. COX4I1: PINDAC triad unique. Both AR structural subunits."},
            {"disease": "SCO2 (COXPD2)",    "shared": "Isolated COX deficiency, AR, neonatal onset",     "distinguishing": "SCO2: HCM 100% via CuA copper metalation — OPPOSITE to COX4I1 (NO HCM). COX4I1: PINDAC multi-system vs SCO2 cardiac-dominant."},
            {"disease": "COA5 (COXPD11)",   "shared": "Isolated COX deficiency, AR",                    "distinguishing": "COA5: HCM 88% CARDINAL; COX4I1: NO HCM. COA5: MT-CO1 assembly factor; COX4I1: structural subunit matrix-facing. Both NO ataxia, NO Fanconi."},
            {"disease": "SCO1 (COXPD4)",    "shared": "Isolated COX deficiency, AR, hepatic involvement", "distinguishing": "SCO1: hepatopathy 100% neonatal hepatic FAILURE (CARDINAL); COX4I1: hepatic transaminase elevation (NOT failure). SCO1: NO pancreatic features; COX4I1: EPI pathognomonic."},
            {"disease": "COX20 (COXPD8)",   "shared": "Isolated COX deficiency, AR",                    "distinguishing": "COX20: ataxia 100% CARDINAL, childhood onset; COX4I1: NO ataxia. COX4I1: PINDAC neonatal/infantile; COX20: childhood neurological with better survival."},
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_id": DISEASE_ID,
        "glossary": [
            {"term": "COX4I1",
             "definition": (
                 "Cytochrome c Oxidase Subunit 4, Isoform 1 — nuclear-encoded structural "
                 "subunit of Complex IV (COX). The ubiquitous, normoxia-expressed isoform of "
                 "CIV subunit 4. 169 aa precursor protein; ~22 aa mitochondrial targeting "
                 "sequence; ~147 aa mature protein; ~17.2 kDa. No transmembrane helix — "
                 "matrix-facing peripheral subunit that contacts the matrix domain of "
                 "MT-CO1 and MT-CO2. Located on chromosome 16q22.1."
             )},
            {"term": "COX4I1 vs COX4I2 isoforms",
             "definition": (
                 "CIV subunit 4 has two isoforms: COX4I1 (ubiquitous — expressed in all "
                 "tissues under normoxia) and COX4I2 (lung/hypoxia-specific — induced by "
                 "HIF-1α under low O₂ to increase CIV efficiency at low oxygen). "
                 "In non-lung tissues: only COX4I1 is expressed — loss of COX4I1 produces "
                 "multi-organ COX deficiency without COX4I2 compensation. "
                 "COX4I1 contains the ATP allosteric regulatory site (inhibition when "
                 "ATP:ADP ratio is high); COX4I2 lacks this site (optimised for hypoxia, "
                 "not ATP feedback regulation). This explains why COX4I1 deficiency disrupts "
                 "energy sensing in addition to OXPHOS capacity."
             )},
            {"term": "COXPD12",
             "definition": (
                 "Mitochondrial Complex IV Deficiency, Nuclear Type 12 — OMIM disease "
                 "designation (#616501) for COX4I1-related COX deficiency. Characterised "
                 "by isolated COX deficiency (15–30% residual), exocrine pancreatic "
                 "insufficiency, dyserythropoietic anaemia, and calvarial hyperostosis "
                 "(PINDAC syndrome). Rare disease — fewer than 30 published patients worldwide."
             )},
            {"term": "PINDAC syndrome",
             "definition": (
                 "Acronym for the pathognomonic phenotype of COX4I1 deficiency: "
                 "Pancreatic Insufficiency aNd Dyserythropoietic Anemia and Calvarial "
                 "hyperostosis. This combination is unique to COX4I1 deficiency among all "
                 "CIV deficiency diseases — no other COXPD produces exocrine pancreatic "
                 "failure, bone marrow dyserythropoiesis, and calvarial bone thickening "
                 "simultaneously. Recognising the PINDAC triad should prompt immediate "
                 "COX4I1 molecular testing (WES/WGS)."
             )},
            {"term": "Exocrine Pancreatic Insufficiency (EPI) in COX4I1",
             "definition": (
                 "The pathognomonic hallmark of COXPD12. The exocrine pancreas is one "
                 "of the most energy-demanding tissues in the body — acinar cells secrete "
                 "digestive enzymes (lipase, amylase, proteases) via a process requiring "
                 "high mitochondrial ATP. Loss of COX4I1 → CIV failure in acinar cells → "
                 "failure of enzyme secretion → EPI. Clinical consequences: steatorrhoea, "
                 "fat-soluble vitamin (A, D, E, K) malabsorption, protein malabsorption, "
                 "weight loss, failure to thrive. Management: PERT (pancreatic enzyme "
                 "replacement therapy) with every meal/snack — this is the most impactful "
                 "treatable component of COX4I1 disease."
             )},
            {"term": "Dyserythropoietic Anaemia (COX4I1)",
             "definition": (
                 "Anaemia with dysplastic erythroid precursors on bone marrow aspirate — "
                 "hallmark of COX4I1 deficiency. Erythropoiesis is highly OXPHOS-dependent "
                 "(dividing erythroblasts require mitochondrial respiration). COX4I1 loss → "
                 "CIV failure in bone marrow erythroblasts → ineffective erythropoiesis → "
                 "dyserythropoietic anaemia. Distinct from: COX10 anaemia (normochromic, "
                 "associated with Fanconi tubulopathy, not BM dysplasia); standard aplastic "
                 "anaemia; haemolytic anaemia. BM aspirate shows multinucleated erythroblasts "
                 "and nuclear bridging. Management: folic acid, iron (if deficient), "
                 "blood transfusion for symptomatic cases."
             )},
            {"term": "Calvarial Hyperostosis (COX4I1)",
             "definition": (
                 "Skull thickening visible on CT/plain X-ray — pathognomonic radiographic "
                 "finding of COX4I1 deficiency. Mechanism: COX4I1 loss → CIV failure in "
                 "osteoblasts → dysregulated bone remodelling → abnormal calvarial bone "
                 "deposition. Calvarial hyperostosis is NOT seen in any other COXPD and "
                 "represents a diagnostic clue on neuroimaging. In contrast to other "
                 "mitochondrial diseases where skull pathology is absent, calvarial "
                 "thickening on head CT should prompt COX4I1 investigation."
             )},
            {"term": "ATP allosteric regulation (COX4I1 site)",
             "definition": (
                 "COX4I1 contains a matrix-facing ATP binding site that enables feedback "
                 "inhibition of CIV activity when the ATP:ADP ratio is high. When cells are "
                 "energy-replete (high ATP), ATP binds COX4I1 → CIV inhibition → reduced "
                 "OXPHOS → prevents over-production of reactive oxygen species (ROS). "
                 "This regulatory function is UNIQUE to COX4I1 among CIV structural subunits "
                 "(COX4I2 lacks this site, as hypoxia does not require ATP feedback inhibition). "
                 "Loss of COX4I1 eliminates this energy-sensing regulation in addition to "
                 "reducing CIV capacity — contributing to multi-organ metabolic dysregulation."
             )},
            {"term": "VPA absolute contraindication (COXPD12)",
             "definition": (
                 "Valproic acid is ABSOLUTELY CONTRAINDICATED in COX4I1 deficiency. "
                 "Triple mechanism: (1) CoA sequestration → inhibits mitochondrial β-oxidation; "
                 "(2) POLG inhibition → mtDNA depletion → further reduction of CIV subunits "
                 "(MT-CO1/CO2/CO3 are mtDNA-encoded); (3) hepatotoxicity risk (synergistic "
                 "with existing hepatic involvement). Seizures in COXPD12 (~38%) must be "
                 "managed with levetiracetam (first-line, renal excretion, no CYP, no mito "
                 "toxicity) or clobazam (adjunct). ACTH/vigabatrin for infantile spasms."
             )},
            {"term": "PERT (Pancreatic Enzyme Replacement Therapy)",
             "definition": (
                 "The most impactful specific treatment for COX4I1/COXPD12. PERT consists "
                 "of enteric-coated pancreatin microspheres (e.g., Creon, Pancreaze) taken "
                 "with every meal and snack. Dosing: typically 500–1000 lipase units/kg/meal, "
                 "titrated to steatorrhoea resolution. PERT corrects fat malabsorption, "
                 "improves caloric absorption, and allows fat-soluble vitamin delivery. "
                 "Fat-soluble vitamins (A, D, E, K) must be supplemented separately "
                 "regardless of PERT response (monitoring levels quarterly). PERT does NOT "
                 "treat the underlying OXPHOS defect but substantially improves nutritional "
                 "status and quality of life."
             )},
        ],
        "clinical_notes": [
            (
                "COX4I1 deficiency (COXPD12) should be suspected in any infant with: "
                "(1) exocrine pancreatic insufficiency (steatorrhoea, low faecal elastase), "
                "(2) dyserythropoietic anaemia (bone marrow dysplasia pattern), "
                "(3) calvarial hyperostosis on skull CT, "
                "(4) isolated Complex IV deficiency on enzyme assay (15-30% residual), "
                "(5) lactic acidosis + developmental delay. "
                "The PINDAC triad (pancreatic + haematological + skeletal) is pathognomonic "
                "and should prompt immediate COX4I1 WES/WGS molecular confirmation."
            ),
            (
                "Exocrine pancreatic management is the most clinically impactful priority. "
                "Start PERT immediately (Creon 500-1000 lipase units/kg/meal) — do NOT wait "
                "for molecular confirmation if EPI is evident. Supplement fat-soluble vitamins "
                "ADEK simultaneously. Monitor faecal elastase-1 (screening) and periodic "
                "faecal fat balance study. Nutritional support via NG/G-tube if feeding "
                "difficulties prevent adequate oral intake. Malnutrition worsens the OXPHOS "
                "burden and accelerates cognitive decline."
            ),
            (
                "Haematological management: Dyserythropoietic anaemia — supplement folic acid "
                "(1-5 mg/day), iron (if deficient from malabsorption). Blood transfusion for "
                "symptomatic severe anaemia (Hb <7 g/dL or symptomatic threshold). "
                "Monitor CBC monthly initially; BM aspirate confirms dysplastic erythropoiesis. "
                "Unlike primary dyserythropoietic anaemia (CDA type I/II/III), there is no "
                "role for interferon-alpha or bone marrow transplantation in the COX4I1 context "
                "(these would not correct the underlying OXPHOS defect)."
            ),
            (
                "Biochemical workup: Muscle biopsy + fibroblast enzyme assay — COX 15-30% "
                "residual with CI/CII/CIII normal is the signature. BN-PAGE for assembled CIV. "
                "Plasma + CSF lactate/pyruvate. Faecal elastase-1 for EPI screening. "
                "Skull CT for calvarial hyperostosis. BM aspirate if anaemia is atypical. "
                "Brain MRI (Leigh pattern is minority — ~20% — but check basal ganglia/brainstem). "
                "EEG if seizures present. Liver function tests, coagulation for hepatic status."
            ),
            (
                "Empiric treatment while awaiting WES: Thiamine 5-10 mg/kg/day IV (MANDATORY) "
                "+ biotin 10-20 mg/day (MANDATORY) — eliminates treatable SLC19A3/BTD mimics. "
                "Start PERT if EPI suspected. CoQ10 ubiquinol + riboflavin + L-carnitine. "
                "VPA ABSOLUTE CI. Propofol ABSOLUTE CI — sevoflurane for any procedures. "
                "GIR 6-8 mg/kg/min perioperative, never fast >4h. If seizures: LEV first-line."
            ),
            (
                "Prognosis: Variable — depends on genotype and nutritional management. "
                "Complete LOF (homozygous deletion/truncating) associated with higher early "
                "mortality (~48% at 1yr). Effective PERT substantially improves nutritional "
                "status and may extend survival. No disease-modifying therapy exists for "
                "the underlying COX4I1 defect. Gene therapy is theoretically feasible "
                "(nuclear-encoded gene) but remains preclinical. Metabolic emergency card "
                "mandatory. Multidisciplinary team: metabolic, GI/nutrition, haematology, "
                "neurology, and cardiology (to exclude HCM annually by ECHO)."
            ),
        ],
        "references": [
            {
                "citation": "Shteyer E et al. (2009). Am J Hum Genet 84(3):413-417.",
                "note": (
                    "First clinical description of human COX4I1 deficiency — biallelic "
                    "deletion of COX4I1 in Israeli Bedouin families with exocrine pancreatic "
                    "insufficiency, dyserythropoietic anaemia, and calvarial hyperostosis "
                    "(PINDAC syndrome). Established COX4I1 as a nuclear-encoded structural "
                    "subunit whose loss causes isolated CIV deficiency with multi-organ phenotype."
                ),
            },
            {
                "citation": "Stroud DA et al. (2015). Cell Metab 21(1):108-119.",
                "note": (
                    "Comprehensive proteomic map of CIV assembly — places COX4I1 in the "
                    "early assembly tier (joins MT-CO1 MITRAC intermediate before heme/copper "
                    "metalation). Demonstrates COX4I1 as an early matrix-domain scaffold. "
                    "COX4I2 assembly behaviour under hypoxia contrasted with COX4I1."
                ),
            },
            {
                "citation": "Acin-Perez R et al. (2011). Cell Metab 13(5):574-584.",
                "note": (
                    "COX4I1 ATP allosteric regulation — demonstrates that the matrix-facing "
                    "ATP binding site on COX4I1 mediates feedback inhibition of CIV activity "
                    "when ATP:ADP ratio is high. This regulatory function is absent in COX4I2, "
                    "explaining tissue-specific metabolic regulation in normoxia vs hypoxia."
                ),
            },
            {
                "citation": "Gorman GS et al. (2016). Nat Rev Dis Primers 2:16080.",
                "note": (
                    "Comprehensive mitochondrial disease epidemiology and clinical management "
                    "review — framework for CIV deficiency management, AED selection in mito "
                    "disease, contraindications, and supportive care applicable to COXPD12."
                ),
            },
            {
                "citation": "Massa V et al. (2008). Am J Hum Genet 82(6):1281-1289.",
                "note": (
                    "COX6B1 (COXPD7) characterisation — structural subunit comparator to "
                    "COX4I1. Both peripheral structural subunits causing isolated CIV deficiency. "
                    "COX6B1: encephalomyopathy/myopathy dominant; COX4I1: PINDAC triad. "
                    "Structural subunit loss vs assembly factor loss compared."
                ),
            },
        ],
        "inheritance_detail": (
            "COX4I1 (COXPD12) is autosomal recessive (AR). Both copies of the COX4I1 gene "
            "must carry pathogenic loss-of-function variants (biallelic) for disease to occur. "
            "Heterozygous carriers are clinically unaffected. Each pregnancy of two carrier "
            "parents carries a 25% recurrence risk. COX4I1 is on chromosome 16q22.1. "
            "A founder deletion allele has been identified in Israeli Bedouin consanguineous "
            "families. Prenatal molecular diagnosis is available once biallelic variants "
            "are confirmed in the proband."
        ),
        "management_summary": (
            "COX4I1/COXPD12 management centres on the PINDAC triad + OXPHOS supportive care: "
            "(1) PANCREATIC: PERT mandatory (Creon 500-1000 u/kg/meal); fat-soluble ADEK vitamins; NG/G-tube nutrition. "
            "(2) HAEMATOLOGICAL: Folic acid + iron; transfusion for severe anaemia. "
            "(3) OXPHOS cocktail: CoQ10 ubiquinol, thiamine (MANDATORY empiric), biotin (MANDATORY empiric), riboflavin, carnitine. "
            "(4) SEIZURES (~38%): LEV first-line — NEVER VPA, NEVER KD. "
            "(5) ENERGY: GIR 6-8 mg/kg/min perioperative — never fast >4h. "
            "(6) ANAESTHESIA: Sevoflurane ONLY — propofol ABSOLUTE CI. "
            "(7) AVOID ABSOLUTELY: VPA, metformin, propofol, linezolid, chloramphenicol, KD."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== COX4I1 COXPD12 OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== COX4I1 COXPD12 BREAKDOWN (first 2 keys) ===")
    bd = get_breakdown()
    print(json.dumps({k: bd[k] for k in list(bd.keys())[:2]}, indent=2))
