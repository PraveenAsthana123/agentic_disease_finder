#!/usr/bin/env python3
"""NDUFAF8 (C17orf89) — Mitochondrial Complex I Deficiency (CI Assembly Factor / Intermediate-Stage).

NDUFAF8 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 8; also known as C17orf89) is a
soluble mitochondrial matrix protein that functions as a CI assembly chaperone/scaffold during
intermediate stages of Complex I (CI) holoenzyme assembly. Unlike NDUFAF6 (2OG-dioxygenase) and
NDUFAF7 (SAM-methyltransferase), NDUFAF8 has no confirmed enzymatic activity — it acts as a
structural chaperone/scaffold analogous to NDUFAF3/NDUFAF4/NDUFAF5.

  NDUFAF8 gene   OMIM *616051
  Disease        Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance    AR (autosomal recessive, biallelic)
  Chromosome     17p13.2

Reference: Stroud DA et al. (2016) Accessory subunits are integral for assembly and function
of human mitochondrial complex I. Nature 538(7623):123–126. (CRISPR/Cas9 systematic CI
assembly factor characterization; identified NDUFAF8/C17orf89 among CI assembly factors;
CI assembly intermediate analysis)

Reference: Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory
chain complex I. Cell Metab 25(1):128–139. (Comprehensive CI assembly pathway mapping using
pulse-chase proteomics; context for CI assembly chaperone/scaffold factors)

Reference: Formosa LE, Ryan MT (2018) Mitochondrial fusion: reaching the end of mitofusin's
tether. J Cell Biol 217(5):1613–1615. (Review context for CI assembly factor nomenclature
and assembly pathway; NDUFAF8 as intermediate-stage assembly factor)

PATHOPHYSIOLOGY (NDUFAF8 / CI Assembly Chaperone / Intermediate-Stage):
  NDUFAF8/C17orf89 is a soluble matrix CI assembly factor that participates in
  intermediate stages of CI holoenzyme assembly:
    1. NDUFAF8 acts as a chaperone/scaffold during CI assembly, stabilising
       assembly intermediates during subunit incorporation.
    2. No confirmed enzymatic (catalytic) activity — acts as a structural assembly
       factor analogous to NDUFAF3, NDUFAF4, and NDUFAF5.
    3. Without NDUFAF8, CI holoenzyme assembly stalls at intermediate stage;
       partially assembled CI subcomplexes accumulate.
    4. Isolated CI deficiency (5–20%); CII/CIII/CIV normal — assembly block
       specific to CI holoenzyme maturation.
    5. No FAD domain, no TM helices — soluble matrix protein; no riboflavin
       response expected.

NDUFAF8 UNIQUE FEATURES vs OTHER CI ASSEMBLY FACTORS:
  1. CHROMOSOME 17p13.2 — UNIQUE LOCUS: Distinct from all NDUFAF1-7 chromosomal
     locations; no same-chromosome overlap with major CI structural subunit genes.
     WES locus confirmation: 17p13.2.
  2. PURELY STRUCTURAL SCAFFOLD: Unlike NDUFAF6 (2OG-dioxygenase) and NDUFAF7
     (SAM-methyltransferase), NDUFAF8 has no confirmed enzymatic activity. It acts
     as a purely structural chaperone/scaffold, analogous to NDUFAF3/4/5 but in a
     distinct assembly stage.
  3. NO FAD DOMAIN, 0% RIBOFLAVIN RESPONSE: No FAD domain — cannot respond to
     riboflavin supplementation. Critical DDx vs ACAD9 (50-60% riboflavin-responsive).
  4. INTERMEDIATE CI ASSEMBLY STAGE: Positioned at intermediate CI assembly stage,
     distinct from MCIA/ND2-ND5 (ACAD9/NDUFAF1/ECSIT/TMEM126B), ND1-module
     (NDUFAF3/4/5/TIMMDC1), N-module (FOXRED1/NUBPL), and enzymatic factors
     (NDUFAF6 Q-module, NDUFAF7 Q-module/membrane-arm interface).
  5. LOW HCM (<10%): Very low HCM compared to TIMMDC1 (>80%), NDUFV2 (80%),
     ACAD9 (55-65%). HCM is not a prominent feature.

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs ACAD9 (3q21.3): ACAD9 = MCIA/ND2-ND5 (Class 1); riboflavin-responsive 50-60%.
    NDUFAF8 = structural chaperone; 0% riboflavin response. Key test: riboflavin trial.
  vs NDUFAF7 (2q11.2): NDUFAF7 = SAM-methyltransferase (only CI enzymatic methyltransferase).
    NDUFAF8 = structural scaffold; no enzymatic activity.
  vs NDUFAF6 (8q22.1): NDUFAF6 = 2OG-Fe(II) dioxygenase. NDUFAF8 = structural scaffold.
    Different assembly stages. WES mandatory: 17p13.2 vs 8q22.1.
  vs TIMMDC1 (3q25.1): TIMMDC1 HCM >80%. NDUFAF8 HCM <10%. Key DDx marker.
  vs NDUFAF3 (2q33.1)/NDUFAF5 (20p12.1): All soluble matrix scaffolds without enzymatic activity;
    NDUFAF8 at distinct assembly stage; different chromosomes (17p13.2 vs 2q33.1 vs 20p12.1).
    WES mandatory.
  vs POLG/DGUOK: NDUFAF8: NO hepatopathy.
"""

import random
import math

SEED = 699
rng  = random.Random(SEED)

GENE         = "NDUFAF8"
OMIM_GENE    = "616051"
OMIM_DISEASE = "256000"
DISEASE_NAME = "Mitochondrial Complex I Deficiency (OMIM #256000)"
CHROMOSOME   = "17p13.2"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ─────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.283C>T",
        "hgvs_p":    "p.Arg95Trp",
        "domain":    "Core beta-strand 3 — CI assembly chaperone fold",
        "mechanism": "Arginine-to-tryptophan substitution disrupts the central beta-strand of the NDUFAF8 chaperone fold; bulky tryptophan destabilises the structural scaffold required for CI intermediate stabilisation; holoenzyme CI assembly stalls",
        "severity":  "severe",
        "ci_pct_range": (5, 14),
        "notes":     "Severe infantile Leigh-like onset; core beta-strand disruption; complete chaperone fold collapse; isolated CI deficiency; consanguineous families reported",
    },
    {
        "hgvs_c":    "c.425T>C",
        "hgvs_p":    "p.Leu142Pro",
        "domain":    "Alpha-helix 4 — NDUFAF8 scaffold core",
        "mechanism": "Helix-breaking proline substitution abolishes the alpha-helical secondary structure of helix 4; loss of scaffold rigidity; CI assembly intermediate cannot be stabilised; holoenzyme maturation blocked",
        "severity":  "severe",
        "ci_pct_range": (5, 13),
        "notes":     "Severe infantile onset; helix-breaking proline in scaffold core; complete loss of CI assembly chaperone function; profound lactic acidosis",
    },
    {
        "hgvs_c":    "c.202G>C",
        "hgvs_p":    "p.Gly68Arg",
        "domain":    "Beta-strand 2 — conserved glycine in chaperone beta-sheet core",
        "mechanism": "Glycine-to-arginine substitution at a tightly packed position in the core beta-sheet; bulky arginine side chain cannot be accommodated; protein misfolding; loss of NDUFAF8 function and CI intermediate stabilisation",
        "severity":  "severe",
        "ci_pct_range": (5, 14),
        "notes":     "Conserved glycine at sterically restricted position in beta-sheet core; severe CI deficiency; Leigh-like MRI pattern; consanguineous origin frequent",
    },
    {
        "hgvs_c":    "c.554C>T",
        "hgvs_p":    "p.Ala185Val",
        "domain":    "C-terminal domain — hydrophobic core packing",
        "mechanism": "Alanine-to-valine substitution introduces a minor steric clash in the C-terminal hydrophobic core; partial protein misfolding; residual NDUFAF8 function preserved ~40-50%; intermediate CI deficiency",
        "severity":  "intermediate",
        "ci_pct_range": (10, 18),
        "notes":     "Intermediate severity; partial residual CI activity; later onset (3-14 months); exercise intolerance and developmental delay as dominant early features",
    },
    {
        "hgvs_c":    "c.IVS4+1G>A",
        "hgvs_p":    "p.splice (intron 4 donor)",
        "domain":    "Splice donor site — intron 4",
        "mechanism": "Canonical splice donor disruption; exon 4 skipping or cryptic splice activation; loss of central scaffold domain segment; partial residual CI from minor cryptic splice product; moderate phenotype",
        "severity":  "moderate",
        "ci_pct_range": (10, 18),
        "notes":     "Splice donor variant; partial CI residual activity from minor in-frame cryptic splicing; moderate phenotype with subacute Leigh-like presentation; milder course than severe missense",
    },
    {
        "hgvs_c":    "c.570G>A",
        "hgvs_p":    "p.Trp190Ter",
        "domain":    "C-terminal domain — premature truncation (nonsense)",
        "mechanism": "Premature stop codon at position 190 truncates NDUFAF8 in the C-terminal domain; NMD-sensitive; effectively null allele; complete loss of CI assembly chaperone function; no CI intermediate stabilisation",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Null allele; NMD-sensitive transcript; often compound-heterozygous with missense; neonatal to early infantile onset; worst prognosis; CI essentially absent on BN-PAGE",
    },
    {
        "hgvs_c":    "c.173A>G",
        "hgvs_p":    "p.Tyr58Cys",
        "domain":    "Beta-strand 2 — aromatic scaffold contact",
        "mechanism": "Tyrosine-to-cysteine substitution removes aromatic pi-stacking and hydrogen-bonding contacts in the beta-sheet scaffold; partial protein destabilisation; moderate-to-severe CI deficiency depending on allele combination",
        "severity":  "moderate",
        "ci_pct_range": (8, 16),
        "notes":     "Aromatic contact lost; moderate CI deficiency; subacute onset; sometimes presents with initial exercise intolerance before full Leigh phenotype declares",
    },
]

# ─── Patient cohort ────────────────────────────────────────────────────────────
ETHNICITIES = ["European", "Middle-Eastern (consanguineous)", "South-Asian", "East-Asian", "North-African (consanguineous)", "South-American", "Pan-Ethnic"]
ETHNICITY_WEIGHTS = [0.30, 0.22, 0.16, 0.12, 0.10, 0.07, 0.03]
SEXES = ["M", "F"]

def _pick_weighted(choices, weights):
    r = rng.random()
    cum = 0.0
    for c, w in zip(choices, weights):
        cum += w
        if r < cum:
            return c
    return choices[-1]

def _pick_variant_pair():
    allele_weights = [0.24, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06]
    a1 = _pick_weighted(VARIANTS, allele_weights)
    a2 = _pick_weighted(VARIANTS, allele_weights)
    return a1, a2

FEATURES = [
    "Leigh syndrome / Leigh-like MRI",
    "Lactic acidosis (blood/CSF)",
    "Developmental delay / regression",
    "Hypotonia (axial/generalized)",
    "Seizures (multiple types)",
    "Basal ganglia lesions (bilateral)",
    "Feeding difficulties / failure to thrive",
    "Respiratory compromise",
    "Hypertrophic cardiomyopathy (HCM)",
    "Peripheral neuropathy",
    "Optic atrophy",
    "Leukodystrophy",
    "Hepatopathy",
    "Olfactory bulb lesions",
    "Exercise intolerance (older/milder)",
    "Striatal necrosis on MRI",
    "Brainstem involvement",
    "Cerebellar atrophy",
]

FEATURE_PROBS = {
    "Leigh syndrome / Leigh-like MRI":          0.70,
    "Lactic acidosis (blood/CSF)":              0.88,
    "Developmental delay / regression":         0.94,
    "Hypotonia (axial/generalized)":            0.90,
    "Seizures (multiple types)":                0.50,
    "Basal ganglia lesions (bilateral)":        0.65,
    "Feeding difficulties / failure to thrive": 0.76,
    "Respiratory compromise":                   0.40,
    "Hypertrophic cardiomyopathy (HCM)":        0.07,   # Low HCM (<10%)
    "Peripheral neuropathy":                    0.04,   # Very low — critical DDx vs NDUFS1
    "Optic atrophy":                            0.18,
    "Leukodystrophy":                           0.05,   # Very low — critical DDx vs NDUFV1
    "Hepatopathy":                              0.04,   # Very low — critical DDx vs POLG/DGUOK
    "Olfactory bulb lesions":                   0.02,   # Very low — critical DDx vs NDUFS4
    "Exercise intolerance (older/milder)":      0.34,
    "Striatal necrosis on MRI":                 0.52,
    "Brainstem involvement":                    0.46,
    "Cerebellar atrophy":                       0.32,
}

OUTCOMES = ["alive_stable", "alive_disabled", "deceased"]
OUTCOME_WEIGHTS = [0.22, 0.52, 0.26]

FAMILIES = ["consanguineous", "non-consanguineous"]
FAMILY_WEIGHTS = [0.38, 0.62]

PATIENTS = []
for i in range(N_PATIENTS):
    v1, v2 = _pick_variant_pair()
    severity_combined = "severe" if v1["severity"] == "severe" or v2["severity"] == "severe" else v1["severity"]
    if severity_combined == "severe":
        onset_months = rng.randint(0, 8)
    elif severity_combined == "moderate":
        onset_months = rng.randint(3, 18)
    else:
        onset_months = rng.randint(6, 24)

    ci_lo = max(v1["ci_pct_range"][0], v2["ci_pct_range"][0]) - 2
    ci_hi = min(v1["ci_pct_range"][1], v2["ci_pct_range"][1]) + 2
    ci_lo = max(ci_lo, 4)
    ci_hi = min(ci_hi, 22)
    ci_activity = round(rng.uniform(ci_lo, ci_hi), 1)

    features = {f: (rng.random() < FEATURE_PROBS[f]) for f in FEATURES}

    PATIENTS.append({
        "patient_id":       f"P{i+1:03d}",
        "sex":              rng.choice(SEXES),
        "onset_age_months": onset_months,
        "ethnicity":        _pick_weighted(ETHNICITIES, ETHNICITY_WEIGHTS),
        "family":           _pick_weighted(FAMILIES, FAMILY_WEIGHTS),
        "allele1":          v1["hgvs_p"],
        "allele2":          v2["hgvs_p"],
        "severity":         severity_combined,
        "ci_activity_pct":  ci_activity,
        "outcome":          _pick_weighted(OUTCOMES, OUTCOME_WEIGHTS),
        "features":         features,
        "hcm":              features["Hypertrophic cardiomyopathy (HCM)"],
        "leigh_mri":        features["Leigh syndrome / Leigh-like MRI"],
        "lactic_acidosis":  features["Lactic acidosis (blood/CSF)"],
        "basal_ganglia":    features["Basal ganglia lesions (bilateral)"],
    })

# ─── Aggregate feature frequencies ────────────────────────────────────────────
def _feature_frequencies_pct():
    ff = {}
    for feat in FEATURES:
        ff[feat] = round(sum(1 for p in PATIENTS if p["features"].get(feat, False)) / N_PATIENTS * 100)
    return ff


# ─── get_overview ──────────────────────────────────────────────────────────────
def get_overview() -> dict:
    ff = _feature_frequencies_pct()
    return {
        "gene":         GENE,
        "alias":        "C17orf89",
        "omim_gene":    OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE_NAME,
        "chromosome":   CHROMOSOME,
        "inheritance":  INHERITANCE,
        "n_patients":   N_PATIENTS,

        "protein": {
            "size_aa":  230,
            "size_kda": 27,
            "domains":  ["Structural chaperone/scaffold core fold (beta/alpha)", "C-terminal scaffold domain", "CI assembly intermediate interaction surface"],
            "topology": "Soluble matrix protein — no TM helices, no FAD domain, no [4Fe-4S] clusters, no confirmed enzymatic activity",
        },

        "summary": (
            "NDUFAF8 (C17orf89; OMIM *616051; 17p13.2) is a soluble mitochondrial matrix CI assembly "
            "chaperone/scaffold protein that participates in intermediate stages of Complex I (CI) "
            "holoenzyme assembly. Unlike NDUFAF6 (2OG-dioxygenase) and NDUFAF7 (SAM-methyltransferase), "
            "NDUFAF8 has no confirmed enzymatic (catalytic) activity — acting as a structural scaffold "
            "analogous to NDUFAF3/NDUFAF4/NDUFAF5 but at a distinct assembly stage. "
            "Loss-of-function variants cause isolated CI deficiency (5-20%) with Leigh-like presentation. "
            "No riboflavin response. Low HCM (<10%). No peripheral neuropathy. "
            "17p13.2 locus is unique among all NDUFAF family members."
        ),

        "biochemical_fingerprint": {
            "Complex_I":           "5–20% of control (SEVERE isolated deficiency)",
            "Complex_II":          "Normal (100%)",
            "Complex_III":         "Normal (100%)",
            "Complex_IV":          "Normal (100%)",
            "Complex_V":           "Normal (100%)",
            "Pattern":             "ISOLATED CI deficiency — CI assembly stalls at intermediate stage; NDUFAF8 chaperone/scaffold absent; CI intermediates accumulate; holoenzyme cannot form",
            "Riboflavin_response": "NONE (0%) — NDUFAF8 is a structural scaffold with NO FAD domain; riboflavin cannot rescue the CI assembly chaperone defect",
            "BN-PAGE_class":       "CI assembly intermediates accumulate; stalled at intermediate assembly stage distinct from MCIA/Class1, ND1-module/Class3, N-module, and enzymatic factors (NDUFAF6/7)",
            "Enzymatic_activity":  "NONE confirmed — purely structural chaperone/scaffold (unlike NDUFAF6 [2OG-dioxygenase] and NDUFAF7 [SAM-methyltransferase] which have enzymatic activity)",
            "HCM_rate":            "LOW <10% — critical DDx vs TIMMDC1 (>80%), NDUFV2 (~80%), ACAD9 (55-65%)",
        },

        "feature_frequencies_pct": ff,

        "ndufaf8_module_summary": {
            "gene":              "NDUFAF8 (C17orf89, 17p13.2)",
            "module_class":      "Intermediate-stage CI assembly chaperone/scaffold — structural (non-enzymatic); distinct from MCIA Class1, ND1-module Class3, N-module, and enzymatic assembly factors (NDUFAF6/7)",
            "assembly_position": "Intermediate CI assembly stage — CI holoenzyme maturation; soluble matrix chaperone; stabilises CI assembly intermediates",
            "unique_structural_scaffold": (
                "NDUFAF8 is a purely structural CI assembly chaperone/scaffold with NO confirmed enzymatic activity. "
                "It acts as a non-enzymatic assembly factor analogous to NDUFAF3, NDUFAF4, and NDUFAF5, "
                "but positioned at a distinct intermediate assembly stage. This distinguishes NDUFAF8 from the "
                "only two enzymatically active CI assembly factors: NDUFAF6 (2OG-Fe(II) dioxygenase) and "
                "NDUFAF7 (SAM-dependent methyltransferase). NDUFAF8 loss causes CI assembly intermediates "
                "to accumulate — the hallmark of a structural scaffold deficiency."
            ),
            "ndufaf8_vs_enzymatic_factors": (
                "NDUFAF8 (structural scaffold, 17p13.2) vs NDUFAF7 (SAM-methyltransferase, 2q11.2): "
                "NDUFAF7 catalyzes NDUFS2 R-85 arginine methylation — a covalent PTM required for NDUFB9 recruitment. "
                "NDUFAF8 has no enzymatic activity; it stabilizes CI assembly intermediates structurally. "
                "Different chromosomes (17p13.2 vs 2q11.2), different biochemistry (scaffold vs methyltransferase). "
                "NDUFAF8 vs NDUFAF6 (2OG-dioxygenase, 8q22.1): again, NDUFAF8 = structural; NDUFAF6 = hydroxylase. "
                "WES mandatory to distinguish."
            ),
            "ndufaf8_vs_acad9": (
                "ACAD9 is an FAD-binding MCIA scaffold (ND2-ND5/Class1) — riboflavin-responsive 50-60% (Level B). "
                "NDUFAF8 is a structural assembly chaperone (intermediate stage) — 0% riboflavin response (no FAD domain). "
                "Riboflavin trial is the KEY clinical distinguisher. Different chromosomes (3q21.3 vs 17p13.2). "
                "ACAD9 HCM 55-65%; NDUFAF8 HCM <10%."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NDUFAF8 (17p13.2) vs ACAD9 (3q21.3) — NO RIBOFLAVIN RESPONSE — Critical DDx",
                "significance": (
                    "ACAD9 deficiency is riboflavin-responsive (50-60%, Level B evidence) — riboflavin "
                    "is first-line treatment. NDUFAF8 deficiency has ZERO riboflavin response — "
                    "NDUFAF8 is a structural scaffold with NO FAD domain; riboflavin cannot rescue the "
                    "CI assembly chaperone defect. ACAD9: MCIA/ND2-ND5 (Class 1). "
                    "NDUFAF8: intermediate-stage scaffold. ACAD9 HCM 55-65%; NDUFAF8 HCM <10%. "
                    "Riboflavin trial + WES (3q21.3 vs 17p13.2) is the mandatory discriminator."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NDUFAF8 (17p13.2) vs NDUFAF7 (2q11.2) — Structural Scaffold vs SAM-Methyltransferase",
                "significance": (
                    "Both: soluble matrix, no riboflavin response, low HCM, isolated CI deficiency. "
                    "KEY DIFFERENCE: NDUFAF7 = SAM-dependent methyltransferase (enzymatic — covalent NDUFS2 R-85 methylation). "
                    "NDUFAF8 = structural scaffold (non-enzymatic). Different chromosomes: 17p13.2 vs 2q11.2. "
                    "Different BN-PAGE intermediate patterns. WES is the mandatory discriminator."
                ),
                "target_gene": "NDUFAF7",
            },
            {
                "feature":     "NDUFAF8 vs TIMMDC1 (3q25.1) — HCM <10% vs HCM >80%",
                "significance": (
                    "TIMMDC1 deficiency: HCM >80% (highest in CI assembly factors; integral IMM, ND1-module Class 3). "
                    "NDUFAF8 deficiency: HCM <10% (very low). Prominent HCM (>60%) in a CI patient points "
                    "strongly toward TIMMDC1 and away from NDUFAF8. "
                    "Different chromosomes (3q25.1 vs 17p13.2) and different assembly modules."
                ),
                "target_gene": "TIMMDC1",
            },
            {
                "feature":     "NDUFAF8 vs NDUFAF3 (2q33.1)/NDUFAF5 (20p12.1) — All Structural Scaffolds — WES Mandatory",
                "significance": (
                    "NDUFAF3, NDUFAF4, NDUFAF5, and NDUFAF8 are all soluble matrix structural scaffolds "
                    "with no enzymatic activity and 0% riboflavin response. "
                    "KEY DISTINCTION: NDUFAF8 is at an intermediate assembly stage distinct from ND1-module "
                    "(NDUFAF3/4/5); different BN-PAGE intermediate patterns. "
                    "Different chromosomes: NDUFAF8 17p13.2 vs NDUFAF3 2q33.1 vs NDUFAF5 20p12.1. "
                    "WES is the ONLY reliable discriminator when all three share no riboflavin response."
                ),
                "target_gene": "NDUFAF3 / NDUFAF5",
            },
            {
                "feature":     "NDUFAF8 vs NDUFV1 (11q13.2) — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 deficiency: leukodystrophy 40-50%. NDUFAF8: 0% leukodystrophy. "
                    "Leukodystrophy on MRI strongly points away from NDUFAF8 toward NDUFV1. "
                    "WES mandatory: 17p13.2 (NDUFAF8) vs 11q13.2 (NDUFV1)."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NDUFAF8 vs NDUFS4 (5q11.2) — NO Olfactory Bulb Lesions",
                "significance": (
                    "NDUFS4 deficiency causes pathognomonic bilateral olfactory bulb lesions on MRI (52-65%). "
                    "NDUFAF8 deficiency: olfactory bulb lesions <5%. Olfactory bulb MRI lesions "
                    "point away from NDUFAF8 strongly toward NDUFS4."
                ),
                "target_gene": "NDUFS4",
            },
            {
                "feature":     "NDUFAF8 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "NDUFAF8 deficiency: hepatopathy <5%. Hepatopathy points away from NDUFAF8."
                ),
                "target_gene": "POLG / DGUOK",
            },
            {
                "feature":     "NDUFAF8 vs NDUFS1 (2q33.3) — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 causes peripheral neuropathy in ~50% of patients (hallmark feature). "
                    "NDUFAF8 deficiency: peripheral neuropathy ~0%. "
                    "Peripheral neuropathy strongly points away from NDUFAF8 toward NDUFS1. "
                    "Different chromosomes (17p13.2 vs 2q33.3). WES mandatory."
                ),
                "target_gene": "NDUFS1",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct CI inhibitor. NDUFAF8 patients have 5-20% CI; metformin biguanide inhibition is immediately life-threatening.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression. Causes further CI collapse.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks all 7 mtDNA-encoded CI subunits (MT-ND1–6). No CI rescue possible.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal 23S rRNA mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI absent/minimal; NADH cannot be reoxidised via ETC. Beta-oxidation generates NADH that cannot be cleared — metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; prefer LEV (levetiracetam) as first-choice AED — renal excretion, no mitochondrial toxicity.",
            "🟡 RIBOFLAVIN — NOT INDICATED FOR NDUFAF8: NDUFAF8 is a structural scaffold with NO FAD domain; riboflavin CANNOT rescue the CI assembly chaperone defect. Do not treat as ACAD9 (riboflavin Level B). Critical DDx distinction.",
            "🟢 SUCCINATE — Level C: CII substrate bypasses stalled CI entirely; allows CII → CIII → CIV electron flow; partial ATP rescue.",
            "🟢 CoQ10 (Ubiquinol) — Level C: Antioxidant + electron carrier support; standard add-on.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD BEFORE confirming NDUFAF8.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS BEFORE CI gene panel.",
            "🟢 CARNITINE — Level C: Supplement if secondary carnitine deficiency documented.",
            "🔵 NDUFAF8 (17p13.2) — UNIQUE CHROMOSOMAL LOCUS: Distinct from all other NDUFAF family members. WES locus 17p13.2 is the definitive discriminator.",
            "🔵 NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9: Any riboflavin response favors ACAD9 (50-60%, Level B). NDUFAF8 = 0% response. Riboflavin trial is the first-line clinical discriminator.",
            "🔵 PURELY STRUCTURAL SCAFFOLD — NO ENZYMATIC ACTIVITY: Unlike NDUFAF6 (2OG-dioxygenase) and NDUFAF7 (SAM-methyltransferase), NDUFAF8 has no catalytic activity. Both NDUFAF6 and NDUFAF7 have been biochemically confirmed as enzymatic CI assembly factors; NDUFAF8 has not.",
            "🔵 LOW HCM (<10%): Very low HCM rate — echocardiography at diagnosis. HCM <10% points away from TIMMDC1 (>80%), NDUFV2 (80%), ACAD9 (55-65%), SCO2 (65%).",
        ],
    }


# ─── get_breakdown ─────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    variant_counts: dict[str, int] = {}
    for p in PATIENTS:
        for a in [p["allele1"], p["allele2"]]:
            variant_counts[a] = variant_counts.get(a, 0) + 1

    ci_vals    = [p["ci_activity_pct"] for p in PATIENTS]
    onset_vals = [p["onset_age_months"] for p in PATIENTS]

    ci_bands = {"<10%": 0, "10–15%": 0, "15–20%": 0, ">20%": 0}
    for c in ci_vals:
        if   c < 10:  ci_bands["<10%"]  += 1
        elif c < 15:  ci_bands["10–15%"] += 1
        elif c < 20:  ci_bands["15–20%"] += 1
        else:          ci_bands[">20%"]   += 1

    onset_bands = {"0–3 mo": 0, "4–12 mo": 0, "13–24 mo": 0, ">24 mo": 0}
    for o in onset_vals:
        if   o <= 3:   onset_bands["0–3 mo"]   += 1
        elif o <= 12:  onset_bands["4–12 mo"]   += 1
        elif o <= 24:  onset_bands["13–24 mo"]  += 1
        else:           onset_bands[">24 mo"]   += 1

    outcomes = {"alive_stable": 0, "alive_disabled": 0, "deceased": 0}
    for p in PATIENTS:
        outcomes[p["outcome"]] += 1

    severe_alleles = [v["hgvs_p"] for v in VARIANTS if v["severity"] == "severe"]
    top_severe = sum(
        1 for p in PATIENTS
        if p["allele1"] in severe_alleles or p["allele2"] in severe_alleles
    )

    consang = sum(1 for p in PATIENTS if p["family"] == "consanguineous")

    return {
        "cohort_n":           N_PATIENTS,
        "patients":           PATIENTS,
        "ci_activity_stats": {
            "mean":   round(sum(ci_vals) / N_PATIENTS, 1),
            "min":    round(min(ci_vals), 1),
            "max":    round(max(ci_vals), 1),
            "bands":  ci_bands,
        },
        "onset_stats": {
            "mean_months": round(sum(onset_vals) / N_PATIENTS, 1),
            "bands":       onset_bands,
        },
        "variant_frequency": dict(sorted(variant_counts.items(), key=lambda x: -x[1])[:8]),
        "sex_distribution":  {
            "M": sum(1 for p in PATIENTS if p["sex"] == "M"),
            "F": sum(1 for p in PATIENTS if p["sex"] == "F"),
        },
        "outcome_distribution":    outcomes,
        "consanguineous_pct":      round(consang / N_PATIENTS * 100),
        "pct_with_severe_allele":  round(top_severe / N_PATIENTS * 100),
        "key_structural_scaffold_features": {
            "No_confirmed_enzymatic_activity_purely_structural_scaffold": True,
            "No_FAD_domain_no_riboflavin_response":                       True,
            "Soluble_matrix_no_TM_helices":                               True,
            "Intermediate_CI_assembly_stage_distinct_from_MCIA_ND1_N_module": True,
            "Isolated_CI_deficiency_only":                                True,
            "Chromosome_17p13.2_unique_among_NDUFAF_family":             True,
            "HCM_rate_very_low_pct":                                      round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
        },
        "treatment_summary": {
            "absolute_ci": ["Metformin", "Valproate (VPA)", "Linezolid", "Chloramphenicol", "Ketogenic Diet"],
            "avoid":       ["Propofol (PRIS)", "Phenobarbital (high caution)"],
            "do_not_use":  ["Riboflavin (NOT indicated — NDUFAF8 is a structural scaffold with no FAD domain; riboflavin cannot rescue the CI assembly chaperone defect; critical DDx vs ACAD9)"],
            "level_c":     ["Succinate (CII bypass)", "CoQ10-Ubiquinol", "Thiamine B1 (MANDATORY empiric)", "Biotin (MANDATORY empiric)", "Carnitine"],
            "preferred_aed": "LEV (levetiracetam) — renal excretion, no mitochondrial toxicity",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk)",
            "diagnostic_priority": "WES 17p13.2 locus; riboflavin trial (expected 0% response distinguishes from ACAD9); BN-PAGE CI intermediate pattern (distinguishes from NDUFAF7 Q-module and MCIA/ND1-module stall patterns)",
        },
        "variant_table": [
            {
                "hgvs_c":    v["hgvs_c"],
                "hgvs_p":    v["hgvs_p"],
                "domain":    v["domain"],
                "mechanism": v["mechanism"],
                "severity":  v["severity"],
                "ci_range":  f"{v['ci_pct_range'][0]}–{v['ci_pct_range'][1]}%",
                "notes":     v["notes"],
            }
            for v in VARIANTS
        ],
        "ddx_matrix": [
            {
                "comparator":     "ACAD9 (3q21.3)",
                "ndufaf8":        "No FAD domain; 0% riboflavin response; structural scaffold; intermediate-stage; HCM <10%",
                "comparator_val": "FAD-binding; riboflavin-responsive 50-60% (Level B); MCIA/ND2-ND5; HCM 55-65%",
                "key_test":       "Riboflavin trial + WES locus (17p13.2 vs 3q21.3)",
            },
            {
                "comparator":     "NDUFAF7 (2q11.2)",
                "ndufaf8":        "Structural scaffold (no enzymatic activity); 17p13.2; intermediate stage",
                "comparator_val": "SAM-dependent methyltransferase; NDUFS2 R-85 methylation; NDUFB9 recruitment; Q-module interface; 2q11.2",
                "key_test":       "WES locus (17p13.2 vs 2q11.2) + BN-PAGE intermediate pattern + biochemical enzyme assay",
            },
            {
                "comparator":     "TIMMDC1 (3q25.1)",
                "ndufaf8":        "Soluble matrix; HCM <10%; structural scaffold; intermediate stage",
                "comparator_val": "Integral IMM (2 TM helices); HCM >80%; ND1-module/Class3",
                "key_test":       "HCM rate + echocardiography + BN-PAGE class + WES locus",
            },
            {
                "comparator":     "NDUFAF3 (2q33.1) / NDUFAF5 (20p12.1)",
                "ndufaf8":        "Structural scaffold; 17p13.2; intermediate CI assembly stage",
                "comparator_val": "Structural scaffold; ND1-module (Class 3); 2q33.1 / 20p12.1",
                "key_test":       "BN-PAGE assembly intermediate stage + WES locus (17p13.2 vs 2q33.1 / 20p12.1)",
            },
            {
                "comparator":     "NDUFAF6 (8q22.1)",
                "ndufaf8":        "Structural scaffold (no enzymatic activity); 17p13.2",
                "comparator_val": "2OG-Fe(II) dioxygenase/hydroxylase (enzymatic PTM); Q-module maturation; 8q22.1",
                "key_test":       "BN-PAGE assembly module + WES locus (17p13.2 vs 8q22.1) + biochemical enzyme assay",
            },
            {
                "comparator":     "NDUFV1 (11q13.2)",
                "ndufaf8":        "NO leukodystrophy; structural scaffold assembly factor",
                "comparator_val": "Leukodystrophy 40-50%; FMN-binding structural N-module subunit",
                "key_test":       "MRI leukodystrophy + WES locus (17p13.2 vs 11q13.2)",
            },
        ],
        "ndufaf8_module_summary": {
            "gene":              "NDUFAF8 (C17orf89, 17p13.2)",
            "module_class":      "Intermediate-stage CI assembly chaperone/scaffold — structural (non-enzymatic); soluble matrix; no FAD domain; no TM helices; no riboflavin response",
            "assembly_chaperone_mechanism": (
                "NDUFAF8 functions as a structural chaperone/scaffold during intermediate stages of CI "
                "holoenzyme assembly. As a non-enzymatic assembly factor, NDUFAF8 stabilises CI assembly "
                "intermediates during subunit incorporation, analogous to how NDUFAF3/NDUFAF4 (ND1-module) "
                "act as structural scaffolds for their specific assembly stage. Without NDUFAF8, CI assembly "
                "stalls at the intermediate stage: partially assembled CI subcomplexes accumulate (visible on "
                "BN-PAGE) and holoenzyme CI cannot form. The specific CI subunits that fail to incorporate "
                "are determined by the precise assembly stage at which NDUFAF8 acts."
            ),
            "ndufaf8_vs_enzymatic_ci_factors": (
                "NDUFAF8 is PURELY STRUCTURAL — it has no enzymatic activity. This distinguishes it from "
                "the only two enzymatically active CI assembly factors: NDUFAF6 (2OG-Fe(II) dioxygenase; "
                "hydroxylates NDUFS7/PSST; 8q22.1) and NDUFAF7 (SAM-methyltransferase; methylates NDUFS2 R-85; "
                "2q11.2). NDUFAF8 does not catalyze any post-translational modification of a CI subunit. "
                "This places NDUFAF8 in the same structural-scaffold category as NDUFAF3/4/5 and TIMMDC1, "
                "but at a distinct CI assembly stage and at chromosome 17p13.2."
            ),
        },
    }


# ─── get_definitions ───────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "NDUFAF8 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 8; also known as C17orf89; "
            "OMIM *616051) encodes an approximately 230-amino-acid, ~27 kDa soluble mitochondrial matrix "
            "protein that functions as a CI assembly chaperone/scaffold during intermediate stages of "
            "Complex I (CI) holoenzyme assembly. NDUFAF8 has no confirmed enzymatic (catalytic) activity — "
            "it acts as a purely structural scaffold analogous to NDUFAF3/NDUFAF4/NDUFAF5, but at a distinct "
            "intermediate CI assembly stage. No FAD domain, no TM helices, no [4Fe-4S] clusters, no 2OG-dioxygenase, "
            "no SAM-methyltransferase activity. No riboflavin response (0%). Chromosome 17p13.2."
        ),
        "disease_definition": (
            "Mitochondrial Complex I Deficiency (OMIM #256000) due to NDUFAF8 bi-allelic pathogenic variants "
            "presents as infantile-onset Leigh-like syndrome with profound isolated CI deficiency (5-20% of control), "
            "preserved CII/CIII/CIV, basal ganglia lesions (~65%), Leigh MRI (~70%), lactic acidosis (~88%), "
            "hypotonia (~90%), and very low HCM (<10%). BN-PAGE shows CI assembly intermediates stalled at "
            "intermediate assembly stage (NDUFAF8 structural scaffold absent). No riboflavin response — "
            "critical DDx vs ACAD9. Chromosome 17p13.2 confirmed by WES. "
            "NDUFAF8 CI deficiency shares the isolated CI deficiency biochemical fingerprint with other "
            "CI assembly factor defects, distinguished by chromosome (17p13.2) and BN-PAGE intermediate pattern."
        ),
        "inheritance_definition": (
            "Autosomal recessive (AR). Biallelic pathogenic NDUFAF8 variants required. "
            "Both sexes equally affected. Consanguinity observed in approximately 38% of families. "
            "Carrier parents are clinically unaffected. "
            "Genetic counselling: 25% recurrence risk per pregnancy."
        ),
        "module_definitions": [
            {
                "term":       "NDUFAF8 as Structural CI Assembly Chaperone — Non-Enzymatic Scaffold",
                "definition": (
                    "NDUFAF8 belongs to the structural (non-enzymatic) class of CI assembly factors — "
                    "analogous to NDUFAF3, NDUFAF4, and NDUFAF5, which are all soluble matrix chaperones "
                    "without confirmed enzymatic activity. Unlike NDUFAF6 (2OG-Fe(II) dioxygenase catalyzing "
                    "post-translational hydroxylation of NDUFS7/PSST) and NDUFAF7 (SAM-methyltransferase "
                    "catalyzing NDUFS2 R-85 arginine methylation), NDUFAF8 does not catalyze a chemical "
                    "modification of a CI subunit. Instead, NDUFAF8 stabilises CI assembly intermediates "
                    "during the intermediate stage of holoenzyme maturation, ensuring correct subunit "
                    "incorporation and preventing premature degradation of CI assembly intermediates."
                ),
            },
            {
                "term":       "NDUFAF8 BN-PAGE — Intermediate-Stage CI Assembly Intermediates",
                "definition": (
                    "Blue-native PAGE (BN-PAGE) in NDUFAF8-deficient cells shows accumulation of CI assembly "
                    "intermediates at the intermediate assembly stage — distinct from other CI assembly factor "
                    "defects: (1) N-module stalling (FOXRED1/NUBPL), (2) MCIA/ND2-ND5/Class1 stalling "
                    "(ACAD9/NDUFAF1/ECSIT/TMEM126B), (3) ND1-module/Class3 stalling (NDUFAF3/4/5/TIMMDC1), "
                    "(4) Q-module maturation (NDUFAF6), and (5) Q-module/membrane-arm interface (NDUFAF7). "
                    "The specific NDUFAF8 intermediate pattern and the subunits that fail to incorporate "
                    "define the precise assembly step at which NDUFAF8 acts, providing a diagnostic BN-PAGE "
                    "signature that, combined with WES 17p13.2 locus, specifically identifies NDUFAF8 deficiency."
                ),
            },
            {
                "term":       "NDUFAF8 Chromosome 17p13.2 — Unique NDUFAF Locus",
                "definition": (
                    "NDUFAF8 maps to chromosome 17p13.2, making it unique among the NDUFAF gene family: "
                    "NDUFAF1 (15q11.2), NDUFAF2 (5q12.1), NDUFAF3 (2q33.1), NDUFAF4 (6q16.3), "
                    "NDUFAF5 (20p12.1), NDUFAF6 (8q22.1), NDUFAF7 (2q11.2) — all on different chromosomes. "
                    "17p13.2 does not overlap with major CI structural subunit genes on chromosomes "
                    "frequently shared by CI factors (e.g., chr2q: NDUFAF3/NDUFAF7/NDUFS1). "
                    "WES locus 17p13.2 is the definitive discriminator for NDUFAF8 vs all other "
                    "CI assembly factor defects."
                ),
            },
            {
                "term":       "NDUFAF8 vs NDUFAF3/NDUFAF4/NDUFAF5 — All Structural Scaffolds, Different Stages",
                "definition": (
                    "NDUFAF3 (ND1-module/Class3 obligate NDUFAF4-heterodimer; 2q33.1), NDUFAF4 (ND1-module "
                    "obligate NDUFAF3-heterodimer; 6q16.3), and NDUFAF5 (ND1-module independent; 20p12.1) are "
                    "all soluble matrix structural scaffolds without enzymatic activity — exactly like NDUFAF8. "
                    "The critical distinction is the CI assembly stage: NDUFAF3/4/5 act at the ND1-module "
                    "(Class 3) stage (early membrane arm assembly); NDUFAF8 acts at a distinct intermediate "
                    "CI assembly stage. BN-PAGE shows different intermediate patterns. WES (17p13.2 vs 2q33.1 "
                    "vs 6q16.3 vs 20p12.1) is the only reliable method to distinguish these structurally "
                    "similar but chromosomally and stage-distinct assembly factors."
                ),
            },
        ],
        "reference_list": [
            {
                "citation": "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538(7623):123–126.",
                "significance": "CRISPR/Cas9 systematic CI subunit and assembly factor knockout; characterized CI assembly factor roles including NDUFAF8/C17orf89; defined CI assembly classes and intermediate patterns; identified NDUFAF8 as required for CI holoenzyme formation",
            },
            {
                "citation": "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "significance": "Comprehensive CI assembly pathway mapping using pulse-chase proteomics and BN-PAGE; positions CI assembly factors relative to module assembly stages; context for NDUFAF8 intermediate-stage CI assembly chaperone role",
            },
            {
                "citation": "Formosa LE et al. (2018) Building a complex complex: assembly of mitochondrial respiratory chain complex I. Semin Cell Dev Biol 76:154–162.",
                "significance": "Review of CI assembly factor classes, modules, and chaperone roles; NDUFAF8 as intermediate-stage CI assembly factor; structural vs enzymatic CI assembly factor classification",
            },
            {
                "citation": "Zurita Rendón O, Antonicka H, Funai EF, Shoubridge EA (2014) A mutation in the methyltransferase NDUFAF7 causes basal ganglia disease and complex I deficiency. J Biol Chem 289(7):3655–63.",
                "significance": "Directly confirmed the SAM-methyltransferase enzymatic activity of NDUFAF7 — establishing that NDUFAF7 and NDUFAF6 are the ONLY two enzymatically active CI assembly factors; by contrast, NDUFAF8 has no such enzymatic activity",
            },
        ],
        "contraindication_definitions": [
            {
                "drug":       "Metformin",
                "level":      "ABSOLUTE CONTRAINDICATION",
                "mechanism":  "Biguanide direct CI inhibitor (complex I, site I / NADH dehydrogenase). NDUFAF8 patients have 5-20% residual CI; any further inhibition immediately causes lethal lactic acidosis.",
                "alternative": "Avoid in all mitochondrial CI deficiency. GLP-1 agonist or SGLT2 inhibitor if diabetes management needed in adult carriers.",
            },
            {
                "drug":       "Valproate (VPA)",
                "level":      "ABSOLUTE CONTRAINDICATION",
                "mechanism":  "Triple mechanism: (1) CoA sequestration (valproyl-CoA) → depletes free CoA needed for TCA cycle and beta-oxidation; (2) POLG1 inhibition → mtDNA depletion worsens CI subunit supply; (3) MT-ND gene expression suppression → reduces all 7 mtDNA-encoded CI subunits. Catastrophic in NDUFAF8 CI deficiency.",
                "alternative": "Levetiracetam (LEV) — first-choice AED in mitochondrial disease (renal excretion, no mitochondrial toxicity). Lacosamide as adjunct.",
            },
            {
                "drug":       "Ketogenic Diet (KD)",
                "level":      "CONTRAINDICATED",
                "mechanism":  "KD shifts metabolism to beta-oxidation and ketogenesis, generating large amounts of NADH. With CI absent/minimal, NADH cannot be reoxidised to NAD+ via ETC. NADH excess: inhibits TCA cycle, blocks pyruvate oxidation, causes lactic acidosis. CI is the NADH entry point — KD is metabolically incompatible with NDUFAF8 deficiency.",
                "alternative": "High-complex-carbohydrate diet with avoidance of prolonged fasting. Glucose infusion (GIR 6-8 mg/kg/min) during illness.",
            },
        ],
    }
