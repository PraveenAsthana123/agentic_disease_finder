#!/usr/bin/env python3
"""NDUFAF7 (C2orf56 / MIDI1IP1) — Mitochondrial Complex I Deficiency MC1DN30 (SAM-Dependent Methyltransferase).

NDUFAF7 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 7; also known as C2orf56) is a
class I SAM-dependent methyltransferase dedicated to Complex I (CI) assembly. It is the ONLY
CI assembly factor with established S-adenosylmethionine (SAM)-dependent methyltransferase
enzymatic activity, catalyzing arginine methylation of NDUFS2 (Q-module subunit, R-114 mature)
required for Q-module assembly and NDUFB9 binding.

  NDUFAF7 gene   OMIM *615898
  Disease        Mitochondrial Complex I Deficiency MC1DN30 (OMIM #618248)
  Inheritance    AR (autosomal recessive, biallelic)
  Chromosome     2q11.2

Reference: Zurita Rendón O, Antonicka H, Funai EF, Shoubridge EA (2014) A mutation in the
methyltransferase NDUFAF7 causes basal ganglia disease and complex I deficiency. J Biol Chem
289(7):3655–63. (First NDUFAF7/C2orf56 CI assembly factor identification and disease gene
characterization; p.Arg321Pro founder mutation)

Reference: Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory
chain complex I. Cell Metab 25(1):128–139. (CI assembly intermediate mapping; Q-module context
of NDUFAF7 and NDUFS2 arginine methylation)

Reference: Rhein VF, Carroll J, Ding S, Fearnley IM, Walker JE (2016) NDUFAF7 methylates
arginine 85 in the NDUFS2 subunit of human complex I. J Biol Chem 291(6):2909–18. (Definitive
methylation site mapping — R-85 in processed mature NDUFS2; SAM-dependent mechanism; NDUFB9
recruitment dependency)

PATHOPHYSIOLOGY (NDUFAF7 / SAM-Dependent Methyltransferase / NDUFS2 Arginine Methylation):
  NDUFAF7/C2orf56 is a class I SAM-dependent methyltransferase that acts during Q-module
  assembly of CI:
    1. NDUFAF7 uses S-adenosylmethionine (SAM) as methyl donor to methylate arginine-85
       of the mature NDUFS2 subunit (equivalent to R-114 including the mitochondrial
       targeting sequence; R-85 in the fully processed mature form per Rhein 2016).
    2. This post-translational arginine methylation of NDUFS2 is required for NDUFB9
       (a membrane arm subunit) to bind to the Q-module assembly intermediate during
       CI assembly progression.
    3. Without NDUFAF7-mediated NDUFS2 R-85 methylation, the Q-module intermediate
       cannot recruit NDUFB9; assembly stalls at the Q-module/membrane-arm interface
       stage; holoenzyme CI cannot form.
    4. Isolated CI deficiency (5–20%); CII/CIII/CIV normal — block is Q-module/
       membrane-arm interface.
    5. No FAD domain, no TM helices — soluble matrix protein with class I SAM-dependent
       methyltransferase fold (Rossmann-like beta/alpha).

NDUFAF7 UNIQUE FEATURES vs OTHER CI ASSEMBLY FACTORS:
  1. ONLY CI ASSEMBLY FACTOR WITH SAM-DEPENDENT METHYLTRANSFERASE ACTIVITY:
     NDUFAF7 is the sole known CI assembly factor that catalyzes S-adenosylmethionine
     (SAM)-dependent arginine methylation. All other CI assembly factors act as chaperones,
     scaffolds, Fe-S delivery factors, or hydroxylases — none as SAM-dependent
     methyltransferases. This is unique among all known CI assembly factors.
  2. POST-TRANSLATIONAL ARGININE METHYLATION OF NDUFS2 (R-85 MATURE):
     The methylation target is NDUFS2 arginine-85 (mature processed form; R-114
     including MTS per Zurita Rendón 2014; confirmed R-85 by Rhein 2016). This
     modification is required for NDUFB9 binding during Q-module/membrane-arm
     interface assembly.
  3. Q-MODULE / MEMBRANE-ARM INTERFACE STAGE:
     NDUFAF7 acts at the Q-module/membrane-arm junction — a stage distinct from:
     N-module (FOXRED1/NUBPL), ND2-ND5/MCIA (ACAD9/NDUFAF1/ECSIT/TMEM126B),
     ND1-module/Class3 (NDUFAF3/4/5/TIMMDC1), and late-stage 2OG (NDUFAF6/Q-module
     NDUFS7 hydroxylation). BN-PAGE shows Q-module/membrane-arm interface intermediates.
  4. NO RIBOFLAVIN RESPONSE (0%): No FAD domain. Riboflavin supplementation CANNOT
     rescue the SAM-methyltransferase defect. Critical DDx vs ACAD9 (50-60%).
  5. LOW HCM (<10%): Very low HCM compared to TIMMDC1 (>80%), NDUFV2 (80%),
     ACAD9 (55-65%). Important DDx marker.
  6. 2q11.2 — UNIQUE CHROMOSOMAL LOCUS; distinct from all other CI assembly factors.

NDUFAF7 vs NDUFAF6 — TWO ENZYMATIC CI ASSEMBLY FACTORS, DIFFERENT MECHANISMS:
  NDUFAF7 (2q11.2): SAM-dependent methyltransferase; methylates NDUFS2 R-85; NDUFB9
    recruitment; Q-module/membrane-arm interface.
  NDUFAF6 (8q22.1): 2OG-Fe(II)-dependent dioxygenase/hydroxylase; hydroxylates NDUFS7/
    PSST; Q-module maturation. Different substrates, different chemistry, different
    chromosomes, different assembly steps. Both: soluble matrix, no FAD, no riboflavin
    response, isolated CI deficiency. WES mandatory to distinguish (2q11.2 vs 8q22.1).

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs ACAD9 (3q21.3): ACAD9 = MCIA/ND2-ND5 (Class 1); riboflavin-responsive 50-60%.
    NDUFAF7 = SAM-methyltransferase; 0% riboflavin response. Key test: riboflavin trial.
  vs NDUFAF6 (8q22.1): Both enzymatic CI assembly factors. NDUFAF7 = methyltransferase/
    NDUFS2. NDUFAF6 = dioxygenase/NDUFS7. Different chromosomes. WES mandatory.
  vs NDUFAF5 (20p12.1): Both no-FAD, no riboflavin response. NDUFAF5 = ND1-module Class3.
    NDUFAF7 = methyltransferase/Q-module. WES mandatory: 2q11.2 vs 20p12.1.
  vs FOXRED1 (11q24.2): FOXRED1 = N-module FAD-oxidoreductase chaperone (no riboflavin).
    NDUFAF7 = Q-module methyltransferase. Different modules, different biochemistry.
  vs TIMMDC1 (3q25.1): TIMMDC1 HCM >80%; NDUFAF7 HCM <10%. Key DDx marker.
  vs NDUFS1 (2q33.3): NDUFS1 peripheral neuropathy ~50%. NDUFAF7: 0%. IMPORTANT:
    NDUFAF7 (2q11.2) and NDUFS1 (2q33.3) are on the SAME CHROMOSOME 2 — WES locus
    is critical: 2q11.2 vs 2q33.3 on the SAME arm (same q-arm, different bands).
  vs POLG/DGUOK: NDUFAF7: NO hepatopathy.
"""

import random
import math

SEED = 697
rng  = random.Random(SEED)

GENE         = "NDUFAF7"
OMIM_GENE    = "615898"
OMIM_DISEASE = "618248"
DISEASE_NAME = "Mitochondrial Complex I Deficiency MC1DN30 (OMIM #618248)"
CHROMOSOME   = "2q11.2"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ─────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.962G>C",
        "hgvs_p":    "p.Arg321Pro",
        "domain":    "C-terminal SAM-binding domain — class I methyltransferase fold",
        "mechanism": "Arginine-to-proline substitution disrupts the C-terminal SAM-binding domain helical structure; helix-breaking proline abolishes SAM cofactor coordination; NDUFS2 R-85 methylation completely eliminated; NDUFB9 cannot bind Q-module",
        "severity":  "severe",
        "ci_pct_range": (5, 14),
        "notes":     "First reported NDUFAF7 pathogenic variant (Zurita Rendón 2014 J Biol Chem); South American consanguineous family; severe infantile CI deficiency; direct disruption of SAM-binding helix",
    },
    {
        "hgvs_c":    "c.455T>C",
        "hgvs_p":    "p.Leu152Pro",
        "domain":    "Beta-strand 4 — central methyltransferase core (Rossmann-like fold)",
        "mechanism": "Helix-breaking proline substitution destroys beta-strand continuity in the central SAM-dependent methyltransferase Rossmann-like fold; protein misfolding; loss of SAM-binding and NDUFS2 methylation activity",
        "severity":  "severe",
        "ci_pct_range": (5, 13),
        "notes":     "Severe infantile Leigh-like onset; helix-breaking proline in methyltransferase core fold; complete loss of enzymatic activity",
    },
    {
        "hgvs_c":    "c.641G>C",
        "hgvs_p":    "p.Gly214Arg",
        "domain":    "Beta-strand 6 — SAM-binding core; GxGxxG-like glycine-rich loop region",
        "mechanism": "Glycine-to-arginine substitution at a conserved glycine in the SAM-binding glycine-rich region; bulky arginine disrupts cofactor binding loop geometry; NDUFS2 methylation severely reduced",
        "severity":  "severe",
        "ci_pct_range": (5, 14),
        "notes":     "Conserved glycine essential for SAM-binding loop flexibility; severe CI deficiency; Leigh-like MRI pattern",
    },
    {
        "hgvs_c":    "c.926C>T",
        "hgvs_p":    "p.Ala309Val",
        "domain":    "C-terminal domain — hydrophobic core packing adjacent to SAM-binding region",
        "mechanism": "Alanine-to-valine substitution disrupts hydrophobic core packing in the C-terminal domain; slight steric clash causes partial misfolding; residual SAM-binding activity reduced 50-70%",
        "severity":  "intermediate",
        "ci_pct_range": (11, 18),
        "notes":     "Intermediate severity; partial NDUFS2 methylation preserved; later onset (4-12 months); exercise intolerance dominant early feature",
    },
    {
        "hgvs_c":    "c.535C>T",
        "hgvs_p":    "p.Arg179Cys",
        "domain":    "Alpha-helix C — NDUFS2 substrate recognition interface",
        "mechanism": "Arginine-to-cysteine substitution at the NDUFS2/R-85 substrate-recognition interface; disrupts electrostatic interactions with NDUFS2; arginine methylation efficiency severely reduced; NDUFB9 recruitment impaired",
        "severity":  "severe",
        "ci_pct_range": (5, 13),
        "notes":     "Substrate-recognition interface mutation; NDUFS2 binding impaired directly; NDUFB9 cannot be recruited; complete assembly stall",
    },
    {
        "hgvs_c":    "c.IVS4+1G>A",
        "hgvs_p":    "p.splice (intron 4 donor)",
        "domain":    "Splice donor site — intron 4",
        "mechanism": "Canonical splice donor disruption; exon 4 skipping or cryptic splice activation; loss of SAM-binding domain segment (exon 4 encodes key methyltransferase core beta-strands); partial residual CI from minor cryptic splice product",
        "severity":  "moderate",
        "ci_pct_range": (10, 18),
        "notes":     "Splice donor variant; partial CI residual activity from minor in-frame cryptic splicing; moderate phenotype with subacute Leigh-like presentation",
    },
    {
        "hgvs_c":    "c.774G>A",
        "hgvs_p":    "p.Trp258Ter",
        "domain":    "C-terminal domain — premature truncation (nonsense)",
        "mechanism": "Premature stop codon truncates protein before C-terminal SAM-binding helix cluster; effectively null allele; complete loss of methyltransferase function; no NDUFS2 methylation; no NDUFB9 recruitment",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Null allele; often compound-heterozygous with missense; neonatal to early infantile onset; worst prognosis",
    },
]

# ─── Patient cohort ────────────────────────────────────────────────────────────
ETHNICITIES = ["European", "South-American", "Middle-Eastern (consanguineous)", "South-Asian", "East-Asian", "North-African (consanguineous)", "Pan-Ethnic"]
ETHNICITY_WEIGHTS = [0.28, 0.22, 0.18, 0.14, 0.08, 0.07, 0.03]
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
    # Weighted allele distribution based on reported frequencies
    allele_weights = [0.28, 0.14, 0.14, 0.12, 0.12, 0.10, 0.10]
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
    "Leigh syndrome / Leigh-like MRI":          0.72,
    "Lactic acidosis (blood/CSF)":              0.88,
    "Developmental delay / regression":         0.94,
    "Hypotonia (axial/generalized)":            0.90,
    "Seizures (multiple types)":                0.52,
    "Basal ganglia lesions (bilateral)":        0.68,
    "Feeding difficulties / failure to thrive": 0.78,
    "Respiratory compromise":                   0.42,
    "Hypertrophic cardiomyopathy (HCM)":        0.08,   # Low HCM (<10%)
    "Peripheral neuropathy":                    0.04,   # Very low — critical DDx vs NDUFS1
    "Optic atrophy":                            0.18,
    "Leukodystrophy":                           0.06,   # Very low — critical DDx vs NDUFV1
    "Hepatopathy":                              0.04,   # Very low — critical DDx vs POLG/DGUOK
    "Olfactory bulb lesions":                   0.02,   # Very low — critical DDx vs NDUFS4
    "Exercise intolerance (older/milder)":      0.32,
    "Striatal necrosis on MRI":                 0.55,
    "Brainstem involvement":                    0.48,
    "Cerebellar atrophy":                       0.34,
}

OUTCOMES = ["alive_stable", "alive_disabled", "deceased"]
OUTCOME_WEIGHTS = [0.22, 0.52, 0.26]

FAMILIES = ["consanguineous", "non-consanguineous"]
FAMILY_WEIGHTS = [0.35, 0.65]

PATIENTS = []
for i in range(N_PATIENTS):
    v1, v2 = _pick_variant_pair()
    severity_combined = "severe" if v1["severity"] == "severe" or v2["severity"] == "severe" else v1["severity"]
    if severity_combined == "severe":
        onset_months = rng.randint(0, 8)
    elif severity_combined == "moderate":
        onset_months = rng.randint(4, 18)
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
        "alias":        "C2orf56 / MIDI1IP1",
        "omim_gene":    OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE_NAME,
        "chromosome":   CHROMOSOME,
        "inheritance":  INHERITANCE,
        "n_patients":   N_PATIENTS,

        "protein": {
            "size_aa":  371,
            "size_kda": 41,
            "domains":  ["Class I SAM-dependent methyltransferase fold (Rossmann-like beta/alpha)", "SAM-binding glycine-rich loop (GxGxxG-like)", "NDUFS2 substrate-recognition interface helix", "C-terminal SAM-binding helix cluster"],
            "topology": "Soluble matrix protein — no TM helices, no FAD domain, no [4Fe-4S] clusters, no 2OG-dioxygenase activity",
        },

        "summary": (
            "NDUFAF7 (C2orf56 / MIDI1IP1; OMIM *615898; 2q11.2) is a class I SAM-dependent methyltransferase "
            "that catalyzes arginine-85 methylation of NDUFS2 (Q-module subunit) during CI assembly. "
            "NDUFAF7 is the ONLY CI assembly factor with SAM-dependent methyltransferase activity — "
            "analogous to NDUFAF6 being the only one with 2OG-dioxygenase activity. "
            "NDUFS2 R-85 methylation is required for NDUFB9 to bind the Q-module/membrane-arm interface, "
            "completing late-stage CI assembly. Loss-of-function variants cause isolated CI deficiency "
            "(5-20%) with Leigh-like presentation, basal ganglia lesions, and low HCM (<10%). "
            "NO riboflavin response. NO peripheral neuropathy. NO leukodystrophy. NO hepatopathy. "
            "Critical DDx note: NDUFAF7 (2q11.2) and NDUFS1 (2q33.3) are on the SAME chromosome 2 — "
            "WES locus discrimination is mandatory (2q11.2 vs 2q33.3)."
        ),

        "biochemical_fingerprint": {
            "Complex_I":              "5–20% of control (SEVERE isolated deficiency)",
            "Complex_II":             "Normal (100%)",
            "Complex_III":            "Normal (100%)",
            "Complex_IV":             "Normal (100%)",
            "Complex_V":              "Normal (100%)",
            "Pattern":                "ISOLATED CI deficiency — Q-module/membrane-arm interface assembly block; NDUFS2 R-85 unmethylated; NDUFB9 cannot bind; Q-module maturation stalled",
            "Riboflavin_response":    "NONE (0%) — NDUFAF7 is a SAM-dependent methyltransferase with NO FAD domain; riboflavin cannot rescue the NDUFS2 arginine methylation defect",
            "BN-PAGE_class":          "Q-module/membrane-arm interface assembly intermediates stalled; distinct from N-module (FOXRED1/NUBPL), MCIA/Class1 (ACAD9), ND1-module/Class3 (NDUFAF3/4/5/TIMMDC1), and NDUFAF6 Q-module intermediates",
            "SAM_methyltransferase_unique": "NDUFAF7 is the ONLY CI assembly factor with SAM-dependent methyltransferase activity — methylates NDUFS2 R-85 for NDUFB9 recruitment; no other CI factor is a methyltransferase",
            "HCM_rate":               "LOW <10% — critical DDx vs TIMMDC1 (>80%), NDUFV2 (~80%), ACAD9 (55-65%), SCO2 (65%)",
            "NDUFS2_R85_methylation": "NDUFAF7 methylates NDUFS2 R-85 (mature form); NDUFB9 binding requires this modification; Q-module/membrane-arm assembly stalls without it",
        },

        "feature_frequencies_pct": ff,

        "ndufaf7_module_summary": {
            "gene":              "NDUFAF7 (C2orf56/MIDI1IP1, 2q11.2)",
            "module_class":      "Q-module/membrane-arm interface assembly — class I SAM-dependent methyltransferase; NDUFS2 R-85 arginine methylation; NDUFB9 recruitment factor",
            "assembly_position": "Q-module/membrane-arm junction — NDUFS2 R-85 methylation; after NDUFAF6 Q-module NDUFS7 hydroxylation in assembly sequence; NDUFB9 binding step",
            "unique_sam_methyltransferase": (
                "NDUFAF7 is the ONLY CI assembly factor with S-adenosylmethionine (SAM)-dependent "
                "methyltransferase activity. Using SAM as methyl donor, NDUFAF7 methylates arginine-85 "
                "of the mature NDUFS2 subunit (Q-module). This arginine methylation is required for "
                "NDUFB9 (a membrane arm accessory subunit) to bind the Q-module assembly intermediate, "
                "enabling the Q-module/membrane-arm interface assembly step. "
                "All other CI assembly factors act as chaperones, scaffolds, Fe-S delivery factors, "
                "or hydroxylases — none catalyze a SAM-dependent methylation of a CI subunit."
            ),
            "ndufaf7_vs_ndufaf6": (
                "NDUFAF7 and NDUFAF6 are the ONLY two CI assembly factors with enzymatic activity: "
                "NDUFAF7 = SAM-dependent methyltransferase; target = NDUFS2 R-85; recruits NDUFB9. "
                "NDUFAF6 = 2OG-Fe(II)-dependent dioxygenase/hydroxylase; target = NDUFS7/PSST; Q-module maturation. "
                "Different substrates (NDUFS2 vs NDUFS7), different chemistries (methyltransfer vs hydroxylation), "
                "different cofactors (SAM vs 2OG+Fe(II)), different chromosomes (2q11.2 vs 8q22.1). "
                "Both: soluble matrix, no FAD, 0% riboflavin response, isolated CI deficiency, low HCM. "
                "WES is the mandatory discriminator."
            ),
            "ndufaf7_vs_acad9": (
                "ACAD9 is an FAD-binding MCIA scaffold (ND2-ND5/Class1) — riboflavin-responsive 50-60% (Level B). "
                "NDUFAF7 is a SAM-methyltransferase (Q-module interface) — 0% riboflavin response (no FAD domain). "
                "Riboflavin trial is the KEY clinical distinguisher. Different modules, different chromosomes "
                "(3q21.3 vs 2q11.2). ACAD9 HCM 55-65%; NDUFAF7 HCM <10%."
            ),
            "ndufs2_r85_methylation_mechanism": (
                "NDUFS2 (49 kDa Q-module subunit) arginine-85 (R-85, mature form) is the methylation "
                "target of NDUFAF7. R-85 is located in a conserved region of NDUFS2 at the interface "
                "between the Q-module matrix domain and the membrane arm. Methylation of R-85 creates "
                "a recognition signal for NDUFB9 (a Type I membrane arm accessory subunit), which "
                "must bind to proceed through the Q-module/membrane-arm interface assembly checkpoint. "
                "Without R-85 methylation, NDUFB9 cannot bind, the Q-module/membrane assembly stalls, "
                "and holoenzyme CI cannot form. This methylation checkpoint is unique to NDUFAF7."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NDUFAF7 (2q11.2) vs ACAD9 (3q21.3) — NO RIBOFLAVIN RESPONSE — Critical DDx",
                "significance": (
                    "ACAD9 deficiency is riboflavin-responsive (50-60%, Level B evidence) — riboflavin "
                    "is first-line treatment. NDUFAF7 deficiency has ZERO riboflavin response — "
                    "NDUFAF7 is a SAM-methyltransferase with NO FAD domain; riboflavin cannot rescue "
                    "the NDUFS2 R-85 methylation defect. ACAD9: MCIA/ND2-ND5 (Class 1). "
                    "NDUFAF7: SAM-methyltransferase/Q-module interface. ACAD9 HCM 55-65%; NDUFAF7 HCM <10%. "
                    "Riboflavin trial + WES (3q21.3 vs 2q11.2) is the mandatory discriminator."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NDUFAF7 (2q11.2) vs NDUFAF6 (8q22.1) — TWO ENZYMATIC CI ASSEMBLY FACTORS — Different Chemistry",
                "significance": (
                    "NDUFAF7 and NDUFAF6 are the ONLY two CI assembly factors with enzymatic activity. "
                    "KEY DIFFERENCES: NDUFAF7 = SAM-methyltransferase; NDUFAF6 = 2OG-dioxygenase. "
                    "Different substrates: NDUFS2 R-85 (NDUFAF7) vs NDUFS7/PSST (NDUFAF6). "
                    "Different cofactors: SAM (NDUFAF7) vs 2OG+Fe(II) (NDUFAF6). "
                    "Different chromosomes: 2q11.2 (NDUFAF7) vs 8q22.1 (NDUFAF6). "
                    "Different BN-PAGE intermediates. WES is the mandatory discriminator."
                ),
                "target_gene": "NDUFAF6",
            },
            {
                "feature":     "NDUFAF7 (2q11.2) vs NDUFS1 (2q33.3) — SAME CHROMOSOME 2 — WES Locus Critical",
                "significance": (
                    "NDUFAF7 (2q11.2) and NDUFS1 (2q33.3) are on the SAME chromosome 2, SAME q-arm. "
                    "CRITICAL DDx: NDUFS1 deficiency causes peripheral neuropathy in ~50% — a hallmark. "
                    "NDUFAF7: peripheral neuropathy 0%. Peripheral neuropathy STRONGLY points toward NDUFS1. "
                    "WES locus discrimination is mandatory: 2q11.2 (NDUFAF7) vs 2q33.3 (NDUFS1). "
                    "These are ~62 Mb apart on chr2q — WES panels resolve them routinely. "
                    "Do not confuse these two chr2q genes in clinical practice."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "NDUFAF7 vs TIMMDC1 (3q25.1) — HCM <10% vs HCM >80%",
                "significance": (
                    "TIMMDC1 deficiency: HCM >80% (highest in CI assembly factors; integral IMM, ND1-module Class 3). "
                    "NDUFAF7 deficiency: HCM <10% (very low). Prominent HCM (>60%) in a CI patient points "
                    "strongly toward TIMMDC1 and away from NDUFAF7. "
                    "Completely different assembly modules: NDUFAF7 SAM-methyltransferase/Q-module vs TIMMDC1 integral-IMM/ND1-module."
                ),
                "target_gene": "TIMMDC1",
            },
            {
                "feature":     "NDUFAF7 vs NDUFV1 (11q13.2) — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 deficiency: leukodystrophy 40-50%. NDUFAF7: 0% leukodystrophy. "
                    "Leukodystrophy on MRI strongly points away from NDUFAF7 toward NDUFV1. "
                    "WES mandatory: 2q11.2 (NDUFAF7) vs 11q13.2 (NDUFV1)."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NDUFAF7 vs NDUFS4 (5q11.2) — NO Olfactory Bulb Lesions",
                "significance": (
                    "NDUFS4 deficiency causes pathognomonic bilateral olfactory bulb lesions on MRI (52-65%). "
                    "NDUFAF7 deficiency: olfactory bulb lesions <5%. Olfactory bulb MRI lesions "
                    "point away from NDUFAF7 strongly toward NDUFS4."
                ),
                "target_gene": "NDUFS4",
            },
            {
                "feature":     "NDUFAF7 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "NDUFAF7 deficiency: hepatopathy <5%. Hepatopathy points away from NDUFAF7."
                ),
                "target_gene": "POLG / DGUOK",
            },
            {
                "feature":     "NDUFAF7 vs NDUFAF5 (20p12.1) — Both No-FAD, No-Riboflavin — WES Mandatory",
                "significance": (
                    "Both NDUFAF7 and NDUFAF5 have no FAD domain and 0% riboflavin response. "
                    "KEY DIFFERENCES: NDUFAF5 = ND1-module/Class3/scaffold. NDUFAF7 = SAM-methyltransferase/Q-module. "
                    "Different assembly modules and BN-PAGE stalling patterns. "
                    "Different chromosomes: 2q11.2 (NDUFAF7) vs 20p12.1 (NDUFAF5). "
                    "WES is the mandatory discriminator when riboflavin response is absent."
                ),
                "target_gene": "NDUFAF5",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct CI inhibitor. NDUFAF7 patients have 5-20% CI; metformin biguanide inhibition is immediately life-threatening.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression. Causes further CI collapse.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks all 7 mtDNA-encoded CI subunits (MT-ND1–6). No CI rescue possible.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal 23S rRNA mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI absent/minimal; NADH cannot be reoxidised via ETC. Beta-oxidation generates NADH that cannot be cleared — metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; prefer LEV (levetiracetam) as first-choice AED — renal excretion, no mitochondrial toxicity.",
            "🟡 RIBOFLAVIN — NOT INDICATED FOR NDUFAF7: NDUFAF7 is a SAM-dependent methyltransferase with NO FAD domain; riboflavin CANNOT rescue the NDUFS2 R-85 methylation defect. Do not treat as ACAD9 (riboflavin Level B). Critical DDx distinction.",
            "🟢 SUCCINATE — Level C: CII substrate bypasses stalled CI entirely; allows CII → CIII → CIV electron flow; partial ATP rescue.",
            "🟢 CoQ10 (Ubiquinol) — Level C: Antioxidant + electron carrier support; standard add-on.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD BEFORE confirming NDUFAF7.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS BEFORE CI gene panel.",
            "🟢 CARNITINE — Level C: Supplement if secondary carnitine deficiency documented.",
            "🔵 NDUFAF7 (2q11.2) — ONLY CI Assembly Factor with SAM-Dependent Methyltransferase Activity: All other CI assembly factors are chaperones, scaffolds, or a dioxygenase (NDUFAF6). NDUFAF7 uniquely catalyzes NDUFS2 R-85 arginine methylation via SAM — a covalent PTM of a CI subunit required for NDUFB9 recruitment.",
            "🔵 NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9: Any riboflavin response favors ACAD9 (50-60%, Level B). NDUFAF7 = 0% response. Riboflavin trial is the first-line clinical discriminator.",
            "🔵 CHROMOSOME 2q11.2 vs NDUFS1 2q33.3 — SAME CHROMOSOME: Both are on chr2q. Clinical DDx: NDUFS1 has peripheral neuropathy ~50%; NDUFAF7 has 0%. WES locus (2q11.2 vs 2q33.3) is mandatory — ~62 Mb separation on the same q-arm.",
            "🔵 LOW HCM (<10%): Very low HCM rate — echocardiography at diagnosis. HCM <10% points away from TIMMDC1 (>80%), NDUFV2 (80%), ACAD9 (55-65%), SCO2 (65%). HCM rate is a critical DDx marker.",
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
        "key_sam_methyltransferase_features": {
            "Only_CI_assembly_factor_with_SAM_methyltransferase_activity": True,
            "NDUFS2_R85_arginine_methylation_target":                      True,
            "NDUFB9_recruitment_requires_NDUFS2_R85_methylation":          True,
            "Q_module_membrane_arm_interface_assembly_step":               True,
            "No_FAD_domain_no_riboflavin_response":                        True,
            "Soluble_matrix_no_TM_helices":                                True,
            "Isolated_CI_deficiency_only":                                 True,
            "HCM_rate_very_low_pct":                                       round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
            "Same_chr2q_as_NDUFS1_critical_WES_locus":                     True,
        },
        "treatment_summary": {
            "absolute_ci": ["Metformin", "Valproate (VPA)", "Linezolid", "Chloramphenicol", "Ketogenic Diet"],
            "avoid":       ["Propofol (PRIS)", "Phenobarbital (high caution)"],
            "do_not_use":  ["Riboflavin (NOT indicated — NDUFAF7 is a SAM-methyltransferase with no FAD domain; riboflavin cannot rescue NDUFS2 R-85 methylation defect; critical DDx vs ACAD9)"],
            "level_c":     ["Succinate (CII bypass)", "CoQ10-Ubiquinol", "Thiamine B1 (MANDATORY empiric)", "Biotin (MANDATORY empiric)", "Carnitine"],
            "preferred_aed": "LEV (levetiracetam) — renal excretion, no mitochondrial toxicity",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk)",
            "diagnostic_priority": "SAM-methyltransferase activity assay; WES 2q11.2 locus; riboflavin trial (expected 0% response distinguishes from ACAD9); peripheral neuropathy evaluation (distinguishes from NDUFS1 2q33.3)",
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
                "ndufaf7":        "No FAD domain; 0% riboflavin response; SAM-methyltransferase; Q-module interface; HCM <10%",
                "comparator_val": "FAD-binding; riboflavin-responsive 50-60% (Level B); MCIA/ND2-ND5; HCM 55-65%",
                "key_test":       "Riboflavin trial + WES locus (2q11.2 vs 3q21.3)",
            },
            {
                "comparator":     "NDUFAF6 (8q22.1)",
                "ndufaf7":        "SAM-methyltransferase; NDUFS2 R-85 substrate; 2q11.2",
                "comparator_val": "2OG-Fe(II) dioxygenase/hydroxylase; NDUFS7/PSST substrate; 8q22.1",
                "key_test":       "WES locus (2q11.2 vs 8q22.1) + BN-PAGE intermediate pattern + biochemical cofactor assay",
            },
            {
                "comparator":     "NDUFS1 (2q33.3) — SAME CHR2q",
                "ndufaf7":        "NO peripheral neuropathy; SAM-methyltransferase assembly factor; 2q11.2",
                "comparator_val": "Peripheral neuropathy ~50%; structural N-module subunit; 2q33.3",
                "key_test":       "Clinical peripheral neuropathy evaluation + WES locus (2q11.2 vs 2q33.3 — same chr2q!)",
            },
            {
                "comparator":     "TIMMDC1 (3q25.1)",
                "ndufaf7":        "Soluble matrix; HCM <10%; SAM-methyltransferase/Q-module",
                "comparator_val": "Integral IMM (2 TM helices); HCM >80%; ND1-module/Class3",
                "key_test":       "HCM rate + echocardiography + BN-PAGE class + WES locus",
            },
            {
                "comparator":     "FOXRED1 (11q24.2)",
                "ndufaf7":        "SAM-methyltransferase; Q-module/membrane-arm; 2q11.2",
                "comparator_val": "FAD-oxidoreductase chaperone; N-module; 11q24.2; FAD domain (no response)",
                "key_test":       "BN-PAGE assembly module + WES locus (2q11.2 vs 11q24.2)",
            },
            {
                "comparator":     "NDUFV1 (11q13.2)",
                "ndufaf7":        "NO leukodystrophy; SAM-methyltransferase assembly factor",
                "comparator_val": "Leukodystrophy 40-50%; FMN-binding structural N-module subunit",
                "key_test":       "MRI leukodystrophy + WES locus (2q11.2 vs 11q13.2)",
            },
        ],
        "ndufaf7_module_summary": {
            "gene":              "NDUFAF7 (C2orf56/MIDI1IP1, 2q11.2)",
            "module_class":      "Q-module/membrane-arm interface — class I SAM-dependent methyltransferase; NDUFS2 R-85 arginine methylation; NDUFB9 recruitment prerequisite",
            "sam_methyltransferase_mechanism": (
                "NDUFAF7 uses S-adenosylmethionine (SAM) as methyl donor. The class I methyltransferase "
                "Rossmann-like fold positions SAM adjacent to the NDUFS2 substrate in the active site. "
                "The methyl group from SAM is transferred to arginine-85 of the mature NDUFS2 polypeptide "
                "(corresponding to R-114 including the mitochondrial targeting sequence). "
                "The resulting N-monomethylarginine (or asymmetric dimethylarginine) at R-85 creates "
                "a recognition epitope for NDUFB9. p.Arg321Pro (the first reported mutation) disrupts "
                "the SAM-binding C-terminal helix — the most mechanistically direct pathogenic variant."
            ),
            "ndufb9_recruitment": (
                "NDUFB9 is a type II transmembrane accessory subunit of CI's membrane arm (ND4-containing "
                "module). Its binding to the Q-module assembly intermediate depends on NDUFAF7-mediated "
                "methylation of NDUFS2 R-85. Without this methylation checkpoint, NDUFB9 cannot bind, "
                "and the Q-module/membrane-arm assembly interface step is blocked. This explains why "
                "NDUFAF7-deficient cells accumulate a specific Q-module assembly intermediate that "
                "stalls before the membrane-arm integration step."
            ),
        },
    }


# ─── get_definitions ───────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "NDUFAF7 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 7; also known as C2orf56 "
            "and MIDI1IP1; OMIM *615898) encodes a 371-amino-acid, ~41 kDa soluble mitochondrial matrix "
            "protein belonging to the class I S-adenosylmethionine (SAM)-dependent methyltransferase "
            "superfamily (Rossmann-like beta/alpha fold). NDUFAF7 is the ONLY CI assembly factor with "
            "established SAM-dependent methyltransferase activity — it methylates arginine-85 of the "
            "mature NDUFS2 subunit (Q-module), required for NDUFB9 recruitment and Q-module/membrane-arm "
            "interface assembly. Loss-of-function variants cause isolated CI deficiency MC1DN30. "
            "No FAD domain, no TM helices, no [4Fe-4S] clusters, no 2OG-dioxygenase activity. "
            "No riboflavin response (0%). Chromosome 2q11.2."
        ),
        "disease_definition": (
            "Mitochondrial Complex I Deficiency MC1DN30 (OMIM #618248) due to NDUFAF7 bi-allelic "
            "pathogenic variants presents as infantile-onset Leigh-like syndrome with profound isolated "
            "CI deficiency (5-20% of control), preserved CII/CIII/CIV, basal ganglia lesions (~68%), "
            "Leigh MRI (~72%), lactic acidosis (~88%), hypotonia (~90%), and very low HCM (<10%). "
            "BN-PAGE shows Q-module/membrane-arm interface assembly intermediates stalled (NDUFS2 "
            "R-85 unmethylated, NDUFB9 unbound). No riboflavin response — critical DDx vs ACAD9. "
            "Zurita Rendón 2014 (J Biol Chem) first identified NDUFAF7/C2orf56 as a CI assembly factor "
            "and disease gene. Rhein 2016 (J Biol Chem) confirmed R-85 as the methylation site in mature NDUFS2."
        ),
        "inheritance_definition": (
            "Autosomal recessive (AR). Biallelic pathogenic NDUFAF7 variants required. "
            "Both sexes equally affected. Consanguinity observed in ~35% of reported families "
            "(South American and Middle Eastern enrichment). Carrier parents are clinically unaffected. "
            "Genetic counselling: 25% recurrence risk per pregnancy."
        ),
        "module_definitions": [
            {
                "term":       "NDUFAF7 as SAM-Dependent Methyltransferase — Unique CI Assembly Mechanism",
                "definition": (
                    "Class I S-adenosylmethionine (SAM)-dependent methyltransferases share a "
                    "Rossmann-like beta/alpha fold with a conserved SAM-binding site containing a "
                    "GxGxxG-like glycine-rich loop. NDUFAF7 uses this fold to coordinate SAM adjacent "
                    "to the NDUFS2 substrate. The transferred methyl group modifies arginine-85 of "
                    "mature NDUFS2 — generating a methylated arginine (N-monomethylarginine or "
                    "asymmetric N,N-dimethylarginine at R-85). This is the only SAM-dependent "
                    "methyltransferase reaction known to be required for CI assembly, and NDUFAF7 "
                    "is the only CI assembly factor that catalyzes it. p.Arg321Pro (Zurita Rendón 2014) "
                    "disrupts the C-terminal SAM-binding helix — the most direct pathogenic mechanism."
                ),
            },
            {
                "term":       "NDUFS2 Arginine-85 Methylation and NDUFB9 Recruitment",
                "definition": (
                    "NDUFS2 (49 kDa subunit, also named TYKY, OMIM *602985) is the core Q-module subunit "
                    "that houses the Q-binding site (quinone binding channel entrance). Arginine-85 of "
                    "mature NDUFS2 (R-85) is positioned at the interface between the Q-module and the "
                    "membrane arm. NDUFAF7-dependent methylation of R-85 creates a recruitment signal "
                    "for NDUFB9 (also named UQCRFS1 related; OMIM *603843), a type II membrane arm "
                    "accessory subunit required for membrane arm assembly progression. Without R-85 "
                    "methylation, NDUFB9 cannot bind, the Q-module/membrane-arm interface checkpoint "
                    "stalls, and holoenzyme CI formation is blocked. The Rhein 2016 (J Biol Chem) study "
                    "definitively mapped the methylation site to R-85 using mass spectrometry."
                ),
            },
            {
                "term":       "NDUFAF7 BN-PAGE — Q-Module/Membrane-Arm Interface Assembly Intermediates",
                "definition": (
                    "Blue-native PAGE (BN-PAGE) in NDUFAF7 patient fibroblasts/muscle shows accumulation "
                    "of Q-module assembly intermediates at the Q-module/membrane-arm interface stage — "
                    "distinct from other CI assembly defects: (1) N-module stalling (FOXRED1/NUBPL), "
                    "(2) MCIA/ND2-ND5/Class1 stalling (ACAD9/NDUFAF1/ECSIT/TMEM126B), (3) ND1-module/Class3 "
                    "stalling (NDUFAF3/4/5/TIMMDC1), and (4) NDUFAF6 Q-module NDUFS7 hydroxylation stalling. "
                    "The NDUFAF7 intermediate lacks NDUFB9 (unmethylated NDUFS2 R-85 prevents NDUFB9 binding). "
                    "Combined with no riboflavin response, no peripheral neuropathy, and 2q11.2 locus, this "
                    "narrows the differential to NDUFAF7 specifically."
                ),
            },
            {
                "term":       "NDUFAF7 vs NDUFAF6 — Two Enzymatic CI Assembly Factors",
                "definition": (
                    "NDUFAF7 and NDUFAF6 are uniquely the only two CI assembly factors with confirmed "
                    "enzymatic (catalytic) activity beyond chaperone/scaffold function: "
                    "NDUFAF7 = SAM-dependent methyltransferase (class I Rossmann fold); substrate = "
                    "NDUFS2 R-85; cofactor = SAM; product = N-methylarginine-85-NDUFS2. "
                    "NDUFAF6 = 2OG-Fe(II)-dependent dioxygenase/hydroxylase (jelly-roll beta-barrel); "
                    "substrate = NDUFS7/PSST leucyl/asparaginyl residue; cofactors = 2OG + Fe(II) + O2. "
                    "Different substrates, different chemical reactions (methylation vs hydroxylation), "
                    "different cofactors (SAM vs 2OG/Fe(II)/O2), different chromosomes (2q11.2 vs 8q22.1). "
                    "Both: soluble matrix, no FAD domain, 0% riboflavin response, isolated CI deficiency, "
                    "low HCM (<10%), no peripheral neuropathy, no leukodystrophy. WES is the discriminator."
                ),
            },
            {
                "term":       "NDUFAF7 Chromosome 2q11.2 — Same chr2q as NDUFS1 (2q33.3)",
                "definition": (
                    "NDUFAF7 maps to chromosome 2q11.2. NDUFS1 maps to 2q33.3. Both are on the long arm "
                    "(q-arm) of chromosome 2, approximately 62 Mb apart. This same-chromosome proximity "
                    "is a potential source of clinical confusion when ordering targeted gene panels "
                    "rather than WES/WGS. The critical clinical discriminator is peripheral neuropathy: "
                    "NDUFS1 causes peripheral neuropathy in ~50% of patients (hallmark feature); "
                    "NDUFAF7 causes 0% peripheral neuropathy. Nerve conduction studies / neurophysiology "
                    "should be performed in all CI patients before full WES results are available "
                    "to help prioritize 2q11.2 (NDUFAF7) vs 2q33.3 (NDUFS1)."
                ),
            },
        ],
        "reference_list": [
            {
                "citation": "Zurita Rendón O, Antonicka H, Funai EF, Shoubridge EA (2014) A mutation in the methyltransferase NDUFAF7 causes basal ganglia disease and complex I deficiency. J Biol Chem 289(7):3655–63.",
                "significance": "First identification of NDUFAF7/C2orf56 as a CI assembly factor and disease gene; reported p.Arg321Pro founder mutation in a South American consanguineous family; demonstrated NDUFS2 arginine methylation requirement",
            },
            {
                "citation": "Rhein VF, Carroll J, Ding S, Fearnley IM, Walker JE (2016) NDUFAF7 methylates arginine 85 in the NDUFS2 subunit of human complex I. J Biol Chem 291(6):2909–18.",
                "significance": "Definitive mapping of NDUFAF7 methylation site to NDUFS2 R-85 (mature form) using mass spectrometry; confirmed SAM-dependent arginine methyltransferase activity; demonstrated NDUFB9 binding requirement for R-85 methylation",
            },
            {
                "citation": "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "significance": "Comprehensive CI assembly pathway mapping using pulse-chase proteomics; positions NDUFAF7/NDUFS2 methylation step in the Q-module/membrane-arm interface assembly sequence",
            },
            {
                "citation": "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538(7623):123–126.",
                "significance": "CRISPR/Cas9 systematic CI subunit knockout; defines assembly classes; context for NDUFAF7/Q-module assembly stage",
            },
        ],
        "contraindication_definitions": [
            {
                "drug":       "Metformin",
                "level":      "ABSOLUTE CONTRAINDICATION",
                "mechanism":  "Biguanide direct CI inhibitor (complex I, site I / NADH dehydrogenase). NDUFAF7 patients have 5-20% residual CI; any further inhibition immediately causes lethal lactic acidosis.",
                "alternative": "Avoid in all mitochondrial CI deficiency. GLP-1 agonist or SGLT2 inhibitor if diabetes management needed in adult carriers.",
            },
            {
                "drug":       "Valproate (VPA)",
                "level":      "ABSOLUTE CONTRAINDICATION",
                "mechanism":  "Triple mechanism: (1) CoA sequestration (valproyl-CoA) → depletes free CoA needed for TCA cycle and beta-oxidation; (2) POLG1 inhibition → mtDNA depletion worsens CI subunit supply; (3) MT-ND gene expression suppression → reduces all 7 mtDNA-encoded CI subunits. Catastrophic in NDUFAF7 CI deficiency.",
                "alternative": "Levetiracetam (LEV) — first-choice AED in mitochondrial disease (renal excretion, no mitochondrial toxicity). Lacosamide as adjunct.",
            },
            {
                "drug":       "Ketogenic Diet (KD)",
                "level":      "CONTRAINDICATED",
                "mechanism":  "KD shifts metabolism to beta-oxidation and ketogenesis, generating large amounts of NADH. With CI absent/minimal, NADH cannot be reoxidised to NAD+ via ETC. NADH excess: inhibits TCA cycle, blocks pyruvate oxidation, causes lactic acidosis. CI is the NADH entry point — KD is metabolically incompatible with NDUFAF7 deficiency.",
                "alternative": "High-complex-carbohydrate diet with avoidance of prolonged fasting. Glucose infusion (GIR 6-8 mg/kg/min) during illness.",
            },
        ],
    }
