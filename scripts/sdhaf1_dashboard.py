#!/usr/bin/env python3
"""SDHAF1 (LYRM8) — Succinate Dehydrogenase Assembly Factor 1 / Complex II Deficiency / Infantile Leukoencephalopathy.

SDHAF1 (Succinate Dehydrogenase Assembly Factor 1; also LYRM8) encodes a 111-amino-acid,
~17 kDa LYR-motif (LYRM) protein that localises to the soluble mitochondrial matrix.
SDHAF1 is the only CII-specific FeS-insertion assembly factor: it delivers [2Fe-2S] and
[4Fe-4S] iron-sulfur clusters to SDHB (the iron-sulfur cluster subunit of Complex II/SDH),
working in concert with the HSC20/HSPA9 chaperone system.

  SDHAF1 gene   OMIM *612848
  Disease       Succinate Dehydrogenase Deficiency / Complex II Deficiency /
                Infantile Leukoencephalopathy (OMIM #252011)
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    19q13.12

Reference: Ghezzi D et al. (2009) SDHAF1, encoding a LYR complex-II specific assembly factor,
is mutated in SDH-defective infantile leukoencephalopathy. Nat Genet 41(6):654–656.
(First SDHAF1 disease gene identification; Italian founder p.Trp85Arg; 9 patients from 5 families)

Reference: Na U et al. (2014) The LYR factors SDHAF1 and SDHAF3 mediate maturation of the
iron-sulfur subunit of succinate dehydrogenase. Cell Metab 20(2):253–266.
(Mechanism: SDHAF1 delivers [2Fe-2S] and [4Fe-4S] FeS clusters to SDHB; works with HSC20/HSPA9 system)

Reference: Maio N, Rouault TA (2016) Mammalian iron-sulfur cluster biogenesis: fundamental
biological insights from yeast and animals. Biochim Biophys Acta 1863(6):1487–1501.
(Review: ISC pathway, LYR protein role in FeS cluster delivery to respiratory chain complexes)

PATHOPHYSIOLOGY (SDHAF1 / CII Assembly Factor / LYR-Motif FeS Delivery):
  SDHAF1/LYRM8 is the only CII-specific LYR-motif FeS insertion factor:
    1. SDHAF1 binds SDHB via its LYR motif and recruits the HSC20/HSPA9 co-chaperone
       system to deliver [2Fe-2S] and [4Fe-4S] clusters into SDHB's three FeS binding sites.
    2. Without SDHAF1, SDHB cannot receive FeS clusters and cannot assemble into the
       SDHA-SDHB catalytic core; CII holoenzyme formation is blocked.
    3. Isolated CII deficiency (SDH activity 10–30% of control); CI, CIII, CIV normal.
    4. CII deficiency → succinate accumulates → elevated succinate in blood, urine, CSF
       and on brain MRS (pathognomonic ~85% of patients).
    5. White-matter (leukoencephalopathy) predominance — NOT Leigh syndrome (gray matter).
    6. NO paraganglioma — critical DDx vs SDHB/SDHC/SDHD (dominant hereditary paraganglioma).
    7. NO HCM, NO peripheral neuropathy, NO olfactory bulb lesions.
    8. KD is contraindicated: beta-oxidation generates FADH2 which feeds into CII;
       deficient CII cannot oxidize FADH2 via ETC → metabolic crisis.

SDHAF1 UNIQUE FEATURES:
  1. ONLY CII-SPECIFIC FeS INSERTION FACTOR: No other assembly factor exclusively
     delivers FeS clusters to SDHB. SDHAF1 is the dedicated FeS-delivery scaffold
     for CII, analogous to NUBPL for CI N-module FeS but specific to SDHB.
  2. LYR-MOTIF (LYRM) PROTEIN: The LYR tripeptide at the N-terminus of the mature
     protein is the HSC20-binding motif. SDHAF1 belongs to the LYR-motif (LYRM)
     protein family — a class of small mitochondrial proteins that recruit the
     HSC20/HSPA9 chaperone system to deliver FeS clusters to target proteins.
  3. ISOLATED CII DEFICIENCY — BIOCHEMICAL FINGERPRINT: CII (SDH) = 10–30%.
     CI/CIII/CIV all normal. Brain MRS elevated succinate ~85% — pathognomonic.
  4. LEUKOENCEPHALOPATHY (WHITE MATTER) NOT LEIGH (GRAY MATTER): SDHAF1 causes
     infantile leukoencephalopathy — progressive white matter disease. SDHA causes
     Leigh syndrome (symmetric gray matter basal ganglia/brainstem). Critical DDx.
  5. NO PARAGANGLIOMA: SDHB/C/D = dominant hereditary paraganglioma. SDHAF1 =
     infantile CII deficiency leukoencephalopathy. Completely different diseases.
  6. 19q13.12 CHROMOSOME: Distinct from SDHA (1p36.1), SDHB (1p36.13), SDHC (1q23.3),
     SDHD (11q23.1), SDHAF2 (11q13.1). WES confirms 19q13.12.

DISTINGUISHING FEATURES vs OTHER CII/SUCCINATE-PATHWAY GENES:
  vs SDHA (1p36.1): SDHA = CII structural FAD-binding subunit; SDHA causes Leigh syndrome
    (gray matter) + paraganglioma. SDHAF1 = leukoencephalopathy (white matter), no paraganglioma.
  vs SDHB (1p36.13)/SDHC (1q23.3)/SDHD (11q23.1): All dominant hereditary paraganglioma.
    Completely different disease from SDHAF1 infantile CII deficiency leukoencephalopathy.
  vs SDHAF2 (PGL2, 11q13.1): Dominant hereditary paraganglioma type 2. Not leukoencephalopathy.
  vs LYRM7 (CIII assembly, 5q33.1): LYRM7 = Complex III deficiency; similar LYR motif but
    different complex. CIII deficiency vs CII deficiency: different biochemical fingerprint.
  vs Fumarate Hydratase (FUMH, 1q42.1): Fumaric aciduria also causes leukoencephalopathy;
    fumaric acid elevated on metabolomics, NOT succinate. Metabolite profile distinguishes.
  vs Canavan disease (ASPA, 17p13.2): White matter; N-acetylaspartate elevated; not succinate.
"""

import random
import math

SEED = 701
rng  = random.Random(SEED)

GENE         = "SDHAF1"
OMIM_GENE    = "612848"
OMIM_DISEASE = "252011"
DISEASE_NAME = "Succinate Dehydrogenase Deficiency / Complex II Deficiency / Infantile Leukoencephalopathy (OMIM #252011)"
CHROMOSOME   = "19q13.12"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ─────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.253T>C",
        "hgvs_p":    "p.Trp85Arg",
        "domain":    "LYR-motif core — tryptophan at FeS-delivery interface",
        "mechanism": (
            "Italian founder mutation. Tryptophan-to-arginine substitution at position 85 disrupts the "
            "core LYR-motif fold required for SDHB docking and FeS cluster delivery. The bulky, charged "
            "arginine side chain clashes with the hydrophobic LYR-motif core, abolishing SDHB contact and "
            "HSC20 co-chaperone recruitment. FeS insertion into SDHB is completely blocked."
        ),
        "severity":  "severe",
        "ci_pct_range": (10, 18),
        "notes": "Italian founder mutation (Ghezzi 2009). LYR-motif core disruption. Complete loss of SDHB FeS cluster delivery. Infantile leukoencephalopathy, elevated brain MRS succinate.",
    },
    {
        "hgvs_c":    "c.200G>C",
        "hgvs_p":    "p.Cys67Ser",
        "domain":    "LYR-motif — conserved cysteine; structural disulfide/fold",
        "mechanism": (
            "Cysteine-to-serine substitution removes the conserved cysteine in the LYR-motif region. "
            "Loss of the thiol group disrupts the structural integrity of the LYRM protein fold and "
            "impairs FeS cluster coordination. SDHAF1 protein stability is severely reduced; SDHB FeS "
            "cluster delivery is blocked."
        ),
        "severity":  "severe",
        "ci_pct_range": (8, 15),
        "notes": "LYR-motif cysteine loss. Structural disulfide disruption. Protein instability. Severe CII deficiency with infantile leukoencephalopathy.",
    },
    {
        "hgvs_c":    "c.163A>G",
        "hgvs_p":    "p.Arg55Gly",
        "domain":    "LYR-motif arginine — R of the LYR tripeptide; FeS-binding interface",
        "mechanism": (
            "Arginine-to-glycine substitution at position 55, which is the 'R' of the eponymous LYR "
            "tripeptide (L-Y-R). The arginine residue is required for HSC20 recognition and FeS cluster "
            "transfer competency. Glycine introduces hypermobility and disrupts the precise geometry of "
            "the FeS-binding interface. SDHB FeS insertion is severely impaired."
        ),
        "severity":  "severe",
        "ci_pct_range": (10, 16),
        "notes": "R of the LYR tripeptide. HSC20 recognition abolished. FeS cluster transfer geometry disrupted. Severe infantile CII deficiency.",
    },
    {
        "hgvs_c":    "c.134C>T",
        "hgvs_p":    "p.Ala45Val",
        "domain":    "SDHAF1 core — hydrophobic packing residue",
        "mechanism": (
            "Alanine-to-valine substitution introduces a minor steric clash in the hydrophobic core of "
            "SDHAF1. Partial protein misfolding with residual SDHAF1 function preserved. SDHB FeS cluster "
            "delivery is reduced but not abolished — intermediate CII deficiency results."
        ),
        "severity":  "intermediate",
        "ci_pct_range": (15, 25),
        "notes": "Core packing disruption. Partial residual SDHAF1 function. Intermediate CII deficiency. Later onset with progressive leukoencephalopathy.",
    },
    {
        "hgvs_c":    "c.IVS2+1G>A",
        "hgvs_p":    "p.splice (intron 2 donor)",
        "domain":    "Splice donor site — intron 2",
        "mechanism": (
            "Canonical splice donor disruption at intron 2 boundary. Exon 2 skipping or cryptic splice "
            "activation leads to partial loss of the central SDHAF1 coding sequence. Minor residual "
            "in-frame cryptic splice product preserves partial SDHB FeS delivery function. Moderate "
            "CII deficiency with some residual SDH activity."
        ),
        "severity":  "moderate",
        "ci_pct_range": (20, 30),
        "notes": "Splice donor intron 2 disruption. Partial CII residual from cryptic splice. Moderate phenotype; milder leukoencephalopathy progression.",
    },
    {
        "hgvs_c":    "c.117G>A",
        "hgvs_p":    "p.Trp39Ter",
        "domain":    "N-terminal region — near-start nonsense / premature truncation",
        "mechanism": (
            "Premature stop codon at position 39 truncates SDHAF1 near the N-terminus, removing the "
            "entire LYR-motif and FeS-delivery scaffold. NMD-sensitive transcript. Effectively a null "
            "allele — no SDHAF1 protein produced; complete loss of SDHB FeS cluster delivery."
        ),
        "severity":  "severe",
        "ci_pct_range": (8, 14),
        "notes": "Near-start nonsense. NMD-sensitive null allele. No SDHAF1 protein. Complete CII assembly block. Severe infantile leukoencephalopathy.",
    },
    {
        "hgvs_c":    "c.82C>T",
        "hgvs_p":    "p.Arg28Cys",
        "domain":    "N-terminal presequence cleavage region — surface charge residue",
        "mechanism": (
            "Arginine-to-cysteine substitution in the N-terminal presequence cleavage region. Disrupts "
            "surface charge distribution important for mitochondrial import processing and SDHAF1 "
            "stability after cleavage. Partial reduction in mature SDHAF1 levels; moderate CII deficiency."
        ),
        "severity":  "moderate",
        "ci_pct_range": (18, 28),
        "notes": "N-terminal presequence cleavage region. Surface charge disruption. Moderate CII deficiency. Slower leukoencephalopathy progression.",
    },
]

# ─── Patient cohort ────────────────────────────────────────────────────────────
ETHNICITIES = ["European (Italian founder)", "European (non-Italian)", "Middle-Eastern (consanguineous)", "South-Asian", "North-African (consanguineous)", "East-Asian", "Pan-Ethnic"]
ETHNICITY_WEIGHTS = [0.28, 0.22, 0.20, 0.14, 0.08, 0.05, 0.03]
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
    allele_weights = [0.28, 0.18, 0.16, 0.14, 0.10, 0.08, 0.06]
    a1 = _pick_weighted(VARIANTS, allele_weights)
    a2 = _pick_weighted(VARIANTS, allele_weights)
    return a1, a2

FEATURES = [
    "Leukoencephalopathy (white matter disease)",
    "Lactic acidosis (blood/CSF)",
    "Brain MRS elevated succinate (pathognomonic)",
    "Developmental delay / regression",
    "Hypotonia (axial/generalized)",
    "Spastic paraplegia",
    "Seizures (multiple types)",
    "Feeding difficulties / failure to thrive",
    "Respiratory compromise",
    "Hypertrophic cardiomyopathy (HCM)",
    "Peripheral neuropathy",
    "Basal ganglia lesions",
    "Hepatopathy",
    "Paraganglioma",
    "Optic atrophy",
    "Macrocephaly",
    "Cerebellar atrophy",
    "Exercise intolerance (older/milder)",
]

FEATURE_PROBS = {
    "Leukoencephalopathy (white matter disease)":    0.92,   # hallmark — white matter
    "Lactic acidosis (blood/CSF)":                   0.80,
    "Brain MRS elevated succinate (pathognomonic)":  0.85,   # ~85% — pathognomonic for CII deficiency
    "Developmental delay / regression":              0.90,
    "Hypotonia (axial/generalized)":                 0.82,
    "Spastic paraplegia":                            0.78,
    "Seizures (multiple types)":                     0.45,
    "Feeding difficulties / failure to thrive":      0.72,
    "Respiratory compromise":                        0.38,
    "Hypertrophic cardiomyopathy (HCM)":             0.02,   # Very low — no HCM in SDHAF1
    "Peripheral neuropathy":                         0.03,   # Very low — no peripheral neuropathy
    "Basal ganglia lesions":                         0.20,   # Minor — mainly white matter not basal ganglia
    "Hepatopathy":                                   0.03,   # Very low — critical DDx vs POLG/DGUOK
    "Paraganglioma":                                 0.00,   # ZERO — critical DDx vs SDHB/C/D
    "Optic atrophy":                                 0.15,
    "Macrocephaly":                                  0.25,
    "Cerebellar atrophy":                            0.30,
    "Exercise intolerance (older/milder)":           0.28,
}

OUTCOMES = ["alive_stable", "alive_disabled", "deceased"]
OUTCOME_WEIGHTS = [0.18, 0.54, 0.28]

FAMILIES = ["consanguineous", "non-consanguineous"]
FAMILY_WEIGHTS = [0.42, 0.58]

PATIENTS = []
for _i in range(N_PATIENTS):
    v1, v2 = _pick_variant_pair()
    severity_combined = "severe" if v1["severity"] == "severe" or v2["severity"] == "severe" else v1["severity"]
    if severity_combined == "severe":
        onset_months = rng.randint(1, 10)
    elif severity_combined == "moderate":
        onset_months = rng.randint(4, 20)
    else:
        onset_months = rng.randint(6, 30)

    # CII activity (10–30% range for SDHAF1)
    ci_lo = max(v1["ci_pct_range"][0], v2["ci_pct_range"][0]) - 3
    ci_hi = min(v1["ci_pct_range"][1], v2["ci_pct_range"][1]) + 3
    ci_lo = max(ci_lo, 8)
    ci_hi = min(ci_hi, 30)
    if ci_lo >= ci_hi:
        ci_hi = ci_lo + 4
    cii_activity = round(rng.uniform(ci_lo, ci_hi), 1)

    features = {f: (rng.random() < FEATURE_PROBS[f]) for f in FEATURES}

    age_at_assessment = onset_months + rng.randint(6, 48)

    PATIENTS.append({
        "patient_id":         f"P{_i+1:03d}",
        "sex":                rng.choice(SEXES),
        "onset_age_months":   onset_months,
        "age_months":         age_at_assessment,
        "ethnicity":          _pick_weighted(ETHNICITIES, ETHNICITY_WEIGHTS),
        "family":             _pick_weighted(FAMILIES, FAMILY_WEIGHTS),
        "allele1":            v1["hgvs_p"],
        "allele2":            v2["hgvs_p"],
        "severity":           severity_combined,
        "cii_activity_pct":   cii_activity,
        "outcome":            _pick_weighted(OUTCOMES, OUTCOME_WEIGHTS),
        "features":           features,
        "hcm":                features["Hypertrophic cardiomyopathy (HCM)"],
        "leukodystrophy":     features["Leukoencephalopathy (white matter disease)"],
        "brain_mrs_succinate": features["Brain MRS elevated succinate (pathognomonic)"],
        "lactic_acidosis":    features["Lactic acidosis (blood/CSF)"],
        "spastic_paraplegia": features["Spastic paraplegia"],
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
        "alias":        "LYRM8",
        "omim_gene":    OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_name": DISEASE_NAME,
        "chromosome":   CHROMOSOME,
        "inheritance":  INHERITANCE,
        "n_patients":   N_PATIENTS,

        "protein": {
            "size_aa":      111,
            "size_kda":     17,
            "localization": "soluble mitochondrial matrix",
            "motif":        "LYR-motif (LYRM)",
            "tm_helices":   0,
            "fad_domain":   False,
            "fes_cluster":  False,
            "domains": [
                "LYR-motif tripeptide (Leu-Tyr-Arg) — HSC20 co-chaperone docking site",
                "SDHB FeS-delivery interface",
                "N-terminal mitochondrial presequence (cleaved)",
            ],
            "topology": (
                "Soluble mitochondrial matrix protein — no TM helices, no FAD domain, "
                "no [4Fe-4S] clusters of its own. LYR-motif LYRM protein. "
                "SDHB FeS cluster delivery scaffold; works with HSC20/HSPA9 chaperone system."
            ),
        },

        "summary": (
            "SDHAF1 (LYRM8; OMIM *612848; 19q13.12) is a 111-aa, ~17 kDa LYR-motif (LYRM) protein "
            "that functions as the only CII-specific FeS insertion assembly factor. SDHAF1 delivers "
            "[2Fe-2S] and [4Fe-4S] iron-sulfur clusters to SDHB (the iron-sulfur cluster subunit of "
            "Complex II/SDH) by recruiting the HSC20/HSPA9 co-chaperone system via its LYR tripeptide. "
            "Without SDHAF1, SDHB cannot receive FeS clusters, CII holoenzyme cannot assemble, and "
            "succinate accumulates — causing infantile leukoencephalopathy with elevated brain MRS "
            "succinate (~85%, pathognomonic). Isolated CII deficiency (10–30%). No paraganglioma. "
            "No HCM. No peripheral neuropathy. KD absolutely contraindicated (FADH2 cannot enter ETC)."
        ),

        "biochemical_fingerprint": {
            "Complex_II_SDH":      "10–30% of control (SEVERE isolated CII deficiency — SDH activity)",
            "Complex_I":           "Normal (100%) — CI unaffected",
            "Complex_III":         "Normal (100%) — CIII unaffected",
            "Complex_IV":          "Normal (100%) — CIV unaffected",
            "Complex_V":           "Normal (100%) — CV unaffected",
            "Pattern":             "ISOLATED CII deficiency — SDHAF1 absent; SDHB cannot receive FeS clusters; SDHA-SDHB catalytic core cannot assemble; succinate accumulates",
            "Succinate_MRS":       "Brain MRS elevated succinate ~85% — PATHOGNOMONIC for CII deficiency; succinate peak visible on 1H-MRS at 2.4 ppm",
            "Succinate_blood_urine": "Elevated succinate in blood, urine, CSF",
            "Riboflavin_response": "NOT INDICATED — SDHAF1 is a LYR-motif scaffold with NO FAD domain; no riboflavin response expected",
            "KD_risk":             "ABSOLUTE CONTRAINDICATION — beta-oxidation FADH2 cannot enter ETC via deficient CII; metabolic crisis",
            "Paraganglioma":       "ZERO — no paraganglioma; critical DDx vs SDHB/SDHC/SDHD (dominant paraganglioma)",
            "HCM_rate":            "VERY LOW <5% — no HCM in SDHAF1",
        },

        "feature_frequencies_pct": ff,

        "sdhaf1_module_summary": {
            "gene":              "SDHAF1 (LYRM8, 19q13.12)",
            "module_class":      "CII Assembly Factor — LYR-motif SDHB FeS cluster delivery scaffold; works with HSC20/HSPA9 chaperone system",
            "assembly_position": "Only CII-specific FeS insertion factor; delivers [2Fe-2S] and [4Fe-4S] to SDHB; required for SDHA-SDHB catalytic core assembly",
            "lyr_motif_mechanism": (
                "SDHAF1 contains a LYR (Leu-Tyr-Arg) tripeptide motif at its N-terminus — the defining "
                "feature of LYR-motif (LYRM) proteins. The LYR motif binds directly to the J-domain of "
                "HSC20, recruiting the HSC20/HSPA9 cochaperone machinery. This co-chaperone complex then "
                "mediates the transfer of [2Fe-2S] and [4Fe-4S] iron-sulfur clusters from the ISC "
                "(iron-sulfur cluster assembly) machinery to the three FeS binding sites of SDHB. "
                "Without SDHAF1, HSC20 cannot be directed to SDHB, FeS insertion fails, and CII assembly stalls."
            ),
            "sdhaf1_vs_sdhaf2_sdhaf3": (
                "SDHAF1 (19q13.12) is the only CII-specific FeS insertion factor and is mutated in "
                "infantile CII deficiency leukoencephalopathy. "
                "SDHAF2 (11q13.1) is an SDH FAD-insertion factor for SDHA — and is the PGL2 gene "
                "(dominant hereditary paraganglioma type 2); SDHAF2 mutations cause paraganglioma, "
                "NOT leukoencephalopathy. "
                "SDHAF3 (1q21.2) is a late-stage CII assembly factor that protects SDHB from oxidative "
                "damage after FeS insertion; SDHAF3 works downstream of SDHAF1 in CII assembly. "
                "Key clinical DDx: SDHAF1 = infantile leukoencephalopathy; SDHAF2 = paraganglioma."
            ),
        },

        "ddx_table": [
            {
                "feature":      "SDHAF1 (19q13.12) vs SDHB/SDHC/SDHD — NO Paraganglioma — Critical DDx",
                "significance": (
                    "SDHB (1p36.13), SDHC (1q23.3), and SDHD (11q23.1) mutations cause dominant hereditary "
                    "paraganglioma/pheochromocytoma. SDHAF1 causes infantile CII deficiency leukoencephalopathy — "
                    "completely different disease. No paraganglioma in SDHAF1. Any paraganglioma/pheo in family "
                    "history immediately points away from SDHAF1. WES chromosome discrimination: 19q13.12 (SDHAF1) "
                    "vs 1p36 (SDHB), 1q23 (SDHC), 11q23 (SDHD)."
                ),
                "target_gene": "SDHB / SDHC / SDHD",
            },
            {
                "feature":      "SDHAF1 vs SDHA (1p36.1) — Leukoencephalopathy vs Leigh Syndrome — Critical DDx",
                "significance": (
                    "SDHA (FAD-binding structural CII subunit) causes Leigh syndrome (symmetric gray matter "
                    "basal ganglia/brainstem lesions) and paraganglioma. "
                    "SDHAF1 causes leukoencephalopathy (progressive white matter disease) — NOT Leigh syndrome. "
                    "MRI pattern discriminates: white matter (SDHAF1) vs gray matter/basal ganglia (SDHA). "
                    "Riboflavin response: SDHA has FAD domain — some riboflavin response possible. "
                    "SDHAF1 has NO FAD domain — no riboflavin response. WES mandatory: 19q13.12 vs 1p36.1."
                ),
                "target_gene": "SDHA",
            },
            {
                "feature":      "SDHAF1 vs SDHAF2 (11q13.1, PGL2) — Infantile CII Deficiency vs Paraganglioma",
                "significance": (
                    "SDHAF2 (PGL2, 11q13.1) = dominant hereditary paraganglioma type 2. SDHAF2 is the FAD "
                    "insertion factor for SDHA. SDHAF2 mutations cause paraganglioma, not infantile "
                    "leukoencephalopathy. SDHAF1 = infantile CII deficiency + leukoencephalopathy, no paraganglioma. "
                    "Both affect CII assembly but at different steps (FAD insertion vs FeS insertion) and "
                    "cause completely different diseases. WES: 19q13.12 (SDHAF1) vs 11q13.1 (SDHAF2)."
                ),
                "target_gene": "SDHAF2",
            },
            {
                "feature":      "SDHAF1 vs LYRM7 (5q33.1) — CII Assembly vs CIII Assembly — Same LYR Motif Family",
                "significance": (
                    "LYRM7 is a LYR-motif protein that delivers FeS clusters to UQCRFS1 (RISP) for "
                    "Complex III assembly — causing CIII deficiency. SDHAF1 delivers FeS to SDHB for "
                    "Complex II. Same LYRM protein family, different target complexes, different biochemistry. "
                    "Biochemical fingerprint distinguishes: isolated CII deficiency (SDHAF1, SDH 10-30%) "
                    "vs isolated CIII deficiency (LYRM7). WES: 19q13.12 vs 5q33.1."
                ),
                "target_gene": "LYRM7",
            },
            {
                "feature":      "SDHAF1 vs Fumarate Hydratase/FUMH (1q42.1) — Succinate vs Fumarate Elevation",
                "significance": (
                    "Fumarate hydratase (FUMH) deficiency causes fumaric aciduria — also presents with "
                    "leukoencephalopathy. KEY METABOLITE DISTINCTION: FUMH deficiency elevates fumaric acid "
                    "in urine/blood/CSF; SDHAF1 deficiency elevates succinate. Brain MRS: SDHAF1 shows "
                    "succinate peak at 2.4 ppm; FUMH shows no succinate peak. Urine organic acids "
                    "and targeted metabolomics are the clinical discriminator."
                ),
                "target_gene": "FUMH",
            },
            {
                "feature":      "SDHAF1 vs Canavan (ASPA, 17p13.2) / MLD (ARSA, 22q13.33) — White Matter DDx",
                "significance": (
                    "Canavan disease (ASPA deficiency) presents with white matter disease but elevates "
                    "N-acetylaspartate (NAA) on MRS — not succinate. MLD (ARSA deficiency) also causes "
                    "white matter disease; sulfatide storage; arylsulfatase A enzyme assay + urine sulfatides "
                    "distinguish. SDHAF1: succinate elevated on MRS (~85%, pathognomonic) + isolated CII "
                    "deficiency on enzyme assay. Succinate MRS peak is the definitive discriminator."
                ),
                "target_gene": "ASPA / ARSA",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct Complex I inhibitor. SDHAF1 patients have isolated CII deficiency; additional ETC stress from CI inhibition worsens metabolic state and risks lactic acidosis crisis.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit suppression. Further compromises already stressed ETC in CII-deficient patients.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA mitoribosomal inhibition. Blocks all mtDNA-encoded subunits. Avoid in all mitochondrial ETC deficiency including CII.",
            "🔴 KETOGENIC DIET (KD) — ABSOLUTELY CONTRAINDICATED: Beta-oxidation generates FADH2 which enters ETC exclusively via Complex II (CII). SDHAF1 patients have CII deficiency — FADH2 cannot be oxidized via ETC. KD triggers severe metabolic crisis in CII deficiency.",
            "🔴 SUCCINATE SUPPLEMENTATION — NOT RECOMMENDED: CII/SDH itself is deficient — exogenous succinate cannot be metabolized via the deficient Complex II. Succinate will further accumulate and worsen toxicity.",
            "🟠 RIBOFLAVIN — NOT INDICATED: SDHAF1 is a LYR-motif FeS delivery scaffold with NO FAD domain. Unlike SDHA (FAD-containing CII subunit), SDHAF1 has no FAD binding site. No riboflavin response expected. Do not treat as SDHA deficiency.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary ETC inhibitor effects. Prefer LEV (levetiracetam) as first-choice AED in mitochondrial disease.",
            "🟢 CoQ10 (Ubiquinol) — Level C: Antioxidant and electron carrier support. FADH2 still needs to enter the quinone pool — CoQ10 support may provide partial benefit.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter deficiency) and BTD BEFORE confirming SDHAF1 as diagnosis.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS. Both cause lactic acidosis + leukoencephalopathy — biotin-responsive mimics.",
            "🟢 CARNITINE — Level C: Supplement if secondary carnitine deficiency documented (avoid long-chain fatty acid accumulation).",
            "🟢 LEV (Levetiracetam) — PREFERRED AED: Renal excretion, no mitochondrial toxicity. First-choice anticonvulsant in SDHAF1 and all mitochondrial disease.",
            "🟢 IV DEXTROSE — GIR 6-8 mg/kg/min during illness: Never fast. Glucose infusion prevents catabolism and FADH2 surge from fat mobilization.",
            "🟢 SEVOFLURANE (not propofol) for anaesthesia: Propofol infusion syndrome (PRIS) risk via CIV inhibition; sevoflurane is the preferred agent.",
            "🔵 BRAIN MRS SUCCINATE PEAK (~85%) — PATHOGNOMONIC: Elevated succinate on 1H-MRS at 2.4 ppm is highly pathognomonic for CII deficiency. Distinguish from Canavan (NAA peak) and fumaric aciduria (fumarate).",
            "🔵 NO PARAGANGLIOMA — Zero risk in SDHAF1: Any paraganglioma/pheochromocytoma immediately points to SDHB/SDHC/SDHD (dominant), not SDHAF1. Critical DDx distinction.",
            "🔵 LEUKOENCEPHALOPATHY not Leigh syndrome: SDHAF1 = white matter disease. SDHA = gray matter (Leigh). MRI pattern is the first discriminator before biochemistry.",
            "🔵 WES 19q13.12 LOCUS: Confirms SDHAF1. Distinct from SDHA (1p36.1), SDHB (1p36.13), SDHC (1q23.3), SDHD (11q23.1), SDHAF2 (11q13.1), LYRM7 (5q33.1). 19q13.12 is unique.",
        ],
    }


# ─── get_breakdown ─────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    variant_counts: dict[str, int] = {}
    for p in PATIENTS:
        for a in [p["allele1"], p["allele2"]]:
            variant_counts[a] = variant_counts.get(a, 0) + 1

    cii_vals   = [p["cii_activity_pct"] for p in PATIENTS]
    onset_vals = [p["onset_age_months"] for p in PATIENTS]

    cii_bands = {"<15%": 0, "15–20%": 0, "20–25%": 0, ">25%": 0}
    for c in cii_vals:
        if   c < 15:  cii_bands["<15%"]   += 1
        elif c < 20:  cii_bands["15–20%"] += 1
        elif c < 25:  cii_bands["20–25%"] += 1
        else:          cii_bands[">25%"]   += 1

    onset_bands = {"0–4 mo": 0, "5–10 mo": 0, "11–20 mo": 0, ">20 mo": 0}
    for o in onset_vals:
        if   o <= 4:   onset_bands["0–4 mo"]   += 1
        elif o <= 10:  onset_bands["5–10 mo"]  += 1
        elif o <= 20:  onset_bands["11–20 mo"] += 1
        else:           onset_bands[">20 mo"]  += 1

    outcomes = {"alive_stable": 0, "alive_disabled": 0, "deceased": 0}
    for p in PATIENTS:
        outcomes[p["outcome"]] += 1

    severe_alleles = [v["hgvs_p"] for v in VARIANTS if v["severity"] == "severe"]
    pct_severe = sum(
        1 for p in PATIENTS
        if p["allele1"] in severe_alleles or p["allele2"] in severe_alleles
    )

    consang = sum(1 for p in PATIENTS if p["family"] == "consanguineous")
    mrs_pos  = sum(1 for p in PATIENTS if p["brain_mrs_succinate"])
    leuko    = sum(1 for p in PATIENTS if p["leukodystrophy"])
    hcm_n    = sum(1 for p in PATIENTS if p["hcm"])

    return {
        "cohort_n":  N_PATIENTS,
        "patients":  PATIENTS,

        "cii_activity_stats": {
            "mean":   round(sum(cii_vals) / N_PATIENTS, 1),
            "min":    round(min(cii_vals), 1),
            "max":    round(max(cii_vals), 1),
            "bands":  cii_bands,
            "label":  "CII/SDH activity (% of control)",
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
        "outcome_distribution":   outcomes,
        "consanguineous_pct":     round(consang / N_PATIENTS * 100),
        "pct_with_severe_allele": round(pct_severe / N_PATIENTS * 100),
        "brain_mrs_succinate_pct": round(mrs_pos / N_PATIENTS * 100),
        "leukoencephalopathy_pct": round(leuko / N_PATIENTS * 100),
        "hcm_pct":                round(hcm_n / N_PATIENTS * 100),

        "key_lyr_motif_features": {
            "LYR_motif_LYRM_protein_SDHB_FeS_delivery":         True,
            "Only_CII_specific_FeS_insertion_assembly_factor":   True,
            "Works_with_HSC20_HSPA9_cochaperone_system":         True,
            "No_FAD_domain_no_riboflavin_response":              True,
            "No_TM_helices_soluble_matrix":                      True,
            "Brain_MRS_succinate_elevated_pct":                  round(mrs_pos / N_PATIENTS * 100),
            "Leukoencephalopathy_white_matter_pct":              round(leuko / N_PATIENTS * 100),
            "HCM_rate_very_low_pct":                             round(hcm_n / N_PATIENTS * 100),
            "Paraganglioma_pct":                                 0,
            "Isolated_CII_deficiency_CI_CIII_CIV_normal":        True,
            "KD_contraindicated_FADH2_CII_deficient":            True,
        },

        "treatment_summary": {
            "absolute_ci": ["Metformin (CI inhibitor)", "Valproate/VPA (triple mechanism)", "Linezolid (mitoribosomal)", "Ketogenic Diet (FADH2 cannot enter ETC via deficient CII)"],
            "not_recommended": ["Succinate supplementation (CII itself is deficient)", "Riboflavin (no FAD domain in SDHAF1)"],
            "avoid":       ["Propofol (PRIS risk via CIV)", "Phenobarbital (high caution, secondary ETC effects)"],
            "level_c":     ["CoQ10-Ubiquinol", "Thiamine B1 (MANDATORY empiric)", "Biotin (MANDATORY empiric)", "Carnitine (if secondary deficiency)"],
            "preferred_aed": "LEV (levetiracetam) — renal excretion, no mitochondrial toxicity",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk)",
            "diet":        "High-complex-carbohydrate; never fast; IV dextrose GIR 6-8 mg/kg/min during illness",
            "diagnostic_priority": "WES 19q13.12 locus; brain MRS (succinate peak 2.4 ppm); SDH enzyme assay (isolated CII 10-30%); urine succinate; exclude SDHA/SDHB/SDHAF2 by clinical phenotype (no paraganglioma, no Leigh MRI)",
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
                "comparator":     "SDHB/SDHC/SDHD (paraganglioma genes)",
                "sdhaf1":         "NO paraganglioma; isolated CII deficiency; infantile leukoencephalopathy; AR; 19q13.12",
                "comparator_val": "Dominant hereditary paraganglioma/pheochromocytoma; adult-onset; NO infantile CII deficiency leukoencephalopathy",
                "key_test":       "Family history (paraganglioma vs not) + WES locus (19q13.12 vs 1p36/1q23/11q23)",
            },
            {
                "comparator":     "SDHA (1p36.1) — Leigh syndrome",
                "sdhaf1":         "Leukoencephalopathy (white matter); no paraganglioma; no FAD domain; no riboflavin response",
                "comparator_val": "Leigh syndrome (gray matter); paraganglioma possible; FAD-binding; riboflavin response possible",
                "key_test":       "MRI pattern (white vs gray matter) + brain MRS + WES (19q13.12 vs 1p36.1)",
            },
            {
                "comparator":     "SDHAF2 (11q13.1, PGL2) — Paraganglioma",
                "sdhaf1":         "Infantile CII deficiency; leukoencephalopathy; FeS insertion factor; 19q13.12",
                "comparator_val": "FAD insertion factor for SDHA; dominant paraganglioma type 2; no infantile leukoencephalopathy; 11q13.1",
                "key_test":       "Clinical phenotype (infantile neuro vs paraganglioma) + WES (19q13.12 vs 11q13.1)",
            },
            {
                "comparator":     "LYRM7 (5q33.1) — CIII assembly",
                "sdhaf1":         "CII deficiency (SDH 10-30%); LYR-motif SDHB FeS delivery; 19q13.12",
                "comparator_val": "CIII deficiency (UQCRFS1 FeS); LYR-motif RISP delivery; 5q33.1",
                "key_test":       "Biochemistry (CII vs CIII deficiency) + WES (19q13.12 vs 5q33.1)",
            },
            {
                "comparator":     "Fumarate Hydratase / FUMH (1q42.1)",
                "sdhaf1":         "Succinate elevated on MRS and metabolomics; CII deficiency",
                "comparator_val": "Fumarate elevated in urine/blood/CSF; fumaric aciduria; no CII deficiency",
                "key_test":       "Urine organic acids (succinate vs fumarate) + brain MRS metabolite peak",
            },
            {
                "comparator":     "Canavan (ASPA, 17p13.2) / MLD (ARSA, 22q13.33)",
                "sdhaf1":         "Succinate peak on MRS; isolated CII deficiency on enzyme assay; 19q13.12",
                "comparator_val": "NAA peak (Canavan) or sulfatides (MLD); normal SDH activity; different enzyme assays",
                "key_test":       "Brain MRS metabolite profile (succinate vs NAA) + SDH enzyme assay + WES",
            },
        ],

        "sdhaf1_module_summary": {
            "gene":         "SDHAF1 (LYRM8, 19q13.12)",
            "module_class": "CII Assembly Factor — LYR-motif SDHB FeS cluster delivery; HSC20/HSPA9 co-chaperone dependent",
            "cii_assembly_pathway": (
                "Complex II (SDH) assembly requires four subunits: SDHA (FAD-binding), SDHB (FeS cluster), "
                "SDHC and SDHD (membrane anchors). Assembly proceeds in stages: "
                "(1) SDHA maturation: SDHAF2 inserts FAD into SDHA; SDHAF4 stabilises SDHA; "
                "(2) SDHB maturation: SDHAF1 delivers [2Fe-2S] and [4Fe-4S] FeS clusters to SDHB; "
                "SDHAF3 protects newly assembled SDHB from oxidative damage; "
                "(3) SDHA-SDHB catalytic core formation: requires complete SDHB FeS maturation; "
                "(4) Integration into SDHC-SDHD membrane anchor at the inner mitochondrial membrane. "
                "SDHAF1 is essential at step 2 — without SDHAF1, SDHB is apo-SDHB (no FeS clusters), "
                "the catalytic core cannot form, and succinate oxidation is blocked."
            ),
        },
    }


# ─── get_definitions ───────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "SDHAF1 (Succinate Dehydrogenase Assembly Factor 1; also LYRM8; OMIM *612848; 19q13.12) encodes "
            "a 111-amino-acid, ~17 kDa LYR-motif (LYRM) protein that localises to the soluble mitochondrial "
            "matrix. SDHAF1 functions as the only CII-specific iron-sulfur (FeS) cluster insertion assembly "
            "factor: it delivers [2Fe-2S] and [4Fe-4S] clusters to the three FeS binding sites of SDHB "
            "(the iron-sulfur cluster subunit of Complex II / succinate dehydrogenase) by recruiting the "
            "HSC20/HSPA9 co-chaperone system via its LYR tripeptide motif. "
            "SDHAF1 has no FAD domain, no TM helices, no FeS clusters of its own, and no other known "
            "enzymatic activity. Its function is entirely dependent on the LYR–HSC20 interaction. "
            "Chromosome 19q13.12. Inheritance AR (biallelic)."
        ),
        "disease_definition": (
            "Succinate Dehydrogenase Deficiency / Complex II Deficiency / Infantile Leukoencephalopathy "
            "(OMIM #252011) due to biallelic SDHAF1 pathogenic variants presents as infantile-onset "
            "progressive leukoencephalopathy (white matter disease) with isolated CII deficiency "
            "(SDH activity 10–30% of control; CI/CIII/CIV normal). "
            "Hallmark findings: elevated brain succinate on MRS (~85%, pathognomonic); progressive spastic "
            "paraplegia; developmental regression; lactic acidosis. "
            "Absence of key features: NO paraganglioma (DDx vs SDHB/SDHC/SDHD); NO Leigh syndrome "
            "(DDx vs SDHA); NO HCM; NO peripheral neuropathy. "
            "MRI: progressive white matter signal abnormality (leukoencephalopathy), not basal ganglia "
            "or brainstem lesions characteristic of Leigh syndrome. "
            "Urine organic acids: elevated succinate. Brain MRS: succinate peak at 2.4 ppm. "
            "Italian founder mutation p.Trp85Arg (c.253T>C) is most common."
        ),
        "inheritance_definition": (
            "Autosomal recessive (AR). Biallelic pathogenic SDHAF1 variants required for disease. "
            "Both sexes equally affected. Consanguinity observed in approximately 42% of reported families. "
            "Italian founder mutation p.Trp85Arg enriched in Italian/Mediterranean populations. "
            "Carrier parents are clinically unaffected. "
            "Genetic counselling: 25% recurrence risk per pregnancy. "
            "NOTE: SDHB, SDHC, SDHD, and SDHAF2 mutations are DOMINANT and cause hereditary paraganglioma — "
            "a completely different mode of inheritance and disease from SDHAF1 (AR, infantile leukoencephalopathy)."
        ),
        "module_definitions": [
            {
                "term":       "LYR-Motif (LYRM) Protein — Definition and HSC20 Recruitment Mechanism",
                "definition": (
                    "LYR-motif proteins (LYRM proteins) are a family of small mitochondrial proteins "
                    "defined by a conserved tripeptide sequence Leu-Tyr-Arg (LYR) near their N-terminus. "
                    "This LYR tripeptide binds directly to the J-domain of HSC20 (HSPA9 co-chaperone), "
                    "recruiting the HSC20/HSPA9/ISCU complex to deliver iron-sulfur (FeS) clusters to "
                    "target apo-proteins. In the case of SDHAF1, the LYR motif directs HSC20 specifically "
                    "to SDHB, enabling [2Fe-2S] and [4Fe-4S] cluster insertion into SDHB's three FeS "
                    "binding sites (S1 [2Fe-2S], S2 [4Fe-4S], S3 [4Fe-4S]). Other LYRM proteins target "
                    "different respiratory chain subunits: LYRM7 targets UQCRFS1 (RISP) for CIII; "
                    "ACN9 targets complex II. SDHAF1 is the only LYRM protein specifically required "
                    "for SDHB FeS maturation."
                ),
            },
            {
                "term":       "SDHB FeS Cluster Delivery — SDHAF1 Mechanism and CII Assembly Pathway",
                "definition": (
                    "SDHB (the iron-sulfur protein subunit of Complex II) contains three FeS clusters: "
                    "one [2Fe-2S] and two [4Fe-4S] clusters, which are essential for electron transfer "
                    "from succinate to ubiquinone. SDHAF1 mediates FeS insertion as follows: "
                    "(1) SDHAF1 binds apo-SDHB via its LYR motif and SDHB-interaction surface; "
                    "(2) SDHAF1 LYR recruits HSC20 (also called HSCB); "
                    "(3) HSC20 interacts with HSPA9 (mortalin/GRP75) and ISCU (scaffold protein) "
                    "to transfer [2Fe-2S] and [4Fe-4S] clusters from ISC machinery to SDHB binding sites; "
                    "(4) SDHAF3 then protects newly FeS-loaded SDHB from oxidative damage; "
                    "(5) FeS-loaded SDHB assembles with SDHA (FAD-loaded, via SDHAF2/SDHAF4) "
                    "to form the SDHA-SDHB catalytic core; "
                    "(6) SDHA-SDHB integrates into SDHC-SDHD membrane anchor. "
                    "Loss of SDHAF1 at step 1 blocks all downstream steps — total CII assembly failure."
                ),
            },
            {
                "term":       "CII Assembly Pathway — SDHAF1, SDHAF2, SDHAF3, SDHAF4 Roles",
                "definition": (
                    "Four SDHAF assembly factors coordinate Complex II biogenesis, each at a distinct stage: "
                    "SDHAF2 (PGL2, 11q13.1): FAD insertion into SDHA; loss → SDHA instability + paraganglioma (dominant). "
                    "SDHAF4 (SDHAF4 gene; soluble matrix): SDHA stabilisation after FAD insertion; loss → SDHA degradation. "
                    "SDHAF1 (LYRM8, 19q13.12): [2Fe-2S] and [4Fe-4S] FeS cluster delivery to SDHB via LYR–HSC20; "
                    "loss → SDHB apo-protein; catalytic core cannot form; CII deficiency; "
                    "infantile leukoencephalopathy (AR). "
                    "SDHAF3 (1q21.2): Protects FeS-loaded SDHB from oxidative damage; loss → SDHB FeS "
                    "degradation; CII deficiency. "
                    "Critical clinical distinction: SDHAF2 causes paraganglioma (dominant); SDHAF1 causes "
                    "infantile leukoencephalopathy (recessive). These are completely different diseases."
                ),
            },
            {
                "term":       "SDHAF1 vs SDHAF2/SDHAF3 — Clinical and Molecular Distinction",
                "definition": (
                    "SDHAF1 (19q13.12, AR): SDHB FeS insertion assembly factor; infantile CII deficiency "
                    "leukoencephalopathy; succinate elevated; no paraganglioma; both alleles must be "
                    "mutated for disease. "
                    "SDHAF2 (11q13.1, PGL2, AD): SDHA FAD insertion factor; dominant hereditary paraganglioma "
                    "type 2 (PGL2); single heterozygous mutation causes paraganglioma; no infantile "
                    "leukoencephalopathy; no CII deficiency in leukocytes (haploinsufficiency). "
                    "SDHAF3 (1q21.2, AR): SDHB protector after FeS insertion; CII deficiency phenotype "
                    "similar to SDHAF1 but distinct molecular step; fewer reported patients. "
                    "Clinical DDx key: Does the patient have paraganglioma/pheo? → SDHB/C/D or SDHAF2. "
                    "Does the patient have infantile leukoencephalopathy + CII deficiency? → SDHAF1 or SDHAF3. "
                    "WES and chromosome locus (19q13.12 vs 11q13.1 vs 1q21.2) are definitive discriminators."
                ),
            },
        ],
        "reference_list": [
            {
                "citation": "Ghezzi D et al. (2009) SDHAF1, encoding a LYR complex-II specific assembly factor, is mutated in SDH-defective infantile leukoencephalopathy. Nat Genet 41(6):654–656.",
                "significance": (
                    "First disease gene identification for SDHAF1. Nine patients from 5 families; "
                    "identified the Italian founder mutation p.Trp85Arg (c.253T>C). Established SDHAF1 "
                    "as the causative gene for isolated CII deficiency infantile leukoencephalopathy. "
                    "Demonstrated elevated brain MRS succinate as pathognomonic marker. Characterized "
                    "SDHAF1 as a LYR-motif protein required for SDHB maturation."
                ),
            },
            {
                "citation": "Na U et al. (2014) The LYR factors SDHAF1 and SDHAF3 mediate maturation of the iron-sulfur subunit of succinate dehydrogenase. Cell Metab 20(2):253–266.",
                "significance": (
                    "Definitive mechanistic study: established that SDHAF1 delivers [2Fe-2S] and [4Fe-4S] "
                    "iron-sulfur clusters to SDHB via the LYR–HSC20/HSPA9 co-chaperone system. "
                    "Showed SDHAF3 acts downstream of SDHAF1, protecting FeS-loaded SDHB. "
                    "Provided biochemical framework distinguishing SDHAF1 (FeS insertion) from SDHAF2 "
                    "(FAD insertion). Confirmed LYR tripeptide as the HSC20-binding determinant."
                ),
            },
            {
                "citation": "Maio N, Rouault TA (2016) Mammalian iron-sulfur cluster biogenesis: fundamental biological insights from yeast and animals. Biochim Biophys Acta 1863(6):1487–1501.",
                "significance": (
                    "Comprehensive review of the ISC pathway and LYR protein family role in FeS cluster "
                    "delivery to respiratory chain complexes. Positions SDHAF1 within the broader context "
                    "of mitochondrial FeS biogenesis: ISC assembly machinery → HSC20/HSPA9/ISCU → "
                    "LYRM proteins (SDHAF1, LYRM7, etc.) → target apo-proteins (SDHB, UQCRFS1). "
                    "Explains why LYR-motif loss (as in SDHAF1 p.Arg55Gly mutating the 'R' of LYR) "
                    "is so catastrophic for FeS delivery."
                ),
            },
        ],
        "contraindication_definitions": [
            {
                "drug":       "Ketogenic Diet (KD)",
                "level":      "ABSOLUTELY CONTRAINDICATED",
                "mechanism":  (
                    "Beta-oxidation of fatty acids generates FADH2, which donates electrons to ubiquinone "
                    "EXCLUSIVELY via Complex II (CII/SDH). In SDHAF1 deficiency, CII is severely deficient "
                    "(10–30% activity). FADH2 generated by KD-driven fat oxidation cannot enter the ETC "
                    "via deficient CII, causing FADH2 accumulation, ETC blockade, and acute metabolic crisis. "
                    "KD is more dangerous in CII deficiency than in CI deficiency."
                ),
                "alternative": "High-complex-carbohydrate diet. IV dextrose GIR 6-8 mg/kg/min during illness. Never fast.",
            },
            {
                "drug":       "Metformin",
                "level":      "ABSOLUTE CONTRAINDICATION",
                "mechanism":  (
                    "Biguanide direct Complex I inhibitor. While SDHAF1 primarily causes CII deficiency, "
                    "additional CI inhibition by metformin further stresses an already impaired ETC, "
                    "risking lactic acidosis crisis. All respiratory chain inhibitors are contraindicated "
                    "in any ETC deficiency."
                ),
                "alternative": "Avoid in all mitochondrial ETC deficiency. For diabetes in adult carriers, consider GLP-1 agonist or SGLT2 inhibitor after specialist review.",
            },
            {
                "drug":       "Valproate (VPA)",
                "level":      "ABSOLUTE CONTRAINDICATION",
                "mechanism":  (
                    "Triple mechanism: (1) CoA sequestration as valproyl-CoA depletes free CoA for TCA cycle; "
                    "(2) POLG1 inhibition causing mtDNA depletion and reduced synthesis of all mtDNA-encoded ETC subunits; "
                    "(3) Suppression of MT-ND gene expression. All three mechanisms worsen ETC function "
                    "in an already CII-deficient patient. VPA-induced hepatotoxicity also reported in "
                    "mitochondrial disease."
                ),
                "alternative": "Levetiracetam (LEV) — renal excretion, no mitochondrial toxicity, first-choice AED. Lacosamide as adjunct if needed.",
            },
            {
                "drug":       "Succinate supplementation",
                "level":      "NOT RECOMMENDED",
                "mechanism":  (
                    "CII (SDH) is the enzyme that metabolizes succinate. In SDHAF1 deficiency, CII is "
                    "deficient (10–30% activity). Exogenous succinate supplementation cannot be oxidized "
                    "by the deficient CII, will further accumulate, and may worsen succinate toxicity. "
                    "Unlike CI deficiency where succinate (CII substrate) can partially bypass the "
                    "defect, in CII deficiency succinate is the substrate of the deficient enzyme itself."
                ),
                "alternative": "CoQ10 (Ubiquinol) Level C — antioxidant support. There is no specific metabolic bypass for CII deficiency analogous to succinate bypass for CI.",
            },
            {
                "drug":       "Riboflavin",
                "level":      "NOT INDICATED",
                "mechanism":  (
                    "Riboflavin (vitamin B2) is a FAD/FMN precursor. Riboflavin supplementation can "
                    "rescue FAD-dependent enzyme defects (e.g., SDHA FAD subunit, ACAD9 FAD-binding "
                    "scaffold for CI). SDHAF1 is a LYR-motif FeS delivery scaffold with NO FAD domain — "
                    "riboflavin has no mechanistic pathway to rescue SDHAF1 dysfunction. "
                    "Do not treat SDHAF1 deficiency as if it were SDHA (FAD-domain) deficiency. "
                    "Critical DDx: riboflavin may benefit SDHA but not SDHAF1."
                ),
                "alternative": "No riboflavin. Focus on CoQ10, thiamine (mandatory empiric), biotin (mandatory empiric).",
            },
        ],
    }
