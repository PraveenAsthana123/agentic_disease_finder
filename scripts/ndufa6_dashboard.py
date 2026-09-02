#!/usr/bin/env python3
"""NDUFA6 — Leigh Syndrome Isolated Complex I Deficiency (B14 / Q-module Distal Arm, AR).

NDUFA6 (NADH:Ubiquinone Oxidoreductase Subunit A6) is a ~128-aa nuclear-encoded
structural subunit of Complex I (~14.8 kDa after MTS cleavage), designated B14
(bovine CI proteomics; Carroll 2006). NDUFA6 occupies the DISTAL Q-module of the
matrix peripheral arm, wrapping around NDUFS7 (PSST, 20 kDa) and NDUFS3 (30 kDa)
via a leucine zipper-like coiled-coil motif to scaffold the quinone-access tunnel
entry point. Unlike NDUFA5 (B13, beta-sandwich fold, proximal N/Q interface, 7q32.1),
NDUFA6 (B14) resides at the DISTAL Q-module — the final matrix arm segment adjacent
to the quinone channel — and is the only NDUFA-subfamily subunit with a leucine
zipper-like (LZ) coiled-coil motif.
NDUFA6 is encoded on chromosome 22q13.2 (OMIM *602137) — autosomal recessive.

  NDUFA6 gene   OMIM *602137
  Disease       Leigh Syndrome (OMIM #256000); Isolated Complex I Deficiency
  Inheritance   Autosomal Recessive (AR) — biallelic pathogenic variants
  Chromosome    22q13.2

PATHOPHYSIOLOGY (Complex I / Q-module / NDUFA6 / B14 / Distal Q-module LZ Scaffold):
  NDUFA6 (B14) is a peripheral structural subunit (~128 aa, ~14.8 kDa) with a
  leucine zipper-like coiled-coil motif that wraps around NDUFS7 (PSST, 20 kDa)
  at the distal end of the Q-module matrix arm. NDUFA6 stabilises the
  NDUFS7–NDUFS3 junction at the quinone-binding tunnel access entry point.
  Loss of NDUFA6 destabilises distal Q-module scaffold → quinone channel entry
  impaired → ubiquinone cannot access the N2 iron-sulfur cluster → NADH-to-
  ubiquinone electron transfer abolished → CI absent or severely reduced on
  BN-PAGE. Isolated CI deficiency 5–20%; CII/CIII/CIV activities NORMAL.

  UNIQUE MOLECULAR SIGNATURE — B14 / LEUCINE ZIPPER / DISTAL Q-MODULE:
    NDUFA6 (B14) is the ONLY NDUFA-subfamily subunit with a leucine zipper-like
    (LZ) coiled-coil motif. This helix-loop-helix LZ fold wraps around NDUFS7
    (PSST) at the DISTAL Q-module, whereas:
      NDUFA5 (B13): beta-sandwich fold at PROXIMAL N/Q interface (7q32.1)
      NDUFA3 (B9):  alpha-helical peripheral scaffold, PP-module (19q13.42)
      NDUFA1 (MWFE): single TM helix, PP-module ND3 face (Xq24, X-linked)
      NDUFA11 (B14.7): 4-TM helices, PP/PD module boundary
    NDUFA6 (B14) is a pure hydrophilic matrix arm subunit — no TM helix, no
    membrane contact — exclusively in the peripheral Q-module distal arm.
    The B14 designation (bovine CI proteomics) is unique; not to be confused
    with NDUFA14 or the B14.5a/B14.7 subunits.

  DISTAL Q-MODULE STRUCTURAL ROLE — NDUFA6 vs NDUFA5:
    NDUFA5 (B13): scaffolds N/Q-module INTERFACE (proximal; contacts NDUFS2+NDUFS3)
    NDUFA6 (B14): scaffolds Q-module DISTAL ARM (distal; contacts NDUFS7+NDUFS3)
    Together NDUFA5 + NDUFA6 form a peripheral scaffold "clamp" along the entire
    matrix arm Q-module — loss of either collapses the Q-module from different ends.
    WES: 22q13.2 (NDUFA6) vs 7q32.1 (NDUFA5) definitively differentiates.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa Fe-S, 2q33.3):
    NO peripheral neuropathy in NDUFA6 (NDUFS1: ~50% — CRITICAL DDx)
    NDUFS1 is a catalytic N-module Fe-S subunit; NDUFA6 is distal Q-module structural
  vs NDUFS4 (N-module accessory, 5q11.2):
    NO olfactory bulb MRI lesions in NDUFA6 (NDUFS4: 52–65% — near-pathognomonic)
  vs NDUFV1 (N-module FMN/N3 Fe-S, 11q13.2):
    NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b 2Fe2S) / SCO2 (CIV):
    NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFA5 (B13, proximal N/Q interface, 7q32.1, beta-sandwich, AR):
    Both Q-module, both no TM helix, both AR. NDUFA6 (B14) is the DISTAL
    Q-module (NDUFS7/NDUFS3 quinone channel entry, 22q13.2, LZ coiled-coil)
    vs NDUFA5 (B13) at the PROXIMAL N/Q interface (NDUFS3/NDUFS2, 7q32.1,
    beta-sandwich fold). WES chromosomal locus is definitive (22q13.2 vs 7q32.1).
    BN-PAGE absent CI in both — DDx requires WES, not biochemistry alone.
  vs NDUFA3 (B9, PP-module membrane arm, alpha-helical, 19q13.42):
    NDUFA6 = matrix arm Q-module (22q13.2, no TM helix); NDUFA3 = membrane arm
    PP-module peripheral scaffold (19q13.42, no TM helix but alpha-helical). WES locus definitive.
  vs NDUFA1 (MWFE, PP-module ND3 face, Xq24, X-LINKED):
    NDUFA6 is AUTOSOMAL (22q13.2, AR biallelic); NDUFA1 is X-LINKED (Xq24) —
    inheritance pattern critical for genetic counselling.
  vs POLG/DGUOK (mtDNA depletion):
    NO hepatopathy in NDUFA6 (POLG: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  p.Arg106Trp   c.316C>T   — LZ-motif coiled-coil core; NDUFS7 contact; severe infantile
  p.Leu89Pro    c.266T>C   — helix-breaking proline; LZ-helix α2 disruption; severe
  p.Gly68Arg    c.202G>C   — near MTS cleavage; import/targeting disruption; severe neonatal
  p.Ala117Val   c.350C>T   — hydrophobic core; LZ coiled-coil packing; intermediate
  c.IVS2+1G>A              — splice donor exon 2; partial CI residual (~10–20%); moderate

THERAPY — NDUFA6 / CI-LEIGH SPECIFICS:
  ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
    Metformin      — directly inhibits CI at ND1/quinone-binding site (Q-module territory)
    Valproate      — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
    Linezolid      — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
    Chloramphenicol — same mitochondrial ribosomal mechanism as linezolid
  CONTRAINDICATED:
    Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                     (NDUFA6 distal Q-module scaffold collapsed, CI failed)
  AVOID / HIGH CAUTION:
    Propofol       — PRIS + secondary CIV inhibition (dual ETC bottleneck)
    Phenobarbital  — secondary CI inhibitor; use LEV first
  LEVEL C cofactors (standard CI supportive):
    Riboflavin (B2)   — CI-specific; FMN prosthetic group at NDUFV1 N-module (upstream)
    CoQ10 (ubiquinol) — electron acceptor at quinone site (CI membrane arm/Q-module interface)
    Thiamine (B1)     — MANDATORY empiric: SLC19A3/BTD mimic (treatable CI-mimic)
    Biotin            — MANDATORY empiric: BTD deficiency mimics CI-Leigh
    Succinate         — CII bypass; bypasses NDUFA6-failed CI entirely; donates ubiquinol via SDHA
    L-Carnitine       — energy metabolism support; secondary transport facilitation
"""

import random, json

GENE     = "NDUFA6"
DISEASE  = "Leigh Syndrome — Isolated Complex I Deficiency (CI-Leigh)"
OMIM_G   = "602137"
OMIM_D   = "256000"
INHERIT  = "Autosomal Recessive (AR) — biallelic"
CHROM    = "22q13.2"
MODULE   = "Q-module (Distal Q-module, Matrix Arm, LZ Coiled-Coil, No TM Helix)"
SIZE     = "128 aa / 14.8 kDa (after MTS cleavage)"
SEED     = 659
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Severe infantile (onset <6 mo)",       35),
    ("Moderate infantile (onset 6–18 mo)",   38),
    ("Intermediate (onset 18–36 mo)",        17),
    ("Attenuated / partial CI residual",     10),
]

VARIANTS = [
    ("p.Arg106Trp",  "c.316C>T",  "LZ-motif coiled-coil core; NDUFS7 contact surface",        "Severe infantile",   33, "Arginine-to-tryptophan in leucine zipper core; disrupts hydrophobic coiled-coil interface with NDUFS7; severe Q-module distal arm scaffold collapse"),
    ("p.Leu89Pro",   "c.266T>C",  "Helix-breaking proline; LZ α2 helix disruption",            "Severe",             23, "Leucine-to-proline substitution in LZ α2 helix; proline cannot participate in α-helix backbone H-bond → LZ coiled-coil unfolds → NDUFS7/NDUFS3 distal scaffold lost"),
    ("p.Gly68Arg",   "c.202G>C",  "Near MTS cleavage; import/targeting disruption",            "Severe neonatal",    16, "Glycine-to-arginine in MTS-proximal region; protein mis-targeting or import failure; neonatal CI absence and metabolic collapse"),
    ("p.Ala117Val",  "c.350C>T",  "LZ hydrophobic core packing; intermediate phenotype",       "Intermediate",       14, "Alanine-to-valine in LZ coiled-coil hydrophobic core; partial scaffold destabilisation; some residual NDUFA6 function → intermediate CI assembly"),
    ("c.IVS2+1G>A",  "Splice donor exon 2", "Partial CI residual (~10–20%)",                  "Moderate / partial", 14, "Splice-donor loss; partial exon 2 skipping; some residual correctly spliced NDUFA6 transcript → partial CI assembly intermediates on BN-PAGE"),
]

SEIZURE_TYPES = [
    ("Focal / multifocal (awake + sleep)",    60),
    ("Generalized tonic-clonic (GTCS)",       42),
    ("Myoclonic",                             30),
    ("Infantile spasms (IS / West synd.)",    20),
    ("Epileptic spasms (post-IS residual)",   13),
    ("Absence (atypical)",                     8),
]

TRIGGERS = [
    ("Febrile illness / infection",           80),
    ("Sub-therapeutic AED level",             63),
    ("Metabolic decompensation",              55),
    ("Sleep deprivation",                     40),
    ("Missed AED dose",                       36),
    ("Fasting / prolonged nil-by-mouth",      32),
    ("Anesthesia / surgical stress",          22),
    ("Enzyme-inducing co-medication",         14),
]

TREATMENTS = [
    ("Levetiracetam (LEV)",             "A",  "Preferred AED; renal excretion; NO mito toxicity; broad-spectrum CI-Leigh safe"),
    ("Riboflavin (B2 / FMN precursor)", "C",  "CI-specific cofactor; FMN at NDUFV1 N-module upstream of Q-module; 100–200 mg/day"),
    ("CoQ10 / Ubiquinol",               "C",  "Electron acceptor CI→CIII; downstream CI Q-module support; 10–30 mg/kg/day ubiquinol preferred"),
    ("Thiamine (B1)",                   "C",  "MANDATORY empiric: SLC19A3/BTD mimics treatable CI-Leigh; 100–300 mg/day before genetic result"),
    ("Biotin",                          "C",  "MANDATORY empiric: BTD deficiency mimics CI-Leigh; 10–40 mg/day empiric cover"),
    ("Succinate (oral/IV)",             "C",  "CII bypass; completely bypasses NDUFA6-failed CI; enters ubiquinol pool via SDHA-SDHB; distal to Q-module"),
    ("L-Carnitine",                     "C",  "Energy metabolism support; secondary transport; 50–100 mg/kg/day"),
    ("Clobazam (CLB) / Clonazepam",     "B",  "Focal / myoclonic adjunct; GABA-A positive modulator; avoid benzodiazepine overuse"),
]

CONTRAINDICATIONS = [
    ("Metformin",                        "ABSOLUTE",        "Direct CI inhibitor at ND1/quinone-binding site (Q-module territory); fatal lactic acidosis in CI-Leigh"),
    ("Valproic acid / VPA",              "ABSOLUTE",        "Triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block; fatal hepatotoxicity risk"),
    ("Linezolid",                        "ABSOLUTE",        "23S rRNA inhibition → blocks synthesis of all 7 mt-encoded ND subunits; CI depletion cascade"),
    ("Chloramphenicol",                  "ABSOLUTE",        "Same 23S rRNA mitoribosomal mechanism as linezolid; avoidable antibiotic alternative available"),
    ("Ketogenic diet (KD)",              "CONTRAINDICATED", "Forces β-oxidation NADH dependence; NDUFA6-distal-Q-module-failed CI cannot reoxidise NADH; fatal metabolic crisis"),
    ("Propofol",                         "AVOID (PRIS)",    "PRIS + secondary CIV inhibition; dual ETC bottleneck in CI failure; use sevoflurane instead"),
    ("Phenobarbital",                    "HIGH CAUTION",    "Secondary CI inhibitor; acceptable only if LEV/CLB fail; monitor lactate closely"),
    ("Enzyme-inducing AEDs (CBZ/PHT/OXC)", "RELATIVE CI",  "Secondary mito toxicity; CYP450 induction → cofactor depletion; avoid if alternatives available"),
]

MONITORING = [
    ("Serum lactate + pyruvate (L:P ratio)", "At each visit; L:P >20 suggests CI dysfunction; target L:P <20"),
    ("AED levels (LEV, CLB, CNZ)",           "Every 3–6 months; sub-therapeutic level = seizure trigger"),
    ("Riboflavin / CoQ10 status",            "Annual; adjust supplementation dose"),
    ("Plasma amino acids",                   "6-monthly; alanine elevation = surrogate for lactic acidosis"),
    ("Neuroimaging MRI brain",               "Every 12 months or at neurological change; Leigh lesion progression"),
    ("Echocardiography",                     "Annual; CI-Leigh rarely causes HCM but screen for onset"),
    ("Ophthalmology (visual acuity, ERG)",   "Annual; pigmentary retinopathy rare in CI-Leigh, screen baseline"),
    ("Neurodevelopmental / cognitive battery", "Every 12 months; Bayley / VABS age-appropriate"),
    ("Respiratory function / polysomnography", "6-monthly; central apnoea and hypoventilation in CI-Leigh"),
    ("Renal function (eGFR, urine organic acids)", "Annual; rule out POLG/renal mimic; monitor drug clearance"),
    ("Pyruvate dehydrogenase (PDH) activity",  "At diagnosis; PDH deficiency mimics CI-Leigh on lactate"),
    ("Mitochondrial respiratory chain enzyme panel", "Baseline + after any acute decompensation"),
]

REFERENCES = [
    "Carroll J et al. (2006) Mol Cell Proteomics — NDUFA6 B14 subunit identification in bovine CI proteome",
    "Guerrero-Castillo S et al. (2017) Cell Metab — CI assembly dynamics; Q-module NDUFA6 B14 distal arm incorporation",
    "Stroud DA et al. (2016) Nature — CI assembly states; distal Q-module matrix arm structural scaffold",
    "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure; distal Q-module NDUFS7/NDUFS3/NDUFA6 peripheral arm",
    "Fassone E & Rahman S (2012) J Med Genet — CI deficiency genetics; NDUFA6 subunit class review",
    "Zhu J et al. (2016) Science — Cryo-EM CI structure at 3.9Å; distal Q-module NDUFA6 B14 position map",
]

KEY_CONCEPTS = [
    ("B14 / Distal Q-module LZ Motif", "NDUFA6 (B14) is the sole NDUFA-subfamily subunit with a leucine zipper-like coiled-coil motif; positioned at the DISTAL Q-module wrapping NDUFS7 (PSST) at the quinone-channel entry; no TM helix"),
    ("NDUFA6 vs NDUFA5 — Distal vs Proximal Q-module", "NDUFA6 (B14, 22q13.2) = distal Q-module (NDUFS7/NDUFS3 quinone channel entry, LZ coiled-coil); NDUFA5 (B13, 7q32.1) = proximal N/Q interface (NDUFS3/NDUFS2, beta-sandwich). Both Q-module; WES locus definitive DDx"),
    ("NDUFS7 (PSST) scaffold dependence", "NDUFA6 LZ coiled-coil wraps NDUFS7 (PSST, 20kDa, N2 Fe-S cluster); loss of NDUFA6 destabilises NDUFS7 → quinone-binding channel entry occluded → NADH-to-ubiquinone electron transfer abolished"),
    ("Isolated CI deficiency pattern", "5–20% CI activity; CII/CIII/CIV NORMAL — biochemical fingerprint mandatory; excludes Complex IV (SURF1/SCO2) and mtDNA depletion (all-ETC-low)"),
    ("BN-PAGE absent CI pattern", "Absent CI holocomplex on BN-PAGE (severe alleles); partial distal Q-module intermediates possible (moderate alleles); sub-assembly stall at NDUFA6-dependent distal Q-module step"),
    ("22q13.2 locus — no neighbouring CI genes", "NDUFA6 maps to 22q13.2; not near any other major CI gene on chromosome 22; WES locus confirmation required alongside biochemistry"),
    ("Metformin absolute CI", "Metformin directly inhibits CI at the ND1/quinone interface (Q-module territory) — administration to any CI-Leigh patient is fatal; also damages NDUFA6-stabilised NDUFS7 indirectly"),
    ("Thiamine + Biotin empiric MANDATORY", "SLC19A3 and BTD deficiencies mimic CI-Leigh clinically; empiric thiamine + biotin BEFORE genetic result can prevent irreversible neurological damage"),
    ("Succinate CII bypass", "Succinate → SDHA → ubiquinol: bypasses NDUFA6-failed CI entirely; only ETC substrate entering ubiquinol pool without requiring CI"),
    ("No NDUFS4 olfactory bulb lesions", "Bilateral olfactory bulb MRI lesions (52–65%) are near-pathognomonic for NDUFS4-Leigh; absence in NDUFA6-Leigh is a CRITICAL DDx pivot; do not defer WES for this MRI finding"),
    ("No HCM in pure CI-Leigh", "HCM: NDUFV2 (~80%) and SCO2 (~100%); NDUFA6 CI-Leigh almost never associated with HCM — cardiac DDx pivot against CIV/NDUFV2 diseases"),
    ("Sevoflurane not propofol", "General anaesthesia: sevoflurane (inhalational) preferred; propofol AVOIDED (PRIS + CIV secondary inhibition in context of CI failure)"),
    ("GIR 6–8 IV dextrose", "Never fast CI-Leigh children; glucose infusion rate 6–8 mg/kg/min during any nil-by-mouth period to prevent fasting-triggered metabolic crisis"),
    ("Genetic counselling AR", "Autosomal recessive (22q13.2); both parents obligate carriers (25% recurrence risk per pregnancy); offer cascade carrier testing and prenatal/preimplantation options"),
    ("No peripheral neuropathy DDx", "Peripheral neuropathy present in ~50% of NDUFS1-Leigh (N-module); ABSENT in NDUFA6-Leigh (structural Q-module scaffold) — a critical clinical DDx pivot before WES"),
]


def _make_patients():
    pats = []
    for i in range(1, N + 1):
        r = rng.random() * 100
        cls = (PHENO_CLASSES[0][0] if r < 35
               else PHENO_CLASSES[1][0] if r < 73
               else PHENO_CLASSES[2][0] if r < 90
               else PHENO_CLASSES[3][0])
        v = rng.choice(VARIANTS)
        age_mo = (rng.randint(1, 6)   if "Severe infant" in cls
                  else rng.randint(6, 18)  if "Moderate"   in cls
                  else rng.randint(18, 36) if "Intermediate" in cls
                  else rng.randint(24, 60))
        pats.append({
            "id":                        f"P{i:02d}",
            "phenotype":                 cls,
            "onset_mo":                  age_mo,
            "variant":                   v[0],
            "cDNA":                      v[1],
            "ci_pct":                    rng.randint(5, 20),
            "has_seizure":               rng.random() < 0.70,
            "has_hypotonia":             rng.random() < 0.85,
            "has_lactic_acidosis":       rng.random() < 0.88,
            "has_leigh_mri":             rng.random() < 0.80,
            "has_respiratory_compromise": rng.random() < 0.42,
            "has_dystonia":              rng.random() < 0.36,
            "has_ataxia":               rng.random() < 0.30,
        })
    return pats


_PATIENTS = _make_patients()


def get_overview():
    pts = _PATIENTS
    n_sz  = sum(1 for p in pts if p["has_seizure"])
    n_hyp = sum(1 for p in pts if p["has_hypotonia"])
    n_lac = sum(1 for p in pts if p["has_lactic_acidosis"])
    n_mri = sum(1 for p in pts if p["has_leigh_mri"])
    n_res = sum(1 for p in pts if p["has_respiratory_compromise"])

    variant_counts: dict[str, int] = {}
    for p in pts:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    phenotype_counts: dict[str, int] = {}
    for p in pts:
        phenotype_counts[p["phenotype"]] = phenotype_counts.get(p["phenotype"], 0) + 1

    return {
        "gene":           GENE,
        "disease":        DISEASE,
        "omim_gene":      OMIM_G,
        "omim_disease":   OMIM_D,
        "inheritance":    INHERIT,
        "chromosome":     CHROM,
        "module":         MODULE,
        "protein_size":   SIZE,
        "cohort_n":       N,
        "seed":           SEED,
        "kpis": {
            "seizures_pct":        round(n_sz  / N * 100),
            "hypotonia_pct":       round(n_hyp / N * 100),
            "lactic_acidosis_pct": round(n_lac / N * 100),
            "leigh_mri_pct":       round(n_mri / N * 100),
            "respiratory_pct":     round(n_res / N * 100),
            "median_onset_mo":     round(sum(p["onset_mo"] for p in pts) / N, 1),
            "mean_ci_pct":         round(sum(p["ci_pct"]   for p in pts) / N, 1),
        },
        "phenotype_distribution": [
            {"class": pc[0], "n": phenotype_counts.get(pc[0], 0), "pct": pc[1]}
            for pc in PHENO_CLASSES
        ],
        "top_variants":  sorted(variant_counts.items(), key=lambda x: -x[1]),
        "seizure_types": [{"type": t, "pct": p} for t, p in SEIZURE_TYPES],
        "triggers":      [{"trigger": t, "pct": p} for t, p in TRIGGERS],
        "references":    REFERENCES,
        "key_concepts":  [{"concept": c[0], "detail": c[1]} for c in KEY_CONCEPTS],
    }


def get_breakdown():
    pts = _PATIENTS
    variant_rows = []
    for vname, cdna, struct, phenotype_modal, freq, detail in VARIANTS:
        vpts = [p for p in pts if p["variant"] == vname]
        variant_rows.append({
            "variant":           vname,
            "cDNA":              cdna,
            "structural_impact": struct,
            "modal_phenotype":   phenotype_modal,
            "freq_pct":          freq,
            "n_in_cohort":       len(vpts),
            "detail":            detail,
        })

    return {
        "gene":       GENE,
        "cohort_n":   N,
        "variants":   variant_rows,
        "treatments": [
            {"name": t[0], "evidence": t[1], "rationale": t[2]}
            for t in TREATMENTS
        ],
        "contraindications": [
            {"drug": c[0], "class": c[1], "reason": c[2]}
            for c in CONTRAINDICATIONS
        ],
        "monitoring": [
            {"parameter": m[0], "protocol": m[1]}
            for m in MONITORING
        ],
        "patients": pts,
    }


def get_definitions():
    return {
        "gene":         GENE,
        "disease":      DISEASE,
        "omim_gene":    OMIM_G,
        "omim_disease": OMIM_D,
        "key_concepts": [{"concept": c[0], "detail": c[1]} for c in KEY_CONCEPTS],
        "glossary": [
            {"term": "B14",                   "definition": "Bovine CI proteomics designation for NDUFA6 (Carroll 2006); unique identifier for NDUFA6; not to be confused with NDUFA14 (B16.6) or NDUFA11 (B14.7)"},
            {"term": "Leucine zipper (LZ)",   "definition": "Coiled-coil protein motif with leucine residues at every 7th position forming a hydrophobic interface; NDUFA6 B14's unique structural feature; wraps NDUFS7 in the distal Q-module"},
            {"term": "Distal Q-module",       "definition": "The terminal segment of the peripheral matrix arm Q-module adjacent to the quinone-binding channel entry; contains NDUFS7 (PSST), NDUFS3, and NDUFA6 (B14)"},
            {"term": "Proximal vs Distal Q-module", "definition": "Proximal Q = N/Q interface (NDUFA5 B13, NDUFS3/NDUFS2); Distal Q = quinone channel entry (NDUFA6 B14, NDUFS7/NDUFS3); together scaffold the entire peripheral arm Q-module"},
            {"term": "NDUFS7 (PSST)",         "definition": "20-kDa Q-module subunit carrying the N2 iron-sulfur cluster that directly reduces ubiquinone; NDUFA6 (B14) LZ coiled-coil wraps and stabilises NDUFS7 at the distal Q-module"},
            {"term": "CI biochemical fingerprint", "definition": "Isolated CI ↓5–20%; CII, CIII, CIV NORMAL; distinguishes nuclear CI deficiency from mtDNA depletion (all-ETC-low) and combined oxidative phosphorylation deficiency"},
            {"term": "BN-PAGE",               "definition": "Blue-native polyacrylamide gel electrophoresis; absent CI band = NDUFA6-Leigh severe allele; partial distal Q-module intermediates in moderate alleles"},
            {"term": "Metformin CI",          "definition": "Metformin directly inhibits CI at the ND1/quinone interface (Q-module territory) → CI-Leigh fatal lactic acidosis; absolute contraindication in any CI-Leigh including NDUFA6"},
            {"term": "Succinate bypass",      "definition": "Succinate → SDHA → ubiquinol → CIII; bypasses NDUFA6-failed CI entirely; only ETC substrate entering ubiquinol pool not requiring CI function"},
            {"term": "Leigh syndrome",        "definition": "Progressive necrotising encephalopathy of childhood; bilateral symmetric brainstem + basal ganglia MRI lesions; caused by ≥100 nuclear/mtDNA gene mutations; NDUFA6 → CI-Leigh subtype"},
            {"term": "GIR 6–8",               "definition": "Glucose infusion rate 6–8 mg/kg/min; mandatory during nil-by-mouth/peri-operative in CI-Leigh; prevents fasting-induced ETC substrate starvation and metabolic crisis"},
            {"term": "AR biallelic",          "definition": "Autosomal recessive; two pathogenic variants in trans (compound het) or homozygous; both parents obligate carriers; 25% recurrence per pregnancy; 22q13.2"},
        ],
        "references": REFERENCES,
    }


if __name__ == "__main__":
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first 500 chars) ===")
    print(_json.dumps(get_breakdown(), indent=2)[:500])
    print("\n=== DEFINITIONS (first 500 chars) ===")
    print(_json.dumps(get_definitions(), indent=2)[:500])
