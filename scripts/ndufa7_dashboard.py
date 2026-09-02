#!/usr/bin/env python3
"""NDUFA7 — Leigh Syndrome Isolated Complex I Deficiency (B14.5a / N-module Peripheral, AR).

NDUFA7 (NADH:Ubiquinone Oxidoreductase Subunit A7) is a ~179-aa nuclear-encoded
structural subunit of Complex I (~20.3 kDa after MTS cleavage), designated B14.5a
(bovine CI proteomics; Carroll 2006). NDUFA7 occupies the PERIPHERAL N-MODULE of
the matrix arm, contacting NDUFS1 (IP1/75kDa — the central Fe-S relay hub carrying
N1b, N4, N5 clusters) and stabilising the NDUFV1–NDUFV2–NDUFS1 N-module core.
NDUFA7 carries NO Fe-S cluster itself — it is a purely structural stabiliser of the
N-module's electron-relay core. NDUFA7 is located on chromosome 19p13.3, distinct
from NDUFA11 (B14.7) at 19q13.33 on the same chromosome — critical DDx naming trap.
NDUFA7 is encoded on chromosome 19p13.3 (OMIM *601796) — autosomal recessive.

  NDUFA7 gene   OMIM *601796
  Disease       Leigh Syndrome (OMIM #256000); Isolated Complex I Deficiency
  Inheritance   Autosomal Recessive (AR) — biallelic pathogenic variants
  Chromosome    19p13.3

PATHOPHYSIOLOGY (Complex I / N-module / NDUFA7 / B14.5a / Peripheral N-module Stabiliser):
  NDUFA7 (B14.5a) is a peripheral structural subunit (~179 aa, ~20.3 kDa) with no
  TM helix, residing in the matrix arm N-module where it contacts NDUFS1 (IP1,
  75kDa) — the central Fe-S electron relay hub carrying N1b, N4, and N5 Fe-S clusters.
  NDUFA7 also contacts NDUFV1 (51kDa, FMN) and NDUFV2 (24kDa, N1b 2Fe-2S) to
  stabilise the tripartite N-module core. NDUFA7 has no Fe-S cluster of its own.
  Loss of NDUFA7 destabilises the NDUFS1 peripheral scaffold → N-module core
  dissociation → NADH oxidation step fails from the N-module side → NADH-to-
  ubiquinone electron relay abolished → CI absent or severely reduced on BN-PAGE.
  Isolated CI deficiency 5–20%; CII/CIII/CIV activities NORMAL.

  UNIQUE MOLECULAR SIGNATURE — B14.5a / N-MODULE / NDUFS1-CONTACT / 19p13.3:
    NDUFA7 (B14.5a) is the ONLY NDUFA-subfamily subunit specifically designated
    B14.5a (as opposed to B14 = NDUFA6, B14.7 = NDUFA11, B14.5b = NDUFA8).
    Its role: stabilising NDUFS1 (IP1/75kDa) — the largest nuclear-encoded CI
    subunit and the central Fe-S relay axis — from the peripheral N-module face.
    NDUFA7 is purely structural (no Fe-S cluster, no catalytic domain, no TM helix).
    This distinguishes it from all catalytic N-module subunits (NDUFV1, NDUFV2,
    NDUFS1 itself) and from all membrane-arm NDUFA subunits (NDUFA1, NDUFA3,
    NDUFA11).

  CRITICAL NAMING TRAP — NDUFA7 (B14.5a) vs NDUFA11 (B14.7) BOTH ON CHROMOSOME 19:
    NDUFA7 (B14.5a): 19p13.3 — N-module peripheral stabiliser (matrix arm)
    NDUFA11 (B14.7): 19q13.33 — PP/PD inter-module membrane boundary (4 TM helices)
    Both on chromosome 19, both NDUFA family, entirely different CI modules.
    WES chromosomal arm (19p vs 19q) is the definitive differentiation tool.
    Naming confusion risk: "B14.5a" vs "B14.7" both contain "14" but refer to
    completely different structural roles; never conflate from the name alone.

  NDUFS1 SCAFFOLD DEPENDENCE — NDUFA7 PERIPHERAL CONTACT:
    NDUFA7 (B14.5a) wraps around NDUFS1 (IP1/75kDa) at the N-module peripheral face.
    NDUFS1 carries N1b, N4, N5 Fe-S clusters — the central relay from NDUFV1-FMN
    through N3, N4, N5 toward N2 (NDUFS7-carried). Loss of NDUFA7 peripheral
    scaffold destabilises NDUFS1 → N-module sub-assembly intermediate stalls on
    BN-PAGE. Unlike NDUFA13 (GRIM-19, NDUFV1-face peripheral) and NDUFA12 (N-Q
    interface), NDUFA7 specifically stabilises the NDUFS1 peripheral face.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa Fe-S, 2q33.3):
    NO peripheral neuropathy in NDUFA7 (NDUFS1: ~50% — CRITICAL DDx)
    NDUFS1 is catalytic with Fe-S clusters; NDUFA7 is structural N-module scaffold
  vs NDUFS4 (N-module accessory, 5q11.2):
    NO olfactory bulb MRI lesions in NDUFA7 (NDUFS4: 52–65% — near-pathognomonic)
  vs NDUFV1 (N-module FMN/N3 Fe-S, 11q13.2):
    NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b 2Fe2S) / SCO2 (CIV):
    NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFA11 (B14.7, 4-TM, PP/PD boundary, 19q13.33):
    NDUFA7 = N-module peripheral (19p13.3, no TM helix); NDUFA11 = PP/PD membrane
    boundary (19q13.33, 4 TM helices). SAME chromosome (19), different arms —
    the naming-and-locus trap; WES chromosomal arm is definitive (19p vs 19q).
  vs NDUFA13 (GRIM-19/B16.6, NDUFV1-face N-module, 19p13.11):
    NDUFA7 (19p13.3) and NDUFA13 (19p13.11): both peripheral N-module, both 19p!
    NDUFA7 contacts NDUFS1; NDUFA13 contacts NDUFV1 face AND has STAT3-inhibitor
    function. WES exact sub-band (19p13.3 vs 19p13.11) is mandatory differentiation.
  vs NDUFA6 (B14, Q-module distal arm, LZ coiled-coil, 22q13.2):
    NDUFA7 = N-module matrix arm peripheral (19p13.3); NDUFA6 = Q-module distal
    scaffold (22q13.2, LZ coiled-coil). Different modules, different chromosomes.
  vs POLG/DGUOK (mtDNA depletion):
    NO hepatopathy in NDUFA7 (POLG: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  p.Arg83Cys    c.247C>T   — N-module NDUFS1 contact surface; severe infantile
  p.Leu62Pro    c.185T>C   — helix-breaking proline in α-helix; severe
  p.Glu27Lys    c.79G>A    — near MTS cleavage; import/targeting disruption; severe neonatal
  p.Ala142Val   c.425C>T   — peripheral core packing; intermediate phenotype
  c.IVS3+1G>A              — splice donor exon 3; partial CI residual (~10–20%); moderate

THERAPY — NDUFA7 / CI-LEIGH SPECIFICS:
  ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
    Metformin      — directly inhibits CI at ND1/quinone-binding site
    Valproate      — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
    Linezolid      — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
    Chloramphenicol — same mitochondrial ribosomal mechanism as linezolid
  CONTRAINDICATED:
    Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                     (NDUFA7 N-module peripheral scaffold collapsed, CI failed)
  AVOID / HIGH CAUTION:
    Propofol       — PRIS + secondary CIV inhibition (dual ETC bottleneck)
    Phenobarbital  — secondary CI inhibitor; use LEV first
  LEVEL C cofactors (standard CI supportive):
    Riboflavin (B2)   — CI-specific; FMN prosthetic group at NDUFV1 (N-module, same module as NDUFA7)
    CoQ10 (ubiquinol) — electron acceptor at quinone site (CI membrane arm/Q-module interface)
    Thiamine (B1)     — MANDATORY empiric: SLC19A3/BTD mimics treatable CI-Leigh
    Biotin            — MANDATORY empiric: BTD deficiency mimics CI-Leigh
    Succinate         — CII bypass; bypasses NDUFA7-failed CI entirely; enters ubiquinol via SDHA
    L-Carnitine       — energy metabolism support; secondary transport facilitation
"""

import random, json

GENE     = "NDUFA7"
DISEASE  = "Leigh Syndrome — Isolated Complex I Deficiency (CI-Leigh)"
OMIM_G   = "601796"
OMIM_D   = "256000"
INHERIT  = "Autosomal Recessive (AR) — biallelic"
CHROM    = "19p13.3"
MODULE   = "N-module (Peripheral N-module Stabiliser, NDUFS1-Contact, No TM Helix)"
SIZE     = "179 aa / 20.3 kDa (after MTS cleavage)"
SEED     = 661
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Severe infantile (onset <6 mo)",       34),
    ("Moderate infantile (onset 6–18 mo)",   38),
    ("Intermediate (onset 18–36 mo)",        18),
    ("Attenuated / partial CI residual",     10),
]

VARIANTS = [
    ("p.Arg83Cys",  "c.247C>T",  "N-module NDUFS1 contact surface; peripheral scaffold disrupted",      "Severe infantile",   34, "Arginine-to-cysteine at NDUFS1 (IP1/75kDa) contact surface; disrupts critical peripheral NDUFA7–NDUFS1 interface; N-module scaffold collapse; severe CI absence"),
    ("p.Leu62Pro",  "c.185T>C",  "Helix-breaking proline in α-helix; N-module fold disruption",         "Severe",             24, "Leucine-to-proline substitution in peripheral arm α-helix; proline cannot participate in backbone H-bond → NDUFA7 fold lost → NDUFS1 peripheral contact abolished"),
    ("p.Glu27Lys",  "c.79G>A",   "Near MTS cleavage; import/targeting disruption",                      "Severe neonatal",    15, "Glutamate-to-lysine in MTS-proximal region; protein mis-targeting or import failure; neonatal CI absence and metabolic collapse"),
    ("p.Ala142Val", "c.425C>T",  "Peripheral N-module core packing; intermediate phenotype",             "Intermediate",       13, "Alanine-to-valine in peripheral core hydrophobic region; partial NDUFA7 stability; some residual NDUFS1 contact → intermediate CI assembly"),
    ("c.IVS3+1G>A", "Splice donor exon 3", "Partial CI residual (~10–20%)",                             "Moderate / partial", 14, "Splice-donor loss; partial exon 3 skipping; some residual correctly spliced NDUFA7 transcript → partial N-module sub-assembly intermediates on BN-PAGE"),
]

SEIZURE_TYPES = [
    ("Focal / multifocal (awake + sleep)",    58),
    ("Generalized tonic-clonic (GTCS)",       40),
    ("Myoclonic",                             28),
    ("Infantile spasms (IS / West synd.)",    18),
    ("Epileptic spasms (post-IS residual)",   12),
    ("Absence (atypical)",                     7),
]

TRIGGERS = [
    ("Febrile illness / infection",           82),
    ("Sub-therapeutic AED level",             60),
    ("Metabolic decompensation",              53),
    ("Sleep deprivation",                     38),
    ("Missed AED dose",                       35),
    ("Fasting / prolonged nil-by-mouth",      30),
    ("Anesthesia / surgical stress",          20),
    ("Enzyme-inducing co-medication",         13),
]

TREATMENTS = [
    ("Levetiracetam (LEV)",             "A",  "Preferred AED; renal excretion; NO mito toxicity; broad-spectrum CI-Leigh safe"),
    ("Riboflavin (B2 / FMN precursor)", "C",  "CI-specific cofactor; FMN at NDUFV1 N-module — same N-module as NDUFA7; 100–200 mg/day"),
    ("CoQ10 / Ubiquinol",               "C",  "Electron acceptor CI→CIII; downstream N-module support; 10–30 mg/kg/day ubiquinol preferred"),
    ("Thiamine (B1)",                   "C",  "MANDATORY empiric: SLC19A3/BTD mimics treatable CI-Leigh; 100–300 mg/day before genetic result"),
    ("Biotin",                          "C",  "MANDATORY empiric: BTD deficiency mimics CI-Leigh; 10–40 mg/day empiric cover"),
    ("Succinate (oral/IV)",             "C",  "CII bypass; completely bypasses NDUFA7-failed CI; enters ubiquinol pool via SDHA-SDHB; distal to N-module failure"),
    ("L-Carnitine",                     "C",  "Energy metabolism support; secondary transport; 50–100 mg/kg/day"),
    ("Clobazam (CLB) / Clonazepam",     "B",  "Focal / myoclonic adjunct; GABA-A positive modulator; avoid benzodiazepine overuse"),
]

CONTRAINDICATIONS = [
    ("Metformin",                        "ABSOLUTE",        "Direct CI inhibitor at ND1/quinone-binding site; fatal lactic acidosis in CI-Leigh"),
    ("Valproic acid / VPA",              "ABSOLUTE",        "Triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block; fatal hepatotoxicity risk"),
    ("Linezolid",                        "ABSOLUTE",        "23S rRNA inhibition → blocks synthesis of all 7 mt-encoded ND subunits; CI depletion cascade"),
    ("Chloramphenicol",                  "ABSOLUTE",        "Same 23S rRNA mitoribosomal mechanism as linezolid; avoidable antibiotic alternative available"),
    ("Ketogenic diet (KD)",              "CONTRAINDICATED", "Forces β-oxidation NADH dependence; NDUFA7-N-module-failed CI cannot reoxidise NADH; fatal metabolic crisis"),
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
    "Carroll J et al. (2006) Mol Cell Proteomics — NDUFA7 B14.5a subunit identification in bovine CI proteome",
    "Guerrero-Castillo S et al. (2017) Cell Metab — CI assembly dynamics; N-module NDUFA7 B14.5a peripheral incorporation",
    "Stroud DA et al. (2016) Nature — CI assembly states; N-module peripheral stabiliser scaffold",
    "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure; N-module NDUFS1 peripheral arm; NDUFA7 B14.5a position",
    "Fassone E & Rahman S (2012) J Med Genet — CI deficiency genetics; NDUFA7 subunit class review",
    "Zhu J et al. (2016) Science — Cryo-EM CI structure at 3.9Å; N-module NDUFA7 B14.5a peripheral contact map",
]

KEY_CONCEPTS = [
    ("B14.5a / N-module NDUFS1 Peripheral Stabiliser", "NDUFA7 (B14.5a) stabilises NDUFS1 (IP1/75kDa — the central Fe-S relay hub, N1b+N4+N5 clusters) at the peripheral N-module face; no Fe-S cluster in NDUFA7 itself; purely structural"),
    ("NDUFA7 vs NDUFA11 — 19p13.3 vs 19q13.33 — SAME chromosome, DIFFERENT arms", "NDUFA7 (B14.5a): 19p13.3, N-module matrix arm peripheral, no TM helix; NDUFA11 (B14.7): 19q13.33, PP/PD membrane module boundary, 4 TM helices. Both on chromosome 19 — chromosomal ARM (p vs q) is the definitive WES pivot"),
    ("NDUFA7 vs NDUFA13 — 19p13.3 vs 19p13.11 — SAME chromosome ARM, DIFFERENT sub-bands", "NDUFA7 (B14.5a, 19p13.3) and NDUFA13 (GRIM-19, 19p13.11) both map to chromosome 19p. NDUFA7 contacts NDUFS1; NDUFA13 contacts NDUFV1 face and has STAT3-inhibitor function. High-resolution WES sub-band mandatory"),
    ("N-module structural hierarchy", "NDUFV1 (FMN, N3) → NDUFV2 (N1b) → NDUFS1 (N1b/N4/N5) → NDUFA7 stabilises NDUFS1 periphery → NDUFA12 stabilises N-Q interface → NDUFA13 stabilises NDUFV1 face; NDUFA7 is specific to the NDUFS1 peripheral scaffold layer"),
    ("No Fe-S cluster in NDUFA7", "NDUFA7 (B14.5a) carries no Fe-S cluster — pure structural stabiliser. DDx from catalytic N-module subunits (NDUFS1, NDUFV1, NDUFV2) which carry Fe-S/FMN: loss of NDUFA7 → N-module scaffold failure; loss of NDUFS1/NDUFV1/NDUFV2 → direct electron-relay failure"),
    ("Isolated CI deficiency pattern", "5–20% CI activity; CII/CIII/CIV NORMAL — biochemical fingerprint mandatory; excludes Complex IV (SURF1/SCO2) and mtDNA depletion (all-ETC-low)"),
    ("BN-PAGE N-module sub-assembly intermediates", "N-module sub-assembly intermediates on BN-PAGE (similar to NDUFA12, NDUFA13, NDUFS4) — distinct from cleaner absent CI in membrane-arm subunits (NDUFA1, NDUFA3, NDUFA11, NDUFB-series); N-module scaffold stall pattern"),
    ("Metformin absolute CI", "Metformin directly inhibits CI at the ND1/quinone interface → CI-Leigh fatal lactic acidosis; absolute contraindication in any CI-Leigh including NDUFA7"),
    ("Thiamine + Biotin empiric MANDATORY", "SLC19A3 and BTD deficiencies mimic CI-Leigh clinically; empiric thiamine + biotin BEFORE genetic result can prevent irreversible neurological damage"),
    ("Succinate CII bypass", "Succinate → SDHA → ubiquinol → CIII: bypasses NDUFA7-failed CI N-module entirely; only ETC substrate entering ubiquinol pool without requiring CI function"),
    ("No NDUFS4 olfactory bulb lesions", "Bilateral olfactory bulb MRI lesions (52–65%) are near-pathognomonic for NDUFS4-Leigh; absence in NDUFA7-Leigh is a CRITICAL DDx pivot; do not defer WES for this MRI finding"),
    ("No HCM in pure CI-Leigh", "HCM: NDUFV2 (~80%) and SCO2 (~100%); NDUFA7 CI-Leigh almost never associated with HCM — cardiac DDx pivot against CIV/NDUFV2 diseases"),
    ("No peripheral neuropathy DDx", "Peripheral neuropathy present in ~50% of NDUFS1-Leigh (N-module catalytic); ABSENT in NDUFA7-Leigh (structural N-module scaffold) — CRITICAL clinical DDx pivot before WES"),
    ("Sevoflurane not propofol", "General anaesthesia: sevoflurane (inhalational) preferred; propofol AVOIDED (PRIS + CIV secondary inhibition in context of CI failure)"),
    ("GIR 6–8 IV dextrose", "Never fast CI-Leigh children; glucose infusion rate 6–8 mg/kg/min during any nil-by-mouth period to prevent fasting-triggered metabolic crisis"),
    ("Genetic counselling AR", "Autosomal recessive (19p13.3); both parents obligate carriers (25% recurrence risk per pregnancy); offer cascade carrier testing and prenatal/preimplantation options"),
]


def _make_patients():
    pats = []
    for i in range(1, N + 1):
        r = rng.random() * 100
        cls = (PHENO_CLASSES[0][0] if r < 34
               else PHENO_CLASSES[1][0] if r < 72
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
            "has_seizure":               rng.random() < 0.72,
            "has_hypotonia":             rng.random() < 0.84,
            "has_lactic_acidosis":       rng.random() < 0.87,
            "has_leigh_mri":             rng.random() < 0.82,
            "has_respiratory_compromise": rng.random() < 0.40,
            "has_dystonia":              rng.random() < 0.34,
            "has_ataxia":               rng.random() < 0.32,
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
            {"term": "B14.5a",                 "definition": "Bovine CI proteomics designation for NDUFA7 (Carroll 2006); distinguishes from B14 (NDUFA6), B14.5b (NDUFA8), and B14.7 (NDUFA11); all carry '14' in their names but are entirely different subunits"},
            {"term": "N-module peripheral stabiliser", "definition": "Structural role of NDUFA7: contacts NDUFS1 (IP1/75kDa) at the peripheral face of the N-module matrix arm; no Fe-S cluster; no catalytic function; purely scaffolding"},
            {"term": "NDUFS1 (IP1/75kDa)",     "definition": "Largest nuclear-encoded CI subunit (727aa, 75kDa); carries Fe-S clusters N1b, N4, N5 — the central NADH-to-ubiquinone electron relay axis; stabilised by NDUFA7 peripherally"},
            {"term": "19p13.3 vs 19q13.33",    "definition": "NDUFA7 (B14.5a) maps to 19p (short arm); NDUFA11 (B14.7) maps to 19q (long arm). Same chromosome (19), different arms — both NDUFA subfamily, completely different CI modules (N-module vs PP/PD membrane boundary)"},
            {"term": "N-module sub-assembly intermediates", "definition": "BN-PAGE pattern in NDUFA7-Leigh: N-module stalls as a sub-assembly intermediate rather than cleanly absent CI (which is seen in membrane-arm subunit losses like NDUFA1, NDUFA11)"},
            {"term": "CI biochemical fingerprint", "definition": "Isolated CI ↓5–20%; CII, CIII, CIV NORMAL; distinguishes nuclear CI deficiency from mtDNA depletion (all-ETC-low) and combined oxidative phosphorylation deficiency"},
            {"term": "BN-PAGE",               "definition": "Blue-native polyacrylamide gel electrophoresis; N-module sub-assembly intermediates in NDUFA7-Leigh; compare with absent CI band in membrane-arm subunit deficiencies"},
            {"term": "Metformin CI",          "definition": "Metformin directly inhibits CI at the ND1/quinone interface → CI-Leigh fatal lactic acidosis; absolute contraindication in any CI-Leigh including NDUFA7"},
            {"term": "Succinate bypass",      "definition": "Succinate → SDHA → ubiquinol → CIII: bypasses NDUFA7-failed CI entirely; only ETC substrate entering ubiquinol pool not requiring CI function"},
            {"term": "Leigh syndrome",        "definition": "Progressive necrotising encephalopathy of childhood; bilateral symmetric brainstem + basal ganglia MRI lesions; caused by ≥100 nuclear/mtDNA gene mutations; NDUFA7 → CI-Leigh subtype"},
            {"term": "GIR 6–8",               "definition": "Glucose infusion rate 6–8 mg/kg/min; mandatory during nil-by-mouth/peri-operative in CI-Leigh; prevents fasting-induced ETC substrate starvation and metabolic crisis"},
            {"term": "AR biallelic",          "definition": "Autosomal recessive; two pathogenic variants in trans (compound het) or homozygous; both parents obligate carriers; 25% recurrence per pregnancy; 19p13.3"},
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
