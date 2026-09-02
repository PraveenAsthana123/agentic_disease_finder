#!/usr/bin/env python3
"""NDUFA5 — Leigh Syndrome Isolated Complex I Deficiency (B13 / Q-module N/Q Interface, AR).

NDUFA5 (NADH:Ubiquinone Oxidoreductase Subunit A5) is a ~116-aa nuclear-encoded
structural subunit of Complex I (~13.5 kDa after MTS cleavage), designated B13
(bovine CI proteomics; Carroll 2006). NDUFA5 occupies the Q-module peripheral arm
at the N-module/Q-module (N/Q) interface of the matrix arm, adopting a beta-sandwich
fold to stabilise the NDUFS3–NDUFS2 subcomplex scaffold. Unlike NDUFA3 (B9, PP-module
peripheral scaffold, 19q13.42) and NDUFA1 (MWFE, PP-module ND3 face, Xq24), NDUFA5
(B13) is exclusively a hydrophilic matrix arm subunit with NO transmembrane helix.
NDUFA5 is encoded on chromosome 7q32.1 (OMIM *603835) — autosomal recessive.

  NDUFA5 gene   OMIM *603835
  Disease       Leigh Syndrome (OMIM #256000); Isolated Complex I Deficiency
  Inheritance   Autosomal Recessive (AR) — biallelic pathogenic variants
  Chromosome    7q32.1

PATHOPHYSIOLOGY (Complex I / Q-module / NDUFA5 / B13 / N-Q Interface Beta-Sandwich):
  NDUFA5 (B13) is a peripheral structural subunit (~116 aa, ~13.5 kDa) with a
  beta-sandwich fold anchoring the N/Q-module interface of the peripheral arm.
  It directly contacts NDUFS3 (30kDa, Q-module) and NDUFS2 (49kDa, Q-module),
  stabilising the NDUFS3–NDUFS2 peripheral arm scaffold. Loss of NDUFA5
  destabilises Q-module N/Q-interface scaffold → absent or severely reduced CI
  holocomplex on BN-PAGE. Sub-assembly intermediates may include partial N/Q-arm
  fragments. Isolated CI deficiency 5–20%; CII/CIII/CIV activities NORMAL.

  UNIQUE MOLECULAR SIGNATURE — B13 / BETA-SANDWICH / Q-MODULE N/Q INTERFACE:
    NDUFA5 (B13) is the only NDUFA-subfamily subunit with a beta-sandwich fold
    anchoring the N/Q-module interface within the matrix arm of Complex I. This
    distinguishes it from all other NDUFA subunits: NDUFA1 (MWFE, single TM,
    PP-module ND3 face), NDUFA3 (B9, alpha-helical peripheral scaffold, PP-module
    ND3/ND4L boundary), and NDUFA11 (B14.7, 4-TM helices, PP-PD boundary).
    NDUFA5 (B13) is a pure hydrophilic matrix arm subunit — no TM helix, no
    membrane contact — exclusively residing in the peripheral Q-module arm.
    The B13 designation (bovine CI proteomics) is unique; not to be confused
    with NDUFA13 (B16.6, different NDUFA subunit, different module).

  Q-MODULE N/Q INTERFACE STRUCTURAL ROLE:
    NDUFA5 (B13) adopts a beta-sandwich fold that bridges the N-module/Q-module
    junction in the peripheral matrix arm. It provides structural scaffolding for
    the NDUFS3 (30kDa) and NDUFS2 (49kDa) iron-sulfur-containing subunits that
    form the quinone-binding tunnel. Loss of NDUFA5 disrupts Q-module N/Q-interface
    integrity → unstable NDUFS3–NDUFS2 subcomplex → impaired quinone-binding
    tunnel → NADH-to-ubiquinone electron transfer failure → CI absent on BN-PAGE.
    Unlike N-module sub-assembly defects (NDUFS1, NDUFS4, NDUFV1), NDUFA5 loss
    primarily affects Q-module peripheral scaffolding at the matrix arm N/Q boundary.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa Fe-S):
    • NO peripheral neuropathy in NDUFA5 (NDUFS1: ~50% — CRITICAL DDx)
    • NDUFS1 is an N-module catalytic iron-sulfur subunit; NDUFA5 is Q-module structural
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA5 (NDUFS4: 52–65% — near-pathognomonic)
  vs NDUFV1 (N-module FMN/N3 Fe-S):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b 2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFA3 (B9, PP-module, 19q13.42, alpha-helical, AR):
    • NDUFA5 (B13) is Q-module matrix arm (7q32.1); NDUFA3 is PP-module membrane
      arm peripheral scaffold (19q13.42) — different structural zones (matrix arm
      Q-module beta-sandwich vs membrane arm PP-module alpha-helical scaffold).
      WES chromosomal locus (7q32.1 vs 19q13.42) is definitive.
  vs NDUFA1 (MWFE, PP-module ND3 face, Xq24, X-LINKED):
    • NDUFA5 is AUTOSOMAL (7q32.1, AR biallelic); NDUFA1 is X-LINKED (Xq24) —
      inheritance pattern critical for genetic counselling. Both AR vs X-linked
      inheritance completely alters family counselling requirements.
  vs NDUFS2 / NDUFS3 (Q-module catalytic core subunits):
    • NDUFA5 is a PERIPHERAL SCAFFOLDING subunit contacting NDUFS2/NDUFS3;
      NDUFS2 and NDUFS3 carry the iron-sulfur clusters N2 and N3 respectively —
      loss of NDUFA5 destabilises the NDUFS3-NDUFS2 complex indirectly vs direct
      loss of Fe-S prosthetic groups in NDUFS2/NDUFS3 deficiencies.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFA5 (POLG: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  p.Arg91Trp   c.271C>T   — beta-strand core; N/Q-interface contact surface; severe infantile
  p.Leu78Pro   c.233T>C   — helix-breaking proline; beta-sandwich strand disruption; severe
  p.Glu53Lys   c.157G>A   — near MTS cleavage; targeting/import stability; severe neonatal
  p.Ala98Val   c.293C>T   — peripheral scaffold beta-sandwich core packing; intermediate
  c.IVS2+1G>A             — splice donor exon 2; partial CI residual (~10–20%); moderate

THERAPY — NDUFA5 / CI-LEIGH SPECIFICS:
  ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
    Metformin      — directly inhibits CI at ND1/quinone-binding site (Q-module territory)
    Valproate      — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
    Linezolid      — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
    Chloramphenicol — same mitochondrial ribosomal mechanism as linezolid
  CONTRAINDICATED:
    Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                     (NDUFA5 Q-module N/Q-interface scaffold collapsed, CI failed)
  AVOID / HIGH CAUTION:
    Propofol       — PRIS + secondary CIV inhibition (dual ETC bottleneck)
    Phenobarbital  — secondary CI inhibitor; use LEV first
  LEVEL C cofactors (standard CI supportive):
    Riboflavin (B2)   — CI-specific; FMN prosthetic group at NDUFV1 N-module (upstream)
    CoQ10 (ubiquinol) — electron acceptor at quinone site (CI membrane arm/Q-module interface)
    Thiamine (B1)     — MANDATORY empiric: SLC19A3/BTD mimic (treatable CI-mimic)
    Biotin            — MANDATORY empiric: BTD mimic (treatable CI-mimic)
    Succinate         — CII bypass; bypasses NDUFA5-failed CI entirely; donates ubiquinol via SDHA
    L-Carnitine       — energy metabolism support; secondary transport facilitation
"""

import random, json

GENE     = "NDUFA5"
DISEASE  = "Leigh Syndrome — Isolated Complex I Deficiency (CI-Leigh)"
OMIM_G   = "603835"
OMIM_D   = "256000"
INHERIT  = "Autosomal Recessive (AR) — biallelic"
CHROM    = "7q32.1"
MODULE   = "Q-module (N/Q Interface, Matrix Arm, Beta-Sandwich)"
SIZE     = "116 aa / 13.5 kDa (after MTS cleavage)"
SEED     = 657
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Severe infantile (onset <6 mo)",  35),
    ("Moderate infantile (onset 6–18 mo)", 38),
    ("Intermediate (onset 18–36 mo)",   17),
    ("Attenuated / partial CI residual", 10),
]

VARIANTS = [
    ("p.Arg91Trp",  "c.271C>T",  "Beta-strand core; N/Q-interface contact",       "Severe infantile",   33, "Beta-strand beta-sandwich core; NDUFS3/NDUFS2 N/Q-interface contact surface disruption"),
    ("p.Leu78Pro",  "c.233T>C",  "Proline disruption of beta-sandwich strand",      "Severe",             24, "Helix-breaking proline substitution; beta-sandwich tertiary structure collapse; Q-module scaffold loss"),
    ("p.Glu53Lys",  "c.157G>A",  "Near MTS cleavage; targeting/import",            "Severe neonatal",    15, "Adjacent to MTS cleavage site; protein mis-targeting or instability; neonatal CI failure"),
    ("p.Ala98Val",  "c.293C>T",  "Beta-sandwich core packing",                     "Intermediate",       14, "Peripheral scaffold beta-sandwich hydrophobic core packing disruption; partial CI assembly"),
    ("c.IVS2+1G>A", "Splice donor exon 2", "Partial CI residual (~10–20%)",        "Moderate / partial", 14, "Splice-donor loss; partial exon 2 skipping; some residual correctly spliced NDUFA5 transcript → partial CI"),
]

SEIZURE_TYPES = [
    ("Focal / multifocal (awake + sleep)", 58),
    ("Generalized tonic-clonic (GTCS)",    40),
    ("Myoclonic",                          28),
    ("Infantile spasms (IS / West synd.)", 18),
    ("Epileptic spasms (post-IS residual)", 12),
    ("Absence (atypical)",                  8),
]

TRIGGERS = [
    ("Febrile illness / infection",    78),
    ("Sub-therapeutic AED level",      65),
    ("Metabolic decompensation",       55),
    ("Sleep deprivation",              42),
    ("Missed AED dose",                38),
    ("Fasting / prolonged nil-by-mouth", 32),
    ("Anesthesia / surgical stress",   20),
    ("Enzyme-inducing co-medication",  14),
]

TREATMENTS = [
    ("Levetiracetam (LEV)",           "A",  "Preferred AED; renal excretion; NO mito toxicity; broad-spectrum CI-Leigh safe"),
    ("Riboflavin (B2 / FMN precursor)", "C", "CI-specific cofactor; FMN at NDUFV1 N-module upstream; 100–200 mg/day"),
    ("CoQ10 / Ubiquinol",             "C",  "Electron acceptor CI→CIII; downstream CI support; 10–30 mg/kg/day ubiquinol preferred"),
    ("Thiamine (B1)",                 "C",  "MANDATORY empiric: SLC19A3/BTD mimics treatable CI-Leigh; 100–300 mg/day before genetic result"),
    ("Biotin",                        "C",  "MANDATORY empiric: BTD deficiency mimics CI-Leigh; 10–40 mg/day empiric cover"),
    ("Succinate (oral/IV)",           "C",  "CII bypass; completely bypasses NDUFA5-failed CI; enters ubiquinol pool via SDHA-SDHB"),
    ("L-Carnitine",                   "C",  "Energy metabolism support; secondary transport; 50–100 mg/kg/day"),
    ("Clobazam (CLB) / Clonazepam",   "B",  "Focal / myoclonic adjunct; GABA-A positive modulator; avoid benzodiazepine overuse"),
]

CONTRAINDICATIONS = [
    ("Metformin", "ABSOLUTE", "Direct CI inhibitor at ND1/quinone-binding site (Q-module territory); causes fatal lactic acidosis"),
    ("Valproic acid / VPA", "ABSOLUTE", "Triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block; fatal hepatotoxicity risk"),
    ("Linezolid", "ABSOLUTE", "23S rRNA inhibition → blocks synthesis of all 7 mt-encoded ND subunits; CI depletion cascade"),
    ("Chloramphenicol", "ABSOLUTE", "Same 23S rRNA mitoribosomal mechanism as linezolid; avoidable antibiotic"),
    ("Ketogenic diet (KD)", "CONTRAINDICATED", "Forces β-oxidation NADH dependence; NDUFA5-Q-module-failed CI cannot reoxidise NADH; fatal metabolic crisis"),
    ("Propofol", "AVOID (PRIS)", "PRIS + secondary CIV inhibition; dual ETC bottleneck; use sevoflurane instead for anaesthesia"),
    ("Phenobarbital", "HIGH CAUTION", "Secondary CI inhibitor; acceptable only if LEV/CLB fail; monitor lactate"),
    ("Enzyme-inducing AEDs (CBZ/PHT/OXC)", "RELATIVE CI", "Secondary mitochondrial toxicity; induce CYP450 → accelerate cofactor depletion; avoid if alternatives exist"),
]

MONITORING = [
    ("Serum lactate + pyruvate (L:P ratio)", "At each visit; L:P >20 suggests CI dysfunction; target L:P <20"),
    ("AED levels (LEV, CLB, CNZ)", "Every 3–6 months; sub-therapeutic level = seizure trigger"),
    ("Riboflavin / CoQ10 status", "Annual; adjust supplementation dose"),
    ("Plasma amino acids", "6-monthly; alanine elevation = surrogate for lactic acidosis"),
    ("Neuroimaging MRI brain", "Every 12 months or at neurological change; Leigh lesion progression"),
    ("Echocardiography", "Annual; CI-Leigh rarely causes HCM but screen for onset"),
    ("Ophthalmology (visual acuity, ERG)", "Annual; pigmentary retinopathy rare in CI-Leigh, screen baseline"),
    ("Neurodevelopmental / cognitive battery", "Every 12 months; Bayley / VABS age-appropriate"),
    ("Respiratory function / polysomnography", "6-monthly; central apnoea and hypoventilation in CI-Leigh"),
    ("Renal function (eGFR, urine organic acids)", "Annual; rule out POLG/renal mimic; monitor drug clearance"),
    ("Pyruvate dehydrogenase (PDH) activity", "At diagnosis; PDH deficiency mimics CI-Leigh on lactate"),
    ("Mitochondrial respiratory chain enzyme panel", "Baseline + after any acute decompensation"),
]

REFERENCES = [
    "Carroll J et al. (2006) MolCell Proteomics — NDUFA5 B13 subunit identification in bovine CI proteome",
    "Guerrero-Castillo S et al. (2017) Cell Metab — CI assembly dynamics; Q-module N/Q-interface NDUFA5 incorporation",
    "Stroud DA et al. (2016) Nature — CI assembly states; matrix arm Q-module structural scaffold",
    "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure; Q-module NDUFS2/NDUFS3/NDUFA5 peripheral arm",
    "Fassone E & Rahman S (2012) J Med Genet — CI deficiency genetics; NDUFA5 subunit class review",
    "Zhu J et al. (2016) Science — Cryo-EM CI structure at 3.9Å; Q-module peripheral subunit map",
]

KEY_CONCEPTS = [
    ("B13 / Q-module N/Q Interface", "NDUFA5 (B13) is the sole NDUFA-subfamily subunit with a beta-sandwich fold at the N-module/Q-module matrix arm interface; no TM helix"),
    ("NDUFA5 vs NDUFA3 (B9)", "NDUFA5 = matrix arm Q-module (7q32.1); NDUFA3 = membrane arm PP-module peripheral scaffold (19q13.42); same NDUFA family, different CI zones"),
    ("NDUFA5 vs NDUFS2/NDUFS3", "NDUFA5 is a PERIPHERAL scaffold subunit (beta-sandwich); NDUFS2/NDUFS3 carry the N2/N3 Fe-S clusters — loss of NDUFA5 destabilises their complex"),
    ("Isolated CI deficiency pattern", "5–20% CI activity; CII/CIII/CIV NORMAL — biochemical fingerprint mandatory for diagnosis; excludes Complex IV (SURF1/SCO2) and mtDNA depletion"),
    ("BN-PAGE pattern", "Absent CI holocomplex on BN-PAGE (severe alleles); partial Q-module intermediates may appear (unlike clean PP-module absence of NDUFA3)"),
    ("NDUFS3-NDUFS2 scaffold dependence", "NDUFA5 loss → NDUFS3-NDUFS2 subcomplex destabilisation → quinone-binding tunnel impaired → NADH-to-ubiquinone electron transfer abolished"),
    ("Metformin absolute CI", "Metformin directly inhibits CI at the ND1/quinone interface (Q-module territory) — administration to any CI-Leigh patient is fatal"),
    ("Thiamine+Biotin empiric mandatory", "SLC19A3 deficiency and BTD deficiency both mimic CI-Leigh clinically; empiric thiamine + biotin BEFORE genetic result can prevent irreversible damage"),
    ("Succinate CII bypass", "Succinate → SDHA → ubiquinol: bypasses NDUFA5-failed CI entirely; sole ETC substrate that does NOT require CI for entry"),
    ("No NDUFS4 olfactory bulb lesions", "Bilateral olfactory bulb MRI lesions (52–65%) are near-pathognomonic for NDUFS4-Leigh; absence in NDUFA5 is a critical DDx pivot"),
    ("No HCM in pure CI-Leigh", "Hypertrophic cardiomyopathy: NDUFV2 (~80%) and SCO2 (~100%); NDUFA5 CI-Leigh is almost never associated with HCM — cardiac DDx pivot"),
    ("Sevoflurane not propofol", "General anaesthesia: sevoflurane (inhalational) preferred; propofol is AVOIDED (PRIS + CIV secondary inhibition in context of CI failure)"),
    ("7q32.1 locus", "NDUFA5 maps to 7q32.1; not near any other major CI gene on chromosome 7; WES locus confirmation required alongside biochemistry"),
    ("GIR 6–8 IV dextrose", "Never fast NDUFA5-Leigh children; glucose infusion rate 6–8 mg/kg/min during any nil-by-mouth period to prevent fasting-triggered metabolic crisis"),
    ("Genetic counselling AR", "Autosomal recessive: both parents are obligate carriers (25% recurrence risk per pregnancy); offer cascade carrier testing and prenatal/preimplantation options"),
]

# ── patient cohort generator ───────────────────────────────────────────────────
def _make_patients():
    pats = []
    for i in range(1, N + 1):
        # phenotype class
        r = rng.random() * 100
        cls = PHENO_CLASSES[0][0] if r < 35 else PHENO_CLASSES[1][0] if r < 73 else PHENO_CLASSES[2][0] if r < 90 else PHENO_CLASSES[3][0]
        # variant
        v = rng.choice(VARIANTS)
        age_mo = rng.randint(1, 6) if "Severe" in cls else rng.randint(6, 18) if "Moderate" in cls else rng.randint(18, 36) if "Intermediate" in cls else rng.randint(24, 60)
        pats.append({
            "id":          f"P{i:02d}",
            "phenotype":   cls,
            "onset_mo":    age_mo,
            "variant":     v[0],
            "cDNA":        v[1],
            "ci_pct":      rng.randint(5, 20),
            "has_seizure": rng.random() < 0.68,
            "has_hypotonia": rng.random() < 0.85,
            "has_lactic_acidosis": rng.random() < 0.88,
            "has_leigh_mri": rng.random() < 0.82,
            "has_respiratory_compromise": rng.random() < 0.42,
            "has_dystonia": rng.random() < 0.35,
            "has_ataxia":  rng.random() < 0.30,
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
            "mean_ci_pct":         round(sum(p["ci_pct"] for p in pts) / N, 1),
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
            "variant":        vname,
            "cDNA":           cdna,
            "structural_impact": struct,
            "modal_phenotype": phenotype_modal,
            "freq_pct":       freq,
            "n_in_cohort":    len(vpts),
            "detail":         detail,
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
        "monitoring":  [
            {"parameter": m[0], "protocol": m[1]}
            for m in MONITORING
        ],
        "patients": pts,
    }


def get_definitions():
    return {
        "gene":       GENE,
        "disease":    DISEASE,
        "omim_gene":  OMIM_G,
        "omim_disease": OMIM_D,
        "key_concepts": [{"concept": c[0], "detail": c[1]} for c in KEY_CONCEPTS],
        "glossary": [
            {"term": "B13",            "definition": "Bovine CI proteomics designation for NDUFA5 (Carroll 2006); unique identifier; not to be confused with NDUFA13 (B16.6)"},
            {"term": "Q-module",       "definition": "Quinone-binding module; distal part of peripheral matrix arm of CI; contains NDUFS2, NDUFS3, NDUFS7, NDUFS8, NDUFA5"},
            {"term": "N/Q interface",  "definition": "Junction between N-module (FMN/Fe-S, NDUFV1/V2) and Q-module (quinone-binding, NDUFS2/S3) within the matrix arm; NDUFA5 (B13) peripheral scaffold"},
            {"term": "Beta-sandwich",  "definition": "Two antiparallel beta-sheets facing each other; NDUFA5 structural fold; provides rigid scaffolding at N/Q-module interface"},
            {"term": "CI biochemical fingerprint", "definition": "Isolated CI ↓5–20%; CII, CIII, CIV NORMAL; distinguishes nuclear CI deficiency from mtDNA depletion (all-ETC-low) and Combined oxidative phosphorylation deficiency"},
            {"term": "BN-PAGE",        "definition": "Blue-native polyacrylamide gel electrophoresis; separates intact respiratory chain complexes; absent CI band = NDUFA5-Leigh severe allele; partial Q-module intermediates in moderate alleles"},
            {"term": "Metformin CI",   "definition": "Metformin directly inhibits the Q-module quinone-binding site (ND1/ubiquinone interface) → CI-Leigh fatal lactic acidosis; absolute contraindication"},
            {"term": "Succinate bypass", "definition": "Oral/IV succinate → SDHA → ubiquinol → CIII; completely bypasses NDUFA5-failed CI; only substrate entering ubiquinol pool that does NOT require CI"},
            {"term": "Leigh syndrome", "definition": "Progressive necrotising encephalopathy of childhood; bilateral symmetric brainstem + basal ganglia MRI lesions; caused by ≥100 nuclear/mtDNA gene mutations; NDUFA5 → CI-Leigh subtype"},
            {"term": "NDUFS3-NDUFS2 subcomplex", "definition": "Core Q-module peripheral arm scaffold containing N2 (NDUFS2) and N3 (NDUFS3) iron-sulfur clusters; NDUFA5 (B13) stabilises this subcomplex at N/Q boundary"},
            {"term": "GIR 6–8",        "definition": "Glucose infusion rate 6–8 mg/kg/min; mandatory during nil-by-mouth/peri-operative in CI-Leigh; prevents fasting-induced ETC substrate starvation and metabolic crisis"},
            {"term": "AR biallelic",   "definition": "Autosomal recessive; two pathogenic variants in trans (compound het) or homozygous; both parents obligate carriers; 25% recurrence per pregnancy"},
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
