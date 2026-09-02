#!/usr/bin/env python3
"""NDUFA8 — Leigh Syndrome Isolated Complex I Deficiency (B14.5b / N-Q Module Boundary, AR).

NDUFA8 (NADH:Ubiquinone Oxidoreductase Subunit A8) is a ~172-aa nuclear-encoded
structural subunit of Complex I (~20.3 kDa after MTS cleavage), designated B14.5b
(bovine CI proteomics; Carroll 2006). NDUFA8 occupies the N-Q MODULE BOUNDARY of
the matrix arm, contacting NDUFS3 (QP-C / 30kDa scaffold — the platform that
positions NDUFS2/N2 and NDUFA9 at the quinone-binding arm junction). NDUFA8
carries NO Fe-S cluster itself — it is a purely structural stabiliser of the
N-Q interface scaffold. NDUFA8 is located on chromosome 9q33.2, distinct from
its closest naming-alike NDUFA7 (B14.5a) at 19p13.3 — critical DDx naming trap:
both are "B14.5x" but different chromosomes, different CI module zones.
NDUFA8 is encoded on chromosome 9q33.2 (OMIM *603649) — autosomal recessive.

  NDUFA8 gene   OMIM *603649
  Disease       Leigh Syndrome (OMIM #256000); Isolated Complex I Deficiency
  Inheritance   Autosomal Recessive (AR) — biallelic pathogenic variants
  Chromosome    9q33.2

PATHOPHYSIOLOGY (Complex I / N-Q Boundary / NDUFA8 / B14.5b / NDUFS3-Contact Scaffold):
  NDUFA8 (B14.5b) is a peripheral structural subunit (~172 aa, ~20.3 kDa) with no
  TM helix, residing at the N-Q MODULE BOUNDARY of the matrix arm where it contacts
  NDUFS3 (QP-C, 30kDa) — the scaffold that positions NDUFS2 (49kDa, N2-4Fe4S
  terminal Fe-S cluster) and NDUFA9 (ETFDH-like, Q-module platform). NDUFA8
  stabilises the NDUFS3 peripheral face at the junction where the N-module hands
  off electrons (NDUFS1-N1b→N4→N5→NDUFS7-N4→NDUFS8-N6a/N6b→NDUFS2-N2→UQ) to
  the Q-module. NDUFA8 has no Fe-S cluster of its own.
  Loss of NDUFA8 destabilises NDUFS3 (QP-C scaffold) → N-Q junction impaired →
  NDUFS2-N2 positioning lost → terminal electron-relay to ubiquinone compromised
  → CI absent or severely reduced on BN-PAGE with N-Q sub-assembly intermediates.
  Isolated CI deficiency 5–20%; CII/CIII/CIV activities NORMAL.

  UNIQUE MOLECULAR SIGNATURE — B14.5b / N-Q BOUNDARY / NDUFS3-CONTACT / 9q33.2:
    NDUFA8 (B14.5b) is the counterpart of NDUFA7 (B14.5a): both ~20.3 kDa, both
    N-module–adjacent structural subunits, both purely scaffold (no Fe-S, no TM).
    NDUFA8 uniquely contacts NDUFS3 (QP-C / 30kDa scaffold) at the N-Q boundary
    — the structural hinge between the matrix arm N-module (NADH oxidation / Fe-S
    relay) and Q-module (quinone reduction). This makes NDUFA8 distinct from every
    other NDUFA-series peripheral subunit:
      NDUFA7 (B14.5a): NDUFS1 (IP1/75kDa Fe-S hub) contact — pure N-module core
      NDUFA8 (B14.5b): NDUFS3 (QP-C 30kDa scaffold) contact — N-Q boundary zone
      NDUFA12 (B17.2): NDUFAF2-paralog — N-Q interface, assembly-factor swap
      NDUFA5 (B13):    Beta-sandwich fold — Q-module matrix arm N-Q interface
    NDUFA8 is the ONLY NDUFA subunit stabilising the NDUFS3 peripheral face directly.

  CRITICAL NAMING TRAP — NDUFA8 (B14.5b) vs NDUFA7 (B14.5a) — B14.5 PREFIX SHARED:
    NDUFA8 (B14.5b): 9q33.2  — N-Q boundary peripheral (NDUFS3-contact, matrix arm)
    NDUFA7 (B14.5a): 19p13.3 — N-module peripheral (NDUFS1-contact, matrix arm core)
    Both designated "B14.5x" (bovine ~14.5 kDa gel band, but true ~20 kDa mature).
    Entirely different chromosomes (9q vs 19p), different CI module zones.
    WES must confirm gene symbol + chromosomal locus: never infer from "B14.5" alone.

  NDUFS3 SCAFFOLD DEPENDENCE — NDUFA8 N-Q BOUNDARY CONTACT:
    NDUFS3 (QP-C, 264aa, 30kDa) positions NDUFS2 (N2 4Fe4S terminal relay) and
    NDUFA9 (ETFDH-like Q-module assembly platform) at the quinone-access tunnel.
    NDUFA8 (B14.5b) stabilises the NDUFS3 peripheral face. Loss of NDUFA8 →
    NDUFS3 sub-assembly destabilised → N-Q boundary collapses → both N-module
    and Q-module intermediates visible on BN-PAGE (broader stall than NDUFA7's
    pure N-module accumulation). NDUFS3 itself (11p11.11) causes a distinct CI
    assembly failure if lost directly (NDUFS3-LOF → Q-module collapse, not N-Q
    peripheral scaffold). NDUFA8 destabilises NDUFS3 from the outside, not inside.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFA7 (B14.5a, N-module NDUFS1-contact, 19p13.3):
    NDUFA8 = N-Q BOUNDARY (NDUFS3 contact, 9q33.2); NDUFA7 = N-MODULE CORE (NDUFS1
    contact, 19p13.3). Same "B14.5" prefix — entirely different chromosomes and CI
    zones. WES chromosomal locus is the DEFINITIVE pivot (9q vs 19p).
  vs NDUFS3 (QP-C/30kDa scaffold, 11p11.11):
    NDUFS3 is the actual N-Q scaffold; NDUFA8 is its PERIPHERAL STABILISER.
    NDUFS3-LOF → full Q-module collapse; NDUFA8-LOF → NDUFS3 scaffold destabilised
    from peripheral face → N-Q boundary failure. Different loci (9q33.2 vs 11p11.11).
  vs NDUFS1 (N-module IP1/75kDa Fe-S, 2q33.3):
    NO peripheral neuropathy in NDUFA8 (NDUFS1: ~50% — CRITICAL DDx)
    NDUFS1 is catalytic with Fe-S clusters; NDUFA8 is structural N-Q scaffold
  vs NDUFS4 (N-module accessory, 5q11.2):
    NO olfactory bulb MRI lesions in NDUFA8 (NDUFS4: 52–65% — near-pathognomonic)
  vs NDUFV1 (N-module FMN/N3 Fe-S, 11q13.2):
    NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b 2Fe2S) / SCO2 (CIV):
    NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFA5 (B13, Q-module N-Q interface, beta-sandwich, 7q32.1):
    NDUFA8 = N-Q boundary from N-module side (NDUFS3 contact); NDUFA5 = Q-module
    N-Q interface from Q-module side (NDUFS3–NDUFS2 subcomplex platform, 7q32.1).
    Same CI junction zone, opposite sides; WES locus (9q vs 7q) is definitive.
  vs POLG/DGUOK (mtDNA depletion):
    NO hepatopathy in NDUFA8 (POLG: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  No common founder variant reported; sporadic/consanguineous families documented
  with homozygous LOF (splice donor loss) and compound heterozygous missense.

THERAPEUTIC NOTES (Inherited CI-Leigh — NDUFA8-specific):
    VPA                — ABSOLUTE CI (triple mechanism; CoA + POLG + ND-subunit expression)
    Metformin          — ABSOLUTE CI (direct CI inhibitor at ND1/Q-module territory)
    Linezolid          — ABSOLUTE CI (23S rRNA → blocks all 7 mtDNA-encoded ND subunits)
    Chloramphenicol    — ABSOLUTE CI (same mechanism as linezolid; avoidable)
    KD                 — CONTRAINDICATED (forces β-oxidation NADH → CI cannot reoxidise)
    Propofol           — AVOID (PRIS + secondary CIV inhibition → dual ETC bottleneck)
    Phenobarbital      — HIGH CAUTION (secondary CI inhibitor; use LEV first)
    Riboflavin (B2)    — CI-specific; FMN at NDUFV1 N-module (upstream of NDUFA8 N-Q zone)
    CoQ10 (ubiquinol)  — electron acceptor at quinone site (CI membrane arm/Q-module)
    Thiamine (B1)      — MANDATORY empiric: SLC19A3/BTD mimics treatable CI-Leigh
    Biotin             — MANDATORY empiric: BTD deficiency mimics CI-Leigh
    Succinate          — CII bypass; bypasses NDUFA8-failed CI entirely; via SDHA → ubiquinol
    L-Carnitine        — energy metabolism support; secondary transport facilitation
"""

import random, json

GENE     = "NDUFA8"
DISEASE  = "Leigh Syndrome — Isolated Complex I Deficiency (CI-Leigh)"
OMIM_G   = "603649"
OMIM_D   = "256000"
INHERIT  = "Autosomal Recessive (AR) — biallelic"
CHROM    = "9q33.2"
MODULE   = "N-Q Module Boundary (Peripheral NDUFS3-Contact Stabiliser, No TM Helix, No Fe-S)"
SIZE     = "172 aa / 20.3 kDa (after MTS cleavage)"
SEED     = 663
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Severe infantile (onset <6 mo)",       34),
    ("Moderate infantile (onset 6–18 mo)",   38),
    ("Intermediate (onset 18–36 mo)",        18),
    ("Attenuated / partial CI residual",     10),
]

VARIANTS = [
    ("p.Arg127Trp",  "c.379C>T",  "N-Q boundary NDUFS3 contact surface; peripheral scaffold disrupted",  "Severe infantile",   34, "Arginine-to-tryptophan at NDUFS3 (QP-C/30kDa) contact surface; disrupts critical NDUFA8–NDUFS3 peripheral interface; N-Q boundary scaffold collapse; severe CI absence"),
    ("p.Leu74Pro",   "c.221T>C",  "Helix-breaking proline in α-helix; N-Q boundary fold disruption",     "Severe",             24, "Leucine-to-proline substitution in peripheral arm α-helix; proline cannot participate in backbone H-bond → NDUFA8 fold lost → NDUFS3 peripheral contact abolished"),
    ("p.Glu38Lys",   "c.112G>A",  "Near MTS cleavage; import/targeting disruption",                      "Severe neonatal",    15, "Glutamate-to-lysine in MTS-proximal region; protein mis-targeting or import failure; neonatal CI absence and metabolic collapse"),
    ("p.Ala135Val",  "c.404C>T",  "N-Q boundary core packing; intermediate phenotype",                   "Intermediate",       13, "Alanine-to-valine in N-Q boundary core hydrophobic region; partial NDUFA8 stability; some residual NDUFS3 contact → intermediate N-Q assembly with partial CI"),
    ("c.IVS2+1G>A",  "Splice donor exon 2", "Partial CI residual (~10–20%)",                            "Moderate / partial", 14, "Splice-donor loss; partial exon 2 skipping; some residual correctly spliced NDUFA8 transcript → partial N-Q sub-assembly intermediates on BN-PAGE"),
]

SEIZURE_TYPES = [
    ("Focal / multifocal (awake + sleep)",    55),
    ("Generalized tonic-clonic (GTCS)",       38),
    ("Myoclonic",                             27),
    ("Infantile spasms (IS / West synd.)",    18),
    ("Epileptic spasms (post-IS residual)",   12),
    ("Absence (atypical)",                     7),
]

TRIGGERS = [
    ("Febrile illness / infection",           80),
    ("Sub-therapeutic AED level",             58),
    ("Metabolic decompensation",              52),
    ("Sleep deprivation",                     37),
    ("Missed AED dose",                       34),
    ("Fasting / prolonged nil-by-mouth",      30),
    ("Anesthesia / surgical stress",          20),
    ("Enzyme-inducing co-medication",         12),
]

TREATMENTS = [
    ("Levetiracetam (LEV)",             "A",  "Preferred AED; renal excretion; NO mito toxicity; broad-spectrum CI-Leigh safe"),
    ("Riboflavin (B2 / FMN precursor)", "C",  "CI-specific cofactor; FMN at NDUFV1 N-module — upstream of NDUFA8 N-Q zone; 100–200 mg/day"),
    ("CoQ10 / Ubiquinol",               "C",  "Electron acceptor CI→CIII; downstream N-Q module support; 10–30 mg/kg/day ubiquinol preferred"),
    ("Thiamine (B1)",                   "C",  "MANDATORY empiric: SLC19A3/BTD mimics treatable CI-Leigh; 100–300 mg/day before genetic result"),
    ("Biotin",                          "C",  "MANDATORY empiric: BTD deficiency mimics CI-Leigh; 10–40 mg/day empiric cover"),
    ("Succinate (oral/IV)",             "C",  "CII bypass; completely bypasses NDUFA8-failed CI; enters ubiquinol pool via SDHA-SDHB; distal to N-Q failure"),
    ("L-Carnitine",                     "C",  "Energy metabolism support; secondary transport; 50–100 mg/kg/day"),
    ("Clobazam (CLB) / Clonazepam",     "B",  "Focal / myoclonic adjunct; GABA-A positive modulator; avoid benzodiazepine overuse"),
]

CONTRAINDICATIONS = [
    ("Metformin",                        "ABSOLUTE",        "Direct CI inhibitor at ND1/quinone-binding site (Q-module territory); fatal lactic acidosis in CI-Leigh"),
    ("Valproic acid / VPA",              "ABSOLUTE",        "Triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block; fatal hepatotoxicity risk"),
    ("Linezolid",                        "ABSOLUTE",        "23S rRNA inhibition → blocks synthesis of all 7 mt-encoded ND subunits; CI depletion cascade"),
    ("Chloramphenicol",                  "ABSOLUTE",        "Same 23S rRNA mitoribosomal mechanism as linezolid; avoidable antibiotic alternative available"),
    ("Ketogenic diet (KD)",              "CONTRAINDICATED", "Forces β-oxidation NADH dependence; NDUFA8-N-Q-failed CI cannot reoxidise NADH; fatal metabolic crisis"),
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
    "Carroll J et al. (2006) Mol Cell Proteomics — NDUFA8 B14.5b subunit identification in bovine CI proteome",
    "Guerrero-Castillo S et al. (2017) Cell Metab — CI assembly dynamics; N-Q boundary NDUFA8 B14.5b peripheral incorporation",
    "Stroud DA et al. (2016) Nature — CI assembly states; N-Q boundary peripheral stabiliser scaffold",
    "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure; NDUFS3 QP-C scaffold; NDUFA8 B14.5b N-Q position",
    "Fassone E & Rahman S (2012) J Med Genet — CI deficiency genetics; NDUFA8 subunit class review",
    "Zhu J et al. (2016) Science — Cryo-EM CI structure at 3.9Å; N-Q boundary NDUFA8 B14.5b peripheral contact map",
]

KEY_CONCEPTS = [
    ("B14.5b / N-Q Boundary NDUFS3 Peripheral Stabiliser", "NDUFA8 (B14.5b) stabilises NDUFS3 (QP-C/30kDa — the N-Q scaffold that positions NDUFS2/N2 and NDUFA9) at the peripheral N-Q boundary face; no Fe-S cluster in NDUFA8 itself; purely structural"),
    ("NDUFA8 (B14.5b) vs NDUFA7 (B14.5a) — 9q33.2 vs 19p13.3 — SAME B14.5 PREFIX, DIFFERENT chromosomes", "NDUFA8 (B14.5b): 9q33.2, N-Q boundary peripheral (NDUFS3 contact); NDUFA7 (B14.5a): 19p13.3, N-module core peripheral (NDUFS1 contact). Both '14.5' in name — entirely different chromosomes and CI module zones. WES gene symbol + chromosomal locus mandatory"),
    ("NDUFA8 (B14.5b, 9q33.2) vs NDUFA5 (B13, Q-module N-Q interface, 7q32.1) — N-Q junction, opposite sides", "NDUFA8 stabilises NDUFS3 from the N-module side of the N-Q boundary; NDUFA5 (B13, beta-sandwich) operates at the Q-module side of the N-Q interface (NDUFS3–NDUFS2 subcomplex platform, 7q32.1). Same junction, opposite faces; WES locus (9q vs 7q) definitive"),
    ("NDUFA8 vs NDUFS3 — peripheral stabiliser vs scaffold itself", "NDUFS3 (QP-C, 11p11.11) IS the N-Q scaffold; NDUFA8 (B14.5b, 9q33.2) STABILISES NDUFS3 from the peripheral face. Loss of NDUFS3 → full Q-module scaffold collapse; loss of NDUFA8 → NDUFS3 peripheral destabilisation → N-Q boundary failure. Different genes, different loci"),
    ("N-Q boundary BN-PAGE pattern — N-module + Q-module sub-assembly intermediates", "NDUFA8-Leigh BN-PAGE shows N-Q boundary sub-assembly intermediates — both N-module and early Q-module accumulations (broader stall than NDUFA7's pure N-module pattern); CONTRAST with cleaner absent CI in membrane-arm subunit losses (NDUFA1, NDUFA3, NDUFA11)"),
    ("No Fe-S cluster in NDUFA8", "NDUFA8 (B14.5b) carries no Fe-S cluster — pure structural N-Q scaffold stabiliser. DDx from catalytic subunits (NDUFS1/NDUFV1/NDUFV2 carry Fe-S/FMN; NDUFS2 carries N2 terminal Fe-S): loss of NDUFA8 → N-Q scaffold failure; not direct electron-relay disruption"),
    ("Isolated CI deficiency pattern", "5–20% CI activity; CII/CIII/CIV NORMAL — biochemical fingerprint mandatory; excludes Complex IV (SURF1/SCO2) and mtDNA depletion (all-ETC-low)"),
    ("Metformin absolute CI", "Metformin directly inhibits CI at the ND1/quinone interface → CI-Leigh fatal lactic acidosis; absolute contraindication in any CI-Leigh including NDUFA8"),
    ("Thiamine + Biotin empiric MANDATORY", "SLC19A3 and BTD deficiencies mimic CI-Leigh clinically; empiric thiamine + biotin BEFORE genetic result can prevent irreversible neurological damage"),
    ("Succinate CII bypass", "Succinate → SDHA → ubiquinol → CIII: bypasses NDUFA8-failed CI N-Q boundary entirely; only ETC substrate entering ubiquinol pool without requiring CI function"),
    ("No NDUFS4 olfactory bulb lesions", "Bilateral olfactory bulb MRI lesions (52–65%) are near-pathognomonic for NDUFS4-Leigh; absence in NDUFA8-Leigh is a CRITICAL DDx pivot"),
    ("No HCM in pure CI-Leigh", "HCM: NDUFV2 (~80%) and SCO2 (~100%); NDUFA8 CI-Leigh almost never associated with HCM — cardiac DDx pivot against CIV/NDUFV2 diseases"),
    ("No peripheral neuropathy DDx", "Peripheral neuropathy present in ~50% of NDUFS1-Leigh (N-module catalytic); ABSENT in NDUFA8-Leigh (structural N-Q scaffold) — CRITICAL clinical DDx pivot before WES"),
    ("Sevoflurane not propofol", "General anaesthesia: sevoflurane (inhalational) preferred; propofol AVOIDED (PRIS + CIV secondary inhibition in context of CI failure)"),
    ("GIR 6–8 IV dextrose", "Never fast CI-Leigh children; glucose infusion rate 6–8 mg/kg/min during any nil-by-mouth period to prevent fasting-triggered metabolic crisis"),
    ("Genetic counselling AR", "Autosomal recessive (9q33.2); both parents obligate carriers (25% recurrence risk per pregnancy); offer cascade carrier testing and prenatal/preimplantation options"),
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
            "has_seizure":               rng.random() < 0.70,
            "has_hypotonia":             rng.random() < 0.82,
            "has_lactic_acidosis":       rng.random() < 0.88,
            "has_leigh_mri":             rng.random() < 0.83,
            "has_respiratory_compromise": rng.random() < 0.38,
            "has_dystonia":              rng.random() < 0.33,
            "has_ataxia":               rng.random() < 0.31,
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
            {"term": "B14.5b",                 "definition": "Bovine CI proteomics designation for NDUFA8 (Carroll 2006); counterpart of B14.5a (NDUFA7); both ~14.5 kDa in original gel, ~20 kDa mature; entirely different chromosomes (9q vs 19p) and CI module zones"},
            {"term": "N-Q boundary peripheral stabiliser", "definition": "Structural role of NDUFA8: contacts NDUFS3 (QP-C/30kDa) at the peripheral face of the N-Q module boundary; no Fe-S cluster; no catalytic function; purely scaffolding the junction between N-module (NADH oxidation) and Q-module (ubiquinone reduction)"},
            {"term": "NDUFS3 (QP-C / 30kDa)",  "definition": "N-Q scaffold subunit (264aa, 30kDa) that positions NDUFS2 (N2 4Fe4S terminal relay) and NDUFA9 (Q-module platform) at the quinone-access tunnel; stabilised by NDUFA8 peripherally"},
            {"term": "9q33.2 vs 19p13.3",      "definition": "NDUFA8 (B14.5b) maps to 9q33.2; NDUFA7 (B14.5a) maps to 19p13.3. Same 'B14.5' bovine proteomics designation — completely different chromosomes and CI module zones. WES must confirm gene symbol + locus"},
            {"term": "N-Q boundary sub-assembly intermediates", "definition": "BN-PAGE pattern in NDUFA8-Leigh: both N-module and early Q-module intermediates accumulate (broader stall than NDUFA7's pure N-module pattern); CONTRAST with cleaner absent CI in membrane-arm subunit losses (NDUFA1, NDUFA3, NDUFA11)"},
            {"term": "CI biochemical fingerprint", "definition": "Isolated CI ↓5–20%; CII, CIII, CIV NORMAL — biochemical fingerprint mandatory; excludes Complex IV (SURF1/SCO2) and mtDNA depletion (all-ETC-low)"},
            {"term": "BN-PAGE",               "definition": "Blue-native polyacrylamide gel electrophoresis; N-Q sub-assembly intermediates in NDUFA8-Leigh at the boundary region; compare with absent CI in membrane-arm subunit deficiencies"},
            {"term": "Metformin CI",          "definition": "Metformin directly inhibits CI at the ND1/quinone interface → CI-Leigh fatal lactic acidosis; absolute contraindication in any CI-Leigh including NDUFA8"},
            {"term": "Succinate bypass",      "definition": "Succinate → SDHA → ubiquinol → CIII: bypasses NDUFA8-failed CI entirely; only ETC substrate entering ubiquinol pool not requiring CI function"},
            {"term": "Leigh syndrome",        "definition": "Progressive necrotising encephalopathy of childhood; bilateral symmetric brainstem + basal ganglia MRI lesions; caused by ≥100 nuclear/mtDNA gene mutations; NDUFA8 → CI-Leigh subtype"},
            {"term": "GIR 6–8",               "definition": "Glucose infusion rate 6–8 mg/kg/min; mandatory during nil-by-mouth/peri-operative in CI-Leigh; prevents fasting-induced ETC substrate starvation and metabolic crisis"},
            {"term": "AR biallelic",          "definition": "Autosomal recessive; two pathogenic variants in trans (compound het) or homozygous; both parents obligate carriers; 25% recurrence per pregnancy; 9q33.2"},
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
