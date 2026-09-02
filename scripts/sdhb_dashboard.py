#!/usr/bin/env python3
"""SDHB — Succinate Dehydrogenase Subunit B (Iron-Sulfur Subunit) /
Paraganglioma 4 (PGL4) — THE malignancy SDH gene.

SDHB (Succinate Dehydrogenase Subunit B; OMIM *185470) encodes the 280-amino-acid,
~32 kDa iron-sulfur subunit of Complex II (succinate dehydrogenase, SDH). SDHB forms
the catalytic core with SDHA and contains three iron-sulfur (FeS) clusters that transfer
electrons from SDHA-bound FADH2 to ubiquinone via the SDHC/D membrane anchor subunits.

  SDHB gene     OMIM *185470
  Protein       Succinate dehydrogenase subunit B (iron-sulfur subunit)
  Size          280 aa, ~32 kDa
  Location      Mitochondrial matrix-facing, attached to IMM via SDHC/D
  Chromosome    1p36.13
  CII role      FeS electron transfer: FADH2 → [2Fe-2S] → [3Fe-4S] → [4Fe-4S] → ubiquinone

FeS Clusters (3 total):
  [2Fe-2S]   Cluster 1 (N1b): Cys70, Cys72, Cys75, Cys101 — proximal to SDHA-SDHB interface
  [3Fe-4S]   Cluster 2 (S2):  Cys148, Cys151, Cys185     — central electron relay
  [4Fe-4S]   Cluster 3 (S3):  Cys208, Cys211, Cys214, Cys217 — proximal to SDHC/D and ubiquinone

Disease: Paraganglioma 4 (PGL4) — OMIM #115310
  Inheritance   AD (autosomal dominant), NOT maternally imprinted
  Penetrance    ~25–35% by age 50 (intermediate: higher than SDHA ~10%, lower than SDHD ~80%)
  Also          Adrenal PCC, extra-adrenal PGL (thoracic/abdominal/pelvic),
                clear-cell/type-2 RCC (15%), SDH-deficient GIST (5%), pituitary adenoma (rare)

KEY CLINICAL FEATURES — SDHB-PGL4:
  MALIGNANCY 20–50%: BY FAR THE HIGHEST OF ALL SDH GENES. SDHB IS THE MALIGNANCY GENE.
  Extra-adrenal PGL predominant: thoracic, abdominal, pelvic — NOT head-neck (unlike SDHD).
  RCC (renal cell carcinoma) 15%: UNIQUE frequency among SDH genes.
  IHC: SDHB null ONLY (SDHA proficient) — unlike SDHA loss (dual SDHA+SDHB null).
  NOT maternally imprinted: biparental transmission (unlike SDHD-PGL1 and SDHAF2-PGL2).
  Sunitinib: best systemic evidence for metastatic SDHB-PGL/PCC (anti-VEGFR/PDGFR).
  Temozolomide: for SDHB tumors with MGMT promoter methylation/silencing.

Reference: Astuti D et al. (2001) Gene mutations in the succinate dehydrogenase subunit SDHB
cause susceptibility to familial phaeochromocytoma and to familial paraganglioma.
Am J Hum Genet 69(1):49–54.
(First report of SDHB germline mutations in PGL4; foundational hereditary PGL/PCC genetics)

Reference: Timmers HJ et al. (2009) New developments in the pathophysiology, diagnosis, and
treatment of pheochromocytoma and paraganglioma.
Clin Endocrinol (Oxf) 70(4):520–531.
(Comprehensive PGL/PCC review including SDHB clinical spectrum and management)

Reference: Jochmanová I et al. (2013) Hypoxia-inducible factor signaling in
pheochromocytoma: turning the rudder in the right direction.
J Natl Cancer Inst 105(17):1270–1283.
(SDH/HIF pseudo-hypoxia pathway; SDHB malignancy mechanism via HIF2α stabilization)

Reference: Crona J, Taieb D, Pacak K (2017) New Perspectives on Pheochromocytoma and
Paraganglioma: Toward a Molecular Classification.
Endocr Rev 38(6):489–515.
(Comprehensive SDHB malignancy meta-analysis; 20–50% malignancy figure well-established)

PATHOPHYSIOLOGY (SDHB — FeS electron chain in CII):

  SDHB in normal CII function:
    1. SDHAF1 delivers FeS clusters to SDHB (via HSC20/HSPA9 chaperone system)
    2. FeS-matured SDHB binds flavinylated SDHA (SDHAF2-flavinylated at His99) → SDHA-SDHB core
    3. SDHA-SDHB binds SDHC-SDHD membrane anchor subunits → CII holoenzyme inserted in IMM
    4. CII: succinate + FAD (SDHA) → fumarate + FADH2; electrons flow:
       FADH2 → [2Fe-2S] (Cys70/72/75/101) → [3Fe-4S] (Cys148/151/185) → [4Fe-4S] (Cys208-217) → ubiquinone
    5. SDHC/D anchor positions ubiquinone binding site adjacent to [4Fe-4S] cluster of SDHB

  SDHB loss-of-function (monoallelic, AD — PGL4):
    1. Heterozygous germline SDHB mutation → haploinsufficiency
    2. In susceptible chromaffin/paraganglionic or renal cells: somatic second-hit (LOH at 1p36.13)
    3. Complete SDHB loss → FeS electron relay blocked → CII inactive → succinate accumulates
    4. Succinate inhibits PHD (prolyl hydroxylase domain) enzymes in cytoplasm
    5. HIF1α/HIF2α not hydroxylated → not degraded by VHL → stabilized → oncogenic pseudo-hypoxia
    6. VEGF, EPO, angiogenic gene transcription → highly vascularized, aggressive paraganglioma
    7. HIGH MALIGNANCY (20–50%): mechanism poorly understood; aggressive pseudo-hypoxia cluster;
       epigenetic silencing; frequent lymph node/bone/liver/lung metastases
    8. NOT IMPRINTED: biparental transmission; maternal and paternal SDHB mutations equally penetrant
    9. PENETRANCE 25–35% by age 50: intermediate; lower than SDHD (80%) but far higher than SDHA (10%)

SDHB UNIQUE FEATURES:
  1. HIGHEST MALIGNANCY OF ALL SDH/PGL GENES: 20–50% in SDHB vs 3–5% SDHD, 5% SDHA,
     1–3% SDHC, 5% SDHAF2. SDHB = THE malignancy gene. Every SDHB tumor must be considered
     potentially malignant until proven otherwise. Surveillance for metastases MANDATORY.
  2. EXTRA-ADRENAL PGL PREDOMINANT: thoracic, abdominal, pelvic paraganglioma, NOT head-neck
     (unlike SDHD/PGL1 where head-neck predominates). Extra-adrenal PGLs have higher malignancy.
  3. RCC (RENAL CELL CARCINOMA) 15%: SDHB-associated RCC is unique among SDH genes for frequency.
     Clear-cell type 2 (ccRCC-T2) and oncocytic morphology. SDHB IHC null in RCC confirms germline.
  4. 1p36.13 LOCUS: same chromosome arm as NDUFV1 (1p33) but different locus. SDHC is at 1q23.3
     (same chromosome, opposite arm). WES mandatory for locus-specific diagnosis.
  5. IHC SDHB NULL ONLY (SDHA proficient): SDHB loss causes only SDHB null on IHC (secondary
     SDHB degradation). SDHA remains proficient (only SDHA loss causes dual SDHA+SDHB null IHC).
     IHC interpretation critical for cascade germline testing.
  6. NOT MATERNALLY IMPRINTED: biparental transmission (unlike SDHD/PGL1 and SDHAF2/PGL2).
     Both maternal and paternal SDHB mutations cause PGL4. Critical for genetic counselling.
  7. SUNITINIB: best systemic therapy evidence for SDHB-PGL/PCC (anti-VEGFR/PDGFR).
     SDHB-PGL highly vascularized (pseudo-hypoxia → VEGF); sunitinib targets VEGFR/PDGFR.
  8. TEMOZOLOMIDE: uniquely active in SDHB tumors with MGMT promoter methylation/silencing.
     Alkylating agent; must confirm MGMT status before use.

DISTINGUISHING FEATURES vs OTHER SDH/PGL GENES:
  vs SDHA (5p15.33): SDHA-PGL5 — malignancy 5% vs SDHB 20–50%. SDHA IHC: dual null (SDHA+SDHB).
    SDHA causes Leigh (AR biallelic); SDHB does NOT cause Leigh. SDHA penetrance 10% vs SDHB 25–35%.
  vs SDHC (1q23.3): SDHC-PGL3 — head-neck PGL predominant vs SDHB extra-adrenal; malignancy 1–3% vs 20–50%.
    Both on chromosome 1 but different arms (1q23.3 vs 1p36.13). SDHC does not cause RCC; SDHB does.
  vs SDHD (11q23.1): SDHD-PGL1 — maternal imprinting (paternal only), penetrance 80%, head-neck predominant.
    SDHB not imprinted, penetrance 25–35%, extra-adrenal predominant. Malignancy 3–5% SDHD vs 20–50% SDHB.
  vs SDHAF2 (11q13.1): SDHAF2-PGL2 — maternal imprinting, penetrance 85–92%, head-neck only.
    SDHB not imprinted, intermediate penetrance, extra-adrenal. Malignancy 5% SDHAF2 vs 20–50% SDHB.
  vs VHL (3p25.3): VHL hemangioblastoma (cerebellum/spine/retina) + ccRCC + PCC; no SDHB hemangioblastoma.
    VHL RCC frequency higher; SDHB RCC 15%. VHL IHC: SDHB proficient (VHL-null IHC). 3p vs 1p loci.
  vs NF1 (17q11.2): NF1 PCC usually benign/unilateral/adrenal; café-au-lait/neurofibromas/Lisch nodules absent in SDHB.
    NF1 no RCC association; SDHB RCC 15%. 17q vs 1p loci.
  vs RET-MEN2 (10q11.21): MEN2A: MTC + PCC + parathyroid; MEN2B: marfanoid + MTC + PCC.
    SDHB: PGL + RCC, NO MTC, NO parathyroid disease. RET 10q11.21 vs SDHB 1p36.13.
"""

import random
import math

SEED         = 707
rng          = random.Random(SEED)

GENE         = "SDHB"
OMIM_GENE    = "185470"
OMIM_DISEASE = "115310"    # Paraganglioma 4 (PGL4)
CHROMOSOME   = "1p36.13"
DISEASE_NAME = (
    "SDHB Succinate Dehydrogenase Subunit B (Iron-Sulfur Subunit) — Paraganglioma 4 (PGL4, "
    "OMIM #115310) — AD, NOT maternally imprinted, penetrance 25–35%, malignancy 20–50% "
    "(HIGHEST of all SDH genes); also adrenal PCC, RCC (15%), SDH-deficient GIST (5%)"
)
N_PATIENTS   = 40

# ─── Pathogenic variants (8 total) ───────────────────────────────────────────
ALL_VARIANTS = [
    {
        "hgvs_c":    "c.590C>G",
        "hgvs_p":    "p.Pro197Arg",
        "domain":    "Structural protein core — central beta-sheet packing",
        "severity_pct": 85,
        "mechanism": (
            "Proline-to-arginine at position 197 in the central structural core of SDHB. "
            "Proline 197 is buried in the hydrophobic core and its rigid pyrrolidine ring "
            "is critical for maintaining the correct tertiary fold of SDHB around the [3Fe-4S] "
            "cluster (Cys148, Cys151, Cys185 region). The large, positively charged arginine "
            "side chain creates steric and electrostatic disruption within the core, destabilizing "
            "the [3Fe-4S] coordinating scaffold. SDHB protein is partially misfolded, impairs "
            "FeS electron relay, and triggers secondary SDHB degradation detectable by IHC "
            "(SDHB null). Severe loss of CII activity → succinate-PHD-HIF2α pseudo-hypoxia."
        ),
        "severity":  "severe",
        "notes": "PGL4. Structural core disruption. Severe 85% activity loss. SDHB null IHC. Malignancy risk 20–50%.",
    },
    {
        "hgvs_c":    "c.302G>A",
        "hgvs_p":    "p.Cys101Tyr",
        "domain":    "[2Fe-2S] cluster ligand Cys101 — proximal FeS cluster coordination",
        "severity_pct": 90,
        "mechanism": (
            "Cysteine-to-tyrosine at position 101, one of the four cysteine ligands (Cys70, "
            "Cys72, Cys75, Cys101) coordinating the [2Fe-2S] cluster (N1b). The [2Fe-2S] "
            "cluster is the first electron acceptor in SDHB, accepting electrons from SDHA-FADH2 "
            "at the SDHA-SDHB interface. Loss of Cys101 thiolate coordination is catastrophic: "
            "the [2Fe-2S] cluster cannot be incorporated during FeS assembly, SDHB cannot fold "
            "correctly, and SDHAF1-mediated FeS delivery to SDHB is aborted. No electron relay "
            "possible from FADH2 through SDHB → complete CII block → severe succinate accumulation "
            "→ HIF2α stabilization → aggressive PGL4 with high malignancy risk."
        ),
        "severity":  "severe",
        "notes": "[2Fe-2S] ligand loss — catastrophic FeS assembly failure. Severe 90% loss. Highest malignancy association. SDHB null IHC.",
    },
    {
        "hgvs_c":    "c.215G>T",
        "hgvs_p":    "p.Gly72Val",
        "domain":    "[2Fe-2S] cluster proximity — Gly72 within FeS coordinating loop (Cys70-Cys75)",
        "severity_pct": 80,
        "mechanism": (
            "Glycine-to-valine at position 72, flanked by [2Fe-2S] coordinating cysteines Cys70 "
            "and Cys75. The conserved glycine at this position has zero side chain (allows the "
            "tight loop geometry required for FeS cluster coordination). Substitution with the "
            "bulky valine side chain sterically distorts the Cys70-Cys72-Cys75 coordinating loop, "
            "impairing [2Fe-2S] cluster geometry and electron transfer efficiency. Partial FeS "
            "assembly may occur, giving some residual CII activity (~20%), but FeS relay is "
            "compromised. In germline heterozygosity, somatic second-hit at 1p36.13 abolishes "
            "remaining function → full CII loss → PGL4."
        ),
        "severity":  "severe",
        "notes": "[2Fe-2S] loop distortion by bulky Val. Severe 80% activity loss. PGL4; extra-adrenal and head-neck PGL both reported.",
    },
    {
        "hgvs_c":    "c.137G>A",
        "hgvs_p":    "p.Arg46Gln",
        "domain":    "SDHB-SDHA interface — N-terminal surface charge",
        "severity_pct": 70,
        "mechanism": (
            "Arginine-to-glutamine at position 46 on the N-terminal surface of SDHB that contacts "
            "SDHA at the SDHA-SDHB interface. Arg46 makes electrostatic contacts with acidic "
            "residues on the SDHA surface (particularly near the SDHA C-terminal docking region). "
            "The glutamine substitution eliminates the positive charge, weakening SDHA-SDHB "
            "interface affinity and reducing CII assembly efficiency. Residual CII activity ~30% "
            "(heterozygous carrier); somatic LOH → complete SDHB loss. Moderately severe. "
            "Associated with PGL4 with somewhat lower malignancy fraction in reported series."
        ),
        "severity":  "moderately_severe",
        "notes": "SDHA-SDHB interface charge disruption. Moderately severe 70% activity loss. Lower (but still substantial) malignancy risk.",
    },
    {
        "hgvs_c":    "c.599C>G",
        "hgvs_p":    "p.Pro200Arg",
        "domain":    "Central helix — helix-breaking proline-to-arginine in SDHB core",
        "severity_pct": 82,
        "mechanism": (
            "Proline-to-arginine at position 200 in a central alpha-helix of SDHB. Prolines "
            "are often conserved structural turn/kink residues (not helix-interior); however, "
            "position 200 is at the N-terminal cap of a helix where proline's rigidity provides "
            "a defined entry geometry. Substituting arginine at a proline creates a helix-beginning "
            "disruption (proline kink replaced by flexible arginine that can adopt multiple rotamers), "
            "destabilizing the helix and the adjacent [3Fe-4S] cluster environment (Cys148, Cys151, "
            "Cys185 in close proximity). SDHB structural integrity severely compromised; degraded "
            "by mitochondrial quality control; SDHB null on IHC."
        ),
        "severity":  "severe",
        "notes": "Helix-proximal proline substitution. Severe 82% loss. Adjacent to [3Fe-4S] cluster environment. SDHB IHC null.",
    },
    {
        "hgvs_c":    "c.272G>C",
        "hgvs_p":    "p.Cys91Ser",
        "domain":    "[2Fe-2S] coordinating Cys91 — SDHB proximal cluster ligand",
        "severity_pct": 88,
        "mechanism": (
            "Cysteine-to-serine at position 91, within the [2Fe-2S] coordinating region of SDHB. "
            "While the canonical four-cysteine ligands of [2Fe-2S] are Cys70, Cys72, Cys75, and "
            "Cys101, Cys91 is part of the loop that scaffolds the cluster coordination geometry. "
            "Serine cannot coordinate iron-sulfur clusters (sulfur replaced by oxygen); the "
            "mutation disrupts the integrity of the [2Fe-2S] cluster loop, impairing cluster "
            "insertion by SDHAF1. Catastrophic: no [2Fe-2S] → no electron relay from FADH2 → "
            "complete CII block → severe HIF2α-driven pseudo-hypoxia → aggressive PGL4."
        ),
        "severity":  "severe",
        "notes": "[2Fe-2S] cluster loop disruption — Cys91Ser catastrophic FeS loss. Severe 88%. High malignancy association. SDHB IHC null.",
    },
    {
        "hgvs_c":    "c.IVS1+1G>A",
        "hgvs_p":    "p.splice_donor_intron1",
        "domain":    "Splice donor site — intron 1; exon 1 encodes N-terminal FeS targeting region",
        "severity_pct": 92,
        "mechanism": (
            "Canonical splice donor site disruption at IVS1+1 (intron 1, position +1). Exon 1 "
            "encodes the N-terminal region of SDHB including the mitochondrial targeting sequence "
            "and the start of the FeS cluster coordinating domain. Splice site mutation causes "
            "intron 1 retention or exon 1 skipping, generating an out-of-frame transcript subject "
            "to nonsense-mediated decay. Results in complete absence of mature SDHB protein — "
            "null allele. With somatic LOH at 1p36.13 (germline heterozygous carrier), complete "
            "loss of SDHB → CII non-functional → severe pseudo-hypoxia → PGL4 with highest "
            "malignancy risk among SDHB mutation classes."
        ),
        "severity":  "severe",
        "notes": "Null splice-site allele. SDHB protein absent. Severe 92% activity loss. Highest malignancy class. NMD of mRNA.",
    },
    {
        "hgvs_c":    "c.732G>A",
        "hgvs_p":    "p.Trp244Ter",
        "domain":    "C-terminal truncation — loss of [4Fe-4S] cluster region (Cys208-Cys217) and SDHC/D interface",
        "severity_pct": 95,
        "mechanism": (
            "Tryptophan-to-stop at codon 244, generating a truncated SDHB protein that retains "
            "the [2Fe-2S] (Cys70-101) and [3Fe-4S] (Cys148-185) cluster regions but loses the "
            "entire [4Fe-4S] cluster domain (Cys208, Cys211, Cys214, Cys217 at residues 208–217) "
            "and the C-terminal SDHC/D anchor interface (residues 230–280). The [4Fe-4S] cluster "
            "is essential for the final electron transfer step to ubiquinone (via SDHC/D). Without "
            "the C-terminal domain, truncated SDHB cannot dock onto SDHC-SDHD and cannot transfer "
            "electrons to ubiquinone even if proximal FeS clusters are intact. Complete CII block. "
            "Null-equivalent functionally. Highest severity in the SDHB variant series."
        ),
        "severity":  "severe",
        "notes": "C-terminal truncation — [4Fe-4S] and SDHC/D interface lost. Functionally null. Severity 95%. Most severe SDHB PGL4 variant.",
    },
]

# ─── Clinical features (PGL4 — 40-patient cohort, seed 707) ──────────────────
CLINICAL_FEATURES = [
    {"feature": "Extra-adrenal PGL (thoracic/abdominal/pelvic)",    "freq_pct": 55},
    {"feature": "Head-neck PGL (HNPGL)",                            "freq_pct": 35},
    {"feature": "Adrenal pheochromocytoma (PCC)",                   "freq_pct": 25},
    {"feature": "Malignant disease (metastases confirmed)",         "freq_pct": 35},
    {"feature": "Bilateral / multicentric disease",                 "freq_pct": 20},
    {"feature": "Renal cell carcinoma (RCC)",                       "freq_pct": 15},
    {"feature": "Biochemical secretion (catecholamines/metanephrines positive)", "freq_pct": 60},
    {"feature": "SDH-deficient GIST",                               "freq_pct":  5},
    {"feature": "Hypertension (secretory PCC/PGL)",                 "freq_pct": 45},
    {"feature": "SDHB null on IHC (SDHA proficient)",              "freq_pct": 95},
    {"feature": "DOTATATE PET-CT positive",                         "freq_pct": 80},
    {"feature": "Bone metastases (malignant subset)",               "freq_pct": 20},
]


def _pick_weighted(choices, weights):
    """Return a weighted random choice using the module-level seeded rng."""
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0
    for c, w in zip(choices, weights):
        cumulative += w
        if r < cumulative:
            return c
    return choices[-1]


def _gen_patients(n: int = 40, seed: int = 707) -> list:
    """Generate n realistic PGL4 (SDHB) patients with seeded random data.

    Each patient has:
      patient_id, age_at_diagnosis_years, sex, variant (hgvs_p), tumor_location,
      malignant (bool), bilateral (bool), rcc (bool, ~15%), secretory (bool, ~60%),
      surveillance_years, ihc_sdhb, ihc_sdha, transmission, notes.
    """
    local_rng = random.Random(seed)

    tumor_locations = [
        "Extra-adrenal PGL (abdominal)",
        "Extra-adrenal PGL (thoracic)",
        "Extra-adrenal PGL (pelvic)",
        "Head-neck PGL (carotid body)",
        "Head-neck PGL (jugulotympanic)",
        "Head-neck PGL (vagal)",
        "Adrenal PCC",
        "Multiple sites",
    ]
    tumor_weights = [22, 10, 8, 15, 10, 5, 15, 15]

    transmission_choices = ["maternal", "paternal", "de_novo"]
    transmission_weights = [40, 40, 20]   # biparental — equal maternal/paternal; ~20% de novo

    patients = []
    for i in range(n):
        local_rng.seed(seed + i * 13 + 3)
        age_dx   = local_rng.randint(18, 65)
        sex      = local_rng.choice(["M", "F"])
        variant  = local_rng.choice(ALL_VARIANTS)
        location = _pick_weighted_local(tumor_locations, tumor_weights, local_rng)
        malignant   = local_rng.random() < 0.35       # 35% malignancy (within 20–50% range)
        bilateral   = local_rng.random() < 0.20       # 20% bilateral
        rcc         = local_rng.random() < 0.15       # 15% RCC (unique SDHB feature)
        secretory   = local_rng.random() < 0.60       # 60% biochemical secretion
        surveillance_yrs = round(local_rng.uniform(1.0, 12.0), 1)
        transmission = _pick_weighted_local(
            transmission_choices, transmission_weights, local_rng
        )

        # Bone metastases only in malignant patients (~55% of malignant)
        bone_mets = malignant and (local_rng.random() < 0.55)
        # DOTATATE positive in ~80% overall
        dotatate_pos = local_rng.random() < 0.80

        patients.append({
            "patient_id":              f"SDHB-PGL4-{i+1:03d}",
            "age_at_diagnosis_years":  age_dx,
            "sex":                     sex,
            "variant_hgvs_p":          variant["hgvs_p"],
            "variant_hgvs_c":          variant["hgvs_c"],
            "variant_domain":          variant["domain"],
            "variant_severity_pct":    variant["severity_pct"],
            "tumor_location":          location,
            "malignant":               malignant,
            "bilateral":               bilateral,
            "rcc":                     rcc,
            "secretory":               secretory,
            "bone_metastases":         bone_mets,
            "dotatate_positive":       dotatate_pos,
            "transmission":            transmission,
            "surveillance_years":      surveillance_yrs,
            "ihc_sdhb":                "null",
            "ihc_sdha":                "proficient",   # SDHA proficient — key DDx vs SDHA loss
            "notes": (
                f"PGL4 SDHB {variant['hgvs_p']}; "
                f"{'malignant' if malignant else 'localized'}; "
                f"{'bilateral; ' if bilateral else ''}"
                f"{'RCC concurrent; ' if rcc else ''}"
                f"{'secretory (metanephrines+); ' if secretory else 'non-secretory; '}"
                f"IHC SDHB null / SDHA proficient"
            ),
        })
    return patients


def _pick_weighted_local(choices, weights, local_rng):
    """Weighted pick using a supplied local_rng (not the module-level rng)."""
    total = sum(weights)
    r = local_rng.uniform(0, total)
    cumulative = 0
    for c, w in zip(choices, weights):
        cumulative += w
        if r < cumulative:
            return c
    return choices[-1]


# ─── get_overview ─────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Return top-level overview of SDHB/PGL4 gene, cohort summary, and key facts."""
    rng.seed(SEED)
    patients = _gen_patients(N_PATIENTS, SEED)

    n_malignant  = sum(1 for p in patients if p["malignant"])
    n_bilateral  = sum(1 for p in patients if p["bilateral"])
    n_rcc        = sum(1 for p in patients if p["rcc"])
    n_secretory  = sum(1 for p in patients if p["secretory"])
    n_extraadren = sum(
        1 for p in patients
        if "Extra-adrenal" in p["tumor_location"]
    )
    n_headneck   = sum(
        1 for p in patients
        if "Head-neck" in p["tumor_location"]
    )
    n_adrenal    = sum(
        1 for p in patients
        if "Adrenal PCC" in p["tumor_location"]
    )
    n_bone_mets  = sum(1 for p in patients if p["bone_metastases"])
    n_dotatate   = sum(1 for p in patients if p["dotatate_positive"])

    ages         = [p["age_at_diagnosis_years"] for p in patients]
    mean_age     = round(sum(ages) / len(ages), 1)

    # Variant frequency in cohort
    variant_counts: dict = {}
    for p in patients:
        v = p["variant_hgvs_p"]
        variant_counts[v] = variant_counts.get(v, 0) + 1
    top_variant = max(variant_counts, key=lambda k: variant_counts[k])

    return {
        "gene":          GENE,
        "omim_gene":     OMIM_GENE,
        "omim_disease":  OMIM_DISEASE,
        "disease_name":  DISEASE_NAME,
        "chromosome":    CHROMOSOME,
        "protein":       "Succinate dehydrogenase subunit B (iron-sulfur subunit)",
        "protein_size":  "280 aa, ~32 kDa",
        "location":      "Mitochondrial matrix-facing; attached to IMM via SDHC/D anchor",
        "fes_clusters":  {
            "[2Fe-2S]": "Cys70, Cys72, Cys75, Cys101 — proximal to SDHA-SDHB interface",
            "[3Fe-4S]": "Cys148, Cys151, Cys185 — central electron relay",
            "[4Fe-4S]": "Cys208, Cys211, Cys214, Cys217 — proximal to SDHC/D and ubiquinone",
        },
        "inheritance":   "AD (autosomal dominant) — NOT maternally imprinted",
        "penetrance":    "25–35% by age 50 (intermediate; higher than SDHA 10%, lower than SDHD 80%)",
        "n_patients":    N_PATIENTS,
        "seed":          SEED,
        "cohort_summary": (
            f"{N_PATIENTS} patients, seed {SEED}. Inheritance AD, single-disease gene (PGL4). "
            f"No AR/biallelic phenotype; all patients have monoallelic germline SDHB + somatic LOH."
        ),

        "cohort_statistics": {
            "n_patients":           N_PATIENTS,
            "mean_age_at_diagnosis_years": mean_age,
            "age_range_years":      [min(ages), max(ages)],
            "malignant_n":          n_malignant,
            "malignant_pct":        round(100 * n_malignant / N_PATIENTS, 1),
            "bilateral_n":          n_bilateral,
            "bilateral_pct":        round(100 * n_bilateral / N_PATIENTS, 1),
            "rcc_n":                n_rcc,
            "rcc_pct":              round(100 * n_rcc / N_PATIENTS, 1),
            "secretory_n":          n_secretory,
            "secretory_pct":        round(100 * n_secretory / N_PATIENTS, 1),
            "extra_adrenal_pgl_n":  n_extraadren,
            "extra_adrenal_pgl_pct": round(100 * n_extraadren / N_PATIENTS, 1),
            "head_neck_pgl_n":      n_headneck,
            "head_neck_pgl_pct":    round(100 * n_headneck / N_PATIENTS, 1),
            "adrenal_pcc_n":        n_adrenal,
            "adrenal_pcc_pct":      round(100 * n_adrenal / N_PATIENTS, 1),
            "bone_metastases_n":    n_bone_mets,
            "bone_metastases_pct":  round(100 * n_bone_mets / N_PATIENTS, 1),
            "dotatate_positive_n":  n_dotatate,
            "dotatate_positive_pct": round(100 * n_dotatate / N_PATIENTS, 1),
            "most_common_variant":  top_variant,
            "n_unique_variants":    len(variant_counts),
        },

        "variant_summary": {
            "n_variants":   len(ALL_VARIANTS),
            "fes_ligand_variants":  ["p.Cys101Tyr", "p.Cys91Ser"],
            "null_variants":        ["c.IVS1+1G>A", "p.Trp244Ter"],
            "structural_variants":  ["p.Pro197Arg", "p.Pro200Arg"],
            "interface_variants":   ["p.Arg46Gln"],
            "proximity_variants":   ["p.Gly72Val"],
            "severity_range":       "70–95% activity loss (all pathogenic)",
        },

        "key_facts": [
            "SDHB = THE malignancy gene: 20–50% malignancy — BY FAR the highest of all SDH genes",
            "Extra-adrenal PGL predominant (thoracic/abdominal/pelvic), NOT head-neck like SDHD",
            "RCC (renal cell carcinoma) in 15% — unique among SDH genes for RCC frequency",
            "IHC: SDHB null ONLY (SDHA proficient) — unlike SDHA loss (dual SDHA+SDHB null)",
            "NOT maternally imprinted: biparental transmission (both maternal and paternal penetrant)",
            "Penetrance 25–35% by age 50 — intermediate (SDHA 10% < SDHB 25–35% < SDHD 80%)",
            "Sunitinib: best systemic evidence for metastatic SDHB-PGL/PCC (anti-VEGFR/PDGFR)",
            "Temozolomide: active in SDHB tumors with MGMT promoter methylation",
            "1p36.13 — SDHC is at 1q23.3 (same chromosome, opposite arm); WES mandatory",
            "Surveillance: annual biochemistry + imaging; DOTATATE PET-CT for systemic staging",
            "All SDHB tumors must be considered potentially malignant until metastases excluded",
            "Alpha-blockade BEFORE beta-blockade pre-op — critical sequence for PCC/secretory PGL",
        ],

        "patients": patients,
    }


# ─── get_breakdown ────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """Return variant breakdown, clinical features, DDx table, and treatment protocols."""
    rng.seed(SEED + 1000)

    # Full variant breakdown
    variant_breakdown = []
    for v in ALL_VARIANTS:
        variant_breakdown.append({
            "hgvs_c":          v["hgvs_c"],
            "hgvs_p":          v["hgvs_p"],
            "domain":          v["domain"],
            "severity":        v["severity"],
            "severity_pct":    v["severity_pct"],
            "mechanism_short": v["mechanism"][:220] + "…",
            "mechanism_full":  v["mechanism"],
            "notes":           v["notes"],
            "phenotype":       "AD PGL4 (monoallelic, penetrance ~25–35%)",
        })

    # Clinical features output
    clinical_features_out = [
        {
            "feature":   f["feature"],
            "freq_pct":  f["freq_pct"],
            "phenotype": "AD_PGL4",
        }
        for f in CLINICAL_FEATURES
    ]

    # DDx table (7 entries)
    ddx_table = [
        {
            "gene":       "SDHA",
            "locus":      "5p15.33",
            "disease":    "PGL5 + Leigh syndrome (AR biallelic) + Carney-Stratakis",
            "key_ddx": (
                "SDHA malignancy 5% vs SDHB 20–50%. SDHA IHC: SDHA null + SDHB null (dual-null); "
                "SDHB IHC: SDHB null ONLY (SDHA proficient) — IHC distinguishes SDHA vs SDHB loss. "
                "SDHA biallelic → Leigh syndrome (AR); SDHB does NOT cause Leigh. "
                "SDHA penetrance 10% vs SDHB 25–35%. SDHA: GIST (Carney-Stratakis 15%) > SDHB GIST (5%)."
            ),
            "malignancy":  "5%",
            "imprinting":  "None (biparental)",
        },
        {
            "gene":       "SDHC",
            "locus":      "1q23.3",
            "disease":    "PGL3 — head-neck paraganglioma, low malignancy",
            "key_ddx": (
                "SDHC head-neck PGL predominant vs SDHB extra-adrenal PGL predominant. "
                "Malignancy 1–3% SDHC vs 20–50% SDHB — most critical DDx. "
                "Both on chromosome 1 but DIFFERENT ARMS: SDHC 1q23.3 vs SDHB 1p36.13 (36 Mb apart). "
                "SDHC does NOT cause RCC; SDHB RCC 15%. No imprinting in either. "
                "IHC SDHB null (SDHA proficient) in both — cannot distinguish by IHC alone; WES mandatory."
            ),
            "malignancy":  "1–3%",
            "imprinting":  "None (biparental)",
        },
        {
            "gene":       "SDHD",
            "locus":      "11q23.1",
            "disease":    "PGL1 — maternal imprinting, head-neck PGL predominant",
            "key_ddx": (
                "SDHD MATERNAL IMPRINTING: paternal transmission only; maternal carriers do NOT transmit disease. "
                "SDHB NOT imprinted: biparental (maternal AND paternal penetrant). "
                "SDHD penetrance ~80% vs SDHB 25–35%. SDHD head-neck PGL predominant vs SDHB extra-adrenal. "
                "Malignancy 3–5% SDHD vs 20–50% SDHB. SDHD 11q vs SDHB 1p — different chromosomes."
            ),
            "malignancy":  "3–5%",
            "imprinting":  "YES (maternal) — paternal transmission only",
        },
        {
            "gene":       "SDHAF2",
            "locus":      "11q13.1",
            "disease":    "PGL2 — maternal imprinting, SDHA flavinylation factor",
            "key_ddx": (
                "SDHAF2 MATERNAL IMPRINTING: paternal transmission only; penetrance 85–92% (highest PGL gene). "
                "SDHB NOT imprinted; penetrance 25–35%. Malignancy 5% SDHAF2 vs 20–50% SDHB. "
                "SDHAF2: head-neck PGL only; SDHB: extra-adrenal predominant + RCC. "
                "SDHAF2 IHC: SDHB null (SDHA proficient, same as SDHB IHC) — WES mandatory."
            ),
            "malignancy":  "5%",
            "imprinting":  "YES (maternal) — paternal only; penetrance 85–92%",
        },
        {
            "gene":       "VHL",
            "locus":      "3p25.3",
            "disease":    "Von Hippel-Lindau — hemangioblastoma + ccRCC + PCC",
            "key_ddx": (
                "VHL: hemangioblastoma (cerebellum/spine/retina) — ABSENT in SDHB. "
                "VHL: ccRCC in ~70%; SDHB: RCC in 15% (much lower frequency). "
                "VHL: direct HIF1α suppressor (VHL ubiquitin ligase). SDHB: indirect via succinate-PHD. "
                "VHL IHC: SDHB proficient (no SDH deficiency). SDHB IHC: SDHB null. "
                "VHL: 3p25.3 vs SDHB: 1p36.13 — different chromosomes. Both AD."
            ),
            "malignancy":  "ccRCC 70%; PCC 10–20% benign",
            "imprinting":  "None (AD, LOH)",
        },
        {
            "gene":       "NF1",
            "locus":      "17q11.2",
            "disease":    "Neurofibromatosis type 1 — café-au-lait, PCC (benign)",
            "key_ddx": (
                "NF1: PCC usually benign, unilateral, adrenal only; no extra-adrenal PGL typical. "
                "NF1: café-au-lait spots, neurofibromas, Lisch nodules — ALL absent in SDHB. "
                "NF1: NO RCC association; SDHB RCC 15%. "
                "NF1: PCC malignancy <5% vs SDHB PGL malignancy 20–50%. "
                "NF1: 17q11.2 vs SDHB: 1p36.13. NF1 IHC: SDHB proficient."
            ),
            "malignancy":  "<5% (PCC benign typical)",
            "imprinting":  "None (AD, de novo common)",
        },
        {
            "gene":       "RET (MEN2)",
            "locus":      "10q11.21",
            "disease":    "MEN2A/MEN2B — MTC + PCC + parathyroid / marfanoid",
            "key_ddx": (
                "MEN2A: medullary thyroid cancer (MTC) + PCC + primary hyperparathyroidism. "
                "MEN2B: marfanoid habitus + MTC + PCC + mucosal neuromas. "
                "SDHB: PGL + RCC; NO MTC, NO parathyroid disease, NO marfanoid features. "
                "MEN2 PCC usually adrenal, bilateral, benign; SDHB extra-adrenal, high malignancy. "
                "RET: 10q11.21 vs SDHB: 1p36.13. Calcitonin elevated in MEN2; normal in SDHB-PGL4."
            ),
            "malignancy":  "MTC high; PCC 5%",
            "imprinting":  "None (AD, de novo in MEN2B)",
        },
    ]

    # Treatment protocols
    treatment = {
        "phenotype": "AD PGL4 / SDHB-associated PGL/PCC/RCC",
        "pre_op_critical_sequence": {
            "rule":   "ALPHA-BLOCKADE BEFORE BETA-BLOCKADE — MANDATORY PRE-OP SEQUENCE",
            "detail": (
                "For adrenal PCC or secretory extra-adrenal PGL: alpha-blockade "
                "(phenoxybenzamine or doxazosin) MUST precede beta-blockade by ≥7–14 days "
                "pre-operatively. Reversing order (beta first) → unopposed alpha "
                "vasoconstriction → hypertensive crisis during surgical manipulation. "
                "This sequence applies to all PGL/PCC regardless of SDH subunit."
            ),
        },
        "recommended_treatments": [
            {
                "drug":      "Surgical resection",
                "dose":      "N/A — first-line for localized PGL/PCC",
                "level":     "A",
                "rationale": (
                    "Complete surgical excision curative for localized disease. "
                    "Laparoscopic adrenalectomy for PCC; open approach for large or "
                    "extra-adrenal PGL. Given malignancy risk 20–50%, pre-op staging "
                    "with DOTATATE PET-CT + CT chest/abdomen/pelvis mandatory."
                ),
            },
            {
                "drug":      "Phenoxybenzamine (alpha-blocker, pre-op)",
                "dose":      "10–40 mg/day, titrated ≥7–14 days pre-op",
                "level":     "A",
                "rationale": (
                    "Non-competitive alpha-adrenoceptor blockade for secretory PCC/PGL. "
                    "Doxazosin (selective alpha-1) used in some centres. "
                    "Must precede beta-blockade (alpha-before-beta rule)."
                ),
            },
            {
                "drug":      "Sunitinib",
                "dose":      "37.5 mg/day (continuous) or 50 mg 4 weeks on / 2 off",
                "level":     "B",
                "rationale": (
                    "BEST SYSTEMIC EVIDENCE for metastatic/unresectable SDHB-PGL/PCC. "
                    "Multi-tyrosine kinase inhibitor: anti-VEGFR-1/2/3 + PDGFR. "
                    "SDHB-PGL highly vascularized (pseudo-hypoxia → VEGF); sunitinib "
                    "targets angiogenic signalling. Phase II data: partial response 25%, "
                    "stable disease 40%; PFS ~13 months in SDHB cohort."
                ),
            },
            {
                "drug":      "177Lu-DOTATATE (PRRT)",
                "dose":      "7.4 GBq q8 weeks × 4 cycles",
                "level":     "B",
                "rationale": (
                    "Peptide receptor radionuclide therapy for SSTR2-positive inoperable "
                    "or metastatic PGL/PCC. Confirm SSTR2 expression with DOTATATE PET-CT "
                    "prior to treatment. Response rate ~30% PR + 50% SD in PGL/PCC series."
                ),
            },
            {
                "drug":      "Temozolomide",
                "dose":      "150–200 mg/m2 days 1–5 q28 days",
                "level":     "C",
                "rationale": (
                    "Alkylating agent; uniquely active in SDHB tumors with MGMT promoter "
                    "methylation/silencing (impairs DNA repair of alkyl adducts). Must confirm "
                    "MGMT status by methylation PCR or pyrosequencing before use. "
                    "Response in MGMT-methylated SDHB tumors: PR 30–40%."
                ),
            },
            {
                "drug":      "Belzutifan (PT2977)",
                "dose":      "120 mg/day",
                "level":     "B — emerging",
                "rationale": (
                    "HIF2α inhibitor — FDA-approved for VHL disease; emerging evidence in SDH-deficient "
                    "PGL/PCC and RCC (succinate-PHD-HIF2α pathway shared). Clinical trials for SDHB "
                    "ongoing. Particularly relevant for SDHB-associated RCC (15% of cohort)."
                ),
            },
            {
                "drug":      "Everolimus",
                "dose":      "10 mg/day",
                "level":     "C",
                "rationale": (
                    "mTOR inhibition for metastatic PGL/PCC. Single-arm phase II data; modest "
                    "activity. Often combined with octreotide (COOPERATE-2 trial). "
                    "Second/third-line option after sunitinib or PRRT."
                ),
            },
            {
                "drug":      "Cabozantinib",
                "dose":      "60 mg/day",
                "level":     "C",
                "rationale": (
                    "Multi-kinase inhibitor: VEGFR2/MET/AXL. Second-line metastatic PGL/PCC "
                    "after sunitinib progression. Case series and phase II data; MET inhibition "
                    "may be relevant in SDHB-RCC (MET pathway active in RCC)."
                ),
            },
        ],
        "surveillance_protocol": [
            "Annual plasma/urine metanephrines + normetanephrines (all SDHB carriers)",
            "Annual MRI/CT abdomen + pelvis (extra-adrenal PGL surveillance)",
            "Annual MRI head-neck (HNPGL surveillance)",
            "DOTATATE PET-CT every 2–3 years for systemic metastasis screening",
            "CT chest annually in malignant-phenotype patients (pulmonary metastases)",
            "Bone scan or whole-body MRI for bone metastases surveillance (malignant)",
            "Annual renal imaging (ultrasound or MRI kidney) — RCC 15% risk",
            "IHC SDHB + SDHA on all resected tumors (SDHB null / SDHA proficient confirms)",
            "Cascade genetic testing of first-degree relatives (biparental — maternal AND paternal)",
            "Surveillance starts age 6–8 (rare paediatric SDHB-PGL reported); routine from age 18",
        ],
        "no_absolute_contraindications": (
            "SDHB-PGL4 has no drug-class ABSOLUTE CONTRAINDICATIONS analogous to SDHA-Leigh "
            "(KD, metformin, valproate) — SDHB is a single-phenotype AD tumor gene, not a "
            "metabolic CII deficiency. Pre-op alpha-before-beta sequence is a CRITICAL "
            "PROTOCOL REQUIREMENT, not a pharmacological contraindication."
        ),
    }

    return {
        "gene":              GENE,
        "omim_gene":         OMIM_GENE,
        "omim_disease":      OMIM_DISEASE,
        "chromosome":        CHROMOSOME,
        "n_variants":        len(ALL_VARIANTS),
        "variant_breakdown": variant_breakdown,
        "clinical_features": clinical_features_out,
        "ddx_table":         ddx_table,
        "treatment":         treatment,
        "pathway_context": {
            "fes_electron_relay": [
                "FADH2 (SDHA) → [2Fe-2S] cluster (Cys70/72/75/101) — proximal, at SDHA-SDHB interface",
                "[2Fe-2S] → [3Fe-4S] cluster (Cys148/151/185) — central SDHB relay",
                "[3Fe-4S] → [4Fe-4S] cluster (Cys208/211/214/217) — distal, near SDHC/D",
                "[4Fe-4S] → ubiquinone (Q) via SDHC/D membrane anchor → ubiquinol (QH2)",
                "Any FeS cluster disruption blocks the complete electron relay → CII inactive",
            ],
            "pseudohypoxia_pathway": (
                "SDHB loss → CII inactive → succinate accumulates in mitochondrial matrix → "
                "succinate exported to cytoplasm → succinate inhibits PHD enzymes (α-KG competitive) → "
                "HIF1α/HIF2α not hydroxylated → not ubiquitinated by VHL → stabilized → nuclear → "
                "VEGF, EPO, angiogenic gene transcription → pseudo-hypoxic tumor → aggressive PGL4 "
                "with high vascularity and malignancy potential"
            ),
            "malignancy_mechanism": (
                "SDHB malignancy 20–50% (vs SDHA 5%, SDHC 1–3%, SDHD 3–5%) — mechanistically "
                "attributed to: (1) extra-adrenal PGL location (retroperitoneal — later detection, "
                "larger tumors); (2) severity of pseudo-hypoxia cluster (Cluster 2: SDH/FH/MDH2 — "
                "most epigenetically reprogrammed); (3) promoter hypermethylation / MGMT silencing "
                "enabling alkylator sensitivity; (4) VEGFR-driven angiogenesis amenable to sunitinib."
            ),
            "sdhaf1_link": (
                "SDHAF1 (LYR-motif protein, 19q13.12) delivers FeS clusters to SDHB (working with "
                "HSC20/HSPA9). SDHAF1 mutations cause CII deficiency / infantile leukoencephalopathy (AR). "
                "SDHB mutations cause PGL4 (AD). Both affect SDHB FeS maturation but at different nodes: "
                "SDHAF1 = assembly factor; SDHB = the subunit itself."
            ),
        },
    }


# ─── get_definitions ─────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """Return complete gene/disease definitions, IHC interpretation, references, monitoring."""
    return {
        "gene": {
            "name":       GENE,
            "full_name":  "Succinate Dehydrogenase Subunit B (Iron-Sulfur Subunit)",
            "omim_gene":  OMIM_GENE,
            "chromosome": CHROMOSOME,
            "size_aa":    280,
            "size_kda":   32,
            "domains": [
                "[2Fe-2S] coordinating domain (Cys70, Cys72, Cys75, Cys101): proximal electron input from SDHA-FADH2",
                "[3Fe-4S] coordinating domain (Cys148, Cys151, Cys185): central electron relay",
                "[4Fe-4S] coordinating domain (Cys208, Cys211, Cys214, Cys217): distal electron output to ubiquinone",
                "N-terminal SDHA-interface surface (Arg46 region): SDHA-SDHB dimer formation",
                "C-terminal SDHC/D anchor interface (residues 240–280): docking onto membrane subunits",
                "N-terminal mitochondrial targeting sequence: cleaved after import",
            ],
            "cofactor":   "Three iron-sulfur clusters ([2Fe-2S], [3Fe-4S], [4Fe-4S]) — inserted by SDHAF1",
            "function": (
                "Electron relay from SDHA-FADH2 through three sequential FeS clusters to ubiquinone; "
                "mediates FADH2 → ubiquinol step in CII and therefore in ETC."
            ),
            "assembly": (
                "SDHAF1 (LYR-motif protein) delivers FeS clusters to SDHB via HSC20/HSPA9 chaperone. "
                "FeS-matured SDHB binds flavinylated SDHA (SDHA already flavinylated by SDHAF2 at His99). "
                "SDHA-SDHB catalytic core then inserts into SDHC-SDHD membrane anchor → CII holoenzyme."
            ),
            "locus_context": (
                "1p36.13 — same chromosome as NDUFV1 (1p33) and SDHC (1q23.3, opposite arm). "
                "Distinct locus; WES mandatory to distinguish from other 1p/1q disease genes."
            ),
        },
        "disease": {
            "pgl4": {
                "omim":        OMIM_DISEASE,
                "name":        "Paraganglioma 4 (PGL4)",
                "inheritance": "AD (autosomal dominant) — NOT maternally imprinted",
                "penetrance":  "25–35% by age 50 (intermediate: SDHA 10% < SDHB 25–35% < SDHD 80%)",
                "malignancy":  "20–50% — HIGHEST OF ALL SDH/PGL GENES",
                "sites": {
                    "extra_adrenal_pgl": "55% — thoracic, abdominal (para-aortic, organ of Zuckerkandl), pelvic",
                    "head_neck_pgl":     "35% — carotid body, jugulotympanic, vagal PGL",
                    "adrenal_pcc":       "25% — functional PCC; bilateral in subset",
                    "rcc":               "15% — clear-cell or oncocytic RCC; SDHB null IHC in RCC confirms germline",
                    "gist":              "5% — SDH-deficient GIST (less than SDHA/Carney-Stratakis)",
                    "pituitary":         "rare — pituitary adenoma reported in small SDHB series",
                },
                "secretion": "60% secretory (catecholamines/metanephrines positive); 40% non-secretory",
                "bilateral": "20% bilateral or multicentric (higher in extra-adrenal PGL)",
                "metastases": (
                    "Bone (most common), lymph nodes, liver, lung. Malignancy defined by "
                    "metastasis in non-chromaffin tissue (bone/liver/lung/LN — not local invasion). "
                    "Bone scan or whole-body MRI mandatory in SDHB carriers."
                ),
                "ihc_pattern": (
                    "SDHB null (loss of SDHB staining); SDHA proficient (SDHA staining retained). "
                    "CRITICAL DDx: SDHA loss → SDHA null + SDHB null. SDHB/C/D loss → SDHB null ONLY."
                ),
                "not_imprinted": (
                    "Unlike SDHD-PGL1 (maternal imprinting, paternal-only transmission) and "
                    "SDHAF2-PGL2 (maternal imprinting, paternal-only): SDHB is NOT imprinted. "
                    "Both maternal and paternal SDHB germline mutations cause PGL4 with equal "
                    "~25–35% penetrance. Critical for genetic counselling (all children at risk, "
                    "not just children of affected fathers)."
                ),
                "surveillance_start": "Age 6–8 years (rare paediatric; routine surveillance from age 18)",
            },
        },
        "imprinting_comparison": {
            "sdhb_pgl4":  "NOT IMPRINTED — biparental; maternal AND paternal mutations penetrant (~25–35%)",
            "sdhd_pgl1":  "MATERNALLY IMPRINTED — paternal transmission only; penetrance ~80%",
            "sdhaf2_pgl2": "MATERNALLY IMPRINTED — paternal transmission only; penetrance ~85–92%",
            "sdha_pgl5":  "NOT IMPRINTED — biparental; penetrance ~10% (lower than SDHB)",
            "sdhc_pgl3":  "NOT IMPRINTED — biparental; head-neck PGL predominant; malignancy 1–3%",
        },
        "malignancy_comparison": {
            "sdhb_pgl4":   "20–50% — HIGHEST",
            "sdhd_pgl1":   "3–5%",
            "sdha_pgl5":   "5%",
            "sdhaf2_pgl2": "5%",
            "sdhc_pgl3":   "1–3% — LOWEST",
            "note": (
                "Malignancy defined as confirmed metastases in non-chromaffin tissue. SDHB tumors "
                "cluster in the pseudo-hypoxia molecular cluster (Cluster 2: SDH/FH/MDH2), which "
                "is the most epigenetically reprogrammed and most aggressive PGL/PCC subtype."
            ),
        },
        "ihc_interpretation": {
            "sdhb_loss":   "SDHB null + SDHA proficient → SDHB mutation (or SDHC/D/AF2 — SDHB secondarily lost)",
            "sdha_loss":   "SDHA null + SDHB null → SDHA mutation (UNIQUE dual-null pattern)",
            "sdhb_null_causes": [
                "SDHB germline mutation (PGL4) — primary SDHB loss",
                "SDHC germline mutation (PGL3) — secondary SDHB degradation",
                "SDHD germline mutation (PGL1) — secondary SDHB degradation",
                "SDHAF2 germline mutation (PGL2) — secondary SDHB degradation",
                "Somatic SDH mutation in sporadic tumor — secondary SDHB degradation",
            ],
            "sdha_null_causes": [
                "SDHA germline or somatic mutation ONLY — no other subunit loss causes SDHA loss",
            ],
            "rationale": (
                "SDHA is required to stabilize SDHB in assembled CII. SDHA loss → secondary SDHB "
                "degradation → dual SDHA+SDHB null. SDHB/C/D losses do not affect SDHA stability "
                "→ SDHB null only. IHC SDHA must be added to all SDH-deficient tumors to distinguish "
                "SDHA loss from SDHB/C/D losses."
            ),
            "clinical_action": (
                "SDHB null on IHC: proceed to germline sequencing of SDHB, SDHC, SDHD, SDHAF2 + SDHA. "
                "If SDHA also null: sequence SDHA germline. "
                "IHC negative (both proficient): low probability of SDH germline but VHL/RET/NF1 possible."
            ),
        },
        "pathway": {
            "fes_assembly_sdhaf1": (
                "SDHAF1 (LYR-motif protein, 19q13.12) is the dedicated SDHB FeS delivery factor. "
                "SDHAF1 works with HSC20 (HSCB) and HSPA9 (mortalin/mtHSP70) chaperones to "
                "transfer [2Fe-2S], [3Fe-4S], and [4Fe-4S] clusters into apo-SDHB during CII assembly. "
                "SDHAF1 mutations cause AR CII deficiency / infantile leukoencephalopathy (white matter). "
                "SDHB mutations cause AD PGL4 (paraganglioma). Both nodes affect SDHB FeS cluster "
                "integrity but produce completely different disease: AR metabolic (SDHAF1) vs AD tumor (SDHB)."
            ),
            "pseudohypoxia_sdhb_mechanism": (
                "Biallelic SDHB loss (germline + somatic LOH) → complete CII inactivity → succinate "
                "accumulates (cannot be oxidized by non-functional CII) → succinate transported to "
                "cytoplasm → competitive inhibition of PHD1/2/3 (prolyl hydroxylase domain enzymes; "
                "α-ketoglutarate normally required as co-substrate) → HIF1α/HIF2α prolyl hydroxylation "
                "blocked → HIF1α/HIF2α not polyubiquitinated by VHL E3 ligase → HIF not degraded by "
                "proteasome → HIF1α/HIF2α stabilized → nuclear translocation → hypoxia response element "
                "(HRE) transcription → VEGF, EPO, GLUT1, LDH, glycolytic genes → pseudo-hypoxic tumor "
                "microenvironment → highly vascularized, aggressive paraganglioma → malignancy 20–50%."
            ),
            "mgmt_temozolomide": (
                "SDHB PGL4 tumors show higher frequency of MGMT promoter methylation/silencing vs "
                "other PGL genes (part of CpG island methylator phenotype / epigenetic reprogramming). "
                "MGMT (O6-methylguanine-DNA methyltransferase) normally repairs O6-alkylguanine adducts "
                "created by temozolomide alkylation. When MGMT is silenced, O6-alkylguanine adducts "
                "persist → replication arrest → apoptosis → tumor response to temozolomide. "
                "MGMT methylation testing mandatory before temozolomide use in SDHB-PGL4."
            ),
        },
        "key_references": [
            {
                "citation": (
                    "Astuti D et al. (2001) Gene mutations in the succinate dehydrogenase subunit "
                    "SDHB cause susceptibility to familial phaeochromocytoma and to familial "
                    "paraganglioma. Am J Hum Genet 69(1):49–54."
                ),
                "relevance": (
                    "First report of germline SDHB mutations in PGL4. Foundational paper "
                    "establishing SDHB as a hereditary PGL/PCC gene. Defines PGL4 as a distinct "
                    "clinical entity from PGL1 (SDHD) and PGL2 (SDHAF2)."
                ),
            },
            {
                "citation": (
                    "Timmers HJ et al. (2009) New developments in the pathophysiology, diagnosis, "
                    "and treatment of pheochromocytoma and paraganglioma. "
                    "Clin Endocrinol (Oxf) 70(4):520–531."
                ),
                "relevance": (
                    "Comprehensive review of PGL/PCC pathophysiology including SDHB clinical "
                    "spectrum, biochemical phenotype, and management. Key reference for SDHB "
                    "clinical features and surveillance recommendations."
                ),
            },
            {
                "citation": (
                    "Jochmanová I et al. (2013) Hypoxia-inducible factor signaling in "
                    "pheochromocytoma: turning the rudder in the right direction. "
                    "J Natl Cancer Inst 105(17):1270–1283."
                ),
                "relevance": (
                    "SDH/HIF pseudo-hypoxia mechanism and SDHB malignancy pathway. Establishes "
                    "HIF2α as the primary oncogenic driver in SDHB pseudo-hypoxia cluster. "
                    "Mechanistic basis for belzutifan (HIF2α inhibitor) use in SDHB-PGL4."
                ),
            },
            {
                "citation": (
                    "Crona J, Taieb D, Pacak K (2017) New Perspectives on Pheochromocytoma "
                    "and Paraganglioma: Toward a Molecular Classification. "
                    "Endocr Rev 38(6):489–515."
                ),
                "relevance": (
                    "Comprehensive SDHB malignancy meta-analysis confirming 20–50% malignancy "
                    "rate (highest of all SDH genes). Establishes molecular cluster classification "
                    "(Cluster 2: pseudo-hypoxia — SDH/FH) and treatment implications for SDHB-PGL4."
                ),
            },
        ],
        "monitoring_protocol": {
            "AD_PGL4": {
                "biochemical": (
                    "Annual plasma free metanephrines + normetanephrines (sensitive for PCC/secretory PGL); "
                    "OR 24-hour urine fractionated metanephrines + catecholamines (acceptable alternative)"
                ),
                "imaging_head_neck": "Annual MRI head-neck (carotid body/jugulotympanic/vagal PGL)",
                "imaging_abdomen_pelvis": "Annual MRI/CT abdomen + pelvis (extra-adrenal/para-aortic PGL surveillance)",
                "imaging_systemic": "DOTATATE PET-CT every 2–3 years; annually in malignant phenotype",
                "renal_surveillance": "Annual renal ultrasound or MRI (RCC risk 15%)",
                "bone_surveillance": (
                    "Whole-body MRI or bone scan in malignant patients "
                    "(bone metastases most common site in SDHB malignant PGL)"
                ),
                "ihc_all_tumors": "SDHA + SDHB IHC on all resected tumors — SDHB null / SDHA proficient confirms",
                "cascade_genetics": (
                    "Cascade testing of first-degree relatives; BIPARENTAL — "
                    "both maternal and paternal lines (NOT imprinted)"
                ),
                "start_age": "Routine surveillance from age 18; consider age 6–8 if family history of paediatric PGL",
                "frequency_malignant": (
                    "Malignant phenotype: CT chest every 6 months + DOTATATE PET-CT annually "
                    "+ bone scan every 12 months"
                ),
            },
        },
    }


if __name__ == "__main__":
    import json

    print("=== SDHB OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']}, OMIM Gene: {ov['omim_gene']}, OMIM Disease: {ov['omim_disease']}")
    print(f"Patients: {ov['n_patients']}, Seed: {ov['seed']}")
    print(f"Chromosome: {ov['chromosome']}, Protein: {ov['protein_size']}")
    cs = ov["cohort_statistics"]
    print(f"Malignant: {cs['malignant_n']}/{ov['n_patients']} ({cs['malignant_pct']}%)")
    print(f"RCC: {cs['rcc_n']}/{ov['n_patients']} ({cs['rcc_pct']}%)")
    print(f"Extra-adrenal PGL: {cs['extra_adrenal_pgl_n']}/{ov['n_patients']} ({cs['extra_adrenal_pgl_pct']}%)")
    print(f"Secretory: {cs['secretory_n']}/{ov['n_patients']} ({cs['secretory_pct']}%)")
    print(f"Bilateral: {cs['bilateral_n']}/{ov['n_patients']} ({cs['bilateral_pct']}%)")
    print(f"Cohort: {ov['cohort_summary']}")

    print("\n=== KEY FACTS ===")
    for kf in ov["key_facts"][:4]:
        print(f"  • {kf}")

    print("\n=== BREAKDOWN ===")
    bd = get_breakdown()
    print(f"Variants: {bd['n_variants']}")
    print(f"Clinical features: {len(bd['clinical_features'])}")
    print(f"DDx entries: {len(bd['ddx_table'])}")
    print(f"Treatment drugs: {len(bd['treatment']['recommended_treatments'])}")

    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    print(list(df.keys()))

    print("\n=== SAMPLE PATIENTS (first 3) ===")
    for p in ov["patients"][:3]:
        print(
            f"  {p['patient_id']}: age {p['age_at_diagnosis_years']}, "
            f"{p['sex']}, {p['variant_hgvs_p']}, {p['tumor_location']}, "
            f"malignant={p['malignant']}, rcc={p['rcc']}, "
            f"secretory={p['secretory']}, bilateral={p['bilateral']}"
        )

    print("\n✅ SDHB dashboard OK")
