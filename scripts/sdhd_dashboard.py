#!/usr/bin/env python3
"""SDHD — Succinate Dehydrogenase Subunit D (Cytochrome b Small Subunit) /
Paraganglioma 1 (PGL1) — AD, MATERNALLY IMPRINTED, Paternal-Transmission-Only.

SDHD (Succinate Dehydrogenase Complex Subunit D; OMIM *602690) encodes the 159-amino-acid,
~17 kDa cytochrome b small subunit of Complex II (succinate dehydrogenase, SDH). Together
with SDHC (the large subunit), SDHD forms the membrane anchor of CII, embedding the
SDHA-SDHB catalytic dimer into the inner mitochondrial membrane (IMM).

  SDHD gene     OMIM *602690
  Protein       Succinate dehydrogenase complex subunit D (cytochrome b small subunit)
  Size          159 aa, ~17 kDa
  Location      Integral inner mitochondrial membrane (IMM), 3 TM helices
  Chromosome    11q23.1
  CII role      Membrane anchor with SDHC; provides axial heme b ligand (His19);
                participates in ubiquinone QP-site binding together with SDHC

Heme b and Ubiquinone Binding:
  Heme b        Single heme b group per CII holoenzyme; axially ligated by SDHD His19
                (and SDHC His91). Essential for CII structural integrity and assembly.
  Ubiquinone    QP (proximal) site: SDHD C-terminal region contributes alongside SDHC
                TM2-TM3 loop (Tyr89, Gly93, Arg136); CoQ10 → CoQH2 here before ETC.

Disease: Paraganglioma 1 (PGL1) — OMIM #168000
  Inheritance   AD (autosomal dominant), MATERNALLY IMPRINTED
                — ONLY PATERNAL transmission causes disease
  Penetrance    ~70–80% by age 50 (paternal allele) — HIGHEST of all SDH genes
  Also          Carney-Stratakis Syndrome (CSS, OMIM #606764) — rare (~5% GIST)

MATERNAL IMPRINTING — SDHD (and SDHAF2):
  The maternal SDHD allele is epigenetically silenced (imprinted) in paraganglionic
  and related neural-crest-derived tissues. Therefore:
    Paternal transmission → child inherits active (unimprinted) SDHD mutation → PGL1
    Maternal transmission → child inherits silenced SDHD allele → CLINICALLY UNAFFECTED
  Female SDHD carriers: their children are NOT at risk of developing PGL1.
  Male SDHD carriers: 50% of children (all sexes) are at risk.

SDHD vs SDHAF2 — BOTH on Chromosome 11, BOTH Maternally Imprinted:
  SDHD  11q23.1  PGL1  ~80% penetrance paternal  ~3-5% malignancy
  SDHAF2 11q13.1 PGL2  ~90% penetrance paternal  ~5%   malignancy
  These are 10Mb apart on chromosome 11q; whole-exome sequencing (WES) mandatory
  to distinguish — cannot be reliably separated by FISH or targeted panels alone.

KEY CLINICAL FEATURES — SDHD-PGL1:
  Head-neck PGL (HNPGL) predominant: carotid body (~65%), jugulotympanic (~42%), vagal (~26%).
  BILATERAL / MULTICENTRIC: ~35–45% — HIGHEST in SDH gene family due to high penetrance.
  MALIGNANCY ~3–5%: low-moderate; higher than SDHC (~1-3%), far below SDHB (20-50%).
  ADRENAL PCC ~15%: secretory; alpha-blockade before beta-blockade CRITICAL.
  IHC: SDHB null (SDHD proficient by IHC — only SDHB staining lost in SDHD-mutant tumor).
  DOTATATE PET-CT: ~75% SSTR2-positive (preferred functional imaging).
  HIGHEST PENETRANCE: ~70-80% paternal — CRITICAL counselling point for male carriers.

Reference: Baysal BE et al. (2000) Mutations in SDHD, a mitochondrial complex II gene,
in hereditary paraganglioma. Science 287(5454):848-851.
(First identification of SDHD germline mutations in hereditary PGL; Dutch founder cohort;
Science landmark paper establishing the SDH-PGL paradigm)

Reference: van Hulsteijn LT et al. (2012) Prevalence of germline SDHB, SDHC, and SDHD
mutations in patients with head-and-neck paragangliomas and pheochromocytomas.
Eur J Hum Genet 20(3):292-7.
(Definitive SDHD penetrance and phenotype analysis; ~70-80% paternal penetrance)

Reference: Havekes B et al. (2009) The association between SDHB, SDHC and SDHD
germline mutations and tumour characteristics in patients with head-and-neck paragangliomas
and phaeochromocytomas: systematic review and meta-analysis.
Eur J Endocrinol 161(3):347-54.
(Meta-analysis; SDHD phenotype characterisation; bilateral PGL prevalence; malignancy 3-5%)

Reference: Crona J, Taieb D, Pacak K (2017) New Perspectives on Pheochromocytoma and
Paraganglioma: Toward a Molecular Classification. Endocr Rev 38(6):489-515.
(Comprehensive SDH classification; SDHD malignancy 3-5%; treatment recommendations)

PATHOPHYSIOLOGY (SDHD — membrane anchor of CII):

  SDHD in normal CII function:
    1. SDHAF2 flavinylates SDHA at His99 (FAD covalent attachment)
    2. SDHAF1 delivers FeS clusters to SDHB (via HSC20/HSPA9 chaperone system)
    3. SDHA-SDHB core forms; SDHC-SDHD membrane anchor assembles in IMM
    4. SDHA-SDHB binds SDHC-SDHD → CII holoenzyme; heme b: SDHC His91 / SDHD His19
    5. SDHD region positions ubiquinone (CoQ10) at QP site → CoQ10 → CoQH2
    6. CII function: succinate + FAD → fumarate + FADH2 → electrons → ubiquinone → ETC

  SDHD loss-of-function (monoallelic, AD — PGL1, MATERNALLY IMPRINTED):
    1. Heterozygous germline SDHD paternal mutation → active allele lost in paraganglionic cells
    2. Somatic second-hit (LOH at 11q23.1) → complete SDHD loss → CII inactive → succinate ↑
    3. Succinate inhibits PHD enzymes → HIF1α/HIF2α stabilised → pseudo-hypoxia
    4. HIF target genes (VEGF, EPO) → vascular, paraganglionic tumour growth
    5. BILATERAL/MULTICENTRIC in ~38%: high penetrance drives second-hit events at multiple sites
    6. MATERNAL IMPRINTING: maternal SDHD allele silenced → only paternal loss is disease-causing
    7. PENETRANCE 70-80% paternal: HIGHEST in SDH gene family — dominant clinical gene
    8. SDHB protein destabilised by SDHD loss → IHC SDHB null (SDHB gene intact)

SDHD UNIQUE FEATURES:
  1. MATERNAL IMPRINTING: only paternal SDHD mutations cause PGL1 — female carriers
     have affected children only from paternal (grandfather) transmission; NOT maternal
  2. HIGHEST PENETRANCE: ~70-80% by age 50 (paternal) — highest of SDH gene family;
     critical counselling for male carriers (all male → children at 50% risk)
  3. CHROMOSOME 11q23.1: same arm as SDHAF2-PGL2 at 11q13.1 (~10Mb apart); both maternally
     imprinted; whole-exome sequencing mandatory to distinguish
  4. HEME B HIS19: SDHD His19 is one of two axial heme b ligands (other: SDHC His91)
  5. BILATERAL MULTICENTRIC: ~38-45% — highest bilateral rate in SDH gene family
  6. LOW-MODERATE MALIGNANCY: ~3-5% — higher than SDHC (1-3%), far below SDHB (20-50%)
  7. IHC: SDHB null ONLY (SDHD not routinely tested by IHC; SDHB loss sufficient for Dx)
  8. HNPGL PREDOMINANT: carotid body, jugulotympanic, vagal — extra-adrenal rare (unlike SDHB)
  9. DUTCH FOUNDER MUTATIONS: Leu12Arg (c.35T>G) and Cys11Tyr (c.32G>A) are common in
     Dutch/European hereditary PGL pedigrees (Baysal 2000 Science cohort)

SDHD vs SDHC KEY DIFFERENCES:
  SDHD (PGL1): MATERNALLY IMPRINTED, ~70-80% penetrance (paternal), 11q23.1, 3 TM helices,
               bilateral ~38%, GIST ~5% (CSS rare)
  SDHC (PGL3): NOT imprinted — biparental, ~50-60% penetrance, 1q23.3, 3 TM helices,
               bilateral ~22%, GIST ~10% (CSS common SDHC feature)

PHARMACOLOGY:
  Alpha-blockade (phenoxybenzamine) BEFORE beta-blockade — CRITICAL pre-op PCC/secretory PGL
  Surgery — primary curative treatment for localised SDHD HNPGL; ENT/skull-base expertise
  177Lu-DOTATATE — SSTR2-positive metastatic/progressive SDHD PGL (~75% SSTR2+)
  Sunitinib — anti-VEGFR/PDGFR; metastatic SDH-deficient PGL (including rare malignant SDHD)
  Belzutifan (HIF2α inhibitor) — emerging; SDH-deficient PGL/RCC
  Surveillance: annual MRI head/neck + chest/abdomen/pelvis; annual catecholamines/metanephrines;
                DOTATATE PET-CT for known/suspected metastatic disease
"""

import random

# ── Module constants ──────────────────────────────────────────────────────────
GENE          = "SDHD"
OMIM_GENE     = "602690"
OMIM_DISEASE  = "168000"   # PGL1
OMIM_CSS      = "606764"   # Carney-Stratakis Syndrome (rare in SDHD)
CHROMOSOME    = "11q23.1"
PROTEIN_SIZE  = "159 aa, ~17 kDa"
TM_HELICES    = "3 TM helices (TM1, TM2, TM3)"
N_PATIENTS    = 40
SEED          = 711
PENETRANCE    = "~70–80% by age 50 (paternal transmission) — HIGHEST SDH gene"
MALIGNANCY    = "~3–5% (low-moderate; below SDHB 20–50%, above SDHC ~1–3%)"
INHERITANCE   = "AD (autosomal dominant), MATERNALLY IMPRINTED — paternal transmission only"
IMPRINTING    = "MATERNALLY IMPRINTED — only paternal SDHD mutation causes PGL1; female carriers' children NOT at risk"

rng = random.Random(SEED)

# ── Pathogenic / likely-pathogenic variants in SDHD ──────────────────────────
VARIANTS = [
    {
        "cDNA": "c.35T>G",
        "protein": "p.Leu12Arg",
        "location": "TM1 helix — N-terminal transmembrane segment",
        "consequence": "Leucine-to-arginine substitution in TM1; introduces positive charge in hydrophobic core; TM helix packing severely disrupted; CII membrane anchor assembly impaired",
        "pathogenicity_pct": 82,
        "severity": "Severe",
        "phenotype": "PGL1 — HNPGL (carotid body predominant); bilateral in ~40%; Baysal 2000 Dutch founder cohort",
        "population": "Dutch/European — major founder mutation in hereditary HNPGL; Baysal 2000 Science cohort",
        "reference": "Baysal BE et al. (2000) Science 287(5454):848-851 — first SDHD PGL1 report; Dutch founder",
    },
    {
        "cDNA": "c.32G>A",
        "protein": "p.Cys11Tyr",
        "location": "TM1 helix — N-terminal (adjacent to Leu12)",
        "consequence": "Cysteine-to-tyrosine in TM1; loss of cysteine thiol; TM1 core packing disrupted; reduced penetrance compared with Leu12Arg (~50-65%)",
        "pathogenicity_pct": 62,
        "severity": "Intermediate",
        "phenotype": "PGL1 — HNPGL; reduced penetrance (~50-65%) vs other SDHD variants; some asymptomatic carriers",
        "population": "Dutch/European — second Dutch founder mutation; penetrance lower than Leu12Arg",
        "reference": "Baysal BE et al. (2000) Science; van Hulsteijn 2012 Eur J Hum Genet — reduced penetrance data",
    },
    {
        "cDNA": "c.56A>G",
        "protein": "p.His19Arg",
        "location": "Heme b axial ligand — TM1 cytoplasmic face",
        "consequence": "Direct loss of heme b axial ligand (His19 → Arg); heme b not coordinated; CII holoenzyme assembly catastrophically impaired; no functional CII produced",
        "pathogenicity_pct": 95,
        "severity": "Severe (catastrophic)",
        "phenotype": "PGL1 — HNPGL; bilateral; high penetrance; heme b loss = most severe CII structural defect in SDHD",
        "population": "Pan-ethnic; de novo and familial",
        "reference": "Baysal BE et al. (2000) Science; Crona 2017 Endocr Rev — heme b ligand functional analysis",
    },
    {
        "cDNA": "c.112C>T",
        "protein": "p.Arg38Ter",
        "location": "TM1-TM2 loop — cytoplasmic region",
        "consequence": "Premature stop codon near N-terminus → null allele; truncated SDHD protein not incorporated into CII; complete loss of SDHD function",
        "pathogenicity_pct": 90,
        "severity": "Severe (null)",
        "phenotype": "PGL1 — HNPGL bilateral; adrenal PCC in 20%; high penetrance ~80%",
        "population": "Pan-ethnic; de novo and inherited",
        "reference": "Havekes B et al. (2009) Eur J Endocrinol; Crona 2017 Endocr Rev",
    },
    {
        "cDNA": "c.276C>G",
        "protein": "p.Asp92Glu",
        "location": "TM3 helix — near SDHC-SDHD interface / heme b coordinating region",
        "consequence": "Conservative substitution but Asp92 is critical for SDHC-SDHD membrane contact and heme b environment; CII holoenzyme destabilised; incomplete loss of function",
        "pathogenicity_pct": 75,
        "severity": "Severe",
        "phenotype": "PGL1 — carotid body PGL predominant; jugulotympanic PGL; occasional adrenal PCC",
        "population": "Southern European — Italian and Spanish families",
        "reference": "Pasini B et al. (2008) Clin Endocrinol (Oxf) 69(5):778-86",
    },
    {
        "cDNA": "c.128G>A",
        "protein": "p.Trp43Ter",
        "location": "TM2 helix — premature truncation",
        "consequence": "Nonsense mutation in TM2 → truncated SDHD lacking TM2, TM3, and SDHC-interface regions; null phenotype; no functional membrane anchor",
        "pathogenicity_pct": 88,
        "severity": "Severe (null)",
        "phenotype": "PGL1 — HNPGL (carotid body + jugulotympanic); vagal PGL in some; bilateral ~35%",
        "population": "Pan-ethnic; multiple independent families",
        "reference": "van Hulsteijn LT et al. (2012) Eur J Hum Genet 20(3):292-7",
    },
    {
        "cDNA": "c.IVS1+2T>C",
        "protein": "splice donor — intron 1",
        "location": "Exon 1-2 boundary — N-terminal TM1 region",
        "consequence": "Splice donor loss → exon 1 skipping or cryptic splice → frameshift → truncated/unstable SDHD; reduced protein level; moderate-severe CII impairment",
        "pathogenicity_pct": 72,
        "severity": "Moderate-Severe",
        "phenotype": "PGL1 — HNPGL; variable penetrance reported (~60-75% paternal); adrenal PCC in ~12%",
        "population": "European; Spanish and German pedigrees",
        "reference": "Pasini B et al. (2008) Clin Endocrinol (Oxf) 69(5):778-86",
    },
    {
        "cDNA": "c.34G>A",
        "protein": "p.Gly12Ser",
        "location": "TM1 helix — adjacent to Cys11 and Leu12 founder mutations",
        "consequence": "Glycine-to-serine in TM1; loss of glycine conformational flexibility in TM helix; moderate TM1 packing disruption; SDHD-SDHC interface partially preserved",
        "pathogenicity_pct": 65,
        "severity": "Intermediate",
        "phenotype": "PGL1 — carotid body PGL; late onset >40 yr; bilateral in ~25%; lower penetrance variant (~60-70%)",
        "population": "European — French and Belgian families",
        "reference": "Havekes B et al. (2009) Eur J Endocrinol 161(3):347-54",
    },
]

# ── Patient cohort (40 patients, seed 711) ────────────────────────────────────
def _pick_weighted(choices, weights, local_rng):
    """Return a weighted random choice."""
    total = sum(weights)
    r = local_rng.random() * total
    cum = 0.0
    for c, w in zip(choices, weights):
        cum += w
        if r <= cum:
            return c
    return choices[-1]


def _gen_patients(n: int = 40, seed: int = 711) -> list:
    """Generate n realistic PGL1 (SDHD) patients with seeded random data."""
    local_rng = random.Random(seed)
    patients = []
    for i in range(n):
        local_rng.seed(seed + i * 17 + 5)

        # Age at diagnosis (HNPGL: typically 25-55 yr; SDHD earlier onset than SDHC due to high penetrance)
        age_dx = int(local_rng.gauss(38, 13))
        age_dx = max(16, min(70, age_dx))

        # Primary tumor location (HNPGL strongly predominant)
        pgl_type = _pick_weighted(
            ["Carotid body PGL", "Jugulotympanic PGL", "Vagal PGL",
             "Adrenal PCC", "Extra-adrenal PGL (thoracic/abdominal)"],
            [65, 42, 26, 15, 10],
            local_rng,
        )

        # Bilateral / multicentric (highest in SDH family ~38%)
        bilateral = local_rng.random() < 0.38

        # Secretory
        secretory = (pgl_type == "Adrenal PCC") or (local_rng.random() < 0.14)

        # Malignant (~4% for SDHD)
        malignant = local_rng.random() < 0.04

        # GIST / Carney-Stratakis (rare in SDHD ~5%)
        gist = local_rng.random() < 0.05

        # DOTATATE PET positive (~75% SSTR2 positive)
        dotatate_pos = local_rng.random() < 0.75

        # Treatment
        treatment = _pick_weighted(
            ["Surgery (curative)", "Surgery + surveillance", "Active surveillance", "Medical management"],
            [65, 22, 10, 3],
            local_rng,
        )

        # Variant
        var = VARIANTS[local_rng.randint(0, len(VARIANTS) - 1)]

        patients.append({
            "id": f"PGL1-{i + 1:03d}",
            "age_at_diagnosis_years": age_dx,
            "pgl_type": pgl_type,
            "bilateral": bilateral,
            "secretory": secretory,
            "malignant": malignant,
            "gist": gist,
            "dotatate_positive": dotatate_pos,
            "treatment": treatment,
            "variant": var["protein"],
            "variant_cdna": var["cDNA"],
            "severity": var["severity"],
        })
    return patients


# ── get_overview ──────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Return top-level overview of SDHD/PGL1 gene, cohort summary, and key facts."""
    patients = _gen_patients(N_PATIENTS, SEED)

    n_hnpgl   = sum(1 for p in patients if "PGL" in p["pgl_type"] and "adrenal" not in p["pgl_type"].lower() and "thoracic" not in p["pgl_type"].lower() and "abdominal" not in p["pgl_type"].lower())
    n_carotid = sum(1 for p in patients if "Carotid" in p["pgl_type"])
    n_jugulo  = sum(1 for p in patients if "Jugulo" in p["pgl_type"])
    n_vagal   = sum(1 for p in patients if "Vagal" in p["pgl_type"])
    n_pcc     = sum(1 for p in patients if "PCC" in p["pgl_type"])
    n_extra   = sum(1 for p in patients if "thoracic" in p["pgl_type"].lower() or "abdominal" in p["pgl_type"].lower())
    n_bilateral = sum(1 for p in patients if p["bilateral"])
    n_secretory = sum(1 for p in patients if p["secretory"])
    n_malignant = sum(1 for p in patients if p["malignant"])
    n_gist      = sum(1 for p in patients if p["gist"])
    n_dotatate  = sum(1 for p in patients if p["dotatate_positive"])
    ages        = [p["age_at_diagnosis_years"] for p in patients]

    def pct(k):
        return round(k / N_PATIENTS * 100, 1)

    var_counts: dict = {}
    for p in patients:
        var_counts[p["variant"]] = var_counts.get(p["variant"], 0) + 1
    top_variants = sorted(var_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "gene": GENE,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "omim_css": OMIM_CSS,
        "chromosome": CHROMOSOME,
        "protein_size": PROTEIN_SIZE,
        "tm_helices": TM_HELICES,
        "inheritance": INHERITANCE,
        "penetrance": PENETRANCE,
        "malignancy": MALIGNANCY,
        "imprinting": IMPRINTING,
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "cohort_summary": (
            f"{N_PATIENTS} patients, seed {SEED}. Inheritance AD, MATERNALLY IMPRINTED — "
            f"paternal transmission only. All patients have monoallelic paternal germline SDHD "
            f"+ somatic LOH at 11q23.1."
        ),
        "cohort_statistics": {
            "n_patients":           N_PATIENTS,
            "hnpgl_pct":            pct(n_hnpgl),
            "carotid_body_pct":     pct(n_carotid),
            "jugulotympanic_pct":   pct(n_jugulo),
            "vagal_pct":            pct(n_vagal),
            "adrenal_pcc_pct":      pct(n_pcc),
            "extra_adrenal_pct":    pct(n_extra),
            "bilateral_pct":        pct(n_bilateral),
            "secretory_pct":        pct(n_secretory),
            "malignant_pct":        pct(n_malignant),
            "gist_pct":             pct(n_gist),
            "dotatate_positive_pct": pct(n_dotatate),
            "age_mean":             round(sum(ages) / len(ages), 1),
            "age_min":              min(ages),
            "age_max":              max(ages),
            "n_unique_variants":    len(set(p["variant"] for p in patients)),
        },
        "cohort_summary_features": [
            {"feature": "Head-neck PGL (any)",          "freq_pct": pct(n_hnpgl)},
            {"feature": "Carotid body PGL",             "freq_pct": pct(n_carotid)},
            {"feature": "Jugulotympanic PGL",           "freq_pct": pct(n_jugulo)},
            {"feature": "Vagal PGL",                    "freq_pct": pct(n_vagal)},
            {"feature": "Adrenal PCC",                  "freq_pct": pct(n_pcc)},
            {"feature": "Extra-adrenal PGL",            "freq_pct": pct(n_extra)},
            {"feature": "Bilateral/multicentric",       "freq_pct": pct(n_bilateral)},
            {"feature": "Secretory (catecholamine)",    "freq_pct": pct(n_secretory)},
            {"feature": "Malignant",                    "freq_pct": pct(n_malignant)},
            {"feature": "GIST (Carney-Stratakis rare)", "freq_pct": pct(n_gist)},
            {"feature": "DOTATATE PET+",                "freq_pct": pct(n_dotatate)},
        ],
        "key_facts": [
            "SDHD (PGL1): MATERNALLY IMPRINTED — ONLY paternal SDHD mutations cause disease.",
            "Female SDHD carriers: children NOT at risk (maternal allele silenced). Male carriers: 50% of children at risk.",
            "HIGHEST PENETRANCE in SDH gene family: ~70-80% by age 50 (paternal transmission).",
            "HNPGL predominant: carotid body (~65%), jugulotympanic (~42%), vagal (~26%).",
            "BILATERAL/MULTICENTRIC: ~38-45% — HIGHEST bilateral rate in SDH gene family.",
            "MALIGNANCY ~3-5%: higher than SDHC (~1-3%), far below SDHB (20-50%).",
            "CRITICAL DDx: SDHD (11q23.1) vs SDHAF2 (11q13.1) — BOTH chr11, BOTH maternally imprinted, ~10Mb apart — WES mandatory.",
            "CRITICAL DDx: SDHD (imprinted) vs SDHB (NOT imprinted) — SDHB extra-adrenal predominant, malignancy 20-50%.",
            "CRITICAL DDx: SDHD (imprinted) vs SDHC (NOT imprinted) — SDHC biparental; female SDHC carrier's children ARE at risk.",
            "Heme b axial ligand: SDHD His19 (with SDHC His91) — direct His19 loss mutations most severe.",
            "IHC: SDHB null (SDHD intact by IHC) — SDHB protein destabilised by SDHD loss even though SDHB gene intact.",
            "Alpha-blockade (phenoxybenzamine) BEFORE beta-blockade: CRITICAL for secretory PGL/PCC pre-op.",
            "DOTATATE PET-CT: ~75% SSTR2-positive; preferred functional imaging for SDHD PGL.",
            "Surveillance: annual MRI head/neck + catecholamines/metanephrines; DOTATATE PET-CT for suspected metastatic.",
            "Dutch founder mutations: Leu12Arg (c.35T>G) and Cys11Tyr (c.32G>A) — hereditary HNPGL cohorts.",
        ],
        "top_variants_cohort": [
            {"variant": v, "count": c, "freq_pct": round(c / N_PATIENTS * 100, 1)}
            for v, c in top_variants
        ],
        "patients": patients,
    }


# ── get_breakdown ─────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """Return variant breakdown and functional analysis."""
    patients = _gen_patients(N_PATIENTS, SEED)
    var_counts: dict = {}
    for p in patients:
        var_counts[p["variant"]] = var_counts.get(p["variant"], 0) + 1

    return {
        "gene": GENE,
        "chromosome": CHROMOSOME,
        "variants": [
            {
                "cDNA": v["cDNA"],
                "protein": v["protein"],
                "location": v["location"],
                "consequence": v["consequence"],
                "pathogenicity_pct": v["pathogenicity_pct"],
                "severity": v["severity"],
                "phenotype": v["phenotype"],
                "population": v["population"],
                "reference": v["reference"],
                "cohort_count": var_counts.get(v["protein"], 0),
            }
            for v in VARIANTS
        ],
        "structural_features": {
            "tm_helices":         3,
            "heme_b":             "His19 (SDHD) — axial ligand; shared with SDHC His91; direct His19 loss = catastrophic CII failure",
            "ubiquinone_qp_site": "SDHD C-terminal region + SDHC TM2-TM3 loop (Tyr89, Gly93, Arg136) — shared CoQ10 binding pocket",
            "sdhc_interface":     "SDHD N-terminal cytoplasmic loop contacts SDHC C-terminal cytoplasmic loop — anchor assembly site",
            "molecular_weight":   "~17 kDa",
            "aa_length":          159,
        },
        "key_ddx": [
            {
                "gene": "SDHAF2 (PGL2)",
                "locus": "11q13.1",
                "ddx_point": "BOTH chr11, BOTH maternally imprinted (~10Mb apart on 11q) — CRITICAL to distinguish by WES; SDHAF2 ~90% penetrance vs SDHD ~80%; SDHAF2 HNPGL only, SDHD also PCC",
                "malignancy": "SDHAF2 ~5%",
                "penetrance": "SDHAF2 ~90% paternal",
            },
            {
                "gene": "SDHB (PGL4)",
                "locus": "1p36.13",
                "ddx_point": "NOT imprinted (both maternal and paternal pathogenic) — SDHB: extra-adrenal PGL predominant; malignancy THE HIGHEST (20-50%); RCC 15%; female SDHB carrier's children at risk",
                "malignancy": "SDHB 20-50%",
                "penetrance": "SDHB ~25-35%",
            },
            {
                "gene": "SDHC (PGL3)",
                "locus": "1q23.3",
                "ddx_point": "NOT imprinted (biparental) — SDHC: female carrier's children at risk (CRITICAL contrast to SDHD); HNPGL; low malignancy ~1-3%; CSS GIST ~10%",
                "malignancy": "SDHC ~1-3%",
                "penetrance": "SDHC ~50-60%",
            },
            {
                "gene": "SDHA (PGL5)",
                "locus": "5p15.33",
                "ddx_point": "NOT imprinted; dual-disease (AR Leigh + AD PGL5); low penetrance ~10%; SDHA IHC null + SDHB null (vs SDHD: SDHA intact, SDHB null only)",
                "malignancy": "SDHA ~5%",
                "penetrance": "SDHA ~10%",
            },
            {
                "gene": "NF1",
                "locus": "17q11.2",
                "ddx_point": "NF1: PCC (adrenal) — usually benign; neurofibromas + café-au-lait + Lisch nodules; HNPGL rare vs SDHD HNPGL predominant; SDHB positive by IHC",
                "malignancy": "NF1 PCC <3%",
                "penetrance": "NF1 >80%",
            },
        ],
        "treatment_summary": {
            "surgery":                  "Primary curative treatment for localised SDHD PGL; ENT/skull base surgery for HNPGL",
            "alpha_blockade_before_beta": "Phenoxybenzamine BEFORE beta-blockade: CRITICAL for secretory PGL/PCC pre-op — beta-first → hypertensive crisis",
            "prrt":                     "177Lu-DOTATATE: SSTR2-positive metastatic/progressive SDHD PGL (~75% SSTR2+)",
            "sunitinib":                "Anti-VEGFR/PDGFR; best systemic evidence for metastatic SDH-deficient PGL (including rare malignant SDHD)",
            "belzutifan":               "HIF2α inhibitor; emerging for unresectable/metastatic SDH-deficient PGL — same pseudo-hypoxia mechanism",
            "surveillance":             "Annual MRI head/neck (primary) + chest/abdomen/pelvis; annual plasma/urine catecholamines + metanephrines; DOTATATE PET-CT for known/suspected metastatic disease",
        },
        "imprinting_counselling": {
            "mechanism":            "Maternal SDHD allele is epigenetically silenced in paraganglionic tissue → only paternal allele expressed → only paternal loss causes PGL1",
            "male_carrier":         "Male carrier → 50% of all children inherit active (paternal) SDHD mutation → at risk; recommend testing all children",
            "female_carrier":       "Female carrier → maternal allele passed to children is silenced → children NOT at risk → surveillance NOT required for children of female carriers",
            "risk_by_sex":          "Risk to child depends on which PARENT carries the mutation, not which sex the child is",
            "vs_sdhaf2":            "SDHAF2 (11q13.1) is ALSO maternally imprinted — identical inheritance rule; clinically indistinguishable by imprinting pattern alone",
            "vs_sdhb_sdhc":         "SDHB and SDHC: NOT imprinted — female carriers' children ARE at risk; critical counselling difference from SDHD",
            "penetrance_paternal":  "~70-80% by age 50 (paternal) — highest in SDH gene family; male carriers must be strongly counselled about child surveillance",
        },
    }


# ── get_definitions ───────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """Return clinical definitions, standards, and references."""
    return {
        "gene": GENE,
        "gene_full_name": "Succinate Dehydrogenase Complex Subunit D (Cytochrome b Small Subunit)",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "omim_css": OMIM_CSS,
        "disease_names": {
            "PGL1": "Paraganglioma 1 (OMIM #168000)",
            "CSS":  "Carney-Stratakis Syndrome (OMIM #606764, rare in SDHD)",
        },
        "chromosome": CHROMOSOME,
        "protein_size": PROTEIN_SIZE,
        "tm_helices": TM_HELICES,
        "inheritance": INHERITANCE,
        "penetrance": PENETRANCE,
        "malignancy": MALIGNANCY,
        "imprinting": IMPRINTING,
        "definitions": [
            {"term": "PGL1", "definition": "Paraganglioma 1 — AD maternally imprinted SDH gene; HNPGL predominant; highest penetrance ~70-80%; OMIM #168000"},
            {"term": "Paraganglioma (PGL)", "definition": "Tumor arising from neural crest-derived paraganglion cells; head-neck or extra-adrenal; 15-30% genetic; SDH genes most common inherited cause"},
            {"term": "HNPGL", "definition": "Head-neck paraganglioma — carotid body, jugulotympanic, vagal; most common SDHD tumour location (>80%)"},
            {"term": "Carotid body PGL", "definition": "PGL at carotid body (bifurcation common carotid artery); most common SDHD location (~65%); Zellballen architecture"},
            {"term": "Jugulotympanic PGL", "definition": "PGL in jugular bulb / middle ear; pulsatile tinnitus; hearing loss; skull base involvement; ~42% SDHD cohort"},
            {"term": "Maternal imprinting", "definition": "Epigenetic silencing of maternal allele in specific tissues → only paternal allele expressed → only paternal mutations cause disease; SDHD and SDHAF2 are maternally imprinted; SDHB and SDHC are NOT"},
            {"term": "Paternal transmission only (SDHD)", "definition": "A father with SDHD mutation: 50% of children at risk. A mother with SDHD mutation: children NOT at risk. Critical for cascade testing and surveillance planning."},
            {"term": "Bilateral/multicentric PGL", "definition": "Multiple synchronous or metachronous PGL; ~38-45% in SDHD — highest in SDH family; requires total-body imaging surveillance; annual MRI mandatory"},
            {"term": "Heme b (His19)", "definition": "Single non-covalent heme b per CII; axially ligated by SDHD His19 (and SDHC His91); required for CII assembly and stability; His19Arg = catastrophic loss"},
            {"term": "IHC SDHB null", "definition": "Absent SDHB staining by immunohistochemistry in SDH-deficient tumours; surrogate for all SDH subunit losses including SDHD; SDHD itself is usually IHC-positive (SDHB loss is the signal)"},
            {"term": "LOH", "definition": "Loss of heterozygosity at 11q23.1 — somatic second-hit inactivating remaining wild-type (maternal) SDHD allele in tumor cells; required for PGL1 tumourigenesis"},
            {"term": "Pseudo-hypoxia", "definition": "HIF1α/HIF2α stabilisation due to succinate-mediated PHD inhibition without true oxygen deficit — oncogenic mechanism in all SDH-deficient PGL including SDHD"},
            {"term": "DOTATATE PET-CT", "definition": "68Ga-DOTATATE PET-CT; preferred functional imaging for SSTR2-positive PGL; ~75% SSTR2+ in SDHD PGL; sensitivity >90% for HNPGL"},
            {"term": "Alpha-blockade before beta", "definition": "Phenoxybenzamine (non-selective alpha) must be established BEFORE any beta-blocker; beta-first causes unopposed alpha → hypertensive crisis; mandatory pre-op for secretory PGL/PCC"},
            {"term": "177Lu-DOTATATE PRRT", "definition": "Peptide receptor radionuclide therapy; 177Lu-labelled somatostatin analogue; for progressive SSTR2-positive metastatic PGL; SDHD PGL ~75% eligible"},
            {"term": "Belzutifan", "definition": "HIF-2α inhibitor (VHL pathway + SDH-deficient tumours); emerging for unresectable metastatic SDH-deficient PGL; same pseudo-hypoxia axis as VHL"},
            {"term": "Dutch founder mutations", "definition": "Leu12Arg (c.35T>G) and Cys11Tyr (c.32G>A) — identified in Baysal 2000 Science Dutch hereditary PGL pedigrees; most common SDHD mutations in European HNPGL cohorts"},
            {"term": "Penetrance (~70-80%)", "definition": "~70-80% of SDHD paternal germline carriers develop PGL1 by age 50 — HIGHEST of SDH gene family; maternal carriers unaffected; absolute risk in children of male carriers is ~35-40%"},
        ],
        "standards": [
            "ENSAT/ENS@T Guidelines — Paraganglioma 2022 (European Network for the Study of Adrenal Tumors)",
            "Endocrine Society Clinical Practice Guidelines — Pheochromocytoma/Paraganglioma 2014 (Lenders et al.)",
            "NCCN Pheochromocytoma/Paraganglioma Guidelines 2024",
            "Baysal BE et al. (2000) Mutations in SDHD, a mitochondrial complex II gene, in hereditary paraganglioma. Science 287(5454):848-851",
            "van Hulsteijn LT et al. (2012) Prevalence of germline SDHB/C/D mutations in patients with head-and-neck paragangliomas and pheochromocytomas. Eur J Hum Genet 20(3):292-7",
            "Havekes B et al. (2009) Association between SDHB/C/D mutations and tumour characteristics. Eur J Endocrinol 161(3):347-54",
            "Crona J, Taieb D, Pacak K (2017) New Perspectives on PCC/PGL. Endocr Rev 38(6):489-515",
            "ACMG/AMP Variant Classification Standards (Richards 2015 Genet Med)",
            "WHO Classification of Endocrine Tumours 2022 — Paraganglioma section",
            "DOTATATE PET-CT — Hofman 2015 Lancet; SSTR2 PGL sensitivity >90%",
            "177Lu-DOTATATE PRRT — Kwekkeboom 2008 J Clin Oncol; metastatic paraganglioma data",
            "Belzutifan — Jonasch 2021 NEJM; VHL + SDH-deficient extension; HIF-2α axis",
        ],
        "references": [
            {
                "citation": "Baysal BE et al. (2000) Mutations in SDHD, a mitochondrial complex II gene, in hereditary paraganglioma. Science 287(5454):848-851.",
                "significance": "Landmark Science paper: first identification of SDHD germline mutations in hereditary PGL; Dutch founder pedigrees; established SDH-PGL paradigm",
            },
            {
                "citation": "van Hulsteijn LT et al. (2012) Prevalence of germline SDHB, SDHC, and SDHD mutations in patients with head-and-neck paragangliomas and pheochromocytomas. Eur J Hum Genet 20(3):292-7.",
                "significance": "Definitive prevalence and penetrance study; SDHD penetrance ~70-80% paternal; bilateral ~38%; HNPGL phenotype dominance confirmed",
            },
            {
                "citation": "Havekes B et al. (2009) The association between SDHB, SDHC and SDHD germline mutations and tumour characteristics. Eur J Endocrinol 161(3):347-54.",
                "significance": "Meta-analysis of SDH gene phenotypes; SDHD malignancy 3-5%; bilateral PGL frequency; age at diagnosis characterisation",
            },
            {
                "citation": "Pasini B et al. (2008) Genetic and clinical characterization of patients with SDHB, SDHC, SDHD and SDHAF2 mutations. Clin Endocrinol (Oxf) 69(5):778-86.",
                "significance": "Comparative SDH gene genotype-phenotype study; SDHD penetrance and bilateral PGL; treatment outcomes",
            },
            {
                "citation": "Crona J, Taieb D, Pacak K (2017) New Perspectives on Pheochromocytoma and Paraganglioma: Toward a Molecular Classification. Endocr Rev 38(6):489-515.",
                "significance": "Comprehensive SDH molecular classification; SDHD malignancy 3-5% confirmed; HNPGL biology; treatment recommendations including PRRT and sunitinib",
            },
            {
                "citation": "Jonasch E et al. (2021) Belzutifan for renal cell carcinoma in von Hippel-Lindau disease. N Engl J Med 385(22):2036-46.",
                "significance": "Belzutifan efficacy; HIF-2α axis directly relevant to SDH-deficient PGL (same pseudo-hypoxia mechanism as SDHD/VHL pathway)",
            },
        ],
    }
