#!/usr/bin/env python3
"""SDHC — Succinate Dehydrogenase Subunit C (Cytochrome b Large Subunit) /
Paraganglioma 3 (PGL3) / Carney-Stratakis Syndrome (CSS) — AD, NOT imprinted.

SDHC (Succinate Dehydrogenase Complex Subunit C; OMIM *602413) encodes the 169-amino-acid,
~15 kDa cytochrome b large subunit of Complex II (succinate dehydrogenase, SDH). SDHC and
SDHD together form the membrane anchor of CII, embedding the SDHA-SDHB catalytic core into
the inner mitochondrial membrane (IMM). SDHC contains 3 transmembrane (TM) helices and is
the larger of the two anchor subunits.

  SDHC gene     OMIM *602413
  Protein       Succinate dehydrogenase complex subunit C (cytochrome b large subunit)
  Size          169 aa, ~15 kDa
  Location      Integral inner mitochondrial membrane (IMM), 3 TM helices
  Chromosome    1q23.3
  CII role      Membrane anchor with SDHD; provides one of two ubiquinone-binding sites;
                coordinates heme b (one per CII holoenzyme, shared SDHC/SDHD interface)

Heme b and Ubiquinone Binding:
  Heme b        Single heme b group, axially ligated by His91 (SDHC) and His19 (SDHD)
                Non-essential for catalysis but required for CII stability and assembly
  Ubiquinone    QP (proximal) site: key residues in SDHC TM2-TM3 loop (Tyr89, Gly93, Arg136)
                plus SDHD; reduction of CoQ10 to CoQH2 here before ETC release

Disease: Paraganglioma 3 (PGL3) — OMIM #605373
  Inheritance   AD (autosomal dominant), NOT maternally imprinted (biparental transmission)
  Penetrance    ~50–60% by age 50 (higher than SDHA ~10%, lower than SDHD ~80%)
  Also          Carney-Stratakis Syndrome (PGL + GIST, OMIM #606764); adrenal PCC (~8%)

Carney-Stratakis Syndrome (CSS):
  SDHC germline mutation + SDH-deficient GIST (without adrenocortical adenoma).
  GIST in ~10% of SDHC carriers. GIST less imatinib-responsive than KIT-mutant GIST.
  Gastric GIST predominant; epithelioid morphology; multinodular.
  Note: SDHA mutations also cause CSS (more common); SDHB and SDHD rare CSS causes.

KEY CLINICAL FEATURES — SDHC-PGL3:
  Head-neck PGL (HNPGL) predominant: carotid body (60%), jugulotympanic (35%), vagal (22%).
  LOW MALIGNANCY ~1–3% — far lower than SDHB (20–50%), lower than SDHD (3–5%).
  Carney-Stratakis Syndrome (PGL + GIST): ~10% GIST in SDHC carriers.
  NOT maternally imprinted: biparental transmission (unlike SDHD-PGL1 and SDHAF2-PGL2).
  IHC: SDHB null (SDHC proficient by IHC — only SDHB staining lost in SDHC-mutant tumor).
  Alpha-blockade BEFORE beta-blockade mandatory pre-op for secretory PGL/PCC.

Reference: Niemann S, Müller U (2000) Mutations in SDHC cause autosomal dominant
paraganglioma, type 3. Nat Genet 26(3):268-70.
(First report of SDHC germline mutations as cause of PGL3; foundational publication)

Reference: Baysal BE et al. (2004) Prevalence of SDHB, SDHC, and SDHD germline mutations
in clinic patients with head-and-neck paragangliomas. J Med Genet 41(9):703-9.
(Multi-gene prevalence study; SDHC less common than SDHD/SDHB in HNPGL cohorts)

Reference: Pasini B et al. (2008) Genetic and clinical characterization of patients with
SDHB, SDHC, SDHD and SDHAF2 mutations. Clin Endocrinol (Oxf) 69(5):778-86.
(Comparative SDH gene study; SDHC penetrance ~50-60%; CSS GIST characterization)

Reference: Crona J, Taieb D, Pacak K (2017) New Perspectives on Pheochromocytoma and
Paraganglioma: Toward a Molecular Classification. Endocr Rev 38(6):489-515.
(Comprehensive SDH classification review; SDHC malignancy ~1-3% confirmed)

PATHOPHYSIOLOGY (SDHC — membrane anchor of CII):

  SDHC in normal CII function:
    1. SDHAF2 flavinylates SDHA at His99 (FAD covalent attachment)
    2. SDHAF1 delivers FeS clusters to SDHB (via HSC20/HSPA9 chaperone system)
    3. SDHA-SDHB core forms; SDHC-SDHD membrane anchor assembles in IMM
    4. SDHA-SDHB binds SDHC-SDHD → CII holoenzyme; heme b between SDHC His91 / SDHD His19
    5. SDHC TM helices position ubiquinone (CoQ10) at QP site → CoQ10 → CoQH2
    6. CII function: succinate + FAD → fumarate + FADH2 → electrons → ubiquinone → ETC

  SDHC loss-of-function (monoallelic, AD — PGL3):
    1. Heterozygous germline SDHC mutation → haploinsufficiency in chromaffin/paraganglionic cells
    2. Somatic second-hit (LOH at 1q23.3) → complete SDHC loss → CII inactive → succinate ↑
    3. Succinate inhibits PHD enzymes → HIF1α/HIF2α stabilised → pseudo-hypoxia
    4. HIF target genes (VEGF, EPO) → vascular, paraganglionic tumour growth
    5. LOW MALIGNANCY (~1-3%): mechanism unclear; possibly lower HIF2α stabilisation than SDHB?
    6. NOT IMPRINTED: biparental — maternal and paternal SDHC mutations equally disease-causing
    7. PENETRANCE 50-60% by age 50: intermediate; higher than SDHA, lower than SDHD
    8. SDHB protein destabilised by SDHC loss → IHC SDHB null (even though SDHB gene intact)

SDHC UNIQUE FEATURES:
  1. 3 TM HELICES: largest membrane anchor subunit (SDHD has 4 TM helices; SDHC has 3)
  2. HEME B HIS91: SDHC His91 is one of two axial heme b ligands (other: SDHD His19)
  3. UBIQUINONE QP SITE: SDHC TM2-TM3 loop + SDHD together form the CoQ10 binding pocket
  4. NOT IMPRINTED: CRITICAL distinguisher from SDHD-PGL1 (maternal imprinting, 11q23.1)
     and SDHAF2-PGL2 (maternal imprinting, 11q13.1) — SDHC has biparental transmission
  5. CARNEY-STRATAKIS: SDHC is a major CSS cause; ~10% GIST in SDHC carriers
  6. LOW MALIGNANCY: ~1-3% vs SDHB 20-50% — use this to reassure SDHC families
  7. SAME CHROMOSOME 1 AS SDHB: SDHC at 1q23.3, SDHB at 1p36.13 — different arms, Chr 1
  8. IHC: SDHB null ONLY (SDHC not routinely tested by IHC; SDHB loss sufficient for diagnosis)
  9. HNPGL PREDOMINANT: 80% head-neck (vs SDHB: extra-adrenal predominant)

SDHC vs SDHD KEY DIFFERENCES:
  SDHC (PGL3): NOT imprinted, ~50-60% penetrance, 1q23.3, 3 TM helices, GIST ~10%
  SDHD (PGL1): Maternally imprinted — only paternal transmission causes disease,
               ~70-80% penetrance paternal, 11q23.1, 4 TM helices, GIST ~5%

PHARMACOLOGY:
  Alpha-blockade (phenoxybenzamine) BEFORE beta-blockade — CRITICAL pre-op PCC/secretory PGL
  Sunitinib — systemic therapy for metastatic PGL (same VEGFR/PDGFR target as SDHB malignant)
  177Lu-DOTATATE — SSTR2-positive metastatic SDHC PGL (rare malignant SDHC)
  Belzutifan (HIF2α inhibitor) — emerging; SDH-deficient PGL/RCC
  Imatinib — LESS RESPONSIVE in CSS SDHC-GIST than in KIT-mutant GIST; sunitinib preferred
  Surveillance: annual MRI neck + chest/abdomen/pelvis; annual catecholamines/metanephrines;
                DOTATATE PET-CT for known/suspected metastatic disease
"""

import random

# ── Module constants ──────────────────────────────────────────────────────────
GENE          = "SDHC"
OMIM_GENE     = "602413"
OMIM_DISEASE  = "605373"   # PGL3
OMIM_CSS      = "606764"   # Carney-Stratakis Syndrome
CHROMOSOME    = "1q23.3"
PROTEIN_SIZE  = "169 aa, ~15 kDa"
TM_HELICES    = "3 TM helices (TM1, TM2, TM3)"
N_PATIENTS    = 40
SEED          = 709
PENETRANCE    = "~50–60% by age 50"
MALIGNANCY    = "~1–3% (low — far below SDHB 20–50%)"
INHERITANCE   = "AD (autosomal dominant), NOT maternally imprinted"
IMPRINTING    = "NOT imprinted — biparental (maternal AND paternal transmission equally penetrant)"

rng = random.Random(SEED)

# ── Pathogenic / likely-pathogenic variants in SDHC ──────────────────────────
VARIANTS = [
    {
        "cDNA": "c.16C>T",
        "protein": "p.Arg6Trp",
        "location": "N-terminal / pre-TM1 region",
        "consequence": "Disrupts signal peptide/mitochondrial import sequence; protein import impaired",
        "pathogenicity_pct": 65,
        "severity": "Moderate",
        "phenotype": "PGL3 — HNPGL (carotid body/jugulotympanic); CSS (GIST) in some families",
        "population": "European — multiple unrelated families; most common SDHC allele in some series",
        "reference": "Baysal 2004 J Med Genet — high frequency in HNPGL clinic cohorts",
    },
    {
        "cDNA": "c.242C>T",
        "protein": "p.Pro81Leu",
        "location": "TM2 helix",
        "consequence": "Helix-breaking proline substitution; TM2 α-helix disrupted; CII assembly impaired",
        "pathogenicity_pct": 82,
        "severity": "Severe",
        "phenotype": "PGL3 — HNPGL; occasional adrenal PCC",
        "population": "Pan-ethnic; de novo and familial",
        "reference": "Pasini 2008 Clin Endocrinol",
    },
    {
        "cDNA": "c.278G>A",
        "protein": "p.Gly93Asp",
        "location": "TM2-TM3 loop — near Tyr89 (ubiquinone QP site) and His91 (heme b ligand)",
        "consequence": "Disrupts ubiquinone QP binding site and heme b His91 coordination simultaneously; CII severely impaired",
        "pathogenicity_pct": 87,
        "severity": "Severe",
        "phenotype": "PGL3 — HNPGL (carotid body predominant); rare CSS GIST",
        "population": "Italian families — Niemann 2000 founding cohort",
        "reference": "Niemann & Müller 2000 Nat Genet — first SDHC mutations reported",
    },
    {
        "cDNA": "c.406C>T",
        "protein": "p.Arg136Trp",
        "location": "TM3 helix",
        "consequence": "Disrupts TM3 structural integrity; SDHC-SDHD interface destabilised; CII holoenzyme assembly fails",
        "pathogenicity_pct": 80,
        "severity": "Severe",
        "phenotype": "PGL3 — HNPGL bilateral; adrenal PCC in ~15%",
        "population": "Pan-ethnic; Dutch and South Asian families",
        "reference": "Pasini 2008 Clin Endocrinol",
    },
    {
        "cDNA": "c.434T>C",
        "protein": "p.Leu145Pro",
        "location": "TM3 helix",
        "consequence": "Helix-breaking proline in TM3; SDHC-SDHD membrane anchor severely disrupted",
        "pathogenicity_pct": 83,
        "severity": "Severe",
        "phenotype": "PGL3 — jugulotympanic PGL; CSS GIST in 2/5 family members",
        "population": "German and British families",
        "reference": "Baysal 2004 J Med Genet",
    },
    {
        "cDNA": "c.IVS4+1G>A",
        "protein": "splice donor — intron 4",
        "location": "Exon 4-5 boundary — TM2-TM3 loop region",
        "consequence": "Splice donor loss → exon 4 skipping → frameshift → null allele; no protein produced",
        "pathogenicity_pct": 90,
        "severity": "Severe (null)",
        "phenotype": "PGL3 — HNPGL bilateral; CSS GIST in some carriers",
        "population": "Pan-ethnic; de novo and familial",
        "reference": "Crona 2017 Endocr Rev",
    },
    {
        "cDNA": "c.501G>A",
        "protein": "p.Trp167Ter",
        "location": "C-terminal cytoplasmic loop",
        "consequence": "Premature stop → C-terminal truncation → SDHC-SDHD binding interface lost; null phenotype",
        "pathogenicity_pct": 92,
        "severity": "Severe (null)",
        "phenotype": "PGL3 — HNPGL; vagal PGL; CSS GIST occasional",
        "population": "Pan-ethnic",
        "reference": "Crona 2017 Endocr Rev",
    },
    {
        "cDNA": "c.131C>T",
        "protein": "p.Ala44Val",
        "location": "TM1 helix",
        "consequence": "Core packing disruption in TM1; mild-moderate CII assembly impairment",
        "pathogenicity_pct": 68,
        "severity": "Intermediate",
        "phenotype": "PGL3 — carotid body PGL; late onset (>40 yr); low malignancy",
        "population": "North African / MENA families",
        "reference": "Pasini 2008 Clin Endocrinol",
    },
]

# ── Patient cohort (40 patients, seed 709) ────────────────────────────────────
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


def _gen_patients(n: int = 40, seed: int = 709) -> list:
    """Generate n realistic PGL3 (SDHC) patients with seeded random data."""
    local_rng = random.Random(seed)
    patients = []
    for i in range(n):
        local_rng.seed(seed + i * 17 + 5)

        # Age at diagnosis (HNPGL: typically 30-55 yr; SDHC slightly older than SDHD)
        age_dx = int(local_rng.gauss(42, 12))
        age_dx = max(18, min(72, age_dx))

        # Primary tumor location (HNPGL predominant)
        pgl_type = _pick_weighted(
            ["Carotid body PGL", "Jugulotympanic PGL", "Vagal PGL",
             "Adrenal PCC", "Extra-adrenal PGL (thoracic/abdominal)"],
            [60, 35, 22, 8, 5],
            local_rng,
        )

        # Bilateral / multicentric
        bilateral = local_rng.random() < 0.22  # ~22%

        # Secretory
        secretory = (pgl_type == "Adrenal PCC") or (local_rng.random() < 0.12)

        # Malignant (~2% for SDHC)
        malignant = local_rng.random() < 0.025

        # GIST (Carney-Stratakis ~10%)
        gist = local_rng.random() < 0.10

        # DOTATATE PET positive (most SDHC PGL are SSTR2 positive)
        dotatate_pos = local_rng.random() < 0.72

        # Treatment
        treatment = _pick_weighted(
            ["Surgery (curative)", "Surgery + surveillance", "Active surveillance", "Medical management"],
            [65, 20, 10, 5],
            local_rng,
        )

        # Variant
        var = VARIANTS[local_rng.randint(0, len(VARIANTS) - 1)]

        patients.append({
            "id": f"PGL3-{i + 1:03d}",
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
    """Return top-level overview of SDHC/PGL3 gene, cohort summary, and key facts."""
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
            f"{N_PATIENTS} patients, seed {SEED}. Inheritance AD, NOT imprinted. "
            f"All patients have monoallelic germline SDHC + somatic LOH at 1q23.3."
        ),
        "cohort_statistics": {
            "n_patients":       N_PATIENTS,
            "hnpgl_pct":        pct(n_hnpgl),
            "carotid_body_pct": pct(n_carotid),
            "jugulotympanic_pct": pct(n_jugulo),
            "vagal_pct":        pct(n_vagal),
            "adrenal_pcc_pct":  pct(n_pcc),
            "extra_adrenal_pct": pct(n_extra),
            "bilateral_pct":    pct(n_bilateral),
            "secretory_pct":    pct(n_secretory),
            "malignant_pct":    pct(n_malignant),
            "gist_pct":         pct(n_gist),
            "dotatate_positive_pct": pct(n_dotatate),
            "age_mean":         round(sum(ages) / len(ages), 1),
            "age_min":          min(ages),
            "age_max":          max(ages),
            "n_unique_variants": len(set(p["variant"] for p in patients)),
        },
        "cohort_summary_features": [
            {"feature": "Head-neck PGL (any)", "freq_pct": pct(n_hnpgl)},
            {"feature": "Carotid body PGL",    "freq_pct": pct(n_carotid)},
            {"feature": "Jugulotympanic PGL",  "freq_pct": pct(n_jugulo)},
            {"feature": "Vagal PGL",           "freq_pct": pct(n_vagal)},
            {"feature": "Adrenal PCC",         "freq_pct": pct(n_pcc)},
            {"feature": "Extra-adrenal PGL",   "freq_pct": pct(n_extra)},
            {"feature": "Bilateral/multicentric", "freq_pct": pct(n_bilateral)},
            {"feature": "Secretory (catecholamine)", "freq_pct": pct(n_secretory)},
            {"feature": "Malignant",           "freq_pct": pct(n_malignant)},
            {"feature": "GIST (Carney-Stratakis)", "freq_pct": pct(n_gist)},
            {"feature": "DOTATATE PET+",       "freq_pct": pct(n_dotatate)},
        ],
        "key_facts": [
            "SDHC (PGL3): head-neck PGL (HNPGL) predominant — carotid body, jugulotympanic, vagal.",
            "LOW MALIGNANCY ~1-3%: far below SDHB (20-50%); use this to reassure SDHC families.",
            "NOT maternally imprinted: biparental — both maternal and paternal SDHC alleles cause disease.",
            "CRITICAL DDx: SDHD (PGL1) at 11q23.1 IS maternally imprinted — SDHC at 1q23.3 is NOT.",
            "Penetrance ~50-60% by age 50 — intermediate; higher than SDHA (~10%), lower than SDHD (~80%).",
            "Carney-Stratakis Syndrome: SDHC mutation + GIST (~10%) — gastric, epithelioid, SDH-deficient GIST.",
            "IHC: SDHB null (SDHC proficient) — SDHB protein destabilised by SDHC loss even though SDHB gene intact.",
            "Surveillance: annual MRI neck (primary) + chest/abdomen/pelvis; annual urine metanephrines/catecholamines.",
            "Alpha-blockade (phenoxybenzamine) BEFORE beta-blockade: CRITICAL for secretory PGL/PCC pre-op.",
            "DOTATATE PET-CT: preferred functional imaging for SDHC PGL (SSTR2+ in ~72%).",
            "SDHC at 1q23.3 vs SDHB at 1p36.13: same chromosome 1, different arms — WES mandatory.",
            "CSS GIST: less imatinib-responsive than KIT-mutant GIST; sunitinib preferred for progressive GIST.",
            "Belzutifan (HIF2α inhibitor): emerging for unresectable/metastatic SDH-deficient PGL.",
            "177Lu-DOTATATE PRRT: for SSTR2-positive progressive/metastatic SDHC PGL (rare but used).",
            "Family cascade: all first-degree relatives should be offered SDHC genetic testing.",
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
            "tm_helices": 3,
            "heme_b": "His91 (SDHC) — axial ligand; shared with SDHD His19",
            "ubiquinone_qp_site": "TM2-TM3 loop residues: Tyr89, Gly93, Arg136 (shared with SDHD)",
            "sdhd_interface": "C-terminal cytoplasmic loop of SDHC contacts SDHD N-terminal cytoplasmic loop",
            "molecular_weight": "~15 kDa",
            "aa_length": 169,
        },
        "key_ddx": [
            {
                "gene": "SDHD (PGL1)",
                "locus": "11q23.1",
                "ddx_point": "SDHD: MATERNALLY IMPRINTED — only paternal transmission causes disease; SDHC: NOT imprinted",
                "malignancy": "SDHD ~3-5%",
                "penetrance": "SDHD ~70-80% paternal transmission",
            },
            {
                "gene": "SDHB (PGL4)",
                "locus": "1p36.13",
                "ddx_point": "Same chromosome 1 as SDHC (1q23.3 vs 1p36.13); SDHB = THE malignancy gene (20-50%); extra-adrenal PGL predominant",
                "malignancy": "SDHB 20-50%",
                "penetrance": "SDHB ~25-35%",
            },
            {
                "gene": "SDHAF2 (PGL2)",
                "locus": "11q13.1",
                "ddx_point": "SDHAF2: MATERNALLY IMPRINTED (like SDHD); HNPGL predominant; malignancy ~5%",
                "malignancy": "SDHAF2 ~5%",
                "penetrance": "SDHAF2 ~90% paternal",
            },
            {
                "gene": "SDHA (PGL5)",
                "locus": "5p15.33",
                "ddx_point": "SDHA: dual-disease (biallelic Leigh + monoallelic PGL5); NOT imprinted; malignancy ~5%",
                "malignancy": "SDHA ~5%",
                "penetrance": "SDHA ~10%",
            },
            {
                "gene": "VHL",
                "locus": "3p25.3",
                "ddx_point": "VHL: haemangioblastoma + ccRCC + PCC (not HNPGL) — SDHB positive by IHC (VHL not SDH pathway)",
                "malignancy": "VHL PCC ~5%",
                "penetrance": "VHL ~80%",
            },
        ],
        "treatment_summary": {
            "surgery": "Primary curative treatment for localised SDHC PGL; ENT/skull base surgery for HNPGL",
            "alpha_blockade_before_beta": "Phenoxybenzamine BEFORE beta-blockade: CRITICAL for secretory PGL/PCC pre-op",
            "prrt": "177Lu-DOTATATE: SSTR2-positive metastatic/progressive SDHC PGL (rare malignant cases)",
            "sunitinib": "Anti-VEGFR/PDGFR; best systemic evidence for metastatic SDH-deficient PGL (including rare malignant SDHC)",
            "belzutifan": "HIF2α inhibitor; emerging for unresectable/metastatic SDH-deficient PGL",
            "gist_treatment": "CSS GIST: sunitinib preferred over imatinib (SDH-deficient GIST less imatinib-responsive than KIT-mutant)",
            "surveillance": "Annual MRI neck + chest/abdomen/pelvis; annual urine metanephrines/catecholamines; DOTATATE PET-CT for known/suspected metastatic",
        },
        "carney_stratakis": {
            "definition": "Carney-Stratakis Syndrome (CSS): PGL + SDH-deficient GIST dyad; OMIM #606764",
            "genes": "SDHC (most common), SDHA, SDHB, SDHD",
            "gist_frequency_sdhc": "~10% of SDHC germline carriers develop GIST",
            "gist_characteristics": "Gastric GIST; epithelioid morphology; multinodular; SDH-deficient (SDHB null by IHC)",
            "gist_vs_kitmutant": "Less imatinib-responsive than KIT-mutant GIST; sunitinib preferred; no NF1/RAS mutation",
            "css_vs_carneys": "CSS ≠ Carney complex (different genes/phenotype); CSS = PGL + GIST without adrenocortical adenoma",
        },
    }


# ── get_definitions ───────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """Return clinical definitions, standards, and references."""
    return {
        "gene": GENE,
        "gene_full_name": "Succinate Dehydrogenase Complex Subunit C (Cytochrome b Large Subunit)",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "disease_names": {
            "PGL3": "Paraganglioma 3 (OMIM #605373)",
            "CSS":  "Carney-Stratakis Syndrome (OMIM #606764)",
        },
        "chromosome": CHROMOSOME,
        "protein_size": PROTEIN_SIZE,
        "tm_helices": TM_HELICES,
        "inheritance": INHERITANCE,
        "penetrance": PENETRANCE,
        "malignancy": MALIGNANCY,
        "imprinting": IMPRINTING,
        "definitions": [
            {"term": "PGL3", "definition": "Paraganglioma 3 — AD non-imprinted SDH gene, HNPGL predominant, OMIM #605373"},
            {"term": "Paraganglioma (PGL)", "definition": "Tumor arising from neural crest-derived paraganglion cells; head-neck or extra-adrenal; 15-30% genetic"},
            {"term": "HNPGL", "definition": "Head-neck paraganglioma — carotid body, jugulotympanic, vagal; most common SDHC tumour location"},
            {"term": "Carotid body PGL", "definition": "PGL at carotid body (bifurcation common carotid artery); most common SDHC HNPGL (~60%)"},
            {"term": "Jugulotympanic PGL", "definition": "PGL in jugular bulb / middle ear; causes pulsatile tinnitus; 35% SDHC cohort"},
            {"term": "Carney-Stratakis Syndrome (CSS)", "definition": "PGL + SDH-deficient GIST dyad; SDHC, SDHA, SDHB, SDHD; OMIM #606764; ≠ Carney complex"},
            {"term": "GIST (CSS)", "definition": "Gastrointestinal stromal tumour in Carney-Stratakis; gastric; epithelioid; SDH-deficient SDHB null; ~10% SDHC carriers"},
            {"term": "Maternal imprinting", "definition": "Epigenetic silencing of maternal allele → only paternal allele expressed. SDHD and SDHAF2 are maternally imprinted; SDHC is NOT"},
            {"term": "Heme b", "definition": "Single non-covalent heme b in CII; axial ligands His91 (SDHC) and His19 (SDHD); required for CII stability; not catalytic"},
            {"term": "Ubiquinone QP site", "definition": "Proximal ubiquinone-binding site in SDHC/SDHD; CoQ10 reduced to CoQH2 here; key residues Tyr89 Gly93 Arg136 (SDHC)"},
            {"term": "Pseudo-hypoxia", "definition": "HIF1α/HIF2α stabilisation due to succinate-mediated PHD inhibition, without true oxygen deficit — oncogenic mechanism in SDH-deficient PGL"},
            {"term": "IHC SDHB null", "definition": "Absent SDHB staining by immunohistochemistry in SDH-deficient tumours; used as surrogate for all SDH subunit losses including SDHC"},
            {"term": "LOH", "definition": "Loss of heterozygosity at 1q23.3 — somatic second-hit inactivating remaining wild-type SDHC allele in tumor cells"},
            {"term": "DOTATATE PET-CT", "definition": "68Ga-DOTATATE PET-CT; preferred functional imaging for SSTR2-positive PGL; sensitivity 90-95% for SDHC PGL"},
            {"term": "Alpha-blockade before beta", "definition": "Phenoxybenzamine (non-selective alpha) must be established BEFORE any beta-blocker; beta-first causes unopposed alpha → hypertensive crisis"},
            {"term": "177Lu-DOTATATE PRRT", "definition": "Peptide receptor radionuclide therapy; 177Lu-labelled somatostatin analogue; for progressive SSTR2-positive metastatic PGL"},
            {"term": "Belzutifan", "definition": "HIF-2α inhibitor (VHL pathway + SDH-deficient tumours); emerging for unresectable metastatic SDH-deficient PGL and ccRCC"},
            {"term": "Penetrance (~50-60%)", "definition": "~50-60% of SDHC germline carriers develop PGL by age 50; intermediate — higher than SDHA 10%, lower than SDHD 80%"},
        ],
        "standards": [
            "ENSAT/ENS@T Guidelines — Paraganglioma 2022 (European Network for the Study of Adrenal Tumors)",
            "Endocrine Society Clinical Practice Guidelines — Pheochromocytoma/Paraganglioma 2014 (Lenders et al.)",
            "NCCN Pheochromocytoma/Paraganglioma Guidelines 2024",
            "Niemann S & Müller U (2000) Mutations in SDHC cause autosomal dominant paraganglioma, type 3. Nat Genet 26(3):268-70",
            "Baysal BE et al. (2004) Prevalence of SDHB, SDHC, and SDHD germline mutations. J Med Genet 41(9):703-9",
            "Pasini B et al. (2008) Genetic and clinical characterization of SDHB/C/D and SDHAF2 patients. Clin Endocrinol 69(5):778-86",
            "Crona J, Taieb D, Pacak K (2017) New Perspectives on PCC/PGL. Endocr Rev 38(6):489-515",
            "ACMG/AMP Variant Classification Standards (Richards 2015 Genet Med)",
            "WHO Classification of Endocrine Tumours 2022 — Paraganglioma section",
            "DOTATATE PET-CT — Hofman 2015 Lancet; SSTR2 PGL sensitivity 90-95%",
            "CSS GIST — Janeway 2011 Science; Pasini 2008 Clin Endocrinol; imatinib resistance data",
            "Belzutifan — Jonasch 2021 NEJM; VHL disease + SDH-deficient extension studies",
        ],
        "references": [
            {
                "citation": "Niemann S, Müller U (2000) Mutations in SDHC cause autosomal dominant paraganglioma, type 3. Nat Genet 26(3):268-70.",
                "significance": "First SDHC/PGL3 report; established SDHC as a paraganglioma susceptibility gene",
            },
            {
                "citation": "Baysal BE et al. (2004) Prevalence of SDHB, SDHC, and SDHD germline mutations in clinic patients with head-and-neck paragangliomas. J Med Genet 41(9):703-9.",
                "significance": "Multi-gene HNPGL cohort; established SDHC prevalence and genotype-phenotype correlations",
            },
            {
                "citation": "Pasini B et al. (2008) Genetic and clinical characterization of patients with SDHB, SDHC, SDHD and SDHAF2 mutations. Clin Endocrinol (Oxf) 69(5):778-86.",
                "significance": "Largest comparative SDH study; SDHC penetrance ~50-60%; CSS GIST frequency ~10%",
            },
            {
                "citation": "Crona J, Taieb D, Pacak K (2017) New Perspectives on Pheochromocytoma and Paraganglioma: Toward a Molecular Classification. Endocr Rev 38(6):489-515.",
                "significance": "Comprehensive molecular classification; SDHC malignancy ~1-3% confirmed; treatment recommendations",
            },
            {
                "citation": "Janeway KA et al. (2011) Defects in succinate dehydrogenase in gastrointestinal stromal tumors lacking KIT and PDGFRA mutations. Science 324(5923):1076-80.",
                "significance": "SDH-deficient GIST identified; foundational CSS biology; imatinib resistance basis",
            },
            {
                "citation": "Jonasch E et al. (2021) Belzutifan for renal cell carcinoma in von Hippel-Lindau disease. N Engl J Med 385(22):2036-46.",
                "significance": "Belzutifan efficacy; HIF-2α axis relevant to SDH-deficient tumours (same pseudo-hypoxia mechanism)",
            },
        ],
    }
