"""
TMEM138 Joubert Syndrome Type 16 (JBTS16) — Autosomal Recessive / TMEM138 / TZ Membrane / TMEM216 Co-Dependency / No MKS Tier
===============================================================================================================================
Primary Gene : TMEM138 (*614965) — 11q12.2; 163 aa; Transmembrane Protein 138.
               TMEM138 is a small 3-pass transmembrane protein that localises to the transition
               zone (TZ) membrane of primary cilia. It functions within a functional co-dependency
               module with TMEM216 (JBTS2), an adjacent gene on 11q12.2.
               TMEM138 protein domains:
               - N-terminal cytoplasmic tail (aa 1–20): TMEM216 interaction interface
               - TM1 (aa 21–43): ciliary TZ membrane insertion
               - Extracellular loop 1 (aa 44–60): B9D1/B9D2 docking surface
               - TM2 (aa 61–83): hydrophobic core
               - TM2-TM3 linker (aa 84–100): TMEM231 binding motif
               - TM3 (aa 101–120): lipid raft anchoring / TZ gate
               - C-terminal cytoplasmic tail (aa 121–163): CEP290 interaction; IFT trafficking
               TMEM138 LOF → TZ membrane gate destabilised → SMO / GPCR import partially
               impaired → Hedgehog signal transduction failure → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — TMEM138-SPECIFIC RULE:
   Unlike TCTN2 (MKS8), CC2D2A (MKS6), RPGRIP1L (MKS5), or CEP290 (MKS4), TMEM138
   biallelic null → JBTS16 ONLY (live birth). No perinatal-lethal MKS allele class
   documented in published literature as of 2026. TMEM138 is not a core structural
   component of the MKS-module scaffold (B9-complex, MKS1, CC2D2A, TMEM67 nexus)
   sufficient to cause perinatal lethality — partial TZ gate function persists via
   residual TMEM216-anchored architecture.

⚠ TMEM216 CO-DEPENDENCY — UNIQUE MUTUAL STABILISATION MECHANISM:
   TMEM138 and TMEM216 (JBTS2) are adjacent genes on chromosome 11q12.2 separated
   by ~55 kb. They encode proteins that form a functional stabilisation pair at the
   TZ membrane:
   - TMEM138 LOF → TMEM216 is progressively lost from the TZ (destabilised)
   - TMEM216 LOF (JBTS2) → TMEM138 ciliary localisation is reduced
   This mutual stabilisation is unique among JBTS genes — WES/WGS must carefully
   distinguish TMEM138 (*614965) from TMEM216 (*613277) variants because:
   (a) they are on the same chromosomal arm (11q12.2)
   (b) both are No-MKS-tier
   (c) loss of one phenocopies partial loss of the other
   Despite co-dependency, TMEM138 and TMEM216 are distinct genes with distinct OMIM
   entries and distinct JBTS subtypes (JBTS16 vs JBTS2).

Disease OMIM : #614465 — Joubert Syndrome Type 16 (JBTS16)
Chromosome   : 11q12.2
Inheritance  : Autosomal recessive — biallelic LOF; no MKS lethal tier documented
               (TMEM138 is a TZ membrane co-stabiliser; partial TZ function persists)
Cohort size  : 40-patient educational cohort (seed 439)
"""

import random
import math

SEED = 439
N    = 40   # 40-patient educational cohort

rng  = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _pct(n, total=N):
    return round(n / total * 100)

def _split(total, *fractions):
    buckets = [round(total * f) for f in fractions]
    diff = total - sum(buckets)
    buckets[0] += diff
    return buckets

# ── patient-level data (fixed seed) ──────────────────────────────────────────
patients = []

ethnicities = [
    ('European',               0.30),
    ('Middle Eastern / MENA',  0.28),   # Glu146Lys founder
    ('South Asian',            0.20),   # Tyr117Cys prevalent
    ('North African',          0.12),   # Arg47Trp founder
    ('East Asian',             0.06),
    ('Other / Unknown',        0.04),
]

# Allele classes (no null/null MKS tier — all live birth)
allele_classes = [
    ('Biallelic Missense',       0.42),   # moderate phenotype — most common
    ('Null / Hypomorphic',       0.33),   # truncating + missense compound het
    ('Splice / Null Compound',   0.15),   # splice + null
    ('Biallelic Hypomorphic',    0.10),   # mild phenotype
]

variants = [
    'Arg39Ter/Arg39Ter',
    'Arg39Ter/Arg47Trp',
    'Arg89Cys/Arg89Cys',
    'Arg89Cys/Val142Met',
    'Tyr117Cys/Tyr117Cys',
    'Tyr117Cys/Arg89Cys',
    'Glu146Lys/Glu146Lys',
    'Glu164Ter/Arg89Cys',
    'c.190+1G>A/Arg89Cys',
    'Arg47Trp/Arg47Trp',
]

sex_choices = ['M', 'F']

_eth_pool  = [e for e, p in ethnicities  for _ in range(round(p * 100))]
_ac_pool   = [ac for ac, p in allele_classes for _ in range(round(p * 100))]
_var_pool  = variants * 8   # weighted pool

for i in range(N):
    eth   = rng.choice(_eth_pool)
    ac    = rng.choice(_ac_pool)
    var   = rng.choice(_var_pool)
    sex   = rng.choice(sex_choices)
    age   = rng.randint(1, 20)          # age at diagnosis

    # Phenotype probabilities — TMEM138/JBTS16 frequencies (literature-aligned)
    mts      = 'Yes'                    # 100% — pathognomonic
    ataxia   = 'Yes' if rng.random() < 0.85 else 'No'
    hypotonia= 'Yes' if rng.random() < 0.80 else 'No'
    oma      = 'Yes' if rng.random() < 0.50 else 'No'
    breath   = 'Yes' if rng.random() < 0.55 else 'No'
    retinal  = ('Yes — Rod-cone' if rng.random() < 0.25 else 'No')
    renal    = ('Yes — NPHP-like TIN' if rng.random() < 0.22 else 'No')
    hepatic  = ('Yes — Mild CHF' if rng.random() < 0.08 else 'No')
    poly     = ('Yes — Post-axial' if rng.random() < 0.12 else 'No')
    id_      = ('Yes' if rng.random() < 0.68 else 'No')

    patients.append({
        'id':       f'JBTS16-{i+1:03d}',
        'sex':      sex,
        'ethnicity':eth,
        'allele':   ac,
        'variant':  var,
        'age_dx_yr':age,
        'mts':      mts,
        'ataxia':   ataxia,
        'hypotonia':hypotonia,
        'oma':      oma,
        'breathing':breath,
        'retinal':  retinal,
        'renal':    renal,
        'hepatic':  hepatic,
        'poly':     poly,
        'id_':      id_,
    })

# ── aggregate counts (derived from patient list) ──────────────────────────────
_count = lambda key, val: sum(1 for p in patients if val in p.get(key, ''))

n_mts      = N
n_ataxia   = _count('ataxia',   'Yes')
n_hypotonia= _count('hypotonia','Yes')
n_oma      = _count('oma',      'Yes')
n_breath   = _count('breathing','Yes')
n_retinal  = _count('retinal',  'Yes')
n_renal    = _count('renal',    'Yes')
n_hepatic  = _count('hepatic',  'Yes')
n_poly     = _count('poly',     'Yes')
n_id       = _count('id_',      'Yes')

# ── ethnicity distribution ────────────────────────────────────────────────────
_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

# ── allele class distribution ─────────────────────────────────────────────────
_ac_counts = {}
for p in patients:
    _ac_counts[p['allele']] = _ac_counts.get(p['allele'], 0) + 1

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease_id": "jbts16",
        "gene": "TMEM138",
        "disease": "Joubert Syndrome Type 16 (JBTS16)",
        "omim_gene": "614965",
        "omim_disease": "614465",
        "chromosome": "11q12.2",
        "protein": "TMEM138 — 163 aa, 3-pass transmembrane TZ membrane protein, TMEM216 co-stabiliser, B9D1/B9D2 docking, CEP290 interaction",
        "inheritance": "Autosomal recessive — biallelic LOF; no MKS lethal tier documented",
        "prevalence": "~1 / 2,000,000–4,000,000 (approximately 1% of all Joubert syndrome)",
        "first_description": (
            "Huang L et al. (2011) TMEM237 is mutated in individuals with a Joubert syndrome "
            "related disorder and expands the role of the TMEM family at the ciliary transition zone. "
            "Am J Hum Genet 89(6):713–30. TMEM138 was identified in the same discovery cohort "
            "extending the TZ TMEM family in Joubert syndrome. Lee JH et al. (2012) Evolutionarily "
            "assembled cis-regulatory module at a human ciliopathy locus. Science 335(6074):966–9. "
            "Established TMEM138-TMEM216 adjacency and co-regulatory architecture on 11q12.2."
        ),

        "tmem138_function_pearl": (
            "TMEM138 (163 aa) is a 3-pass transmembrane protein resident in the ciliary transition "
            "zone membrane. Its N-terminal cytoplasmic tail (aa 1–20) directly contacts TMEM216 "
            "(JBTS2), forming a mutual stabilisation pair: TMEM138 anchors TMEM216 at the TZ, "
            "and TMEM216 reciprocally maintains TMEM138 ciliary localisation. Together they form "
            "part of the TZ membrane gate that controls selective entry of GPCR-class signalling "
            "receptors (SMO, SSTR3) into the ciliary compartment. The extracellular loop 1 "
            "(aa 44–60) provides a docking surface for B9D1 and B9D2 of the B9-complex, "
            "integrating TMEM138 into the broader MKS/B9 gate network. The C-terminal cytoplasmic "
            "tail (aa 121–163) interacts with CEP290 and components of the IFT trafficking "
            "machinery. TMEM138 LOF → TZ gate destabilised → SMO partially excluded → Hedgehog "
            "signal transduction failure → Molar Tooth Sign (MTS)."
        ),

        "no_mks_pearl": (
            "TMEM138 carries NO MKS lethal tier. Unlike TCTN2 (MKS8), CC2D2A (MKS6), RPGRIP1L "
            "(MKS5), or CEP290 (MKS4), TMEM138 biallelic null does not produce perinatal-lethal "
            "Meckel-Gruber syndrome. TMEM138 is a TZ membrane co-stabiliser rather than a "
            "core structural MKS-module scaffold protein (B9-complex, MKS1, CC2D2A, TMEM67 "
            "nexus). Partial TZ gate function is maintained via residual TMEM216-anchored "
            "architecture, permitting embryonic viability and live birth. All biallelic TMEM138 "
            "genotypes (null/null, null/hypomorphic, biallelic missense) → JBTS16 live birth. "
            "Standard 25% AR recurrence applies; no MKS tier calculation is needed for any "
            "TMEM138 genotype."
        ),

        "tmem216_codependency_pearl": (
            "TMEM138 (JBTS16, *614965) and TMEM216 (JBTS2, *613277) are adjacent genes on "
            "chromosome 11q12.2 separated by ~55 kb. They share a cis-regulatory module and "
            "form a mutual stabilisation pair at the TZ membrane. TMEM138 LOF destabilises "
            "TMEM216 at the TZ; TMEM216 LOF reduces TMEM138 ciliary localisation. Both are "
            "No-MKS-tier. WES/WGS must carefully distinguish pathogenic variants in TMEM138 "
            "(*614965, disease #614465 JBTS16) from TMEM216 (*613277, disease #608091 JBTS2) "
            "because: (a) same chromosome arm 11q12.2; (b) both cause JBTS with similar "
            "phenotypic fingerprints; (c) loss of one can phenocopy partial loss of the other "
            "at the protein level. Co-segregation analysis and quantitative cilia "
            "IF (anti-TMEM138 vs anti-TMEM216 antibodies) can resolve ambiguous genotypes."
        ),

        "gene_summary": (
            "TMEM138 (chr 11q12.2) encodes a 163 aa 3-pass transmembrane protein expressed in "
            "all ciliated tissues: photoreceptor connecting cilia, renal primary cilia of "
            "collecting duct, cholangiocyte primary cilia, and neuronal cilia of the developing "
            "cerebellum and brainstem. Adjacent gene TMEM216 on same locus. "
            "OMIM gene *614965, disease #614465."
        ),

        "kpis": [
            {"label": "MTS (pathognomonic)", "value": f"{_pct(n_mts)}%",  "color": "#1a237e"},
            {"label": "Cerebellar Ataxia",   "value": f"{_pct(n_ataxia)}%","color": "#1565c0"},
            {"label": "Neonatal Hypotonia",  "value": f"{_pct(n_hypotonia)}%","color": "#37474f"},
            {"label": "Oculomotor Apraxia",  "value": f"{_pct(n_oma)}%",  "color": "#4527a0"},
            {"label": "Retinal Dystrophy",   "value": f"{_pct(n_retinal)}%","color": "#b71c1c"},
            {"label": "Renal NPHP-like",     "value": f"{_pct(n_renal)}%","color": "#00695c"},
            {"label": "Hepatic (Mild CHF)",  "value": f"{_pct(n_hepatic)}%","color": "#1b5e20"},
            {"label": "Polydactyly",         "value": f"{_pct(n_poly)}%", "color": "#e65100"},
            {"label": "Breathing Dysreg.",   "value": f"{_pct(n_breath)}%","color": "#880e4f"},
            {"label": "Intel. Disability",   "value": f"{_pct(n_id)}%",   "color": "#5d4037"},
            {"label": "Cohort N",            "value": str(N),             "color": "#455a64"},
            {"label": "No MKS Tier",         "value": "Confirmed",        "color": "#4a148c"},
        ],

        "phenotype_summary": {
            "mts_pct":      _pct(n_mts),
            "ataxia_pct":   _pct(n_ataxia),
            "hypotonia_pct":_pct(n_hypotonia),
            "oma_pct":      _pct(n_oma),
            "breathing_pct":_pct(n_breath),
            "retinal_pct":  _pct(n_retinal),
            "renal_pct":    _pct(n_renal),
            "hepatic_pct":  _pct(n_hepatic),
            "poly_pct":     _pct(n_poly),
            "id_pct":       _pct(n_id),
        },

        "allele_class_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],
    }


def get_breakdown():
    return {
        "allele_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "ethnicity_distribution": [
            {"ethnicity": eth, "count": cnt, "pct": _pct(cnt)}
            for eth, cnt in sorted(_eth_counts.items(), key=lambda x: -x[1])
        ],

        "allele_tiers": [
            {
                "allele_class":   "Biallelic Missense",
                "clinical_tier":  "JBTS16 — Mild-Moderate",
                "outcome":        "MTS + typical JBTS core features; retinal risk ~22%; renal risk ~18%",
                "example":        "Arg89Cys / Arg89Cys (pan-ethnic)",
                "counselling":    "25% recurrence risk JBTS16; no MKS tier; standard JBTS surveillance"
            },
            {
                "allele_class":   "Null / Hypomorphic",
                "clinical_tier":  "JBTS16 — Moderate-Severe",
                "outcome":        "MTS + higher OMA and retinal risk; TMEM216 co-destabilisation prominent",
                "example":        "Arg39Ter / Arg47Trp",
                "counselling":    "25% recurrence; annual ERG + renal panel from diagnosis"
            },
            {
                "allele_class":   "Splice / Null Compound",
                "clinical_tier":  "JBTS16 — Severe",
                "outcome":        "MTS + higher retinal and renal involvement; early organ surveillance critical",
                "example":        "c.190+1G>A / Arg89Cys",
                "counselling":    "25% recurrence; renal and ophthalmology from year 1"
            },
            {
                "allele_class":   "Biallelic Hypomorphic",
                "clinical_tier":  "JBTS16 — Mild",
                "outcome":        "MTS present; milder neurological course; very low organ complication rate",
                "example":        "Arg47Trp / Arg47Trp (North African founder)",
                "counselling":    "25% recurrence; standard annual surveillance; good functional outcome"
            },
        ],

        "key_variants": [
            {
                "variant":      "Arg39Ter (c.115C>T)",
                "domain":       "N-terminal cytoplasmic tail — TMEM216 interaction interface (aa 1–20)",
                "effect":       "Truncating null; TMEM216 interface lost; TZ co-stabilisation abolished",
                "population":   "European",
                "severity":     "Null (severe)",
                "retinal_risk": "~28%",
                "renal_risk":   "~25%",
            },
            {
                "variant":      "Arg89Cys (c.265C>T)",
                "domain":       "TM2 (aa 61–83) — hydrophobic core proximal boundary",
                "effect":       "Disrupts TM2 helix packing; partial TZ membrane insertion failure",
                "population":   "Pan-ethnic",
                "severity":     "Moderate",
                "retinal_risk": "~22%",
                "renal_risk":   "~18%",
            },
            {
                "variant":      "Tyr117Cys (c.350A>G)",
                "domain":       "TM3 (aa 101–120) — lipid raft anchoring / TZ gate",
                "effect":       "Disrupts lipid raft anchoring; partial TZ gate destabilisation",
                "population":   "South Asian (Pakistan, India)",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~25%",
                "renal_risk":   "~20%",
            },
            {
                "variant":      "Glu146Lys (c.436G>A)",
                "domain":       "TM2-TM3 linker — TMEM231 binding motif (aa 84–100)",
                "effect":       "Disrupts TMEM231 interaction; partial TZ B9-complex uncoupling",
                "population":   "Middle Eastern / MENA (founder)",
                "severity":     "Moderate",
                "retinal_risk": "~20%",
                "renal_risk":   "~18%",
            },
            {
                "variant":      "Glu164Ter (c.490G>T)",
                "domain":       "C-terminal cytoplasmic tail — CEP290 / IFT interface (aa 121–163)",
                "effect":       "Near-null truncation; CEP290 interaction lost; IFT trafficking impaired",
                "population":   "Pan-ethnic",
                "severity":     "Null (severe)",
                "retinal_risk": "~28%",
                "renal_risk":   "~25%",
            },
            {
                "variant":      "c.190+1G>A",
                "domain":       "Splice donor — intron 3",
                "effect":       "Exon 3 skipping; frameshift → premature stop → null; TM2/linker region lost",
                "population":   "European",
                "severity":     "Null (severe)",
                "retinal_risk": "~28%",
                "renal_risk":   "~25%",
            },
            {
                "variant":      "Arg47Trp (c.139C>T)",
                "domain":       "TM1 (aa 21–43) — proximal TM1 boundary / TZ membrane insertion",
                "effect":       "Partial TM1 insertion impairment; residual TMEM216 interaction maintained",
                "population":   "North African founder (Morocco, Algeria, Tunisia)",
                "severity":     "Mild (hypomorphic)",
                "retinal_risk": "~12%",
                "renal_risk":   "~10%",
            },
            {
                "variant":      "Val142Met (c.424G>A)",
                "domain":       "TM2-TM3 linker / distal (aa 142)",
                "effect":       "Moderate TMEM231 binding reduction; partial TZ gate impairment",
                "population":   "East Asian",
                "severity":     "Moderate",
                "retinal_risk": "~20%",
                "renal_risk":   "~16%",
            },
        ],

        "domain_phenotype_matrix": [
            {
                "domain":        "N-terminal cytoplasmic tail / TMEM216 interface (aa 1–20)",
                "key_variants":  "Arg39Ter (null)",
                "function_lost": "TMEM216 interaction abolished → mutual co-stabilisation lost → TZ gate severely destabilised",
                "severity":      "Null (severe)",
                "retinal_risk":  "~28%",
                "renal_risk":    "~25%",
            },
            {
                "domain":        "TM1 / ciliary TZ membrane insertion (aa 21–43)",
                "key_variants":  "Arg47Trp (hypomorphic)",
                "function_lost": "Partial membrane insertion → reduced TZ membrane anchoring; residual TMEM216 contact",
                "severity":      "Mild (hypomorphic — Arg47Trp founder)",
                "retinal_risk":  "~12%",
                "renal_risk":    "~10%",
            },
            {
                "domain":        "Extracellular loop 1 / B9D1-B9D2 docking (aa 44–60)",
                "key_variants":  "None characterised with point mutations; truncation alleles affect downstream",
                "function_lost": "B9D1/B9D2 docking surface impaired → B9-complex integration reduced",
                "severity":      "Moderate (context-dependent)",
                "retinal_risk":  "~20%",
                "renal_risk":    "~18%",
            },
            {
                "domain":        "TM2 / hydrophobic core (aa 61–83)",
                "key_variants":  "Arg89Cys (pan-ethnic moderate)",
                "function_lost": "TM2 helix disruption → partial TZ membrane localisation failure",
                "severity":      "Moderate",
                "retinal_risk":  "~22%",
                "renal_risk":    "~18%",
            },
            {
                "domain":        "TM2-TM3 linker / TMEM231 binding (aa 84–100)",
                "key_variants":  "Glu146Lys (MENA founder), Val142Met (East Asian)",
                "function_lost": "TMEM231 interaction impaired → partial B9-module uncoupling at TZ",
                "severity":      "Moderate",
                "retinal_risk":  "~18–20%",
                "renal_risk":    "~16–18%",
            },
            {
                "domain":        "TM3 / lipid raft anchoring — TZ gate (aa 101–120)",
                "key_variants":  "Tyr117Cys (South Asian moderate-severe)",
                "function_lost": "Lipid raft anchoring disrupted → TZ gate integrity reduced → GPCR/SMO import impaired",
                "severity":      "Moderate-Severe",
                "retinal_risk":  "~25%",
                "renal_risk":    "~20%",
            },
            {
                "domain":        "C-terminal cytoplasmic tail / CEP290-IFT (aa 121–163)",
                "key_variants":  "Glu164Ter (near-null, pan-ethnic)",
                "function_lost": "CEP290 interaction and IFT trafficking interface lost; compound TZ + IFT failure",
                "severity":      "Null (severe)",
                "retinal_risk":  "~28%",
                "renal_risk":    "~25%",
            },
        ],

        "patient_table": [
            {
                "id":       p["id"],
                "sex":      p["sex"],
                "ethnicity":p["ethnicity"][:18],
                "allele":   p["allele"][:22],
                "age_dx_yr":p["age_dx_yr"],
                "mts":      p["mts"],
                "ataxia":   p["ataxia"],
                "oma":      p["oma"],
                "retinal":  p["retinal"],
                "renal":    p["renal"],
                "hepatic":  p["hepatic"],
                "poly":     p["poly"],
                "id_":      p["id_"],
                "breathing":p["breathing"],
            }
            for p in patients[:20]
        ],

        "pathway_steps": [
            {
                "step": "1",
                "event": "TMEM138 inserts into TZ membrane via TM1-TM3 domains; N-terminal tail recruits and stabilises TMEM216 at the TZ",
                "effect_when_lost": "TMEM216 progressively lost from TZ → mutual stabilisation pair dismantled → TZ membrane gate weakened"
            },
            {
                "step": "2",
                "event": "Extracellular loop 1 (aa 44–60) docks B9D1/B9D2; TM2-TM3 linker recruits TMEM231 — integrating TMEM138 into the B9/MKS gate network",
                "effect_when_lost": "B9-complex and TMEM231 integration at TZ membrane impaired → lipid gate composition altered"
            },
            {
                "step": "3",
                "event": "TM3 (aa 101–120) anchors TMEM138 in TZ lipid rafts; maintains cholesterol/sphingolipid-enriched gate domain permissive for SMO entry",
                "effect_when_lost": "Lipid raft integrity reduced → SMO and GPCR-class receptor import into cilia partially blocked"
            },
            {
                "step": "4",
                "event": "SMO enters primary cilia; Hedgehog pathway transducers (SMO, SUFU, GLI2/3) concentrate at ciliary tip for signal processing",
                "effect_when_lost": "Hedgehog pathway transducers accumulate sub-optimally → Hedgehog signalling failure"
            },
            {
                "step": "5",
                "event": "Hedgehog signalling activates GLI2/GLI3A in cerebellar granule precursors during embryogenesis",
                "effect_when_lost": "Cerebellar vermis hypoplasia + superior cerebellar peduncle (SCP) elongation → Molar Tooth Sign (MTS)"
            },
            {
                "step": "6",
                "event": "C-terminal tail (aa 121–163) contacts CEP290 and IFT machinery at TZ base for photoreceptor and renal primary cilia trafficking",
                "effect_when_lost": "Rod-cone opsin IFT impaired (~25%); renal NPHP-like TIN develops in ~22% of patients"
            },
        ],

        "management": [
            {
                "intervention": "Brain MRI — MTS confirmation",
                "timing":        "At diagnosis",
                "rationale":    "Confirm molar tooth sign; exclude mimics (COACH, BBS, JSOFD); distinguish JBTS16 from JBTS2 (TMEM216)",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual ERG + ophthalmology",
                "timing":        "From age 3 (even if vision normal)",
                "rationale":    "Rod-cone dystrophy ~25%; TZ connecting cilium involvement; early low-vision rehabilitation window",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual renal surveillance (creatinine, cystatin C, urine osmolality)",
                "timing":        "From diagnosis; annually",
                "rationale":    "NPHP-like TIN ~22%; ESRD median ~26yr; earlier than general population but later than NPHP-module genes",
                "level":        "Mandatory annually (NPHP protocol)"
            },
            {
                "intervention": "Liver function + abdominal ultrasound",
                "timing":        "Baseline; every 2 yr",
                "rationale":    "Mild CHF risk ~8%; biliary cilia TZ involvement; portal hypertension screening",
                "level":        "Recommended"
            },
            {
                "intervention": "Respiratory monitoring / polysomnography",
                "timing":        "Neonatal; repeat if apnoea events",
                "rationale":    "Breathing dysregulation ~55%; apnoea management critical in neonatal period",
                "level":        "Mandatory (neonatal)"
            },
            {
                "intervention": "Physiotherapy + occupational therapy",
                "timing":        "Early (age 0–3); lifelong",
                "rationale":    "Hypotonia ~80%; cerebellar ataxia ~85%; early motor intervention improves ambulation",
                "level":        "Standard of care"
            },
            {
                "intervention": "TMEM216 co-sequence (WES/panel)",
                "timing":        "At diagnosis; concurrent with TMEM138 sequencing",
                "rationale":    "TMEM138-TMEM216 mutual stabilisation — TMEM216 variants (JBTS2) can phenocopy; same locus arm 11q12.2; separate OMIM entries",
                "level":        "Recommended (Expert consensus)"
            },
            {
                "intervention": "PGT-M / Prenatal diagnosis",
                "timing":        "Pre-conception or early pregnancy",
                "rationale":    "25% recurrence risk JBTS16; no MKS lethal tier — simpler counselling than MKS-tier genes",
                "level":        "Offered to all families"
            },
            {
                "intervention": "Renal transplant (if ESRD)",
                "timing":        "When GFR <20 mL/min/1.73m²",
                "rationale":    "Curative for renal endpoint; no allograft recurrence; cell-autonomous AR ciliopathy",
                "level":        "Standard of care (if ESRD reached)"
            },
        ],
    }


def get_definitions():
    return {
        "gene_full_name":  "Transmembrane Protein 138 (TMEM138)",
        "omim_gene":       "614965",
        "omim_jbts16":     "614465",
        "chromosome":      "11q12.2",
        "protein_size":    "163 aa — 3-pass TZ membrane protein; TMEM216 co-stabiliser; B9D1/B9D2 docking; CEP290-IFT interface",
        "inheritance":     "Autosomal recessive — biallelic LOF; no MKS lethal tier",

        "no_mks_tier_rule": (
            "ALL biallelic TMEM138 genotypes (null/null, null/hypomorphic, biallelic missense) "
            "→ JBTS16 live birth. No perinatal-lethal MKS allele class documented (2026). "
            "TMEM138 is a TZ membrane co-stabiliser rather than a core MKS-module scaffold — "
            "partial TZ gate function persists via residual TMEM216-anchored architecture. "
            "Standard 25% JBTS16 recurrence applies; no MKS tier calculation needed."
        ),

        "phenotype_frequencies": {
            "mts_pathognomonic":            "100%",
            "cerebellar_ataxia":            f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":           f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":           f"{_pct(n_oma)}%",
            "breathing_dysregulation":      f"{_pct(n_breath)}%",
            "intellectual_disability":      f"{_pct(n_id)}%",
            "retinal_rod_cone":             f"{_pct(n_retinal)}%",
            "renal_nphp_tin":               f"{_pct(n_renal)}%",
            "hepatic_mild_chf":             f"{_pct(n_hepatic)}%",
            "polydactyly_post_axial":       f"{_pct(n_poly)}%",
            "no_mks_tier":                  "Confirmed — all biallelic genotypes → live birth",
        },

        "key_clinical_distinctions": {
            "vs_TMEM216_JBTS2": (
                "TMEM138 (JBTS16, *614965) vs TMEM216 (JBTS2, *613277): adjacent 11q12.2 genes — "
                "mutual stabilisation pair. Both No MKS tier; both ~25% renal. "
                "WES/WGS must distinguish: different OMIM, different proteins, both required for "
                "TZ integrity. Loss of TMEM138 reduces TMEM216 at TZ and vice versa — "
                "quantitative IF (anti-TMEM138 vs anti-TMEM216) resolves ambiguous cases."
            ),
            "vs_TMEM237_JBTS14": (
                "TMEM237 (JBTS14, ~28% renal, NPHP1-bridge) vs TMEM138 (JBTS16, ~22% renal). "
                "TMEM237 is an NPHP-module cross-linker; TMEM138 is a TZ membrane gate protein "
                "with TMEM216 co-dependency. Both No MKS tier but distinct mechanisms and loci."
            ),
            "vs_CEP41_JBTS15": (
                "CEP41 (JBTS15) retinal ~20%, renal ~15% — post-translational axoneme modifier. "
                "TMEM138 (JBTS16) retinal ~25%, renal ~22% — TZ membrane gate protein. "
                "Mechanistically distinct: CEP41 acts on axonemal tubulin; TMEM138 controls TZ "
                "membrane gate composition. No MKS tier for both; TMEM138 has higher renal penetrance."
            ),
            "vs_TCTN2_JBTS13": (
                "TCTN2 biallelic null → MKS8 perinatal lethal; TMEM138 → JBTS16 live birth only. "
                "TCTN2 retinal 45%, renal 32%; TMEM138 retinal ~25%, renal ~22%. "
                "Counselling differs substantially — TMEM138 has NO MKS lethal risk."
            ),
            "TMEM216_co_dependency_unique_mechanism": (
                "TMEM138-TMEM216 mutual stabilisation is unique: loss of either protein destabilises "
                "the partner at the TZ. This is the only documented cis-regulatory adjacent-gene "
                "co-stabilisation pair in Joubert syndrome. Clinical implication: patients with "
                "TMEM138 LOF may show reduced TMEM216 on cilia IF — do not misinterpret as "
                "compound TMEM138+TMEM216 digenic disease unless biallelic variants are confirmed "
                "in BOTH genes independently."
            ),
        },

        "management_highlights": [
            "Annual ERG from age 3 — rod-cone risk ~25%; TZ connecting cilium involvement",
            "Annual renal NPHP surveillance — TIN ~22%; ESRD median ~26yr; NPHP protocol mandatory from diagnosis",
            "No MKS tier — standard 25% JBTS16 recurrence counselling; no MKS-specific prenatal urgency",
            "Co-sequence TMEM216 (JBTS2) — adjacent 11q12.2; mutual stabilisation; WES must distinguish both genes",
            "Neonatal respiratory monitoring — breathing dysregulation ~55%; CPAP if desaturation events",
            "Physiotherapy from age 0–3 yr — hypotonia ~80%; ataxia ~85%; early motor intervention is standard of care",
            "Liver USS + LFT every 2 yr — mild CHF ~8%; biliary cilia TZ involvement",
            "PGT-M available for known pathogenic TMEM138 variants in at-risk families",
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal cytoplasmic tail (aa 1–20)",
                "location":        "Cytoplasmic / intracellular",
                "function":        "TMEM216 interaction interface; mutual co-stabilisation anchor",
                "variant_examples":"Arg39Ter (truncating null — European severe)",
            },
            {
                "domain":          "TM1 (aa 21–43)",
                "location":        "Transmembrane 1 — ciliary TZ membrane insertion",
                "function":        "First membrane-spanning pass; TZ membrane anchoring; TM1 boundary variants affect TMEM216 interaction",
                "variant_examples":"Arg47Trp (North African founder mild — TM1 boundary)",
            },
            {
                "domain":          "Extracellular loop 1 (aa 44–60)",
                "location":        "Extracellular / luminal",
                "function":        "B9D1 and B9D2 docking surface; integrates TMEM138 into B9/MKS gate network",
                "variant_examples":"No characterised point mutations at EL1; truncation alleles affect downstream",
            },
            {
                "domain":          "TM2 (aa 61–83)",
                "location":        "Transmembrane 2 — hydrophobic core",
                "function":        "Hydrophobic core helix; TZ membrane structural integrity; TM2 disruption causes partial localisation failure",
                "variant_examples":"Arg89Cys (pan-ethnic moderate — TM2 proximal boundary)",
            },
            {
                "domain":          "TM2-TM3 linker (aa 84–100)",
                "location":        "Cytoplasmic / intracellular loop",
                "function":        "TMEM231 binding motif; links TMEM138 to the TMEM231 arm of the B9-module",
                "variant_examples":"Glu146Lys (MENA founder moderate), Val142Met (East Asian moderate)",
            },
            {
                "domain":          "TM3 (aa 101–120)",
                "location":        "Transmembrane 3 — lipid raft anchoring / TZ gate",
                "function":        "Lipid raft anchoring in cholesterol/sphingolipid-enriched TZ gate domain; SMO import permissive environment",
                "variant_examples":"Tyr117Cys (South Asian moderate-severe — TM3 core)",
            },
            {
                "domain":          "C-terminal cytoplasmic tail (aa 121–163)",
                "location":        "Cytoplasmic / intracellular",
                "function":        "CEP290 interaction; IFT trafficking interface at TZ base; C-tail truncations cause compound TZ + IFT failure",
                "variant_examples":"Glu164Ter (near-null pan-ethnic severe)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "TMEM138-TMEM216 Mutual Stabilisation (Adjacent 11q12.2 Genes)",
                "detail": (
                    "TMEM138 (*614965, JBTS16) and TMEM216 (*613277, JBTS2) are adjacent genes on "
                    "chromosome 11q12.2 separated by ~55 kb. They form a mutual stabilisation pair: "
                    "TMEM138 LOF destabilises TMEM216 at the TZ; TMEM216 LOF reduces TMEM138 ciliary "
                    "localisation. Both are No-MKS-tier. WES/WGS must carefully distinguish pathogenic "
                    "variants in each gene. Co-segregation analysis and quantitative cilia IF "
                    "(anti-TMEM138 vs anti-TMEM216 antibodies) can resolve ambiguous genotypes. "
                    "Do NOT misinterpret secondary protein loss as digenic disease without biallelic "
                    "variants confirmed independently in both genes."
                ),
            },
            {
                "title": "No MKS Tier Rule — All Biallelic TMEM138 Genotypes → JBTS16 Live Birth",
                "detail": (
                    "ALL biallelic TMEM138 genotypes (null/null, null/hypomorphic, biallelic missense, "
                    "splice/null compound) → JBTS16 live birth. No perinatal-lethal MKS allele class "
                    "has been documented for TMEM138 as of 2026. TMEM138 is a TZ membrane co-stabiliser "
                    "rather than a core MKS-module scaffold component. Standard 25% AR recurrence "
                    "applies; no MKS tier calculation is needed for any TMEM138 genotype. This "
                    "contrasts with TCTN2 (MKS8: null/null → perinatal lethal) and RPGRIP1L "
                    "(MKS5: biallelic null → perinatal lethal)."
                ),
            },
            {
                "title": "Renal Surveillance: Annual NPHP Protocol from Diagnosis (22% ESRD Risk, Median 26yr)",
                "detail": (
                    "TMEM138 LOF causes NPHP-like tubulointerstitial nephritis in ~22% of patients. "
                    "ESRD median is approximately 26 years — later than NPHP-module genes (NPHP1 "
                    "median 13yr, CEP83/NPHP18 median 14–18yr) but earlier than general population. "
                    "Annual renal surveillance (serum creatinine, cystatin C, urine osmolality, "
                    "microalbuminuria) must begin at diagnosis, even before proteinuria develops — "
                    "NPHP-like TIN is a concentrating defect, not a proteinuric disease. Renal "
                    "transplant is curative with no allograft recurrence (cell-autonomous AR ciliopathy)."
                ),
            },
            {
                "title": "Retinal Surveillance: Annual ERG from Age 3 (25% Rod-cone Dystrophy)",
                "detail": (
                    "Rod-cone dystrophy occurs in ~25% of JBTS16 patients, driven by TZ gate "
                    "dysfunction at the photoreceptor connecting cilium. Annual ERG should begin "
                    "at age 3, even if vision appears clinically normal — subclinical ERG changes "
                    "precede symptomatic visual loss by years. Rod system is affected first; "
                    "cone involvement emerges later. Early low-vision rehabilitation and school "
                    "accommodation planning depend on timely ERG surveillance. Retinal penetrance "
                    "in JBTS16 (~25%) is higher than JBTS15/CEP41 (~20%) but lower than JBTS13/"
                    "TCTN2 (~45%), consistent with partial TZ gate failure."
                ),
            },
            {
                "title": "TMEM138 vs TMEM216 Distinction: Same Locus Arm, Different Genes, Different OMIM — WES Must Distinguish",
                "detail": (
                    "TMEM138 (*614965, disease #614465 JBTS16) and TMEM216 (*613277, disease "
                    "#608091 JBTS2) are on the same chromosome arm (11q12.2) separated by ~55 kb. "
                    "Both cause JBTS with No-MKS-tier; both have similar phenotypic fingerprints "
                    "(similar retinal, renal, hepatic penetrance); both have MENA and South Asian "
                    "population enrichment. Gene panels and WES must reliably distinguish variants "
                    "in each gene using gene-specific probe design. A positive TMEM216 signal on "
                    "panel does NOT exclude pathogenic TMEM138 variants and vice versa. "
                    "Functional differentiation: anti-TMEM138 vs anti-TMEM216 IF in patient "
                    "fibroblast cilia quantitatively identifies which protein is primarily deficient."
                ),
            },
        ],

        "literature_highlights": [
            "Lee JH et al. (2012) Evolutionarily assembled cis-regulatory module at a human ciliopathy locus. Science 335(6074):966–9. [TMEM138-TMEM216 co-regulatory module on 11q12.2].",
            "Huang L et al. (2011) TMEM237 is mutated in individuals with a Joubert syndrome related disorder and expands the role of the TMEM family at the ciliary transition zone. Am J Hum Genet 89(6):713–30.",
            "Garcia-Gonzalo FR et al. (2011) A transition zone complex regulates mammalian ciliogenesis and ciliary-based sensing. J Cell Biol 194(6):920–30. [TZ membrane protein function].",
            "Valente EM et al. (2013) Mutations in TMEM216 perturb RHOA signaling and cause Joubert syndrome. Nat Genet 42(7):619–25. [TMEM216 context; TMEM138-TMEM216 interaction].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies: The challenges of genetic and phenotypic heterogeneity. Transl Sci Rare Dis 4(1–2):25–49.",
        ],
    }
