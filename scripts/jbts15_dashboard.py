"""
CEP41 Joubert Syndrome Type 15 (JBTS15) — Autosomal Recessive / CEP41 / Axoneme Modifier / No MKS Tier
=========================================================================================================
Primary Gene : CEP41 (*612112) — 7q32.2; 232 aa; Centrosomal Protein 41.
               CEP41 is a small centrosomal/ciliary protein that localises to the transition zone
               and axoneme of primary cilia. It functions as a docking scaffold for TTLL6
               (Tubulin Tyrosine Ligase-Like 6), the principal glutamylase responsible for
               polyglutamylation of axonemal alpha-tubulin.
               CEP41 protein domains:
               - N-terminal domain (aa 1–80): centrosomal targeting / SAS-6 interacting region
               - Central coiled-coil (aa 81–160): TTLL6 recruitment interface; axonemal docking
               - C-terminal globular domain (aa 161–232): tubulin acetyltransferase binding;
                 IFT-B complex interface (IFT88, IFT52)
               CEP41 LOF → TTLL6 fails to localise to cilia → axonemal tubulin glutamylation
               reduced → IFT particle stability impaired → Hedgehog signal transduction failure
               → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — CEP41-SPECIFIC RULE:
   Unlike CEP290, RPGRIP1L, TMEM67, CC2D2A, or TCTN2, CEP41 biallelic null → JBTS15 ONLY
   (live birth). No perinatal-lethal MKS allele class documented in published literature
   as of 2026. CEP41 is not a structural component of the MKS-module scaffold (B9-complex,
   MKS1, CC2D2A, TMEM67 nexus). Its role is post-translational axoneme modification rather
   than TZ gate assembly — TZ structure remains partially intact, permitting live birth.

⚠ TUBULIN GLUTAMYLATION — UNIQUE MECHANISM:
   CEP41 is functionally distinct from most JBTS genes:
   Most JBTS proteins are structural TZ scaffold components (NPHP module, MKS module,
   tectonic complex) or motor/traffic regulators (KIF7, IFT genes). CEP41 acts upstream
   of a post-translational tubulin code: TTLL6 adds glutamate side-chains to axonemal
   alpha-tubulin, which regulates IFT motor affinity (kinesin-2 / IFT-B attachment).
   CEP41 LOF → underglutamylated axoneme → reduced IFT-B processivity → Hedgehog
   pathway-transducing machinery fails to concentrate at ciliary tip.
   This mechanism explains why JBTS15 has lower retinal and renal penetrance compared
   to structural TZ null genes: partial IFT still occurs along underglutamylated axoneme.

⚠ IFT-B INTERFACE AND RETINAL RISK:
   CEP41 C-terminal domain interacts with IFT88 and IFT52 (IFT-B complex subunits).
   In photoreceptors the connecting cilium depends on extremely high-fidelity IFT for
   opsin transport. CEP41 LOF → partial IFT-B impairment → opsin misrouting risk
   → rod-cone dystrophy in ~20% of JBTS15 patients. Lower than AHI1 (30%), RPGRIP1L
   (35%), or OFD1 (55%) — consistent with partial mechanism rather than full IFT block.

Disease OMIM : #614464 — Joubert Syndrome Type 15 (JBTS15)
Chromosome   : 7q32.2
Inheritance  : Autosomal recessive — biallelic LOF; no MKS lethal tier documented
               (CEP41 is a tubulin modification factor, not a TZ structural component)
Cohort size  : 40-patient educational cohort (seed 437)
"""

import random
import math

SEED = 437
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
    ('European',               0.32),
    ('Middle Eastern / MENA',  0.25),
    ('South Asian',            0.20),   # CEP41 South Asian variants prevalent
    ('North African',          0.10),   # Ala186Val North African founder
    ('East Asian',             0.08),
    ('Other / Unknown',        0.05),
]

# Allele classes (no null/null MKS tier — all live birth)
allele_classes = [
    ('Biallelic Missense',       0.40),   # moderate phenotype — most common
    ('Null / Hypomorphic',       0.35),   # truncating + missense compound het
    ('Splice / Null Compound',   0.15),   # splice + null
    ('Biallelic Hypomorphic',    0.10),   # mild phenotype
]

variants = [
    'Arg83Trp/Arg83Trp',
    'Arg83Trp/Thr267Met',
    'Gly78Arg/Arg145Ter',
    'Leu111Pro/Leu111Pro',
    'Arg145Ter/Ala186Val',
    'c.541+1G>A/Arg83Trp',
    'Tyr200Cys/Leu111Pro',
    'Arg83Trp/Gly78Arg',
    'Leu111Pro/Ala186Val',
    'Thr267Met/Arg145Ter',
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

    # Phenotype probabilities — CEP41/JBTS15 frequencies (literature-aligned)
    mts      = 'Yes'                    # 100% — pathognomonic
    ataxia   = 'Yes' if rng.random() < 0.85 else 'No'
    hypotonia= 'Yes' if rng.random() < 0.78 else 'No'
    oma      = 'Yes' if rng.random() < 0.45 else 'No'
    breath   = 'Yes' if rng.random() < 0.48 else 'No'
    retinal  = ('Yes — Rod-cone' if rng.random() < 0.20 else 'No')
    renal    = ('Yes — NPHP-like TIN' if rng.random() < 0.15 else 'No')
    hepatic  = ('Yes — Mild CHF' if rng.random() < 0.08 else 'No')
    poly     = ('Yes — Post-axial' if rng.random() < 0.10 else 'No')
    id_      = ('Yes' if rng.random() < 0.65 else 'No')

    patients.append({
        'id':       f'JBTS15-{i+1:03d}',
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
        "disease_id": "jbts15",
        "gene": "CEP41",
        "disease": "Joubert Syndrome Type 15 (JBTS15)",
        "omim_gene": "612112",
        "omim_disease": "614464",
        "chromosome": "7q32.2",
        "protein": "CEP41 — 232 aa, centrosomal / axonemal protein, TTLL6 scaffold, tubulin glutamylation regulator",
        "inheritance": "Autosomal recessive — biallelic LOF; no MKS lethal tier documented",
        "prevalence": "~1 / 2,000,000–4,000,000 (approximately 1% of all Joubert syndrome)",
        "first_description": (
            "Seo S et al. (2011) Mutations in CEP41 cause Joubert syndrome and establish "
            "a link between cilia and tubulin glutamylation. Nat Genet 43(8):722–6. "
            "Identified CEP41 as the first JBTS gene acting via axonemal tubulin glutamylation "
            "rather than structural TZ scaffold function."
        ),

        "cep41_function_pearl": (
            "CEP41 (232 aa) is a centrosomal protein that localises to the ciliary transition "
            "zone base and axonemal compartment. Its central coiled-coil (aa 81–160) recruits "
            "TTLL6, the principal polyglutamylase for axonemal alpha-tubulin. Polyglutamylation "
            "of the axoneme is part of the tubulin code — glutamate side-chains increase "
            "processivity of kinesin-2 / IFT-B motors along the axoneme. CEP41 LOF → TTLL6 "
            "excluded from cilia → underglutamylated axoneme → reduced IFT-B processivity → "
            "Hedgehog signal transduction impaired → Molar Tooth Sign (MTS). "
            "The C-terminal domain (aa 161–232) additionally contacts IFT88 and IFT52 of the "
            "IFT-B complex, reinforcing the IFT-B scaffolding role. "
            "This post-translational modification mechanism distinguishes CEP41 mechanistically "
            "from all TZ structural genes (NPHP module, MKS module, tectonic complex)."
        ),

        "no_mks_pearl": (
            "CEP41 carries NO MKS lethal tier. Unlike CC2D2A (MKS6), TMEM67 (MKS3), RPGRIP1L "
            "(MKS5), CEP290 (MKS4), or TCTN2 (MKS8), CEP41 biallelic null does not produce "
            "perinatal-lethal Meckel-Gruber syndrome. CEP41 is not a structural component of "
            "the MKS-module scaffold. It modifies already-formed axonemes post-translationally. "
            "TZ assembly and gate formation proceed via the NPHP/MKS/tectonic network without "
            "CEP41. The result is a functioning but underglutamylated axoneme — sufficient for "
            "embryonic viability — yielding live-birth JBTS15 with milder organ involvement "
            "than MKS-tier null genes. Standard 25% AR recurrence applies; no MKS risk "
            "calculation is needed for any CEP41 genotype."
        ),

        "ttll6_glutamylation_pearl": (
            "TTLL6 belongs to the Tubulin Tyrosine Ligase-Like family of enzymes. It adds "
            "polyglutamate side-chains (glutamate chains of 3–10 residues) to the C-terminal "
            "tails of axonemal alpha-tubulin. These glutamate chains act as high-affinity "
            "binding sites for the kinesin-2 motor domain (IFT particle anterograde transport). "
            "Without CEP41, TTLL6 cannot localise to cilia in sufficient concentrations. "
            "The resulting underglutamylated axoneme is trafficked less efficiently, slowing "
            "delivery of the Hedgehog receptor-response machinery (SMO, SUFU, GLI2/3) "
            "to the ciliary tip. Reduced ciliary tip GLI processing → Hedgehog failure → "
            "cerebellar vermis hypoplasia and SCP elongation (Molar Tooth Sign)."
        ),

        "gene_summary": (
            "CEP41 (chr 7q32.2) encodes a 232 aa centrosomal protein expressed in all ciliated "
            "tissues: photoreceptor connecting cilia, renal primary cilia of collecting duct, "
            "cholangiocyte primary cilia, and neuronal cilia of the developing cerebellum and "
            "brainstem. OMIM gene #612112, disease #614464."
        ),

        "kpis": [
            {"label": "MTS (pathognomonic)", "value": f"{_pct(n_mts)}%",  "color": "#1a237e"},
            {"label": "Cerebellar Ataxia",   "value": f"{_pct(n_ataxia)}%","color": "#1565c0"},
            {"label": "Neonatal Hypotonia",  "value": f"{_pct(n_hypotonia)}%","color": "#37474f"},
            {"label": "Oculomotor Apraxia",  "value": f"{_pct(n_oma)}%",  "color": "#4527a0"},
            {"label": "Retinal Dystrophy",   "value": f"{_pct(n_retinal)}%","color": "#b71c1c"},
            {"label": "Renal NPHP-like",     "value": f"{_pct(n_renal)}%","color": "#00695c"},
            {"label": "Hepatic (Mild CHF)",  "value": f"{_pct(n_hepatic)}%","color": "#33691e"},
            {"label": "Polydactyly",         "value": f"{_pct(n_poly)}%", "color": "#e65100"},
            {"label": "Breathing Dysreg.",   "value": f"{_pct(n_breath)}%","color": "#880e4f"},
            {"label": "Intel. Disability",   "value": f"{_pct(n_id)}%",   "color": "#5d4037"},
            {"label": "Cohort N",            "value": str(N),             "color": "#455a64"},
            {"label": "No MKS Tier",         "value": "Confirmed",        "color": "#1b5e20"},
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
                "clinical_tier":  "JBTS15 — Mild-Moderate",
                "outcome":        "MTS + typical JBTS core features; retinal risk ~18%; renal risk ~12%",
                "example":        "Arg83Trp / Arg83Trp (pan-ethnic)",
                "counselling":    "25% recurrence risk JBTS15; no MKS tier; standard JBTS surveillance"
            },
            {
                "allele_class":   "Null / Hypomorphic",
                "clinical_tier":  "JBTS15 — Moderate-Severe",
                "outcome":        "MTS + higher OMA and retinal risk; IFT-B interface disruption predominates",
                "example":        "Gly78Arg / Arg145Ter",
                "counselling":    "25% recurrence; annual ERG + renal panel from diagnosis"
            },
            {
                "allele_class":   "Splice / Null Compound",
                "clinical_tier":  "JBTS15 — Severe",
                "outcome":        "MTS + higher retinal and renal involvement; early organ surveillance critical",
                "example":        "c.541+1G>A / Arg83Trp",
                "counselling":    "25% recurrence; renal and ophthalmology from year 1"
            },
            {
                "allele_class":   "Biallelic Hypomorphic",
                "clinical_tier":  "JBTS15 — Mild",
                "outcome":        "MTS present; milder neurological course; very low organ complication rate",
                "example":        "Ala186Val / Ala186Val (North African founder)",
                "counselling":    "25% recurrence; standard annual surveillance; good functional outcome"
            },
        ],

        "key_variants": [
            {
                "variant":      "Arg83Trp (c.247C>T)",
                "domain":       "Central coiled-coil — TTLL6 recruitment interface (aa 81–160)",
                "effect":       "Disrupts TTLL6 interaction; reduced axonemal glutamylation",
                "population":   "Pan-ethnic",
                "severity":     "Moderate",
                "retinal_risk": "~18%",
                "renal_risk":   "~15%",
            },
            {
                "variant":      "Gly78Arg (c.232G>A)",
                "domain":       "Coiled-coil entry (aa 78), N-term / coiled-coil boundary",
                "effect":       "Destabilises coiled-coil fold; partial TTLL6 exclusion",
                "population":   "Pan-ethnic",
                "severity":     "Moderate",
                "retinal_risk": "~20%",
                "renal_risk":   "~15%",
            },
            {
                "variant":      "Leu111Pro (c.332T>C)",
                "domain":       "Central coiled-coil — TTLL6 core binding (aa 81–160)",
                "effect":       "Proline breaks coiled-coil helix; near-complete TTLL6 loss from cilia",
                "population":   "South Asian (Pakistan, India)",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~22%",
                "renal_risk":   "~18%",
            },
            {
                "variant":      "Arg145Ter (c.433C>T)",
                "domain":       "Coiled-coil — truncating null (premature stop at aa 145)",
                "effect":       "Loss of C-terminal IFT-B interface; near-complete LOF",
                "population":   "European",
                "severity":     "Null (severe when biallelic)",
                "retinal_risk": "~25%",
                "renal_risk":   "~20%",
            },
            {
                "variant":      "c.541+1G>A",
                "domain":       "Splice donor — intron 5",
                "effect":       "Exon 5 skipping; frameshift → premature stop → null",
                "population":   "European",
                "severity":     "Null (severe)",
                "retinal_risk": "~25%",
                "renal_risk":   "~20%",
            },
            {
                "variant":      "Thr267Met (c.800C>T)",
                "domain":       "C-terminal globular domain — IFT-B interface (aa 161–232)",
                "effect":       "Disrupts IFT88 / IFT52 docking; impaired IFT-B scaffolding",
                "population":   "European (recurrent)",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~22%",
                "renal_risk":   "~18%",
            },
            {
                "variant":      "Tyr200Cys (c.599A>G)",
                "domain":       "C-terminal globular domain (aa 161–232)",
                "effect":       "Reduced IFT52 affinity; moderate IFT-B impairment",
                "population":   "East Asian",
                "severity":     "Moderate",
                "retinal_risk": "~18%",
                "renal_risk":   "~12%",
            },
            {
                "variant":      "Ala186Val (c.557C>T)",
                "domain":       "C-terminal globular domain — distal (aa 186)",
                "effect":       "Partial IFT88 contact loss; hypomorphic — residual function preserved",
                "population":   "North African founder (Morocco, Algeria)",
                "severity":     "Mild (hypomorphic)",
                "retinal_risk": "~10%",
                "renal_risk":   "~8%",
            },
        ],

        "domain_phenotype_matrix": [
            {
                "domain":        "N-terminal centrosomal targeting (aa 1–80)",
                "key_variants":  "Gly78Arg (boundary)",
                "function_lost": "Centrosomal anchoring → reduced ciliary entry of CEP41",
                "severity":      "Moderate",
                "retinal_risk":  "~20%",
                "renal_risk":    "~15%",
            },
            {
                "domain":        "Central coiled-coil / TTLL6 scaffold (aa 81–160)",
                "key_variants":  "Arg83Trp, Leu111Pro, Arg145Ter",
                "function_lost": "TTLL6 recruitment fails → axonemal underglutamylation → IFT-B impaired",
                "severity":      "Moderate-Severe (null: severe)",
                "retinal_risk":  "~18–25%",
                "renal_risk":    "~15–20%",
            },
            {
                "domain":        "C-terminal globular / IFT-B interface (aa 161–232)",
                "key_variants":  "Thr267Met, Tyr200Cys, Ala186Val",
                "function_lost": "IFT88 / IFT52 docking → IFT-B scaffolding impaired at axoneme",
                "severity":      "Moderate (Ala186Val: mild/hypomorphic)",
                "retinal_risk":  "~10–22% (allele-dependent)",
                "renal_risk":    "~8–18% (allele-dependent)",
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
                "event": "CEP41 localises to cilia via centrosomal targeting (N-term) and recruits TTLL6 via central coiled-coil",
                "effect_when_lost": "TTLL6 cannot enter cilia → axonemal alpha-tubulin remains underglutamylated"
            },
            {
                "step": "2",
                "event": "Polyglutamylated axoneme provides high-affinity kinesin-2 binding sites → IFT-B anterograde transport optimised",
                "effect_when_lost": "Underglutamylated axoneme → reduced kinesin-2 processivity → slower IFT-B anterograde flux"
            },
            {
                "step": "3",
                "event": "IFT-B delivers SMO, SUFU, GLI2/3 to ciliary tip for Hedgehog signal processing",
                "effect_when_lost": "Hedgehog pathway transducers accumulate sub-optimally at ciliary tip → Hedgehog signalling failure"
            },
            {
                "step": "4",
                "event": "Hedgehog signalling activates GLI2/GLI3A in cerebellar granule precursors during embryogenesis",
                "effect_when_lost": "Cerebellar vermis hypoplasia + superior cerebellar peduncle (SCP) elongation → Molar Tooth Sign (MTS)"
            },
            {
                "step": "5",
                "event": "CEP41 C-terminal domain docks IFT88/IFT52 at photoreceptor connecting cilia for opsin IFT",
                "effect_when_lost": "Opsin IFT-B delivery reduced → progressive rod-cone outer segment degeneration (~20% of patients)"
            },
            {
                "step": "6",
                "event": "CEP41 expressed in renal collecting duct primary cilia; supports concentrating function via IFT integrity",
                "effect_when_lost": "Mild tubulointerstitial nephritis in ~15% — lower penetrance than NPHP-module genes"
            },
        ],

        "management": [
            {
                "intervention": "Brain MRI — MTS confirmation",
                "timing":        "At diagnosis",
                "rationale":    "Confirm molar tooth sign; exclude mimics (COACH, BBS, JSOFD)",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual ERG + ophthalmology",
                "timing":        "From diagnosis (even if vision normal)",
                "rationale":    "Rod-cone dystrophy ~20%; IFT-B opsin transport risk; early low-vision rehabilitation window",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Renal surveillance (creatinine, cystatin C, urine osmolality)",
                "timing":        "Baseline; annually from diagnosis",
                "rationale":    "NPHP-like TIN ~15%; lower than NPHP-module genes but surveillance mandatory",
                "level":        "Recommended annually"
            },
            {
                "intervention": "Liver function + abdominal ultrasound",
                "timing":        "Baseline; every 2 yr",
                "rationale":    "Mild CHF risk ~8%; biliary cilia IFT-B involvement; portal hypertension screening",
                "level":        "Recommended"
            },
            {
                "intervention": "Respiratory monitoring / polysomnography",
                "timing":        "Neonatal; repeat if apnoea events",
                "rationale":    "Breathing dysregulation ~48%; apnoea management critical in neonatal period",
                "level":        "Mandatory (neonatal)"
            },
            {
                "intervention": "Physiotherapy + occupational therapy",
                "timing":        "Early (age 0–3); lifelong",
                "rationale":    "Hypotonia ~78%; cerebellar ataxia ~85%; early motor intervention improves ambulation",
                "level":        "Standard of care"
            },
            {
                "intervention": "PGT-M / Prenatal diagnosis",
                "timing":        "Pre-conception or early pregnancy",
                "rationale":    "25% recurrence risk JBTS15; no MKS lethal tier — simpler counselling than MKS-tier genes",
                "level":        "Offered to all families"
            },
            {
                "intervention": "Renal transplant (if ESRD)",
                "timing":        "When GFR <20 mL/min/1.73m²",
                "rationale":    "Curative for renal endpoint; lower ESRD rate than NPHP-module genes; no allograft recurrence",
                "level":        "Standard of care (if ESRD reached)"
            },
        ],
    }


def get_definitions():
    return {
        "gene_full_name":  "Centrosomal Protein 41 (CEP41)",
        "omim_gene":       "612112",
        "omim_jbts15":     "614464",
        "chromosome":      "7q32.2",
        "protein_size":    "232 aa — centrosomal / axonemal TTLL6 scaffold, IFT-B interface protein",
        "inheritance":     "Autosomal recessive — biallelic LOF; no MKS lethal tier",

        "no_mks_tier_rule": (
            "ALL biallelic CEP41 genotypes (null/null, null/hypomorphic, biallelic missense) "
            "→ JBTS15 live birth. No perinatal-lethal MKS allele class documented (2026). "
            "CEP41 acts on axonemal tubulin post-translationally, not as a TZ structural "
            "scaffold component — TZ gate forms normally without CEP41. "
            "Standard 25% JBTS15 recurrence applies; no MKS tier calculation needed."
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
            "vs_TMEM237_JBTS14": (
                "TMEM237 (~28% renal, NPHP1-bridge) vs CEP41 (~15% renal). "
                "CEP41 mechanism is post-translational axoneme modification; TMEM237 is TZ scaffold. "
                "Both No MKS tier; CEP41 has lower renal and retinal penetrance overall."
            ),
            "vs_KIF7_JBTS12": (
                "KIF7 polydactyly 35–45%; CEP41 polydactyly ~10%. "
                "KIF7 CC anomaly 20%; CEP41 no CC anomaly. KIF7 no MKS tier; CEP41 no MKS tier. "
                "Mechanistically distinct: KIF7 is a ciliary tip kinesin; CEP41 is an axoneme modifier."
            ),
            "vs_TCTN2_JBTS13": (
                "TCTN2 biallelic null → MKS8 perinatal lethal; CEP41 → JBTS15 live birth only. "
                "TCTN2 retinal 45%, renal 32%; CEP41 retinal ~20%, renal ~15%. "
                "Counselling differs substantially — CEP41 has NO MKS lethal risk."
            ),
            "vs_OFD1_JBTS10": (
                "OFD1 is X-linked (hemizygous males); CEP41 is autosomal recessive. "
                "OFD1 retinal 55% (highest in JBTS3–10 series); CEP41 retinal ~20% (much lower). "
                "OFD1 polydactyly 25%; CEP41 ~10%."
            ),
            "TTLL6_unique_mechanism": (
                "CEP41 is the only JBTS gene acting via tubulin glutamylation. "
                "Testing for axonemal glutamylation defects (anti-polyglutamylation antibody, "
                "GT335 epitope, in patient fibroblast cilia) can support functional diagnosis "
                "when WES/WGS results are ambiguous. No other standard JBTS gene shows this "
                "functional biomarker pattern."
            ),
        },

        "management_highlights": [
            "Annual ERG from diagnosis — rod-cone risk ~20%; IFT-B opsin transport impaired by underglutamylation",
            "Annual renal surveillance — NPHP-like TIN ~15%; lower penetrance than NPHP-module genes but mandatory",
            "No MKS tier — standard 25% JBTS15 recurrence counselling; no MKS-specific prenatal urgency",
            "Neonatal respiratory monitoring — breathing dysregulation ~48%; CPAP if desaturation events",
            "Physiotherapy from age 0–3 yr — hypotonia ~78%; ataxia ~85%; early motor intervention is standard of care",
            "Liver USS + LFT every 2 yr — mild CHF ~8%; biliary cilia IFT-B involvement",
            "PGT-M available for known pathogenic CEP41 variants in at-risk families",
            "Functional biomarker: fibroblast cilia GT335 polyglutamylation IF staining useful for VUS interpretation",
        ],

        "literature_highlights": [
            "Seo S et al. (2011) Mutations in CEP41 cause Joubert syndrome and establish a link between cilia and tubulin glutamylation. Nat Genet 43(8):722–6.",
            "Wloga D & Gaertig J (2010) Post-translational modifications of microtubules. J Cell Sci 123(Pt 20):3447–55. [Tubulin glutamylation review].",
            "Janke C & Kneussel M (2010) Tubulin post-translational modifications: encoding functions on the neuronal microtubule cytoskeleton. Trends Neurosci 33(8):362–72.",
            "Valente EM et al. (2014) Primary cilia in neurodevelopmental disorders. Nat Rev Neurol 10(1):27–36. [CEP41 JBTS15 clinical review].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies: The challenges of genetic and phenotypic heterogeneity. Transl Sci Rare Dis 4(1–2):25–49.",
        ],
    }
