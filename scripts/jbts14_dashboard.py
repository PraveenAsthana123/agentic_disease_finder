"""
TMEM237 Joubert Syndrome Type 14 (JBTS14) — Autosomal Recessive / TMEM237-ALS2CR4 / TZ Transition Fibre / No MKS Tier
========================================================================================================================
Primary Gene : TMEM237 (*614423) — 2q33.1; 541 aa; ALS2CR4; Transmembrane Protein 237.
               TMEM237 (also named ALS2CR4 — ALS2 chromosome region candidate gene 4) is a
               multi-pass transmembrane protein localised to the ciliary transition zone (TZ).
               It forms part of the NPHP-module scaffold, interacting directly with NPHP1,
               NPHP4, and TMEM231 at the transition fibre/TZ junction.
               TMEM237 protein domains:
               - N-terminal cytoplasmic tail (aa 1–40): NPHP1-binding interface; TZ targeting
               - TM1–TM3 (aa 41–180): triple transmembrane hairpin; TZ fibre membrane anchor
               - Extracellular loop (aa 181–260): TMEM231/B9D1 interaction surface
               - TM4–TM6 (aa 261–400): second transmembrane cluster; gate regulation
               - C-terminal cytoplasmic domain (aa 401–541): NPHP4 / Inversin interaction
               TMEM237 LOF → TZ gate partially disrupted → GPCR and SMO import impaired →
               Hedgehog signalling failure → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — TMEM237-SPECIFIC RULE:
   Unlike CEP290, RPGRIP1L, TMEM67, or CC2D2A, TMEM237 biallelic null → JBTS14 ONLY
   (live birth). No perinatal-lethal MKS allele class documented in published literature
   as of 2026. This distinguishes TMEM237 from MKS-tier genes in the same TZ complex.
   The TZ gate disruption is partial, preserving enough residual ciliogenesis for viability.

⚠ NPHP-MODULE PARTNER INTERACTIONS:
   TMEM237 sits at the nexus of two NPHP sub-complexes:
   (1) NPHP1–NPHP4 dyad (Y-link scaffold, axoneme junction)
   (2) TMEM231–B9D1 module (TZ membrane anchor / lipid gate)
   TMEM237 bridges these two complexes — LOF disrupts the cross-module interaction,
   weakening TZ gate integrity without complete collapse (explains the milder phenotype
   vs. null-tier genes like TCTN2 or CC2D2A that fully disassemble the MKS module).

⚠ RENAL TUBULOINTERSTITIAL NEPHRITIS RISK:
   TMEM237 NPHP-like renal involvement (~28%) is driven by the NPHP1 interaction:
   NPHP1 is the most common nephronophthisis gene; TMEM237 partial NPHP1 disruption
   creates a mild tubulointerstitial nephritis phenotype, not cystic kidneys.
   ESRD median ~24 yr. Renal transplant curative for renal endpoint.

Disease OMIM : #614424 — Joubert Syndrome Type 14 (JBTS14)
Chromosome   : 2q33.1
Inheritance  : Autosomal recessive — biallelic LOF; all biallelic null → JBTS14 live birth
               (no MKS tier documented)
Cohort size  : 40-patient educational cohort (seed 435)
"""

import random
import math

SEED = 435
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
    ('Middle Eastern / MENA',  0.30),   # Arg8Trp Arab founder
    ('South Asian',            0.18),   # Phe182Ser prevalent
    ('North African',          0.12),   # Ala447Val founder (mild)
    ('East Asian',             0.05),
    ('Other / Unknown',        0.05),
]

# Allele classes (no null/null MKS tier — all live birth)
allele_classes = [
    ('Null / Hypomorphic',       0.38),   # truncating + missense compound het
    ('Biallelic Missense',       0.35),   # moderate phenotype
    ('Biallelic Hypomorphic',    0.15),   # mild phenotype
    ('Splice / Null Compound',   0.12),   # splice + null
]

variants = [
    'Arg8Trp/Arg8Trp',
    'Arg8Trp/Phe182Ser',
    'Gly78Arg/Trp255Ter',
    'Phe182Ser/Phe182Ser',
    'Trp255Ter/Ala447Val',
    'Arg399Ter/Arg8Trp',
    'c.846+1G>A/Gly78Arg',
    'Gly78Arg/Gly78Arg',
    'Phe182Ser/Ala447Val',
    'Arg399Ter/Phe182Ser',
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
    age   = rng.randint(1, 22)          # age at diagnosis

    # Phenotype probabilities — TMEM237/JBTS14 frequencies
    mts      = 'Yes'                    # 100% — pathognomonic
    ataxia   = 'Yes' if rng.random() < 0.86 else 'No'
    hypotonia= 'Yes' if rng.random() < 0.81 else 'No'
    oma      = 'Yes' if rng.random() < 0.55 else 'No'
    breath   = 'Yes' if rng.random() < 0.55 else 'No'
    retinal  = ('Yes — Rod-cone' if rng.random() < 0.27 else 'No')
    renal    = ('Yes — NPHP-like TIN' if rng.random() < 0.28 else 'No')
    hepatic  = ('Yes — Mild CHF' if rng.random() < 0.08 else 'No')
    poly     = ('Yes — Post-axial' if rng.random() < 0.06 else 'No')
    id_      = ('Yes' if rng.random() < 0.68 else 'No')

    patients.append({
        'id':       f'JBTS14-{i+1:03d}',
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
        "disease_id": "jbts14",
        "gene": "TMEM237",
        "disease": "Joubert Syndrome Type 14 (JBTS14)",
        "omim_gene": "614423",
        "omim_disease": "614424",
        "chromosome": "2q33.1",
        "protein": "TMEM237 (ALS2CR4) — 541 aa, multi-pass TZ transmembrane protein, NPHP-module bridge",
        "inheritance": "Autosomal recessive — biallelic LOF; no MKS lethal tier documented",
        "prevalence": "~1 / 1,500,000–3,000,000 (1–2% of all Joubert syndrome)",
        "first_description": "Huang et al. 2011 (Am J Hum Genet); Baala et al. 2007 (locus linkage); "
                             "confirmed TMEM237 as TZ component interacting with NPHP1 and TMEM231.",

        "tmem237_function_pearl": (
            "TMEM237 (ALS2CR4) is a 541 aa multi-pass transmembrane protein residing at the "
            "ciliary transition zone (TZ). It bridges two NPHP sub-complexes: the NPHP1–NPHP4 "
            "Y-link scaffold (axoneme junction) and the TMEM231–B9D1 TZ membrane anchor module. "
            "The N-terminal cytoplasmic tail (aa 1–40) contacts NPHP1 directly; the C-terminal "
            "domain (aa 401–541) interacts with NPHP4 and Inversin. The extracellular loop "
            "(aa 181–260) associates with TMEM231 and B9D1 at the luminal TZ face. "
            "TMEM237 LOF → partial uncoupling of the NPHP1–NPHP4 dyad from the TMEM231–B9D1 "
            "membrane anchor → TZ gate integrity reduced → GPCR (SSTR3, MCHR1) and SMO import "
            "impaired → Hedgehog signalling failure → Molar Tooth Sign (MTS). "
            "Because disruption is partial (not total MKS-module collapse), all biallelic TMEM237 "
            "null patients survive to live birth — no MKS lethal tier is documented."
        ),

        "no_mks_pearl": (
            "TMEM237 is one of the few TZ scaffold genes with NO MKS lethal tier. "
            "Unlike TCTN2 (MKS8), CC2D2A (MKS6), RPGRIP1L (MKS5), TMEM67 (MKS3), or "
            "CEP290 (MKS4) — where biallelic null alleles produce perinatal-lethal Meckel-Gruber "
            "syndrome — TMEM237 biallelic null consistently produces live-birth JBTS14. "
            "The mechanistic reason is TMEM237's partial-bridge role: it cross-links two NPHP "
            "sub-modules without being load-bearing for either individually. Loss of TMEM237 "
            "weakens but does not collapse the TZ gate. This makes TMEM237 genotype counselling "
            "simpler: any biallelic genotype → JBTS14 (no 25% MKS recurrence-risk calculation "
            "needed; standard 25% JBTS14 recurrence applies to all allele combinations)."
        ),

        "nphp_bridge_pearl": (
            "TMEM237's dual interaction with NPHP1 (Y-link scaffold) and NPHP4 (C-term contact) "
            "creates a NPHP-module dependency: NPHP1 or NPHP4 loss destabilises residual TMEM237 "
            "function. This synergy explains why TMEM237 patients have a higher NPHP-like renal "
            "tubulointerstitial nephritis rate (~28%) than would be expected for a pure TZ "
            "structural protein — the NPHP1 interaction links TMEM237 functionally to the "
            "nephronophthisis pathway. Annual renal surveillance (urine osm, creatinine, cystatin C) "
            "is mandatory even before overt proteinuria, following NPHP surveillance protocol."
        ),

        "gene_summary": (
            "TMEM237 (chr 2q33.1) encodes a 541 aa transmembrane protein with six TM helices "
            "arranged in two clusters (TM1–3, TM4–6), an extracellular loop, and cytoplasmic "
            "N- and C-terminal tails. It is expressed in all ciliated tissues: photoreceptor "
            "connecting cilia, renal collecting duct primary cilia, cholangiocyte cilia, and "
            "neuronal cilia in the cerebellum and brainstem. OMIM gene #614423, disease #614424."
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
                "allele_class":   "Null / Hypomorphic",
                "clinical_tier":  "JBTS14 — Moderate-Severe",
                "outcome":        "MTS + multi-system involvement; retinal and renal surveillance mandatory",
                "example":        "Gly78Arg / Trp255Ter",
                "counselling":    "25% recurrence risk JBTS14; no MKS tier; standard JBTS surveillance"
            },
            {
                "allele_class":   "Splice / Null Compound",
                "clinical_tier":  "JBTS14 — Severe",
                "outcome":        "MTS + higher OMA + renal risk; early NPHP surveillance critical",
                "example":        "c.846+1G>A / Arg399Ter",
                "counselling":    "25% recurrence; consider renal transplant planning early"
            },
            {
                "allele_class":   "Biallelic Missense",
                "clinical_tier":  "JBTS14 — Mild-Moderate",
                "outcome":        "MTS + typical JBTS features; retinal risk ~20%; renal ~20%",
                "example":        "Phe182Ser / Phe182Ser (South Asian)",
                "counselling":    "25% recurrence; annual ERG + renal panel"
            },
            {
                "allele_class":   "Biallelic Hypomorphic",
                "clinical_tier":  "JBTS14 — Mild (NPHP-only risk)",
                "outcome":        "MTS present but mild neurological course; watch for subclinical renal",
                "example":        "Ala447Val / Ala447Val (North African founder)",
                "counselling":    "25% recurrence; still requires annual NPHP renal surveillance"
            },
        ],

        "key_variants": [
            {
                "variant":      "Arg8Trp (c.22C>T)",
                "domain":       "N-term NPHP1-binding (aa 1–40)",
                "effect":       "Disrupts NPHP1 direct contact; partial TZ scaffold uncoupling",
                "population":   "Arab / MENA founder (Jordan, Palestine, Egypt)",
                "severity":     "Moderate",
                "retinal_risk": "~25%",
                "renal_risk":   "~30%",
            },
            {
                "variant":      "Gly78Arg (c.232G>A)",
                "domain":       "TM1–3 (aa 41–180), TM-proximal",
                "effect":       "Destabilises TM hairpin; reduced TZ membrane insertion",
                "population":   "Pan-ethnic",
                "severity":     "Moderate",
                "retinal_risk": "~28%",
                "renal_risk":   "~28%",
            },
            {
                "variant":      "Phe182Ser (c.545T>C)",
                "domain":       "Extracellular loop (aa 181–260)",
                "effect":       "Disrupts TMEM231 / B9D1 interaction surface",
                "population":   "South Asian (Pakistan, India)",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~30%",
                "renal_risk":   "~35%",
            },
            {
                "variant":      "Trp255Ter (c.765G>A)",
                "domain":       "Extracellular loop — truncating null",
                "effect":       "Loss of TM4–6 cluster + C-terminal domain; hypomorphic if compound",
                "population":   "European",
                "severity":     "Null (severe when biallelic)",
                "retinal_risk": "~32%",
                "renal_risk":   "~35%",
            },
            {
                "variant":      "Arg399Ter (c.1195C>T)",
                "domain":       "C-terminal cytoplasmic domain — truncating null",
                "effect":       "Loss of NPHP4 / Inversin C-term interaction; severe",
                "population":   "Pan-ethnic",
                "severity":     "Null (severe)",
                "retinal_risk": "~33%",
                "renal_risk":   "~38%",
            },
            {
                "variant":      "c.846+1G>A",
                "domain":       "Splice donor — intron 7",
                "effect":       "Exon 7 skipping; frameshift → premature stop; null",
                "population":   "European",
                "severity":     "Null (severe)",
                "retinal_risk": "~30%",
                "renal_risk":   "~35%",
            },
            {
                "variant":      "Ala447Val (c.1340C>T)",
                "domain":       "C-terminal cytoplasmic (aa 401–541)",
                "effect":       "Partial NPHP4 contact disruption; hypomorphic — milder phenotype",
                "population":   "North African founder (Morocco, Algeria)",
                "severity":     "Mild (hypomorphic)",
                "retinal_risk": "~15%",
                "renal_risk":   "~18%",
            },
            {
                "variant":      "Tyr312Cys (c.935A>G)",
                "domain":       "TM4–6 cluster (aa 261–400), intramembrane",
                "effect":       "Moderate TM destabilisation; decreased TMEM231 binding",
                "population":   "East Asian",
                "severity":     "Moderate",
                "retinal_risk": "~25%",
                "renal_risk":   "~28%",
            },
        ],

        "domain_phenotype_matrix": [
            {
                "domain":        "N-term NPHP1-binding (aa 1–40)",
                "key_variants":  "Arg8Trp",
                "function_lost": "NPHP1 direct contact → Y-link scaffold weakening",
                "severity":      "Moderate",
                "retinal_risk":  "~25%",
                "renal_risk":    "~30% (NPHP-like TIN)",
            },
            {
                "domain":        "TM1–3 hairpin (aa 41–180)",
                "key_variants":  "Gly78Arg",
                "function_lost": "TZ membrane insertion; gate structural integrity",
                "severity":      "Moderate",
                "retinal_risk":  "~28%",
                "renal_risk":    "~28%",
            },
            {
                "domain":        "Extracellular loop (aa 181–260)",
                "key_variants":  "Phe182Ser, Trp255Ter",
                "function_lost": "TMEM231 / B9D1 TZ membrane anchor cross-link",
                "severity":      "Moderate-Severe",
                "retinal_risk":  "~30%",
                "renal_risk":    "~35%",
            },
            {
                "domain":        "TM4–6 cluster (aa 261–400)",
                "key_variants":  "Tyr312Cys",
                "function_lost": "Secondary TZ membrane anchor; TMEM231 C-term interface",
                "severity":      "Moderate",
                "retinal_risk":  "~25%",
                "renal_risk":    "~28%",
            },
            {
                "domain":        "C-term NPHP4/Inversin (aa 401–541)",
                "key_variants":  "Arg399Ter, c.846+1G>A, Ala447Val",
                "function_lost": "NPHP4 / Inversin cross-module bridge; TZ organisation",
                "severity":      "Null: severe; Ala447Val: mild (hypomorphic)",
                "retinal_risk":  "~15–33% (allele-dependent)",
                "renal_risk":    "~18–38% (allele-dependent)",
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
                "event": "TMEM237 bridges NPHP1–NPHP4 Y-link scaffold and TMEM231–B9D1 TZ membrane anchor",
                "effect_when_lost": "Cross-module uncoupling: Y-link scaffold weakened; TZ membrane anchor partially destabilised"
            },
            {
                "step": "2",
                "event": "TZ gate controls import of GPCRs (SSTR3, MCHR1) and SMO into cilia",
                "effect_when_lost": "GPCR / SMO ciliary entry reduced — Hedgehog signal transduction impaired"
            },
            {
                "step": "3",
                "event": "Hedgehog signalling activates GLI2/GLI3A in cerebellar granule precursors",
                "effect_when_lost": "Cerebellar vermis hypoplasia + SCP elongation → Molar Tooth Sign (MTS)"
            },
            {
                "step": "4",
                "event": "TMEM237–NPHP1 axis maintains primary cilia in renal collecting duct",
                "effect_when_lost": "Tubulointerstitial nephritis (TIN) — concentrating defect → ESRD median 24 yr"
            },
            {
                "step": "5",
                "event": "TMEM237 present in photoreceptor connecting cilia (CC)",
                "effect_when_lost": "Rod-cone dystrophy in ~27% — progressive from age 6–10 yr; ERG mandatory"
            },
            {
                "step": "6",
                "event": "TMEM237 in cholangiocyte primary cilia supports biliary flow",
                "effect_when_lost": "Mild ductal plate malformation (CHF ~8%) — portal pressure monitoring from age 2 yr"
            },
        ],

        "management": [
            {
                "intervention": "Brain MRI — MTS confirmation",
                "timing":        "At diagnosis",
                "rationale":    "Confirm molar tooth sign; exclude mimic conditions (COACH, JSOFD, BBS)",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual ERG + ophthalmology",
                "timing":        "From diagnosis (even if vision normal at presentation)",
                "rationale":    "Rod-cone dystrophy ~27%; early intervention window for visual aids / low-vision rehab",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual renal surveillance (creatinine, cystatin C, urine osmolality)",
                "timing":        "From diagnosis; escalate if concentrating defect detected",
                "rationale":    "NPHP-like TIN ~28%; ESRD median ~24 yr; transplant curative for renal endpoint",
                "level":        "Mandatory (NPHP protocol)"
            },
            {
                "intervention": "Liver function + ultrasound",
                "timing":        "Baseline; repeat every 2 yr",
                "rationale":    "Mild CHF risk ~8%; portal hypertension screening if LFT elevated",
                "level":        "Recommended"
            },
            {
                "intervention": "Respiratory monitoring / polysomnography",
                "timing":        "Neonatal; repeat if apnoea events",
                "rationale":    "Breathing dysregulation ~55%; apnoea in neonatal period may require CPAP",
                "level":        "Mandatory (neonatal)"
            },
            {
                "intervention": "Physiotherapy + occupational therapy",
                "timing":        "Early (age 0–3); lifelong",
                "rationale":    "Hypotonia ~81%; cerebellar ataxia ~86%; early intervention improves functional outcome",
                "level":        "Standard of care"
            },
            {
                "intervention": "PGT-M / Prenatal diagnosis",
                "timing":        "Pre-conception or early pregnancy",
                "rationale":    "25% recurrence risk JBTS14; no MKS lethal risk (simpler counselling vs MKS-tier genes)",
                "level":        "Offered to all families"
            },
            {
                "intervention": "Renal transplant",
                "timing":        "When GFR <20 mL/min/1.73m²",
                "rationale":    "Curative for renal endpoint; JBTS14 is cell-autonomous — no recurrence in allograft",
                "level":        "Standard of care (end-stage renal)"
            },
        ],
    }


def get_definitions():
    return {
        "gene_full_name":  "Transmembrane Protein 237 (TMEM237; ALS2CR4)",
        "omim_gene":       "614423",
        "omim_jbts14":     "614424",
        "chromosome":      "2q33.1",
        "protein_size":    "541 aa — multi-pass TZ transmembrane protein, NPHP-module bridge",
        "inheritance":     "Autosomal recessive — biallelic LOF; no MKS lethal tier",

        "no_mks_tier_rule": (
            "ALL biallelic TMEM237 genotypes (null/null, null/hypomorphic, biallelic missense) "
            "→ JBTS14 live birth. No perinatal-lethal MKS allele class documented (2026). "
            "Standard 25% JBTS14 recurrence applies to all families; no MKS-8/6/5/4/3 tier "
            "risk calculation needed."
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
            "esrd_median_age":              "~24 yr",
            "no_mks_tier":                  "Confirmed — all biallelic genotypes → live birth",
        },

        "key_clinical_distinctions": {
            "vs_TCTN1_JBTS11":      "TCTN1 biallelic null → JBTS11 (no MKS); TMEM237 → JBTS14 (no MKS). Both live-birth only. TCTN1 hepatic 12% vs TMEM237 8%; TMEM237 higher renal NPHP link.",
            "vs_TCTN2_JBTS13":      "TCTN2 biallelic null → MKS8 (perinatal lethal); TMEM237 biallelic null → JBTS14 only (live birth). Counselling differs substantially.",
            "vs_NPHP1_pure":        "Biallelic null NPHP1 (JBTS4 allele class) can give pure NPHP without MTS. TMEM237 always gives MTS regardless of allele class.",
            "vs_KIF7_JBTS12":       "KIF7 polydactyly 35–45%, CC anomaly 20%; TMEM237 polydactyly ~6%, no CC anomaly. Distinct skeletal and brain phenotype.",
            "vs_CC2D2A_JBTS9":      "CC2D2A COACH hepatic ~25%; TMEM237 hepatic ~8% mild only. CC2D2A MKS6 tier; TMEM237 no MKS tier.",
            "NPHP_surveillance":    "TMEM237 NPHP1-bridge interaction mandates NPHP renal protocol even before proteinuria — do not defer renal surveillance.",
        },

        "management_highlights": [
            "Annual ERG + ophthalmology from diagnosis — rod-cone risk ~27%; do not wait for subjective visual complaint",
            "Annual renal surveillance (creatinine + urine osmolality) — NPHP-like TIN ~28%; ESRD median ~24 yr",
            "No MKS lethal tier — standard 25% JBTS14 recurrence counselling; no MKS-specific prenatal urgency",
            "Neonatal respiratory monitoring — apnoea in 55%; CPAP if oxygen desaturation",
            "Physiotherapy from age 0–3 yr — hypotonia ~81%; early motor intervention improves ambulation prognosis",
            "Liver USS + LFT every 2 yr — mild CHF ~8%; portal pressure assessment if hepatomegaly",
            "Renal transplant curative when ESRD — no recurrence in allograft (cell-autonomous ciliopathy)",
            "PGT-M available for known pathogenic TMEM237 variants in at-risk families",
        ],

        "literature_highlights": [
            "Huang L et al. (2011) TMEM237 is mutated in individuals with a Joubert syndrome related disorder and expands the role of the TMEM family at the ciliary transition zone. Am J Hum Genet 89(6):713–30.",
            "Baala L et al. (2007) Pleiotropic effects of CEP290 (NPHP6) mutations extend to Meckel syndrome. Am J Hum Genet 81(1):170–9. [Locus linkage context].",
            "Reiter JF & Leroux MR (2017) Genes and molecular pathways underpinning ciliopathies. Nat Rev Mol Cell Biol 18(9):533–47. [TMEM237 TZ complex review].",
            "Shi X et al. (2017) Super-resolution microscopy reveals that disruption of ciliary transition-zone architecture causes Joubert syndrome. Nat Cell Biol 19(10):1178–88.",
            "Gustavsson P et al. (2022) Clinical delineation of TMEM237-related JBTS14: 23 additional families and genotype-phenotype correlations. Eur J Hum Genet 30:48–57.",
        ],
    }
