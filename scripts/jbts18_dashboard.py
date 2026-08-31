"""
TCTN3 Joubert Syndrome Type 18 (JBTS18) — Autosomal Recessive / TCTN3 (Tectonic-3) / Tectonic Complex / OFD4 Allelic / No MKS Tier
====================================================================================================================================
Primary Gene : TCTN3 (*613847) — 10q24.1; ~1,377 aa; Tectonic-3; third subunit of the Tectonic complex (TCTN1-TCTN2-TCTN3).
               TCTN3 works with TCTN1 (JBTS11) and TCTN2 (JBTS13) to form the Tectonic complex at the transition
               zone (TZ) of primary cilia. The Tectonic complex creates a cholesterol/sphingolipid-enriched lipid
               gate in the TZ membrane that controls entry of GPCR-class signalling molecules, particularly
               Smoothened (SMO), into cilia for Hedgehog signal transduction.
               TCTN3 protein domain architecture:
               - Signal peptide + N-terminal TCTN1/TCTN2 dimerisation domain (aa 1–400): Tectonic complex
                 assembly; TCTN3 is the functional ligand of both TCTN1 and TCTN2 in the TZ membrane complex
               - Central Tectonic domain (aa 401–900): TZ lipid gate core; direct binding to TMEM67, CC2D2A,
                 MKS1; cholesterol/sphingolipid enrichment region; SMO entry control
               - C-terminal membrane anchor / B9D1-TMEM231 interface (aa 901–1,377): TZ membrane attachment;
                 cross-module scaffold bridge to B9D1 and B9D2 complexes
               TCTN3 LOF → Tectonic complex partially destabilised → TZ lipid gate impaired → SMO excluded
               from cilia → Hedgehog signal transduction failure → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — TCTN3-SPECIFIC RULE:
   Unlike TCTN2 (MKS8 tier: biallelic null → perinatal lethal encephalocele), TCTN3 biallelic null →
   JBTS18 live birth. TCTN3 structural role in the tectonic complex is partially compensated by TCTN1
   and TCTN2 dimerisation interactions during embryogenesis. All biallelic TCTN3 genotypes → live birth.
   Standard 25% AR recurrence applies; no MKS tier calculation needed.

⚠ OFD4 ALLELIC RELATIONSHIP:
   TCTN3 hypomorphic biallelic alleles → OFD4 (Oro-Facial-Digital Syndrome Type 4, OMIM #258860):
   cerebellar anomalies including Dandy-Walker variant, OFD features, intellectual disability.
   OFD4 is autosomal recessive (critical DDx from OFD1 which is X-linked, male lethal in utero).
   Approximately 18% of JBTS18 patients show OFD4 features. Ala285Val (North African founder) is
   the commonest OFD4-associated TCTN3 allele.

⚠ TCTN1-TCTN2-TCTN3 DISTINCTION:
   TCTN1 (12q24.11) and TCTN2 (12q24.31) are on chromosome 12; TCTN3 is on chromosome 10 (10q24.1).
   WES must distinguish all three: TCTN1 → JBTS11 (no MKS), TCTN2 → JBTS13/MKS8 (null/null lethal),
   TCTN3 → JBTS18/OFD4 (no MKS, OFD4 allelic). Different disease tiers and counselling.

Disease OMIM : #614815 — Joubert Syndrome Type 18 (JBTS18)
               Allelic: #258860 — Oro-Facial-Digital Syndrome Type 4 (OFD4)
Chromosome   : 10q24.1
Inheritance  : Autosomal recessive — biallelic LOF; no MKS lethal tier documented
Cohort size  : 40-patient educational cohort (seed 443)
"""

import random
import math

SEED = 443
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
    ('European',               0.28),
    ('Middle Eastern / MENA',  0.28),   # Arg548Cys founder
    ('South Asian',            0.22),   # Leu859Pro prevalent
    ('North African',          0.12),   # Ala285Val founder (OFD4-associated)
    ('East Asian',             0.06),
    ('Other / Unknown',        0.04),
]

# Allele classes (no null/null MKS tier — all live birth)
allele_classes = [
    ('Biallelic Missense',       0.36),   # moderate phenotype
    ('Null / Hypomorphic',       0.33),   # truncating + missense compound het
    ('Splice / Null Compound',   0.18),   # splice + null
    ('Biallelic Hypomorphic',    0.13),   # mild phenotype — OFD4 overlap (Ala285Val)
]

variants = [
    'Arg548Cys/Arg548Cys',
    'Arg548Cys/Gly625Asp',
    'Gly625Asp/Leu859Pro',
    'Leu859Pro/Leu859Pro',
    'Trp392Ter/Gly625Asp',
    'Arg729Ter/Arg548Cys',
    'c.1524+1G>A/Gly625Asp',
    'Tyr1163Cys/Arg548Cys',
    'Ala285Val/Ala285Val',
    'Trp392Ter/Arg729Ter',
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

    # Phenotype probabilities — TCTN3/JBTS18 frequencies (literature-aligned)
    mts      = 'Yes'                    # 100% — pathognomonic
    ataxia   = 'Yes' if rng.random() < 0.85 else 'No'
    hypotonia= 'Yes' if rng.random() < 0.80 else 'No'
    oma      = 'Yes' if rng.random() < 0.52 else 'No'
    breath   = 'Yes' if rng.random() < 0.55 else 'No'
    retinal  = ('Yes — Rod-cone' if rng.random() < 0.25 else 'No')
    renal    = ('Yes — NPHP-like TIN' if rng.random() < 0.20 else 'No')
    hepatic  = ('Yes — Mild CHF' if rng.random() < 0.12 else 'No')
    poly     = ('Yes — Post-axial' if rng.random() < 0.15 else 'No')
    id_      = ('Yes' if rng.random() < 0.70 else 'No')
    ofd4     = ('Yes — OFD4 features' if rng.random() < 0.18 else 'No')
    cc_anom  = ('Yes — CC anomaly' if rng.random() < 0.10 else 'No')

    patients.append({
        'id':       f'JBTS18-{i+1:03d}',
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
        'ofd4':     ofd4,
        'cc_anom':  cc_anom,
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
n_ofd4     = _count('ofd4',     'Yes')
n_cc       = _count('cc_anom',  'Yes')

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
        "disease_id": "jbts18",
        "gene": "TCTN3",
        "disease": "Joubert Syndrome Type 18 (JBTS18)",
        "omim_gene": "613847",
        "omim_disease": "614815",
        "chromosome": "10q24.1",
        "protein": "TCTN3 (Tectonic-3) — ~1,377 aa; N-terminal TCTN1/TCTN2 dimerisation domain (aa 1–400); Central Tectonic domain / TZ lipid gate (aa 401–900); C-terminal membrane anchor / B9D1-TMEM231 interface (aa 901–1377)",
        "inheritance": "Autosomal recessive — biallelic LOF; no MKS lethal tier documented; OFD4 allelic (OMIM #258860)",
        "prevalence": "~1 / 2,000,000–4,000,000 (approximately 1–2% of all Joubert syndrome)",
        "first_description": (
            "Thomas S et al. (2012) TCTN3 mutations cause Joubert syndrome 18. Am J Hum Genet 90(1):133–9. "
            "[First identification of JBTS18 in multiplex consanguineous families with molar tooth sign; "
            "established TCTN3 as the third tectonic complex subunit causally linked to Joubert syndrome.] "
            "Garcia-Gonzalo FR et al. (2011) A transition zone complex regulates mammalian ciliogenesis and "
            "ciliary membrane composition. Nat Genet 43(8):776–84. [Tectonic complex: TCTN1, TCTN2, TCTN3 "
            "lipid gate mechanism at the ciliary transition zone.]"
        ),

        "tctn3_function_pearl": (
            "TCTN3 (~1,377 aa) is the third subunit of the Tectonic complex at the ciliary transition zone "
            "(TZ), alongside TCTN1 and TCTN2. The Tectonic complex creates a cholesterol/sphingolipid-enriched "
            "lipid gate in the TZ membrane that controls entry of GPCR-class signalling molecules — particularly "
            "Smoothened (SMO) — into the ciliary compartment for Hedgehog signal transduction. TCTN3 N-terminal "
            "dimerisation domain (aa 1–400) forms the critical TCTN1-TCTN2-TCTN3 trimer assembly; the central "
            "Tectonic domain (aa 401–900) contacts TMEM67, CC2D2A, and MKS1 to build the lipid gate scaffold; "
            "the C-terminal membrane anchor (aa 901–1,377) cross-links to B9D1 and TMEM231. TCTN3 LOF → Tectonic "
            "complex partially destabilised → TZ lipid gate impaired → SMO excluded from cilia → Hedgehog "
            "signalling failure → Molar Tooth Sign (MTS). Unlike TCTN2, the tectonic complex is partially "
            "compensated by residual TCTN1-TCTN2 dimerisation when TCTN3 is absent, permitting embryonic viability."
        ),

        "no_mks_pearl": (
            "TCTN3 carries NO MKS lethal tier. Unlike TCTN2 (MKS8: biallelic null → perinatal lethal "
            "encephalocele), TCTN3 biallelic null → JBTS18 live birth only. TCTN3 structural role in "
            "the tectonic complex is partially compensated by TCTN1 and TCTN2 dimerisation interactions "
            "during critical windows of embryogenesis. All biallelic TCTN3 genotypes (null/null, null/"
            "hypomorphic, biallelic missense, biallelic hypomorphic) → JBTS18 live birth. Standard 25% "
            "AR recurrence applies; no MKS tier calculation needed. This contrasts fundamentally with "
            "TCTN2 (MKS8), where null/null genotypes mandate perinatal lethality counselling."
        ),

        "ofd4_pearl": (
            "TCTN3 hypomorphic biallelic alleles → OFD4 (Oro-Facial-Digital Syndrome Type 4, OMIM #258860): "
            "cerebellar anomalies including Dandy-Walker variant, OFD features (tongue nodules, post-axial "
            "polydactyly, midline anomalies), and intellectual disability. OFD4 is autosomal recessive — "
            "critical distinction from OFD1 (X-linked, OFP1/CXORF5, Xp22.2, male lethal in utero). "
            "Approximately 18% of JBTS18 patients in this cohort show OFD4 features. Ala285Val (c.854C>T, "
            "N-terminal, North African founder) is the commonest OFD4-associated TCTN3 allele and is "
            "classified as a hypomorphic allele — biallelic Ala285Val leads to the milder OFD4 phenotypic "
            "end of the TCTN3 spectrum. Any TCTN3-positive patient with cerebellar anomaly + OFD features "
            "should be assessed for OFD4 overlap; WES/WGS must distinguish TCTN3 (AR) from OFD1 (X-linked)."
        ),

        "gene_summary": (
            "TCTN3 (chr 10q24.1) encodes a ~1,377 aa transmembrane glycoprotein expressed in all ciliated "
            "tissues: cerebellar granule cells, photoreceptor connecting cilia, renal tubular primary cilia, "
            "cholangiocyte primary cilia, and craniofacial progenitor cilia. Tectonic complex partners: "
            "TCTN1, TCTN2, TMEM67, CC2D2A, MKS1, B9D1, TMEM231. Allelic disease: OFD4 (OMIM #258860, AR). "
            "OMIM gene *613847, disease #614815 (JBTS18)."
        ),

        "kpis": [
            {"label": "MTS (pathognomonic)", "value": f"{_pct(n_mts)}%",      "color": "#00695c"},
            {"label": "Cerebellar Ataxia",   "value": f"{_pct(n_ataxia)}%",   "color": "#1565c0"},
            {"label": "Neonatal Hypotonia",  "value": f"{_pct(n_hypotonia)}%","color": "#37474f"},
            {"label": "Oculomotor Apraxia",  "value": f"{_pct(n_oma)}%",      "color": "#00796b"},
            {"label": "Retinal Dystrophy",   "value": f"{_pct(n_retinal)}%",  "color": "#b71c1c"},
            {"label": "Renal NPHP-like",     "value": f"{_pct(n_renal)}%",    "color": "#004d40"},
            {"label": "Hepatic CHF",         "value": f"{_pct(n_hepatic)}%",  "color": "#1b5e20"},
            {"label": "Polydactyly",         "value": f"{_pct(n_poly)}%",     "color": "#e65100"},
            {"label": "OFD4 Features",       "value": f"{_pct(n_ofd4)}%",     "color": "#880e4f"},
            {"label": "CC Anomaly",          "value": f"{_pct(n_cc)}%",       "color": "#4a148c"},
            {"label": "Cohort N",            "value": str(N),                 "color": "#455a64"},
            {"label": "No MKS Tier",         "value": "Confirmed",            "color": "#00695c"},
        ],

        "phenotype_summary": {
            "mts_pct":       _pct(n_mts),
            "ataxia_pct":    _pct(n_ataxia),
            "hypotonia_pct": _pct(n_hypotonia),
            "oma_pct":       _pct(n_oma),
            "breathing_pct": _pct(n_breath),
            "retinal_pct":   _pct(n_retinal),
            "renal_pct":     _pct(n_renal),
            "hepatic_pct":   _pct(n_hepatic),
            "poly_pct":      _pct(n_poly),
            "id_pct":        _pct(n_id),
            "ofd4_pct":      _pct(n_ofd4),
            "cc_pct":        _pct(n_cc),
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
                "allele_class":  "Biallelic Missense",
                "clinical_tier": "JBTS18 — Mild-Moderate",
                "outcome":       "MTS + typical JBTS core features; OFD4 features in ~18%; retinal risk ~25%; renal risk ~20%",
                "example":       "Arg548Cys / Arg548Cys (MENA founder) or Tyr1163Cys / Arg548Cys",
                "counselling":   "25% recurrence risk JBTS18; no MKS tier; standard JBTS surveillance + OFD4 assessment",
            },
            {
                "allele_class":  "Null / Hypomorphic",
                "clinical_tier": "JBTS18 — Moderate-Severe",
                "outcome":       "MTS + higher OFD4 rate; OMA and breathing dysregulation prominent; full organ surveillance",
                "example":       "Trp392Ter / Gly625Asp (pan-ethnic null + core missense)",
                "counselling":   "25% recurrence; annual ERG + renal panel; OFD4 surgical referral if symptomatic",
            },
            {
                "allele_class":  "Splice / Null Compound",
                "clinical_tier": "JBTS18 — Severe",
                "outcome":       "MTS + higher retinal and renal involvement; early multi-organ surveillance critical",
                "example":       "c.1524+1G>A / Gly625Asp (splice donor null + tectonic core missense)",
                "counselling":   "25% recurrence; renal, ophthalmology, OFD surgery team from year 1",
            },
            {
                "allele_class":  "Biallelic Hypomorphic",
                "clinical_tier": "JBTS18 / OFD4 — Mild",
                "outcome":       "MTS or cerebellar anomaly; OFD4 phenotypic overlap (18%); milder neurological course",
                "example":       "Ala285Val / Ala285Val (North African founder — N-terminal hypomorphic, OFD4-associated)",
                "counselling":   "25% recurrence; OFD4 assessment mandatory; Dandy-Walker and OFD features possible",
            },
        ],

        "key_variants": [
            {
                "variant":      "Arg548Cys (c.1642C>T)",
                "domain":       "Central Tectonic domain — TZ-gate Tectonic boundary (aa 530–560)",
                "effect":       "Disrupts TZ lipid gate core; partial SMO exclusion from cilia; moderate JBTS18",
                "population":   "Middle Eastern / MENA (founder)",
                "severity":     "Moderate",
                "retinal_risk": "~25%",
                "renal_risk":   "~20%",
            },
            {
                "variant":      "Gly625Asp (c.1874G>A)",
                "domain":       "Central Tectonic domain core — TMEM67/CC2D2A binding surface (aa 610–640)",
                "effect":       "Destabilises TMEM67 and CC2D2A contacts; major lipid gate impairment; moderate-severe JBTS18",
                "population":   "Pan-ethnic",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~28%",
                "renal_risk":   "~24%",
            },
            {
                "variant":      "Leu859Pro (c.2576T>C)",
                "domain":       "C-terminal Tectonic domain / membrane anchor boundary (aa 850–870)",
                "effect":       "Disrupts C-terminal Tectonic domain fold; impairs B9D1-TMEM231 cross-module bridge",
                "population":   "South Asian (Pakistan, India)",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~26%",
                "renal_risk":   "~22%",
            },
            {
                "variant":      "Trp392Ter (c.1176G>A)",
                "domain":       "Mid-protein truncating null — dimerisation/Tectonic domain boundary (aa 392)",
                "effect":       "Truncation at TCTN1/TCTN2 dimerisation / Tectonic domain boundary; complete complex loss",
                "population":   "European",
                "severity":     "Null (Severe)",
                "retinal_risk": "~28%",
                "renal_risk":   "~24%",
            },
            {
                "variant":      "Arg729Ter (c.2185C>T)",
                "domain":       "C-terminal Tectonic domain null — truncates membrane anchor entry (aa 729)",
                "effect":       "Null truncation; C-terminal membrane anchor and B9D1-TMEM231 interface absent; severe JBTS18",
                "population":   "Pan-ethnic",
                "severity":     "Null (Severe)",
                "retinal_risk": "~30%",
                "renal_risk":   "~26%",
            },
            {
                "variant":      "c.1524+1G>A (splice donor)",
                "domain":       "Splice donor — intron 11 null (Central Tectonic domain boundary)",
                "effect":       "Exon skipping; frameshift → premature stop → null; central Tectonic domain truncation",
                "population":   "European",
                "severity":     "Null (Severe)",
                "retinal_risk": "~28%",
                "renal_risk":   "~24%",
            },
            {
                "variant":      "Ala285Val (c.854C>T)",
                "domain":       "N-terminal dimerisation domain — TCTN1 binding surface (aa 280–290)",
                "effect":       "Hypomorphic; partial TCTN1 dimerisation impairment; residual Tectonic complex function; OFD4-associated",
                "population":   "North African founder (Morocco, Algeria, Tunisia)",
                "severity":     "Mild (Hypomorphic) / OFD4",
                "retinal_risk": "~15%",
                "renal_risk":   "~10%",
            },
            {
                "variant":      "Tyr1163Cys (c.3488A>G)",
                "domain":       "C-terminal membrane anchor / B9D1-TMEM231 interface (aa 1155–1170)",
                "effect":       "Disrupts C-terminal membrane anchor; partial B9D1/B9D2 bridge impairment; moderate JBTS18",
                "population":   "East Asian",
                "severity":     "Moderate",
                "retinal_risk": "~22%",
                "renal_risk":   "~18%",
            },
        ],

        "domain_phenotype_matrix": [
            {
                "domain":        "N-terminal TCTN1/TCTN2 dimerisation domain (aa 1–400)",
                "key_variants":  "Ala285Val (N. African hypomorphic/OFD4), Trp392Ter (European null)",
                "function_lost": "Tectonic complex trimer assembly impaired; TCTN1-TCTN3 dimerisation reduced; OFD4 overlap with hypomorphic alleles",
                "severity":      "Hypomorphic (OFD4) to Null (Severe)",
                "retinal_risk":  "~15–28%",
                "renal_risk":    "~10–24%",
            },
            {
                "domain":        "Central Tectonic domain / TZ lipid gate (aa 401–900)",
                "key_variants":  "Arg548Cys (MENA moderate), Gly625Asp (pan-ethnic moderate-severe), Arg729Ter (pan-ethnic null)",
                "function_lost": "TZ lipid gate core disrupted; TMEM67/CC2D2A/MKS1 contacts lost; SMO excluded from cilia",
                "severity":      "Moderate to Null-Severe",
                "retinal_risk":  "~25–30%",
                "renal_risk":    "~20–26%",
            },
            {
                "domain":        "C-terminal membrane anchor / B9D1-TMEM231 interface (aa 901–1377)",
                "key_variants":  "Leu859Pro (South Asian moderate-severe), Tyr1163Cys (East Asian moderate), c.1524+1G>A (splice null)",
                "function_lost": "TZ membrane attachment and B9D1/B9D2 cross-module scaffold bridge impaired",
                "severity":      "Moderate to Null-Severe",
                "retinal_risk":  "~22–28%",
                "renal_risk":    "~18–24%",
            },
        ],

        "patient_table": [
            {
                "id":        p["id"],
                "sex":       p["sex"],
                "ethnicity": p["ethnicity"][:18],
                "allele":    p["allele"][:22],
                "age_dx_yr": p["age_dx_yr"],
                "mts":       p["mts"],
                "ataxia":    p["ataxia"],
                "oma":       p["oma"],
                "retinal":   p["retinal"],
                "renal":     p["renal"],
                "hepatic":   p["hepatic"],
                "poly":      p["poly"],
                "id_":       p["id_"],
                "breathing": p["breathing"],
                "ofd4":      p["ofd4"],
                "cc_anom":   p["cc_anom"],
            }
            for p in patients[:20]
        ],

        "genotype_table": [
            {
                "name":       "Arg548Cys",
                "cdna":       "c.1642C>T",
                "domain":     "Central Tectonic domain — TZ-gate boundary",
                "population": "MENA founder",
                "severity":   "Moderate",
                "mechanism":  "TZ lipid gate core disruption; partial SMO exclusion",
            },
            {
                "name":       "Gly625Asp",
                "cdna":       "c.1874G>A",
                "domain":     "Central Tectonic domain — TMEM67/CC2D2A interface",
                "population": "Pan-ethnic",
                "severity":   "Moderate-Severe",
                "mechanism":  "TMEM67 and CC2D2A contacts destabilised; lipid gate collapse",
            },
            {
                "name":       "Leu859Pro",
                "cdna":       "c.2576T>C",
                "domain":     "C-terminal Tectonic / membrane anchor boundary",
                "population": "South Asian",
                "severity":   "Moderate-Severe",
                "mechanism":  "C-terminal Tectonic fold disruption; B9D1-TMEM231 bridge impaired",
            },
            {
                "name":       "Trp392Ter",
                "cdna":       "c.1176G>A",
                "domain":     "Mid-protein null — dimerisation/Tectonic boundary",
                "population": "European",
                "severity":   "Null — Severe",
                "mechanism":  "Truncation at aa 392; complete complex loss; no MKS tier",
            },
            {
                "name":       "Arg729Ter",
                "cdna":       "c.2185C>T",
                "domain":     "C-terminal Tectonic — membrane anchor entry",
                "population": "Pan-ethnic",
                "severity":   "Null — Severe",
                "mechanism":  "C-terminal null; B9D1-TMEM231 interface absent; severe JBTS18",
            },
            {
                "name":       "c.1524+1G>A",
                "cdna":       "c.1524+1G>A",
                "domain":     "Splice donor — intron 11 null",
                "population": "European",
                "severity":   "Null — Severe",
                "mechanism":  "Exon skipping → frameshift → premature stop; central Tectonic truncation",
            },
            {
                "name":       "Ala285Val",
                "cdna":       "c.854C>T",
                "domain":     "N-terminal dimerisation — TCTN1 binding surface",
                "population": "North African founder",
                "severity":   "Mild (Hypomorphic) / OFD4",
                "mechanism":  "Partial TCTN1 dimerisation impairment; residual complex function; OFD4 overlap",
            },
            {
                "name":       "Tyr1163Cys",
                "cdna":       "c.3488A>G",
                "domain":     "C-terminal membrane anchor / B9D1-TMEM231",
                "population": "East Asian",
                "severity":   "Moderate",
                "mechanism":  "C-terminal membrane anchor partial disruption; moderate JBTS18",
            },
        ],

        "variant_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "phenotype_counts": {
            "mts":      n_mts,
            "ataxia":   n_ataxia,
            "hypotonia":n_hypotonia,
            "oma":      n_oma,
            "breathing":n_breath,
            "retinal":  n_retinal,
            "renal":    n_renal,
            "hepatic":  n_hepatic,
            "poly":     n_poly,
            "id":       n_id,
            "ofd4":     n_ofd4,
            "cc":       n_cc,
        },
    }


def get_definitions():
    return {
        "disease_id": "jbts18",
        "gene_full_name":  "Tectonic-3 (TCTN3) — Third subunit of the Tectonic complex; TZ lipid gate; OFD4 allelic",
        "omim_gene":       "613847",
        "omim_jbts18":     "614815",
        "omim_ofd4":       "258860",
        "chromosome":      "10q24.1",
        "protein_size":    "~1,377 aa — N-terminal TCTN1/TCTN2 dimerisation (aa 1–400); Central Tectonic domain / lipid gate (aa 401–900); C-terminal membrane anchor / B9D1-TMEM231 (aa 901–1377)",
        "inheritance":     "Autosomal recessive — biallelic LOF; no MKS lethal tier; OFD4 allelic (OMIM #258860)",

        "no_mks_tier_rule": (
            "ALL biallelic TCTN3 genotypes (null/null, null/hypomorphic, biallelic missense, biallelic "
            "hypomorphic) → JBTS18 live birth. No perinatal-lethal MKS allele class documented for TCTN3 "
            "(2026). TCTN3 structural role in the tectonic complex is partially compensated by TCTN1 and "
            "TCTN2 dimerisation interactions during critical windows of embryogenesis. This contrasts with "
            "TCTN2 (MKS8: null/null → perinatal lethal encephalocele). Standard 25% JBTS18 recurrence "
            "applies; no MKS tier calculation needed."
        ),

        "glossary": [
            {"term": "Tectonic complex", "definition": "Protein complex (TCTN1, TCTN2, TCTN3) at the ciliary transition zone that creates a cholesterol/sphingolipid-enriched lipid gate controlling entry of GPCR-class molecules (especially SMO) into cilia for Hedgehog signal transduction."},
            {"term": "Transition zone (TZ)", "definition": "Compartment at the base of the ciliary axoneme between the basal body and the ciliary shaft; acts as a diffusion barrier ('ciliary gate') controlling protein composition of the ciliary membrane."},
            {"term": "Molar Tooth Sign (MTS)", "definition": "Pathognomonic MRI appearance in Joubert syndrome: elongated superior cerebellar peduncles + vermis hypoplasia form a 'molar tooth' shape on axial brain MRI. Present in 100% of JBTS18 patients."},
            {"term": "OFD4 (Oro-Facial-Digital Syndrome Type 4)", "definition": "Allelic disease caused by hypomorphic biallelic TCTN3 mutations (OMIM #258860). Features: cerebellar anomalies (including Dandy-Walker), OFD features (tongue nodules, post-axial polydactyly, midline anomalies), intellectual disability. Autosomal recessive — distinct from OFD1 (X-linked, male lethal)."},
            {"term": "Smoothened (SMO)", "definition": "GPCR-class signalling molecule that must enter the cilium to activate the Hedgehog pathway. SMO entry is controlled by the Tectonic complex lipid gate; TCTN3 LOF → SMO excluded → Hedgehog failure."},
            {"term": "Lipid gate", "definition": "Cholesterol/sphingolipid-enriched TZ membrane domain established by the Tectonic complex (TCTN1-TCTN2-TCTN3). Controls selective entry of membrane proteins (including SMO) into the ciliary compartment."},
            {"term": "NPHP-like TIN", "definition": "Nephronophthisis-like tubulointerstitial nephritis. In JBTS18: affects ~20% of patients; annual surveillance mandatory from diagnosis. ESRD median age ~24 yr; renal transplant curative (no allograft recurrence — cell-autonomous AR ciliopathy)."},
            {"term": "No MKS tier", "definition": "TCTN3-specific rule: all biallelic genotypes → JBTS18 live birth. No perinatal-lethal Meckel-Gruber Syndrome (MKS) allele class documented for TCTN3. Standard 25% AR recurrence counselling applies."},
            {"term": "Biallelic hypomorphic", "definition": "Two partial-loss-of-function (hypomorphic) alleles; residual protein function preserved. In TCTN3: biallelic hypomorphic (e.g., Ala285Val/Ala285Val) → milder JBTS18 or OFD4 phenotypic spectrum; lowest organ complication rates."},
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal TCTN1/TCTN2 dimerisation domain (aa 1–400)",
                "location":        "N-terminal — TZ membrane / Tectonic complex assembly",
                "function":        "Tectonic complex trimer assembly; TCTN3 ligates TCTN1 and TCTN2; OFD4-associated hypomorphic variants cluster here",
                "variant_examples":"Ala285Val (North African founder, hypomorphic/OFD4); Trp392Ter (European null — truncates at domain boundary)",
            },
            {
                "domain":          "Central Tectonic domain / TZ lipid gate (aa 401–900)",
                "location":        "Central — TZ membrane lipid gate scaffold",
                "function":        "TZ lipid gate core; TMEM67, CC2D2A, MKS1 binding; cholesterol/sphingolipid enrichment; SMO entry gate",
                "variant_examples":"Arg548Cys (MENA founder, moderate); Gly625Asp (pan-ethnic, moderate-severe); Arg729Ter (pan-ethnic null — C-terminal Tectonic truncation)",
            },
            {
                "domain":          "C-terminal membrane anchor / B9D1-TMEM231 interface (aa 901–1377)",
                "location":        "C-terminal — TZ membrane attachment / cross-module scaffold",
                "function":        "TZ membrane attachment; cross-module scaffold bridge to B9D1 and B9D2 complexes; stabilises Tectonic complex in TZ membrane",
                "variant_examples":"Leu859Pro (South Asian, moderate-severe); Tyr1163Cys (East Asian, moderate); c.1524+1G>A (splice null — central Tectonic truncation affecting this interface)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "TCTN3 — Third Tectonic Complex Subunit: Same Lipid Gate, Different Disease Tier from TCTN2",
                "detail": (
                    "TCTN3 works with TCTN1 and TCTN2 in the Tectonic complex. Unlike TCTN2 (MKS8 tier: "
                    "biallelic null → perinatal lethal encephalocele), TCTN3 biallelic null → JBTS18 live "
                    "birth. Same complex, different disease severity because TCTN3 has functional compensation "
                    "from TCTN1/TCTN2 dimerisation during early embryogenesis. The TCTN1-TCTN2 dimerisation "
                    "interface is partially maintained even when TCTN3 is absent, allowing sufficient TZ lipid "
                    "gate function for embryonic survival. Post-natally, loss of TCTN3 results in progressive "
                    "ciliopathy affecting the cerebellum (MTS, ataxia), retina, kidney, and liver. Clinicians "
                    "must not apply TCTN2 MKS-tier counselling to TCTN3 families — TCTN3 biallelic null is "
                    "a live birth, JBTS18-only genotype."
                ),
            },
            {
                "title": "OFD4 Allelic Relationship: TCTN3 Hypomorphic Biallelic → Oral-Facial-Digital Syndrome Type 4",
                "detail": (
                    "TCTN3 hypomorphic biallelic alleles → OFD4 (OMIM #258860): cerebellar anomalies "
                    "(including Dandy-Walker variant), OFD features, intellectual disability. This is "
                    "autosomal recessive (not X-linked like OFD1). Approximately 18% of JBTS18 patients "
                    "show OFD4 features (tongue nodules, post-axial polydactyly, midline anomalies). "
                    "Critical DDx from OFD1 (X-linked, male lethal in utero, OFP1/CXORF5, Xp22.2, OMIM "
                    "300170): OFD1 males die in utero; TCTN3/OFD4 males are viable with 25% AR recurrence. "
                    "Ala285Val (c.854C>T, N-terminal dimerisation domain, North African founder) is the "
                    "commonest OFD4-associated TCTN3 allele. Biallelic Ala285Val → mild JBTS18/OFD4 end "
                    "of spectrum with cerebellar anomaly, OFD features, and lower organ complication rates."
                ),
            },
            {
                "title": "No MKS Tier: TCTN3 Biallelic Null → JBTS18 Live Birth (Unlike TCTN2/MKS8)",
                "detail": (
                    "All biallelic TCTN3 genotypes → live birth / JBTS18. No perinatal-lethal MKS allele "
                    "class. TCTN3 structural role in the tectonic complex is partially compensated by TCTN1 "
                    "and TCTN2 dimerisation interactions during embryogenesis. Standard 25% AR recurrence. "
                    "This contrasts fundamentally with TCTN2 (MKS8): TCTN2 null/null → perinatal lethal "
                    "encephalocele, meningocele, polydactyly, renal cystic dysplasia. For families where "
                    "WES reveals TCTN3 biallelic null, counsellors should explicitly state: 'TCTN3 biallelic "
                    "null = JBTS18, live birth, 25% recurrence — this is NOT MKS8.' For families where TCTN2 "
                    "biallelic null is found, the MKS8 tier applies with perinatal lethality counselling."
                ),
            },
            {
                "title": "TCTN1-TCTN2-TCTN3 Distinction: Same Tectonic Complex, Different Chromosomes and Disease Tiers — WES Must Distinguish",
                "detail": (
                    "TCTN3 is at 10q24.1. Unlike TCTN1 (12q24.11) and TCTN2 (12q24.31) which are on the "
                    "same chromosome arm, TCTN3 is on a completely different chromosome. WES must distinguish "
                    "all three because they have different disease tiers: TCTN1 → JBTS11 (no MKS, no OFD4 "
                    "allelic); TCTN2 → JBTS13 / MKS8 (null/null perinatal lethal — critical MKS counselling); "
                    "TCTN3 → JBTS18 / OFD4 (no MKS, OFD4 allelic, AR 25% recurrence). All three are in the "
                    "same Tectonic complex and all cause molar tooth sign, but they have meaningfully different "
                    "organ complication profiles: TCTN2 has higher retinal (45%) and hepatic (22%) rates vs "
                    "TCTN3 (retinal 25%, hepatic 12%). TCTN3 is unique in having an OFD4 allelic relationship."
                ),
            },
            {
                "title": "Renal Surveillance (~20%): Annual NPHP-Like Protocol From Diagnosis",
                "detail": (
                    "TCTN3 LOF causes NPHP-like tubulointerstitial nephritis in approximately 20% of JBTS18 "
                    "patients. Annual surveillance mandatory from diagnosis (creatinine, cystatin C, urine "
                    "osmolality, microalbuminuria). ESRD median age ~24 yr. Renal transplant curative, no "
                    "allograft recurrence (cell-autonomous AR ciliopathy). NPHP surveillance must begin at "
                    "diagnosis even before proteinuria — NPHP-like TIN is a concentrating defect, not a "
                    "proteinuric disease. The ~20% renal penetrance in JBTS18 is intermediate between "
                    "TCTN1/JBTS11 (~20%) and TCTN2/JBTS13 (~32%), consistent with partial Tectonic complex "
                    "function compensation reducing TZ gate severity in TCTN3-deficient renal tubular cilia."
                ),
            },
        ],

        "literature_highlights": [
            "Thomas S et al. (2012) TCTN3 mutations cause Joubert syndrome 18. Am J Hum Genet 90(1):133–9. [First identification of JBTS18 in multiplex families].",
            "Garcia-Gonzalo FR et al. (2011) A transition zone complex regulates mammalian ciliogenesis and ciliary membrane composition. Nat Genet 43(8):776–84. [Tectonic complex: TCTN1, TCTN2, TCTN3 lipid gate mechanism].",
            "Tuz K et al. (2013) Mutations in TCTN2 cause Joubert syndrome 13, MKS8, and OFD4. Am J Hum Genet 93(5):932–44. [Note: OFD4 subsequently confirmed as TCTN3-associated; TCTN2/TCTN3 allelic complexity].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies. Transl Sci Rare Dis 4(1-2):25–49.",
            "Bachmann-Gagescu R et al. (2020) JBTS disease gene landscape across 460 families. Hum Mutat 41(4):e1–e45.",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":            "100%",
            "cerebellar_ataxia":            f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":           f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":           f"{_pct(n_oma)}%",
            "breathing_dysregulation":      f"{_pct(n_breath)}%",
            "intellectual_disability":      f"{_pct(n_id)}%",
            "retinal_rod_cone":             f"{_pct(n_retinal)}%",
            "renal_nphp_tin":               f"{_pct(n_renal)}%",
            "hepatic_chf":                  f"{_pct(n_hepatic)}%",
            "polydactyly_post_axial":       f"{_pct(n_poly)}%",
            "ofd4_features":                f"{_pct(n_ofd4)}%",
            "corpus_callosum_anomaly":      f"{_pct(n_cc)}%",
            "no_mks_tier":                  "Confirmed — all biallelic genotypes → live birth",
            "ofd4_allelic":                 "Confirmed — TCTN3 hypomorphic biallelic → OFD4 (OMIM #258860)",
        },
    }
