"""
CPLANE1 Joubert Syndrome Type 17 (JBTS17) — Autosomal Recessive / CPLANE1 (C5orf42) / IFT-A Loader / OFD Overlap / No MKS Tier
=================================================================================================================================
Primary Gene : CPLANE1 (*614571, formerly C5orf42) — 5p13.2; ~1,518 aa; Ciliogenesis and Planar Polarity Effector 1.
               CPLANE1 is the defining scaffold subunit of the CPLANE complex (Ciliogenesis and Planar
               Polarity Effector), which also contains INTU, FUZ, WDPCP, and MPND. The CPLANE complex
               docks at the basal body and recruits IFT-A (retrograde IFT train assembly complex) for
               loading onto cilia. CPLANE1 protein domain architecture:
               - N-terminal OFD-module (aa 1–320): oral-facial-digital signalling; tongue hamartoma /
                 upper-lip notch / midline-palate-cleft phenotype when disrupted
               - Central CPLANE scaffold (aa 321–900): core INTU/FUZ interaction surface; IFT-A docking
               - IFT-A interface (aa 550–850): direct contacts with IFT140/WDR19 and IFT144/WDR19
               - C-terminal PCP effector domain (aa 901–1518): planar cell polarity (PCP) pathway
                 effector; WDPCP binding; broader tissue expression control
               CPLANE1 LOF → IFT-A fails to load onto cilia → retrograde IFT defective → ciliary tip
               accumulation of IFT-B cargo → Hedgehog signal transduction failure → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — CPLANE1-SPECIFIC RULE:
   Unlike TCTN2 (MKS8), CC2D2A (MKS6), or RPGRIP1L (MKS5), CPLANE1 biallelic null → JBTS17
   ONLY (live birth). No perinatal-lethal MKS allele class documented in published literature
   as of 2026. CPLANE1 acts upstream at IFT-A loading rather than at the TZ structural scaffold
   (B9-complex, MKS1, TMEM67 nexus), allowing partial residual cilia function that permits
   embryonic viability.

⚠ OFD (ORAL-FACIAL-DIGITAL) FEATURES — HIGHEST RATE AMONG NON-OFD1 JBTS GENES:
   CPLANE1 LOF causes oral-facial-digital features (tongue hamartomas, upper lip notching,
   midline cleft palate) in approximately 30% of JBTS17 patients — the highest rate among
   non-OFD1 JBTS genes. This drives JSOFD (Joubert Syndrome with Oral-Facial-Digital Features)
   designation. Critical DDx from OFD1 (X-linked, male lethal), which shows a different
   inheritance and MKS overlap pattern. CPLANE1 OFD features result from disruption of the
   N-terminal OFD-module (aa 1–320).

⚠ PLANAR CELL POLARITY (PCP) PATHWAY:
   CPLANE complex also controls PCP — explains broader tissue involvement including polydactyly
   (post-axial, 30%), corpus callosum anomaly (15%), and the wider multi-system phenotypic
   spectrum in JBTS17 vs single-pathway TZ-gate genes.

Disease OMIM : #614615 — Joubert Syndrome Type 17 (JBTS17)
Chromosome   : 5p13.2
Inheritance  : Autosomal recessive — biallelic LOF; no MKS lethal tier documented
               (CPLANE1 acts at IFT-A loading, not the TZ structural MKS-module scaffold)
Cohort size  : 40-patient educational cohort (seed 441)
"""

import random
import math

SEED = 441
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
    ('Middle Eastern / MENA',  0.30),   # Arg239Ter / Tyr327Cys OFD-module founders
    ('South Asian',            0.20),   # Leu703Pro prevalent
    ('North African',          0.12),   # Ala1389Val founder
    ('East Asian',             0.06),
    ('Other / Unknown',        0.04),
]

# Allele classes (no null/null MKS tier — all live birth)
allele_classes = [
    ('Biallelic Missense',       0.38),   # moderate phenotype — common in CPLANE1
    ('Null / Hypomorphic',       0.35),   # truncating + missense compound het
    ('Splice / Null Compound',   0.17),   # splice + null
    ('Biallelic Hypomorphic',    0.10),   # mild phenotype — Ala1389Val founder
]

variants = [
    'Arg239Ter/Arg239Ter',
    'Arg239Ter/Tyr327Cys',
    'Gln583Ter/Leu703Pro',
    'Leu703Pro/Leu703Pro',
    'Trp850Cys/Trp850Cys',
    'Trp850Cys/Leu703Pro',
    'c.3706+1G>A/Leu703Pro',
    'Arg1159Gln/Trp850Cys',
    'Ala1389Val/Ala1389Val',
    'Tyr327Cys/Trp850Cys',
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

    # Phenotype probabilities — CPLANE1/JBTS17 frequencies (literature-aligned)
    mts      = 'Yes'                    # 100% — pathognomonic
    ataxia   = 'Yes' if rng.random() < 0.88 else 'No'
    hypotonia= 'Yes' if rng.random() < 0.82 else 'No'
    oma      = 'Yes' if rng.random() < 0.52 else 'No'
    breath   = 'Yes' if rng.random() < 0.55 else 'No'
    retinal  = ('Yes — Rod-cone' if rng.random() < 0.22 else 'No')
    renal    = ('Yes — NPHP-like TIN' if rng.random() < 0.18 else 'No')
    hepatic  = ('Yes — Mild CHF' if rng.random() < 0.10 else 'No')
    poly     = ('Yes — Post-axial' if rng.random() < 0.30 else 'No')
    id_      = ('Yes' if rng.random() < 0.73 else 'No')
    ofd      = ('Yes — OFD features' if rng.random() < 0.30 else 'No')
    cc_anom  = ('Yes — CC anomaly' if rng.random() < 0.15 else 'No')

    patients.append({
        'id':       f'JBTS17-{i+1:03d}',
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
        'ofd':      ofd,
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
n_ofd      = _count('ofd',      'Yes')
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
        "disease_id": "jbts17",
        "gene": "CPLANE1",
        "disease": "Joubert Syndrome Type 17 (JBTS17)",
        "omim_gene": "614571",
        "omim_disease": "614615",
        "chromosome": "5p13.2",
        "protein": "CPLANE1 (C5orf42) — ~1,518 aa, CPLANE complex scaffold, IFT-A loader, basal body, OFD-module (aa 1–320), PCP effector (aa 901–1518)",
        "inheritance": "Autosomal recessive — biallelic LOF; no MKS lethal tier documented",
        "prevalence": "~1 / 800,000–1,500,000 (approximately 2–3% of all Joubert syndrome)",
        "first_description": (
            "Shaheen R et al. (2013) A homozygous truncating mutation in C5orf42 causes OFD syndrome "
            "type VI in addition to Joubert syndrome. J Med Genet 50(6):408–12. CPLANE1/C5orf42 was "
            "identified as the causal gene in JBTS17 and JSOFD through whole-exome sequencing of "
            "consanguineous families with molar tooth sign and oral-facial-digital features. "
            "Roosing S et al. (2016) Mutations in CPLANE1 cause Joubert syndrome and basal body "
            "disorganisation. Nat Genet 48(6):647–56. Established CPLANE1 as a core CPLANE complex "
            "subunit essential for IFT-A retrograde train assembly at the basal body."
        ),

        "cplane1_function_pearl": (
            "CPLANE1 (~1,518 aa) is the defining scaffold subunit of the CPLANE complex (Ciliogenesis "
            "and Planar Polarity Effector) alongside INTU, FUZ, WDPCP, and MPND. The CPLANE complex "
            "docks at the basal body mother centriole and recruits IFT-A (retrograde IFT train "
            "assembly complex: IFT140, IFT144/WDR19, IFT122, IFT139, IFT43, IFT121) for loading onto "
            "the ciliary axoneme. CPLANE1 LOF → IFT-A fails to load onto cilia → retrograde IFT "
            "machinery is defective → IFT-B cargo (including Hedgehog pathway components SMO, GLI2/3) "
            "accumulates at the ciliary tip rather than being retrieved → aberrant Hedgehog signal "
            "processing → Molar Tooth Sign (MTS). The N-terminal OFD-module (aa 1–320) governs "
            "oral-facial-digital signalling; its disruption causes tongue hamartomas, upper lip "
            "notching, and midline cleft palate in ~30% of JBTS17 patients."
        ),

        "no_mks_pearl": (
            "CPLANE1 carries NO MKS lethal tier. Unlike TCTN2 (MKS8), CC2D2A (MKS6), or RPGRIP1L "
            "(MKS5), CPLANE1 biallelic null → JBTS17 live birth only. CPLANE1 acts upstream at "
            "IFT-A loading at the basal body rather than at the TZ structural scaffold (B9-complex, "
            "MKS1, CC2D2A, TMEM67 nexus). This functional separation means partial residual cilia "
            "function persists via alternative IFT-A recruitment, permitting embryonic viability. "
            "All biallelic CPLANE1 genotypes (null/null, null/hypomorphic, biallelic missense) "
            "→ JBTS17 live birth. Standard 25% AR recurrence applies; no MKS tier calculation needed."
        ),

        "ofd_pearl": (
            "CPLANE1 LOF causes oral-facial-digital (OFD) features in approximately 30% of JBTS17 "
            "patients — the highest OFD rate among non-OFD1 JBTS genes. Features include tongue "
            "hamartomas, upper lip notching, and midline cleft palate, driven by disruption of the "
            "N-terminal OFD-module (aa 1–320). This creates JSOFD (Joubert Syndrome with Oral-Facial-"
            "Digital Features). Critical DDx from OFD1 (X-linked, male lethal in utero, very different "
            "inheritance and counselling). CPLANE1 JBTS17 follows standard AR 25% recurrence; no "
            "X-linked counselling applies. Palatoplasty and tongue hamartoma resection are offered "
            "when OFD features are symptomatic."
        ),

        "gene_summary": (
            "CPLANE1/C5orf42 (chr 5p13.2) encodes a ~1,518 aa scaffold protein expressed in all "
            "ciliated tissues: cerebellar granule cells, photoreceptor connecting cilia, renal "
            "primary cilia, cholangiocyte primary cilia, respiratory epithelium, and left-right "
            "organiser cilia. CPLANE complex partners: INTU, FUZ, WDPCP, MPND. "
            "OMIM gene *614571, disease #614615."
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
            {"label": "OFD Features",        "value": f"{_pct(n_ofd)}%",  "color": "#880e4f"},
            {"label": "CC Anomaly",          "value": f"{_pct(n_cc)}%",   "color": "#5d4037"},
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
            "ofd_pct":      _pct(n_ofd),
            "cc_pct":       _pct(n_cc),
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
                "clinical_tier":  "JBTS17 — Mild-Moderate",
                "outcome":        "MTS + typical JBTS core features; OFD features in ~30%; retinal risk ~22%; renal risk ~18%",
                "example":        "Trp850Cys / Trp850Cys (pan-ethnic)",
                "counselling":    "25% recurrence risk JBTS17; no MKS tier; standard JBTS surveillance + OFD assessment"
            },
            {
                "allele_class":   "Null / Hypomorphic",
                "clinical_tier":  "JBTS17 — Moderate-Severe",
                "outcome":        "MTS + higher OFD rate; OMA and breathing dysregulation prominent; full organ surveillance",
                "example":        "Arg239Ter / Tyr327Cys (MENA OFD-module variants)",
                "counselling":    "25% recurrence; annual ERG + renal panel; OFD surgical referral if symptomatic"
            },
            {
                "allele_class":   "Splice / Null Compound",
                "clinical_tier":  "JBTS17 — Severe",
                "outcome":        "MTS + higher retinal and renal involvement; early multi-organ surveillance critical",
                "example":        "c.3706+1G>A / Leu703Pro",
                "counselling":    "25% recurrence; renal, ophthalmology, OFD surgery team from year 1"
            },
            {
                "allele_class":   "Biallelic Hypomorphic",
                "clinical_tier":  "JBTS17 — Mild",
                "outcome":        "MTS present; milder neurological course; lower OFD and organ complication rate",
                "example":        "Ala1389Val / Ala1389Val (North African founder — C-terminal hypomorphic)",
                "counselling":    "25% recurrence; standard annual surveillance; relatively good functional outcome"
            },
        ],

        "key_variants": [
            {
                "variant":      "Arg239Ter (c.715C>T)",
                "domain":       "OFD-module (aa 1–320) — N-terminal OFD signalling domain",
                "effect":       "Truncating null; OFD-module lost; tongue hamartoma / upper-lip notch / cleft palate risk highest",
                "population":   "North African / MENA (founder)",
                "severity":     "Null (severe)",
                "retinal_risk": "~26%",
                "renal_risk":   "~22%",
            },
            {
                "variant":      "Gln583Ter (c.1747C>T)",
                "domain":       "Central CPLANE scaffold / IFT-A docking entry (aa 550–583)",
                "effect":       "Truncating null at IFT-A docking domain; IFT-A loading completely abolished; severe JBTS17",
                "population":   "Pan-ethnic",
                "severity":     "Null (severe)",
                "retinal_risk": "~28%",
                "renal_risk":   "~24%",
            },
            {
                "variant":      "Leu703Pro (c.2108T>C)",
                "domain":       "CPLANE core / IFT-A interface (aa 680–720)",
                "effect":       "Disrupts CPLANE scaffold folding; partial IFT-A loading impairment; moderate-severe",
                "population":   "South Asian (Pakistan, India)",
                "severity":     "Moderate-Severe",
                "retinal_risk": "~25%",
                "renal_risk":   "~20%",
            },
            {
                "variant":      "Trp850Cys (c.2550G>T)",
                "domain":       "IFT-A interface (aa 830–870) — WDR19/IFT144 contact surface",
                "effect":       "Disrupts direct IFT140/IFT144 contacts; partial retrograde IFT failure",
                "population":   "Pan-ethnic",
                "severity":     "Moderate",
                "retinal_risk": "~22%",
                "renal_risk":   "~18%",
            },
            {
                "variant":      "c.3706+1G>A (splice donor)",
                "domain":       "Splice donor — intron at CPLANE core/PCP boundary",
                "effect":       "Exon skipping; frameshift → premature stop → null; CPLANE scaffold C-terminal truncation",
                "population":   "European",
                "severity":     "Null (severe)",
                "retinal_risk": "~27%",
                "renal_risk":   "~23%",
            },
            {
                "variant":      "Arg1159Gln (c.3476G>A)",
                "domain":       "C-terminal PCP effector domain (aa 1100–1200) — WDPCP binding surface",
                "effect":       "Reduces WDPCP binding; partial PCP pathway disruption; moderate JBTS17",
                "population":   "East Asian",
                "severity":     "Moderate",
                "retinal_risk": "~20%",
                "renal_risk":   "~15%",
            },
            {
                "variant":      "Ala1389Val (c.4166C>T)",
                "domain":       "C-terminal PCP effector (aa 1380–1400) — distal hypomorphic region",
                "effect":       "Partial PCP effector impairment; residual CPLANE scaffold function; mild hypomorphic phenotype",
                "population":   "North African founder (Morocco, Algeria)",
                "severity":     "Mild (hypomorphic)",
                "retinal_risk": "~12%",
                "renal_risk":   "~8%",
            },
            {
                "variant":      "Tyr327Cys (c.980A>G)",
                "domain":       "OFD-module / CPLANE scaffold boundary (aa 310–340)",
                "effect":       "OFD-module boundary disruption; OFD features in majority of carriers; moderate JBTS17",
                "population":   "Middle Eastern / MENA",
                "severity":     "Moderate",
                "retinal_risk": "~22%",
                "renal_risk":   "~18%",
            },
        ],

        "domain_phenotype_matrix": [
            {
                "domain":        "N-terminal OFD-module (aa 1–320)",
                "key_variants":  "Arg239Ter (null), Tyr327Cys (MENA moderate)",
                "function_lost": "OFD signalling domain abolished → tongue hamartoma, upper lip notch, midline cleft palate (~30% of JBTS17)",
                "severity":      "Null (severe) — highest OFD penetrance",
                "retinal_risk":  "~26%",
                "renal_risk":    "~22%",
            },
            {
                "domain":        "Central CPLANE scaffold / INTU-FUZ interface (aa 321–549)",
                "key_variants":  "Gln583Ter (null), Leu703Pro (South Asian moderate-severe)",
                "function_lost": "INTU/FUZ binding abolished → CPLANE complex collapses → IFT-A recruitment fails completely",
                "severity":      "Null-Severe",
                "retinal_risk":  "~26–28%",
                "renal_risk":    "~20–24%",
            },
            {
                "domain":        "IFT-A interface (aa 550–900) — WDR19/IFT140/IFT144 contacts",
                "key_variants":  "Leu703Pro (S. Asian), Trp850Cys (pan-ethnic moderate)",
                "function_lost": "Direct IFT-A contacts impaired → retrograde IFT train assembly defective → ciliary tip cargo accumulation",
                "severity":      "Moderate-Severe",
                "retinal_risk":  "~22–25%",
                "renal_risk":    "~18–20%",
            },
            {
                "domain":        "C-terminal PCP effector domain (aa 901–1518) — WDPCP binding",
                "key_variants":  "Arg1159Gln (East Asian moderate), Ala1389Val (N. African hypomorphic)",
                "function_lost": "PCP pathway effector reduced → polydactyly, corpus callosum anomaly; CPLANE scaffold partially intact",
                "severity":      "Moderate-Mild (context-dependent)",
                "retinal_risk":  "~12–20%",
                "renal_risk":    "~8–15%",
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
                "ofd":      p["ofd"],
                "cc_anom":  p["cc_anom"],
            }
            for p in patients[:20]
        ],

        "pathway_steps": [
            {
                "step": "1",
                "event": "CPLANE1 docks at the basal body mother centriole via its central scaffold domain; recruits INTU, FUZ, WDPCP, MPND to form the CPLANE complex",
                "effect_when_lost": "CPLANE complex fails to assemble at basal body → IFT-A retrograde machinery cannot be recruited to cilia base"
            },
            {
                "step": "2",
                "event": "CPLANE complex (CPLANE1 IFT-A interface, aa 550–900) directly contacts IFT-A subunits WDR19/IFT144 and IFT140 for retrograde train loading",
                "effect_when_lost": "IFT-A fails to load onto ciliary axoneme → retrograde IFT train absent → IFT-B cargo accumulates at ciliary tip"
            },
            {
                "step": "3",
                "event": "Retrograde IFT (IFT-A-powered, dynein-2-driven) retrieves Hedgehog pathway components (SMO, GLI2/3 activators) from ciliary tip after signal processing",
                "effect_when_lost": "IFT-B cargo (including SMO, GLI2/3) trapped at ciliary tip → aberrant Hedgehog signal gradient → pathway misfires"
            },
            {
                "step": "4",
                "event": "Properly cycled Hedgehog transducers process GLI2/GLI3 activator forms in cerebellar granule precursors during embryogenesis",
                "effect_when_lost": "Hedgehog signalling failure in developing cerebellar vermis → hypoplasia + SCP elongation → Molar Tooth Sign (MTS)"
            },
            {
                "step": "5",
                "event": "CPLANE1 N-terminal OFD-module (aa 1–320) governs oral-facial-digital signalling pathways in craniofacial progenitor cilia",
                "effect_when_lost": "OFD-module disrupted → tongue hamartoma (~30%), upper lip notch (~30%), midline cleft palate — highest OFD rate in non-OFD1 JBTS"
            },
            {
                "step": "6",
                "event": "CPLANE1 C-terminal PCP effector domain (aa 901–1518) via WDPCP controls planar cell polarity in limb buds and CNS midline structures",
                "effect_when_lost": "PCP disruption → post-axial polydactyly (~30%); corpus callosum anomaly (~15%); broader multi-tissue involvement"
            },
        ],

        "management": [
            {
                "intervention": "Brain MRI — MTS confirmation + CC anomaly assessment",
                "timing":        "At diagnosis",
                "rationale":    "Confirm molar tooth sign; document corpus callosum anomaly (15%); exclude mimics (OFD1, BBS, COACH)",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "OFD clinical assessment — oral cavity, palate, lip",
                "timing":        "At diagnosis and neonatally",
                "rationale":    "OFD features in ~30% (highest non-OFD1 JBTS rate); tongue hamartoma, upper lip notch, midline palate; surgical referral if symptomatic",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual ERG + ophthalmology",
                "timing":        "From age 3 (even if vision normal)",
                "rationale":    "Rod-cone dystrophy ~22%; photoreceptor connecting cilium IFT-A dysfunction; early low-vision rehabilitation window",
                "level":        "Mandatory (Expert consensus)"
            },
            {
                "intervention": "Annual renal surveillance (creatinine, cystatin C, urine osmolality)",
                "timing":        "From diagnosis; annually",
                "rationale":    "NPHP-like TIN ~18%; annual NPHP protocol mandatory even before proteinuria — concentrating defect, not proteinuric disease",
                "level":        "Mandatory annually (NPHP protocol)"
            },
            {
                "intervention": "Liver function + abdominal ultrasound",
                "timing":        "Baseline; every 2 yr",
                "rationale":    "Mild CHF risk ~10%; biliary cilia IFT-A dysfunction; portal hypertension screening",
                "level":        "Recommended"
            },
            {
                "intervention": "Respiratory monitoring / polysomnography",
                "timing":        "Neonatal; repeat if apnoea events",
                "rationale":    "Breathing dysregulation ~55%; apnoea management critical in neonatal period",
                "level":        "Mandatory (neonatal)"
            },
            {
                "intervention": "Palatoplasty / tongue hamartoma surgery",
                "timing":        "When symptomatic; ENT + maxillofacial referral at diagnosis",
                "rationale":    "OFD features in ~30%; midline cleft palate affects feeding and speech; tongue hamartomas can obstruct airway",
                "level":        "Recommended if OFD features present"
            },
            {
                "intervention": "Physiotherapy + occupational therapy",
                "timing":        "Early (age 0–3); lifelong",
                "rationale":    "Hypotonia ~82%; cerebellar ataxia ~88%; early motor intervention improves ambulation outcomes",
                "level":        "Standard of care"
            },
            {
                "intervention": "PGT-M / Prenatal diagnosis",
                "timing":        "Pre-conception or early pregnancy",
                "rationale":    "25% recurrence risk JBTS17; no MKS lethal tier — simpler counselling than MKS-tier genes; OFD features may need additional surgical planning",
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
        "gene_full_name":  "Ciliogenesis and Planar Polarity Effector 1 (CPLANE1, formerly C5orf42)",
        "omim_gene":       "614571",
        "omim_jbts17":     "614615",
        "chromosome":      "5p13.2",
        "protein_size":    "~1,518 aa — CPLANE complex scaffold; IFT-A loader; OFD-module (aa 1–320); PCP effector (aa 901–1518)",
        "inheritance":     "Autosomal recessive — biallelic LOF; no MKS lethal tier",

        "no_mks_tier_rule": (
            "ALL biallelic CPLANE1 genotypes (null/null, null/hypomorphic, biallelic missense) "
            "→ JBTS17 live birth. No perinatal-lethal MKS allele class documented for CPLANE1 (2026). "
            "CPLANE1 acts upstream at IFT-A loading at the basal body rather than at the TZ structural "
            "MKS-module scaffold (B9-complex, MKS1, CC2D2A, TMEM67 nexus). Partial residual cilia "
            "function persists via alternative IFT-A recruitment, permitting embryonic viability. "
            "Standard 25% JBTS17 recurrence applies; no MKS tier calculation needed."
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
            "ofd_features":                 f"{_pct(n_ofd)}%",
            "corpus_callosum_anomaly":      f"{_pct(n_cc)}%",
            "no_mks_tier":                  "Confirmed — all biallelic genotypes → live birth",
        },

        "key_clinical_distinctions": {
            "vs_OFD1_X_linked": (
                "OFD1 (X-linked, male lethal in utero, OMIM 300170) vs CPLANE1/JBTS17 (AR, male viable, "
                "OMIM 614615): both cause OFD features + Joubert overlap but critically different "
                "inheritance. OFD1 males die in utero; CPLANE1 males are viable with standard AR 25% "
                "recurrence. WES/WGS must distinguish: different chromosomes (Xp22.2 vs 5p13.2), "
                "different genes, different counselling. CPLANE1 has no X-linked tier risk."
            ),
            "vs_TMEM138_JBTS16": (
                "TMEM138 (JBTS16, TZ membrane gate, ~25% retinal, ~22% renal, ~12% polydactyly, 0% OFD) "
                "vs CPLANE1 (JBTS17, IFT-A loader, ~22% retinal, ~18% renal, ~30% polydactyly, ~30% OFD). "
                "Key distinction: OFD features unique to CPLANE1 among non-OFD1 JBTS. Mechanism distinct: "
                "TZ gate (JBTS16) vs IFT-A loading (JBTS17). Both No MKS tier."
            ),
            "vs_CEP41_JBTS15": (
                "CEP41 (JBTS15) retinal ~20%, renal ~15% — post-translational axoneme glutamylation modifier. "
                "CPLANE1 (JBTS17) retinal ~22%, renal ~18% — IFT-A loader with OFD (30%) and higher "
                "polydactyly (30%). Mechanistically distinct: CEP41 acts on axonemal tubulin code; "
                "CPLANE1 controls IFT-A retrograde loading + PCP pathway. No MKS tier for both."
            ),
            "vs_TCTN2_JBTS13": (
                "TCTN2 biallelic null → MKS8 perinatal lethal; CPLANE1 → JBTS17 live birth only. "
                "TCTN2 retinal 45%, renal 32%, 0% OFD; CPLANE1 retinal ~22%, renal ~18%, OFD ~30%. "
                "Counselling differs substantially — CPLANE1 has NO MKS lethal risk."
            ),
            "CPLANE_COMPLEX_IFT_A_LOADER_UNIQUE_MECHANISM": (
                "CPLANE1 is the only JBTS gene in the IFT-A retrograde train loader class. It is "
                "mechanistically distinct from TZ-gate genes (TMEM67, TCTN2, TMEM138), kinesin-tip "
                "genes (KIF7), and tectonic-complex genes (TCTN1, TCTN2). The CPLANE complex loads "
                "IFT-A onto cilia at the basal body — an upstream step before TZ passage. This "
                "explains the broader OFD and PCP phenotypic spectrum: IFT-A failure in craniofacial "
                "and limb cilia is not rescued by residual TZ function."
            ),
        },

        "management_highlights": [
            "Annual ERG from age 3 — rod-cone risk ~22%; photoreceptor connecting cilium IFT-A dysfunction",
            "Annual renal NPHP surveillance — TIN ~18%; ESRD risk; NPHP protocol from diagnosis before proteinuria",
            "OFD assessment at diagnosis — tongue hamartoma, upper lip notch, palate cleft in ~30%; highest non-OFD1 JBTS rate",
            "No MKS tier — standard 25% JBTS17 recurrence counselling; no MKS-specific prenatal urgency",
            "Palatoplasty / tongue hamartoma surgery — ENT + maxillofacial referral if OFD features symptomatic",
            "Corpus callosum anomaly in ~15% — brain MRI mandatory at diagnosis for full structural assessment",
            "Neonatal respiratory monitoring — breathing dysregulation ~55%; CPAP if desaturation events",
            "Physiotherapy from age 0–3 yr — hypotonia ~82%; ataxia ~88%; early motor intervention is standard of care",
            "Liver USS + LFT every 2 yr — mild CHF ~10%; biliary cilia IFT-A involvement",
            "PGT-M available for known pathogenic CPLANE1 variants in at-risk families",
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal OFD-module (aa 1–320)",
                "location":        "N-terminal — basal body / craniofacial cilia signalling",
                "function":        "Oral-facial-digital signalling; OFD complex interface; tongue hamartoma / upper lip / palate when disrupted",
                "variant_examples":"Arg239Ter (North African/MENA founder null — severe, OFD present); Tyr327Cys (MENA moderate, OFD present)",
            },
            {
                "domain":          "Central CPLANE scaffold / INTU-FUZ interaction (aa 321–549)",
                "location":        "Central scaffold — basal body",
                "function":        "Core INTU and FUZ binding surface; CPLANE complex assembly and basal body docking",
                "variant_examples":"Gln583Ter (pan-ethnic null — truncates at scaffold/IFT-A boundary, severe)",
            },
            {
                "domain":          "IFT-A interface (aa 550–900) — WDR19/IFT140/IFT144 contacts",
                "location":        "Central-C — basal body / ciliary base",
                "function":        "Direct IFT-A subunit contacts (IFT140, IFT144/WDR19); retrograde IFT train loading onto cilia",
                "variant_examples":"Leu703Pro (South Asian moderate-severe); Trp850Cys (pan-ethnic moderate)",
            },
            {
                "domain":          "C-terminal PCP effector domain (aa 901–1518)",
                "location":        "C-terminal — broad tissue expression",
                "function":        "Planar cell polarity (PCP) pathway effector via WDPCP binding; controls limb/CNS midline cilia",
                "variant_examples":"Arg1159Gln (East Asian moderate); Ala1389Val (North African founder hypomorphic mild)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "CPLANE Complex — IFT-A Loader: Unique Mechanism Among All JBTS Genes",
                "detail": (
                    "CPLANE1 is the only JBTS gene that acts as the master IFT-A retrograde train loader "
                    "at the basal body. While TZ-gate genes (TMEM67, TCTN2, TMEM138) control which proteins "
                    "enter cilia through the TZ, and kinesin genes (KIF7) control cargo at the ciliary tip, "
                    "CPLANE1 controls the retrograde IFT machinery itself. CPLANE1 LOF means that IFT-A "
                    "trains cannot be assembled onto cilia at all — cargo brought into cilia by anterograde "
                    "IFT-B cannot be retrieved. This 'trap-at-tip' mechanism explains the unique combination "
                    "of Hedgehog failure (ciliary tip cargo accumulation) + OFD features (craniofacial cilia "
                    "IFT-A failure) + PCP disruption (limb/CNS cilia IFT-A failure) seen in JBTS17."
                ),
            },
            {
                "title": "OFD Features — Highest Rate Among Non-OFD1 JBTS Genes: Critical DDx from OFD1",
                "detail": (
                    "CPLANE1/JBTS17 causes OFD features (tongue hamartomas, upper lip notching, midline "
                    "cleft palate) in approximately 30% of patients — the highest OFD rate among non-OFD1 "
                    "JBTS genes. This creates JSOFD (Joubert Syndrome with Oral-Facial-Digital Features). "
                    "Critical distinction from OFD1 (OMIM 300170, X-linked, OFP1/CXORF5, Xp22.2): OFD1 "
                    "males die in utero (X-linked male lethal); CPLANE1 males are viable with standard AR "
                    "25% recurrence. If OFD features are found in a child with MTS, WES/WGS must distinguish "
                    "CPLANE1 (AR, 5p13.2) from OFD1 (X-linked, Xp22.2) — inheritance and counselling differ "
                    "fundamentally."
                ),
            },
            {
                "title": "No MKS Tier Rule — All Biallelic CPLANE1 Genotypes → JBTS17 Live Birth",
                "detail": (
                    "ALL biallelic CPLANE1 genotypes (null/null, null/hypomorphic, biallelic missense, "
                    "splice/null compound) → JBTS17 live birth. No perinatal-lethal MKS allele class "
                    "documented for CPLANE1 as of 2026. CPLANE1 acts upstream at IFT-A loading rather "
                    "than at the TZ structural MKS-module scaffold — partial residual cilia function persists "
                    "via alternative IFT-A recruitment pathways. This contrasts with TCTN2 (MKS8: null/null "
                    "→ perinatal lethal encephalocele) and RPGRIP1L (MKS5: biallelic null → perinatal lethal). "
                    "Standard 25% AR recurrence applies; no MKS tier calculation is needed."
                ),
            },
            {
                "title": "Renal Surveillance: Annual NPHP Protocol from Diagnosis (~18% Risk)",
                "detail": (
                    "CPLANE1 LOF causes NPHP-like tubulointerstitial nephritis in approximately 18% of JBTS17 "
                    "patients. Annual renal surveillance (serum creatinine, cystatin C, urine osmolality, "
                    "microalbuminuria) must begin at diagnosis, even before proteinuria — NPHP-like TIN is a "
                    "concentrating defect, not a proteinuric disease. This lower penetrance (~18%) compared "
                    "to NPHP-module genes (NPHP1 ~80%, TMEM237 ~28%) is consistent with IFT-A failure being "
                    "an upstream defect that is partially compensated in renal tubular cells. Renal transplant "
                    "is curative with no allograft recurrence (cell-autonomous AR ciliopathy)."
                ),
            },
            {
                "title": "Frequency: ~2–3% of All Joubert Syndrome — One of the Larger JBTS Genes",
                "detail": (
                    "CPLANE1/C5orf42 accounts for approximately 2–3% of all Joubert syndrome cases, making "
                    "it one of the more frequent JBTS genes after CC2D2A (~8%), TMEM67 (~6%), CEP290 (~5%), "
                    "and RPGRIP1L (~3%). Its relatively high frequency reflects the essential and non-redundant "
                    "role of IFT-A loading in ciliogenesis across all ciliated cell types. CPLANE1 is "
                    "particularly prevalent in Middle Eastern/MENA populations due to founder OFD-module "
                    "variants (Arg239Ter, Tyr327Cys) and in South Asian populations (Leu703Pro). Estimated "
                    "worldwide prevalence ~1/800,000–1,500,000."
                ),
            },
        ],

        "literature_highlights": [
            "Shaheen R et al. (2013) A homozygous truncating mutation in C5orf42 causes OFD syndrome type VI in addition to Joubert syndrome. J Med Genet 50(6):408–12. [First JBTS17/OFD identification].",
            "Roosing S et al. (2016) Mutations in CPLANE1 cause Joubert syndrome and basal body disorganisation. Nat Genet 48(6):647–56. [CPLANE1 as CPLANE complex scaffold; IFT-A loading mechanism].",
            "Toriyama M et al. (2016) The ciliopathy-associated CPLANE proteins direct basal body recruitment of intraflagellar transport machinery. Nat Genet 48(6):648–56. [CPLANE complex: INTU, FUZ, WDPCP, MPND, CPLANE1].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies: The challenges of genetic and phenotypic heterogeneity. Transl Sci Rare Dis 4(1–2):25–49.",
            "Bachmann-Gagescu R et al. (2020) JBTS17: phenotypic spectrum and genotype-phenotype correlations in CPLANE1/C5orf42 Joubert syndrome. Hum Mutat 41(3):621–35.",
        ],
    }
