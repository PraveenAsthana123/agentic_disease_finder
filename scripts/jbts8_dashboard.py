"""
ARL13B Joubert Syndrome Type 8 (JBTS8) — GTPase / INPP5E Trafficking / No-MKS
=============================================================================
Primary Gene : ARL13B (*608922) — 3q11.1; 428 aa; ADP-Ribosylation Factor-Like
               GTPase 13B (also JBTS8-GTPase; Caenorhabditis elegans ortholog: ARL-13)
               ARL13B is a CILIARY MEMBRANE RAFT GTPase / INPP5E TRAFFICKING CHAPERONE.
               ARL13B contains: N-terminal GTPase domain (aa 1–250; switch I R79/Y86,
               switch II, effector-binding platform), ALPS motif (Amphipathic Lipid-Packing
               Sensor, aa 251–280; ciliary membrane curvature sensing / membrane targeting),
               and C-terminal coiled-coil (aa 280–428; ARL3 GEF activity / protein docking).
               ARL13B is constitutively GTP-bound (very low intrinsic GTPase activity).
               It localises to the ciliary membrane raft and recruits INPP5E to cilia
               (via the INPP5E CAAX motif and ARL13B effector loop). INPP5E converts
               PI(4,5)P₂ → PI(4)P at the ciliary membrane, maintaining PI(4)P-rich
               ciliary identity. ARL13B loss → INPP5E cannot reach cilia → PI(4,5)P₂
               accumulates at ciliary membrane → SMOOTHENED/GPCR misrouting → Hedgehog
               failure → Molar Tooth Sign (MTS), cerebellar vermis hypoplasia.
               ARL13B also acts as a GEF (Guanine nucleotide Exchange Factor) for ARL3,
               activating ARL3-GTP which then displaces lipidated cargoes from PDEδ/UNC119,
               releasing INPP5E/PDE6D-dependent prenylated proteins into cilia.
               UNIQUE FEATURE: ARL13B has NO MKS-lethal tier. Unlike CEP290 (MKS4),
               TMEM67 (MKS3), RPGRIP1L (MKS5), CC2D2A (MKS6), or TCTN2 (MKS8) —
               biallelic ARL13B null variants cause JBTS8 (not Meckel-Gruber syndrome).
               This makes ARL13B allele-class interpretation simpler: all biallelic
               LOF → JBTS8. No severity stratification by allele class required.
               ~1% of all Joubert syndrome cases worldwide.
Disease OMIM : #612291 — Joubert Syndrome 8 (JBTS8)
               Only one OMIM disease entry — no MKS/NPHP/BBS allele tiers.
Chromosome   : 3q11.1
Inheritance  : Autosomal Recessive — biallelic LOF; all biallelic LOF → JBTS8
               (no lethal allele tier, unlike most other JBTS genes)
Prevalence   : ~1% of all Joubert syndrome cases
               ~1/2,000,000–5,000,000 worldwide (JBTS8 phenotype)

⚠ KEY DIAGNOSTIC PEARL — NO MKS TIER (ARL13B UNIQUE):
ARL13B is the ONLY major Joubert syndrome gene WITHOUT an MKS-lethal allele tier.
Unlike CEP290 (MKS4), RPGRIP1L (MKS5), or TMEM67 (MKS3), biallelic null ARL13B
variants do NOT cause Meckel-Gruber syndrome. This means:
(1) All biallelic ARL13B LOF variants (regardless of allele class: null/null,
    null/missense, missense/missense) cause JBTS8 — no prenatal lethality.
(2) Do NOT defer genetic counselling expecting MKS risk for subsequent pregnancies:
    sibling recurrence risk is JBTS8, not MKS.
(3) ARL13B variants can appear in MKS gene panels without causing MKS — report
    as JBTS8, NOT as MKS5/MKS3/MKS4 mis-classification.

⚠ INPP5E TRAFFICKING AXIS — DIAGNOSTIC IMPLICATION:
ARL13B and INPP5E (JBTS1 gene) function in the SAME ciliary pathway:
ARL13B recruits INPP5E → INPP5E converts PI(4,5)P₂ → PI(4)P at ciliary tip.
PI(4)P = ciliary identity lipid. Loss of either gene → identical PI(4,5)P₂
accumulation → SMOOTHENED misrouting → MTS. Implication: patients with
biallelic INPP5E variants (JBTS1) and patients with biallelic ARL13B variants
(JBTS8) have overlapping ciliary PI metabolism defects. WES panels must
sequence BOTH genes; ARL13B null variants that disrupt INPP5E binding (switch I /
effector loop) are high-confidence pathogenic even if not recurrent.

⚠ Arg79Gln NORTH AFRICAN / EUROPEAN FOUNDER ALLELE (c.236G>A):
p.Arg79Gln is the most common ARL13B disease allele in North African and
Southern European cohorts. Arg79 sits in the switch I region (P-loop adjacent);
Gln substitution reduces GTP-effector coupling without abolishing GTP binding —
hypomorphic/partial LOF. Most JBTS8 patients from Tunisia, Morocco, and Southern
Europe carry p.Arg79Gln in homozygosity or as compound het with a truncating null.
Biallelic p.Arg79Gln = mild JBTS8 (MTS + mild ataxia + oculomotor apraxia).

Protein Structure — ARL13B (428 aa; ciliary membrane GTPase)
--------------------------------------------------------------
Domain 1: GTPase domain (aa 1–250)              — ARF/ARL family GTP-binding fold
           Switch I (aa 70–90, DTXGHR):
             Arg79Gln — switch I P-loop edge; North African/European founder; JBTS8 mild
             Tyr86Cys — switch I; MENA; JBTS8 moderate
           Switch II (aa 130–145, DPTG):
             Ala130Val — switch II; European; JBTS8 moderate
           Effector loop (aa 185–210):
             Arg200Cys — ALPS-proximal effector; pan-ethnic; JBTS8 moderate-severe
Domain 2: ALPS motif (aa 251–280)               — Amphipathic Lipid-Packing Sensor
           Required for ciliary membrane raft association; helix-loop-helix membrane sensor
           Leu261Pro — ALPS core; disrupts membrane curvature sensing; pan-ethnic; JBTS8
Domain 3: C-terminal coiled-coil (aa 280–428)  — ARL3 GEF activity; INPP5E docking
           c.246+2T>C — splice donor intron 3 (truncating null); European; JBTS8
           Leu374Pro — CC C-terminal; ARL3-GEF interface; South Asian; JBTS8 moderate

Key pathogenic variant classes (ARL13B / JBTS8):
1. p.Arg79Gln (c.236G>A):   Switch I GTPase; North African/European founder; hypomorphic
2. p.Tyr86Cys (c.257A>G):   Switch I GTPase; MENA; moderate JBTS8
3. p.Arg200Cys (c.598C>T):  ALPS-proximal effector; pan-ethnic; moderate-severe JBTS8
4. c.246+2T>C (splice):     Intron 3 splice donor null; European; JBTS8
5. p.Leu374Pro (c.1121T>C): C-terminal CC; ARL3-GEF interface; South Asian; JBTS8

JBTS8 Allele-Phenotype Rule (SIMPLIFIED — no MKS tier):
  All biallelic LOF (null/null, null/missense, missense/missense) → JBTS8
  Arg79Gln/Arg79Gln → mild JBTS8 (MTS + mild ataxia + OMA; lower retinal/renal)
  Null/missense     → moderate JBTS8 (MTS + ataxia + variable OMA/retinal)
  Null/null         → moderate-severe JBTS8 (MTS + higher retinal/renal frequency)
  NO MKS tier exists for ARL13B — all biallelic LOF survive to live birth

Original discovery: Cantagrel V. et al., AJHG 2008 — ARL13B identified in Tunisian
  families with Joubert syndrome; p.Arg79Gln identified as founder allele.
"""

import random
import math

SEED = 423
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
    ('North African',                  0.32),  # Arg79Gln founder enriched (Tunisia/Morocco)
    ('European',                       0.28),  # Arg79Gln + splice null European
    ('Middle Eastern / MENA',          0.20),  # Tyr86Cys / Arg200Cys
    ('South Asian',                    0.12),  # Leu374Pro / compound het
    ('Other / Unknown',                0.05),
    ('East Asian',                     0.03),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous p.Arg79Gln (North African/European founder — JBTS8 mild)',
    'Compound het: null (c.246+2T>C) + p.Arg79Gln (JBTS8 moderate)',
    'Compound het: p.Tyr86Cys + p.Arg200Cys (JBTS8 moderate)',
    'Compound het: null + missense (JBTS8 moderate-severe)',
    'Compound het: p.Leu374Pro + p.Arg79Gln (JBTS8 mild — South Asian)',
]
allele_fracs = [0.35, 0.28, 0.18, 0.12, 0.07]
allele_pool = []
for ac, frac in zip(allele_classes, allele_fracs):
    allele_pool.extend([ac] * round(frac * N))
while len(allele_pool) < N:
    allele_pool.append(allele_classes[0])
rng.shuffle(allele_pool)

age_dx_pool = (
    [rng.randint(0, 1) for _ in range(16)] +  # neonatal/infantile (MTS on MRI)
    [rng.randint(2, 8) for _ in range(18)] +   # early childhood
    [rng.randint(9, 20) for _ in range(6)]     # later (milder / Arg79Gln hom)
)
rng.shuffle(age_dx_pool)

for i in range(N):
    eth = eth_pool[i]
    allele = allele_pool[i]
    age_dx = age_dx_pool[i]
    is_arg79_hom = 'Homozygous p.Arg79Gln' in allele
    # phenotype flags
    mts          = True                           # 100% — MTS pathognomonic
    ataxia       = rng.random() < 0.88
    hypotonia    = rng.random() < 0.85
    oma          = rng.random() < 0.50
    retinal      = rng.random() < (0.20 if is_arg79_hom else 0.33)   # lower in mild
    hepatic      = False                          # NO hepatic fibrosis in ARL13B
    renal        = rng.random() < (0.08 if is_arg79_hom else 0.20)   # lower in mild
    polydactyly  = rng.random() < 0.05           # very rare
    id_          = rng.random() < 0.70
    breathing    = rng.random() < 0.55
    esrd_age     = rng.randint(18, 35) if renal else None
    patients.append({
        "id":            f"JBTS8-{i+1:03d}",
        "ethnicity":     eth,
        "allele_class":  allele,
        "age_dx_yr":     age_dx,
        "mts":           mts,
        "ataxia":        ataxia,
        "hypotonia":     hypotonia,
        "oma":           oma,
        "retinal":       retinal,
        "hepatic":       hepatic,      # always False for ARL13B
        "renal":         renal,
        "polydactyly":   polydactyly,
        "id_":           id_,
        "breathing":     breathing,
        "esrd_age":      esrd_age,
        "is_arg79_hom":  is_arg79_hom,
    })

# ── aggregate counts ──────────────────────────────────────────────────────────
n_mts         = N           # 100%
n_ataxia      = sum(1 for p in patients if p["ataxia"])
n_hypotonia   = sum(1 for p in patients if p["hypotonia"])
n_oma         = sum(1 for p in patients if p["oma"])
n_retinal     = sum(1 for p in patients if p["retinal"])
n_renal       = sum(1 for p in patients if p["renal"])
n_polydactyly = sum(1 for p in patients if p["polydactyly"])
n_id          = sum(1 for p in patients if p["id_"])
n_breathing   = sum(1 for p in patients if p["breathing"])
n_arg79_hom   = sum(1 for p in patients if p["is_arg79_hom"])

# ── API builders ──────────────────────────────────────────────────────────────
def get_overview():
    allele_dist = {}
    for p in patients:
        allele_dist[p["allele_class"]] = allele_dist.get(p["allele_class"], 0) + 1

    return {
        "gene":     "ARL13B",
        "disease":  "Joubert Syndrome Type 8 (JBTS8)",
        "omim_gene": "608922",
        "omim_disease": "612291",
        "chromosome": "3q11.1",
        "protein":  "428 aa — GTPase domain / ALPS motif / C-terminal coiled-coil (ARL3 GEF)",
        "inheritance": "Autosomal Recessive — biallelic LOF; ALL biallelic LOF → JBTS8 (no MKS tier)",
        "prevalence": "~1% of all Joubert syndrome; ~1/2,000,000–5,000,000 worldwide",
        "hallmark":  "Molar Tooth Sign (MTS) — 100%; cerebellar vermis hypoplasia; INPP5E trafficking axis",
        "no_mks_unique": (
            "ARL13B is the ONLY major Joubert gene WITHOUT an MKS-lethal tier. "
            "Biallelic null ARL13B → JBTS8 (live birth), NOT Meckel-Gruber syndrome. "
            "No allele-class tier stratification required — all biallelic LOF = JBTS8."
        ),
        "critical_diagnostic_pearl": (
            "p.Arg79Gln (c.236G>A) — switch I GTPase domain — is the MOST COMMON ARL13B allele "
            "in North African (Tunisian, Moroccan) and Southern European populations. "
            "Homozygous p.Arg79Gln causes mild JBTS8 (MTS + mild ataxia + OMA; low retinal/renal frequency). "
            "Confirmed founder allele by Cantagrel et al. 2008 (AJHG). Low frequency in gnomAD; "
            "clinical MTS on MRI is essential for pathogenicity confirmation. "
            "Unlike RPGRIP1L-Ala229Thr or TMEM67-Cys615Arg, Arg79Gln does NOT have a higher-severity "
            "sibling risk (no MKS concern); recurrence risk is JBTS8 only."
        ),
        "inpp5e_axis_pearl": (
            "ARL13B and INPP5E (JBTS1 / OMIM #243200) work in the SAME ciliary phosphoinositide pathway. "
            "ARL13B recruits INPP5E to the ciliary membrane; INPP5E converts PI(4,5)P₂ → PI(4)P. "
            "Loss of either ARL13B (JBTS8) or INPP5E (JBTS1) → PI(4,5)P₂ accumulation → "
            "SMOOTHENED/GPCR ciliary exclusion → Hedgehog failure → Molar Tooth Sign. "
            "Genetically, JBTS1 (INPP5E) and JBTS8 (ARL13B) are functionally equivalent for MTS generation. "
            "Always sequence both genes in unsolved JBTS panels."
        ),
        "first_description": "Cantagrel et al., AJHG 2008 — ARL13B identified as JBTS8 gene; p.Arg79Gln North African founder allele described",
        "gene_summary": (
            "ARL13B is a constitutively GTP-bound ARF/ARL-family GTPase that localises to the ciliary "
            "membrane raft. Its primary function is chaperoning INPP5E to cilia (via effector loop / "
            "INPP5E CAAX interaction), maintaining PI(4)P-rich ciliary identity and Hedgehog competence. "
            "ARL13B also acts as a GEF for ARL3 (C-terminal coiled-coil), activating ARL3-GTP which "
            "displaces prenylated cargoes (INPP5E, PDE6) from PDEδ/UNC119, routing them to cilia. "
            "Loss of ARL13B disrupts both direct INPP5E recruitment and ARL3-GTP-mediated trafficking."
        ),
        "cohort_size": N,
        "kpis": [
            {"label": "Molar Tooth Sign",       "value": f"{n_mts}/{N} (100%)",           "color": "#0d47a1"},
            {"label": "Cerebellar Ataxia",      "value": f"{n_ataxia}/{N} ({_pct(n_ataxia)}%)", "color": "#1565c0"},
            {"label": "Neonatal Hypotonia",     "value": f"{n_hypotonia}/{N} ({_pct(n_hypotonia)}%)", "color": "#283593"},
            {"label": "Oculomotor Apraxia",     "value": f"{n_oma}/{N} ({_pct(n_oma)}%)", "color": "#4527a0"},
            {"label": "Intellectual Disability","value": f"{n_id}/{N} ({_pct(n_id)}%)",   "color": "#6a1b9a"},
            {"label": "Breathing Dysreg.",      "value": f"{n_breathing}/{N} ({_pct(n_breathing)}%)", "color": "#880e4f"},
            {"label": "Retinal Dystrophy",      "value": f"{n_retinal}/{N} ({_pct(n_retinal)}%)", "color": "#ad1457"},
            {"label": "Renal (TIN)",            "value": f"{n_renal}/{N} ({_pct(n_renal)}%)", "color": "#00695c"},
            {"label": "Polydactyly",            "value": f"{n_polydactyly}/{N} ({_pct(n_polydactyly)}%)", "color": "#37474f"},
            {"label": "NO Hepatic Fibrosis",    "value": "0/40 (0%)",                     "color": "#1b5e20"},
            {"label": "NO MKS Tier",            "value": "Unique — all LOF → JBTS8",      "color": "#e65100"},
            {"label": "Arg79Gln Hom.",          "value": f"{n_arg79_hom}/{N} ({_pct(n_arg79_hom)}%)", "color": "#f57f17"},
        ],
        "allele_class_distribution": [
            {"allele_class": k, "count": v, "pct": _pct(v)} for k, v in allele_dist.items()
        ],
        "phenotype_summary": {
            "mts_pct":         100,
            "ataxia_pct":      _pct(n_ataxia),
            "hypotonia_pct":   _pct(n_hypotonia),
            "oma_pct":         _pct(n_oma),
            "retinal_pct":     _pct(n_retinal),
            "renal_pct":       _pct(n_renal),
            "polydactyly_pct": _pct(n_polydactyly),
            "id_pct":          _pct(n_id),
            "breathing_pct":   _pct(n_breathing),
            "hepatic_pct":     0,   # always 0 for ARL13B
        },
    }


def get_breakdown():
    # ── Ethnicity distribution ──────────────────────────────────────────────
    eth_counts = {}
    for p in patients:
        eth_counts[p["ethnicity"]] = eth_counts.get(p["ethnicity"], 0) + 1

    # ── Allele distribution ──────────────────────────────────────────────────
    allele_dist = {}
    for p in patients:
        allele_dist[p["allele_class"]] = allele_dist.get(p["allele_class"], 0) + 1

    # ── Variant table ────────────────────────────────────────────────────────
    key_variants = [
        {
            "variant":    "p.Arg79Gln (c.236G>A)",
            "domain":     "Switch I — GTPase domain (aa 79)",
            "effect":     "Hypomorphic — partial effector coupling loss; GTP binding intact",
            "population": "North African founder (Tunisian/Moroccan); Southern European",
            "phenotype":  "JBTS8 mild (MTS + mild ataxia + OMA; lower retinal/renal)",
            "frequency":  "~35-40% of all JBTS8 alleles; most common worldwide",
            "omim_class": "Hypomorphic / partially functional",
        },
        {
            "variant":    "p.Tyr86Cys (c.257A>G)",
            "domain":     "Switch I — GTPase domain (aa 86)",
            "effect":     "Moderate LOF — switch I cysteine disrupts effector binding",
            "population": "MENA (Lebanon, Iran, Egypt)",
            "phenotype":  "JBTS8 moderate (MTS + ataxia + OMA; moderate retinal risk)",
            "frequency":  "~15% of MENA JBTS8 alleles",
            "omim_class": "Pathogenic",
        },
        {
            "variant":    "p.Arg200Cys (c.598C>T)",
            "domain":     "ALPS-proximal effector loop (aa 200)",
            "effect":     "Severe LOF — disrupts INPP5E effector binding + ALPS-adjacent membrane targeting",
            "population": "Pan-ethnic",
            "phenotype":  "JBTS8 moderate-severe (MTS + ataxia + higher retinal/renal)",
            "frequency":  "~18% of JBTS8 alleles; second most common pan-ethnic",
            "omim_class": "Pathogenic — high confidence",
        },
        {
            "variant":    "c.246+2T>C (splice donor intron 3)",
            "domain":     "Intron 3 splice donor — truncating null",
            "effect":     "Null — exon 3 skipping → frameshift → NMD → complete ARL13B loss",
            "population": "European (French, Italian, German)",
            "phenotype":  "JBTS8 moderate-severe (compound het with missense)",
            "frequency":  "~12% of European JBTS8 alleles",
            "omim_class": "Pathogenic null",
        },
        {
            "variant":    "p.Leu374Pro (c.1121T>C)",
            "domain":     "C-terminal coiled-coil — ARL3 GEF interface (aa 374)",
            "effect":     "Moderate LOF — impairs ARL3 GEF activity; INPP5E indirect trafficking failure",
            "population": "South Asian (India, Pakistan, Bangladesh)",
            "phenotype":  "JBTS8 moderate (MTS + ataxia + OMA; variable retinal)",
            "frequency":  "~8% of South Asian JBTS8 alleles",
            "omim_class": "Pathogenic",
        },
    ]

    # ── Per-patient table (first 20) ─────────────────────────────────────────
    patient_rows = [
        {
            "id":          p["id"],
            "ethnicity":   p["ethnicity"],
            "allele":      p["allele_class"],
            "age_dx_yr":   p["age_dx_yr"],
            "mts":         "Yes",       # always
            "ataxia":      "Yes" if p["ataxia"]     else "No",
            "oma":         "Yes" if p["oma"]         else "No",
            "retinal":     "Yes" if p["retinal"]     else "No",
            "renal":       f"Yes (ESRD~{p['esrd_age']}yr)" if p["renal"] else "No",
            "id_":         "Yes" if p["id_"]         else "No",
            "breathing":   "Yes" if p["breathing"]   else "No",
            "hepatic":     "No (never)",
            "mks_tier":    "N/A — no MKS tier",
        }
        for p in patients[:20]
    ]

    # ── Domain-phenotype matrix ──────────────────────────────────────────────
    domain_matrix = [
        {
            "domain":         "Switch I (aa 70–90)",
            "key_variants":   "Arg79Gln, Tyr86Cys",
            "function_lost":  "Effector coupling / INPP5E effector loop interaction",
            "severity":       "Mild–Moderate JBTS8",
            "retinal_risk":   "Low (Arg79Gln hom) / Moderate (Tyr86Cys)",
            "renal_risk":     "Low",
        },
        {
            "domain":         "ALPS motif (aa 251–280)",
            "key_variants":   "Leu261Pro, Arg200Cys (adjacent)",
            "function_lost":  "Ciliary membrane curvature sensing / raft targeting",
            "severity":       "Moderate–Severe JBTS8",
            "retinal_risk":   "Moderate–High",
            "renal_risk":     "Moderate",
        },
        {
            "domain":         "C-terminal coiled-coil (aa 280–428)",
            "key_variants":   "Leu374Pro, splice null (intron 3)",
            "function_lost":  "ARL3 GEF activity / prenylated cargo release to cilia",
            "severity":       "Moderate JBTS8",
            "retinal_risk":   "Moderate",
            "renal_risk":     "Low–Moderate",
        },
    ]

    # ── Pathway table ────────────────────────────────────────────────────────
    pathway_steps = [
        {"step": 1, "event": "ARL13B-GTP at ciliary membrane raft (constitutive)",
         "effect_when_lost": "ARL13B absent → no GTP-bound scaffold at ciliary membrane"},
        {"step": 2, "event": "ARL13B recruits INPP5E to cilia via CAAX/effector loop",
         "effect_when_lost": "INPP5E cannot localise to cilia → PI(4,5)P₂ accumulates at ciliary tip"},
        {"step": 3, "event": "INPP5E converts PI(4,5)P₂ → PI(4)P at ciliary membrane",
         "effect_when_lost": "PI(4)P-rich ciliary identity lost → GPCR/SMOOTHENED misrouting"},
        {"step": 4, "event": "ARL13B activates ARL3 (GEF) → ARL3-GTP displaces PDEδ-cargo",
         "effect_when_lost": "ARL3-GTP not produced → prenylated cargoes (INPP5E, PDE6) stuck in cytosol"},
        {"step": 5, "event": "SMOOTHENED enters cilia normally (PI(4)P-dependent)",
         "effect_when_lost": "SMOOTHENED excluded → GLI not activated → Hedgehog signalling fails"},
        {"step": 6, "event": "Hedgehog signalling → cerebellar vermis development",
         "effect_when_lost": "Cerebellar vermis hypoplasia → Molar Tooth Sign (MTS) on MRI"},
    ]

    # ── Treatment / management table ────────────────────────────────────────
    management = [
        {
            "intervention":    "Brain MRI — MTS confirmation",
            "timing":          "At diagnosis (neonatal if suspected prenatally)",
            "rationale":       "MTS (elongated SCPs + vermis hypoplasia) is pathognomonic for JBTS8",
            "level":           "Mandatory",
        },
        {
            "intervention":    "Ophthalmology / Electroretinogram (ERG)",
            "timing":          "Annual from diagnosis",
            "rationale":       "Rod-cone dystrophy in ~30%; INPP5E-ARL13B axis; early ERG detects subclinical retinal degeneration",
            "level":           "Level A (annual surveillance)",
        },
        {
            "intervention":    "Renal ultrasound + eGFR",
            "timing":          "Annual from diagnosis",
            "rationale":       "Tubulointerstitial nephritis (TIN) in ~15%; ESRD risk median ~25yr in affected",
            "level":           "Level A (annual surveillance)",
        },
        {
            "intervention":    "Polysomnography / sleep study",
            "timing":          "At diagnosis; repeat if breathing concerns",
            "rationale":       "Breathing dysregulation (episodic apnea/hyperpnea) in ~55%; brainstem respiratory centre involvement",
            "level":           "Level B",
        },
        {
            "intervention":    "Physiotherapy — cerebellar ataxia",
            "timing":          "From diagnosis, ongoing",
            "rationale":       "Cerebellar ataxia in ~88%; core strengthening + balance training",
            "level":           "Level A",
        },
        {
            "intervention":    "Occupational therapy",
            "timing":          "From 12 months, ongoing",
            "rationale":       "Fine motor + ADL support; oculomotor apraxia compensation strategies",
            "level":           "Level A",
        },
        {
            "intervention":    "Speech + feeding therapy",
            "timing":          "If hypotonia/dysphagia present",
            "rationale":       "Neonatal hypotonia in ~85%; feeding difficulties common in infancy",
            "level":           "Level B",
        },
        {
            "intervention":    "Renal transplant (if ESRD)",
            "timing":          "At ESRD (eGFR <15); usually 3rd–4th decade",
            "rationale":       "CURATIVE for renal endpoint; neurological/retinal NOT corrected by transplant",
            "level":           "Level A (when indicated)",
        },
        {
            "intervention":    "Genetic counselling — NO MKS risk",
            "timing":          "At diagnosis (immediate family + siblings)",
            "rationale":       "Recurrence = JBTS8 (25%); NOT MKS. Unlike TMEM67/CEP290/RPGRIP1L, no prenatal lethal allele tier. "
                               "Prenatal MRI/ultrasound for subsequent pregnancies monitors MTS (3rd trimester), not encephalocele.",
            "level":           "Mandatory counselling",
        },
    ]

    return {
        "cohort_size":            N,
        "ethnicity_distribution": [{"ethnicity": k, "count": v, "pct": _pct(v)}
                                   for k, v in sorted(eth_counts.items(), key=lambda x: -x[1])],
        "allele_distribution":    [{"allele_class": k, "count": v, "pct": _pct(v)}
                                   for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "key_variants":           key_variants,
        "domain_phenotype_matrix":domain_matrix,
        "pathway_steps":          pathway_steps,
        "management":             management,
        "patient_table":          patient_rows,
        "phenotype_counts": {
            "mts":         n_mts,
            "ataxia":      n_ataxia,
            "hypotonia":   n_hypotonia,
            "oma":         n_oma,
            "retinal":     n_retinal,
            "renal":       n_renal,
            "polydactyly": n_polydactyly,
            "id":          n_id,
            "breathing":   n_breathing,
            "hepatic":     0,
        },
    }


def get_definitions():
    return {
        "arl13b":          "ADP-Ribosylation Factor-Like GTPase 13B — ciliary membrane raft GTPase; INPP5E trafficking chaperone; ARL3 GEF",
        "jbts8":           "Joubert Syndrome Type 8 (OMIM #612291) — ARL13B LOF; all biallelic LOF → JBTS8; no MKS tier (unique)",
        "mts":             "Molar Tooth Sign — pathognomonic brain MRI; elongated superior cerebellar peduncles + cerebellar vermis hypoplasia/aplasia",
        "inpp5e":          "Inositol Polyphosphate 5-Phosphatase E — JBTS1 gene; converts PI(4,5)P₂ → PI(4)P at ciliary membrane; recruited by ARL13B",
        "alps_motif":      "Amphipathic Lipid-Packing Sensor — helix-loop-helix; senses membrane curvature; required for ARL13B ciliary membrane raft targeting",
        "arl3_gef":        "ARL3 Guanine Nucleotide Exchange Factor (GEF) — ARL13B C-terminal CC activates ARL3 (GDP→GTP); ARL3-GTP displaces prenylated cargoes from PDEδ/UNC119 for ciliary delivery",
        "switch_i":        "Switch I region (GTPase domain, aa 70–90) — conformational state coupled to GTP/GDP; effector binding platform; Arg79 and Tyr86 are key effector-coupling residues",
        "pip2_pip":        "PI(4,5)P₂ → PI(4)P conversion — INPP5E reaction at ciliary tip; PI(4)P = ciliary identity lipid; PI(4,5)P₂ accumulation (ARL13B/INPP5E loss) → SMOOTHENED exclusion",
        "smo_hedgehog":    "SMOOTHENED (SMO) — GPCR that requires ciliary localisation for Hedgehog pathway activation; excluded from cilia when PI(4,5)P₂ accumulates → GLI not activated → MTS",
        "no_mks_unique":   "ARL13B uniquely lacks an MKS-lethal allele tier — all biallelic LOF (null/null, null/missense, missense/missense) cause JBTS8, not Meckel-Gruber syndrome",
        "arg79gln_pearl":  "p.Arg79Gln (c.236G>A) — switch I; most common ARL13B allele; North African/Southern European founder; hypomorphic; biallelic → mild JBTS8 (not MKS); Cantagrel 2008",
        "oma":             "Oculomotor Apraxia — horizontal gaze initiation failure; compensatory head thrusting; ~50% in JBTS8",
        "rod_cone":        "Rod-cone dystrophy — photoreceptor degeneration; INPP5E-ARL13B PI(4)P axis in connecting cilium; ERG mandatory annual surveillance in JBTS8",
        "tin":             "Tubulointerstitial Nephritis (TIN) — interstitial fibrosis + tubular atrophy; mild renal involvement in ~15% JBTS8; ESRD risk median ~25yr (lower than JBTS4/JBTS5)",
        "therapy_status":  "No disease-modifying therapy 2026 for ARL13B/JBTS8; renal transplant CURATIVE for renal endpoint; neurological/retinal NOT corrected; active research: PI metabolism modulators",
        "inheritance":     "Autosomal Recessive — biallelic loss-of-function; all biallelic LOF = JBTS8; recurrence risk 25%; carrier parents phenotypically normal",
        "frequency":       "~1% of all Joubert syndrome; ~1/2,000,000–5,000,000 worldwide; rare but important as NO-MKS-tier prototype",
        "cantagrel_2008":  "Cantagrel V. et al. (AJHG 2008) — original discovery of ARL13B as JBTS8 gene; identified p.Arg79Gln founder allele in North African JBTS cohort",
        "related_genes":   "INPP5E (JBTS1 — same pathway); AHI1 (JBTS3); NPHP1 (JBTS4); CEP290 (JBTS5); TMEM67 (JBTS6 — MKS3); RPGRIP1L (JBTS7 — MKS5); ARL3 (ARL13B GEF substrate)",
    }
