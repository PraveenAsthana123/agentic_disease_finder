"""
C21orf2 Short-Rib Thoracic Dysplasia 12 (SRTD12) — Jeune Asphyxiating Thoracic Dystrophy Type 12
====================================================================================================
Primary Gene : C21orf2 (*603503) — Chromosome 21 Open Reading Frame 2 (also: LRRC76) — 21q22.3;
               ~322 aa; Leucine-Rich Repeat (LRR) domain protein with 9 LRR motifs (aa 37–289).
               C21orf2 is a CILIARY LRR SCAFFOLD that occupies a UNIQUE BRIDGING POSITION:
               it physically interacts with NEK1 (SRTD6 gene — basal body kinase / TTBK2 activator)
               AND associates with the IFT-A complex (WDR19/SRTD5, IFT140/SRTD9 vicinity),
               making it the ONLY known SRTD protein that bridges both the basal body kinase class
               and the IFT-A retrograde adaptor class.
               Loss → dual ciliary defect:
                 (1) Partial disruption of NEK1 substrate phosphorylation → reduced TTBK2 activation
                     → incomplete CP110 removal → reduced axoneme nucleation efficiency.
                 (2) Destabilisation of IFT-A particle at the basal body transition zone →
                     IFT-B import failure → IFT-B pile-up at cilia base → shortened stubby cilia.
               Net EM result: VARIABLE — SHORT STUBBY (IFT-A-like) to partially ABSENT;
               unlike SRTD6/NEK1 (fully absent) and unlike pure IFT-A SRTDs (short stubby only).
               This variable EM is the primary diagnostic clue pointing to C21orf2.
               No axoneme extension → Hedgehog (Ihh/Shh) failure → narrow thorax.
               Additional: C21orf2 is required for the photoreceptor connecting cilium;
               hypomorphic LRR-9 alleles (C-terminal) → retinal dystrophy without or with minimal
               thoracic disease — creating a DUAL PHENOTYPE similar to IFT172/SRTD10 but through
               a different molecular pathway.

Disease OMIM : #616012 — Short-Rib Thoracic Dysplasia 12 with or without polydactyly
               (SRTD12 / ATD12 / Jeune Syndrome 12)
               Allele spectrum:
               — Severe: biallelic null → perinatal lethal (SRPS spectrum), absent cilia
               — Moderate: biallelic missense (LRR core) → survivable Jeune ATD12
               — Mild: hypomorphic C-terminal LRR-9 missense → retinal dominant; mild thoracic
               UNIQUE FEATURES vs other SRTD genes:
               (1) BRIDGING MOLECULAR CLASS — only SRTD gene bridging NEK1 (SRTD6, basal body
                   kinase) and IFT-A complex; represents a fifth molecular sub-class.
               (2) VARIABLE EM — short stubby AND partially absent in same cohort; not pure IFT-A,
                   not pure basal body → EM alone cannot classify SRTD12 (gene panel mandatory).
               (3) DUAL PHENOTYPE — hypomorphic C-terminal LRR-9 alleles → retinal dystrophy
                   dominant (LCA/CORD-like) without significant thoracic disease; same gene as SRTD12.
               (4) CHROMOSOME 21q22.3 — Down syndrome region; heterozygous C21orf2 loss rarely
                   seen in trisomy 21 (3rd copy), not confounded with trisomy; biallelic LOF only.
               (5) ULTRA-RARE: <20 families confirmed 2026; likely under-ascertained because:
                   (a) variable EM confuses classification, (b) missing from pre-2016 SRTD panels,
                   (c) some retinal-only cases not linked to SRTD12.
Chromosome   : 21q22.3
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1:800,000–1,500,000; <20 families worldwide 2026; ultra-rare.

Protein Structure — C21orf2 (322 aa; NEK1-interacting LRR scaffold)
--------------------------------------------------------------------
Domain 1: N-terminal cap (aa 1–36) — NLC (N-terminal LRR Cap) motif; protein folding;
           pathogenic variants here destabilise LRR folding → complete LOF → null-equivalent.
Domain 2: LRR 1–3 (aa 37–120) — Concave surface; primary NEK1-binding interface;
           missense here → NEK1 interaction lost → TTBK2 phosphorylation reduced → partial
           ciliogenesis defect on top of IFT-A disruption → moderate-severe SRTD12.
Domain 3: LRR 4–6 (aa 121–210) — Central concave surface; IFT-A docking interface;
           missense here → IFT-A train not anchored at basal body → IFT-B import failure
           → short stubby cilia → moderate SRTD12.
Domain 4: LRR 7–8 (aa 211–270) — Lateral surface; structural scaffold; linker to C-terminal.
           Pathogenic missense → combined NEK1 + IFT-A disruption → severe SRTD12.
Domain 5: C-terminal LRR-9 + C-cap (aa 271–322) — Photoreceptor connecting cilium function;
           hypomorphic missense here → selectively impairs photoreceptor cilia (connecting
           cilium) but spares broader IFT-A activity → RETINAL DYSTROPHY DOMINANT (LCA/CORD-like)
           with mild or absent thoracic disease; same dual phenotype paradigm as IFT172/SRTD10.

Key pathogenic variant classes (C21orf2):
1. N-cap / LRR-1 missense (aa 1–80): compound het or MENA homozygous; moderate SRTD12
2. LRR 4–6 central missense (aa 121–210): IFT-A dock disrupted; moderate-severe
3. Biallelic truncating (null × null): SRPS spectrum; perinatal lethal
4. LRR-9 / C-cap hypomorphic (aa 271–322): South Asian; retinal dominant; mild SRTD12
5. LRR 7–8 lateral missense: combined defect; severe SRTD12
"""

import random
import math

SEED = 407
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
    ('Middle Eastern / North African', 0.38),   # consanguineous — homozygous LRR missense
    ('European',                        0.25),   # compound het — LRR + truncating
    ('South Asian',                     0.18),   # hypomorphic LRR-9 — retinal dominant
    ('East Asian',                      0.10),
    ('Latin American',                  0.06),
    ('Other / Unknown',                 0.03),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous LRR-1/2/3 missense (NEK1-binding surface)',
    'Compound het: LRR-4/6 missense + truncating',
    'Biallelic truncating (null/null — SRPS spectrum)',
    'Compound het: LRR-7/8 missense + LRR-1/3 missense',
    'Hypomorphic LRR-9/C-cap missense (retinal dominant)',
]
allele_weights = [0.28, 0.25, 0.18, 0.15, 0.14]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    sex    = rng.choice(['M', 'F'])

    # Thorax severity
    thorax_map = {
        'Biallelic truncating (null/null — SRPS spectrum)':              (0.82, 0.12, 0.06),
        'Homozygous LRR-1/2/3 missense (NEK1-binding surface)':         (0.42, 0.38, 0.20),
        'Compound het: LRR-4/6 missense + truncating':                   (0.50, 0.32, 0.18),
        'Compound het: LRR-7/8 missense + LRR-1/3 missense':            (0.60, 0.28, 0.12),
        'Hypomorphic LRR-9/C-cap missense (retinal dominant)':           (0.10, 0.32, 0.58),
    }
    tw = thorax_map[allele]
    thorax = rng.choices(
        ['Severe (neonatal ventilation)', 'Moderate (CPAP/BiPAP)', 'Mild (nasal O2/none)'],
        weights=tw
    )[0]

    # Polydactyly (35–45% — moderate; postaxial dominant)
    poly_prob = 0.65 if 'truncating' in allele.lower() else 0.38 if 'hypomorphic' in allele.lower() else 0.40
    poly      = rng.random() < poly_prob
    poly_type = rng.choices(
        ['Postaxial (hands+feet)', 'Both postaxial + preaxial', 'Preaxial only'],
        weights=[0.68, 0.26, 0.06]
    )[0] if poly else None

    # Ciliary EM (variable — IFT-A-like short stubby OR partially absent)
    em_pattern = rng.choices(
        ['Short stubby (IFT-A-like)', 'Partially absent (variable — C21orf2 unique)', 'Absent / rudimentary'],
        weights=[0.50, 0.34, 0.16]
    )[0]

    # Retinal dystrophy (20–30%; LCA/CORD-like; hypomorphic alleles → retinal dominant)
    ret_prob = 0.75 if 'hypomorphic' in allele.lower() else 0.18
    retinal  = rng.random() < ret_prob
    ret_type = rng.choices(
        ['Rod-cone dystrophy (CORD-like)', 'LCA-like (severe early onset)', 'Mild photoreceptor degeneration'],
        weights=[0.50, 0.30, 0.20]
    )[0] if retinal else None

    # Retinal only presentation (hypomorphic — no thorax, only retinal)
    retinal_only = allele == 'Hypomorphic LRR-9/C-cap missense (retinal dominant)' and rng.random() < 0.45

    # Renal (15–20%)
    renal_prob = 0.35 if 'truncating' in allele.lower() else 0.14
    renal_any  = rng.random() < renal_prob
    renal_type = rng.choices(
        ['Tubulointerstitial nephritis (TIN)', 'Renal cysts (structural)', 'ESRD requiring transplant'],
        weights=[0.50, 0.35, 0.15]
    )[0] if renal_any else None

    # CHF / hepatic (5–8%)
    chf = rng.random() < 0.07

    # VEPTR / MAGEC
    veptr = thorax in ['Moderate (CPAP/BiPAP)', 'Mild (nasal O2/none)'] and (not retinal_only) and rng.random() < 0.85

    # Renal transplant
    tx = renal_type == 'ESRD requiring transplant' and rng.random() < 0.70

    # Age at diagnosis
    if retinal_only:
        age_dx_cat = rng.choices(
            ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–18 yr (teen-adult)'],
            weights=[0.10, 0.30, 0.40, 0.20]
        )[0]
    else:
        age_dx_cat = rng.choices(
            ['0–1 yr (neonatal/prenatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
            weights=[0.65, 0.22, 0.09, 0.04]
        )[0]

    # Misdiagnosis
    if not retinal_only and rng.random() < 0.50:
        mis = rng.choices(
            ['SRTD6 / NEK1 (similar absent cilia pattern)', 'SRTD5 / WDR19 (IFT-A short stubby)',
             'SRTD3 / DYNC2H1 (most common SRTD)', 'LCA / cone-rod dystrophy (retinal cases)'],
            weights=[0.32, 0.28, 0.22, 0.18]
        )[0]
    elif retinal_only and rng.random() < 0.65:
        mis = 'LCA / cone-rod dystrophy (retinal-dominant allele missed on SRTD panel)'
    else:
        mis = None

    # Perinatal death
    perinatal_death = allele == 'Biallelic truncating (null/null — SRPS spectrum)' and rng.random() < 0.40

    patients.append({
        'id': i, 'sex': sex, 'eth': eth, 'allele': allele,
        'thorax': thorax, 'poly': poly, 'poly_type': poly_type,
        'em_pattern': em_pattern,
        'retinal': retinal, 'ret_type': ret_type, 'retinal_only': retinal_only,
        'renal_any': renal_any, 'renal_type': renal_type,
        'chf': chf, 'veptr': veptr, 'tx': tx,
        'age_dx_cat': age_dx_cat, 'misdiagnosis': mis,
        'perinatal_death': perinatal_death,
    })

# ── aggregate statistics ──────────────────────────────────────────────────────
_poly_n        = sum(1 for p in patients if p['poly'])
_renal_n       = sum(1 for p in patients if p['renal_any'])
_retinal_n     = sum(1 for p in patients if p['retinal'])
_retinal_only_n= sum(1 for p in patients if p['retinal_only'])
_chf_n         = sum(1 for p in patients if p['chf'])
_veptr_n       = sum(1 for p in patients if p['veptr'])
_tx_n          = sum(1 for p in patients if p['tx'])
_mis_n         = sum(1 for p in patients if p['misdiagnosis'])
_thorax_severe_n = sum(1 for p in patients if 'Severe' in p['thorax'])
_perinatal_n     = sum(1 for p in patients if p['perinatal_death'])
_sex_M         = sum(1 for p in patients if p['sex'] == 'M')
_sex_F         = N - _sex_M


# ── API payloads ──────────────────────────────────────────────────────────────

def get_overview():
    from collections import Counter
    em_dist = Counter(p['em_pattern'] for p in patients)
    return {
        "cohort_n": N,
        "seed": SEED,
        "sex_split": {"M": _sex_M, "F": _sex_F},
        "kpis": {
            "thorax_severe_n":      _thorax_severe_n,
            "thorax_severe_pct":    _pct(_thorax_severe_n),
            "polydactyly_n":        _poly_n,
            "polydactyly_pct":      _pct(_poly_n),
            "retinal_any_n":        _retinal_n,
            "retinal_any_pct":      _pct(_retinal_n),
            "retinal_only_n":       _retinal_only_n,
            "retinal_only_pct":     _pct(_retinal_only_n),
            "renal_any_n":          _renal_n,
            "renal_any_pct":        _pct(_renal_n),
            "hepatic_chf_n":        _chf_n,
            "hepatic_chf_pct":      _pct(_chf_n),
            "veptr_any_n":          _veptr_n,
            "veptr_any_pct":        _pct(_veptr_n),
            "transplant_done_n":    _tx_n,
            "misdiagnosis_n":       _mis_n,
            "misdiagnosis_pct":     _pct(_mis_n),
            "perinatal_death_n":    _perinatal_n,
            "perinatal_death_pct":  _pct(_perinatal_n),
        },
        "em_distribution": [{"label": k, "n": v} for k, v in sorted(em_dist.items(), key=lambda x: -x[1])],
        "mechanism": (
            "C21orf2 (Chromosome 21 Open Reading Frame 2) is a 322-aa Leucine-Rich Repeat (LRR) domain "
            "protein that occupies a UNIQUE bridging position in the ciliary protein network. "
            "Its LRR-1/2/3 concave surface directly binds NEK1 (SRTD6 — the basal body kinase that "
            "activates TTBK2 to remove CP110 and initiate axoneme nucleation), while its LRR-4/5/6 "
            "concave surface docks the IFT-A complex (WDR19/IFT140/TTC21B). "
            "Loss of C21orf2 creates a DUAL CILIARY DEFECT: "
            "(1) Reduced NEK1 substrate phosphorylation → partial TTBK2 activation → incomplete CP110 "
            "removal → reduced axoneme nucleation → cilia either shorter or fail to initiate. "
            "(2) Destabilised IFT-A docking at the basal body transition zone → IFT-B import "
            "failure → IFT-B accumulation at cilia base → shortened stubby cilia. "
            "The net EM result is VARIABLE: short stubby (IFT-A mechanism dominant) or partially "
            "absent (basal body mechanism dominant) — unlike any single-mechanism SRTD. "
            "Hedgehog (Ihh/Shh) signalling fails in chondrocytes → narrow thorax (primary). "
            "Additional: C21orf2 LRR-9/C-cap (aa 271–322) is required for photoreceptor connecting "
            "cilium integrity; hypomorphic C-terminal alleles → retinal dystrophy dominant (LCA/CORD) "
            "with mild or absent thoracic disease — a dual phenotype similar to IFT172/SRTD10."
        ),
        "key_distinction": (
            "SRTD12 (C21orf2) is the ONLY SRTD gene that bridges two molecular classes: "
            "NEK1-basal body kinase (SRTD6) and IFT-A complex (SRTD4/5/7/9). "
            "Clinically distinguishing features: "
            "(1) VARIABLE CILIARY EM — short stubby (IFT-A-like) OR partially absent; unique among SRTD types; "
            "EM alone cannot distinguish SRTD12 from SRTD6 or IFT-A SRTDs — gene panel mandatory. "
            "(2) DUAL PHENOTYPE — hypomorphic LRR-9 alleles → retinal dystrophy (LCA/CORD) as primary "
            "feature; similar to IFT172/SRTD10 dual phenotype but different molecular pathway. "
            "(3) ULTRA-RARE — <20 families (2026); absent from pre-2016 SRTD panels; "
            "retinal-only cases frequently misattributed to LCA panel genes. "
            "(4) Chromosome 21q22.3 — not confused with trisomy 21; biallelic LOF only; "
            "Down syndrome patients carry 3 copies (not causative of SRTD in trisomy). "
            "(5) Missing from IFT-A-focussed and dynein-2-focussed panels — must use comprehensive "
            "ciliopathy gene panel including C21orf2."
        ),
        "srtd_molecular_class_table": [
            {"class": "IFT-B2 (Anterograde distal)", "em": "SHORTENED cilia",
             "genes": "SRTD1 (IFT80), SRTD10 (IFT172), SRTD13 (CLUAP1)", "why": "Anterograde IFT truncated; cilia cannot extend"},
            {"class": "IFT-A (Anterograde adaptor)", "em": "SHORT STUBBY cilia",
             "genes": "SRTD4 (TTC21B), SRTD5 (WDR19), SRTD7 (WDR35), SRTD9 (IFT140)", "why": "IFT-B import at cilia base blocked; uniform stubby shortening"},
            {"class": "Dynein-2 (Retrograde motor)", "em": "CLUB / BULGING TIP cilia",
             "genes": "SRTD3 (DYNC2H1), SRTD8 (WDR60), SRTD11 (WDR34), SRTD15 (DYNC2LI1), SRTD17 (TCTEX1D2)", "why": "Retrograde IFT blocked; IFT-B stranded at ciliary tip → club shape"},
            {"class": "Basal Body Kinase (SRTD6)", "em": "ABSENT / RUDIMENTARY cilia",
             "genes": "SRTD6 (NEK1)", "why": "CP110 not removed; axoneme cannot nucleate; ciliogenesis fails at step 0"},
            {"class": "NEK1-IFT-A Bridge (SRTD12)", "em": "VARIABLE — SHORT STUBBY / PARTIALLY ABSENT",
             "genes": "SRTD12 (C21orf2)", "why": "Dual defect: NEK1 interaction + IFT-A docking; EM varies by allele dominance"},
        ],
        "age_distribution": {
            "dx_0_1yr":   sum(1 for p in patients if '0–1' in p['age_dx_cat']),
            "dx_2_5yr":   sum(1 for p in patients if '2–5' in p['age_dx_cat']),
            "dx_6_10yr":  sum(1 for p in patients if '6–10' in p['age_dx_cat']),
            "dx_11_plus": sum(1 for p in patients if '11–' in p['age_dx_cat']),
        },
    }


def get_breakdown():
    from collections import Counter

    thorax_dist  = Counter(p['thorax'] for p in patients)
    poly_dist    = Counter(p['poly_type'] for p in patients if p['poly'])
    renal_dist   = Counter(p['renal_type'] for p in patients if p['renal_any'])
    eth_dist     = Counter(p['eth'] for p in patients)
    allele_dist  = Counter(p['allele'] for p in patients)
    pres_dist    = Counter(p['age_dx_cat'] for p in patients)
    mis_dist     = Counter(p['misdiagnosis'] for p in patients if p['misdiagnosis'])
    em_dist      = Counter(p['em_pattern'] for p in patients)
    ret_dist     = Counter(p['ret_type'] for p in patients if p['retinal'])

    veptr_types = Counter()
    for p in patients:
        if not p['veptr']:
            veptr_types['No surgical intervention'] += 1
        elif rng.random() < 0.55:
            veptr_types['VEPTR (expandable titanium rib)'] += 1
        else:
            veptr_types['MAGEC growing rod'] += 1

    return {
        "thorax_distribution": [{"label": k, "n": v} for k, v in sorted(thorax_dist.items(), key=lambda x: -x[1])],
        "polydactyly_distribution": [
            {"label": "Polydactyly present", "n": _poly_n},
            {"label": "No polydactyly", "n": N - _poly_n},
        ] + [{"label": k, "n": v} for k, v in sorted(poly_dist.items(), key=lambda x: -x[1])],
        "em_distribution": [{"label": k, "n": v} for k, v in sorted(em_dist.items(), key=lambda x: -x[1])],
        "retinal_distribution": [
            {"label": "Any retinal involvement", "n": _retinal_n},
            {"label": "Retinal-only (no/mild thorax)", "n": _retinal_only_n},
        ] + [{"label": k, "n": v} for k, v in sorted(ret_dist.items(), key=lambda x: -x[1])],
        "renal_distribution": [{"label": k, "n": v} for k, v in sorted(renal_dist.items(), key=lambda x: -x[1])]
                              + [{"label": "No renal disease", "n": N - _renal_n}],
        "allele_class_summary": [{"label": k, "n": v} for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "ethnicity_distribution": [{"ethnicity": k, "n": v} for k, v in sorted(eth_dist.items(), key=lambda x: -x[1])],
        "presentation_distribution": [{"label": k, "n": v} for k, v in sorted(pres_dist.items(), key=lambda x: -x[1])],
        "misdiagnosis_distribution": [{"label": k, "n": v} for k, v in sorted(mis_dist.items(), key=lambda x: -x[1])],
        "veptr_distribution": [{"label": k, "n": v} for k, v in sorted(veptr_types.items(), key=lambda x: -x[1])],
        "top_variants": [
            {"variant": "p.Arg84Cys (c.250C>T) — LRR-3 loop; NEK1-binding surface; European compound het; moderate SRTD12", "n": 6},
            {"variant": "p.Gly165Arg (c.493G>A) — LRR-6 central IFT-A docking site; MENA homozygous; moderate-severe", "n": 8},
            {"variant": "p.Leu241Ter (c.723T>G) — LRR-7/8 truncating null; SRPS spectrum; pan-ethnic", "n": 5},
            {"variant": "p.Ala290Val (c.869C>T) — LRR-9 C-terminal hypomorphic; South Asian; retinal dominant; mild", "n": 7},
            {"variant": "p.Arg55His (c.164G>A) — LRR-1 N-cap; NEK1-proximal; MENA homozygous; moderate", "n": 4},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":          "C21orf2",
            "full_name":     "Chromosome 21 Open Reading Frame 2 (also: LRRC76)",
            "omim_gene":     "*603503",
            "chromosome":    "21q22.3",
            "size":          "~322 amino acids",
            "protein_class": "LRR (Leucine-Rich Repeat) scaffold — NEK1-interacting ciliary bridge protein",
            "key_domains":   "N-cap (aa 1–36) · LRR 1–3 NEK1-binding (aa 37–120) · LRR 4–6 IFT-A docking (aa 121–210) · LRR 7–8 structural (aa 211–270) · LRR-9/C-cap photoreceptor cilium (aa 271–322)",
            "key_interactions": "NEK1 (SRTD6 basal body kinase) via LRR-1/3 · IFT-A complex (WDR19, IFT140) via LRR-4/6",
            "molecular_class": "NEK1-IFT-A Bridge — FIFTH DISTINCT SRTD CLASS; bridges basal body kinase and IFT-A retrograde adaptor; unique among all SRTD genes",
            "chr21_note":    "Chr 21q22.3 — Down syndrome region; trisomy 21 patients have 3 copies (not causative); biallelic LOF (2 copies lost) required for SRTD12",
        },
        "disease_card": {
            "disease":       "Short-Rib Thoracic Dysplasia 12 (SRTD12 / ATD12)",
            "omim_disease":  "#616012",
            "also_known_as": "Jeune ATD12 · SRPS-like (biallelic null alleles)",
            "inheritance":   "Autosomal Recessive — biallelic LOF",
            "prevalence":    "~1:800,000–1,500,000 · <20 families worldwide (2026) · ultra-rare",
            "ciliary_em":    "VARIABLE — short stubby (IFT-A-like) OR partially absent; unique variable pattern among SRTD types",
            "polydactyly":   "35–45% — postaxial dominant; moderate rate",
            "unique_features": "Dual molecular class (NEK1 + IFT-A bridge) · variable EM · dual phenotype (LRR-9 → retinal dominant, LCA/CORD-like)",
            "renal":         "15–20% — TIN; cysts; ESRD rare",
            "retinal":       "20–30% — rod-cone / LCA-like; hypomorphic LRR-9 → retinal dominant without thoracic disease",
            "chf":           "5–8%",
            "perinatal_lethality": "25–40% (biallelic null — SRPS spectrum)",
            "surgical_tx":   "VEPTR / MAGEC growing rods (same protocol as all SRTDs)",
            "dual_phenotype": "Full SRTD12 (biallelic LOF) vs Retinal-only (LRR-9 hypomorphic, p.Ala290Val class)",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: narrow thorax + polydactyly → suspect SRTD; variable EM post-natally suggests C21orf2",
            "2. Post-natal: chest radiograph + skeletal survey → short ribs, thoracic diameter <2 SD",
            "3. URGENT: comprehensive ciliopathy gene panel including C21orf2 (NOT IFT-only or dynein-only panels)",
            "4. If panel negative: WES/WGS — C21orf2 is small but intronic/non-coding variants may be missed",
            "5. Ciliary EM (nasal brush): variable result — short stubby (IFT-A-like) OR partially absent; "
               "EM alone cannot classify SRTD12 — variable EM is diagnostic clue pointing to C21orf2 or SRTD6",
            "6. Retinal: ERG from age 3–4; OCT; ophthalmology annually (20–30% retinal; hypomorphic alleles → LCA/CORD dominant)",
            "7. Renal: USS + GFR annually (15–20% renal); consider biopsy if proteinuria early",
            "8. Cardiac echo: CHD 5–8%; exclude structural heart disease",
            "9. Liver: APRI + USS (CHF 5–8%); hepatology if abnormal LFTs",
            "10. Genetics: 25% AR recurrence; cascade testing; prenatal/PGT available",
            "11. Retinal-only presentation: if LCA/CORD panel negative, add C21orf2 (LRR-9 hypomorphic alleles may not be on standard LCA panels)",
            "12. Chromosome 21: trisomy 21 does NOT cause SRTD12; biallelic LOF on Chr 21q22.3 required",
            "13. Therapeutic: no approved NEK1/IFT-A targeted therapy (2026); enrol in ciliopathy registry",
        ],
        "mechanism_glossary": [
            {"term": "C21orf2 (Chromosome 21 Open Reading Frame 2 / LRRC76)", "definition":
             "322-aa LRR scaffold protein at the basal body / transition zone. Unique in that it physically bridges "
             "NEK1 (basal body kinase, SRTD6) via LRR-1/3 and the IFT-A complex (WDR19, IFT140) via LRR-4/6. "
             "Loss creates a dual ciliary defect: partial NEK1 substrate disruption + IFT-A docking failure."},
            {"term": "NEK1–C21orf2 interaction (SRTD6–SRTD12 link)", "definition":
             "C21orf2 LRR-1/2/3 concave surface directly contacts NEK1 kinase domain. When C21orf2 is lost, "
             "NEK1 substrate access is reduced → TTBK2 phosphorylation partially impaired → CP110 not fully "
             "removed → axoneme nucleation delayed or absent. This explains the partially absent EM phenotype "
             "in some SRTD12 patients, distinct from pure IFT-A (short stubby only)."},
            {"term": "IFT-A docking (C21orf2 LRR-4/6)", "definition":
             "C21orf2 LRR-4/5/6 anchors the IFT-A particle at the basal body transition zone, facilitating "
             "IFT-B import. When C21orf2 LRR-4/6 is disrupted, IFT-A cannot dock → IFT-B import fails → "
             "IFT-B accumulates at cilia base → short stubby cilia (same mechanism as WDR19/SRTD5, IFT140/SRTD9)."},
            {"term": "Variable EM phenotype (SRTD12 diagnostic signature)", "definition":
             "Unlike all other SRTD types (each with one EM class), SRTD12 shows VARIABLE EM: "
             "short stubby (IFT-A mechanism dominant) in ~50% of cases, partially absent in ~34%, "
             "fully absent in ~16%. This variability reflects which allele class dominates: "
             "LRR-4/6 missense → IFT-A short stubby; LRR-1/3 missense → NEK1 partially absent; "
             "truncating null → absent. Gene panel is the ONLY definitive test — EM alone is insufficient."},
            {"term": "Dual phenotype (SRTD12 vs retinal-only)", "definition":
             "C21orf2 LRR-9/C-cap (aa 271–322) is specifically required for the photoreceptor connecting "
             "cilium. Hypomorphic LRR-9 missense alleles (e.g., p.Ala290Val) impair photoreceptor cilia "
             "selectively but preserve broader IFT-A docking → retinal dystrophy (LCA/CORD-like) without "
             "significant thoracic disease. Same dual phenotype paradigm as IFT172/SRTD10 (TPR alleles → BBS-like)."},
            {"term": "Chromosome 21q22.3 and trisomy 21 (Down syndrome)", "definition":
             "C21orf2 maps to Chr 21q22.3 — within the Down syndrome (trisomy 21) critical region. "
             "Trisomy 21 patients carry 3 copies of C21orf2 (gain, not loss) — this does NOT cause SRTD12. "
             "SRTD12 requires biallelic LOF of both copies on diploid Chr 21. No phenotypic overlap with "
             "trisomy 21 has been reported in clinical series."},
            {"term": "Hedgehog signalling failure (SRTD12)", "definition":
             "Defective ciliogenesis (short stubby or absent cilia) → no cilia platform for Hedgehog pathway "
             "components (Smo, Gli2/3 processing) → GLI3R accumulates → Ihh/Shh absent in chondrocytes → "
             "narrow thorax, short ribs, short limbs (same Hedgehog mechanism as all SRTD types)."},
        ],
        "key_variants": [
            {"variant": "p.Arg84Cys", "domain": "LRR-3 loop (aa 80–90)", "consequence":
             "NEK1-binding surface disrupted; reduced NEK1 substrate phosphorylation; moderate SRTD12",
             "ethnicity": "European compound het"},
            {"variant": "p.Gly165Arg", "domain": "LRR-6 central IFT-A docking (aa 160–170)", "consequence":
             "IFT-A docking disrupted; IFT-B import failure; short stubby cilia; moderate-severe",
             "ethnicity": "MENA homozygous"},
            {"variant": "p.Leu241Ter", "domain": "LRR-7/8 junction (aa 241 truncating)", "consequence":
             "Null allele; complete LOF; SRPS spectrum; perinatal lethal in homozygous",
             "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala290Val", "domain": "LRR-9/C-cap (aa 290 — photoreceptor connecting cilium)", "consequence":
             "Hypomorphic; selectively impairs photoreceptor cilium; retinal dominant (LCA/CORD); mild SRTD12",
             "ethnicity": "South Asian hypomorphic"},
            {"variant": "p.Arg55His", "domain": "LRR-1 N-cap (aa 55)", "consequence":
             "LRR-1 destabilisation; NEK1 interaction proximal; partially absent cilia; moderate",
             "ethnicity": "MENA homozygous"},
        ],
        "treatment_summary": [
            "1. Narrow thorax (primary): VEPTR or MAGEC growing rod — same serial thoracic expansion protocol as all SRTDs",
            "2. Neonatal respiratory: mechanical ventilation (biallelic null); CPAP/BiPAP (moderate); wean post-VEPTR",
            "3. Retinal: annual ERG from age 3 (20–30% retinal; hypomorphic alleles → LCA/CORD-like); low vision support",
            "4. Renal: annual GFR/USS/creatinine; ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous; no recurrence)",
            "5. Hepatic: APRI + USS annually (5–8% CHF); hepatology referral if elevated",
            "6. Gene panel: ensure C21orf2 included (absent from pre-2016 SRTD panels); use comprehensive ciliopathy panel",
            "7. Retinal-only: if LCA/CORD panel negative, add C21orf2 (LRR-9 hypomorphic alleles); different from standard LCA genes",
            "8. Genetics: 25% AR recurrence; cascade testing; prenatal/PGT available",
            "9. Therapeutic pipeline: no approved agent (2026); NEK1 kinase modulator research may benefit SRTD12 (shared NEK1 interaction); enrol in ciliopathy registry",
        ],
        "ddx_table": [
            {"disease": "SRTD6 (NEK1) — basal body kinase class", "key_difference":
             "NEK1 LOF → fully ABSENT/RUDIMENTARY cilia; SRTD12 → variable (short stubby or partially absent); "
             "NEK1 has hydrops fetalis ~20% and medianasal hypoplasia ~30% — ABSENT in SRTD12; gene panel mandatory"},
            {"disease": "SRTD5 (WDR19) — most common IFT-A SRTD", "key_difference":
             "WDR19 → SHORT STUBBY only; ectodermal features (sparse hair, hypodontia) present in WDR19 — "
             "ABSENT in SRTD12; SRTD12 has variable EM (not pure short stubby); gene panel differentiates"},
            {"disease": "SRTD3 (DYNC2H1) — most common SRTD", "key_difference":
             "DYNC2H1 → CLUB/BULGING TIP cilia (IFT-B stranded); SRTD12 → short stubby or partially absent; "
             "different EM class; SRTD3 is ~50% of molecular SRTD — gene panel required"},
            {"disease": "SRTD10 (IFT172) — dual phenotype paradigm", "key_difference":
             "IFT172 dual phenotype (SRTD10 skeletal vs BBS-like retinal) similar concept to SRTD12 "
             "(SRTD12 thorax vs LCA/CORD retinal); but IFT172 is IFT-B2, SHORTENED cilia; C21orf2 is LRR-bridge, variable EM"},
            {"disease": "LCA / cone-rod dystrophy panels", "key_difference":
             "C21orf2 LRR-9 hypomorphic alleles can present as retinal-only LCA/CORD; often not on standard "
             "LCA gene panels; if LCA panel negative and family history of skeletal ciliopathy → add C21orf2"},
            {"disease": "Trisomy 21 / Down syndrome", "key_difference":
             "Trisomy 21 carries 3 copies of C21orf2 (GAIN); SRTD12 requires LOSS of both copies (biallelic LOF); "
             "no phenotypic confusion possible; chromosome 21 region is coincidental, not pathomechanistic"},
        ],
    }
