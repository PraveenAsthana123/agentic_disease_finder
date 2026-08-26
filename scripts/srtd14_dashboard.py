"""
IFT43 Short-Rib Thoracic Dysplasia 14 (SRTD14) — Jeune Asphyxiating Thoracic Dystrophy Type 14
=================================================================================================
Primary Gene : IFT43 (*614068) — also C14orf179 — Intraflagellar Transport 43 Homolog;
               ~362 aa; SMALLEST IFT-A subunit; entirely alpha-helical (no WD40 / no TPR);
               14q24.3.
               IFT43 is the small peripheral stabilizer of the IFT-A complex:
                 Alpha-helix 1–2 (N-terminal stabilization domain, aa 1–80): docks onto
                   IFT122 WD10 blade upper surface; provides electrostatic clamp for the
                   C-terminal propeller face of IFT122 (WDR10 / SRTD2); loss → WD10–12
                   face of IFT122 becomes partially disordered → ectodermal cilia sensitive.
                 Alpha-helix 3–4 (central docking module, aa 81–200): contacts IFT122 WD11
                   and WD12 C-terminal surface; stabilises the entire 12-blade propeller
                   C-cap; required for full IFT-A peripheral cohesion.
                 Alpha-helix 5 (C-terminal ectodermal specificity domain, aa 201–362):
                   required selectively for ectodermal cilia (hair follicle, ameloblast /
                   dental, eccrine gland) where cilia are uniquely IFT43-sensitive;
                   hypomorphic variants here → CED3-like phenotype without thoracic disease.
               Loss of IFT43 → partial IFT-A peripheral destabilisation → IFT-A C-cap
               disordered → IFT-B import fails at transition zone → SHORT STUBBY cilia
               (IFT-A class EM — identical to SRTD2/4/5/7/9; gene panel required).
               Ectodermal-specific sensitivity: IFT43 is uniquely required for ectodermal
               cilium integrity (hair follicle, ameloblast, eccrine gland); the C-terminal
               helix-5 domain is ectodermal-cilia-specific. Full biallelic LOF → SRTD14
               (narrow thorax + skeletal disease). Hypomorphic C-terminal alleles → CED3-like
               ectodermal phenotype without significant thoracic disease.
               IFT43 IS THE SIXTH AND SMALLEST SUBUNIT OF THE IFT-A COMPLEX — completing the
               IFT-A subunit table (WDR19/SRTD5, IFT140/SRTD9, WDR35/SRTD7, TTC21B/SRTD4,
               IFT122/SRTD2, IFT43/SRTD14). Absent from most pre-2016 IFT-A SRTD panels.

Disease OMIM : #616546 — Short-Rib Thoracic Dysplasia 14 with or without Polydactyly
               (SRTD14 / ATD14 / Jeune Syndrome 14)
               Allele spectrum:
               — Severe: biallelic null → perinatal lethal (SRPS spectrum); absent or
                 minimal cilia
               — Moderate: compound het / MENA homozygous missense (α-helix 1–4) →
                 full SRTD14 (narrow thorax, polydactyly, short stubby cilia)
               — Mild: hypomorphic C-terminal α-helix-5 missense → ectodermal phenotype
                 without thoracic disease (CED3-like); sparse hair, hypodontia
               UNIQUE FEATURES vs other IFT-A SRTDs:
               (1) SMALLEST IFT-A SUBUNIT (362 aa): the only entirely alpha-helical IFT-A
                   subunit — no WD40 repeats, no TPR; stabilises IFT122-WD10-12 C-cap.
               (2) ECTODERMAL CILIA SELECTIVITY: the C-terminal α-helix-5 region is uniquely
                   required for ectodermal cilia; hypomorphic variants here → sparse hair /
                   hypodontia without thoracic disease (same paradigm as SRTD2/CED2 and
                   SRTD5/CED1 — but via the IFT43 stabilizer rather than the IFT122 hub).
               (3) IFT-A COMPLETION: IFT43 is the last unaccounted IFT-A subunit to gain a
                   dedicated SRTD designation; completing the molecular IFT-A SRTD table.
               (4) ULTRA-RARE: <10–15 families worldwide (2026); absent from pre-2016
                   SRTD panels; often misdiagnosed as unclassified IFT-A SRTD.
               (5) LOWER POLYDACTYLY RATE: ~25–35% (vs 40–50% for SRTD2/5); reflects
                   milder Hedgehog disruption in partial IFT-A LOF vs core subunit LOF.
Chromosome   : 14q24.3
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1:1,000,000+; <10–15 families worldwide (2026); ultra-rare.

Protein Structure — IFT43 (~362 aa; alpha-helical; IFT-A peripheral stabilizer)
----------------------------------------------------------------------------------
Domain 1: Alpha-helix 1–2 (aa 1–80) — IFT122-WD10 C-cap upper surface docking;
           electrostatic clamp stabilising WD10 blade; primary IFT-A C-cap anchor.
           Missense here → WD10 face destabilised → IFT-A peripheral assembly weakened
           → moderate SRTD14 (thorax + polydactyly).
Domain 2: Alpha-helix 3–4 central docking module (aa 81–200) — contacts IFT122 WD11
           and WD12; stabilises entire C-terminal propeller cap; required for IFT-A
           peripheral cohesion. Missense → more severe IFT-A peripheral loss → moderate-
           severe SRTD14.
Domain 3: Alpha-helix 5 C-terminal (aa 201–362) — ectodermal-cilia-specific domain;
           selectively required for hair follicle, ameloblast, eccrine gland cilium
           integrity. Hypomorphic missense here → ectodermal phenotype without thoracic
           disease (CED3-like); C-terminal tail residues maintain ectodermal protein
           interactions distinct from skeletal cilia.

Key pathogenic variant classes (IFT43 / SRTD14):
1. Alpha-helix 1 missense (IFT122-WD10 dock, aa 30–80): European compound het; moderate SRTD14
2. Alpha-helix 3–4 missense (central dock aa 90–200): MENA homozygous; moderate-severe SRTD14
3. Biallelic truncating (null/null): SRPS spectrum; perinatal lethal
4. C-terminal helix-5 hypomorphic (ectodermal specificity, aa 220–362): South Asian; CED3-like
5. Alpha-helix 3 compound het missense: IFT122 WD11–WD12 face; moderate SRTD14
"""

import random
import math

SEED = 411
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
    ('Middle Eastern / North African', 0.40),   # consanguineous — homozygous α3–4 missense
    ('European',                        0.25),   # compound het — α1-2 missense + truncating
    ('South Asian',                     0.20),   # hypomorphic C-terminal helix-5 (CED3-like)
    ('East Asian',                      0.08),
    ('African',                         0.07),
]

allele_classes = [
    ('Compound het — missense + truncating',            0.32),
    ('Homozygous missense (consanguineous)',            0.35),
    ('Biallelic truncating (null/null)',                0.12),
    ('Compound het — two missense',                    0.14),
    ('Hypomorphic C-terminal compound het',            0.07),
]

thorax_cats = [
    ('Severe (Z-score < −4)',     0.20),
    ('Moderate (Z-score −2 to −4)', 0.45),
    ('Mild (Z-score −1 to −2)',   0.23),
    ('Absent / CED3-only',        0.12),   # hypomorphic alleles → no thoracic disease
]

misdiagnosis_labels = [
    'Unclassified IFT-A SRTD (EM only — short stubby)',
    'SRTD2 (IFT122) — same IFT-A class, short stubby EM',
    'SRTD5 (WDR19) — ectodermal overlap (sparse hair)',
    'Ectodermal dysplasia syndrome (CED3-like alleles)',
    'SRTD9 (IFT140) — IFT-A class, no CED',
    None,   # not misdiagnosed
]

ced3_features = [
    'Sparse/fine hair (ectodermal cilia: hair follicle)',
    'Hypodontia / oligodontia (ameloblast cilia)',
    'Nail dysplasia / nail plate thinning',
    'Narrow forehead / frontal bossing (mild)',
    'Brachydactyly (distal phalanx shortening)',
]

_poly_n          = round(N * 0.28)   # 28% polydactyly — lower than most IFT-A SRTDs
_renal_n         = round(N * 0.15)   # 15% — TIN / ESRD
_retinal_n       = round(N * 0.08)   # 8%
_hepatic_chf_n   = round(N * 0.05)   # 5% — CHF
_veptr_n         = round(N * 0.68)   # 68% had VEPTR or MAGEC
_perinatal_n     = round(N * 0.20)   # 20% biallelic null → perinatal lethal
_ced3_n          = round(N * 0.22)   # 22% have ectodermal (CED3-like) features
_ced3_only_n     = round(N * 0.10)   # 10% ectodermal-only (no thoracic disease)
_misdiagnosis_n  = round(N * 0.55)   # 55% misdiagnosed initially

for i in range(N):
    r = rng.random()
    eth = ethnicities[0][0]
    cumulative = 0
    for name, prob in ethnicities:
        cumulative += prob
        if r < cumulative:
            eth = name
            break

    r2 = rng.random()
    allele = allele_classes[0][0]
    cumulative = 0
    for name, prob in allele_classes:
        cumulative += prob
        if r2 < cumulative:
            allele = name
            break

    r3 = rng.random()
    thorax = thorax_cats[0][0]
    cumulative = 0
    for name, prob in thorax_cats:
        cumulative += prob
        if r3 < cumulative:
            thorax = name
            break

    poly      = i < _poly_n
    poly_type = (rng.choice(['Postaxial', 'Postaxial — bilateral', 'Preaxial (rare)'])
                 if poly else None)
    renal_any = i < _renal_n
    renal_type = (rng.choice(['Tubulointerstitial nephritis (TIN)', 'Cortical cysts + TIN',
                               'ESRD (transplant candidate)'])
                  if renal_any else None)
    retinal   = i < _retinal_n
    hepatic   = i < _hepatic_chf_n
    veptr     = i < _veptr_n
    perinatal = i < _perinatal_n

    # ectodermal (CED3-like) features
    ced3      = i < _ced3_n
    ced3_only = i < _ced3_only_n
    ced3_feats = (rng.sample(ced3_features, rng.randint(2, 4)) if ced3 else [])

    mis = rng.choice(misdiagnosis_labels) if i < _misdiagnosis_n else None

    dx_age = (rng.uniform(0, 0.5) if perinatal
              else rng.uniform(0.1, 3) if 'Severe' in thorax or 'Moderate' in thorax
              else rng.uniform(0.5, 8))
    if dx_age < 1:
        age_dx_cat = '0–1 yr (neonatal/infant)'
    elif dx_age < 4:
        age_dx_cat = '2–5 yr (early childhood)'
    elif dx_age < 12:
        age_dx_cat = '6–11 yr (school age)'
    else:
        age_dx_cat = '12+ yr (adolescent/adult)'

    em_pattern = rng.choice([
        'Short stubby — IFT-A class (typical SRTD14)',
        'Short stubby — IFT-A class (typical SRTD14)',
        'Short stubby with partial C-cap disorganisation',
        'Near-normal cilia (hypomorphic C-terminal alleles)',
    ])

    patients.append({
        'id':          f'P{i+1:02d}',
        'eth':         eth,
        'allele':      allele,
        'thorax':      thorax,
        'poly':        poly,
        'poly_type':   poly_type,
        'renal_any':   renal_any,
        'renal_type':  renal_type,
        'retinal':     retinal,
        'hepatic':     hepatic,
        'veptr':       veptr,
        'perinatal':   perinatal,
        'ced3':        ced3,
        'ced3_only':   ced3_only,
        'ced3_features': ced3_feats,
        'misdiagnosis': mis,
        'age_dx_cat':  age_dx_cat,
        'em_pattern':  em_pattern,
    })


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    transplant_done = sum(1 for p in patients if p['renal_any'] and p['renal_type'] and 'ESRD' in p['renal_type'])
    misdiagnosis_n  = sum(1 for p in patients if p['misdiagnosis'])

    return {
        "cohort_n": N,
        "seed": SEED,
        "gene": "IFT43 (C14orf179)",
        "disease": "SRTD14 — Short-Rib Thoracic Dysplasia 14",
        "mechanism": (
            "IFT43 is the SIXTH and SMALLEST subunit (~362 aa) of the IFT-A retrograde adaptor complex — "
            "an entirely alpha-helical stabilizer of IFT122 (SRTD2) at its WD10–12 C-terminal propeller "
            "face. The N-terminal alpha-helices 1–2 (aa 1–80) dock onto IFT122-WD10 upper surface; "
            "helices 3–4 (aa 81–200) contact IFT122-WD11/WD12 and stabilise the IFT-A C-cap; the "
            "C-terminal helix-5 (aa 201–362) is uniquely required for ectodermal cilia (hair follicle, "
            "ameloblast, eccrine gland). Loss of IFT43 → IFT-A peripheral C-cap destabilised → IFT-B "
            "import fails at transition zone → SHORT STUBBY cilia (same IFT-A EM class as SRTD2/4/5/7/9). "
            "Hypomorphic C-terminal alleles → ectodermal phenotype without thoracic disease (CED3-like)."
        ),
        "key_distinction": (
            "SRTD14 (IFT43) is the COMPLETION of the IFT-A SRTD subunit table: "
            "WDR19/SRTD5 (outer arm), IFT140/SRTD9 (scaffold), WDR35/SRTD7 (inner arm), "
            "TTC21B/SRTD4 (adaptor bridge), IFT122/SRTD2 (hub), IFT43/SRTD14 (peripheral C-cap stabilizer). "
            "UNIQUE FEATURES: (1) SMALLEST IFT-A subunit — entirely alpha-helical, no WD40/TPR; "
            "(2) ECTODERMAL CILIA SELECTIVITY — C-terminal helix-5 domain selectively required for "
            "ectodermal cilia (sparse hair, hypodontia without thoracic disease in hypomorphic alleles); "
            "(3) LOWEST POLYDACTYLY RATE of all IFT-A SRTDs (~28%); "
            "(4) ULTRA-RARE — <10–15 families worldwide (2026); absent from pre-2016 SRTD panels; "
            "(5) 55% misdiagnosis rate — short stubby EM indistinguishable from SRTD2/4/5/7/9 without gene panel."
        ),
        "kpis": {
            "thorax_severe_n":     sum(1 for p in patients if 'Severe' in p['thorax']),
            "thorax_severe_pct":   _pct(sum(1 for p in patients if 'Severe' in p['thorax'])),
            "polydactyly_n":       _poly_n,
            "polydactyly_pct":     _pct(_poly_n),
            "renal_any_n":         _renal_n,
            "renal_any_pct":       _pct(_renal_n),
            "ced3_n":              _ced3_n,
            "ced3_pct":            _pct(_ced3_n),
            "ced3_only_n":         _ced3_only_n,
            "ced3_only_pct":       _pct(_ced3_only_n),
            "retinal_any_n":       _retinal_n,
            "retinal_any_pct":     _pct(_retinal_n),
            "hepatic_chf_n":       _hepatic_chf_n,
            "hepatic_chf_pct":     _pct(_hepatic_chf_n),
            "veptr_any_n":         _veptr_n,
            "veptr_any_pct":       _pct(_veptr_n),
            "perinatal_death_n":   _perinatal_n,
            "perinatal_death_pct": _pct(_perinatal_n),
            "misdiagnosis_n":      misdiagnosis_n,
            "misdiagnosis_pct":    _pct(misdiagnosis_n),
            "transplant_done_n":   transplant_done,
        },
        "em_distribution": [
            {"label": "Short stubby — IFT-A class (typical SRTD14)",                    "n": 28},
            {"label": "Short stubby with partial C-cap disorganisation",                  "n": 7},
            {"label": "Near-normal cilia (hypomorphic C-terminal helix-5 alleles)",       "n": 5},
        ],
        "ift_a_subunit_table": [
            {"subunit": "WDR19 (IFT144)",   "gene_srtd": "WDR19 / SRTD5",         "role": "Outer arm; N-WD40 contacts IFT-A core; also NPHP13 alleles", "omim": "*608151"},
            {"subunit": "IFT140",            "gene_srtd": "IFT140 / SRTD9",         "role": "Scaffold subunit; N-WD40 contacts WDR35; C-TPR contacts WDR19", "omim": "*614620"},
            {"subunit": "WDR35 (IFT121)",    "gene_srtd": "WDR35 / SRTD7",         "role": "Inner arm; WD40 propeller; contacts IFT140 N-face", "omim": "*613602"},
            {"subunit": "TTC21B (IFT139)",   "gene_srtd": "TTC21B / SRTD4",        "role": "Adaptor bridge; TRP repeat; bridges WDR19-C-tail → IFT122-WD5-7; also NPHP12 alleles", "omim": "*612014"},
            {"subunit": "IFT122 (WDR10)",    "gene_srtd": "IFT122 / SRTD2",        "role": "ARM HUB; 12-blade WD40 propeller; receives WDR19 (WD1-4) + TTC21B (WD5-7); IFT43 stabilizes WD10-12", "omim": "*606045"},
            {"subunit": "IFT43 (C14orf179)", "gene_srtd": "IFT43 / SRTD14 (THIS GENE)", "role": "Peripheral C-cap stabilizer; alpha-helical; docks IFT122-WD10-12; ectodermal-cilia-specific C-terminal domain", "omim": "*614068"},
        ],
        "srtd_molecular_class_table": [
            {"class": "IFT-A retrograde adaptor",     "em": "SHORT STUBBY cilia",
             "genes": "SRTD2 (IFT122) · SRTD4 (TTC21B) · SRTD5 (WDR19) · SRTD7 (WDR35) · SRTD9 (IFT140) · SRTD14 (IFT43 — THIS)",
             "why": "IFT-A C-cap or core destabilised → IFT-B import failure → cilia cannot extend beyond short stub"},
            {"class": "Dynein-2 retrograde motor",    "em": "CLUB / BULGING TIP",
             "genes": "SRTD3 (DYNC2H1) · SRTD8 (WDR60) · SRTD11 (WDR34) · SRTD15 (DYNC2LI1) · SRTD17 (TCTEX1D2)",
             "why": "Retrograde IFT fails; IFT-B cargo accumulates at ciliary tip → bulging tip EM"},
            {"class": "IFT-B2 anterograde subcomplex","em": "SHORTENED cilia",
             "genes": "SRTD1 (IFT80) · SRTD10 (IFT172) · SRTD13 (CLUAP1)",
             "why": "IFT-B2 assembly fails; kinesin-2 tip anchor lost; cilia truncated but not fully absent"},
            {"class": "Basal Body Kinase",             "em": "ABSENT / RUDIMENTARY",
             "genes": "SRTD6 (NEK1)",
             "why": "CP110 not removed; axoneme nucleation fails at step 0; ciliogenesis cannot initiate"},
            {"class": "NEK1-IFT-A Bridge",            "em": "VARIABLE — SHORT STUBBY / PARTIALLY ABSENT",
             "genes": "SRTD12 (C21orf2)",
             "why": "Dual defect bridging NEK1 and IFT-A; EM varies with allele dominance"},
        ],
        "age_distribution": {
            "dx_0_1yr":   sum(1 for p in patients if '0–1' in p['age_dx_cat']),
            "dx_2_5yr":   sum(1 for p in patients if '2–5' in p['age_dx_cat']),
            "dx_6_10yr":  sum(1 for p in patients if '6–11' in p['age_dx_cat']),
            "dx_11_plus": sum(1 for p in patients if '12+' in p['age_dx_cat']),
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

    # CED3 feature distribution
    ced3_feat_all = []
    for p in patients:
        ced3_feat_all.extend(p['ced3_features'])
    ced3_feat_dist = Counter(ced3_feat_all)

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
            {"label": "No polydactyly",      "n": N - _poly_n},
        ] + [{"label": k, "n": v} for k, v in sorted(poly_dist.items(), key=lambda x: -x[1])],
        "em_distribution": [{"label": k, "n": v} for k, v in sorted(em_dist.items(), key=lambda x: -x[1])],
        "ced3_distribution": [
            {"label": "CED3-like ectodermal phenotype",          "n": _ced3_n},
            {"label": "CED3-only (no/minimal thoracic disease)", "n": _ced3_only_n},
        ] + [{"label": k, "n": v} for k, v in sorted(ced3_feat_dist.items(), key=lambda x: -x[1])],
        "renal_distribution": [{"label": k, "n": v} for k, v in sorted(renal_dist.items(), key=lambda x: -x[1])]
                              + [{"label": "No renal disease", "n": N - _renal_n}],
        "allele_class_summary": [{"label": k, "n": v} for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "ethnicity_distribution": [{"ethnicity": k, "n": v} for k, v in sorted(eth_dist.items(), key=lambda x: -x[1])],
        "presentation_distribution": [{"label": k, "n": v} for k, v in sorted(pres_dist.items(), key=lambda x: -x[1])],
        "misdiagnosis_distribution": [{"label": k, "n": v} for k, v in sorted(mis_dist.items(), key=lambda x: -x[1])],
        "veptr_distribution": [{"label": k, "n": v} for k, v in sorted(veptr_types.items(), key=lambda x: -x[1])],
        "top_variants": [
            {"variant": "p.Arg45His (c.134G>A) — α-helix 1; IFT122-WD10 upper-surface dock; European compound het; moderate SRTD14",       "n": 7},
            {"variant": "p.Leu112Phe (c.334C>T) — α-helix 3 central dock; IFT122-WD11/WD12 contact; MENA homozygous; moderate-severe SRTD14", "n": 8},
            {"variant": "p.Gln198Ter (c.592C>T) — truncating null; SRPS spectrum; pan-ethnic",                                                  "n": 5},
            {"variant": "p.Ala289Val (c.866C>T) — α-helix 5 C-terminal; ectodermal-cilia-specific; South Asian hypomorphic; CED3-like mild", "n": 6},
            {"variant": "p.Lys78Glu (c.232A>G) — α-helix 3; IFT122-WD11 C-face docking; MENA homozygous; moderate SRTD14",                   "n": 4},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":          "IFT43",
            "aliases":       "C14orf179",
            "full_name":     "Intraflagellar Transport 43 Homolog (Chromosome 14 Open Reading Frame 179)",
            "omim_gene":     "*614068",
            "chromosome":    "14q24.3",
            "size":          "~362 amino acids",
            "protein_class": "Entirely alpha-helical (no WD40, no TPR); IFT-A peripheral C-cap stabilizer — SMALLEST IFT-A subunit",
            "key_domains":   "α-helix 1–2 (IFT122-WD10 dock, aa 1–80) · α-helix 3–4 central docking module (IFT122-WD11/WD12, aa 81–200) · α-helix 5 C-terminal ectodermal-cilia-specific domain (aa 201–362)",
            "key_interactions": "IFT122/WDR10 (SRTD2) via WD10–12 C-cap face (primary) · IFT144/WDR19 (SRTD5) peripheral (indirect) · IFT139/TTC21B (SRTD4) peripheral (indirect)",
            "molecular_class": "IFT-A peripheral stabilizer — SHORT STUBBY cilia; EM identical to SRTD2/4/5/7/9; gene panel required for definitive subtype",
            "ced3_note": "Alpha-helix-5 C-terminal hypomorphic alleles → CED3-like ectodermal phenotype (sparse hair, hypodontia, brachydactyly); ectodermal cilia sensitivity without full IFT-A destabilisation; same CED1/CED2 paradigm via IFT43 peripheral stabilizer",
        },
        "disease_card": {
            "disease":       "Short-Rib Thoracic Dysplasia 14 (SRTD14 / ATD14)",
            "omim_disease":  "#616546",
            "also_known_as": "Jeune ATD14 · CED3-like (OMIM ref pending, hypomorphic C-terminal alleles) · SRPS-like (biallelic null)",
            "inheritance":   "Autosomal Recessive — biallelic LOF",
            "prevalence":    "~1:1,000,000+; <10–15 families worldwide (2026); ultra-rare",
            "ciliary_em":    "SHORT STUBBY (IFT-A class) — identical to SRTD2/4/5/7/9; cannot distinguish by EM; gene panel required",
            "polydactyly":   "~25–35% — postaxial dominant; LOWER RATE than most IFT-A SRTDs (partial IFT-A peripheral loss)",
            "unique_features": "Smallest IFT-A subunit (362 aa; entirely alpha-helical) · IFT-A C-cap completion (6th and final IFT-A SRTD gene) · Ectodermal cilia specificity (CED3-like hypomorphic alleles) · Ultra-rare (<15 families 2026)",
            "renal":         "~15% — TIN; cortical cysts; ESRD (lower than SRTD2/5 due to partial IFT-A peripheral LOF)",
            "retinal":       "~8% — mild rod-cone; no photoreceptor-specific allele",
            "chf":           "~5%",
            "perinatal_lethality": "~20% (biallelic null — SRPS spectrum)",
            "surgical_tx":   "VEPTR / MAGEC growing rods (standard SRTD protocol)",
            "ced3_phenotype": "Sparse/fine hair · hypodontia/oligodontia · nail dysplasia · narrow forehead · brachydactyly (hypomorphic α-helix-5; mild/absent thorax)",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: narrow thorax + polydactyly → suspect SRTD; short stubby cilia post-natally → IFT-A class (EM CANNOT distinguish SRTD14 from SRTD2/4/5/7/9 — gene panel mandatory)",
            "2. Post-natal: chest radiograph + skeletal survey → short ribs, thoracic diameter <2 SD",
            "3. URGENT: comprehensive ciliopathy gene panel including IFT43/C14orf179 — NOT on pre-2016 SRTD panels; full IFT-A subunit coverage mandatory (IFT43 often missing from older panels)",
            "4. CED3-like presentation: if ectodermal dysplasia (sparse hair + hypodontia + brachydactyly) without clear thoracic disease → include IFT43 on panel (hypomorphic α-helix-5 alleles); CED2(IFT122) and CED1(WDR19) should also be on panel",
            "5. Ciliary EM (nasal brush): short stubby (same as SRTD2/4/5/7/9); IFT-B accumulates at cilia base; EM confirms IFT-A class but gene panel required for subtype; hypomorphic C-terminal alleles may show near-normal cilia",
            "6. Retinal: ERG from age 3; OCT; annual ophthalmology (8% mild rod-cone; no retinal-only allele in IFT43)",
            "7. Renal: annual USS + GFR (15%); ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous; no recurrence)",
            "8. Cardiac echo: CHD ~5%; exclude structural heart disease",
            "9. Ectodermal: dental panoramic X-ray; dermatology for hair/nail dysplasia (CED3-like cohort)",
            "10. Genetics: 25% AR recurrence; cascade testing; prenatal / PGT available",
            "11. IFT-A panel completeness: confirm IFT43/C14orf179 alongside WDR19, IFT140, WDR35, TTC21B, IFT122 — all 6 IFT-A core SRTD genes must be on panel for complete IFT-A SRTD coverage",
            "12. Therapeutic: no approved agent (2026); enrol in ciliopathy / rare-disease registry; IFT43 stabilisation is a research target",
        ],
        "mechanism_glossary": [
            {"term": "IFT43 (C14orf179) — IFT-A peripheral C-cap stabilizer", "definition":
             "~362-aa entirely alpha-helical protein; the SIXTH and SMALLEST subunit of the IFT-A "
             "retrograde adaptor complex. It docks onto IFT122 (SRTD2) at the WD10–12 C-terminal "
             "propeller face, providing a peripheral electrostatic C-cap. Loss of IFT43 → partial "
             "IFT-A peripheral assembly weakened → IFT-B import reduced → SHORT STUBBY cilia "
             "(same EM class as SRTD2/4/5/7/9; gene panel required for subtype diagnosis)."},
            {"term": "IFT-A C-cap (IFT122-WD10-12 + IFT43 stabilizer module)", "definition":
             "The C-terminal WD10–12 blades of IFT122 (SRTD2) + IFT43 form the IFT-A peripheral "
             "C-cap module. IFT43 helices 1–4 clamp the WD10–12 blade face; helix-5 is ectodermal-cilia-"
             "specific. This module stabilises the outer IFT-A scaffold and is required for "
             "ectodermal cilia (hair follicle, ameloblast, eccrine gland) in addition to the full "
             "IFT-A assembly needed for skeletal cilia."},
            {"term": "CED3-like ectodermal phenotype (IFT43 hypomorphic C-terminal alleles)", "definition":
             "IFT43 α-helix-5 (aa 201–362) is uniquely required for ectodermal cilia. Hypomorphic "
             "missense variants here selectively impair ectodermal cilia without destabilising the "
             "full IFT-A complex → sparse hair, hypodontia, nail dysplasia, brachydactyly; "
             "thoracic disease absent or very mild. Same paradigm as IFT122/CED2 and WDR19/CED1 — "
             "but via the IFT43 peripheral stabilizer rather than the IFT122 hub or WDR19 outer arm."},
            {"term": "IFT-A complete subunit table (SRTD14 completes the IFT-A SRTD series)", "definition":
             "All six IFT-A subunits with SRTD designations: WDR19 (SRTD5) + IFT140 (SRTD9) + "
             "WDR35 (SRTD7) + TTC21B (SRTD4) + IFT122 (SRTD2) + IFT43 (SRTD14). SRTD14 is the "
             "sixth and final IFT-A subunit SRTD — completing the molecular IFT-A SRTD table. "
             "IFT43 was absent from all pre-2016 SRTD panels."},
            {"term": "Short stubby cilia — IFT-A class (SRTD2/4/5/7/9/14)", "definition":
             "IFT-A destabilisation → IFT-B import fails at the transition zone → IFT-B cargo "
             "accumulates at cilia base → cilia form only a short stubby remnant. This EM pattern "
             "is identical across all six IFT-A SRTDs (SRTD2/4/5/7/9/14); distinguishable ONLY "
             "by gene panel, not by EM morphology alone."},
            {"term": "Hedgehog signalling failure (SRTD14)", "definition":
             "IFT43 LOF → partial IFT-A peripheral loss → short stubby cilia → insufficient "
             "ciliary platform for Hedgehog signal transduction (Smo, Gli processing) → "
             "GLI3R accumulates → Ihh/Shh absent in chondrocytes → narrow thorax, short ribs, "
             "limb shortening. Same mechanism as all IFT-A SRTDs; partial vs core LOF accounts "
             "for the slightly lower polydactyly rate in SRTD14."},
        ],
        "key_variants": [
            {"variant": "p.Arg45His", "domain": "α-helix 1 (aa 40–50; IFT122-WD10 upper-surface dock)", "consequence":
             "IFT122-WD10 electrostatic clamp disrupted; IFT-A C-cap partially disordered; moderate SRTD14",
             "ethnicity": "European compound het"},
            {"variant": "p.Leu112Phe", "domain": "α-helix 3 central dock (aa 108–118; IFT122-WD11/WD12 contact)", "consequence":
             "Central IFT43–IFT122 interface weakened; IFT-A C-cap cohesion reduced; MENA homozygous; moderate-severe SRTD14",
             "ethnicity": "Middle Eastern / MENA homozygous"},
            {"variant": "p.Gln198Ter", "domain": "α-helix 4/5 junction truncating (aa 198 null)", "consequence":
             "Null allele; complete IFT43 LOF; SRPS spectrum; perinatal lethal if homozygous",
             "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala289Val", "domain": "α-helix 5 C-terminal ectodermal domain (aa 285–295)", "consequence":
             "Hypomorphic; ectodermal-cilia-specific domain disrupted; CED3-like sparse hair / hypodontia; mild/absent thoracic disease",
             "ethnicity": "South Asian hypomorphic"},
            {"variant": "p.Lys78Glu", "domain": "α-helix 3 (aa 75–85; IFT122-WD11 C-face docking)", "consequence":
             "IFT122-WD11 docking surface disrupted; IFT-A peripheral C-cap weakened; moderate SRTD14",
             "ethnicity": "MENA homozygous"},
        ],
        "treatment_summary": [
            "1. Narrow thorax (primary): VEPTR or MAGEC growing rod — standard serial thoracic expansion protocol for all SRTDs",
            "2. Neonatal respiratory: mechanical ventilation (biallelic null SRPS); CPAP/BiPAP (moderate); wean post-VEPTR",
            "3. Renal: annual GFR/USS/creatinine (~15%); ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous; no recurrence)",
            "4. Retinal: annual ERG from age 3 (~8% mild rod-cone); low-vision support if needed",
            "5. Ectodermal (CED3-like cohort): dental panoramic X-ray; paediatric dentistry (hypodontia); dermatology (sparse hair, nail dysplasia)",
            "6. Hepatic: APRI + USS annually (~5% CHF); hepatology if elevated LFTs",
            "7. Gene panel: ensure IFT43/C14orf179 included — absent from pre-2016 SRTD panels; full 6-subunit IFT-A coverage (IFT43 + IFT122 + WDR19 + IFT140 + WDR35 + TTC21B) mandatory",
            "8. Genetics: 25% AR recurrence; cascade testing; prenatal / PGT available",
            "9. Therapeutic pipeline: no approved agent (2026); IFT43 C-cap stabilisation is a research target; enrol in ciliopathy registry",
        ],
        "ddx_table": [
            {"disease": "SRTD2 (IFT122) — IFT-A ARM hub", "key_difference":
             "IFT122 (SRTD2) is the primary IFT43 interaction partner — its WD10-12 face docks IFT43; "
             "both produce short stubby cilia (IFT-A class); SRTD2 has CED2 dual phenotype (WD10-12 hypomorphic); "
             "SRTD14/IFT43 has CED3-like ectodermal phenotype (α-helix-5 hypomorphic); gene panel differentiates"},
            {"disease": "SRTD5 (WDR19) — IFT-A outer arm; most common IFT-A SRTD", "key_difference":
             "WDR19 is the most commonly mutated IFT-A SRTD gene (~100–200 families); both IFT-A class (short stubby); "
             "WDR19/SRTD5 has CED1 and NPHP13 alleles — IFT43/SRTD14 has CED3-like (no NPHP allele); "
             "IFT43 is 50–100x rarer; gene panel mandatory"},
            {"disease": "SRTD4 (TTC21B) — IFT-A adaptor bridge", "key_difference":
             "TTC21B bridges WDR19 to IFT122; TTC21B (SRTD4) has JOUBERT MTS and NPHP12 alleles — "
             "IFT43/SRTD14 has neither; both IFT-A short stubby; gene panel differentiates"},
            {"disease": "SRTD9 (IFT140) — IFT-A scaffold", "key_difference":
             "IFT140 scaffold subunit contacts WDR35 and WDR19; IFT140/SRTD9 has no CED phenotype — "
             "IFT43/SRTD14 has CED3-like ectodermal features (helix-5 alleles); both IFT-A short stubby; "
             "gene panel mandatory"},
            {"disease": "SRTD7 (WDR35) — IFT-A inner arm", "key_difference":
             "WDR35 directly docks IFT140-N-WD40; WDR35/SRTD7 has no CED phenotype, no NPHP — "
             "IFT43/SRTD14 has CED3-like; similar polydactyly rate; both IFT-A short stubby; gene panel required"},
            {"disease": "CED1 (WDR19 hypomorphic) / CED2 (IFT122 hypomorphic)", "key_difference":
             "All three CEDs share overlapping ectodermal features (sparse hair, hypodontia, brachydactyly); "
             "CED1 = WDR19 outer-arm hypomorphic; CED2 = IFT122 WD10-12 hypomorphic; CED3-like = IFT43 helix-5 hypomorphic; "
             "gene panel is the ONLY definitive differentiator; CED1/CED2 are far more common than CED3-like/IFT43"},
            {"disease": "SRTD3 (DYNC2H1) — most common SRTD overall", "key_difference":
             "DYNC2H1 → club/bulging tip cilia (IFT-B stranded at tip) — DIFFERENT EM CLASS from IFT43 short stubby; "
             "SRTD3 ~50% of all molecular SRTD; no CED phenotype; gene panel confirms; no ectodermal features"},
        ],
    }
