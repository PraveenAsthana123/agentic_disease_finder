"""
KIAA0586 Short-Rib Thoracic Dysplasia 16 (SRTD16) — Jeune Asphyxiating Thoracic Dystrophy Type 16
=====================================================================================================
Primary Gene : KIAA0586 (*610178) — also TALPID3, CPLANE1 — Centrosomal/Centriolar Scaffold Protein;
               ~1,624 aa; large multi-domain coiled-coil centriolar scaffold; distal appendage protein.
               KIAA0586 localizes to the distal appendages of the mother centriole and is required for:
                 CC1 domain (N-terminal, aa 1–400): centriolar localization; distal appendage anchoring;
                   centriole-to-basal-body transition; required for initial ciliary membrane docking.
                 CC2 domain (central, aa 401–900): CPLANE complex assembly; contacts INTU and FUZ;
                   coordinates planar cell polarity (PCP) effectors with ciliogenesis machinery.
                 CPLANE interface (aa 700–1,100): INTU/FUZ/WDPCP interaction surface; required for
                   basal body maturation and IFT complex recruitment to the ciliary base.
                 CC3 domain (C-terminal, aa 1,100–1,624): IFT protein recruitment coordination;
                   functional coordination with NEK1 for CP110 removal; hypomorphic alleles here
                   → Joubert syndrome-23 (JBTS23: cerebellar vermis hypoplasia, molar tooth sign)
                   without full SRTD skeletal disease.
               Loss of KIAA0586 → basal body maturation fails → IFT-A and IFT-B proteins cannot be
               recruited to the basal body → ciliogenesis aborts at or immediately after step 0 →
               ABSENT cilia or rudimentary stubs (sixth distinct SRTD molecular class).
               KIAA0586 IS THE DEFINING MEMBER OF THE SIXTH SRTD MOLECULAR CLASS:
               Centriolar/CPLANE scaffold — distinct from IFT-A (SRTD2/4/5/7/9/14), IFT-B2
               (SRTD1/10/13), Dynein-2 (SRTD3/8/11/15/17), Basal Body Kinase (SRTD6/NEK1),
               and NEK1–IFT-A Bridge (SRTD12/C21orf2).

Disease OMIM : #617098 — Short-Rib Thoracic Dysplasia 16 with or without Polydactyly
               (SRTD16 / ATD16 / Jeune Syndrome 16)
               Allele spectrum:
               — Severe: biallelic null → perinatal lethal (SRPS-like spectrum); absent cilia
               — Moderate-severe: compound het / MENA homozygous missense (CC1–CC2) →
                 full SRTD16 (narrow thorax, polydactyly, absent/rudimentary cilia)
               — Mild (Joubert): hypomorphic C-terminal CC3 missense → JBTS23 (cerebellar
                 vermis hypoplasia, molar tooth sign, ataxia) without full thoracic disease
               UNIQUE FEATURES vs other SRTDs:
               (1) SIXTH DISTINCT MOLECULAR CLASS — Centriolar/CPLANE Scaffold: not IFT-A,
                   not IFT-B2, not Dynein-2, not Basal Body Kinase, not NEK1–IFT-A Bridge.
               (2) JOUBERT SYNDROME ALLELES (JBTS23): hypomorphic C-terminal CC3 missense →
                   JBTS23 cerebellar phenotype (MTS + ataxia); absent in all other SRTDs.
               (3) HIGH POLYDACTYLY — 55%: postaxial dominant + occasional preaxial;
                   second-highest rate after NEK1/SRTD6 (65–75%).
               (4) ABSENT/RUDIMENTARY CILIA EM — same as SRTD6/NEK1; distinct from IFT-A
                   short stubby, Dynein-2 club, IFT-B2 shortened.
               (5) ULTRA-RARE: <20 families worldwide (2026); absent from pre-2016 SRTD panels.
               (6) CPLANE COMPLEX: overlap with Bardet-Biedl (BBS, CPLANE dysfunction) but
                   BBS has obesity/no thorax; gene panel resolves.
Chromosome   : 14q23.1
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1:1,000,000+; <20 families worldwide (2026); ultra-rare.

Protein Structure — KIAA0586 (~1,624 aa; multi-domain coiled-coil centriolar scaffold)
----------------------------------------------------------------------------------------
Domain 1: CC1 N-terminal coiled-coil (aa 1–400) — distal appendage anchoring; centriole-to-
           basal-body transition; centriolar membrane docking initiation. Missense → basal body
           maturation delayed but partially functional → moderate SRTD16.
Domain 2: CC2 central coiled-coil + CPLANE interface (aa 401–1,100) — CPLANE complex assembly;
           INTU/FUZ/WDPCP interaction; basal body IFT recruitment coordination. Missense → CPLANE
           partially uncoupled → IFT recruitment reduced → absent cilia; moderate-severe SRTD16.
Domain 3: CC3 C-terminal coiled-coil (aa 1,100–1,624) — IFT protein recruitment; NEK1 functional
           coordination for CP110 cap removal; required for IFT platform assembly. Hypomorphic
           missense → partial loss; ciliary platform reduces but does not abort → JBTS23 (Joubert)
           without thoracic skeletal disease.

Key pathogenic variant classes (KIAA0586 / SRTD16):
1. CC1 missense (distal appendage, aa 100–400): MENA homozygous; moderate SRTD16
2. CC1/CC2 junction compound het: European; moderate-severe SRTD16
3. Biallelic truncating (null/null): SRPS-like; perinatal lethal
4. C-terminal CC3 hypomorphic (Joubert domain, aa 1,050–1,624): South Asian; JBTS23 without SRTD
5. CPLANE interface missense (aa 450–1,000): Middle Eastern homozygous; moderate SRTD16
"""

import random
import math

SEED = 413
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
ethnicities = [
    ('Middle Eastern / MENA (consanguineous)',  0.38),
    ('South Asian',                             0.22),
    ('European (compound het)',                 0.20),
    ('East Asian',                              0.08),
    ('African / Sub-Saharan',                   0.07),
    ('Latin American',                          0.05),
]

allele_classes = [
    ('Compound het missense + truncating (CC1-CC2)',  0.32),
    ('Homozygous missense — MENA (CC1-CC2)',          0.28),
    ('Biallelic truncating null (SRPS spectrum)',      0.18),
    ('Hypomorphic CC3 — Joubert-only alleles',        0.12),
    ('Compound het missense — European (CC1-CC2)',    0.10),
]

thorax_cats = [
    ('Severe — narrow chest, respiratory failure (neonatal)', 0.30),
    ('Moderate — thoracic constriction, recurrent infections', 0.40),
    ('Mild — borderline narrow thorax, ambulatory',           0.18),
    ('Absent (Joubert-only CC3 hypomorphic)',                 0.12),
]

misdiagnosis_labels = [
    'SRTD6 (NEK1) — same absent cilia EM; different gene',
    'Joubert Syndrome (JBTS) — Joubert alleles misclassified as pure JBTS',
    'Unclassified ciliopathy (absent cilia; rare gene not on panel)',
    'SRTD3 (DYNC2H1) — most-common SRTD; different EM class (club cilia)',
    'BBS (Bardet-Biedl) — CPLANE complex overlap; no obesity in SRTD16',
    None,   # not misdiagnosed
]

joubert_features = [
    'Cerebellar vermis hypoplasia (JBTS23 hypomorphic)',
    'Molar tooth sign (MTS) on brain MRI',
    'Ataxia / truncal instability',
    'Oculomotor apraxia',
    'Episodic hyperpnea (respiratory irregularity)',
]

_poly_n          = round(N * 0.55)   # 55% polydactyly — high; 2nd after NEK1-SRTD6
_renal_n         = round(N * 0.22)   # 22% — TIN / ESRD
_retinal_n       = round(N * 0.18)   # 18% — higher due to Joubert overlap
_hepatic_chf_n   = round(N * 0.07)   # 7% — CHF
_veptr_n         = round(N * 0.65)   # 65% had VEPTR or MAGEC
_perinatal_n     = round(N * 0.28)   # 28% biallelic null → perinatal lethal
_joubert_n       = round(N * 0.12)   # 12% have Joubert (JBTS23) features
_hydrops_n       = round(N * 0.10)   # 10% hydrops fetalis
_misdiagnosis_n  = round(N * 0.60)   # 60% misdiagnosed initially

patients = []

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
    poly_type = (rng.choice(['Postaxial', 'Postaxial — bilateral', 'Postaxial + Preaxial (rare)', 'Preaxial only (rare)'])
                 if poly else None)
    renal_any = i < _renal_n
    renal_type = (rng.choice(['Tubulointerstitial nephritis (TIN)', 'Cortical cysts + TIN',
                               'ESRD (transplant candidate)', 'Nephronophthisis-like (JBTS23 overlap)'])
                  if renal_any else None)
    retinal   = i < _retinal_n
    hepatic   = i < _hepatic_chf_n
    veptr     = i < _veptr_n
    perinatal = i < _perinatal_n
    joubert   = i < _joubert_n
    hydrops   = i < _hydrops_n

    joubert_feats = (rng.sample(joubert_features, rng.randint(2, 4)) if joubert else [])

    mis = rng.choice(misdiagnosis_labels) if i < _misdiagnosis_n else None

    dx_age = (rng.uniform(0, 0.5) if perinatal
              else rng.uniform(0.1, 3) if 'Severe' in thorax or 'Moderate' in thorax
              else rng.uniform(0.5, 10))
    if dx_age < 1:
        age_dx_cat = '0–1 yr (neonatal/infant)'
    elif dx_age < 4:
        age_dx_cat = '2–5 yr (early childhood)'
    elif dx_age < 12:
        age_dx_cat = '6–11 yr (school age)'
    else:
        age_dx_cat = '12+ yr (adolescent/adult — Joubert-only alleles)'

    em_pattern = rng.choice([
        'Absent cilia — KIAA0586 basal body class (typical SRTD16)',
        'Absent cilia — KIAA0586 basal body class (typical SRTD16)',
        'Rudimentary basal body stubs — no axoneme extension',
        'Short rudimentary axoneme — partial CC3 Joubert-allele function',
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
        'joubert':     joubert,
        'joubert_features': joubert_feats,
        'hydrops':     hydrops,
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
        "gene": "KIAA0586 (TALPID3 / CPLANE1)",
        "disease": "SRTD16 — Short-Rib Thoracic Dysplasia 16",
        "mechanism": (
            "KIAA0586 (TALPID3/CPLANE1) is a large centriolar scaffold protein (~1,624 aa; multiple "
            "coiled-coil domains) that localizes to the distal appendages of the mother centriole. "
            "It is required for: (1) centriole-to-basal-body transition and distal appendage maturation; "
            "(2) CPLANE complex assembly (INTU, FUZ, WDPCP interaction) — coordinates planar cell "
            "polarity effectors with ciliogenesis; (3) IFT-A and IFT-B protein recruitment to the "
            "ciliary base (basal body platform); (4) functional coordination with NEK1 (SRTD6) for "
            "CP110 removal from the mother centriole. Loss of KIAA0586 → basal body maturation fails "
            "→ IFT components cannot assemble at the ciliary base → ciliogenesis aborts at or "
            "immediately after step 0 → ABSENT cilia or rudimentary stubs. This defines the SIXTH "
            "distinct SRTD molecular class: Centriolar/CPLANE scaffold."
        ),
        "key_distinction": (
            "SRTD16 (KIAA0586) introduces three unique features absent from all other SRTDs: "
            "(1) SIXTH MOLECULAR CLASS — Centriolar/CPLANE scaffold: KIAA0586 is neither IFT-A, "
            "IFT-B2, Dynein-2, Basal Body Kinase (NEK1), nor NEK1–IFT-A Bridge (C21orf2) — it is "
            "the centriolar scaffold that recruits IFT machinery to the basal body. "
            "(2) JOUBERT SYNDROME ALLELES (JBTS23): hypomorphic C-terminal CC3 missense → JBTS23 "
            "cerebellar phenotype (MTS + ataxia) WITHOUT full thoracic SRTD — the ONLY SRTD with "
            "confirmed Joubert hypomorphic alleles in the same gene. "
            "(3) SECOND-HIGHEST POLYDACTYLY (~55%): postaxial dominant with occasional preaxial; "
            "surpassed only by NEK1/SRTD6 (65–75%). "
            "ABSENT/RUDIMENTARY CILIA EM — same as SRTD6/NEK1; gene panel differentiates. "
            "Ultra-rare: <20 families worldwide (2026); absent from pre-2016 SRTD panels."
        ),
        "kpis": {
            "thorax_severe_n":     sum(1 for p in patients if 'Severe' in p['thorax']),
            "thorax_severe_pct":   _pct(sum(1 for p in patients if 'Severe' in p['thorax'])),
            "polydactyly_n":       _poly_n,
            "polydactyly_pct":     _pct(_poly_n),
            "renal_any_n":         _renal_n,
            "renal_any_pct":       _pct(_renal_n),
            "joubert_n":           _joubert_n,
            "joubert_pct":         _pct(_joubert_n),
            "hydrops_n":           _hydrops_n,
            "hydrops_pct":         _pct(_hydrops_n),
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
            {"label": "Absent cilia — KIAA0586 basal body class (typical SRTD16)",             "n": 24},
            {"label": "Rudimentary basal body stubs — no axoneme extension",                    "n": 11},
            {"label": "Short rudimentary axoneme — partial CC3 Joubert-allele function",         "n": 5},
        ],
        "srtd_molecular_class_table": [
            {"class": "IFT-A retrograde adaptor",        "em": "SHORT STUBBY cilia",
             "genes": "SRTD2 (IFT122) · SRTD4 (TTC21B) · SRTD5 (WDR19) · SRTD7 (WDR35) · SRTD9 (IFT140) · SRTD14 (IFT43)",
             "why": "IFT-A C-cap or core destabilised → IFT-B import failure → cilia cannot extend"},
            {"class": "Dynein-2 retrograde motor",       "em": "CLUB / BULGING TIP",
             "genes": "SRTD3 (DYNC2H1) · SRTD8 (WDR60) · SRTD11 (WDR34) · SRTD15 (DYNC2LI1) · SRTD17 (TCTEX1D2)",
             "why": "Retrograde IFT fails; IFT-B cargo accumulates at tip → bulging club"},
            {"class": "IFT-B2 anterograde subcomplex",  "em": "SHORTENED cilia",
             "genes": "SRTD1 (IFT80) · SRTD10 (IFT172) · SRTD13 (CLUAP1)",
             "why": "IFT-B2 assembly fails; kinesin-2 tip anchor lost; cilia truncated"},
            {"class": "Basal Body Kinase",               "em": "ABSENT / RUDIMENTARY",
             "genes": "SRTD6 (NEK1)",
             "why": "NEK1 kinase fails to phosphorylate TTBK2; CP110 not removed; axoneme nucleation fails"},
            {"class": "NEK1–IFT-A Bridge",              "em": "VARIABLE — SHORT STUBBY / PARTIALLY ABSENT",
             "genes": "SRTD12 (C21orf2)",
             "why": "Dual defect bridging NEK1 and IFT-A; EM varies by allele dominance"},
            {"class": "Centriolar/CPLANE Scaffold",      "em": "ABSENT / RUDIMENTARY STUBS",
             "genes": "SRTD16 (KIAA0586 — THIS GENE)",
             "why": "Basal body maturation fails; IFT-A/B cannot be recruited → ciliogenesis aborts at step 0; JBTS23 Joubert alleles"},
        ],
        "cplane_complex_table": [
            {"component": "KIAA0586 (TALPID3)", "gene_srtd": "KIAA0586 / SRTD16 (THIS GENE)", "role": "Centriolar scaffold; distal appendage; IFT recruitment platform; NEK1 coordination", "omim": "*610178"},
            {"component": "INTU",               "gene_srtd": "INTU (no SRTD designation)",     "role": "Basal body scaffold; CPLANE hub; WDPCP interaction", "omim": "*243310"},
            {"component": "FUZ",                "gene_srtd": "FUZ (JBTS/BBS alleles)",          "role": "Ciliogenesis effector; CPLANE member; PCP effector", "omim": "*610622"},
            {"component": "WDPCP (FRITZ)",      "gene_srtd": "WDPCP (BBS17 alleles)",           "role": "WD40 planar polarity effector; CPLANE member", "omim": "*613580"},
            {"component": "NEK1",               "gene_srtd": "NEK1 / SRTD6",                    "role": "Basal body kinase; CP110 removal; TTBK2 phosphorylation; functionally coordinates with KIAA0586", "omim": "*604588"},
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

    joubert_feat_all = []
    for p in patients:
        joubert_feat_all.extend(p['joubert_features'])
    joubert_feat_dist = Counter(joubert_feat_all)

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
        "joubert_distribution": [
            {"label": "Joubert (JBTS23) alleles — MTS + ataxia", "n": _joubert_n},
            {"label": "No Joubert features",                      "n": N - _joubert_n},
        ] + [{"label": k, "n": v} for k, v in sorted(joubert_feat_dist.items(), key=lambda x: -x[1])],
        "renal_distribution": [{"label": k, "n": v} for k, v in sorted(renal_dist.items(), key=lambda x: -x[1])]
                              + [{"label": "No renal disease", "n": N - _renal_n}],
        "allele_class_summary": [{"label": k, "n": v} for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "ethnicity_distribution": [{"ethnicity": k, "n": v} for k, v in sorted(eth_dist.items(), key=lambda x: -x[1])],
        "presentation_distribution": [{"label": k, "n": v} for k, v in sorted(pres_dist.items(), key=lambda x: -x[1])],
        "misdiagnosis_distribution": [{"label": k, "n": v} for k, v in sorted(mis_dist.items(), key=lambda x: -x[1])],
        "veptr_distribution": [{"label": k, "n": v} for k, v in sorted(veptr_types.items(), key=lambda x: -x[1])],
        "top_variants": [
            {"variant": "p.Thr208Ile (c.623C>T) — CC1 domain (distal appendage, aa 200–220); MENA homozygous consanguineous; moderate SRTD16",                   "n": 8},
            {"variant": "p.Arg369His (c.1106G>A) — CC1/CC2 junction (aa 360–380); European compound het; moderate-severe SRTD16",                                  "n": 7},
            {"variant": "p.Glu752Ter (c.2254G>T) — CC2 central truncating null (aa 752); SRPS-like; pan-ethnic perinatal lethal",                                  "n": 5},
            {"variant": "p.Leu1082Pro (c.3245T>C) — CC3 C-terminal Joubert domain (aa 1,080–1,090); South Asian hypomorphic; JBTS23-like without SRTD thorax",    "n": 5},
            {"variant": "p.Arg483Gln (c.1448G>A) — CPLANE interface CC2 (aa 480–490); Middle Eastern homozygous consanguineous; moderate SRTD16",                  "n": 6},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":          "KIAA0586",
            "aliases":       "TALPID3, CPLANE1",
            "full_name":     "KIAA0586 — Centriolar/Ciliogenesis and Planar Polarity Effectors Scaffold Protein (TALPID3)",
            "omim_gene":     "*610178",
            "chromosome":    "14q23.1",
            "size":          "~1,624 amino acids",
            "protein_class": "Large multi-domain coiled-coil centriolar scaffold; distal appendage protein; CPLANE complex member",
            "key_domains":   "CC1 N-terminal (aa 1–400; distal appendage anchor) · CC2 central + CPLANE interface (aa 401–1,100; INTU/FUZ/WDPCP contact; IFT recruitment) · CC3 C-terminal (aa 1,100–1,624; NEK1 coordination; IFT platform; Joubert hypomorphic domain)",
            "key_interactions": "INTU (CPLANE hub) · FUZ (ciliogenesis effector) · WDPCP/FRITZ (planar polarity) · NEK1/SRTD6 (basal body kinase; CP110 removal coordination) · IFT-A complex (recruitment to basal body)",
            "molecular_class": "Centriolar/CPLANE scaffold — SIXTH distinct SRTD molecular class; ABSENT/RUDIMENTARY cilia EM; gene panel required for subtype",
            "joubert_note": "CC3 C-terminal hypomorphic alleles → JBTS23 Joubert phenotype (cerebellar vermis hypoplasia, molar tooth sign, ataxia) without full SRTD thoracic disease; ONLY SRTD gene with confirmed Joubert hypomorphic alleles",
        },
        "disease_card": {
            "disease":       "Short-Rib Thoracic Dysplasia 16 (SRTD16 / ATD16)",
            "omim_disease":  "#617098",
            "also_known_as": "Jeune ATD16 · JBTS23 (hypomorphic C-terminal CC3 alleles) · SRPS-like (biallelic null)",
            "inheritance":   "Autosomal Recessive — biallelic LOF",
            "prevalence":    "~1:1,000,000+; <20 families worldwide (2026); ultra-rare",
            "ciliary_em":    "ABSENT cilia or rudimentary basal body stubs — Centriolar/CPLANE scaffold class; same EM class as SRTD6/NEK1; gene panel required",
            "polydactyly":   "~55% — postaxial dominant, occasional preaxial; SECOND-HIGHEST rate after NEK1/SRTD6 (65–75%)",
            "unique_features": "Sixth distinct SRTD molecular class (Centriolar/CPLANE scaffold) · Joubert syndrome alleles (JBTS23; CC3 hypomorphic) — ONLY SRTD gene · Absent/rudimentary cilia (same as SRTD6/NEK1) · CPLANE complex overlap with BBS",
            "renal":         "~22% — TIN; NPHP-like (Joubert overlap); cortical cysts; ESRD",
            "retinal":       "~18% — rod-cone; higher due to Joubert/CPLANE overlap",
            "chf":           "~7%",
            "hydrops":       "~10% — hydrops fetalis (lower than NEK1/SRTD6 ~20%)",
            "perinatal_lethality": "~28% (biallelic null — SRPS-like spectrum)",
            "surgical_tx":   "VEPTR / MAGEC growing rods (standard SRTD protocol)",
            "joubert_phenotype": "Cerebellar vermis hypoplasia · Molar tooth sign (MTS) on brain MRI · Ataxia / truncal instability · Oculomotor apraxia · Episodic hyperpnea (JBTS23 CC3-hypomorphic; absent/mild thorax)",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: narrow thorax + polydactyly → suspect SRTD; absent cilia post-natally → Centriolar/CPLANE scaffold class (same as SRTD6); KIAA0586 often absent from pre-2016 SRTD panels",
            "2. Post-natal: chest radiograph + skeletal survey → short ribs, thoracic diameter <2 SD; brain MRI if Joubert features suspected",
            "3. URGENT: comprehensive ciliopathy gene panel including KIAA0586/TALPID3 — NOT on pre-2016 SRTD panels; confirm KIAA0586 alongside NEK1 (SRTD6) and C21orf2 (SRTD12) for basal-body-class coverage",
            "4. Joubert MTS: if cerebellar vermis hypoplasia + molar tooth sign → include KIAA0586 on JBTS panel (hypomorphic CC3 alleles → JBTS23 without SRTD thorax); JBTS23 often missing KIAA0586 before 2016",
            "5. Ciliary EM (nasal brush): absent cilia or rudimentary basal body stubs — same as SRTD6/NEK1; confirms Centriolar/CPLANE scaffold class; gene panel required to distinguish SRTD16 from SRTD6",
            "6. Retinal: ERG from age 3; OCT; annual ophthalmology (~18% rod-cone; JBTS23 overlap increases retinal risk)",
            "7. Renal: annual USS + GFR (~22%); ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous; no recurrence); NPHP-like histology possible in Joubert alleles",
            "8. Cardiac echo: CHD ~7%; hydrops evaluation prenatally (~10%)",
            "9. Neurology: brain MRI for Joubert MTS if CC3 alleles suspected; developmental assessment; ataxia evaluation",
            "10. Genetics: 25% AR recurrence; cascade testing; prenatal / PGT available",
            "11. Panel completeness: confirm KIAA0586 alongside NEK1 (SRTD6); both have absent EM; KIAA0586 also on JBTS23 panel",
            "12. Therapeutic: no approved agent (2026); CPLANE complex is a research target; enrol in ciliopathy / rare-disease registry",
        ],
        "mechanism_glossary": [
            {"term": "KIAA0586 (TALPID3/CPLANE1) — Centriolar/CPLANE Scaffold", "definition":
             "~1,624-aa multi-domain coiled-coil centriolar scaffold protein localizing to distal "
             "appendages of the mother centriole. Required for basal body maturation, CPLANE complex "
             "assembly, and recruitment of IFT-A/IFT-B components to the ciliary base. Loss → "
             "ciliogenesis aborts at step 0 → ABSENT cilia (sixth distinct SRTD molecular class)."},
            {"term": "CPLANE complex (Ciliogenesis and Planar Polarity Effectors)", "definition":
             "Protein complex including KIAA0586 (TALPID3), INTU, FUZ, and WDPCP/FRITZ. "
             "Coordinates ciliogenesis initiation with planar cell polarity (PCP) signalling at "
             "the basal body. KIAA0586 is the centriolar scaffold subunit; INTU and FUZ are "
             "effectors. CPLANE dysfunction links SRTDs, Joubert syndrome, and BBS."},
            {"term": "Joubert Syndrome 23 (JBTS23) — KIAA0586 hypomorphic CC3 alleles", "definition":
             "Hypomorphic missense variants in the KIAA0586 C-terminal CC3 domain (aa 1,100–1,624) → "
             "partial loss; ciliary platform reduced but not absent → cerebellar vermis hypoplasia + "
             "molar tooth sign (MTS) + ataxia (Joubert phenotype) without full SRTD thoracic skeletal "
             "disease. JBTS23 is the Joubert designation for KIAA0586 hypomorphic alleles — "
             "SRTD16 is the full biallelic LOF designation. KIAA0586 is the ONLY SRTD gene with "
             "confirmed Joubert syndrome alleles."},
            {"term": "Absent cilia — Centriolar/CPLANE scaffold class (SRTD6/SRTD16)", "definition":
             "Both NEK1/SRTD6 (basal body kinase) and KIAA0586/SRTD16 (centriolar scaffold) produce "
             "absent/rudimentary cilia; the EM pattern is indistinguishable between these two classes. "
             "Mechanism differs: NEK1 fails to phosphorylate TTBK2 → CP110 not removed → ciliogenesis "
             "never initiates; KIAA0586 loses the basal body scaffold → IFT proteins cannot be "
             "recruited → ciliogenesis aborts at or just after CP110 removal. Gene panel required."},
            {"term": "Hedgehog signalling failure (SRTD16)", "definition":
             "KIAA0586 LOF → absent cilia → complete failure of ciliary Hedgehog platform → "
             "Smo activation fails; Gli3R accumulates; Ihh/Shh absent in chondrocytes → "
             "narrow thorax, short ribs, limb shortening. High polydactyly rate (55%) reflects "
             "near-complete Hedgehog failure, similar to NEK1/SRTD6."},
        ],
        "key_variants": [
            {"variant": "p.Thr208Ile", "domain": "CC1 distal appendage domain (aa 200–220)", "consequence":
             "Distal appendage anchoring weakened; basal body maturation delayed; IFT recruitment reduced; moderate SRTD16",
             "ethnicity": "MENA homozygous consanguineous"},
            {"variant": "p.Arg369His", "domain": "CC1/CC2 junction (aa 360–380)", "consequence":
             "CPLANE complex assembly disrupted at CC1-CC2 interface; IFT-A/B recruitment partially lost; moderate-severe SRTD16",
             "ethnicity": "European compound het"},
            {"variant": "p.Glu752Ter", "domain": "CC2 central truncating null (aa 752)", "consequence":
             "Null allele; complete KIAA0586 LOF; SRPS-like spectrum; perinatal lethal if homozygous",
             "ethnicity": "Pan-ethnic"},
            {"variant": "p.Leu1082Pro", "domain": "CC3 C-terminal Joubert domain (aa 1,080–1,090)", "consequence":
             "Hypomorphic; IFT platform and NEK1 coordination partially preserved; ciliary platform reduced → JBTS23 Joubert MTS without SRTD thoracic disease",
             "ethnicity": "South Asian hypomorphic"},
            {"variant": "p.Arg483Gln", "domain": "CPLANE interface CC2 (aa 480–490)", "consequence":
             "CPLANE scaffold disrupted at INTU/FUZ binding interface; IFT recruitment severely reduced; moderate SRTD16",
             "ethnicity": "Middle Eastern homozygous"},
        ],
        "treatment_summary": [
            "1. Narrow thorax (primary): VEPTR or MAGEC growing rod — standard serial thoracic expansion protocol for all SRTDs",
            "2. Neonatal respiratory: mechanical ventilation (biallelic null SRPS-like); CPAP/BiPAP (moderate); hydrops management prenatally (~10%)",
            "3. Renal: annual GFR/USS/creatinine (~22%); ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous; no recurrence); NPHP-like workup in Joubert alleles",
            "4. Retinal: annual ERG from age 3 (~18% rod-cone; Joubert overlap); low-vision support",
            "5. Neurology: brain MRI for cerebellar vermis / MTS in CC3 alleles (JBTS23); developmental assessment; ataxia physiotherapy",
            "6. Hepatic: APRI + USS annually (~7% CHF); hepatology if elevated LFTs",
            "7. Gene panel: ensure KIAA0586/TALPID3 included alongside NEK1 (SRTD6); both absent cilia; also include KIAA0586 on JBTS23 panel for Joubert CC3 alleles",
            "8. Genetics: 25% AR recurrence; cascade testing; prenatal / PGT available",
            "9. Therapeutic pipeline: no approved agent (2026); CPLANE complex modulation is a research target; enrol in ciliopathy registry",
        ],
        "ddx_table": [
            {"disease": "SRTD6 (NEK1) — Basal Body Kinase; most similar EM class", "key_difference":
             "NEK1 (SRTD6) is the basal body kinase that phosphorylates TTBK2 to remove CP110; "
             "KIAA0586 (SRTD16) is the centriolar scaffold that recruits IFT machinery after CP110 removal. "
             "Both produce absent/rudimentary cilia (EM indistinguishable). NEK1/SRTD6 has hydrops 20% + "
             "medial nasal hypoplasia 30%; SRTD16 has Joubert JBTS23 alleles (absent in SRTD6). Gene panel MANDATORY."},
            {"disease": "Joubert Syndrome 23 (JBTS23) — same gene (KIAA0586) hypomorphic alleles", "key_difference":
             "JBTS23 and SRTD16 are allelic conditions: JBTS23 = hypomorphic CC3 C-terminal missense → "
             "partial ciliary platform → cerebellar vermis hypoplasia + MTS + ataxia; no/mild thorax. "
             "SRTD16 = biallelic LOF → absent cilia → full thoracic skeletal disease. "
             "KIAA0586 is the ONLY SRTD gene with confirmed Joubert hypomorphic alleles."},
            {"disease": "SRTD12 (C21orf2) — NEK1–IFT-A Bridge; different class but EM overlap", "key_difference":
             "C21orf2 bridges NEK1 kinase and IFT-A complex; SRTD12 has variable EM (short stubby / partially absent); "
             "SRTD16 has absent/rudimentary cilia. C21orf2 has dual LCA/CORD retinal dominant alleles absent in SRTD16. "
             "Both rare; gene panel differentiates."},
            {"disease": "BBS (Bardet-Biedl Syndrome) — CPLANE complex overlap", "key_difference":
             "CPLANE dysfunction links SRTD16 and BBS (WDPCP/BBS17, FUZ alleles overlap BBS CPLANE). "
             "BBS hallmarks: obesity, polydactyly, retinal dystrophy, intellectual disability — NO narrow thorax. "
             "SRTD16: narrow thorax + polydactyly + absent cilia; NO obesity. Gene panel resolves."},
            {"disease": "SRTD3 (DYNC2H1) — most common SRTD; different EM class", "key_difference":
             "DYNC2H1 → club/bulging tip cilia (IFT-B stranded at tip — Dynein-2 class) — DIFFERENT EM CLASS "
             "from absent cilia of SRTD16. SRTD3 ~50% of all molecular SRTDs; no Joubert alleles. Gene panel confirms."},
            {"disease": "Joubert Syndrome (other genes — JBTS1/INPP5E, JBTS2/TMEM216, etc.)", "key_difference":
             "Pure Joubert syndromes lack narrow thorax; JBTS23/KIAA0586 is the ONLY Joubert gene that "
             "also causes an SRTD when fully lost. If cerebellar vermis hypoplasia (MTS) is present with "
             "thoracic narrowing, KIAA0586 is the top candidate. Gene panel confirms."},
        ],
    }
