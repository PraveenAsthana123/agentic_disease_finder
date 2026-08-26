"""
BBS4 Bardet-Biedl Syndrome Type 4 — BBS4 (Tetratricopeptide Repeat Domain 8) Dashboard
=========================================================================================
Primary Gene : BBS4 (*600374) — Bardet-Biedl Syndrome 4 Protein;
               519 aa; 8 tetratricopeptide repeat (TPR) domains (aa 42–519);
               N-terminal cap (aa 1–41); 15q24.1.
               BBS4 is a STRUCTURAL SUBUNIT of the BBSome octameric complex,
               occupying a PERIPHERAL POSITION adjacent to BBS5 (PH-domain
               lipid-binding) and BBIP10/BBIP1 (dynein regulatory):
                 TPR repeats 1–4 (aa 42–250): scaffold for BBS5 (PH domain)
                   interaction; BBS4 Gly180 contacts BBS5 PH-domain β-strand —
                   Gly180Ser → steric clash → BBS4–BBS5 interface disrupted;
                 TPR repeats 5–8 (aa 251–519): BBIP10 and pericentriolar
                   satellite (PCM1) interaction domain; BBS4 Pro247 maintains
                   TPR stacking geometry — Pro247Ser → TPR5–8 partially unfolds
                   → satellite association reduced;
                 Leu147 (TPR3): hydrophobic packing of TPR3 core; Leu147Pro →
                   TPR3 misfolded → cascading TPR4–5 destabilisation → BBS5
                   contact surface lost.
               BBSome pericentriolar satellite (PCM1) localisation mechanism:
                 BBS4 TPR5–8 tethers the nascent BBSome to pericentriolar
                 material (PCM1 granules) during assembly → BBSome oligomerises
                 at pericentrin scaffold → mature BBSome migrates to ciliary
                 base (IFT-docking zone) → IFT-B complex picks up BBSome for
                 axonemal transport.
               BBS4 LOF → BBSome fails to anchor to PCM1 pericentriolar
               satellites → BBSome assembly is impaired at the satellite step
               (upstream of ARL6/BBS3 membrane recruitment) → soluble BBSome
               subunits accumulate in cytoplasm → all BBSome-dependent GPCR
               ciliary trafficking fails → multi-system BBS.
               KEY DISTINCTION from BBS3 (ARL6): BBS4 failure is at PCM1
               satellite anchoring (assembly prerequisite), not at membrane
               recruitment; unlike BBS3 where BBSome assembles but cannot dock,
               in BBS4 the BBSome does not complete assembly at pericentriolar
               satellites → failure is earlier in the biogenesis pathway.
               KEY DISTINCTION from BBS2: BBS2 LOF collapses the BBS2–BBS7
               structural core dimer (first assembly step), whereas BBS4 LOF
               disrupts the satellite-tethering step (later step, BBS5–BBIP10
               integration); BBSome partial intermediates may persist in BBS4.

Disease OMIM : #209900 — Bardet-Biedl Syndrome (BBS4 allelic disorder)
               All BBS types share OMIM #209900; molecular subtype defined by gene.
               Allele spectrum:
               — Severe: biallelic null (truncating/null-null) → no BBS4 →
                 complete satellite-tethering failure; severe multi-system BBS
               — Moderate: compound het (TPR missense + truncating) → partial
                 satellite association; moderate multi-system BBS
               — Mild: hypomorphic TPR4–8 missense → reduced satellite tethering;
                 attenuated multi-system BBS; possible retinal-only or oligosyndromic
               UNIQUE FEATURES vs BBS1, BBS2, BBS3:
               (1) PERICENTRIOLAR SATELLITE ANCHOR: BBS4 is the only BBSome
                   subunit that directly links the assembling BBSome to PCM1
                   pericentriolar granules — a function shared with no other BBS
                   gene; loss of PCM1 satellite localisation distinguishes BBS4
                   LOF at the cellular level (immunofluorescence: BBS4 IF pattern
                   lost; compare BBS2 where core scaffold is absent)
               (2) BBSome PARTIAL INTERMEDIATE: unlike BBS2 where BBSome
                   completely misassembles, BBS4 LOF may leave partial BBSome
                   intermediates (BBS1–BBS2–BBS7 trimer) in cytoplasm — these
                   partial complexes are non-functional but detectable; BBS4 is
                   required to complete the BBS5–BBIP10 module integration
               (3) FREQUENCY ~2–4% of molecularly confirmed BBS: rarer than BBS1
                   (~25%) or BBS2 (~10–20%); under 100 families worldwide (2026)
               (4) NO SINGLE DOMINANT ETHNIC FOUNDER: allele spectrum pan-ethnic;
                   Leu147Pro has slight European enrichment; no founder comparable
                   to BBS1 M390R or BBS2 R631P (Bedouin)
               (5) PERICENTRIOLAR SATELLITE IF AS DIAGNOSTIC TOOL: BBS4
                   immunofluorescence (anti-BBS4 antibody showing PCM1-colocalised
                   signal) is ABSENT in BBS4 LOF; this cellular diagnostic tool
                   is unique to BBS4 and can confirm pathogenicity of BBS4 VUS
                   in patient-derived fibroblasts
Chromosome   : 15q24.1
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous);
               tri-allelic BBS in ~4% (peripheral BBSome position)
Prevalence   : ~1:100,000–160,000 worldwide (BBS overall); BBS4 accounts for
               ~2–4% of molecularly confirmed BBS cases; under 100 families (2026)
"""

import random

SEED = 339
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
    ('European',                                     0.33),  # Leu147Pro slight enrichment
    ('Middle Eastern / North African',               0.27),  # Arg287Ter
    ('South Asian',                                  0.20),  # Pro247Ser
    ('East Asian / Other',                           0.13),
    ('Latin American',                               0.07),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('European')
rng.shuffle(eth_pool)

# Allele classes
allele_classes = ['null/null', 'TPR missense / truncating', 'compound het (2 missense)']
allele_weights = [0.40, 0.42, 0.18]

# Allele pool (weighted)
def _pick_allele():
    r = rng.random()
    cum = 0.0
    for cls, w in zip(allele_classes, allele_weights):
        cum += w
        if r < cum:
            return cls
    return allele_classes[-1]

# Clinical parameters (BBS4 literature values, ~2–4% of BBS)
# Rod-cone: 100%; postaxial poly: ~65%; obesity: ~80%; cognitive: ~50%;
# renal: ~45%; hypogonadism: ~65%; anosmia: ~60%; CHD: ~6%; tri-allelic: ~4%
for i in range(N):
    eth  = eth_pool[i]
    allele = _pick_allele()
    poly   = rng.random() < 0.65
    obese  = rng.random() < 0.80
    cogn   = rng.random() < 0.50
    renal  = rng.random() < 0.45
    hypo   = rng.random() < 0.65
    anos   = rng.random() < 0.60
    chd    = rng.random() < 0.06

    # retinal: always rod-cone; stage at diagnosis
    ret_stage = rng.choices(
        ['Early (ERG reduced, minimal VA loss)', 'Moderate (VF constriction, ERG extinguished)',
         'Advanced (tunnel vision, VA ≤ 0.1)', 'End-stage (NLP / hand motion)'],
        weights=[0.20, 0.38, 0.30, 0.12]
    )[0]

    # renal sub-type
    renal_type = None
    if renal:
        renal_type = rng.choices(
            ['Renal cysts', 'Calyceal clubbing', 'Horseshoe kidney', 'CKD Gr 3+'],
            weights=[0.40, 0.30, 0.18, 0.12]
        )[0]

    # polydactyly sub-type
    poly_type = None
    if poly:
        poly_type = rng.choices(
            ['Postaxial hands only', 'Postaxial hands+feet', 'Postaxial feet only'],
            weights=[0.38, 0.45, 0.17]
        )[0]

    # triallelic (BBS4 + second BBS gene variant)
    triallelic = rng.random() < 0.04

    # top recurrent variant (educational)
    variant = rng.choices(
        ['p.Leu147Pro (c.440T>C)', 'p.Arg287Ter (c.859C>T)',
         'p.Gly180Ser (c.538G>A)', 'p.Pro247Ser (c.739C>T)', 'p.Trp410Ter (c.1230G>A)'],
        weights=[0.28, 0.24, 0.20, 0.16, 0.12]
    )[0]

    # age at diagnosis (yr)
    dx_age = rng.choices(
        ['0–4', '5–11', '12–17', '18+'],
        weights=[0.15, 0.35, 0.30, 0.20]
    )[0]

    # initial misdiagnosis
    misdiag = rng.choices(
        ['Retinitis Pigmentosa (isolated)', 'Alström Syndrome', 'Laurence-Moon Syndrome',
         'Non-syndromic obesity', 'Cohen Syndrome', 'Correct BBS (first referral)'],
        weights=[0.28, 0.12, 0.10, 0.15, 0.05, 0.30]
    )[0]

    patients.append({
        'ethnicity': eth, 'allele_class': allele, 'polydactyly': poly,
        'poly_type': poly_type, 'obesity': obese, 'cognitive': cogn,
        'renal': renal, 'renal_type': renal_type, 'hypogonadism': hypo,
        'anosmia': anos, 'chd': chd, 'retinal_stage': ret_stage,
        'triallelic': triallelic, 'variant': variant, 'dx_age': dx_age,
        'misdiagnosis': misdiag,
    })


# ── API handlers ──────────────────────────────────────────────────────────────
def get_overview():
    poly_n   = sum(1 for p in patients if p['polydactyly'])
    obese_n  = sum(1 for p in patients if p['obesity'])
    cogn_n   = sum(1 for p in patients if p['cognitive'])
    renal_n  = sum(1 for p in patients if p['renal'])
    hypo_n   = sum(1 for p in patients if p['hypogonadism'])
    anos_n   = sum(1 for p in patients if p['anosmia'])
    chd_n    = sum(1 for p in patients if p['chd'])
    tri_n    = sum(1 for p in patients if p['triallelic'])
    esrd_n   = sum(1 for p in patients
                   if p['renal'] and p['renal_type'] == 'CKD Gr 3+')
    misdiag_n = sum(1 for p in patients
                    if p['misdiagnosis'] != 'Correct BBS (first referral)')

    stages_cnt = {}
    for p in patients:
        stages_cnt[p['retinal_stage']] = stages_cnt.get(p['retinal_stage'], 0) + 1
    retinal_dist = [
        {'label': s, 'n': stages_cnt.get(s, 0)}
        for s in ['Early (ERG reduced, minimal VA loss)',
                  'Moderate (VF constriction, ERG extinguished)',
                  'Advanced (tunnel vision, VA ≤ 0.1)',
                  'End-stage (NLP / hand motion)']
    ]

    age_bins = {'0–4': 0, '5–11': 0, '12–17': 0, '18+': 0}
    for p in patients:
        age_bins[p['dx_age']] += 1

    bbs_pathway_comparison = [
        {'gene': 'BBS1',          'role': 'Cargo recognition β-propeller; BBSome ARL6-docking platform', 'function_class': 'Cargo Adaptor', 'omim': '209901', 'frequency_pct': '25'},
        {'gene': 'BBS2',          'role': 'Core structural WD40/coiled-coil; BBS2–BBS7 spine dimer',      'function_class': 'Core Scaffold',  'omim': '606151', 'frequency_pct': '10–20'},
        {'gene': 'BBS3 / ARL6',   'role': 'Small GTPase; recruits intact BBSome to ciliary membrane',     'function_class': 'GTPase Switch',  'omim': '608845', 'frequency_pct': '3–5'},
        {'gene': 'BBS4 ← THIS',   'role': 'TPR scaffold; tethers nascent BBSome to PCM1 pericentriolar satellites; BBS5–BBIP10 integration', 'function_class': 'Satellite Anchor', 'omim': '600374', 'frequency_pct': '2–4'},
        {'gene': 'BBS5',          'role': 'PH-domain lipid-binding; BBSome membrane contact (PtdIns)',    'function_class': 'Lipid Sensor',   'omim': '603650', 'frequency_pct': '2–3'},
        {'gene': 'BBS7',          'role': 'WD40/coiled-coil; BBS2–BBS7 core dimer partner',              'function_class': 'Core Scaffold',  'omim': '607590', 'frequency_pct': '2–3'},
        {'gene': 'BBS8 / TTC8',   'role': 'TPR repeats; IFT-B docking; transitions BBSome to IFT trains','function_class': 'IFT Adaptor',    'omim': '608132', 'frequency_pct': '2–3'},
        {'gene': 'BBS9 / PTHB1',  'role': 'Platform subunit; bridges BBS2–BBS7 and BBS4–BBS5 modules',  'function_class': 'Platform Bridge', 'omim': '607968', 'frequency_pct': '2–3'},
        {'gene': 'BBS10',         'role': 'CCT/TRiC-like chaperonin; assists BBSome assembly',           'function_class': 'Chaperonin',     'omim': '610148', 'frequency_pct': '20–25'},
    ]

    return {
        'cohort_n': N,
        'seed': SEED,
        'kpis': {
            'polydactyly_n': poly_n,   'polydactyly_pct': _pct(poly_n),
            'obesity_n': obese_n,      'obesity_pct': _pct(obese_n),
            'cognitive_n': cogn_n,     'cognitive_pct': _pct(cogn_n),
            'renal_any_n': renal_n,    'renal_any_pct': _pct(renal_n),
            'hypogonadism_n': hypo_n,  'hypogonadism_pct': _pct(hypo_n),
            'anosmia_n': anos_n,       'anosmia_pct': _pct(anos_n),
            'chd_n': chd_n,            'chd_pct': _pct(chd_n),
            'triallelic_n': tri_n,     'triallelic_pct': _pct(tri_n),
            'esrd_n': esrd_n,
            'misdiagnosis_n': misdiag_n, 'misdiagnosis_pct': _pct(misdiag_n),
            'retinal_endstage_n': stages_cnt.get('End-stage (NLP / hand motion)', 0),
            'retinal_endstage_pct': _pct(stages_cnt.get('End-stage (NLP / hand motion)', 0)),
        },
        'mechanism': (
            "BBS4 protein contains 8 TPR (tetratricopeptide repeat) domains (aa 42–519) "
            "that scaffold the BBS5 (PH-domain lipid-binding) and BBIP10 (dynein regulatory) "
            "subunits within the assembling BBSome. Critically, BBS4 TPR5–8 tethers the "
            "nascent BBSome to PCM1 pericentriolar satellites — protein granules at the "
            "pericentriolar material where BBSome subunits concentrate for stepwise assembly. "
            "BBS4 LOF → BBSome cannot anchor to PCM1 satellites → BBS5–BBIP10 module fails "
            "to integrate → BBSome partial intermediates (BBS1–BBS2–BBS7 trimer) accumulate "
            "but cannot complete → mature BBSome absent from ciliary base → GPCR cargo "
            "(LepR, SSTR3, MCHR1) fails to enter cilia → multi-system BBS."
        ),
        'key_distinction': (
            "BBS4 uniquely anchors BBSome assembly to PCM1 pericentriolar satellites — "
            "a cellular function not shared by any other BBS gene. "
            "vs BBS2: BBS2 LOF collapses the BBS2–BBS7 core dimer (step 0); BBS4 LOF "
            "fails at pericentriolar satellite tethering (later step); partial BBSome "
            "intermediates may persist in BBS4 but not BBS2. "
            "vs BBS3/ARL6: ARL6 LOF leaves a fully assembled BBSome stranded in the "
            "cytoplasm (cannot reach ciliary membrane); BBS4 LOF prevents BBSome from "
            "assembling completely (earlier biogenesis defect). "
            "vs BBS1: BBS1 is the cargo recognition subunit (recognises GPCR sorting "
            "signals); BBS4 is a structural satellite-tethering scaffold — entirely "
            "different functional class. "
            "Diagnostic tool unique to BBS4: anti-BBS4 immunofluorescence in patient "
            "fibroblasts shows absence of PCM1-colocalised BBS4 puncta — confirmatory "
            "for BBS4 LOF VUS classification."
        ),
        'bbs_pathway_comparison': bbs_pathway_comparison,
        'retinal_distribution': retinal_dist,
        'age_distribution': {
            'dx_0_4yr': age_bins['0–4'],
            'dx_5_11yr': age_bins['5–11'],
            'dx_12_17yr': age_bins['12–17'],
            'dx_18plus': age_bins['18+'],
        },
    }


def get_breakdown():
    systemic = [
        ('Rod-cone dystrophy (rod first, 100%)', N, 100),
        ('Postaxial polydactyly',   sum(1 for p in patients if p['polydactyly']),   _pct(sum(1 for p in patients if p['polydactyly']))),
        ('Obesity / hyperphagia',   sum(1 for p in patients if p['obesity']),        _pct(sum(1 for p in patients if p['obesity']))),
        ('Hypogonadism',            sum(1 for p in patients if p['hypogonadism']),   _pct(sum(1 for p in patients if p['hypogonadism']))),
        ('Anosmia / hyposmia',      sum(1 for p in patients if p['anosmia']),        _pct(sum(1 for p in patients if p['anosmia']))),
        ('Cognitive / LD',          sum(1 for p in patients if p['cognitive']),      _pct(sum(1 for p in patients if p['cognitive']))),
        ('Renal anomaly',           sum(1 for p in patients if p['renal']),          _pct(sum(1 for p in patients if p['renal']))),
        ('CHD',                     sum(1 for p in patients if p['chd']),             _pct(sum(1 for p in patients if p['chd']))),
    ]
    systemic_burden = [{'feature': f, 'n': n, 'pct': pct} for f, n, pct in systemic]

    eth_cnt = {}
    for p in patients:
        eth_cnt[p['ethnicity']] = eth_cnt.get(p['ethnicity'], 0) + 1
    ethnicity_distribution = [
        {'ethnicity': eth, 'n': cnt}
        for eth, cnt in sorted(eth_cnt.items(), key=lambda x: -x[1])
    ]

    allele_cnt = {}
    for p in patients:
        allele_cnt[p['allele_class']] = allele_cnt.get(p['allele_class'], 0) + 1
    allele_class_summary = [
        {'label': cls, 'n': cnt}
        for cls, cnt in sorted(allele_cnt.items(), key=lambda x: -x[1])
    ]

    stage_cnt = {}
    for p in patients:
        stage_cnt[p['retinal_stage']] = stage_cnt.get(p['retinal_stage'], 0) + 1
    retinal_stage_distribution = [
        {'label': s, 'n': stage_cnt.get(s, 0)}
        for s in ['Early (ERG reduced, minimal VA loss)',
                  'Moderate (VF constriction, ERG extinguished)',
                  'Advanced (tunnel vision, VA ≤ 0.1)',
                  'End-stage (NLP / hand motion)']
    ]

    renal_cnt = {}
    for p in patients:
        if p['renal'] and p['renal_type']:
            renal_cnt[p['renal_type']] = renal_cnt.get(p['renal_type'], 0) + 1
    renal_cnt['No renal anomaly'] = N - sum(1 for p in patients if p['renal'])
    renal_distribution = [
        {'label': k, 'n': v}
        for k, v in sorted(renal_cnt.items(), key=lambda x: -x[1])
    ]

    poly_cnt = {}
    for p in patients:
        if p['polydactyly'] and p['poly_type']:
            poly_cnt[p['poly_type']] = poly_cnt.get(p['poly_type'], 0) + 1
    poly_cnt['No polydactyly'] = N - sum(1 for p in patients if p['polydactyly'])
    polydactyly_distribution = [
        {'label': k, 'n': v}
        for k, v in sorted(poly_cnt.items(), key=lambda x: -x[1])
    ]

    age_cnt = {}
    for p in patients:
        age_cnt[p['dx_age']] = age_cnt.get(p['dx_age'], 0) + 1
    presentation_distribution = [
        {'label': f'{g} yr', 'n': age_cnt.get(g, 0)}
        for g in ['0–4', '5–11', '12–17', '18+']
    ]

    misdiag_cnt = {}
    for p in patients:
        misdiag_cnt[p['misdiagnosis']] = misdiag_cnt.get(p['misdiagnosis'], 0) + 1
    misdiagnosis_distribution = [
        {'label': k, 'n': v}
        for k, v in sorted(misdiag_cnt.items(), key=lambda x: -x[1])
    ]

    var_cnt = {}
    for p in patients:
        var_cnt[p['variant']] = var_cnt.get(p['variant'], 0) + 1
    top_variants = [
        {'variant': k, 'n': v}
        for k, v in sorted(var_cnt.items(), key=lambda x: -x[1])
    ]

    return {
        'systemic_burden': systemic_burden,
        'ethnicity_distribution': ethnicity_distribution,
        'allele_class_summary': allele_class_summary,
        'retinal_stage_distribution': retinal_stage_distribution,
        'renal_distribution': renal_distribution,
        'polydactyly_distribution': polydactyly_distribution,
        'presentation_distribution': presentation_distribution,
        'misdiagnosis_distribution': misdiagnosis_distribution,
        'top_variants': top_variants,
    }


def get_definitions():
    return {
        'gene_card': {
            'Gene':              'BBS4',
            'Aliases':           'BBS4 protein; tetratricopeptide repeat protein (BBS4-TPR)',
            'OMIM Gene':         '*600374',
            'Protein length':    '519 aa',
            'Domains':           '8 TPR repeats (aa 42–519); N-terminal cap (aa 1–41)',
            'Chromosome':        '15q24.1',
            'mRNA':              'NM_033028.4',
            'Protein':           'NP_149018.1',
            'Function':          'TPR scaffold tethers assembling BBSome to PCM1 pericentriolar satellites; integrates BBS5–BBIP10 module',
            'BBSome role':       'Peripheral structural subunit (satellite anchor); not part of BBS2–BBS7 core dimer',
            'Interactors':       'BBS5 (PH domain), BBIP10, PCM1, BBS9 (platform bridge), BBS8/TTC8',
            'Expression':        'Ubiquitous; high in RPE, photoreceptors, renal tubules, olfactory neurons',
            'Mouse model':       'Bbs4−/− (Mykytyn 2004): rod-cone degeneration, obesity, anosmia, infertility',
        },
        'disease_card': {
            'Disease':           'Bardet-Biedl Syndrome Type 4',
            'OMIM Disease':      '#209900',
            'Locus':             'BBS4 (15q24.1)',
            'Inheritance':       'Autosomal Recessive; biallelic LOF',
            'Prevalence':        '~1:100,000–160,000 (BBS overall); BBS4 ~2–4% of BBS',
            'Molecular class':   'BBSome pericentriolar satellite anchor (TPR structural subunit)',
            'Key features':      'Rod-cone (rod first), postaxial polydactyly, truncal obesity, renal anomalies, hypogonadism, anosmia, cognitive LD',
            'Retinal pattern':   'Rod-cone dystrophy (nyctalopia first); ERG scotopic extinguished before photopic; same as BBS1/BBS2/BBS3',
            'Obesity mechanism': 'LepR mis-trafficking (satiety signalling failure); hyperphagia; BMI 29–45',
            'Tri-allelic':       '~4% (lower than BBS2 ~9%; BBS4 peripheral BBSome position)',
            'Unique feature':    'PCM1-colocalised BBS4 IF puncta absent in patient fibroblasts; diagnostic for BBS4 VUS',
            'Gene panel':        'Full BBS panel (20–24 genes) + MLPA/CNV mandatory; BBS4 alone detected in <4% without panel',
            'Frequency':         '~2–4% of molecularly confirmed BBS; under 100 families worldwide (2026)',
        },
        'key_variants': [
            {'variant': 'p.Leu147Pro (c.440T>C)',  'domain': 'TPR3 hydrophobic core',          'consequence': 'TPR3 misfolding → BBS5 contact surface lost; TPR4–5 cascade destabilisation', 'ethnicity': 'European (slight enrichment)'},
            {'variant': 'p.Arg287Ter (c.859C>T)',  'domain': 'TPR6 (C-terminal module)',        'consequence': 'Truncating null → no BBIP10 interaction domain; complete BBS4 LOF',             'ethnicity': 'Pan-ethnic; Middle Eastern enriched'},
            {'variant': 'p.Gly180Ser (c.538G>A)',  'domain': 'TPR4 / BBS5 interface',          'consequence': 'Steric clash at BBS4–BBS5 PH-domain contact → BBS5 integration fails',          'ethnicity': 'South Asian; Middle Eastern'},
            {'variant': 'p.Pro247Ser (c.739C>T)',  'domain': 'TPR5–6 junction (PCM1 anchor)',   'consequence': 'TPR stacking geometry disrupted → pericentriolar satellite association reduced',  'ethnicity': 'South Asian; European'},
            {'variant': 'p.Trp410Ter (c.1230G>A)', 'domain': 'TPR8 (C-terminus)',              'consequence': 'Truncating null → no BBIP10 dynein regulatory domain; complete LOF',             'ethnicity': 'European; North African'},
        ],
        'diagnostic_workup': [
            '1. Clinical: confirm ≥4 primary BBS features (rod-cone dystrophy, polydactyly, obesity, renal anomaly, hypogonadism, anosmia/cognitive LD)',
            '2. Ophthalmology: ERG (scotopic extinguished before photopic = rod-first pattern); fundus photography; OCT for outer retinal layer loss',
            '3. Genetic: full BBS gene panel (20–24 genes, next-gen sequencing + MLPA for CNV); BBS4 detected only by panel, not targeted single-gene',
            '4. BBS4 VUS confirmation: anti-BBS4 immunofluorescence in patient fibroblasts — absence of PCM1-colocalised BBS4 puncta confirms LOF',
            '5. Renal ultrasound: cysts, horseshoe kidney, calyceal clubbing; renal function (eGFR, urine protein)',
            '6. Metabolic: fasting glucose, HbA1c, lipids; insulin resistance screen (HOMA-IR)',
            '7. Endocrine: LH, FSH, testosterone/oestradiol, AMH; growth hormone if short stature',
            '8. Olfactory testing: UPSIT or Sniffin Sticks (anosmia 60%)',
            '9. Cardiac echo: exclude structural CHD (6%)',
            '10. Neurodevelopmental assessment: IQ, learning disability spectrum (50%)',
        ],
        'treatment_summary': [
            '1. Retinal: no approved gene therapy for BBS4 (2026); low vision aids; tinted lenses; yearly ophthalmology; avoid vitamin A supplementation (no evidence in BBS)',
            '2. Obesity: GLP-1 RA (semaglutide/liraglutide) — primary pharmacotherapy; orlistat adjunct; bariatric surgery if BMI >40 with co-morbidities; LepR mis-trafficking mechanism rationale for GLP-1 RA',
            '3. Renal: annual eGFR + urine albumin; ACE inhibitor if CKD/proteinuria; renal transplant for ESRD (curative, no recurrence — BBS4 LOF is cell-autonomous)',
            '4. Polydactyly: surgical excision in infancy (neonatal/early paediatric); orthopaedic referral',
            '5. Hypogonadism: testosterone replacement (males, Tanner stage assessment); oestrogen/progesterone for females; subfertility counselling',
            '6. Cognitive: early intervention (speech, OT, educational support); IEP/504 plan; annual neurodevelopmental review',
            '7. Anosmia: safety alert (gas, smoke, food hygiene); olfactory training (limited evidence)',
            '8. Endocrine: annual HbA1c + lipids from age 10; metformin + GLP-1 RA for T2D-pattern insulin resistance',
            '9. Genetic counselling: AR inheritance; 25% recurrence risk; cascade testing for siblings',
            '10. BBS4 VUS management: BBS4 IF fibroblast assay for VUS classification before initiating treatment',
        ],
        'ddx_table': [
            {'disease': 'BBS1 (BBS1 gene)',        'key_difference': 'BBS1 = cargo recognition β-propeller; M390R European founder; ~25% of BBS; clinically identical — gene panel mandatory to distinguish'},
            {'disease': 'BBS2 (BBS2 gene)',         'key_difference': 'BBS2 = BBS2–BBS7 core dimer (first assembly step); R631P Bedouin founder; BBSome misassembles at step 0 vs BBS4 satellite-tethering failure'},
            {'disease': 'BBS3 / ARL6',             'key_difference': 'ARL6 = GTPase membrane recruiter; BBSome assembles normally but cannot dock membrane; BBS4 failure is earlier (satellite anchoring)'},
            {'disease': 'BBS10 (chaperonin)',       'key_difference': 'BBS10 = CCT/TRiC chaperonin; ~20–25% of BBS; no structural BBSome subunit; chaperonin-class distinct from BBS4 TPR scaffold'},
            {'disease': 'Alström Syndrome (ALMS1)', 'key_difference': 'No polydactyly; cone-rod (NOT rod-first); infantile DCM; SNHL; BBS4 has polydactyly + rod-first + no DCM'},
            {'disease': 'Laurence-Moon Syndrome',   'key_difference': 'Spastic paraplegia + progressive neurological; NO polydactyly; overlap in obesity + retinal; distinct gene (PNPLA6)'},
            {'disease': 'McKusick-Kaufman (MKKS)',  'key_difference': 'Hydrometrocolpos (females) + CHD + polydactyly; no retinal dystrophy; BBS6/MKKS is a chaperonin-like protein'},
            {'disease': 'Joubert Syndrome (JBTS)',  'key_difference': 'Molar tooth sign on MRI; cerebellar vermis hypoplasia; no polydactyly in most; NPHP-class renal not cystic BBS pattern'},
        ],
        'mechanism_glossary': [
            {'term': 'Tetratricopeptide Repeat (TPR) Domain',
             'definition': 'A structural motif of 34-aa degenerate repeats that form antiparallel α-helices stacked in a superhelix; mediates protein–protein interactions via concave groove contacts. BBS4 contains 8 TPR repeats (aa 42–519) that create binding surfaces for BBS5 (PH domain), BBIP10 (dynein regulatory subunit), and PCM1 (pericentriolar satellite scaffold).'},
            {'term': 'Pericentriolar Material (PCM) / PCM1 Satellites',
             'definition': 'Electron-dense protein granules (PCM1, pericentrin, PCNT) surrounding the centrosome that serve as assembly platforms for multi-protein complexes. BBS4 TPR5–8 tethers the assembling BBSome to PCM1 granules, concentrating BBSome subunits for stepwise oligomerisation. Loss of BBS4 causes dispersion of BBSome subunits away from PCM1 foci — detectable by anti-BBS4 immunofluorescence.'},
            {'term': 'BBSome Octameric Complex',
             'definition': 'A coat-like complex of 8 subunits (BBS1, BBS2, BBS4, BBS5, BBS7, BBS8/TTC8, BBS9/PTHB1, BBIP10) that acts as a cargo adaptor for GPCR trafficking into and out of primary cilia. Assembly is stepwise: BBS2–BBS7 core dimer → BBS9 platform → BBS4–BBS5–BBIP10 peripheral module (BBS4-dependent satellite tethering) → BBS1 cargo recognition → ARL6/BBS3 recruits assembled BBSome to ciliary membrane.'},
            {'term': 'BBS5 (PH-domain subunit)',
             'definition': 'Pleckstrin-homology (PH) domain BBSome subunit; binds phosphoinositides (PtdIns(3)P, PtdIns(4,5)P2) at the ciliary membrane, anchoring BBSome cargo to the lipid bilayer. BBS5 is recruited to the BBSome via BBS4 TPR1–4 contact (Gly180 interface); BBS4 LOF → BBS5 cannot integrate → lipid anchoring of BBSome cargo fails.'},
            {'term': 'BBIP10 / BBIP1 (dynein regulatory subunit)',
             'definition': 'BBSome-interacting protein 10; the smallest BBSome subunit; directly regulates cytoplasmic dynein-2 activity for retrograde IFT. BBIP10 is tethered to BBSome via BBS4 TPR5–8. BBS4 LOF → BBIP10 not integrated → retrograde IFT regulation impaired → IFT-B/dynein-2 imbalance.'},
            {'term': 'LepR Mis-trafficking (obesity mechanism)',
             'definition': 'Leptin receptor (LepR) must cycle into and out of primary cilia of hypothalamic neurons to transduce satiety signalling. BBSome loads LepR onto IFT trains for ciliary entry via BBS1-cargo recognition. BBS4 LOF → BBSome assembly incomplete → LepR stranded at ciliary base → LepR fails to signal → leptin resistance → hyperphagia → truncal obesity (BMI 29–45 in BBS4 cohort).'},
            {'term': 'Rod-Cone Dystrophy (rod-first pattern)',
             'definition': 'Photoreceptor degeneration in which rod cells (scotopic, peripheral, night vision) degenerate before cone cells (photopic, central, colour). ERG: scotopic amplitude extinguished first. All BBSome-null syndromes (BBS1–BBS4, etc.) share rod-first pattern — distinguishing from Alström Syndrome (cone-rod, CONE-first). BBS4 retinal degeneration is indistinguishable clinically from BBS1/BBS2/BBS3; gene panel is mandatory.'},
            {'term': 'Tri-allelic BBS',
             'definition': 'Rare (oligogenic) mode of BBS where two pathogenic alleles in one BBS gene plus one pathogenic allele in a second BBS gene are required for full phenotype expression. BBS4 tri-allelic rate ~4% (lower than BBS2 ~9%) reflecting BBS4\'s peripheral BBSome position — one null BBS4 allele has less dominant-negative effect on the two-hit BBS2–BBS7 core dimer.'},
        ],
    }
