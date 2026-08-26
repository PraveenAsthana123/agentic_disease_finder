"""
BBS5 Bardet-Biedl Syndrome Type 5 — BBS5 (Pleckstrin Homology Domain Lipid Sensor) Dashboard
==============================================================================================
Primary Gene : BBS5 (*603650) — Bardet-Biedl Syndrome 5 Protein;
               520 aa; N-terminal coiled-coil domain (aa 1–90) + central PH
               (Pleckstrin Homology) domain (aa 91–300) + C-terminal α-helical
               domain (aa 301–520); 2q31.1.
               BBS5 is a PERIPHERAL STRUCTURAL SUBUNIT of the BBSome octameric
               complex — the sole LIPID SENSOR that anchors the assembled BBSome
               to phosphoinositide lipids at the ciliary membrane base:
                 N-terminal coiled-coil (aa 1–90): homodimerisation interface and
                   BBS4-TPR1–4 contact platform; cc-domain residues Leu138 and
                   Val142 pack into BBS4-TPR groove — Leu138Pro → coiled-coil
                   destabilised → BBS4–BBS5 module integration disrupted;
                 PH domain (aa 91–300): canonical split β-barrel; binds
                   PtdIns(3)P and PtdIns(4,5)P2 at the ciliary transition zone
                   inner leaflet; Ala242 sits in the D4/D5 lipid-binding loop
                   (β5–β6) — Ala242Val → steric block → PtdIns contact lost →
                   BBSome cannot anchor to ciliary membrane lipids; Arg217 bridges
                   BBS4-Gly180 contact surface — Arg217Trp → BBS4–BBS5 interface
                   broken;
                 C-terminal α-helical domain (aa 301–520): contacts BBS9
                   (platform bridge subunit); Glu303Ter truncates entire C-term →
                   no BBS9 contact → BBSome platform cannot assemble.
               BBSome membrane anchoring mechanism (BBS5-dependent step):
                 After BBSome assembles at PCM1 pericentriolar satellites (BBS4
                 tethering step) → ARL6/BBS3-GTP recruits assembled BBSome to
                 ciliary membrane → BBS5 PH domain inserts into PtdIns-enriched
                 inner leaflet of the ciliary transition zone membrane → BBSome
                 is anchored for IFT-B docking and cargo loading.
               BBS5 LOF → BBSome assembles normally at PCM1 satellites (BBS4
               intact) → ARL6/BBS3-GTP delivers BBSome to ciliary membrane zone
               → BBSome CANNOT anchor to PtdIns lipids (PH domain absent) →
               BBSome detaches / fails to engage IFT-B complex → GPCR cargo
               (LepR, SSTR3, MCHR1) cannot enter cilia → multi-system BBS.
               KEY DISTINCTION from BBS4: BBS4 LOF prevents satellite tethering
               (BBSome assembly incomplete); BBS5 LOF allows complete assembly
               AND ARL6 delivery but FAILS at membrane lipid anchoring — one
               assembly step later. BBS5-null cells show intact PCM1-colocalised
               BBS4 satellite pattern (confirming BBS4 is present and functional).
               KEY DISTINCTION from BBS3 (ARL6): ARL6 is EXTERNAL to the BBSome
               (a GTPase recruiter); BBS5 is an INTRINSIC BBSome subunit; in
               BBS3-null cells the BBSome is fully assembled but ARL6 cannot
               recruit it; in BBS5-null cells ARL6-GTP recruitment occurs but
               the BBSome cannot anchor via PH–PtdIns contact — mechanistically
               distinct failure points at the membrane docking stage.
               KEY DISTINCTION from BBS1: BBS1 is the cargo recognition β-propeller
               (recognises GPCR cytoplasmic tail sorting signals); BBS5 is the
               lipid anchor — different functional classes within the same complex.
               KEY DISTINCTION from BBS2/BBS7 core: BBS2–BBS7 dimer is the
               structural backbone (step 0 of assembly); BBS5 is a peripheral
               module (post-satellite step); BBS5 failure does not collapse the
               BBS2–BBS7 core dimer.

Disease OMIM : #209900 — Bardet-Biedl Syndrome (BBS5 allelic disorder)
               All BBS types share OMIM #209900; molecular subtype defined by gene.
               Allele spectrum:
               — Severe: biallelic null (null/null) → no BBS5 protein → complete
                 PH-domain membrane anchoring failure; full multi-system BBS
               — Moderate: compound het (PH missense + truncating) → partial PH
                 domain lipid binding; moderate multi-system BBS
               — Mild: hypomorphic PH missense (reduced PtdIns affinity) →
                 attenuated BBSome membrane anchoring; attenuated multi-system BBS;
                 possible retinal-only or oligosyndromic presentations reported
               UNIQUE FEATURES vs BBS1, BBS2, BBS3, BBS4:
               (1) INTRINSIC BBSome LIPID SENSOR: BBS5 is the only BBSome subunit
                   that directly contacts phosphoinositide lipids (PtdIns(3)P,
                   PtdIns(4,5)P2) via its PH domain at the ciliary inner leaflet —
                   the intrinsic membrane anchor; no other BBS gene encodes a PH
                   domain within the BBSome
               (2) LATE MEMBRANE ANCHORING FAILURE: unlike BBS4 (satellite
                   tethering, earlier) and BBS3 (ARL6 recruitment, external GTPase),
                   BBS5 failure occurs AFTER complete BBSome assembly AND ARL6-GTP
                   recruitment — the failure is specifically at PtdIns lipid contact
               (3) BBS4 SATELLITE PATTERN INTACT IN BBS5-NULL CELLS: because BBS4
                   is functional, anti-BBS4 IF in BBS5-null patient fibroblasts
                   shows NORMAL PCM1-colocalised BBS4 puncta — distinguishing BBS5
                   LOF from BBS4 LOF (where BBS4 IF is absent)
               (4) FREQUENCY ~2–3% of molecularly confirmed BBS: rarer than BBS1
                   (~25%), BBS2 (~10–20%), BBS10 (~20–25%); under 50 families
                   worldwide (2026) — the rarest of the five characterised here
               (5) NO DOMINANT ETHNIC FOUNDER: allele spectrum pan-ethnic; Ala242Val
                   has slight European enrichment; no single-population founder
                   comparable to BBS1-M390R (European) or BBS2-R631P (Bedouin)
Chromosome   : 2q31.1
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous);
               tri-allelic BBS in ~3% (peripheral BBSome position, similar to BBS4)
Prevalence   : ~1:100,000–160,000 worldwide (BBS overall); BBS5 accounts for
               ~2–3% of molecularly confirmed BBS cases; under 50 families (2026)
"""

import random

SEED = 341
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
    ('European',                                     0.35),  # Ala242Val slight enrichment
    ('Middle Eastern / North African',               0.25),  # Arg217Trp
    ('South Asian',                                  0.20),  # Leu138Pro
    ('East Asian / Other',                           0.12),  # Ile184Thr
    ('Latin American',                               0.08),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('European')
rng.shuffle(eth_pool)

# Allele classes
allele_classes = ['null/null', 'PH missense / truncating', 'compound het (2 missense)']
allele_weights = [0.35, 0.45, 0.20]

def _pick_allele():
    r = rng.random()
    cum = 0.0
    for cls, w in zip(allele_classes, allele_weights):
        cum += w
        if r < cum:
            return cls
    return allele_classes[-1]

# Clinical parameters (BBS5 literature values, ~2–3% of BBS)
# Rod-cone: 100%; postaxial poly: ~60%; obesity: ~75%; cognitive: ~42%;
# renal: ~38%; hypogonadism: ~58%; anosmia: ~55%; CHD: ~5%; tri-allelic: ~3%
for i in range(N):
    eth    = eth_pool[i]
    allele = _pick_allele()
    poly   = rng.random() < 0.60
    obese  = rng.random() < 0.75
    cogn   = rng.random() < 0.42
    renal  = rng.random() < 0.38
    hypo   = rng.random() < 0.58
    anos   = rng.random() < 0.55
    chd    = rng.random() < 0.05

    # retinal: always rod-cone; stage at diagnosis
    ret_stage = rng.choices(
        ['Early (ERG reduced, minimal VA loss)', 'Moderate (VF constriction, ERG extinguished)',
         'Advanced (tunnel vision, VA ≤ 0.1)', 'End-stage (NLP / hand motion)'],
        weights=[0.22, 0.36, 0.28, 0.14]
    )[0]

    # renal sub-type
    renal_type = None
    if renal:
        renal_type = rng.choices(
            ['Renal cysts', 'Calyceal clubbing', 'Horseshoe kidney', 'CKD Gr 3+'],
            weights=[0.42, 0.28, 0.18, 0.12]
        )[0]

    # polydactyly sub-type
    poly_type = None
    if poly:
        poly_type = rng.choices(
            ['Postaxial hands only', 'Postaxial hands+feet', 'Postaxial feet only'],
            weights=[0.40, 0.42, 0.18]
        )[0]

    # triallelic (BBS5 + second BBS gene variant)
    triallelic = rng.random() < 0.03

    # top recurrent variant (educational)
    variant = rng.choices(
        ['p.Ala242Val (c.725C>T)', 'p.Arg217Trp (c.649C>T)',
         'p.Leu138Pro (c.413T>C)', 'p.Glu303Ter (c.907G>T)', 'p.Ile184Thr (c.551T>C)'],
        weights=[0.30, 0.25, 0.20, 0.15, 0.10]
    )[0]

    # age at diagnosis (yr)
    dx_age = rng.choices(
        ['0–4', '5–11', '12–17', '18+'],
        weights=[0.13, 0.33, 0.32, 0.22]
    )[0]

    # initial misdiagnosis
    misdiag = rng.choices(
        ['Retinitis Pigmentosa (isolated)', 'Alström Syndrome', 'Laurence-Moon Syndrome',
         'Non-syndromic obesity', 'Cohen Syndrome', 'Correct BBS (first referral)'],
        weights=[0.30, 0.10, 0.10, 0.14, 0.05, 0.31]
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
    poly_n    = sum(1 for p in patients if p['polydactyly'])
    obese_n   = sum(1 for p in patients if p['obesity'])
    cogn_n    = sum(1 for p in patients if p['cognitive'])
    renal_n   = sum(1 for p in patients if p['renal'])
    hypo_n    = sum(1 for p in patients if p['hypogonadism'])
    anos_n    = sum(1 for p in patients if p['anosmia'])
    chd_n     = sum(1 for p in patients if p['chd'])
    tri_n     = sum(1 for p in patients if p['triallelic'])
    esrd_n    = sum(1 for p in patients
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
        {'gene': 'BBS1',          'role': 'Cargo recognition β-propeller; ARL6-GTP docking platform; recognises GPCR cytoplasmic tails', 'function_class': 'Cargo Adaptor',      'omim': '209901', 'frequency_pct': '25'},
        {'gene': 'BBS2',          'role': 'Core structural WD40/coiled-coil; BBS2–BBS7 structural spine dimer (first assembly step)',    'function_class': 'Core Scaffold',      'omim': '606151', 'frequency_pct': '10–20'},
        {'gene': 'BBS3 / ARL6',   'role': 'Small GTPase (external); GTP-loaded recruits assembled BBSome to ciliary membrane',          'function_class': 'GTPase Recruiter',   'omim': '608845', 'frequency_pct': '3–5'},
        {'gene': 'BBS4',          'role': 'TPR scaffold; tethers assembling BBSome to PCM1 pericentriolar satellites (earlier defect)',  'function_class': 'Satellite Anchor',   'omim': '600374', 'frequency_pct': '2–4'},
        {'gene': 'BBS5 ← THIS',   'role': 'PH domain lipid sensor; anchors assembled BBSome to PtdIns(3)P/PtdIns(4,5)P2 at ciliary inner leaflet; BBS4 co-module partner', 'function_class': 'Lipid Sensor / Membrane Anchor', 'omim': '603650', 'frequency_pct': '2–3'},
        {'gene': 'BBS7',          'role': 'WD40/coiled-coil; BBS2–BBS7 core dimer partner (structural backbone)',                       'function_class': 'Core Scaffold',      'omim': '607590', 'frequency_pct': '2–3'},
        {'gene': 'BBS8 / TTC8',   'role': 'TPR repeats; IFT-B docking; transitions BBSome to IFT-B trains at ciliary entry',            'function_class': 'IFT Adaptor',        'omim': '608132', 'frequency_pct': '2–3'},
        {'gene': 'BBS9 / PTHB1',  'role': 'Platform bridge; connects BBS2–BBS7 core to BBS4–BBS5 peripheral module',                   'function_class': 'Platform Bridge',    'omim': '607968', 'frequency_pct': '2–3'},
        {'gene': 'BBS10',         'role': 'CCT/TRiC-like chaperonin; assists BBSome subunit folding and assembly',                      'function_class': 'Chaperonin',         'omim': '610148', 'frequency_pct': '20–25'},
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
            "BBS5 protein contains a central PH (Pleckstrin Homology) domain (aa 91–300) "
            "that directly binds phosphoinositide lipids — PtdIns(3)P and PtdIns(4,5)P2 — "
            "enriched at the inner leaflet of the ciliary transition zone membrane. "
            "BBS5 is the sole intrinsic lipid sensor within the BBSome octameric complex. "
            "After the BBSome assembles at PCM1 pericentriolar satellites (BBS4-dependent step) "
            "and ARL6/BBS3-GTP recruits the intact BBSome to the ciliary membrane zone, "
            "BBS5 PH domain inserts into the PtdIns-enriched inner leaflet, anchoring the BBSome "
            "for IFT-B docking and GPCR cargo loading. "
            "BBS5 LOF → BBSome assembles normally (BBS4 intact, PCM1 satellite pattern preserved) "
            "→ ARL6/BBS3-GTP delivers BBSome to ciliary membrane zone → BBSome CANNOT anchor to "
            "PtdIns lipids → BBSome detaches before IFT-B engagement → GPCR cargo (LepR, SSTR3, "
            "MCHR1) fails to enter cilia → multi-system BBS. "
            "N-terminal coiled-coil (aa 1–90) also contacts BBS4 TPR1–4 (BBS4-Gly180 surface), "
            "forming the BBS4–BBS5 peripheral module that integrates with BBS9 (platform bridge)."
        ),
        'key_distinction': (
            "BBS5 is the ONLY intrinsic BBSome subunit with a PH domain — the sole "
            "phosphoinositide lipid sensor of the complex. "
            "vs BBS4: BBS4 LOF prevents satellite tethering (BBSome assembly incomplete, "
            "PCM1-BBS4 IF absent); BBS5 LOF allows complete BBSome assembly at PCM1 "
            "satellites (BBS4 IF NORMAL in BBS5-null cells) — failure occurs one step later "
            "at membrane lipid anchoring. Anti-BBS4 IF in BBS5-null fibroblasts: NORMAL "
            "(diagnostic to distinguish BBS5 from BBS4 LOF). "
            "vs BBS3/ARL6: ARL6 is EXTERNAL to BBSome (GTPase recruiter); BBS5 is "
            "INTRINSIC to BBSome (structural subunit); in BBS3-null cells, BBSome is "
            "assembled and cannot reach the membrane; in BBS5-null cells, BBSome reaches "
            "the membrane zone but fails the intrinsic PH–PtdIns anchoring step. "
            "vs BBS1: BBS1 is the cargo recognition subunit (GPCR sorting signals); BBS5 "
            "is the membrane lipid anchor — different functional classes within the same complex. "
            "vs BBS2/BBS7 core: BBS2–BBS7 dimer is the structural backbone (step 0); "
            "BBS5 is a peripheral module (post-satellite step); BBS5-null does NOT collapse "
            "the BBS2–BBS7 core dimer — partial BBSome intermediates may be detectable."
        ),
        'bbs_pathway_comparison': bbs_pathway_comparison,
        'retinal_distribution': retinal_dist,
        'age_distribution': {
            'dx_0_4yr':   age_bins['0–4'],
            'dx_5_11yr':  age_bins['5–11'],
            'dx_12_17yr': age_bins['12–17'],
            'dx_18plus':  age_bins['18+'],
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
            'Gene':              'BBS5',
            'Aliases':           'BBS5 protein; Bardet-Biedl syndrome 5 protein',
            'OMIM Gene':         '*603650',
            'Protein length':    '520 aa',
            'Domains':           'N-terminal coiled-coil (aa 1–90); PH domain (aa 91–300); C-terminal α-helical domain (aa 301–520)',
            'Chromosome':        '2q31.1',
            'mRNA':              'NM_152384.2',
            'Protein':           'NP_689597.1',
            'Function':          'PH domain binds PtdIns(3)P / PtdIns(4,5)P2 at ciliary transition zone membrane; intrinsic BBSome lipid anchor; BBS4 co-module partner',
            'BBSome role':       'Peripheral structural subunit (lipid sensor / membrane anchor); forms BBS4–BBS5–BBIP10 peripheral module; contacts BBS4 (TPR1–4) and BBS9 (platform)',
            'Interactors':       'BBS4 (TPR1–4 via cc-domain), BBS9 (platform bridge via C-term), BBIP10, BBS1 (indirect via BBS9 bridge), ARL6/BBS3 (indirect)',
            'Expression':        'Ubiquitous; high in RPE, photoreceptors, renal tubules, olfactory epithelium, hypothalamic neurons',
            'Mouse model':       'Bbs5−/− (Li 2004): rod-cone degeneration, obesity, anosmia, male infertility; BBS4 IF pattern intact confirming satellite-tethering step unaffected',
        },
        'disease_card': {
            'Disease':           'Bardet-Biedl Syndrome Type 5',
            'OMIM Disease':      '#209900',
            'Locus':             'BBS5 (2q31.1)',
            'Inheritance':       'Autosomal Recessive; biallelic LOF',
            'Prevalence':        '~1:100,000–160,000 (BBS overall); BBS5 ~2–3% of BBS; under 50 families worldwide (2026)',
            'Molecular class':   'BBSome intrinsic PH-domain lipid sensor (peripheral structural subunit)',
            'Key features':      'Rod-cone (rod first), postaxial polydactyly, truncal obesity, renal anomalies, hypogonadism, anosmia, cognitive LD',
            'Retinal pattern':   'Rod-cone dystrophy (nyctalopia first); ERG scotopic extinguished before photopic; same as BBS1/BBS2/BBS3/BBS4',
            'Obesity mechanism': 'LepR mis-trafficking (satiety signalling failure); hyperphagia; BMI 28–44; lipid anchoring defect of BBSome at hypothalamic neuronal cilia',
            'Tri-allelic':       '~3% (similar to BBS4 ~4%; peripheral BBSome position)',
            'Unique feature':    'Anti-BBS4 IF in BBS5-null patient fibroblasts shows NORMAL PCM1 puncta — distinguishes BBS5 from BBS4 LOF; only BBSome subunit with intrinsic PH-domain lipid binding',
            'Gene panel':        'Full BBS panel (20–24 genes) + MLPA/CNV mandatory; BBS5 clinically indistinguishable from BBS1–BBS4 by phenotype alone',
            'Frequency':         '~2–3% of molecularly confirmed BBS; under 50 families worldwide (2026)',
        },
        'key_variants': [
            {'variant': 'p.Ala242Val (c.725C>T)',  'domain': 'PH domain β5–β6 loop (D4/D5 lipid-binding)',  'consequence': 'Steric block at PtdIns contact → PH domain cannot bind ciliary membrane PtdIns(3)P/PtdIns(4,5)P2; BBSome detaches at membrane anchoring step', 'ethnicity': 'European (slight enrichment)'},
            {'variant': 'p.Arg217Trp (c.649C>T)',  'domain': 'PH domain / BBS4-Gly180 contact surface',     'consequence': 'BBS4–BBS5 interface disrupted → BBS5 cannot integrate into BBS4 peripheral module; PH domain correctly folded but module assembly fails', 'ethnicity': 'Middle Eastern / North African'},
            {'variant': 'p.Leu138Pro (c.413T>C)',  'domain': 'N-terminal coiled-coil (BBS4-TPR contact)',    'consequence': 'Coiled-coil destabilised → BBS4 TPR1–4 contact lost → BBS4–BBS5 peripheral module fails to form; early misfolding phenotype', 'ethnicity': 'South Asian'},
            {'variant': 'p.Glu303Ter (c.907G>T)',  'domain': 'C-terminal α-helical (BBS9 contact domain)',   'consequence': 'Truncating null → no BBS9 (platform bridge) contact → BBSome platform cannot assemble around BBS5; complete BBS5 LOF', 'ethnicity': 'Pan-ethnic; European / East Asian'},
            {'variant': 'p.Ile184Thr (c.551T>C)',  'domain': 'PH domain β3–β4 (lipid-binding pocket core)', 'consequence': 'Hydrophobic core of PH β-barrel disrupted → lipid-binding pocket collapses → membrane anchoring lost; reduced thermal stability', 'ethnicity': 'European; East Asian'},
        ],
        'diagnostic_workup': [
            '1. Clinical: confirm ≥4 primary BBS features (rod-cone dystrophy, polydactyly, obesity, renal anomaly, hypogonadism, anosmia/cognitive LD)',
            '2. Ophthalmology: ERG (scotopic extinguished before photopic = rod-first pattern); fundus photography; OCT for outer retinal layer loss; visual field (tubular constriction)',
            '3. Genetic: full BBS gene panel (20–24 genes, NGS + MLPA for CNV); BBS5 detected only by panel, not targeted single-gene testing',
            '4. BBS5 vs BBS4 cell assay: anti-BBS4 IF in patient fibroblasts — NORMAL PCM1-colocalised BBS4 puncta = BBS5 LOF (not BBS4); ABSENT puncta = BBS4 LOF',
            '5. Renal ultrasound: cysts, horseshoe kidney, calyceal clubbing; renal function (eGFR, urine protein/albumin)',
            '6. Metabolic: fasting glucose, HbA1c, lipids; insulin resistance screen (HOMA-IR); BMI trajectory',
            '7. Endocrine: LH, FSH, testosterone/oestradiol, AMH; growth hormone if short stature',
            '8. Olfactory testing: UPSIT or Sniffin Sticks (anosmia 55%)',
            '9. Cardiac echo: exclude structural CHD (5%); no infantile DCM (distinguishes from Alström)',
            '10. Neurodevelopmental assessment: IQ, learning disability spectrum (42%)',
        ],
        'treatment_summary': [
            '1. Retinal: no approved gene therapy for BBS5 (2026); low vision aids; tinted lenses; yearly ophthalmology; avoid vitamin A supplementation (no evidence in BBS ciliopathy)',
            '2. Obesity: GLP-1 RA (semaglutide/liraglutide) — primary pharmacotherapy for BBS5 LepR mis-trafficking; orlistat adjunct; bariatric surgery if BMI >40 with co-morbidities',
            '3. Renal: annual eGFR + urine albumin; ACE inhibitor if CKD/proteinuria; renal transplant for ESRD (curative, no recurrence — BBS5 LOF is cell-autonomous)',
            '4. Polydactyly: surgical excision in infancy (neonatal/early paediatric); orthopaedic referral for functional impairment',
            '5. Hypogonadism: testosterone replacement (males, Tanner stage assessment); oestrogen/progesterone for females; subfertility counselling; ART as appropriate',
            '6. Cognitive: early intervention (speech, OT, educational support); IEP/504 plan; annual neurodevelopmental review',
            '7. Anosmia: safety alert (gas, smoke, food hygiene); olfactory training (limited evidence in ciliopathy anosmia)',
            '8. Endocrine: annual HbA1c + lipids from age 10; metformin + GLP-1 RA for T2D-pattern insulin resistance',
            '9. Genetic counselling: AR inheritance; 25% recurrence risk; cascade testing for siblings; phenotypic overlap with BBS1–BBS4 discussed',
            '10. BBS5 VUS management: anti-BBS4 IF fibroblast assay; PH domain lipid-binding biochemical assay (PtdIns pull-down); cilia morphology (scanning EM) for VUS classification',
        ],
        'ddx_table': [
            {'disease': 'BBS1 (BBS1 gene)',        'key_difference': 'BBS1 = cargo recognition β-propeller; M390R European founder (~70% BBS1 alleles); ~25% of BBS; clinically identical — gene panel mandatory to distinguish'},
            {'disease': 'BBS2 (BBS2 gene)',         'key_difference': 'BBS2 = BBS2–BBS7 core dimer (step 0 of assembly); R631P Bedouin founder; BBSome misassembles at step 0 vs BBS5 membrane anchoring failure (later)'},
            {'disease': 'BBS3 / ARL6',             'key_difference': 'ARL6 is EXTERNAL GTPase recruiter; BBSome fully assembles and ARL6 delivers it to membrane; BBS5 LOF causes PH–PtdIns anchoring failure at the same membrane zone — external vs intrinsic failure'},
            {'disease': 'BBS4 (TPR scaffold)',      'key_difference': 'BBS4 = satellite tethering (earlier defect, assembly step); anti-BBS4 IF shows ABSENT puncta in BBS4-null but NORMAL puncta in BBS5-null — key cellular discriminator'},
            {'disease': 'BBS10 (chaperonin)',       'key_difference': 'BBS10 = CCT/TRiC chaperonin; ~20–25% of BBS; no structural BBSome subunit; chaperonin-class (fold-assist) distinct from BBS5 PH-domain lipid sensor'},
            {'disease': 'Alström Syndrome (ALMS1)', 'key_difference': 'No polydactyly; cone-rod (NOT rod-first); infantile DCM; SNHL; ALMS1 is a large centrosomal/ciliary scaffold (no PH domain); BBS5 has polydactyly + rod-first + no DCM + no SNHL'},
            {'disease': 'Laurence-Moon Syndrome',   'key_difference': 'Spastic paraplegia + progressive neurological deterioration; NO polydactyly; overlap in obesity + retinal; distinct gene (PNPLA6 / PHARC pathway)'},
            {'disease': 'McKusick-Kaufman (MKKS)',  'key_difference': 'Hydrometrocolpos (females) + CHD + polydactyly; no retinal dystrophy; BBS6/MKKS is a chaperonin-like protein; no PH domain'},
        ],
        'mechanism_glossary': [
            {'term': 'Pleckstrin Homology (PH) Domain',
             'definition': 'A ~100-aa structural module forming a split β-barrel (7 β-strands + 1 C-terminal α-helix) with a positively charged cleft that specifically binds phosphoinositide lipids. BBS5 PH domain (aa 91–300) binds PtdIns(3)P and PtdIns(4,5)P2 enriched at the inner leaflet of the ciliary transition zone membrane, anchoring the assembled BBSome. The D4/D5 loop (β5–β6) harbours Ala242, whose substitution to Val sterically blocks phosphate head-group contacts.'},
            {'term': 'Phosphoinositides (PtdIns) at the Ciliary Membrane',
             'definition': 'Phosphatidylinositol phospholipids enriched at specific membrane compartments: PtdIns(4,5)P2 marks the periciliary / transition zone inner leaflet; PtdIns(3)P marks early endosomes and ciliary base. BBS5 PH domain binds both, anchoring the BBSome at the transition zone membrane inner leaflet — a step that precedes IFT-B docking and GPCR cargo loading. Loss of BBS5 PH–PtdIns contact → BBSome detaches before cargo loading.'},
            {'term': 'BBSome Octameric Complex',
             'definition': 'A coat-like complex of 8 subunits (BBS1, BBS2, BBS4, BBS5, BBS7, BBS8/TTC8, BBS9/PTHB1, BBIP10) that acts as a cargo adaptor for GPCR trafficking into and out of primary cilia. BBSome biogenesis is stepwise: (0) BBS2–BBS7 core dimer; (1) BBS9 platform bridges core to periphery; (2) BBS4-dependent satellite tethering integrates BBS5–BBIP10 peripheral module; (3) ARL6/BBS3-GTP recruits BBSome to ciliary membrane; (4) BBS5 PH domain anchors BBSome to PtdIns; (5) IFT-B loads BBSome for axonemal transport. BBS5 is required at step 4.'},
            {'term': 'BBS4–BBS5 Peripheral Module',
             'definition': 'A functional sub-complex within the BBSome where BBS4 TPR repeats 1–4 (via Gly180 contact surface) bind BBS5 N-terminal coiled-coil (Leu138 region) and BBS5 PH domain (Arg217 surface). Together with BBIP10 (dynein regulatory subunit anchored by BBS4 TPR5–8), these three form the BBS4–BBS5–BBIP10 peripheral module on the BBSome face. BBS9 (platform bridge) connects this module to the BBS2–BBS7 core. Loss of either BBS4 or BBS5 disrupts this module; BBS5-null retains BBS4 satellite localisation, while BBS4-null loses BBS5 integration.'},
            {'term': 'LepR Mis-trafficking (obesity mechanism)',
             'definition': 'Leptin receptor (LepR) must cycle into and out of primary cilia of hypothalamic POMC/AgRP neurons to transduce satiety signalling. BBSome loads LepR onto IFT trains (via BBS1 cargo recognition). BBS5 LOF → BBSome cannot anchor to ciliary membrane → LepR fails IFT loading → LepR stranded at ciliary base → leptin resistance → hyperphagia → truncal obesity (BMI 28–44 in BBS5 cohort). GLP-1 RA (semaglutide) is preferred pharmacotherapy.'},
            {'term': 'Rod-Cone Dystrophy (rod-first pattern)',
             'definition': 'Photoreceptor degeneration in which rod cells (scotopic, peripheral, night vision) degenerate before cone cells (photopic, central, colour). ERG: scotopic amplitude extinguished first. All BBSome-null syndromes (BBS1–BBS5, etc.) share rod-first pattern — distinguishing from Alström Syndrome (cone-rod, CONE-first). BBS5 retinal degeneration is clinically indistinguishable from BBS1/BBS2/BBS3/BBS4; gene panel is mandatory for subtype diagnosis.'},
            {'term': 'Tri-allelic BBS',
             'definition': 'Rare (oligogenic) mode of BBS where two pathogenic alleles in one BBS gene plus one pathogenic allele in a second BBS gene are required for full phenotype expression. BBS5 tri-allelic rate ~3% (similar to BBS4 ~4%; both peripheral subunits with less dominant-negative impact on the BBS2–BBS7 core dimer than central subunit mutations). Tri-allelic BBS5 cases may involve BBS1 or BBS9 as the second-hit gene.'},
        ],
    }
