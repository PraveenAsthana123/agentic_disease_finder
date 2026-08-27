"""
CEP290 Joubert Syndrome Type 5 (JBTS5) — Centrosomal Protein 290 / NPHP6 / MKS4 / BBS14 / LCA10
===================================================================================================
Primary Gene : CEP290 (*610142) — 12q21.32; ~2479 aa; Centrosomal Protein 290 kDa
               CEP290 (Centrosomal Protein 290 kDa) is the MOST COMMON Joubert syndrome gene,
               accounting for ~10–15% of all JBTS cases. CEP290 localises to the TRANSITION ZONE
               (TZ) Y-links and basal body / centrosome. CEP290 forms a complex with NPHP5 (IQCB1),
               PCM1, and other TZ proteins to regulate ciliary gate integrity and protein
               trafficking. Loss of CEP290 → TZ Y-link assembly fails → GPCRs, Smoothened, and
               other ciliary cargo CANNOT enter the cilium → impaired Hedgehog, somatostatin,
               and cAMP signalling → Molar Tooth Sign (MTS), cerebellar vermis hypoplasia.
               CEP290 is uniquely PLEIOTROPIC: allele class and cellular context govern which
               spectrum disease manifests (LCA10, JBTS5, NPHP6, MKS4, BBS14).
Disease OMIM : #610188 — Joubert Syndrome 5 (JBTS5)
               Also: #610188 = JBTS5 · #615454 = NPHP6 · #611134 = MKS4 · #209900 = BBS14
               Also: #613142 = Leber Congenital Amaurosis 10 (LCA10) — retinal only, IVS26 allele
Chromosome   : 12q21.32
Inheritance  : Autosomal Recessive — biallelic LOF; ALLELE CLASS GOVERNS DISEASE TIER
Prevalence   : ~10–15% of all Joubert syndrome cases (most common JBTS gene worldwide)
               ~1/500,000–700,000 worldwide (JBTS5 phenotype)
               CEP290 NPHP6: ~1/1,000,000 · MKS4 lethal: ~1/150,000 births (founder)

⚠ KEY DIAGNOSTIC PEARL — IVS26 INTRONIC ALLELE (LCA10):
The most common CEP290 disease allele in European LCA10 is c.2991+1655A>G (IVS26), a deep
INTRONIC mutation that creates a cryptic splice site in intron 26 of CEP290 and inserts a
128-bp pseudo-exon. This allele is INVISIBLE to standard exon-focused panels (WES, gene panel);
detectable ONLY by RNA studies, cDNA analysis, genome sequencing, or targeted IVS26 assay.
In JBTS5, c.2991+1655A>G typically appears as one allele of a compound heterozygote with a
truncating second allele. Allele class GOVERNS severity: IVS26/IVS26 → LCA10 (retinal only),
IVS26/truncating → JBTS5, biallelic truncating → NPHP6/MKS4 (severe).

Protein Structure — CEP290 / Nephrocystin-6 (2479 aa; TZ Y-link / centrosomal scaffold)
------------------------------------------------------------------------------------------
Domain 1: N-terminal CC repeats (aa 1–200)        — Self-association; centrosome targeting
Domain 2: SMC-like hinge (aa 200–600)             — Ciliary gate scaffolding; PCM1 contact
Domain 3: Central CC1 (aa 600–1100)               — KIF3A kinesin binding; IFT entry platform
Domain 4: Central CC2 (aa 1100–1700)              — NPHP5 (IQCB1) binding; retinal critical
           cc.2991+1655 IVS26 inserts pseudo-exon → frameshift in reading frame; truncates
           mRNA → hypomorphic; retinal photoreceptors most sensitive (LCA10 if biallelic IVS26)
Domain 5: C-terminal CC (aa 1700–2479)            — Calmodulin binding; TZ Y-link anchoring
           Trp1891Ter and Arg1933Ter — the two most common truncating JBTS5/NPHP6 null alleles

Key pathogenic variant classes (CEP290):
1. c.2991+1655A>G (IVS26, deep intronic): creates pseudo-exon; European enriched;
   IVS26/truncating → JBTS5; IVS26/IVS26 → LCA10 (retina only); INVISIBLE to WES/panel
2. p.Trp1891Ter (c.5673G>A): truncating null; JBTS5/NPHP6; pan-ethnic
3. p.Arg1933Ter (c.5797C>T): truncating null; MKS4 if biallelic; JBTS5 if compound het
4. p.Glu2089Lys (c.6265G>A): missense; MENA; hypomorphic → JBTS5 milder
5. p.Cys89Tyr (c.266G>A): N-term missense; South Asian; PCM1 contact disrupted
6. p.Leu2195Pro (c.6584T>C): C-term missense; European; calmodulin-binding domain

JBTS5 Allele-Phenotype Tier Rule:
  Biallelic NULL (two truncating): MKS4 (Meckel-Gruber lethal spectrum)
  One NULL + one HYPOMORPHIC (IVS26 or missense): JBTS5 (MTS + retinal + renal variable)
  Biallelic IVS26 / biallelic mild missense: LCA10 (retinal only, NO MTS)
  Hypomorphic + truncating (NPHP5 domain): NPHP6 (renal only → ESRD)
"""

import random
import math

SEED = 417
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
    ('European',                       0.45),   # IVS26 enriched
    ('Middle Eastern / North African', 0.25),   # consanguineous; various alleles
    ('South Asian',                    0.15),   # Cys89Tyr enriched
    ('Ashkenazi Jewish',               0.05),
    ('East Asian',                     0.05),
    ('Other / Unknown',                0.05),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het: c.2991+1655A>G (IVS26) + truncating null (JBTS5 tier)',
    'Compound het: truncating null + missense hypomorphic',
    'Compound het: missense + missense (both hypomorphic)',
    'Compound het: IVS26 + C-term missense (Leu2195Pro)',
    'Compound het: splice site + truncating null',
]
allele_fracs = [0.38, 0.28, 0.18, 0.10, 0.06]
allele_pool = []
for ac, frac in zip(allele_classes, allele_fracs):
    allele_pool.extend([ac] * round(frac * N))
while len(allele_pool) < N:
    allele_pool.append(allele_classes[0])
rng.shuffle(allele_pool)

age_dx_pool = (
    [rng.randint(0, 1) for _ in range(18)] +    # neonatal/infantile (MTS on MRI)
    [rng.randint(2, 8) for _ in range(14)] +    # early childhood
    [rng.randint(9, 20) for _ in range(8)]      # later (retinal/renal-first)
)
rng.shuffle(age_dx_pool)

for i in range(N):
    eth = eth_pool[i]
    allele = allele_pool[i]
    age_dx = age_dx_pool[i]
    has_ivs26 = 'IVS26' in allele
    # CEP290 has HIGH retinal involvement (~57%)
    has_retinal = rng.random() < 0.57        # rod-cone dystrophy (highest JBTS type)
    has_renal = rng.random() < 0.35          # NPHP-type tubulointerstitial
    has_oma = rng.random() < 0.55            # oculomotor apraxia (intermediate)
    has_ataxia = rng.random() < 0.90         # cerebellar ataxia (very common)
    has_hypotonia = rng.random() < 0.88      # neonatal hypotonia
    has_breathing = rng.random() < 0.60      # episodic apnea / breathing dysregulation
    has_id = rng.random() < 0.73             # intellectual disability
    has_liver = rng.random() < 0.08          # hepatic fibrosis (rare, MKS4 overlap)
    has_poly = rng.random() < 0.08           # polydactyly (rare, BBS14 overlap)
    ivs26_missed = has_ivs26 and (i < 7)     # 17% — IVS26 invisible to standard panels
    patients.append({
        'id': f'JBTS5-{i+1:03d}',
        'ethnicity': eth,
        'allele_class': allele,
        'age_diagnosis': age_dx,
        'mts': True,                          # all JBTS5 have MTS (allele class selects for it)
        'retinal_dystrophy': has_retinal,
        'renal_nphp': has_renal,
        'oculomotor_apraxia': has_oma,
        'cerebellar_ataxia': has_ataxia,
        'neonatal_hypotonia': has_hypotonia,
        'breathing_dysregulation': has_breathing,
        'intellectual_disability': has_id,
        'hepatic_fibrosis': has_liver,
        'polydactyly': has_poly,
        'ivs26_allele': has_ivs26,
        'ivs26_missed_panel': ivs26_missed,
    })


# ── API response builders ─────────────────────────────────────────────────────
def get_overview():
    retinal_n  = sum(1 for p in patients if p['retinal_dystrophy'])
    renal_n    = sum(1 for p in patients if p['renal_nphp'])
    oma_n      = sum(1 for p in patients if p['oculomotor_apraxia'])
    ataxia_n   = sum(1 for p in patients if p['cerebellar_ataxia'])
    hypotonia_n = sum(1 for p in patients if p['neonatal_hypotonia'])
    breathe_n  = sum(1 for p in patients if p['breathing_dysregulation'])
    id_n       = sum(1 for p in patients if p['intellectual_disability'])
    liver_n    = sum(1 for p in patients if p['hepatic_fibrosis'])
    poly_n     = sum(1 for p in patients if p['polydactyly'])
    ivs26_n    = sum(1 for p in patients if p['ivs26_allele'])
    missed_n   = sum(1 for p in patients if p['ivs26_missed_panel'])
    age_mean   = round(sum(p['age_diagnosis'] for p in patients) / N, 1)

    kpis = [
        {'label': 'Cohort size', 'value': str(N), 'color': '#1a237e'},
        {'label': 'Molar Tooth Sign', 'value': '100%', 'color': '#1a237e'},
        {'label': 'Retinal Dystrophy', 'value': f'{_pct(retinal_n)}%', 'color': '#b71c1c'},
        {'label': 'Renal (NPHP-type)', 'value': f'{_pct(renal_n)}%', 'color': '#006064'},
        {'label': 'Oculomotor Apraxia', 'value': f'{_pct(oma_n)}%', 'color': '#e65100'},
        {'label': 'Cerebellar Ataxia', 'value': f'{_pct(ataxia_n)}%', 'color': '#37474f'},
        {'label': 'Neonatal Hypotonia', 'value': f'{_pct(hypotonia_n)}%', 'color': '#4a148c'},
        {'label': 'Breathing Dysreg.', 'value': f'{_pct(breathe_n)}%', 'color': '#880e4f'},
        {'label': 'Intellectual Disab.', 'value': f'{_pct(id_n)}%', 'color': '#1b5e20'},
        {'label': 'IVS26 allele carrier', 'value': f'{_pct(ivs26_n)}%', 'color': '#f57f17'},
        {'label': 'IVS26 panel missed', 'value': f'{_pct(missed_n)}%', 'color': '#c62828'},
        {'label': 'Mean age Dx (yr)', 'value': str(age_mean), 'color': '#37474f'},
    ]

    return {
        'kpis': kpis,
        'hallmark': (
            'Molar Tooth Sign (MTS) — elongated Superior Cerebellar Peduncles (SCP) with '
            'cerebellar vermis hypoplasia on axial brain MRI — is PATHOGNOMONIC for JBTS5. '
            'JBTS5 (CEP290) is the MOST COMMON Joubert syndrome gene, accounting for ~10–15% '
            'of all JBTS cases worldwide. CEP290 is uniquely pleiotropic: the same gene causes '
            'LCA10 (retinal only), JBTS5 (MTS + retinal + renal), NPHP6 (renal), MKS4 '
            '(lethal), or BBS14 depending solely on ALLELE CLASS and cellular context.'
        ),
        'critical_diagnostic_pearl': (
            'The IVS26 allele (c.2991+1655A>G) — the most common CEP290 disease allele in '
            'European LCA10/JBTS5 (~38% of JBTS5 European alleles) — is INVISIBLE TO WES AND '
            'STANDARD GENE PANELS. It is a deep intronic mutation creating a 128-bp pseudo-exon '
            'by aberrant splicing. Detection requires: RNA/cDNA studies, genome sequencing '
            '(WGS), or a targeted IVS26-specific assay. When CEP290 is clinically suspected '
            'but panel/WES is negative, request IVS26 targeted testing or WGS. Failure to '
            'detect = diagnostic miss in ~38% of JBTS5 European patients.'
        ),
        'allele_phenotype_rule': (
            'ALLELE CLASS → DISEASE TIER: '
            '(1) Biallelic NULL → MKS4 (Meckel-Gruber, lethal). '
            '(2) One NULL + one HYPOMORPHIC (IVS26 or missense) → JBTS5 (MTS + variable retinal/renal). '
            '(3) Biallelic IVS26 or biallelic mild missense → LCA10 (retinal ONLY, NO MTS). '
            '(4) Hypomorphic + truncating in NPHP5-binding domain → NPHP6 (renal → ESRD). '
            '(5) Hypomorphic pair → BBS14 (obesity + retinal, rarest tier). '
            'THIS RULE MUST BE APPLIED BEFORE ASSIGNING A DIAGNOSIS — same two CEP290 alleles '
            'in siblings can cause different tiers if allele severity differs.'
        ),
        'prevalence': '~10–15% of all Joubert syndrome cases (most common JBTS gene worldwide); ~1/500,000–700,000 (JBTS5 phenotype)',
        'first_description': 'Sayer et al. 2006 (Am J Hum Genet 79:882–892) — CEP290 / NPHP6 identified as cause of Senior-Loken syndrome; Valente et al. 2006 (Nat Genet 38:623–625) — CEP290 / JBTS5 linkage; den Hollander et al. 2006 (Nat Genet 38:875–877) — IVS26 allele in LCA10',
    }


def get_breakdown():
    # Allele class distribution
    allele_dist = {}
    for p in patients:
        a = p['allele_class']
        allele_dist[a] = allele_dist.get(a, 0) + 1
    allele_rows = sorted(allele_dist.items(), key=lambda x: -x[1])

    # Ethnicity distribution
    eth_dist = {}
    for p in patients:
        e = p['ethnicity']
        eth_dist[e] = eth_dist.get(e, 0) + 1
    eth_rows = sorted(eth_dist.items(), key=lambda x: -x[1])

    # Age at diagnosis distribution
    age_bins = {'0–1 yr': 0, '2–4 yr': 0, '5–10 yr': 0, '11–20 yr': 0}
    for p in patients:
        a = p['age_diagnosis']
        if a <= 1:
            age_bins['0–1 yr'] += 1
        elif a <= 4:
            age_bins['2–4 yr'] += 1
        elif a <= 10:
            age_bins['5–10 yr'] += 1
        else:
            age_bins['11–20 yr'] += 1

    # Feature co-occurrence matrix
    features = ['retinal_dystrophy', 'renal_nphp', 'oculomotor_apraxia',
                'cerebellar_ataxia', 'neonatal_hypotonia', 'breathing_dysregulation',
                'intellectual_disability']
    feature_labels = ['Retinal Dystr.', 'Renal NPHP', 'Ocul. Apraxia',
                      'Cereb. Ataxia', 'Neo. Hypotonia', 'Breathing Dysreg.', 'Intell. Disab.']
    feat_counts = {fl: sum(1 for p in patients if p[f]) for f, fl in zip(features, feature_labels)}

    # IVS26 diagnostic miss impact
    ivs26_pts = [p for p in patients if p['ivs26_allele']]
    ivs26_missed = [p for p in patients if p['ivs26_missed_panel']]

    return {
        'allele_distribution': [
            {'allele_class': ac, 'count': cnt, 'pct': _pct(cnt)}
            for ac, cnt in allele_rows
        ],
        'ethnicity_distribution': [
            {'ethnicity': e, 'count': cnt, 'pct': _pct(cnt)}
            for e, cnt in eth_rows
        ],
        'age_at_diagnosis': [
            {'bin': b, 'count': c, 'pct': _pct(c)}
            for b, c in age_bins.items()
        ],
        'feature_prevalence': [
            {'feature': fl, 'count': feat_counts[fl], 'pct': _pct(feat_counts[fl])}
            for fl in feature_labels
        ],
        'ivs26_summary': {
            'ivs26_carriers': len(ivs26_pts),
            'ivs26_pct': _pct(len(ivs26_pts)),
            'panel_missed': len(ivs26_missed),
            'panel_missed_pct': _pct(len(ivs26_missed)),
            'message': (
                f'{len(ivs26_missed)} of {N} patients ({_pct(len(ivs26_missed))}%) had their '
                f'IVS26 allele MISSED by standard gene panel / WES — requiring RNA/WGS rescue. '
                'IVS26 carriers need targeted deep-intronic or genome-sequencing confirmation.'
            ),
        },
        'tier_comparison': [
            {'tier': 'JBTS5 (MTS + retinal + renal)', 'allele_class': 'One NULL + one hypomorphic (IVS26 or missense)', 'retinal': '~57%', 'renal': '~35%', 'mts': 'YES'},
            {'tier': 'LCA10 (retinal only)', 'allele_class': 'Biallelic IVS26 / biallelic mild missense', 'retinal': '~100%', 'renal': 'Rare', 'mts': 'NO'},
            {'tier': 'NPHP6 (renal → ESRD)', 'allele_class': 'Hypomorphic + truncating (NPHP5 domain)', 'retinal': '~30%', 'renal': '~100%', 'mts': 'Variable'},
            {'tier': 'MKS4 (lethal)', 'allele_class': 'Biallelic NULL (two truncating)', 'retinal': 'N/A', 'renal': 'N/A', 'mts': 'N/A (lethal)'},
            {'tier': 'BBS14 (obesity + retinal)', 'allele_class': 'Biallelic hypomorphic pair', 'retinal': '~80%', 'renal': '~20%', 'mts': 'Rare'},
        ],
        'sample_patients': [
            {
                'id': p['id'],
                'ethnicity': p['ethnicity'],
                'allele_class': p['allele_class'][:60] + ('…' if len(p['allele_class']) > 60 else ''),
                'age_dx': p['age_diagnosis'],
                'retinal': 'Yes' if p['retinal_dystrophy'] else 'No',
                'renal': 'Yes' if p['renal_nphp'] else 'No',
                'oma': 'Yes' if p['oculomotor_apraxia'] else 'No',
                'ivs26': 'Yes' if p['ivs26_allele'] else 'No',
                'ivs26_missed': 'YES ⚠' if p['ivs26_missed_panel'] else 'No',
            }
            for p in patients[:20]
        ],
    }


def get_definitions():
    return {
        'gene': {
            'symbol': 'CEP290',
            'full_name': 'Centrosomal Protein 290 kDa',
            'aliases': ['NPHP6', 'BBS14', 'MKS4', 'JBTS5-gene', 'LCA10-gene', 'POC3', '3H11Ag'],
            'omim_gene': '*610142',
            'chromosome': '12q21.32',
            'protein_size': '2479 aa (~290 kDa)',
            'expression': 'Ubiquitous; highest in retinal photoreceptors, renal tubular cells, neurons',
            'subcellular_location': 'Transition zone (TZ) Y-links; centrosome / basal body; axoneme base',
        },
        'diseases': [
            {
                'omim': '#610188',
                'name': 'Joubert Syndrome 5 (JBTS5)',
                'allele_class': 'One NULL + one HYPOMORPHIC (IVS26 or missense)',
                'key_features': 'Molar Tooth Sign, retinal dystrophy (~57%), renal (~35%), cerebellar ataxia',
            },
            {
                'omim': '#613142',
                'name': 'Leber Congenital Amaurosis 10 (LCA10)',
                'allele_class': 'Biallelic IVS26 / biallelic mild missense',
                'key_features': 'Retinal dystrophy only (severe early-onset); NO MTS; candidate for ASO therapy',
            },
            {
                'omim': '#615454',
                'name': 'Nephronophthisis 6 (NPHP6)',
                'allele_class': 'Hypomorphic + truncating (NPHP5-binding C-terminal)',
                'key_features': 'Tubulointerstitial nephritis → ESRD; retinal ~30%; variable MTS',
            },
            {
                'omim': '#611134',
                'name': 'Meckel-Gruber / MKS4 (lethal)',
                'allele_class': 'Biallelic NULL (two truncating/frameshift)',
                'key_features': 'Lethal ciliopathy: occipital encephalocele, polycystic kidneys, polydactyly',
            },
            {
                'omim': '#209900',
                'name': 'Bardet-Biedl Syndrome 14 (BBS14)',
                'allele_class': 'Biallelic hypomorphic pair',
                'key_features': 'Obesity, retinal dystrophy, polydactyly; BBSome partially functional',
            },
        ],
        'key_variants': [
            {
                'variant': 'c.2991+1655A>G (IVS26)',
                'protein': 'pseudo-exon insertion / frameshift (hypomorphic)',
                'location': 'Intron 26 — deep intronic cryptic splice site',
                'population': 'European (enriched)',
                'disease_tier': 'LCA10 (biallelic) / JBTS5 (compound het + truncating)',
                'note': 'INVISIBLE to WES/panel — requires RNA, WGS, or targeted IVS26 assay',
            },
            {
                'variant': 'p.Trp1891Ter (c.5673G>A)',
                'protein': 'Truncating null (C-terminal CC)',
                'location': 'Exon 41, C-terminal coiled-coil / calmodulin-binding region',
                'population': 'Pan-ethnic',
                'disease_tier': 'JBTS5 (compound het) / NPHP6',
                'note': 'Common truncating allele; disrupts NPHP5 interaction surface',
            },
            {
                'variant': 'p.Arg1933Ter (c.5797C>T)',
                'protein': 'Truncating null (C-terminal)',
                'location': 'Exon 42, C-terminal domain',
                'population': 'Pan-ethnic',
                'disease_tier': 'MKS4 (biallelic) / JBTS5 (compound het)',
                'note': 'Most severe null allele — biallelic → MKS4 lethal',
            },
            {
                'variant': 'p.Glu2089Lys (c.6265G>A)',
                'protein': 'Missense — calmodulin-binding domain disruption',
                'location': 'Exon 44',
                'population': 'MENA / North African (enriched)',
                'disease_tier': 'JBTS5 (hypomorphic)',
                'note': 'Partial loss of calmodulin binding; milder JBTS5 phenotype',
            },
            {
                'variant': 'p.Cys89Tyr (c.266G>A)',
                'protein': 'Missense — N-terminal PCM1/centrosome contact',
                'location': 'Exon 3, N-terminal SMC domain',
                'population': 'South Asian',
                'disease_tier': 'JBTS5 (hypomorphic)',
                'note': 'Disrupts PCM1 binding; partial centrosome localisation loss',
            },
            {
                'variant': 'p.Leu2195Pro (c.6584T>C)',
                'protein': 'Missense — C-terminal TZ-anchoring surface',
                'location': 'Exon 45, C-terminal CC',
                'population': 'European',
                'disease_tier': 'JBTS5 (milder, compound het with IVS26)',
                'note': 'Hypomorphic; destabilises C-terminal TZ anchoring; compatible with MTS',
            },
        ],
        'molecular_mechanism': (
            'CEP290 localises to the ciliary TRANSITION ZONE (TZ) Y-links and the centrosomal '
            'PCM (pericentriolar material). CEP290 forms a complex with NPHP5 (IQCB1) that is '
            'CRITICAL for retinal photoreceptor connecting cilium integrity (explains high LCA/retinal '
            'burden). CEP290 also interacts with PCM1, RAB8A, KIF3A, and other IFT regulators. '
            'Loss of CEP290 → TZ Y-link scaffold fails → ciliary gate becomes permissive (leaky) → '
            'GPCRs, Smoothened, and other cargo CANNOT specifically enrich at cilia → impaired '
            'Hedgehog, somatostatin, and cAMP signalling → Molar Tooth Sign and cerebellar vermis '
            'hypoplasia. In photoreceptors, CEP290/NPHP5 complex loss → connecting cilium gate '
            'fails → rhodopsin and other outer-segment cargo traffick wrongly → photoreceptor death. '
            'Severity depends on residual CEP290 function (allele class).'
        ),
        'treatment': {
            'renal': 'ACEi/ARB for proteinuria; dialysis → renal transplant (CURATIVE for renal endpoint; brain and retinal NOT corrected by transplant)',
            'retinal': 'Low vision aids; retinal gene therapy clinical trials ongoing (ASO targeting IVS26 pseudo-exon for LCA10 — EDIT-101/sepofarsen Phase 2/3 — FIRST in-vivo CRISPR for eye disease); ERG monitoring',
            'neurological': 'OT/PT for ataxia; speech therapy; special education (ID management)',
            'breathing': 'NICU monitoring; caffeine for apnea; sleep study (central apnea pattern)',
            'aso_therapy': (
                'SEPOFARSEN (QR-110) — antisense oligonucleotide targeting IVS26 pseudo-exon to '
                'restore correct CEP290 splicing in retinal photoreceptors. Phase 2/3 trials for '
                'LCA10 (c.2991+1655A>G / IVS26 allele carriers). First in-vivo CRISPR therapy '
                '(EDIT-101) also targets IVS26 in photoreceptors. ONLY applicable to IVS26 '
                'carriers; biallelic IVS26 preferred for ASO monotherapy.'
            ),
        },
        'surveillance': [
            'Brain MRI (MTS verification) at diagnosis — with SCP measurement',
            'Full ophthalmology + ERG (photopic and scotopic) — rod-cone pattern',
            'Annual renal function: creatinine, eGFR, urinalysis, renal US',
            'Molecular genetics: IVS26 targeted assay if panel/WES negative',
            'Audiometry: sensorineural hearing loss rare but reported',
            'Annual developmental assessment / neuropsychological evaluation',
            'Echocardiography if congenital heart defect suspected',
        ],
        'glossary': {
            'MTS': 'Molar Tooth Sign — pathognomonic JBTS brain MRI finding: elongated SCP + vermis hypoplasia',
            'IVS26': 'Intron 26 deep intronic variant c.2991+1655A>G — hypomorphic; creates pseudo-exon; invisible to WES',
            'CEP290': 'Centrosomal Protein 290 kDa — TZ Y-link scaffold; most common JBTS gene',
            'NPHP5': 'Nephrocystin-5 (IQCB1) — CEP290 binding partner; critical for retinal photoreceptor CC integrity',
            'TZ': 'Transition Zone — ciliary gate between basal body and axoneme; regulated by CEP290',
            'LCA10': 'Leber Congenital Amaurosis 10 — CEP290 retinal-only disease (biallelic mild/IVS26 alleles)',
            'NPHP6': 'Nephronophthisis 6 — CEP290 renal-dominant disease (C-terminal truncating + hypomorphic)',
            'MKS4': 'Meckel-Gruber Syndrome type 4 — CEP290 biallelic null — lethal ciliopathy',
            'BBS14': 'Bardet-Biedl Syndrome 14 — CEP290 biallelic hypomorphic — BBSome partial dysfunction',
            'SCP': 'Superior Cerebellar Peduncle — elongated in MTS on brain MRI',
            'ASO': 'Antisense Oligonucleotide — sepofarsen/QR-110 targets IVS26 pseudo-exon for LCA10',
            'CRISPR': 'EDIT-101 — in-vivo CRISPR editing of IVS26 in photoreceptors (clinical trial)',
            'CC': 'Coiled-Coil domain — CEP290 has multiple CC domains for protein-protein interactions',
            'PCM1': 'Pericentriolar Material 1 — CEP290 binding partner at centrosome/basal body',
            'ERG': 'Electroretinogram — rod-then-cone pattern extinguished in CEP290 retinopathy',
        },
    }
