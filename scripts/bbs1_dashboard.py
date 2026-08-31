"""
BBS1 Bardet-Biedl Syndrome Type 1 — Autosomal Recessive / BBS1 / BBSome Cargo-Recognition Subunit / Met390Arg European Founder / Most Common BBS Gene
=======================================================================================================================================================
Primary Gene : BBS1 (*209901) — Bardet-Biedl Syndrome 1 Protein; ~593 aa; 11q13.2;
               N-terminal beta-propeller platform (aa 1–300) + Central Met390 scaffold
               region / BBS9-bridge interface (aa 301–450) + C-terminal ARM cargo-
               recognition domain / Rab8-GTP interaction (aa 451–593).
               BBS1 is the CARGO-RECOGNITION subunit of the BBSome octameric complex:
                 N-terminal beta-propeller (aa 1–300): forms the PLATFORM layer of
                   BBSome; contacts BBS4 (TPR) and BBS5 (PH domain) on the platform
                   face; structurally indispensable for BBSome lattice formation.
                 Central scaffold / BBS9-bridge region (aa 301–450): contacts BBS9
                   (PTHB1) C-terminal helix — BBS9 is the structural bridge from the
                   BBS2–BBS7 core dimer to BBS1 (cargo arm); Met390 (aa 390) lies at
                   the BBS9 interface; Met390Arg disrupts BBS9 docking → BBSome cannot
                   present cargo-recognition face to the IFT-B machinery.
                 C-terminal ARM-repeat cargo-recognition domain (aa 451–593): direct
                   interaction with Rab8-GTP (Rab8a-GDP→GTP switch recruits BBSome to
                   cilia base); binds LHFPL4, Arl6/BBS3 (BBSome–IFT-B adaptor);
                   V458G / Arg521Ter alleles ablate Rab8 interaction.
               BBS1 LOF pathway:
               BBS1 biallelic LOF → BBSome assembles structurally (BBS2–BBS7 scaffold
               still forms) BUT cargo-recognition arm fails → BBSome binds IFT-B but
               cannot recognise GPCR/cargo tails → LepR, SSTR3, MCHR1, ORP3, olfactory
               receptor proteins mis-trafficked → multi-system ciliopathy (BBS syndrome).
               This contrasts with BBS2 LOF (structural core) where BBSome fails to
               assemble at step 0. BBS1 LOF = functional BBSome without cargo docking.

⚠ MET390ARG (c.1169T>G) — EUROPEAN FOUNDER — MOST COMMON BBS MUTATION WORLDWIDE:
   Met390Arg at the BBS9-interface / central scaffold region is the most prevalent
   BBS mutation globally. In Northern European populations: carrier frequency ~1:70–100
   in molecularly confirmed BBS families; Met390Arg accounts for ~70–80% of European
   BBS1 alleles. Homozygous Met390Arg → moderate BBS1: retinal dystrophy (rod-FIRST),
   obesity, polydactyly (~55%), renal structural anomalies (~45%), learning disability
   (~60%). Partial residual BBS9 docking retained → somewhat milder multi-system
   expressivity vs biallelic null (early truncating). Most important BBS allele for
   population-level carrier screening in European ancestry.

⚠ BBS1 IS THE MOST COMMON BBS GENE (~20–25% OF ALL MOLECULARLY CONFIRMED BBS):
   Among >21 BBS loci, BBS1 (and BBS10) are the two most common, each accounting
   for ~20–25% of all molecularly confirmed BBS cases in European populations.
   BBS10 is enriched in North African populations; BBS1 is the European-predominant gene.
   Consequence: every BBS panel must include BBS1 in the primary tier. Single-gene
   BBS1 testing is clinically appropriate only in populations with high M390R frequency
   (Northern European); elsewhere, full BBS multigene panel is mandatory.

⚠ NO MKS TIER — BBS1 LOF IS NOT LETHAL:
   BBS1 biallelic null does NOT cause Meckel-Gruber Syndrome (MKS). The TZ structural
   gate (B9-complex, tectonic module) is intact in BBS1 LOF — BBSome is a ciliary
   coat/transport complex, not a TZ gate structural element. All BBS1 patients are
   liveborn. MKS requires collapse of TZ structural scaffolding (B9D1, B9D2, MKS1,
   RPGRIP1L, TMEM67 — entirely different protein complexes). BBS1 LOF = ciliary cargo
   mis-trafficking WITHOUT gate collapse.

⚠ ROD-CONE RETINAL DYSTROPHY — ROD-FIRST PROGRESSION (MOST PENETRANT FEATURE):
   Retinal rod-cone dystrophy is the earliest and most penetrant BBS1 feature (~93%).
   Rod photoreceptor outer-segment disc proteins (rhodopsin, ROM1) mis-trafficked →
   rod degeneration begins before cone. Presentation: night blindness (typically
   childhood), peripheral ring scotoma, progressive to legal blindness by 3rd–4th decade.
   ERG: scotopic amplitude extinguished >> photopic (distinguishes from Alstrom
   cone-rod). Annual ERG + OCT from age 3. Gene therapy / retinal neuroprotection
   trials active (CRISPR-BBS1 preclinical 2023–2024). Mandatory AAO genetics referral.

⚠ OBESITY — LEPR MIS-TRAFFICKING / HYPOTHALAMIC LEPTIN RESISTANCE:
   LepR (Leptin Receptor) cannot enter hypothalamic neuronal cilia without BBSome
   cargo recognition. Without ciliary LepR, satiety signalling from adipose fails →
   persistent hyperphagia from infancy → truncal obesity (BMI 30–50 by adolescence
   in 90%). First-line: GLP-1 RA (semaglutide / liraglutide) + metformin. Bariatric
   surgery for adult BMI ≥ 40 (sleeve gastrectomy preferred — early satiety restored
   even with mis-trafficked LepR via peripheral GLP-1/PYY axes).

Disease OMIM : #209900 — Bardet-Biedl Syndrome (BBS gene-class shared number)
               Gene OMIM: *209901 (BBS1); all BBS types share #209900; molecular
               subtype defined by causative gene (BBS1 here).
Chromosome   : 11q13.2
Inheritance  : Autosomal Recessive — biallelic LOF; Met390Arg homozygous (European
               consanguineous/founder); compound het (Met390Arg + truncating/splice);
               tri-allelic BBS in ~5% (BBS1 × 2 + BBS modifier at BBS4/BBS5/BBS9 locus)
Prevalence   : BBS overall ~1:100,000–160,000; BBS1 ~20–25% of molecularly confirmed
               BBS cases. European-ancestry predominant.
"""

import random
import math

SEED = 461
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
    ('European (non-consanguineous)',              0.40),  # Met390Arg European founder
    ('Middle Eastern / MENA (consanguineous)',     0.25),
    ('North African (consanguineous)',             0.15),
    ('South Asian (consanguineous)',               0.12),
    ('East Asian / Other',                         0.08),
]

allele_classes = [
    ('Homozygous Met390Arg (European founder)',            0.30),
    ('Met390Arg + frameshift/truncating',                 0.28),
    ('Compound het — two missense (non-Met390Arg)',        0.18),
    ('Met390Arg + splice variant',                        0.12),
    ('Biallelic truncating (null/null)',                   0.07),
    ('Tri-allelic BBS (BBS1 × 2 + modifier)',             0.05),
]

variants = [
    'Met390Arg/Met390Arg',                  # European founder homozygous; moderate
    'Met390Arg/c.1169+1G>A',               # European founder + splice; moderate-severe
    'Met390Arg/c.1052_1053insA',            # founder + frameshift; severe
    'Met390Arg/Arg99Ter',                   # founder + early truncating; severe
    'Met390Arg/Glu549Ter',                  # founder + C-terminal truncating; moderate-severe
    'Met390Arg/Leu295Pro',                  # founder + central scaffold; moderate-severe
    'Ile246Val/Met390Arg',                  # beta-propeller mild + founder; moderate
    'Ala428Thr/Met390Arg',                  # North African cluster + founder; moderate
    'Gly436Arg/Ile246Val',                  # two missense; moderate
    'Tyr222Cys/Ala428Thr',                  # South Asian compound; moderate-severe
    'c.47_50del/Met390Arg',                 # N-terminal frameshift + founder; severe
    'Arg393Gln/Met390Arg',                  # near-Met390 compound; moderate-severe
    'Val458Gly/Met390Arg',                  # ARM cargo-domain + founder; moderate-severe
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(3, 48)
    sex = _rng_p.choice(['M', 'F'])

    rod_cone     = _rng_p.random() < 0.93   # rod-cone dystrophy; most penetrant feature
    obesity      = _rng_p.random() < 0.91   # LepR mis-trafficking; hyperphagia
    poly         = _rng_p.random() < 0.65   # post-axial polydactyly; hands/feet
    renal        = _rng_p.random() < 0.50   # structural anomalies (cysts, calyceal clubbing)
    cognitive    = _rng_p.random() < 0.63   # learning disability; neuronal cilia
    hypogonadism = _rng_p.random() < 0.77   # males: cryptorchidism; females: irregular menses
    anosmia      = _rng_p.random() < 0.62   # olfactory cilia BBSome absent
    chd          = _rng_p.random() < 0.07   # congenital heart disease; rare in BBS1
    hepatic      = _rng_p.random() < 0.06   # liver fibrosis; rare BBS1 feature
    esrd         = _rng_p.random() < 0.10   # ESRD when renal structural + obesity-related CKD
    triallelic   = ac == 'Tri-allelic BBS (BBS1 × 2 + modifier)'
    misdiagnosed = _rng_p.random() < 0.22   # late/incorrect Dx (Alstrom, PCOS, nonsyndromic RP)

    # Retinal stage (rod-first progression)
    if rod_cone:
        if age < 10:
            ret_stage = 'Scotopic ERG reduced — early rod loss'
        elif age < 20:
            ret_stage = 'Night blindness + ring scotoma'
        elif age < 35:
            ret_stage = 'Advanced field loss — cone involvement'
        else:
            ret_stage = 'Legal blindness / end-stage'
    else:
        ret_stage = 'No retinal disease (rare non-penetrant)'

    # Age at diagnosis
    if poly and age < 5:
        dx_route = 'Neonatal polydactyly + obesity'
    elif rod_cone and age < 12:
        dx_route = 'Childhood retinal screening'
    elif obesity and cognitive and age < 20:
        dx_route = 'Obesity + cognitive workup'
    else:
        dx_route = 'Adult multi-system review'

    patients.append({
        'id':           f'BBS1-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'rod_cone':     rod_cone,
        'ret_stage':    ret_stage,
        'obesity':      obesity,
        'poly':         poly,
        'renal':        renal,
        'cognitive':    cognitive,
        'hypogonadism': hypogonadism,
        'anosmia':      anosmia,
        'chd':          chd,
        'hepatic':      hepatic,
        'esrd':         esrd,
        'triallelic':   triallelic,
        'misdiagnosed': misdiagnosed,
        'dx_route':     dx_route,
    })

# ── aggregate counts ─────────────────────────────────────────────────────────
n_rod_cone     = sum(1 for p in patients if p['rod_cone'])
n_obesity      = sum(1 for p in patients if p['obesity'])
n_poly         = sum(1 for p in patients if p['poly'])
n_renal        = sum(1 for p in patients if p['renal'])
n_cognitive    = sum(1 for p in patients if p['cognitive'])
n_hypogonadism = sum(1 for p in patients if p['hypogonadism'])
n_anosmia      = sum(1 for p in patients if p['anosmia'])
n_chd          = sum(1 for p in patients if p['chd'])
n_hepatic      = sum(1 for p in patients if p['hepatic'])
n_esrd         = sum(1 for p in patients if p['esrd'])
n_triallelic   = sum(1 for p in patients if p['triallelic'])
n_misdiagnosed = sum(1 for p in patients if p['misdiagnosed'])

_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

_ac_counts = {}
for p in patients:
    _ac_counts[p['allele_class']] = _ac_counts.get(p['allele_class'], 0) + 1

_ret_stage_counts = {}
for p in patients:
    if p['rod_cone']:
        _ret_stage_counts[p['ret_stage']] = _ret_stage_counts.get(p['ret_stage'], 0) + 1

_dx_route_counts = {}
for p in patients:
    _dx_route_counts[p['dx_route']] = _dx_route_counts.get(p['dx_route'], 0) + 1

# ── BBSome complex reference table ────────────────────────────────────────────
BBSOME_TABLE = [
    {'subunit': 'BBS1 [THIS DASHBOARD]', 'role': 'Cargo-recognition ARM / Rab8 docking / BBS9 bridge', 'omim': '*209901', 'frequency_pct': 22},
    {'subunit': 'BBS2',                  'role': 'Structural core — BBS2–BBS7 WD40 dimer spine',        'omim': '*606151', 'frequency_pct': 15},
    {'subunit': 'BBS3 (ARL6)',           'role': 'Small GTPase — recruits BBSome to cilia base',        'omim': '*608845', 'frequency_pct': 1},
    {'subunit': 'BBS4',                  'role': 'TPR domain — platform layer; centrosome anchor',      'omim': '*600374', 'frequency_pct': 2},
    {'subunit': 'BBS5',                  'role': 'PH domain — PI(3)P / membrane interaction platform',  'omim': '*603650', 'frequency_pct': 1},
    {'subunit': 'BBS7',                  'role': 'WD40 — BBS2–BBS7 core dimer (structural partner)',    'omim': '*607590', 'frequency_pct': 2},
    {'subunit': 'BBS8 (TTC8)',           'role': 'TPR domain — outer coat; BBSome IFT-B docking arm',   'omim': '*608132', 'frequency_pct': 3},
    {'subunit': 'BBS9 (PTHB1)',          'role': 'Structural bridge — BBS2–BBS7 → BBS1 cargo arm',      'omim': '*607968', 'frequency_pct': 4},
    {'subunit': 'BBS18 (BBIP1/BBIP10)', 'role': 'Assembly chaperone — stabilises BBSome scaffold',     'omim': '*613605', 'frequency_pct': 1},
]

# Retinal stage distribution
RETINAL_DIST = []
stage_order = ['Scotopic ERG reduced — early rod loss', 'Night blindness + ring scotoma',
               'Advanced field loss — cone involvement', 'Legal blindness / end-stage',
               'No retinal disease (rare non-penetrant)']
for s in stage_order:
    cnt = _ret_stage_counts.get(s, 0) + (N - n_rod_cone if s == 'No retinal disease (rare non-penetrant)' else 0)
    if cnt > 0:
        RETINAL_DIST.append({'label': s, 'n': cnt})

# Age distribution
age_buckets = {'age_0_9': 0, 'age_10_19': 0, 'age_20_34': 0, 'age_35_plus': 0}
for p in patients:
    if p['age'] < 10:    age_buckets['age_0_9'] += 1
    elif p['age'] < 20:  age_buckets['age_10_19'] += 1
    elif p['age'] < 35:  age_buckets['age_20_34'] += 1
    else:                age_buckets['age_35_plus'] += 1


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id": "bbs1",

        "kpis": {
            "total_patients":      N,
            "rod_cone_pct":        _pct(n_rod_cone),
            "obesity_pct":         _pct(n_obesity),
            "poly_pct":            _pct(n_poly),
            "renal_pct":           _pct(n_renal),
            "cognitive_pct":       _pct(n_cognitive),
            "hypogonadism_pct":    _pct(n_hypogonadism),
            "anosmia_pct":         _pct(n_anosmia),
            "chd_pct":             _pct(n_chd),
            "hepatic_pct":         _pct(n_hepatic),
            "esrd_pct":            _pct(n_esrd),
            "triallelic_pct":      _pct(n_triallelic),
            "misdiagnosed_pct":    _pct(n_misdiagnosed),
            "no_mks_tier":         True,
        },

        "alerts": {
            "met390arg_founder": (
                "MET390ARG (c.1169T>G) — MOST COMMON BBS MUTATION WORLDWIDE: Met390Arg at the BBS9-interface "
                "scaffold region (aa 390) is the European founder allele accounting for ~70–80% of European "
                "BBS1 alleles. Carrier frequency ~1:70–100 in molecularly confirmed Northern European BBS "
                "families. Homozygous → moderate BBS1. Compound with splice/truncating → moderate-severe. "
                "Met390Arg carrier screening recommended in Northern European ancestry. Full BBS multigene "
                "panel mandatory in all other populations — Met390Arg is NOT a universal BBS1 founder allele."
            ),
            "rod_cone_rod_first": (
                "ROD-CONE DYSTROPHY — ROD FIRST (MOST PENETRANT BBS1 FEATURE ~93%): Night blindness before "
                "central vision loss; ERG scotopic amplitude extinguished >> photopic. Distinguishes from "
                "Alstrom syndrome (cone-rod: daytime photophobia first) and from non-syndromic RP. Annual "
                "ERG + OCT from age 3. Mandatory AAO genetics referral. CRISPR-based BBS1 retinal gene "
                "therapy in preclinical development (2023–2024)."
            ),
            "lepr_mis_trafficking_obesity": (
                "OBESITY — LEPR MIS-TRAFFICKING / HYPOTHALAMIC LEPTIN RESISTANCE (~91%): LepR cannot "
                "enter hypothalamic cilia without BBS1 cargo-recognition. Hyperphagia from infancy; "
                "truncal BMI 30–50 by adolescence. GLP-1 RA (semaglutide) + metformin first-line. "
                "Bariatric surgery for adult BMI ≥ 40 (sleeve gastrectomy restores peripheral satiety "
                "via GLP-1/PYY even with defective ciliary LepR). Obesity monitoring mandatory from "
                "infancy — early GLP-1 RA trials ongoing in paediatric BBS."
            ),
            "bbs1_most_common_gene": (
                "BBS1 (~20–25% OF ALL BBS) — MOST COMMON BBS GENE IN EUROPEAN POPULATIONS: BBS1 and "
                "BBS10 each account for ~20–25% of molecularly confirmed BBS. BBS1 is European-predominant; "
                "BBS10 is North-African-predominant. Full BBS multigene panel (≥20 genes) mandatory; single-"
                "gene BBS1 testing only appropriate if Met390Arg probability is high (European ancestry, "
                "positive BBS phenotype). Tri-allelic BBS (BBS1 × 2 + modifier at BBS4/BBS5/BBS9) in ~5%."
            ),
        },

        "key_facts": [
            "BBS1 (~593 aa) — cargo-recognition subunit of BBSome octamer; 11q13.2; OMIM *209901",
            "Met390Arg (c.1169T>G) — most common BBS mutation worldwide; European founder; ~70–80% of European BBS1 alleles",
            "BBSome role: BBS1 LOF → BBSome assembles structurally but cargo-recognition arm (Rab8 docking) fails",
            "Rod-cone retinal dystrophy ~93% — rod-FIRST; scotopic ERG extinguished before photopic",
            "Obesity ~91% — LepR mis-trafficking / hypothalamic leptin resistance; GLP-1 RA first-line",
            "Post-axial polydactyly ~65% (hands/feet); renal structural anomalies ~50% (cysts, calyceal clubbing)",
            "Cognitive/learning disability ~63%; hypogonadism ~77%; anosmia/hyposmia ~62%",
            "No MKS tier — BBSome cargo dysfunction does NOT collapse TZ B9-complex structural gate",
            "Most common BBS gene in European populations (~20–25% of molecularly confirmed BBS)",
            "Tri-allelic BBS (BBS1 × 2 + BBS4/BBS5/BBS9 modifier) in ~5% — full panel mandatory",
        ],

        "patients": [
            {
                "id":           p['id'],
                "age":          p['age'],
                "sex":          p['sex'],
                "ethnicity":    p['ethnicity'],
                "allele_class": p['allele_class'],
                "variant":      p['variant'],
                "rod_cone":     p['rod_cone'],
                "ret_stage":    p['ret_stage'],
                "obesity":      p['obesity'],
                "poly":         p['poly'],
                "renal":        p['renal'],
                "cognitive":    p['cognitive'],
                "hypogonadism": p['hypogonadism'],
                "anosmia":      p['anosmia'],
                "chd":          p['chd'],
                "hepatic":      p['hepatic'],
                "esrd":         p['esrd'],
                "triallelic":   p['triallelic'],
                "misdiagnosed": p['misdiagnosed'],
                "dx_route":     p['dx_route'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "bbs1",

        "ethnicity_distribution": [
            {"ethnicity": eth, "count": cnt, "pct": _pct(cnt)}
            for eth, cnt in sorted(_eth_counts.items(), key=lambda x: -x[1])
        ],

        "allele_class_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "phenotype_summary": {
            "rod_cone":     {"n": n_rod_cone,     "pct": _pct(n_rod_cone)},
            "obesity":      {"n": n_obesity,      "pct": _pct(n_obesity)},
            "poly":         {"n": n_poly,         "pct": _pct(n_poly)},
            "renal":        {"n": n_renal,        "pct": _pct(n_renal)},
            "cognitive":    {"n": n_cognitive,    "pct": _pct(n_cognitive)},
            "hypogonadism": {"n": n_hypogonadism, "pct": _pct(n_hypogonadism)},
            "anosmia":      {"n": n_anosmia,      "pct": _pct(n_anosmia)},
            "chd":          {"n": n_chd,          "pct": _pct(n_chd)},
            "hepatic":      {"n": n_hepatic,      "pct": _pct(n_hepatic)},
            "esrd":         {"n": n_esrd,         "pct": _pct(n_esrd)},
            "triallelic":   {"n": n_triallelic,   "pct": _pct(n_triallelic)},
            "misdiagnosed": {"n": n_misdiagnosed, "pct": _pct(n_misdiagnosed)},
        },

        "retinal_stage_distribution": RETINAL_DIST,

        "age_distribution": [
            {"label": k.replace("_", " "), "n": v}
            for k, v in age_buckets.items()
        ],

        "dx_route_distribution": [
            {"route": route, "n": cnt}
            for route, cnt in sorted(_dx_route_counts.items(), key=lambda x: -x[1])
        ],

        "bbsome_table": BBSOME_TABLE,

        "notable_variants": [
            {
                "name":       "Met390Arg",
                "cdna":       "c.1169T>G",
                "domain":     "Central scaffold / BBS9-bridge region (aa 390) — Met390 BBS9-docking residue",
                "population": "European — founder allele; most common BBS mutation worldwide; ~70–80% of European BBS1 alleles",
                "severity":   "Moderate (hypomorphic)",
                "mechanism":  "Met-to-Arg substitution at aa 390 disrupts the BBS9 (PTHB1) C-terminal helix docking surface. Met390 is a conserved hydrophobic contact residue at the BBS1–BBS9 interface; Arg390 introduces a bulky charged sidechain incompatible with hydrophobic BBS9 pocket. BBSome assembles (BBS2–BBS7 scaffold intact) but BBS9 bridge is unstable → BBS1 cargo-recognition arm cannot be presented to IFT-B machinery. Residual partial BBS9 docking (~25–35% WT) from non-Met390 BBS9 contact residues → moderate (not severe) BBS1 phenotype. Homozygous → moderate multi-system BBS: rod-cone dystrophy, obesity, polydactyly ~55%, renal ~45%, learning disability ~60%. Northern European carrier frequency ~1:70–100 in BBS families; general population estimate ~1:150–200.",
            },
            {
                "name":       "Ile246Val",
                "cdna":       "c.736A>G",
                "domain":     "N-terminal beta-propeller platform (aa 246) — WD40-repeat scaffold face",
                "population": "Pan-ethnic — mild hypomorphic allele; enriched in compound het BBS1 families",
                "severity":   "Mild–Moderate (hypomorphic)",
                "mechanism":  "Conservative Ile-to-Val substitution in the N-terminal beta-propeller scaffold face. Val246 has a slightly reduced sidechain volume vs Ile246 at the BBS4 (TPR) docking surface. Partial loss of BBSome platform stability. As compound het with Met390Arg → moderate BBS1 (Met390Arg drives severity; Ile246Val contributes modestly). Homozygous Ile246Val → very mild BBS1 phenotype (some carriers reach adulthood without full BBS diagnosis; late retinal onset). Key actionable: Ile246Val/Ile246Val carriers may be classified as isolated non-syndromic RP — BBS1 molecular diagnosis changes surveillance protocol (obesity/renal/hypogonadism monitoring added).",
            },
            {
                "name":       "Ala428Thr",
                "cdna":       "c.1282G>A",
                "domain":     "Central scaffold / BBS9-bridge region (aa 428) — between Met390 and ARM domain",
                "population": "North African founder cluster — moderate-prevalent in Maghreb families",
                "severity":   "Moderate",
                "mechanism":  "Ala-to-Thr introduces a hydroxyl group into the hydrophobic BBS9-interface groove adjacent to Met390 (aa 390–430 hydrophobic pocket). Disrupts BBS9-docking similarly to Met390Arg but without the charged Arg effect. BBS9-bridge partially destabilised; cargo-recognition arm misdirected. Moderate BBS1: rod-cone dystrophy, obesity, polydactyly ~60%, renal ~48%. North African regional cluster — enriched in Moroccan, Algerian, Tunisian consanguineous families. Compound with Met390Arg (European/North African trans-ethnic) → moderate-severe BBS1.",
            },
            {
                "name":       "Glu549Ter",
                "cdna":       "c.1645G>T",
                "domain":     "C-terminal ARM cargo-recognition domain (aa 549) — Rab8-GTP binding region",
                "population": "Pan-ethnic — compound het with Met390Arg; moderate-severe phenotype",
                "severity":   "Moderate–Severe (truncating C-terminal)",
                "mechanism":  "Premature stop at aa 549 removes the distal ARM-repeat cargo-recognition domain (aa 549–593: Rab8-GTP interface, LHFPL4 binding, Arl6/BBS3 adaptor contact). Truncated BBS1(1–548) can still dock to BBS9 (Met390–BBS9 region intact) but cannot engage Rab8-GTP → BBSome cannot be recruited to cilia base by Rab8 switch. Functional BBSome complex forms but fails to dock at cilia transition zone base → severe cargo mis-trafficking. As compound with Met390Arg → moderate-severe: early-onset rod-cone, obesity, full multi-system involvement. Biallelic Glu549Ter → severe BBS1.",
            },
            {
                "name":       "Leu295Pro",
                "cdna":       "c.884T>C",
                "domain":     "N-terminal beta-propeller / central scaffold junction (aa 295)",
                "population": "Middle Eastern / MENA (consanguineous) — regional cluster",
                "severity":   "Moderate–Severe",
                "mechanism":  "Pro substitution at aa 295 at the beta-propeller/scaffold junction introduces rigidity destroying the WD40 blade-turn geometry. BBS4 (TPR) docking to the platform face is partially lost. BBSome platform layer misdocked → downstream BBS8 (TTC8) TPR interaction weakened → partial BBSome coat assembly with non-functional cargo arm. Homozygous → moderate-severe BBS1 with higher renal penetrance (~60%) and earlier retinal onset. Enriched in consanguineous MENA families. Compound with Met390Arg → moderate-severe BBS1 in trans-ethnic families.",
            },
            {
                "name":       "Val458Gly",
                "cdna":       "c.1373T>G",
                "domain":     "C-terminal ARM cargo-recognition domain (aa 458) — first ARM repeat",
                "population": "Pan-ethnic",
                "severity":   "Moderate–Severe",
                "mechanism":  "Val-to-Gly at the first ARM repeat (aa 458) introduces excessive flexibility in the cargo-recognition domain. ARM repeats form a right-handed superhelix; Gly at position 458 disrupts the helix-packing geometry and ARM supercoil stability. Rab8-GTP binding surface distorted → Rab8 recruitment to BBSome impaired. As compound het with Met390Arg → moderate-severe BBS1: rod-cone onset age 6–8, obesity by age 5, polydactyly with bilateral post-axial supernumerary digits. Renal ~55%; cognitive moderate.",
            },
            {
                "name":       "Arg99Ter",
                "cdna":       "c.295C>T",
                "domain":     "N-terminal beta-propeller (aa 99) — early truncating null allele",
                "population": "Pan-ethnic — biallelic null if paired with truncating; JBTS24-like severity if compound",
                "severity":   "Severe (null allele)",
                "mechanism":  "Early stop at aa 99 removes essentially all functional BBS1 protein (N-terminal 99 aa fragment has no known function). Biallelic Arg99Ter → severe BBS1: complete BBSome cargo-recognition failure; early retinal onset (<5 years); early-onset truncal obesity; bilateral polydactyly; renal ~65%; ESRD risk ~20%. As compound het with Met390Arg → severity is Met390Arg-limited (partial BBS9 docking from Met390Arg allele rescues catastrophic null). Not independently sufficient for Met390Arg-like moderate phenotype — compound severity between Met390Arg and null biallelic.",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":     "bbs1",
        "gene_full_name": "BBS1 — Bardet-Biedl Syndrome 1 Protein; BBSome Cargo-Recognition Subunit; ARM-Repeat Rab8-GTP Docking Arm; BBS9-Bridge Interface; Met390Arg European Founder; 11q13.2",
        "omim_gene":      "209901",
        "omim_disease":   "209900",
        "chromosome":     "11q13.2",
        "protein_size": (
            "~593 aa — N-terminal beta-propeller platform (aa 1–300; BBS4/BBS5 docking face; "
            "BBSome lattice scaffold); central scaffold / BBS9-bridge region (aa 301–450; "
            "Met390 BBS9-docking residue; pathogenic missense hotzone); C-terminal ARM cargo-"
            "recognition domain (aa 451–593; Rab8-GTP interface; LHFPL4 / Arl6-BBS3 adaptor)"
        ),
        "inheritance": "Autosomal recessive — biallelic damaging LOF; Met390Arg homozygous (European founder); compound het (Met390Arg + truncating/splice); tri-allelic BBS (~5%); NO MKS lethal tier",

        "bbsome_cargo_recognition_rule": (
            "BBS1 is the CARGO-RECOGNITION subunit of the BBSome octameric ciliary coat complex. "
            "Unlike BBS2 (structural core scaffold: BBS2–BBS7 dimer, BBSome fails to assemble if BBS2 lost), "
            "BBS1 LOF allows BBSome to assemble structurally (BBS2–BBS7–BBS9 scaffold intact) but the "
            "cargo-recognition ARM domain fails → BBSome cannot dock Rab8-GTP at the cilia base → "
            "GPCRs (LepR, SSTR3, MCHR1), photoreceptor membrane proteins, and olfactory receptors "
            "cannot enter cilia → multi-system BBS ciliopathy. The distinction: BBS2 LOF = assembly failure; "
            "BBS1 LOF = docking failure with intact assembly. Clinical phenotype is identical — gene panel required."
        ),

        "glossary": [
            {
                "term": "BBS1 (Bardet-Biedl Syndrome 1 protein)",
                "definition": (
                    "BBS1 (gene; protein BBS1; OMIM *209901). ~593 aa ciliary coat adaptor at 11q13.2. "
                    "The cargo-recognition subunit of the BBSome octameric complex. Three functional modules: "
                    "N-terminal beta-propeller platform (aa 1–300, BBS4/BBS5 scaffold contact), central "
                    "BBS9-bridge region (aa 301–450, Met390 BBS9-docking hotspot), and C-terminal ARM "
                    "cargo-recognition domain (aa 451–593, Rab8-GTP binding). BBS1 LOF → BBSome structurally "
                    "intact but cargo-docking arm non-functional → GPCRs mis-trafficked → BBS syndrome. "
                    "Most common BBS gene in European populations (~20–25% of confirmed BBS). Met390Arg is "
                    "the most prevalent BBS mutation worldwide. Autosomal recessive. 11q13.2 locus."
                ),
            },
            {
                "term": "BBSome octameric complex",
                "definition": (
                    "The BBSome is a coat-like protein complex of 8 subunits (BBS1, BBS2, BBS3/ARL6, BBS4, "
                    "BBS5, BBS7, BBS8/TTC8, BBS9/PTHB1) that mediates ciliary GPCR trafficking via IFT-B. "
                    "Structure: BBS2–BBS7 WD40 dimer forms the core scaffold spine; BBS9 (bridge subunit) "
                    "links BBS2–BBS7 core to BBS1 (cargo arm); BBS4 and BBS5 (platform layer) anchor BBSome "
                    "to the cilia base membrane; BBS8 (TPR) docks BBSome to IFT-B machinery. Function: "
                    "BBSome recognises cargo tail sequences (GPCR cytoplasmic tails, photoreceptor membrane "
                    "proteins) via BBS1 ARM domain + BBS7 WD40 face, then couples to IFT-B for anterograde "
                    "ciliary transport. Without functional BBSome: LepR (satiety), SSTR3/MCHR1 (neuronal), "
                    "rhodopsin (photoreceptor), olfactory receptors — all mis-trafficked → BBS multi-system. "
                    "BBS1 is the cargo-recognition subunit; BBS2 is the structural scaffold. Clinically "
                    "indistinguishable phenotypes require multigene panel (≥20 BBS genes)."
                ),
            },
            {
                "term": "Met390Arg (c.1169T>G) — most common BBS mutation worldwide",
                "definition": (
                    "Met390Arg at aa 390 (BBS9-bridge interface, central scaffold region) is the European "
                    "founder allele for BBS1 and the most prevalent BBS mutation globally. Estimated Northern "
                    "European carrier frequency ~1:70–100 in BBS families (~1:150–200 general population). "
                    "Met390 is a conserved hydrophobic residue in the BBS1–BBS9 docking groove; Arg390 "
                    "introduces a bulky charged sidechain disrupting BBS9 hydrophobic contact → partial "
                    "BBS9-bridge instability (hypomorphic, ~25–35% residual docking). Homozygous Met390Arg → "
                    "moderate BBS1: rod-cone dystrophy, obesity, polydactyly ~55%, renal ~45%. Compound "
                    "with truncating → moderate-severe. Met390Arg carrier screening is clinically appropriate "
                    "for Northern European ancestry couples; not sufficient as sole BBS1 test outside Europe. "
                    "CRISPR correction of Met390Arg in mouse retina (Dilan et al. 2023) → proof-of-concept "
                    "for future retinal gene therapy."
                ),
            },
            {
                "term": "ARM-repeat cargo-recognition domain (BBS1 C-terminal, aa 451–593)",
                "definition": (
                    "Armadillo (ARM) repeats are ~42 aa helical repeat units stacking into a right-handed "
                    "superhelix that forms an extended binding groove for protein cargo tails. BBS1 C-terminal "
                    "ARM domain (aa 451–593): contains the Rab8-GTP binding surface — Rab8 switches from GDP "
                    "(inactive) to GTP at the periciliary membrane and recruits BBSome to the cilia base for "
                    "cargo loading. Also binds LHFPL4 (ciliary lipid scramblase cargo) and interacts with "
                    "Arl6/BBS3 GTPase adaptor (bridges BBSome to IFT-B train). Pathogenic variants in the "
                    "ARM domain (Val458Gly, Glu549Ter, Arg521Ter): disrupt Rab8 docking → BBSome cannot "
                    "initiate ciliary cargo loading. ARM domain variants are functionally more severe than "
                    "Met390Arg (BBS9-bridge) because ARM domain is the final cargo-recognition step."
                ),
            },
            {
                "term": "LepR mis-trafficking and BBS obesity",
                "definition": (
                    "Leptin receptor (LepR) must enter hypothalamic arcuate nucleus neuronal primary cilia "
                    "to transduce adipose-derived leptin signal and suppress appetite (satiety signalling). "
                    "BBSome (cargo-recognition by BBS1) is required for LepR ciliary trafficking via IFT-B "
                    "anterograde transport. BBS1 LOF → LepR cannot enter cilia → leptin cannot signal → "
                    "persistent hyperphagia from infancy → truncal obesity (BMI 30–50) in 91% of BBS1 patients. "
                    "This is NOT simple hyperinsulinism — LepR ciliary mis-trafficking is the primary mechanism. "
                    "Adipose-derived leptin is elevated (hyperleptinemia) but cannot signal. GLP-1 RA "
                    "(semaglutide, liraglutide) acts via peripheral GLP-1R on vagal afferents — bypasses "
                    "ciliary LepR → significant BMI reduction (~10–15%) in BBS. Bariatric surgery similarly "
                    "effective via PYY/GLP-1 axis independent of ciliary LepR."
                ),
            },
            {
                "term": "Rod-cone retinal dystrophy (rod-FIRST) vs cone-rod (Alstrom)",
                "definition": (
                    "BBS1 retinal dystrophy is rod-FIRST (rod-cone): rod photoreceptors degenerate before "
                    "cone photoreceptors. Rods are most numerous in the peripheral retina and most dependent "
                    "on BBSome for outer-segment disc protein trafficking (rhodopsin, ROM1). Presentation: "
                    "night blindness (nyctalopia) as first symptom, typically childhood; peripheral ring "
                    "scotoma progressing centrally; daytime visual acuity preserved until late. ERG: scotopic "
                    "(rod) amplitude extinguished >> photopic (cone) amplitude — pathognomonic rod-cone pattern. "
                    "Distinguishes from Alstrom syndrome (ALMS1 mutations) which is cone-ROD: daytime "
                    "photophobia and central vision loss FIRST, then peripheral. Also distinguishes from "
                    "LEBER congenital amaurosis (extinguished from birth). Annual ERG + OCT from age 3 is "
                    "mandatory for BBS1 surveillance."
                ),
            },
            {
                "term": "No MKS tier (BBS1)",
                "definition": (
                    "BBS1 biallelic null does NOT cause Meckel-Gruber Syndrome (MKS). The transition zone "
                    "(TZ) structural gate — B9-complex (B9D1/JBTS19, B9D2/JBTS34, MKS1/JBTS28), tectonic "
                    "module (TCTN1-3), and RPGRIP1L — remain structurally intact in BBS1 LOF. MKS requires "
                    "TZ gate collapse (occludin-barrier failure → soluble ciliary compartment lost → lethal "
                    "multi-organ malformation). BBSome is a ciliary COAT/TRANSPORT complex — it rides atop "
                    "the TZ gate using the intact gate as its substrate. BBS1 LOF = cargo mis-trafficking "
                    "with structurally intact TZ. All BBS1 patients are liveborn; no prenatal hydrocephalus, "
                    "anencephaly, or renal cyst explosion seen in MKS. No MKS prenatal counselling needed "
                    "for BBS1 families. Compare: B9D1/JBTS19 (MKS allelic), RPGRIP1L/JBTS7 (MKS allelic) — "
                    "entirely different protein complexes at the TZ structural gate."
                ),
            },
            {
                "term": "Tri-allelic BBS inheritance in BBS1",
                "definition": (
                    "Tri-allelic BBS (also: 'oligogenic BBS' or 'third-site modification') occurs when two "
                    "BBS1 alleles (recessive-tier) are combined with a heterozygous modifier allele at a "
                    "second BBS locus (BBS4, BBS5, BBS9 — all direct BBS1 scaffold/bridge partners). Estimated "
                    "~5% of BBS1 families. Phenotype: more severe multi-system involvement than biallelic "
                    "Met390Arg alone — earlier retinal onset, higher renal penetrance, more pronounced cognitive "
                    "features. Mechanistic basis: third allele destabilises the BBS9-bridge or BBS4/BBS5 "
                    "platform contact, reducing residual BBSome function below the Met390Arg-rescued threshold. "
                    "Clinical implication: family members with one BBS1 allele + one BBS4/BBS5/BBS9 allele "
                    "are not BBS-affected carriers — they require phenotyping only if a third allele is added. "
                    "Full sequencing of all BBSome subunit genes mandatory in tri-allelic-suspect families."
                ),
            },
        ],
    }
