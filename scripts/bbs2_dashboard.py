"""
BBS2 Bardet-Biedl Syndrome Type 2 — Bardet-Biedl Syndrome 2 Protein Dashboard
===============================================================================
Primary Gene : BBS2 (*606151) — Bardet-Biedl Syndrome 2 protein;
               ~721 aa; N-terminal WD40 beta-propeller (aa 1–400) +
               C-terminal coiled-coil BBSome interface domain (aa 400–721);
               16q13.
               BBS2 is a structural CORE subunit of the BBSome complex:
                 WD40 beta-propeller (aa 1–400): forms the structural
                   scaffold base; stacks directly onto BBS7 WD40 face to
                   create the BBS2–BBS7 dimer — the central spine of the
                   BBSome coat layer; loss of WD40 integrity → BBSome
                   misassembles at step 0.
                 N-terminal coiled-coil / WD40 transition (aa 350–500):
                   contacts BBS9 (PTHB1) C-terminal helix; BBS9 bridges
                   BBS2 to BBS1 (cargo recognition subunit); disruption
                   here → BBSome intact structurally but cargo recognition
                   lost (hypomorphic-like presentation).
                 C-terminal coiled-coil / BBSome interface (aa 500–721):
                   packs against BBS7 coiled-coil; also contacts BBS8 (TTC8)
                   N-terminal TPR; stabilises the outer BBSome coat; R631P
                   (Bedouin founder) disrupts this packing → BBSome assembly
                   fails.
               Loss of BBS2 → BBSome fails to assemble → IFT-mediated
               GPCR/ciliary cargo trafficking fails → GPCRs (LepR, SSTR3,
               MCHR1, ORP3) cannot enter cilia → multi-system disease:
               rod-cone degeneration, obesity, polydactyly, renal, cognitive,
               hypogonadism, anosmia.

Disease OMIM : #209900 — Bardet-Biedl Syndrome (BBS2 allelic disorder)
               All BBS types share OMIM #209900; molecular subtype is defined
               by the causative gene.
               Allele spectrum:
               — Severe: biallelic null (early truncating) → complete BBSome
                 LOF; early-onset rod-cone, severe obesity, full multi-system
               — Moderate: compound het missense (WD40 core + CC interface)
                 → partial BBSome assembly; moderate-severe multi-system
               — Mild-moderate: hypomorphic CC-interface alleles (R631P
                 heterozygous + second allele) → functional BBSome reduced
                 but not absent; clinical expressivity varies
               UNIQUE FEATURES vs BBS1:
               (1) NO M390R EUROPEAN FOUNDER: BBS2 lacks the BBS1 M390R
                   founder mutation that accounts for 70% of European BBS1
                   alleles; BBS2 allele spectrum is more heterogeneous
               (2) R631P BEDOUIN/ARABIAN FOUNDER: p.Arg631Pro (c.1892G>C)
                   is enriched 4–6× in Arabian Peninsula / Bedouin
                   populations and accounts for ~30–45% of Middle Eastern
                   BBS2 alleles
               (3) BBSome CORE MODULE: BBS2 (with BBS7 and BBS9) forms the
                   structural core scaffold of BBSome; LOF disrupts the entire
                   complex assembly unlike peripheral subunit LOF
               (4) FREQUENCY ~10–20%: second or third most common BBS gene
                   after BBS1 (~25%) and BBS10 (~25%); prevalence varies
                   by population
               (5) TRI-ALLELIC BBS: BBS2 participates in tri-allelic
                   inheritance (two BBS2 alleles + one variant in BBS1/7/9)
                   more frequently than other BBS types due to its core
                   scaffold role
Chromosome   : 16q13
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or
               homozygous consanguineous); tri-allelic BBS in ~8%
Prevalence   : ~1:100,000–160,000 worldwide (BBS overall); BBS2 accounts
               for ~10–20% of molecularly confirmed BBS cases
"""

import random
import math

SEED = 335
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
    ('Middle Eastern / Bedouin / Arabian Peninsula', 0.42),  # R631P founder enrichment
    ('European',                                     0.25),  # scattered allele spectrum
    ('North African',                                0.15),  # Maghreb, some founder overlap
    ('South Asian',                                  0.10),
    ('East Asian / Other',                           0.08),
]

allele_classes = [
    ('Compound het — missense + truncating',               0.35),
    ('Homozygous missense (consanguineous — R631P)',       0.28),  # Bedouin founder
    ('Biallelic truncating (null/null)',                   0.12),
    ('Compound het — two missense',                       0.16),
    ('Tri-allelic BBS (BBS2 × 2 + modifier)',             0.09),
]

retinal_stage = [
    ('Early — reduced scotopic ERG; minimal visual loss',     0.20),
    ('Moderate — extinguished scotopic; preserved photopic',  0.38),
    ('Advanced — rod-cone both affected; tunnel vision',      0.30),
    ('End-stage — bare light perception',                     0.12),
]

renal_types = [
    'Calyceal clubbing (corticomedullary junction)',
    'Renal cysts (cortical / corticomedullary)',
    'Horseshoe kidney',
    'Chronic kidney disease (CKD stage 2–3)',
    'ESRD — transplant listed or completed',
    'Recurrent UTI + vesicoureteral reflux',
]

misdiagnosis_labels = [
    'Non-syndromic retinitis pigmentosa (rod-cone without systemic features initially)',
    'Alstrom Syndrome (obesity + rod-cone; BBS2 lacks cardiomyopathy/deafness)',
    'Laurence-Moon Syndrome (historical term; BBS2 more common)',
    'MKKS / BBS6 (McKusick-Kaufman Syndrome overlap — tri-allelic BBS2)',
    'Usher Syndrome (early retinal without cognition; BBS2 has learning disability)',
    None,   # not misdiagnosed
]

_poly_n      = round(N * 0.70)   # 70% post-axial polydactyly (same as BBS generally)
_renal_n     = round(N * 0.50)   # 50% structural renal anomaly or CKD
_cognitive_n = round(N * 0.55)   # 55% learning disability / cognitive impairment
_obesity_n   = round(N * 0.88)   # 88% obesity / BMI ≥ 30 by adolescence
_hypo_n      = round(N * 0.70)   # 70% hypogonadism (males: cryptorchidism/small testes)
_anosmia_n   = round(N * 0.62)   # 62% anosmia / hyposmia (olfactory cilia)
_chd_n       = round(N * 0.07)   # 7% congenital heart defect (similar to BBS1)
_triallelicn = round(N * 0.09)   # 9% tri-allelic
_mis_n       = round(N * 0.45)   # 45% misdiagnosed initially

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
    ret_stage = retinal_stage[0][0]
    cumulative = 0
    for name, prob in retinal_stage:
        cumulative += prob
        if r3 < cumulative:
            ret_stage = name
            break

    poly      = i < _poly_n
    poly_type = (rng.choice(['Postaxial — bilateral hands', 'Postaxial — hands + feet',
                              'Postaxial — unilateral', 'Preaxial (rare variant)'])
                 if poly else None)
    renal_any = i < _renal_n
    renal_type = (rng.choice(renal_types) if renal_any else None)
    cognitive = i < _cognitive_n
    obese     = i < _obesity_n
    hypo      = i < _hypo_n
    anosmia   = i < _anosmia_n
    chd       = i < _chd_n
    triallelic = i < _triallelicn

    mis = rng.choice(misdiagnosis_labels) if i < _mis_n else None

    # Age at diagnosis — retinal symptoms often first sign
    dx_age = (rng.uniform(0.5, 5) if 'Early' in ret_stage
              else rng.uniform(3, 12) if 'Moderate' in ret_stage
              else rng.uniform(8, 20) if 'Advanced' in ret_stage
              else rng.uniform(15, 35))
    if dx_age < 5:
        age_dx_cat = '0–4 yr (infant/preschool)'
    elif dx_age < 12:
        age_dx_cat = '5–11 yr (childhood)'
    elif dx_age < 18:
        age_dx_cat = '12–17 yr (adolescent)'
    else:
        age_dx_cat = '18+ yr (adult)'

    bmi = (rng.uniform(30, 50) if obese else rng.uniform(22, 29))

    patients.append({
        'id':           f'P{i+1:02d}',
        'eth':          eth,
        'allele':       allele,
        'ret_stage':    ret_stage,
        'poly':         poly,
        'poly_type':    poly_type,
        'renal_any':    renal_any,
        'renal_type':   renal_type,
        'cognitive':    cognitive,
        'obese':        obese,
        'bmi':          round(bmi, 1),
        'hypo':         hypo,
        'anosmia':      anosmia,
        'chd':          chd,
        'triallelic':   triallelic,
        'misdiagnosis': mis,
        'age_dx_cat':   age_dx_cat,
    })


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    esrd_n     = sum(1 for p in patients if p['renal_any'] and p['renal_type'] and 'ESRD' in p['renal_type'])
    mis_n      = sum(1 for p in patients if p['misdiagnosis'])
    endstage_n = sum(1 for p in patients if 'End-stage' in p['ret_stage'])

    return {
        "cohort_n": N,
        "seed": SEED,
        "gene": "BBS2",
        "disease": "Bardet-Biedl Syndrome Type 2 (BBS2)",
        "mechanism": (
            "BBS2 (~721 aa) is a STRUCTURAL CORE subunit of the BBSome octameric complex. "
            "Its N-terminal WD40 beta-propeller (aa 1–400) forms the BBSome coat base, stacking "
            "directly onto BBS7 WD40 to create the BBS2–BBS7 dimer — the central scaffold spine. "
            "The C-terminal coiled-coil (aa 400–721) packs against BBS7 CC and contacts BBS8 TPR, "
            "stabilising the outer BBSome coat; BBS9 bridges BBS2 to BBS1 (cargo recognition). "
            "Loss of BBS2 → BBSome fails to assemble → IFT-mediated GPCR ciliary trafficking fails "
            "→ LepR, SSTR3, MCHR1, ORP3 cannot enter cilia → obesity (LepR), retinal degeneration "
            "(photoreceptor cilia), cognitive impairment (neuronal cilia), polydactyly (Hh signalling), "
            "renal anomalies, hypogonadism, anosmia (olfactory cilia)."
        ),
        "key_distinction": (
            "BBS2 vs BBS1: NO M390R EUROPEAN FOUNDER (BBS1 M390R accounts for 70% of European BBS1 alleles; "
            "BBS2 lacks this — allele spectrum is more heterogeneous). "
            "R631P BEDOUIN FOUNDER: p.Arg631Pro (c.1892G>C) enriched 4–6× in Arabian Peninsula / Bedouin "
            "populations; ~30–45% of Middle Eastern BBS2 alleles. "
            "CORE SCAFFOLD ROLE: BBS2–BBS7 dimer is the structural spine of BBSome; complete LOF → "
            "catastrophic BBSome misassembly (unlike peripheral subunit LOF which may leave partial complex). "
            "TRI-ALLELIC BBS: BBS2 participates in tri-allelic inheritance more than most BBS types "
            "due to its core position; two BBS2 alleles + one variant in BBS1/7/9 → complete BBSome failure. "
            "FREQUENCY ~10–20% of all BBS; 2nd–3rd most common after BBS1 (~25%) and BBS10 (~25%)."
        ),
        "kpis": {
            "polydactyly_n":      _poly_n,
            "polydactyly_pct":    _pct(_poly_n),
            "obesity_n":          _obesity_n,
            "obesity_pct":        _pct(_obesity_n),
            "cognitive_n":        _cognitive_n,
            "cognitive_pct":      _pct(_cognitive_n),
            "renal_any_n":        _renal_n,
            "renal_any_pct":      _pct(_renal_n),
            "hypogonadism_n":     _hypo_n,
            "hypogonadism_pct":   _pct(_hypo_n),
            "anosmia_n":          _anosmia_n,
            "anosmia_pct":        _pct(_anosmia_n),
            "retinal_endstage_n": endstage_n,
            "retinal_endstage_pct": _pct(endstage_n),
            "chd_n":              _chd_n,
            "chd_pct":            _pct(_chd_n),
            "triallelic_n":       _triallelicn,
            "triallelic_pct":     _pct(_triallelicn),
            "misdiagnosis_n":     mis_n,
            "misdiagnosis_pct":   _pct(mis_n),
            "esrd_n":             esrd_n,
        },
        "bbsome_subunit_table": [
            {"subunit": "BBS1",      "role": "Cargo recognition — β-propeller + ARM; binds Rab8a/Rab11; ONLY subunit with direct lipid cargo selectivity", "omim": "*209901", "frequency_pct": "~25"},
            {"subunit": "BBS2 (THIS)", "role": "Structural core scaffold — WD40 propeller base; BBS2–BBS7 dimer = BBSome spine; bridges BBS9 → BBS1", "omim": "*606151", "frequency_pct": "~10–20"},
            {"subunit": "BBS4",      "role": "Tetratricopeptide repeat (TPR); peripheral adaptor; contacts BBSome periphery; connects to BBS2 via BBS9", "omim": "*600374", "frequency_pct": "~3–5"},
            {"subunit": "BBS5",      "role": "PH-domain scaffold; lipid-sensing adaptor; stabilises BBSome on ciliary membrane; rare cause of BBS", "omim": "*603650", "frequency_pct": "~2–3"},
            {"subunit": "BBS7",      "role": "WD40 propeller — BBS2–BBS7 core dimer partner; BBS7 LOF identical clinical phenotype to BBS2", "omim": "*607590", "frequency_pct": "~3–5"},
            {"subunit": "BBS8 (TTC8)", "role": "Tetratricopeptide repeat (TPR); N-TPR contacts BBS2 C-terminal CC; outer BBSome coat stabiliser", "omim": "*608132", "frequency_pct": "~3–5"},
            {"subunit": "BBS9 (PTHB1)", "role": "Structural bridge — links BBS2 WD40/CC junction to BBS1 ARM; collapse here = cargo recognition failure", "omim": "*607968", "frequency_pct": "~4–6"},
            {"subunit": "BBS18 (BBIP10)", "role": "Small BBSome subunit; stabilises BBSome–IFT-B interface; required for BBSome–ciliary membrane docking", "omim": "*613605", "frequency_pct": "<2"},
        ],
        "retinal_distribution": [
            {"label": "Early — reduced scotopic ERG; minimal visual loss",     "n": round(N * 0.20)},
            {"label": "Moderate — extinguished scotopic; preserved photopic",  "n": round(N * 0.38)},
            {"label": "Advanced — rod-cone both affected; tunnel vision",       "n": round(N * 0.30)},
            {"label": "End-stage — bare light perception",                      "n": round(N * 0.12)},
        ],
        "age_distribution": {
            "dx_0_4yr":   sum(1 for p in patients if '0–4' in p['age_dx_cat']),
            "dx_5_11yr":  sum(1 for p in patients if '5–11' in p['age_dx_cat']),
            "dx_12_17yr": sum(1 for p in patients if '12–17' in p['age_dx_cat']),
            "dx_18plus":  sum(1 for p in patients if '18+' in p['age_dx_cat']),
        },
    }


def get_breakdown():
    from collections import Counter

    ret_dist   = Counter(p['ret_stage'] for p in patients)
    poly_dist  = Counter(p['poly_type'] for p in patients if p['poly'])
    renal_dist = Counter(p['renal_type'] for p in patients if p['renal_any'])
    eth_dist   = Counter(p['eth'] for p in patients)
    allele_dist = Counter(p['allele'] for p in patients)
    pres_dist  = Counter(p['age_dx_cat'] for p in patients)
    mis_dist   = Counter(p['misdiagnosis'] for p in patients if p['misdiagnosis'])

    # average BMI
    avg_bmi = round(sum(p['bmi'] for p in patients) / N, 1)

    return {
        "retinal_stage_distribution": [{"label": k, "n": v} for k, v in sorted(ret_dist.items(), key=lambda x: -x[1])],
        "polydactyly_distribution": [
            {"label": "Polydactyly present", "n": _poly_n},
            {"label": "No polydactyly",      "n": N - _poly_n},
        ] + [{"label": k, "n": v} for k, v in sorted(poly_dist.items(), key=lambda x: -x[1])],
        "renal_distribution": [{"label": k, "n": v} for k, v in sorted(renal_dist.items(), key=lambda x: -x[1])]
                              + [{"label": "No renal disease", "n": N - _renal_n}],
        "allele_class_summary": [{"label": k, "n": v} for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "ethnicity_distribution": [{"ethnicity": k, "n": v} for k, v in sorted(eth_dist.items(), key=lambda x: -x[1])],
        "presentation_distribution": [{"label": k, "n": v} for k, v in sorted(pres_dist.items(), key=lambda x: -x[1])],
        "misdiagnosis_distribution": [{"label": k, "n": v} for k, v in sorted(mis_dist.items(), key=lambda x: -x[1])],
        "systemic_burden": [
            {"feature": "Obesity (BMI ≥ 30 by adolescence)",       "n": _obesity_n,   "pct": _pct(_obesity_n),   "avg_bmi": avg_bmi},
            {"feature": "Post-axial polydactyly",                   "n": _poly_n,      "pct": _pct(_poly_n)},
            {"feature": "Hypogonadism",                             "n": _hypo_n,      "pct": _pct(_hypo_n)},
            {"feature": "Anosmia / hyposmia",                       "n": _anosmia_n,   "pct": _pct(_anosmia_n)},
            {"feature": "Cognitive / learning disability",          "n": _cognitive_n, "pct": _pct(_cognitive_n)},
            {"feature": "Renal anomaly (structural or CKD)",        "n": _renal_n,     "pct": _pct(_renal_n)},
            {"feature": "Congenital heart defect (CHD)",            "n": _chd_n,       "pct": _pct(_chd_n)},
            {"feature": "Tri-allelic BBS modifier",                 "n": _triallelicn, "pct": _pct(_triallelicn)},
        ],
        "top_variants": [
            {"variant": "p.Arg631Pro (c.1892G>C) — C-terminal coiled-coil (aa 625–640); Bedouin/Arabian founder; BBSome outer coat disrupted; moderate-severe BBS2", "n": 10},
            {"variant": "p.Trp515Ter (c.1545G>A) — WD40/CC junction truncating null; biallelic LOF; European pan-ethnic; severe BBS2",                               "n": 6},
            {"variant": "p.Arg592Gln (c.1775G>A) — WD40 terminal blade (aa 585–600); BBS9-bridge interface; North African/European; moderate-severe BBS2",           "n": 7},
            {"variant": "p.Leu226Pro (c.677T>C) — WD40 core propeller (blade 5); BBS7-stacking interface; pan-ethnic compound het; moderate BBS2",                   "n": 5},
            {"variant": "p.Pro483Ser (c.1447C>T) — WD40 terminal / CC transition; BBS9-contact area; compound het; moderate BBS2",                                   "n": 4},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":          "BBS2",
            "aliases":       "Bardet-Biedl Syndrome 2",
            "full_name":     "Bardet-Biedl Syndrome 2 Protein",
            "omim_gene":     "*606151",
            "chromosome":    "16q13",
            "size":          "~721 amino acids",
            "protein_class": "WD40 beta-propeller (N-terminal, aa 1–400) + coiled-coil BBSome interface domain (C-terminal, aa 400–721)",
            "key_domains":   "WD40 beta-propeller (structural scaffold, BBS7-stacking surface) · N-WD40/CC transition (BBS9 bridge contact) · C-terminal coiled-coil (BBS7 CC + BBS8 TPR interface; R631P Bedouin founder site)",
            "key_interactions": "BBS7 (core dimer partner; WD40 stacking + CC packing) · BBS9/PTHB1 (bridge to BBS1 cargo recognition) · BBS8/TTC8 (N-TPR contact at BBS2 C-CC) · BBS1 (indirect; via BBS9 bridge)",
            "molecular_class": "BBSome structural CORE module — BBS2–BBS7–BBS9 trimer forms the scaffold spine; BBS2 LOF → catastrophic BBSome misassembly",
            "founder_note":  "p.Arg631Pro (c.1892G>C) is the Bedouin/Arabian Peninsula founder mutation in BBS2; 4–6× enriched vs other populations; ~30–45% of Middle Eastern BBS2 alleles. BBS2 lacks the BBS1 M390R European founder equivalent.",
        },
        "disease_card": {
            "disease":       "Bardet-Biedl Syndrome Type 2 (BBS2)",
            "omim_disease":  "#209900",
            "also_known_as": "BBS2 · Laurence-Moon-Bardet-Biedl syndrome 2 (historical)",
            "inheritance":   "Autosomal Recessive — biallelic LOF; tri-allelic BBS in ~8–9%",
            "prevalence":    "~1:100,000–160,000 (BBS overall); BBS2 ~10–20% of molecularly confirmed BBS",
            "cardinal_features": "Rod-cone retinal dystrophy · Post-axial polydactyly (70%) · Obesity/hyperphagia (88%) · Renal anomalies (50%) · Cognitive impairment (55%) · Hypogonadism (70%)",
            "retinal":       "Rod-cone progressive degeneration — ROD FIRST (scotopic ERG extinguishes before photopic); RP-like fundus; same ERG pattern as BBS1 (rod-cone vs Alstrom cone-rod)",
            "obesity":       "Truncal/central obesity; hyperphagia; BMI 30–50 by adolescence; LepR mis-trafficking → satiety failure; GLP-1 RA + bariatric surgery considered",
            "polydactyly":   "~70% post-axial (extra digit on ulnar/fibular side); bilateral hands > feet; may be surgically managed in infancy",
            "renal":         "~50% structural anomalies: calyceal clubbing, cysts, horseshoe kidney, CKD; ESRD possible; annual monitoring essential",
            "cognitive":     "~55% — mild to moderate learning disability; neuronal cilia BBSome failure; BBS2 cognitive profile similar to BBS1",
            "hypogonadism":  "~70% — males: cryptorchidism, small testes; females: irregular menses, ovarian cysts; hypogonadotropic pattern",
            "anosmia":       "~62% — olfactory cilia BBSome dysfunction; hyposmia to complete anosmia; often underreported",
            "chd":           "~7% CHD (ASD, VSD); NOT the infantile cardiomyopathy seen in Alstrom; cardiac echo in all BBS2 patients",
            "triallelic":    "~8–9% tri-allelic BBS: two BBS2 alleles + one variant in BBS1/7/9; more severe phenotype; BBSome completely non-functional",
        },
        "diagnostic_workup": [
            "1. Clinical diagnosis: rod-cone ERG (scotopic before photopic) + obesity + post-axial polydactyly in a child → suspect BBS; obtain comprehensive BBS gene panel (minimum 20 BBS genes)",
            "2. Gene panel: BBS2-specific (chr 16q13) — confirm biallelic LOF; include BBS7 (BBS2–BBS7 dimer partner; similar phenotype); always send a full BBS panel (BBS1–BBS22) not single-gene sequencing",
            "3. ERG: extinguished scotopic before photopic — rod-cone (NOT cone-rod as in Alstrom); annual ERG from age 3–5; fundus photography + OCT for photoreceptor layer monitoring",
            "4. Renal: annual renal USS + serum creatinine/GFR from birth; structural anomalies (calyceal clubbing, cysts) at 50%; ACEi/ARB if proteinuria; CKD management; ESRD → transplant",
            "5. Obesity: dietitian referral from diagnosis; GLP-1 receptor agonist (semaglutide) if BMI ≥ 35 and ≥ 12 years; bariatric surgery considered in adults with severe obesity + comorbidities",
            "6. Polydactyly: paediatric orthopaedics for surgical correction (infancy or early childhood)",
            "7. Hypogonadism: LH/FSH/testosterone/oestradiol; cryptorchidism correction; hormone replacement if indicated",
            "8. Anosmia: smell testing (UPSIT) at diagnosis; counselling (gas safety, olfactory hazards)",
            "9. Cardiac: echo at diagnosis (7% CHD); annual review in childhood; not DCM (Alstrom) — rule out structural disease",
            "10. Cognitive: neuropsychological assessment; IEP (Individual Education Plan); learning support; BBS2 similar cognitive profile to BBS1 (mild-moderate LD in 55%)",
            "11. Tri-allelic BBS: if more severe than expected for biallelic BBS2, expand panel to BBS1/7/9 for modifier allele; tri-allelic pedigree analysis",
            "12. Genetics: 25% AR recurrence; BBS2 R631P carrier screening in Bedouin/Arabian communities; PGT available",
        ],
        "mechanism_glossary": [
            {"term": "BBSome octameric complex (BBS1/2/4/5/7/8/9/18)", "definition":
             "Stable 8-subunit coat complex that traffics GPCRs and signalling receptors into/out "
             "of cilia via IFT. BBS2 (WD40 + CC) + BBS7 (WD40) form the structural core dimer. "
             "BBS9 bridges BBS2 to BBS1 (cargo recognition). BBS8 (TPR) contacts BBS2 CC. "
             "BBSome travels along axoneme with IFT trains as a cargo transporter."},
            {"term": "BBS2–BBS7 core dimer (BBSome scaffold spine)", "definition":
             "BBS2 N-terminal WD40 propeller stacks face-to-face onto BBS7 WD40 propeller; "
             "their C-terminal coiled-coils pack in parallel to form the BBSome structural spine. "
             "This dimer is the first assembly step of BBSome. Loss of BBS2 → BBS7 cannot form "
             "the dimer → rest of BBSome cannot assemble → complete BBSome LOF."},
            {"term": "BBS9 bridge (BBS2 → BBS1 cargo recognition link)", "definition":
             "BBS9/PTHB1 contacts BBS2 at the WD40/CC transition (aa 350–500) and bridges to "
             "BBS1 ARM domain (cargo recognition subunit). Disruption of the BBS2–BBS9 interface "
             "(hypomorphic alleles) → BBSome assembles but cannot recognise ciliary cargo → "
             "receptors fail to enter cilia despite partial BBSome integrity."},
            {"term": "R631P Bedouin/Arabian founder (c.1892G>C)", "definition":
             "p.Arg631Pro is the major Middle Eastern BBS2 founder mutation; located in the "
             "C-terminal coiled-coil (aa 625–640) at the BBS7 CC packing interface; "
             "Arg631 provides a salt bridge to BBS7 Glu at the CC dimer interface — Pro631 "
             "disrupts the helix and CC packing → BBSome outer coat unstable → LOF. "
             "Carrier frequency ~1/30–1/50 in Arabian Peninsula Bedouin populations."},
            {"term": "LepR mis-trafficking (obesity mechanism)", "definition":
             "Leptin receptor (LepR) requires BBSome-mediated ciliary trafficking for "
             "hypothalamic satiety signalling. BBS2 LOF → BBSome absent → LepR cannot "
             "exit cilia in response to leptin → leptin resistance → hyperphagia and "
             "obesity. This is the same mechanism as BBS1 obesity; GLP-1 RA addresses "
             "downstream satiety signalling independent of ciliary LepR."},
            {"term": "Rod-cone retinal degeneration (BBS2)", "definition":
             "Photoreceptor outer segments are modified primary cilia. BBS2 LOF → BBSome "
             "absent from photoreceptor cilia → rhodopsin and GPCR trafficking fails → "
             "outer segment membrane disruption → rod photoreceptors degenerate FIRST "
             "(scotopic ERG extinguishes before photopic). Same rod-cone pattern as BBS1; "
             "contrasts with Alstrom (cone-rod, cone first)."},
            {"term": "Tri-allelic BBS (BBS2 × 2 + modifier)", "definition":
             "~8–9% of BBS2 cases have a third pathogenic allele in a second BBS gene "
             "(most commonly BBS1, BBS7, or BBS9 — the BBS2 interaction partners). "
             "Two BBS2 alleles alone may cause complete BBSome LOF; a third allele in a "
             "core partner amplifies severity. Tri-allelic BBS2 often presents with earlier "
             "onset and more severe renal / retinal involvement."},
        ],
        "key_variants": [
            {"variant": "p.Arg631Pro", "domain": "C-terminal coiled-coil (BBS7-packing interface, aa 625–640)",
             "consequence": "Bedouin/Arabian Peninsula founder; Pro631 disrupts CC helix packing at BBS7 interface → BBSome outer coat unstable → BBSome LOF; moderate-severe BBS2",
             "ethnicity": "Bedouin / Arabian Peninsula (homozygous in consanguineous)"},
            {"variant": "p.Trp515Ter", "domain": "WD40/CC transition truncating (aa 515 null)",
             "consequence": "Early truncating null; no BBSome scaffold; complete BBS2 LOF; severe multi-system BBS",
             "ethnicity": "Pan-ethnic / European compound het"},
            {"variant": "p.Arg592Gln", "domain": "WD40 terminal blade / BBS9-bridge contact (aa 585–600)",
             "consequence": "BBS9–BBS2 bridge interface disrupted; BBSome may partially assemble but cargo recognition lost; moderate-severe BBS2",
             "ethnicity": "North African / European"},
            {"variant": "p.Leu226Pro", "domain": "WD40 core propeller blade 5 (aa 220–235; BBS7-stacking face)",
             "consequence": "BBS2–BBS7 core dimer destabilised; BBSome scaffold spine fails at step 1; moderate BBS2",
             "ethnicity": "Pan-ethnic compound het"},
            {"variant": "p.Pro483Ser", "domain": "WD40 terminal / CC transition (aa 478–490; BBS9-contact region)",
             "consequence": "BBS2–BBS9 bridge contact weakened; intermediate BBSome assembly defect; compound het; moderate BBS2",
             "ethnicity": "Pan-ethnic compound het"},
        ],
        "treatment_summary": [
            "1. Retinal: annual ERG + fundus + OCT; low-vision rehabilitation; orientation/mobility training; avoid high-intensity light exposure",
            "2. Obesity: early dietitian referral; low-calorie high-protein diet; GLP-1 RA (semaglutide/liraglutide) if BMI ≥ 35 and ≥ 12 yr; bariatric surgery in adults with BMI ≥ 40 + comorbidities; target LepR-independent satiety pathways",
            "3. Polydactyly: paediatric orthopaedic surgery for post-axial digit removal (infancy preferred); functional hand assessment",
            "4. Renal: annual USS + GFR/creatinine; nephrology referral if CKD; ACEi/ARB for proteinuria; ESRD → transplant planning",
            "5. Hypogonadism: testosterone replacement (males) or oestrogen–progesterone (females) if needed; cryptorchidism orchidopexy in infancy",
            "6. Anosmia: smell training; gas safety (LPG detector); occupational counselling re: odour-dependent work",
            "7. Cognitive: neuropsychological testing at school age; IEP; special education support; speech-language therapy if verbal LD",
            "8. Cardiac: echo at diagnosis; annual auscultation; NOT cardiomyopathy (unlike Alstrom); structural CHD correction if present",
            "9. Tri-allelic BBS2: more intensive monitoring across all systems; genetic counselling re: reduced recurrence predictability",
            "10. Gene panel: full BBS panel (20–24 genes) mandatory; BBS2 single-gene testing misses BBS2 compound-het + other locus co-mutation; confirm via RNA functional assay if VUS",
            "11. Therapeutic pipeline (2026): BBSome cargo-trafficking modulators (preclinical); gene therapy for photoreceptor cilia; semaglutide/tirzepatide (standard obesity care); no approved ciliopathy-specific treatment",
        ],
        "ddx_table": [
            {"disease": "BBS1 (BBS1 gene) — most common BBS (~25%)", "key_difference":
             "BBS1 has M390R European founder (70% of European BBS1 alleles); BBS2 lacks this — "
             "BBS2 allele spectrum more heterogeneous; both rod-cone (rod first); same BBSome pathway; "
             "gene panel is the ONLY differentiator at clinical level; BBS2 has R631P Bedouin founder"},
            {"disease": "BBS10 — second most common BBS (~25%); chaperonin-like (CCT)",
             "key_difference":
             "BBS10 is a chaperonin-like protein (NOT a BBSome structural subunit); BBS10 assists "
             "BBSome assembly as an HSP60-like chaperone; BBS10 patients have HIGHER obesity rate and "
             "MORE severe renal disease than BBS2; gene panel mandatory; clinically indistinguishable"},
            {"disease": "BBS7 (BBS7 gene) — BBS2–BBS7 core dimer partner", "key_difference":
             "BBS7 is the direct dimer partner of BBS2 (BBS2–BBS7 WD40 spine); loss of BBS7 has "
             "nearly identical phenotype to BBS2 (same BBSome scaffold failure); gene panel required; "
             "tri-allelic BBS2+BBS7 produces complete BBSome scaffold loss"},
            {"disease": "Alstrom Syndrome (ALMS1)", "key_difference":
             "Alstrom: CONE-ROD (cone first) vs BBS2 rod-cone (rod first) — ERG is the key discriminator; "
             "Alstrom has infantile cardiomyopathy (DCM) — BBS2 does NOT; Alstrom has sensorineural "
             "deafness — BBS2 does NOT; Alstrom NO polydactyly — BBS2 70% polydactyly; "
             "Alstrom NO cognitive impairment — BBS2 55%; ALMS1 is a single gene"},
            {"disease": "Laurence-Moon Syndrome (historical)", "key_difference":
             "Laurence-Moon (historical eponym) was originally described with paraplegia/spastic diplegia "
             "and NO polydactyly; Bardet-Biedl (with polydactyly, no paraplegia) is now the correct term "
             "for BBS2; most 'Laurence-Moon' patients historically are now reclassified as BBS"},
            {"disease": "MKKS / BBS6 (McKusick-Kaufman Syndrome)", "key_difference":
             "MKKS causes hydrometrocolpos + polydactyly + CHD in neonates (McKusick-Kaufman); "
             "if BBS features emerge post-childhood → BBS6; MKKS is a chaperonin-like protein; "
             "overlap with BBS2 tri-allelic; gene panel resolves"},
        ],
    }
