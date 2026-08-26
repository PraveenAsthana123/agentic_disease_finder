"""
BBS3 Bardet-Biedl Syndrome Type 3 — ARL6 (ADP Ribosylation Factor Like GTPase 6) Dashboard
=============================================================================================
Primary Gene : ARL6 (*608845) — ADP Ribosylation Factor Like GTPase 6;
               also known as BBS3;
               ~186 aa; single GTPase domain (Arf-like); myristoylated N-terminus;
               3q11.2.
               ARL6 is FUNDAMENTALLY DIFFERENT from BBS2: it is NOT a structural
               subunit of the BBSome octameric complex. Instead:
                 GTPase domain (aa 1–186): Arf/ARL-family GTPase;
                   P-loop (aa 23–35): GTP/GDP nucleotide binding; Thr31 (P-loop)
                   contacts Mg²⁺-GTP phosphate — Thr31Ile → GTP binding abolished;
                   Switch I (aa 36–55): GTP-γ-phosphate sensor; conformational
                   change on GTP binding opens the BBS1-binding surface;
                   Switch II (aa 72–97): Leu93 stabilises switch II β-strand
                   stacking — Leu93Pro → switch II disordered → GTP hydrolysis
                   conformation lost → GDP-locked ARL6 cannot engage BBS1;
                   Interswitch region (aa 110–130): contacts BBS1 β-propeller
                   directly — Arg122 forms a salt bridge to BBS1 Asp89; Arg122Trp
                   → BBS1 interface disrupted → BBSome cannot be recruited to
                   ciliary membrane;
                 C-terminal amphipathic helix (aa 160–186): membrane anchor;
                   Ile195 (in extended isoform) positions the ARL6-GTP complex at
                   the ciliary base membrane (PtdIns(3)P-enriched).
               BBSome membrane recruitment mechanism:
                 ARL6-GTP (active, membrane-bound) → binds BBS1 β-propeller →
                 recruits intact BBSome to ciliary membrane → BBSome concentrates
                 GPCR cargo (LepR, SSTR3, MCHR1, ORP3) at ciliary base →
                 IFT-B trains carry cargo along axoneme.
               ARL6 LOF → ARL6 remains GDP-locked or absent → BBSome assembles
               normally in cytoplasm but CANNOT dock to ciliary membrane →
               BBSome-dependent GPCR ciliary trafficking fails → multi-system BBS.
               KEY DISTINCTION: BBSome is intact in BBS3 (unlike BBS2 where BBSome
               itself fails to assemble); the DELIVERY of BBSome to the membrane
               is defective. This has therapeutic implications: restoring ARL6-GTP
               activity could rescue BBSome trafficking without fixing BBSome itself.

Disease OMIM : #209900 — Bardet-Biedl Syndrome (ARL6/BBS3 allelic disorder)
               All BBS types share OMIM #209900; molecular subtype defined by gene.
               Allele spectrum:
               — Severe: biallelic null (truncating/null-null) → complete ARL6 LOF;
                 no GTP cycle → BBSome never reaches membrane; severe multi-system
               — Moderate: compound het (P-loop + switch II missense) → residual
                 GTP binding; partial BBSome recruitment; moderate multi-system
               — Mild: hypomorphic switch II or C-terminal helix alleles → reduced
                 but not absent GTP hydrolysis; BBSome partially recruited;
                 attenuated multi-system BBS
               UNIQUE FEATURES vs BBS1 and BBS2:
               (1) ONLY GTPase IN BBS PATHWAY: ARL6/BBS3 is the sole Arf-family
                   small GTPase in the BBSome recruitment mechanism; not a structural
                   BBSome subunit — it acts as the MOLECULAR SWITCH for BBSome
                   membrane docking; this is a unique functional class
               (2) BBSome ASSEMBLES NORMALLY in BBS3: unlike BBS2 where BBSome
                   misassembles, the BBSome octamer is intact in BBS3; failure is
                   at the membrane recruitment step — ARL6 is upstream of BBSome
                   membrane contact, not inside the complex
               (3) FREQUENCY ~3–5% of molecularly confirmed BBS: rare compared
                   to BBS1 (~25%) and BBS2 (~10–20%); ultra-rare: under 100–150
                   families worldwide (2026)
               (4) DRUG-TARGETABLE GTPASE CYCLE: small molecule GTPase modulators
                   (GEF activators to shift ARL6 toward GTP-bound state) represent
                   a potential therapeutic avenue unique to BBS3 — not applicable
                   to structural BBSome subunit disorders (BBS2, BBS7, etc.)
               (5) NO DOMINANT ETHNIC FOUNDER: unlike BBS1 M390R (European) or
                   BBS2 R631P (Bedouin), BBS3/ARL6 lacks a single dominant founder;
                   allele spectrum pan-ethnic with slight enrichment of Leu93Pro
                   in European populations
Chromosome   : 3q11.2
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous);
               tri-allelic BBS in ~5% (lower than BBS2 because ARL6 is peripheral)
Prevalence   : ~1:100,000–160,000 worldwide (BBS overall); ARL6/BBS3 accounts
               for ~3–5% of molecularly confirmed BBS cases; under 150 families
"""

import random

SEED = 337
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
    ('European',                                     0.35),  # Leu93Pro slight enrichment
    ('Middle Eastern / North African',               0.28),  # Arg122Trp interswitch
    ('South Asian',                                  0.18),  # Ile195Thr C-terminal
    ('East Asian / Other',                           0.12),
    ('Latin American',                               0.07),
]

allele_classes = [
    ('Compound het — missense + truncating',             0.38),
    ('Compound het — two missense',                      0.27),
    ('Homozygous missense (consanguineous)',             0.22),  # MENA consanguinity
    ('Biallelic truncating (null/null)',                 0.08),
    ('Tri-allelic BBS (ARL6 × 2 + modifier)',           0.05),
]

retinal_stage = [
    ('Early — reduced scotopic ERG; minimal visual loss',     0.22),
    ('Moderate — extinguished scotopic; preserved photopic',  0.36),
    ('Advanced — rod-cone both affected; tunnel vision',      0.29),
    ('End-stage — bare light perception',                     0.13),
]

renal_types = [
    'Calyceal clubbing (corticomedullary junction)',
    'Renal cysts (cortical)',
    'CKD stage 2–3 (tubulointerstitial)',
    'Horseshoe kidney',
    'ESRD — transplant listed or completed',
    'Recurrent UTI + vesicoureteral reflux',
]

misdiagnosis_labels = [
    'Non-syndromic retinitis pigmentosa (rod-cone without systemic features)',
    'Alstrom Syndrome (obesity + rod-cone; ARL6 lacks DCM/deafness)',
    'Laurence-Moon Syndrome (historical term; BBS3 more common)',
    'Usher Syndrome (early retinal; BBS3 lacks SNHL)',
    'BBS1 (clinically identical; only gene panel differentiates)',
    None,   # not misdiagnosed
]

_poly_n      = round(N * 0.63)   # 63% post-axial polydactyly (slightly lower than BBS1/BBS2)
_renal_n     = round(N * 0.38)   # 38% structural renal anomaly or CKD
_cognitive_n = round(N * 0.45)   # 45% learning disability / cognitive impairment
_obesity_n   = round(N * 0.78)   # 78% obesity — slightly lower than BBS2 (88%)
_hypo_n      = round(N * 0.62)   # 62% hypogonadism
_anosmia_n   = round(N * 0.58)   # 58% anosmia / hyposmia
_chd_n       = round(N * 0.05)   # 5% CHD (lower than BBS1/BBS2 7%)
_triallelicn = round(N * 0.05)   # 5% tri-allelic (lower than BBS2 9%)
_mis_n       = round(N * 0.43)   # 43% misdiagnosed initially

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
    renal_any  = i < _renal_n
    renal_type = (rng.choice(renal_types) if renal_any else None)
    cognitive  = i < _cognitive_n
    obese      = i < _obesity_n
    hypo       = i < _hypo_n
    anosmia    = i < _anosmia_n
    chd        = i < _chd_n
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

    bmi = (rng.uniform(29, 50) if obese else rng.uniform(21, 28))

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
        "gene": "ARL6 (BBS3)",
        "disease": "Bardet-Biedl Syndrome Type 3 (BBS3 / ARL6)",
        "mechanism": (
            "ARL6 (~186 aa) is the ONLY small GTPase in the BBSome membrane-recruitment pathway. "
            "Unlike BBS2 (a structural BBSome scaffold subunit), ARL6 acts as a MOLECULAR SWITCH: "
            "ARL6-GTP (active, ciliary membrane-bound) directly contacts the BBS1 β-propeller "
            "(Arg122 interswitch → BBS1 Asp89 salt bridge), recruiting the intact BBSome octamer "
            "to the PtdIns(3)P-enriched ciliary base membrane. Without ARL6-GTP, BBSome assembles "
            "normally in the cytoplasm but CANNOT dock to the ciliary membrane → GPCR cargo "
            "(LepR, SSTR3, MCHR1, ORP3) cannot be concentrated at the ciliary base for IFT-B "
            "transport → multi-system BBS disease. "
            "BBSome is structurally INTACT in BBS3 — the defect is upstream at membrane recruitment, "
            "not within the complex (contrasting with BBS2 where BBSome fails to assemble)."
        ),
        "key_distinction": (
            "BBS3 (ARL6) vs BBS1 and BBS2: "
            "UNIQUE GTPase CLASS — ARL6 is the only Arf-family GTPase in the BBS pathway; BBS1 is a "
            "β-propeller cargo-recognition subunit; BBS2 is a WD40/CC structural scaffold; ARL6 is a "
            "peripheral membrane GTPase acting as molecular switch for BBSome docking. "
            "BBSome INTACT in BBS3: assembly of the 8-subunit BBSome proceeds normally; failure is at "
            "the membrane-docking step (ARL6-GTP → BBS1 interface); this creates a potential drug "
            "target (GEF activators to restore ARL6-GTP state). "
            "FREQUENCY ~3–5% of all BBS: rare compared to BBS1 (~25%) and BBS2 (~10–20%); "
            "under 150 families worldwide (2026). "
            "NO DOMINANT ETHNIC FOUNDER: Leu93Pro mildly enriched in Europeans; Arg122Trp in "
            "Middle East/North Africa; Ile195Thr in South Asia — but none reaches BBS1 M390R "
            "(70% European) or BBS2 R631P (30–45% MENA) prevalence."
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
        "bbs_pathway_comparison": [
            {"gene": "BBS1",       "role": "Cargo recognition — β-propeller + ARM; binds Rab8a/Rab11; ARL6-GTP contacts BBS1 β-propeller to recruit BBSome", "function_class": "Structural BBSome subunit", "omim": "*209901", "frequency_pct": "~25"},
            {"gene": "BBS2",       "role": "Structural core scaffold — WD40 propeller; BBS2–BBS7 dimer = BBSome spine; LOF → BBSome misassembles", "function_class": "Structural BBSome subunit", "omim": "*606151", "frequency_pct": "~10–20"},
            {"gene": "ARL6/BBS3 (THIS)", "role": "Peripheral membrane GTPase — ARL6-GTP contacts BBS1 to RECRUIT BBSome to ciliary membrane; BBSome assembles normally; failure is membrane docking", "function_class": "BBSome membrane recruiter (GTPase)", "omim": "*608845", "frequency_pct": "~3–5"},
            {"gene": "BBS4",       "role": "TPR adaptor — peripheral BBSome subunit; structural connector; rarely causes BBS", "function_class": "Structural BBSome subunit", "omim": "*600374", "frequency_pct": "~3–5"},
            {"gene": "BBS7",       "role": "WD40 propeller — BBS2–BBS7 core dimer partner; identical clinical phenotype to BBS2", "function_class": "Structural BBSome subunit", "omim": "*607590", "frequency_pct": "~3–5"},
            {"gene": "BBS9",       "role": "Structural bridge — BBS2 → BBS1 cargo recognition link; BBS9 LOF → cargo recognition failure", "function_class": "Structural BBSome subunit", "omim": "*607968", "frequency_pct": "~4–6"},
            {"gene": "BBS10",      "role": "Chaperonin-like (HSP60-family) — assists BBSome assembly; NOT a BBSome subunit; BBS10 LOF → assembly failure", "function_class": "BBSome assembly chaperone", "omim": "*610148", "frequency_pct": "~25"},
        ],
        "retinal_distribution": [
            {"label": "Early — reduced scotopic ERG; minimal visual loss",     "n": round(N * 0.22)},
            {"label": "Moderate — extinguished scotopic; preserved photopic",  "n": round(N * 0.36)},
            {"label": "Advanced — rod-cone both affected; tunnel vision",       "n": round(N * 0.29)},
            {"label": "End-stage — bare light perception",                      "n": round(N * 0.13)},
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
            {"feature": "Obesity (BMI ≥ 29 by adolescence)",         "n": _obesity_n,   "pct": _pct(_obesity_n),   "avg_bmi": avg_bmi},
            {"feature": "Post-axial polydactyly",                     "n": _poly_n,      "pct": _pct(_poly_n)},
            {"feature": "Hypogonadism",                               "n": _hypo_n,      "pct": _pct(_hypo_n)},
            {"feature": "Anosmia / hyposmia",                         "n": _anosmia_n,   "pct": _pct(_anosmia_n)},
            {"feature": "Cognitive / learning disability",            "n": _cognitive_n, "pct": _pct(_cognitive_n)},
            {"feature": "Renal anomaly (structural or CKD)",          "n": _renal_n,     "pct": _pct(_renal_n)},
            {"feature": "Congenital heart defect (CHD)",              "n": _chd_n,       "pct": _pct(_chd_n)},
            {"feature": "Tri-allelic BBS modifier",                   "n": _triallelicn, "pct": _pct(_triallelicn)},
        ],
        "top_variants": [
            {"variant": "p.Leu93Pro (c.278T>C) — switch II region (aa 88–98); European; switch II β-strand disordered → GDP-locked ARL6 → BBSome not recruited; moderate BBS3", "n": 9},
            {"variant": "p.Arg122Trp (c.364C>T) — interswitch region (aa 118–128); Middle Eastern/North African; Arg122 salt bridge to BBS1 Asp89 lost → BBSome membrane docking abolished; moderate-severe BBS3", "n": 8},
            {"variant": "p.Thr31Ile (c.92C>T) — P-loop nucleotide-binding (aa 23–35); pan-ethnic; Thr31 contacts Mg²⁺-GTP; Ile31 → GTP binding abolished → complete ARL6 LOF; severe BBS3", "n": 6},
            {"variant": "p.Ile195Thr (c.584T>C) — C-terminal membrane anchor (aa 190–200); South Asian; reduces PtdIns(3)P ciliary base tethering; partial BBSome recruitment; moderate BBS3", "n": 5},
            {"variant": "p.Glu209Ter (c.625G>T) — truncating null; pan-ethnic; complete ARL6 LOF; severe multi-system BBS3", "n": 4},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":          "ARL6",
            "aliases":       "BBS3; ADP Ribosylation Factor Like GTPase 6",
            "full_name":     "ADP Ribosylation Factor Like GTPase 6 (Bardet-Biedl Syndrome 3 protein)",
            "omim_gene":     "*608845",
            "chromosome":    "3q11.2",
            "size":          "~186 amino acids",
            "protein_class": "Arf/ARL-family small GTPase; single GTPase domain; myristoylated N-terminus; NO WD40, NO TPR, NO coiled-coil",
            "key_domains":   "P-loop (aa 23–35; nucleotide binding; Thr31 contacts Mg²⁺-GTP) · Switch I (aa 36–55; GTP-γ-phosphate sensor; opens BBS1 surface on GTP binding) · Switch II (aa 72–97; Leu93 stabilises β-strand stacking for GTP hydrolysis conformation) · Interswitch region (aa 110–130; Arg122 salt bridge to BBS1 Asp89 — direct BBS1 contact for BBSome recruitment) · C-terminal amphipathic helix (aa 160–186; ciliary base membrane anchor)",
            "key_interactions": "BBS1 β-propeller (Arg122 interswitch → BBS1 Asp89; GTP-dependent — ARL6-GTP only) · PtdIns(3)P-enriched ciliary base membrane (C-terminal helix anchor) · INPP5E (ciliary PI phosphatase; regulates PtdIns(3)P; modulates ARL6 membrane dwell time)",
            "molecular_class": "BBSome MEMBRANE RECRUITER (GTPase) — NOT a structural BBSome subunit; acts as molecular switch upstream of BBSome membrane docking; ARL6 LOF → BBSome intact but cannot reach ciliary membrane",
            "founder_note":  "No single dominant ethnic founder. p.Leu93Pro (switch II) mildly enriched in Europeans; p.Arg122Trp (interswitch) enriched in Middle East/North Africa; p.Ile195Thr (C-terminal) in South Asia. Contrast: BBS1 M390R (70% European alleles) and BBS2 R631P (30–45% MENA alleles).",
        },
        "disease_card": {
            "disease":       "Bardet-Biedl Syndrome Type 3 (BBS3 / ARL6)",
            "omim_disease":  "#209900",
            "also_known_as": "BBS3 · ARL6-associated BBS · Laurence-Moon-Bardet-Biedl syndrome 3 (historical)",
            "inheritance":   "Autosomal Recessive — biallelic LOF; tri-allelic BBS in ~5%",
            "prevalence":    "~1:100,000–160,000 (BBS overall); ARL6/BBS3 ~3–5% of molecularly confirmed BBS; under 150 families worldwide (2026)",
            "cardinal_features": "Rod-cone retinal dystrophy · Post-axial polydactyly (63%) · Obesity/hyperphagia (78%) · Renal anomalies (38%) · Cognitive impairment (45%) · Hypogonadism (62%)",
            "retinal":       "Rod-cone progressive degeneration — ROD FIRST (scotopic ERG extinguishes before photopic); RP-like fundus; same rod-cone ERG pattern as BBS1 and BBS2; contrasts with Alstrom (cone-rod, cone first)",
            "obesity":       "Truncal/central obesity; hyperphagia; ~78% (somewhat lower than BBS2 88%, BBS1 ~80%); LepR ciliary trafficking fails → satiety failure; GLP-1 RA + dietary management",
            "polydactyly":   "~63% post-axial (slightly lower than BBS1/BBS2 70%); extra digit on ulnar/fibular side; surgical correction in infancy",
            "renal":         "~38% structural anomalies (lower than BBS2 50%): calyceal clubbing, cortical cysts, CKD; ESRD less common than BBS2; annual monitoring",
            "cognitive":     "~45% — mild to moderate learning disability (lower than BBS2 55%); neuronal cilia BBSome trafficking failure; similar educational support needs",
            "hypogonadism":  "~62% — males: cryptorchidism, small testes; females: irregular menses, ovarian cysts; hypogonadotropic pattern; hormone replacement as needed",
            "anosmia":       "~58% — olfactory cilia BBSome dysfunction; hyposmia to complete anosmia; often underreported",
            "chd":           "~5% CHD (ASD, VSD); NOT infantile cardiomyopathy (Alstrom); cardiac echo at diagnosis",
            "triallelic":    "~5% tri-allelic BBS: two ARL6 alleles + one variant in BBS1/BBS4/BBS7; lower than BBS2 (9%) because ARL6 is peripheral — modifier alleles in BBSome core have smaller additive effect on ARL6-GTP function",
        },
        "diagnostic_workup": [
            "1. Clinical diagnosis: rod-cone ERG (scotopic before photopic) + obesity + post-axial polydactyly → suspect BBS; obtain full BBS gene panel (minimum 20–24 genes); BBS3/ARL6 ~3–5% of confirmed BBS",
            "2. Gene panel: ARL6-specific (chr 3q11.2) — confirm biallelic LOF in ARL6; always include full BBS1–BBS22 panel (not ARL6 alone — BBS3 clinically indistinguishable from BBS1/BBS2 without genetics)",
            "3. ERG: extinguished scotopic before photopic — rod-cone (NOT cone-rod as in Alstrom); annual ERG from age 3–5; OCT for photoreceptor layer thinning; fundus photography",
            "4. Renal: annual renal USS + serum creatinine/GFR; structural anomalies (~38%); ACEi/ARB for proteinuria; CKD management",
            "5. Obesity: dietitian from diagnosis; GLP-1 RA (semaglutide) if BMI ≥ 35 and ≥ 12 yr; bariatric surgery in adults; target LepR-independent satiety",
            "6. Polydactyly: paediatric orthopaedics for post-axial digit correction (infancy preferred)",
            "7. Hypogonadism: LH/FSH/testosterone/oestradiol; cryptorchidism orchidopexy; hormone replacement if indicated",
            "8. Anosmia: smell testing (UPSIT); gas safety counselling (LPG detector)",
            "9. Cardiac: echo at diagnosis (5% CHD); not cardiomyopathy (Alstrom); rule out structural disease",
            "10. Cognitive: neuropsychological assessment at school age; IEP; learning support",
            "11. ARL6 GTPase activity (research): functional GTPase assay (GTP-binding + hydrolysis) to classify variant as GDP-locked vs partial-activity hypomorph — guides prognosis and future GEF-activator trial eligibility",
            "12. Genetics: 25% AR recurrence; genetic counselling; PGT available; ARL6 VUS: classify by GTPase domain position (P-loop/switch I/II/interswitch = likely pathogenic; C-terminal = partial phenotype possible)",
        ],
        "mechanism_glossary": [
            {"term": "ARL6 GTPase cycle (BBSome switch)", "definition":
             "ARL6 cycles between GTP-bound (active) and GDP-bound (inactive) states. "
             "ACTIVE: ARL6-GTP inserts N-terminal myristoyl group into ciliary base membrane → "
             "Switch I/II conformational change exposes BBS1-binding surface (Arg122 interswitch) → "
             "BBS1 β-propeller contacts ARL6 → intact BBSome docks to ciliary membrane → "
             "GPCR cargo concentrated for IFT-B transport. "
             "INACTIVE: ARL6-GDP in cytoplasm → BBS1 surface not exposed → BBSome floats in "
             "cytoplasm without membrane contact → cargo trafficking fails."},
            {"term": "BBSome membrane recruitment (ARL6 → BBS1 → BBSome)", "definition":
             "ARL6-GTP binds BBS1 β-propeller via Arg122 (ARL6 interswitch) → Asp89 (BBS1) salt bridge. "
             "This docking event anchors the intact BBSome octamer to the ciliary base membrane "
             "(PtdIns(3)P-enriched region at axoneme entry zone). "
             "BBSome then laterally diffuses along the membrane to concentrate GPCR cargo "
             "(LepR, SSTR3, MCHR1) for IFT-B-mediated anterograde/retrograde transport. "
             "ARL6 LOF: BBSome fully assembled in cytoplasm but remains cytoplasmic."},
            {"term": "BBSome-intact mechanism (unique to BBS3 vs BBS2)", "definition":
             "In BBS2 (structural scaffold LOF), the BBSome octamer itself FAILS TO ASSEMBLE — "
             "BBS2 LOF prevents BBS2–BBS7 dimer formation, collapsing the entire complex. "
             "In BBS3 (ARL6 LOF), BBSome assembles NORMALLY as an octamer (BBS1/2/4/5/7/8/9/18 "
             "intact); the failure is UPSTREAM at membrane recruitment. "
             "Therapeutic implication: a GEF activator that shifts ARL6 toward GTP-bound state "
             "could rescue BBSome membrane docking WITHOUT fixing the BBSome itself — "
             "a target unique to BBS3 not applicable to BBS2, BBS7, or BBS9 disorders."},
            {"term": "Leu93Pro (switch II destabilisation)", "definition":
             "Leu93 sits on the switch II β-strand (aa 88–98); its hydrophobic side chain "
             "stacks against the GTPase core, stabilising the GTP-hydrolysis conformation. "
             "Pro93 inserts a helix-breaking residue into the strand → switch II disordered → "
             "ARL6 cannot complete GTP hydrolysis cycle properly → GDP-locked; "
             "GDP-locked ARL6 cannot expose the BBS1-binding interswitch surface → "
             "BBSome not recruited to ciliary membrane → BBS3."},
            {"term": "Arg122 interswitch (BBS1 molecular interface)", "definition":
             "Arg122 on the ARL6 interswitch region (aa 110–130) forms a salt bridge with "
             "BBS1 Asp89 on the BBS1 β-propeller. This interaction is GTP-dependent: "
             "only ARL6-GTP exposes Arg122 for BBS1 contact. "
             "Arg122Trp: bulky aromatic Trp disrupts the ARL6 interswitch backbone geometry "
             "and abolishes the Arg122–BBS1 Asp89 salt bridge → BBSome not recruited to "
             "ciliary membrane → severe BBS3."},
            {"term": "GEF-activator therapeutic concept (BBS3-specific)", "definition":
             "A Guanine nucleotide Exchange Factor (GEF) activator for ARL6 would shift "
             "the ARL6 GDP→GTP equilibrium toward the GTP-bound active state. "
             "If the patient has a hypomorphic ARL6 allele (partial GTPase activity, e.g. "
             "Ile195Thr or mild switch II missenses), a GEF activator could rescue sufficient "
             "ARL6-GTP to restore BBSome membrane docking. This approach is not applicable "
             "to null/truncating BBS3 alleles and is distinct from BBSome-subunit therapies."},
        ],
        "key_variants": [
            {"variant": "p.Leu93Pro", "domain": "Switch II region (aa 88–98; GTP hydrolysis β-strand)",
             "consequence": "Switch II strand disordered by Pro93; GDP-locked ARL6; BBS1-binding surface not exposed; BBSome not recruited; moderate-severe BBS3",
             "ethnicity": "European (mildly enriched)"},
            {"variant": "p.Arg122Trp", "domain": "Interswitch region (aa 118–128; BBS1 Asp89 contact)",
             "consequence": "Arg122–BBS1 Asp89 salt bridge abolished; BBSome membrane docking fails; moderate-severe BBS3",
             "ethnicity": "Middle Eastern / North African"},
            {"variant": "p.Thr31Ile", "domain": "P-loop nucleotide-binding (aa 23–35; Mg²⁺-GTP contact)",
             "consequence": "Thr31 contacts Mg²⁺ coordinating GTP γ-phosphate; Ile31 → GTP binding abolished → complete ARL6 LOF; severe multi-system BBS",
             "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ile195Thr", "domain": "C-terminal amphipathic helix (membrane anchor, aa 190–200)",
             "consequence": "Partial loss of PtdIns(3)P ciliary base tethering; residual BBSome membrane recruitment; moderate BBS3 (hypomorphic)",
             "ethnicity": "South Asian"},
            {"variant": "p.Glu209Ter", "domain": "C-terminal truncating null",
             "consequence": "Complete ARL6 LOF; no membrane anchor or GTPase domain; severe biallelic null BBS3",
             "ethnicity": "Pan-ethnic"},
        ],
        "treatment_summary": [
            "1. Retinal: annual ERG + fundus photography + OCT; low-vision rehabilitation; orientation/mobility training; avoid intense light",
            "2. Obesity: early dietitian referral; GLP-1 RA (semaglutide/liraglutide) if BMI ≥ 35 and ≥ 12 yr; bariatric surgery in adults; target LepR-independent satiety pathways",
            "3. Polydactyly: paediatric orthopaedic surgery for post-axial digit removal (infancy preferred); functional hand therapy",
            "4. Renal: annual USS + GFR/creatinine; nephrology if CKD; ACEi/ARB for proteinuria; ESRD → transplant planning",
            "5. Hypogonadism: hormone replacement (testosterone/oestrogen-progesterone); cryptorchidism orchidopexy in infancy",
            "6. Anosmia: smell training; gas safety (LPG detector); occupational counselling",
            "7. Cognitive: neuropsychological testing at school age; IEP; special education support",
            "8. Cardiac: echo at diagnosis (5% CHD); not cardiomyopathy; structural CHD correction if present",
            "9. ARL6 GEF-activator pipeline (2026, preclinical): small molecules to shift ARL6 GDP→GTP equilibrium; applicable to hypomorphic alleles (Ile195Thr, partial switch II); not to nulls (Thr31Ile, Glu209Ter)",
            "10. Gene panel: full BBS panel (20–24 genes) mandatory; ARL6 single-gene testing insufficient; confirm VUS with GTPase functional assay (GTP-binding + BBS1 pull-down)",
        ],
        "ddx_table": [
            {"disease": "BBS1 (BBS1 gene) — most common BBS (~25%)", "key_difference":
             "BBS1 is a CARGO RECOGNITION subunit (β-propeller + ARM); ARL6-GTP directly contacts "
             "BBS1 β-propeller to recruit BBSome; BBS1 M390R European founder (70% of European BBS1) "
             "vs BBS3 no dominant founder; both rod-cone; gene panel is the ONLY clinical differentiator"},
            {"disease": "BBS2 (BBS2 gene) — ~10–20% of BBS; BBSome core scaffold", "key_difference":
             "BBS2 is a STRUCTURAL BBSome subunit (WD40 + CC); BBS2 LOF → BBSome MISASSEMBLES; "
             "BBS3 LOF → BBSome ASSEMBLES normally but cannot reach ciliary membrane; "
             "BBS2 R631P Bedouin founder vs BBS3 no dominant founder; BBS3 obesity slightly lower; "
             "gene panel mandatory"},
            {"disease": "BBS10 — ~25% of BBS; chaperonin-like (CCT/HSP60)", "key_difference":
             "BBS10 is a chaperonin-like assembly factor for BBSome; BBS10 more severe obesity and "
             "renal disease than BBS3; BBS10 most common BBS gene (with BBS1); "
             "gene panel mandatory; clinically indistinguishable from BBS3"},
            {"disease": "Alstrom Syndrome (ALMS1)", "key_difference":
             "Alstrom: CONE-ROD (cone first) vs BBS3 rod-cone (rod first); Alstrom infantile DCM "
             "(cardiomyopathy, 60%) — BBS3 CHD only 5%; Alstrom SNHL — BBS3 no SNHL; "
             "Alstrom NO polydactyly — BBS3 63% polydactyly; ALMS1 single gene"},
            {"disease": "Joubert Syndrome (JBTS; INPP5E, TMEM67, AHI1, CEP290, others)", "key_difference":
             "Joubert has molar tooth sign (MTS) on brain MRI — BBS3/ARL6 does NOT cause Joubert "
             "despite ciliopathy overlap (ARL6 ciliary base membrane; not transition zone); "
             "Joubert: cerebellar vermis hypoplasia + ataxia; BBS3: no MTS; gene panel resolves"},
        ],
    }
