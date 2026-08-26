"""
IFT140 Short-Rib Thoracic Dysplasia 9 (SRTD9) — Jeune Asphyxiating Thoracic Dystrophy Type 9
================================================================================================
Primary Gene : IFT140 (*614620) — 16p13.3; ~1462 aa; Intraflagellar Transport Protein 140
               IFT140 is a core structural subunit of the IFT-A complex (6-subunit retrograde
               adapter complex: WDR19/IFT144, IFT140, WDR35/IFT121, IFT122, TTC21B/IFT139, IFT43).
               IFT-A complex rides on IFT trains and is required for RETROGRADE IFT (ciliary tip
               → cell body, powered by dynein-2) and for proper ANTEROGRADE IFT-B cargo import
               at the cilia base. IFT140 is the second-largest IFT-A subunit (after WDR19) and
               forms the structural scaffold: N-terminal WD40 β-propeller (aa 1–500) interfaces
               with WDR35/IFT121; C-terminal TPR (tetratricopeptide repeat) domain (aa 901–1462)
               contacts WDR19/IFT144 and dynein-2 (DYNC2H1). IFT140 loss → IFT-A complex
               instability → retrograde IFT impaired and IFT-B import defect at cilia base →
               SHORT/STUBBY CILIA (distinct from dynein-2 SRTDs which show club/bulging tips) →
               Hedgehog (Ihh/Shh) signalling failure in chondrocytes → narrow thorax, short ribs.
               IFT-A complex hierarchy: WDR19/IFT144 (SRTD5) + IFT140 (SRTD9) + WDR35/IFT121
               (SRTD7) + TTC21B/IFT139 (SRTD4/NPHP12) + IFT122 (CED2-like) + IFT43.
Disease OMIM : #266920 — Short-Rib Thoracic Dysplasia 9 With or Without Polydactyly (SRTD9)
               Also called: Asphyxiating Thoracic Dystrophy 9 (ATD9); Jeune syndrome type 9
               Severe biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum
               WDR19/IFT144 (SRTD5/NPHP13) is the most common IFT-A SRTD; IFT140 (SRTD9) is
               second most common IFT-A SRTD
Chromosome   : 16p13.3
Inheritance  : Autosomal Recessive — biallelic LOF (compound het missense+truncating most common)
Prevalence   : ~1/200,000–500,000; ~50–100 families reported worldwide (2026);
               Second most common IFT-A SRTD after WDR19 (SRTD5/NPHP13);
               More common than dynein-2 tail SRTDs (SRTD15, SRTD17) but less common than
               SRTD3 (DYNC2H1) which accounts for ~50% of molecularly confirmed SRTD.

KEY DIAGNOSTIC DISTINCTION — EM MORPHOLOGY:
----------------------------------------------
IFT140 (SRTD9) is an IFT-A subunit, NOT a dynein-2 motor subunit.
EM finding: SHORT/STUBBY CILIA with IFT particle accumulation at the transition zone / cilia base.
This is DIFFERENT from SRTD3/8/11/15/17 (dynein-2 SRTDs) which show BULGING/CLUB CILIA TIPS
(IFT-B piles up at the tip because the dynein-2 retrograde motor cannot return IFT trains).
Teaching point: Club/bulging tip → dynein-2 SRTD. Short/stubby cilia → IFT-A SRTD (IFT140).
Gene panel is MANDATORY in both cases — EM alone cannot distinguish within each group.

Protein Structure — IFT140 (1462 aa; IFT-A complex scaffold)
-------------------------------------------------------------
Domain 1: N-terminal WD40 β-propeller domain (aa 1–500) — IFT-A assembly platform;
          WDR35/IFT121-binding interface; most pathogenic missense variants cluster here;
          ~10 WD40 repeats forming an extended β-propeller
Domain 2: Central linker / IFT-B interaction region (aa 501–900) — connects WD40 to TPR;
          contacts IFT-B complex; dynein-2 (DYNC2H1) heavy chain docking; moderate variant zone
Domain 3: C-terminal TPR (tetratricopeptide repeat) domain (aa 901–1462) — WDR19/IFT144-binding
          interface; IFT-A complex nucleation; mutations here → mild-moderate SRTD9 or CED-like

Key pathogenic variant classes (IFT140):
1. WD40 propeller missense (aa 1–500): compound het; moderate SRTD9; most common class
2. Linker region missense (aa 501–900): moderate-severe; IFT-B interaction impaired
3. Truncating/null (any domain): biallelic null → SRPS spectrum; most severe
4. Hypomorphic WD40 edge (aa 400–500): mild; surviving adults; renal-dominant late presentation
5. C-terminal TPR missense (aa 901–1462): mild-moderate; partial WDR19 binding preserved
"""

import random
import math

SEED = 391
N    = 40   # 40-patient educational cohort

rng  = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _pct(n, total=N):
    return round(n / total * 100)

def _split(total, *fractions):
    """Distribute 'total' into len(fractions) buckets deterministically."""
    buckets = [round(total * f) for f in fractions]
    diff = total - sum(buckets)
    buckets[0] += diff
    return buckets

# ── patient-level data (fixed seed) ──────────────────────────────────────────
patients = []
ethnicities = [
    ('European',                        0.32),   # compound het; most common in published cohorts
    ('Middle Eastern / North African',  0.28),   # consanguineous homozygous
    ('South Asian',                     0.18),
    ('East Asian',                      0.10),
    ('Latin American',                  0.07),
    ('Other / Unknown',                 0.05),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het missense + truncating',
    'Homozygous missense (WD40 propeller)',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense',
    'Homozygous hypomorphic (WD40 edge)',
]
allele_weights = [0.38, 0.27, 0.15, 0.13, 0.07]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.20, 0.48, 0.32])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial', 'None'],
        weights=[0.22, 0.20, 0.06, 0.52])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.15, 0.12, 0.73])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.40, 0.60])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.12, 0.06, 0.82])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.07, 0.05, 0.88])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.30, 0.12, 0.58])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.52, 0.20, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.18, 0.30, 0.52])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.20, 0.22, 0.38, 0.20])[0]

    misdiagnosis = rng.choices(
        ['EVC (Ellis-van Creveld)', 'Thanatophoric dysplasia (prenatal)',
         'BBS (Bardet-Biedl)', 'SRTD3/DYNC2H1 (gene panel corrected)', 'SRTD5/WDR19 (gene panel corrected)', 'None'],
        weights=[0.08, 0.07, 0.05, 0.12, 0.07, 0.61])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.36, 0.28, 0.22, 0.14])[0]

    sex = rng.choice(['M', 'F'])

    patients.append(dict(
        ethnicity    = eth,
        allele       = allele,
        null_allele  = null,
        thorax_sev   = thorax_sev,
        poly_status  = poly_status,
        renal        = renal,
        ckd_stage    = ckd_stage,
        retinal      = retinal,
        hepatic      = hepatic,
        veptr        = veptr,
        tx_renal     = tx_renal,
        resp_mgmt    = resp_mgmt,
        presentation = presentation,
        misdiagnosis = misdiagnosis,
        age_dx       = age_dx,
        sex          = sex,
    ))

# ── aggregate counts ──────────────────────────────────────────────────────────
def _cnt(field, value):
    return sum(1 for p in patients if p[field] == value)

def _cnt_partial(field, value):
    return sum(1 for p in patients if value in p[field])

thorax_severe_n   = _cnt_partial('thorax_sev', 'Severe')
polydactyly_n     = sum(1 for p in patients if p['poly_status'] != 'None')
renal_any_n       = sum(1 for p in patients if p['renal'] != 'None')
retinal_any_n     = sum(1 for p in patients if p['retinal'] != 'None')
hepatic_chf_n     = sum(1 for p in patients if p['hepatic'] != 'None')
veptr_any_n       = sum(1 for p in patients if p['veptr'] != 'Observation only')
transplant_done_n = _cnt('tx_renal', 'Renal transplant (done)')
misdiagnosis_n    = sum(1 for p in patients if p['misdiagnosis'] != 'None')

sex_M = sum(1 for p in patients if p['sex'] == 'M')
sex_F = N - sex_M

age_dx_0_1   = _cnt('age_dx', '0–1 yr (neonatal)')
age_dx_2_5   = _cnt('age_dx', '2–5 yr (infant)')
age_dx_6_10  = _cnt('age_dx', '6–10 yr (child)')
age_dx_11_16 = _cnt('age_dx', '11–16 yr (teen)')

# ── public API helpers ────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease":    "IFT140 Short-Rib Thoracic Dysplasia 9 (SRTD9 / ATD9)",
        "gene":       "IFT140 (*614620) — 16p13.3 — 1462 aa — IFT-A Complex Core Structural Subunit",
        "omim_gene":  "614620",
        "omim_disease": "266920",
        "chromosome": "16p13.3",
        "cohort_n":   N,
        "seed":       SEED,
        "kpis": {
            "thorax_severe_n":    thorax_severe_n,
            "thorax_severe_pct":  _pct(thorax_severe_n),
            "polydactyly_n":      polydactyly_n,
            "polydactyly_pct":    _pct(polydactyly_n),
            "renal_any_n":        renal_any_n,
            "renal_any_pct":      _pct(renal_any_n),
            "retinal_any_n":      retinal_any_n,
            "retinal_any_pct":    _pct(retinal_any_n),
            "hepatic_chf_n":      hepatic_chf_n,
            "hepatic_chf_pct":    _pct(hepatic_chf_n),
            "veptr_any_n":        veptr_any_n,
            "veptr_any_pct":      _pct(veptr_any_n),
            "transplant_done_n":  transplant_done_n,
            "misdiagnosis_n":     misdiagnosis_n,
            "misdiagnosis_pct":   _pct(misdiagnosis_n),
        },
        "sex_split":  {"M": sex_M, "F": sex_F},
        "age_distribution": {
            "dx_0_1yr":   age_dx_0_1,
            "dx_2_5yr":   age_dx_2_5,
            "dx_6_10yr":  age_dx_6_10,
            "dx_11_16yr": age_dx_11_16,
        },
        "mechanism": (
            "IFT140 loss destabilises the IFT-A complex (6 subunits: WDR19/IFT144, IFT140, "
            "WDR35/IFT121, IFT122, TTC21B/IFT139, IFT43). Without IFT140, IFT-A cannot properly "
            "assemble at the cilia base or bind dynein-2 for retrograde IFT. Result: IFT-B cargo "
            "import into cilia is impaired at the transition zone; retrograde IFT train recycling "
            "fails. EM: SHORT/STUBBY CILIA with IFT-particle accumulation at the cilia base / "
            "transition zone — DISTINCT from dynein-2 SRTDs (club/bulging TIP). Hedgehog (Ihh/Shh) "
            "signalling fails in chondrocytes → NARROW THORAX (primary), short ribs, short limbs. "
            "Secondary: renal TIN/ESRD (~20–30% survivors), rod-cone dystrophy (~15–20%), CHF (~8–12%)."
        ),
        "key_distinction": (
            "IFT140 (SRTD9) belongs to the IFT-A COMPLEX — not the dynein-2 motor. This is the "
            "critical mechanistic distinction from SRTD3/8/11/15/17 (dynein-2 SRTDs). EM shows "
            "SHORT/STUBBY CILIA (not club/bulging tips). IFT140 is the second-largest IFT-A subunit "
            "(1462 aa; after WDR19/IFT144 at 1342 aa) and the second most common IFT-A SRTD. "
            "WD40 β-propeller domain (N-terminal) binds WDR35/IFT121; C-terminal TPR domain binds "
            "WDR19/IFT144 — IFT140 is the structural bridge between the two major IFT-A halves. "
            "Polydactyly is slightly more frequent (~45–50%) than in dynein-2 tail SRTDs. "
            "Renal/retinal involvement: moderate (between SRTD3/SRTD8 severity and SRTD17)."
        ),
        "ifta_subunit_table": [
            {"subunit": "WDR19/IFT144", "role": "Largest IFT-A subunit; scaffold for IFT-A complex; IFT140 C-TPR partner", "srtd": "SRTD5 (also NPHP13, CED1)", "omim_gene": "608151", "chr": "4p14",   "freq": "Most common IFT-A SRTD; ~1% of all SRTD"},
            {"subunit": "IFT140",       "role": "Core structural scaffold; WD40→WDR35 binding; TPR→WDR19 binding; dynein-2 DYNC2H1 docking", "srtd": "SRTD9",             "omim_gene": "614620", "chr": "16p13.3", "freq": "~0.5–1% of SRTD; 2nd most common IFT-A SRTD"},
            {"subunit": "WDR35/IFT121", "role": "IFT-A complex WD40 subunit; IFT140 WD40 partner; anterograde IFT-B bridge", "srtd": "SRTD7",             "omim_gene": "613602", "chr": "2p24.1",  "freq": "~0.3–0.5% of SRTD"},
            {"subunit": "TTC21B/IFT139","role": "IFT-A retrograde adapter; Hedgehog pathway (Gli processing); major NPHP gene", "srtd": "SRTD4 / NPHP12",  "omim_gene": "612014", "chr": "2q24.3",  "freq": "~0.5–1% of SRTD"},
            {"subunit": "IFT122",       "role": "IFT-A structural subunit; CED2-like cranioectodermal dysplasia", "srtd": "CED2-like",          "omim_gene": "606045", "chr": "3q21.3",  "freq": "very rare"},
            {"subunit": "IFT43",        "role": "Smallest IFT-A subunit; stabilises IFT-A complex", "srtd": "—",                  "omim_gene": "614068", "chr": "14q24.3", "freq": "not SRTD-associated"},
        ],
    }


def get_breakdown():
    def _dist(field):
        counts = {}
        for p in patients:
            v = p[field]
            counts[v] = counts.get(v, 0) + 1
        return [{"label": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    def _eth_dist():
        counts = {}
        for p in patients:
            e = p['ethnicity']
            counts[e] = counts.get(e, 0) + 1
        return [{"ethnicity": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    return {
        "thorax_distribution": _dist('thorax_sev'),
        "polydactyly_distribution": [
            {"label": "Postaxial (bilateral)",  "n": _cnt('poly_status', 'Postaxial (bilateral)')},
            {"label": "Postaxial (unilateral)", "n": _cnt('poly_status', 'Postaxial (unilateral)')},
            {"label": "Preaxial",               "n": _cnt('poly_status', 'Preaxial')},
            {"label": "None",                   "n": _cnt('poly_status', 'None')},
        ],
        "renal_distribution": _dist('renal'),
        "ckd_stage_distribution": [
            {"label": "CKD 4",         "n": _cnt('ckd_stage', 'CKD 4')},
            {"label": "CKD 5 / ESRD",  "n": _cnt('ckd_stage', 'CKD 5 / ESRD')},
            {"label": "None / not applicable", "n": _cnt('ckd_stage', 'None')},
        ],
        "retinal_distribution": _dist('retinal'),
        "hepatic_distribution": _dist('hepatic'),
        "veptr_distribution": _dist('veptr'),
        "presentation_distribution": _dist('presentation'),
        "misdiagnosis_distribution": _dist('misdiagnosis'),
        "allele_class_summary": _dist('allele'),
        "ethnicity_distribution": _eth_dist(),
        "treatment_renal": _dist('tx_renal'),
        "respiratory_management": _dist('resp_mgmt'),
        "top_variants": [
            {"variant": "p.Arg830Gln (c.2489G>A) — WD40 propeller repeat 11 (WDR35 interface) — European compound het — moderate SRTD9", "n": rng.randint(4, 6)},
            {"variant": "p.Cys769Ser (c.2306G>C) — WD40/linker junction (IFT-B interaction) — Middle Eastern homozygous — moderate-severe", "n": rng.randint(3, 5)},
            {"variant": "p.Gln1014Ter (c.3040C>T) — TPR domain truncating null — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala497Val (c.1490C>T) — WD40 propeller edge (aa 490–510) — South Asian hypomorphic — mild adult", "n": rng.randint(2, 3)},
            {"variant": "p.Thr1218Met (c.3653C>T) — C-terminal TPR (WDR19/IFT144 interface) — MENA homozygous — moderate", "n": rng.randint(1, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "IFT140 — Intraflagellar Transport 140",
            "also_known_as":    "IFT140; WDPCP? No. IFT140 = IFT140; also referenced as IFT-A core scaffold",
            "omim_gene":        "*614620",
            "chromosome":       "16p13.3",
            "protein_size":     "1462 aa",
            "protein_class":    "IFT-A complex core structural subunit — WD40 β-propeller + central linker + C-terminal TPR scaffold",
            "domain_structure":  "N-terminal WD40 β-propeller (aa 1–500; ~10 WD40 repeats; WDR35/IFT121-binding) + central linker/IFT-B contact region (aa 501–900) + C-terminal TPR domain (aa 901–1462; WDR19/IFT144-binding)",
            "function":         "Core IFT-A complex scaffold bridging WDR35/IFT121 (WD40 partner) and WDR19/IFT144 (TPR partner); IFT-A complex assembly; retrograde IFT cargo adapter; dynein-2 DYNC2H1 docking at cilia base",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "WDR35/IFT121 (SRTD7; N-terminal WD40 face) + WDR19/IFT144 (SRTD5; C-terminal TPR domain) + DYNC2H1 (SRTD3; linker region)",
            "expression":       "Cilia (primary non-motile 9+0); chondrocytes; renal tubular epithelium; retinal photoreceptor connecting cilium; developing long bones",
        },
        "disease_card": {
            "omim_disease":       "#266920",
            "full_name":          "Short-Rib Thoracic Dysplasia 9 With or Without Polydactyly",
            "alternative_names":  "SRTD9; Asphyxiating Thoracic Dystrophy 9 (ATD9); Jeune syndrome type 9",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":        "~45–50% (postaxial most common; preaxial rare ~5%; slightly more frequent than SRTD17)",
            "renal":              "TIN + corticomedullary cysts → ESRD in ~20–30% of survivors",
            "retinal":            "Rod-cone dystrophy in ~15–20% (connecting cilium IFT failure)",
            "hepatic_chf":        "Ductal plate malformation → CHF in ~8–12%",
            "situs_inversus":     "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":        "ABSENT — no brainstem/vermis midline defect",
            "em_finding":         "SHORT/STUBBY CILIA with IFT-particle accumulation at transition zone/cilia base — DISTINCT from dynein-2 SRTDs (club tip)",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); frequency similar to SRTD11",
            "renal_transplant":   "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":         "~1/200,000–500,000; ~50–100 families worldwide (2026); 2nd most common IFT-A SRTD",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD spectrum",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones, trident acetabulum",
            "3. GENE PANEL (mandatory first-line): Include IFT140 alongside DYNC2H1, WDR19, WDR35, TTC21B, IFT172; WES if panel negative",
            "4. WES + CNV analysis: IFT140 is 1462 aa; large gene — small indels/CNVs may escape standard panel; CNV analysis mandatory",
            "5. Skeletal ciliopathy EM (skin/fibroblast biopsy): SHORT/STUBBY CILIA with IFT accumulation at cilia base/TZ — DIFFERENT from dynein-2 SRTDs (club tip); EM morphology helps direct gene panel interpretation",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "7. ERG annually from age 6 (rod-cone dystrophy ~15–20%); ophthalmology referral",
            "8. Liver USS + APRI if hepatic signs (CHF ~8–12%); avoid hepatotoxins",
            "9. Echocardiography (CHD less common than EVC; exclude secondary cardiac effects of chronic hypoxia)",
            "10. MLPA/CNV for IFT140 deletions if sequencing negative but phenotype compelling",
        ],
        "mechanism_glossary": [
            {"term": "IFT140",          "definition": "Intraflagellar Transport 140 — 1462 aa core IFT-A complex scaffold subunit. N-terminal WD40 β-propeller binds WDR35/IFT121; C-terminal TPR domain binds WDR19/IFT144. Structural bridge between the two major IFT-A halves."},
            {"term": "IFT-A complex",   "definition": "6-subunit retrograde adapter complex (WDR19/IFT144, IFT140, WDR35/IFT121, TTC21B/IFT139, IFT122, IFT43) required for retrograde IFT (tip→base) via dynein-2 and for IFT-B import at the cilia base. Loss → short/stubby cilia."},
            {"term": "Retrograde IFT",  "definition": "Transport of IFT trains from the ciliary tip back to the cell body, powered by cytoplasmic dynein-2. IFT-A complex is the adapter that recruits dynein-2 (via DYNC2H1) to IFT trains at the tip for retrograde run."},
            {"term": "Short/stubby cilia (EM)", "definition": "EM finding in IFT-A SRTDs including IFT140 (SRTD9): cilia fail to grow to normal length; IFT particles accumulate at the transition zone/cilia base rather than bulging at the tip. CONTRAST: dynein-2 SRTDs show CLUB/BULGING tips."},
            {"term": "Club/bulging ciliary tip (EM)", "definition": "EM finding in DYNEIN-2 SRTDs (SRTD3/8/11/15/17): IFT-B piles up at the ciliary tip because the dynein-2 retrograde motor cannot bring trains back. Tip swells/bulges. NOT seen in IFT-A SRTDs like IFT140 (SRTD9)."},
            {"term": "Transition zone (TZ)", "definition": "Y-link region at the cilia base between the basal body and the axoneme; acts as a gate for IFT entry. In IFT-A SRTDs, IFT particles often accumulate here rather than entering the axoneme properly."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling processed in primary cilia. IFT140 loss → short cilia → impaired Gli2 activation and Gli3R processing → Hedgehog signal failure → short ribs, narrow thorax, short limbs."},
            {"term": "Cell-autonomous",   "definition": "The IFT defect is intrinsic to the organ's own cells. After renal transplantation, the donor kidney (donor IFT140+) functions normally — CURATIVE for renal disease, no recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Arg830Gln",  "domain": "WD40 β-propeller repeat 11 (WDR35/IFT121 interface; aa 820–840)", "consequence": "Disrupts WDR35/IFT121 binding; European compound het; moderate SRTD9; most common class",     "ethnicity": "European (compound het)"},
            {"variant": "p.Cys769Ser",  "domain": "WD40/linker junction (IFT-B interaction surface; aa 760–780)",      "consequence": "Disulphide disruption; impairs linker-IFT-B contact; Middle Eastern homozygous; moderate-severe", "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Gln1014Ter", "domain": "C-terminal TPR domain (truncating null; aa 1014)",                   "consequence": "Truncates WDR19/IFT144-binding TPR domain; null allele; SRPS spectrum if biallelic; pan-ethnic",   "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala497Val",  "domain": "WD40 β-propeller edge (aa 490–510; WD40/linker transition)",         "consequence": "Hypomorphic; partial WDR35 binding preserved; mild; adult survival; renal-dominant late",           "ethnicity": "South Asian (compound het)"},
            {"variant": "p.Thr1218Met", "domain": "C-terminal TPR (WDR19/IFT144 binding helix; aa 1210–1225)",          "consequence": "Disrupts IFT144 docking; MENA homozygous; moderate; partial IFT-A complex formation",              "ethnicity": "MENA (homozygous)"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months through growth",
            "2. Respiratory: neonatal mechanical ventilation if severe thoracic restriction; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous IFT defect)",
            "4. Retinal: annual ERG from age 6 (rod-cone dystrophy ~15–20%, higher than SRTD17); ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF (~8–12%); hepatology referral; avoid hepatotoxins; portal hypertension monitoring",
            "6. Genetics: cascade testing of parents + siblings; prenatal/preimplantation genetics (25% recurrence); consanguinity risk counselling",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",      "key_difference": "Most common SRTD (~50%); DYNEIN-2 motor (not IFT-A); EM: CLUB cilia (bulging tip) vs SHORT cilia in IFT140; gene panel resolves"},
            {"disease": "SRTD5 / NPHP13 (WDR19)", "key_difference": "Most common IFT-A SRTD (same mechanistic group as IFT140); WDR19 binds IFT140 C-TPR domain; broader phenotype (CED ectodermal features in WDR19 null); gene panel resolves"},
            {"disease": "SRTD7 (WDR35/IFT121)", "key_difference": "Same IFT-A complex; WDR35 binds IFT140 WD40 N-terminal; direct binding partner; both cause IFT-A SRTD; gene panel differentiates"},
            {"disease": "SRTD8 (WDR60)",        "key_difference": "DYNEIN-2 intermediate chain (not IFT-A); EM: CLUB cilia vs SHORT cilia; retinal/renal more common in WDR60; gene panel"},
            {"disease": "SRTD11 (WDR34)",       "key_difference": "DYNEIN-2 intermediate chain (not IFT-A); EM: CLUB cilia vs SHORT cilia; gene panel"},
            {"disease": "SRTD17 (TCTEX1D2)",    "key_difference": "DYNEIN-2 light chain (not IFT-A); extremely rare vs IFT140; EM: CLUB tip vs SHORT cilia; less renal/retinal than IFT140"},
            {"disease": "Ellis-van Creveld (EVC)", "key_difference": "CHD present in ~60% EVC (rare in SRTD9); EVC ectodermal features (nails, teeth); IFT-A confirmed by gene panel"},
            {"disease": "BBS (Bardet-Biedl)",   "key_difference": "No narrow thorax in BBS; obesity + hypogonadism absent in SRTD9; BBS involves BBSome/IFT-B — different IFT complex"},
            {"disease": "NPHP (nephronophthisis)", "key_difference": "NPHP → renal is PRIMARY (not secondary); no narrow thorax/short ribs in NPHP; different ciliary compartment (TZ not IFT)"},
            {"disease": "Joubert syndrome (JBTS)", "key_difference": "Molar tooth sign (MTS) on MRI — ABSENT in SRTD9; cerebellar vermis hypoplasia in JBTS; MTS never occurs in SRTD"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD9 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
