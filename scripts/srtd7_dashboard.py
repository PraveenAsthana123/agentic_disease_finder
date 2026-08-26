"""
WDR35 Short-Rib Thoracic Dysplasia 7 (SRTD7) — Jeune Asphyxiating Thoracic Dystrophy Type 7
================================================================================================
Primary Gene : WDR35 (*613602) — 2p24.1; ~1181 aa; WD-Repeat Domain 35 / IFT121
               WDR35 (also called IFT121) is a core structural subunit of the IFT-A complex
               (6-subunit retrograde adapter: WDR19/IFT144, IFT140, WDR35/IFT121, IFT122,
               TTC21B/IFT139, IFT43). WDR35 is a 14-blade WD40 β-propeller. Its C-face
               directly docks onto the N-terminal WD40 domain of IFT140 (SRTD9), making WDR35
               and IFT140 direct structural partners. WDR35 also bridges the IFT-A complex to
               IFT-B anterograde machinery (anterograde bridge function). WDR35 loss → IFT-A
               complex instability (IFT140 loses WD40 docking partner) → retrograde IFT
               failure + IFT-B import defect at cilia base → SHORT/STUBBY CILIA (same class as
               SRTD5/WDR19, SRTD9/IFT140) → Hedgehog (Ihh/Shh) signalling failure in
               chondrocytes → narrow thorax, short ribs.
               IFT-A complex hierarchy: WDR19/IFT144 (SRTD5) + IFT140 (SRTD9) + WDR35/IFT121
               (SRTD7) + TTC21B/IFT139 (SRTD4/NPHP12) + IFT122 (CED2-like) + IFT43.
Disease OMIM : #614091 — Short-Rib Thoracic Dysplasia 7 With or Without Polydactyly (SRTD7)
               Also called: Asphyxiating Thoracic Dystrophy 7 (ATD7); Jeune syndrome type 7
               Severe biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum
               WDR35 is the third most common IFT-A SRTD (after WDR19/SRTD5 and IFT140/SRTD9)
               WDR35 is the DIRECT IFT140 WD40 BINDING PARTNER within the IFT-A complex
Chromosome   : 2p24.1
Inheritance  : Autosomal Recessive — biallelic LOF (compound het missense+truncating most common)
Prevalence   : ~1/300,000–700,000; ~30–50 families reported worldwide (2026);
               Third most common IFT-A SRTD after WDR19 (SRTD5/NPHP13) and IFT140 (SRTD9);
               Much less common than SRTD3 (DYNC2H1) which accounts for ~50% of all SRTD.

KEY DIAGNOSTIC DISTINCTION — EM MORPHOLOGY:
----------------------------------------------
WDR35 (SRTD7) is an IFT-A subunit, NOT a dynein-2 motor subunit.
EM finding: SHORT/STUBBY CILIA with IFT particle accumulation at the transition zone / cilia base.
This is DIFFERENT from SRTD3/8/11/15/17 (dynein-2 SRTDs) which show BULGING/CLUB CILIA TIPS
(IFT-B piles up at the tip because the dynein-2 retrograde motor cannot return IFT trains).
Teaching point: Club/bulging tip → dynein-2 SRTD. Short/stubby cilia → IFT-A SRTD (WDR35).
Gene panel is MANDATORY in both cases — EM alone cannot distinguish within each group.
WITHIN IFT-A SRTDs: WDR35 (SRTD7) cannot be distinguished from WDR19 (SRTD5) or IFT140 (SRTD9)
by EM alone — all show short/stubby cilia. Gene panel resolves.

Protein Structure — WDR35 (1181 aa; 14-blade WD40 β-propeller; IFT-A bridge)
-------------------------------------------------------------------------------
Domain 1: N-terminal extension (aa 1–120) — IFT-A complex entry point; stabilises N-terminal
          WD40 blades; N-terminal variants often hypomorphic; flexible loop contacts IFT43
Domain 2: WD40 β-propeller blades 1–7 (aa 121–580) — IFT-A assembly; connects to TTC21B/IFT139
          and IFT122 within the IFT-A complex; central β-propeller core; most pathogenic variants
Domain 3: WD40 β-propeller blades 8–14 / C-face (aa 581–1181) — DIRECT IFT140 SRTD9 BINDING;
          C-face of WDR35 docks onto N-terminal WD40 domain of IFT140; loss disrupts IFT140
          docking → IFT-A complex destabilisation; most severe clinical variants cluster here

Key pathogenic variant classes (WDR35):
1. Compound het missense + truncating (mixed): most common; European; moderate SRTD7
2. Homozygous missense (WD40 β-propeller): consanguineous MENA/South Asian; moderate-severe
3. Homozygous truncating (null allele): biallelic null → SRPS spectrum; most severe
4. Compound het two missense: moderate; IFT140-binding face partially preserved
5. Homozygous hypomorphic (N-terminal/edge WD40): mild; surviving adults; rare
"""

import random
import math

SEED = 395
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
    ('European',                        0.34),   # compound het; most common in published cohorts
    ('Middle Eastern / North African',  0.27),   # consanguineous homozygous
    ('South Asian',                     0.17),
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
    'Homozygous missense (WD40 β-propeller)',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense',
    'Homozygous hypomorphic (N-terminal)',
]
allele_weights = [0.40, 0.26, 0.16, 0.12, 0.06]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.18, 0.50, 0.32])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial', 'None'],
        weights=[0.20, 0.18, 0.04, 0.58])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.12, 0.10, 0.78])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.42, 0.58])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.10, 0.05, 0.85])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.06, 0.04, 0.90])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.30, 0.12, 0.58])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.50, 0.22, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.16, 0.30, 0.54])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.18, 0.22, 0.40, 0.20])[0]

    misdiagnosis = rng.choices(
        ['EVC (Ellis-van Creveld)', 'Thanatophoric dysplasia (prenatal)',
         'BBS (Bardet-Biedl)', 'SRTD3/DYNC2H1 (gene panel corrected)',
         'SRTD9/IFT140 (gene panel corrected)', 'SRTD5/WDR19 (gene panel corrected)', 'None'],
        weights=[0.08, 0.06, 0.04, 0.12, 0.09, 0.06, 0.55])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.34, 0.30, 0.22, 0.14])[0]

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
        "disease":    "WDR35 Short-Rib Thoracic Dysplasia 7 (SRTD7 / ATD7)",
        "gene":       "WDR35 (*613602) — 2p24.1 — 1181 aa — IFT-A Complex WD40 Subunit / IFT121",
        "omim_gene":  "613602",
        "omim_disease": "614091",
        "chromosome": "2p24.1",
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
            "WDR35/IFT121 (1181 aa) is a 14-blade WD40 β-propeller subunit of the IFT-A complex "
            "(6 subunits: WDR19/IFT144, IFT140, WDR35/IFT121, IFT122, TTC21B/IFT139, IFT43). "
            "WDR35 is the DIRECT structural binding partner of IFT140 (SRTD9): the C-face of "
            "WDR35's β-propeller docks onto the N-terminal WD40 domain of IFT140. This interaction "
            "is the molecular linchpin of the IFT-A complex half containing WDR35 and IFT140. "
            "WDR35 also bridges the IFT-A complex to IFT-B anterograde cargo (anterograde bridge). "
            "Without WDR35, IFT140 loses its WD40-face docking partner → IFT-A complex instability "
            "→ retrograde IFT failure + IFT-B import defect at cilia base. "
            "EM: SHORT/STUBBY CILIA with IFT-particle accumulation at the transition zone/base — "
            "DISTINCT from dynein-2 SRTDs (SRTD3/8/11/15/17) which show CLUB/BULGING TIPS. "
            "Hedgehog (Ihh/Shh) signalling fails in chondrocytes → NARROW THORAX (primary), "
            "short ribs, short limbs. Secondary: renal TIN/ESRD (~15–25% survivors), "
            "rod-cone dystrophy (~10–15%), CHF (~5–10%)."
        ),
        "key_distinction": (
            "WDR35 (SRTD7) is the DIRECT IFT140 (SRTD9) BINDING PARTNER within the IFT-A complex. "
            "This direct protein-protein interaction (WDR35 C-face β-propeller ↔ IFT140 N-terminal "
            "WD40 domain) is what makes the SRTD7/SRTD9 pair the closest molecular relatives in "
            "the IFT-A SRTD group. Both belong to the IFT-A complex (NOT dynein-2 motor). "
            "EM: SHORT/STUBBY CILIA (same as WDR19/SRTD5 and IFT140/SRTD9; all IFT-A SRTDs). "
            "WDR35 is the third most common IFT-A SRTD (~30–50 families; ~1/300K–700K). "
            "Polydactyly in ~40–45%. Renal/retinal involvement moderate (slightly less than "
            "SRTD9/IFT140; similar to or slightly less than SRTD5/WDR19). "
            "IFT-A SRTDs cannot be distinguished from each other by EM — gene panel is mandatory."
        ),
        "ifta_subunit_table": [
            {"subunit": "WDR19/IFT144", "role": "Largest IFT-A subunit; central WD40 scaffold; contacts IFT140 C-TPR, TTC21B C-tail, IFT122 N-terminal", "srtd": "SRTD5 (also NPHP13, CED1)", "omim_gene": "608151", "chr": "4p14",   "freq": "Most common IFT-A SRTD; ~1% of all SRTD"},
            {"subunit": "IFT140",       "role": "Core structural scaffold; N-terminal WD40 binds WDR35; C-terminal TPR binds WDR19; structural bridge between both IFT-A halves", "srtd": "SRTD9",             "omim_gene": "614620", "chr": "16p13.3", "freq": "~0.5–1% of SRTD; 2nd most common IFT-A SRTD"},
            {"subunit": "WDR35/IFT121", "role": "14-blade WD40 β-propeller; DIRECT IFT140 WD40-face binding partner; IFT-A↔IFT-B anterograde bridge", "srtd": "SRTD7",             "omim_gene": "613602", "chr": "2p24.1",  "freq": "~0.3–0.5% of SRTD; 3rd most common IFT-A SRTD"},
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
            {"variant": "p.Thr1205Met (c.3614C>T) — C-terminal WD40 propeller blade 14 (IFT140 C-face binding surface) — European compound het — moderate SRTD7", "n": rng.randint(4, 6)},
            {"variant": "p.Pro209Leu (c.626C>T) — N-terminal WD40 extension / blade 2 (IFT-A complex entry; IFT43 contact) — Middle Eastern homozygous — moderate-severe", "n": rng.randint(3, 5)},
            {"variant": "p.Arg549Ter (c.1645C>T) — central WD40 β-propeller (blade 7) truncating null — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala812Thr (c.2434G>A) — WD40 blade 10 edge (IFT-B anterograde bridge region) — South Asian hypomorphic — mild adult", "n": rng.randint(2, 3)},
            {"variant": "p.Cys983Tyr (c.2948G>A) — C-terminal WD40 blade 12 (IFT140 interface helix) — MENA homozygous — moderate", "n": rng.randint(1, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "WDR35 — WD-Repeat Domain 35 (also: IFT121 / IFT-A complex component)",
            "also_known_as":    "IFT121; WDR35; WD repeat-containing protein 35",
            "omim_gene":        "*613602",
            "chromosome":       "2p24.1",
            "protein_size":     "1181 aa",
            "protein_class":    "IFT-A complex WD40 subunit — 14-blade WD40 β-propeller + N-terminal extension; direct IFT140 (SRTD9) binding partner",
            "domain_structure":  "N-terminal extension (aa 1–120; IFT43 contact; IFT-A assembly) + WD40 β-propeller blades 1–7 (aa 121–580; IFT-B anterograde bridge; TTC21B/IFT139 and IFT122 contacts) + WD40 β-propeller blades 8–14 / C-face (aa 581–1181; DIRECT IFT140 SRTD9 WD40 N-terminal domain binding — molecular linchpin of this IFT-A half)",
            "function":         "IFT-A complex structural subunit; 14-blade WD40 β-propeller whose C-face directly docks onto IFT140's N-terminal WD40 domain (most critical IFT-A binary interaction in this sub-complex); bridges IFT-A complex to IFT-B anterograde cargo; required for retrograde IFT and anterograde cargo import at cilia base",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT140/SRTD9 (SRTD9; direct C-face WD40 docking — most critical binary contact) + WDR19/IFT144 (SRTD5; same IFT-A complex) + TTC21B/IFT139 (SRTD4/NPHP12) + IFT43 (N-terminal contact)",
            "expression":       "Cilia (primary non-motile 9+0); chondrocytes; renal tubular epithelium; retinal photoreceptor connecting cilium; developing long bones",
        },
        "disease_card": {
            "omim_disease":       "#614091",
            "full_name":          "Short-Rib Thoracic Dysplasia 7 With or Without Polydactyly",
            "alternative_names":  "SRTD7; Asphyxiating Thoracic Dystrophy 7 (ATD7); Jeune syndrome type 7",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":        "~40–45% (postaxial most common; preaxial rare ~3%; similar to IFT140-SRTD9, slightly less than WDR19-SRTD5)",
            "renal":              "TIN + corticomedullary cysts → ESRD in ~15–25% of survivors",
            "retinal":            "Rod-cone dystrophy in ~10–15% (connecting cilium IFT failure)",
            "hepatic_chf":        "Ductal plate malformation → CHF in ~5–10%",
            "situs_inversus":     "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":        "ABSENT — no brainstem/vermis midline defect",
            "em_finding":         "SHORT/STUBBY CILIA with IFT-particle accumulation at transition zone/cilia base — DISTINCT from dynein-2 SRTDs (club tip); SAME CLASS as SRTD5/WDR19 and SRTD9/IFT140",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); frequency similar to SRTD9/SRTD11",
            "renal_transplant":   "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":         "~1/300,000–700,000; ~30–50 families worldwide (2026); 3rd most common IFT-A SRTD",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD spectrum",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones, trident acetabulum",
            "3. GENE PANEL (mandatory first-line): Include WDR35 alongside DYNC2H1, WDR19, IFT140, TTC21B, IFT172; WES if panel negative",
            "4. WES + CNV analysis: WDR35 is 1181 aa; large gene — small indels/CNVs may escape standard panel; CNV analysis mandatory",
            "5. Skeletal ciliopathy EM (skin/fibroblast biopsy): SHORT/STUBBY CILIA with IFT accumulation at cilia base/TZ — DIFFERENT from dynein-2 SRTDs (club tip); SAME EM CLASS as SRTD5 and SRTD9 — gene panel essential within IFT-A group",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "7. ERG annually from age 6 (rod-cone dystrophy ~10–15%); ophthalmology referral",
            "8. Liver USS + APRI if hepatic signs (CHF ~5–10%); avoid hepatotoxins",
            "9. Echocardiography (CHD less common than EVC; exclude secondary cardiac effects of chronic hypoxia)",
            "10. MLPA/CNV for WDR35 deletions if sequencing negative but phenotype compelling",
        ],
        "mechanism_glossary": [
            {"term": "WDR35 / IFT121",  "definition": "WD-Repeat Domain 35 — 1181 aa, 14-blade WD40 β-propeller IFT-A complex subunit. C-face directly docks onto IFT140 N-terminal WD40 domain (most critical binary interaction in the WDR35-IFT140 half of IFT-A). N-terminal extension contacts IFT43. Central blades bridge IFT-A to IFT-B anterograde cargo."},
            {"term": "IFT-A complex",   "definition": "6-subunit retrograde adapter complex (WDR19/IFT144, IFT140, WDR35/IFT121, TTC21B/IFT139, IFT122, IFT43) required for retrograde IFT (tip→base) via dynein-2 and for IFT-B import at the cilia base. Loss → short/stubby cilia."},
            {"term": "Retrograde IFT",  "definition": "Transport of IFT trains from the ciliary tip back to the cell body, powered by cytoplasmic dynein-2. IFT-A complex is the adapter that recruits dynein-2 (via DYNC2H1) to IFT trains at the tip for the retrograde run."},
            {"term": "Short/stubby cilia (EM)", "definition": "EM finding in IFT-A SRTDs including WDR35 (SRTD7), IFT140 (SRTD9), WDR19 (SRTD5): cilia fail to grow to normal length; IFT particles accumulate at the transition zone/cilia base. CONTRAST: dynein-2 SRTDs (SRTD3/8/11/15/17) show CLUB/BULGING TIPS. Within IFT-A SRTDs, EM cannot distinguish — gene panel mandatory."},
            {"term": "Club/bulging ciliary tip (EM)", "definition": "EM finding in DYNEIN-2 SRTDs (SRTD3/8/11/15/17): IFT-B piles up at the ciliary tip because the dynein-2 retrograde motor cannot bring trains back. Tip swells/bulges. NOT seen in IFT-A SRTDs like WDR35 (SRTD7)."},
            {"term": "WDR35-IFT140 binary interaction", "definition": "The most critical molecular interaction for this SRTD7-SRTD9 sub-complex: the C-face of WDR35's 14-blade WD40 β-propeller docks onto the N-terminal WD40 domain of IFT140. This interaction is the linchpin of the WDR35+IFT140 half of the IFT-A complex. Pathogenic WDR35 variants in blades 8–14 preferentially disrupt this interaction."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling processed in primary cilia. WDR35 loss → short cilia → impaired Gli2 activation and Gli3R processing → Hedgehog signal failure → short ribs, narrow thorax, short limbs."},
            {"term": "Cell-autonomous",   "definition": "The IFT defect is intrinsic to the organ's own cells. After renal transplantation, the donor kidney (donor WDR35+) functions normally — CURATIVE for renal disease, no recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Thr1205Met",  "domain": "WD40 β-propeller blade 14 / C-face (IFT140 N-terminal WD40 binding surface; aa 1195–1215)", "consequence": "Disrupts direct IFT140 WD40 docking; European compound het; moderate SRTD7; most common class",     "ethnicity": "European (compound het)"},
            {"variant": "p.Pro209Leu",   "domain": "N-terminal WD40 extension / blade 2 (IFT-A complex entry; IFT43 contact; aa 200–220)",      "consequence": "Destabilises N-terminal IFT-A entry; impairs IFT43 contact; Middle Eastern homozygous; moderate-severe", "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Arg549Ter",   "domain": "WD40 β-propeller blade 7 (central; truncating null; aa 549)",                                "consequence": "Truncates central propeller; null allele; SRPS spectrum if biallelic; pan-ethnic",   "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala812Thr",   "domain": "WD40 blade 10 edge (IFT-B anterograde bridge region; aa 805–820)",                           "consequence": "Hypomorphic; partial IFT-B bridge preserved; mild; adult survival; renal-dominant late",           "ethnicity": "South Asian (compound het)"},
            {"variant": "p.Cys983Tyr",   "domain": "C-terminal WD40 blade 12 (IFT140 binding helix; aa 975–995)",                               "consequence": "Disulphide disruption; impairs IFT140 C-face docking; MENA homozygous; moderate",              "ethnicity": "MENA (homozygous)"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months through growth",
            "2. Respiratory: neonatal mechanical ventilation if severe thoracic restriction; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous IFT defect)",
            "4. Retinal: annual ERG from age 6 (rod-cone dystrophy ~10–15%); ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF (~5–10%); hepatology referral; avoid hepatotoxins; portal hypertension monitoring",
            "6. Genetics: cascade testing of parents + siblings; prenatal/preimplantation genetics (25% recurrence); consanguinity risk counselling",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",      "key_difference": "Most common SRTD (~50%); DYNEIN-2 motor (not IFT-A); EM: CLUB cilia (bulging tip) vs SHORT cilia in WDR35; gene panel resolves"},
            {"disease": "SRTD5 / NPHP13 (WDR19)", "key_difference": "Most common IFT-A SRTD (same class as WDR35); both show short/stubby cilia on EM — cannot distinguish by EM alone; WDR19 has unique ectodermal features (sparse hair, hypodontia) absent in SRTD7; gene panel required"},
            {"disease": "SRTD9 (IFT140)",       "key_difference": "WDR35 and IFT140 are DIRECT BINDING PARTNERS within IFT-A — the closest molecular relatives; both IFT-A; both short/stubby cilia; cannot distinguish by EM; gene panel essential; IFT140 is slightly more common (~50–100 families vs ~30–50)"},
            {"disease": "SRTD8 (WDR60)",        "key_difference": "DYNEIN-2 intermediate chain (not IFT-A); EM: CLUB cilia vs SHORT cilia; retinal/renal more common in WDR60; gene panel"},
            {"disease": "SRTD11 (WDR34)",       "key_difference": "DYNEIN-2 intermediate chain (not IFT-A); EM: CLUB cilia vs SHORT cilia; gene panel"},
            {"disease": "SRTD17 (TCTEX1D2)",    "key_difference": "DYNEIN-2 light chain (not IFT-A); extremely rare vs WDR35; EM: CLUB tip vs SHORT cilia; less renal/retinal than WDR35"},
            {"disease": "Ellis-van Creveld (EVC)", "key_difference": "CHD present in ~60% EVC (rare in SRTD7); EVC ectodermal features (nails, teeth); IFT-A confirmed by gene panel"},
            {"disease": "BBS (Bardet-Biedl)",   "key_difference": "No narrow thorax in BBS; obesity + hypogonadism absent in SRTD7; BBS involves BBSome/IFT-B — different IFT complex"},
            {"disease": "NPHP (nephronophthisis)", "key_difference": "NPHP → renal is PRIMARY (not secondary); no narrow thorax/short ribs in NPHP; different ciliary compartment (TZ not IFT)"},
            {"disease": "Joubert syndrome (JBTS)", "key_difference": "Molar tooth sign (MTS) on MRI — ABSENT in SRTD7; cerebellar vermis hypoplasia in JBTS; MTS never occurs in SRTD"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD7 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
