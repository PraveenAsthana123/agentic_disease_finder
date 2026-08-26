"""
WDR19 Short-Rib Thoracic Dysplasia 5 (SRTD5) — Jeune Asphyxiating Thoracic Dystrophy Type 5
==============================================================================================
Primary Gene : WDR19 (*608151) — 4p14; ~1342 aa; WD Repeat Domain 19 (also: IFT144, KIAA1638)
               WDR19/IFT144 is the LARGEST subunit of the IFT-A complex (6-subunit retrograde
               adapter complex: WDR19/IFT144, IFT140, WDR35/IFT121, IFT122, TTC21B/IFT139, IFT43).
               WDR19 scaffolds the CORE of IFT-A: its WD40 β-propeller forms the central
               platform on which IFT140 (SRTD9, TPR domain contacts) and IFT122 dock.
               WDR19 loss → IFT-A complex instability → retrograde IFT impaired + IFT-B import
               defect at cilia base → SHORT/STUBBY CILIA → Hedgehog (Ihh/Shh) failure →
               NARROW THORAX, short ribs, short limbs.
               IFT-A complex hierarchy: WDR19/IFT144 (SRTD5) + IFT140 (SRTD9) + WDR35/IFT121
               (SRTD7) + TTC21B/IFT139 (SRTD4/NPHP12) + IFT122 (CED2-like) + IFT43.
Disease OMIM : #614376 — Short-Rib Thoracic Dysplasia 5 With or Without Polydactyly (SRTD5)
               Also called: Asphyxiating Thoracic Dystrophy 5 (ATD5); Jeune syndrome type 5
               SAME gene also causes:
               • NPHP13 (#614377) — nephronophthisis type 13 (renal-dominant, no/mild thorax)
               • CED1 overlap — cranioectodermal dysplasia type 1 (Sensenbrenner syndrome features)
               Severe biallelic null → Short-Rib Polydactyly Syndrome (SRPS) spectrum
               WDR19/IFT144 is THE MOST COMMON IFT-A SRTD (~1% of all molecularly confirmed SRTD)
Chromosome   : 4p14
Inheritance  : Autosomal Recessive — biallelic LOF (compound het missense+truncating most common)
Prevalence   : ~1/150,000–300,000; ~100–200 families reported worldwide (2026);
               Most common IFT-A SRTD; ~2–3× more common than IFT140 (SRTD9);
               Less common than DYNC2H1 (SRTD3, ~50% of all SRTD) but the dominant IFT-A gene

KEY DIAGNOSTIC DISTINCTION — EM MORPHOLOGY + ECTODERMAL FEATURES:
--------------------------------------------------------------------
WDR19 (SRTD5) is an IFT-A subunit, NOT a dynein-2 motor subunit.
EM finding: SHORT/STUBBY CILIA (same as all IFT-A SRTDs).
DISTINGUISHING FEATURE vs other IFT-A SRTDs: ECTODERMAL FEATURES (CED-like) in hypomorphic /
null alleles — sparse hair (hypotrichosis), hypodontia/small teeth, dental anomalies — present
in ~20–30% of SRTD5 patients. This overlap with CED1 (Sensenbrenner syndrome) is unique to
WDR19 (and IFT122) among SRTD genes — NOT seen in SRTD9/7/4. Gene panel distinguishes.
Teaching point: IFT-A SRTD + ectodermal features → think WDR19 (SRTD5) first.

Protein Structure — WDR19/IFT144 (1342 aa; IFT-A complex LARGEST subunit)
---------------------------------------------------------------------------
Domain 1: N-terminal WD40 β-propeller repeat cluster (aa 1–500) — IFT-A assembly scaffold;
          contacts IFT43 (peripheral) and IFT122; most common missense variants in SRTD5;
          ~12 WD40 repeats forming a multi-blade β-propeller; homozygous truncating here → SRPS
Domain 2: Central WD40 β-propeller / IFT140-contact platform (aa 501–900) — direct IFT140
          (SRTD9) C-terminal TPR domain-binding interface; IFT-B complex contacts; DYNC2H1 docking;
          mutations here → moderate-severe SRTD5; NPHP13 alleles cluster here (hypomorphic)
Domain 3: C-terminal α-helical regulatory tail (aa 901–1342) — IFT-A complex nucleation;
          contacts TTC21B/IFT139 (SRTD4); ectodermal-phenotype variants cluster here (CED overlap)

Key pathogenic variant classes (WDR19):
1. N-terminal WD40 compound het missense (aa 1–500): most common class; moderate SRTD5
2. Central WD40 homozygous missense (aa 501–900): consanguineous; moderate-severe; NPHP13 overlap
3. Truncating/null (any domain): biallelic null → SRPS spectrum; most severe
4. C-terminal hypomorphic missense (aa 901–1342): mild; NPHP13-dominant (renal only); CED overlap
5. Compound het truncating + hypomorphic: mixed severity; thorax + renal ± ectodermal
"""

import random
import math

SEED = 393
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
    ('European',                        0.30),   # compound het; most common published
    ('Middle Eastern / North African',  0.30),   # consanguineous homozygous; high frequency
    ('South Asian',                     0.20),
    ('East Asian',                      0.10),
    ('Latin American',                  0.06),
    ('Other / Unknown',                 0.04),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het missense + truncating',
    'Homozygous missense (central WD40)',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense',
    'Homozygous hypomorphic (C-terminal; NPHP13-dominant)',
]
allele_weights = [0.35, 0.28, 0.15, 0.14, 0.08]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.22, 0.46, 0.32])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial', 'None'],
        weights=[0.25, 0.22, 0.08, 0.45])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.18, 0.12, 0.70])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.38, 0.62])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.15, 0.08, 0.77])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.10, 0.05, 0.85])[0]

    ectodermal = rng.choices(
        ['Sparse hair + hypodontia (CED-like)', 'Sparse hair only', 'Hypodontia / small teeth only', 'None'],
        weights=[0.12, 0.10, 0.08, 0.70])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.32, 0.13, 0.55])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.50, 0.22, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.19, 0.32, 0.49])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.22, 0.24, 0.36, 0.18])[0]

    misdiagnosis = rng.choices(
        ['EVC (Ellis-van Creveld)', 'Thanatophoric dysplasia (prenatal)',
         'BBS (Bardet-Biedl)', 'SRTD3/DYNC2H1 (gene panel corrected)',
         'SRTD9/IFT140 (gene panel corrected)', 'CED/Sensenbrenner (ectodermal features)', 'None'],
        weights=[0.07, 0.06, 0.05, 0.10, 0.08, 0.07, 0.57])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.37, 0.28, 0.21, 0.14])[0]

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
        ectodermal   = ectodermal,
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

thorax_severe_n    = _cnt_partial('thorax_sev', 'Severe')
polydactyly_n      = sum(1 for p in patients if p['poly_status'] != 'None')
renal_any_n        = sum(1 for p in patients if p['renal'] != 'None')
retinal_any_n      = sum(1 for p in patients if p['retinal'] != 'None')
hepatic_chf_n      = sum(1 for p in patients if p['hepatic'] != 'None')
ectodermal_any_n   = sum(1 for p in patients if p['ectodermal'] != 'None')
veptr_any_n        = sum(1 for p in patients if p['veptr'] != 'Observation only')
transplant_done_n  = _cnt('tx_renal', 'Renal transplant (done)')
misdiagnosis_n     = sum(1 for p in patients if p['misdiagnosis'] != 'None')

sex_M = sum(1 for p in patients if p['sex'] == 'M')
sex_F = N - sex_M

age_dx_0_1   = _cnt('age_dx', '0–1 yr (neonatal)')
age_dx_2_5   = _cnt('age_dx', '2–5 yr (infant)')
age_dx_6_10  = _cnt('age_dx', '6–10 yr (child)')
age_dx_11_16 = _cnt('age_dx', '11–16 yr (teen)')

# ── public API helpers ────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease":    "WDR19 Short-Rib Thoracic Dysplasia 5 (SRTD5 / ATD5)",
        "gene":       "WDR19 (*608151) — 4p14 — 1342 aa — IFT-A Complex LARGEST Subunit (also: IFT144, KIAA1638)",
        "omim_gene":  "608151",
        "omim_disease": "614376",
        "omim_disease_nphp13": "614377",
        "chromosome": "4p14",
        "cohort_n":   N,
        "seed":       SEED,
        "kpis": {
            "thorax_severe_n":     thorax_severe_n,
            "thorax_severe_pct":   _pct(thorax_severe_n),
            "polydactyly_n":       polydactyly_n,
            "polydactyly_pct":     _pct(polydactyly_n),
            "renal_any_n":         renal_any_n,
            "renal_any_pct":       _pct(renal_any_n),
            "retinal_any_n":       retinal_any_n,
            "retinal_any_pct":     _pct(retinal_any_n),
            "hepatic_chf_n":       hepatic_chf_n,
            "hepatic_chf_pct":     _pct(hepatic_chf_n),
            "ectodermal_any_n":    ectodermal_any_n,
            "ectodermal_any_pct":  _pct(ectodermal_any_n),
            "veptr_any_n":         veptr_any_n,
            "veptr_any_pct":       _pct(veptr_any_n),
            "transplant_done_n":   transplant_done_n,
            "misdiagnosis_n":      misdiagnosis_n,
            "misdiagnosis_pct":    _pct(misdiagnosis_n),
        },
        "sex_split":  {"M": sex_M, "F": sex_F},
        "age_distribution": {
            "dx_0_1yr":   age_dx_0_1,
            "dx_2_5yr":   age_dx_2_5,
            "dx_6_10yr":  age_dx_6_10,
            "dx_11_16yr": age_dx_11_16,
        },
        "mechanism": (
            "WDR19/IFT144 loss destabilises the IFT-A complex (6 subunits: WDR19/IFT144, IFT140, "
            "WDR35/IFT121, IFT122, TTC21B/IFT139, IFT43). WDR19 is the LARGEST IFT-A subunit (1342 aa) "
            "and the central scaffold on which IFT140 (C-TPR domain) and IFT122 dock. Without WDR19, "
            "IFT-A cannot assemble at the cilia base or bind dynein-2 for retrograde IFT. Result: "
            "IFT-B cargo import into cilia is impaired at the transition zone; retrograde IFT train "
            "recycling fails. EM: SHORT/STUBBY CILIA with IFT-particle accumulation at the cilia "
            "base / transition zone — DISTINCT from dynein-2 SRTDs (club/bulging TIP). Hedgehog "
            "(Ihh/Shh) signalling fails in chondrocytes → NARROW THORAX (primary), short ribs, "
            "short limbs. Secondary: renal TIN/ESRD (~25–35% survivors), rod-cone dystrophy "
            "(~15–25%), CHF (~10–15%), ECTODERMAL FEATURES — sparse hair/hypodontia (~20–30%)."
        ),
        "key_distinction": (
            "WDR19 (SRTD5) is the MOST COMMON IFT-A SRTD (~1% of all SRTD; ~100–200 families 2026). "
            "It is the LARGEST IFT-A subunit (1342 aa) and the central assembly platform — IFT140 "
            "(SRTD9) C-terminal TPR domain docks directly onto WDR19. KEY CLINICAL DISTINCTION vs other "
            "IFT-A SRTDs: ECTODERMAL FEATURES (CED-like) — sparse hair, hypodontia, dental anomalies "
            "— in ~20–30% of SRTD5 patients. This overlap with CED1/Sensenbrenner syndrome is unique "
            "to WDR19 (and IFT122) among SRTD genes. WDR19 ALSO causes NPHP13 (#614377) — "
            "nephronophthisis without thoracic involvement (hypomorphic C-terminal alleles; renal-"
            "dominant). Polydactyly ~50–60% (higher than SRTD9 ~45–50%). EM: SHORT/STUBBY CILIA."
        ),
        "ifta_subunit_table": [
            {"subunit": "WDR19/IFT144", "role": "LARGEST IFT-A subunit; central scaffold; IFT140 C-TPR binding; IFT122 platform; dynein-2 DYNC2H1 contact", "srtd": "SRTD5 (also NPHP13, CED1-overlap)", "omim_gene": "608151", "chr": "4p14",    "freq": "MOST COMMON IFT-A SRTD; ~1% of all SRTD; ~100–200 families 2026"},
            {"subunit": "IFT140",       "role": "Core structural scaffold; WD40→WDR35 binding; TPR→WDR19 binding; dynein-2 DYNC2H1 docking", "srtd": "SRTD9",                         "omim_gene": "614620", "chr": "16p13.3", "freq": "2nd most common IFT-A SRTD; ~50–100 families 2026"},
            {"subunit": "WDR35/IFT121", "role": "IFT-A complex WD40 subunit; IFT140 WD40 N-terminal partner; anterograde IFT-B bridge", "srtd": "SRTD7",                            "omim_gene": "613602", "chr": "2p24.1",  "freq": "~0.3–0.5% of SRTD"},
            {"subunit": "TTC21B/IFT139","role": "IFT-A retrograde adapter; Hedgehog pathway (Gli processing); major NPHP gene", "srtd": "SRTD4 / NPHP12",                           "omim_gene": "612014", "chr": "2q24.3",  "freq": "~0.5–1% of SRTD"},
            {"subunit": "IFT122",       "role": "IFT-A structural subunit; CED2-like cranioectodermal dysplasia", "srtd": "CED2-like",                                               "omim_gene": "606045", "chr": "3q21.3",  "freq": "very rare"},
            {"subunit": "IFT43",        "role": "Smallest IFT-A subunit; stabilises IFT-A complex", "srtd": "—",                                                                     "omim_gene": "614068", "chr": "14q24.3", "freq": "not SRTD-associated"},
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
            {"label": "CKD 4",              "n": _cnt('ckd_stage', 'CKD 4')},
            {"label": "CKD 5 / ESRD",       "n": _cnt('ckd_stage', 'CKD 5 / ESRD')},
            {"label": "None / not applicable", "n": _cnt('ckd_stage', 'None')},
        ],
        "retinal_distribution": _dist('retinal'),
        "hepatic_distribution": _dist('hepatic'),
        "ectodermal_distribution": _dist('ectodermal'),
        "veptr_distribution": _dist('veptr'),
        "presentation_distribution": _dist('presentation'),
        "misdiagnosis_distribution": _dist('misdiagnosis'),
        "allele_class_summary": _dist('allele'),
        "ethnicity_distribution": _eth_dist(),
        "treatment_renal": _dist('tx_renal'),
        "respiratory_management": _dist('resp_mgmt'),
        "top_variants": [
            {"variant": "p.Ser1116Leu (c.3347C>T) — C-terminal WD40 repeat (IFT140-TPR contact; aa 1110–1130) — European compound het — moderate SRTD5", "n": rng.randint(4, 7)},
            {"variant": "p.Arg787Trp (c.2359C>T) — Central WD40 platform (IFT140-binding surface; aa 780–800) — Middle Eastern homozygous — moderate-severe SRTD5", "n": rng.randint(4, 6)},
            {"variant": "p.Glu251Ter (c.751G>T) — N-terminal WD40 truncating null — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala388Thr (c.1162G>A) — N-terminal WD40 edge (aa 380–400) — South Asian hypomorphic — mild; NPHP13-dominant", "n": rng.randint(2, 4)},
            {"variant": "p.Arg897Gln (c.2690G>A) — Central/C-tail junction (TTC21B contact; aa 890–910) — MENA homozygous — moderate; CED-overlap ectodermal", "n": rng.randint(2, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "WDR19 — WD Repeat Domain 19 (also: IFT144, KIAA1638)",
            "also_known_as":    "IFT144; KIAA1638; DYF-2 (C. elegans ortholog)",
            "omim_gene":        "*608151",
            "chromosome":       "4p14",
            "protein_size":     "1342 aa",
            "protein_class":    "IFT-A complex LARGEST subunit — multi-blade WD40 β-propeller scaffold (N-terminal + central domains) + C-terminal α-helical regulatory tail",
            "domain_structure":  "N-terminal WD40 β-propeller (aa 1–500; ~12 WD40 repeats; IFT43 + IFT122 scaffold) + Central WD40 platform (aa 501–900; IFT140 C-TPR docking; IFT-B contact; DYNC2H1 interface) + C-terminal α-helical tail (aa 901–1342; TTC21B/IFT139 contact; CED-overlap variants here)",
            "function":         "Central scaffold of the IFT-A complex (6-subunit retrograde adapter); binds IFT140 (SRTD9) via WD40 central platform; nucleates IFT-A assembly; retrograde IFT cargo adapter; dynein-2 DYNC2H1 docking; required for IFT-B import at cilia base transition zone",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT140 (SRTD9; C-terminal TPR docks onto WDR19 central WD40) + IFT122 (N-terminal WD40 platform) + TTC21B/IFT139 (SRTD4; C-tail contact) + DYNC2H1 (SRTD3; central platform)",
            "expression":       "Cilia (primary non-motile 9+0); chondrocytes; renal tubular epithelium (NPHP13 presentation); retinal photoreceptor connecting cilium; ectodermal cells (hair follicles, dental lamina — CED overlap); developing long bones",
            "allelic_diseases":  "SRTD5 (#614376) — thoracic + polydactyly dominant; NPHP13 (#614377) — nephronophthisis renal-dominant (hypomorphic C-terminal alleles); CED1-overlap (ectodermal features in null/severe)",
        },
        "disease_card": {
            "omim_disease":       "#614376",
            "full_name":          "Short-Rib Thoracic Dysplasia 5 With or Without Polydactyly",
            "alternative_names":  "SRTD5; Asphyxiating Thoracic Dystrophy 5 (ATD5); Jeune syndrome type 5",
            "allelic_conditions": "NPHP13 (#614377) — same gene, renal-dominant phenotype; CED1-overlap with ectodermal features",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":        "~50–60% (postaxial most common; preaxial ~8%; highest polydactyly frequency among IFT-A SRTDs)",
            "renal":              "TIN + corticomedullary cysts → ESRD in ~25–35% of survivors (higher than SRTD9; overlap with NPHP13 alleles)",
            "retinal":            "Rod-cone dystrophy in ~15–25% (connecting cilium IFT failure; comparable to SRTD9)",
            "hepatic_chf":        "Ductal plate malformation → CHF in ~10–15%",
            "ectodermal":         "CED-LIKE FEATURES in ~20–30% — sparse hair (hypotrichosis), hypodontia, small/peg-shaped teeth — UNIQUE TO WDR19 (and IFT122) among SRTD genes; guides clinical suspicion to WDR19",
            "situs_inversus":     "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":        "ABSENT — no brainstem/vermis midline defect (contrast with JBTS ciliopathies)",
            "em_finding":         "SHORT/STUBBY CILIA with IFT-particle accumulation at transition zone/cilia base — SAME AS ALL IFT-A SRTDs; DISTINCT from dynein-2 SRTDs (club/bulging tip)",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); higher SRPS frequency than SRTD9 due to larger gene and more null alleles in literature",
            "renal_transplant":   "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":         "~1/150,000–300,000; ~100–200 families worldwide (2026); MOST COMMON IFT-A SRTD",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD spectrum",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones, trident acetabulum; check for ectodermal signs (hair, teeth)",
            "3. GENE PANEL (mandatory first-line): Include WDR19 alongside DYNC2H1, IFT140, WDR35, TTC21B, IFT172; WES if panel negative. WDR19 must be top IFT-A candidate.",
            "4. WES + CNV analysis: WDR19 is large (1342 aa, large genomic span at 4p14); CNV deletion analysis mandatory in panel-negative SRTD5-phenotype cases",
            "5. Skeletal ciliopathy EM (skin/fibroblast biopsy): SHORT/STUBBY CILIA with IFT accumulation at cilia base/TZ — same as all IFT-A SRTDs; gene panel needed to distinguish WDR19 from IFT140",
            "6. Ectodermal examination: sparse hair (hypotrichosis), hypodontia, small/peg-shaped teeth — if present, raises WDR19 probability significantly; refer to dermatology + dentistry",
            "7. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis (NPHP13 alleles may be renal-only presentation)",
            "8. ERG annually from age 6 (rod-cone dystrophy ~15–25%); ophthalmology referral",
            "9. Liver USS + APRI if hepatic signs (CHF ~10–15%); avoid hepatotoxins",
            "10. NPHP13 screen in WDR19 heterozygotes with unexplained nephronophthisis — compound het including hypomorphic C-terminal allele may present as isolated renal disease",
        ],
        "mechanism_glossary": [
            {"term": "WDR19/IFT144",       "definition": "WD Repeat Domain 19 — 1342 aa, largest IFT-A complex subunit. WD40 β-propeller central platform docks IFT140 (C-TPR) and IFT122; C-tail contacts TTC21B/IFT139. Loss → IFT-A complex cannot assemble → SHORT/STUBBY CILIA."},
            {"term": "IFT-A complex",       "definition": "6-subunit retrograde adapter complex (WDR19/IFT144, IFT140, WDR35/IFT121, TTC21B/IFT139, IFT122, IFT43) required for retrograde IFT (tip→base) via dynein-2 and for IFT-B import at the cilia base. WDR19 is the central scaffold."},
            {"term": "NPHP13",              "definition": "Allelic condition caused by hypomorphic WDR19 C-terminal alleles — presents as isolated nephronophthisis (renal interstitial nephritis → ESRD) WITHOUT significant thoracic dysplasia. Key: WDR19 can cause SRTD5 OR NPHP13 depending on allele severity."},
            {"term": "CED1-overlap",        "definition": "Cranioectodermal dysplasia type 1 (Sensenbrenner syndrome) features — sparse hair, hypodontia, small teeth — found in ~20–30% of WDR19 patients. Caused by WDR19 null/severe alleles affecting ectodermal primary cilia. UNIQUE to WDR19 (and IFT122) among SRTD genes."},
            {"term": "Short/stubby cilia (EM)", "definition": "EM finding in IFT-A SRTDs including WDR19 (SRTD5): cilia fail to grow to normal length; IFT particles accumulate at the transition zone/cilia base. CONTRAST: dynein-2 SRTDs (SRTD3/8/11/15/17) show CLUB/BULGING TIPS."},
            {"term": "Transition zone (TZ)", "definition": "Y-link region at the cilia base between the basal body and the axoneme; gate for IFT entry. In IFT-A SRTDs (including WDR19), IFT particles accumulate here rather than entering the axoneme."},
            {"term": "Hedgehog (Ihh/Shh)",  "definition": "Growth plate signalling processed in primary cilia. WDR19 loss → short cilia → impaired Gli2 activation and Gli3R processing → Hedgehog signal failure → short ribs, narrow thorax, short limbs."},
            {"term": "Cell-autonomous",     "definition": "The IFT defect is intrinsic to the organ's own cells. After renal transplantation (for NPHP13 renal failure), the donor kidney functions normally — CURATIVE, no recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Ser1116Leu", "domain": "C-terminal WD40 repeat (IFT140-TPR contact surface; aa 1110–1130)", "consequence": "Disrupts IFT140 C-TPR/WDR19 binding; European compound het; moderate SRTD5; most reported European class", "ethnicity": "European (compound het)"},
            {"variant": "p.Arg787Trp",  "domain": "Central WD40 platform (IFT140-binding surface; aa 780–800)",          "consequence": "Disrupts IFT140 docking on WDR19 central platform; Middle Eastern homozygous; moderate-severe SRTD5",       "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Glu251Ter",  "domain": "N-terminal WD40 β-propeller (truncating null; aa 251)",                "consequence": "Truncates IFT43/IFT122 scaffolding domain; biallelic null → SRPS spectrum; most severe",                    "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala388Thr",  "domain": "N-terminal WD40 edge (aa 380–400; WD40/central transition)",           "consequence": "Hypomorphic; partial IFT122 binding preserved; mild; renal-dominant NPHP13 presentation in adults",            "ethnicity": "South Asian (compound het)"},
            {"variant": "p.Arg897Gln",  "domain": "Central/C-tail junction (TTC21B/IFT139 contact helix; aa 890–910)",    "consequence": "Disrupts TTC21B docking; MENA homozygous; moderate SRTD5; ectodermal CED-like features co-present",            "ethnicity": "MENA (homozygous)"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months through growth",
            "2. Respiratory: neonatal mechanical ventilation if severe thoracic restriction; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous IFT defect). NPHP13 alleles: renal may be presenting/dominant feature",
            "4. Retinal: annual ERG from age 6 (rod-cone dystrophy ~15–25%); ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF (~10–15%); hepatology referral; avoid hepatotoxins; portal hypertension monitoring",
            "6. Ectodermal: dermatology referral for hypotrichosis; paediatric dentistry for hypodontia (dental prosthetics early — hypodontia causes feeding + speech delay); CED-like features do not worsen after childhood",
            "7. Genetics: cascade testing of parents + siblings; prenatal/preimplantation genetics (25% recurrence); NPHP13 allele counselling for C-terminal hypomorphic carriers; consanguinity risk counselling",
            "8. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology, dermatology, dentistry (ectodermal MDT unique to WDR19)",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",         "key_difference": "Most common SRTD (~50%); DYNEIN-2 motor (not IFT-A); EM: CLUB cilia (bulging tip) vs SHORT cilia in WDR19; NO ectodermal features in SRTD3; gene panel resolves"},
            {"disease": "SRTD9 (IFT140)",          "key_difference": "Same IFT-A complex; IFT140 C-TPR domain docks directly onto WDR19; both cause IFT-A SHORT CILIA; WDR19 (SRTD5) is more common; ectodermal features ONLY in WDR19 (not IFT140); gene panel differentiates"},
            {"disease": "SRTD7 (WDR35/IFT121)",    "key_difference": "Same IFT-A complex; WDR35 contacts IFT140 (not WDR19 directly); WDR19 has ectodermal CED-overlap; gene panel differentiates"},
            {"disease": "NPHP13 (WDR19)",          "key_difference": "SAME GENE — hypomorphic C-terminal WDR19 alleles cause isolated nephronophthisis (renal-dominant, no/mild thorax). Compound het truncating + hypomorphic → renal + thorax overlap. Gene determines allele severity"},
            {"disease": "CED1/Sensenbrenner (IFT122)", "key_difference": "Ectodermal features (hair + teeth) also in IFT122 (CED2-like); CED1 has LESS thoracic involvement than SRTD5; WDR19 has MORE renal/NPHP13 overlap; gene panel resolves"},
            {"disease": "SRTD8 (WDR60)",           "key_difference": "DYNEIN-2 intermediate chain (not IFT-A); EM: CLUB cilia vs SHORT cilia; NO ectodermal features in WDR60; gene panel"},
            {"disease": "Ellis-van Creveld (EVC)",  "key_difference": "CHD present in ~60% EVC (rare in SRTD5); EVC ectodermal features (nails, teeth) may overlap — KEY: CHD absent in WDR5; IFT-A confirmed by gene panel"},
            {"disease": "BBS (Bardet-Biedl)",       "key_difference": "No narrow thorax in BBS; obesity + hypogonadism absent in SRTD5; BBSome/IFT-B (not IFT-A) complex"},
            {"disease": "NPHP (other genes)",       "key_difference": "NPHP → renal is PRIMARY (not secondary); no narrow thorax/short ribs in non-WDR19 NPHP; check WDR19 in NPHP13 before diagnosing other NPHP genes"},
            {"disease": "Joubert syndrome (JBTS)",  "key_difference": "Molar tooth sign (MTS) on MRI — ABSENT in SRTD5/WDR19; cerebellar vermis hypoplasia in JBTS; MTS never occurs in SRTD"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD5 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("Ectodermal:", b["ectodermal_distribution"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
