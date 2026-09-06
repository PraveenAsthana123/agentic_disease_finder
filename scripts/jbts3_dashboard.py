"""
AHI1 Joubert Syndrome Type 3 (JBTS3) — Jouberin / Transition Zone Scaffold
=============================================================================
Primary Gene : AHI1 (*608894) — 6q23.3; ~1196 aa; Abelson Helper Integration Site 1 (Jouberin)
               AHI1 (Jouberin) is a TRANSITION ZONE (TZ) scaffold protein.
               AHI1 forms a tripartite complex with NPHP1 (JBTS4/NPHP1) and NPHP4 at the
               ciliary transition zone Y-links. Loss of AHI1 → TZ gate assembly defective
               → SSTR3 (somatostatin receptor 3), MCHR1 (melanin-concentrating hormone
               receptor 1), and Smo (Smoothened) FAIL to enter neuronal cilia →
               impaired Hedgehog, somatostatin, and MCH receptor signalling in neurons →
               Molar Tooth Sign (MTS) on brain MRI, cerebellar vermis hypoplasia,
               oculomotor apraxia (OMA), neonatal hypotonia, cerebellar ataxia.
               Cilia are structurally intact (9+2 axoneme preserved) but TZ gate function
               is compromised — a pure gating defect, not a structural cilia defect.
Disease OMIM : #608629 — Joubert Syndrome 3 (JBTS3)
Chromosome   : 6q23.3
Inheritance  : Autosomal Recessive — biallelic LOF (homozygous consanguineous or compound het)
Prevalence   : ~10–15% of all Joubert syndrome cases; JBTS overall ~1/80,000–100,000 births;
               JBTS3 specific ~1/600,000–800,000 worldwide

Protein Structure — AHI1 / Jouberin (1196 aa; transition zone scaffold)
-------------------------------------------------------------------------
Domain 1: N-terminal WD40 beta-propeller (aa 1–400)   — 7 WD40 repeats; forms beta-propeller
           scaffold; NPHP1 binding surface; Arg830Trp founder allele is in the coiled-coil
           junction adjacent to this domain
Domain 2: Coiled-coil domain (aa 400–700)              — NPHP4 interaction; Arg830Trp
           (c.2488C>T) lies in the coiled-coil/WD40-junction; most common pathogenic cluster
Domain 3: C-terminal SH3 domain (aa 1050–1196)        — protein-protein interaction;
           Arg1041Trp cluster; retinal dystrophy-prominent alleles in this region

Key pathogenic variant classes (AHI1):
1. Homozygous Arg830Trp (c.2488C>T): Ashkenazi Jewish founder; ~1/92 carrier frequency in AJ;
   most common AHI1 variant worldwide; moderate-severe MTS + OMA; coiled-coil/WD40-junction
2. Compound het missense + truncating: European/Pan-ethnic; variable severity; most common non-AJ
3. Compound het two missense: European/MENA; moderate MTS; variable OMA
4. Compound het with splice variant: MENA/South Asian; variable; splice loss of TZ scaffold
5. Homozygous truncating (severe/null): biallelic null; severe MTS + retinal + OMA; consanguineous
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
    """Distribute 'total' into len(fractions) buckets deterministically."""
    buckets = [round(total * f) for f in fractions]
    diff = total - sum(buckets)
    buckets[0] += diff
    return buckets

# ── patient-level data (fixed seed) ──────────────────────────────────────────
patients = []
ethnicities = [
    ('Ashkenazi Jewish',               0.35),   # Arg830Trp founder dominant
    ('European',                       0.30),   # compound het most common
    ('Middle Eastern / North African', 0.20),   # consanguineous enrichment
    ('South Asian',                    0.08),
    ('East Asian',                     0.04),
    ('Other / Unknown',                0.03),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous Arg830Trp (Ashkenazi founder)',
    'Compound het missense + truncating',
    'Compound het two missense',
    'Compound het with splice variant',
    'Homozygous truncating (severe/null)',
]
allele_weights = [0.20, 0.30, 0.22, 0.18, 0.10]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'truncating (severe' in allele.lower() or allele == 'Homozygous truncating (severe/null)'

    mts_status = rng.choices(
        ['Present (classic Molar Tooth Sign on MRI)', 'Present (partial/subtle MTS)', 'Absent/not imaged'],
        weights=[0.80, 0.15, 0.05])[0]

    hypotonia = rng.choices(
        ['Neonatal hypotonia (severe)', 'Neonatal hypotonia (moderate)', 'Absent'],
        weights=[0.55, 0.33, 0.12])[0]

    oculomotor_apraxia = rng.choices(
        ['Present (OMA — horizontal gaze initiation failure)', 'Partial/mild OMA', 'Absent'],
        weights=[0.55, 0.20, 0.25])[0]

    ataxia = rng.choices(
        ['Cerebellar ataxia (gait + limb)', 'Gait ataxia only', 'Absent/too young to assess'],
        weights=[0.68, 0.22, 0.10])[0]

    id_severity = rng.choices(
        ['Moderate-severe ID', 'Mild-moderate ID', 'Borderline/normal cognition'],
        weights=[0.55, 0.27, 0.18])[0]

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Cone-rod dystrophy', 'Reduced ERG only', 'None'],
        weights=[0.30, 0.10, 0.12, 0.48])[0]

    breathing = rng.choices(
        ['Episodic apnea + hyperpnea (alternating)', 'Apnea alone (neonatal)', 'Hyperpnea alone', 'None'],
        weights=[0.35, 0.20, 0.10, 0.35])[0]

    renal = rng.choices(
        ['Renal cysts (structural)', 'Renal tubular anomaly', 'None'],
        weights=[0.05, 0.03, 0.92])[0]

    corpus_callosum = rng.choices(
        ['Agenesis (complete)', 'Hypoplasia (partial)', 'Normal'],
        weights=[0.12, 0.23, 0.65])[0]

    presentation = rng.choices(
        ['Neonatal hypotonia + breathing dysregulation', 'Developmental delay + ataxia (infant)',
         'Brain MRI abnormality found incidentally', 'Oculomotor apraxia noted at school age'],
        weights=[0.35, 0.38, 0.17, 0.10])[0]

    misdiagnosis = rng.choices(
        ['Dandy-Walker syndrome', 'Cerebellar hypoplasia NOS',
         'Joubert-like without gene diagnosis', 'Pontocerebellar hypoplasia',
         'Benign external hydrocephalus', 'None'],
        weights=[0.15, 0.12, 0.10, 0.06, 0.04, 0.53])[0]

    age_dx_mo = rng.choices(
        ['0–6 mo (neonatal)', '7–24 mo (infant)', '2–5 yr (early childhood)', '6–12 yr (school age)'],
        weights=[0.28, 0.35, 0.25, 0.12])[0]

    sex = rng.choice(['M', 'F'])

    patients.append(dict(
        ethnicity          = eth,
        allele_class       = allele,
        null_allele        = null,
        mts_status         = mts_status,
        hypotonia          = hypotonia,
        oculomotor_apraxia = oculomotor_apraxia,
        ataxia             = ataxia,
        id_severity        = id_severity,
        retinal            = retinal,
        breathing_dysregulation = breathing,
        renal              = renal,
        corpus_callosum    = corpus_callosum,
        age_diagnosis_mo   = age_dx_mo,
        initial_presentation = presentation,
        misdiagnosis       = misdiagnosis,
        sex                = sex,
    ))

# ── aggregate counts ──────────────────────────────────────────────────────────
def _cnt(field, value):
    return sum(1 for p in patients if p[field] == value)

def _cnt_partial(field, value):
    return sum(1 for p in patients if value in p[field])

mts_present_n         = sum(1 for p in patients if 'Present' in p['mts_status'])
hypotonia_n           = sum(1 for p in patients if p['hypotonia'] != 'Absent')
oma_n                 = sum(1 for p in patients if 'Present' in p['oculomotor_apraxia'] or 'Partial' in p['oculomotor_apraxia'])
ataxia_n              = sum(1 for p in patients if p['ataxia'] != 'Absent/too young to assess')
id_n                  = sum(1 for p in patients if 'ID' in p['id_severity'])
retinal_n             = sum(1 for p in patients if p['retinal'] != 'None')
breathing_n           = sum(1 for p in patients if p['breathing_dysregulation'] != 'None')
renal_n               = sum(1 for p in patients if p['renal'] != 'None')
corpus_callosum_n     = sum(1 for p in patients if p['corpus_callosum'] != 'Normal')
misdiagnosis_n        = sum(1 for p in patients if p['misdiagnosis'] != 'None')

sex_M = sum(1 for p in patients if p['sex'] == 'M')
sex_F = N - sex_M

age_0_6mo  = _cnt('age_diagnosis_mo', '0–6 mo (neonatal)')
age_7_24mo = _cnt('age_diagnosis_mo', '7–24 mo (infant)')
age_2_5yr  = _cnt('age_diagnosis_mo', '2–5 yr (early childhood)')
age_6_12yr = _cnt('age_diagnosis_mo', '6–12 yr (school age)')

# ── public API helpers ────────────────────────────────────────────────────────

def get_overview():
    def _eth_dist():
        counts = {}
        for p in patients:
            e = p['ethnicity']
            counts[e] = counts.get(e, 0) + 1
        return [{"ethnicity": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    def _allele_dist():
        counts = {}
        for p in patients:
            a = p['allele_class']
            counts[a] = counts.get(a, 0) + 1
        return [{"label": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    return {
        "disease":          "AHI1 Joubert Syndrome Type 3 (JBTS3)",
        "gene":             "AHI1 (*608894) — 6q23.3 — 1196 aa — Jouberin / Transition Zone Scaffold",
        "gene_omim":        "608894",
        "disease_omim":     "608629",
        "chromosome":       "6q23.3",
        "inheritance":      "Autosomal Recessive — biallelic LOF",
        "protein_length":   "1196 aa",
        "cohort_n":         N,
        "seed":             SEED,
        "total":            N,
        "confirmed_jbts3":  N,
        "molar_tooth_mri":          mts_present_n,
        "neonatal_hypotonia":       hypotonia_n,
        "oculomotor_apraxia":       oma_n,
        "cerebellar_ataxia":        ataxia_n,
        "id_moderate_severe":       id_n,
        "retinal_dystrophy":        retinal_n,
        "breathing_dysregulation":  breathing_n,
        "renal_disease":            renal_n,
        "corpus_callosum":          corpus_callosum_n,
        "kpis": {
            "mts_n":              mts_present_n,
            "mts_pct":            _pct(mts_present_n),
            "hypotonia_n":        hypotonia_n,
            "hypotonia_pct":      _pct(hypotonia_n),
            "oma_n":              oma_n,
            "oma_pct":            _pct(oma_n),
            "ataxia_n":           ataxia_n,
            "ataxia_pct":         _pct(ataxia_n),
            "id_n":               id_n,
            "id_pct":             _pct(id_n),
            "retinal_n":          retinal_n,
            "retinal_pct":        _pct(retinal_n),
            "breathing_n":        breathing_n,
            "breathing_pct":      _pct(breathing_n),
            "renal_n":            renal_n,
            "renal_pct":          _pct(renal_n),
            "corpus_callosum_n":  corpus_callosum_n,
            "corpus_callosum_pct": _pct(corpus_callosum_n),
            "misdiagnosis_n":     misdiagnosis_n,
            "misdiagnosis_pct":   _pct(misdiagnosis_n),
        },
        "sex_split":  {"M": sex_M, "F": sex_F},
        "age_distribution": {
            "dx_0_6mo":  age_0_6mo,
            "dx_7_24mo": age_7_24mo,
            "dx_2_5yr":  age_2_5yr,
            "dx_6_12yr": age_6_12yr,
        },
        "mechanism_summary": (
            "AHI1 (Jouberin) is a transition zone (TZ) scaffold protein at the base of primary cilia. "
            "AHI1 forms a tripartite complex with NPHP1 and NPHP4 at ciliary TZ Y-links. "
            "Loss of AHI1 → TZ gate assembly defective → neuronal GPCRs (SSTR3, MCHR1) and Smo "
            "FAIL to enter cilia → impaired Hedgehog/somatostatin/MCH receptor signalling in neurons "
            "→ Molar Tooth Sign (MTS) on brain MRI (elongated superior cerebellar peduncles + "
            "cerebellar vermis hypoplasia + deepened interpeduncular fossa). Cilia are structurally "
            "intact (9+2 axoneme preserved) — a pure TZ gating defect."
        ),
        "key_feature": (
            "MOLAR TOOTH SIGN (MTS) on brain MRI is PATHOGNOMONIC for Joubert syndrome. "
            "Elongated superior cerebellar peduncles + cerebellar vermis hypoplasia + deepened "
            "interpeduncular fossa = MTS. Oculomotor apraxia (OMA) ~75% — horizontal gaze "
            "initiation failure. NO polydactyly, NO obesity, NO narrow thorax, renal very rare "
            "(distinguishes JBTS3 from BBS, SRTD, NPHP)."
        ),
        "founder_allele": "Arg830Trp (c.2488C>T) — Ashkenazi Jewish founder; ~1/92 carrier frequency in AJ",
        "frequency_jbts3": "~10–15% of all Joubert syndrome cases; ~1/600,000–800,000 worldwide",
        "frequency_all_jbts": "JBTS overall ~1/80,000–100,000 births",
        "ethnicity_breakdown": _eth_dist(),
        "allele_class_breakdown": _allele_dist(),
        "top_variants": [
            {"variant": "p.Arg830Trp (c.2488C>T)", "domain": "Coiled-coil/WD40-junction",
             "ethnicity": "Ashkenazi Jewish (founder ~1/92 carrier)", "severity": "Moderate-severe MTS + OMA"},
            {"variant": "p.Val443del (c.1327_1329delGTG)", "domain": "WD40 propeller loop",
             "ethnicity": "European (compound het)", "severity": "Moderate"},
            {"variant": "p.Gln604Ter (c.1810C>T)", "domain": "Truncating null — WD40/CC junction",
             "ethnicity": "Pan-ethnic", "severity": "Severe MTS + OMA + retinal"},
            {"variant": "p.Arg1041Trp (c.3121C>T)", "domain": "SH3 domain",
             "ethnicity": "European", "severity": "Retinal dystrophy prominent"},
            {"variant": "p.Trp904Ter (c.2712G>A)", "domain": "Coiled-coil truncation",
             "ethnicity": "MENA / South Asian", "severity": "Severe"},
        ],
        "distinguishing_from_bbs": (
            "NO polydactyly (0%), NO obesity — these are defining BBS features. "
            "JBTS3 has Molar Tooth Sign (never seen in BBS). Retinal dystrophy less frequent in JBTS3 (~52%) vs BBS (~85%)."
        ),
        "distinguishing_from_nphp": (
            "Renal disease very rare in JBTS3 (8%) vs NPHP (defining). "
            "JBTS3 AHI1 mutations do not carry NPHP-class alleles — no tubulointerstitial nephritis. "
            "MTS on MRI is present in JBTS3, not in NPHP."
        ),
        "distinguishing_from_srtd": (
            "NO narrow thorax, NO skeletal dysplasia, NO polydactyly. "
            "SRTD affects IFT-A/IFT-B/dynein-2 in chondrocytes; JBTS3 affects TZ gating in neurons. "
            "MTS is ABSENT in SRTD — its presence immediately points to JBTS, not SRTD."
        ),
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
        "per_patient": [
            {
                "id":                   i + 1,
                "ethnicity":            p['ethnicity'],
                "allele_class":         p['allele_class'],
                "mts_status":           p['mts_status'],
                "hypotonia":            p['hypotonia'],
                "oculomotor_apraxia":   p['oculomotor_apraxia'],
                "ataxia":               p['ataxia'],
                "id_severity":          p['id_severity'],
                "retinal":              p['retinal'],
                "breathing_dysregulation": p['breathing_dysregulation'],
                "renal":                p['renal'],
                "corpus_callosum":      p['corpus_callosum'],
                "age_diagnosis_mo":     p['age_diagnosis_mo'],
                "initial_presentation": p['initial_presentation'],
                "misdiagnosis":         p['misdiagnosis'],
                "sex":                  p['sex'],
            }
            for i, p in enumerate(patients)
        ],
        "mts_distribution":            _dist('mts_status'),
        "hypotonia_distribution":      _dist('hypotonia'),
        "oma_distribution":            _dist('oculomotor_apraxia'),
        "ataxia_distribution":         _dist('ataxia'),
        "id_distribution":             _dist('id_severity'),
        "retinal_distribution":        _dist('retinal'),
        "breathing_distribution":      _dist('breathing_dysregulation'),
        "renal_distribution":          _dist('renal'),
        "corpus_callosum_distribution": _dist('corpus_callosum'),
        "presentation_distribution":   _dist('initial_presentation'),
        "misdiagnosis_distribution":   _dist('misdiagnosis'),
        "allele_class_summary":        _dist('allele_class'),
        "ethnicity_distribution":      _eth_dist(),
        "age_distribution":            _dist('age_diagnosis_mo'),
        "top_variants": [
            {"variant": "p.Arg830Trp (c.2488C>T) — Coiled-coil/WD40-junction — Ashkenazi Jewish founder — moderate-severe MTS + OMA",     "n": rng.randint(6, 9)},
            {"variant": "p.Val443del (c.1327_1329delGTG) — WD40 propeller loop — European — compound het — moderate",                     "n": rng.randint(3, 5)},
            {"variant": "p.Gln604Ter (c.1810C>T) — Truncating null — pan-ethnic — severe MTS + OMA + retinal",                            "n": rng.randint(2, 4)},
            {"variant": "p.Arg1041Trp (c.3121C>T) — SH3 domain — European — retinal dystrophy prominent",                                 "n": rng.randint(2, 3)},
            {"variant": "p.Trp904Ter (c.2712G>A) — Coiled-coil truncation — MENA/South Asian — severe",                                   "n": rng.randint(1, 3)},
        ],
        "summary_stats": {
            "total":               N,
            "mts_present_n":       mts_present_n,
            "mts_present_pct":     _pct(mts_present_n),
            "hypotonia_n":         hypotonia_n,
            "oma_n":               oma_n,
            "ataxia_n":            ataxia_n,
            "id_n":                id_n,
            "retinal_n":           retinal_n,
            "breathing_n":         breathing_n,
            "renal_n":             renal_n,
            "corpus_callosum_n":   corpus_callosum_n,
            "misdiagnosis_n":      misdiagnosis_n,
            "no_polydactyly":      True,
            "no_obesity":          True,
        },
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":            "AHI1 — Abelson Helper Integration Site 1 (also: Jouberin)",
            "also_known_as":   "Jouberin; JBTS3 gene; AHI-1",
            "omim_gene":       "*608894",
            "chromosome":      "6q23.3",
            "protein_size":    "1196 aa",
            "protein_class":   "Transition zone scaffold protein — ciliary Y-link organiser",
            "domain_structure": (
                "N-terminal WD40 beta-propeller (aa 1–400) — 7 WD40 repeats; NPHP1 binding surface; "
                "Coiled-coil domain (aa 400–700) — NPHP4 interaction; Arg830Trp pathogenic cluster at CC/WD40-junction; "
                "C-terminal SH3 domain (aa 1050–1196) — protein-protein interaction; Arg1041Trp retinal-prominent cluster"
            ),
            "function": (
                "AHI1 (Jouberin) scaffolds the ciliary transition zone (TZ) Y-links by forming a "
                "tripartite complex with NPHP1 and NPHP4. This TZ gate complex controls which "
                "receptors and signalling proteins gain access to the ciliary compartment. "
                "Loss of AHI1 → TZ gate incomplete → SSTR3, MCHR1, Smo fail to enter neuronal cilia "
                "→ impaired Hh/somatostatin/MCH receptor signalling → cerebellar vermis hypoplasia + MTS."
            ),
            "inheritance":   "Autosomal Recessive — biallelic LOF",
            "key_contacts":  "NPHP1 (JBTS4 / NPHP1) via WD40 beta-propeller; NPHP4 via coiled-coil; Jouberin complex",
            "expression":    "Brain (cerebellar vermis, brainstem); retina (photoreceptors); kidney (tubular, low level); ciliated cells",
        },
        "disease_card": {
            "omim_disease":      "#608629",
            "full_name":         "Joubert Syndrome 3 (JBTS3)",
            "alternative_names": "Cerebello-oculo-renal syndrome 3; CORS3; Joubert-AHI1",
            "primary_finding":   "MOLAR TOOTH SIGN (MTS) on brain MRI — PATHOGNOMONIC for Joubert syndrome; elongated SCPs + cerebellar vermis hypoplasia + deepened interpeduncular fossa",
            "oculomotor_apraxia": "~75% (horizontal gaze initiation failure — neuronal cilia SSTR3/MCHR1 failure)",
            "neonatal_hypotonia": "~88% (truncal hypotonia; neonatal feeding difficulty)",
            "cerebellar_ataxia":  "~90% (gait + limb ataxia; cerebellar vermis dysgenesis)",
            "intellectual_disability": "~82% (moderate-severe; variable; some AJ Arg830Trp have milder ID)",
            "retinal_dystrophy":  "~52% (rod-cone > cone-rod; ERG mandatory; lower than BBS ~85%)",
            "breathing_dysregulation": "~65% (episodic apnea alternating with hyperpnea; neonatal period; usually resolves by 6 months)",
            "renal_disease":      "VERY RARE ~8% — minor structural cysts only; NO NPHP-type tubulointerstitial nephritis",
            "corpus_callosum":    "~35% (agenesis or hypoplasia; brain MRI mandatory)",
            "polydactyly":        "ABSENT (0%) — critical distinguisher from BBS, SRTD, and other ciliopathies with polydactyly",
            "obesity":            "ABSENT — critical distinguisher from BBS",
            "narrow_thorax":      "ABSENT — critical distinguisher from SRTD",
            "misdiagnosis":       "Dandy-Walker syndrome; cerebellar hypoplasia NOS; pontocerebellar hypoplasia; Joubert-like without gene dx",
            "founder_allele":     "Arg830Trp (c.2488C>T) — Ashkenazi Jewish founder; ~1/92 carrier frequency in AJ; most common AHI1 variant worldwide",
            "prevalence":         "~10–15% of all Joubert syndrome; ~1/600,000–800,000 worldwide",
        },
        "diagnostic_workup": [
            "1. Brain MRI (MANDATORY FIRST-LINE): Molar Tooth Sign (MTS) = pathognomonic for Joubert syndrome. Axial T1/T2 at posterior fossa — elongated superior cerebellar peduncles + cerebellar vermis hypoplasia + deepened interpeduncular fossa. MTS present in ~95% JBTS3.",
            "2. Ophthalmology + ERG: retinal dystrophy in ~52%; rod-cone > cone-rod pattern; ERG mandatory at diagnosis and annually from age 3. Oculomotor apraxia clinical exam.",
            "3. GENE PANEL (mandatory): AHI1, CEP290, TMEM67/MKS3, RPGRIP1L, CC2D2A, INPP5E, TMEM216, KIF7, TCTN1-3 — comprehensive JBTS panel resolves allele. AHI1 Arg830Trp may be flagged as Ashkenazi Jewish founder on targeted ancestry panel.",
            "4. Renal USS: renal cysts very rare in JBTS3 (~8%); renal USS at diagnosis and every 3 years; if cysts present, consider CEP290 (JBTS5) or TMEM67 (JBTS6) DDx.",
            "5. Developmental assessment: intellectual disability screening; physiotherapy/OT assessment for ataxia and hypotonia.",
            "6. Breathing monitoring (neonatal): polysomnography if episodic apnea/hyperpnea; usually resolves by 6 months; NICU if severe.",
            "7. WES + CNV if JBTS gene panel negative — AHI1 intragenic rearrangements are rare; MLPA for CNV detection.",
            "8. Genetics: Ashkenazi Jewish carrier testing for Arg830Trp; cascade family testing; preimplantation genetics (25% recurrence AR); consanguinity counselling.",
            "9. Echocardiography NOT routinely required (cardiac involvement uncommon in JBTS3 — distinguishes from BBS).",
            "10. DDx evaluation: if polydactyly present → reconsider BBS; if narrow thorax → reconsider SRTD; if ESRD → reconsider NPHP or CEP290-JBTS5.",
        ],
        "mechanism_glossary": [
            {"term": "AHI1 (Jouberin)", "definition": "1196 aa transition zone scaffold protein (WD40 + coiled-coil + SH3). Forms tripartite TZ complex with NPHP1 and NPHP4. Loss → TZ gate incomplete → neuronal GPCR failure → MTS + OMA + ataxia."},
            {"term": "Transition Zone (TZ)", "definition": "Proximal region of the cilium above the basal body. Contains Y-shaped protein bridges (Y-links) connecting axoneme to ciliary membrane. Acts as a selective gate controlling protein entry and exit from the ciliary compartment."},
            {"term": "Y-links", "definition": "Electron-dense Y-shaped connections between outer doublet microtubules and the ciliary membrane in the TZ. Composed of NPHP proteins and MKS/TCTN modules. AHI1 is a key Y-link organiser."},
            {"term": "Molar Tooth Sign (MTS)", "definition": "Pathognomonic brain MRI finding for Joubert syndrome: elongated superior cerebellar peduncles (SCPs) + cerebellar vermis hypoplasia/aplasia + deepened interpeduncular fossa, creating a 'molar tooth' appearance on axial MRI. ABSENT in BBS, SRTD, NPHP."},
            {"term": "SSTR3 / MCHR1", "definition": "Somatostatin receptor 3 (SSTR3) and melanin-concentrating hormone receptor 1 (MCHR1) — neuronal GPCRs that require AHI1-mediated TZ gating to enter neuronal cilia. Loss → impaired somatostatin and MCH signalling in brain circuits regulating motor, cognitive, and autonomic functions."},
            {"term": "Oculomotor Apraxia (OMA)", "definition": "Inability to initiate voluntary horizontal eye movements (gaze initiation failure). Due to brainstem/cerebellar vermis dysfunction from TZ gating failure in neurons. Present in ~75% JBTS3. Clinically: compensatory head thrusts to 'unlock' gaze."},
            {"term": "Neonatal Breathing Dysregulation", "definition": "Episodic apnea alternating with hyperpnea (fast breathing) in neonates with JBTS3. Due to brainstem breathing centre dysfunction (Smo/Hh signalling failure in reticular formation neurons). Usually resolves by 3–6 months."},
            {"term": "Ashkenazi Jewish founder Arg830Trp", "definition": "p.Arg830Trp (c.2488C>T) in AHI1 coiled-coil/WD40-junction. Carrier frequency ~1/92 in Ashkenazi Jewish population. Most common AHI1 variant worldwide. Homozygotes: moderate-severe JBTS3 with MTS + OMA; occasional retinal involvement."},
        ],
        "key_variants": [
            {"variant": "p.Arg830Trp (c.2488C>T)", "domain": "Coiled-coil/WD40-junction (aa ~830)", "consequence": "Disrupts NPHP4-AHI1 coiled-coil interaction; TZ gate partially impaired; moderate-severe MTS + OMA; founder allele", "ethnicity": "Ashkenazi Jewish (homozygous founder; ~1/92 carrier)"},
            {"variant": "p.Val443del (c.1327_1329delGTG)", "domain": "WD40 propeller loop (aa 443)", "consequence": "In-frame deletion; WD40 blade integrity impaired; NPHP1 binding surface weakened; moderate severity", "ethnicity": "European (compound het most common)"},
            {"variant": "p.Gln604Ter (c.1810C>T)", "domain": "Truncating null — WD40/coiled-coil junction", "consequence": "Premature stop; null allele; full TZ gate failure; severe MTS + OMA + retinal dystrophy", "ethnicity": "Pan-ethnic"},
            {"variant": "p.Arg1041Trp (c.3121C>T)", "domain": "SH3 domain (aa 1041)", "consequence": "SH3 protein-interaction domain; retinal dystrophy prominent; moderate neurological; European", "ethnicity": "European"},
            {"variant": "p.Trp904Ter (c.2712G>A)", "domain": "Coiled-coil truncation (aa 904)", "consequence": "Truncating within coiled-coil; null allele; severe; NPHP4-AHI1 contact lost; MENA/South Asian enrichment", "ethnicity": "MENA / South Asian (consanguineous)"},
        ],
        "treatment_summary": [
            "1. No disease-modifying therapy for JBTS3 (2026) — symptomatic and supportive management only.",
            "2. BREATHING DYSREGULATION (neonatal): NICU monitoring; pulse oximetry; caffeine for apnea of prematurity-like episodes; polysomnography; usually resolves by 3–6 months. Rarely requires prolonged ventilatory support.",
            "3. HYPOTONIA + ATAXIA: physiotherapy from diagnosis; occupational therapy for fine motor; hydrotherapy for gait ataxia; adaptive equipment (walkers, orthotics) as needed.",
            "4. DEVELOPMENTAL / INTELLECTUAL DISABILITY: early intervention programme; speech-language therapy (dysarthria + language delay); special education support; neuropsychological assessment.",
            "5. OCULOMOTOR APRAXIA: low vision support; visual aids; occupational therapy for reading/navigation. No pharmacological treatment for OMA.",
            "6. RETINAL DYSTROPHY (~52%): annual ERG from age 3; ophthalmology retinal specialist; low vision aids; Gene therapy trials (CEP290-LCA10 precedent; no AHI1-specific trial 2026); vitamin A supplementation monitoring.",
            "7. RENAL (very rare ~8%): annual renal USS; creatinine monitoring; if cysts found, consult nephrology. No NPHP-type nephronophthisis management needed in JBTS3 — renal disease is minor structural, not progressive ESRD.",
            "8. GENETICS: Ashkenazi Jewish carrier panel includes Arg830Trp; cascade family testing; prenatal USS for structural brain anomalies; preimplantation genetic testing (25% recurrence AR).",
            "9. MDT: paediatric neurology, ophthalmology, physiotherapy, OT, speech-language therapy, genetics, nephrology (if renal), neonatology (breathing, neonatal).",
        ],
        "ddx_table": [
            {"disease": "Bardet-Biedl Syndrome (BBS)", "key_difference": "BBS has polydactyly (~70%) + obesity — both ABSENT in JBTS3. BBS retinal dystrophy ~85% vs JBTS3 ~52%. MTS on MRI ABSENT in BBS — its presence confirms JBTS, not BBS."},
            {"disease": "NPHP (Nephronophthisis)", "key_difference": "Renal disease is DEFINING in NPHP (progressive tubulointerstitial nephritis → ESRD) — VERY RARE in JBTS3 (~8%). MTS absent in pure NPHP. JBTS3 can have NPHP overlap if CEP290/RPGRIP1L alleles — AHI1 is not NPHP-dominant."},
            {"disease": "SRTD (Short-Rib Thoracic Dysplasia)", "key_difference": "Narrow thorax + polydactyly are hallmarks of SRTD — BOTH ABSENT in JBTS3. MTS present in JBTS3, absent in SRTD. Different affected ciliary compartment: TZ (JBTS3) vs IFT-A/dynein-2 (SRTD)."},
            {"disease": "Dandy-Walker Syndrome", "key_difference": "Dandy-Walker: cystic dilatation of 4th ventricle + absent cerebellar vermis — different from MTS. No ciliopathy genetics. MTS pattern on MRI distinguishes JBTS from Dandy-Walker."},
            {"disease": "Pontocerebellar Hypoplasia (PCH)", "key_difference": "PCH: progressive pontine + cerebellar atrophy (TSEN, RARS2 genes). MTS absent. Usually more severe/fatal course. No ciliary genetics."},
            {"disease": "JBTS5 (CEP290)", "key_difference": "CEP290 biallelic null → MKS4 (lethal) or LCA10 (retinal-only). Hypomorphic → JBTS5. Renal involvement more common in CEP290-JBTS5 vs AHI1-JBTS3. Gene panel differentiates."},
            {"disease": "JBTS6 (TMEM67/MKS3)", "key_difference": "TMEM67 → liver fibrosis more prominent (Meckel-like overlap); renal involvement higher than AHI1-JBTS3. Allele class governs: null → MKS3, hypomorphic → JBTS6. Gene panel required."},
            {"disease": "Cerebellar hypoplasia NOS", "key_difference": "Many causes (prenatal infection, FOXC1, VLDLR). MTS is specific to JBTS — non-MTS cerebellar hypoplasia is NOT Joubert syndrome. Brain MRI pattern + gene panel resolves."},
        ],
        "jbts_gene_comparison": [
            {"gene": "AHI1",    "jbts": "JBTS3", "omim_gene": "608894", "chr": "6q23.3",  "module": "TZ scaffold (NPHP1/NPHP4 complex)", "frequency": "~10–15% all JBTS", "renal": "Very rare ~8%", "retinal": "~52%"},
            {"gene": "INPP5E",  "jbts": "JBTS1", "omim_gene": "613037", "chr": "9q34.3",  "module": "Ciliary PI(3,4,5)P3 phosphatase", "frequency": "~5–8%", "renal": "Rare", "retinal": "~30%"},
            {"gene": "TMEM216", "jbts": "JBTS2", "omim_gene": "613277", "chr": "11q13.1", "module": "TZ membrane protein (MKS module)", "frequency": "~2–3%", "renal": "Moderate", "retinal": "~50%"},
            {"gene": "NPHP1",   "jbts": "JBTS4", "omim_gene": "607100", "chr": "2q13",    "module": "TZ Y-link (NPHP module) — AHI1 partner", "frequency": "~1–3%", "renal": "HIGH (NPHP1 = NPHP dominant)", "retinal": "~40%"},
            {"gene": "CEP290",  "jbts": "JBTS5", "omim_gene": "610142", "chr": "12q21.32","module": "TZ transition plate + Y-link", "frequency": "~10–15%", "renal": "Moderate ~25%", "retinal": "~70%"},
            {"gene": "TMEM67",  "jbts": "JBTS6", "omim_gene": "609884", "chr": "8q22.1",  "module": "TZ membrane (MKS/TCTN module)", "frequency": "~5–8%", "renal": "High + liver", "retinal": "~55%"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== JBTS3 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("MTS:", b["mts_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("Ethnicity:", b["ethnicity_distribution"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
