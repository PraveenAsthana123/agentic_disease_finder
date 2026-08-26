"""
TTC21B Short-Rib Thoracic Dysplasia 4 (SRTD4) — Jeune Asphyxiating Thoracic Dystrophy Type 4
================================================================================================
Primary Gene : TTC21B (*612014) — 2q24.3; ~1315 aa; Tetratricopeptide Repeat Domain 21B
               Also known as IFT139, TTC21, JBTS12, NPHP12.
               TTC21B is a TPR (tetratricopeptide repeat) domain protein and IFT-A complex
               subunit. It acts as an adaptor bridging WDR19/IFT144 (SRTD5) C-tail to IFT122
               (CED2) within the IFT-A complex core. TPR repeats mediate protein–protein
               interactions; N-terminal TPR domain contacts IFT122; C-terminal region contacts
               WDR19 C-tail. Loss → IFT-A complex assembly failure → retrograde IFT from
               ciliary tip impaired → IFT-B cannot be imported at cilia base/transition zone →
               SHORT STUBBY CILIA (EM) → Hedgehog signal failure in chondrocytes → narrow
               thorax, short ribs, short limbs.
               IFT-A subunit hierarchy: IFT140 (SRTD9) + WDR19/IFT144 (SRTD5) + WDR35/IFT121
               (SRTD7) + TTC21B/IFT139 (SRTD4) + IFT122 (CED2) + IFT43.
               UNIQUE: TTC21B causes THREE distinct ciliopathy phenotypes depending on allele
               severity: SRTD4 (Jeune ATD4 — thoracic-dominant, severe biallelic LOF), NPHP12
               (renal-only, hypomorphic C-terminal alleles; same as WDR19/NPHP13 paradigm), and
               JBTS12 (Joubert syndrome type 12 — hypomorphic alleles with cerebellar vermis
               involvement; unique among SRTD genes: TTC21B bridges SRTD and Joubert spectra).
Disease OMIM : #613819 — Short-Rib Thoracic Dysplasia 4 With or Without Polydactyly (SRTD4/ATD4)
               #613820 — Nephronophthisis 12 (NPHP12; renal-only hypomorphic TTC21B alleles)
               Also called: Asphyxiating Thoracic Dystrophy 4 (ATD4); Jeune syndrome type 4
               Severe biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum.
               JBTS12: Joubert syndrome type 12 (*JBTS12) — some hypomorphic alleles cause MTS.
Chromosome   : 2q24.3
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1/300,000–1,000,000; ~50–100 families reported worldwide (2026);
               Second most common IFT-A SRTD gene after WDR19/IFT144 (SRTD5/NPHP13);
               ~1–2% of molecularly confirmed SRTD cases.

Protein Structure — TTC21B (1315 aa; IFT-A TPR adaptor)
---------------------------------------------------------
Domain 1: N-terminal TPR cluster I (aa 1–300)    — IFT122 contact surface; scaffold docking
Domain 2: Central TPR cluster II (aa 301–700)    — IFT-A complex core; WDR19/IFT144 C-tail
           binding surface; most pathogenic missense cluster; aa 450–650 hotspot
Domain 3: C-terminal TPR cluster III (aa 701–1100) — WDR19 C-tail junction; allele severity
           determinant; hypomorphic missense here → NPHP12 (renal-only) or JBTS12
Domain 4: C-terminal unstructured tail (aa 1101–1315) — IFT train regulation; dispensable
           for SRTD4; hypomorphic variants → mild phenotype only

Key pathogenic variant classes (TTC21B):
1. Central TPR missense (aa 301–700): compound het or homozygous; moderate SRTD4; most common
2. Compound het: truncating + missense (any domain): moderate to severe SRTD4
3. Biallelic null (truncating + truncating): SRPS spectrum; perinatal lethal
4. Hypomorphic C-terminal missense (aa 701–1100): NPHP12 (renal-only, no narrow thorax);
   or JBTS12 (Joubert, cerebellar vermis hypoplasia, molar tooth sign)
5. Homozygous hypomorphic (C-terminal): mildest; late-onset renal or Joubert
"""

import random
import math

SEED = 397
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
    ('Middle Eastern / North African', 0.32),   # consanguineous homozygous
    ('European',                        0.28),   # compound het most common
    ('South Asian',                     0.20),   # hypomorphic NPHP12 alleles common
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
    'Homozygous missense (central TPR)',
    'Compound het two missense',
    'Homozygous truncating (null / SRPS)',
    'Homozygous hypomorphic (C-terminal / NPHP12)',
]
allele_weights = [0.35, 0.27, 0.18, 0.12, 0.08]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.20, 0.50, 0.30])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'None'],
        weights=[0.20, 0.18, 0.62])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.14, 0.12, 0.74])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.42, 0.58])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.08, 0.06, 0.86])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.05, 0.05, 0.90])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.30, 0.12, 0.58])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.48, 0.24, 0.28])[0]
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
         'BBS (Bardet-Biedl)', 'SRTD5/WDR19 (gene panel corrected)',
         'JBTS12 / Joubert (C-terminal allele)', 'None'],
        weights=[0.08, 0.07, 0.05, 0.12, 0.05, 0.63])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.35, 0.30, 0.22, 0.13])[0]

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
        "disease":    "TTC21B Short-Rib Thoracic Dysplasia 4 (SRTD4 / ATD4)",
        "gene":       "TTC21B (*612014) — 2q24.3 — 1315 aa — IFT-A TPR Adaptor (IFT139)",
        "omim_gene":  "612014",
        "omim_disease": "613819",
        "omim_disease_nphp12": "613820",
        "chromosome": "2q24.3",
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
            "TTC21B loss removes an IFT-A adaptor TPR protein (IFT139) that bridges WDR19/IFT144 "
            "(SRTD5) C-tail to IFT122 within the IFT-A complex. Without TTC21B, IFT-A complex "
            "assembly is disrupted → retrograde IFT from ciliary tip to cell body impaired → "
            "IFT-B CANNOT be imported at the cilia base/transition zone → SHORT STUBBY CILIA "
            "(EM; same as all IFT-A SRTDs: WDR19-SRTD5, IFT140-SRTD9, WDR35-SRTD7) → "
            "Hedgehog (Ihh/Shh) signalling fails in chondrocytes → NARROW THORAX (primary), "
            "short ribs, short limbs. Secondary: renal TIN/ESRD (~20–30% survivors), "
            "rod-cone dystrophy (~10–15%), CHF (~8%)."
        ),
        "key_distinction": (
            "TTC21B (SRTD4) is UNIQUE among SRTD genes in causing THREE distinct ciliopathy "
            "phenotypes: (1) SRTD4/ATD4 — thoracic-dominant severe biallelic LOF; "
            "(2) NPHP12 — renal-only hypomorphic C-terminal alleles (no narrow thorax), same "
            "paradigm as WDR19/NPHP13; (3) JBTS12 — Joubert syndrome type 12 (molar tooth sign "
            "on MRI, cerebellar vermis hypoplasia), hypomorphic alleles. TTC21B is one of the "
            "very few genes that bridge the SRTD and Joubert ciliopathy spectra. SHORT STUBBY "
            "cilia on EM (not club/bulging) — the IFT-A signature distinct from dynein-2 SRTDs. "
            "No ectodermal features (ectodermal is unique to WDR19/SRTD5 among IFT-A SRTDs). "
            "~1–2% of all SRTD; ~50–100 families worldwide (2026); second most common IFT-A SRTD."
        ),
        "ift_a_subunit_table": [
            {"subunit": "IFT140",   "role": "Core scaffold WD40 β-propeller; binds WDR35 N-term + WDR19 C-TPR",    "srtd": "SRTD9",  "omim_gene": "614620", "chr": "16p13.3", "freq": "~2nd IFT-A"},
            {"subunit": "WDR19",    "role": "LARGEST IFT-A subunit; central WD40 platform; docks IFT140+TTC21B",   "srtd": "SRTD5",  "omim_gene": "608151", "chr": "4p14",    "freq": "~1% SRTD (most common IFT-A)"},
            {"subunit": "WDR35",    "role": "14-blade WD40 β-propeller; C-face docks IFT140 N-terminal domain",     "srtd": "SRTD7",  "omim_gene": "613602", "chr": "2p24.1",  "freq": "3rd IFT-A"},
            {"subunit": "TTC21B",   "role": "TPR adaptor (IFT139); bridges WDR19 C-tail to IFT122",                 "srtd": "SRTD4",  "omim_gene": "612014", "chr": "2q24.3",  "freq": "~1–2% SRTD (2nd IFT-A)"},
            {"subunit": "IFT122",   "role": "WD40 β-propeller; IFT-A stability; TTC21B N-term contact",             "srtd": "CED2",   "omim_gene": "606045", "chr": "3q21.3",  "freq": "CED2 (rare SRTD)"},
            {"subunit": "IFT43",    "role": "Small IFT-A subunit; stability; WDR35 N-term contact",                  "srtd": "CED3",   "omim_gene": "614068", "chr": "14q24.3", "freq": "CED3 (very rare)"},
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
            {"variant": "p.Arg484Cys (c.1450C>T) — central TPR cluster II (IFT122 contact) — European compound het — moderate SRTD4", "n": rng.randint(4, 6)},
            {"variant": "p.Gly605Arg (c.1813G>A) — central TPR α-helix (WDR19 C-tail binding) — Middle Eastern homozygous — moderate-severe SRTD4", "n": rng.randint(3, 5)},
            {"variant": "p.Trp1100Ter (c.3300G>A) — truncating, null — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala775Val (c.2324C>T) — C-terminal TPR III edge — South Asian hypomorphic — NPHP12 renal-dominant mild", "n": rng.randint(2, 3)},
            {"variant": "p.Arg924Gln (c.2771G>A) — WDR19 C-tail junction region — MENA homozygous — moderate SRTD4", "n": rng.randint(1, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "TTC21B — Tetratricopeptide Repeat Domain 21B",
            "also_known_as":    "IFT139; TTC21; JBTS12; NPHP12; IFT-A TPR adaptor",
            "omim_gene":        "*612014",
            "chromosome":       "2q24.3",
            "protein_size":     "1315 aa",
            "protein_class":    "IFT-A complex TPR (tetratricopeptide repeat) domain protein",
            "domain_structure":  "N-terminal TPR cluster I (aa 1–300, IFT122 contact) + central TPR cluster II (aa 301–700, WDR19 C-tail binding, pathogenic hotspot) + C-terminal TPR cluster III (aa 701–1100, allele severity determinant) + unstructured C-tail (aa 1101–1315)",
            "function":         "Bridges WDR19/IFT144 (SRTD5) C-tail to IFT122 within IFT-A complex; critical adaptor for IFT-A assembly; retrograde IFT competence depends on TTC21B integrity",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT122 (N-terminal TPR domain); WDR19/IFT144 SRTD5 (C-tail junction; central TPR)",
            "expression":       "Primary non-motile cilia (9+0); chondrocytes; renal tubular epithelium; retinal photoreceptor connecting cilium; cerebellar granule cells (explains JBTS12 with hypomorphic alleles)",
            "three_phenotypes":  "SRTD4 (#613819) — severe thoracic; NPHP12 (#613820) — renal-only hypomorphic; JBTS12 — Joubert MTS hypomorphic",
        },
        "disease_card": {
            "omim_disease_srtd4":  "#613819 — SRTD4 (ATD4) — severe biallelic LOF",
            "omim_disease_nphp12": "#613820 — NPHP12 — hypomorphic C-terminal alleles; renal-only",
            "full_name":           "Short-Rib Thoracic Dysplasia 4 With or Without Polydactyly",
            "alternative_names":   "SRTD4; Asphyxiating Thoracic Dystrophy 4 (ATD4); Jeune syndrome type 4; NPHP12 (renal phenotype); JBTS12 (Joubert phenotype)",
            "primary_finding":     "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":         "~35–40% (postaxial; moderate frequency among IFT-A SRTDs)",
            "renal":               "TIN + corticomedullary cysts → ESRD in ~20–30% of survivors; NPHP12 alleles → renal-only (no thorax)",
            "retinal":             "Rod-cone dystrophy in ~10–15% (connecting cilium IFT failure)",
            "hepatic_chf":         "Ductal plate malformation → CHF in ~8%",
            "situs_inversus":      "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":         "PRESENT in JBTS12 alleles (hypomorphic C-terminal) — UNIQUE among SRTD4 genotypes; absent in full SRTD4/null alleles",
            "ectodermal":          "ABSENT — no sparse hair, hypodontia, or dystrophic nails (ectodermal is unique to WDR19/SRTD5 among IFT-A SRTDs)",
            "srps_spectrum":       "Biallelic null alleles → SRPS (perinatal lethal); severity similar to SRTD5/WDR19",
            "renal_transplant":    "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment":  "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":          "~1/300,000–1,000,000; ~50–100 families worldwide (2026); ~1–2% of SRTD",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones; short stubby cilia phenotype (not club-tip — IFT-A signature)",
            "3. GENE PANEL (mandatory first-line): IFT140, WDR19, WDR35, TTC21B, IFT122, WDR60, DYNC2H1, WDR34 — TTC21B MUST be included; SRTD4 clinically identical to SRTD5/WDR19",
            "4. For NPHP phenotype: renal USS + creatinine; TTC21B on NPHP gene panel (NPHP12); co-sequence with WDR19 (NPHP13) — digenic and modifier effects possible",
            "5. WES + CNV if panel negative — TTC21B is large (1315 aa); CNV/deletion analysis mandatory",
            "6. Ciliopathy EM (skin/fibroblast biopsy): SHORT STUBBY cilia — IFT-A class; NOT club/bulging tip (that is dynein-2). Distinguishes IFT-A SRTDs from SRTD3/8/11/15/17",
            "7. Brain MRI if hypomorphic alleles suspected: check for molar tooth sign (JBTS12); absent in full SRTD4 null alleles",
            "8. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "9. ERG annually from age 6 (rod-cone dystrophy ~10–15%); ophthalmology referral",
            "10. Liver USS + APRI if hepatic signs (CHF ~8%)",
            "11. MLPA/CNV for TTC21B deletions if sequencing negative but phenotype compelling",
        ],
        "mechanism_glossary": [
            {"term": "TTC21B (IFT139)",     "definition": "Tetratricopeptide Repeat Domain 21B — a 1315 aa IFT-A complex adaptor protein with multiple TPR helix-turn-helix repeats. Bridges WDR19/IFT144 C-tail to IFT122 within the IFT-A complex core. Also known as IFT139, JBTS12, NPHP12."},
            {"term": "TPR domain",          "definition": "Tetratricopeptide repeat — a tandem helix-turn-helix structural motif mediating protein–protein interactions. TTC21B's TPR repeats form a curved solenoid that docks sequentially onto WDR19 and IFT122 within the IFT-A complex."},
            {"term": "IFT-A complex",       "definition": "Intraflagellar transport complex A — the retrograde IFT machinery that moves from ciliary tip to cell body, powered by dynein-2. Required for importing IFT-B (anterograde complex) at the cilia base. Failure → IFT-B cannot enter cilia → short stubby cilia."},
            {"term": "Short stubby cilia",  "definition": "EM finding in IFT-A SRTDs: cilia are short and stubby but do NOT show bulging/club tips (that is dynein-2). IFT-B fails to be imported at the base, so cilia are uniformly shortened. Identical across SRTD4/5/7/9 — gene panel mandatory within IFT-A group."},
            {"term": "Retrograde IFT",      "definition": "Intraflagellar transport from ciliary tip back to cell body powered by dynein-2. IFT-A is required for the anterograde-to-retrograde transition at the tip AND for IFT-B import at the ciliary base."},
            {"term": "NPHP12 paradigm",     "definition": "Hypomorphic C-terminal alleles of TTC21B cause renal-only disease (nephronophthisis type 12) without narrow thorax — same allele-severity paradigm as WDR19/NPHP13. Both genes are dual SRTD/NPHP genes."},
            {"term": "JBTS12",              "definition": "Joubert syndrome type 12 — caused by specific hypomorphic TTC21B alleles. Shows molar tooth sign (MTS) on MRI, cerebellar vermis hypoplasia. UNIQUE to TTC21B among SRTD genes: bridges SRTD and Joubert ciliopathy spectra. Absent in full SRTD4 null alleles."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling pathway processed at the ciliary tip. IFT-A failure → Hedgehog components cannot be transported → GLI3R accumulates → short ribs, narrow thorax, short limbs (SRTD phenotype in chondrocytes)."},
            {"term": "Cell-autonomous",    "definition": "The IFT defect is intrinsic to the transplanted organ's cells. After renal transplantation, the new kidney (donor's TTC21B+) functions normally — CURATIVE for renal disease, no recurrence post-transplant."},
        ],
        "key_variants": [
            {"variant": "p.Arg484Cys",  "domain": "Central TPR cluster II (aa 301–700; IFT122 contact surface)", "consequence": "TPR helix distortion; IFT122 binding impaired; European compound het; moderate SRTD4",     "ethnicity": "European"},
            {"variant": "p.Gly605Arg",  "domain": "Central TPR α-helix (WDR19 C-tail binding, aa 590–620)",     "consequence": "WDR19 C-tail contact disrupted; Middle Eastern homozygous; moderate-severe SRTD4",              "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Trp1100Ter", "domain": "C-terminal truncating (null allele)",                          "consequence": "Premature stop; biallelic null → SRPS spectrum; pan-ethnic",                                  "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala775Val",  "domain": "C-terminal TPR cluster III (aa 701–1100; allele severity)",   "consequence": "Hypomorphic; South Asian; NPHP12 renal-dominant mild; no narrow thorax",                     "ethnicity": "South Asian"},
            {"variant": "p.Arg924Gln",  "domain": "WDR19 C-tail junction region (C-terminal TPR III)",           "consequence": "WDR19 C-tail junction missense; MENA homozygous; moderate SRTD4; renal secondary",          "ethnicity": "MENA"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months",
            "2. Respiratory: neonatal mechanical ventilation if severe null alleles; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous). NPHP12 alleles → renal-only management; no thoracic intervention needed",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF; hepatology referral; avoid hepatotoxic drugs",
            "6. JBTS12 alleles: neurology review; brain MRI at diagnosis and every 3–5 yr; ataxia/oculomotor assessment; breathing irregularity monitoring",
            "7. Genetics: cascade testing of parents + siblings; TTC21B allele-severity counselling (SRTD4 vs NPHP12 vs JBTS12 phenotype depends on specific variant combination); prenatal/preimplantation genetics (25% recurrence risk)",
            "8. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology; neurology if JBTS12 suspected",
        ],
        "ddx_table": [
            {"disease": "SRTD5 (WDR19)",      "key_difference": "Most common IFT-A SRTD; same IFT-A mechanism and short stubby cilia EM; WDR19 unique: ectodermal features (CED1) — absent in SRTD4; gene panel resolves; WDR19 contacts TTC21B C-tail directly"},
            {"disease": "SRTD9 (IFT140)",     "key_difference": "IFT140 is the core scaffold WD40; TTC21B is the TPR adaptor; same IFT-A short stubby cilia; gene panel mandatory within IFT-A group"},
            {"disease": "SRTD7 (WDR35)",      "key_difference": "WDR35 is a 14-blade β-propeller; TTC21B is a TPR protein; same IFT-A mechanism; gene panel required"},
            {"disease": "SRTD3 (DYNC2H1)",    "key_difference": "Most common SRTD (~50%); DYNEIN-2 (not IFT-A); CLUB cilia on EM (not short stubby) — key EM distinction from SRTD4"},
            {"disease": "NPHP12 (TTC21B)",    "key_difference": "SAME GENE — hypomorphic C-terminal TTC21B alleles → renal-only (no narrow thorax); allele severity distinguishes SRTD4 vs NPHP12"},
            {"disease": "JBTS12 (TTC21B)",    "key_difference": "SAME GENE — hypomorphic TTC21B alleles → Joubert MTS (cerebellar vermis hypoplasia on MRI); ABSENT in full SRTD4 null alleles — UNIQUE feature of TTC21B"},
            {"disease": "NPHP13 (WDR19)",     "key_difference": "WDR19 hypomorphic alleles → renal-only NPHP13; TTC21B → NPHP12; same paradigm but different genes; WDR19 also has CED1 ectodermal alleles (absent in TTC21B)"},
            {"disease": "Ellis-van Creveld (EVC)", "key_difference": "CHD present in ~60% EVC (rare in SRTD4); EVC ectodermal features (teeth, nails); EVC is not IFT-A ciliopathy"},
            {"disease": "Bardet-Biedl (BBS)",  "key_difference": "No narrow thorax in BBS; obesity + hypogonadism absent in SRTD4; BBSome is a different ciliary complex from IFT-A"},
            {"disease": "Joubert (JBTS, other genes)", "key_difference": "Other JBTS genes (CEP290, RPGRIP1L, etc.) do not cause narrow thorax; TTC21B/JBTS12 is unique in bridging SRTD and Joubert spectra"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD4 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
