"""
TCTEX1D2 Short-Rib Thoracic Dysplasia 17 (SRTD17) — Jeune Asphyxiating Thoracic Dystrophy Type 17
=====================================================================================================
Primary Gene : TCTEX1D2 (*617819) — 3q21.3; ~142 aa; T-Complex Protein 1-Like Domain Containing 2
               TCTEX1D2 is a dynein-2 light chain that binds the C-terminal β-propeller face of
               WDR60 (SRTD8) within the cytoplasmic dynein-2 retrograde IFT motor complex.
               TCTEX1D2 stabilises WDR60 within the dynein-2 tail; loss → WDR60 destabilisation
               → partial dynein-2 complex assembly failure → retrograde IFT impairment →
               IFT-B accumulation at cilia tip (bulging/club morphology on EM) →
               Hedgehog signal failure in chondrocytes → narrow thorax, short ribs, short limbs.
               Dynein-2 subunit hierarchy: DYNC2H1 (SRTD3) + DYNC2LI1 (SRTD15) +
               WDR34 (SRTD11) + WDR60 (SRTD8) + TCTEX1D2 (SRTD17) + DYNLRB1/DYNLRB2 (light).
Disease OMIM : #617405 — Short-Rib Thoracic Dysplasia 17 With or Without Polydactyly (SRTD17)
               Also called: Asphyxiating Thoracic Dystrophy 17 (ATD17)
               Severe biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum
Chromosome   : 3q21.3
Inheritance  : Autosomal Recessive — biallelic LOF (homozygous consanguineous or compound het)
Prevalence   : ~1/2,000,000–5,000,000; ~5–15 families reported worldwide (2026);
               Extremely rare: among the rarest dynein-2-subunit SRTD genes;
               Fifth known dynein-2-subunit SRTD gene after DYNC2H1 (SRTD3), WDR60 (SRTD8),
               WDR34 (SRTD11), and DYNC2LI1 (SRTD15)

Protein Structure — TCTEX1D2 (142 aa; dynein-2 light chain)
-------------------------------------------------------------
Domain 1: N-terminal T-complex fold (aa 1–60)   — dimerisation interface; dynein-2 complex docking
Domain 2: Core T-complex 1-like domain (aa 61–110) — WDR60 β-propeller C-face binding surface;
           missense cluster in aa 70–100; loss here decouples WDR60 from dynein-2 tail
Domain 3: C-terminal regulatory helix (aa 111–142) — IFT train assembly regulation; phospho-site

Key pathogenic variant classes (TCTEX1D2):
1. WDR60-interface missense (aa 61–110): compound heterozygous; moderate SRTD17; most common
2. Homozygous truncating (null): biallelic null; SRPS spectrum; consanguineous
3. N-terminal missense (aa 1–60): dimerisation failure; severe to moderate
4. Hypomorphic C-terminal (aa 111–142): mild; surviving adults; renal-dominant presentation
"""

import random
import math

SEED = 389
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
    ('Middle Eastern / North African', 0.40),   # highest consanguinity; homozygous dominant
    ('European',                        0.24),   # compound het most common
    ('South Asian',                     0.20),
    ('East Asian',                      0.08),
    ('Latin American',                  0.05),
    ('Other / Unknown',                 0.03),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous missense (WDR60-interface)',
    'Compound het missense + truncating',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense',
    'Homozygous hypomorphic (mild)',
]
allele_weights = [0.35, 0.28, 0.17, 0.13, 0.07]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.18, 0.50, 0.32])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'None'],
        weights=[0.14, 0.12, 0.74])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.10, 0.10, 0.80])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.45, 0.55])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.05, 0.03, 0.92])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.04, 0.04, 0.92])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.27, 0.10, 0.63])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.50, 0.22, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.15, 0.28, 0.57])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.18, 0.22, 0.40, 0.20])[0]

    misdiagnosis = rng.choices(
        ['EVC (Ellis-van Creveld)', 'Thanatophoric dysplasia (prenatal)',
         'BBS (Bardet-Biedl)', 'SRTD8 / gene panel corrected', 'None'],
        weights=[0.08, 0.07, 0.05, 0.18, 0.62])[0]

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
        "disease":    "TCTEX1D2 Short-Rib Thoracic Dysplasia 17 (SRTD17 / ATD17)",
        "gene":       "TCTEX1D2 (*617819) — 3q21.3 — 142 aa — Dynein-2 Light Chain (T-Complex 1-Like)",
        "omim_gene":  "617819",
        "omim_disease": "617405",
        "chromosome": "3q21.3",
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
            "TCTEX1D2 loss removes the dynein-2 light chain that stabilises WDR60 (SRTD8) within "
            "the dynein-2 tail complex. Without TCTEX1D2, WDR60 is destabilised → partial dynein-2 "
            "complex assembly failure → retrograde IFT from ciliary tip to cell body impaired → "
            "IFT-B pile-up at cilia tip (bulging/club morphology on EM) → Hedgehog (Ihh/Shh) "
            "signalling fails in chondrocytes → NARROW THORAX (primary), short ribs, short limbs. "
            "Secondary: renal TIN/ESRD (~15–20% survivors), rod-cone dystrophy (~8%), CHF (~5%)."
        ),
        "key_distinction": (
            "TCTEX1D2 (SRTD17) is the SMALLEST subunit in the dynein-2 complex (142 aa) — a "
            "T-complex protein 1-like domain light chain docking onto the WDR60 β-propeller C-face. "
            "SRTD17 is extremely rare (~5–15 families worldwide, 2026) and is the fifth dynein-2 "
            "subunit gene associated with SRTD. Homozygous consanguineous families predominate "
            "(Middle Eastern / North African). Phenotype is generally moderate-to-mild compared to "
            "SRTD3 (DYNC2H1); secondary organ involvement (renal, retinal, hepatic) is less "
            "frequent. Gene panel sequencing is mandatory — clinical overlap with SRTD3/8/11/15 "
            "is complete; TCTEX1D2 must be included on the SRTD gene panel."
        ),
        "dynein2_subunit_table": [
            {"subunit": "DYNC2H1",  "role": "AAA+ motor heavy chain (retrograde force)",                         "srtd": "SRTD3",  "omim_gene": "603297", "chr": "11q22.3", "freq": "~50% of SRTD"},
            {"subunit": "DYNC2LI1", "role": "Scaffold light intermediate chain; bridges DYNC2H1 to WDR34",      "srtd": "SRTD15", "omim_gene": "617248", "chr": "2p21",    "freq": "~1–2% of SRTD"},
            {"subunit": "WDR34",    "role": "WD40 6-blade β-propeller intermediate chain (opposite to WDR60)",  "srtd": "SRTD11", "omim_gene": "604126", "chr": "9q34.11", "freq": "~3–5% of SRTD"},
            {"subunit": "WDR60",    "role": "WD40 7-blade β-propeller intermediate chain (opposite to WDR34)",  "srtd": "SRTD8",  "omim_gene": "615462", "chr": "7q36.3",  "freq": "~5–10% of SRTD"},
            {"subunit": "TCTEX1D2", "role": "Light chain; T-complex 1-like domain; WDR60 C-face binding",       "srtd": "SRTD17", "omim_gene": "617819", "chr": "3q21.3",  "freq": "very rare (~5–15 families)"},
            {"subunit": "DYNLRB1/2","role": "Roadblock-type light chains",                                        "srtd": "—",      "omim_gene": "607167", "chr": "20q13.32","freq": "not SRTD-associated"},
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
            {"variant": "p.Ile71Asn (c.212T>A) — T-complex core (WDR60 interface) — Middle Eastern homozygous — moderate SRTD17", "n": rng.randint(3, 5)},
            {"variant": "p.Gly93Glu (c.278G>A) — WDR60-binding surface — compound het European — moderate SRTD17",              "n": rng.randint(2, 4)},
            {"variant": "p.Arg108Ter (c.322C>T) — truncating, null — SRPS spectrum — pan-ethnic",                                "n": rng.randint(2, 3)},
            {"variant": "p.Thr54Ile (c.161C>T) — N-terminal T-complex fold — South Asian hypomorphic — mild",                   "n": rng.randint(1, 3)},
            {"variant": "p.Leu120Pro (c.359T>C) — C-terminal regulatory helix — MENA homozygous — severe",                      "n": rng.randint(1, 2)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "TCTEX1D2 — T-Complex Protein 1-Like Domain Containing 2",
            "also_known_as":    "TCTEX type 1 domain containing 2; dynein-2 light chain",
            "omim_gene":        "*617819",
            "chromosome":       "3q21.3",
            "protein_size":     "142 aa",
            "protein_class":    "Dynein-2 light chain — T-complex 1-like domain",
            "domain_structure":  "N-terminal T-complex dimerisation fold (aa 1–60) + core WDR60-interface domain (aa 61–110) + C-terminal regulatory helix (aa 111–142)",
            "function":         "Stabilises WDR60 within the dynein-2 tail; essential for full retrograde IFT motor assembly; contacts WDR60 β-propeller C-face",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "WDR60 SRTD8 (C-terminal β-propeller face)",
            "expression":       "Cilia (primary non-motile 9+0); chondrocytes; renal tubular epithelium; retinal photoreceptor connecting cilium",
        },
        "disease_card": {
            "omim_disease":       "#617405",
            "full_name":          "Short-Rib Thoracic Dysplasia 17 With or Without Polydactyly",
            "alternative_names":  "SRTD17; Asphyxiating Thoracic Dystrophy 17 (ATD17); Jeune syndrome type 17",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":        "~25% (postaxial; less frequent than SRTD3/8/11/15)",
            "renal":              "TIN + corticomedullary cysts → ESRD in ~15–20% of survivors",
            "retinal":            "Rod-cone dystrophy in ~8% (connecting cilium IFT failure)",
            "hepatic_chf":        "Ductal plate malformation → CHF in ~5%",
            "situs_inversus":     "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":        "ABSENT — no brainstem/vermis midline defect",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); less common than SRTD3",
            "renal_transplant":   "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":         "~1/2,000,000–5,000,000; ~5–15 families worldwide (2026); extremely rare",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones",
            "3. GENE PANEL (mandatory first-line): DYNC2H1, DYNC2LI1, WDR34, WDR60, TCTEX1D2 — TCTEX1D2 must be included; clinical overlap is complete",
            "4. WES + CNV if panel negative — small gene; small deletions may be missed by NGS alone",
            "5. Skeletal ciliopathy EM (skin/fibroblast biopsy): bulging/club cilia tip — same as SRTD3/8/11/15",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "7. ERG annually from age 6 (rod-cone dystrophy ~8%); ophthalmology referral",
            "8. Liver USS + APRI if hepatic signs (CHF ~5%)",
            "9. Echocardiography if CHD suspected",
            "10. MLPA for TCTEX1D2 deletions if sequencing negative but phenotype compelling (gene is small — CNV analysis mandatory)",
        ],
        "mechanism_glossary": [
            {"term": "TCTEX1D2",      "definition": "T-Complex Protein 1-Like Domain Containing 2 — the smallest dynein-2 subunit (142 aa); a light chain that docks onto the WDR60 β-propeller C-face to stabilise WDR60 in the dynein-2 tail."},
            {"term": "Retrograde IFT", "definition": "Intraflagellar transport from ciliary tip back to cell body, powered by dynein-2. Required for recycling IFT trains and turning off Hedgehog signalling at the tip."},
            {"term": "T-complex fold", "definition": "Structural domain shared with testis-specific T-complex proteins; in TCTEX1D2 adapted for dynein-2 light chain function and WDR60 binding rather than spermatogenesis."},
            {"term": "Club/bulging cilia", "definition": "EM finding: IFT-B accumulates at the ciliary tip when retrograde IFT fails → tip swells into a club shape. Identical finding in SRTD3, 8, 11, 15, 17 — cannot distinguish by EM alone."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling pathway processed at the ciliary tip. Failure → GLI3R not processed → short ribs, narrow thorax, short limbs (SRTD phenotype)."},
            {"term": "SRPS spectrum",      "definition": "Short-Rib Polydactyly Syndrome — perinatal-lethal SRTD alleles (biallelic null). Less common in SRTD17 than SRTD3 (DYNC2H1) due to partial assembly rescue."},
            {"term": "WDR60 (SRTD8)",      "definition": "Direct binding partner of TCTEX1D2 (WDR60 β-propeller C-face). Loss of TCTEX1D2 secondarily destabilises WDR60 — SRTD17 and SRTD8 have overlapping but distinct molecular basis."},
            {"term": "Cell-autonomous",    "definition": "The IFT defect is intrinsic to the transplanted organ's cells. After renal transplantation, the new kidney (donor's TCTEX1D2+) functions normally — CURATIVE for renal disease, no recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Ile71Asn",  "domain": "T-complex core / WDR60 interface (aa 61–110)", "consequence": "Disrupts WDR60 C-face binding; compound het or homozygous; moderate SRTD17",          "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Gly93Glu",  "domain": "WDR60-binding surface (aa 80–100)",            "consequence": "Charge reversal at WDR60 interface; European compound het; moderate SRTD17",           "ethnicity": "European"},
            {"variant": "p.Arg108Ter", "domain": "Near-C-terminal (truncating)",                  "consequence": "Null allele; SRPS spectrum in biallelic null; pan-ethnic",                              "ethnicity": "Pan-ethnic"},
            {"variant": "p.Thr54Ile",  "domain": "N-terminal T-complex fold (aa 40–60)",          "consequence": "Hypomorphic; mild; surviving adults; renal-dominant late presentation",                "ethnicity": "South Asian"},
            {"variant": "p.Leu120Pro", "domain": "C-terminal regulatory helix (aa 111–142)",      "consequence": "Disrupts regulatory helix; MENA homozygous; severe thorax; WDR60 contact affected",   "ethnicity": "MENA"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months",
            "2. Respiratory: neonatal mechanical ventilation if severe; CPAP/NIV for moderate thorax; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous)",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF; hepatology referral; avoid hepatotoxic drugs",
            "6. Genetics: cascade testing of parents + siblings; consanguinity risk counselling; prenatal/preimplantation genetics for recurrence risk (25%)",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",    "key_difference": "Most common SRTD (~50%); same phenotype; heavier secondary organ involvement; DYNC2H1 is motor, TCTEX1D2 is light chain"},
            {"disease": "SRTD8 (WDR60)",      "key_difference": "WDR60 is the direct binding partner of TCTEX1D2; loss of TCTEX1D2 secondarily destabilises WDR60; gene panel is the only differentiator"},
            {"disease": "SRTD11 (WDR34)",     "key_difference": "WDR34 is on the opposite side of the dynein-2 tail from WDR60; gene panel resolves"},
            {"disease": "SRTD15 (DYNC2LI1)",  "key_difference": "DYNC2LI1 is a scaffold chain, TCTEX1D2 is a light chain; both affect dynein-2 tail stability; gene panel required"},
            {"disease": "Ellis-van Creveld (EVC)", "key_difference": "CHD present in ~60% EVC (rare in SRTD17); EVC ectodermal features (teeth, nails) absent in SRTD17"},
            {"disease": "BBS (Bardet-Biedl)", "key_difference": "No narrow thorax in BBS; obesity + hypogonadism absent in SRTD17; BBS-specific gene panel"},
            {"disease": "Thanatophoric dysplasia", "key_difference": "Dominant FGFR3 mutation; cloverleaf skull; telephone-receiver femora; not ciliary"},
            {"disease": "NPHP (nephronophthisis)", "key_difference": "NPHP → renal is PRIMARY; no short ribs in NPHP; different ciliary compartment (TZ not IFT)"},
            {"disease": "Joubert syndrome (JBTS)", "key_difference": "Molar tooth sign (MTS) on MRI — ABSENT in SRTD17; cerebellar vermis hypoplasia in JBTS not SRTD"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD17 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
