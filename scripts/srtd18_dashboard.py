"""
IFT43 Short-Rib Thoracic Dysplasia 18 (SRTD18) — Jeune Asphyxiating Thoracic Dystrophy Type 18
=================================================================================================
Primary Gene : IFT43 (*614068) — 14q24; ~378 aa; Intraflagellar Transport Protein 43
               IFT43 is the SMALLEST subunit of the IFT-A satellite module (peripheral IFT-A).
               IFT43 directly contacts the N-terminal WD40 blade of IFT121 (WDR35, SRTD7) and
               anchors the satellite module to the IFT-A complex periphery. Loss of IFT43
               → IFT121 destabilised within the satellite module → entire IFT-A satellite
               weakened → retrograde IFT (tip-to-base) impaired → SHORT-STUBBY CILIA (IFT-A
               class on EM, same as SRTD2/4/5/7/9) → IFT-B precursor accumulation at cilia
               base → Hedgehog signal failure in chondrocytes → narrow thorax, short ribs,
               shortened limbs, campomelia (bowing of long bones — DISTINCTIVE SRTD18 feature).
               Allele-specific dual phenotype: severe biallelic LOF → SRTD18/SRPS; hypomorphic
               → CED3 (Cranioectodermal Dysplasia 3 / Sensenbrenner syndrome with ectodermal
               features: craniosynostosis, sparse hair, dental anomalies, narrow thorax).
               IFT-A satellite hierarchy: IFT121 (SRTD7) + IFT43 (SRTD18) — the only two
               IFT-A satellite core components (distinct from ARM-hub: IFT144/IFT140/IFT122).
Disease OMIM : #617866 — Short-Rib Thoracic Dysplasia 18 With Polydactyly (SRTD18)
               Also called: Asphyxiating Thoracic Dystrophy 18 (ATD18)
               Severe null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum
               Hypomorphic alleles → CED3 (Cranioectodermal Dysplasia 3; Sensenbrenner syndrome)
Chromosome   : 14q24
Inheritance  : Autosomal Recessive — biallelic LOF (homozygous consanguineous or compound het)
Prevalence   : ~1/1,000,000–3,000,000; ~10–25 families reported worldwide (2026);
               Very rare: among the rarest IFT-A satellite SRTD genes;
               Second IFT-A satellite-core SRTD gene after IFT121/WDR35 (SRTD7)

Protein Structure — IFT43 (378 aa; IFT-A satellite peripheral subunit)
------------------------------------------------------------------------
Domain 1: N-terminal adaptor module (aa 1–110)  — IFT121 N-terminal WD40 blade-1-3 binding;
           missense cluster aa 70–110; loss decouples satellite from IFT121; W174R nearby
Domain 2: Central linker (aa 111–200)           — IFT121-contact interface; main pathogenic
           missense cluster (W174R in this region); allosteric communication to IFT-A ARM hub
Domain 3: C-terminal anchoring domain (aa 201–378) — positions IFT43 at IFT-A periphery;
           CED3 hypomorphic alleles cluster here (reduced but not absent IFT-A satellite)

Key pathogenic variant classes (IFT43):
1. IFT121-interface missense (aa 70–200): homozygous; moderate-to-severe SRTD18; campomelia
2. Compound het missense + splice: compound het; variable severity; European predominant
3. Homozygous truncating (null): biallelic null; SRPS spectrum; consanguineous
4. C-terminal hypomorphic missense (aa 201–378): CED3/Sensenbrenner syndrome spectrum; mild
5. Compound het: one null + one hypomorphic: intermediate SRTD18; variable campomelia
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
    ('Middle Eastern / North African', 0.38),   # highest consanguinity; W174R homozygous dominant
    ('European',                        0.26),   # compound het most common; CED3 hypomorphic
    ('South Asian',                     0.18),
    ('East Asian',                      0.10),
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
    'Homozygous missense (IFT121-interface)',
    'Compound het missense + splice/truncating',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense',
    'C-terminal hypomorphic (CED3 spectrum)',
]
allele_weights = [0.34, 0.27, 0.16, 0.15, 0.08]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele
    ced3   = 'CED3' in allele or 'hypomorphic' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.20, 0.48, 0.32])[0]

    campomelia = rng.choices(
        ['Present (bilateral bowing)', 'Present (unilateral)', 'Absent'],
        weights=[0.35, 0.15, 0.50])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial', 'None'],
        weights=[0.22, 0.14, 0.04, 0.60])[0]

    ced3_features = rng.choices(
        ['Craniosynostosis + sparse hair + dental', 'Sparse hair + dental only', 'None'],
        weights=[0.05, 0.04, 0.91])[0] if ced3 else 'None'

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'Cystic (no TIN)', 'None'],
        weights=[0.12, 0.10, 0.05, 0.73])[0]

    ckd_stage = (
        rng.choices(['CKD 3–4', 'CKD 5 / ESRD'], weights=[0.40, 0.60])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.08, 0.05, 0.87])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Hepatic fibrosis (CED3)', 'Elevated LFTs only', 'None'],
        weights=[0.05, 0.04, 0.04, 0.87])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.30, 0.12, 0.58])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.48, 0.24, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.17, 0.28, 0.55])[0]

    presentation = rng.choices(
        ['Prenatal (short limbs + campomelia + polyhydramnios)', 'Neonatal respiratory failure',
         'Infant thorax + limb + bowing anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.22, 0.20, 0.38, 0.20])[0]

    misdiagnosis = rng.choices(
        ['SRTD7 / WDR35 (IFT-A satellite)', 'SRTD5 / WDR19 (IFT-A ARM hub)',
         'EVC (Ellis-van Creveld)', 'Thanatophoric dysplasia (prenatal)',
         'CED1 / CED2 (Sensenbrenner — before IFT43 sequenced)', 'None'],
        weights=[0.18, 0.08, 0.07, 0.06, 0.05, 0.56])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.38, 0.28, 0.22, 0.12])[0]

    sex = rng.choice(['M', 'F'])

    patients.append(dict(
        ethnicity    = eth,
        allele       = allele,
        null_allele  = null,
        ced3         = ced3,
        thorax_sev   = thorax_sev,
        campomelia   = campomelia,
        poly_status  = poly_status,
        ced3_features= ced3_features,
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
campomelia_n      = sum(1 for p in patients if p['campomelia'] != 'Absent')
polydactyly_n     = sum(1 for p in patients if p['poly_status'] != 'None')
renal_any_n       = sum(1 for p in patients if p['renal'] != 'None')
retinal_any_n     = sum(1 for p in patients if p['retinal'] != 'None')
hepatic_chf_n     = sum(1 for p in patients if p['hepatic'] != 'None')
ced3_features_n   = sum(1 for p in patients if p['ced3_features'] != 'None')
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
        "disease":    "IFT43 Short-Rib Thoracic Dysplasia 18 (SRTD18 / ATD18)",
        "gene":       "IFT43 (*614068) — 14q24 — 378 aa — IFT-A Satellite Core Component (Smallest IFT-A Subunit)",
        "omim_gene":  "614068",
        "omim_disease": "617866",
        "chromosome": "14q24",
        "cohort_n":   N,
        "seed":       SEED,
        "kpis": {
            "thorax_severe_n":    thorax_severe_n,
            "thorax_severe_pct":  _pct(thorax_severe_n),
            "campomelia_n":       campomelia_n,
            "campomelia_pct":     _pct(campomelia_n),
            "polydactyly_n":      polydactyly_n,
            "polydactyly_pct":    _pct(polydactyly_n),
            "renal_any_n":        renal_any_n,
            "renal_any_pct":      _pct(renal_any_n),
            "retinal_any_n":      retinal_any_n,
            "retinal_any_pct":    _pct(retinal_any_n),
            "hepatic_chf_n":      hepatic_chf_n,
            "hepatic_chf_pct":    _pct(hepatic_chf_n),
            "ced3_features_n":    ced3_features_n,
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
            "IFT43 loss removes the SMALLEST IFT-A satellite component, which directly contacts "
            "the N-terminal WD40 blade of IFT121 (WDR35, SRTD7). Without IFT43, IFT121 is "
            "destabilised within the satellite module → the IFT-A satellite collapses → retrograde "
            "IFT (tip-to-base) is impaired → SHORT-STUBBY CILIA on EM (IFT-A class — same as "
            "SRTD2/4/5/7/9) → IFT-B precursor accumulates at cilia base → Hedgehog (Ihh/Shh) "
            "signalling fails in chondrocytes → NARROW THORAX (primary), short ribs, short limbs. "
            "DISTINCTIVE: campomelia (bowing of long bones) in ~50% — highest frequency among "
            "IFT-A SRTDs. Secondary: renal TIN/cysts (~27%), retinal dystrophy (~13%), CHF (~9%). "
            "Dual phenotype: null alleles → SRTD18/SRPS; hypomorphic → CED3 (Sensenbrenner)."
        ),
        "key_distinction": (
            "IFT43 (SRTD18) is the SECOND IFT-A satellite core component SRTD gene after "
            "IFT121/WDR35 (SRTD7). IFT43 is the SMALLEST IFT-A subunit (378 aa) — the peripheral "
            "anchor of the satellite module. SRTD18 is UNIQUE among IFT-A SRTDs for: (1) campomelia "
            "(bowing) as a distinctive EM/radiograph finding; (2) allele-specific dual phenotype: "
            "biallelic null → SRTD18, hypomorphic → CED3/Sensenbrenner syndrome (craniosynostosis "
            "+ sparse hair + dental anomalies). The IFT43-W174R missense (Arabian Peninsula founder) "
            "is the most cited allele. Short-stubby cilia on EM are IFT-A class — identical to "
            "SRTD5/7/9, so gene panel sequencing is the ONLY differentiator. "
            "IFT43 must be included on any comprehensive skeletal ciliopathy gene panel."
        ),
        "ift_a_satellite_table": [
            {"subunit": "IFT144 (WDR19)",  "role": "ARM hub — IFT-A core scaffold, binds IFT140 and IFT122", "srtd": "SRTD5",  "omim_gene": "608151", "chr": "4p14",    "freq": "~10–15% IFT-A SRTD"},
            {"subunit": "IFT140",           "role": "ARM hub — bridges IFT144 and IFT122; IFT-A core",       "srtd": "SRTD9",  "omim_gene": "614620", "chr": "16p13.3", "freq": "~10–15% IFT-A SRTD"},
            {"subunit": "IFT139 (TTC21B)", "role": "ARM hub — retrograde IFT; anterograde cargo release",   "srtd": "SRTD4",  "omim_gene": "612014", "chr": "2q24.3",  "freq": "~5–8% IFT-A SRTD"},
            {"subunit": "IFT122",           "role": "ARM hub — bridges IFT144 N-face and IFT43 satellite",   "srtd": "SRTD2",  "omim_gene": "606045", "chr": "3q21.3",  "freq": "~5–8% IFT-A SRTD"},
            {"subunit": "IFT121 (WDR35)",  "role": "Satellite — direct IFT43 binding partner; N-WD40",     "srtd": "SRTD7",  "omim_gene": "613602", "chr": "2p24.1",  "freq": "~5–8% IFT-A SRTD"},
            {"subunit": "IFT43",            "role": "Satellite — SMALLEST subunit; anchors satellite to IFT121", "srtd": "SRTD18", "omim_gene": "614068", "chr": "14q24",   "freq": "very rare (~10–25 families)"},
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
        "campomelia_distribution": [
            {"label": "Present (bilateral bowing)",  "n": _cnt('campomelia', 'Present (bilateral bowing)')},
            {"label": "Present (unilateral)",        "n": _cnt('campomelia', 'Present (unilateral)')},
            {"label": "Absent",                      "n": _cnt('campomelia', 'Absent')},
        ],
        "polydactyly_distribution": [
            {"label": "Postaxial (bilateral)",  "n": _cnt('poly_status', 'Postaxial (bilateral)')},
            {"label": "Postaxial (unilateral)", "n": _cnt('poly_status', 'Postaxial (unilateral)')},
            {"label": "Preaxial",               "n": _cnt('poly_status', 'Preaxial')},
            {"label": "None",                   "n": _cnt('poly_status', 'None')},
        ],
        "renal_distribution": _dist('renal'),
        "ckd_stage_distribution": [
            {"label": "CKD 3–4",          "n": _cnt('ckd_stage', 'CKD 3–4')},
            {"label": "CKD 5 / ESRD",     "n": _cnt('ckd_stage', 'CKD 5 / ESRD')},
            {"label": "None / not applicable", "n": _cnt('ckd_stage', 'None')},
        ],
        "retinal_distribution": _dist('retinal'),
        "hepatic_distribution": _dist('hepatic'),
        "ced3_distribution": _dist('ced3_features'),
        "veptr_distribution": _dist('veptr'),
        "presentation_distribution": _dist('presentation'),
        "misdiagnosis_distribution": _dist('misdiagnosis'),
        "allele_class_summary": _dist('allele'),
        "ethnicity_distribution": _eth_dist(),
        "treatment_renal": _dist('tx_renal'),
        "respiratory_management": _dist('resp_mgmt'),
        "top_variants": [
            {"variant": "p.Trp174Arg (c.520T>C) — Central linker / IFT121-interface — Arabian Peninsula founder — SRTD18 + campomelia", "n": rng.randint(4, 6)},
            {"variant": "p.Glu98Lys (c.292G>A) — N-terminal adaptor IFT121-blade contact — MENA homozygous — severe SRTD18",           "n": rng.randint(2, 4)},
            {"variant": "p.Arg243Ter (c.727C>T) — Truncating null — C-terminal anchoring lost — SRPS spectrum — pan-ethnic",           "n": rng.randint(2, 3)},
            {"variant": "p.Leu312Pro (c.935T>C) — C-terminal anchoring domain — CED3/Sensenbrenner hypomorphic — European",             "n": rng.randint(1, 3)},
            {"variant": "p.Gly78Val (c.233G>T) — N-terminal adaptor IFT121 blade-2 — compound het South Asian — moderate SRTD18",       "n": rng.randint(1, 2)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "IFT43 — Intraflagellar Transport Protein 43",
            "also_known_as":    "C14orf179; KIAA0602 (partial); IFT-A satellite core component",
            "omim_gene":        "*614068",
            "chromosome":       "14q24",
            "protein_size":     "378 aa",
            "protein_class":    "IFT-A satellite peripheral subunit — smallest IFT-A component",
            "domain_structure":  (
                "N-terminal adaptor module (aa 1–110) — IFT121 N-terminal WD40 blade-1-3 binding; "
                "Central linker (aa 111–200) — IFT121-contact interface; W174R pathogenic cluster; "
                "C-terminal anchoring domain (aa 201–378) — positions satellite at IFT-A periphery; "
                "CED3/Sensenbrenner hypomorphic cluster"
            ),
            "function":         (
                "Anchors the IFT-A satellite module by directly binding IFT121 (WDR35) N-terminal "
                "WD40 domain. Loss of IFT43 destabilises IFT121 → IFT-A satellite collapses → "
                "retrograde IFT (tip-to-base) failure → short-stubby cilia (IFT-A EM class)"
            ),
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT121/WDR35 (SRTD7) N-terminal WD40 blades 1–3; IFT122 peripheral face",
            "expression":       "Cilia (primary non-motile 9+0); chondrocytes; renal tubular epithelium; retinal connecting cilium; craniofacial suture cells",
        },
        "disease_card": {
            "omim_disease":       "#617866",
            "full_name":          "Short-Rib Thoracic Dysplasia 18 With Polydactyly",
            "alternative_names":  "SRTD18; Asphyxiating Thoracic Dystrophy 18 (ATD18); Jeune syndrome type 18",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "campomelia":         "~50% (bowing of long bones — DISTINCTIVE SRTD18 feature; highest rate among IFT-A SRTDs)",
            "polydactyly":        "~40% (postaxial; higher frequency than SRTD17; preaxial in 4%)",
            "renal":              "TIN + corticomedullary cysts → ESRD in ~22% of survivors",
            "retinal":            "Rod-cone dystrophy in ~13% (connecting cilium IFT failure)",
            "hepatic_chf":        "CHF / hepatic fibrosis in ~9% (ductal plate malformation; CED3 overlap)",
            "ced3_overlap":       "CED3/Sensenbrenner syndrome in hypomorphic alleles: craniosynostosis + sparse hair + dental anomalies + narrow thorax",
            "situs_inversus":     "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":        "ABSENT — no brainstem/vermis midline defect",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); less common than SRTD3",
            "renal_transplant":   "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention); campomelia correction (osteotomy) if severe",
            "prevalence":         "~1/1,000,000–3,000,000; ~10–25 families worldwide (2026); very rare",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, campomelia (bowing), polyhydramnios → suspect SRTD18 (campomelia highly suggestive vs. other IFT-A SRTDs)",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened/bowed tubular bones; radiograph pattern with campomelia distinguishes SRTD18",
            "3. GENE PANEL (mandatory first-line): IFT144/WDR19, IFT140, IFT122, IFT121/WDR35, IFT43 — IFT43 MUST be included; clinical overlap with all IFT-A SRTDs is complete",
            "4. WES + CNV if panel negative — small gene; intragenic deletions may be missed by NGS; MLPA for IFT43 CNV analysis",
            "5. Skeletal ciliopathy EM (skin/fibroblast): SHORT-STUBBY CILIA — IFT-A class (cannot distinguish SRTD2/4/5/7/9/18 by EM alone; gene panel resolves)",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "7. ERG annually from age 6 (rod-cone dystrophy ~13%); ophthalmology referral",
            "8. Liver USS + APRI if hepatic signs; CED3 overlap: cranial USS/CT for craniosynostosis; dermatology for sparse hair",
            "9. Echocardiography if CHD suspected",
            "10. CED3/Sensenbrenner evaluation: cranial suture assessment + dental exam + hair examination if hypomorphic alleles suspected",
        ],
        "mechanism_glossary": [
            {"term": "IFT43",               "definition": "Intraflagellar Transport Protein 43 — 378 aa; the SMALLEST IFT-A subunit; peripheral anchor of the IFT-A satellite module docking onto IFT121/WDR35."},
            {"term": "IFT-A satellite module", "definition": "Peripheral sub-complex of IFT-A formed by IFT121 (WDR35) and IFT43. Distinct from the ARM-hub (IFT144/IFT140/IFT122). Loss of either satellite component → IFT-A satellite collapse."},
            {"term": "Retrograde IFT",      "definition": "Intraflagellar transport from ciliary tip back to cell body (tip-to-base), mediated by dynein-2. IFT-A disruption impairs retrograde IFT → short-stubby cilia morphology on EM."},
            {"term": "Short-stubby cilia",  "definition": "EM finding in IFT-A gene defects: cilia are shortened and stubby (not club/bulging like dynein-2 defects). Identical in SRTD2/4/5/7/9/18 — cannot distinguish by EM; gene panel required."},
            {"term": "Campomelia",          "definition": "Bowing (angular deformity) of long bones (femur, tibia, fibula, humerus). DISTINCTIVE in SRTD18 — highest frequency among IFT-A SRTDs (~50%). Mechanism: IFT-A failure → chondrocyte Hh signalling failure → asymmetric periosteal remodelling."},
            {"term": "CED3 / Sensenbrenner", "definition": "Cranioectodermal Dysplasia type 3 — caused by hypomorphic IFT43 alleles. Features: craniosynostosis, sparse hair, hypodontia/dental anomalies, narrow thorax, hepatic fibrosis. Allele class governs SRTD18 vs. CED3."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate and craniofacial signalling pathway processed at the ciliary tip. IFT-A failure → GLI processing failure → short ribs, narrow thorax, campomelia (SRTD) and craniosynostosis (CED3)."},
            {"term": "Cell-autonomous",    "definition": "The IFT defect is intrinsic to each organ's own cells. After renal transplantation, the donor kidney (IFT43+) functions normally — CURATIVE, no recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Trp174Arg (W174R)",  "domain": "Central linker / IFT121-contact (aa 111–200)",    "consequence": "Weakens IFT43-IFT121 docking; most common SRTD18 allele; Arabian Peninsula founder; campomelia present", "ethnicity": "Arabian Peninsula / MENA (homozygous)"},
            {"variant": "p.Glu98Lys",            "domain": "N-terminal adaptor — IFT121 blade-1 contact",    "consequence": "Charge reversal disrupts IFT121 blade-1 binding; MENA homozygous; severe SRTD18",                   "ethnicity": "MENA (homozygous)"},
            {"variant": "p.Arg243Ter",           "domain": "C-terminal anchoring domain (truncating null)",  "consequence": "Loss of anchoring domain; null allele; SRPS spectrum in biallelic null; pan-ethnic",               "ethnicity": "Pan-ethnic"},
            {"variant": "p.Leu312Pro",           "domain": "C-terminal anchoring domain (hypomorphic)",     "consequence": "Hypomorphic; residual IFT43 → CED3/Sensenbrenner spectrum; mild thorax; craniosynostosis",         "ethnicity": "European"},
            {"variant": "p.Gly78Val",            "domain": "N-terminal adaptor IFT121 blade-2 surface",     "consequence": "Missense; compound het with truncating; moderate SRTD18; campomelia in ~40%",                      "ethnicity": "South Asian"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months",
            "2. CAMPOMELIA: orthopaedic osteotomy for severe bowing if functional impairment; night splinting mild cases; physiotherapy to prevent contracture",
            "3. Respiratory: neonatal mechanical ventilation if severe thorax; CPAP/NIV for moderate thorax; wean as thorax expands with VEPTR",
            "4. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous)",
            "5. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy 2026",
            "6. Hepatic: APRI + USS annually if CHF/fibrosis suspected; hepatology referral; avoid hepatotoxic drugs; liver transplant for CED3 hepatic failure (rare)",
            "7. CED3/Sensenbrenner (hypomorphic alleles): craniofacial surgery for craniosynostosis; dental surveillance; dermatology (sparse hair); multidisciplinary CED clinic",
            "8. Genetics: cascade testing; consanguinity counselling; prenatal/preimplantation genetics (25% recurrence); CED3 vs. SRTD18 allele class counselling",
            "9. MDT: paediatric orthopaedics (VEPTR + osteotomy), nephrology, respiratory, ophthalmology, genetics, hepatology, craniofacial surgery (CED3)",
        ],
        "ddx_table": [
            {"disease": "SRTD7 (IFT121/WDR35)",  "key_difference": "IFT43 direct binding partner — satellite partner; campomelia less frequent in SRTD7; only gene panel resolves"},
            {"disease": "SRTD5 (WDR19/IFT144)", "key_difference": "IFT-A ARM hub; most common IFT-A SRTD (~10–15%); no CED3 dual phenotype; same short-stubby cilia EM class; gene panel required"},
            {"disease": "SRTD9 (IFT140)",       "key_difference": "IFT-A ARM hub; same short-stubby cilia EM; campomelia rare in SRTD9 vs. common in SRTD18; gene panel resolves"},
            {"disease": "SRTD2 (IFT122)",       "key_difference": "IFT-A ARM hub; CED2 dual phenotype (same allele-class concept); IFT122 is larger ARM-hub vs. IFT43 satellite; gene panel required"},
            {"disease": "CED1 / CED2",           "key_difference": "Cranioectodermal Dysplasia — same allele-class spectrum concept; CED1=IFT122, CED2=WDR35/IFT121; CED3=IFT43; gene panel is the ONLY differentiator"},
            {"disease": "SRTD3 (DYNC2H1)",      "key_difference": "Dynein-2 motor; most common SRTD (~50%); club/bulging cilia (NOT short-stubby — EM distinguishes); no CED3 overlap"},
            {"disease": "EVC (Ellis-van Creveld)", "key_difference": "CHD present in ~60% EVC; ectodermal features (hair/nails/teeth) in EVC differ from CED3/IFT43; EVC2/EVC genes; not IFT-A"},
            {"disease": "Thanatophoric dysplasia", "key_difference": "Dominant FGFR3 mutation; cloverleaf skull; telephone-receiver femora; campomelia in TD is different mechanism"},
            {"disease": "Joubert syndrome (JBTS)", "key_difference": "Molar tooth sign (MTS) on MRI — ABSENT in SRTD18; different ciliary compartment (TZ/CEP290 class not IFT-A)"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD18 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("Campomelia:", b["campomelia_distribution"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
