"""
IFT80 Short-Rib Thoracic Dysplasia 1 (SRTD1) — Jeune Asphyxiating Thoracic Dystrophy Type 1
================================================================================================
Primary Gene : IFT80 (*611229) — 3q25.33; ~801 aa; Intraflagellar Transport Protein 80
               IFT80 is a WD40 β-propeller protein and a subunit of the IFT-B2 subcomplex
               (the distal, tip-proximal half of the IFT-B anterograde transport complex).
               IFT-B2 contains: IFT80, IFT88, IFT57, IFT38/CLUAP1, IFT172, IFT25, IFT27.
               IFT80 directly contacts IFT38/CLUAP1 (SRTD13 gene) and IFT88 within IFT-B2.
               Loss → IFT-B2 cannot assemble → IFT-B anterograde complex truncated → kinesin-2
               (KIF3A/KIF3B/KAP) cannot transport the full IFT-B train toward the ciliary tip →
               SHORTENED CILIA (EM; distinct from IFT-A short stubby and dynein-2 club cilia) →
               Hedgehog (Ihh/Shh) signal components cannot reach tip for processing → GLI3R
               accumulates in chondrocytes → NARROW THORAX, short ribs, short limbs.
               HISTORIC: IFT80 was the FIRST gene identified for Jeune ATD/SRTD (Beales et al.,
               2007, Nature Genetics), making SRTD1 the founding genetic entry of the SRTD series.

Disease OMIM : #208500 — Asphyxiating Thoracic Dystrophy 1 (ATD1 / SRTD1 / Jeune syndrome type 1)
               Original historic OMIM number for Jeune syndrome, now assigned to IFT80-caused cases.
               Severe biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum.
               No separate renal-only (NPHP) or Joubert phenotype described for IFT80 (unlike TTC21B).
Chromosome   : 3q25.33
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1/300,000–600,000; ~20–30 families reported worldwide (2026);
               RARE even among SRTD genes; ~0.5–1% of molecularly confirmed SRTD cases;
               Much rarer than DYNC2H1/SRTD3 (~50%) and WDR19/SRTD5 (~1%).

Protein Structure — IFT80 (801 aa; IFT-B2 WD40 β-propeller)
--------------------------------------------------------------
Domain 1: N-terminal WD40 β-propeller (aa 1–400)  — forms canonical 7-blade WD40 propeller;
           IFT38/CLUAP1 contact surface; most pathogenic missense cluster; blades 3–6 hotspot
Domain 2: Central linker / hinge (aa 401–550)      — flexible region connecting propeller to
           C-terminal domain; missense here → partial IFT-B2 disruption; moderate phenotype
Domain 3: C-terminal extension (aa 551–801)        — IFT88 contact surface; IFT-B2 stabilisation;
           hypomorphic missense here → milder phenotype; IFT-B2 assembly partially maintained

Key pathogenic variant classes (IFT80):
1. WD40 blade 3–6 missense (aa 120–320): compound het or homozygous; moderate SRTD1; most common
2. Compound het: truncating + missense (any domain): moderate to severe SRTD1
3. Biallelic null (truncating + truncating): SRPS spectrum; perinatal lethal
4. Hypomorphic C-terminal missense (aa 551–801): milder SRTD1; late survival; renal secondary
5. Homozygous hypomorphic: mildest; adult survival with moderate restrictive lung disease
"""

import random
import math

SEED = 399
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
    ('Middle Eastern / North African', 0.30),   # consanguineous homozygous
    ('European',                        0.30),   # compound het most common
    ('South Asian',                     0.20),   # hypomorphic C-terminal common
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
    'Homozygous missense (WD40 blade 3–6)',
    'Compound het two missense',
    'Homozygous truncating (null / SRPS)',
    'Hypomorphic C-terminal missense',
]
allele_weights = [0.33, 0.28, 0.20, 0.12, 0.07]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.22, 0.48, 0.30])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial (rare)', 'None'],
        weights=[0.22, 0.24, 0.04, 0.50])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.10, 0.12, 0.78])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.40, 0.60])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.06, 0.06, 0.88])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.04, 0.05, 0.91])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.32, 0.10, 0.58])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.50, 0.20, 0.30])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.20, 0.28, 0.52])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.22, 0.20, 0.38, 0.20])[0]

    misdiagnosis = rng.choices(
        ['SRTD3/DYNC2H1 (most common SRTD — gene panel corrected)',
         'EVC (Ellis-van Creveld)',
         'Thanatophoric dysplasia (prenatal)',
         'BBS (Bardet-Biedl)',
         'SRTD5/WDR19 (IFT-A mistaken — gene panel corrected)',
         'None'],
        weights=[0.14, 0.07, 0.06, 0.04, 0.05, 0.64])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.36, 0.30, 0.22, 0.12])[0]

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
        "disease":    "IFT80 Short-Rib Thoracic Dysplasia 1 (SRTD1 / ATD1 / Jeune Syndrome)",
        "gene":       "IFT80 (*611229) — 3q25.33 — 801 aa — IFT-B2 WD40 β-Propeller (IFT-B anterograde)",
        "omim_gene":  "611229",
        "omim_disease": "208500",
        "chromosome": "3q25.33",
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
            "IFT80 loss disables the IFT-B2 subcomplex — the distal (tip-proximal) half of the "
            "anterograde IFT-B complex. Without IFT80, IFT-B2 cannot assemble around its "
            "IFT38/CLUAP1 and IFT88 contacts, truncating the IFT-B train. Kinesin-2 (KIF3A/B/KAP) "
            "drives an incomplete IFT-B train toward the ciliary tip → IFT-B cargo (including "
            "Hedgehog signalling components Gli2/Gli3, SMO, Patched) cannot be transported to the "
            "tip for processing → SHORTENED CILIA (EM; different from IFT-A short stubby cilia "
            "and dynein-2 club/bulging-tip cilia) → Hedgehog (Ihh/Shh) processing fails in "
            "chondrocytes → GLI3R accumulates → NARROW THORAX (primary), short ribs, short limbs. "
            "Secondary: renal TIN/ESRD (~15–22% survivors), rod-cone dystrophy (~8–12%), CHF (~5%)."
        ),
        "key_distinction": (
            "IFT80 (SRTD1) is the FOUNDING SRTD GENE — first identified by Beales et al. (2007, "
            "Nature Genetics), establishing the ciliopathy basis of Jeune syndrome. IFT80 encodes "
            "an IFT-B2 WD40 β-propeller subunit — the ONLY SRTD gene in the IFT-B2 subcomplex "
            "alongside SRTD10 (IFT172) and SRTD13 (CLUAP1/IFT38). Ciliary EM shows SHORTENED "
            "cilia — distinct from IFT-A short stubby (SRTD4/5/7/9) and dynein-2 club/bulging "
            "(SRTD3/8/11/15/17). NO NPHP or Joubert phenotype (unlike TTC21B/SRTD4). NO "
            "ectodermal features (unlike WDR19/SRTD5). RAREST of all current SRTD subsets: "
            "~20–30 families worldwide (2026); ~0.5–1% of SRTD."
        ),
        "ift_b2_subunit_table": [
            {"subunit": "IFT80",      "role": "WD40 β-propeller; IFT38/CLUAP1 + IFT88 contact; IFT-B2 assembly scaffold",      "srtd": "SRTD1",  "omim_gene": "611229", "chr": "3q25.33",  "freq": "~0.5–1% SRTD (founding gene)"},
            {"subunit": "IFT172",     "role": "Largest IFT-B2 subunit; WD40+TPR; kinesin-2 anchoring at ciliary tip",           "srtd": "SRTD10", "omim_gene": "607386", "chr": "2p23.3",   "freq": "~1–2% SRTD"},
            {"subunit": "CLUAP1/IFT38", "role": "IFT-B2 hub; directly contacts IFT80; IFT-B1/B2 linchpin",                     "srtd": "SRTD13", "omim_gene": "616470", "chr": "16q23.1",  "freq": "Very rare (<10 families)"},
            {"subunit": "IFT88",      "role": "IFT-B2 structural scaffold; contacts IFT80 C-terminal; Hedgehog signalling",     "srtd": "—",      "omim_gene": "600595", "chr": "13q12.11", "freq": "No SRTD reported (may be lethal)"},
            {"subunit": "IFT57",      "role": "IFT-B2 coiled-coil; contacts IFT-B1 via IFT52 bridge",                          "srtd": "—",      "omim_gene": "605502", "chr": "3p21.31",  "freq": "No SRTD reported"},
            {"subunit": "IFT25",      "role": "Small IFT-B2 subunit; Hedgehog signalling; binds IFT27 (Rab-like GTPase)",       "srtd": "—",      "omim_gene": "613286", "chr": "17q21.2",  "freq": "Retinal only (not SRTD)"},
            {"subunit": "IFT27",      "role": "Rab-like GTPase; IFT-B2 membrane-signalling; pairs with IFT25",                  "srtd": "—",      "omim_gene": "615504", "chr": "22q12.3",  "freq": "Bardet-Biedl-like (not SRTD)"},
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
            {"label": "Preaxial (rare)",        "n": _cnt('poly_status', 'Preaxial (rare)')},
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
            {"variant": "p.Arg498Cys (c.1492C>T) — WD40 blade 6 (IFT38/CLUAP1 contact surface) — European compound het — moderate SRTD1", "n": rng.randint(4, 6)},
            {"variant": "p.Thr376Met (c.1127C>T) — WD40 blade 4 (IFT88 C-terminal contact zone) — Middle Eastern homozygous — moderate-severe SRTD1", "n": rng.randint(3, 5)},
            {"variant": "p.Gln598Ter (c.1792C>T) — truncating null — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala614Val (c.1841C>T) — C-terminal extension (IFT88 stabilisation) — South Asian hypomorphic — mild SRTD1", "n": rng.randint(2, 3)},
            {"variant": "p.Lys256Glu (c.766A>G) — WD40 blade 3 (CLUAP1/IFT38 core interface) — MENA homozygous — moderate SRTD1", "n": rng.randint(1, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "IFT80 — Intraflagellar Transport Protein 80",
            "also_known_as":    "WDR56 (historical); IFT-B2 component; founding SRTD gene",
            "omim_gene":        "*611229",
            "chromosome":       "3q25.33",
            "protein_size":     "801 aa",
            "protein_class":    "IFT-B2 subcomplex WD40 β-propeller protein (anterograde IFT)",
            "domain_structure": "N-terminal WD40 β-propeller (aa 1–400; IFT38/CLUAP1 + IFT88 contacts; pathogenic hotspot blades 3–6) + central linker/hinge (aa 401–550; flexible; moderate missense phenotype) + C-terminal extension (aa 551–801; IFT88 stabilisation; hypomorphic domain)",
            "function":         "Core scaffold of IFT-B2 subcomplex (distal half of IFT-B anterograde complex); contacts IFT38/CLUAP1 and IFT88 within IFT-B2; loss truncates IFT-B train → anterograde IFT (kinesin-2 driven) fails → shortened cilia → Hedgehog processing failure",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT38/CLUAP1 (SRTD13; direct N-terminal WD40 contact); IFT88 (C-terminal extension contact; IFT-B2 structural anchor)",
            "expression":       "Primary non-motile cilia (9+0); chondrocytes (growth plate); renal tubular epithelium; retinal photoreceptor connecting cilium",
            "historic_note":    "FOUNDING SRTD GENE — first identified by Beales et al. (2007, Nature Genetics) in patients with Jeune ATD; established the IFT/ciliopathy genetic basis of SRTD. This discovery launched the molecular era of SRTD classification.",
            "ift_b_complex":    "IFT-B anterograde complex has two sub-complexes: IFT-B1 (core, kinesin-docking) and IFT-B2 (distal, tip-proximal, Hedgehog signal assembly). IFT80 is exclusively IFT-B2.",
        },
        "disease_card": {
            "omim_disease":     "#208500 — Asphyxiating Thoracic Dystrophy 1 (ATD1 / SRTD1 / Jeune syndrome type 1)",
            "full_name":        "Short-Rib Thoracic Dysplasia 1 With or Without Polydactyly",
            "alternative_names": "SRTD1; Asphyxiating Thoracic Dystrophy 1 (ATD1); Jeune syndrome type 1; Jeune ATD (historic, pre-molecular era name)",
            "primary_finding":  "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":      "~50% (postaxial most common; preaxial rare ~4%)",
            "renal":            "TIN + corticomedullary cysts → ESRD in ~15–22% survivors; NO NPHP phenotype (unlike TTC21B/SRTD4)",
            "retinal":          "Rod-cone dystrophy in ~8–12% (connecting cilium IFT-B2 failure; less frequent than IFT-A SRTDs)",
            "hepatic_chf":      "Ductal plate malformation → CHF in ~5% (less frequent than SRTD3)",
            "situs_inversus":   "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":      "ABSENT — no Molar Tooth Sign described for IFT80 (unlike TTC21B/SRTD4 which causes JBTS12)",
            "ectodermal":       "ABSENT — no sparse hair, hypodontia, or dystrophic nails (ectodermal is unique to WDR19/SRTD5 among SRTD genes)",
            "srps_spectrum":    "Biallelic null alleles → SRPS (perinatal lethal); less frequent than SRTD3",
            "renal_transplant": "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":       "~1/300,000–600,000; ~20–30 families worldwide (2026); ~0.5–1% of SRTD (rarest SRTD class)",
            "historic_significance": "FIRST SRTD GENE EVER IDENTIFIED — Beales et al. 2007 Nature Genetics. Established IFT/ciliogenesis pathway as the molecular cause of Jeune ATD, enabling gene panel diagnosis of the entire SRTD spectrum.",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones; SHORTENED cilia (EM) — distinct from IFT-A short stubby (no IFT-B import deficit at base) and dynein-2 club tip",
            "3. GENE PANEL (mandatory first-line): IFT80 MUST be included alongside DYNC2H1, WDR19, IFT140, WDR35, WDR60, WDR34, TTC21B, TCTEX1D2, DYNC2LI1, IFT172, CLUAP1 — SRTD1 clinically identical to all other SRTDs; gene panel is the ONLY way to distinguish",
            "4. WES + CNV if panel negative — IFT80 3q25.33; CNV deletion analysis mandatory for large gene deletions",
            "5. Ciliopathy EM (skin/fibroblast biopsy): SHORTENED cilia — IFT-B2 failure; shorter than normal but NOT club/bulging tip (dynein-2) and NOT short stubby (IFT-A). Subtle difference from IFT-A; gene panel always required",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually; NO NPHP variant expected for IFT80",
            "7. ERG annually from age 6 (rod-cone dystrophy ~8–12%); ophthalmology referral",
            "8. Liver USS + APRI if hepatic signs suspected (CHF ~5%); liver biopsy if APRI elevated",
            "9. Brain MRI: NO Molar Tooth Sign expected (unlike TTC21B/JBTS12); normal posterior fossa",
            "10. MLPA/CNV for IFT80 3q25.33 deletions if sequencing negative but phenotype compelling",
        ],
        "mechanism_glossary": [
            {"term": "IFT80",           "definition": "Intraflagellar Transport Protein 80 — 801 aa WD40 β-propeller; IFT-B2 subcomplex scaffold; contacts IFT38/CLUAP1 (N-terminal WD40 blade 3–6) and IFT88 (C-terminal extension). Loss truncates IFT-B2 and the entire IFT-B anterograde train."},
            {"term": "IFT-B2 subcomplex", "definition": "The distal (tip-proximal) half of the IFT-B anterograde complex: IFT80 + IFT88 + IFT57 + IFT38/CLUAP1 + IFT172 + IFT25 + IFT27. IFT-B2 sits at the 'anterograde tip' of the IFT train, positioning Hedgehog components for tip processing."},
            {"term": "IFT-B complex",   "definition": "The anterograde IFT complex, driven by kinesin-2 (KIF3A/KIF3B/KAP) from the ciliary base to the tip. Consists of IFT-B1 (core/proximal: IFT52, IFT46, IFT88, IFT70, IFT52, IFT22) and IFT-B2 (distal/tip-proximal). IFT-B transports axonemal building blocks and Hedgehog pathway components to the ciliary tip."},
            {"term": "Shortened cilia", "definition": "EM finding in IFT-B SRTDs (SRTD1/IFT80, SRTD10/IFT172, SRTD13/CLUAP1): cilia are shortened because IFT-B anterograde delivery to the tip is impaired — axonemal building blocks cannot be delivered. Distinct from IFT-A short stubby (IFT-B cannot enter at base) and dynein-2 club/bulging tip (retrograde failure → IFT-B stranded at tip)."},
            {"term": "Anterograde IFT", "definition": "Kinesin-2-driven transport from the ciliary base to the tip. Delivers: IFT-A complex, outer dynein arms, axonemal tubulin, Hedgehog signal components (SMO, Gli2/3, Patched). IFT-B2 carries the Hedgehog components at the distal tip-proximal region."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling pathway that requires ciliary tip delivery for processing of Gli2/Gli3 transcription factors. IFT80 loss → Gli components cannot reach the tip → GLI3R accumulates → Ihh/Shh response impaired in chondrocytes → short ribs, narrow thorax, short limbs."},
            {"term": "IFT38 / CLUAP1", "definition": "CLUAP1 (IFT38) — IFT-B2 hub protein; directly contacts IFT80 N-terminal WD40 blades 3–6; bridges IFT-B1 and IFT-B2. Loss causes SRTD13 (very rare). IFT38/CLUAP1 variants are the closest molecular neighbours of IFT80 within IFT-B2."},
            {"term": "Cell-autonomous",  "definition": "The IFT-B2 defect is intrinsic to the cell. After renal transplantation, the donor kidney (IFT80+) functions normally — CURATIVE for renal disease; no recurrence post-transplant."},
        ],
        "key_variants": [
            {"variant": "p.Arg498Cys",  "domain": "WD40 blade 6 (IFT38/CLUAP1 contact surface, aa 480–510)", "consequence": "Disrupts WD40 surface contacting IFT38/CLUAP1; IFT-B2 assembly impaired; European compound het; moderate SRTD1",     "ethnicity": "European"},
            {"variant": "p.Thr376Met",  "domain": "WD40 blade 4 (IFT88 C-terminal contact zone, aa 360–390)", "consequence": "IFT88-IFT80 interface disrupted; IFT-B2 partially destabilised; Middle Eastern homozygous; moderate-severe SRTD1", "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Gln598Ter",  "domain": "Truncating null (central linker/hinge region)",             "consequence": "Premature stop; biallelic null → SRPS spectrum; pan-ethnic",                                                       "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala614Val",  "domain": "C-terminal extension (IFT88 stabilisation region, aa 600–650)", "consequence": "Hypomorphic; South Asian; milder SRTD1; partial IFT-B2 assembly maintained; late survival",                   "ethnicity": "South Asian"},
            {"variant": "p.Lys256Glu",  "domain": "WD40 blade 3 (CLUAP1/IFT38 core interface, aa 240–270)",   "consequence": "CLUAP1 docking disrupted; MENA homozygous; moderate SRTD1; renal secondary",                                      "ethnicity": "MENA"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months; SRTD1 thorax mechanics same as other SRTDs",
            "2. Respiratory: neonatal mechanical ventilation if severe null alleles; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous). NOTE: NO NPHP allele variant in IFT80 — all SRTD1 renal disease is secondary to thoracic SRTD, not a separate NPHP phenotype",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF; hepatology referral; avoid hepatotoxic drugs",
            "6. Genetics: cascade testing of parents + siblings; no NPHP or Joubert allele-severity counselling needed (unlike TTC21B); standard AR 25% recurrence risk; prenatal/preimplantation genetics",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology; NO neurology unless an unrelated comorbidity",
            "8. HISTORIC COUNSELLING: Families with IFT80/SRTD1 can be informed they carry the variant in the founding Jeune syndrome gene — the 2007 discovery that launched the IFT/ciliopathy molecular era for this disease",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",    "key_difference": "Most common SRTD (~50%); DYNEIN-2 (not IFT-B2); CLUB cilia on EM (IFT-B stranded at tip) — distinct from SRTD1 shortened cilia; gene panel resolves"},
            {"disease": "SRTD5 (WDR19)",      "key_difference": "Most common IFT-A SRTD; IFT-A complex (not IFT-B2); short STUBBY cilia (IFT-B cannot enter at base); ectodermal features UNIQUE to WDR19 (absent in SRTD1); gene panel required"},
            {"disease": "SRTD9 (IFT140)",     "key_difference": "IFT-A core scaffold (not IFT-B2); short stubby cilia (IFT-A); no ectodermal; no NPHP; gene panel required"},
            {"disease": "SRTD4 (TTC21B)",     "key_difference": "IFT-A TPR adaptor (not IFT-B2); short stubby cilia; THREE phenotypes (SRTD4/NPHP12/JBTS12) — IFT80 causes SRTD1 only (no NPHP/Joubert variant)"},
            {"disease": "SRTD10 (IFT172)",    "key_difference": "SAME IFT-B2 COMPLEX — IFT172 is the LARGEST IFT-B2 subunit; shortened cilia (same EM class as IFT80); gene panel is the ONLY way to distinguish SRTD1 from SRTD10"},
            {"disease": "SRTD13 (CLUAP1)",    "key_difference": "SAME IFT-B2 COMPLEX — CLUAP1/IFT38 directly contacts IFT80; shortened cilia; very rare (<10 families); gene panel mandatory"},
            {"disease": "EVC (Ellis-van Creveld)", "key_difference": "CHD present in ~60% EVC (rare in SRTD1); EVC ectodermal features (teeth, nails, hair); EVC is a hedgehog-cilia pathway disorder but not IFT-B2 based"},
            {"disease": "Bardet-Biedl (BBS)",  "key_difference": "No narrow thorax in BBS; obesity + hypogonadism absent in SRTD1; BBSome is a separate cilia transport complex (not IFT-B2)"},
            {"disease": "Thanatophoric dysplasia", "key_difference": "Dominant FGFR3 gain-of-function (not AR IFT-B2 ciliopathy); thanatophoric has curved femurs, macrocephaly, cloverleaf skull — absent in SRTD1"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD1 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
