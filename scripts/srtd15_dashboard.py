"""
DYNC2LI1 Short-Rib Thoracic Dysplasia 15 (SRTD15) — Jeune Asphyxiating Thoracic Dystrophy Type 15
=====================================================================================================
Primary Gene : DYNC2LI1 (*617248) — 2p21; ~492 aa; Dynein Cytoplasmic 2 Light Intermediate Chain 1
               DYNC2LI1 is the scaffold light intermediate chain (also known as D2LIC / LIC3)
               of the cytoplasmic dynein-2 complex, the RETROGRADE intraflagellar transport (IFT)
               motor. DYNC2LI1 directly contacts WDR34 (SRTD11) via its C-terminal domain, bridges
               the DYNC2H1 heavy chain AAA+ motor to the WDR34/WDR60 intermediate chain adapters,
               and is essential for stable dynein-2 complex assembly.
               Loss of DYNC2LI1 → WDR34 destabilisation → retrograde IFT failure →
               Hedgehog signal failure in chondrocytes → narrow thorax, short ribs, short limbs.
               Dynein-2 subunit hierarchy: DYNC2H1 (SRTD3) + DYNC2LI1 (SRTD15) +
               WDR34 (SRTD11) + WDR60 (SRTD8) + TCTEX1D2 (SRTD17) + DYNLRB1/DYNLRB2 (light).
Disease OMIM : #617127 — Short-Rib Thoracic Dysplasia 15 With or Without Polydactyly (SRTD15)
               Also called: Asphyxiating Thoracic Dystrophy 15 (ATD15)
               Severe biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum
Chromosome   : 2p21
Inheritance  : Autosomal Recessive — biallelic LOF (homozygous missense or compound het most common)
Prevalence   : ~1/500,000–2,000,000; ~10–20 families reported worldwide (2026);
               Very rare: ~1–2% of all molecularly confirmed SRTD;
               Fourth most common dynein-2-subunit SRTD gene after DYNC2H1 (SRTD3, ~50%),
               WDR60 (SRTD8, ~5–10%), and WDR34 (SRTD11, ~3–5%)

Protein Structure — DYNC2LI1 (492 aa; dynein-2 scaffold light intermediate chain)
-----------------------------------------------------------------------------------
Domain 1: N-terminal α/β fold (aa 1–120)    — DYNC2H1 stem contact; initiates dynein-2 tail assembly
Domain 2: Central RAS-like GTPase fold (aa 121–360) — structural scaffold; no GTPase activity;
           dominant negative missense cluster in aa 200–300
Domain 3: WDR34 docking helix (aa 361–420)  — direct contact with WDR34 C-terminal interface helix;
           loss here decouples WDR34 from the dynein-2 tail → SRTD11 variant phenocopies SRTD15
Domain 4: C-terminal regulatory tail (aa 421–492) — IFT particle handoff; phosphorylation-regulated

Key pathogenic variant classes (DYNC2LI1):
1. RAS-fold missense (aa 200–300): compound heterozygous; moderate SRTD15 phenotype; most common
2. WDR34-docking truncating: C-terminal truncations; severe; decouples WDR34 → SRPS spectrum
3. N-stem null alleles: homozygous consanguineous; severe; entire dynein-2 tail lost
4. Hypomorphic missense (aa 350–420): mild; surviving adults; renal dominant presentation
"""

import random
import math

SEED = 387
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
    ('Middle Eastern / North African', 0.32),   # highest consanguinity; homozygous prevalent
    ('European',                        0.28),   # compound het most common
    ('South Asian',                     0.18),
    ('East Asian',                      0.10),
    ('Latin American',                  0.08),
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
    'Homozygous missense (RAS-fold)',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense',
    'Homozygous hypomorphic (mild)',
]
allele_weights = [0.33, 0.28, 0.16, 0.15, 0.08]

for i in range(N):
    eth   = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null  = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.22, 0.52, 0.26])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'None'],
        weights=[0.20, 0.16, 0.64])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.14, 0.14, 0.72])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.40, 0.60])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.08, 0.05, 0.87])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.07, 0.05, 0.88])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.30, 0.12, 0.58])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.55, 0.20, 0.25])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.18, 0.30, 0.52])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.22, 0.24, 0.38, 0.16])[0]

    misdiagnosis = rng.choices(
        ['EVC (Ellis-van Creveld)', 'Thanatophoric dysplasia (prenatal)',
         'BBS (Bardet-Biedl)', 'SRTD3 / gene panel corrected', 'None'],
        weights=[0.10, 0.08, 0.05, 0.15, 0.62])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.40, 0.30, 0.20, 0.10])[0]

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
        "disease":    "DYNC2LI1 Short-Rib Thoracic Dysplasia 15 (SRTD15 / ATD15)",
        "gene":       "DYNC2LI1 (*617248) — 2p21 — 492 aa — Dynein-2 Scaffold Light Intermediate Chain",
        "omim_gene":  "617248",
        "omim_disease": "617127",
        "chromosome": "2p21",
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
            "DYNC2LI1 loss removes the scaffold light intermediate chain that bridges the DYNC2H1 "
            "AAA+ motor (SRTD3) to the WDR34/WDR60 intermediate chain adapters. Without DYNC2LI1, "
            "WDR34 (SRTD11) detaches from the dynein-2 tail → entire complex destabilises → "
            "retrograde IFT from ciliary tip to cell body stalls → IFT-B pile-up at cilia tip "
            "(bulging/club morphology on EM) → Hedgehog (Ihh/Shh) signalling fails in chondrocytes "
            "→ NARROW THORAX (primary), short ribs, short limbs. Secondary: renal TIN/ESRD "
            "(~25–30% survivors), rod-cone dystrophy (~10–15%), CHF (~8%)."
        ),
        "key_distinction": (
            "DYNC2LI1 (SRTD15) is the ONLY scaffold subunit in the dynein-2 complex — it does not "
            "form a β-propeller (unlike WDR34/WDR60) but instead acts as a structural bridge. "
            "Its RAS-fold GTPase domain has NO GTPase activity (structural only). Biallelic null "
            "→ SRPS spectrum. Very rare: ~10–20 families worldwide (2026). Gene panel sequencing "
            "is mandatory — clinical overlap with SRTD3/8/11 is complete; gene identity is the "
            "only reliable differentiator."
        ),
        "dynein2_subunit_table": [
            {"subunit": "DYNC2H1",   "role": "AAA+ motor heavy chain (retrograde force)",                        "srtd": "SRTD3",  "omim_gene": "603297", "chr": "11q22.3", "freq": "~50% of SRTD"},
            {"subunit": "DYNC2LI1",  "role": "Scaffold light intermediate chain; bridges DYNC2H1 to WDR34",     "srtd": "SRTD15", "omim_gene": "617248", "chr": "2p21",    "freq": "~1–2% of SRTD"},
            {"subunit": "WDR34",     "role": "WD40 6-blade β-propeller intermediate chain (opposite to WDR60)", "srtd": "SRTD11", "omim_gene": "604126", "chr": "9q34.11", "freq": "~3–5% of SRTD"},
            {"subunit": "WDR60",     "role": "WD40 7-blade β-propeller intermediate chain (opposite to WDR34)", "srtd": "SRTD8",  "omim_gene": "615462", "chr": "7q36.3",  "freq": "~5–10% of SRTD"},
            {"subunit": "TCTEX1D2", "role": "Light chain (contacts WDR60 β-propeller)",                         "srtd": "SRTD17", "omim_gene": "617819", "chr": "3q21.3",  "freq": "very rare"},
            {"subunit": "DYNLRB1/2","role": "Roadblock-type light chains",                                       "srtd": "—",      "omim_gene": "607167", "chr": "20q13.32","freq": "not SRTD-associated"},
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
            {"label": "Postaxial (bilateral)", "n": _cnt('poly_status', 'Postaxial (bilateral)')},
            {"label": "Postaxial (unilateral)", "n": _cnt('poly_status', 'Postaxial (unilateral)')},
            {"label": "None", "n": _cnt('poly_status', 'None')},
        ],
        "renal_distribution": _dist('renal'),
        "ckd_stage_distribution": [
            {"label": "CKD 4", "n": _cnt('ckd_stage', 'CKD 4')},
            {"label": "CKD 5 / ESRD", "n": _cnt('ckd_stage', 'CKD 5 / ESRD')},
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
            {"variant": "p.Arg275Gln (c.824G>A) — RAS-fold core — Middle Eastern homozygous — moderate SRTD15", "n": rng.randint(3, 5)},
            {"variant": "p.Leu154Pro (c.461T>C) — N/RAS junction — consanguineous — severe SRTD15", "n": rng.randint(2, 4)},
            {"variant": "p.Glu174Ter (c.520G>T) — truncating, null — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 3)},
            {"variant": "p.Arg322Cys (c.964C>T) — WDR34-docking helix — European compound het — moderate", "n": rng.randint(2, 3)},
            {"variant": "p.Ala307Val (c.920C>T) — RAS-fold edge — South Asian — hypomorphic mild adult", "n": rng.randint(1, 2)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":              "DYNC2LI1 — Dynein Cytoplasmic 2 Light Intermediate Chain 1",
            "also_known_as":     "D2LIC, LIC3",
            "omim_gene":         "*617248",
            "chromosome":        "2p21",
            "protein_size":      "492 aa",
            "protein_class":     "Scaffold light intermediate chain — dynein-2 complex",
            "domain_structure":  "N-terminal α/β fold (DYNC2H1 contact) + central RAS-like GTPase fold (structural only, no GTPase activity) + WDR34-docking helix + C-terminal regulatory tail",
            "function":          "Bridges DYNC2H1 AAA+ motor to WDR34/WDR60 intermediate chain adapters; essential for dynein-2 complex assembly and stability",
            "inheritance":       "Autosomal Recessive — biallelic LOF",
            "key_contacts":      "DYNC2H1 (N-terminal), WDR34 SRTD11 (C-terminal docking helix)",
            "expression":        "Cilia (primary non-motile 9+0); chondrocytes; renal tubular epithelium; photoreceptor connecting cilium",
        },
        "disease_card": {
            "omim_disease":        "#617127",
            "full_name":           "Short-Rib Thoracic Dysplasia 15 With or Without Polydactyly",
            "alternative_names":   "SRTD15; Asphyxiating Thoracic Dystrophy 15 (ATD15); Jeune syndrome type 15",
            "primary_finding":     "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe alleles",
            "polydactyly":         "~35% (postaxial most common; bilateral > unilateral)",
            "renal":               "TIN + corticomedullary cysts → ESRD in ~25–30% of survivors",
            "retinal":             "Rod-cone dystrophy in ~10–15% (connecting cilium IFT failure)",
            "hepatic_chf":         "Ductal plate malformation → CHF in ~8%",
            "situs_inversus":      "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":         "ABSENT — no brainstem/vermis midline defect",
            "srps_spectrum":       "Biallelic null alleles → SRPS (perinatal lethal)",
            "renal_transplant":    "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment":  "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":          "~1/500,000–2,000,000; ~10–20 families worldwide (2026); ~1–2% of all SRTD",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones",
            "3. GENE PANEL (mandatory first-line): DYNC2H1, DYNC2LI1, WDR34, WDR60, TCTEX1D2 — clinical overlap is complete",
            "4. WES + CNV if panel negative — DYNC2LI1 deletions reported",
            "5. Skeletal ciliopathy EM (if skin/fibroblast biopsy): bulging/club cilia tip — same as SRTD3/8/11",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "7. ERG annually from age 6 (rod-cone dystrophy ~10–15%); ophthalmology referral",
            "8. Liver USS + APRI if hepatic signs (CHF ~8%)",
            "9. Echocardiography if CHD suspected (less common than EVC)",
            "10. MLPA for DYNC2LI1 deletions if sequencing negative but phenotype compelling",
        ],
        "mechanism_glossary": [
            {"term": "DYNC2LI1",    "definition": "Dynein Cytoplasmic 2 Light Intermediate Chain 1 — scaffold bridge of the dynein-2 retrograde IFT motor; unique in that its RAS-fold domain has NO GTPase activity (structural role only)."},
            {"term": "Retrograde IFT", "definition": "Intraflagellar transport from ciliary tip back to cell body, powered by dynein-2. Required for recycling IFT trains and turning off Hedgehog signalling at the tip."},
            {"term": "RAS-like GTPase fold", "definition": "Structural domain conserved in DYNC2LI1 from the Ras superfamily, but catalytically dead — functions as a rigid scaffold for dynein-2 assembly, not a signalling GTPase."},
            {"term": "Club/bulging cilia", "definition": "EM finding: IFT-B accumulates at the ciliary tip when retrograde IFT fails → tip swells into a club shape. Identical finding in SRTD3, SRTD8, SRTD11, SRTD15 — cannot distinguish by EM alone."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling pathway processed at the ciliary tip. Failure → GLI3R not processed → short ribs, narrow thorax, short limbs (SRTD phenotype)."},
            {"term": "SRPS spectrum",      "definition": "Short-Rib Polydactyly Syndrome — perinatal-lethal SRTD alleles (biallelic null). Occurs across SRTD3/8/11/15/17; frequency and severity differ by subunit."},
            {"term": "WDR34 (SRTD11)",    "definition": "Direct contact partner of DYNC2LI1 C-terminal docking helix. Loss of DYNC2LI1 secondarily destabilises WDR34 — SRTD15 and SRTD11 have overlapping severity."},
            {"term": "Cell-autonomous",    "definition": "The IFT defect is intrinsic to the transplanted organ's cells. After renal transplantation, the new kidney (donor's DYNC2LI1+) functions normally — CURATIVE for renal disease, no recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Arg275Gln",  "domain": "RAS-fold core (aa 200–300)", "consequence": "Destabilises central scaffold; compound het or homozygous; moderate SRTD15",      "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Leu154Pro",  "domain": "N/RAS junction (aa 121–160)", "consequence": "Disrupts N-terminal α/β fold; severe; consanguineous families",                  "ethnicity": "South Asian / MENA"},
            {"variant": "p.Glu174Ter",  "domain": "RAS-fold (truncating)",       "consequence": "Null allele; SRPS spectrum in biallelic null; pan-ethnic",                         "ethnicity": "Pan-ethnic"},
            {"variant": "p.Arg322Cys",  "domain": "WDR34-docking helix (aa 361–420)", "consequence": "Decouples WDR34/SRTD11 contact; European compound het with truncating; moderate SRTD15", "ethnicity": "European"},
            {"variant": "p.Ala307Val",  "domain": "RAS-fold edge (aa 300–360)",  "consequence": "Hypomorphic; mild; surviving adults; renal-dominant late presentation",            "ethnicity": "South Asian"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months",
            "2. Respiratory: neonatal mechanical ventilation if severe; CPAP/NIV for moderate thorax; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous)",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF; hepatology referral; avoid hepatotoxic drugs",
            "6. Genetics: cascade testing of parents + siblings; carrier frequency ~1/1,000 in consanguineous populations; prenatal/preimplantation genetics for recurrence risk (25%)",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",    "key_difference": "Most common SRTD (~50%); same phenotype; gene panel differentiates; DYNC2H1 is the motor, DYNC2LI1 is the scaffold"},
            {"disease": "SRTD8 (WDR60)",      "key_difference": "WDR60 is the 7-blade β-propeller opposite WDR34; phenotypically identical; gene panel mandatory"},
            {"disease": "SRTD11 (WDR34)",     "key_difference": "WDR34 directly contacts DYNC2LI1; loss of DYNC2LI1 secondarily destabilises WDR34 — can mimic SRTD11; gene panel resolves"},
            {"disease": "Ellis-van Creveld (EVC)", "key_difference": "CHD present in ~60% EVC (rare in SRTD15); EVC on 4p16 (not 2p21); ectodermal features (teeth, nails) in EVC"},
            {"disease": "BBS (Bardet-Biedl)", "key_difference": "No narrow thorax in BBS; obesity + hypogonadism present; BBS-specific gene panel"},
            {"disease": "Thanatophoric dysplasia", "key_difference": "Dominant FGFR3 mutation; cloverleaf skull; telephone-receiver femora; not ciliary"},
            {"disease": "NPHP (nephronophthisis)", "key_difference": "NPHP → renal is PRIMARY (not narrow thorax); no short ribs; different ciliary compartment (TZ, not IFT)"},
            {"disease": "Joubert syndrome (JBTS)", "key_difference": "Molar tooth sign (MTS) on MRI — ABSENT in SRTD15; cerebellar vermis hypoplasia in JBTS, not SRTD15"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD15 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
