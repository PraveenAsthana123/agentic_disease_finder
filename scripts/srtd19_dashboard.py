"""
CEP120 Short-Rib Thoracic Dysplasia 19 (SRTD19) — Jeune Asphyxiating Thoracic Dystrophy Type 19
==================================================================================================
Primary Gene : CEP120 (*613446) — 5q23.2; ~1,007 aa; Centrosomal Protein 120 (CCDC100)
               CEP120 is a centriole ELONGATION factor that works downstream of CPAP/CENPJ
               to extend the central cartwheel of the procentriole. CEP120 directly binds
               CPAP and recruits CEP135 + SSNA1/NA14 to the nascent centriole wall.
               Loss of CEP120 → centriole elongation fails → SHORT centrioles → dysfunctional
               BASAL BODIES → ciliogenesis is impaired → SHORT CILIA (reduced length; basal
               body anchoring unstable) → Hedgehog signal failure in chondrocytes → NARROW
               THORAX, short ribs, shortened limbs (SRTD19 core phenotype).
               Distinct from IFT-A/B cargo transport defects (SRTD2–18): CEP120 is a
               CENTRIOLE ASSEMBLY factor, not a transport subunit. Cilia are SHORT but
               structurally present (weak basal body docking), unlike IFT-A short-stubby
               (retrograde failure) or dynein-2 club/bulging (tip accumulation).
               Severe biallelic LOF alleles → SRTD19/SRPS spectrum (perinatal lethal).
               Intermediate/hypomorphic alleles → moderate SRTD19 (liveborn, VEPTR).
               Very hypomorphic alleles → JBTS31 (Joubert Syndrome Type 31) — mild
               cerebellar vermis hypoplasia + MTS; basal body partially functional.
Disease OMIM : #617895 — Short-Rib Thoracic Dysplasia 19 With or Without Polydactyly (SRTD19)
               Also allelic: JBTS31 (Joubert Syndrome Type 31, OMIM #617562)
               Also called: Asphyxiating Thoracic Dystrophy 19 (ATD19)
Chromosome   : 5q23.2
Inheritance  : Autosomal Recessive — biallelic LOF (homozygous consanguineous or compound het)
Prevalence   : ~1/1,500,000–4,000,000; ~15–30 families reported worldwide (2026);
               Very rare: among the rarest SRTD subtypes; CENTRIOLE ASSEMBLY class

Protein Structure — CEP120 (1,007 aa; centriole elongation factor / centrosomal protein)
-----------------------------------------------------------------------------------------
Domain 1: N-terminal CPAP-binding domain (aa 1–200)   — direct CPAP/CENPJ interaction;
           procentriole wall seeding; Arg200 is the most common pathogenic residue;
           MENA/Arabian Peninsula founder Arg200Gln cluster (aa 180–220)
Domain 2: Central coiled-coil 1 (aa 201–550)           — CEP135 interaction; SSNA1/NA14
           recruitment; self-oligomerization for centriole ring assembly;
           South Asian cluster (Leu408Pro); pan-ethnic Arg329Cys
Domain 3: Central coiled-coil 2 (aa 551–750)           — TULP3 interaction; ciliary
           transport coordination; Gly605Glu (European)
Domain 4: C-terminal domain (aa 751–1,007)             — PCM anchoring; subdistal
           appendage positioning; basal body membrane anchorage;
           truncating alleles → null → SRPS spectrum

Key pathogenic variant classes (CEP120):
1. CPAP-binding missense (aa 180–220): homozygous; moderate-severe SRTD19; MENA founder
2. CC1 structural missense (aa 300–450): compound het; variable severity; South Asian / European
3. CC2 missense (aa 551–750): European predominant; moderate
4. Homozygous truncating (null): biallelic null; SRPS spectrum; consanguineous / pan-ethnic
5. Mild hypomorphic (any domain): JBTS31 spectrum — mild MTS + molar tooth; rare
"""

import random
import math

SEED = 415
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
    ('Middle Eastern / North African', 0.35),   # Arg200Gln founder; consanguinity predominant
    ('European',                        0.28),   # Gly605Glu CC2; compound het most common
    ('South Asian',                     0.20),   # Leu408Pro CC1 cluster; consanguineous
    ('East Asian',                      0.10),
    ('Latin American',                  0.05),
    ('Other / Unknown',                 0.02),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous missense (CPAP-binding domain)',
    'Compound het missense + splice/truncating',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense (CC1 + CC2)',
    'Mild hypomorphic (JBTS31 spectrum)',
]
allele_weights = [0.32, 0.28, 0.16, 0.16, 0.08]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null_a = 'null' in allele.lower() or 'SRPS' in allele
    jbts31 = 'JBTS31' in allele or 'hypomorphic' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.16 if null_a else 0.18, 0.50, 0.32 if not null_a else 0.34])[0]
    if null_a:
        thorax_sev = rng.choices(
            ['Severe (perinatal/neonatal respiratory failure)', 'Moderate'],
            weights=[0.75, 0.25])[0]

    polydactyly = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial', 'None'],
        weights=[0.30, 0.25, 0.04, 0.41])[0]

    renal = rng.choices(
        ['Renal cysts (corticomedullary)', 'NPHP-like / nephronophthisis', 'None'],
        weights=[0.15, 0.10, 0.75])[0]

    ckd_stage = 'None'
    if renal != 'None':
        ckd_stage = rng.choices(['CKD 3–4', 'CKD 5 / ESRD', 'None'],
                                 weights=[0.50, 0.30, 0.20])[0]

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Mild rod involvement (ERG borderline)', 'None'],
        weights=[0.10, 0.05, 0.85])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Mild fibrosis', 'None'],
        weights=[0.06, 0.02, 0.92])[0]

    jbts31_features = rng.choices(
        ['Cerebellar vermis hypoplasia (mild MTS)', 'None'],
        weights=[0.08 if jbts31 else 0.03, 1 - (0.08 if jbts31 else 0.03)])[0]

    veptr = rng.choices(
        ['VEPTR inserted', 'VEPTR planned', 'MAGEC growing rods', 'Conservative', 'SRPS (perinatal)'],
        weights=[0.35, 0.12, 0.10, 0.35, 0.08])[0]
    if null_a:
        veptr = 'SRPS (perinatal)'

    presentation = rng.choices(
        ['Prenatal USS + genetic diagnosis', 'Postnatal skeletal survey', 'Newborn respiratory'],
        weights=[0.45, 0.38, 0.17])[0]

    tx_renal = 'None'
    if ckd_stage == 'CKD 5 / ESRD':
        tx_renal = rng.choices(
            ['Renal transplant (curative)', 'Haemodialysis (awaiting Tx)', 'Peritoneal dialysis'],
            weights=[0.60, 0.25, 0.15])[0]

    resp_mgmt = rng.choices(
        ['VEPTR + no ventilation', 'CPAP / NIV', 'Mechanical ventilation (neonatal)', 'None (mild)'],
        weights=[0.38, 0.22, 0.12, 0.28])[0]
    if null_a:
        resp_mgmt = 'Mechanical ventilation (neonatal)'

    misdiagnosis = rng.choices(
        ['Initially SRTD3 (DYNC2H1)', 'Initially SRTD5 (WDR19)', 'Initially BBS', 'None'],
        weights=[0.18, 0.14, 0.06, 0.62])[0]

    sex  = rng.choice(['M', 'F'])
    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.55, 0.25, 0.14, 0.06])[0]

    patients.append({
        'id':           f'SRTD19-P{i+1:02d}',
        'sex':          sex,
        'age_dx':       age_dx,
        'ethnicity':    eth,
        'allele':       allele,
        'thorax_sev':   thorax_sev,
        'poly_status':  polydactyly,
        'renal':        renal,
        'ckd_stage':    ckd_stage,
        'retinal':      retinal,
        'hepatic':      hepatic,
        'jbts31_feats': jbts31_features,
        'veptr':        veptr,
        'presentation': presentation,
        'tx_renal':     tx_renal,
        'resp_mgmt':    resp_mgmt,
        'misdiagnosis': misdiagnosis,
    })

# ── aggregate counts ──────────────────────────────────────────────────────────
def _cnt(field, val):
    return sum(1 for p in patients if p[field] == val)

def _cnt_any(field, vals):
    return sum(1 for p in patients if p[field] in vals)

sex_M  = _cnt('sex', 'M')
sex_F  = _cnt('sex', 'F')

thorax_severe_n = _cnt('thorax_sev', 'Severe (perinatal/neonatal respiratory failure)')
polydactyly_n   = _cnt_any('poly_status', ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial'])
renal_any_n     = _cnt_any('renal', ['Renal cysts (corticomedullary)', 'NPHP-like / nephronophthisis'])
retinal_any_n   = _cnt_any('retinal', ['Rod-cone dystrophy (ERG confirmed)', 'Mild rod involvement (ERG borderline)'])
hepatic_chf_n   = _cnt('hepatic', 'CHF (ductal plate malformation)')
jbts31_n        = _cnt('jbts31_feats', 'Cerebellar vermis hypoplasia (mild MTS)')
veptr_any_n     = _cnt_any('veptr', ['VEPTR inserted', 'VEPTR planned', 'MAGEC growing rods'])
transplant_done_n = _cnt('tx_renal', 'Renal transplant (curative)')
misdiagnosis_n  = _cnt_any('misdiagnosis', ['Initially SRTD3 (DYNC2H1)', 'Initially SRTD5 (WDR19)', 'Initially BBS'])
srps_n          = _cnt('veptr', 'SRPS (perinatal)')

age_dx_0_1   = _cnt('age_dx', '0–1 yr (neonatal)')
age_dx_2_5   = _cnt('age_dx', '2–5 yr (infant)')
age_dx_6_10  = _cnt('age_dx', '6–10 yr (child)')
age_dx_11_16 = _cnt('age_dx', '11–16 yr (teen)')

# ── public API helpers ────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease":    "CEP120 Short-Rib Thoracic Dysplasia 19 (SRTD19 / ATD19)",
        "gene":       "CEP120 (*613446) — 5q23.2 — 1,007 aa — Centriole Elongation Factor (CCDC100)",
        "omim_gene":  "613446",
        "omim_disease": "617895",
        "also_allelic": "JBTS31 (Joubert Syndrome Type 31, OMIM #617562)",
        "chromosome": "5q23.2",
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
            "jbts31_n":           jbts31_n,
            "jbts31_pct":         _pct(jbts31_n),
            "veptr_any_n":        veptr_any_n,
            "veptr_any_pct":      _pct(veptr_any_n),
            "srps_n":             srps_n,
            "srps_pct":           _pct(srps_n),
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
            "CEP120 loss disrupts centriole ELONGATION — not IFT transport. CEP120 binds CPAP/CENPJ "
            "to extend the central cartwheel of the procentriole. Without CEP120, centrioles remain "
            "SHORT → dysfunctional BASAL BODIES → ciliogenesis impaired (cilia form but are short "
            "and poorly anchored) → Hedgehog (Ihh/Shh) signalling fails in chondrocytes → NARROW "
            "THORAX (primary), short ribs, rhizomelia. DISTINCTIVE: cilia are SHORT but NOT the "
            "short-stubby IFT-A class (no retrograde IFT cargo) nor club/bulging dynein-2 tip "
            "accumulation — instead a centriole-assembly phenotype with weakly anchored short cilia. "
            "Secondary: renal cysts (~25%), retinal dystrophy (~15%), hepatic fibrosis (~8%). "
            "Allelic with JBTS31 (hypomorphic) — mild cerebellar vermis hypoplasia + MTS (~8%)."
        ),
        "key_distinction": (
            "CEP120 (SRTD19) is the FIRST major centriole ELONGATION FACTOR SRTD gene — mechanistically "
            "distinct from all IFT-A (SRTD2–14, SRTD18) and dynein-2 (SRTD3) subtypes. "
            "CEP120 acts upstream of IFT: without a functional basal body, IFT loading itself is "
            "impaired. EM: cilia are short but NOT the IFT-A 'short-stubby' pattern — distinguishable "
            "by ultrastructure and by CPAP-interaction pathway functional assays. "
            "JBTS31 allelism: the SRTD19–JBTS31 spectrum is governed by allele class (null → SRPS; "
            "missense → SRTD19; mild hypomorphic → JBTS31). This positions CEP120 in the SRTD–JBTS "
            "overlap group (cf. KIAA0586/TALPID3 JBTS23/SRTD16). "
            "CEP120 must be included on any comprehensive skeletal ciliopathy gene panel."
        ),
        "centriole_elongation_table": [
            {"factor": "CPAP (CENPJ)",    "role": "Central cartwheel seed — recruits CEP120; MCPH6 when LOF", "disease": "MCPH6 (biallelic null)", "omim_gene": "609279", "chr": "13q12.12"},
            {"factor": "CEP120",          "role": "Centriole elongation — CPAP-bound; CEP135 recruiter",      "disease": "SRTD19 / JBTS31",        "omim_gene": "613446", "chr": "5q23.2"},
            {"factor": "CEP135",          "role": "Centriole wall scaffold — recruited by CEP120",             "disease": "MCPH8 (biallelic)",       "omim_gene": "611423", "chr": "4q12"},
            {"factor": "SSNA1/NA14",      "role": "Centriole wall stability — CEP120 interaction partner",     "disease": "No OMIM disease (2026)",  "omim_gene": "607283", "chr": "1p34.3"},
            {"factor": "SAS-6 (SASS6)",   "role": "Cartwheel spoke — the central hub; procentriole seeding",   "disease": "MCPH14 (biallelic)",      "omim_gene": "609321", "chr": "1p21.2"},
            {"factor": "STIL",            "role": "Centriole duplication — CPAP binding partner",               "disease": "MCPH7; HPE7",             "omim_gene": "181590", "chr": "1p33"},
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
            {"label": "CKD 3–4",              "n": _cnt('ckd_stage', 'CKD 3–4')},
            {"label": "CKD 5 / ESRD",         "n": _cnt('ckd_stage', 'CKD 5 / ESRD')},
            {"label": "None / not applicable", "n": _cnt('ckd_stage', 'None')},
        ],
        "retinal_distribution": _dist('retinal'),
        "hepatic_distribution": _dist('hepatic'),
        "jbts31_distribution": _dist('jbts31_feats'),
        "veptr_distribution": _dist('veptr'),
        "presentation_distribution": _dist('presentation'),
        "misdiagnosis_distribution": _dist('misdiagnosis'),
        "allele_class_summary": _dist('allele'),
        "ethnicity_distribution": _eth_dist(),
        "treatment_renal": _dist('tx_renal'),
        "respiratory_management": _dist('resp_mgmt'),
        "top_variants": [
            {"variant": "p.Arg200Gln (c.599G>A) — CPAP-binding domain (aa 180–220) — MENA/Arabian Peninsula founder — moderate-severe SRTD19", "n": rng.randint(4, 6)},
            {"variant": "p.Leu408Pro (c.1223T>C) — CC1 coiled-coil misfolding — South Asian homozygous — moderate-severe SRTD19",               "n": rng.randint(2, 4)},
            {"variant": "p.Gly605Glu (c.1814G>A) — CC1/CC2 junction — European compound het — moderate SRTD19",                                 "n": rng.randint(2, 3)},
            {"variant": "p.Arg329Cys (c.985C>T) — CC1 surface — pan-ethnic moderate SRTD19",                                                    "n": rng.randint(1, 3)},
            {"variant": "p.Glu788Ter (c.2362G>T) — C-terminal truncating null — SRPS spectrum — pan-ethnic",                                    "n": rng.randint(1, 2)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "CEP120 — Centrosomal Protein 120 (CCDC100)",
            "also_known_as":    "CCDC100; C5orf37 (partial); CEP120 centriole elongation factor",
            "omim_gene":        "*613446",
            "chromosome":       "5q23.2",
            "protein_size":     "1,007 aa",
            "protein_class":    "Centriole elongation factor — centrosomal / basal body component",
            "domain_structure": (
                "N-terminal CPAP-binding domain (aa 1–200) — direct CPAP/CENPJ interaction; "
                "procentriole elongation seeding; Arg200 pathogenic cluster (MENA founder); "
                "Central coiled-coil 1 (aa 201–550) — CEP135 and SSNA1/NA14 recruitment; "
                "self-oligomerization ring for centriole wall; South Asian Leu408Pro cluster; "
                "Central coiled-coil 2 (aa 551–750) — TULP3 interaction; ciliary transport linkage; "
                "C-terminal domain (aa 751–1,007) — PCM anchoring; subdistal appendage positioning; "
                "truncating alleles → null → SRPS spectrum"
            ),
            "function": (
                "CEP120 extends the central cartwheel of the procentriole by binding CPAP/CENPJ "
                "and recruiting CEP135 + SSNA1/NA14 to the nascent centriole wall. Loss → short "
                "centrioles → dysfunctional basal bodies → impaired ciliogenesis → short cilia "
                "(basal body anchoring weak; IFT partially preserved but impaired). "
                "Hedgehog signalling fails → SRTD19 skeletal phenotype."
            ),
            "inheritance":  "Autosomal Recessive — biallelic LOF",
            "key_contacts": "CPAP/CENPJ (procentriole elongation), CEP135 (scaffold), SSNA1/NA14 (stability), TULP3 (ciliary transport)",
            "expression":   "Centrosome (centriolar wall); basal body; chondrocytes; renal tubular epithelium; retinal photoreceptor connecting cilium",
        },
        "disease_card": {
            "omim_disease":       "#617895",
            "full_name":          "Short-Rib Thoracic Dysplasia 19 With or Without Polydactyly",
            "alternative_names":  "SRTD19; Asphyxiating Thoracic Dystrophy 19 (ATD19); Jeune syndrome type 19",
            "also_allelic":       "JBTS31 (Joubert Syndrome Type 31, #617562) — same gene CEP120; allele-class governs phenotype",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal/perinatal respiratory failure in severe (null) alleles",
            "polydactyly":        "~55% (postaxial predominant; bilateral in ~30%, unilateral ~25%, preaxial <5%)",
            "renal":              "Renal cysts (corticomedullary) + NPHP-like in ~25%; ESRD in subset (~8%)",
            "retinal":            "Rod-cone dystrophy in ~15% (basal body ciliogenesis failure at connecting cilium)",
            "hepatic_chf":        "CHF / ductal plate malformation in ~8%",
            "jbts31_overlap":     "Mild cerebellar vermis hypoplasia (MTS) in ~8% — JBTS31 spectrum; hypomorphic alleles",
            "situs_inversus":     "ABSENT — primary non-motile cilia (9+0); no nodal motile cilia defect",
            "ced_sensenbrenner":  "ABSENT — unlike IFT43 (SRTD18), CEP120 LOF does not cause cranioectodermal features",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); less common than SRTD3 (DYNC2H1)",
            "renal_transplant":   "CURATIVE — cell-autonomous centriole/basal body defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":         "~1/1,500,000–4,000,000; ~15–30 families worldwide (2026); very rare",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polydactyly, polyhydramnios → suspect SRTD19; no campomelia (unlike SRTD18/IFT43)",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones; less campomelia than SRTD18 (IFT43)",
            "3. GENE PANEL (mandatory first-line): include CEP120 — clinical overlap with SRTD3/5/7/9/18 is extensive; EM cannot distinguish CEP120 from IFT-A class",
            "4. WES + CNV if panel negative — CEP120 intragenic deletions possible; MLPA for CNV",
            "5. Centriole EM (fibroblast/skin): SHORT CILIA with weak basal body anchoring — distinct from IFT-A short-stubby (retrograde IFT) and dynein-2 club/bulging; confirms centriole-assembly phenotype",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis",
            "7. ERG annually from age 6 (rod-cone dystrophy ~15%); ophthalmology referral",
            "8. Brain MRI: vermis + brainstem if JBTS31 spectrum suspected (hypomorphic alleles — mild MTS)",
            "9. Liver USS + APRI if hepatic signs (CHF ~8%)",
            "10. Echocardiography if CHD suspected",
        ],
        "mechanism_glossary": [
            {"term": "CEP120",               "definition": "Centrosomal Protein 120 — 1,007 aa centriole elongation factor; binds CPAP/CENPJ and recruits CEP135 + SSNA1/NA14 to extend the procentriole central cartwheel."},
            {"term": "Centriole elongation", "definition": "The process by which the procentriole grows from ~100 nm (SAS-6 cartwheel stub) to full ~450 nm length, dependent on CPAP–CEP120–CEP135 axis. CEP120 LOF → short centrioles."},
            {"term": "Basal body",           "definition": "The mother centriole that anchors to the plasma membrane and templates the ciliary axoneme. Short/dysfunctional centrioles → impaired basal body docking → short or absent cilia."},
            {"term": "Short cilia",          "definition": "CEP120-class EM finding: cilia are shortened but structurally present (weak basal body anchoring). Distinct from IFT-A short-stubby (cargo accumulation) and dynein-2 club/bulging (tip accumulation)."},
            {"term": "CPAP / CENPJ",         "definition": "Centrosomal P4.1-associated protein — seeds the procentriole cartwheel; recruits CEP120 for elongation. Biallelic LOF causes MCPH6 (microcephaly), not SRTD."},
            {"term": "JBTS31 allelism",      "definition": "Hypomorphic CEP120 alleles cause Joubert Syndrome Type 31 (mild MTS + cerebellar vermis hypoplasia) instead of SRTD19. Allele class governs spectrum: null → SRPS; missense → SRTD19; mild hypomorphic → JBTS31."},
            {"term": "Hedgehog (Ihh/Shh)",   "definition": "Growth plate signalling pathway processed at the ciliary tip. CEP120 LOF → dysfunctional cilia → GLI processing failure → SRTD skeletal phenotype (narrow thorax, short ribs, polydactyly)."},
            {"term": "Cell-autonomous",       "definition": "Centriole/basal body defect intrinsic to each cell. Renal transplantation with donor kidney (CEP120+) is CURATIVE — no post-transplant recurrence."},
        ],
        "key_variants": [
            {"variant": "p.Arg200Gln (c.599G>A)",  "domain": "CPAP-binding domain (aa 180–220)",       "consequence": "Disrupts CPAP–CEP120 interface; most common SRTD19 allele; MENA/Arabian Peninsula founder",    "ethnicity": "MENA / Arabian Peninsula (homozygous)"},
            {"variant": "p.Leu408Pro (c.1223T>C)",  "domain": "CC1 coiled-coil (structural misfolding)", "consequence": "Destabilises CC1 helical bundle; CEP135 recruitment impaired; South Asian homozygous",          "ethnicity": "South Asian (homozygous)"},
            {"variant": "p.Gly605Glu (c.1814G>A)",  "domain": "CC1/CC2 junction",                        "consequence": "Disrupts CC1–CC2 linker; TULP3 interaction partially lost; European compound het",             "ethnicity": "European"},
            {"variant": "p.Arg329Cys (c.985C>T)",   "domain": "CC1 surface (SSNA1/NA14 contact)",        "consequence": "Reduces SSNA1 recruitment; moderate SRTD19; pan-ethnic",                                       "ethnicity": "Pan-ethnic"},
            {"variant": "p.Glu788Ter (c.2362G>T)",  "domain": "C-terminal domain (truncating null)",     "consequence": "Loss of PCM anchoring domain; null allele → SRPS spectrum in biallelic null; pan-ethnic",      "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala501Val (c.1502C>T)",  "domain": "CC1 (mild hypomorphic)",                  "consequence": "Partial residual CEP120 → JBTS31 spectrum (mild MTS) when compound with hypomorph",            "ethnicity": "European / pan-ethnic"},
            {"variant": "c.1344+1G>A (splice)",      "domain": "CC1/CC2 splice junction",                 "consequence": "Exon skip → partial protein; moderate-severe SRTD19; European predominant",                    "ethnicity": "European"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months",
            "2. Respiratory: neonatal mechanical ventilation if severe thorax; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — cell-autonomous (no recurrence)",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS if CHF/fibrosis suspected; hepatology referral; avoid hepatotoxic drugs",
            "6. JBTS31 spectrum (hypomorphic alleles): brain MRI for vermis/MTS; neurodevelopmental surveillance; ataxia physiotherapy; oculomotor apraxia support",
            "7. Genetics: cascade testing; consanguinity counselling; prenatal/preimplantation genetics (25% recurrence); SRTD19 vs. JBTS31 allele-class counselling",
            "8. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, neurology (JBTS31 subset)",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",     "key_difference": "Dynein-2 motor (~50% all SRTD); club/bulging cilia EM (dynein-2 class) vs. CEP120 short-anchoring; most common SRTD; gene panel resolves"},
            {"disease": "SRTD5 (WDR19/IFT144)","key_difference": "IFT-A ARM hub; short-stubby cilia (IFT-A class) vs. CEP120 short-anchoring; no JBTS31 allelism; gene panel resolves"},
            {"disease": "SRTD16 (KIAA0586)",   "key_difference": "CPLANE1 — also allelic with JBTS17; same SRTD–JBTS dual-spectrum concept as CEP120; gene panel differentiates"},
            {"disease": "SRTD18 (IFT43)",      "key_difference": "IFT-A satellite; campomelia ~50% (distinctive) vs. CEP120 no campomelia; CED3 dual phenotype vs. JBTS31 dual; gene panel required"},
            {"disease": "JBTS23 (KIAA0586)",   "key_difference": "Same gene as SRTD16; severe alleles → SRTD; hypomorphic → JBTS23 (cf. CEP120: severe → SRTD19, hypomorphic → JBTS31)"},
            {"disease": "BBS (BBSome genes)",   "key_difference": "BBSome ciliary cargo transport; obesity + RP + polydactyly; no narrow thorax/short ribs; no skeletal dysplasia"},
        ],
    }
