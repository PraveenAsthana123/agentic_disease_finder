"""
TCTEX1D2 Short-Rib Thoracic Dysplasia 20 (SRTD20) — Jeune Asphyxiating Thoracic Dystrophy Type 20
====================================================================================================
Primary Gene : TCTEX1D2 (*617022) — 3q21.3; ~276 aa; Tctex1 Domain-Containing Protein 2 (DYNLT2B)
               TCTEX1D2 is the dynein-2-specific Tctex-type light chain of the cytoplasmic
               dynein-2 motor complex. It bridges the dynein-2 intermediate chains (WDR34/SRTD11,
               WDR60/SRTD8) and the heavy chain (DYNC2H1/SRTD3), stabilising the complete
               retrograde IFT motor complex at the ciliary base.
               Loss of TCTEX1D2 → dynein-2 motor complex destabilised → retrograde IFT
               (tip-to-base transport) fails → IFT-B cargo (IFT25, IFT27, IFT81) accumulates
               at the CILIARY TIP → CLUB/BULGING cilia on EM (dynein-2 class — same pattern as
               SRTD3/DYNC2H1, SRTD8/WDR60, SRTD11/WDR34, SRTD15/DYNC2LI1).
               GLI processing at the ciliary tip fails → Hedgehog (Ihh/Shh) signal collapse
               in growth plate chondrocytes → NARROW THORAX, short ribs, shortened limbs
               (primary skeletal SRTD20 phenotype).
               Null alleles → SRPS spectrum (perinatal lethal; most severe dynein-2 outcome).
               TCTEX1D2 is the FIFTH canonical dynein-2 complex subunit gene causing SRTD:
               joining DYNC2H1 (heavy chain; SRTD3), WDR34 (LIC2; SRTD11),
               WDR60 (IC1; SRTD8), and DYNC2LI1 (LIC1; SRTD15).
Disease OMIM : #617925 — Short-Rib Thoracic Dysplasia 20 With or Without Polydactyly (SRTD20)
               Also called: Asphyxiating Thoracic Dystrophy 20 (ATD20); Jeune syndrome type 20
Chromosome   : 3q21.3
Inheritance  : Autosomal Recessive — biallelic LOF (homozygous consanguineous or compound het)
Prevalence   : ~1/2,000,000–6,000,000; ~10–20 families reported worldwide (2026);
               Among the rarest SRTD subtypes; pure DYNEIN-2 LIGHT CHAIN class

Protein Structure — TCTEX1D2 (276 aa; dynein-2 Tctex-type light chain / DYNLT2B)
-----------------------------------------------------------------------------------
Domain 1: Tctex1 N-lobe (aa 1–90)     — forms the β-sheet sandwich core common to all Tctex
           light chains; DYNC2H1 heavy-chain tether interface; critical for motor assembly;
           MENA/Arabian Peninsula founder Arg78Gln (aa 70–95) — ablates heavy-chain contact
Domain 2: Tctex1 C-lobe (aa 91–200)   — WDR34/WDR60 intermediate-chain docking surface;
           dimerisation face; South Asian Leu142Pro (AA 135–155) cluster; pan-ethnic Gly118Arg
Domain 3: C-terminal linker (aa 201–276) — IFT-B peripheral contact; cargo-adaptor coordination;
           European Ala221Val + truncating Glu258Ter (null → SRPS spectrum)

Key pathogenic variant classes (TCTEX1D2):
1. Tctex1 N-lobe missense (aa 70–95): DYNC2H1 contact disrupted; MENA/ArabianPeninsula founder
2. Tctex1 C-lobe missense (aa 110–160): WDR34/WDR60 docking impaired; S.Asian/pan-ethnic
3. Homozygous truncating (null alleles): dynein-2 assembly collapse; SRPS spectrum; pan-ethnic
4. Compound het missense + truncating: most common European genotype; moderate-severe SRTD20
5. C-terminal linker missense (aa 200–276): IFT-B contact partial; European; moderate SRTD20
"""

import random
import math

SEED = 491
N    = 40   # 40-patient educational cohort

rng  = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _pct(n, total=N):
    return round(n / total * 100)

def _split(total, *fractions):
    buckets = [round(total * f) for f in fractions]
    diff = total - sum(buckets)
    buckets[0] += diff
    return buckets

# ── patient-level data (fixed seed) ──────────────────────────────────────────
patients = []
ethnicities = [
    ('Middle Eastern / North African', 0.33),   # Arg78Gln founder; consanguinity predominant
    ('European',                        0.30),   # Ala221Val C-lobe; compound het most common
    ('South Asian',                     0.22),   # Leu142Pro C-lobe cluster; consanguineous
    ('East Asian',                      0.09),
    ('Latin American',                  0.04),
    ('Other / Unknown',                 0.02),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous N-lobe missense (DYNC2H1-contact disrupted)',
    'Compound het missense + splice/truncating',
    'Homozygous truncating (null / SRPS)',
    'Compound het two missense (N-lobe + C-lobe)',
    'C-terminal linker missense (moderate)',
]
allele_weights = [0.30, 0.30, 0.15, 0.15, 0.10]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null_a = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.20 if null_a else 0.18, 0.48, 0.32 if not null_a else 0.10])[0]
    if null_a:
        thorax_sev = rng.choices(
            ['Severe (perinatal/neonatal respiratory failure)', 'Moderate'],
            weights=[0.80, 0.20])[0]

    polydactyly = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial', 'None'],
        weights=[0.22, 0.20, 0.04, 0.54])[0]

    renal = rng.choices(
        ['Renal cysts (corticomedullary)', 'NPHP-like / nephronophthisis', 'None'],
        weights=[0.18, 0.12, 0.70])[0]

    ckd_stage = 'None'
    if renal != 'None':
        ckd_stage = rng.choices(['CKD 3–4', 'CKD 5 / ESRD', 'None'],
                                 weights=[0.55, 0.25, 0.20])[0]

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Mild rod involvement (ERG borderline)', 'None'],
        weights=[0.12, 0.04, 0.84])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Mild fibrosis', 'None'],
        weights=[0.07, 0.02, 0.91])[0]

    situs_inv = rng.choices(
        ['Situs inversus totalis', 'None'],
        weights=[0.02, 0.98])[0]  # dynein-2 — primary non-motile, no nodal; rare incidental

    veptr = rng.choices(
        ['VEPTR inserted', 'VEPTR planned', 'MAGEC growing rods', 'Conservative', 'SRPS (perinatal)'],
        weights=[0.32, 0.12, 0.12, 0.36, 0.08])[0]
    if null_a:
        veptr = 'SRPS (perinatal)'

    cilia_em = rng.choices(
        ['Club/bulging tip (dynein-2 class)', 'Club/bulging + IFT-B distal accumulation', 'Not done'],
        weights=[0.40, 0.25, 0.35])[0]

    presentation = rng.choices(
        ['Prenatal USS + genetic diagnosis', 'Postnatal skeletal survey', 'Newborn respiratory'],
        weights=[0.42, 0.40, 0.18])[0]

    tx_renal = 'None'
    if ckd_stage == 'CKD 5 / ESRD':
        tx_renal = rng.choices(
            ['Renal transplant (curative)', 'Haemodialysis (awaiting Tx)', 'Peritoneal dialysis'],
            weights=[0.58, 0.28, 0.14])[0]

    resp_mgmt = rng.choices(
        ['VEPTR + no ventilation', 'CPAP / NIV', 'Mechanical ventilation (neonatal)', 'None (mild)'],
        weights=[0.36, 0.20, 0.14, 0.30])[0]
    if null_a:
        resp_mgmt = 'Mechanical ventilation (neonatal)'

    misdiagnosis = rng.choices(
        ['Initially SRTD3 (DYNC2H1)', 'Initially SRTD8 (WDR60)', 'Initially SRTD11 (WDR34)', 'None'],
        weights=[0.22, 0.12, 0.08, 0.58])[0]

    sex  = rng.choice(['M', 'F'])
    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.52, 0.28, 0.14, 0.06])[0]

    patients.append({
        'id':           f'SRTD20-P{i+1:02d}',
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
        'situs_inv':    situs_inv,
        'veptr':        veptr,
        'cilia_em':     cilia_em,
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
veptr_any_n     = _cnt_any('veptr', ['VEPTR inserted', 'VEPTR planned', 'MAGEC growing rods'])
transplant_done_n = _cnt('tx_renal', 'Renal transplant (curative)')
misdiagnosis_n  = _cnt_any('misdiagnosis', ['Initially SRTD3 (DYNC2H1)', 'Initially SRTD8 (WDR60)', 'Initially SRTD11 (WDR34)'])
srps_n          = _cnt('veptr', 'SRPS (perinatal)')
club_cilia_n    = _cnt_any('cilia_em', ['Club/bulging tip (dynein-2 class)', 'Club/bulging + IFT-B distal accumulation'])

age_dx_0_1   = _cnt('age_dx', '0–1 yr (neonatal)')
age_dx_2_5   = _cnt('age_dx', '2–5 yr (infant)')
age_dx_6_10  = _cnt('age_dx', '6–10 yr (child)')
age_dx_11_16 = _cnt('age_dx', '11–16 yr (teen)')

# ── public API helpers ────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease":    "TCTEX1D2 Short-Rib Thoracic Dysplasia 20 (SRTD20 / ATD20)",
        "gene":       "TCTEX1D2 (*617022) — 3q21.3 — 276 aa — Dynein-2 Tctex-type Light Chain (DYNLT2B)",
        "omim_gene":  "617022",
        "omim_disease": "617925",
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
            "srps_n":             srps_n,
            "srps_pct":           _pct(srps_n),
            "club_cilia_n":       club_cilia_n,
            "club_cilia_pct":     _pct(club_cilia_n),
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
            "TCTEX1D2 loss destabilises the dynein-2 retrograde IFT motor complex. TCTEX1D2 is the "
            "dynein-2-specific Tctex-type light chain (DYNLT2B); it bridges the heavy chain (DYNC2H1/SRTD3) "
            "and the intermediate chains (WDR34/SRTD11, WDR60/SRTD8) within the assembled dynein-2 motor. "
            "Without TCTEX1D2, dynein-2 assembly is incomplete → RETROGRADE IFT fails → IFT-B cargo "
            "(IFT25, IFT27, IFT81) accumulates at the CILIARY TIP → CLUB/BULGING cilia on EM "
            "(DYNEIN-2 CLASS; identical pattern to SRTD3/DYNC2H1 and SRTD8/WDR60). "
            "Ciliary GLI2/GLI3 processing at the tip is disrupted → Hedgehog (Ihh/Shh) signal collapse "
            "in growth plate chondrocytes → NARROW THORAX, short ribs, shortened limbs (SRTD20). "
            "Secondary organ involvement: renal cysts/NPHP-like (~30%), retinal dystrophy (~16%), "
            "hepatic CHF (~9%). SITUS INVERSUS ABSENT — primary non-motile cilia (9+0 ultrastructure); "
            "TCTEX1D2 is dynein-2-specific, not dynein-1 / motile ciliary dynein."
        ),
        "key_distinction": (
            "TCTEX1D2 (SRTD20) is the FIFTH canonical dynein-2 subunit SRTD gene, completing the "
            "known dynein-2 motor subunit catalogue: DYNC2H1 (heavy chain; SRTD3), DYNC2LI1 (LIC1; SRTD15), "
            "WDR34 (LIC2; SRTD11), WDR60 (IC1; SRTD8), TCTEX1D2 (Tctex light chain; SRTD20). "
            "EM: CLUB/BULGING cilia at tip — the dynein-2 class EM hallmark — shared with SRTD3/8/11/15. "
            "Unlike IFT-A subtype (short-stubby cilia: SRTD2/4/5/7/9/13/14/18) or centriole-assembly "
            "subtype (short-anchoring: SRTD19/CEP120). "
            "NO JBTS ALLELISM confirmed for TCTEX1D2 (2026) — unlike CEP120 (→JBTS31) or KIAA0586 (→JBTS23). "
            "GENE PANEL is MANDATORY — clinical and EM overlap with SRTD3 (DYNC2H1, ~50% of all SRTD) "
            "is complete; TCTEX1D2 is the rarest dynein-2 SRTD subunit and is frequently missed on older panels."
        ),
        "dynein2_complex_table": [
            {"subunit": "DYNC2H1 (heavy chain)", "role": "ATPase motor — drives retrograde IFT; assembles dynein-2 scaffold", "disease": "SRTD3 (~50% all SRTD)", "omim_gene": "603297", "chr": "11q22.3"},
            {"subunit": "DYNC2LI1 (LIC1)",       "role": "Light intermediate chain 1 — motor regulation; cargo release",    "disease": "SRTD15",               "omim_gene": "617089", "chr": "2p21"},
            {"subunit": "WDR34 (LIC2)",           "role": "Light intermediate chain 2 — intermediate chain scaffold",        "disease": "SRTD11",               "omim_gene": "613363", "chr": "9q34.11"},
            {"subunit": "WDR60 (IC1)",            "role": "Intermediate chain 1 — TCTEX1D2 docking; structural scaffold",   "disease": "SRTD8",                "omim_gene": "615462", "chr": "7q36.3"},
            {"subunit": "TCTEX1D2 (Tctex LC)", "role": "Tctex-type light chain — DYNC2H1 + WDR60 bridge; motor stabiliser", "disease": "SRTD20 (THIS)",        "omim_gene": "617022", "chr": "3q21.3"},
            {"subunit": "DYNLL1/2 (roadblock LC)","role": "Roadblock light chain — dynein-2 peripheral cargo adaptor",       "disease": "No OMIM disease (2026)","omim_gene": "617082", "chr": "12q24.31"},
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
        "cilia_em_distribution": _dist('cilia_em'),
        "veptr_distribution": _dist('veptr'),
        "presentation_distribution": _dist('presentation'),
        "misdiagnosis_distribution": _dist('misdiagnosis'),
        "allele_class_summary": _dist('allele'),
        "ethnicity_distribution": _eth_dist(),
        "treatment_renal": _dist('tx_renal'),
        "respiratory_management": _dist('resp_mgmt'),
        "top_variants": [
            {"variant": "p.Arg78Gln (c.233G>A) — Tctex1 N-lobe (aa 70–95) — MENA/Arabian Peninsula founder — ablates DYNC2H1 contact — moderate-severe SRTD20", "n": rng.randint(4, 6)},
            {"variant": "p.Leu142Pro (c.425T>C) — Tctex1 C-lobe (aa 135–155) — South Asian homozygous — WDR60 docking impaired — moderate-severe SRTD20",         "n": rng.randint(2, 4)},
            {"variant": "p.Gly118Arg (c.352G>C) — Tctex1 C-lobe core — pan-ethnic — dimerisation disrupted — moderate SRTD20",                                     "n": rng.randint(2, 3)},
            {"variant": "p.Ala221Val (c.662C>T) — C-terminal linker — European compound het — IFT-B contact partial loss — moderate SRTD20",                       "n": rng.randint(1, 3)},
            {"variant": "p.Glu258Ter (c.772G>T) — C-terminal truncating null — dynein-2 assembly collapse — SRPS spectrum — pan-ethnic",                           "n": rng.randint(1, 2)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "TCTEX1D2 — Tctex1 Domain-Containing Protein 2",
            "also_known_as":    "DYNLT2B; TCTEX1D2; hTctex1d2 (human); dynein-2 Tctex-type light chain",
            "omim_gene":        "*617022",
            "chromosome":       "3q21.3",
            "protein_size":     "276 aa",
            "protein_class":    "Dynein-2 light chain — cytoplasmic dynein-2 retrograde IFT motor subunit",
            "domain_structure": (
                "Tctex1 N-lobe (aa 1–90) — β-sheet sandwich core conserved across all Tctex light chains; "
                "DYNC2H1 heavy-chain tether interface; Arg78 is pathogenic hotspot (MENA founder); "
                "Tctex1 C-lobe (aa 91–200) — WDR34/WDR60 intermediate-chain docking surface; "
                "dimerisation face; Leu142Pro (South Asian); Gly118Arg (pan-ethnic); "
                "C-terminal linker (aa 201–276) — IFT-B peripheral contact; cargo adaptor coordination; "
                "Ala221Val (European moderate); Glu258Ter truncating null → SRPS spectrum"
            ),
            "function": (
                "TCTEX1D2 stabilises the assembled cytoplasmic dynein-2 motor complex by bridging the "
                "heavy chain (DYNC2H1) and intermediate chains (WDR34, WDR60). Dynein-2 powers retrograde "
                "IFT (tip-to-base transport). TCTEX1D2 loss → dynein-2 instability → retrograde IFT failure "
                "→ IFT-B cargo accumulates at ciliary tip → club/bulging cilia on EM (dynein-2 class) → "
                "Hedgehog signal failure in chondrocytes → SRTD20 skeletal phenotype."
            ),
            "inheritance":  "Autosomal Recessive — biallelic LOF",
            "key_contacts": "DYNC2H1 (heavy chain — Tctex1 N-lobe), WDR60 (IC1 — C-lobe), WDR34 (LIC2 — C-lobe), IFT-B periphery (C-terminal linker)",
            "expression":   "Ciliated cells (motile and primary); centrosome/basal body; chondrocytes; renal tubular epithelium; retinal photoreceptor connecting cilium",
        },
        "disease_card": {
            "omim_disease":       "#617925",
            "full_name":          "Short-Rib Thoracic Dysplasia 20 With or Without Polydactyly",
            "alternative_names":  "SRTD20; Asphyxiating Thoracic Dystrophy 20 (ATD20); Jeune syndrome type 20",
            "dynein2_class":      "FIFTH dynein-2 subunit SRTD gene (joining SRTD3/DYNC2H1, SRTD8/WDR60, SRTD11/WDR34, SRTD15/DYNC2LI1)",
            "primary_finding":    "NARROW THORAX — pathognomonic; neonatal/perinatal respiratory failure in severe (null) alleles",
            "cilia_em":           "CLUB/BULGING cilia at tip (dynein-2 class) — IFT-B cargo accumulation (IFT25/IFT27/IFT81) at distal tip; no short-stubby (IFT-A) or short-anchoring (centriole-assembly) features",
            "polydactyly":        "~46% (postaxial predominant; bilateral ~22%, unilateral ~20%, preaxial ~4%)",
            "renal":              "Renal cysts (corticomedullary) + NPHP-like ~30%; ESRD in subset (~8%)",
            "retinal":            "Rod-cone dystrophy in ~16% (dynein-2 failure at connecting cilium — IFT cargo accumulation in outer segment)",
            "hepatic_chf":        "CHF / ductal plate malformation in ~9%",
            "jbts_allelism":      "NOT confirmed (2026) — no JBTS allelic counterpart identified; pure SRTD20 spectrum",
            "situs_inversus":     "ABSENT — TCTEX1D2 is dynein-2-specific (primary non-motile cilia); no nodal motile cilia defect",
            "srps_spectrum":      "Biallelic null alleles → SRPS (perinatal lethal); same severity as SRTD3/DYNC2H1 null",
            "renal_transplant":   "CURATIVE — cell-autonomous dynein-2/basal body defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention); same approach as all SRTD subtypes",
            "prevalence":         "~1/2,000,000–6,000,000; ~10–20 families worldwide (2026); rarest canonical dynein-2 SRTD subtype",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polydactyly, polyhydramnios → suspect SRTD20 (identical to SRTD3); no campomelia (unlike SRTD18/IFT43)",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones; EM phenotype identical to SRTD3 — gene panel is the ONLY differentiator",
            "3. GENE PANEL (mandatory first-line): include TCTEX1D2 — many older panels lack this subunit; comprehensive dynein-2 panel needed: DYNC2H1, WDR60, WDR34, DYNC2LI1, TCTEX1D2",
            "4. WES + CNV if panel negative — TCTEX1D2 copy number variants possible; MLPA for 3q21.3 region",
            "5. Cilia EM (fibroblast/skin): CLUB/BULGING cilia at tip — dynein-2 class; IFT-B cargo accumulation at distal tip; confirms retrograde IFT failure; identical EM to SRTD3/8/11/15",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually from diagnosis; NPHP panel if NPHP-like",
            "7. ERG annually from age 6 (rod-cone dystrophy ~16%); ophthalmology referral",
            "8. Brain MRI: NOT routinely required (no JBTS allelism confirmed for TCTEX1D2 as of 2026)",
            "9. Liver USS + APRI if hepatic signs (CHF ~9%)",
            "10. Cardiac echo if CHD suspected; no situs inversus expected",
        ],
        "mechanism_glossary": [
            {"term": "TCTEX1D2",           "definition": "Tctex1 Domain-Containing Protein 2 — 276 aa dynein-2-specific Tctex-type light chain (DYNLT2B). Bridges DYNC2H1 heavy chain and WDR60/WDR34 intermediate chains. Loss destabilises dynein-2 retrograde IFT motor."},
            {"term": "Dynein-2 complex",   "definition": "The cytoplasmic motor that powers retrograde IFT (tip-to-base). Composed of: DYNC2H1 (heavy chain), DYNC2LI1 + WDR34 (light intermediate chains), WDR60 (intermediate chain), TCTEX1D2 + DYNLL1/2 (light chains). Loss of any subunit → retrograde IFT failure."},
            {"term": "Club/bulging cilia", "definition": "Dynein-2 class EM finding: cilia tip swells due to IFT-B cargo accumulation (IFT25, IFT27, IFT81) unable to return to base. Shared by SRTD3/8/11/15/20 — all dynein-2 subunit mutations show this EM phenotype."},
            {"term": "Retrograde IFT",     "definition": "Tip-to-base transport of IFT-B trains powered by cytoplasmic dynein-2. Cargo includes deactivated kinesin-2, used IFT subunits, and Hh signalling components post-processing. Failure → tip accumulation → Hh disruption → SRTD skeletal phenotype."},
            {"term": "Tctex1 domain",      "definition": "Conserved β-sheet sandwich fold shared by Tctex-type light chains across dynein-1 (DYNLT1/DYNLT3) and dynein-2 (TCTEX1D2). In dynein-2, the Tctex1 fold provides the DYNC2H1-docking interface and WDR60 contact surface."},
            {"term": "Cell-autonomous",    "definition": "TCTEX1D2 defect intrinsic to each ciliated cell. Renal transplantation with donor kidney (TCTEX1D2+) is CURATIVE — no post-transplant recurrence. Standard principle across all SRTD subtypes."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate chondrocyte signalling pathway processed at the ciliary tip. Dynein-2 failure → GLI processing fails → Ihh/Shh signal collapse → SRTD skeletal dysplasia (narrow thorax, short ribs, polydactyly)."},
        ],
        "key_variants": [
            {"variant": "p.Arg78Gln (c.233G>A)",  "domain": "Tctex1 N-lobe (aa 70–95)",          "consequence": "Ablates DYNC2H1 heavy-chain contact; most common SRTD20 allele; MENA/Arabian Peninsula founder", "ethnicity": "MENA / Arabian Peninsula (homozygous)"},
            {"variant": "p.Leu142Pro (c.425T>C)",  "domain": "Tctex1 C-lobe (aa 135–155)",         "consequence": "Destabilises C-lobe β-sheet; WDR60 docking lost; South Asian homozygous; moderate-severe",      "ethnicity": "South Asian (homozygous)"},
            {"variant": "p.Gly118Arg (c.352G>C)",  "domain": "Tctex1 C-lobe core",                 "consequence": "Disrupts dimerisation face; pan-ethnic; moderate SRTD20",                                        "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala221Val (c.662C>T)",  "domain": "C-terminal linker",                  "consequence": "Partial IFT-B contact loss; European compound het; moderate SRTD20",                             "ethnicity": "European"},
            {"variant": "p.Glu258Ter (c.772G>T)",  "domain": "C-terminal truncating null",          "consequence": "Loss of C-terminal linker; null allele → SRPS spectrum in biallelic null; pan-ethnic",          "ethnicity": "Pan-ethnic"},
            {"variant": "c.218+1G>A (splice)",      "domain": "Tctex1 N-lobe/C-lobe junction splice","consequence": "Exon skip → partial protein; moderate-severe SRTD20; European predominant",                    "ethnicity": "European"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months; same approach for all SRTD subtypes",
            "2. Respiratory: neonatal mechanical ventilation if severe thorax; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — cell-autonomous (no recurrence)",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support; no disease-modifying therapy (2026)",
            "5. Hepatic: APRI + USS if CHF/fibrosis suspected; hepatology referral; avoid hepatotoxic drugs",
            "6. Brain MRI: not routinely required (no confirmed JBTS allelism for TCTEX1D2 as of 2026); consider if ataxia or MTS features present clinically",
            "7. Genetics: cascade testing; consanguinity counselling; prenatal/preimplantation genetics (25% recurrence); TCTEX1D2 allele-class discussion (null → SRPS risk; missense → SRTD20)",
            "8. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1)",     "key_difference": "Heavy chain (~50% all SRTD); club/bulging cilia EM — IDENTICAL to SRTD20; most common SRTD; comprehensive dynein-2 panel (DYNC2H1 + TCTEX1D2) resolves"},
            {"disease": "SRTD8 (WDR60)",        "key_difference": "Intermediate chain 1; same dynein-2 club/bulging EM class; WDR60 is TCTEX1D2's direct docking partner — gene panel only differentiator"},
            {"disease": "SRTD11 (WDR34)",       "key_difference": "Light intermediate chain 2; same dynein-2 class EM; WDR34 also contacts TCTEX1D2 C-lobe — gene panel differentiates"},
            {"disease": "SRTD15 (DYNC2LI1)",    "key_difference": "Light intermediate chain 1; same dynein-2 retrograde class; rarer than SRTD3 but commoner than SRTD20 — comprehensive panel required"},
            {"disease": "SRTD5 (WDR19/IFT144)","key_difference": "IFT-A ARM hub; SHORT-STUBBY cilia (IFT-A class) vs. club/bulging (dynein-2) — EM distinguishes; gene panel confirms"},
            {"disease": "SRTD19 (CEP120)",      "key_difference": "Centriole elongation (not dynein-2); SHORT-ANCHORING cilia (distinct from club/bulging); JBTS31 allelism present vs. absent in SRTD20"},
        ],
    }
