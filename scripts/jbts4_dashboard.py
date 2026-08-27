"""
NPHP1 Joubert Syndrome Type 4 (JBTS4) — Nephrocystin-1 / TZ Y-Link / NPHP Dual Phenotype
===========================================================================================
Primary Gene : NPHP1 (*607100) — 2q13; ~732 aa; Nephrocystin-1
               NPHP1 (Nephrocystin-1) is a TRANSITION ZONE (TZ) Y-link scaffold protein.
               NPHP1 forms a tripartite complex with AHI1 (Jouberin/JBTS3) and NPHP4 at the
               ciliary transition zone Y-links. Loss of NPHP1 → TZ Y-link structure fails
               → GPCRs (SSTR3, MCHR1) and Smoothened FAIL to enter neuronal cilia
               → impaired Hedgehog, somatostatin, and MCH receptor signalling in neurons →
               Molar Tooth Sign (MTS) on brain MRI, cerebellar vermis hypoplasia.
               Cilia are structurally intact (9+2 axoneme preserved) — a pure TZ gating defect.
Disease OMIM : #609583 — Joubert Syndrome 4 (JBTS4)
               Also: #256100 — Nephronophthisis 1 (NPHP1) — biallelic null alleles → pure renal
               Also: #266900 — Senior-Løken Syndrome type 1 (SLS1) — rare retinal + renal overlap
Chromosome   : 2q13
Inheritance  : Autosomal Recessive — biallelic LOF; ALLELE CLASS GOVERNS phenotype
Prevalence   : ~1–3% of all Joubert syndrome cases; JBTS overall ~1/80,000–100,000 births;
               JBTS4 ~1/2,500,000–8,000,000 worldwide (true JBTS phenotype)
               NPHP1 (pure nephronophthisis): ~1/50,000 — the most common cause of NPHP worldwide

⚠ KEY DIAGNOSTIC PEARL — LARGE HOMOZYGOUS DELETION:
The most common NPHP1 pathogenic allele worldwide (~50–60% of all NPHP1 disease alleles) is a
LARGE ~610 kb HOMOZYGOUS DELETION of the NPHP1 gene region at Chr 2q13.  This deletion is
INVISIBLE TO WES/PANEL SEQUENCING — detected ONLY by MLPA, aCGH, or long-read WGS.
Failure to test by MLPA in a negative gene panel = diagnostic miss.  Always add MLPA
when NPHP1 is clinically suspected and panel is negative.

Protein Structure — NPHP1 / Nephrocystin-1 (732 aa; transition zone Y-link scaffold)
---------------------------------------------------------------------------------------
Domain 1: N-terminal coiled-coil (CC1, aa 1–200)   — NPHP4 binding surface; TZ Y-link anchor
Domain 2: SH3 domain (aa ~250–310)                 — AHI1 (Jouberin/JBTS3 partner), PSTPIP-1,
           CD2AP interaction; interacts with NPHP4 focal adhesion signalling
Domain 3: C-terminal coiled-coil (CC2, aa ~450–650)— NPHP4 heterodimerisation; TZ anchoring;
           most truncating pathogenic variants fall proximal to or within CC2
Domain 4: C-terminus (aa ~650–732)                 — TZ transition plate anchoring; IQ-motif
           for calmodulin binding; calmodulin regulates NPHP1 TZ localisation

Key pathogenic variant classes (NPHP1):
1. Large ~610kb homozygous NPHP1 deletion (Chr 2q13): European most common; MLPA/aCGH required;
   biallelic null → NPHP1 (pure nephronophthisis, NO MTS) — NOT JBTS4
2. Compound het: large deletion + hypomorphic missense (Thr323Met): JBTS4 compound het;
   European most common JBTS4 allele class; MTS + renal
3. Frameshift / truncating compound het: biallelic null → NPHP1 (no MTS); single null + hypomorph → JBTS4
4. Homozygous Arg328Ter: rare biallelic null; pure NPHP phenotype; pan-ethnic
5. Compound het missense–missense (hypomorphic pair): JBTS4 with milder course

JBTS4 vs NPHP1 Allele-Phenotype Rule:
  Biallelic NULL (two truncating/deletion) → NPHP1 (pure nephronophthisis, NO brain MTS)
  One NULL + one HYPOMORPHIC (e.g., Thr323Met) → JBTS4 (MTS + renal NPHP-type)
  Two HYPOMORPHIC → JBTS4 mild form (rare)
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
    ('European',                       0.45),   # large deletion + hypomorph most common
    ('Middle Eastern / North African', 0.25),   # consanguineous enrichment
    ('South Asian',                    0.15),
    ('Ashkenazi Jewish',               0.06),
    ('East Asian',                     0.05),
    ('Other / Unknown',                0.04),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het: large deletion + Thr323Met hypomorphic (JBTS4 class)',
    'Compound het: frameshift null + Thr323Met hypomorphic',
    'Compound het: missense + missense (both hypomorphic)',
    'Homozygous frameshift null (misclassified JBTS4)',
    'Compound het: splice null + hypomorphic missense',
]
allele_fracs = [0.42, 0.25, 0.15, 0.10, 0.08]
allele_pool = []
for ac, frac in zip(allele_classes, allele_fracs):
    allele_pool.extend([ac] * round(frac * N))
while len(allele_pool) < N:
    allele_pool.append(allele_classes[0])
rng.shuffle(allele_pool)

age_dx_pool = (
    [rng.randint(0, 2) for _ in range(20)] +     # neonatal / infantile (MTS on MRI)
    [rng.randint(3, 10) for _ in range(12)] +     # childhood
    [rng.randint(11, 25) for _ in range(8)]       # older (renal-first presentation)
)
rng.shuffle(age_dx_pool)

for i in range(N):
    eth = eth_pool[i]
    allele = allele_pool[i]
    age_dx = age_dx_pool[i]
    has_mts = allele not in ['Homozygous frameshift null (misclassified JBTS4)']
    has_renal = rng.random() < 0.45          # ~45% NPHP-type renal disease (high vs JBTS3 8%)
    has_retinal = rng.random() < 0.38        # ~38% rod-cone dystrophy
    has_oma = rng.random() < 0.40            # ~40% oculomotor apraxia (lower than JBTS3 75%)
    has_ataxia = rng.random() < 0.88         # ~88% cerebellar ataxia
    has_hypotonia = rng.random() < 0.85      # ~85% neonatal hypotonia
    has_breathing = rng.random() < 0.55      # ~55% breathing dysregulation (episodic apnea)
    has_id = rng.random() < 0.70             # ~70% intellectual disability
    has_liver = rng.random() < 0.05          # ~5% hepatic fibrosis (rare, MKS-overlap alleles)
    mlpa_missed = (i < 6)                    # 15% — deletion missed before MLPA
    patients.append({
        'id': f'JBTS4-{i+1:03d}',
        'ethnicity': eth,
        'allele_class': allele,
        'age_diagnosis': age_dx,
        'mts': has_mts,
        'renal_nphp': has_renal,
        'retinal_dystrophy': has_retinal,
        'oculomotor_apraxia': has_oma,
        'cerebellar_ataxia': has_ataxia,
        'neonatal_hypotonia': has_hypotonia,
        'breathing_dysregulation': has_breathing,
        'intellectual_disability': has_id,
        'hepatic_fibrosis': has_liver,
        'mlpa_required_for_dx': mlpa_missed,
    })


# ── API response builders ─────────────────────────────────────────────────────
def get_overview():
    mts_n     = sum(1 for p in patients if p['mts'])
    renal_n   = sum(1 for p in patients if p['renal_nphp'])
    retinal_n = sum(1 for p in patients if p['retinal_dystrophy'])
    oma_n     = sum(1 for p in patients if p['oculomotor_apraxia'])
    ataxia_n  = sum(1 for p in patients if p['cerebellar_ataxia'])
    hypotonia_n = sum(1 for p in patients if p['neonatal_hypotonia'])
    breath_n  = sum(1 for p in patients if p['breathing_dysregulation'])
    id_n      = sum(1 for p in patients if p['intellectual_disability'])
    mlpa_n    = sum(1 for p in patients if p['mlpa_required_for_dx'])

    median_dx = sorted(p['age_diagnosis'] for p in patients)[N // 2]

    return {
        "disease": "Joubert Syndrome Type 4 (JBTS4)",
        "gene": "NPHP1",
        "gene_full": "Nephrocystin-1 (NPHP1) — Transition Zone Y-Link Scaffold",
        "omim_gene": "607100",
        "omim_disease": "609583",
        "chromosome": "2q13",
        "inheritance": "Autosomal Recessive (biallelic LOF — allele class governs phenotype)",
        "cohort_n": N,
        "seed": SEED,
        "kpis": [
            {"label": "Molar Tooth Sign",       "value": f"{_pct(mts_n)}%",  "note": "MTS on brain MRI — pathognomonic JBTS"},
            {"label": "Renal NPHP-type",        "value": f"{_pct(renal_n)}%","note": "Tubuloint. nephritis → ESRD risk; HIGH vs JBTS3 8%"},
            {"label": "Retinal Dystrophy",      "value": f"{_pct(retinal_n)}%","note": "Rod-cone ERG mandatory"},
            {"label": "Oculomotor Apraxia",     "value": f"{_pct(oma_n)}%",  "note": "LOWER than JBTS3 (~75%) — ~40% in JBTS4"},
            {"label": "Cerebellar Ataxia",      "value": f"{_pct(ataxia_n)}%","note": "Near-universal"},
            {"label": "Neonatal Hypotonia",     "value": f"{_pct(hypotonia_n)}%","note": "Presenting feature"},
            {"label": "Breathing Dysregulation","value": f"{_pct(breath_n)}%","note": "Episodic apnea/hyperpnea; neonatal"},
            {"label": "Intellectual Disability","value": f"{_pct(id_n)}%",   "note": "Moderate-severe range"},
            {"label": "MLPA Diagnostic",        "value": f"{_pct(mlpa_n)}%", "note": "Large deletion INVISIBLE to WES — MLPA essential"},
        ],
        "median_age_diagnosis_yr": median_dx,
        "hallmark": "Molar Tooth Sign (MTS) on axial brain MRI + HIGH renal NPHP-type disease (~45%)",
        "critical_diagnostic_pearl": (
            "Large ~610 kb homozygous NPHP1 deletion is the most common pathogenic allele (~50–60% of NPHP1 disease "
            "alleles worldwide) and is INVISIBLE TO WES/PANEL SEQUENCING. MLPA or aCGH is MANDATORY when NPHP1 is "
            "suspected and gene panel is negative. Failure to test = diagnostic miss."
        ),
        "allele_phenotype_rule": (
            "Biallelic NULL → NPHP1 (pure nephronophthisis, NO MTS). "
            "One NULL + one HYPOMORPHIC (Thr323Met) → JBTS4 (MTS + renal NPHP). "
            "Allele class governs phenotype — identical gene, opposite brain phenotype."
        ),
        "prevalence": "~1–3% of all JBTS; JBTS4 ~1/2.5–8M worldwide; NPHP1 pure ~1/50,000",
        "frequency_in_jbts": "~1–3% of all Joubert syndrome",
        "first_description": "Parisi et al., 2004 (NPHP1 del found in JBTS patients); Valente et al., 2006",
    }


def get_breakdown():
    # Allele-class distribution
    allele_counts = {}
    for p in patients:
        allele_counts[p['allele_class']] = allele_counts.get(p['allele_class'], 0) + 1
    allele_summary = [{"class": k, "n": v, "pct": _pct(v)} for k, v in sorted(allele_counts.items(), key=lambda x: -x[1])]

    # Ethnicity
    eth_counts = {}
    for p in patients:
        eth_counts[p['ethnicity']] = eth_counts.get(p['ethnicity'], 0) + 1
    eth_dist = [{"ethnicity": k, "n": v, "pct": _pct(v)} for k, v in sorted(eth_counts.items(), key=lambda x: -x[1])]

    # Age at diagnosis distribution
    age_bins = {'0–2 yr (neonatal/infantile)': 0, '3–10 yr (childhood)': 0, '11–25 yr (late)': 0}
    for p in patients:
        a = p['age_diagnosis']
        if a <= 2:
            age_bins['0–2 yr (neonatal/infantile)'] += 1
        elif a <= 10:
            age_bins['3–10 yr (childhood)'] += 1
        else:
            age_bins['11–25 yr (late)'] += 1
    age_dist = [{"bin": k, "n": v, "pct": _pct(v)} for k, v in age_bins.items()]

    # Feature co-occurrence matrix (selected)
    renal_and_retinal = sum(1 for p in patients if p['renal_nphp'] and p['retinal_dystrophy'])
    renal_and_mts     = sum(1 for p in patients if p['renal_nphp'] and p['mts'])
    mts_and_oma       = sum(1 for p in patients if p['mts'] and p['oculomotor_apraxia'])
    mts_and_ataxia    = sum(1 for p in patients if p['mts'] and p['cerebellar_ataxia'])

    # MTS distribution
    mts_present = sum(1 for p in patients if p['mts'])
    mts_absent  = N - mts_present

    # MLPA impact
    mlpa_required = sum(1 for p in patients if p['mlpa_required_for_dx'])

    return {
        "allele_class_summary": allele_summary,
        "ethnicity_distribution": eth_dist,
        "age_at_diagnosis_distribution": age_dist,
        "mts_distribution": [
            {"label": "MTS present (JBTS4 confirmed)", "n": mts_present, "pct": _pct(mts_present)},
            {"label": "MTS absent (biallelic null, NPHP1 phenotype)", "n": mts_absent, "pct": _pct(mts_absent)},
        ],
        "feature_cooccurrence": [
            {"pair": "Renal NPHP + Retinal dystrophy (SLS1-like)",     "n": renal_and_retinal, "pct": _pct(renal_and_retinal)},
            {"pair": "Renal NPHP + Molar Tooth Sign (JBTS4 hallmark)", "n": renal_and_mts,     "pct": _pct(renal_and_mts)},
            {"pair": "MTS + Oculomotor Apraxia",                        "n": mts_and_oma,       "pct": _pct(mts_and_oma)},
            {"pair": "MTS + Cerebellar Ataxia (near-universal)",        "n": mts_and_ataxia,    "pct": _pct(mts_and_ataxia)},
        ],
        "mlpa_impact": {
            "missed_by_wes_alone": mlpa_required,
            "pct_missed": _pct(mlpa_required),
            "note": (
                "Large ~610kb NPHP1 deletion INVISIBLE to WES; these patients diagnosed only after MLPA. "
                "Without MLPA, 15% of JBTS4 cohort would have remained undiagnosed."
            ),
        },
        "renal_severity_detail": {
            "nphp_present": sum(1 for p in patients if p['renal_nphp']),
            "nphp_pct": _pct(sum(1 for p in patients if p['renal_nphp'])),
            "note": (
                "NPHP-type tubulointerstitial nephritis: progressive fibrosis → ESRD by 3rd decade. "
                "Renal USS + creatinine/eGFR annual from diagnosis. "
                "Transplant is curative for renal endpoint; does NOT correct brain/retinal."
            ),
        },
    }


def get_definitions():
    return {
        "gene_card": {
            "gene": "NPHP1",
            "full_name": "Nephrocystin-1",
            "omim": "607100",
            "chromosome": "2q13",
            "protein_size": "732 aa",
            "function": (
                "Transition Zone (TZ) Y-link scaffold: forms tripartite complex with AHI1 (Jouberin/JBTS3) "
                "and NPHP4 at ciliary TZ Y-links. Maintains TZ gate → controls GPCR entry "
                "(SSTR3, MCHR1, Smo) into neuronal and renal tubular cilia. "
                "Also localises to focal adhesions and interacts with paxillin, PSTPIP-1, CD2AP "
                "in kidney tubular epithelium."
            ),
            "domains": [
                {"name": "CC1 — N-terminal coiled-coil (aa 1–200)",  "role": "NPHP4 binding; TZ Y-link anchor; CC1–CC1 homodimerisation"},
                {"name": "SH3 domain (aa ~250–310)",                   "role": "AHI1 (Jouberin) binding; PSTPIP-1, CD2AP, p130Cas interaction; focal adhesion signalling"},
                {"name": "CC2 — C-terminal coiled-coil (aa ~450–650)","role": "NPHP4 heterodimerisation; TZ anchoring platform; most null variants proximal to CC2"},
                {"name": "IQ-motif / C-terminus (aa ~650–732)",        "role": "Calmodulin binding; calmodulin regulates NPHP1 TZ localisation dynamically"},
            ],
            "mechanism_of_disease": (
                "Loss of NPHP1 → TZ Y-link scaffold incomplete → NPHP4–AHI1–NPHP1 tripartite TZ complex fails "
                "→ TZ gate function lost → GPCRs (SSTR3, MCHR1) and Smoothened cannot enter neuronal cilia "
                "→ Molar Tooth Sign, cerebellar vermis hypoplasia. In renal tubular cells: "
                "NPHP1 loss → TZ gate + focal adhesion signalling impaired → progressive "
                "tubulointerstitial nephritis → medullary cyst → ESRD."
            ),
            "allele_phenotype_switch": (
                "BIALLELIC NULL (two truncating or large deletion) → NPHP1 (#256100): "
                "pure nephronophthisis, NO Molar Tooth Sign, NO cerebellar MTS. "
                "ONE NULL + ONE HYPOMORPHIC (Thr323Met, Val617Met) → JBTS4 (#609583): "
                "MTS + renal NPHP-type disease. This allele-class switch is unique to NPHP1 "
                "among all JBTS genes and is critical for correct phenotype prediction."
            ),
        },
        "key_variants": [
            {"variant": "Large ~610 kb homozygous NPHP1 deletion",     "domain": "Entire NPHP1 gene deleted (Chr 2q13 inverted repeat)",                      "consequence": "Biallelic null → NPHP1 (nephronophthisis, NO MTS). Undetectable by WES — MLPA mandatory. Most common NPHP1 allele worldwide (~50–60%).", "ethnicity": "European (most common homozygous), global"},
            {"variant": "p.Thr323Met (c.968C>T)",                       "domain": "CC1–SH3 junction (aa 323); NPHP4-contact residue",                           "consequence": "Hypomorphic missense; partial TZ scaffold function retained; when compound het with null → JBTS4 phenotype (MTS). Key JBTS4 allele.", "ethnicity": "European (compound het with deletion or frameshift)"},
            {"variant": "p.Arg328Ter (c.982C>T)",                       "domain": "CC1–SH3 junction truncating null (aa 328)",                                   "consequence": "Premature stop; null allele; biallelic null → NPHP1; compound het + hypomorph → JBTS4. Pan-ethnic.", "ethnicity": "Pan-ethnic"},
            {"variant": "p.Leu57AlafsX18 (c.169_170delCT)",            "domain": "N-terminus frameshift null (aa 57)",                                          "consequence": "Early frameshift; NMD; null allele; biallelic → NPHP1; compound het with hypomorph → JBTS4.", "ethnicity": "Pan-ethnic / European"},
            {"variant": "p.Val617Met (c.1849G>A)",                      "domain": "IQ-motif adjacent (aa 617); calmodulin-binding region",                       "consequence": "Hypomorphic; partial calmodulin-NPHP1 binding retained; JBTS4-associated when compound het with null; mild-moderate MTS.", "ethnicity": "European / MENA"},
            {"variant": "Compound het: deletion + Thr323Met",           "domain": "One null (610kb del) + one hypomorphic (Thr323Met)",                          "consequence": "Most common JBTS4 genotype worldwide (~42% of JBTS4 cohort). MTS + renal NPHP-type. Requires MLPA to detect deletion allele.", "ethnicity": "European (most common JBTS4 genotype)"},
        ],
        "treatment_summary": [
            "1. No disease-modifying therapy for JBTS4 (2026) — symptomatic, supportive, organ-specific management.",
            "2. MOLAR TOOTH SIGN / BRAIN: MRI at diagnosis; annual neurology follow-up; no curative intervention for cerebellar vermis hypoplasia.",
            "3. RENAL NPHP-TYPE (~45%): Annual renal USS from diagnosis; creatinine/eGFR every 6 months from age 5; ACE inhibitor/ARB if proteinuria; renal transplant is CURATIVE for the renal endpoint (does not correct brain/retinal disease); list early when eGFR declining.",
            "4. BREATHING DYSREGULATION (~55%): Neonatal NICU monitoring; pulse oximetry; caffeine for apnea episodes; polysomnography before discharge. Usually resolves by 6 months.",
            "5. HYPOTONIA + ATAXIA: Physiotherapy from diagnosis; occupational therapy; hydrotherapy; adaptive gait aids (walker, AFO) as needed.",
            "6. INTELLECTUAL DISABILITY (~70%): Early intervention; speech-language therapy; special education; neuropsychological assessment by age 3.",
            "7. OCULOMOTOR APRAXIA (~40%): Low vision support; visual aids; occupational therapy for reading/navigation.",
            "8. RETINAL DYSTROPHY (~38%): Annual ERG from age 3; ophthalmology retinal specialist; low vision aids; no NPHP1-specific gene therapy trial (2026).",
            "9. DIAGNOSTIC CRITICAL: Always order MLPA when NPHP1 is clinically suspected and gene panel is negative — the ~610kb deletion is WES-invisible. Report MLPA result before closing the diagnostic workup.",
            "10. GENETICS: Carrier frequency of NPHP1 deletion ~1/150 in European; cascade family testing; prenatal USS (MTS detectable from 18–20 wk); preimplantation genetic testing (25% recurrence AR).",
            "11. MDT: Paediatric neurology, nephrology, ophthalmology, physiotherapy, OT, speech-language therapy, genetics, neonatology.",
        ],
        "ddx_table": [
            {"disease": "JBTS3 (AHI1)", "key_difference": "AHI1 is the NPHP1 binding partner — same TZ complex, same MTS. JBTS3: OMA ~75% (higher than JBTS4 ~40%); renal very rare ~8% (vs JBTS4 ~45%). Gene panel differentiates. Ashkenazi Arg830Trp founder vs NPHP1 large deletion."},
            {"disease": "NPHP1 (pure nephronophthisis)", "key_difference": "SAME GENE, biallelic null alleles → NO MTS, NO cerebellar brain phenotype. Pure tubulointerstitial nephritis → ESRD. MLPA shows homozygous deletion. Allele class distinguishes JBTS4 from NPHP1 completely."},
            {"disease": "Senior-Løken Syndrome type 1 (SLS1)", "key_difference": "NPHP1-caused SLS1: renal NPHP + retinal dystrophy WITHOUT MTS. Intermediate phenotype between NPHP1 and JBTS4. Allele class governs. MRI brain required to distinguish from JBTS4."},
            {"disease": "JBTS5 (CEP290)", "key_difference": "CEP290: MTS + retinal dystrophy ~70% (higher than JBTS4 ~38%); renal ~25%. LCA10 (retinal-only) if biallelic null. CEP290 is second most common JBTS gene. Gene panel differentiates."},
            {"disease": "Bardet-Biedl Syndrome (BBS)", "key_difference": "BBS: polydactyly + obesity — both ABSENT in JBTS4. MTS ABSENT in BBS. Retinal dystrophy in BBS ~85%. Both can have renal anomalies — MTS on MRI confirms JBTS, not BBS."},
            {"disease": "SRTD (Short-Rib Thoracic Dysplasia)", "key_difference": "SRTD: narrow thorax + rib-shortening pathognomonic — ABSENT in JBTS4. MTS absent in SRTD. Polydactyly common in SRTD, absent in JBTS4."},
            {"disease": "MKS (Meckel-Gruber Syndrome)", "key_difference": "MKS: lethal; encephalocele + polydactyly + renal cystic dysplasia. NPHP1 null alleles do NOT cause MKS (no encephalocele). MKS is caused by MKS1, CC2D2A, TMEM67 biallelic null — not NPHP1."},
            {"disease": "Dandy-Walker Syndrome", "key_difference": "Dandy-Walker: cystic 4th ventricle dilatation + absent cerebellar vermis — different MRI pattern from MTS. No ciliopathy genetics (usually). MTS pattern on MRI confirms JBTS4, not Dandy-Walker."},
        ],
        "nphp1_module_comparison": [
            {"gene": "NPHP1",  "jbts": "JBTS4",  "omim_gene": "607100", "chr": "2q13",    "module": "TZ Y-link scaffold (AHI1/NPHP4 complex)", "allele_switch": "Null→NPHP1; Null+Hypomorph→JBTS4", "renal": "HIGH ~45% (NPHP-dominant)", "retinal": "~38%", "oma": "~40%"},
            {"gene": "AHI1",   "jbts": "JBTS3",  "omim_gene": "608894", "chr": "6q23.3",  "module": "TZ Y-link scaffold (NPHP1/NPHP4 partner)", "allele_switch": "No dual phenotype", "renal": "Very rare ~8%", "retinal": "~52%", "oma": "~75%"},
            {"gene": "NPHP4",  "jbts": "JBTS",   "omim_gene": "607215", "chr": "1p36.31", "module": "TZ Y-link core (NPHP1/AHI1 partner)",    "allele_switch": "Null→NPHP4; JBTS rare", "renal": "Moderate ~30%", "retinal": "~45%", "oma": "~50%"},
            {"gene": "INPP5E", "jbts": "JBTS1",  "omim_gene": "613037", "chr": "9q34.3",  "module": "Ciliary PI(3,4,5)P3 phosphatase",         "allele_switch": "No dual phenotype", "renal": "Rare", "retinal": "~30%", "oma": "~25%"},
            {"gene": "CEP290", "jbts": "JBTS5",  "omim_gene": "610142", "chr": "12q21.32","module": "TZ transition plate + Y-link",             "allele_switch": "Null→LCA10/MKS4; Hypomorph→JBTS5", "renal": "~25%", "retinal": "~70%", "oma": "~55%"},
            {"gene": "TMEM67", "jbts": "JBTS6",  "omim_gene": "609884", "chr": "8q22.1",  "module": "TZ membrane (MKS/TCTN module)",            "allele_switch": "Null→MKS3; Hypomorph→JBTS6", "renal": "High+liver", "retinal": "~55%", "oma": "~50%"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== JBTS4 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("MTS:", b["mts_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("Ethnicity:", b["ethnicity_distribution"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
