"""
NEK1 Short-Rib Thoracic Dysplasia 6 (SRTD6) — Jeune Asphyxiating Thoracic Dystrophy Type 6
==============================================================================================
Primary Gene : NEK1 (*604588) — NIMA-Related Kinase 1 — 4q33; ~1258 aa;
               Serine/threonine kinase at the basal body / mother centriole.
               NEK1 is the ONLY ciliogenesis kinase whose loss causes SRTD.
               It is NOT an IFT subunit (not IFT-A, not IFT-B2); NOT a dynein-2
               motor subunit — it belongs to a distinct fourth molecular class:
               BASAL BODY KINASE / CILIOGENESIS INITIATOR.
               NEK1 phosphorylates TTBK2 (Tau Tubulin Kinase 2) at distal appendages,
               which is the master switch for cilia assembly initiation.  NEK1 also
               phosphorylates transition-zone proteins (CEP164, RPGRIP1L, NPHP4) and
               is required for CP110 removal from the mother centriole to allow axoneme
               nucleation.
               Loss → TTBK2 not activated → CP110 not removed → axoneme cannot nucleate
               → ABSENT or RUDIMENTARY cilia (no elongation whatsoever).
               Ciliary EM: ABSENT / RUDIMENTARY cilia stubs — fourth and
               mechanistically DISTINCT EM class from:
                 IFT-B2 (SHORTENED), IFT-A (SHORT STUBBY), Dynein-2 (CLUB/BULGING TIP).
               No axoneme → Hedgehog (Ihh/Shh) signaling completely absent →
               GLI3R maximally accumulates in chondrocytes → NARROW THORAX (severest).
               Additional dual role: NEK1 phosphorylates ATRIP/CHEK1 → DNA damage
               response checkpoint; this is NOT expressed as a clinical phenotype in
               biallelic LOF but contributes to genomic instability in heterozygous
               carriers (not clinically actionable 2026).

Disease OMIM : #263520 — Short-Rib Thoracic Dysplasia 6 with or without polydactyly
               (SRTD6 / ATD6 / Majewski syndrome / Short-Rib Polydactyly Syndrome type II,
               Majewski type / SRPS type II).
               SRTD6 encompasses the full NEK1-related SRTD spectrum:
               — Severe end: SRPS type II (Majewski): perinatal lethal, biallelic null,
                 hydrops fetalis, absent cilia, severe medianasal hypoplasia.
               — Moderate: Jeune ATD6: survivable but severe thoracic restrictive disease.
               — Mild end: Hypomorphic kinase missense → adult survivors.
               NEK1 is on the NIMA kinase family (same family as NPHP9/NEK8) but NEK1
               has a distinct substrate set and ciliopathy phenotype from NEK8/NPHP9.
               UNIQUE FEATURES vs all other SRTD genes:
               (1) ABSENT/RUDIMENTARY cilia EM — unique fourth EM class.
               (2) HYDROPS FETALIS ~20%: lymphatic cilia are NEK1-dependent;
                   impaired lymphatic ciliogenesis → fetal hydrops → unique among SRTD.
               (3) MEDIANASAL HYPOPLASIA ~30%: Majewski phenotypic feature;
                   midline facial cleft/nasal hypoplasia; NOT seen in other SRTD types.
               (4) LARYNGEAL STENOSIS/HYPOPLASIA ~15%: airway compromise beyond
                   narrow thorax; unique to SRTD6.
               (5) HIGHEST POLYDACTYLY RATE: 65–75% — highest of ALL SRTD genes;
                   both postaxial AND preaxial (vs mostly postaxial in other SRTDs).
               (6) DUAL KINASE ROLE: NEK1 kinase domain mutation → SRTD6;
                   NEK1 is NOT an IFT component — gene panel must explicitly include NEK1.
Chromosome   : 4q33
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1:500,000–1,000,000; ~30–60 families reported worldwide as of 2026.
               Under-ascertained: NEK1 was absent from early SRTD panels focussed on
               IFT subunits.  Polydactyly-prominent neonatal deaths historically
               classified as "Majewski SRPS type II" may not have had NEK1 confirmed
               molecularly.

Protein Structure — NEK1 (1258 aa; basal body kinase / ciliogenesis master switch)
------------------------------------------------------------------------------------
Domain 1: N-lobe kinase domain (aa 1–130)
           ATP-binding pocket; glycine-rich P-loop (GXGXXG motif); Lys42 catalytic.
           Pathogenic missense hotspot: kinase activity abolished → moderate SRTD6.
Domain 2: C-lobe kinase domain (aa 131–270)
           DFG activation loop (Asp179-Phe180-Gly181); substrate-binding cleft;
           catalytic Asp179 essential for phospho-transfer.
           DFG motif missense → severe loss of kinase activity → severe SRTD6.
Domain 3: Kinase-coiled-coil linker region (aa 271–600)
           Bridge between kinase output and coiled-coil structural domain;
           hypomorphic missense here → partial TTBK2 phosphorylation preserved → mild SRTD6.
Domain 4: N-terminal coiled-coil (aa 601–900)
           Dimerization; basal body localisation signal; NEK1 dimerises for full activity.
           Loss of dimerisation → kinase partially active; moderate-severe SRTD6.
Domain 5: C-terminal coiled-coil / regulatory domain (aa 900–1258)
           Interaction with TTBK2 (Tau Tubulin Kinase 2) and CEP164;
           hypomorphic missense → retained basal body localisation but reduced TTBK2
           phosphorylation → mild; some adult survivors possible.

Key pathogenic variant classes (NEK1):
1. Kinase N-lobe missense (aa 1–130): compound het or homozygous; moderate SRTD6
2. DFG activation loop missense (aa 131–270): MENA homozygous; moderate-severe
3. Biallelic truncating (null × null): SRPS type II spectrum; perinatal lethal (40–50%)
4. Kinase-linker missense (aa 271–600): South Asian hypomorphic; mild; adult survivors
5. C-terminal coiled-coil missense (aa 900–1258): Middle Eastern; severe
"""

import random
import math

SEED = 405
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
    ('Middle Eastern / North African', 0.35),  # consanguineous — homozygous kinase missense common
    ('European',                        0.25),  # compound het
    ('South Asian',                     0.20),  # hypomorphic linker alleles
    ('East Asian',                      0.10),
    ('Latin American',                  0.07),
    ('Other / Unknown',                 0.03),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Homozygous kinase domain missense (N-lobe aa 1–130)',
    'Compound het missense + truncating',
    'Biallelic truncating (null/null — SRPS II spectrum)',
    'Compound het two missense',
    'Hypomorphic kinase-linker missense (aa 271–600)',
]
allele_weights = [0.30, 0.28, 0.20, 0.12, 0.10]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    sex    = rng.choice(['M', 'F'])

    # Thorax severity
    thorax_map = {
        'Biallelic truncating (null/null — SRPS II spectrum)': (0.85, 0.10, 0.05),
        'Homozygous kinase domain missense (N-lobe aa 1–130)': (0.45, 0.40, 0.15),
        'Compound het missense + truncating': (0.55, 0.30, 0.15),
        'Compound het two missense': (0.35, 0.40, 0.25),
        'Hypomorphic kinase-linker missense (aa 271–600)': (0.15, 0.40, 0.45),
    }
    tw = thorax_map[allele]
    thorax = rng.choices(['Severe (neonatal ventilation)', 'Moderate (CPAP/BiPAP)', 'Mild (nasal O2/none)'], weights=tw)[0]

    # Polydactyly (65–75% — highest among SRTD types)
    poly_prob = 0.70 if 'truncating' in allele.lower() else 0.62 if 'missense' in allele else 0.55
    poly      = rng.random() < poly_prob
    poly_type = rng.choices(
        ['Postaxial (hands+feet)', 'Both postaxial + preaxial', 'Preaxial only'],
        weights=[0.55, 0.35, 0.10]
    )[0] if poly else None

    # Hydrops fetalis (UNIQUE to SRTD6 — ~20%; lymphatic cilia fail)
    hydrops_prob = 0.40 if 'truncating' in allele.lower() else 0.12
    hydrops = rng.random() < hydrops_prob

    # Medianasal hypoplasia (Majewski feature — ~30%)
    mnh_prob = 0.60 if 'truncating' in allele.lower() else 0.18
    med_nasal = rng.random() < mnh_prob

    # Laryngeal stenosis (~15%)
    lary_prob = 0.35 if 'truncating' in allele.lower() else 0.08
    laryngeal = rng.random() < lary_prob

    # Renal (35–45%)
    renal_prob = 0.60 if 'truncating' in allele.lower() else 0.32
    renal_any  = rng.random() < renal_prob
    renal_type = rng.choices(
        ['Renal cysts (structural)', 'Tubulointerstitial nephritis (TIN)', 'ESRD requiring transplant'],
        weights=[0.50, 0.35, 0.15]
    )[0] if renal_any else None

    # Retinal (15–25%)
    ret_prob = 0.30 if 'truncating' in allele.lower() else 0.15
    retinal  = rng.random() < ret_prob

    # CHF / hepatic (10–15%)
    chf      = rng.random() < 0.13

    # VEPTR / MAGEC (almost all survivors with moderate/mild thorax)
    veptr = thorax in ['Moderate (CPAP/BiPAP)', 'Mild (nasal O2/none)'] and rng.random() < 0.88

    # Renal transplant (subset of ESRD)
    tx = renal_type == 'ESRD requiring transplant' and rng.random() < 0.75

    # Age at diagnosis
    age_dx_cat = rng.choices(
        ['0–1 yr (neonatal/prenatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.70, 0.18, 0.08, 0.04]
    )[0]

    # Misdiagnosis
    if rng.random() < 0.45:
        mis = rng.choices(
            ['SRTD3 / DYNC2H1 (no EM distinction at birth)', 'Ellis-van Creveld (polydactyly + short limbs)',
             'Hydrolethalus syndrome (hydrops + absent cilia)', 'Saldino-Noonan SRPS (polydactyly + skeletal)'],
            weights=[0.38, 0.28, 0.22, 0.12]
        )[0]
    else:
        mis = None

    # Perinatal death
    perinatal_death = allele == 'Biallelic truncating (null/null — SRPS II spectrum)' and rng.random() < 0.48

    patients.append({
        'id': i, 'sex': sex, 'eth': eth, 'allele': allele,
        'thorax': thorax, 'poly': poly, 'poly_type': poly_type,
        'hydrops': hydrops, 'med_nasal': med_nasal, 'laryngeal': laryngeal,
        'renal_any': renal_any, 'renal_type': renal_type,
        'retinal': retinal, 'chf': chf, 'veptr': veptr, 'tx': tx,
        'age_dx_cat': age_dx_cat, 'misdiagnosis': mis,
        'perinatal_death': perinatal_death,
    })

# ── aggregate statistics ──────────────────────────────────────────────────────
_poly_n    = sum(1 for p in patients if p['poly'])
_renal_n   = sum(1 for p in patients if p['renal_any'])
_retinal_n = sum(1 for p in patients if p['retinal'])
_chf_n     = sum(1 for p in patients if p['chf'])
_veptr_n   = sum(1 for p in patients if p['veptr'])
_tx_n      = sum(1 for p in patients if p['tx'])
_mis_n     = sum(1 for p in patients if p['misdiagnosis'])
_hydrops_n = sum(1 for p in patients if p['hydrops'])
_mnh_n     = sum(1 for p in patients if p['med_nasal'])
_lary_n    = sum(1 for p in patients if p['laryngeal'])
_thorax_severe_n = sum(1 for p in patients if 'Severe' in p['thorax'])
_perinatal_n     = sum(1 for p in patients if p['perinatal_death'])
_sex_M     = sum(1 for p in patients if p['sex'] == 'M')
_sex_F     = N - _sex_M

# ── API payloads ──────────────────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_n": N,
        "seed": SEED,
        "sex_split": {"M": _sex_M, "F": _sex_F},
        "kpis": {
            "thorax_severe_n":    _thorax_severe_n,
            "thorax_severe_pct":  _pct(_thorax_severe_n),
            "polydactyly_n":      _poly_n,
            "polydactyly_pct":    _pct(_poly_n),
            "hydrops_n":          _hydrops_n,
            "hydrops_pct":        _pct(_hydrops_n),
            "med_nasal_n":        _mnh_n,
            "med_nasal_pct":      _pct(_mnh_n),
            "renal_any_n":        _renal_n,
            "renal_any_pct":      _pct(_renal_n),
            "retinal_any_n":      _retinal_n,
            "retinal_any_pct":    _pct(_retinal_n),
            "hepatic_chf_n":      _chf_n,
            "hepatic_chf_pct":    _pct(_chf_n),
            "veptr_any_n":        _veptr_n,
            "veptr_any_pct":      _pct(_veptr_n),
            "transplant_done_n":  _tx_n,
            "misdiagnosis_n":     _mis_n,
            "misdiagnosis_pct":   _pct(_mis_n),
            "perinatal_death_n":  _perinatal_n,
            "perinatal_death_pct":_pct(_perinatal_n),
            "laryngeal_n":        _lary_n,
            "laryngeal_pct":      _pct(_lary_n),
        },
        "mechanism": (
            "NEK1 (NIMA-Related Kinase 1) is a serine/threonine kinase at the basal body / mother centriole. "
            "It is NOT an IFT complex subunit (not IFT-A, not IFT-B2) and NOT a dynein-2 motor subunit — it is the ONLY "
            "ciliogenesis kinase whose loss causes SRTD. NEK1 phosphorylates TTBK2 (Tau Tubulin Kinase 2) at distal "
            "appendages of the mother centriole — the master switch for CP110 removal and axoneme nucleation. "
            "Loss of NEK1 → TTBK2 not activated → CP110 persists at centriole tip → axoneme cannot nucleate → "
            "ABSENT or RUDIMENTARY cilia (no elongation). Absent cilia → Hedgehog (Ihh/Shh) "
            "signalling completely absent → GLI3R maximally accumulates in chondrocytes → severe NARROW THORAX. "
            "NEK1 also phosphorylates transition-zone scaffold proteins (CEP164, RPGRIP1L) and is required for "
            "lymphatic cilia assembly — explaining the UNIQUE fetal hydrops (~20%) seen only in SRTD6 among all SRTD types. "
            "Secondary dual role: NEK1 phosphorylates ATRIP/CHEK1 in DNA damage response; not a clinical phenotype in "
            "biallelic LOF patients but contributes to the functional rationale for kinase-activating therapeutic strategies."
        ),
        "key_distinction": (
            "SRTD6 (NEK1) occupies a UNIQUE fourth molecular class — distinct from IFT-B2 (shortened cilia: SRTD1/10/13), "
            "IFT-A (short stubby cilia: SRTD4/5/7/9), and Dynein-2 (club cilia: SRTD3/8/11/15/17). "
            "NEK1 loss causes ABSENT/RUDIMENTARY cilia — ciliogenesis fails before axoneme nucleation. "
            "Clinically distinguishing features: (1) ABSENT cilia on EM — not club, not short, not stubby; "
            "(2) Hydrops fetalis ~20% — UNIQUE to SRTD6 among all SRTD types (lymphatic cilia); "
            "(3) Medianasal hypoplasia ~30% — Majewski feature; absent in other SRTD types; "
            "(4) Laryngeal stenosis ~15% — airway compromise beyond thorax restriction; "
            "(5) HIGHEST polydactyly rate (65–75%) — postaxial + preaxial — in all SRTD types; "
            "(6) NEK1 is a kinase — NOT on early IFT-focussed SRTD panels; gene panel must include NEK1."
        ),
        "srtd_molecular_class_table": [
            {"class": "IFT-B2 (Anterograde distal)", "em": "SHORTENED cilia", "genes": "SRTD1 (IFT80), SRTD10 (IFT172), SRTD13 (CLUAP1)", "why": "Anterograde IFT truncated; cilia cannot extend; Hedgehog cargo undelivered"},
            {"class": "IFT-A (Anterograde adaptor)", "em": "SHORT STUBBY cilia", "genes": "SRTD4 (TTC21B), SRTD5 (WDR19), SRTD7 (WDR35), SRTD9 (IFT140)", "why": "IFT-B import at cilia base blocked; uniform stubby shortening"},
            {"class": "Dynein-2 (Retrograde motor)", "em": "CLUB / BULGING TIP cilia", "genes": "SRTD3 (DYNC2H1), SRTD8 (WDR60), SRTD11 (WDR34), SRTD15 (DYNC2LI1), SRTD17 (TCTEX1D2)", "why": "Retrograde IFT blocked; IFT-B stranded at ciliary tip → club shape"},
            {"class": "Basal Body Kinase (SRTD6)", "em": "ABSENT / RUDIMENTARY cilia", "genes": "SRTD6 (NEK1)", "why": "Ciliogenesis initiation fails; TTBK2 not activated; CP110 not removed; axoneme cannot nucleate"},
        ],
        "age_distribution": {
            "dx_0_1yr":   sum(1 for p in patients if '0–1' in p['age_dx_cat']),
            "dx_2_5yr":   sum(1 for p in patients if '2–5' in p['age_dx_cat']),
            "dx_6_10yr":  sum(1 for p in patients if '6–10' in p['age_dx_cat']),
            "dx_11_16yr": sum(1 for p in patients if '11–16' in p['age_dx_cat']),
        },
    }


def get_breakdown():
    from collections import Counter

    thorax_dist = Counter(p['thorax'] for p in patients)
    poly_dist   = Counter(p['poly_type'] for p in patients if p['poly'])
    renal_dist  = Counter(p['renal_type'] for p in patients if p['renal_any'])
    eth_dist    = Counter(p['eth'] for p in patients)
    allele_dist = Counter(p['allele'] for p in patients)
    pres_dist   = Counter(p['age_dx_cat'] for p in patients)
    mis_dist    = Counter(p['misdiagnosis'] for p in patients if p['misdiagnosis'])
    poly_absent = N - _poly_n

    veptr_types = Counter()
    for p in patients:
        if not p['veptr']:
            veptr_types['No surgical intervention'] += 1
        elif rng.random() < 0.55:
            veptr_types['VEPTR (expandable titanium rib)'] += 1
        else:
            veptr_types['MAGEC growing rod'] += 1

    return {
        "thorax_distribution": [{"label": k, "n": v} for k, v in sorted(thorax_dist.items(), key=lambda x: -x[1])],
        "polydactyly_distribution": [
            {"label": "Polydactyly present", "n": _poly_n},
            {"label": "No polydactyly", "n": poly_absent},
        ] + [{"label": k, "n": v} for k, v in sorted(poly_dist.items(), key=lambda x: -x[1])],
        "hydrops_medianasal": [
            {"label": "Hydrops fetalis (unique SRTD6)", "n": _hydrops_n},
            {"label": "Medianasal hypoplasia (Majewski)", "n": _mnh_n},
            {"label": "Laryngeal stenosis", "n": _lary_n},
            {"label": "Perinatal death (null/null)", "n": _perinatal_n},
        ],
        "renal_distribution": [{"label": k, "n": v} for k, v in sorted(renal_dist.items(), key=lambda x: -x[1])]
                              + [{"label": "No renal disease", "n": N - _renal_n}],
        "allele_class_summary": [{"label": k, "n": v} for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "ethnicity_distribution": [{"ethnicity": k, "n": v} for k, v in sorted(eth_dist.items(), key=lambda x: -x[1])],
        "presentation_distribution": [{"label": k, "n": v} for k, v in sorted(pres_dist.items(), key=lambda x: -x[1])],
        "misdiagnosis_distribution": [{"label": k, "n": v} for k, v in sorted(mis_dist.items(), key=lambda x: -x[1])],
        "veptr_distribution": [{"label": k, "n": v} for k, v in sorted(veptr_types.items(), key=lambda x: -x[1])],
        "top_variants": [
            {"variant": "p.Thr141Met (c.422C>T) — kinase N-lobe; European compound het; moderate", "n": 6},
            {"variant": "p.Arg271Gln (c.812G>A) — DFG activation loop; MENA homozygous; moderate-severe", "n": 8},
            {"variant": "p.Glu415Ter (c.1243G>T) — kinase domain truncating; SRPS II; pan-ethnic", "n": 7},
            {"variant": "p.Arg629Trp (c.1885C>T) — kinase-linker junction; South Asian hypomorphic; mild", "n": 5},
            {"variant": "p.Leu851Pro (c.2552T>C) — coiled-coil; Middle Eastern; severe", "n": 4},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":          "NEK1",
            "full_name":     "NIMA-Related Kinase 1 (Never In Mitosis A-Related Kinase 1)",
            "omim_gene":     "*604588",
            "chromosome":    "4q33",
            "size":          "~1258 amino acids",
            "protein_class": "Serine/threonine kinase — basal body / ciliogenesis kinase",
            "key_domains":   "Kinase domain (aa 1–270; N-lobe + C-lobe + DFG motif) · Coiled-coil (aa 601–1258; dimerization + TTBK2 binding)",
            "key_substrates":"TTBK2 (master cilia switch) · CEP164 (distal appendage scaffold) · RPGRIP1L (TZ) · ATRIP/CHEK1 (DNA damage)",
            "molecular_class":"Basal body kinase — NOT IFT-A, NOT IFT-B2, NOT Dynein-2 — DISTINCT FOURTH CLASS",
            "nima_family":   "NIMA family (same as NPHP9/NEK8, but distinct substrate and phenotype); NEK1 ≠ NEK8",
        },
        "disease_card": {
            "disease":       "Short-Rib Thoracic Dysplasia 6 (SRTD6 / ATD6)",
            "omim_disease":  "#263520",
            "also_known_as": "Majewski syndrome · SRPS type II · Jeune ATD6 (survivable moderate alleles)",
            "inheritance":   "Autosomal Recessive — biallelic LOF",
            "prevalence":    "~1:500,000–1,000,000 · ~30–60 families worldwide (2026)",
            "ciliary_em":    "ABSENT / RUDIMENTARY cilia (fourth EM class — unique among SRTD types)",
            "polydactyly":   "65–75% — HIGHEST rate of ALL SRTD genes; both postaxial + preaxial",
            "unique_features":"Hydrops fetalis ~20% (lymphatic cilia); medianasal hypoplasia ~30% (Majewski); laryngeal stenosis ~15%",
            "renal":         "35–45% — renal cysts + TIN; NO NPHP allele series",
            "retinal":       "15–25% — rod-cone dystrophy",
            "chf":           "10–15%",
            "perinatal_lethality":"40–50% (biallelic null — SRPS II spectrum)",
            "surgical_tx":   "VEPTR / MAGEC growing rods — same as all SRTDs; serial expansion",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: narrow thorax + polydactyly (postaxial + preaxial) + hydrops/ascites → suspect SRTD6/NEK1",
            "2. Post-natal: chest radiograph + skeletal survey → short ribs, thoracic diameter <2 SD below mean",
            "3. Confirm medianasal hypoplasia / laryngeal assessment (unique SRTD6 features)",
            "4. URGENT: extended ciliopathy gene panel including NEK1 (must not be IFT-only panel)",
            "5. If panel negative: WES/WGS — NEK1 is large gene (1258 aa); deep intronic/structural variants missed on panels",
            "6. Ciliary EM (nasal brush/bronchial): expect ABSENT or rudimentary cilia stubs (NOT club, NOT stubby, NOT shortened)",
            "7. Renal USS (cysts vs TIN); annual GFR/creatinine; consider renal biopsy if proteinuria early",
            "8. Ophthalmology: ERG from age 4 (rod-cone 15–25%); low vision support",
            "9. Cardiac echo: CHD 10–15%; CHD NOT a Majewski-defining feature but must be excluded",
            "10. Liver: APRI + USS annually (CHF 10–15%); hepatology if abnormal",
            "11. Laryngoscopy: if stridor beyond expected thoracic restriction — laryngeal stenosis 15%",
            "12. Genetics: 25% AR recurrence risk; cascade testing; prenatal diagnosis / PGT available",
            "13. Therapeutic: NEK1 kinase activators in pre-clinical pipeline (2026) — no approved agent",
        ],
        "mechanism_glossary": [
            {"term": "NEK1 (NIMA-Related Kinase 1)", "definition": "Serine/threonine kinase at the basal body. Phosphorylates TTBK2 to initiate ciliogenesis. Also involved in DNA damage checkpoint (ATRIP/CHEK1). Loss → cilia cannot form (ABSENT/RUDIMENTARY). Fourth and distinct SRTD molecular class beyond IFT-B2, IFT-A, Dynein-2."},
            {"term": "TTBK2 (Tau Tubulin Kinase 2)", "definition": "NEK1 substrate at the mother centriole distal appendage. When NEK1 activates TTBK2, TTBK2 phosphorylates CEP164 and removes CP110 from the centriole tip — the master switch that allows axoneme nucleation. NEK1 loss → TTBK2 inactive → CP110 not removed → no cilia."},
            {"term": "CP110 (Centriolar Coiled-Coil Protein 110)", "definition": "Cap protein at the mother centriole tip that must be removed (by TTBK2 downstream of NEK1) to allow axoneme nucleation. Persistence of CP110 → blocked ciliogenesis → absent cilia in SRTD6."},
            {"term": "ABSENT / RUDIMENTARY cilia (SRTD6 EM signature)", "definition": "Unique fourth EM class among SRTD types. Contrast: Dynein-2 SRTDs → club/bulging tip (IFT-B stranded); IFT-A SRTDs → short stubby (IFT-B base pile-up); IFT-B2 SRTDs → shortened (anterograde truncated). In SRTD6/NEK1: axoneme nucleation fails entirely; only rudimentary stubs or no cilia at all."},
            {"term": "Hydrops fetalis (SRTD6-unique)", "definition": "~20% of SRTD6 patients — unique among all SRTD types. NEK1 is required for lymphatic cilia assembly; absent lymphatic cilia → impaired fetal lymph drainage → fetal hydrops/ascites. Not caused by cardiac failure (CHD rate not elevated above background)."},
            {"term": "Medianasal hypoplasia (Majewski feature)", "definition": "~30% of SRTD6 patients — the 'Majewski' feature. Midline nasal hypoplasia; sometimes median cleft between nasal wings. Absent in all other SRTD types. Originally characterised in Majewski syndrome (SRPS type II = NEK1) as a defining feature distinguishing it from Saldino-Noonan (SRPS type I)."},
            {"term": "Hedgehog signalling failure (SRTD6)", "definition": "ABSENT cilia in SRTD6 → no cilia platform for Hedgehog pathway → complete absence of GLI processing → GLI3R maximally accumulates → Ihh/Shh signalling completely absent in chondrocytes → most severe Hedgehog signalling failure of all SRTD types (no cilia at all, vs shortened/clubbed in others)."},
            {"term": "NIMA kinase family (NEK1 vs NEK8)", "definition": "NEK1 and NEK8 (NPHP9) are both NIMA-related kinases but with distinct roles: NEK8/NPHP9 → inversin compartment; phosphorylates BICC1/Anks6; causes nephronophthisis with situs inversus, CHF, no thoracic dysplasia. NEK1 → basal body kinase; phosphorylates TTBK2/CEP164; causes SRTD6 with narrow thorax, polydactyly, hydrops. Different substrates, different ciliopathy spectra."},
        ],
        "key_variants": [
            {"variant": "p.Thr141Met", "domain": "Kinase N-lobe (aa 1–130)", "consequence": "Reduced ATP binding; partial kinase activity; moderate SRTD6", "ethnicity": "European compound het"},
            {"variant": "p.Arg271Gln", "domain": "DFG activation loop (aa 131–270)", "consequence": "DFG motif disrupted; near-complete kinase abolition; moderate-severe", "ethnicity": "MENA homozygous"},
            {"variant": "p.Glu415Ter", "domain": "Kinase domain (truncating)", "consequence": "Null allele; complete LOF; SRPS II perinatal spectrum", "ethnicity": "Pan-ethnic"},
            {"variant": "p.Arg629Trp", "domain": "Kinase-linker (aa 271–600)", "consequence": "Partial TTBK2 phosphorylation retained; mild SRTD6; adult survivors", "ethnicity": "South Asian hypomorphic"},
            {"variant": "p.Leu851Pro", "domain": "Coiled-coil (aa 601–900)", "consequence": "Dimerisation disrupted; basal body mislocalisation; severe", "ethnicity": "Middle Eastern homozygous"},
        ],
        "treatment_summary": [
            "1. Narrow thorax (primary): VEPTR or MAGEC growing rod — serial thoracic expansion; first-line surgical; same protocol as all SRTDs",
            "2. Neonatal respiratory: mechanical ventilation (null alleles; severe); CPAP/BiPAP (moderate); wean as thorax expands post-VEPTR",
            "3. Laryngeal stenosis: endoscopic assessment + dilation if severe; tracheostomy rarely required; ENT mandatory",
            "4. Hydrops/fetal: antenatal ECHO + Doppler; postnatal diuretics; no specific fetal intervention 2026; delivery planning",
            "5. Renal: annual GFR/USS/creatinine; ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous; no recurrence); NO NPHP allele counselling (NEK1 has no renal-only allele series)",
            "6. Retinal: annual ERG from age 4; ophthalmology; low vision support (15–25%); standard surveillance",
            "7. Hepatic: APRI + USS annually; hepatology if CHF suspected; avoid hepatotoxics",
            "8. Gene panel: ensure NEK1 included (NOT on older IFT-only SRTD panels); cascade testing; 25% AR recurrence; PGT available",
            "9. Therapeutic pipeline: NEK1 kinase activator / mRNA therapy (pre-clinical 2026); no approved agent; enrol in registry",
        ],
        "ddx_table": [
            {"disease": "SRTD3 (DYNC2H1) — most common SRTD", "key_difference": "DYNC2H1: CLUB/BULGING TIP cilia (IFT-B stranded at tip); no hydrops; no medianasal hypoplasia; gene panel differentiates"},
            {"disease": "SRTD5 (WDR19) — most common IFT-A SRTD", "key_difference": "WDR19: SHORT STUBBY cilia; ectodermal features (sparse hair, hypodontia) ABSENT in SRTD6; gene panel differentiates"},
            {"disease": "Hydrolethalus syndrome (HYLS1)", "key_difference": "HYLS1: also absent/rudimentary cilia (basal body); but severe acrania/anencephaly, brain malformations — ABSENT in SRTD6; lethal vs SRTD6 survivors"},
            {"disease": "Ellis-van Creveld (EVC/EVC2)", "key_difference": "EVC: CHD (ASD/VSD) in 60% — NOT a feature of SRTD6; EVC has SHH-pathway gain-of-function mechanism; ectodermal features (EVC) absent SRTD6"},
            {"disease": "Saldino-Noonan SRPS (SRPS type I) — various dynein-2 genes", "key_difference": "SRPS type I: caused by dynein-2 subunit genes (DYNC2H1 null etc.); club cilia; no medianasal hypoplasia; no hydrops; gene panel mandatory"},
            {"disease": "NPHP9 (NEK8) — same NIMA kinase family", "key_difference": "NEK8: inversin compartment kinase; causes nephronophthisis with situs inversus + CHF — NO narrow thorax, NO polydactyly; completely different substrates (BICC1 vs TTBK2)"},
        ],
    }
