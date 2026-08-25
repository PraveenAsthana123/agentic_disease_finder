#!/usr/bin/env python3
"""GA2 (Glutaric Acidemia Type II) / MADD (Multiple Acyl-CoA Dehydrogenase Deficiency) Dashboard.

ETFA / ETFB / ETFDH genes encode the Electron Transfer Flavoprotein system:
  ETF (ETFA + ETFB heterodimer):
    - ETFA: 333 aa; 15q23–q25; FAD-binding alpha subunit; mitochondrial matrix
    - ETFB: 255 aa; 19q13.41; AMP-binding beta subunit; mitochondrial matrix
    - ETF heterodimer accepts electrons (as FADH2) from ALL acyl-CoA dehydrogenases:
      SCAD, MCAD, VLCAD, LCAD, ACADSB, IVD, GCD (glutaryl-CoA DH), DHODH, DMGDH, SARDH

  ETFDH (ETF:Ubiquinone Oxidoreductase):
    - 617 aa; 4q32.1; FAD + Fe/S cluster + CoQ10 binding; inner mitochondrial membrane
    - Transfers electrons from ETF-FADH2 → ubiquinone (CoQ10) → respiratory chain Complex III

  ETF/ETFDH = THE COMMON ELECTRON CONDUIT for ALL mitochondrial acyl-CoA dehydrogenases:
    Every acyl-CoA dehydrogenase step (short/medium/very-long/long/isovaleryl/glutaryl)
    requires ETF → ETFDH → CoQ10 to re-oxidise the enzyme FAD cofactor.
    LOSS OF ETF OR ETFDH → ALL acyl-CoA dehydrogenases SIMULTANEOUSLY BLOCKED.

GA2 METABOLIC BLOCK — PAN-ACYLCARNITINEMIA:
  ETFA/ETFB/ETFDH LOF → ETF cannot accept FADH2 → ALL dehydrogenases accumulate
  their acyl-CoA substrates → ALL acylcarnitines accumulate:
    C4  (butyrylcarnitine)         — from SCAD block
    C5  (isovalerylcarnitine)      — from IVD block [C5 = KEY GA2 vs isolated IVD/SCAD]
    C8  (octanoylcarnitine)        — from MCAD block
    C10 (decanoylcarnitine)        — from MCAD block (C10)
    C12 (dodecanoylcarnitine)      — from LCAD/MCAD block
    C14:1 (tetradecenoylcarnitine) — from VLCAD block
    C16 (palmitoylcarnitine)       — from VLCAD/LCAD block
    C5-DC (glutarylcarnitine)      — from GCD block [GA2 vs GA1 distinction]
  → "PAN-ACYLCARNITINEMIA" = DIAGNOSTIC PATTERN on tandem MS/MS NBS

URINE ORGANIC ACIDS (GA2):
  Ethylmalonic acid (EMA)           ↑↑ (MUCH higher than in SCAD — often >300 mmol/mol Cr)
  Glutaric acid (C5-DC)             ↑ (also elevated in GA1 — distinguish by NBS acylcarnitines)
  2-Hydroxyglutaric acid            ↑ (minor)
  Adipic acid (C6-dicarboxylic)     ↑
  Suberic acid (C8-dicarboxylic)    ↑
  Sebacic acid (C10-dicarboxylic)   ↑
  5-Hydroxyhexanoic acid            ↑
  Isovalerylglycine                 ↑ (IVD block component)
  Ethylmalonic acid > 300 mmol/mol Cr + pan-acylcarnitinemia = diagnostic for severe GA2

CLINICAL TYPES (3 types by severity + gene):
  TYPE I   (ETFA or ETFB; neonatal severe; WITH congenital anomalies):
    - Polycystic kidneys, facial dysmorphism, brain malformations, hepatomegaly
    - Overwhelming metabolic acidosis, hypoglycaemia, hyperammonemia at birth
    - Death in first days–weeks of life; not amenable to treatment
    - MOST SEVERE; no riboflavin response

  TYPE II  (ETFA or ETFB; neonatal severe; WITHOUT congenital anomalies):
    - Severe hypoketotic hypoglycaemia + metabolic acidosis in neonatal period
    - Hepatomegaly; cardiomyopathy common
    - Better prognosis than Type I if metabolic crisis survived; variable riboflavin response
    - ETFA p.Arg191Cys → some riboflavin response possible

  TYPE III (ETFB or ETFDH; mild / late-onset; RIBOFLAVIN-RESPONSIVE MADD = RR-MADD):
    - Onset: infancy to adult (median presentation: young adulthood)
    - Proximal myopathy, muscle weakness, exercise intolerance, fatigue
    - Episodes of metabolic decompensation (hypoglycaemia, vomiting, metabolic acidosis)
    - RIBOFLAVIN 100–300 mg/day → DRAMATIC RESPONSE (hallmark of RR-MADD):
        Acylcarnitines NORMALISE within weeks
        EMA NORMALISES within weeks
        Myopathy IMPROVES/RESOLVES within months
    - ETFDH mutations: p.Arg191Trp (Asian founder ~50% East Asian), p.Arg191Cys (European),
      p.Pro456Leu (Northern European), p.Asp128Asn, p.Gly116Arg
    - Majority of GA2 cases identified in modern era are RR-MADD (Type III ETFDH)

KEY CLINICAL FACTS (HIGHEST YIELD):
  1. PAN-ACYLCARNITINEMIA = C4+C5+C8+C10+C12+C14:1+C16+C5-DC ALL elevated = DIAGNOSTIC
  2. EMA = ethylmalonic acid; MUCH higher than in SCAD (>300 vs <50 mmol/mol Cr in SCAD)
  3. TYPE III (RR-MADD) = ETFDH mutations; RIBOFLAVIN 100–300 mg/day → DRAMATIC response
  4. VPA = ABSOLUTE CONTRAINDICATION in ALL types (blocks ETF pathway + respiratory chain)
  5. KD = ABSOLUTE CONTRAINDICATION (ALL fat oxidation is blocked)
  6. C5 elevated — GA2 KEY POSITIVE vs SCAD (SCAD has C4 only; GA2 has C4+C5+C8...)
  7. C8 ELEVATED in GA2 (unlike SCAD where C8 normal) — GA2 vs SCAD key difference
  8. Riboflavin INDICATED in Type III/RR-MADD — unlike SCAD (riboflavin Level B, less response)
  9. CoQ10 supplementation Level B in ETFDH mutations (ETFDH binds CoQ10)
 10. Congenital anomalies (polycystic kidneys) = Type I (ETFA/ETFB null)
 11. Glutaric acid in urine — also in GA1; distinguish: GA2 pan-acylcarnitinemia vs GA1 C5-DC isolated
 12. p.Arg191Trp = East Asian founder ETFDH variant (50% of RR-MADD cases in Asian populations)
 13. ETFDH mutations in adult-onset myopathy — riboflavin trial mandatory before muscle biopsy
 14. Emergency: IV glucose + riboflavin + avoid fat loading (Type II/III decompensation)
 15. FADH2 generated by acyl-CoA dehydrogenases CANNOT reach ubiquinone → energy failure

OMIM Disease: #231680 (Glutaric Acidemia Type II)
OMIM Genes:   ETFA *608053 (15q23–q25) · ETFB *130410 (19q13.41) · ETFDH *231675 (4q32.1)
Inheritance:  Autosomal Recessive (AR), biallelic LOF of ETFA or ETFB or ETFDH
Prevalence:   ~1:100,000–250,000 (combined all types); RR-MADD (Type III) is most common

FATTY ACID BETA-OXIDATION — WHERE ETF/ETFDH FIT:
  Step 1 (all chain lengths): Acyl-CoA → trans-2-Enoyl-CoA + FADH2    [via SCAD/MCAD/VLCAD/IVD/GCD]
                              FADH2 → ETF(FADH2) → ETFDH → CoQ10      [ELECTRON CONDUIT ← GA2 block]
  Step 2: trans-2-Enoyl-CoA → L-3-Hydroxyacyl-CoA                     [via ECHS1/HADHA]
  Step 3: L-3-Hydroxyacyl-CoA → 3-Ketoacyl-CoA + NADH                 [via HADH/HADHA]
  Step 4: 3-Ketoacyl-CoA → Acyl-CoA(–2C) + Acetyl-CoA                 [via HADHB/ACAT1]

  GA2 block: ETFA/ETFB/ETFDH LOF → Step 1 FADH2 CANNOT be reoxidised
  → ALL acyl-CoA dehydrogenases BACK-INHIBITED (cofactor FAD NOT regenerated)
  → ALL chain-length beta-oxidation BLOCKED simultaneously
  → Pan-acylcarnitinemia; pan-dicarboxylic aciduria
"""

import random

SEED = 277
random.seed(SEED)

# ── Variant table (ETFA / ETFB / ETFDH) ─────────────────────────────────────
VARIANTS = [
    {"variant": "ETFDH p.Arg191Trp (c.571C>T)", "freq": 30, "gene": "ETFDH", "domain": "FAD-binding domain",
     "type": "III (RR-MADD)", "phenotype": "Adult myopathy, riboflavin-responsive",
     "note": "East Asian founder variant; ~50% of Asian RR-MADD; p.Arg191 critical for FAD binding"},
    {"variant": "ETFDH p.Arg191Cys (c.571C>T)", "freq": 12, "gene": "ETFDH", "domain": "FAD-binding domain",
     "type": "III (RR-MADD)", "phenotype": "Childhood/adult myopathy; moderate riboflavin response",
     "note": "European; allelic at same position as p.Arg191Trp — different amino acid change"},
    {"variant": "ETFDH p.Pro456Leu (c.1367C>T)", "freq": 8, "gene": "ETFDH", "domain": "Fe/S cluster region",
     "type": "III (RR-MADD)", "phenotype": "Adult myopathy; riboflavin-responsive",
     "note": "Northern European; affects Fe/S cluster coordination → electron transfer impaired"},
    {"variant": "ETFDH p.Asp128Asn (c.382G>A)", "freq": 6, "gene": "ETFDH", "domain": "FAD binding",
     "type": "III (RR-MADD)", "phenotype": "Variable onset myopathy; riboflavin-responsive",
     "note": "Disrupts FAD-binding pocket geometry; partial residual activity retained"},
    {"variant": "ETFDH p.Gly116Arg (c.346G>A)", "freq": 5, "gene": "ETFDH", "domain": "FAD binding core",
     "type": "III (RR-MADD)", "phenotype": "Childhood onset; riboflavin-responsive",
     "note": "Core FAD-binding fold disrupted; residual CoQ10 binding preserved"},
    {"variant": "ETFA p.Arg191Cys (c.571C>T)", "freq": 8, "gene": "ETFA", "domain": "FAD-binding domain",
     "type": "II–III", "phenotype": "Neonatal or late-onset; partial riboflavin response",
     "note": "ETFA allelic; some ETFA missense respond partially to riboflavin"},
    {"variant": "ETFA c.IVS7+1G>A (splice-site)", "freq": 7, "gene": "ETFA", "domain": "Splice junction",
     "type": "I–II", "phenotype": "Neonatal severe; no riboflavin response",
     "note": "Null splice; absent ETF alpha subunit → no ETF heterodimer formation"},
    {"variant": "ETFB p.Gly116Arg", "freq": 5, "gene": "ETFB", "domain": "AMP-binding domain",
     "type": "II", "phenotype": "Neonatal without congenital anomalies; poor response",
     "note": "AMP-binding disrupted → ETF heterodimer destabilised"},
    {"variant": "ETFA p.His314Arg", "freq": 4, "gene": "ETFA", "domain": "FAD interface",
     "type": "I", "phenotype": "Neonatal severe with congenital anomalies; lethal",
     "note": "FAD interface disrupted; Type I with polycystic kidneys documented"},
    {"variant": "ETFB Exon 5 deletion", "freq": 3, "gene": "ETFB", "domain": "Full exon deletion",
     "type": "I", "phenotype": "Neonatal severe with congenital anomalies",
     "note": "Null allele; complete absence ETFB → no ETF heterodimer"},
    {"variant": "ETFDH p.Arg128Gln", "freq": 7, "gene": "ETFDH", "domain": "FAD binding",
     "type": "III", "phenotype": "Adult onset; riboflavin-responsive; exercise intolerance",
     "note": "Partial residual ETFDH activity; riboflavin restores near-normal function"},
    {"variant": "Other compound het", "freq": 5, "gene": "ETFA/ETFB/ETFDH", "domain": "Various",
     "type": "II–III", "phenotype": "Variable",
     "note": "Compound heterozygous combinations; phenotype depends on residual activity"},
]

# ── Phenotype distribution ────────────────────────────────────────────────────
PHENOTYPE_DIST = {
    "Type III (RR-MADD / ETFDH) — adult myopathy":       16,
    "Type III (RR-MADD / ETFDH) — childhood onset":        9,
    "Type II (neonatal severe, no anomalies)":              7,
    "Type I (neonatal severe, congenital anomalies)":       5,
    "Type III (partial riboflavin response)":               3,
}


def _make_patient(i):
    """Generate a synthetic GA2/MADD patient record."""
    rng = random.Random(SEED + i * 37)

    # Assign phenotype
    if i < 16:
        ph = "Type III (RR-MADD / ETFDH)"
        gene = "ETFDH"
        variant = rng.choice([
            "p.Arg191Trp/p.Arg191Trp", "p.Arg191Trp/p.Pro456Leu",
            "p.Arg191Cys/p.Arg191Trp", "p.Asp128Asn/p.Gly116Arg",
            "p.Arg128Gln/p.Pro456Leu",
        ])
        rr_madd = True
        riboflavin_resp = True
        onset_age = rng.randint(15, 45)
        c4 = round(rng.uniform(1.5, 6.0), 1)
        c5 = round(rng.uniform(1.0, 4.0), 1)
        c8 = round(rng.uniform(0.8, 3.5), 1)
        c16 = round(rng.uniform(4.0, 14.0), 1)
        ema = round(rng.uniform(80, 600), 0)
        ga = round(rng.uniform(20, 200), 0)
        glucose = round(rng.uniform(2.0, 4.5), 1)
        cardiomyopathy = False
        congen_anomalies = False
        crisis = rng.random() < 0.3
        seizures = rng.random() < 0.15
        myopathy = True
    elif i < 25:
        ph = "Type III ETFDH — childhood"
        gene = "ETFDH"
        variant = rng.choice(["p.Arg191Cys/p.Arg191Cys", "p.Arg191Trp/p.Gly116Arg", "p.Arg191Cys/p.Asp128Asn"])
        rr_madd = True
        riboflavin_resp = True
        onset_age = rng.randint(2, 14)
        c4 = round(rng.uniform(2.0, 8.0), 1)
        c5 = round(rng.uniform(1.5, 5.5), 1)
        c8 = round(rng.uniform(1.0, 5.0), 1)
        c16 = round(rng.uniform(5.0, 18.0), 1)
        ema = round(rng.uniform(150, 800), 0)
        ga = round(rng.uniform(40, 300), 0)
        glucose = round(rng.uniform(1.8, 4.0), 1)
        cardiomyopathy = rng.random() < 0.2
        congen_anomalies = False
        crisis = rng.random() < 0.5
        seizures = rng.random() < 0.2
        myopathy = True
    elif i < 32:
        ph = "Type II neonatal (no anomalies)"
        gene = rng.choice(["ETFA", "ETFB"])
        variant = rng.choice([
            "ETFA p.Arg191Cys/p.Arg191Cys",
            "ETFB p.Gly116Arg/c.IVS7+1G>A",
            "ETFA p.Arg191Cys/splice",
            "ETFB Exon5del/p.Gly116Arg",
        ])
        rr_madd = False
        riboflavin_resp = False
        onset_age = 0
        c4 = round(rng.uniform(3.0, 12.0), 1)
        c5 = round(rng.uniform(2.5, 8.0), 1)
        c8 = round(rng.uniform(2.0, 8.0), 1)
        c16 = round(rng.uniform(8.0, 25.0), 1)
        ema = round(rng.uniform(400, 2000), 0)
        ga = round(rng.uniform(100, 600), 0)
        glucose = round(rng.uniform(0.5, 2.5), 1)
        cardiomyopathy = rng.random() < 0.6
        congen_anomalies = False
        crisis = True
        seizures = rng.random() < 0.4
        myopathy = False
    else:
        ph = "Type I neonatal (with congenital anomalies)"
        gene = rng.choice(["ETFA", "ETFB"])
        variant = rng.choice([
            "ETFA p.His314Arg/splice",
            "ETFB Exon5del/null",
            "ETFA null/null",
        ])
        rr_madd = False
        riboflavin_resp = False
        onset_age = 0
        c4 = round(rng.uniform(5.0, 18.0), 1)
        c5 = round(rng.uniform(4.0, 12.0), 1)
        c8 = round(rng.uniform(3.0, 10.0), 1)
        c16 = round(rng.uniform(10.0, 35.0), 1)
        ema = round(rng.uniform(800, 3000), 0)
        ga = round(rng.uniform(200, 1200), 0)
        glucose = round(rng.uniform(0.2, 1.5), 1)
        cardiomyopathy = rng.random() < 0.8
        congen_anomalies = True
        crisis = True
        seizures = rng.random() < 0.5
        myopathy = False

    return {
        "id":              f"GA2-{SEED}-{i+1:02d}",
        "phenotype":       ph,
        "gene":            gene,
        "variant":         variant,
        "onset_age":       onset_age,
        "c4_umol":         c4,
        "c5_umol":         c5,
        "c8_umol":         c8,
        "c16_umol":        c16,
        "ema_mmol_cr":     ema,
        "glutaric_mmol":   ga,
        "glucose_mmol":    glucose,
        "cardiomyopathy":  cardiomyopathy,
        "congen_anomalies": congen_anomalies,
        "myopathy":        myopathy,
        "seizures":        seizures,
        "metabolic_crisis": crisis,
        "rr_madd":         rr_madd,
        "riboflavin_resp": riboflavin_resp,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)
    type3   = sum(1 for p in PATIENTS if "III" in p["phenotype"])
    type2   = sum(1 for p in PATIENTS if "Type II" in p["phenotype"])
    type1   = sum(1 for p in PATIENTS if "Type I" in p["phenotype"])
    rr_madd = sum(1 for p in PATIENTS if p["rr_madd"])
    seizures_n = sum(1 for p in PATIENTS if p["seizures"])
    cardio_n   = sum(1 for p in PATIENTS if p["cardiomyopathy"])
    congen_n   = sum(1 for p in PATIENTS if p["congen_anomalies"])
    crisis_n   = sum(1 for p in PATIENTS if p["metabolic_crisis"])
    myopathy_n = sum(1 for p in PATIENTS if p["myopathy"])
    riboflavin_n = sum(1 for p in PATIENTS if p["riboflavin_resp"])

    avg_c4  = round(sum(p["c4_umol"]  for p in PATIENTS) / n, 2)
    avg_c5  = round(sum(p["c5_umol"]  for p in PATIENTS) / n, 2)
    avg_c8  = round(sum(p["c8_umol"]  for p in PATIENTS) / n, 2)
    avg_ema = round(sum(p["ema_mmol_cr"] for p in PATIENTS) / n, 0)

    return {
        "n_patients": n,
        "seed": SEED,
        "gene": "ETFA (15q23–q25) / ETFB (19q13.41) / ETFDH (4q32.1)",
        "locus": "ETFA: 15q23–q25 · ETFB: 19q13.41 · ETFDH: 4q32.1",
        "protein": (
            "ETFA — 333 aa ETF alpha subunit (FAD-binding); "
            "ETFB — 255 aa ETF beta subunit (AMP-binding); "
            "ETFDH — 617 aa ETF:Ubiquinone Oxidoreductase (FAD + Fe/S + CoQ10)"
        ),
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF of ETFA or ETFB or ETFDH",
        "omim_gene": "ETFA *608053 / ETFB *130410 / ETFDH *231675",
        "omim_disease": "#231680",
        "prevalence": "~1:100,000–250,000 (all types combined); RR-MADD (Type III ETFDH) most common in modern era",
        "primary_nbs_marker": (
            "PAN-ACYLCARNITINEMIA: C4↑ + C5↑ + C8↑ + C10↑ + C12↑ + C14:1↑ + C16↑ + C5-DC↑ "
            "(ALL chain-length acylcarnitines elevated simultaneously — diagnostic pattern)"
        ),
        "urine_oa_hallmarks": [
            "Ethylmalonic acid (EMA) ↑↑ — MUCH higher than SCAD (often >300 mmol/mol Cr)",
            "Glutaric acid ↑ — also in GA1; distinguish by pan-acylcarnitinemia",
            "2-Hydroxyglutaric acid ↑",
            "Adipic acid ↑",
            "Suberic acid ↑",
            "Sebacic acid ↑",
            "Isovalerylglycine ↑ (IVD block component)",
            "5-Hydroxyhexanoic acid ↑",
        ],
        "rr_madd_key": (
            "RIBOFLAVIN-RESPONSIVE MADD (RR-MADD) = Type III ETFDH mutations. "
            "Riboflavin 100–300 mg/day → DRAMATIC normalisation of acylcarnitines + EMA + myopathy "
            "(response within days–weeks; hallmark of ETFDH deficiency)"
        ),
        "key_negatives": [
            "C8 ELEVATED (unlike SCAD where C8 normal) — pan-acylcarnitinemia distinguishes GA2 from SCAD",
            "C5 ELEVATED (unlike SCAD where C5 normal) — IVD block component in GA2",
            "EMA >> SCAD levels (>300 vs <50 mmol/mol Cr) — key quantitative distinction",
            "No isolated C8+HG+SG (unlike MCAD with pathognomonic glycine conjugates)",
            "Glutaric acid in urine (also GA1) — pan-acylcarnitinemia distinguishes GA2 from GA1",
        ],
        "absolute_ci": ["VPA (blocks ETF pathway + mitochondrial respiration — ALL types)", "KD (ALL fat oxidation BLOCKED)"],
        "first_line_treatment": (
            "Type III/RR-MADD: Riboflavin 100–300 mg/day (Level A — DRAMATIC response). "
            "All types: Avoid fasting; emergency IV glucose + riboflavin; L-carnitine if C0 depleted."
        ),
        "clinical_summary": (
            f"GA2/MADD (Glutaric Acidemia Type II) is caused by deficiency of the ETF electron-conduit system "
            f"(ETFA/ETFB heterodimer) or ETFDH (ETF:ubiquinone oxidoreductase), which is required for ALL "
            f"mitochondrial acyl-CoA dehydrogenases to re-oxidise their FAD cofactor. This simultaneously "
            f"blocks SCAD (C4), MCAD (C8), VLCAD (C14:1), IVD (C5), and glutaryl-CoA DH (C5-DC) → "
            f"pan-acylcarnitinemia on NBS. Three clinical types: Type I (ETFA/ETFB, neonatal + congenital "
            f"anomalies; n={type1}), Type II (ETFA/ETFB, neonatal without anomalies; n={type2}), Type III "
            f"(ETFDH, mild/late-onset RR-MADD; n={type3}). The majority in the modern era are Type III "
            f"(RR-MADD), defined by DRAMATIC riboflavin 100–300 mg/day response (n={rr_madd} riboflavin-"
            f"responsive). Seizures n={seizures_n}; metabolic crisis n={crisis_n}; "
            f"cardiomyopathy n={cardio_n}; congenital anomalies n={congen_n}. "
            f"VPA = ABSOLUTE CI (all types). KD = ABSOLUTE CI (all fat oxidation blocked)."
        ),
        "kpis": {
            "total_patients":   n,
            "type3_rr_madd":    type3,
            "type2_n":          type2,
            "type1_n":          type1,
            "rr_madd_n":        rr_madd,
            "riboflavin_resp_n": riboflavin_n,
            "seizures_n":       seizures_n,
            "cardiomyopathy_n": cardio_n,
            "congen_anomalies_n": congen_n,
            "crisis_n":         crisis_n,
            "myopathy_n":       myopathy_n,
            "avg_c4_umol":      avg_c4,
            "avg_c5_umol":      avg_c5,
            "avg_c8_umol":      avg_c8,
            "avg_ema_mmol_cr":  avg_ema,
        },
        "phenotype_distribution": {k: v for k, v in PHENOTYPE_DIST.items()},
    }


def get_breakdown():
    sample = PATIENTS[:10]

    biomarkers = {
        "C4_butyrylcarnitine": {
            "label":     "C4 (Butyrylcarnitine) — SCAD block component",
            "direction": "ELEVATED",
            "normal":    "<0.5 µmol/L",
            "status":    "↑↑ (ELEVATED — SCAD arm blocked)",
            "color":     "danger",
            "rationale": (
                "C4 elevation from SCAD being blocked. In GA2, C4 is MUCH higher than in isolated SCAD "
                "and occurs ALONGSIDE C5, C8, C14:1, C16 elevation (pan-acylcarnitinemia). "
                "Isolated C4 elevation (without C5/C8 co-elevation) argues for SCAD, not GA2."
            ),
        },
        "C5_isovalerylcarnitine": {
            "label":     "C5 (Isovalerylcarnitine) — IVD block component [KEY POSITIVE vs SCAD]",
            "direction": "ELEVATED",
            "normal":    "<0.3 µmol/L",
            "status":    "↑↑ (ELEVATED — IVD arm blocked)",
            "color":     "danger",
            "rationale": (
                "C5 elevation is the KEY POSITIVE differentiating GA2 from SCAD. "
                "IVD (isovaleryl-CoA dehydrogenase) requires ETF for FADH2 re-oxidation. "
                "SCAD does NOT elevate C5 (SCAD only elevates C4). GA2 elevates BOTH C4 AND C5."
            ),
        },
        "C8_octanoylcarnitine": {
            "label":     "C8 (Octanoylcarnitine) — MCAD block component [KEY POSITIVE vs SCAD]",
            "direction": "ELEVATED",
            "normal":    "<0.3 µmol/L",
            "status":    "↑↑ (ELEVATED — MCAD arm blocked)",
            "color":     "danger",
            "rationale": (
                "C8 elevation is a KEY POSITIVE differentiating GA2 from SCAD. "
                "SCAD does NOT elevate C8 (SCAD = C4 only). GA2 elevates C8 (MCAD arm blocked). "
                "Presence of C8 alongside C4 strongly argues for GA2 over isolated SCAD deficiency."
            ),
        },
        "C16_palmitoylcarnitine": {
            "label":     "C16 (Palmitoylcarnitine) — VLCAD block component",
            "direction": "ELEVATED",
            "normal":    "<2.0 µmol/L",
            "status":    "↑↑ (ELEVATED — VLCAD arm blocked)",
            "color":     "danger",
            "rationale": (
                "C16 elevation from the VLCAD arm being blocked. Long-chain acylcarnitines "
                "(C14:1, C16, C18:1) elevated in GA2 confirms all chain-length dehydrogenases are affected. "
                "Long-chain acylcarnitines are NOT elevated in SCAD or MCAD."
            ),
        },
        "EMA_ethylmalonic_acid": {
            "label":     "EMA (Ethylmalonic Acid) — MUCH higher than SCAD [KEY QUANTITATIVE DIFF]",
            "direction": "ELEVATED ↑↑↑",
            "normal":    "<25 mmol/mol Cr",
            "status":    "MARKEDLY ELEVATED (often >300–3000 mmol/mol Cr in Type I/II)",
            "color":     "danger",
            "rationale": (
                "EMA is elevated in BOTH SCAD and GA2. However, EMA in GA2 is MUCH higher "
                "(often >300 mmol/mol Cr, up to 3000 in neonatal severe) vs SCAD (<50 mmol/mol Cr). "
                "Quantitative EMA measurement is critical for the SCAD vs GA2 differential. "
                "EMA elevation does NOT occur in isolated MCAD, VLCAD, LCHAD, or ACAT1."
            ),
        },
        "Glutaric_acid_urine": {
            "label":     "Glutaric Acid (Urine) — ALSO in GA1; distinguish by acylcarnitines",
            "direction": "ELEVATED",
            "normal":    "<10 mmol/mol Cr",
            "status":    "↑ (ELEVATED — glutaryl-CoA DH arm blocked)",
            "color":     "warning",
            "rationale": (
                "Glutaric acid is elevated because GCD (glutaryl-CoA dehydrogenase) is also blocked (ETF-dependent). "
                "Glutaric acid is also elevated in GA1 (glutaryl-CoA DH primary deficiency). "
                "Key distinction: GA1 has isolated C5-DC on NBS + no pan-acylcarnitinemia; "
                "GA2 has PAN-ACYLCARNITINEMIA (C4+C5+C8+C16 all elevated)."
            ),
        },
        "C0_free_carnitine": {
            "label":     "C0 (Free Carnitine) — secondary depletion",
            "direction": "LOW",
            "normal":    "15–50 µmol/L",
            "status":    "↓ (secondary depletion from pan-acylcarnitinemia)",
            "color":     "warning",
            "rationale": (
                "C0 depletion is secondary to massive acylcarnitine conjugation. "
                "L-carnitine supplementation is indicated when C0 <10 µmol/L. "
                "Unlike MCAD, secondary carnitine depletion in GA2 can be profound due to pan-acylcarnitinemia."
            ),
        },
        "Adipic_Suberic_Sebacic": {
            "label":     "Adipic / Suberic / Sebacic Acids (Urine) — dicarboxylic acids",
            "direction": "ELEVATED",
            "normal":    "<10 mmol/mol Cr each",
            "status":    "↑ (elevated in GA2 — omega oxidation of backed-up medium chain FAs)",
            "color":     "warning",
            "rationale": (
                "Adipic (C6), suberic (C8), and sebacic (C10) dicarboxylic acids are elevated "
                "because medium-chain fatty acids that cannot be beta-oxidised undergo peroxisomal "
                "omega-oxidation instead. This dicarboxylic acid pattern is characteristic of GA2 "
                "and also of MCAD — but in GA2, ALL chain lengths show this pattern."
            ),
        },
    }

    phenotype_patterns = [
        {
            "type": "Type III (RR-MADD / ETFDH)",
            "prevalence": "~62%",
            "onset": "Adolescence–adult",
            "c4": "1.5–6.0 µmol/L",
            "c5": "1.0–4.0 µmol/L",
            "c8": "0.8–3.5 µmol/L",
            "c16": "4.0–14.0 µmol/L",
            "ema": "80–600 mmol/mol Cr",
            "glucose": "2.0–4.5 mmol/L",
            "riboflavin_response": "DRAMATIC (hallmark)",
            "prognosis": "Excellent with riboflavin; myopathy resolves",
            "gene": "ETFDH",
        },
        {
            "type": "Type II neonatal (no congenital anomalies)",
            "prevalence": "~18%",
            "onset": "Neonatal (day 1–5)",
            "c4": "3.0–12.0 µmol/L",
            "c5": "2.5–8.0 µmol/L",
            "c8": "2.0–8.0 µmol/L",
            "c16": "8.0–25.0 µmol/L",
            "ema": "400–2000 mmol/mol Cr",
            "glucose": "0.5–2.5 mmol/L",
            "riboflavin_response": "None / partial (ETFA p.Arg191Cys)",
            "prognosis": "Guarded; depends on metabolic crisis management; cardiomyopathy common",
            "gene": "ETFA / ETFB",
        },
        {
            "type": "Type I neonatal (with congenital anomalies)",
            "prevalence": "~12%",
            "onset": "Neonatal (<24 hours)",
            "c4": "5.0–18.0 µmol/L",
            "c5": "4.0–12.0 µmol/L",
            "c8": "3.0–10.0 µmol/L",
            "c16": "10.0–35.0 µmol/L",
            "ema": "800–3000 mmol/mol Cr",
            "glucose": "0.2–1.5 mmol/L",
            "riboflavin_response": "None",
            "prognosis": "Death in days–weeks; polycystic kidneys + brain malformations; no curative Rx",
            "gene": "ETFA / ETFB (null alleles)",
        },
        {
            "type": "Type III ETFDH childhood onset",
            "prevalence": "~23%",
            "onset": "Infancy–childhood",
            "c4": "2.0–8.0 µmol/L",
            "c5": "1.5–5.5 µmol/L",
            "c8": "1.0–5.0 µmol/L",
            "c16": "5.0–18.0 µmol/L",
            "ema": "150–800 mmol/mol Cr",
            "glucose": "1.8–4.0 mmol/L",
            "riboflavin_response": "DRAMATIC",
            "prognosis": "Good with riboflavin ± L-carnitine; early treatment prevents myopathy progression",
            "gene": "ETFDH",
        },
    ]

    treatment_table = [
        {
            "intervention":  "Riboflavin 100–300 mg/day (Vitamin B2)",
            "level":         "Level A — FIRST-LINE for Type III (RR-MADD); Level B for Type II (ETFA p.Arg191Cys)",
            "rationale":     "FAD precursor; restores partial ETF/ETFDH function in missense variants; DRAMATIC response in ETFDH mutations — acylcarnitines + EMA normalise within days–weeks; myopathy resolves within months",
            "contraindication": None,
        },
        {
            "intervention":  "Avoid fasting",
            "level":         "Level A — ALL types",
            "rationale":     "Fasting forces FAO which is globally blocked; triggers hypoketotic hypoglycaemia and metabolic crisis; fasting duration limits: 0–1y (<4 hr), 1–3y (<6 hr), >3y (<8 hr)",
            "contraindication": None,
        },
        {
            "intervention":  "IV Glucose 10% + Riboflavin IV/enteral (acute crisis)",
            "level":         "Level A — acute decompensation",
            "rationale":     "Glucose at GIR 8–12 mg/kg/min suppresses catabolism; riboflavin administered simultaneously; reverses hypoketotic hypoglycaemia and metabolic acidosis",
            "contraindication": None,
        },
        {
            "intervention":  "L-Carnitine 50–100 mg/kg/day",
            "level":         "Level A if C0 <10 µmol/L; Level B for maintenance",
            "rationale":     "Conjugates backed-up acyl-CoA species as acylcarnitines for renal excretion; replaces secondary depletion from pan-acylcarnitinemia; reduces acyl-CoA toxicity",
            "contraindication": None,
        },
        {
            "intervention":  "CoQ10 100–300 mg/day",
            "level":         "Level B — especially for ETFDH mutations",
            "rationale":     "ETFDH binds CoQ10 for electron transfer; CoQ10 supplementation may augment residual ETFDH activity; evidence in ETFDH RR-MADD supports adjunct to riboflavin",
            "contraindication": None,
        },
        {
            "intervention":  "Low-fat diet / fat restriction (symptomatic Type II/III)",
            "level":         "Level B — adjunct during acute phases",
            "rationale":     "Reduces acyl-CoA substrate load; not as strict as in VLCAD/LCHAD where long-chain fat is primary danger; MCT oil is NOT beneficial in GA2 (MCT oxidation also blocked)",
            "contraindication": "MCT oil is INEFFECTIVE in GA2 (unlike VLCAD/LCHAD) — MCT is still acyl-CoA substrate requiring ETF",
        },
        {
            "intervention":  "VPA (Valproic acid)",
            "level":         "ABSOLUTE CONTRAINDICATION — ALL types",
            "rationale":     "VPA = valproyl-CoA + beta-oxidation products that directly block ETF/ETFDH; additionally depletes carnitine + inhibits respiratory chain Complex I; FATAL in Type I/II; severe decompensation in Type III",
            "contraindication": "ABSOLUTE CI — ALL types (Types I, II, III)",
        },
        {
            "intervention":  "Ketogenic Diet (KD)",
            "level":         "ABSOLUTE CONTRAINDICATION — ALL types",
            "rationale":     "KD requires ALL fat oxidation pathways; EVERY acyl-CoA dehydrogenase is blocked in GA2 (ETF conduit absent); KD → massive acylcarnitine accumulation + energy failure → catastrophic decompensation",
            "contraindication": "ABSOLUTE CI — ALL types",
        },
    ]

    key_differentials = {
        "GA2_vs_SCAD": (
            "EMA elevated in BOTH; but GA2 EMA is MUCH higher (>300 vs <50 mmol/mol Cr). "
            "GA2 has C5 + C8 elevated (IVD + MCAD arms also blocked); SCAD has C4 only. "
            "Riboflavin response: Type III GA2 → DRAMATIC; isolated SCAD → modest at best."
        ),
        "GA2_vs_MCAD": (
            "C8 elevated in both; GA2 has PAN-acylcarnitinemia (C4+C5+C8+C16 all up). "
            "MCAD has C8+HG+SG+PPG (pathognomonic glycine conjugates) — absent in GA2. "
            "MCAD: C4 and C14 NORMAL; GA2: C4 and C14:1 BOTH elevated."
        ),
        "GA2_vs_GA1": (
            "Glutaric acid elevated in BOTH. GA1 has ISOLATED C5-DC (glutarylcarnitine) on NBS, "
            "NO pan-acylcarnitinemia, NO C4/C8 elevation. GA2 has pan-acylcarnitinemia (C4+C5+C8+C16). "
            "Macrocephaly + striatal injury (putamen) = GA1 hallmark — ABSENT in GA2."
        ),
        "GA2_vs_IVD": (
            "C5 elevated in both; GA2 has PAN-acylcarnitinemia (C4+C5+C8+C16). "
            "Isolated IVD: C5 elevated ONLY; isovalerylglycine dominant urine OA; C4/C8/C16 NORMAL. "
            "Isovaleryl smell (sweaty feet) = IVD hallmark, not GA2."
        ),
        "RR-MADD_vs_Polymyositis": (
            "ETFDH RR-MADD mimics adult inflammatory myopathy (weakness, elevated CK). "
            "Key: RR-MADD has elevated acylcarnitines (C4/C5/C8) + EMA on urine OA. "
            "Riboflavin trial MANDATORY before muscle biopsy in any adult proximal myopathy + elevated CK."
        ),
        "Type_I_vs_Type_II": (
            "Both neonatal severe; Type I has CONGENITAL ANOMALIES (polycystic kidneys, facial dysmorphism, "
            "brain malformations) = ETF heterodimer structurally absent. "
            "Type II has NO congenital anomalies; better prognosis; some riboflavin response (ETFA p.Arg191Cys)."
        ),
    }

    exam_pearls = [
        "PAN-ACYLCARNITINEMIA (C4+C5+C8+C10+C12+C14:1+C16+C5-DC ALL elevated) = DIAGNOSTIC for GA2",
        "C5 ELEVATED = KEY POSITIVE vs SCAD (SCAD = C4 only; GA2 = C4+C5+C8+C14:1+C16)",
        "C8 ELEVATED = KEY POSITIVE vs SCAD (SCAD = C4 only; C8 normal in isolated SCAD)",
        "EMA >> SCAD levels: GA2 EMA >300 mmol/mol Cr; SCAD EMA <50 mmol/mol Cr",
        "TYPE III (RR-MADD) = ETFDH mutations; RIBOFLAVIN 100–300 mg/day = DRAMATIC response (Level A)",
        "Type I = congenital anomalies (polycystic kidneys, brain malformations) = ETFA/ETFB null alleles",
        "VPA = ABSOLUTE CI (ALL types) — blocks ETF pathway directly; FATAL in neonatal types",
        "KD = ABSOLUTE CI (ALL types) — ALL fat oxidation blocked; contrast: KD HELPS PDH, ABSOLUTE CI here",
        "MCT oil = INEFFECTIVE in GA2 (unlike VLCAD/LCHAD) — MCT still requires ETF for beta-oxidation",
        "CoQ10 100–300 mg/day adjunct for ETFDH mutations (ETFDH binds CoQ10)",
        "p.Arg191Trp (ETFDH) = East Asian founder; ~50% of Asian RR-MADD patients",
        "Riboflavin trial MANDATORY before muscle biopsy in adult proximal myopathy + elevated acylcarnitines",
        "Glutaric acid in urine: GA2 vs GA1 distinction by pan-acylcarnitinemia (GA2) vs isolated C5-DC (GA1)",
        "FA conduit concept: ETF → ETFDH is the common ELECTRON ACCEPTOR for ALL acyl-CoA dehydrogenases",
        "Emergency: IV glucose 10% GIR 8–12 mg/kg/min + IV/enteral riboflavin + avoid fat loading",
    ]

    return {
        "biomarkers": biomarkers,
        "phenotype_patterns": phenotype_patterns,
        "treatment_table": treatment_table,
        "patient_sample": sample,
        "variant_table": VARIANTS,
        "key_differentials": key_differentials,
        "exam_pearls": exam_pearls,
    }


def get_definitions():
    return {
        "disease_name": "GA2 (Glutaric Acidemia Type II) / MADD (Multiple Acyl-CoA Dehydrogenase Deficiency)",
        "gene": "ETFA (15q23–q25) / ETFB (19q13.41) / ETFDH (4q32.1)",
        "locus": "ETFA: 15q23–q25 · ETFB: 19q13.41 · ETFDH: 4q32.1",
        "omim_gene": "ETFA *608053 / ETFB *130410 / ETFDH *231675",
        "omim_disease": "#231680",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF of ETFA or ETFB or ETFDH",
        "protein": (
            "ETFA — 333 aa FAD-binding alpha subunit of ETF (mitochondrial matrix). "
            "ETFB — 255 aa AMP-binding beta subunit of ETF (mitochondrial matrix). "
            "ETFDH — 617 aa ETF:Ubiquinone Oxidoreductase (FAD + Fe/S cluster + CoQ10 binding; "
            "inner mitochondrial membrane — electron transfer conduit from ETF to respiratory chain)."
        ),
        "enzymatic_function": (
            "ETF (ETFA+ETFB heterodimer) accepts FADH2 from ALL mitochondrial acyl-CoA dehydrogenases "
            "(SCAD, MCAD, VLCAD, LCAD, IVD, GCD, DHODH, DMGDH, SARDH). "
            "ETFDH then re-oxidises ETF-FADH2 by transferring electrons to ubiquinone (CoQ10) → "
            "electrons enter respiratory chain at Complex III level. "
            "GA2 = block in this common conduit → ALL acyl-CoA dehydrogenases SIMULTANEOUSLY BACK-INHIBITED "
            "→ PAN-ACYLCARNITINEMIA (C4+C5+C8+C10+C12+C14:1+C16+C5-DC all elevated)."
        ),
        "pathway": "Mitochondrial fatty acid beta-oxidation (Step 1 electron conduit) + amino acid catabolism (IVD/GCD)",
        "metabolic_block": (
            "ETFA/ETFB/ETFDH LOF → ETF-FADH2 cannot be transferred to ubiquinone → "
            "ALL acyl-CoA dehydrogenases cannot re-oxidise their FAD cofactor → "
            "ALL step-1 reactions (all chain lengths) are SIMULTANEOUSLY BLOCKED → "
            "C4, C5, C8, C10, C12, C14:1, C16, C18, C5-DC ALL accumulate → PAN-ACYLCARNITINEMIA. "
            "Urine: EMA ↑↑, glutaric acid ↑, adipic/suberic/sebacic (dicarboxylic acids) ↑, isovalerylglycine ↑."
        ),
        "nbs_marker": "PAN-ACYLCARNITINEMIA: C4+C5+C8+C10+C12+C14:1+C16+C5-DC ALL elevated on tandem MS/MS",
        "clinical_types": {
            "Type_I": (
                "ETFA/ETFB null alleles; neonatal severe; congenital anomalies (polycystic kidneys, "
                "facial dysmorphism, brain malformations, hepatomegaly); death in days–weeks; "
                "overwhelming metabolic acidosis + hypoglycaemia from birth; no riboflavin response."
            ),
            "Type_II": (
                "ETFA/ETFB missense/hypomorphic; neonatal severe WITHOUT congenital anomalies; "
                "severe hypoketotic hypoglycaemia + metabolic acidosis; cardiomyopathy common; "
                "ETFA p.Arg191Cys: some riboflavin response possible; guarded prognosis."
            ),
            "Type_III_RR-MADD": (
                "ETFDH mutations (most common: p.Arg191Trp, p.Arg191Cys, p.Pro456Leu); "
                "mild/late onset (infancy to adult); proximal myopathy, exercise intolerance, fatigue; "
                "episodic metabolic decompensation (hypoglycaemia); DRAMATIC riboflavin response: "
                "100–300 mg/day normalises acylcarnitines + EMA + myopathy within weeks–months."
            ),
        },
        "rr_madd_definition": (
            "Riboflavin-Responsive MADD (RR-MADD) = Type III GA2 caused by ETFDH missense mutations "
            "with residual protein folding. Riboflavin (FAD precursor) supplements the cofactor pool, "
            "partially restores ETFDH tertiary structure and FAD binding → near-normal electron transfer → "
            "complete clinical and biochemical remission. Response is DRAMATIC (hallmark test). "
            "p.Arg191Trp (East Asian founder, ~50% of Asian RR-MADD) and p.Arg191Cys (European) are the "
            "most common RR-MADD variants. CoQ10 as adjunct may augment ETFDH function in some patients."
        ),
        "treatments": {
            "riboflavin": "Level A (Type III RR-MADD): 100–300 mg/day; DRAMATIC response in ETFDH mutations",
            "avoid_fasting": "Level A (all types): fasting triggers crisis; duration limits by age",
            "iv_glucose": "Level A (acute crisis): GIR 8–12 mg/kg/min suppresses catabolism",
            "l_carnitine": "Level A if C0 <10 µmol/L; Level B for maintenance",
            "coq10": "Level B (ETFDH): 100–300 mg/day adjunct; ETFDH binds CoQ10",
            "contraindications": [
                "VPA = ABSOLUTE CI ALL TYPES (blocks ETF/ETFDH directly; lethal in neonatal types)",
                "KD = ABSOLUTE CI ALL TYPES (all fat oxidation blocked; no MCT workaround)",
                "MCT oil = INEFFECTIVE (MCT still requires ETF; contrast with VLCAD/LCHAD where MCT bypasses block)",
            ],
        },
        "prevalence": "~1:100,000–250,000 (all types); RR-MADD (Type III ETFDH) most common in modern era",
        "common_variants": {
            "ETFDH p.Arg191Trp (c.571C>T)": (
                "~30% of GA2 cohort; East Asian founder; ~50% of Asian RR-MADD; "
                "critical FAD-binding residue; DRAMATIC riboflavin response"
            ),
            "ETFDH p.Arg191Cys (c.571C>T allele 2)": (
                "~12%; European; allelic with p.Arg191Trp — same codon different substitution; "
                "moderate riboflavin response; seen in Type II and Type III"
            ),
            "ETFDH p.Pro456Leu (c.1367C>T)": (
                "~8%; Northern European; Fe/S cluster region; riboflavin-responsive Type III"
            ),
        },
        "key_exam_facts": [
            "PAN-ACYLCARNITINEMIA (C4+C5+C8+C14:1+C16 ALL elevated) = diagnostic for GA2/MADD",
            "C5 ELEVATED = KEY POSITIVE vs SCAD (SCAD = C4 only; IVD arm also blocked in GA2)",
            "C8 ELEVATED = KEY POSITIVE vs SCAD (SCAD C8 NORMAL; GA2 C8 elevated — MCAD arm blocked)",
            "EMA >> SCAD: GA2 EMA >300 mmol/mol Cr; SCAD EMA <50 mmol/mol Cr",
            "Type III (RR-MADD) = ETFDH mutations; riboflavin 100–300 mg/day = DRAMATIC response (Level A)",
            "Type I = congenital anomalies (polycystic kidneys, brain malformations) = ETFA/ETFB null",
            "VPA = ABSOLUTE CI (ALL types) — blocks ETF pathway + respiratory chain",
            "KD = ABSOLUTE CI (ALL types) — ALL fat oxidation blocked",
            "MCT oil INEFFECTIVE in GA2 (unlike VLCAD/LCHAD) — MCT still requires ETF",
            "CoQ10 adjunct for ETFDH mutations (ETFDH binds CoQ10 for electron transfer)",
            "p.Arg191Trp (ETFDH) = East Asian founder; ~50% Asian RR-MADD patients",
            "Riboflavin trial MANDATORY before muscle biopsy in adult proximal myopathy + acylcarnitinemia",
            "GA2 vs GA1: GA2 = pan-acylcarnitinemia; GA1 = isolated C5-DC + NO C4/C8/C16 elevation",
            "Electron conduit: ETF → ETFDH → CoQ10 is common to ALL acyl-CoA dehydrogenases",
            "Emergency: IV glucose 10% + riboflavin (enteral or IV) + carnitine; NEVER VPA or KD",
        ],
        "glossary": {
            "GA2": "Glutaric Acidemia Type II (OMIM #231680) — pan-acylcarnitinemia from ETF/ETFDH deficiency",
            "MADD": "Multiple Acyl-CoA Dehydrogenase Deficiency — synonym for GA2; reflects simultaneous block of all dehydrogenases",
            "RR-MADD": "Riboflavin-Responsive MADD — Type III GA2 (ETFDH mutations); DRAMATIC riboflavin response",
            "ETF": "Electron Transfer Flavoprotein — ETFA+ETFB heterodimer; accepts FADH2 from all acyl-CoA dehydrogenases",
            "ETFA": "ETF alpha subunit (333 aa, 15q23–q25) — FAD-binding; forms heterodimer with ETFB",
            "ETFB": "ETF beta subunit (255 aa, 19q13.41) — AMP-binding; forms heterodimer with ETFA",
            "ETFDH": "ETF:Ubiquinone Oxidoreductase (617 aa, 4q32.1) — FAD+Fe/S+CoQ10; transfers electrons from ETF to CoQ10",
            "PAN-ACYLCARNITINEMIA": "Simultaneous elevation of C4+C5+C8+C10+C12+C14:1+C16+C5-DC — diagnostic pattern for GA2",
            "EMA": "Ethylmalonic acid — elevated in BOTH SCAD and GA2; MUCH higher in GA2 (>300 vs <50 mmol/mol Cr)",
            "IVD": "Isovaleryl-CoA Dehydrogenase — one of the FAO enzymes requiring ETF; IVD block → C5 elevation in GA2",
            "GCD": "Glutaryl-CoA Dehydrogenase — ETF-dependent; GCD block in GA2 → glutaric acid elevation (also in GA1)",
            "CoQ10": "Coenzyme Q10 (ubiquinone) — electron acceptor for ETFDH; CoQ10 supplementation supports ETFDH function",
            "FAD": "Flavin Adenine Dinucleotide — cofactor of all acyl-CoA dehydrogenases + ETF + ETFDH; riboflavin (B2) is precursor",
            "Riboflavin": "Vitamin B2 → FAD; restores FAD-binding in ETFDH missense mutants → near-normal electron transfer in RR-MADD",
            "NBS": "Newborn Screening — pan-acylcarnitinemia on tandem MS/MS is the diagnostic NBS signal for GA2",
        },
        "references": [
            "Angle B et al. (1993). Type II glutaric aciduria. Brain Dev 15(6):462-467.",
            "Olsen RK et al. (2007). ETFDH mutations as a major cause of riboflavin-responsive MADD. Brain 130(Pt 8):2045-2054.",
            "Yamada K et al. (2006). ETFA and ETFB mutations in glutaric aciduria type II. J Inherit Metab Dis 29(1):93-96.",
            "Liang WC et al. (2009). Riboflavin-responsive oxidative phosphorylation complex I deficiency caused by defective ACAD9. Brain 132(Pt 8):2170-2179.",
            "Prasun P (2021). Multiple Acyl-CoA Dehydrogenase Deficiency. In: Adam MP et al., eds. GeneReviews. Seattle: Univ Washington.",
            "OMIM #231680 — Glutaric Acidemia Type II (ETFA *608053 / ETFB *130410 / ETFDH *231675). omim.org.",
            "Frerman FE, Goodman SI (2001). Defects of electron transfer flavoprotein and ETF-ubiquinone oxidoreductase. Scriver CR et al., Metabolic Basis of Inherited Disease, 8th ed.",
        ],
    }
