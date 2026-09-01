#!/usr/bin/env python3
"""LRPPRC — Leigh Syndrome French-Canadian Type (LSFC) + Combined Complex I & IV Deficiency.

LRPPRC (Leucine-Rich Pentatricopeptide Repeat-Containing, also LRP130 / GP130) encodes
a 1394 aa nuclear-encoded mitochondrial matrix protein that forms the LRPPRC–SLIRP complex,
which is required to stabilise and co-ordinately polyadenylate ALL 13 mitochondrially-encoded
mRNAs (MT-CO1, MT-CO2, MT-CO3, MT-ND1–6, MT-ATP6, MT-ATP8, MT-CYB).  Loss of LRPPRC
causes rapid degradation of multiple mt-mRNAs → reduced translation of all mt-encoded OXPHOS
subunits → combined Complex I + Complex IV (COX) deficiency.

  LRPPRC gene      OMIM *607544
  Disease          Leigh Syndrome French-Canadian Type (LSFC, OMIM #220111 / *607544-related)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       2p21

FOUNDER POPULATION: Saguenay–Lac-Saint-Jean (SLSJ) region, Quebec, Canada.
  Carrier frequency in SLSJ: ~1 in 40
  Affected births in SLSJ:   ~1 in 2000 — one of the highest rates of a rare mito disease globally
  Founder mutation:           p.Ala354Val (c.1061C>T) — in N-terminal PPR-3 repeat domain

PATHOPHYSIOLOGY (LRPPRC / all mt-mRNAs / Combined I + IV):
  LRPPRC PPR domain binds the SLIRP (SRA Stem-Loop Interacting RNA Binding Protein) co-factor:
    • LRPPRC–SLIRP complex associates with all 13 mt-mRNA 3' ends
    • Promotes poly(A) tail addition by MTPAP (mt-poly(A) polymerase)
    • Prevents premature mt-mRNA degradation by mitochondrial RNA degradosome
    • Stabilised mt-mRNAs: MT-CO1, MT-CO2, MT-CO3, MT-ND1–6, MT-ATP6, MT-ATP8, MT-CYB

  Loss of LRPPRC:
    → ALL 13 mt-mRNAs are rapidly degraded (poly-A tail lost → PNPase/SUV3 degradation)
    → ↓↓ translation of ALL mtDNA-encoded OXPHOS subunits
    → Combined Complex I (NADH dehydrogenase) + Complex IV (COX) deficiency
    → Complex II (fully nuclear-encoded) and Complex III partially preserved
    → Isolated COX deficiency: NOT present — COMBINED CI+CIV distinguishes from SURF1/COX10/SCO2

DISTINGUISHING FEATURE — EPISODIC METABOLIC CRISES:
  • CARDINAL DISTINGUISHING FEATURE of LSFC: episodic acute metabolic crises
  • Triggers: febrile illness, infections, physiological stress, fasting, surgery
  • During crisis: severe lactic acidosis (lactate >8-15 mmol/L), encephalopathy,
    rapid neurological deterioration; can be rapidly fatal
  • Between crises: relatively stable; some developmental progress possible
  • FRENCH-CANADIAN founder disease; now also reported worldwide (non-SLSJ)

BIOCHEMICAL FINGERPRINT (distinguishes from isolated COX diseases):
  • Complex I: 20-40% of control → REDUCED (combined defect — KEY DDx from SURF1/COX10/SCO2)
  • Complex IV (COX): 15-35% of control → REDUCED
  • Complex II: NORMAL (nuclear-encoded — SDHA/B/C/D not affected)
  • Complex III: partially preserved (some mt-encoded CYB reduction, often mild)
  → COMBINED I + IV = LRPPRC signature; isolated COX = SURF1/SCO2/COX10/COX15

References:
  Mootha VK et al. Nat Genet. 2003;33(2):192–196. (LSFC gene identification)
  Sasarman F et al. EMBO J. 2010;29(17):2966–2976. (LRPPRC–SLIRP complex mechanism)
  Merante F et al. Mol Genet Metab. 1993;49(3):185–189. (LSFC original description)
  Falk MJ et al. J Med Genet. 2013;50(3):148–158. (LSFC natural history, SLSJ cohort)
"""
from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 605
DISEASE_ID   = "lrpprc"
DISEASE_NAME = "LRPPRC Leigh Syndrome French-Canadian Type — LSFC (Combined Complex I + IV Deficiency)"
GENE         = "LRPPRC"
PROTEIN      = "LRPPRC — 1394 aa, mitochondrial matrix, PPR-domain mt-mRNA stabiliser (all 13 mt-mRNAs)"
OMIM_GENE    = "*607544"
OMIM_DISEASE = "#220111 / *607544 (Leigh Syndrome French-Canadian Type, LSFC)"
CHROMOSOME   = "2p21"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Infantile–early childhood (6 months–3 years); episodic metabolic crises CARDINAL"
COHORT_SIZE  = 40
COLOR        = "#1a237e"   # deep indigo — founder disease; distinct from teal TACO1 / red-magenta SCO2
LIGHT        = "#e8eaf6"

# Genotype pool
GENO_A354V_HOM  = "p.Ala354Val homozygous (c.1061C>T) — SLSJ founder; most common SLSJ; moderate severity"
GENO_A354V_NULL = "p.Ala354Val / truncating (compound het) — SLSJ; more severe than homozygous"
GENO_NULL_NULL  = "Biallelic truncating (compound het) — non-SLSJ; severe; no LRPPRC protein"
GENO_MISS_MISS  = "Missense / missense (compound het) — non-SLSJ worldwide; variable severity"
GENO_DEL_MISS   = "Large deletion / missense (compound het) — rare; moderate-severe"

GENO_POOL    = [GENO_A354V_HOM, GENO_A354V_NULL, GENO_NULL_NULL, GENO_MISS_MISS, GENO_DEL_MISS]
GENO_WEIGHTS = [0.40,            0.25,             0.10,            0.18,           0.07]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient LRPPRC/LSFC cohort (seed-605)."""
    return random.Random(SEED)


# ── Patient cohort ────────────────────────────────────────────────────────────
_TX_POOL = [
    "IV Dextrose GIR 6-8 (crisis)",
    "CoQ10/Ubiquinol",
    "Riboflavin B2",
    "Thiamine B1 (empiric, mandatory)",
    "Biotin (empiric, mandatory)",
    "NaHCO3 (lactic acidosis)",
    "LEV (seizures)",
    "Aggressive fever management (crisis prevention)",
    "Carnitine (secondary deficiency)",
    "Liver monitoring / hepatic support",
    "Continuous enteral feeds (prevent fasting)",
    "NIV/BiPAP (respiratory support in crisis)",
]

_OUTCOMES = [
    "Alive — stable between crises, moderate developmental impairment (SLSJ founder, childhood)",
    "Alive — recurrent crises, progressive disability; ongoing support",
    "Alive — relatively mild (compound het missense), adolescent/adult",
    "Died — acute metabolic crisis + lactic acidosis + respiratory failure (infantile)",
    "Died — progressive neurological failure after recurrent crises (childhood/teen)",
]
_OUT_WEIGHTS = [0.30, 0.25, 0.15, 0.18, 0.12]


def _generate_cohort(rng: random.Random) -> list[dict[str, Any]]:
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno           = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex            = rng.choice(["M", "F"])
        onset_mo       = rng.choices([6, 8, 10, 12, 15, 18, 24, 30, 36],
                                      weights=[5, 8, 10, 14, 12, 14, 14, 10, 7])[0]
        onset_yr       = round(onset_mo / 12, 1)
        lactate_base   = round(rng.uniform(2.5, 7.0), 1)   # baseline elevated
        lactate_crisis = round(rng.uniform(8.0, 18.0), 1)  # during crisis
        coxI_pct       = rng.randint(20, 45)                # Complex I (combined)
        coxIV_pct      = rng.randint(15, 35)                # Complex IV (combined)

        has_crises      = rng.random() < 0.92   # CARDINAL — episodic metabolic crises
        has_leigh_mri   = rng.random() < 0.75
        has_regression  = rng.random() < 0.85
        has_hypotonia   = rng.random() < 0.80
        has_cognitive   = rng.random() < 0.80
        has_seizures    = rng.random() < 0.55
        has_ataxia      = rng.random() < 0.50
        has_hepatopathy = rng.random() < 0.45   # mild-moderate; KEY DDx SCO1 (100% severe neonatal)
        has_nystagmus   = rng.random() < 0.35
        has_resp        = rng.random() < 0.40   # during crisis
        has_hcm         = rng.random() < 0.08   # RARE — KEY DDx SCO2 (100%)
        has_tubulopathy = rng.random() < 0.10   # RARE — KEY DDx COX10 (65%)
        has_optic       = rng.random() < 0.25
        has_facial      = rng.random() < 0.30   # mild facial features (low nasal bridge)

        feat_list = ["Lactic acidosis (baseline elevated)", "Combined CI + CIV deficiency (DISTINGUISHING)"]
        if has_crises:      feat_list.append("Episodic metabolic crises (CARDINAL — fever-triggered)")
        if has_regression:  feat_list.append("Psychomotor regression")
        if has_hypotonia:   feat_list.append("Hypotonia")
        if has_cognitive:   feat_list.append("Cognitive delay / intellectual disability")
        if has_leigh_mri:   feat_list.append("Leigh/Leigh-like MRI")
        if has_seizures:    feat_list.append("Seizures")
        if has_ataxia:      feat_list.append("Ataxia")
        if has_hepatopathy: feat_list.append("Hepatopathy (mild-moderate)")
        if has_nystagmus:   feat_list.append("Nystagmus")
        if has_resp:        feat_list.append("Respiratory compromise (crisis)")
        if has_optic:       feat_list.append("Optic atrophy")
        if has_facial:      feat_list.append("Mild facial features (low nasal bridge)")
        if has_hcm:         feat_list.append("HCM (RARE)")
        if has_tubulopathy: feat_list.append("Renal tubulopathy (RARE)")

        txs     = rng.sample(_TX_POOL, k=rng.randint(3, 6))
        outcome = rng.choices(_OUTCOMES, weights=_OUT_WEIGHTS)[0]

        patients.append({
            "id":              f"LRPPRC-{i:03d}",
            "geno":            geno,
            "sex":             sex,
            "onset_yr":        onset_yr,
            "lactate_base":    lactate_base,
            "lactate_crisis":  lactate_crisis,
            "coxI_pct":        coxI_pct,
            "coxIV_pct":       coxIV_pct,
            "has_crises":      has_crises,
            "has_leigh_mri":   has_leigh_mri,
            "has_regression":  has_regression,
            "has_hypotonia":   has_hypotonia,
            "has_cognitive":   has_cognitive,
            "has_seizures":    has_seizures,
            "has_ataxia":      has_ataxia,
            "has_hepatopathy": has_hepatopathy,
            "has_nystagmus":   has_nystagmus,
            "has_resp":        has_resp,
            "has_hcm":         has_hcm,
            "has_tubulopathy": has_tubulopathy,
            "has_optic":       has_optic,
            "has_facial":      has_facial,
            "features":        ", ".join(feat_list[:7]),
            "treatments":      ", ".join(txs[:5]),
            "outcome":         outcome,
        })
    return patients


# ── Overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    died  = sum(1 for p in patients if p["outcome"].startswith("Died"))
    alive = COHORT_SIZE - died

    def _pct(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feature_frequencies = {
        "Episodic Metabolic Crises (CARDINAL DISTINGUISHING — fever-triggered)":  _pct("has_crises"),
        "Lactic Acidosis — Baseline Elevated (100% in LSFC)":                     100,
        "Combined Complex I + Complex IV Deficiency (DISTINGUISHING vs COX-only)": 100,
        "Psychomotor Regression":                                                  _pct("has_regression"),
        "Hypotonia":                                                               _pct("has_hypotonia"),
        "Cognitive Delay / Intellectual Disability":                               _pct("has_cognitive"),
        "Leigh / Leigh-like MRI":                                                  _pct("has_leigh_mri"),
        "Seizures":                                                                _pct("has_seizures"),
        "Ataxia":                                                                  _pct("has_ataxia"),
        "Hepatopathy — Mild-Moderate (KEY DDx SCO1 is 100% Severe Neonatal)":     _pct("has_hepatopathy"),
        "Respiratory Compromise (during crisis)":                                  _pct("has_resp"),
        "Nystagmus":                                                               _pct("has_nystagmus"),
        "Optic Atrophy":                                                           _pct("has_optic"),
        "Mild Facial Features (low nasal bridge)":                                 _pct("has_facial"),
        "HCM (RARE — KEY DDx SCO2 100% / COX15 78%)":                            _pct("has_hcm"),
        "Renal Tubulopathy (RARE — KEY DDx COX10 65%)":                           _pct("has_tubulopathy"),
        "NO HCM Dominant (KEY DDx SCO2/COX15)":                                   100,
        "NO Severe Neonatal Hepatic Failure (KEY DDx SCO1 — 100%)":               100,
        "NO Iron Overload (KEY DDx GRACILE)":                                      100,
        "French-Canadian Founder (SLSJ p.Ala354Val — 40% alleles in cohort)":     round(sum(1 for p in patients if "SLSJ" in p["geno"] or "founder" in p["geno"].lower()) / COHORT_SIZE * 100),
        "Alive (stable between crises in survivors)":                              round(alive / COHORT_SIZE * 100),
    }

    kpis = [
        {"label": "Cohort (n)",         "value": COHORT_SIZE,                                                        "color": COLOR},
        {"label": "Episodic Crises",    "value": f"{feature_frequencies['Episodic Metabolic Crises (CARDINAL DISTINGUISHING — fever-triggered)']}%", "color": "#c62828"},
        {"label": "Leigh MRI",          "value": f"{feature_frequencies['Leigh / Leigh-like MRI']}%",                "color": "#6a1b9a"},
        {"label": "Hepatopathy",        "value": f"{feature_frequencies['Hepatopathy — Mild-Moderate (KEY DDx SCO1 is 100% Severe Neonatal)']}%", "color": "#e65100"},
        {"label": "Hypotonia",          "value": f"{feature_frequencies['Hypotonia']}%",                             "color": COLOR},
        {"label": "Combined I+IV",      "value": "100%",                                                             "color": "#1565c0"},
        {"label": "Fatal",              "value": f"{round(died/COHORT_SIZE*100)}%",                                  "color": "#b71c1c"},
        {"label": "Seed",               "value": f"#{SEED}",                                                         "color": "#455a64"},
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "severity": "ABSOLUTE CI — ESPECIALLY DANGEROUS IN LRPPRC/LSFC",
            "mechanism": (
                "Triple mechanism — each compounded by LRPPRC/LSFC pathology:\n"
                "1. CoA SEQUESTRATION: VPA forms valproyl-CoA → sequesters mitochondrial CoA pool "
                "   → depletes acetyl-CoA and succinyl-CoA needed for TCA cycle and OXPHOS. "
                "   LRPPRC patients already have impaired OXPHOS (combined CI+CIV); CoA depletion "
                "   tips the metabolic balance into irreversible lactic crisis.\n"
                "2. POLG INHIBITION: VPA inhibits mitochondrial polymerase gamma (POLG) "
                "   → reduces mtDNA copy number → fewer mt-mRNA templates → further reduces "
                "   the already-depleted mt-mRNA pool that LRPPRC is failing to stabilise. "
                "   This compounds the LRPPRC molecular defect at the DNA template level.\n"
                "3. HEPATOTOXICITY: VPA is directly hepatotoxic; LRPPRC/LSFC patients have "
                "   mild-moderate hepatopathy (~45%) at baseline. Adding a hepatotoxic drug "
                "   can precipitate acute hepatic failure in LRPPRC patients even at standard doses.\n"
                "Use LEV (renal excretion, no hepatic metabolism, no mito toxicity) as first-line AED."
            ),
        },
        {
            "drug": "Metformin",
            "severity": "ABSOLUTE CI — CATASTROPHIC IN COMBINED CI+CIV LRPPRC DISEASE",
            "mechanism": (
                "Metformin is a Complex I inhibitor. LRPPRC/LSFC disease already causes "
                "COMBINED Complex I + Complex IV deficiency (20-40% of normal for Complex I). "
                "Adding metformin directly inhibits the residual Complex I activity:\n"
                "  • Blocks NADH → ubiquinone electron transfer at CI\n"
                "  • Combined CI inhibition (metformin) + baseline CI deficiency (LRPPRC) → "
                "    complete functional block of Complex I\n"
                "  • Massive NADH accumulation → pyruvate → lactate conversion → fatal lactic acidosis\n"
                "  • This is MORE dangerous in LRPPRC than in isolated COX diseases (SURF1/COX10/SCO2) "
                "    where Complex I is intact\n"
                "If glucose intolerance develops in long-term LRPPRC survivors: insulin only. "
                "Never use metformin, canagliflozin, or other medications that stress OXPHOS."
            ),
        },
        {
            "drug": "Linezolid",
            "severity": "ABSOLUTE CI",
            "mechanism": (
                "Linezolid (oxazolidinone) inhibits the mitochondrial 23S rRNA-equivalent (mt-LSU 16S "
                "rRNA), blocking elongation step of ALL mitochondrial ribosomal translation. "
                "In LRPPRC disease, mt-mRNAs are already depleted (all 13 are unstable without LRPPRC). "
                "Linezolid eliminates translation of the residual mt-mRNAs that are still present:\n"
                "  • LRPPRC deficiency → ↓ mt-mRNA stability → reduced but not absent mt-mRNA pool\n"
                "  • Linezolid → blocks ribosomal translation of what little mt-mRNA remains\n"
                "  • Combined: near-complete shutdown of all mt-encoded OXPHOS subunit synthesis\n"
                "  • All 5 respiratory complexes collapse simultaneously (unlike isolated COX diseases)\n"
                "Alternative antibiotics: vancomycin (MRSA), daptomycin, beta-lactams. "
                "Chloramphenicol has the SAME mechanism and is equally CONTRAINDICATED."
            ),
        },
        {
            "drug": "Propofol",
            "severity": "ABSOLUTE CI during crisis; AVOID at all times (PRIS + hepatic risk)",
            "mechanism": (
                "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex IV (cytochrome aa3 site) "
                "and uncouples beta-oxidation. In LRPPRC patients:\n"
                "1. COMBINED CI+CIV already deficient — propofol's direct COX inhibition compounds this\n"
                "2. LRPPRC patients have hepatopathy (~45%) → impaired propofol clearance (hepatic "
                "   glucuronidation) → propofol accumulates → higher systemic exposure → greater PRIS risk\n"
                "3. During a metabolic crisis: any additional OXPHOS inhibition may be rapidly lethal\n"
                "Alternative anaesthesia: sevoflurane (volatile) — safe in mito disease; "
                "dexmedetomidine for sedation. If propofol is unavoidable (RSI emergency), restrict "
                "to single induction dose only; NEVER use propofol infusion in LRPPRC/LSFC."
            ),
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "severity": "CONTRAINDICATED",
            "mechanism": (
                "High-fat diet requires beta-oxidation → FADH2 and NADH → Complex I/II → Complex IV. "
                "LRPPRC/LSFC has COMBINED Complex I + Complex IV deficiency:\n"
                "  • Beta-oxidation produces FADH2: enters at Complex II → Complex III → COX (CIV)\n"
                "    → With CIV deficient, reducing equivalents accumulate → lactic acidosis\n"
                "  • Beta-oxidation also produces NADH: enters at Complex I → same pathway\n"
                "    → With CI ALSO deficient, NADH cannot be re-oxidised → severe reducing equivalent "
                "      accumulation → fatal lactic crisis\n"
                "  • KD is uniquely dangerous in LRPPRC because BOTH electron-entry complexes are reduced\n"
                "  • Additionally, high-fat diet may stress the already-compromised liver in LRPPRC\n"
                "IV dextrose (GIR 6-8 mg/kg/min) is the preferred energy substrate. "
                "High-carbohydrate, low-fat oral diet between crises."
            ),
        },
        {
            "drug": "Fasting / Caloric Restriction",
            "severity": "DANGEROUS — MAJOR CRISIS TRIGGER (maintain continuous glucose at all times)",
            "mechanism": (
                "Fasting is the most preventable acute crisis trigger in LSFC/LRPPRC:\n"
                "  • Fasting forces gluconeogenesis and beta-oxidation as glucose falls\n"
                "  • Both pathways generate NADH/FADH2 that LRPPRC-deficient OXPHOS cannot re-oxidise\n"
                "  • Result: rapid lactate accumulation → metabolic crisis within hours of fasting onset\n\n"
                "Crisis prevention protocol:\n"
                "  1. NEVER fast LSFC patients — continuous enteral/parenteral glucose mandatory\n"
                "  2. During ANY intercurrent illness: IV dextrose GIR 6-8 mg/kg/min IMMEDIATELY\n"
                "  3. Pre-operative fasting: minimal (2h clear fluids); switch to IV dextrose\n"
                "  4. Fever management: aggressive antipyretics + cooling — fever doubles metabolic rate\n"
                "  5. Emergency card: all LSFC patients should carry a 'metabolic emergency' card "
                "     instructing immediate IV glucose and LEV at any ER presentation\n\n"
                "The episodic crisis pattern of LSFC is directly tied to fasting/fever; "
                "families must be educated that ANY illness = immediate glucose infusion."
            ),
        },
    ]

    return {
        "gene":              GENE,
        "protein":           PROTEIN,
        "disease":           DISEASE_NAME,
        "omim_gene":         OMIM_GENE,
        "omim_disease":      OMIM_DISEASE,
        "chromosome":        CHROMOSOME,
        "inheritance":       INHERITANCE,
        "onset":             ONSET,
        "cohort":            f"{COHORT_SIZE} patients · seed-{SEED} · LRPPRC biallelic (LSFC — Leigh + Episodic Crises + Combined CI+CIV)",
        "mechanism": (
            "LRPPRC encodes a 1394 aa mitochondrial matrix protein built from ~33 pentatricopeptide "
            "repeat (PPR) motifs that form a right-handed superhelix to grip single-stranded RNA. "
            "LRPPRC partners with SLIRP (SRA Stem-Loop Interacting RNA Binding Protein) to form "
            "the LRPPRC–SLIRP ribonucleoprotein complex. This complex stabilises ALL 13 "
            "mitochondrially-encoded mRNAs (MT-CO1, CO2, CO3, ND1-ND6, ATP6, ATP8, CYB) by "
            "recruiting MTPAP (mt-poly-A-polymerase) to add poly-A tails and by shielding the "
            "3' ends from the mt-RNA degradosome (SUPV3L1/PNPase). Without LRPPRC, all 13 "
            "mt-mRNAs lose their poly-A tails and are rapidly degraded. Translation of ALL "
            "mt-encoded OXPHOS subunits falls sharply: MT-CO1/CO2/CO3 (Complex IV core), "
            "MT-ND1-6 (Complex I core), MT-ATP6/8 (Complex V), MT-CYB (Complex III). "
            "This produces a COMBINED Complex I + Complex IV deficiency — the defining "
            "biochemical fingerprint of LRPPRC/LSFC, contrasting with the ISOLATED Complex IV "
            "deficiency of SURF1, SCO1, SCO2, COX10, COX15, and TACO1. "
            "The residual OXPHOS capacity is sufficient for baseline energy needs between crises, "
            "but ANY intercurrent illness, fever, or fasting increases ATP demand beyond this "
            "reduced capacity → acute lactic acidosis crisis → encephalopathy → rapid deterioration. "
            "The French-Canadian founder mutation p.Ala354Val in the PPR-3 repeat reduces but does "
            "not abolish LRPPRC protein, explaining the episodic-stable phenotype (rather than "
            "continuous severe decline as in complete-null SURF1/SCO2)."
        ),
        "crisis_note": (
            "EPISODIC METABOLIC CRISES — CARDINAL DISTINGUISHING FEATURE OF LRPPRC/LSFC:\n"
            "Episodic acute metabolic crises are present in ~92% of LRPPRC/LSFC patients:\n"
            "  • TRIGGERS: febrile illness (most common), infections, surgery, fasting, extreme stress\n"
            "  • DURING CRISIS: severe lactic acidosis (lactate 8-18 mmol/L), encephalopathy, "
            "    rapid neurological deterioration, can progress to coma and death within hours\n"
            "  • BETWEEN CRISES: relatively stable; some developmental progress possible; "
            "    baseline lactate mildly elevated (2.5-7 mmol/L) but not acutely dangerous\n"
            "  • FREQUENCY: 1-5+ crises per year; each crisis carries ~15-20% mortality risk\n"
            "  • KEY DDx: distinguishes LSFC from TACO1 (no crises; slower progressive decline), "
            "    SURF1 (continuous decline, respiratory dominant), SCO2 (cardiac dominant, fatal first year)\n\n"
            "Crisis management protocol:\n"
            "  1. IV dextrose STAT: GIR 6-8 mg/kg/min — suppress gluconeogenesis + beta-oxidation\n"
            "  2. Aggressive fever control: acetaminophen + ibuprofen + cooling blankets\n"
            "  3. NaHCO3: if pH <7.2 or BE < −12 (lactic acidosis correction)\n"
            "  4. LEV IV: if seizures occur (ABSOLUTE CI VPA)\n"
            "  5. Avoid fasting: maintain continuous glucose infusion until crisis resolves\n"
            "  6. ICU admission: for severe crisis (lactate >10, encephalopathy)\n"
            "  7. NEVER: propofol, linezolid, metformin, KD, VPA during crisis"
        ),
        "kpis":               kpis,
        "feature_frequencies": feature_frequencies,
        "contraindications":  contraindications,
    }


# ── Breakdown (patients + feature frequencies) ─────────────────────────────
def get_breakdown() -> dict[str, Any]:
    rng      = _rng()
    patients = _generate_cohort(rng)

    def _pct2(key: str) -> int:
        return round(sum(1 for p in patients if p[key]) / COHORT_SIZE * 100)

    feat_freq = {
        "Episodic Metabolic Crises (CARDINAL DISTINGUISHING)":                  _pct2("has_crises"),
        "Lactic Acidosis — Baseline Elevated (100%)":                           100,
        "Combined Complex I + Complex IV Deficiency (DISTINGUISHING)":          100,
        "Psychomotor Regression":                                               _pct2("has_regression"),
        "Hypotonia":                                                            _pct2("has_hypotonia"),
        "Cognitive Delay / Intellectual Disability":                            _pct2("has_cognitive"),
        "Leigh / Leigh-like MRI":                                              _pct2("has_leigh_mri"),
        "Seizures":                                                             _pct2("has_seizures"),
        "Ataxia":                                                               _pct2("has_ataxia"),
        "Hepatopathy — Mild-Moderate (KEY DDx SCO1 100% Severe Neonatal)":     _pct2("has_hepatopathy"),
        "Respiratory Compromise (during crisis)":                               _pct2("has_resp"),
        "Nystagmus":                                                            _pct2("has_nystagmus"),
        "Optic Atrophy":                                                        _pct2("has_optic"),
        "Mild Facial Features (low nasal bridge)":                              _pct2("has_facial"),
        "HCM (RARE — KEY DDx SCO2 100% / COX15 78%)":                         _pct2("has_hcm"),
        "Renal Tubulopathy (RARE — KEY DDx COX10 65%)":                        _pct2("has_tubulopathy"),
        "NO HCM Dominant (KEY DDx SCO2/COX15)":                               100,
        "NO Severe Neonatal Hepatic Failure (KEY DDx SCO1)":                   100,
        "NO Iron Overload (KEY DDx GRACILE)":                                  100,
        "Died":                                                                 round(sum(1 for p in patients if p["outcome"].startswith("Died")) / COHORT_SIZE * 100),
    }

    return {
        "patients":            patients,
        "feature_frequencies": feat_freq,
    }


# ── Definitions ────────────────────────────────────────────────────────────
def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "LRPPRC — PPR-Domain mt-mRNA Stabiliser (1394 aa, 2p21)",
            "definition": (
                "LRPPRC (Leucine-Rich Pentatricopeptide Repeat-Containing, OMIM *607544) encodes "
                "a 1394 amino acid mitochondrial matrix protein:\n\n"
                "Structural domain:\n"
                "  • ~33 tandem pentatricopeptide repeat (PPR) motifs — each 35 aa; together form "
                "    a right-handed superhelix with a concave inner surface that grips single-stranded RNA\n"
                "  • PPR proteins are the largest protein family in plants (>400 members) but "
                "    LRPPRC is one of few mammalian PPR proteins targeting mitochondrial RNA\n"
                "  • N-terminal mitochondrial targeting sequence (MTS): cleaved upon mitochondrial import\n\n"
                "Molecular function:\n"
                "  • LRPPRC binds SLIRP (SRA Stem-Loop Interacting RNA Binding Protein) to form "
                "    the LRPPRC–SLIRP complex — a ribonucleoprotein 'coat' for mt-mRNAs\n"
                "  • Recruits MTPAP (mt-poly-A polymerase) to add poly-A tails to all 13 mt-mRNAs\n"
                "  • Poly-A tails protect mt-mRNAs from degradation by the mt-RNA degradosome "
                "    (SUPV3L1 RNA helicase + PNPase polynucleotide phosphorylase complex)\n"
                "  • Targets ALL 13 mtDNA-encoded mRNAs: MT-CO1, MT-CO2, MT-CO3, "
                "    MT-ND1–6, MT-ATP6, MT-ATP8, MT-CYB\n\n"
                "Consequence of LRPPRC loss:\n"
                "  • All 13 mt-mRNAs lose poly-A tails → rapid degradation by mt-degradosome\n"
                "  • Translation of ALL mt-encoded OXPHOS subunits reduced:\n"
                "    - Complex I (ND1-6): 20-40% of control → COMBINED CI deficiency\n"
                "    - Complex IV (COX1-3): 15-35% of control → COMBINED CIV deficiency\n"
                "    - Complex III/V: partially affected\n"
                "  • Complex II: NORMAL (all 4 SDHA/B/C/D subunits are nuclear-encoded — not affected)\n"
                "  • Distinguishes from ALL other COX diseases: COMBINED I+IV vs isolated IV"
            ),
        },
        {
            "term": "LRPPRC vs TACO1 — All mt-mRNAs vs MT-CO1 Specific (Critical DDx)",
            "definition": (
                "LRPPRC and TACO1 both affect mt-mRNA handling but differ fundamentally:\n\n"
                "LRPPRC (2p21) — BROAD, all mt-mRNAs:\n"
                "  • Stabilises ALL 13 mt-mRNAs via LRPPRC–SLIRP complex and poly-A addition\n"
                "  • Loss → rapid degradation of MT-CO1, MT-CO2, MT-CO3, MT-ND1-6, MT-CYB, etc.\n"
                "  • Biochemistry: COMBINED Complex I + Complex IV deficiency\n"
                "  • Clinical: LSFC — French-Canadian Leigh + EPISODIC CRISES + hepatopathy; "
                "    infantile/early-childhood onset; crises with stable intervals\n\n"
                "TACO1 (17q23.3) — NARROW, MT-CO1 only:\n"
                "  • Activates TRANSLATION of MT-CO1 mRNA specifically (mt-ribosome recruitment)\n"
                "  • MT-CO1 mRNA is present but poorly translated; other mt-mRNAs unaffected\n"
                "  • Biochemistry: ISOLATED Complex IV deficiency (I, II, III NORMAL)\n"
                "  • Clinical: CHILDHOOD-ONSET (3-8yr, LATER) + Dysarthria CARDINAL + progressive "
                "    ataxia; NO episodic crisis pattern; milder, more stable decline\n\n"
                "Diagnostic algorithm:\n"
                "  • Leigh + Combined CI+CIV + EPISODIC CRISES + ± hepatopathy → LRPPRC\n"
                "  • Leigh + Isolated CIV + CHILDHOOD onset + Dysarthria (NO crises) → TACO1\n"
                "  • Leigh + Isolated CIV + Infantile + respiratory/HCM → SURF1/SCO2/COX15\n"
                "  • Leigh + Isolated CIV + Infantile + tubulopathy → COX10\n\n"
                "mt-RNA biology DDx:\n"
                "  LRPPRC → mt-mRNA stabilisation (poly-A) — post-transcriptional global\n"
                "  TACO1  → mt-mRNA translational activation — translational MT-CO1 specific\n"
                "  MTIF2  → mt-translation initiation factor (a different entry point)"
            ),
        },
        {
            "term": "LSFC Founder Mutation p.Ala354Val — Saguenay-Lac-Saint-Jean Quebec",
            "definition": (
                "The Leigh Syndrome French-Canadian type (LSFC) founder mutation:\n\n"
                "p.Ala354Val (c.1061C>T) in LRPPRC:\n"
                "  • Located in PPR-3 repeat domain (N-terminal region of the PPR superhelix)\n"
                "  • Reduces LRPPRC protein stability (~25-40% of normal protein level retained)\n"
                "  • DOES NOT abolish LRPPRC entirely → explains episodic stable phenotype "
                "    (as opposed to continuous severe disease in complete-null alleles)\n"
                "  • Founder effect: Saguenay-Lac-Saint-Jean region, Quebec, Canada:\n"
                "    - Carrier frequency: ~1/40 in SLSJ (vs <1/1000 worldwide)\n"
                "    - Disease frequency: ~1/2000 births in SLSJ\n"
                "    - Homozygous p.Ala354Val: moderate severity; episodic crises; "
                "      some patients survive into adulthood\n"
                "    - Compound het (p.Ala354Val / null truncating): more severe\n\n"
                "Gene identification:\n"
                "  • Mootha VK et al. (2003) Nat Genet 33:192: mapped LSFC to LRPPRC by "
                "    positional cloning in the genetically isolated SLSJ population\n"
                "  • Merante F et al. (1993) Mol Genet Metab 49:185: clinical description LSFC\n\n"
                "Non-SLSJ LRPPRC patients:\n"
                "  • Worldwide patients now reported with compound het missense/truncating\n"
                "  • Similar phenotype: Leigh + episodic crises + combined CI+CIV\n"
                "  • Gene panels and WES identify LRPPRC outside SLSJ population"
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "LRPPRC Genotype–Phenotype — Founder Allele and Null Allele Series",
            "definition": (
                "LRPPRC allele severity determines LSFC clinical severity:\n\n"
                "p.Ala354Val / p.Ala354Val (homozygous founder):\n"
                "  • 25-40% residual LRPPRC protein; partial mt-mRNA stabilisation retained\n"
                "  • Moderate severity: episodic crises (1-3/year), stable between crises\n"
                "  • Some patients survive into adolescence or adulthood\n"
                "  • Mt-mRNA levels reduced but not absent: 30-50% of normal\n\n"
                "p.Ala354Val / null (compound het, most SLSJ severely affected):\n"
                "  • One allele: 25-40% protein; one allele: no contribution\n"
                "  • More severe: crises more frequent (3-5+/year), often more severe\n"
                "  • Higher mortality in first decade than homozygous founder\n\n"
                "Null / null (biallelic truncating, non-SLSJ):\n"
                "  • No residual LRPPRC protein\n"
                "  • Most severe: early and severe lactic acidosis; high neonatal/infantile mortality\n"
                "  • mt-mRNA levels very severely depleted\n\n"
                "General genotype–phenotype principle:\n"
                "  • More residual LRPPRC function → more stable mt-mRNA → milder, episodic phenotype\n"
                "  • Complete absence → continuous severe disease\n"
                "  • ALL LRPPRC patients: AVOID VPA, metformin, linezolid, propofol, KD, fasting"
            ),
        },
        {
            "term": "Combined Complex I + Complex IV Deficiency — LRPPRC Biochemical Signature",
            "definition": (
                "The biochemical fingerprint of LRPPRC/LSFC distinguishes it from all other "
                "COX-deficiency diseases:\n\n"
                "LRPPRC/LSFC (COMBINED):\n"
                "  • Complex I (CI): 20-40% of control — REDUCED (ND1-6 subunits depleted)\n"
                "  • Complex IV (CIV): 15-35% of control — REDUCED (COX1-3 subunits depleted)\n"
                "  • Complex II: NORMAL (100%) — nuclear-encoded (SDHA/B/C/D unaffected by LRPPRC)\n"
                "  • Complex III: variable, mildly reduced (MT-CYB reduced, nuclear subunits intact)\n"
                "  • Complex V: partially reduced (MT-ATP6/8 affected)\n\n"
                "Isolated COX-deficiency diseases (for comparison):\n"
                "  • SURF1, SCO1, SCO2, COX10, COX15, TACO1: Complex I NORMAL\n"
                "  • Only CIV reduced; CI remains fully intact\n\n"
                "Diagnostic implication:\n"
                "  • Muscle biopsy + respiratory chain enzyme analysis:\n"
                "    - CI+CIV both low → LRPPRC (or POLG, TWNK, MPV17 — check mtDNA depletion)\n"
                "    - CIV only low + CI normal → SURF1/SCO2/COX10/COX15/TACO1\n"
                "  • Northern blot / mt-mRNA quantitation: all 13 mt-mRNAs reduced → LRPPRC\n"
                "  • Gene panel: confirm with LRPPRC biallelic variants"
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "LSFC — Disease Natural History and Crisis Mortality Risk",
            "definition": (
                "LSFC natural history based on SLSJ cohort studies (Falk 2013, Morin 2011):\n\n"
                "Onset:\n"
                "  • 80% present by 12-18 months of age\n"
                "  • Initial: hypotonia, poor feeding, mild developmental delay\n"
                "  • First crisis: often triggered by first febrile illness\n\n"
                "Episodic crisis pattern:\n"
                "  • Frequency: 1-5+ crises/year — highly variable\n"
                "  • Each crisis: median 5-10 days of acute illness; ICU admission often needed\n"
                "  • Crisis mortality: ~15-20% per untreated crisis; reduced by aggressive IV glucose\n"
                "  • Between crises: 60-70% have minimal residual deficits initially; "
                "    progressive neurological damage accumulates with each crisis\n\n"
                "Long-term trajectory:\n"
                "  • Survivors accumulate neurological deficits: developmental regression, "
                "    cognitive impairment, ataxia, seizures\n"
                "  • ~30-35% die in first 2 years (crisis-related)\n"
                "  • ~40-50% survive to adolescence; increasing disability between crises\n"
                "  • Some SLSJ p.Ala354Val homozygous patients survive to 3rd–4th decade "
                "    (rare but documented); requires intensive crisis prevention\n\n"
                "Factors that improve prognosis:\n"
                "  1. Immediate IV glucose for ANY febrile illness\n"
                "  2. Metabolic emergency card carried at all times\n"
                "  3. Metabolic specialist coordination with local ER\n"
                "  4. Fever prevention (immunisations, flu vaccine, mask in viral season)"
            ),
        },
        {
            "term": "LRPPRC DDx from SCO1 Hepatopathy — Both Have Liver Involvement, Opposite Severity",
            "definition": (
                "Hepatic involvement in LRPPRC/LSFC vs SCO1 — critical differential:\n\n"
                "LRPPRC/LSFC — MILD-MODERATE HEPATOPATHY (~45%):\n"
                "  • Hepatomegaly: present in some, not universal\n"
                "  • Liver enzymes: mildly to moderately elevated during crises\n"
                "  • No coagulopathy at baseline\n"
                "  • NOT the dominant feature — crisis encephalopathy dominates\n"
                "  • Does NOT cause neonatal fulminant hepatic failure\n\n"
                "SCO1 — NEONATAL HEPATIC FAILURE (100%):\n"
                "  • CARDINAL FEATURE: severe hepatomegaly, coagulopathy, cholestasis "
                "    from day 1-7 of life (neonatal onset)\n"
                "  • 85% die in first year from hepatic + neurological failure\n"
                "  • Liver is the PRIMARY failing organ, not the brain\n"
                "  • Isolated COX (Complex IV) deficiency — NOT combined CI+CIV\n\n"
                "Algorithm:\n"
                "  Combined CI+CIV + Leigh + episodic crises + mild hepatopathy → LRPPRC\n"
                "  Isolated CIV + NEONATAL HEPATIC FAILURE 100% + coagulopathy → SCO1\n\n"
                "Additional DDx point:\n"
                "  • DGUOK, MPV17, POLG cause hepatocerebral mtDNA depletion syndrome "
                "    with both liver + brain and mtDNA depletion — check mtDNA copy number "
                "    before assuming LRPPRC (LRPPRC does NOT deplete mtDNA)"
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "LEV — Preferred AED in LRPPRC/LSFC Disease",
            "definition": (
                "Levetiracetam (LEV) is the AED of choice in LRPPRC/LSFC:\n\n"
                "1. RENAL excretion: 66% unchanged via kidney; no hepatic CYP metabolism\n"
                "   → Critical in LRPPRC since patients have mild-moderate hepatopathy (~45%)\n"
                "   → No hepatotoxic metabolites; no CYP induction\n"
                "2. No mitochondrial toxicity: LEV does not inhibit CI, CII, CIII, or CIV\n"
                "   → Safe in combined CI+CIV deficiency\n"
                "3. Cardiac-safe: no conduction effects\n"
                "4. IV formulation: essential during LSFC metabolic crisis with seizures\n"
                "5. Broad-spectrum: focal + generalised + myoclonic\n\n"
                "Crisis seizure protocol:\n"
                "  • LEV IV loading: 20-40 mg/kg over 15 minutes\n"
                "  • SIMULTANEOUS: IV dextrose GIR 6-8 — treat the crisis, not just the seizure\n"
                "  • NEVER VPA (ABSOLUTE CI: CoA sequestration + hepatotoxicity + POLG)\n\n"
                "AVOID in LRPPRC:\n"
                "  • VPA: ABSOLUTE CI (see above)\n"
                "  • Phenobarbital: Complex I inhibitor — further reduces already-deficient CI\n"
                "  • Phenytoin: hepatic metabolism (risky with hepatopathy)\n"
                "  • Carbamazepine: CYP3A4 inducer; hepatic; avoid in hepatopathy"
            ),
        },
        {
            "term": "Crisis Prevention Protocol — Fever + Fasting Management in LSFC",
            "definition": (
                "The single most important intervention in LRPPRC/LSFC is crisis PREVENTION:\n\n"
                "FEVER MANAGEMENT (crisis trigger #1):\n"
                "  • Acetaminophen: 15 mg/kg q4-6h — first-line fever reduction\n"
                "  • Ibuprofen: 10 mg/kg q6-8h — add if acetaminophen insufficient\n"
                "    (caution with hepatopathy; avoid if elevated baseline transaminases)\n"
                "  • Cooling blanket: if temperature >38.5°C despite antipyretics\n"
                "  • Any fever >38°C in LSFC child: IV glucose initiated within 1 hour\n\n"
                "GLUCOSE MAINTENANCE (crisis trigger #2 = fasting):\n"
                "  • Between crises: high-carbohydrate oral diet; avoid >2h fasting\n"
                "  • Pre-procedure: cornstarch supplement 1g/kg at bedtime if early NPO\n"
                "  • Any GI illness → IV dextrose GIR 6-8 mg/kg/min IMMEDIATELY\n"
                "  • Target blood glucose: 5-10 mmol/L during illness\n\n"
                "IMMUNISATION STRATEGY:\n"
                "  • All routine vaccines — recommended and accelerated schedule if possible\n"
                "  • Annual influenza vaccine (flu is common crisis trigger)\n"
                "  • RSV prophylaxis (palivizumab) in infants (<2y)\n"
                "  • COVID-19 vaccine for patients >12 months\n\n"
                "EMERGENCY CARD PROTOCOL:\n"
                "  • LSFC Emergency Card carried at all times\n"
                "  • ER instructions: IV dextrose GIR 6-8 immediately; LEV if seizures; "
                "    NEVER VPA, propofol, linezolid, fasting\n"
                "  • Metabolic centre phone number on card (24h oncall)"
            ),
        },
        {
            "term": "CoQ10/Ubiquinol, Riboflavin, Thiamine, Biotin — Cofactor Therapy in LRPPRC",
            "definition": (
                "Standard mitochondrial cofactor therapy in LRPPRC (all Level C evidence):\n\n"
                "CoQ10 / Ubiquinol:\n"
                "  • 300-600 mg/day adults; 10-30 mg/kg/day children; ubiquinol preferred\n"
                "  • Mobile electron carrier: CI → ubiquinone → CIII → cytochrome c → CIV\n"
                "  • Supplemental CoQ10 may enhance residual respiratory chain function "
                "    between crises; evidence remains Level C\n\n"
                "Riboflavin (B2): 100-400 mg/day\n"
                "  • FMN and FAD: essential cofactors for both Complex I (FMN-N1) and "
                "    Complex II (FAD-SDHA) — particularly relevant given CI+CIV combined defect\n\n"
                "Thiamine (B1): 100-300 mg/day — MANDATORY empiric in ALL Leigh\n"
                "  • PDH and alpha-KGDH cofactor; SLC19A3 (THTR2) deficiency mimics Leigh\n"
                "  • Give empirically until molecular diagnosis confirmed\n\n"
                "Biotin: 5-20 mg/day — MANDATORY empiric\n"
                "  • Biotinidase deficiency (BTD) is a CURABLE Leigh mimic\n"
                "  • Give empirically until biotinidase enzyme activity confirmed\n\n"
                "Carnitine: 50-100 mg/kg/day\n"
                "  • Secondary carnitine deficiency common in OXPHOS and during crises\n\n"
                "Note: None of these cofactors prevent or abort acute metabolic crises. "
                "IV glucose remains the only evidence-based crisis intervention. "
                "Cofactors aim to optimise inter-crisis OXPHOS function."
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2)[:2000])
    print("\n=== BREAKDOWN (patients[:3]) ===")
    bk = get_breakdown()
    print(json.dumps({"patients": bk["patients"][:3], "feature_frequencies": bk["feature_frequencies"]}, indent=2))
    print("\n=== DEFINITIONS (first term) ===")
    df = get_definitions()
    print(df["pharmacology"][0]["term"])
