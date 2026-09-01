#!/usr/bin/env python3
"""MELAS — Mitochondrial Encephalomyopathy, Lactic Acidosis, and Stroke-like Episodes.

Primary mtDNA disease — maternal inheritance:
  MELAS Syndrome  OMIM #540000
  Primary gene: MT-TL1 OMIM *590050 (mitochondrial tRNA-Leucine-1-UUA/UUG)
  Primary mutation: m.3243A>G (~80%); also m.3271T>C (~10%), m.3252A>G (~5%), others (~5%)

PATHOPHYSIOLOGY (tRNA-Leu / multi-complex OXPHOS defect):
MT-TL1 encodes mitochondrial tRNA-Leu(UUA/UUG). This tRNA is required for translation
of all 13 mtDNA-encoded OXPHOS subunits (7 Complex I, 1 Complex III, 3 Complex IV,
2 Complex V). The m.3243A>G mutation in the anticodon stem-loop impairs tRNA:
  1. WOBBLE MODIFICATION — loss of 5-taurinomethyluridine (tm5U) at the wobble position,
     so UUA and UUG Leu codons are mistranslated → defective OXPHOS subunit assembly
  2. tRNA STABILITY — reduced steady-state tRNA-Leu level → global OXPHOS translation
     failure proportional to heteroplasmy burden
  3. COMPLEX I + IV MOST AFFECTED — CI loses ND subunits; CIV loses CO subunits;
     mitochondrial membrane potential collapses → ATP synthesis fails

STROKE-LIKE EPISODE (SLE) MECHANISM — NOT ischemic:
SLEs are caused by neuronal hyperexcitability (seizures drive metabolic demand beyond
OXPHOS capacity) and vasoregulatory failure (mtDNA-mutant vascular smooth muscle and
endothelium cannot sustain NO-mediated vasodilation → neuronal energy failure in
cortical areas with highest metabolic demand: occipital/parietal cortex).
Key distinction from ischemic stroke: SLE crosses arterial territories, is T2/FLAIR-bright
in cortex without restricted diffusion in acute phase (pseudo-infarct; DWI positive later),
and spreads/migrates over days.

HETEROPLASMY THRESHOLD:
  <60% mutant  → Asymptomatic or subclinical (MIDD isolated / minor features)
  60-80% mutant → Full MELAS triad (SLE + Encephalopathy + Lactic Acidosis)
  >80% mutant  → Severe / early-onset MELAS, high SLE frequency
Blood heteroplasmy UNDERESTIMATES true neuronal/muscle heteroplasmy — urine epithelial
cells or muscle biopsy are more accurate for tissue burden.

MELAS CLINICAL PRESENTATION (OMIM #540000):
  1. Stroke-like episodes (SLE) — 100% (defining criterion); occipital/parietal dominant;
     scotoma, cortical blindness, hemiplegia; MRI: T2/FLAIR cortical ± subcortical; migrates;
     DWI: cytotoxic/vasogenic mixed; precipitated by infection, fasting, metabolic stress
  2. Encephalopathy — 95%; recurrent headaches (migraine-like), vomiting; progressive
     cognitive decline; dementia in late-stage; cortical atrophy on serial MRI
  3. Lactic acidosis — 95%; elevated plasma/CSF lactate (>2.5 mmol/L); L:P ratio >20;
     elevated alanine; acute metabolic crisis can be life-threatening
  4. Seizures — 70%; focal occipital > GTCS; EEG: focal slowing, epileptiform discharges;
     often electroclinical at onset of SLE; triggers for SLE
  5. Sensorineural hearing loss (SNHL) — 75%; high-frequency first; bilateral progressive;
     may predate SLE by years; cochlear OXPHOS failure; cochlear implant effective
  6. Diabetes mellitus — 45%; maternally inherited diabetes and deafness (MIDD subtype);
     m.3243A>G accounts for 1-2% of all diabetes; typically thin, lean DM
  7. Short stature — 60%; growth hormone insufficiency + myopathy
  8. Myopathy — 70%; exercise intolerance, proximal weakness; ragged red fibers (RRF)
     on Gomori trichrome (succinate dehydrogenase-rich mitochondrial accumulation);
     COX-negative fibers; EMG: myopathic pattern; CK mildly elevated
  9. Cardiomyopathy — 30%; Wolff-Parkinson-White (WPW) pattern 15%; HCM 15%;
     conduction defects; regular cardiology monitoring with ECG + echo
  10. Peripheral neuropathy — 40%; sensory > motor; axonal; EMG-verified
  11. Psychiatric — 35%; anxiety, depression, psychosis; secondary to cortical atrophy

MIDD (Maternally Inherited Diabetes and Deafness) — MELAS SPECTRUM:
m.3243A>G at low heteroplasmy (<60%) can cause isolated MIDD without full MELAS triad;
accounts for 1-2% of all type 2 diabetes in general population (unrecognized).

KEY DIFFERENTIAL DIAGNOSIS:
  1. Ischemic stroke (young) — arterial territory, restricted DWI acute, no lactic acidosis,
     no ragged red fibers, no maternal inheritance, no SNHL
  2. CADASIL (NOTCH3) — white matter disease, migraine, no lactic acidosis, no RRF,
     AD inheritance, GOM deposits on skin biopsy
  3. MERRF (MT-TK) — myoclonic epilepsy + RRF; NO SLE; NO lactic acidosis crisis;
     ataxia dominant; cerebellar atrophy; m.8344A>G
  4. Mitochondrial stroke (POLG/Alpers) — hepatopathy 80%, VPA-CI same reason
  5. Encephalitis — fever, CSF pleocytosis, no lactic acidosis, no maternal pattern
  6. MSIIT (MT-ND5 m.13513G>A) — overlap MELAS + Leigh + LHON; ND5 gene

ABSOLUTE DRUG CONTRAINDICATIONS:
  VPA (valproate) — ABSOLUTE CI ALL MELAS:
    Mechanism: (a) CoA sequestration as valproyl-CoA → depletes free CoA required for
    PDH/KGDH (already impaired in MELAS) → blocks TCA cycle → worsens lactic acidosis;
    (b) POLG1 inhibition → worsens mtDNA depletion; (c) direct hepatotoxicity compounded
    by mitochondrial liver failure. Alternative: LEV (preferred), CLB, PER, ZNS.
  Metformin — ABSOLUTE CI ALL MELAS (including MIDD diabetes):
    Mechanism: Complex I inhibition → worsens OXPHOS failure → fatal lactic acidosis.
    L:P ratio already elevated; metformin stacks on pre-existing lactic acidosis risk.
    Use insulin, glipizide, or DPP-4 inhibitors instead. NEVER metformin in any
    confirmed or suspected m.3243A>G carrier.
  Linezolid — ABSOLUTE CI (all mitochondrial diseases):
    Mechanism: Inhibits mitochondrial 23S rRNA → blocks mtDNA-encoded protein synthesis
    → depletes all OXPHOS complexes. Causes DION, optic neuropathy, lactic acidosis.
    Use alternative antibiotics; if Gram+ MRSA is unavoidable, prefer daptomycin/tigecycline.

DRUGS TO AVOID:
  Propofol — AVOID (PRIS): propofol infusion syndrome worsened by impaired FAO in
    mitochondrial disease; Complex I inhibition compounds energy failure; use sevoflurane.
  Statins — HIGH CAUTION (CoQ10 depletion): worsen mitochondrial myopathy + rhabdomyolysis;
    if cardiovascular indication imperative, use lowest dose + CoQ10 300-600mg + CK monitoring.
  Aminoglycosides — AVOID: cochlear OXPHOS failure amplified; worsens SNHL irreversibly.
  Chloramphenicol — AVOID: mitochondrial 23S rRNA inhibitor (same mechanism as linezolid).
  Beta-blockers — CAUTION: block catecholamine-mediated mitochondrial biogenesis; reduce
    exercise tolerance; avoid in MELAS unless strong cardiac indication.
  Barbiturates (phenobarbital) — AVOID: Complex I inhibition; worsens mitochondrial energy crisis.
  Glucocorticoids — CAUTION: can precipitate hyperglycaemia (MIDD-DM); impair mito biogenesis.
  CBZ/OXC/PHT — AVOID: hepatotoxic Na-channel blockers; impair OXPHOS; PHT reduces CoQ10.

TREATMENTS (evidence-based):
  L-Arginine (IV + oral) — Level B: nitric oxide (NO) precursor; vasodilates cortical
    vessels during SLE (Koga 2005 Neurology RCT; El-Hattab 2012 Ann Neurol).
    Acute SLE: 0.5g/kg IV over 30-60 min (repeat q8-12h during acute phase).
    Maintenance: 0.15-0.30g/kg/day oral (monitor for hyperammonaemia at high doses).
    Mechanism: L-arginine → eNOS/nNOS → NO → cGMP → VSM relaxation → SLE attenuation.
  L-Citrulline — Level C: upstream NO precursor; better bioavailability (bypasses hepatic
    arginase); 0.2-0.3g/kg/day; alternative to L-arginine for maintenance.
  CoQ10 / Ubiquinol — Level C: electron carrier; bypass mechanism between CI→CIII;
    300-1200mg/day in divided doses; ubiquinol (reduced form) preferred; Shoffner 1989.
  Riboflavin (B2) — Level C: FMN/FAD cofactor for Complex I and II; 100-400mg/day.
  Thiamine (B1) — Level C: PDH/alpha-KGDH cofactor; reduces lactic acidosis burden; 100-200mg/day.
  L-Carnitine — Level C: long-chain FA transport; 50-100mg/kg/day; helps with myopathy.
  Taurine — Level B (MELAS-specific): tRNA modification at wobble position (5-taurinomethyluridine
    synthesis requires taurine); some evidence for SLE frequency reduction (Rikimaru 2012);
    3-6g/day; low toxicity profile; mechanism is disease-specific (not general mito supplement).
  LEV (levetiracetam) — Preferred AED: renal excretion, no mito toxicity, minimal
    drug interactions; SV2A mechanism; 1000-3000mg/day in divided doses.
  CLB (clobazam) — Alternative AED: CYP3A4/CYP2C19; no hepatic mito toxicity; 10-30mg/day.
  Perampanel — Level C add-on: AMPA receptor antagonist; hepatic but no mito toxicity.
  Insulin — for MIDD diabetes (metformin ABSOLUTE CI); target HbA1c <7.5%.
  Acute crisis protocol: IV dextrose GIR 8-10, IV arginine 0.5g/kg, thiamine 100mg IV,
    avoid fasting >4h, avoid hypotension, treat trigger infections aggressively, ICU monitoring.

KEY REFERENCES:
  Goto 1990 (Biochem Biophys Res Commun) — Discovery of m.3243A>G in MELAS
  Pavlakis 1984 (Ann Neurol) — First MELAS clinical description
  Koga 2005 (Neurology) — L-arginine RCT for acute SLE (Level B)
  El-Hattab 2012 (Ann Neurol) — L-arginine comprehensive MELAS review
  Goto 1994 (Nat Genet) — MT-TL1 tRNA modification defect mechanism
  DiMauro 2004 (Mitochondrion) — mtDNA disease review
  van der Knaap 1998 (AJNR) — MELAS MRI stroke-like episode characterisation
  Rikimaru 2012 (J Neurol) — Taurine supplementation MELAS
"""
from __future__ import annotations
import random
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────
SEED        = 587
N_PATIENTS  = 40
DISEASE     = "MELAS (Mitochondrial Encephalomyopathy, Lactic Acidosis, Stroke-like Episodes)"
GENE        = "MT-TL1"
OMIM_GENE   = "*590050"
OMIM_DIS    = "#540000"
LOCUS       = "Mitochondrial DNA (mtDNA) — maternal inheritance"
PROTEIN     = "tRNA-Leu(UUA/UUG) — 72 nucleotides, anticodon stem-loop, wobble modification"

# Mutation pool
MUTATION_POOL    = [
    "m.3243A>G (p.tRNA-Leu — anticodon stem-loop, wobble-U34 modification loss)",
    "m.3271T>C (p.tRNA-Leu — discriminator stem, similar mechanism, milder)",
    "m.3252A>G (p.tRNA-Leu — anticodon loop, least common primary MELAS)",
    "Other MT-TL1 / MT-ND5 variants",
]
MUTATION_WEIGHTS = [0.80, 0.10, 0.05, 0.05]

# Phenotype strata
PHENO_FULL   = "Full MELAS Triad (SLE + Encephalopathy + Lactic Acidosis)"
PHENO_PART   = "Partial / Oligosymptomatic MELAS"
PHENO_MIDD   = "MIDD (Maternally Inherited Diabetes and Deafness)"


def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient MELAS cohort (seed-587)."""
    return random.Random(SEED)


def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient MT-TL1 MELAS disease cohort (seed-587).

    ~70% Full MELAS triad, ~20% Partial, ~10% MIDD.
    """
    patients = []
    for i in range(N_PATIENTS):
        pid = f"MELAS-{i+1:03d}"
        het = rng.uniform(40, 98)   # blood heteroplasmy %
        # Phenotype by heteroplasmy
        if het >= 70:
            phenotype = PHENO_FULL
        elif het >= 50:
            phenotype = PHENO_PART
        else:
            phenotype = PHENO_MIDD

        mut = rng.choices(MUTATION_POOL, weights=MUTATION_WEIGHTS)[0]
        sex = "F" if rng.random() < 0.55 else "M"   # slight F predominance in MELAS series
        age_dx = rng.gauss(28, 12)
        age_dx = max(4, min(65, age_dx))

        # Clinical features
        sle             = phenotype == PHENO_FULL or rng.random() < 0.30
        encephalopathy  = sle or rng.random() < 0.60
        lactic_acidosis = rng.random() < (0.95 if phenotype == PHENO_FULL else 0.55 if phenotype == PHENO_PART else 0.20)
        seizures        = rng.random() < (0.70 if phenotype == PHENO_FULL else 0.35 if phenotype == PHENO_PART else 0.10)
        snhl            = rng.random() < 0.75
        diabetes        = rng.random() < (0.40 if phenotype == PHENO_FULL else 0.50 if phenotype == PHENO_PART else 0.80)
        short_stature   = rng.random() < (0.60 if phenotype != PHENO_MIDD else 0.30)
        myopathy        = rng.random() < (0.70 if phenotype == PHENO_FULL else 0.45 if phenotype == PHENO_PART else 0.20)
        cardiomyopathy  = rng.random() < (0.30 if phenotype == PHENO_FULL else 0.15 if phenotype == PHENO_PART else 0.10)
        neuropathy      = rng.random() < (0.40 if phenotype == PHENO_FULL else 0.20 if phenotype == PHENO_PART else 0.10)
        psychiatric     = rng.random() < (0.35 if phenotype == PHENO_FULL else 0.20 if phenotype == PHENO_PART else 0.10)

        # Treatment
        on_arginine    = sle and rng.random() < 0.75
        on_citrulline  = rng.random() < 0.40
        on_coq10       = rng.random() < 0.70
        on_riboflavin  = rng.random() < 0.55
        on_thiamine    = rng.random() < 0.50
        on_taurine     = rng.random() < 0.45
        on_carnitine   = rng.random() < 0.50
        on_lev         = seizures and rng.random() < 0.75
        vpa_exposure   = rng.random() < 0.06   # inadvertent VPA — always adverse event
        met_exposure   = diabetes and rng.random() < 0.08  # metformin before diagnosis — always adverse
        insulin_use    = diabetes and rng.random() < 0.80

        n_sle = rng.randint(1, 8) if sle else 0
        patients.append({
            "id": pid,
            "mutation": mut,
            "heteroplasmy_pct": round(het, 1),
            "phenotype": phenotype,
            "sex": sex,
            "age_at_dx": round(age_dx, 1),
            "sle": sle,
            "n_sle": n_sle,
            "encephalopathy": encephalopathy,
            "lactic_acidosis": lactic_acidosis,
            "seizures": seizures,
            "snhl": snhl,
            "diabetes": diabetes,
            "short_stature": short_stature,
            "myopathy": myopathy,
            "cardiomyopathy": cardiomyopathy,
            "neuropathy": neuropathy,
            "psychiatric": psychiatric,
            "on_arginine": on_arginine,
            "on_citrulline": on_citrulline,
            "on_coq10": on_coq10,
            "on_riboflavin": on_riboflavin,
            "on_thiamine": on_thiamine,
            "on_taurine": on_taurine,
            "on_carnitine": on_carnitine,
            "on_lev": on_lev,
            "vpa_exposure": vpa_exposure,
            "met_exposure": met_exposure,
            "insulin_use": insulin_use,
        })
    return patients


def get_overview() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)
    n = len(patients)

    n_full    = sum(1 for p in patients if p["phenotype"] == PHENO_FULL)
    n_part    = sum(1 for p in patients if p["phenotype"] == PHENO_PART)
    n_midd    = sum(1 for p in patients if p["phenotype"] == PHENO_MIDD)
    n_sle     = sum(1 for p in patients if p["sle"])
    n_la      = sum(1 for p in patients if p["lactic_acidosis"])
    n_snhl    = sum(1 for p in patients if p["snhl"])
    n_dm      = sum(1 for p in patients if p["diabetes"])
    n_myo     = sum(1 for p in patients if p["myopathy"])
    n_sei     = sum(1 for p in patients if p["seizures"])
    n_vpa     = sum(1 for p in patients if p["vpa_exposure"])
    n_met     = sum(1 for p in patients if p["met_exposure"])
    avg_dx    = sum(p["age_at_dx"] for p in patients) / n
    avg_het   = sum(p["heteroplasmy_pct"] for p in patients) / n

    return {
        "gene": GENE,
        "protein": PROTEIN,
        "disease": DISEASE,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DIS,
        "chromosome": LOCUS,
        "inheritance": "Maternal (mtDNA — ALL children of carrier mother inherit mutation; "
                       "paternal transmission does NOT occur for mtDNA)",
        "onset": "Childhood to young adult (mean ~28 yr in this cohort; range 4-65 yr; "
                 "neonatal onset possible in >90% heteroplasmy)",
        "mechanism": (
            "MT-TL1 encodes mitochondrial tRNA-Leu(UUA/UUG). The m.3243A>G mutation "
            "impairs two critical tRNA functions: (1) 5-taurinomethyluridine (tm5U) "
            "wobble modification is lost, causing mistranslation of UUA/UUG leucine "
            "codons in all 13 mtDNA-encoded OXPHOS subunits; (2) tRNA steady-state "
            "levels are reduced. Complex I (7 mtDNA subunits: ND1-3,4,4L,5,6) and "
            "Complex IV (3 subunits: CO1-3) are most severely affected, producing "
            "combined CI+CIV deficiency. Reduced mitochondrial membrane potential leads "
            "to cortical energy failure and impaired vascular NO synthesis, causing the "
            "distinctive stroke-like episodes that cross arterial boundaries."
        ),
        "mtdna_pattern": (
            "mtDNA POINT MUTATION (not deletion/rearrangement — key DDx vs. Pearson/KSS). "
            "m.3243A>G heteroplasmy in MT-TL1 (nucleotide 3243 of 16,569 mtDNA bp). "
            "Blood heteroplasmy UNDERESTIMATES tissue burden — urine epithelial cells "
            "or muscle biopsy preferred for accurate quantification. "
            "Threshold concept: <60% subclinical/MIDD; 60-80% full MELAS; >80% severe. "
            "Offspring heteroplasmy is unpredictable due to mitochondrial bottleneck."
        ),
        "cohort": f"40 patients · seed-{SEED}",
        "kpis": [
            {"label": "Full MELAS Triad", "value": f"{n_full}/{n}", "color": "#b71c1c"},
            {"label": "Partial / MIDD", "value": f"{n_part+n_midd}/{n}", "color": "#e65100"},
            {"label": "Stroke-like Episodes", "value": f"{n_sle}/{n}", "color": "#c62828"},
            {"label": "Lactic Acidosis", "value": f"{n_la}/{n}", "color": "#6a1b9a"},
            {"label": "SNHL", "value": f"{n_snhl}/{n}", "color": "#1565c0"},
            {"label": "Diabetes (DM)", "value": f"{n_dm}/{n}", "color": "#00695c"},
            {"label": "Myopathy (RRF)", "value": f"{n_myo}/{n}", "color": "#4e342e"},
            {"label": "Seizures", "value": f"{n_sei}/{n}", "color": "#283593"},
            {"label": "VPA Exposure (CI)", "value": str(n_vpa), "color": "#d50000"},
            {"label": "Metformin Exposure (CI)", "value": str(n_met), "color": "#bf360c"},
            {"label": "Mean Age at Dx (yr)", "value": f"{avg_dx:.1f}", "color": "#37474f"},
            {"label": "Mean Blood Heteroplasmy", "value": f"{avg_het:.1f}%", "color": "#558b2f"},
        ],
        "contraindications": [
            {
                "drug": "VPA (Valproate)",
                "severity": "ABSOLUTE CI — ALL MELAS/MT-TL1",
                "mechanism": "CoA sequestration (valproyl-CoA) → PDH/KGDH cofactor depletion → "
                             "worsens lactic acidosis; POLG inhibition → mtDNA depletion; "
                             "hepatotoxicity compounded by mitochondrial liver failure risk. "
                             "NEVER use VPA in any confirmed or suspected m.3243A>G carrier.",
            },
            {
                "drug": "Metformin",
                "severity": "ABSOLUTE CI — ALL MELAS (including MIDD)",
                "mechanism": "Direct Complex I inhibition → worsens OXPHOS failure → fatal "
                             "lactic acidosis. L:P ratio already elevated in MELAS. Use "
                             "insulin, glipizide, or DPP-4 inhibitors for MIDD-DM.",
            },
            {
                "drug": "Linezolid",
                "severity": "ABSOLUTE CI — All Mitochondrial Diseases",
                "mechanism": "Mitochondrial 23S rRNA inhibition → blocks mtDNA-encoded protein "
                             "synthesis → pan-OXPHOS depletion; causes DION + lactic acidosis. "
                             "Use daptomycin/tigecycline for MRSA if unavoidable.",
            },
            {
                "drug": "Propofol",
                "severity": "AVOID — PRIS Risk",
                "mechanism": "Propofol infusion syndrome: Complex I inhibition + impaired FAO "
                             "in mito disease → fatal cardiac failure. Use sevoflurane for GA.",
            },
            {
                "drug": "Statins",
                "severity": "HIGH CAUTION — CoQ10 Depletion",
                "mechanism": "HMG-CoA reductase inhibition depletes CoQ10 → worsens mito "
                             "myopathy + rhabdomyolysis risk. If essential: lowest dose + "
                             "CoQ10 300-600mg + monthly CK monitoring.",
            },
            {
                "drug": "Aminoglycosides",
                "severity": "AVOID — Cochlear OXPHOS Amplification",
                "mechanism": "Cochlear hair cell OXPHOS failure is amplified; worsens SNHL "
                             "irreversibly. Use alternative antibiotics.",
            },
            {
                "drug": "Chloramphenicol",
                "severity": "AVOID — Mitochondrial 23S rRNA",
                "mechanism": "Same mechanism as linezolid; avoid in all mitochondrial disease.",
            },
            {
                "drug": "Phenobarbital / Barbiturates",
                "severity": "AVOID — Complex I Inhibition",
                "mechanism": "Mitochondrial Complex I inhibition worsens energy crisis during SLE.",
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    rows = []
    for p in patients:
        feats = []
        if p["sle"]:           feats.append(f"SLE×{p['n_sle']}")
        if p["encephalopathy"]: feats.append("Enceph")
        if p["lactic_acidosis"]: feats.append("LA")
        if p["seizures"]:       feats.append("Sz")
        if p["snhl"]:           feats.append("SNHL")
        if p["diabetes"]:       feats.append("DM")
        if p["myopathy"]:       feats.append("Myo")
        if p["cardiomyopathy"]: feats.append("CM")
        if p["neuropathy"]:     feats.append("Neuropathy")
        if p["short_stature"]:  feats.append("Short")
        if p["psychiatric"]:    feats.append("Psych")
        txs = []
        if p["on_arginine"]:   txs.append("Arg")
        if p["on_citrulline"]: txs.append("Cit")
        if p["on_coq10"]:      txs.append("CoQ10")
        if p["on_riboflavin"]: txs.append("B2")
        if p["on_thiamine"]:   txs.append("B1")
        if p["on_taurine"]:    txs.append("Tau")
        if p["on_carnitine"]:  txs.append("Carnitine")
        if p["on_lev"]:        txs.append("LEV")
        alerts = []
        if p["vpa_exposure"]:  alerts.append("⚠ VPA-CI exposure")
        if p["met_exposure"]:  alerts.append("⚠ Metformin-CI exposure")
        rows.append({
            "id": p["id"],
            "mutation": p["mutation"][:40],
            "het": p["heteroplasmy_pct"],
            "phenotype": p["phenotype"][:30],
            "sex": p["sex"],
            "age_dx": p["age_at_dx"],
            "features": ", ".join(feats) or "—",
            "treatments": ", ".join(txs) or "—",
            "alerts": "; ".join(alerts) if alerts else "None",
        })

    n = len(patients)
    ff = {
        "Stroke-like Episodes (SLE)":  round(100 * sum(1 for p in patients if p["sle"]) / n),
        "Encephalopathy":              round(100 * sum(1 for p in patients if p["encephalopathy"]) / n),
        "Lactic Acidosis":             round(100 * sum(1 for p in patients if p["lactic_acidosis"]) / n),
        "Seizures":                    round(100 * sum(1 for p in patients if p["seizures"]) / n),
        "Sensorineural Hearing Loss":  round(100 * sum(1 for p in patients if p["snhl"]) / n),
        "Myopathy (RRF)":              round(100 * sum(1 for p in patients if p["myopathy"]) / n),
        "Diabetes Mellitus":           round(100 * sum(1 for p in patients if p["diabetes"]) / n),
        "Short Stature":               round(100 * sum(1 for p in patients if p["short_stature"]) / n),
        "Cardiomyopathy/WPW":          round(100 * sum(1 for p in patients if p["cardiomyopathy"]) / n),
        "Peripheral Neuropathy":       round(100 * sum(1 for p in patients if p["neuropathy"]) / n),
        "Psychiatric":                 round(100 * sum(1 for p in patients if p["psychiatric"]) / n),
    }
    return {"patients": rows, "feature_frequencies": ff}


def get_definitions() -> dict[str, Any]:
    return {
        "pharmacology": [
            {
                "term": "L-Arginine (IV + Oral) — Level B MELAS",
                "definition": (
                    "MECHANISM: L-arginine is the substrate for nitric oxide synthase (eNOS/nNOS) → NO → "
                    "soluble guanylate cyclase → cGMP → vascular smooth muscle relaxation. In MELAS, "
                    "vascular endothelium and smooth muscle carry the m.3243A>G mutation and cannot "
                    "maintain adequate NO synthesis during metabolic stress, contributing to the "
                    "vasoregulatory failure underlying stroke-like episodes (SLEs).\n"
                    "EVIDENCE: Koga 2005 (Neurology RCT) showed IV arginine during acute SLE reduces "
                    "neurological damage and SLE frequency. El-Hattab 2012 (Ann Neurol) extended the "
                    "evidence to oral maintenance.\n"
                    "DOSING: Acute SLE — 0.5g/kg IV over 30-60 min (repeat q8-12h during acute phase). "
                    "Maintenance — 0.15-0.30g/kg/day oral arginine or equivalent as L-citrulline "
                    "(0.2-0.3g/kg/day; better bioavailability).\n"
                    "SAFETY: Monitor ammonia at high doses (>0.5g/kg/day); arginine can cause "
                    "hyperammonaemia via urea cycle activation. Avoid in urea cycle defects."
                ),
            },
            {
                "term": "L-Citrulline — Level C MELAS",
                "definition": (
                    "MECHANISM: L-citrulline is converted to L-arginine in the kidney (bypasses hepatic "
                    "arginase), providing a sustained NO precursor supply with better oral bioavailability "
                    "than arginine. Preferred for maintenance therapy.\n"
                    "DOSING: 0.2-0.3g/kg/day oral in divided doses.\n"
                    "SAFETY: Lower risk of hyperammonaemia vs. arginine; GI tolerance generally good."
                ),
            },
            {
                "term": "Taurine — Level B MELAS (Disease-Specific Mechanism)",
                "definition": (
                    "MECHANISM: Taurine is required for biosynthesis of 5-taurinomethyluridine (tm5U) "
                    "at the wobble position (U34) of tRNA-Leu(UUA/UUG). The m.3243A>G mutation "
                    "impairs this modification. Taurine supplementation partially restores wobble "
                    "modification, improving translation fidelity of UUA/UUG codons in OXPHOS subunits. "
                    "This mechanism is specific to MT-TL1 MELAS — not a generic mitochondrial supplement.\n"
                    "EVIDENCE: Rikimaru 2012 (J Neurol): taurine 3g/day reduced SLE frequency "
                    "in a MELAS cohort; case series support.\n"
                    "DOSING: 3-6g/day oral in divided doses. Excellent safety profile.\n"
                    "NOTE: Only relevant for MT-TL1 mutations affecting the wobble-U34 modification."
                ),
            },
            {
                "term": "Valproate (VPA) — ABSOLUTE CONTRAINDICATION in ALL MELAS",
                "definition": (
                    "THREE INDEPENDENT MECHANISMS of harm in MELAS:\n"
                    "(1) CoA SEQUESTRATION: Valproate conjugates with CoA → valproyl-CoA; "
                    "depletes free CoA required as PDH and alpha-KGDH cofactors (already "
                    "impaired in MELAS due to CI dysfunction); worsens lactic acidosis acutely.\n"
                    "(2) POLG1 INHIBITION: VPA inhibits mtDNA polymerase gamma → worsens "
                    "mtDNA depletion; can trigger acute liver failure (Alpers phenocopy).\n"
                    "(3) HEPATOTOXICITY: Synergistic hepatotoxicity with mito liver failure.\n"
                    "ALTERNATIVE AEDs: LEV (preferred — renal excretion), CLB, perampanel, ZNS.\n"
                    "KEY POINT: MELAS-MIDD patients with diabetes who present to non-specialist "
                    "neurologists may be started on VPA for first seizure — always check for "
                    "maternal diabetes + deafness before prescribing VPA in any adult-onset seizure."
                ),
            },
            {
                "term": "Metformin — ABSOLUTE CONTRAINDICATION in ALL MELAS (including MIDD)",
                "definition": (
                    "MECHANISM: Metformin directly inhibits mitochondrial Complex I → reduces NADH "
                    "oxidation → worsens the existing OXPHOS failure in MELAS → fatal lactic acidosis.\n"
                    "CLINICAL SIGNIFICANCE: MIDD (maternally inherited diabetes and deafness) is a "
                    "common MELAS spectrum manifestation; such patients often present to endocrinology "
                    "rather than neurology and may receive metformin as first-line type-2 diabetes "
                    "treatment WITHOUT recognition of their mitochondrial diagnosis.\n"
                    "Metformin may precipitate life-threatening lactic acidosis (L >10 mmol/L) "
                    "in an m.3243A>G carrier with any concurrent infection, dehydration, or "
                    "contrast medium exposure.\n"
                    "ALTERNATIVES: Insulin (safest); glipizide/gliclazide (sulphonylurea); "
                    "DPP-4 inhibitors (sitagliptin — no mito toxicity); GLP-1 analogues (weight benefit).\n"
                    "SCREENING: Any patient with maternal-pattern type-2 DM + sensorineural hearing "
                    "loss + thin/lean body habitus → test m.3243A>G IMMEDIATELY before starting metformin."
                ),
            },
            {
                "term": "Levetiracetam (LEV) — Preferred AED in MELAS",
                "definition": (
                    "Levetiracetam is the preferred antiepileptic drug in MELAS:\n"
                    "(1) RENAL EXCRETION (66% unchanged renal; 27% hepatic hydrolysis) → no "
                    "cytochrome P450 induction; no mitochondrial hepatotoxicity\n"
                    "(2) SV2A MECHANISM → reduces synaptic vesicle release; no GABA/sodium "
                    "channel interference; no complex I inhibition\n"
                    "(3) NO COA SEQUESTRATION → does not worsen lactic acidosis\n"
                    "(4) IV FORMULATION available for acute seizure/SLE control\n"
                    "DOSING: 500-1500mg BD (1000-3000mg/day); titrate by 500mg q2-4 weeks.\n"
                    "ADVERSE EFFECTS: Irritability, mood disturbance (add B6 pyridoxine 50mg/day "
                    "if behavioural side effects occur)."
                ),
            },
            {
                "term": "CoQ10/Ubiquinol — Level C MELAS",
                "definition": (
                    "Coenzyme Q10 (ubiquinone/ubiquinol) shuttles electrons from CI and CII to CIII, "
                    "providing a bypass mechanism that partially compensates for CI/CIV deficiency.\n"
                    "DOSING: 300-1200mg/day ubiquinol (reduced form; better absorption) in divided doses.\n"
                    "EVIDENCE: Level C (observational + case series; no RCT in MELAS-specific cohort).\n"
                    "SAFETY: Excellent profile; may reduce warfarin efficacy (monitor INR). Statins "
                    "deplete CoQ10 — mandatory supplementation if statin co-prescribed."
                ),
            },
            {
                "term": "Propofol — AVOID (PRIS) in MELAS",
                "definition": (
                    "Propofol Infusion Syndrome (PRIS): propofol inhibits mitochondrial Complex I "
                    "and impairs long-chain fatty acid beta-oxidation. In mitochondrial disease, "
                    "these pathways are already compromised; propofol at anaesthetic doses (>4mg/kg/h "
                    "for >48h) can cause fatal PRIS: metabolic acidosis, rhabdomyolysis, arrhythmia, "
                    "cardiac failure.\n"
                    "ALTERNATIVE: Sevoflurane or isoflurane for general anaesthesia.\n"
                    "NOTE: Even single-dose induction doses are used cautiously in MELAS; brief "
                    "induction may be permissible with careful monitoring, but AVOID infusion."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "MT-TL1 Gene — tRNA-Leu(UUA/UUG)",
                "definition": (
                    "MT-TL1 encodes mitochondrial tRNA-Leucine-1, which decodes UUA and UUG "
                    "leucine codons in all 13 mtDNA-encoded OXPHOS subunits.\n"
                    "STRUCTURE: 72-nucleotide cloverleaf secondary structure; anticodon stem-loop "
                    "contains U34 (wobble position) which requires 5-taurinomethyluridine (tm5U) "
                    "modification for faithful decoding.\n"
                    "LOCATION: mtDNA nucleotide 3230-3304 (light strand).\n"
                    "m.3243A>G: Most common MELAS mutation (80%); disrupts tm5U34 modification "
                    "and reduces steady-state tRNA level → global OXPHOS translation failure.\n"
                    "DISTINCT FROM MT-TL2: MT-TL2 (tRNA-Leu-CUN) at nucleotides 12266-12336 "
                    "causes a different, rarer mitochondrial myopathy (CPEO-associated)."
                ),
            },
            {
                "term": "Heteroplasmy and Tissue Threshold",
                "definition": (
                    "HETEROPLASMY: Co-existence of wild-type and mutant mtDNA copies within a "
                    "single cell. The ratio determines whether the energetic threshold for "
                    "cellular dysfunction is exceeded.\n"
                    "THRESHOLD CONCEPT: <60% mutant → subclinical (MIDD); 60-80% → MELAS; "
                    ">80% → severe MELAS. IMPORTANT: This is tissue-specific — blood often "
                    "shows LOWER heteroplasmy than muscle, brain, or renal tubular cells.\n"
                    "DIAGNOSTIC IMPLICATION: A negative blood result does NOT exclude MELAS. "
                    "Test urine pellet, muscle biopsy, or buccal cells if clinical suspicion high.\n"
                    "DRIFT: Blood heteroplasmy declines with age (mitotic selection against "
                    "mutant mtDNA in haematopoietic stem cells); muscle heteroplasmy is more stable.\n"
                    "BOTTLENECK: Offspring heteroplasmy is unpredictable; a mother with 50% "
                    "blood heteroplasmy may have offspring ranging from 0% to >90%."
                ),
            },
            {
                "term": "Mitochondrial Inheritance — MELAS vs. Nuclear OXPHOS Genes",
                "definition": (
                    "MELAS from MT-TL1 mutation follows strict MATERNAL inheritance:\n"
                    "  • ALL children (both sexes) of an affected mother inherit the mutation\n"
                    "  • FATHERS do NOT transmit — paternal mtDNA is eliminated after fertilisation\n"
                    "  • Maternal relatives (grandmothers, aunts, cousins) may carry same mutation "
                    "at varying heteroplasmy → variable expressivity within families\n"
                    "CONTRAST WITH NUCLEAR OXPHOS DISEASES: SURF1 (Leigh disease), POLG, SCO2 "
                    "etc. follow Mendelian (usually AR) inheritance — family pattern is different.\n"
                    "CLINICAL TIP: Young adult stroke + maternal DM + SNHL → always consider "
                    "m.3243A>G before diagnosing 'cryptogenic stroke'."
                ),
            },
            {
                "term": "Stroke-like Episode (SLE) — Pathogenesis and MRI",
                "definition": (
                    "SLEs are NOT conventional ischemic strokes:\n"
                    "MECHANISM: Neuronal hyperexcitability (seizure onset) drives metabolic "
                    "demand beyond OXPHOS capacity in cortical regions with highest energy "
                    "requirements (occipital/parietal cortex, visual association areas). "
                    "Simultaneously, vascular smooth muscle and endothelium carrying the "
                    "m.3243A>G mutation fail to maintain NO-mediated vasodilation, causing "
                    "cortical ischaemia that does not respect arterial boundaries.\n"
                    "MRI SIGNATURE: T2/FLAIR cortical signal, often parieto-occipital; "
                    "CROSSES VASCULAR TERRITORIES (key DDx from ischemic stroke); migrates "
                    "over days to weeks; DWI may show restricted diffusion later (cytotoxic "
                    "component) but initially vasogenic; cortical laminar necrosis in chronic stage.\n"
                    "DISTINGUISHING FEATURES vs. ISCHEMIC STROKE: No arterial distribution, "
                    "elevated lactate, family history, ragged-red fibers, SNHL, young age, "
                    "positive mtDNA mutation on blood/muscle testing."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "MELAS Syndrome — OMIM #540000",
                "definition": (
                    "MELAS = Mitochondrial Encephalomyopathy, Lactic Acidosis, and Stroke-like Episodes.\n"
                    "DEFINING CRITERIA (Pavlakis 1984 + updated 1992):\n"
                    "  A. Stroke-like episodes before age 40\n"
                    "  B. Encephalopathy (seizures and/or dementia)\n"
                    "  C. Mitochondrial myopathy (ragged-red fibers, lactic acidosis)\n"
                    "SUPPORTING FEATURES: maternal inheritance, SNHL, short stature, DM.\n"
                    "PREVALENCE: ~1/6,000-10,000; m.3243A>G prevalence in population ~1/400 "
                    "(most subclinical). Most common mitochondrial disease presenting with "
                    "stroke-like episodes.\n"
                    "NATURAL HISTORY: Progressive; SLE frequency increases without treatment; "
                    "progressive cognitive decline; median survival ~35-45 years from symptom "
                    "onset; cardiac/respiratory failure are common causes of death."
                ),
            },
            {
                "term": "MIDD — Maternally Inherited Diabetes and Deafness",
                "definition": (
                    "MIDD is the m.3243A>G phenotype at LOW blood heteroplasmy (<60%).\n"
                    "PREVALENCE: m.3243A>G accounts for 1-2% of all type 2 diabetes in "
                    "general population — most cases are UNRECOGNISED.\n"
                    "CLINICAL FEATURES: Lean/thin-body type 2 DM (not obese); SNHL (bilateral "
                    "high-frequency, often precedes DM diagnosis); maternal family history of DM "
                    "and/or deafness; no SLE (too low heteroplasmy).\n"
                    "DIAGNOSTIC TRAP: MIDD patients receive metformin without testing for "
                    "m.3243A>G → risk of fatal lactic acidosis.\n"
                    "SCREENING RULE: Maternal-pattern DM + SNHL + lean body = test m.3243A>G "
                    "BEFORE starting metformin."
                ),
            },
            {
                "term": "Ragged Red Fibers (RRF) — Muscle Biopsy in MELAS",
                "definition": (
                    "Ragged Red Fibers are the histological hallmark of mitochondrial myopathy.\n"
                    "MECHANISM: Mitochondrial proliferation compensates for OXPHOS failure → "
                    "accumulation of structurally abnormal mitochondria at the subsarcolemmal "
                    "region → red 'ragged' appearance on Gomori modified trichrome stain.\n"
                    "SDH STAINING: Ragged red = succinate dehydrogenase-rich (SDH is "
                    "nuclear-encoded, preserved in mtDNA disease) = reliably positive. "
                    "COX (cytochrome c oxidase) staining: COX-negative fibers in CIV deficiency.\n"
                    "CLINICAL USE: Muscle biopsy + electron microscopy = gold standard for "
                    "confirming mitochondrial myopathy; however, genetic testing (next-generation "
                    "sequencing of MT-TL1) has largely supplanted biopsy for diagnosis."
                ),
            },
            {
                "term": "Key DDx: MELAS vs. Ischemic Stroke vs. CADASIL vs. MERRF",
                "definition": (
                    "MELAS vs. ISCHEMIC STROKE (young adult):\n"
                    "  MELAS: Crosses vascular territories; lactic acidosis; RRF; maternal history; "
                    "SNHL; DM; positive mt DNA mutation. STROKE: Arterial distribution; restricted "
                    "DWI from onset; no lactic acidosis; no RRF.\n\n"
                    "MELAS vs. CADASIL (NOTCH3):\n"
                    "  MELAS: Cortical/subcortical mixed; lactic acidosis; RRF; mtDNA; maternal. "
                    "CADASIL: White matter disease; anterior temporal/external capsule; no lactic "
                    "acidosis; no RRF; GOM deposits on skin biopsy; AD inheritance.\n\n"
                    "MELAS vs. MERRF (MT-TK m.8344A>G):\n"
                    "  MELAS: SLE (defining); occipital dominant; DM; SNHL. "
                    "MERRF: Myoclonic epilepsy dominant; NO SLE; cerebellar atrophy; ataxia "
                    "dominant; WPW rare; different mt gene.\n\n"
                    "MELAS vs. MT-ND5 m.13513G>A (MSIIT — MELAS/LHON/Leigh overlap):\n"
                    "  Both can have SLE; ND5 mutation may also show optic nerve involvement "
                    "(LHON-like) and basal ganglia Leigh pattern; genetic sequencing differentiates."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Prescribing Safety Summary — MELAS (MT-TL1 m.3243A>G)",
                "definition": (
                    "ABSOLUTE CONTRAINDICATIONS (use is always wrong):\n"
                    "  • VPA (valproate): CoA sequestration + POLG inhibition + hepatotoxicity\n"
                    "  • Metformin: Complex I inhibition → fatal lactic acidosis (including MIDD)\n"
                    "  • Linezolid: Mitochondrial 23S rRNA inhibition → pan-OXPHOS depletion\n"
                    "  • Chloramphenicol: Same mechanism as linezolid\n\n"
                    "DRUGS TO AVOID (strong caution; use only if no alternative):\n"
                    "  • Propofol infusion: PRIS — fatal in prolonged use in mito disease\n"
                    "  • Phenobarbital: Complex I inhibition\n"
                    "  • Aminoglycosides: Irreversible worsening of SNHL\n"
                    "  • CBZ/OXC/PHT: Hepatotoxic Na-channel blockers; PHT depletes CoQ10\n"
                    "  • Glucocorticoids: Hyperglycaemia (MIDD-DM) + impair mito biogenesis\n\n"
                    "CAUTION (monitor carefully):\n"
                    "  • Statins: CoQ10 depletion → worsens myopathy; mandatory CoQ10 supplement\n"
                    "  • Beta-blockers: Reduce exercise tolerance; avoid if myopathy dominant\n"
                    "  • IV contrast: Ensure hydration; transient contrast nephropathy can "
                    "  precipitate lactic acidosis crisis\n\n"
                    "PREFERRED TREATMENTS:\n"
                    "  AED: LEV (first-line); CLB or perampanel (second-line)\n"
                    "  Acute SLE: IV arginine 0.5g/kg + IV dextrose GIR 8-10 + thiamine 100mg IV\n"
                    "  Maintenance: Oral citrulline/arginine + CoQ10/ubiquinol + taurine + B2 + B1\n"
                    "  DM (MIDD): Insulin + DPP-4 inhibitor (NEVER metformin)\n"
                    "  Antibiotic safety: Avoid linezolid/chloramphenicol/aminoglycosides; "
                    "prefer cephalosporins/carbapenems/daptomycin\n\n"
                    "TRIGGER AVOIDANCE (prevent SLE):\n"
                    "  • NEVER fast >4 hours (use IV dextrose if NBM for procedure)\n"
                    "  • Treat infections EARLY AND AGGRESSIVELY\n"
                    "  • Avoid hypotension during surgery or sepsis\n"
                    "  • Avoid prolonged physical exhaustion without carbohydrate supplementation\n"
                    "  • Genetic counselling: Reproductive options include PGT-M for known carriers"
                ),
            },
        ],
    }
