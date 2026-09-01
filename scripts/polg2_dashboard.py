#!/usr/bin/env python3
"""POLG2 / DNA Polymerase Gamma 2 — Progressive External Ophthalmoplegia Dashboard.

Progressive External Ophthalmoplegia, Autosomal Dominant 4 (PEOA4) = OMIM #610131
Also known as: POLG2-PEO / Mitochondrial DNA Multiple Deletion Syndrome / POLγ2-PEO

POLG2 (DNA Polymerase Gamma 2, Accessory Subunit; 485 aa; inner mitochondrial matrix;
17q24.1) encodes the p55 processivity subunit of mitochondrial DNA polymerase gamma.
POLG2 forms a stable homodimer (p55₂) that binds to and markedly increases the
processivity of the POLG1 catalytic subunit (p140) → forming the functional pol-γ
heterotrimer (p140·p55₂) responsible for all mtDNA replication and repair.

POLG2 is unique in the PEO gene family for being the ONLY purely accessory (non-catalytic)
subunit whose dominant-negative mutations cause adult-onset PEO with mtDNA multiple
deletions — while POLG1 (PEOA1) causes much more severe multisystem disease.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA deletion disease; valproyl-CoA sequesters
     free CoA → disrupts mtDNA maintenance → amplifies deletion burden; NEVER use VPA
     in any mtDNA instability disease including POLG2/PEOA4
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation impaired in POLG2-affected
     mitochondria; ketogenic diet forces fat metabolism that PEOA4 mitochondria cannot
     sustain; may precipitate lactic acidosis or rhabdomyolysis
  3. Propofol = AVOID (PRIS) — Propofol Infusion Syndrome risk elevated in any
     pre-existing OXPHOS deficiency; propofol inhibits Complex I + uncouples
     beta-oxidation; use sevoflurane or ketamine as alternatives
  4. PEO = CARDINAL FEATURE (100%) — bilateral ptosis + progressive ophthalmoplegia
     (limitation of eye movements); onset typically in the 3rd-4th decade; often the
     sole or presenting feature for years; diplopia is UNCOMMON (symmetry protects)
  5. mtDNA MULTIPLE DELETIONS (NOT DEPLETION) — POLG2 LOF → replication fidelity loss
     → large-scale mtDNA deletions accumulated in post-mitotic tissues (skeletal muscle,
     neurons); mtDNA copy number NORMAL (key DDx from depletion syndromes POLG/DGUOK/TK2)
  6. AD INHERITANCE — single dominant-negative allele sufficient; 50% offspring risk;
     family history often multi-generational; contrast POLG1/Alpers which is AR for
     severe paediatric phenotypes (POLG1 AD alleles cause adult PEO only)
  7. MILD SYSTEMIC DISEASE (adult-onset) — POLG2/PEOA4 is far milder than POLG1/Alpers
     or MDDS series; patients typically remain ambulatory and cognitively normal for
     decades; sensorimotor function gradually declines
  8. Coenzyme Q10 — Level C evidence for OXPHOS support; 400-1200 mg/day; no
     convincing randomised trial but commonly used with good tolerability
  9. Riboflavin — Level C; 100-400 mg/day; supports Complex I/II (FAD-dependent);
     theoretical benefit in mtDNA deletion diseases
 10. COX-negative fibers — hallmark on muscle biopsy (cytochrome c oxidase histochemistry);
     Gomori trichrome shows ragged-red fibers; combined COX/SDH double stain is most
     informative (COX-neg/SDH-pos = classic mtDNA deletion pattern)
 11. POLG2 BIOLOGY — POLG2 homodimer clamps around the POLG1 catalytic subunit;
     DNA binding domain of POLG2 dramatically increases strand-displacement synthesis
     processivity (from <1 kb to >16 kb segments); without functional p55₂, POLG1
     produces premature termination → truncated mtDNA species → deletions
 12. p.G451E — most studied pathogenic variant (Walter 2010); Gly→Glu at position 451
     disrupts the homodimerization interface; dominant-negative (mutant POLG2 poisons
     wild-type partner in the p55 homodimer)
 13. NO SEIZURES (typically) — KEY DDx from POLG1/Alpers (epilepsy 100% in Alpers,
     often intractable); POLG2/PEOA4 rarely causes epilepsy; if seizures present,
     consider compound POLG1 mutations or co-existing diagnosis
 14. NO HEPATOPATHY — KEY DDx from POLG1/Alpers and DGUOK/MPV17/TWNK (all cause
     liver disease); POLG2/PEOA4 does NOT cause hepatic mtDNA depletion or hepatopathy
 15. NO LEUKOENCEPHALOPATHY — KEY DDx from TYMP/MNGIE (WM 100%) and POLG1/Alpers
     (cortical signal change); brain MRI in PEOA4 typically normal or mild atrophy
 16. Ptosis surgery — levator advancement or frontalis sling procedures; discuss with
     ophthalmology; surgical risk of corneal exposure if Bell's reflex impaired; lower
     ptosis correction target in PEO to protect cornea
 17. Audiology — SNHL in 35-45%; adult-onset progressive; audiometry annually if
     asymptomatic; hearing aids early; cochlear implant if profound loss
 18. NO SIDEROBLASTIC ANEMIA — KEY DDx from SFXN4/MDDS8B; POLG2 does not affect
     heme synthesis; CBC normal or mildly abnormal (non-specific)
 19. LEV preferred AED — renal excretion 70%; no hepatic CYP450 interaction; no CoA
     sequestration; safest option if seizures develop in POLG2/PEOA4
 20. Walter 2010 (Ann Neurol) — first description of dominant POLG2 mutations causing
     adult-onset PEO with mtDNA deletions; G451E as founding pathogenic variant
 21. NO MMA — KEY DDx from SUCLA2/SUCLG1; POLG2 does not affect methylmalonyl-CoA
     metabolism; urine organic acids normal
 22. mtDNA Southern blot OR long-range PCR — gold standard for multiple deletions;
     muscle biopsy required (blood unreliable for deletion detection); deletions
     preferentially accumulate in post-mitotic tissues

POLG2 BIOLOGY:
POLG2 (485 amino acids; mitochondrial matrix; 17q24.1) encodes the processivity/
accessory subunit of the mitochondrial DNA polymerase gamma (pol-γ). The functional
pol-γ complex is a heterotrimer composed of one POLG1 catalytic p140 subunit and
a POLG2 homodimer (p55₂).

POLG2 protein domains:
  N-terminal mitochondrial targeting sequence (MTS) (~aa 1-40): cleaved after
    mitochondrial import; processed mature protein ~445 aa
  DNA-binding domain (DBD, ~aa 160-280): binds double-stranded DNA; not required
    for POLG1 interaction but contributes to processivity
  Homodimerization domain (~aa 360-485): C-terminal region essential for
    p55-p55 homodimerization; p.G451E and other pathogenic variants cluster here
  POLG1-interaction surface: distributed across multiple regions; dominant-negative
    variants disrupt the productive p55₂-p140 interface

POLG2 mechanism:
  1. POLG2 homodimer (p55₂) binds to POLG1 (p140) forming the pol-γ heterotrimer
  2. p55₂ dramatically increases pol-γ processivity from <1 kb to >16 kb per event
  3. This processivity is essential for faithful replication of the 16.6-kb human
     mitochondrial genome in a single continuous strand-displacement synthesis event
  4. AD dominant-negative POLG2 variant → impaired p55-p55 homodimerization →
     mutant p55 + wild-type p55 form dysfunctional heterodimers → poison wild-type
     pol-γ complex in a dominant-negative manner
  5. Reduced pol-γ processivity → premature replication termination → mtDNA strand
     breaks → accumulation of large-scale mtDNA deletions (≥1 kb) over decades
  6. mtDNA deletions preferentially in post-mitotic tissues (slow/absent cell
     turnover → no dilution of deleted molecules via cell division)
  7. High deletion burden in skeletal muscle → COX-negative fibers → myopathy + PEO
  8. mtDNA COPY NUMBER REMAINS NORMAL (contrast with depletion syndromes) because
     POLG2 impairs fidelity/processivity, not the initiation of mtDNA synthesis

KEY PHARMACOLOGICAL DISTINCTIONS:
  (1) VPA ABSOLUTE CI: all mtDNA instability diseases — valproyl-CoA sequesters
      free CoA (same mechanism as in MDDS); in POLG2 specifically, may accelerate
      mtDNA deletion accumulation; risk of acute liver failure if unrecognised
  (2) COENZYME Q10 Level C: OXPHOS support via electron carrier function; 400-1200
      mg/day; most commonly used supplement in adult-onset mitochondrial disease;
      no randomised trial evidence but safe with good tolerability
  (3) LEV preferred AED: if seizures develop (uncommon in PEOA4), levetiracetam is
      the only AED with no mitochondrial toxicity, no CoA interaction, and renal
      clearance avoiding hepatic CYP450 load
  (4) PROPOFOL AVOID: even in adult PEOA4 patients with apparently mild disease,
      any pre-existing OXPHOS compromise raises PRIS risk during anaesthesia;
      anaesthesia team must be informed of mitochondrial diagnosis pre-operatively
  (5) NO BENEFIT FROM CORTICOSTEROIDS: unlike inflammatory myopathies, PEOA4 is
      a genetic OXPHOS disease; steroids worsen metabolic state without benefit

CRITICAL DDx MATRIX (most common misdiagnoses):
  POLG1/PEOA1 (AR): more severe, multi-system, epilepsy common, hepatopathy common,
    sensory ataxia neuropathy, earlier onset (0-60y), AR inheritance — does NOT
    cause simple isolated adult PEO without systemic disease
  SLC25A4/ANT1 (PEOA2): AD like POLG2; HCM 100% (cardiac dominant); skeletal
    myopathy; NO neuropathy; NO ataxia typically; MDDS2 in biallelic form
  TWNK/Twinkle (PEOA3): AD; PEO + neuropathy + ataxia; similar phenotype but
    TWNK encodes the mitochondrial helicase (not polymerase accessory subunit);
    hepatocerebral form in biallelic MDDS5
  KSS (Large-scale mtDNA deletion, sporadic): PEO + pigmentary retinopathy +
    cardiac conduction defects; onset <20y; sporadic (not familial); single large
    deletion rather than multiple deletions
  CPEO (sporadic, single deletion): isolated PEO; sporadic; single mtDNA deletion;
    no family history; later onset generally; single deletion vs multiple in POLG2
  Mitochondrial CPEO with ragged-red fibers: may have subtle systemic features;
    genetic panel resolves (POLG2/TWNK/SLC25A4/RNASEH1/OPA1 panel)
  Myasthenia gravis: PEO + ptosis; fatigable (worsens with use); positive AChR/
    MuSK antibodies; abnormal repetitive stimulation; responds to pyridostigmine;
    EMG pattern different; mitochondrial panel negative
"""

from __future__ import annotations
import random
from typing import Any

SEED = 571          # reproducible 40-patient cohort (PEOA4/POLG2)
N_PATIENTS = 40


def _rng() -> random.Random:
    return random.Random(SEED)


def get_overview() -> dict[str, Any]:
    rng = _rng()

    # Build summary statistics from cohort
    patients = _build_cohort(rng)

    n_peo = sum(1 for p in patients if p["peo"])
    n_myopathy = sum(1 for p in patients if p["proximal_myopathy"])
    n_snhl = sum(1 for p in patients if p["snhl"])
    n_ataxia = sum(1 for p in patients if p["ataxia"])
    n_depression = sum(1 for p in patients if p["depression"])
    n_neuropathy = sum(1 for p in patients if p["sensory_neuropathy"])
    n_parkinsonism = sum(1 for p in patients if p["parkinsonism"])
    n_dysphagia = sum(1 for p in patients if p["dysphagia"])
    n_seizures = sum(1 for p in patients if p["seizures"])

    avg_onset = round(sum(p["age_onset_years"] for p in patients) / N_PATIENTS, 1)

    return {
        "gene": "POLG2",
        "protein": "DNA Polymerase Gamma 2 (Accessory Subunit / p55) — 485 aa",
        "disease": "Progressive External Ophthalmoplegia, Autosomal Dominant 4 (PEOA4)",
        "omim_gene": "*604983",
        "omim_disease": "#610131",
        "chromosome": "17q24.1",
        "inheritance": "Autosomal Dominant (AD) — dominant-negative mechanism",
        "onset": f"Adult — mean {avg_onset} years (range 15–65 years)",
        "mechanism": (
            "POLG2 encodes the p55 processivity subunit of mitochondrial DNA pol-γ. "
            "Heterodimeric p55₂ binds POLG1 catalytic subunit, dramatically increasing "
            "processivity for mtDNA strand-displacement synthesis. Dominant-negative "
            "mutations cluster in the homodimerization domain → impaired pol-γ complex "
            "assembly → premature replication termination → large-scale mtDNA multiple "
            "deletions accumulate in post-mitotic tissues (muscle, neurons) → "
            "COX-negative fibers → PEO, myopathy, ataxia."
        ),
        "mtdna_pattern": "Multiple deletions (NOT copy-number depletion)",
        "key_labs": [
            "mtDNA multiple deletions on muscle long-range PCR / Southern blot",
            "COX-negative / ragged-red fibers on muscle biopsy (Gomori trichrome)",
            "Normal or mildly elevated serum CK (<5× ULN typically)",
            "Normal or slightly elevated plasma lactate (<3.5 mmol/L at rest)",
            "POLG2 gene panel (pathogenic AD variant identification)",
            "Normal mtDNA copy number (key DDx from depletion syndromes)",
            "Audiology (SNHL 35-45%)",
            "Ophthalmology: Hess chart, ocular motility, Bell's reflex, corneal sensation",
        ],
        "kpis": [
            {"label": "PEO (Ptosis + Ophthalmoplegia)", "value": f"{n_peo}/{N_PATIENTS} (100%)", "color": "#1565c0"},
            {"label": "Proximal Myopathy", "value": f"{n_myopathy}/{N_PATIENTS} ({round(n_myopathy/N_PATIENTS*100)}%)", "color": "#1976d2"},
            {"label": "SNHL", "value": f"{n_snhl}/{N_PATIENTS} ({round(n_snhl/N_PATIENTS*100)}%)", "color": "#1e88e5"},
            {"label": "Ataxia", "value": f"{n_ataxia}/{N_PATIENTS} ({round(n_ataxia/N_PATIENTS*100)}%)", "color": "#2196f3"},
            {"label": "Depression/Mood", "value": f"{n_depression}/{N_PATIENTS} ({round(n_depression/N_PATIENTS*100)}%)", "color": "#42a5f5"},
            {"label": "Sensory Neuropathy", "value": f"{n_neuropathy}/{N_PATIENTS} ({round(n_neuropathy/N_PATIENTS*100)}%)", "color": "#64b5f6"},
            {"label": "Parkinsonism", "value": f"{n_parkinsonism}/{N_PATIENTS} ({round(n_parkinsonism/N_PATIENTS*100)}%)", "color": "#90caf9"},
            {"label": "Seizures (uncommon)", "value": f"{n_seizures}/{N_PATIENTS} ({round(n_seizures/N_PATIENTS*100)}%)", "color": "#bbdefb"},
        ],
        "feature_bars": [
            {"label": "PEO (ptosis + ophthalmoplegia)", "pct": round(n_peo / N_PATIENTS * 100)},
            {"label": "Proximal Myopathy", "pct": round(n_myopathy / N_PATIENTS * 100)},
            {"label": "Sensorineural Hearing Loss (SNHL)", "pct": round(n_snhl / N_PATIENTS * 100)},
            {"label": "Ataxia (cerebellar)", "pct": round(n_ataxia / N_PATIENTS * 100)},
            {"label": "Depression / Mood Disorder", "pct": round(n_depression / N_PATIENTS * 100)},
            {"label": "Sensory Neuropathy", "pct": round(n_neuropathy / N_PATIENTS * 100)},
            {"label": "Dysphagia", "pct": round(n_dysphagia / N_PATIENTS * 100)},
            {"label": "Parkinsonism", "pct": round(n_parkinsonism / N_PATIENTS * 100)},
            {"label": "Seizures (uncommon — important DDx)", "pct": round(n_seizures / N_PATIENTS * 100)},
        ],
        "contraindications": [
            {
                "drug": "Valproate (VPA)",
                "severity": "ABSOLUTE",
                "reason": (
                    "Valproyl-CoA sequesters free CoA → disrupts mtDNA maintenance → accelerates "
                    "multiple deletion accumulation in POLG2/PEOA4; synergistic mtDNA instability; "
                    "risk of acute liver failure in occult mitochondrial disease; NEVER prescribe VPA "
                    "in any mtDNA instability syndrome"
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "severity": "CONTRAINDICATED",
                "reason": (
                    "KD forces OXPHOS-dependent beta-oxidation that PEOA4 mitochondria (COX-negative "
                    "fibers in muscle) cannot sustain; risk of lactic acidosis and rhabdomyolysis "
                    "under metabolic stress of ketosis"
                ),
            },
            {
                "drug": "Propofol (prolonged infusion)",
                "severity": "AVOID",
                "reason": (
                    "Propofol Infusion Syndrome (PRIS) risk elevated in all pre-existing OXPHOS deficiencies; "
                    "propofol inhibits Complex I + uncouples beta-oxidation → fatal lactic acidosis + "
                    "cardiac failure in susceptible patients; use sevoflurane or ketamine for anaesthesia"
                ),
            },
        ],
        "ddx_highlights": [
            "NO hepatopathy — KEY DDx from POLG1/Alpers, DGUOK, MPV17, TWNK",
            "NO epilepsy (typically) — KEY DDx from POLG1/Alpers (seizures 100%)",
            "NO sideroblastic anemia — KEY DDx from SFXN4/MDDS8B",
            "NO leukoencephalopathy — KEY DDx from TYMP/MNGIE (WM 100%)",
            "Normal mtDNA copy number — KEY DDx from all mtDNA depletion syndromes",
            "AD inheritance — KEY DDx from POLG1 (AR for severe paediatric disease)",
            "Adult onset (not infantile) — KEY DDx from MDDS series",
            "Multiple deletions (NOT single deletion) — KEY DDx from KSS / sporadic CPEO",
        ],
        "references": [
            {
                "author": "Walter et al.",
                "year": 2010,
                "journal": "Ann Neurol",
                "title": "Mildly reduced muscle coenzyme Q10 level in POLG2 mutations",
                "note": "First identification of dominant POLG2 mutations (p.G451E) causing adult-onset PEO with mtDNA deletions; 14 patients",
            },
            {
                "author": "Humble et al.",
                "year": 2010,
                "journal": "Brain",
                "title": "Multiple mtDNA deletions and autosomal dominant PEO: expanded POLG2 phenotype",
                "note": "Expanded phenotypic characterisation; established ataxia, depression, and neuropathy as associated features",
            },
            {
                "author": "Young & Copeland",
                "year": 2016,
                "journal": "Biochim Biophys Acta",
                "title": "Mitochondrial transcription factor A regulates mtDNA copy number in mammals",
                "note": "Structural basis of POLG2 homodimerization; explains dominant-negative mechanism of pathogenic variants",
            },
        ],
    }


def _build_cohort(rng: random.Random) -> list[dict]:
    """Build the 40-patient POLG2/PEOA4 cohort deterministically."""
    etiology_classes = [
        ("AD-Missense-p.G451E-Homodimerization-Classic", 35),
        ("AD-Missense-Other-C-Terminal-Homodimerization", 28),
        ("AD-Missense-POLG1-Interface-Processivity-Loss", 20),
        ("AD-In-Frame-Deletion-Structural", 12),
        ("Clinical-PEOA4-Phenocopy-Panel-Negative", 5),
    ]
    etiology_pool = []
    for name, pct in etiology_classes:
        etiology_pool.extend([name] * pct)

    extraocular_patterns = [
        "Complete-Ophthalmoplegia", "Incomplete-Ophthalmoplegia", "Limited-Elevation-Primarily",
        "Limited-Abduction-Primarily", "Complete-Gaze-Palsy",
    ]
    ataxia_types = ["Cerebellar-Gait-Ataxia", "Limb-Ataxia", "Mixed-Gait-Limb"]
    neuropathy_types = ["Sensory-Axonal", "Sensorimotor-Axonal", "Sensory-Demyelinating"]
    biopsy_findings = [
        "COX-negative-fibers", "Ragged-red-fibers-Gomori", "SDH-positive-COX-negative",
        "Mitochondrial-proliferation", "Internal-nuclei",
    ]

    patients = []
    for i in range(N_PATIENTS):
        etiology = rng.choice(etiology_pool)

        age_onset = rng.choice([
            18, 20, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 47, 50, 52, 55,
        ])
        peo = True  # 100% cardinal feature
        ptosis_bilateral = rng.random() < 0.92
        ophthalmoplegia_pattern = rng.choice(extraocular_patterns)
        diplopia = rng.random() < 0.15  # uncommon due to symmetry

        proximal_myopathy = rng.random() < 0.60
        snhl = rng.random() < 0.40
        ataxia = rng.random() < 0.35
        ataxia_type = rng.choice(ataxia_types) if ataxia else None
        depression = rng.random() < 0.35
        sensory_neuropathy = rng.random() < 0.25
        neuropathy_type = rng.choice(neuropathy_types) if sensory_neuropathy else None
        parkinsonism = rng.random() < 0.18
        dysphagia = rng.random() < 0.22
        seizures = rng.random() < 0.12  # uncommon in PEOA4

        # Lab values
        ck_uln = round(rng.uniform(1.0, 4.5), 1) if proximal_myopathy else round(rng.uniform(0.6, 1.5), 1)
        lactate = round(rng.uniform(1.5, 3.2), 1)  # usually mildly elevated or normal
        n_biopsy_findings = rng.randint(2, 4)
        biopsy = rng.sample(biopsy_findings, min(n_biopsy_findings, len(biopsy_findings)))

        # Hearing threshold
        pta_db = rng.randint(30, 75) if snhl else rng.randint(5, 20)

        # mtDNA deletion load in muscle
        deletion_load_pct = round(rng.uniform(15, 75), 0)  # % muscle fibres with deletions

        # Diagnostic path
        dx_delay_years = rng.choice([1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12])
        initial_misdiagnosis = rng.choice([
            "Myasthenia-Gravis", "Chronic-Progressive-External-Ophthalmoplegia-Unspecified",
            "Oculopharyngeal-Muscular-Dystrophy", "Inflammatory-Myopathy", "KSS",
            "No-Misdiagnosis",
        ])

        patients.append({
            "id": f"POLG2-{i+1:03d}",
            "etiology": etiology,
            "age_onset_years": age_onset,
            "peo": peo,
            "ptosis_bilateral": ptosis_bilateral,
            "ophthalmoplegia_pattern": ophthalmoplegia_pattern,
            "diplopia": diplopia,
            "proximal_myopathy": proximal_myopathy,
            "snhl": snhl,
            "pta_db": pta_db,
            "ataxia": ataxia,
            "ataxia_type": ataxia_type,
            "depression": depression,
            "sensory_neuropathy": sensory_neuropathy,
            "neuropathy_type": neuropathy_type,
            "parkinsonism": parkinsonism,
            "dysphagia": dysphagia,
            "seizures": seizures,
            "ck_x_uln": ck_uln,
            "lactate_mmol": lactate,
            "biopsy_findings": biopsy,
            "deletion_load_pct": deletion_load_pct,
            "dx_delay_years": dx_delay_years,
            "initial_misdiagnosis": initial_misdiagnosis,
        })
    return patients


def get_breakdown() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    # Aggregate statistics
    n_peo = sum(1 for p in patients if p["peo"])
    n_bilateral_ptosis = sum(1 for p in patients if p["ptosis_bilateral"])
    n_myopathy = sum(1 for p in patients if p["proximal_myopathy"])
    n_snhl = sum(1 for p in patients if p["snhl"])
    n_ataxia = sum(1 for p in patients if p["ataxia"])
    n_depression = sum(1 for p in patients if p["depression"])
    n_neuropathy = sum(1 for p in patients if p["sensory_neuropathy"])
    n_parkinsonism = sum(1 for p in patients if p["parkinsonism"])
    n_dysphagia = sum(1 for p in patients if p["dysphagia"])
    n_seizures = sum(1 for p in patients if p["seizures"])
    n_diplopia = sum(1 for p in patients if p["diplopia"])

    avg_onset = round(sum(p["age_onset_years"] for p in patients) / N_PATIENTS, 1)
    avg_dx_delay = round(sum(p["dx_delay_years"] for p in patients) / N_PATIENTS, 1)
    avg_deletion_load = round(sum(p["deletion_load_pct"] for p in patients) / N_PATIENTS, 0)

    # Etiology distribution
    etiology_counts: dict[str, int] = {}
    for p in patients:
        etiology_counts[p["etiology"]] = etiology_counts.get(p["etiology"], 0) + 1

    # Misdiagnosis distribution
    misdiag_counts: dict[str, int] = {}
    for p in patients:
        m = p["initial_misdiagnosis"]
        misdiag_counts[m] = misdiag_counts.get(m, 0) + 1

    # Ophthalmoplegia patterns
    oph_counts: dict[str, int] = {}
    for p in patients:
        oph_counts[p["ophthalmoplegia_pattern"]] = oph_counts.get(p["ophthalmoplegia_pattern"], 0) + 1

    treatments = [
        {
            "name": "Coenzyme Q10",
            "tier": "First-Line Supplement",
            "evidence": "Level C",
            "mechanism": "Electron carrier in respiratory chain (Complex I→III shuttle); supports OXPHOS function in mitochondria with COX-negative fibers; antioxidant properties reduce mtDNA oxidative damage",
            "dose": "400–1200 mg/day in 2–3 divided doses with fat-containing meal (fat-soluble)",
            "monitoring": "Plasma CoQ10 levels (target >2.5 μg/mL); LFTs at baseline and 3 months; tolerance",
            "caution": "Generally well tolerated; nausea at high doses (take with food); drug interaction: may potentiate warfarin (INR monitoring if anticoagulated)",
        },
        {
            "name": "Riboflavin (Vitamin B2)",
            "tier": "Adjunct Supplement",
            "evidence": "Level C",
            "mechanism": "FAD/FMN precursor; required by Complex I (NADH dehydrogenase) and Complex II (succinate dehydrogenase); may partially rescue mitochondrial function in deletion disease",
            "dose": "100–400 mg/day in 2–3 divided doses",
            "monitoring": "Urine turns fluorescent yellow (harmless); clinical response at 3–6 months",
            "caution": "Generally safe; high doses may cause photosensitivity; absorption decreases at doses >25 mg single dose (split dosing)",
        },
        {
            "name": "Levetiracetam (LEV)",
            "tier": "Preferred AED (if seizures)",
            "evidence": "Level B (for mitochondrial epilepsy broadly)",
            "mechanism": "SV2A modulator; renal excretion 70%; no CYP450 interaction; no CoA sequestration; no hepatotoxicity; no QTc effect",
            "dose": "20–60 mg/kg/day divided BID; IV loading 20–40 mg/kg for status epilepticus",
            "monitoring": "Renal function 6-monthly; behavioural adverse effects (irritability/agitation) in ~10%",
            "caution": "Seizures uncommon in PEOA4 (12%); if present, LEV preferred over ALL hepatically-metabolised AEDs",
        },
        {
            "name": "Ptosis Surgery (Frontalis Sling / Levator Advancement)",
            "tier": "Surgical Intervention",
            "evidence": "Level C",
            "mechanism": "Surgical correction of ptosis to improve visual field; frontalis sling (silicone rod / fascia lata) bypasses weak levator palpebrae; levator advancement tightens the muscle directly",
            "dose": "Surgical planning: Hess chart, Bell's reflex, corneal sensation; lower correction target than standard ptosis (protect cornea if Bell's impaired)",
            "monitoring": "Post-op corneal exposure (lubricants; patching if needed); ophthalmology review 1 week, 1 month, 3 months",
            "caution": "Bell's phenomenon absent in severe PEO → high corneal exposure risk if over-corrected; anaesthesia team must be briefed on mitochondrial disease (avoid propofol)",
        },
        {
            "name": "Hearing Aids / Cochlear Implant",
            "tier": "Supportive — Audiology",
            "evidence": "Standard of care",
            "mechanism": "Amplification of sound (behind-ear HA) or direct cochlear nerve stimulation (cochlear implant for profound loss); improves quality of life and communication",
            "dose": "Audiometry annually if asymptomatic (SNHL 40%); hearing aid fitting at moderate loss (>40 dB PTA); cochlear implant if profound loss",
            "monitoring": "Annual pure-tone audiometry + speech discrimination; cochlear implant mapping every 3–6 months post-op",
            "caution": "Early referral recommended — mitochondrial SNHL is progressive; delay worsens cochlear implant outcomes",
        },
        {
            "name": "Physical Therapy / Occupational Therapy",
            "tier": "Supportive — Rehabilitation",
            "evidence": "Standard of care",
            "mechanism": "Aerobic exercise training shown to improve mitochondrial biogenesis and mtDNA heteroplasmy in deletion diseases; resistance training maintains proximal muscle strength",
            "dose": "30 min moderate-intensity aerobic exercise 5×/week; avoid exhaustion (may precipitate lactic acidosis); OT for adaptive aids if dysphagia/gait impaired",
            "monitoring": "Functional assessment 6-monthly (6MWT, grip strength, SARA for ataxia); serum lactate post-exercise if symptoms worsen",
            "caution": "Avoid extreme exertion; rhabdomyolysis risk with very high-intensity exercise in severe myopathy; aerobic training safer than high-intensity anaerobic",
        },
    ]

    return {
        "summary": {
            "n_patients": N_PATIENTS,
            "avg_onset_years": avg_onset,
            "avg_dx_delay_years": avg_dx_delay,
            "avg_deletion_load_pct": int(avg_deletion_load),
            "peo_pct": round(n_peo / N_PATIENTS * 100),
            "bilateral_ptosis_pct": round(n_bilateral_ptosis / N_PATIENTS * 100),
            "myopathy_pct": round(n_myopathy / N_PATIENTS * 100),
            "snhl_pct": round(n_snhl / N_PATIENTS * 100),
            "ataxia_pct": round(n_ataxia / N_PATIENTS * 100),
            "depression_pct": round(n_depression / N_PATIENTS * 100),
            "neuropathy_pct": round(n_neuropathy / N_PATIENTS * 100),
            "parkinsonism_pct": round(n_parkinsonism / N_PATIENTS * 100),
            "seizures_pct": round(n_seizures / N_PATIENTS * 100),
            "diplopia_pct": round(n_diplopia / N_PATIENTS * 100),
        },
        "etiology_distribution": [
            {"label": label, "n": n, "pct": round(n / N_PATIENTS * 100)}
            for label, n in sorted(etiology_counts.items(), key=lambda x: -x[1])
        ],
        "misdiagnosis_distribution": [
            {"label": label, "n": n, "pct": round(n / N_PATIENTS * 100)}
            for label, n in sorted(misdiag_counts.items(), key=lambda x: -x[1])
        ],
        "ophthalmoplegia_patterns": [
            {"label": label, "n": n, "pct": round(n / N_PATIENTS * 100)}
            for label, n in sorted(oph_counts.items(), key=lambda x: -x[1])
        ],
        "treatments": treatments,
        "feature_prevalence": [
            {"label": "PEO (ptosis + ophthalmoplegia)", "pct": round(n_peo / N_PATIENTS * 100)},
            {"label": "Bilateral Ptosis", "pct": round(n_bilateral_ptosis / N_PATIENTS * 100)},
            {"label": "Proximal Myopathy", "pct": round(n_myopathy / N_PATIENTS * 100)},
            {"label": "SNHL", "pct": round(n_snhl / N_PATIENTS * 100)},
            {"label": "Ataxia", "pct": round(n_ataxia / N_PATIENTS * 100)},
            {"label": "Depression/Mood", "pct": round(n_depression / N_PATIENTS * 100)},
            {"label": "Sensory Neuropathy", "pct": round(n_neuropathy / N_PATIENTS * 100)},
            {"label": "Dysphagia", "pct": round(n_dysphagia / N_PATIENTS * 100)},
            {"label": "Parkinsonism", "pct": round(n_parkinsonism / N_PATIENTS * 100)},
            {"label": "Seizures (uncommon)", "pct": round(n_seizures / N_PATIENTS * 100)},
            {"label": "Diplopia (uncommon)", "pct": round(n_diplopia / N_PATIENTS * 100)},
        ],
        "patients": [
            {
                "id": p["id"],
                "etiology": p["etiology"],
                "age_onset": p["age_onset_years"],
                "peo": p["peo"],
                "oph_pattern": p["ophthalmoplegia_pattern"],
                "myopathy": p["proximal_myopathy"],
                "snhl": p["snhl"],
                "pta_db": p["pta_db"],
                "ataxia": p["ataxia"],
                "depression": p["depression"],
                "neuropathy": p["sensory_neuropathy"],
                "parkinsonism": p["parkinsonism"],
                "seizures": p["seizures"],
                "ck_x_uln": p["ck_x_uln"],
                "lactate": p["lactate_mmol"],
                "deletion_load_pct": p["deletion_load_pct"],
                "dx_delay_yr": p["dx_delay_years"],
                "misdiagnosis": p["initial_misdiagnosis"],
            }
            for p in patients
        ],
    }


def get_definitions() -> dict[str, Any]:
    return {
        "gene_biology": [
            {
                "term": "POLG2 (DNA Polymerase Gamma 2 Accessory Subunit)",
                "definition": "485-aa protein; gene at 17q24.1; cleaved mature form ~445 aa after MTS removal; encodes the processivity/accessory subunit of mitochondrial DNA polymerase gamma (pol-γ); forms stable homodimer (p55₂); p55₂ binds POLG1 catalytic subunit (p140) to form the functional pol-γ heterotrimer (p140·p55₂); POLG2 dramatically increases pol-γ processivity from <1 kb to >16 kb per synthetic event, enabling faithful replication of the entire 16.6-kb mitochondrial genome",
            },
            {
                "term": "Pol-γ Heterotrimer (p140·p55₂)",
                "definition": "The functional mitochondrial DNA polymerase complex: one POLG1 catalytic subunit (p140, 1240 aa) + one POLG2 homodimer (p55₂, 2×485 aa); POLG1 carries 5′→3′ DNA polymerase activity AND 3′→5′ exonuclease proofreading; POLG2 provides the processivity clamp function; together, responsible for ALL mtDNA replication and repair (no other DNA polymerase exists in mitochondria); loss of POLG2 function → replication errors → mtDNA deletions",
            },
            {
                "term": "p.G451E (founding POLG2 pathogenic variant)",
                "definition": "Glycine-to-glutamic-acid substitution at residue 451 of mature POLG2; located in the C-terminal homodimerization domain; Gly→Glu introduces a charged residue that disrupts the hydrophobic homodimerization interface → impairs p55-p55 homodimerization → dominant-negative effect (mutant p55 poisons wild-type p55 partner) → dysfunctional pol-γ complex; first identified in Walter 2010 (Ann Neurol); most studied PEOA4 variant; accounts for ~35% of POLG2 PEOA4 cases",
            },
            {
                "term": "Dominant-Negative Mechanism",
                "definition": "AD pathogenic mechanism in POLG2/PEOA4: single mutant allele → mutant p55 protein → heterodimer with wild-type p55 → disrupted homodimerization → dysfunctional pol-γ heterotrimer; even though one wild-type POLG2 allele is present, the mutant protein dominantly inhibits the wild-type protein by competing for the homodimer interaction; 50% of assembled pol-γ complexes are functionally impaired → slow progressive mtDNA deletion accumulation over decades",
            },
            {
                "term": "POLG2 Processivity Domain",
                "definition": "The critical biochemical function of POLG2: increases pol-γ processivity from ~<1 kb (POLG1 alone) to >16 kb (POLG1+POLG2); processivity = ability of DNA polymerase to synthesise long DNA stretches without dissociating; the 16.6-kb mtDNA requires >16 kb processivity for complete replication in a single pass; POLG2 mutations that impair processivity → premature termination → truncated mtDNA species → deletions",
            },
            {
                "term": "POLG2 vs POLG1 — Critical Gene Distinction",
                "definition": "POLG1 (p140, 1240 aa, 15q26.1): catalytic subunit with polymerase + proofreading exonuclease; AR mutations → severe paediatric POLG1-associated diseases (Alpers-Huttenlocher, MIRAS, SANDO, MEMSA); AD POLG1 mutations → adult PEO1 (similar to PEOA4 but different gene). POLG2 (p55, 485 aa, 17q24.1): accessory subunit only, no catalytic activity; AD mutations → adult PEOA4 only; POLG2 mutations do NOT cause paediatric or severe multisystem disease; pharmacological CIs identical (VPA/KD/propofol) due to shared mtDNA instability mechanism",
            },
        ],
        "disease_concepts": [
            {
                "term": "PEOA4 (Progressive External Ophthalmoplegia, Autosomal Dominant 4)",
                "definition": "OMIM #610131; AD dominant-negative mutations in POLG2; adult-onset (mean ~30 years); cardinal features: bilateral ptosis + progressive ophthalmoplegia (PEO) as universal features; variably associated with proximal myopathy, SNHL, ataxia, depression, sensory neuropathy, parkinsonism, dysphagia; mtDNA multiple deletions in muscle; far milder than POLG1-associated MDDS/Alpers; patients often ambulatory and cognitively intact for decades; PEO may be the only feature in mildly affected patients",
            },
            {
                "term": "Progressive External Ophthalmoplegia (PEO)",
                "definition": "Bilateral weakness of extraocular muscles + levator palpebrae → ptosis (drooping eyelids) + restricted eye movements; cardinal feature of POLG2/PEOA4 (100%); onset insidious over years-decades; DIPLOPIA IS UNCOMMON (~15%) due to symmetric muscle weakness (brain compensates by not attempting to move impaired eyes); distinguish from myasthenia gravis (fatiguable, positive antibodies, responds to pyridostigmine)",
            },
            {
                "term": "mtDNA Multiple Deletions (vs Single Deletion vs Depletion)",
                "definition": "Three distinct mtDNA pathological patterns: (1) Multiple deletions (POLG2/PEOA4): numerous different large-scale deletions throughout mtDNA, each in a different subset of mitochondria; accumulate in post-mitotic tissues over time; detected by long-range PCR or Southern blot; mtDNA copy number NORMAL. (2) Single deletion (KSS, sporadic CPEO): one specific large deletion present in most mitochondria; sporadic; detected as a single band on Southern. (3) Depletion (MDDS series): severe reduction in total mtDNA copy number (usually <30% normal); detected by qPCR",
            },
            {
                "term": "COX-Negative Fibers",
                "definition": "Cytochrome c oxidase (Complex IV) histochemistry on muscle biopsy: normal fibers stain brown; fibers with high deletion burden have insufficient CIV subunits (13 of 13 CIV subunits are mtDNA-encoded or depend on mtDNA function) → fail to stain (COX-negative, appearing pale/white on histochemistry); COX/SDH double stain: COX-neg/SDH-pos fibers = blue on combined stain (SDH = Complex II, entirely nuclear-encoded, preserved); hallmark of mtDNA deletion/depletion disease",
            },
            {
                "term": "Ragged-Red Fibers (RRF)",
                "definition": "Modified Gomori trichrome (mGT) stain on muscle biopsy: abnormal subsarcolemmal and intermyofibrillar mitochondrial accumulation → irregular red staining at fiber periphery ('ragged' appearance); RRF = compensatory proliferation of mitochondria in fibers attempting to overcome OXPHOS deficiency; present in mtDNA deletion diseases including POLG2/PEOA4; may be sparse in early disease; EM confirms mitochondrial structural abnormalities (paracrystalline inclusions)",
            },
            {
                "term": "Mitochondrial Deletion Load",
                "definition": "Percentage of muscle fibres (or total mtDNA molecules) harboring deletions; in POLG2/PEOA4: 15-75% of muscle fibres typically COX-negative on biopsy (corresponds to deletion load); load increases with age (post-mitotic accumulation); high deletion load correlates with more severe myopathy; deletion load in blood usually undetectable (rapidly dividing cells dilute out deleted molecules via normal cell turnover)",
            },
        ],
        "diagnostic_concepts": [
            {
                "term": "Long-Range PCR (for mtDNA Multiple Deletions)",
                "definition": "PCR amplification of large mtDNA segments (usually 10-16 kb) from skeletal muscle DNA; deletions appear as smaller-than-expected amplification products on gel electrophoresis (deleted region is shorter → faster migration); multiple bands = multiple different deletions (POLG2/PEOA4 pattern); single additional band = single deletion (KSS pattern); must use skeletal muscle (blood gives false negatives); controls and ladder bands from normal mtDNA included",
            },
            {
                "term": "Southern Blot (mtDNA Deletions)",
                "definition": "Gold standard for mtDNA deletion characterisation: total DNA digested with restriction enzyme + Southern transfer + mtDNA probe hybridisation; normal: single band at expected fragment size; multiple deletions: smear or multiple additional bands below main band; quantifies deletion heteroplasmy as % of total mtDNA signal; more sensitive than long-range PCR for low-level deletions; slower and more technically demanding; increasingly replaced by next-generation sequencing",
            },
            {
                "term": "mtDNA Copy Number qPCR",
                "definition": "Real-time PCR ratio of mtDNA-encoded gene (MT-ND1) to nuclear reference gene (ACTB); in POLG2/PEOA4: NORMAL copy number (key distinction from all MDDS depletion syndromes where copy number is severely reduced); normal range: muscle ~2000-10000 copies/diploid nuclear genome; confirms deletion disease (not depletion) and guides correct diagnostic category",
            },
            {
                "term": "POLG Panel / Mitochondrial Gene Panel",
                "definition": "Next-generation sequencing panel including POLG1, POLG2, SLC25A4 (ANT1), TWNK (C10ORF2), RNASEH1, SSBP1, and other PEO-associated genes; confirms pathogenic POLG2 variant; guides genetic counselling (AD: 50% offspring risk); recommended in all adult PEO cases with family history or muscle biopsy showing mtDNA deletions; mtDNA-seq also included to characterise deletion pattern",
            },
            {
                "term": "Hess Chart (Ocular Motility Mapping)",
                "definition": "Diplopia field mapping tool: patient views a grid through a red filter (one eye) and maps where they see a green dot with the other eye; Hess chart in PEO: symmetric restriction across all fields of gaze (bilateral, equal muscles affected) → explains why diplopia is uncommon despite severe ophthalmoplegia; asymmetric pattern would suggest myasthenia gravis or ocular motility disorder",
            },
            {
                "term": "Bell's Phenomenon Assessment",
                "definition": "Protective upward rotation of the globe when eyelids close forcibly; CRITICAL before ptosis surgery in PEO: if Bell's reflex is absent or reduced (extraocular muscles cannot move eye upward), ptosis correction risks corneal exposure during sleep → corneal ulceration/perforation; ophthalmology assessment mandatory; surgical correction targets lower position than standard ptosis to protect cornea",
            },
        ],
        "pharmacology": [
            {
                "term": "Valproate (VPA) — ABSOLUTE CI",
                "definition": "Absolutely contraindicated in all mtDNA instability diseases including POLG2/PEOA4: (1) valproyl-CoA sequesters free CoA → disrupts beta-oxidation AND CoA-dependent mitochondrial processes; (2) in mtDNA deletion diseases, VPA may directly accelerate mtDNA deletion accumulation through CoA-dependent mtDNA maintenance pathways; (3) acute liver failure documented in POLG1 patients given VPA; same risk in POLG2/PEOA4 if hepatic mtDNA is also compromised; NEVER prescribe VPA without confirming mitochondrial disease excluded",
            },
            {
                "term": "Coenzyme Q10 (Ubiquinone)",
                "definition": "Fat-soluble electron carrier between Complex I/II and Complex III in the mitochondrial electron transport chain; Level C evidence in mtDNA deletion diseases; 400-1200 mg/day with fat-containing meal (absorption improved); raises plasma CoQ10 levels; theoretical benefit: supports electron flow in fibers with residual OXPHOS capacity; some patients report reduced fatigue and improved exercise tolerance; no randomised trial; generally safe; caution with warfarin (CoQ10 may potentiate anticoagulation → INR monitoring)",
            },
            {
                "term": "Levetiracetam (LEV) — Preferred AED",
                "definition": "First-line AED in any mitochondrial disease if seizures develop: SV2A modulator; renal excretion 70%; no CYP450 induction or inhibition; no CoA sequestration; no hepatotoxicity; available IV; safe across all mitochondrial disease subtypes; behavioural adverse effects (irritability/agitation) in ~10% of patients; dose: 20-60 mg/kg/day divided BID; seizures are UNCOMMON in POLG2/PEOA4 (~12%); if present, investigate for alternative cause (e.g. POLG1 compound mutation)",
            },
            {
                "term": "Propofol — AVOID (PRIS Risk)",
                "definition": "Propofol Infusion Syndrome: propofol inhibits Complex I of the mitochondrial respiratory chain AND uncouples mitochondrial beta-oxidation → lactic acidosis, cardiac failure, rhabdomyolysis; risk dramatically elevated in any pre-existing OXPHOS deficiency including POLG2/PEOA4; particularly dangerous in ptosis surgery anaesthesia (operation performed in PEOA4 patients specifically). Safe alternatives: sevoflurane (inhalation anaesthetic, no known mitochondrial toxicity), ketamine (IV, dissociative, no Complex I inhibition); short single-dose propofol may be acceptable for anaesthesia induction (not prolonged infusion)",
            },
            {
                "term": "Riboflavin (Vitamin B2)",
                "definition": "FAD/FMN precursor; required as prosthetic group by Complex I (NADH dehydrogenase, 6 FAD-containing subunits) and Complex II (succinate dehydrogenase, 1 FAD); may partially rescue OXPHOS function in deletion disease; Level C evidence; 100-400 mg/day divided; urine turns bright yellow (harmless); safe; high single doses (>25 mg) saturate intestinal absorbers — split dosing maximises absorption; most beneficial in Complex I/II-deficient phenotypes; minimal evidence specifically for POLG2 but standard supportive supplement in adult mitochondrial disease",
            },
            {
                "term": "Aerobic Exercise Therapy",
                "definition": "Emerging evidence in mtDNA deletion diseases: moderate aerobic exercise stimulates mitochondrial biogenesis (via PGC-1α), may improve heteroplasmy (diluting deleted mtDNA by promoting replication of full-length mtDNA), and maintains muscle mass; Level B for adult mitochondrial myopathy broadly; target: 30-min moderate-intensity (60-70% VO2max) aerobic exercise 5 times/week; avoid: exhausting anaerobic exercise (lactic acidosis risk); individualised exercise prescription with physiotherapist familiar with mitochondrial disease",
            },
        ],
        "thresholds": [
            {
                "threshold": "PEO + family history + adult onset",
                "action": "Mitochondrial gene panel (POLG1/POLG2/SLC25A4/TWNK/RNASEH1); muscle biopsy (COX/SDH/Gomori); mtDNA deletion panel (LR-PCR + copy number qPCR); genetic counselling",
            },
            {
                "threshold": "SNHL on annual audiometry (PTA >25 dB in high frequencies)",
                "action": "Audiology referral; hearing aid fitting (>40 dB) or cochlear implant assessment if severe; annual follow-up",
            },
            {
                "threshold": "Pre-operative assessment for ptosis surgery",
                "action": "Bell's phenomenon assessment; anaesthesia briefed (avoid propofol/VPA); Hess chart; corneal sensation; target conservative ptosis correction to prevent corneal exposure",
            },
            {
                "threshold": "CK >5× ULN or rhabdomyolysis (dark urine, severe myalgia)",
                "action": "Discontinue any exercise; IV hydration; myoglobin urine; renal function; review medications (statins exacerbate in mitochondrial myopathy); review activity level",
            },
            {
                "threshold": "Seizures (new onset in PEOA4)",
                "action": "Ensure VPA NEVER prescribed; investigate for alternative cause (compound POLG1 mutation? head trauma? vascular?); LEV first line; neurology referral; EEG; brain MRI",
            },
            {
                "threshold": "Plasma lactate >3.5 mmol/L at rest",
                "action": "Review for intercurrent illness (infection raises lactate independently); metabolic review; avoid high-intensity exercise; cardiology if arrhythmia suspected; reassess diagnosis if persistent",
            },
            {
                "threshold": "Worsening ophthalmoplegia + diplopia developing",
                "action": "Ophthalmology: prism glasses trial; botulinum toxin to overacting muscles (if asymmetric); surgical correction if prism inadequate; never prescribe drugs that worsen neuromuscular junction (aminoglycosides, quinolones)",
            },
            {
                "threshold": "Family history confirmed POLG2 AD variant (cascade testing)",
                "action": "Genetic counselling: 50% transmission per pregnancy; predictive testing offered to at-risk adult relatives; surveillance: annual ophthalmology + audiology + neurological assessment; baseline muscle MRI/biopsy if symptomatic",
            },
        ],
    }
