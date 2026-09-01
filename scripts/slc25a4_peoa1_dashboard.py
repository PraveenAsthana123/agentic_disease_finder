#!/usr/bin/env python3
"""SLC25A4 / ANT1 — Progressive External Ophthalmoplegia Autosomal Dominant 1 (PEOA1).

SLC25A4-Related Mitochondrial Disease — AD adPEO (ANT1 dominant-negative):
  Progressive External Ophthalmoplegia Autosomal Dominant 1 (PEOA1)
  OMIM Disease #157640  ·  OMIM Gene SLC25A4 *103220

SLC25A4 (ANT1; 298 aa; 4q35.1) encodes Adenine Nucleotide Translocator 1, the
dominant isoform in adult heart and skeletal muscle. ANT1 forms a homodimer in the
inner mitochondrial membrane and exchanges ADP (into matrix) for ATP (into cytoplasm)
— the final step in oxidative phosphorylation that delivers ATP to the cytoplasm.

TWO COMPLETELY DIFFERENT SLC25A4 DISEASES:
  AR biallelic LOF → SLC25A4 MDDS2 (#615418):
    complete loss of ANT1 → severe cardiomyopathic mtDNA DEPLETION;
    infantile-fatal; HCM 100%; mtDNA copy number drops dramatically.
  AD heterozygous dominant negative → SLC25A4 PEOA1 (#157640):
    dominant negative missense in TM domain → partial ADP/ATP exchange impairment
    → dNTP pool imbalance → replication stalling → mtDNA MULTIPLE DELETIONS (not
    depletion); adult-onset; PEO + exercise intolerance + proximal myopathy; no HCM.

DOMINANT NEGATIVE MECHANISM (PEOA1):
  ANT1 functions as a homodimer (two 298-aa protomers in antiparallel orientation,
  each with 6 transmembrane helices, embedded in the IMM). The ADP/ATP exchange
  cycle requires precise conformational switching between the c-state (cytoplasmic
  open, binds ADP) and m-state (matrix open, releases ADP, takes up ATP).
  Dominant negative missense variants (typically in TM3 or TM5) disrupt one
  subunit in the ANT1 homodimer, but the mutant subunit co-assembles with the
  WT subunit → the heterodimer has impaired conformational mobility → reduced
  ADP/ATP exchange capacity in mitochondria of skeletal muscle and heart.
  The MATRIX consequence: ATP/ADP ratio inside the mitochondrial matrix falls →
  reduced substrate availability for mtDNA maintenance enzymes (pol-γ, TWNK,
  dNTP-synthesising enzymes) → dNTP pool imbalance → replication fork stalling
  at regions flanked by direct repeats → error-prone DSB repair at direct-repeat
  microhomologies → progressive MULTIPLE mtDNA DELETIONS accumulate in post-mitotic
  tissues (skeletal muscle, brain, heart; negligible in liver/kidney where ANT2/ANT3
  compensate for ANT1 deficiency — explains hepatic sparing identical to AR MDDS2).

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — identical mechanism to all other mtDNA
     instability diseases: valproyl-CoA sequesters free CoA → disrupts mtDNA
     replication machinery → accelerates multiple deletion burden; NEVER use VPA
     in any SLC25A4/ANT1 disease (AR or AD form)
  2. KD = CONTRAINDICATED — skeletal muscle COX-negative fibers (deletions) cannot
     sustain OXPHOS-dependent fat oxidation; ketogenic metabolic stress → lactic
     acidosis + myalgia + rhabdomyolysis; intact complex I–IV required for KD
  3. Propofol = AVOID (PRIS) — propofol inhibits Complex I + uncouples
     beta-oxidation in OXPHOS-deficient cells → fatal Propofol Infusion Syndrome
     risk in pre-existing mitochondrial disease; use sevoflurane or ketamine
  4. PEOA1 ≠ MDDS2 — CRITICAL DISTINCTION: PEOA1 (AD, heterozygous) causes
     multiple DELETIONS + adult PEO; MDDS2 (AR, biallelic) causes DEPLETION +
     infantile cardiomyopathy; SAME GENE (SLC25A4 4q35.1) but opposite allele
     dosage → diametrically opposite phenotypes; the dominant negative "poisons"
     the ADP/ATP exchange step without eliminating it; biallelic LOF eliminates it
  5. PEO 100% — bilateral ptosis + progressive ophthalmoplegia = CARDINAL feature
     of PEOA1; same as POLG2/PEOA4 and DNA2/PEOA5; Bell's phenomenon assessment
     MANDATORY before ptosis surgery (corneal exposure risk if absent)
  6. EXERCISE INTOLERANCE ~85% — hallmark of ANT1-mediated disease; reduced lactate
     threshold; exercise-induced myalgia + fatigue + CK rise; lactic acidosis with
     moderate exertion; aerobic training programme improves mitochondrial biogenesis
  7. PROXIMAL MYOPATHY ~80% — hip-girdle + shoulder-girdle weakness; CK normal or
     mildly elevated (<5× ULN); COX-negative fibers on Gomori trichrome + COX/SDH
     double stain; ragged-red fibers in severely affected fibres
  8. NO HEPATOPATHY — KEY DDx from POLG1/Alpers (80% hepatopathy), DGUOK, MPV17,
     TWNK AR; ANT1 AD PEOA1 spares liver because ANT2/ANT3 compensate in hepatocytes
  9. NO HCM — KEY DDx from SLC25A4 AR MDDS2 (HCM 100%); PEOA1 has rare mild
     cardiomyopathy (<10%); HCM in infancy is exclusive to AR loss-of-function MDDS2
 10. mtDNA MULTIPLE DELETIONS (NOT DEPLETION) — long-range PCR on muscle (blood
     unreliable); copy number NORMAL (key DDx from all MDDS/depletion syndromes);
     identical molecular fingerprint to POLG2/DNA2/TWNK-adPEO/RNASEH1-ARCO
 11. KAUKONEN 2000 (Science 289:133) — first identification of heterozygous ANT1
     mutations in Finnish families with adPEO; 3 mutations (p.A114P, p.V289M,
     p.L98P) in TM domain; mtDNA multiple deletions on muscle biopsy; established
     ANT1 as a cause of adPEO (before this, all adPEO was "chromosomally unknown")
 12. EXERCISE TRAINING — Level B evidence (mitochondrial myopathy, including adPEO);
     aerobic exercise increases mitochondrial biogenesis; improves exercise capacity;
     reduces proportion of mutant-deleted fibers (heteroplasmy shift in satellite cells);
     30 min aerobic × 5/week; avoid anaerobic HIIT (rhabdomyolysis risk)
 13. SNHL ~35% — progressive sensorineural hearing loss; more common than in POLG2
     (~40%) but similar; audiogram annually; cochlear implant if severe (AVOID
     propofol for anaesthesia in susceptible patients)
 14. NO LEUKOENCEPHALOPATHY — KEY DDx from TYMP/MNGIE (white matter 100%);
     brain MRI in PEOA1 shows cerebellar atrophy at most; no WM disease
 15. CoQ10 Level C — 400–1200 mg/day; supports residual OXPHOS function in
     deletion-bearing fibers; standard supplementation for all mtDNA deletion diseases

SLC25A4 / ANT1 BIOLOGY:
SLC25A4 (298 amino acids; 4q35.1) is the predominant adenine nucleotide translocator
isoform in adult heart and skeletal muscle (ANT1), where it constitutes up to 10%
of inner mitochondrial membrane protein.
Structure:
  6 transmembrane helices (TM1–TM6): form the hydrophilic cavity of the carrier
  3 α-helical matrix loops (ML1, ML2, ML3): connect pairs of TM helices
  3 hydrophilic intermembrane space (IMS) loops: connect TM2-TM3, TM4-TM5, TM6-TM1
  Conserved MCF (Mitochondrial Carrier Family) signature: [RKX][RKX][FWY](20-30aa)D
  PEOA1 missense cluster: TM3 (p.A114P, p.L98P) and TM5 (p.V289M) disrupt TM helix
  packing → conformational rigidity → impaired c-state/m-state transition of dimer

ANT1 function:
  c-state: cytoplasmic-open binding site exposed; ADP from cytoplasm enters cavity
  Conformational transition: ADP binding triggers closure of IMS side, opening of
    matrix side (requires WT amphipathic helix mobility in TM3/TM5)
  m-state: matrix-open; releases ADP into matrix; takes up ATP (made by ATP synthase)
  Dominant negative: mutant p.A114P (proline substitution in TM3 α-helix) → kink in
    TM3 → helix mobility impaired → conformational switching slowed in the heterodimer
    → reduced ADP/ATP exchange → matrix ATP/ADP ratio falls → dNTP pool imbalance

Tissue specificity:
  ANT1: heart >> skeletal muscle; low in liver, kidney, brain
  ANT2: ubiquitous (proliferating cells); absent in post-mitotic cells
  ANT3: ubiquitous at low levels; maintains residual exchange in liver/kidney
  ANT1 knockdown in post-mitotic muscle = irreplaceable → phenotype severe
  ANT2/ANT3 compensate in liver/kidney → explains hepatic sparing in PEOA1 and MDDS2
"""

from __future__ import annotations

import random
from typing import Any

SEED = 577          # reproducible 40-patient cohort (SLC25A4-AD-PEOA1-adPEO)
N_PATIENTS = 40


def _rng() -> random.Random:
    return random.Random(SEED)


def get_overview() -> dict[str, Any]:
    rng = _rng()

    patients = _build_cohort(rng)

    n_peo = sum(1 for p in patients if p["peo"])
    n_exercise_intol = sum(1 for p in patients if p["exercise_intolerance"])
    n_myopathy = sum(1 for p in patients if p["proximal_myopathy"])
    n_snhl = sum(1 for p in patients if p["snhl"])
    n_ataxia = sum(1 for p in patients if p["ataxia"])
    n_depression = sum(1 for p in patients if p["depression"])
    n_dysphagia = sum(1 for p in patients if p["dysphagia"])
    n_cardiomyopathy = sum(1 for p in patients if p["cardiomyopathy"])

    avg_onset = round(sum(p["age_onset_years"] for p in patients) / N_PATIENTS, 1)

    return {
        "gene": "SLC25A4",
        "protein": "Adenine Nucleotide Translocator 1 (ANT1) — 298 aa",
        "disease": (
            "SLC25A4-Related Mitochondrial Disease — "
            "Progressive External Ophthalmoplegia Autosomal Dominant 1 (PEOA1 / adPEO-ANT1)"
        ),
        "omim_gene": "*103220",
        "omim_disease": "#157640 (PEOA1; AD heterozygous dominant-negative missense; Kaukonen 2000 Science)",
        "chromosome": "4q35.1",
        "inheritance": (
            "Autosomal Dominant (AD) — heterozygous dominant-negative missense in TM3 or TM5; "
            "single mutant allele sufficient to impair homodimer ADP/ATP exchange; "
            "CRITICAL DDx from SLC25A4 AR MDDS2 (biallelic LOF → HCM + depletion)"
        ),
        "onset": f"Adult — mean {avg_onset} years (range 25–65 years); earlier onset in TM3-null-like alleles",
        "mechanism": (
            "Heterozygous missense variant in TM3 or TM5 of the ANT1 homodimer → "
            "impaired proline-disrupted amphipathic helix mobility → slowed c-state/m-state "
            "conformational switching → reduced matrix ADP/ATP exchange capacity → dNTP pool "
            "imbalance in mitochondrial matrix → replication fork stalling at direct-repeat sequences → "
            "error-prone DSB repair → progressive large-scale mtDNA multiple deletions in post-mitotic "
            "skeletal muscle and cardiac muscle → COX-negative fibers → PEO + exercise intolerance + "
            "proximal myopathy. Normal mtDNA copy number (deletion NOT depletion)."
        ),
        "mtdna_pattern": (
            "Multiple deletions (NOT copy-number depletion) — muscle long-range PCR required; "
            "blood unreliable (poor sensitivity); mtDNA copy number NORMAL (key DDx from MDDS depletion series)"
        ),
        "key_labs": [
            "mtDNA multiple deletions on muscle long-range PCR / Southern blot (blood unreliable — deletions tissue-specific)",
            "COX-negative / ragged-red fibers on muscle biopsy (Gomori trichrome + COX/SDH double stain)",
            "Normal mtDNA copy number on quantitative PCR (key DDx from MDDS2 AR — depletion below 20%)",
            "SLC25A4 heterozygous missense panel — dominant negative TM3/TM5 variant in single allele; NGS mito panel",
            "CK: normal to mildly elevated (<5× ULN); lactate: elevated at rest and disproportionately with exercise",
            "Cardiology: ECG + ECHO annually (cardiomyopathy rare <10% in PEOA1; contrast HCM 100% in MDDS2)",
            "Ophthalmology: Hess chart, ocular motility, Bell's phenomenon, ptosis degree, corneal sensation",
            "Audiogram annually (SNHL ~35%; progressive sensorineural)",
            "Brain MRI: cerebellar atrophy (if ataxia); white matter SPARED (DDx from TYMP/MNGIE leukoencephalopathy)",
            "NCS/EMG: axonal sensory neuropathy pattern if neuropathy present (~15%); myopathic changes in severe cases",
        ],
        "kpis": [
            {"label": "PEO (Ptosis + Ophthalmoplegia)", "value": f"{n_peo}/{N_PATIENTS} ({round(n_peo/N_PATIENTS*100)}%)", "color": "#1b5e20"},
            {"label": "Exercise Intolerance", "value": f"{n_exercise_intol}/{N_PATIENTS} ({round(n_exercise_intol/N_PATIENTS*100)}%)", "color": "#2e7d32"},
            {"label": "Proximal Myopathy", "value": f"{n_myopathy}/{N_PATIENTS} ({round(n_myopathy/N_PATIENTS*100)}%)", "color": "#388e3c"},
            {"label": "SNHL", "value": f"{n_snhl}/{N_PATIENTS} ({round(n_snhl/N_PATIENTS*100)}%)", "color": "#43a047"},
            {"label": "Ataxia (mild)", "value": f"{n_ataxia}/{N_PATIENTS} ({round(n_ataxia/N_PATIENTS*100)}%)", "color": "#4caf50"},
            {"label": "Depression/Mood", "value": f"{n_depression}/{N_PATIENTS} ({round(n_depression/N_PATIENTS*100)}%)", "color": "#66bb6a"},
            {"label": "Dysphagia", "value": f"{n_dysphagia}/{N_PATIENTS} ({round(n_dysphagia/N_PATIENTS*100)}%)", "color": "#81c784"},
            {"label": "Cardiomyopathy (rare)", "value": f"{n_cardiomyopathy}/{N_PATIENTS} ({round(n_cardiomyopathy/N_PATIENTS*100)}%)", "color": "#a5d6a7"},
        ],
        "feature_bars": [
            {"label": "PEO (bilateral ptosis + ophthalmoplegia — CARDINAL)", "pct": round(n_peo / N_PATIENTS * 100)},
            {"label": "Exercise Intolerance (hallmark — reduced lactate threshold)", "pct": round(n_exercise_intol / N_PATIENTS * 100)},
            {"label": "Proximal Myopathy (hip + shoulder girdle)", "pct": round(n_myopathy / N_PATIENTS * 100)},
            {"label": "Sensorineural Hearing Loss (SNHL)", "pct": round(n_snhl / N_PATIENTS * 100)},
            {"label": "Ataxia (cerebellar, mild)", "pct": round(n_ataxia / N_PATIENTS * 100)},
            {"label": "Depression / Mood Disorder", "pct": round(n_depression / N_PATIENTS * 100)},
            {"label": "Dysphagia", "pct": round(n_dysphagia / N_PATIENTS * 100)},
            {"label": "Cardiomyopathy (RARE — key DDx from AR MDDS2 HCM 100%)", "pct": round(n_cardiomyopathy / N_PATIENTS * 100)},
        ],
        "contraindications": [
            {
                "drug": "Valproate (VPA)",
                "severity": "ABSOLUTE",
                "reason": (
                    "Valproyl-CoA sequesters free CoA → disrupts mtDNA replication machinery "
                    "(CoA required for pol-γ activity and dNTP metabolism) → accelerates mtDNA "
                    "multiple deletion accumulation in ANT1-PEOA1 skeletal muscle; additional risk: "
                    "occult mitochondrial disease → valproate-induced Reye-like hepatotoxicity; "
                    "mechanism identical to VPA CI in all mtDNA instability diseases; "
                    "NEVER prescribe VPA in SLC25A4-PEOA1 (or any ANT1-related disease)"
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "severity": "CONTRAINDICATED",
                "reason": (
                    "KD forces OXPHOS-dependent beta-oxidation (>70% energy from fat); "
                    "COX-negative fibers in PEOA1 skeletal muscle cannot sustain the metabolic "
                    "demand; impaired ADP/ATP exchange (dominant negative ANT1) further limits "
                    "fatty acid oxidation capacity; risk of lactic acidosis + rhabdomyolysis under "
                    "ketotic stress; intact respiratory chain required for fat oxidation"
                ),
            },
            {
                "drug": "Propofol (prolonged infusion)",
                "severity": "AVOID",
                "reason": (
                    "Propofol Infusion Syndrome (PRIS): propofol inhibits Complex I + uncouples "
                    "beta-oxidation → fatal lactic acidosis + cardiac failure in susceptible patients "
                    "with pre-existing OXPHOS deficiency; elevated risk in all mtDNA deletion diseases "
                    "including SLC25A4-PEOA1; preferred alternatives: sevoflurane or ketamine"
                ),
            },
        ],
        "ddx_highlights": [
            "PEOA1 ≠ MDDS2 — SAME GENE (SLC25A4) but OPPOSITE phenotypes: AD missense = multiple deletions + adult PEO; AR LOF = depletion + infantile HCM",
            "NO HCM — KEY DDx from SLC25A4 AR MDDS2 (HCM 100%, fatal before age 2); cardiomyopathy RARE in PEOA1 (<10%)",
            "NO hepatopathy — KEY DDx from POLG1/Alpers, DGUOK, MPV17, TWNK AR; ANT2/ANT3 compensate in liver",
            "NO leukoencephalopathy — KEY DDx from TYMP/MNGIE (diffuse WM 100%); PEOA1 MRI shows cerebellar atrophy only",
            "Normal mtDNA copy number — KEY DDx from all mtDNA depletion syndromes (copy number drops in MDDS2/POLG/DGUOK/TK2)",
            "Exercise intolerance HALLMARK (~85%) — more prominent in PEOA1 than in POLG2/PEOA4; reduced lactate threshold; aerobic training improves",
            "Ataxia MILD (~20%) — much less than RNASEH1-ARCO (~85%) and DNA2/PEOA5 (~55%); PEO is primary not ataxia",
            "TM3/TM5 dominant negative cluster (p.A114P, p.V289M, p.L98P) — Finnish founder effect; same 4q35.1 locus as MDDS2",
        ],
        "references": [
            {
                "author": "Kaukonen J et al.",
                "year": 2000,
                "journal": "Science",
                "title": "Role of adenine nucleotide translocator 1 in mtDNA maintenance",
                "note": (
                    "First identification of heterozygous ANT1 missense mutations in Finnish families "
                    "with autosomal dominant adPEO + multiple mtDNA deletions; 3 mutations identified: "
                    "p.A114P (TM3), p.V289M (TM5), p.L98P (TM3); established ANT1/SLC25A4 as a cause of "
                    "adPEO alongside TWNK (then unknown); demonstrated that dominant negative ANT1 mutants "
                    "cause deletions not depletion; Science 289:133-136"
                ),
            },
            {
                "author": "Graham BH et al.",
                "year": 1997,
                "journal": "Nat Genet",
                "title": "A mouse model for mitochondrial myopathy and cardiomyopathy resulting from a deficiency in the heart/muscle isoform of the adenine nucleotide translocator",
                "note": (
                    "Ant1-knockout mouse: homozygous Ant1-/- → skeletal myopathy + cardiomyopathy + "
                    "ragged-red fibers + multiple mtDNA deletions; established ANT1 as essential for "
                    "mtDNA integrity in post-mitotic tissues; heart-predominant expression confirmed; "
                    "heterozygous Ant1+/- showed partial phenotype (like AD dominance)"
                ),
            },
            {
                "author": "Napoli L et al.",
                "year": 2001,
                "journal": "Ann Neurol",
                "title": "A novel missense adenine nucleotide translocator-1 gene mutation in a Greek adPEO family",
                "note": (
                    "Additional ANT1 mutations in non-Finnish PEOA1 families; expanded genotype-phenotype; "
                    "confirmed that exercise intolerance and proximal myopathy are the predominant "
                    "functional deficits after PEO; underscored importance of long-range PCR on muscle "
                    "(blood deletions absent in most families)"
                ),
            },
        ],
    }


def _build_cohort(rng: random.Random) -> list[dict]:
    """Build the 40-patient SLC25A4-AD-PEOA1 cohort deterministically (seed 577)."""
    etiology_classes = [
        ("AD-Missense-TM3-p.A114P-Finnish-Founder", 30),
        ("AD-Missense-TM5-p.V289M-Finnish-Founder", 25),
        ("AD-Missense-TM3-Other-Non-Finnish", 20),
        ("AD-Missense-TM5-Other-Non-Finnish", 15),
        ("AD-Phenocopy-SLC25A4-Panel-Negative-adPEO", 10),
    ]
    etiology_pool: list[str] = []
    for name, pct in etiology_classes:
        etiology_pool.extend([name] * pct)

    extraocular_patterns = [
        "Complete-Ophthalmoplegia-All-Directions",
        "Incomplete-Ophthalmoplegia-Bilateral-Elevation-Limited",
        "Incomplete-Ophthalmoplegia-Bilateral-All-Directions",
        "Limited-Abduction-Bilateral",
        "Bilateral-Ptosis-Only-No-Ophthalmoplegia-Yet",
    ]
    misdiagnosis_pool = [
        "Myasthenia-Gravis-Seronegative",
        "CPEO-Sporadic-Adult-Onset-Unclassified",
        "Mitochondrial-Myopathy-Unspecified",
        "POLG-Related-adPEO",
        "Limb-Girdle-Muscular-Dystrophy",
        "Oculopharyngeal-Muscular-Dystrophy-OPMD",
    ]

    patients = []
    for i in range(N_PATIENTS):
        pid = f"PEOA1-ANT1-{i+1:03d}"
        etiology = rng.choice(etiology_pool)
        age_onset = round(rng.gauss(38, 10), 0)
        age_onset = max(25, min(65, int(age_onset)))

        peo = True  # 100% cardinal
        ptosis_bilateral = True
        oph_pattern = rng.choice(extraocular_patterns)
        exercise_intolerance = rng.random() < 0.85
        proximal_myopathy = rng.random() < 0.80
        snhl = rng.random() < 0.35
        ataxia = rng.random() < 0.20
        depression = rng.random() < 0.25
        dysphagia = rng.random() < 0.20
        cardiomyopathy = rng.random() < 0.08   # RARE in PEOA1 (contrast with HCM 100% in MDDS2)
        sensory_neuropathy = rng.random() < 0.15
        parkinsonism = rng.random() < 0.10
        seizures = rng.random() < 0.07

        ck_x_uln = round(rng.uniform(0.8, 4.5), 1)   # normal to mildly elevated
        lactate_rest_mmol = round(rng.uniform(1.2, 3.5), 1)
        deletion_load_pct = round(rng.uniform(15, 65), 0)  # % of fibres COX-negative
        dx_delay_years = round(rng.uniform(2, 12), 1)
        misdiagnosis = rng.choice(misdiagnosis_pool)

        patients.append({
            "id": pid,
            "etiology": etiology,
            "age_onset_years": age_onset,
            "peo": peo,
            "ptosis_bilateral": ptosis_bilateral,
            "ophthalmoplegia_pattern": oph_pattern,
            "exercise_intolerance": exercise_intolerance,
            "proximal_myopathy": proximal_myopathy,
            "snhl": snhl,
            "ataxia": ataxia,
            "depression": depression,
            "dysphagia": dysphagia,
            "cardiomyopathy": cardiomyopathy,
            "sensory_neuropathy": sensory_neuropathy,
            "parkinsonism": parkinsonism,
            "seizures": seizures,
            "ck_x_uln": ck_x_uln,
            "lactate_rest_mmol": lactate_rest_mmol,
            "deletion_load_pct": int(deletion_load_pct),
            "dx_delay_years": dx_delay_years,
            "initial_misdiagnosis": misdiagnosis,
        })
    return patients


def get_breakdown() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    n_peo = sum(1 for p in patients if p["peo"])
    n_exercise = sum(1 for p in patients if p["exercise_intolerance"])
    n_myopathy = sum(1 for p in patients if p["proximal_myopathy"])
    n_snhl = sum(1 for p in patients if p["snhl"])
    n_ataxia = sum(1 for p in patients if p["ataxia"])
    n_depression = sum(1 for p in patients if p["depression"])
    n_dysphagia = sum(1 for p in patients if p["dysphagia"])
    n_cardiomyopathy = sum(1 for p in patients if p["cardiomyopathy"])
    n_neuropathy = sum(1 for p in patients if p["sensory_neuropathy"])
    n_parkinsonism = sum(1 for p in patients if p["parkinsonism"])
    n_seizures = sum(1 for p in patients if p["seizures"])

    avg_onset = round(sum(p["age_onset_years"] for p in patients) / N_PATIENTS, 1)
    avg_dx_delay = round(sum(p["dx_delay_years"] for p in patients) / N_PATIENTS, 1)
    avg_deletion_load = round(sum(p["deletion_load_pct"] for p in patients) / N_PATIENTS, 0)

    etiology_counts: dict[str, int] = {}
    for p in patients:
        etiology_counts[p["etiology"]] = etiology_counts.get(p["etiology"], 0) + 1

    misdiag_counts: dict[str, int] = {}
    for p in patients:
        m = p["initial_misdiagnosis"]
        misdiag_counts[m] = misdiag_counts.get(m, 0) + 1

    oph_counts: dict[str, int] = {}
    for p in patients:
        oph_counts[p["ophthalmoplegia_pattern"]] = oph_counts.get(p["ophthalmoplegia_pattern"], 0) + 1

    treatments = [
        {
            "name": "Coenzyme Q10",
            "tier": "First-Line Supplement",
            "evidence": "Level C",
            "mechanism": (
                "Electron carrier in the mitochondrial respiratory chain (Complex I→III shuttle); "
                "supports OXPHOS function in COX-deficient fibers; antioxidant may reduce oxidative "
                "mtDNA damage and slow secondary deletion accumulation; no evidence of deletion reversal "
                "but improves cellular energetics in deletion-bearing fibers; standard supplementation "
                "for all mtDNA deletion diseases including SLC25A4-PEOA1"
            ),
            "dose": "400–1200 mg/day in 2–3 divided doses with fat-containing meal (fat-soluble vitamin)",
            "monitoring": "Plasma CoQ10 levels (target >2.5 μg/mL); LFTs at 3 months; GI tolerance",
            "caution": "Well tolerated; nausea at high doses (take with food); may potentiate warfarin (monitor INR)",
        },
        {
            "name": "Aerobic Exercise Training",
            "tier": "First-Line Supportive — Exercise Intolerance",
            "evidence": "Level B (mitochondrial myopathy/adPEO)",
            "mechanism": (
                "Aerobic exercise induces mitochondrial biogenesis (PGC-1α activation) in "
                "non-deleted muscle fibres → increases total mitochondrial content → improves "
                "exercise capacity despite residual deletion burden; activates satellite cells → "
                "regenerating fibres inherit lower deletion proportions (heteroplasmy shift); "
                "improves lactate threshold; reduces fatigue; safe if intensity is aerobic (not HIIT)"
            ),
            "dose": (
                "30 min moderate aerobic exercise × 5/week (cycling/swimming/walking); "
                "intensity: 60–75% max heart rate or Borg RPE 12–14 (moderate); "
                "incremental increase over 8–12 weeks; annual reassessment with 6MWT and CPET"
            ),
            "monitoring": "6-minute walk test (6MWT) 6-monthly; CK post-exercise (avoid rhabdomyolysis); CPET annually",
            "caution": "Avoid high-intensity interval training (HIIT) — anaerobic threshold exceeded → rhabdomyolysis risk in COX-deficient fibres",
        },
        {
            "name": "Levetiracetam (LEV)",
            "tier": "Preferred AED (if seizures)",
            "evidence": "Level B (mitochondrial epilepsy broadly)",
            "mechanism": (
                "SV2A modulator; renal excretion 70%; no CYP450 interaction; no CoA sequestration; "
                "no hepatotoxicity; no QTc prolongation; safest option in mtDNA deletion disease "
                "where VPA is absolutely contraindicated; seizures uncommon in PEOA1 (~7%)"
            ),
            "dose": "20–60 mg/kg/day divided BID; IV loading 20–40 mg/kg for status epilepticus",
            "monitoring": "Renal function 6-monthly; behavioural AEs (irritability) ~10%; CBC annually",
            "caution": "Seizures uncommon in SLC25A4-PEOA1 (<8%); if present, LEV preferred over all hepatically-metabolised AEDs; VPA ABSOLUTE CI",
        },
        {
            "name": "Ptosis Surgery (Frontalis Sling / Levator Advancement)",
            "tier": "Surgical Intervention",
            "evidence": "Level C",
            "mechanism": (
                "Surgical correction of bilateral ptosis when visual field obstruction is confirmed; "
                "frontalis sling (silicone rod / fascia lata) or levator advancement tightens the "
                "weak levator palpebrae superioris; anaesthesia team MUST be briefed: propofol AVOID, "
                "mitochondrial disease card carried"
            ),
            "dose": "Pre-op: Hess chart + ocular motility + Bell's reflex + corneal sensation — all MANDATORY before surgery",
            "monitoring": "Post-op corneal exposure monitoring; lubricating eye drops; 1-week, 1-month, 3-month review",
            "caution": (
                "Bell's phenomenon ABSENT in severe PEO → corneal exposure and exposure keratitis if "
                "ptosis over-corrected; lower correction target when Bell's absent; AVOID propofol "
                "(PRIS risk); use sevoflurane or ketamine for anaesthesia"
            ),
        },
        {
            "name": "Riboflavin (Vitamin B2)",
            "tier": "Adjunct Supplement",
            "evidence": "Level C",
            "mechanism": (
                "FAD/FMN precursor; required by Complex I (NADH dehydrogenase) and Complex II; "
                "supports residual respiratory chain capacity in non-deleted fibres; may partially "
                "compensate for reduced OXPHOS function downstream of impaired ANT1 ADP/ATP exchange"
            ),
            "dose": "100–400 mg/day in 2–3 divided doses",
            "monitoring": "Urine turns fluorescent yellow (harmless); clinical response at 3–6 months; no liver toxicity",
            "caution": "Generally safe; absorption decreases at single doses >25 mg — split dosing required for full effect",
        },
        {
            "name": "Speech and Language Therapy (SLT)",
            "tier": "Supportive — Dysphagia",
            "evidence": "Standard of care",
            "mechanism": (
                "Compensatory swallowing strategies reduce aspiration risk; texture modification "
                "and bolus control for pharyngeal dysphagia; videofluoroscopy quantifies aspiration; "
                "PEG if severe dysphagia and weight loss; AVOID propofol for any procedure"
            ),
            "dose": "SLT assessment at diagnosis; review if dysphagia develops; annual reassessment",
            "monitoring": "Weight and nutritional status; aspiration pneumonia surveillance",
            "caution": "Dysphagia in PEOA1 is pharyngeal/oesophageal weakness (not cerebellar origin as in RNASEH1); SLT assessment tailored accordingly",
        },
    ]

    return {
        "summary": {
            "n_patients": N_PATIENTS,
            "avg_onset_years": avg_onset,
            "avg_dx_delay_years": avg_dx_delay,
            "avg_deletion_load_pct": int(avg_deletion_load),
            "peo_pct": round(n_peo / N_PATIENTS * 100),
            "exercise_intol_pct": round(n_exercise / N_PATIENTS * 100),
            "myopathy_pct": round(n_myopathy / N_PATIENTS * 100),
            "snhl_pct": round(n_snhl / N_PATIENTS * 100),
            "ataxia_pct": round(n_ataxia / N_PATIENTS * 100),
            "depression_pct": round(n_depression / N_PATIENTS * 100),
            "dysphagia_pct": round(n_dysphagia / N_PATIENTS * 100),
            "cardiomyopathy_pct": round(n_cardiomyopathy / N_PATIENTS * 100),
            "neuropathy_pct": round(n_neuropathy / N_PATIENTS * 100),
            "parkinsonism_pct": round(n_parkinsonism / N_PATIENTS * 100),
            "seizures_pct": round(n_seizures / N_PATIENTS * 100),
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
            {"label": "PEO (bilateral ptosis + ophthalmoplegia — CARDINAL 100%)", "pct": round(n_peo / N_PATIENTS * 100)},
            {"label": "Exercise Intolerance (hallmark — reduced lactate threshold)", "pct": round(n_exercise / N_PATIENTS * 100)},
            {"label": "Proximal Myopathy (hip + shoulder girdle)", "pct": round(n_myopathy / N_PATIENTS * 100)},
            {"label": "Sensorineural Hearing Loss (SNHL)", "pct": round(n_snhl / N_PATIENTS * 100)},
            {"label": "Ataxia (cerebellar, mild)", "pct": round(n_ataxia / N_PATIENTS * 100)},
            {"label": "Depression / Mood Disorder", "pct": round(n_depression / N_PATIENTS * 100)},
            {"label": "Dysphagia", "pct": round(n_dysphagia / N_PATIENTS * 100)},
            {"label": "Cardiomyopathy (RARE — key DDx from AR MDDS2 HCM 100%)", "pct": round(n_cardiomyopathy / N_PATIENTS * 100)},
            {"label": "Sensory Neuropathy (axonal, mild)", "pct": round(n_neuropathy / N_PATIENTS * 100)},
            {"label": "Parkinsonism (mild, partial L-DOPA)", "pct": round(n_parkinsonism / N_PATIENTS * 100)},
            {"label": "Seizures (uncommon)", "pct": round(n_seizures / N_PATIENTS * 100)},
        ],
        "patients": [
            {
                "id": p["id"],
                "etiology": p["etiology"],
                "age_onset": p["age_onset_years"],
                "peo": p["peo"],
                "oph_pattern": p["ophthalmoplegia_pattern"],
                "exercise_intol": p["exercise_intolerance"],
                "myopathy": p["proximal_myopathy"],
                "snhl": p["snhl"],
                "ataxia": p["ataxia"],
                "cardiomyopathy": p["cardiomyopathy"],
                "neuropathy": p["sensory_neuropathy"],
                "seizures": p["seizures"],
                "ck_x_uln": p["ck_x_uln"],
                "lactate": p["lactate_rest_mmol"],
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
                "term": "SLC25A4 / ANT1",
                "definition": (
                    "Solute Carrier Family 25 Member 4 / Adenine Nucleotide Translocator 1; "
                    "298 amino acids; 4q35.1. ANT1 is the dominant isoform in adult heart and "
                    "skeletal muscle (constitutes ~10% of inner mitochondrial membrane protein). "
                    "Forms a homodimer; exchanges ADP (cytoplasm→matrix) for ATP (matrix→cytoplasm) "
                    "— the final step coupling oxidative phosphorylation to cytoplasmic ATP delivery. "
                    "Essential for post-mitotic cell survival; ANT2/ANT3 compensate in liver/kidney."
                ),
            },
            {
                "term": "Dominant Negative Mechanism (PEOA1)",
                "definition": (
                    "Heterozygous missense in TM3 (e.g., p.A114P — proline kink disrupts helix "
                    "amphipathicity) or TM5 (e.g., p.V289M — valine→methionine in transmembrane "
                    "packing interface) of one ANT1 allele. The mutant protomer co-assembles with "
                    "WT protomer in the homodimer. The heterodimer has impaired c-state/m-state "
                    "conformational switching → reduced ADP/ATP exchange in ~50% of ANT1 dimers → "
                    "dNTP pool imbalance → multiple mtDNA deletions. NOT haploinsufficiency."
                ),
            },
            {
                "term": "ANT1 vs ANT2 vs ANT3 (Isoform Specificity)",
                "definition": (
                    "ANT1 (SLC25A4): heart + skeletal muscle dominant; post-mitotic; 4q35.1. "
                    "ANT2 (SLC25A5): ubiquitous but absent in non-proliferating/differentiated cells; Xq24. "
                    "ANT3 (SLC25A6): ubiquitous at low levels including liver/kidney; Xp22.32. "
                    "ANT1 disease (PEOA1, MDDS2) is post-mitotic tissue-specific because ANT2/ANT3 "
                    "compensate in mitotically active cells. Explains why PEOA1 spares liver."
                ),
            },
            {
                "term": "mtDNA Multiple Deletions (PEOA1 molecular pattern)",
                "definition": (
                    "Large-scale mtDNA deletions (1–8 kb), often involving the D-loop or the "
                    "major arc region, accumulating over years in post-mitotic muscle and brain. "
                    "Multiple different deletion breakpoints (not a single deletion) — distinguish "
                    "from KSS / sporadic CPEO (single large-deletion) and MDDS (copy number falls). "
                    "Detected by long-range PCR (muscle preferred; blood insensitive) or Southern "
                    "blot. mtDNA copy number NORMAL."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "PEOA1 — Progressive External Ophthalmoplegia Autosomal Dominant 1",
                "definition": (
                    "OMIM #157640. AD heterozygous SLC25A4 missense → adPEO + multiple mtDNA "
                    "deletions in muscle. Kaukonen 2000 (Science). PEO + exercise intolerance + "
                    "proximal myopathy cardinal triad. Distinct from MDDS2 (AR biallelic LOF → "
                    "HCM + depletion). Same gene, opposite allele effect."
                ),
            },
            {
                "term": "MDDS2 vs PEOA1 — The Two-Disease Paradox of SLC25A4",
                "definition": (
                    "Two diseases from the same gene: (1) MDDS2 (#615418): AR biallelic LOF → "
                    "complete loss of ANT1 → HCM 100% + mtDNA depletion + infantile-fatal; "
                    "(2) PEOA1 (#157640): AD heterozygous dominant-negative missense → partial "
                    "ANT1 dysfunction → multiple deletions + adult-onset PEO + exercise intolerance. "
                    "This is one of the most striking allelic heterogeneity examples in mitochondrial genetics."
                ),
            },
            {
                "term": "COX-Negative Fibres",
                "definition": (
                    "Cytochrome c oxidase (Complex IV)-negative fibres on COX/SDH double stain "
                    "muscle biopsy. COX-neg fibres retain SDH (Complex II activity, nDNA-encoded) "
                    "→ appear blue. Pathognomonic for mtDNA deletion/depletion disease. In PEOA1, "
                    "COX-negative fibre proportion correlates with deletion load and clinical severity."
                ),
            },
            {
                "term": "Exercise Intolerance in ANT1 Disease",
                "definition": (
                    "Hallmark of PEOA1: reduced lactate threshold; disproportionate lactic acidosis "
                    "with moderate exertion; exercise-induced myalgia + fatigue + CK elevation. "
                    "Mechanism: ANT1 dominant negative reduces ADP/ATP exchange → ATP delivery "
                    "limited → anaerobic glycolysis recruited earlier → lactate rises earlier. "
                    "Aerobic training reverses this partially by inducing mitochondrial biogenesis."
                ),
            },
            {
                "term": "Bell's Phenomenon",
                "definition": (
                    "Upward and outward rotation of the eyeball on attempted eyelid closure "
                    "(protective reflex). In PEO patients with ptosis surgery consideration: if "
                    "Bell's phenomenon is ABSENT (severe PEO), the cornea is not protected when "
                    "eyelids do not close fully → corneal exposure + keratitis risk after ptosis "
                    "surgery. MANDATORY pre-operative assessment before any ptosis repair in PEOA1."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "VPA Absolute Contraindication — Mechanism in mtDNA Deletion Disease",
                "definition": (
                    "Valproic acid → valproyl-CoA (via mitochondrial beta-oxidation) → sequesters "
                    "free intramitochondrial CoA → CoA depletion disrupts the TWNK-POLG-dNTP "
                    "machinery (CoA required for pol-γ priming reactions and for dNTP biosynthesis "
                    "enzyme cofactors) → accelerates pre-existing mtDNA multiple deletion accumulation. "
                    "Irreversible in post-mitotic tissue. Additionally: occult mitochondrial disease "
                    "predisposes to VPA-induced Reye-like hepatotoxicity. NEVER use in SLC25A4-PEOA1."
                ),
            },
            {
                "term": "PRIS — Propofol Infusion Syndrome",
                "definition": (
                    "Propofol inhibits Complex I (NADH-CoQ reductase) + disrupts beta-oxidation "
                    "in the inner mitochondrial membrane → fatal lactic acidosis + myocardial failure "
                    "in susceptible patients with pre-existing OXPHOS dysfunction. SLC25A4-PEOA1 "
                    "patients have impaired ADP/ATP exchange + COX-negative fibers → PRIS risk elevated. "
                    "Avoid propofol for all anaesthesia procedures. Use sevoflurane or ketamine."
                ),
            },
            {
                "term": "Preferred AED — Levetiracetam (LEV)",
                "definition": (
                    "Seizures uncommon in PEOA1 (<8%) but if present: LEV preferred. "
                    "Mechanism: SV2A modulator; renal elimination 70% (unchanged); no CYP450 "
                    "interactions; no CoA sequestration; no hepatotoxicity; no mitochondrial "
                    "toxicity demonstrated. Safe in all mtDNA deletion diseases. "
                    "VPA ABSOLUTE CI; avoid hepatically-metabolised AEDs (PHT, CBZ) if possible."
                ),
            },
            {
                "term": "KD — Contraindicated in All mtDNA Deletion/Depletion Diseases",
                "definition": (
                    "Ketogenic diet forces >70% energy from fat via beta-oxidation, which is "
                    "OXPHOS-dependent (requires intact Complex I–IV). COX-negative fibres in "
                    "PEOA1 cannot sustain this metabolic burden → lactic acidosis + rhabdomyolysis. "
                    "Impaired ANT1 ADP/ATP exchange also limits beta-oxidation capacity. "
                    "Contraindicated in all mtDNA instability diseases: PEOA1, MDDS2, POLG, TWNK, "
                    "DNA2, POLG2, RNASEH1, TK2, etc."
                ),
            },
        ],
    }
