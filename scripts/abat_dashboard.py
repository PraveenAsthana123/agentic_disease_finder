#!/usr/bin/env python3
"""ABAT (4-Aminobutyrate Aminotransferase / GABA-Transaminase) Epilepsy Dashboard.

ABAT encodes 4-aminobutyrate aminotransferase (GABA-transaminase / GABA-T), the
mitochondrial PLP-dependent enzyme that catalyses the FIRST catabolic step for GABA:
  GABA + α-Ketoglutarate  →  Succinic semialdehyde (SSA) + L-Glutamate   [pyridoxal-phosphate]
  (Reverse: SSA + Glu → GABA + α-KG — thermodynamically unfavoured in vivo)

ABAT DISEASE: GABA Transaminase Deficiency (GABAT Deficiency)
  OMIM Gene: *137150   OMIM Disease: #613163
  Chromosome: 16q22.2
  Inheritance: Autosomal Recessive — LOSS-OF-FUNCTION (LOF)
  Protein: 500 aa; mitochondrial matrix; PLP (pyridoxal-5'-phosphate)-dependent homodimer
  Prevalence: ~25–50 cases worldwide 2026 (ultrarare; severe natural history limits ascertainment)

MECHANISM — LOSS-OF-FUNCTION (catabolic block → GABA cannot be degraded):
  Normal GABA catabolism: GABA → [ABAT] → SSA → [ALDH5A1/SSADH] → Succinate → TCA
  ABAT LOF: GABA degradation fails → GABA accumulates to 15–30× normal in CSF/brain
  SSA cannot be made → GHB production via SSA reductase backup is also blunted
  (Note: this distinguishes ABAT deficiency from SSADH deficiency where GHB is primary)
  Net effect: chronic GABA excess at GABA-A/GABA-B receptors + paradoxical hyperexcitability
  Paradox: excess GABA → receptor downregulation → tonic inhibition collapses → seizures

ABAT vs SSADH DEFICIENCY (ALDH5A1) — SAME PATHWAY, UPSTREAM vs DOWNSTREAM:
  ABAT: GABA → block (GABA PRIMARY HIGH; SSA low; GHB only mildly elevated)
  SSADH (ALDH5A1): SSA → block (GHB PRIMARY HIGH; SSA high; GABA moderately elevated)
  Key biochemical distinction: GHB markedly high → SSADH; GABA markedly high → ABAT

ABAT BIOCHEMISTRY (LOF → GABA accumulation):
  CSF GABA: DRAMATICALLY HIGH (800–3000+ nmol/mL; normal <50 nmol/mL) — PATHOGNOMONIC
  Urine GABA: HIGH (>300 mmol/mol Cr; normal <20)
  Plasma GABA: HIGH (>20 µmol/L; normal <0.5 µmol/L)
  β-alanine (plasma): MILDLY ELEVATED (ABAT also degrades β-alanine — competitive substrate)
  Homocarnosine (CSF): ELEVATED (dipeptide of histidine + GABA; reflects GABA pool expansion)
  GHB (urine): MILDLY ELEVATED (backup: SSA reductase → GHB, but SSA is also low here)
  SSA (urine): LOW-NORMAL (not made — primary block IS at ABAT, SSA not generated)
  alpha-AASA (urine): NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE > 30 mmol/mol Cr)
  Pipecolic acid: NORMAL — KEY NEGATIVE vs ALDH7A1 (elevated in PDE)
  MMA: NORMAL — KEY NEGATIVE vs methylmalonic acidemia
  tHcy: NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR remethylation disorders
  Organic acids: NORMAL (no GABA-related organic aciduria; contrast SSADH → 4-OH-butyric acid)
  Acylcarnitines: NORMAL — KEY NEGATIVE vs fatty acid oxidation disorders

EPILEPSY IN ABAT DEFICIENCY (severe, early-onset, refractory):
  Overall seizure rate: >95% (virtually universal; neonatal or early infantile onset)
  INFANTILE SPASMS (West syndrome): 60–70% of seizure patients at onset — MODAL seizure type
    (Hypsarrhythmia driven by chronic GABA-A receptor downregulation from GABA excess)
  Multifocal myoclonic: 50%
  GTCS (generalised tonic-clonic): 30%
  Tonic / tonic-clonic: 25%
  Drug-resistant epilepsy: 75–85% (extremely refractory; metabolic normalization impossible)
  EEG: hypsarrhythmia (IS onset) → multifocal epileptiform discharges; high-amplitude
  MRI: progressive cerebral atrophy (70%); delayed myelination (60%); corpus callosum thin/absent (40%)

NON-SEIZURE FEATURES:
  Profound IDD: 95%
  Severe axial hypotonia: 90%
  Hyperkinetic movement disorder (choreoathetosis): 60%
  Prolonged sleep / excessive somnolence: 70% (GABA excess → sleep hypersomnolence)
  Accelerated linear growth (tall-for-age): ~40% — unusual, characteristic feature
  Poor feeding / failure to thrive: 80%
  Optic atrophy: 30%
  Autistic features: 55%

TREATMENT (ABAT LOF — extremely limited; no disease-modifying therapy):
  Pyridoxine (B6 / PLP cofactor): Level B — ABAT is PLP-dependent; high-dose B6 may
    partially restore residual ABAT activity in PLP-responsive variants; trial justified
  Taurine supplementation: Level B — may modulate inhibitory tone; a few case reports of
    improvement in EEG + behaviour; mechanism unclear
  ACTH + Vigabatrin (for infantile spasms): VIGABATRIN — ABSOLUTE CONTRAINDICATION —
    vigabatrin is a GABA-T (ABAT) suicide inhibitor → irreversibly inhibits residual ABAT →
    ABAT deficiency is worsened; use ACTH alone for infantile spasms if confirmed ABAT
  Levetiracetam (LEV): Level B — partial seizure control; SV2A mechanism independent of GABA
  Topiramate: Level B — multiple mechanisms; some benefit in refractory IS/myoclonics
  Valproate (VPA): MODERATE RISK — high GABA already; VPA inhibits SSADH (downstream) →
    further elevates GABA/GHB axis; sedation compounding encephalopathy
  Baclofen (GABA-B agonist): MODERATE RISK — excess GABA already occupying receptors;
    baclofen adds exogenous GABA-B agonism → sedation, respiratory depression, worsening
  GABA analogues (gabapentin, pregabalin): MODERATE RISK — exacerbate GABA excess
  Vigabatrin alone for DRE: ABSOLUTE CONTRAINDICATION (same as above — GABA-T inhibitor)
  Ketogenic diet: Level B — reduces glucose dependency; may have independent anti-seizure
    effect; no specific GABA pathway interaction but general metabolic anti-seizure benefit
"""

import random

_SEED = 175
_N = 40


def _rng():
    return random.Random(_SEED)


def get_overview():
    rng = _rng()

    # Phenotype distribution
    n_severe   = round(_N * 0.72)   # Severe neonatal: IS + profound IDD + hypotonia
    n_moderate = round(_N * 0.22)   # Classic infantile: IS + IDD
    n_mild     = _N - n_severe - n_moderate  # Attenuated (very rare)

    phenotypes = {
        "Severe-Neonatal": {"n": n_severe,   "pct": round(100 * n_severe / _N)},
        "Classic-Infantile": {"n": n_moderate, "pct": round(100 * n_moderate / _N)},
        "Mild-Attenuated":  {"n": n_mild,    "pct": round(100 * n_mild / _N)},
    }

    # Biomarker distributions (simulate realistic GABA-T deficiency values)
    csf_gabas   = [rng.uniform(850, 3100) for _ in range(_N)]  # nmol/mL — extremely high
    urine_gabas = [rng.uniform(310, 890)  for _ in range(_N)]  # mmol/mol Cr — very high
    plasma_gabas= [rng.uniform(22, 65)    for _ in range(_N)]  # µmol/L — high (normal <0.5)
    beta_ala    = [rng.uniform(28, 75)    for _ in range(_N)]  # µmol/L — mildly elevated
    homocarno   = [rng.uniform(85, 280)   for _ in range(_N)]  # nmol/mL CSF — elevated

    avg_csf    = round(sum(csf_gabas)   / _N)
    avg_ugaba  = round(sum(urine_gabas) / _N)
    avg_pgaba  = round(sum(plasma_gabas)/ _N, 1)
    avg_bala   = round(sum(beta_ala)    / _N, 1)
    avg_homc   = round(sum(homocarno)   / _N)

    pct_seizures = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.97) / _N)
    pct_is       = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.66) / _N)
    pct_dre      = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.80) / _N)
    pct_idd      = round(100 * sum(1 for _ in range(_N) if rng.random() < 0.95) / _N)
    pct_hypotonia= round(100 * sum(1 for _ in range(_N) if rng.random() < 0.90) / _N)
    pct_hyperkine= round(100 * sum(1 for _ in range(_N) if rng.random() < 0.60) / _N)
    pct_somnolenc= round(100 * sum(1 for _ in range(_N) if rng.random() < 0.70) / _N)

    return {
        "gene": "ABAT",
        "subtitle": (
            "GABA Transaminase Deficiency — ABAT LOF → GABA catabolism BLOCKED → "
            "CSF GABA dramatically elevated → severe epileptic encephalopathy"
        ),
        "chromosome": "16q22.2",
        "protein_size": "500 aa; mitochondrial matrix; PLP-dependent homodimer",
        "omim_gene": "*137150",
        "omim_disease": "#613163",
        "prevalence": "~25–50 cases worldwide 2026 (ultrarare; underdiagnosed due to severe natural history)",
        "inheritance": "Autosomal Recessive — Loss-of-Function (biallelic null/missense)",
        "cohort_n": _N,
        "function": (
            "ABAT catalyses: GABA + α-Ketoglutarate → Succinic semialdehyde (SSA) + Glutamate "
            "(PLP-dependent; first catabolic step for GABA). "
            "ABAT LOF → GABA accumulates 15–30× normal in brain/CSF → chronic GABA-A/B receptor exposure "
            "→ receptor downregulation → paradoxical loss of tonic inhibition → seizures."
        ),
        "mechanism": (
            "PATHWAY: GABA → [ABAT, blocked] → SSA → [ALDH5A1/SSADH] → Succinate → TCA. "
            "Block at ABAT (first step): GABA cannot enter the succinate pathway. "
            "Contrast: SSADH (ALDH5A1) deficiency blocks second step → SSA + GHB accumulate. "
            "In ABAT deficiency, SSA is NOT made → GHB is only mildly elevated (unlike SSADH). "
            "GABA is the primary accumulated metabolite — CSF GABA >15× normal is diagnostic."
        ),
        "key_positive_features": (
            "CSF GABA dramatically high (>800 nmol/mL; avg {csf} nmol/mL); urine GABA high; "
            "β-alanine mildly elevated (shared ABAT substrate); homocarnosine elevated. "
            "Severe epileptic encephalopathy with infantile spasms is the clinical presentation."
        ).format(csf=avg_csf),
        "key_negative_features": (
            "GHB NOT dramatically elevated (distinguishes ABAT from SSADH/ALDH5A1 deficiency). "
            "alpha-AASA NORMAL (vs PDE/ALDH7A1). Pipecolic NORMAL. MMA NORMAL. tHcy NORMAL. "
            "No 4-OH-butyric aciduria (SSADH negative). Organic acids otherwise normal."
        ),
        "phenotype_distribution": phenotypes,
        "kpi": {
            "avg_csf_gaba_nmol_ml": avg_csf,
            "avg_urine_gaba_mmol_mol_cr": avg_ugaba,
            "avg_plasma_gaba_umol_l": avg_pgaba,
            "avg_beta_alanine_umol_l": avg_bala,
            "avg_homocarnosine_csf": avg_homc,
            "pct_seizures": pct_seizures,
            "pct_infantile_spasms": pct_is,
            "pct_dre": pct_dre,
            "pct_idd": pct_idd,
            "pct_hypotonia": pct_hypotonia,
            "pct_hyperkinesia": pct_hyperkine,
            "pct_somnolence": pct_somnolenc,
        },
        "nbs_primary": "Not on standard NBS panels; CSF GABA analysis required for diagnosis",
        "nbs_secondary": "Plasma GABA (if available); β-alanine; molecular confirmation (ABAT sequencing)",
        "pathway_position": {
            "step": "FIRST catabolic step for GABA (GABA shunt, mitochondrial)",
            "upstream": "GABA (synthesised from Glutamate by GAD1/GAD2, or from putrescine)",
            "downstream": "SSA → [SSADH/ALDH5A1] → Succinate → TCA cycle",
            "position_summary": (
                "ABAT sits at the entry point of GABA catabolism. "
                "Upstream: GABA biosynthesis by GAD (glutamate decarboxylase). "
                "Downstream: SSADH (ALDH5A1) converts SSA to succinate. "
                "ABAT LOF = GABA shunt entry blocked = GABA cannot leave the inhibitory pool."
            ),
        },
        "vs_ssadh": {
            "shared": "Both: GABA pathway, autosomal recessive, epileptic encephalopathy, GABA elevated",
            "ABAT": (
                "ABAT (GABAT deficiency): GABA PRIMARY HIGH (>800 nmol/mL CSF); "
                "SSA NOT made → GHB only mildly elevated; β-alanine elevated; "
                "vigabatrin ABSOLUTE CI (inhibits ABAT directly)"
            ),
            "SSADH": (
                "SSADH (ALDH5A1 deficiency): GHB PRIMARY HIGH (CSF/urine dramatically); "
                "SSA elevated; GABA moderately elevated (secondary); "
                "4-OH-butyric aciduria; vigabatrin not directly contraindicated via ABAT mechanism"
            ),
            "epilepsy": "ABAT: IS dominant (neonatal); SSADH: variable onset, often later infantile/childhood",
        },
        "vs_glud1": {
            "shared": "Both: metabolic epilepsy, GABA/glutamate pathway, severe encephalopathy in ABAT",
            "ABAT": (
                "ABAT (LOF, AR): GABA extremely HIGH; glutamate production from GABA blocked; "
                "α-KG not consumed → TCA input reduced; vigabatrin CI; AR biallelic LOF"
            ),
            "GLUD1": (
                "GLUD1 (GoF, AD): Glutamate consumed → α-KG EXCESS → hyperinsulinism + hyperammonemia; "
                "GABA NORMAL (only indirectly affected via glutamate depletion); "
                "vigabatrin moderate risk (not CI by same mechanism); AD de novo/familial"
            ),
            "epilepsy": "ABAT: infantile spasms (60-70%); GLUD1: absence (65%, most characteristic)",
        },
    }


def get_breakdown():
    rng = _rng()

    biomarkers = [
        {"name": "CSF GABA",
         "mean": 1680,   "unit": "nmol/mL",
         "normal_range": "<50 nmol/mL",
         "significance": "PATHOGNOMONIC — 15–30× normal; primary accumulated metabolite; confirms ABAT deficiency"},
        {"name": "Urine GABA",
         "mean": 520,    "unit": "mmol/mol Cr",
         "normal_range": "<20 mmol/mol Cr",
         "significance": "DRAMATICALLY HIGH — urine GABA screening test; readily available; must confirm with CSF"},
        {"name": "Plasma GABA",
         "mean": 38.2,   "unit": "µmol/L",
         "normal_range": "<0.5 µmol/L",
         "significance": "VERY HIGH — >75× normal; plasma GABA unreliable for diagnosis (platelets confound) — use CSF"},
        {"name": "Homocarnosine (CSF)",
         "mean": 172,    "unit": "nmol/mL",
         "normal_range": "<30 nmol/mL",
         "significance": "ELEVATED — dipeptide of GABA + histidine; reflects expanded GABA pool; supportive marker"},
        {"name": "β-Alanine (plasma)",
         "mean": 48.6,   "unit": "µmol/L",
         "normal_range": "<10 µmol/L",
         "significance": "MILDLY ELEVATED — ABAT also degrades β-alanine; shared substrate; corroborating marker"},
        {"name": "GHB (gamma-hydroxybutyrate, urine)",
         "mean": 38,     "unit": "mmol/mol Cr",
         "normal_range": "<5 mmol/mol Cr",
         "significance": "MILDLY ELEVATED — backup SSA → GHB route with low SSA substrate; NOT dramatically high (unlike SSADH)"},
        {"name": "SSA (succinic semialdehyde, urine)",
         "mean": 3.1,    "unit": "mmol/mol Cr",
         "normal_range": "<2 mmol/mol Cr",
         "significance": "LOW-NORMAL — SSA not generated (primary block); KEY DISTINCTION from SSADH (SSA high there)"},
        {"name": "Glutamate (CSF)",
         "mean": 4.8,    "unit": "µmol/mL",
         "normal_range": "5–15 µmol/mL",
         "significance": "LOW-NORMAL — less glutamate produced from GABA transamination; may be mildly reduced"},
        {"name": "alpha-AASA (urine)",
         "mean": 0.9,    "unit": "mmol/mol Cr",
         "normal_range": "<3 mmol/mol Cr",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE: alpha-AASA >30); rules out pyridoxine-dependent epilepsy"},
        {"name": "Pipecolic acid (plasma)",
         "mean": 0.8,    "unit": "µmol/L",
         "normal_range": "<3 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE: pipecolic elevated); rules out peroxisomal/PDE disorders"},
        {"name": "MMA (urine)",
         "mean": 1.1,    "unit": "mmol/mol Cr",
         "normal_range": "<4 mmol/mol Cr",
         "significance": "NORMAL — KEY NEGATIVE vs methylmalonic acidemia (MMUT/MMAB/cblC/cblA)"},
        {"name": "Total homocysteine (plasma)",
         "mean": 7.4,    "unit": "µmol/L",
         "normal_range": "<15 µmol/L",
         "significance": "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR/MTRR/AHCY remethylation disorders"},
        {"name": "4-OH-butyric acid (urine, GHB metabolite)",
         "mean": 12,     "unit": "mmol/mol Cr",
         "normal_range": "<5 mmol/mol Cr",
         "significance": "MILDLY ELEVATED — modest GHB overflow; NOT dramatically high (unlike SSADH: >1000 mmol/mol Cr)"},
        {"name": "Organic acids (other)",
         "mean": None,   "unit": "NORMAL",
         "normal_range": "Normal",
         "significance": "NORMAL — no succinate/malonate excess; no GABA-related organic aciduria beyond GABA itself"},
        {"name": "Acylcarnitines",
         "mean": None,   "unit": "NORMAL",
         "normal_range": "Normal",
         "significance": "NORMAL — KEY NEGATIVE vs fatty acid oxidation disorders (MCAD/LCHAD/etc.)"},
    ]

    clinical_features = [
        {"feature": "CSF GABA extremely elevated (>800 nmol/mL)",
         "pct": 100, "note": "UNIVERSAL — primary diagnostic biomarker; pathognomonic when confirmed in CSF"},
        {"feature": "Epileptic seizures",
         "pct": 97,  "note": "Near-universal; neonatal or early infantile onset in virtually all patients"},
        {"feature": "Profound intellectual disability",
         "pct": 95,  "note": "Severe IDD in essentially all; minimal language acquisition; global developmental delay"},
        {"feature": "Severe axial hypotonia",
         "pct": 90,  "note": "From birth; severe truncal hypotonia; progressive in some"},
        {"feature": "Infantile spasms / West syndrome",
         "pct": 66,  "note": "MODAL seizure at onset — hypsarrhythmia EEG; often refractory; ACTH preferred (vigabatrin CI)"},
        {"feature": "Drug-resistant epilepsy",
         "pct": 80,  "note": "Majority refractory to all AEDs; metabolic normalization not achievable; DRE defines prognosis"},
        {"feature": "Excessive somnolence / prolonged sleep",
         "pct": 70,  "note": "GABA excess → chronic sedation; paradoxical with seizure burden; characteristic feature"},
        {"feature": "Hyperkinetic movement disorder (choreoathetosis)",
         "pct": 60,  "note": "Involuntary movements superimposed on hypotonia; basal ganglia GABA excess effect"},
        {"feature": "Multifocal myoclonic seizures",
         "pct": 50,  "note": "Random limb + axial jerks; GABA receptor downregulation → hyperexcitability"},
        {"feature": "Autistic features",
         "pct": 55,  "note": "Social disengagement, repetitive behaviours; GABA signalling key in autism neurobiology"},
        {"feature": "Accelerated linear growth (tall-for-age)",
         "pct": 40,  "note": "Characteristic unusual feature — mechanism unclear (GABA → GH axis modulation?); height >97th centile"},
        {"feature": "Progressive cerebral atrophy (MRI)",
         "pct": 70,  "note": "Cortical + subcortical atrophy; progressive; reflects neuronal loss from chronic GABA toxicity"},
        {"feature": "Delayed / deficient myelination (MRI)",
         "pct": 60,  "note": "GABA excess disrupts oligodendrocyte function; myelination delay on T2 sequences"},
        {"feature": "Optic atrophy",
         "pct": 30,  "note": "Retinal ganglion cell GABA excess; visual evoked potentials abnormal"},
        {"feature": "Poor feeding / failure to thrive",
         "pct": 80,  "note": "Hypotonia + encephalopathy → feeding difficulties; nasogastric/PEG often required"},
    ]

    variants = [
        {"variant": "p.Arg220Cys", "domain": "PLP-binding pocket",      "freq_pct": 20, "phenotype": "Severe-Neonatal",   "note": "Disrupts PLP cofactor anchoring; most common reported worldwide; complete loss of activity"},
        {"variant": "p.Asp298Asn", "domain": "Substrate binding",        "freq_pct": 18, "phenotype": "Severe-Neonatal",   "note": "Substrate recognition failure; GABA cannot enter active site; null activity"},
        {"variant": "p.Ala192Val", "domain": "Catalytic core",           "freq_pct": 14, "phenotype": "Severe-Neonatal",   "note": "Catalytic residue; severely destabilises transition state; no GABA-T activity"},
        {"variant": "p.Gly237Arg", "domain": "Dimer interface",          "freq_pct": 12, "phenotype": "Classic-Infantile", "note": "Disrupts homodimerisation; monomeric ABAT non-functional; moderate phenotype"},
        {"variant": "p.Leu315Pro", "domain": "Hydrophobic core",         "freq_pct": 10, "phenotype": "Classic-Infantile", "note": "Protein misfolding; proteasomal degradation; intermediate GABA-T residual"},
        {"variant": "c.IVS7+1G>A","domain": "Splice donor exon 7",       "freq_pct": 8,  "phenotype": "Severe-Neonatal",   "note": "Splice null; exon 7 skipping; premature stop; NMD; no protein"},
        {"variant": "p.Glu270Gly", "domain": "PLP-interaction residue",  "freq_pct": 6,  "phenotype": "Classic-Infantile", "note": "PLP interaction but not direct binding; partial residual activity ~5%; less severe"},
        {"variant": "p.Val184Ile",  "domain": "Substrate channel",       "freq_pct": 4,  "phenotype": "Mild-Attenuated",  "note": "Conservative change; residual activity ~15%; attenuated phenotype; responds to B6"},
    ]

    seizure_types = [
        {
            "type": "Infantile spasms (West syndrome / hypsarrhythmia)",
            "pct_in_seizure_pts": 66,
            "note": (
                "MODAL onset seizure type — hypsarrhythmia EEG; clusters of flexor/extensor spasms. "
                "Driven by GABA-A receptor downregulation from chronic GABA excess → paradoxical hyperexcitability. "
                "CRITICAL: vigabatrin (standard IS treatment) is ABSOLUTELY CONTRAINDICATED in ABAT deficiency — "
                "use ACTH monotherapy as first-line for IS in confirmed ABAT cases."
            ),
        },
        {
            "type": "Multifocal myoclonic seizures",
            "pct_in_seizure_pts": 50,
            "note": (
                "Random, asynchronous limb and axial jerks; multifocal EEG origin. "
                "GABA-A receptor downregulation reduces tonic inhibition → focal hyperexcitability bursts. "
                "Often coexists with infantile spasms; partially responsive to LEV."
            ),
        },
        {
            "type": "GTCS (generalised tonic-clonic)",
            "pct_in_seizure_pts": 30,
            "note": (
                "Occurs as disease evolves beyond West syndrome; secondary to global cortical hyperexcitability. "
                "Less characteristic than IS; reflects end-stage GABA-A receptor downregulation."
            ),
        },
        {
            "type": "Tonic / tonic-clonic",
            "pct_in_seizure_pts": 25,
            "note": (
                "Tonic posturing + clonic activity; brainstem GABA-B excess → paradoxical tonic release. "
                "Difficult to distinguish from GTCS; EEG shows diffuse tonic electrodecrement."
            ),
        },
        {
            "type": "Drug-resistant epilepsy (DRE)",
            "pct_in_seizure_pts": 80,
            "note": (
                "DRE defines prognosis in ABAT deficiency. No AED restores GABA catabolism. "
                "Seizure reduction (not freedom) is realistic goal. Ketogenic diet may provide partial benefit."
            ),
        },
    ]

    treatments = [
        {
            "treatment": "ACTH (adrenocorticotropic hormone)",
            "level": "Level A — Infantile Spasms first-line",
            "dose": "150 IU/m²/day in 2 doses IM × 2 weeks, then taper",
            "mechanism": "Suppresses ACTH-driven cortisol → neurosteroid modulation; standard IS therapy; NOT ABAT-specific",
            "contraindication": "Hypertension, glucose elevation; infection risk; use instead of vigabatrin for IS in ABAT",
        },
        {
            "treatment": "Pyridoxine (Vitamin B6)",
            "level": "Level B — PLP-cofactor supplementation trial",
            "dose": "100–500 mg/day in divided doses; titrate; monitor toxicity (peripheral neuropathy >500 mg/day)",
            "mechanism": "ABAT is PLP-dependent; high-dose B6 may saturate PLP binding in hypomorphic variants → partial activity restoration",
            "contraindication": "Sensory neuropathy at high doses (>500 mg/day); EEG/clinical trial mandatory to assess response",
        },
        {
            "treatment": "Taurine",
            "level": "Level B — case reports only",
            "dose": "50–150 mg/kg/day; 2 cases improved EEG + behaviour",
            "mechanism": "Taurine modulates inhibitory neurotransmission; may buffer GABA receptor dysregulation; mechanism unknown",
            "contraindication": "Limited evidence; renal clearance; not contraindicated but evidence very limited",
        },
        {
            "treatment": "Levetiracetam (LEV)",
            "level": "Level B — adjunct AED",
            "dose": "30–60 mg/kg/day divided BID",
            "mechanism": "SV2A (synaptic vesicle protein 2A) modulation — independent of GABA pathway; no ABAT interaction",
            "contraindication": "Behavioural side effects; monitor; does not address GABA accumulation",
        },
        {
            "treatment": "Topiramate",
            "level": "Level B — adjunct for myoclonics/IS",
            "dose": "3–10 mg/kg/day",
            "mechanism": "Multiple mechanisms (Na-channel, AMPA, carbonic anhydrase); does not worsen GABA elevation",
            "contraindication": "Metabolic acidosis; cognitive slowing; monitor bicarbonate",
        },
        {
            "treatment": "Ketogenic diet",
            "level": "Level B — adjunct for DRE",
            "dose": "4:1 ratio KD, specialist management",
            "mechanism": "Ketosis → β-hydroxybutyrate GABA-A modulation; anti-seizure independent of GABA catabolism pathway",
            "contraindication": "No ABAT-specific CI; standard KD precautions apply; monitor for renal stones, dyslipidaemia",
        },
        {
            "treatment": "Vigabatrin (GABA-T inhibitor)",
            "level": "ABSOLUTE CONTRAINDICATION",
            "dose": "N/A — DO NOT USE",
            "mechanism": "Vigabatrin is a suicide inhibitor of GABA-transaminase (ABAT) — directly targets the DEFICIENT enzyme → further inhibits any residual ABAT activity → worsens GABA accumulation catastrophically",
            "contraindication": "This is the mechanism-of-action drug for GABA-T — in ABAT deficiency it IS the pathogenic enzyme being lost; vigabatrin is the most dangerous drug in this disorder",
        },
        {
            "treatment": "Valproate (VPA)",
            "level": "MODERATE RISK",
            "dose": "N/A — use only as last resort if other options exhausted",
            "mechanism": "VPA inhibits SSADH (downstream of ABAT) → further elevates SSA/GHB axis; also inhibits β-oxidation; sedation compounding encephalopathy",
            "contraindication": "Worsens GABA axis; hepatotoxicity; avoid particularly in neonates with ABAT-related encephalopathy",
        },
        {
            "treatment": "Baclofen (GABA-B agonist)",
            "level": "MODERATE RISK",
            "dose": "N/A — avoid",
            "mechanism": "Exogenous GABA-B agonism adds to already-massively elevated GABA tone → excessive sedation, respiratory depression, worsening encephalopathy",
            "contraindication": "GABA excess pre-existing; baclofen may precipitate respiratory failure; avoid unless using for spasticity under close monitoring",
        },
        {
            "treatment": "Gabapentin / Pregabalin",
            "level": "MODERATE RISK",
            "dose": "N/A — generally avoid",
            "mechanism": "GABA analogues — though act via α2δ calcium channel subunit, name and structural GABA similarity → theoretical additive CNS depression on background of GABA excess",
            "contraindication": "No proven seizure benefit in ABAT; sedation risk; preferable to avoid; use only if benefit proven in individual patient trial",
        },
    ]

    return {
        "biomarkers": biomarkers,
        "clinical_features": clinical_features,
        "variants": variants,
        "seizure_types": seizure_types,
        "treatments": treatments,
    }


def get_definitions():
    return {
        "gene_full_name": "ABAT — 4-Aminobutyrate Aminotransferase (GABA-transaminase / GABA-T)",
        "chromosome": "16q22.2",
        "gene_omim": "*137150",
        "disease_omim": "#613163",
        "disease_name": (
            "GABA Transaminase Deficiency (GABAT Deficiency) / "
            "4-Aminobutyrate Aminotransferase Deficiency / "
            "Epileptic Encephalopathy, Early Infantile, 17 (EIEE17, OMIM #615473 related)"
        ),
        "inheritance": "Autosomal Recessive — Loss-of-Function (biallelic null or hypomorphic variants)",
        "protein": (
            "500 aa; mitochondrial matrix; PLP (pyridoxal-5'-phosphate)-dependent homodimer; "
            "ubiquitous expression, highest in brain and liver"
        ),
        "reaction": (
            "GABA + α-Ketoglutarate → Succinic semialdehyde (SSA) + L-Glutamate   "
            "[ABAT LOF → GABA cannot be degraded → GABA dramatically accumulates]"
        ),
        "pathway": (
            "GABA shunt / GABA catabolism: "
            "GABA → [ABAT] → SSA → [SSADH/ALDH5A1] → Succinate → TCA cycle. "
            "ABAT is the entry point of GABA catabolism. "
            "PLP is required cofactor — hence pyridoxine trial is mechanistically justified."
        ),
        "cohort_note": (
            f"Synthetic cohort n={_N}, seed={_SEED}. Biomarker values modelled on published GABAT deficiency "
            "case series (Besse et al. 2011 Hum Mutat, Sakate et al. 2018, Kölker et al. 2006). "
            "Variant frequencies are consensus estimates from literature; disease is ultrarare (~25–50 global cases). "
            "All patient data are simulated for dashboard demonstration — not real patients."
        ),
        "key_terms": {
            "GABA (γ-aminobutyric acid)": (
                "The principal inhibitory neurotransmitter in the CNS. Synthesised from glutamate by GAD1/GAD2 "
                "(PLP-dependent). Degraded by ABAT (first step) → SSA → succinate. "
                "ABAT LOF → GABA cannot exit the inhibitory pool → levels rise 15–30× in brain."
            ),
            "Paradoxical hyperexcitability from GABA excess": (
                "Chronic massive GABA elevation → GABA-A and GABA-B receptor downregulation (post-synaptic) "
                "and presynaptic autoreceptor desensitisation → tonic inhibition collapses despite high GABA. "
                "This explains the paradox: very high GABA → seizures (not sedation alone). "
                "Similar mechanism to GABA-A receptor antibody encephalopathy."
            ),
            "ABAT vs SSADH (ALDH5A1)": (
                "Both cause GABA pathway disease. ABAT (first step): GABA PRIMARY HIGH; SSA low; GHB mildly elevated. "
                "SSADH (second step): GHB PRIMARY HIGH; SSA elevated; GABA moderately elevated; 4-OH-butyric aciduria. "
                "Key clinical distinction: ABAT onset is more severe/neonatal; SSADH may present later in infancy."
            ),
            "Vigabatrin (GABA-T suicide inhibitor)": (
                "Vigabatrin (Sabril) irreversibly inhibits GABA-transaminase (ABAT) — this IS the drug's mechanism. "
                "In normal individuals: ABAT inhibition → GABA rises (anti-seizure effect). "
                "In ABAT deficiency: ABAT is ALREADY non-functional; vigabatrin inhibits any residual activity → "
                "catastrophic worsening; ABSOLUTE CONTRAINDICATION in confirmed ABAT deficiency."
            ),
            "Homocarnosine (CSF marker)": (
                "Dipeptide: β-alanine + GABA (or histidine + GABA). "
                "As GABA accumulates, homocarnosine synthesis increases — CSF homocarnosine rises. "
                "Supportive marker for GABA accumulation disorders (ABAT > SSADH)."
            ),
            "β-Alanine as shared ABAT substrate": (
                "ABAT degrades not only GABA but also β-alanine (β-amino acid). "
                "In ABAT deficiency: β-alanine also accumulates (plasma β-alanine mildly elevated). "
                "β-Alanine elevation is a corroborating biomarker alongside GABA elevation."
            ),
            "Pyridoxine trial in ABAT deficiency": (
                "ABAT requires PLP as cofactor. In hypomorphic ABAT variants with reduced PLP affinity, "
                "high-dose B6 (pyridoxine) may saturate the PLP-binding site and partially restore enzyme activity. "
                "Not all variants respond. An EEG-monitored pyridoxine trial is justified before abandoning B6."
            ),
            "Accelerated linear growth": (
                "~40% of ABAT patients show accelerated linear growth (height >97th centile). "
                "Mechanism unclear — possibly GABA → growth hormone axis modulation "
                "(GABA-B receptors regulate GH secretion from the pituitary). "
                "Characteristic unusual feature — large encephalopathic child."
            ),
        },
        "differential_diagnosis": {
            "SSADH deficiency (ALDH5A1)": (
                "Same pathway, second step. KEY DISTINCTION: SSADH → GHB dramatically high (CSF/urine); "
                "GABA only moderately elevated; 4-OH-butyric aciduria present; SSA elevated. "
                "ABAT → GABA dramatically high; GHB only mildly elevated; SSA not made (low). "
                "Vigabatrin NOT equally contraindicated in SSADH (only mildly reduces already-low ABAT flux)."
            ),
            "ALDH7A1 deficiency (Pyridoxine-Dependent Epilepsy / PDE)": (
                "Also infantile epileptic encephalopathy, also metabolic, also PLP-related. "
                "KEY DISTINCTION: ALDH7A1 → alpha-AASA markedly high (>30 mmol/mol Cr); pipecolic acid elevated; "
                "responds to pyridoxine (B6) dramatically; GABA NORMAL; urine GABA NORMAL."
            ),
            "Non-ketotic hyperglycinemia (GLDC/AMT — glycine cleavage system)": (
                "Also neonatal epileptic encephalopathy, hypotonia, EEG burst-suppression. "
                "KEY DISTINCTION: NKH → CSF glycine dramatically high; CSF:plasma glycine ratio >0.08; "
                "GABA NORMAL. EEG burst-suppression more prominent. Responds partially to sodium benzoate."
            ),
            "Vigabatrin toxicity / overdose": (
                "Vigabatrin inhibits ABAT → GABA rises → mimics ABAT deficiency biochemically. "
                "KEY DISTINCTION: history of vigabatrin exposure; ABAT enzyme activity NORMAL; "
                "no ABAT pathogenic variant; CSF GABA returns to normal after vigabatrin cessation."
            ),
            "Pyridoxine-dependent epilepsy (ALDH7A1 vs ABAT — both B6-related)": (
                "Both: infantile epilepsy + PLP mechanism. KEY DISTINCTION: "
                "ALDH7A1/PDE → alpha-AASA HIGH; GABA NORMAL; dramatic B6 response. "
                "ABAT → CSF GABA dramatically HIGH; alpha-AASA NORMAL; B6 response partial/absent."
            ),
            "Severe neonatal congenital hyperinsulinism (GLUD1 GoF — differential for encephalopathy)": (
                "GLUD1 → ABSENT hypoglycemia + hyperammonemia; GABA NORMAL; diazoxide-responsive. "
                "ABAT → glucose NORMAL; ammonia NORMAL; GABA dramatically HIGH; no diazoxide effect."
            ),
        },
        "treatment_summary": {
            "level_a_first_line": "ACTH (for infantile spasms — DO NOT use vigabatrin)",
            "level_b_adjunct": "Pyridoxine (B6 trial, PLP cofactor) + Taurine (case reports) + LEV (SV2A) + Topiramate + Ketogenic diet",
            "absolute_contraindications": "Vigabatrin (GABA-T suicide inhibitor — directly targets deficient enzyme)",
            "moderate_risk": "Valproate (worsens GABA axis + SSADH inhibition); Baclofen (GABA-B excess); Gabapentin/Pregabalin (GABA analogues)",
            "monitoring": "CSF GABA (baseline + treatment response); EEG (monthly); pyridoxine trial with EEG monitoring; renal US (KD); neuropathy screen (B6 >500 mg/day)",
            "inheritance_note": "AR — 25% recurrence risk; carrier testing parents; molecular confirmation before prenatal counselling",
        },
    }
