"""
SLC25A22 Epilepsy — Mitochondrial Glutamate Carrier 1 / DEE3 / EIEE3 / 11p15.5
=================================================================================
40-patient cohort · SLC25A22 (11p15.5) · AR biallelic only

SLC25A22 BIOLOGY:
SLC25A22 (11p15.5) encodes Mitochondrial Glutamate Carrier 1 (GC1), a 323-amino-acid
electrogenic glutamate/H⁺ antiporter embedded in the mitochondrial inner membrane.
It is the principal route for cytoplasmic glutamate entry into the mitochondrial matrix,
feeding the TCA cycle (via transamination → α-ketoglutarate), the malate–aspartate
shuttle, and the GABA synthesis precursor pool. LOF → catastrophic neonatal epileptic
encephalopathy (Ohtahara syndrome) with a unique metabolic signature: elevated plasma
and CSF glutamate that is pathognomonic in the neonatal DEE context.

SLC25A22 — STRUCTURE (323 aa, mitochondrial inner membrane):
  MODULE 1 (aa 1-110): First ~100 aa repeat unit — TM1 (aa 7-26) and TM2 (aa 35-56);
    carries conserved P-x-[D/E]-x-x-[K/R] motif (MCF signature repeat 1);
    substrate-binding residue Arg71 (contacts glutamate α-carboxylate).
  MODULE 2 (aa 111-218): Second repeat unit — TM3 (aa 116-137) and TM4 (aa 147-167);
    Asp123 forms salt bridge in transport cycle; Leu200 is structural TM4 anchor.
  MODULE 3 (aa 219-323): Third repeat unit — TM5 (aa 222-243) and TM6 (aa 252-273);
    Gly236 (TM5 kink-forming glycine, loss destabilises glutamate channel); Thr276 (TM6
    expression anchor). All 6 TM helices form a barrel-like substrate translocation
    channel; 3 odd-numbered helices form the matrix-side gate, 3 even-numbered form
    the IMS-side gate — alternating-access mechanism.
  MATRIX LOOPS: ml1 (aa 57-115) and ml2 (aa 168-221) — matrix-facing; regulate access;
    Arg85 in ml1 (matrix-facing) directs glutamate into translocation barrel.
  IMS LOOPS: il1 (aa 27-34) and il2 (aa 138-146) — inter-membrane space side.

SLC25A22 MECHANISM — GLUTAMATE IMPORT / GABA PRECURSOR SUPPLY:
  Step 1: Cytoplasmic glutamate (20-120 µmol/L; EXCITATORY neurotransmitter precursor)
          binds IMS-side gate of GC1 barrel → conformational change to occluded state.
  Step 2: One H⁺ from matrix binds matrix-side → counter-transport (electrogenic: net
          negative charge moved inward driven by mitochondrial membrane potential ΔΨm
          −180 mV). Glutamate released into matrix.
  Step 3a: Matrix glutamate → GOT2 (aspartate aminotransferase 2) catalyses:
           Glutamate + OAA → α-KG + Aspartate → TCA cycle (α-KG enters at isocitrate
           step). Critical for malate–aspartate shuttle (neuronal bioenergetics).
  Step 3b: Matrix glutamate → GDH (glutamate dehydrogenase) → α-KG + NH₄⁺ → TCA.
  Step 3c: GABA synthesis: cytoplasmic glutamate → GAD1/GAD2 (requires pyridoxal
           phosphate B₆) → GABA. Reduced mitochondrial clearance of glutamate →
           cytoplasmic glutamate pool expands → NMDA/AMPA receptor overactivation
           BEFORE GAD1/2 can convert it (GABA synthesis not impaired directly, but
           the precursor oversupply combined with excitotoxic receptor activation
           drives the encephalopathy).
  SLC25A22 LOF CONSEQUENCE:
    • Cytoplasmic glutamate ACCUMULATES → plasma and CSF glutamate elevated
      (plasma >200 µmol/L; normal <100; measured as fasting plasma amino acids).
    • Sustained NMDAR + AMPAR overactivation → excitotoxic cascade → neonatal
      burst-suppression (Ohtahara syndrome).
    • Mitochondrial TCA cycle substrate deprivation → bioenergetic failure in
      neurons (high-energy-demand cells most vulnerable).
    • GABA synthesis impaired indirectly: cytoplasmic glutamate excess → receptor
      desensitisation; mitochondrial energy failure → Na⁺/K⁺-ATPase failure →
      chloride homeostasis disruption (NKCC1 immature, GABA depolarising neonatally).
    • Net: catastrophic neonatal seizure threshold collapse → Ohtahara burst-suppression.

PHENOTYPIC SPECTRUM:
  AR BIALLELIC NULL: Most severe. Neonatal onset (day 0-14). Ohtahara syndrome
    (burst-suppression EEG). Complete GC1 absence. Elevated plasma glutamate >200
    µmol/L. Profound ID (IQ <20 in survivors). ~35% year-1 mortality. MRI: bilateral
    symmetric signal changes in basal ganglia (globus pallidus and putamen) and
    thalamus — distinctive from PLCB1 (diffuse cortical atrophy). ACTH+VGB Level A.
    Pyridoxine trial MANDATORY before diagnosis locked (B6-responsive neonatal seizures
    is the key clinical mimic with similar EEG).
  AR BIALLELIC HYPOMORPHIC: Partial GC1 activity (~15-40% residual). West syndrome
    onset (3-9 months). Hypsarrhythmia but NOT burst-suppression. Moderate-severe ID.
    Plasma glutamate moderately elevated (120-180 µmol/L). Longer survival. ACTH+VGB
    responsive. KD beneficial (reduces glycolytic substrate → less cytoplasmic
    glutamate excitotoxicity via reduced glycolysis).
  MIGRATORY FOCAL SEIZURES (LOF compound het): A subset of compound heterozygous
    patients present with migrating partial seizures of infancy (MPSI-like) rather
    than pure Ohtahara. EEG shows multifocal migrating ictal pattern (cf. KCNT1/SCN1A
    MPSI), but with elevated glutamate distinguishing SLC25A22. Profound ID. Rare (~10%)
  PHENOCOPY (SLC25A22-negative): Clinical Ohtahara or West with elevated plasma
    glutamate where SLC25A22 sequencing is negative. May reflect GLUD1, GLS, or
    unknown glutamate metabolism genes. Broad metabolic + genetic panel mandatory.

DISTINGUISHING SLC25A22 FROM PLCB1 / STXBP1 / NKH (NEONATAL DEE DDx):
  SLC25A22 (AR): Elevated plasma glutamate >200 µmol/L (PATHOGNOMONIC);
               bilateral symmetric BG/thalamic MRI signal changes;
               pyridoxine trial 100 mg IV mandatory; CSF glutamate elevated.
  PLCB1 (AR): Plasma glutamate NORMAL; diffuse cortical atrophy MRI (not BG);
               IP3/DAG pathway; no B6 response; somatic mosaic subtype (FCD IIb).
  STXBP1 (AD de novo): Plasma amino acids NORMAL; vesicle fusion pathway;
               STXBP1 protein reduced on immunoblot; most common neonatal DEE gene;
               no metabolic signature.
  NKH/GLDC (AR): Elevated plasma GLYCINE (not glutamate); CSF:plasma glycine >0.08;
               EEG burst-suppression with characteristic hiccups; sodium benzoate Rx.
  B6-DEPENDENT SEIZURES (ALDH7A1): Elevated pipecolic acid; responds immediately
               to pyridoxine 100 mg IV (diagnostic); plasma α-AASA elevated.
  KCNQ2 (AD de novo): Tonic asymmetric neonatal seizures; NOT burst-suppression
               predominantly; normal plasma amino acids; CBZ/PHT HELPFUL (opposite
               of SLC25A22).

CONTRAINDICATED DRUGS:
  PHENYTOIN / CARBAMAZEPINE / OXCARBAZEPINE:
    Na-channel blockers worsen burst-suppression in SLC25A22 Ohtahara (deepens the
    suppression phase by reducing Na⁺ channel availability during already silent
    phase). ABSOLUTE CONTRAINDICATION in confirmed SLC25A22 neonatal DEE.
  VALPROATE (VPA): POLG screen MANDATORY before VPA. Fatal Alpers-Huttenlocher
    hepatic failure in POLG carriers. VPA inhibits mitochondrial β-oxidation,
    compounding the bioenergetic failure in SLC25A22 (mitochondrial carrier LOF).
    Particular concern: SLC25A22 patients have mitochondrial dysfunction as part
    of pathophysiology — VPA mitochondrial toxicity risk is heightened.
  VIGABATRIN (VGB): REMS programme mandatory (visual field restriction). Maximum
    16 weeks for infantile spasms; annual ophthalmology thereafter.
  LAMOTRIGINE: Avoid in West/LGS evolution — may worsen myoclonic component.

REFERENCES:
  Molinari F et al. (2005) Impaired mitochondrial glutamate transport in autosomal
    recessive neonatal myoclonic epilepsy. Am J Hum Genet 76:334-339. PMID 15592994.
  Molinari F et al. (2009) Mutations in mSLC25A22 in neonatal epilepsy with
    suppression-bursts. Ann Neurol 65:630-635. PMID 19489073.
  Lemattre C et al. (2019) SLC25A22 gain-of-function mutation as a cause of
    neonatal epileptic encephalopathy. Mol Genet Genomic Med 7:e887. PMID 31397989.
  Poduri A, Lowenstein D (2011) Epilepsy genetics — past, present, and future.
    Curr Opin Genet Dev 21:325-332 (context: mitochondrial carrier epilepsies).
  ILAE Gene Classification (2022): SLC25A22 — DEE3 / EIEE3 (OMIM 609304).
"""

import random

random.seed(507)

# ── ETIOLOGY CATALOG ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "SLC25A22-AR-Biallelic-Null",
        "n_target": 22,
        "description": (
            "AR biallelic null (homozygous nonsense/frameshift or compound heterozygous "
            "null alleles). Complete GC1 absence. Neonatal Ohtahara syndrome: "
            "burst-suppression day 0-14. Profound ID. ~35% year-1 mortality (respiratory "
            "failure during burst phase). Plasma glutamate >200 µmol/L. Bilateral "
            "symmetric BG/thalamic MRI signal changes. ACTH+VGB Level A."
        ),
        "typical_variant": "Arg71Cys (c.211C>T) homozygous / c.520del frameshift / Arg85Cys compound het",
        "inheritance": "AR biallelic null",
        "functional_deficit": "Complete GC1 absence; plasma glutamate >200 µmol/L; bilateral BG MRI changes",
    },
    {
        "category": "SLC25A22-AR-Biallelic-Hypomorphic",
        "n_target": 10,
        "description": (
            "AR biallelic with at least one hypomorphic missense allele (~15-40% residual "
            "GC1 activity). West syndrome onset 3-9 months (hypsarrhythmia, NOT "
            "burst-suppression typically). Moderate-severe ID. Plasma glutamate "
            "moderately elevated (120-180 µmol/L). Longer survival than null. "
            "ACTH+VGB responsive. KD beneficial."
        ),
        "typical_variant": "Gly236Trp / Thr276Ile (TM6, reduced expression, partial transport)",
        "inheritance": "AR biallelic hypomorphic",
        "functional_deficit": "Partial GC1 transport (15-40% residual); moderately elevated plasma glutamate",
    },
    {
        "category": "SLC25A22-Migratory-Focal",
        "n_target": 4,
        "description": (
            "Compound heterozygous LOF presenting as migrating partial seizures of "
            "infancy (MPSI-like). EEG shows multifocal migrating ictal pattern. "
            "Elevated plasma glutamate distinguishes from KCNT1/SCN1A MPSI. "
            "Profound ID. Rare presentation (~10% of SLC25A22 cohorts)."
        ),
        "typical_variant": "Leu200Pro + Arg71Cys compound het (different functional domains)",
        "inheritance": "AR compound heterozygous",
        "functional_deficit": "Partial/null GC1; migrating focal pattern (distinct from pure Ohtahara)",
    },
    {
        "category": "SLC25A22-Phenocopy",
        "n_target": 4,
        "description": (
            "Clinical Ohtahara or West syndrome with elevated plasma glutamate; "
            "SLC25A22 sequencing + MLPA negative. May reflect GLUD1 GOF, GLS LOF, "
            "or undiscovered glutamate metabolism genes. Broad metabolic + gene panel "
            "mandatory. Empirical pyridoxine + pyridoxal phosphate trial."
        ),
        "typical_variant": "No pathogenic SLC25A22 variant identified",
        "inheritance": "Unknown (phenocopy)",
        "functional_deficit": "Not established — alternative glutamate pathway",
    },
]

# ── PATIENT COHORT  (40 patients, seed 507) ──────────────────────────────────
def _build_cohort():
    rng = random.Random(507)
    pts = []
    pid = 1
    for ec in ETIOLOGY_CATALOG:
        n = ec["n_target"]
        for _ in range(n):
            cat = ec["category"]
            is_null = cat == "SLC25A22-AR-Biallelic-Null"
            is_hypo = cat == "SLC25A22-AR-Biallelic-Hypomorphic"
            is_migr = cat == "SLC25A22-Migratory-Focal"
            is_pheno = cat == "SLC25A22-Phenocopy"

            age_onset_days = (
                rng.randint(0, 14) if is_null else
                rng.randint(60, 270) if is_hypo else
                rng.randint(3, 90) if is_migr else
                rng.randint(0, 180)
            )

            burst_sup = rng.random() < (0.92 if is_null else 0.10 if is_hypo else 0.30 if is_migr else 0.45)
            ohtahara  = burst_sup and rng.random() < (0.95 if is_null else 0.30 if is_migr else 0.50)
            west_syn  = (not ohtahara) and rng.random() < (0.05 if is_null else 0.85 if is_hypo else 0.20 if is_migr else 0.35)
            hyps      = west_syn and rng.random() < 0.88
            migr_foc  = is_migr and rng.random() < 0.80
            eeg_abnl  = burst_sup or hyps or migr_foc or rng.random() < 0.70

            # Plasma glutamate
            pla_glut = (
                rng.randint(200, 380) if is_null else
                rng.randint(120, 195) if is_hypo else
                rng.randint(130, 220) if is_migr else
                rng.randint(60, 140)   # phenocopy may or may not be elevated
            )
            elev_glut = pla_glut > 150

            # MRI
            mri_done  = rng.random() < 0.94
            bg_change = mri_done and rng.random() < (0.78 if is_null else 0.40 if is_hypo else 0.50 if is_migr else 0.20)
            cort_atr  = mri_done and rng.random() < (0.55 if is_null else 0.25 if is_hypo else 0.30 if is_migr else 0.20)
            thal_chg  = bg_change and rng.random() < 0.72

            # ID
            profound_id = rng.random() < (0.88 if is_null else 0.50 if is_hypo else 0.75 if is_migr else 0.40)
            any_id      = profound_id or rng.random() < 0.90

            # Treatment
            b6_trial   = rng.random() < 0.92   # nearly universal mandatory
            b6_resp    = b6_trial and rng.random() < 0.04  # very rare response in true SLC25A22
            plp_trial  = b6_trial and not b6_resp and rng.random() < 0.60
            acth_vgb   = (ohtahara or west_syn) and rng.random() < 0.82
            kd_tried   = rng.random() < (0.55 if is_null else 0.68 if is_hypo else 0.50 if is_migr else 0.30)
            polg_test  = rng.random() < 0.88

            yr1_mort   = rng.random() < (0.35 if is_null else 0.06 if is_hypo else 0.18 if is_migr else 0.12)

            sex = rng.choice(["M", "F"])
            pts.append({
                "patient_id": f"SLC25A22-{pid:03d}",
                "sex": sex,
                "category": cat,
                "age_onset_days": age_onset_days,
                "ohtahara_syndrome": ohtahara,
                "west_syndrome": west_syn,
                "migratory_focal_seizures": migr_foc,
                "burst_suppression": burst_sup,
                "hypsarrhythmia": hyps,
                "eeg_abnormal": eeg_abnl,
                "plasma_glutamate_umol_L": pla_glut,
                "elevated_plasma_glutamate": elev_glut,
                "mri_done": mri_done,
                "bg_mri_changes": bg_change,
                "thalamic_changes": thal_chg,
                "cortical_atrophy": cort_atr,
                "profound_id": profound_id,
                "any_id": any_id,
                "b6_pyridoxine_trial": b6_trial,
                "b6_responsive": b6_resp,
                "plp_trial": plp_trial,
                "acth_vgb_given": acth_vgb,
                "kd_tried": kd_tried,
                "polg_tested": polg_test,
                "yr1_mortality": yr1_mort,
            })
            pid += 1
    return pts


PATIENTS = _build_cohort()

# ── TREATMENTS ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Pyridoxine (B₆) 100 mg IV — MANDATORY FIRST-LINE TRIAL",
        "level": "Level A (evidence) — mandatory in ALL neonatal DEE before diagnosis locked. "
                 "SLC25A22 rarely responds (<5%), but B6-responsive seizures (ALDH7A1, PNPO) "
                 "are treatable mimics. Positive response = immediate seizure cessation within "
                 "30 min; EEG response within 24 h. Must be given before any anticonvulsant.",
    },
    {
        "drug": "Pyridoxal Phosphate (PLP) 30 mg/kg/day oral — if B6 trial fails",
        "level": "Level B — PNPO deficiency (a close mimic) responds to PLP but NOT pyridoxine. "
                 "PLP trial after negative B6 response is mandatory before ruling out B6-metabolism "
                 "disorders. Duration: 3-5 days; assess EEG response.",
    },
    {
        "drug": "ACTH + Vigabatrin (UKISS protocol)",
        "level": "Level A — for infantile spasms / West syndrome evolution. Standard UKISS dose: "
                 "ACTH 150 IU/m²/day (2 weeks) + VGB 50 mg/kg/day (maintained). VGB REMS "
                 "mandatory (visual field restriction); ophthalmology review every 3 months.",
    },
    {
        "drug": "Ketogenic Diet (KD) — 4:1 or modified Atkins",
        "level": "Level B — reduces glycolytic substrate → less cytoplasmic glutamate from "
                 "glucose-driven excitatory neurotransmission. Shifts neuronal energy metabolism "
                 "to mitochondrial β-oxidation (less demand on GC1 for TCA substrate). "
                 "RD dietitian + metabolic team mandatory. ALWAYS check for FAO defects before KD.",
    },
    {
        "drug": "Phenobarbital — second-line (if pyridoxine + PLP negative)",
        "level": "Level B — moderate evidence for neonatal seizure control. Acts via GABA-A "
                 "potentiation. Not mechanism-targeted but available and partially effective "
                 "for burst phase. Avoid if EEG burst-suppression worsening.",
    },
    {
        "drug": "Levetiracetam — adjunct",
        "level": "Level C — SV2A ligand; reduces synaptic vesicle cycling; additive benefit "
                 "in neonatal DEE. Safe profile; no mitochondrial toxicity. Add after first-line "
                 "failure. Does not worsen burst-suppression.",
    },
    {
        "drug": "Clonazepam — adjunct (benzodiazepine bridge)",
        "level": "Level C — for acute seizure clusters / status epilepticus. GABA-A positive "
                 "allosteric modulator. Short-term use; tolerance develops. Buccal or IV. "
                 "Not as primary maintenance therapy.",
    },
    {
        "drug": "Sodium Benzoate — if NKH remains in differential",
        "level": "Level B (for NKH) — glycine cleavage system cofactor; NOT mechanism-targeted "
                 "for SLC25A22. Use only if plasma glycine >600 µmol/L (NKH DDx unresolved) "
                 "while genetic confirmation pending.",
    },
]

# ── CONTRAINDICATIONS ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "reason": (
            "ABSOLUTE CI in Ohtahara (burst-suppression). Na-channel blockers deepen the "
            "suppression phase by reducing Na⁺ channel availability during the already-silent "
            "inter-burst interval. May prolong burst duration without shortening bursts. "
            "Contraindicated even if tonic component mimics KCNQ2 (where CBZ is helpful)."
        ),
    },
    {
        "drug": "Valproate (VPA) without POLG screen",
        "reason": (
            "POLG sequencing MANDATORY before VPA. Fatal Alpers-Huttenlocher hepatic failure "
            "in POLG carriers. Additional concern specific to SLC25A22: VPA inhibits "
            "mitochondrial β-oxidation, compounding the bioenergetic failure caused by "
            "the GC1 LOF. Heightened mitochondrial toxicity risk in SLC25A22."
        ),
    },
    {
        "drug": "Vigabatrin (VGB) >16 weeks without REMS ophthalmology",
        "reason": (
            "VGB causes irreversible peripheral visual field constriction (30-40% patients "
            "with prolonged use). REMS programme: ophthalmology assessment every 3 months "
            "during IS treatment; annual thereafter. Maximum 16 weeks for IS protocol "
            "per UKISS without confirmed ophthalmology monitoring plan."
        ),
    },
    {
        "drug": "Lamotrigine in West/LGS evolution",
        "reason": (
            "Risk of myoclonic worsening in Lennox-Gastaut spectrum. Also SJS risk "
            "with rapid titration (slower protocol required). Not first-line in DEE3."
        ),
    },
    {
        "drug": "Sodium Valproate as monotherapy without POLG + FAO screen",
        "reason": (
            "In SLC25A22 specifically: GC1 LOF → mitochondrial TCA substrate depletion → "
            "cells more dependent on fatty acid oxidation as backup energy. VPA inhibits "
            "CPT1/LCAD → removes the FAO backup. Combination of GC1 LOF + VPA → acute "
            "mitochondrial energy crisis. Always confirm POLG + FAO status first."
        ),
    },
]

# ── MONITORING ────────────────────────────────────────────────────────────────
MONITORING = [
    {
        "timepoint": "Neonatal (Day 0-14)",
        "action": (
            "EEG STAT (burst-suppression characterisation); plasma amino acids (glutamate, "
            "glycine); CSF amino acids; pyridoxine 100 mg IV trial; PLP trial if B6 negative; "
            "MRI brain (bilateral BG signal changes); biotinidase; plasma lactate/ammonia; "
            "SLC25A22 sequencing + broad neonatal DEE panel simultaneous; POLG screen "
            "before any VPA; head circumference; APGAR/neonatal vitals."
        ),
    },
    {
        "timepoint": "3 Months",
        "action": (
            "Plasma amino acids (glutamate level on treatment); EEG evolution check "
            "(burst-suppression → hypsarrhythmia transition?); developmental milestone "
            "assessment; KD efficacy (ketone body levels if on KD: target β-OHB 2-5 mmol/L); "
            "ACTH taper per UKISS schedule; VGB ophthalmology baseline."
        ),
    },
    {
        "timepoint": "6 Months",
        "action": (
            "MRI brain follow-up (BG signal evolution; cortical myelination); "
            "developmental assessment (BSID-III or equivalent); seizure diary review; "
            "EEG (interictal evolution); plasma amino acids (glutamate monitoring on KD); "
            "VGB ophthalmology; metabolic labs (lactate, ammonia) on KD."
        ),
    },
    {
        "timepoint": "12 Months",
        "action": (
            "Comprehensive neurodevelopmental evaluation (occupational therapy, speech, "
            "physiotherapy); MRI (12-month myelination milestone + BG signal assessment); "
            "plasma amino acids annualised; EEG (spike-wave evolution); "
            "medication review (phenobarbital wean if seizure-free >3 months); "
            "genetic counselling for family (recurrence risk 25% AR siblings); "
            "cascade carrier testing of parents confirmed."
        ),
    },
    {
        "timepoint": "24 Months",
        "action": (
            "Annual review: developmental trajectory; seizure classification update; "
            "EEG (spike morphology, background); MRI (2-year cortical architecture); "
            "KD continuation decision (reassess benefit:risk); VGB ophthalmology annual; "
            "plasma glutamate (long-term metabolic monitoring); "
            "educational needs assessment (EHCP equivalent)."
        ),
    },
    {
        "timepoint": "Ongoing (Annual)",
        "action": (
            "Annual plasma amino acids (glutamate target <150 µmol/L on treatment); "
            "EEG; neuroimaging every 2-3 years; VGB ophthalmology annually; "
            "metabolic review (if KD: lipid profile, renal function, growth); "
            "transition planning (adolescent → adult service) from age 14; "
            "AED review for seizure freedom potential or reduction."
        ),
    },
]

# ── LIFECYCLE ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Neonatal (day 0-14)",
        "events": "Burst-suppression EEG; ohtahara; apnoea; poor feeding",
        "key_action": "Pyridoxine 100 mg IV; PLP trial; EEG + MRI + plasma AA STAT",
    },
    {
        "stage": "Early Infantile (1-3 months)",
        "events": "Persistent Ohtahara or transition to West syndrome",
        "key_action": "ACTH + VGB if IS; KD initiation; POLG screen before VPA",
    },
    {
        "stage": "Late Infantile (3-12 months)",
        "events": "West syndrome (hypsarrhythmia); developmental arrest",
        "key_action": "UKISS ACTH+VGB completion; KD efficacy review; plasma glutamate monitoring",
    },
    {
        "stage": "Toddler (1-3 years)",
        "events": "Lennox-Gastaut evolution (some); developmental plateau",
        "key_action": "EEG + AED review; multidisciplinary developmental input; annual MRI",
    },
    {
        "stage": "School age (4-10 years)",
        "events": "Ongoing drug-resistant epilepsy; intellectual disability",
        "key_action": "EHCP / educational plan; seizure rescue medication; annual metabolic review",
    },
    {
        "stage": "Adolescent / Adult",
        "events": "Continued DEE; transition to adult neurology; comorbid psychiatric needs",
        "key_action": "Transition planning from age 14; VGB ophthalmology; adult metabolic review",
    },
]

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "metric": "Plasma Glutamate",
        "normal": "<100 µmol/L",
        "alert_value": ">150 µmol/L (moderate elevation)",
        "action": "Investigate SLC25A22; confirm with fasting plasma amino acid profile",
        "critical_value": ">200 µmol/L (diagnostic in neonatal DEE context)",
    },
    {
        "metric": "CSF Glutamate",
        "normal": "<20 µmol/L",
        "alert_value": ">25 µmol/L",
        "action": "Confirm elevated CSF glutamate; paired plasma/CSF glutamate ratio",
        "critical_value": ">40 µmol/L (consistent with SLC25A22 LOF)",
    },
    {
        "metric": "Pyridoxine Trial Response",
        "normal": "No seizure cessation within 30 min",
        "alert_value": "Partial EEG improvement",
        "action": "Consider PNPO deficiency (PLP trial); check pipecolic acid + α-AASA for ALDH7A1",
        "critical_value": "Complete seizure cessation within 30-60 min → B6-responsive seizures confirmed",
    },
    {
        "metric": "KD Ketone Bodies (β-OHB)",
        "normal": "<0.5 mmol/L (non-ketotic)",
        "alert_value": "<1.0 mmol/L (inadequate ketosis)",
        "action": "Adjust KD ratio / caloric intake; dietitian review",
        "critical_value": ">5 mmol/L (hyperketo — reduce ratio or carbohydrate allowance)",
    },
    {
        "metric": "CSF:Plasma Glycine Ratio",
        "normal": "<0.06",
        "alert_value": ">0.06",
        "action": "Investigate NKH (GLDC/AMT/GCSH); plasma glycine >600 µmol/L confirms NKH DDx",
        "critical_value": ">0.08 (NKH diagnostic threshold; exclude before locking SLC25A22 diagnosis)",
    },
]

# ── DEFINITIONS ───────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "SLC25A22 / GC1",
        "definition": (
            "Solute Carrier Family 25, Member 22 — Mitochondrial Glutamate Carrier 1. "
            "323-aa electrogenic glutamate/H⁺ antiporter in mitochondrial inner membrane. "
            "Imports cytoplasmic glutamate into mitochondrial matrix for TCA cycle and "
            "malate–aspartate shuttle. LOF → cytoplasmic glutamate accumulation + "
            "mitochondrial bioenergetic deficit → neonatal epileptic encephalopathy. "
            "OMIM Gene 609302 / Disease DEE3 614563."
        ),
    },
    {
        "term": "Mitochondrial Carrier Family (MCF / SLC25)",
        "definition": (
            "Family of ~50 transport proteins in the mitochondrial inner membrane, all "
            "sharing a tripartite structure of three ~100-aa repeat modules, each with "
            "two transmembrane helices. Facilitate substrate exchange between cytoplasm "
            "and mitochondrial matrix. SLC25A22 is the glutamate carrier; others include "
            "ANT (SLC25A4/5, ADP/ATP), citrin (SLC25A13, aspartate/glutamate), "
            "DIC (SLC25A10, dicarboxylate), and AGC (SLC25A12/13, aspartate/glutamate)."
        ),
    },
    {
        "term": "Ohtahara Syndrome (DEE / EIEE)",
        "definition": (
            "Earliest-onset epileptic encephalopathy (day 0-3 months). Characteristic EEG: "
            "burst-suppression pattern (high-amplitude bursts of mixed slow-spike-wave ~1-3 s "
            "alternating with suppression periods ~2-5 s). Profound ID. High neonatal "
            "mortality. Multiple genetic causes: STXBP1 (most common), KCNQ2, SLC25A22, "
            "PLCB1, ARX, plus structural and metabolic causes. Genetic aetiology mandatory "
            "workup given treatment implications (e.g., CBZ helpful in KCNQ2, harmful in others)."
        ),
    },
    {
        "term": "Malate–Aspartate Shuttle",
        "definition": (
            "Key mitochondrial/cytoplasmic cycle that regenerates cytoplasmic NAD⁺ and "
            "transfers reducing equivalents into the mitochondrial matrix. Requires "
            "aspartate (exported from mitochondria via AGC1/SLC25A12) and malate. "
            "SLC25A22 LOF impairs matrix glutamate supply for transamination, reducing "
            "shuttle efficiency → impaired neuronal NADH regeneration → bioenergetic stress."
        ),
    },
    {
        "term": "Pyridoxine (B₆) Mandatory Trial",
        "definition": (
            "100 mg pyridoxine IV given to ALL neonates with unexplained seizures/DEE before "
            "specific diagnosis. B6-responsive neonatal seizures (ALDH7A1/antiquitin deficiency) "
            "is an immediately treatable cause with overlapping EEG. Positive response: "
            "seizure cessation within 30 min + EEG normalisation within 24 h. Negative "
            "response does NOT exclude SLC25A22. Follow with PLP trial (PNPO deficiency mimic)."
        ),
    },
    {
        "term": "Pyridoxal Phosphate (PLP) Trial",
        "definition": (
            "Active form of vitamin B₆; required as co-factor for GAD1/GAD2 (GABA synthesis) "
            "and >150 other enzymes. PNPO (pyridox(am)ine phosphate oxidase) deficiency → "
            "cannot synthesise PLP from B₆ → responds to PLP but NOT pyridoxine. "
            "PLP 30 mg/kg/day oral for 3-5 days after negative B6 IV trial. "
            "OMIM 610090. Must exclude before locking SLC25A22 as sole diagnosis."
        ),
    },
    {
        "term": "Elevated Plasma Glutamate (>150 µmol/L)",
        "definition": (
            "Key diagnostic biomarker in SLC25A22 DEE3. Fasting plasma amino acid profile "
            "measured by ion-exchange chromatography. Normal plasma glutamate <100 µmol/L. "
            "SLC25A22 null: typically >200 µmol/L. Hypomorphic: 120-180 µmol/L. "
            "Critical DDx: in NKH the elevated amino acid is GLYCINE (not glutamate). "
            "In STXBP1/KCNQ2 plasma amino acids are NORMAL. Elevated glutamate in neonatal "
            "DEE = SLC25A22 until proven otherwise."
        ),
    },
    {
        "term": "Bilateral Symmetric BG/Thalamic MRI Changes",
        "definition": (
            "Bilateral symmetric T2/FLAIR signal changes in basal ganglia (globus pallidus, "
            "putamen) and thalamus in SLC25A22 DEE3. Reflects selective vulnerability of "
            "basal ganglia neurons to mitochondrial energy failure (high metabolic demand). "
            "Key MRI distinction: PLCB1 → diffuse cortical atrophy (no BG changes); "
            "SLC25A22 → bilateral BG ± thalamic changes (no focal cortical dysplasia). "
            "Pattern also seen in: GLUT1 deficiency, biotin-thiamine-responsive BD, "
            "organic acidurias — metabolic MRI differential is broad."
        ),
    },
    {
        "term": "POLG Mandatory Screen",
        "definition": (
            "POLG (polymerase gamma; mitochondrial DNA polymerase) mutation carriers develop "
            "Alpers-Huttenlocher syndrome with VPA: fatal hepatic failure within weeks of VPA "
            "initiation. Screen BEFORE any VPA prescription. Critical in SLC25A22 since "
            "GC1 LOF already stresses mitochondrial function — compound POLG hit + VPA "
            "carries particularly high hepatotoxic risk. POLG sequencing + liver function "
            "baseline mandatory before VPA consideration."
        ),
    },
    {
        "term": "Migratory Partial Seizures of Infancy (MPSI-like)",
        "definition": (
            "EEG pattern: continuous migration of ictal discharges from one hemisphere "
            "to another, with multifocal onset. Classic genetic causes: KCNT1 (gain-of-function), "
            "SCN1A, SCN2A, TBC1D24. SLC25A22 compound heterozygous LOF can produce an "
            "MPSI-like pattern. Key DDx: elevated plasma glutamate (SLC25A22) vs normal "
            "amino acids with potassium channel phenotype (KCNT1). Quinidine trial (KCNT1) "
            "should NOT be given empirically without gene confirmation."
        ),
    },
    {
        "term": "Ketogenic Diet (KD) — Mechanism in SLC25A22",
        "definition": (
            "KD provides ketone bodies (β-OHB, acetoacetate) as alternative neuronal fuel, "
            "bypassing glycolysis and reducing pyruvate/glutamate generation from glucose. "
            "In SLC25A22: less cytoplasmic glutamate from reduced glycolytic flux → "
            "reduced NMDA/AMPA excitotoxic drive. β-OHB enters TCA directly as acetyl-CoA "
            "without requiring glutamate transaminase route → partially compensates for "
            "GC1 deficit. Evidence: Level B (observational) for SLC25A22. Always screen "
            "for fatty acid oxidation defects before KD initiation."
        ),
    },
    {
        "term": "NKH (Non-Ketotic Hyperglycinemia) DDx",
        "definition": (
            "GLDC/AMT/GCSH mutations. Elevated plasma GLYCINE (>600 µmol/L) and CSF:plasma "
            "glycine ratio >0.08. EEG burst-suppression + hiccups (pathognomonic combination). "
            "Plasma amino acids distinguish: NKH = glycine↑ (not glutamate); "
            "SLC25A22 = glutamate↑ (not glycine). Sodium benzoate (NKH-specific). "
            "Both are AR, neonatal onset, burst-suppression — clinical overlap is high, "
            "metabolic profile is the discriminator."
        ),
    },
    {
        "term": "ALDH7A1 (B6-Responsive Seizures / Antiquitin)",
        "definition": (
            "Alpha-aminoadipic semialdehyde dehydrogenase deficiency. AR. Neonatal seizures "
            "immediately responsive to pyridoxine 100 mg IV (seizure cessation ≤30 min). "
            "Biomarkers: elevated plasma pipecolic acid + urine α-aminoadipic semialdehyde "
            "(α-AASA). Positive B6 trial in neonatal DEE = presumptive ALDH7A1 until confirmed. "
            "Pyridoxine 100 mg IV is thus diagnostic AND therapeutic — mandatory before "
            "assuming SLC25A22 or any other neonatal DEE gene."
        ),
    },
    {
        "term": "GC1 vs GC2 (SLC25A22 vs SLC25A18)",
        "definition": (
            "SLC25A22 encodes GC1 (Mitochondrial Glutamate Carrier 1, 323 aa, 11p15.5). "
            "SLC25A18 encodes GC2 (Mitochondrial Glutamate Carrier 2, 11p14.3). "
            "Both transport glutamate but differ in expression pattern: GC1 is enriched "
            "in neurons (brain > liver); GC2 is more ubiquitous. LOF mutations in "
            "SLC25A22 (GC1) cause DEE3. SLC25A18 (GC2) mutations have not been causally "
            "linked to epilepsy but disrupt hepatic glutamate metabolism."
        ),
    },
    {
        "term": "Arg71Cys Founder Variant",
        "definition": (
            "p.Arg71Cys (c.211C>T) — the most common SLC25A22 pathogenic variant. "
            "Arg71 is in the substrate-binding pocket of Module 1, directly coordinating "
            "the α-carboxylate of glutamate. Cys substitution abolishes substrate binding "
            "(null-equivalent). Founder effect in the Arabian Peninsula / Middle East "
            "(consanguineous families). Homozygous Arg71Cys → classic Ohtahara null "
            "presentation. Also seen compound-het with frameshift alleles."
        ),
    },
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _pct(pts, key):
    n = len(pts)
    if n == 0:
        return 0
    return round(100 * sum(1 for p in pts if p.get(key)) / n)


def _mean(pts, key):
    vals = [p[key] for p in pts if isinstance(p.get(key), (int, float))]
    if not vals:
        return 0
    return round(sum(vals) / len(vals))


# ── API FUNCTIONS ─────────────────────────────────────────────────────────────
def get_overview():
    pts = PATIENTS
    n = len(pts)
    etiol_dist = []
    for ec in ETIOLOGY_CATALOG:
        cat_pts = [p for p in pts if p["category"] == ec["category"]]
        etiol_dist.append({
            "etiology": ec["category"].replace("SLC25A22-", "").replace("-", " "),
            "n": len(cat_pts),
            "pct": round(100 * len(cat_pts) / n),
        })
    treat_summary = [
        {"drug": t["drug"].split(" —")[0].split(" (")[0], "level": t["level"][:100]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {
            "timepoint": m["timepoint"],
            "action": m["action"][:85] + "…" if len(m["action"]) > 85 else m["action"],
        }
        for m in MONITORING[:5]
    ]
    return {
        "gene": "SLC25A22",
        "chromosome": "11p15.5",
        "omim_gene": "609302",
        "omim_disease": "609304",
        "protein": "Mitochondrial Glutamate Carrier 1 (GC1)",
        "aa_length": 323,
        "domains": (
            "Module 1 (aa 1-110, TM1-TM2, Arg71 glutamate binding) + "
            "Module 2 (aa 111-218, TM3-TM4, Asp123 transport cycle) + "
            "Module 3 (aa 219-323, TM5-TM6, Gly236 kink + Thr276 expression anchor)"
        ),
        "inheritance": "AR biallelic only (null + hypomorphic); no de novo / no somatic mosaic subtype",
        "disease_spectrum": "DEE3 / EIEE3 — Ohtahara (null) → West (hypomorphic) → Migratory focal (compound-het)",
        "unique_feature": (
            "Elevated plasma glutamate (>200 µmol/L) is the pathognomonic biomarker "
            "— measures fasting plasma amino acids. Bilateral symmetric BG/thalamic MRI "
            "changes (not cortical atrophy). Pyridoxine 100 mg IV trial MANDATORY before "
            "locking diagnosis (B6-responsive neonatal seizures is key treatable mimic). "
            "PHT/CBZ/OXC ABSOLUTE CI. VPA heightened mitochondrial toxicity risk in GC1 LOF."
        ),
        "cohort_seed": 507,
        "kpis": {
            "n_patients": n,
            "ohtahara_pct": _pct(pts, "ohtahara_syndrome"),
            "west_syndrome_pct": _pct(pts, "west_syndrome"),
            "migratory_focal_pct": _pct(pts, "migratory_focal_seizures"),
            "burst_suppression_pct": _pct(pts, "burst_suppression"),
            "hypsarrhythmia_pct": _pct(pts, "hypsarrhythmia"),
            "eeg_abnormal_pct": _pct(pts, "eeg_abnormal"),
            "elevated_plasma_glut_pct": _pct(pts, "elevated_plasma_glutamate"),
            "mean_plasma_glut_umol_L": _mean(pts, "plasma_glutamate_umol_L"),
            "bg_mri_changes_pct": _pct(pts, "bg_mri_changes"),
            "thalamic_changes_pct": _pct(pts, "thalamic_changes"),
            "profound_id_pct": _pct(pts, "profound_id"),
            "any_id_pct": _pct(pts, "any_id"),
            "b6_trial_pct": _pct(pts, "b6_pyridoxine_trial"),
            "acth_vgb_pct": _pct(pts, "acth_vgb_given"),
            "kd_tried_pct": _pct(pts, "kd_tried"),
            "mri_done_pct": _pct(pts, "mri_done"),
            "yr1_mortality_pct": _pct(pts, "yr1_mortality"),
            "polg_tested_pct": _pct(pts, "polg_tested"),
        },
        "etiology_distribution": etiol_dist,
        "treatments_summary": treat_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS[:5],
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
    }


def get_breakdown():
    pts = PATIENTS
    by_cat = {}
    for p in pts:
        c = p["category"].replace("SLC25A22-", "").replace("-", " ")
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(p)

    breakdown = []
    for cat, cat_pts in by_cat.items():
        n = len(cat_pts)
        breakdown.append({
            "category": cat,
            "n": n,
            "ohtahara_pct": _pct(cat_pts, "ohtahara_syndrome"),
            "west_pct": _pct(cat_pts, "west_syndrome"),
            "migratory_focal_pct": _pct(cat_pts, "migratory_focal_seizures"),
            "burst_suppression_pct": _pct(cat_pts, "burst_suppression"),
            "hypsarrhythmia_pct": _pct(cat_pts, "hypsarrhythmia"),
            "elevated_plasma_glut_pct": _pct(cat_pts, "elevated_plasma_glutamate"),
            "mean_plasma_glut": _mean(cat_pts, "plasma_glutamate_umol_L"),
            "bg_mri_pct": _pct(cat_pts, "bg_mri_changes"),
            "profound_id_pct": _pct(cat_pts, "profound_id"),
            "any_id_pct": _pct(cat_pts, "any_id"),
            "acth_vgb_pct": _pct(cat_pts, "acth_vgb_given"),
            "kd_pct": _pct(cat_pts, "kd_tried"),
            "yr1_mortality_pct": _pct(cat_pts, "yr1_mortality"),
        })

    etiol_details = [
        {
            "category": ec["category"].replace("SLC25A22-", "").replace("-", " "),
            "typical_variant": ec["typical_variant"],
            "inheritance": ec["inheritance"],
            "functional_deficit": ec["functional_deficit"],
            "description": ec["description"],
        }
        for ec in ETIOLOGY_CATALOG
    ]

    summary = {
        "ohtahara_pct": _pct(pts, "ohtahara_syndrome"),
        "west_pct": _pct(pts, "west_syndrome"),
        "burst_suppression_pct": _pct(pts, "burst_suppression"),
        "hypsarrhythmia_pct": _pct(pts, "hypsarrhythmia"),
        "elevated_plasma_glut_pct": _pct(pts, "elevated_plasma_glutamate"),
        "mean_plasma_glut_umol_L": _mean(pts, "plasma_glutamate_umol_L"),
        "bg_mri_changes_pct": _pct(pts, "bg_mri_changes"),
        "profound_id_pct": _pct(pts, "profound_id"),
        "acth_vgb_pct": _pct(pts, "acth_vgb_given"),
        "kd_pct": _pct(pts, "kd_tried"),
        "b6_trial_pct": _pct(pts, "b6_pyridoxine_trial"),
        "yr1_mortality_pct": _pct(pts, "yr1_mortality"),
        "polg_tested_pct": _pct(pts, "polg_tested"),
    }

    return {
        "gene": "SLC25A22",
        "chromosome": "11p15.5",
        "cohort_size": len(pts),
        "cohort_seed": 507,
        "summary": summary,
        "by_category": breakdown,
        "etiology_details": etiol_details,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
    }


def get_definitions():
    return {
        "gene": "SLC25A22",
        "chromosome": "11p15.5",
        "protein": "Mitochondrial Glutamate Carrier 1 (GC1)",
        "omim_gene": "609302",
        "omim_disease": "609304",
        "disease_name": "DEE3 / EIEE3 — Developmental and Epileptic Encephalopathy 3",
        "inheritance": "AR biallelic (null + hypomorphic); strictly recessive — no de novo, no somatic mosaic",
        "definitions": DEFINITIONS,
        "key_ddx": [
            "PLCB1 (20p12.3): plasma glutamate NORMAL; diffuse cortical atrophy MRI (no BG); "
            "IP3/DAG pathway (not mitochondrial); somatic mosaic / FCD IIb subtype absent in SLC25A22",
            "STXBP1 (9q34.11): plasma amino acids NORMAL; vesicle fusion pathway; AD de novo; "
            "most common neonatal DEE gene; no metabolic signature",
            "NKH/GLDC (AR): elevated GLYCINE (not glutamate); CSF:plasma glycine >0.08; "
            "hiccups pathognomonic; sodium benzoate Rx — key metabolic discriminator",
            "KCNQ2 (20q13.33): tonic asymmetric neonatal seizures; NOT burst-suppression "
            "predominantly; CBZ/PHT HELPFUL (opposite of SLC25A22)",
            "ALDH7A1 B6-responsive: pyridoxine 100 mg IV → immediate seizure cessation; "
            "elevated pipecolic acid + α-AASA; treatable — mandatory exclusion trial",
            "PNPO deficiency: responds to PLP but NOT B₆; OMIM 610090; PLP 30 mg/kg/day "
            "trial after negative B6 result before locking SLC25A22",
        ],
        "mandatory_workup": [
            "Plasma amino acids (FASTING): glutamate >150 µmol/L diagnostic in neonatal DEE",
            "CSF amino acids: CSF glutamate elevation confirms central nervous system involvement",
            "Pyridoxine 100 mg IV STAT trial (MANDATORY in all neonatal DEE — treatable mimics)",
            "Pyridoxal phosphate (PLP) 30 mg/kg/day if B6 trial negative (PNPO exclusion)",
            "SLC25A22 sequencing + MLPA (gene panel or targeted in neonatal DEE workup)",
            "Broad neonatal DEE panel: STXBP1 / KCNQ2 / ARX / PLCB1 / CDKL5 / SCN1A simultaneous",
            "EEG STAT: burst-suppression vs hypsarrhythmia vs migrating pattern characterisation",
            "MRI brain (3T): bilateral BG/thalamic signal changes + cortical architecture",
            "Biotinidase activity (cheap, treatable, critical exclusion)",
            "Plasma lactate, ammonia, organic acids (mitochondrial DDx broad screen)",
            "CSF:plasma glycine ratio (NKH exclusion; normal <0.06)",
            "Plasma pipecolic acid + urine α-AASA (ALDH7A1 confirmation if B6-responsive)",
            "POLG sequencing MANDATORY before any VPA consideration",
            "Cascade genetic testing: parents (carriers); siblings (25% recurrence risk)",
        ],
        "standards": [
            "OMIM 609304 (DEE3 / EIEE3) — SLC25A22",
            "Molinari et al. (2005) Am J Hum Genet 76:334-339 (original SLC25A22 DEE paper)",
            "Molinari et al. (2009) Ann Neurol 65:630-635 (neonatal epilepsy + suppression-bursts)",
            "ILAE Gene Classification (2022): SLC25A22 — definitive DEE gene",
            "UKISS protocol (ACTH + VGB for infantile spasms)",
            "VGB REMS programme (visual field; max 16 weeks IS use)",
            "POLG Working Group guidelines (pre-VPA screening)",
            "SEQC2 Variant Curation Guidelines (ClinGen SLC25A22 expert panel)",
        ],
        "five_key_facts": [
            "Elevated plasma glutamate (>200 µmol/L) on fasting amino acids is the pathognomonic "
            "biomarker — distinguishes SLC25A22 DEE3 from PLCB1 (normal glutamate) and NKH "
            "(elevated glycine not glutamate)",
            "Bilateral symmetric BG/thalamic MRI signal changes (not diffuse cortical atrophy) "
            "is the imaging signature — reflects selective vulnerability of BG neurons to "
            "mitochondrial GC1 LOF bioenergetic failure",
            "Pyridoxine 100 mg IV trial is MANDATORY before locking diagnosis — B6-responsive "
            "seizures (ALDH7A1) and PNPO deficiency are immediately treatable mimics with "
            "overlapping neonatal burst-suppression EEG",
            "PHT/CBZ/OXC ABSOLUTE CONTRAINDICATED (worsen Ohtahara burst-suppression); "
            "VPA carries heightened mitochondrial toxicity risk in GC1 LOF — POLG screen mandatory",
            "Strictly AR (no de novo, no somatic mosaic) — recurrence risk 25% for siblings; "
            "cascade carrier testing of parents and at-risk relatives is mandatory",
        ],
    }
