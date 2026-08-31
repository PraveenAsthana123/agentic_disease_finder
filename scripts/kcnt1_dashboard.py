"""
KCNT1 Epilepsy — Na+-Activated K+ Channel / SLACK / KNa1.1 / DEE14 / MMPSI / 9q34.3
=======================================================================================
40-patient cohort · KCNT1 (9q34.3) · GoF de novo + GoF familial + GoF AR + Phenocopy

KCNT1 BIOLOGY:
KCNT1 (9q34.3) encodes the sodium-activated potassium channel KNa1.1 (also known as
SLACK — Sequence Like A Calcium-activated K+ channel — or SLO2.2). It is a large-conductance
Na+-activated K+ channel of 1237 amino acids expressed predominantly in neurons,
particularly in the cerebral cortex, hippocampus, brainstem, and spinal cord.
Under physiological conditions, KNa1.1 provides an adaptation current: Na+ influx during
action potentials activates the channel → K+ efflux → membrane hyperpolarisation.
This adaptation current limits burst firing and provides network-level homeostasis.
Pathogenic gain-of-function (GoF) mutations lower the Na+ activation threshold → channels
open at resting Na+ concentrations → excessive constitutive K+ efflux. Paradoxically,
this dysregulates glutamate release from presynaptic terminals, disrupts glial potassium
buffering, and entrains pathological network synchrony — producing malignant epileptic
encephalopathy rather than the expected hyperpolarisation.

KCNT1 — STRUCTURE (1237 aa):
  N-TERMINAL (aa 1-40): Short intracellular N-terminus; regulatory phosphorylation sites.
  S1-S4 VOLTAGE-SENSING-LIKE DOMAIN (aa 41-180): Four transmembrane segments with charged
    residues; voltage-sensing-like but KNa1.1 is not strictly voltage-gated at physiological
    voltages; S4 contains positively charged residues that contribute to gating.
  S5-PORE-S6 DOMAIN (aa 181-310): Forms the conduction pore with K+-selectivity filter
    (GYG motif); S5 is the outer helix, S6 is the inner helix (gate). The selectivity
    filter is identical in architecture to other Kv and KCa channels.
  LARGE C-TERMINAL DOMAIN (aa 311-1237): ~75% of the protein; contains:
    - RCK1 domain (Regulator of K+ Conductance 1, aa 340-490): primary Na+ binding site;
      Na+ coordinates to backbone carbonyl oxygens of conserved residues; GoF mutations
      cluster heavily in RCK1 (e.g. R428Q, Y796H, A913V).
    - Linker region (aa 491-694): flexible loop between RCK1 and RCK2.
    - RCK2 domain (aa 695-840): secondary Na+ binding site; binds NADP+ (regulatory);
      NAD+ binding in RCK1/RCK2 modulates channel gating; both RCK domains form an
      octameric gating ring (two tetramers) in the open-channel state.
    - C-terminal tail (aa 841-1237): contains protein interaction motifs; interacts with
      Fragile X Mental Retardation Protein (FMRP) — KNa1.1/FMRP complex modulates
      presynaptic glutamate release; GoF disrupts this modulatory complex.

KCNT1 MECHANISM — GoF PATHOPHYSIOLOGY:
  Normal: Na+ enters during action potential bursts → Na+ binds RCK1 → channel opens →
          K+ efflux → after-hyperpolarisation → burst termination (adaptation current).
  GoF: Lower Na+ threshold → channel open at baseline Na+ → excessive K+ efflux at rest →
       paradoxical excitability via:
       (1) Disruption of FMRP-mediated presynaptic glutamate release regulation
           → uncontrolled glutamatergic excitation;
       (2) Glial K+ buffering dysregulation (KNa1.1 is also expressed in glia) →
           elevated extracellular [K+] → further neuronal depolarisation;
       (3) Network synchrony entrainment → Malignant Migrating Partial Seizures of
           Infancy (MMPSI) pattern on EEG: continuous migration of ictal discharges
           across both hemispheres.
  Quinidine: Open-channel blocker of KNa1.1. Enters the open pore and blocks K+ flux.
    ONLY effective when the channel is in the GoF-open state. Contraindicated in LoF.
    QTc monitoring mandatory.

PHENOTYPIC SPECTRUM:
  GoF DE NOVO (MMPSI / DEE14): Most severe. Neonatal to 3-month onset. MMPSI pattern on
    EEG: continuous multifocal migration of ictal discharges across both hemispheres.
    High seizure burden (>50 seizures/day common). Profound ID. Drug-resistant (>3 AEDs
    failed typical). No metabolic biomarker (unlike SLC25A22: no elevated plasma glutamate;
    unlike NKH: no elevated glycine). MRI initially normal → progressive cortical atrophy.
    Quinidine trial indicated if GoF confirmed. ~40% of KCNT1 epilepsy cohort.
  GoF AD FAMILIAL (ADNFLE2 — Autosomal Dominant Nocturnal Frontal Lobe Epilepsy type 2):
    Childhood onset. Hypermotor seizures from sleep (nocturnal frontal lobe semiology):
    tonic asymmetric posturing, cycling movements, vocalisation during NREM sleep.
    Milder cognitive trajectory vs MMPSI. Familial (autosomal dominant inheritance).
    Genetic family cascade testing mandatory. Quinidine trial may help in refractory cases.
    ~20% of KCNT1 epilepsy cohort. OMIM disease 615005.
  GoF AR BIALLELIC (SEVERE DEE): Both alleles carry GoF mutations (extremely rare).
    Neonatal onset. Ohtahara-like burst-suppression. Very severe. Only a few reported
    families globally. ~15% of KCNT1 severe DEE cohort in specialist centres.
  PHENOCOPY (KCNT1-NEGATIVE MMPSI): MMPSI-like clinical and EEG picture without
    pathogenic KCNT1 variant. DDx: SCN1A (febrile seizures + migrating pattern in SMEI
    subtype), SCN8A (rapid progression, different EEG morphology), ATP1A3 (alternating
    hemiplegia of childhood — AHC, triggered by infection/fever, hemiplegic episodes not
    seizures). Broad epilepsy gene panel mandatory. ~25% of clinical MMPSI in large cohorts.

DISTINGUISHING KCNT1 FROM KEY DDx:
  KCNT1 (GoF): No metabolic biomarker; migrating ictal EEG; quinidine GoF-specific;
               CBZ/OXC/PHT ABSOLUTE CI (worsen MMPSI); normal plasma amino acids.
  SCN1A Dravet: Febrile/temperature-sensitive seizures; SMEI EEG not migrating pattern;
               stiripentol + clobazam + VPA (Dravet-specific); SCN1A = most common DEE gene.
  SLC25A22 DEE3: Elevated plasma glutamate >200 umol/L (pathognomonic); BG/thalamic MRI;
               AR biallelic (no GoF); pyridoxine trial mandatory; no quinidine.
  ATP1A3 AHC: Na+/K+-ATPase alpha-3 subunit; alternating hemiplegia (triggered episodes,
               not ictal); flunarizine Rx; no migrating EEG seizure pattern.
  NKH: Elevated plasma glycine + CSF:plasma glycine ratio >0.08; hiccups; AR glycine
               cleavage; sodium benzoate Rx; KCNT1 = glycine NORMAL.

CONTRAINDICATED DRUGS:
  CARBAMAZEPINE / OXCARBAZEPINE / PHENYTOIN:
    ABSOLUTE CONTRAINDICATION in KCNT1 MMPSI/DEE14 (GoF). Na-channel blockers reduce
    the Na+ influx that normally activates KNa1.1 adaptation current, dysregulating
    network synchrony and worsening the migrating ictal pattern. Case reports of acute
    seizure escalation within 24-48h of CBZ in KCNT1 MMPSI.
  LAMOTRIGINE: Na-channel modulator — avoid in KCNT1 MMPSI/GoF channelopathy.
  VALPROATE without POLG screen: POLG sequencing MANDATORY before any VPA.
    Fatal Alpers-Huttenlocher hepatic failure in POLG carriers.

QUINIDINE THERAPY PROTOCOL:
  Drug: Quinidine sulfate — open-channel blocker of KNa1.1.
  Indication: KCNT1 GoF mutations ONLY (confirmed by in vitro or computational GoF evidence).
  Dose: Start 15 mg/kg/day oral divided every 6h; titrate to plasma level 2-5 ug/mL.
  Monitoring: ECG before initiation (baseline QTc); QTc at every dose change; plasma level
    at steady state (5 days); liver function; monthly QTc in maintenance.
  Stop if: QTc >450 ms (absolute) or >60 ms increase from baseline.

REFERENCES:
  Barcia G et al. (2012) De novo gain-of-function KCNT1 channel mutations cause
    malignant migrating partial seizures of infancy. Nat Genet 44:1255-1259. PMID 23086397.
  Milligan CJ et al. (2014) KCNT1 gain of function in 2 epilepsy phenotypes.
    Ann Neurol 76:826-834. PMID 25346106.
  Rizzo F et al. (2016) Unique quinidine effects on KCNT1 channels.
    Neurology 86:1063-1071. PMID 26888995.
  ILAE Gene Classification (2022): KCNT1 — DEE14 / MMPSI (OMIM 614959), ADNFLE2 (615005).
"""

import random

random.seed(509)

# ── ETIOLOGY CATALOG ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "KCNT1-GoF-AD-DeNovo-MMPSI",
        "n_target": 16,
        "description": (
            "De novo gain-of-function heterozygous KCNT1 mutations. Onset <3 months (typically "
            "within first 6 months). MMPSI: continuous multifocal migration of ictal activity "
            "across both hemispheres on EEG. High seizure burden (>50/day common). Profound ID. "
            "Drug-resistant (>3 AEDs failed). No metabolic biomarker (amino acids normal). "
            "Quinidine trial indicated — open-channel block GoF-specific. CBZ/OXC/PHT/LTG "
            "ABSOLUTE CI (worsen migrating pattern). ~40% of KCNT1 epilepsy cohort."
        ),
        "typical_variant": "R428Q (RCK1), Y796H (RCK2 linker), A913V (RCK2), M896I — de novo heterozygous",
        "inheritance": "AD de novo GoF",
        "functional_deficit": (
            "GoF KNa1.1 channel: open at resting Na+ → excessive K+ efflux → paradoxical "
            "excitation via FMRP/glutamate dysregulation + glial K+ buffering failure → MMPSI"
        ),
    },
    {
        "category": "KCNT1-GoF-AD-ADNFLE",
        "n_target": 8,
        "description": (
            "Autosomal dominant familial GoF KCNT1 mutations. Autosomal dominant nocturnal "
            "frontal lobe epilepsy type 2 (ADNFLE2, OMIM 615005). Onset childhood (2-20 years). "
            "Hypermotor nocturnal seizures: tonic asymmetric posturing, cycling movements, "
            "vocalisation from NREM sleep. Milder cognitive trajectory than MMPSI. Familial "
            "with autosomal dominant inheritance — cascade genetic testing of family essential. "
            "Quinidine may help in refractory ADNFLE2. ~20% of KCNT1 epilepsy cohort."
        ),
        "typical_variant": "Y796H (also in MMPSI), R474H (RCK1), T314A (N-terminal C-domain junction)",
        "inheritance": "AD familial GoF",
        "functional_deficit": (
            "GoF KNa1.1: dysregulated frontal network synchrony during NREM sleep → "
            "hypermotor nocturnal seizures; milder than MMPSI due to partial GoF effect"
        ),
    },
    {
        "category": "KCNT1-GoF-AR-Biallelic-SevereDE",
        "n_target": 6,
        "description": (
            "Biallelic GoF KCNT1 mutations — both alleles carry GoF variants (extremely rare; "
            "only a handful of families reported globally). Neonatal onset. Ohtahara-like severe "
            "epileptic encephalopathy with burst-suppression. Very severe ID. May combine "
            "MMPSI pattern with burst-suppression elements. More severe than heterozygous "
            "GoF suggesting compound-dosage effect on KNa1.1 dysregulation. ~15% of KCNT1 "
            "cohort in specialist epilepsy genetics centres."
        ),
        "typical_variant": "Compound GoF: R428Q/A456V biallelic or homozygous missense in consanguineous families",
        "inheritance": "AR biallelic GoF (extremely rare)",
        "functional_deficit": (
            "Biallelic GoF: maximal KNa1.1 dysregulation → neonatal burst-suppression + "
            "MMPSI pattern; most severe KCNT1 phenotype"
        ),
    },
    {
        "category": "KCNT1-Phenocopy",
        "n_target": 10,
        "description": (
            "MMPSI-like clinical and EEG phenotype without identified KCNT1 pathogenic variant. "
            "Broad epilepsy gene panel mandatory: SCN1A (febrile seizures + SMEI subtype), "
            "SCN8A (rapid progression, different EEG), ATP1A3 (alternating hemiplegia — AHC, "
            "not truly ictal migrations), TBC1D24 (DOOR syndrome + MMPSI-like), SCN2A "
            "(neonatal epilepsy + autism). ~25% of clinical MMPSI in large cohorts. "
            "Empirical AED trial without quinidine (gene-confirmation required for quinidine)."
        ),
        "typical_variant": "No KCNT1 pathogenic variant; SCN1A/SCN8A/ATP1A3/TBC1D24 DDx",
        "inheritance": "Unknown (phenocopy); SCN1A = de novo/AD most common",
        "functional_deficit": "Not KCNT1 — alternative ion channel or transporter mechanism",
    },
]

# ── PATIENT COHORT (40 patients, seed 509) ──────────────────────────────────
def _build_cohort():
    rng = random.Random(509)
    pts = []
    pid = 1
    for ec in ETIOLOGY_CATALOG:
        n = ec["n_target"]
        for _ in range(n):
            cat = ec["category"]
            is_mmpsi  = cat == "KCNT1-GoF-AD-DeNovo-MMPSI"
            is_adnfle = cat == "KCNT1-GoF-AD-ADNFLE"
            is_ar     = cat == "KCNT1-GoF-AR-Biallelic-SevereDE"
            is_pheno  = cat == "KCNT1-Phenocopy"

            # Onset in days
            age_onset_days = (
                rng.randint(0, 90)     if is_mmpsi  else
                rng.randint(730, 5110) if is_adnfle else   # 2-14 years
                rng.randint(0, 28)     if is_ar     else
                rng.randint(0, 180)                         # phenocopy
            )

            mmpsi               = rng.random() < (0.92 if is_mmpsi else 0.05 if is_adnfle else 0.70 if is_ar else 0.40)
            adnfle              = rng.random() < (0.05 if is_mmpsi else 0.90 if is_adnfle else 0.10 if is_ar else 0.10)
            ohtahara_like       = rng.random() < (0.15 if is_mmpsi else 0.02 if is_adnfle else 0.75 if is_ar else 0.20)
            burst_suppression   = ohtahara_like and rng.random() < (0.80 if is_ar else 0.60 if is_mmpsi else 0.30)
            migrating_focal_eeg = mmpsi and rng.random() < (0.90 if is_mmpsi else 0.60 if is_ar else 0.30)
            hypsarrhythmia      = rng.random() < (0.10 if is_mmpsi else 0.02 if is_adnfle else 0.20 if is_ar else 0.12)
            drug_resistant      = rng.random() < (0.92 if is_mmpsi else 0.35 if is_adnfle else 0.95 if is_ar else 0.65)
            n_aeds_failed       = (
                rng.randint(3, 7) if is_mmpsi  else
                rng.randint(1, 3) if is_adnfle else
                rng.randint(3, 8) if is_ar     else
                rng.randint(1, 5)
            )

            # Quinidine
            quinidine_tried     = rng.random() < (0.72 if is_mmpsi else 0.25 if is_adnfle else 0.55 if is_ar else 0.10)
            quinidine_responded = quinidine_tried and rng.random() < (0.38 if is_mmpsi else 0.45 if is_adnfle else 0.30 if is_ar else 0.10)

            acth_vgb_given      = rng.random() < (0.45 if is_mmpsi else 0.05 if is_adnfle else 0.60 if is_ar else 0.25)
            kd_tried            = rng.random() < (0.55 if is_mmpsi else 0.20 if is_adnfle else 0.65 if is_ar else 0.30)
            polg_tested         = rng.random() < 0.85
            profound_id         = rng.random() < (0.82 if is_mmpsi else 0.08 if is_adnfle else 0.90 if is_ar else 0.40)
            any_id              = profound_id or rng.random() < (0.95 if is_mmpsi else 0.35 if is_adnfle else 0.98 if is_ar else 0.60)
            qtc_monitored       = quinidine_tried or rng.random() < 0.30
            eeg_migrating_pattern = migrating_focal_eeg or rng.random() < (0.10 if is_adnfle else 0.05)
            seizure_free        = rng.random() < (0.02 if is_mmpsi else 0.35 if is_adnfle else 0.02 if is_ar else 0.15)

            sex = rng.choice(["M", "F"])
            pts.append({
                "patient_id":             f"KCNT1-{pid:03d}",
                "sex":                    sex,
                "category":               cat,
                "age_onset_days":         age_onset_days,
                "mmpsi":                  mmpsi,
                "adnfle":                 adnfle,
                "ohtahara_like":          ohtahara_like,
                "burst_suppression":      burst_suppression,
                "migrating_focal_eeg":    migrating_focal_eeg,
                "hypsarrhythmia":         hypsarrhythmia,
                "drug_resistant":         drug_resistant,
                "n_aeds_failed":          n_aeds_failed,
                "quinidine_tried":        quinidine_tried,
                "quinidine_responded":    quinidine_responded,
                "acth_vgb_given":         acth_vgb_given,
                "kd_tried":               kd_tried,
                "polg_tested":            polg_tested,
                "profound_id":            profound_id,
                "any_id":                 any_id,
                "qtc_monitored":          qtc_monitored,
                "eeg_migrating_pattern":  eeg_migrating_pattern,
                "seizure_free":           seizure_free,
            })
            pid += 1
    return pts


PATIENTS = _build_cohort()

# ── TREATMENTS ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Quinidine Sulfate — GoF-SPECIFIC open-channel block of KNa1.1",
        "level": (
            "Level B-C (case series + small open-label trials). ONLY for confirmed KCNT1 GoF "
            "mutations. Oral quinidine sulfate, start 15 mg/kg/day divided every 6h. Target "
            "plasma level 2-5 ug/mL. ECG before initiation (baseline QTc) and with every "
            "dose change. Hold if QTc >450 ms or rise >60 ms from baseline. Assess EEG "
            "migrating pattern response at 4-8 weeks. Responder rate ~30-40%. NOT to be given "
            "empirically without confirmed KCNT1 GoF variant status."
        ),
    },
    {
        "drug": "ACTH + Vigabatrin (UKISS Level A) — for IS/West component",
        "level": (
            "Level A — standard UKISS protocol for infantile spasms component or West syndrome "
            "transition. ACTH 150 IU/m2/day (2 weeks) + VGB 50 mg/kg/day maintained. VGB REMS "
            "mandatory (visual field restriction); ophthalmology review every 3 months. Not "
            "mechanism-targeted for KCNT1 GoF migrating pattern but addresses IS component "
            "when hypsarrhythmia is present."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD) — 4:1 or modified Atkins",
        "level": (
            "Level B (observational evidence in KCNT1 MMPSI). Reduces network excitability via "
            "ketone body metabolism; may dampen glutamatergic excitation via reduced glycolysis. "
            "RD dietitian + metabolic team mandatory. Screen for fatty acid oxidation defects "
            "before initiation. Some MMPSI patients show EEG improvement; not curative. "
            "Monitor beta-OHB (target 2-5 mmol/L), growth, lipid profile."
        ),
    },
    {
        "drug": "Phenobarbital — first rescue/bridge (not curative for MMPSI)",
        "level": (
            "Level B for neonatal seizure control. GABA-A potentiation. Available in neonatal "
            "period and provides partial seizure reduction. Not mechanism-targeted for KCNT1 "
            "GoF. Useful as bridge while quinidine is initiated or while KD is established. "
            "IV loading: 20 mg/kg IV over 30 min. Avoid if causing respiratory depression."
        ),
    },
    {
        "drug": "Clobazam — adjunct benzodiazepine",
        "level": (
            "Level C adjunct. 1,5-benzodiazepine with GABA-A modulatory activity. Lower sedation "
            "profile than clonazepam. Additive benefit in MMPSI for reducing seizure clusters. "
            "Start 0.1-0.3 mg/kg/day; tolerance may develop over weeks-months. Buccal midazolam "
            "for acute cluster rescue. Not disease-modifying."
        ),
    },
    {
        "drug": "Levetiracetam — adjunct SV2A ligand",
        "level": (
            "Level C adjunct. SV2A vesicle protein ligand; reduces neurotransmitter release. "
            "Safe mitochondrial profile; no hepatic concerns. Additive in MMPSI. May paradoxically "
            "worsen behaviour/irritability in some patients. 20-60 mg/kg/day in divided doses. "
            "Does not worsen migrating EEG pattern."
        ),
    },
    {
        "drug": "ABSOLUTE CI: CBZ / OXC / PHT / LTG — worsen MMPSI, pro-convulsant in GoF",
        "level": (
            "ABSOLUTE CONTRAINDICATION in KCNT1 MMPSI/DEE14. Na-channel blockers (CBZ, OXC, PHT) "
            "block Na+ influx → reduce Na+ available to activate KNa1.1 adaptation current → "
            "dysregulates network-level burst termination → documented worsening of migrating "
            "ictal pattern in KCNT1. Case reports of acute seizure escalation within 24-48h "
            "of CBZ initiation in MMPSI. LTG also Na-channel modulator — avoid in MMPSI GoF. "
            "Key DDx: CBZ IS HELPFUL in KCNQ2 — making CBZ response a discriminator."
        ),
    },
]

# ── CONTRAINDICATIONS ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
        "reason": (
            "ABSOLUTE CONTRAINDICATION in KCNT1 MMPSI/DEE14 (GoF). Na-channel blockers reduce "
            "the Na+ influx that normally activates KNa1.1 adaptation current, dysregulating "
            "network synchrony and worsening the migrating ictal pattern. Multiple case reports "
            "of acute seizure escalation within 24-48h of CBZ in KCNT1 MMPSI. Critical "
            "distinction: CBZ IS HELPFUL in KCNQ2 neonatal DEE — use as discriminator."
        ),
    },
    {
        "drug": "Lamotrigine (LTG)",
        "reason": (
            "Na-channel modulator — avoid in KCNT1 MMPSI/GoF channelopathy. Same mechanism "
            "as CBZ/OXC (Na+ channel block). Risk of worsening migrating ictal pattern. "
            "Additionally risk of myoclonic worsening in Lennox-Gastaut evolution. Stevens-Johnson "
            "syndrome risk with rapid titration in DEE context."
        ),
    },
    {
        "drug": "Quinidine WITHOUT confirmed KCNT1 GoF status",
        "reason": (
            "Quinidine is GoF-specific. Prescribing without confirmed GoF variant status "
            "risks QTc prolongation (potentially fatal Torsades de Pointes arrhythmia) without "
            "potential benefit. Always confirm KCNT1 GoF variant by in vitro functional "
            "assessment or strong computational GoF prediction before initiating quinidine. "
            "QTc >450 ms = absolute contraindication regardless of GoF status."
        ),
    },
    {
        "drug": "Valproate (VPA) without POLG screen",
        "reason": (
            "POLG sequencing MANDATORY before any VPA in DEE. Fatal Alpers-Huttenlocher "
            "hepatic failure in POLG carriers. VPA is not specifically contraindicated in "
            "KCNT1 by mechanism (no mitochondrial involvement unlike SLC25A22), but POLG "
            "screen is universally mandatory before VPA in all DEE patients per protocol."
        ),
    },
]

# ── MONITORING ────────────────────────────────────────────────────────────────
MONITORING = [
    {
        "timepoint": "Pre-Quinidine Initiation",
        "action": (
            "ECG (12-lead): baseline QTc measurement. QTc >450 ms = contraindication to "
            "quinidine. Electrolytes (K+, Mg2+ — hypokalemia/hypomagnesemia increase QTc risk). "
            "Liver function. Complete blood count. KCNT1 GoF variant functional confirmation. "
            "Plasma quinidine level at steady state (day 5). EEG baseline (migrating pattern "
            "frequency, ictal burden). Video-EEG recording for migrating ictal documentation."
        ),
    },
    {
        "timepoint": "Quinidine Dose Titration (Every 2-4 weeks)",
        "action": (
            "ECG QTc at every dose increase. Target quinidine plasma level 2-5 ug/mL. "
            "Hold quinidine if QTc >450 ms or rise >60 ms from baseline. Monitor for "
            "cinchonism (tinnitus, headache, visual disturbance). Seizure diary review. "
            "EEG at 4-8 weeks to assess migrating pattern response."
        ),
    },
    {
        "timepoint": "3 Months",
        "action": (
            "EEG (migrating pattern evolution); quinidine plasma level; QTc; developmental "
            "milestone assessment (Bayley or BSID-III); seizure diary; ACTH taper per UKISS "
            "if IS component; KD ketone levels if on KD (target beta-OHB 2-5 mmol/L); "
            "VGB ophthalmology baseline if VGB initiated."
        ),
    },
    {
        "timepoint": "6 Months",
        "action": (
            "MRI brain (initial and 6-month evolution — KCNT1 MMPSI may show progressive "
            "cortical atrophy); comprehensive developmental assessment; seizure frequency "
            "review; EEG (migrating pattern resolution or evolution); quinidine response "
            "decision (continue/stop/escalate); KD metabolic labs (lipid, renal, growth)."
        ),
    },
    {
        "timepoint": "12 Months Annual",
        "action": (
            "Comprehensive neurodevelopmental evaluation (OT, speech, PT); MRI 12-month; "
            "QTc annual on quinidine maintenance; VGB ophthalmology annual; EEG background "
            "evolution; AED rationalisation review; genetic counselling (de novo: low recurrence "
            "<1%; AD familial: 50% offspring risk; AR biallelic: 25% sibling risk)."
        ),
    },
    {
        "timepoint": "Ongoing (Annual)",
        "action": (
            "Annual QTc on quinidine; EEG; VGB ophthalmology annual; developmental trajectory; "
            "seizure classification update; neuroimaging every 2-3 years (cortical atrophy "
            "progression); transition planning from age 14 for MMPSI patients; AED review "
            "for seizure-free potential; cascade genetic testing for familial ADNFLE2."
        ),
    },
]

# ── LIFECYCLE ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Neonatal (day 0-28)",
        "events": "MMPSI-like EEG; continuous migrating focal seizures; ohtahara-like burst-suppression (AR biallelic)",
        "key_action": "Broad epilepsy panel STAT; avoid CBZ/OXC/PHT; phenobarbital bridge; confirm GoF before quinidine",
    },
    {
        "stage": "Early Infantile (1-3 months)",
        "events": "MMPSI established; high seizure burden; migrating ictal EEG",
        "key_action": "Quinidine initiation if GoF confirmed + QTc safe; KD consideration; POLG screen before VPA",
    },
    {
        "stage": "Late Infantile (3-12 months)",
        "events": "MMPSI evolution; IS component in some; progressive cortical atrophy possible",
        "key_action": "ACTH+VGB if IS component (UKISS); KD efficacy review; quinidine response assessment",
    },
    {
        "stage": "Toddler (1-3 years)",
        "events": "Ongoing drug-resistant epilepsy; developmental arrest; potential LGS evolution",
        "key_action": "EEG + AED review; multidisciplinary developmental input; quinidine continuation decision",
    },
    {
        "stage": "School age / ADNFLE (4-20 years)",
        "events": "MMPSI survivors: intellectual disability + ongoing seizures; ADNFLE onset in this window",
        "key_action": "EHCP / educational plan; ADNFLE: sleep EEG; family cascade genetic testing (ADNFLE2)",
    },
    {
        "stage": "Adult",
        "events": "ADNFLE persists; MMPSI survivors in adult care; comorbid psychiatric needs",
        "key_action": "Transition to adult neurology; ongoing QTc monitoring on quinidine; reproductive counselling",
    },
]

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "metric": "QTc on Quinidine",
        "normal": "<430 ms",
        "alert_value": "430-450 ms (increased monitoring)",
        "action": "Do not escalate dose; repeat ECG; check electrolytes (K+, Mg2+)",
        "critical_value": ">450 ms — WITHHOLD quinidine immediately; urgent cardiology review",
    },
    {
        "metric": "Quinidine Plasma Level",
        "normal": "2-5 ug/mL (therapeutic)",
        "alert_value": "<2 ug/mL (subtherapeutic) or >5 ug/mL (toxic risk)",
        "action": "Adjust dose; re-check level after 5 days (steady state)",
        "critical_value": ">6 ug/mL (cinchonism risk: tinnitus, headache, visual disturbance, arrhythmia)",
    },
    {
        "metric": "KD Ketone Bodies (beta-OHB)",
        "normal": "<0.5 mmol/L (non-ketotic)",
        "alert_value": "<2 mmol/L (inadequate ketosis for seizure control)",
        "action": "Adjust KD ratio; reduce carbohydrate allowance; dietitian review",
        "critical_value": ">5 mmol/L (hyperketo — reduce ratio or increase carbohydrate)",
    },
    {
        "metric": "Plasma Amino Acids (DDx screen)",
        "normal": "Normal range — glutamate <100 umol/L; glycine <400 umol/L",
        "alert_value": "Elevated glutamate or glycine (DDx trigger — unexpected in KCNT1)",
        "action": "Elevated glutamate → SLC25A22 DDx; elevated glycine → NKH DDx; both NORMAL in KCNT1",
        "critical_value": "Glutamate >200 umol/L → reconsider KCNT1 diagnosis; SLC25A22 more likely",
    },
    {
        "metric": "Seizure Burden (MMPSI ictal events/day)",
        "normal": "Goal: >=50% reduction from baseline",
        "alert_value": "No change or worsening at 4 weeks on quinidine",
        "action": "Reconsider quinidine dose; check plasma level; review GoF variant confirmation",
        "critical_value": "Acute worsening within 48h of any new AED — suspect Na-channel blocker given inadvertently",
    },
]

# ── DEFINITIONS ───────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "KCNT1 / SLACK / KNa1.1 / SLO2.2",
        "definition": (
            "KCNT1 (9q34.3) encodes the sodium-activated potassium channel KNa1.1 (also called "
            "SLACK — Sequence Like A Calcium-activated K+ channel — or SLO2.2). 1237 amino acids. "
            "Large-conductance K+ channel activated by intracellular Na+ (not voltage primarily). "
            "OMIM Gene 608042 / Disease DEE14/MMPSI 614959 / ADNFLE2 615005."
        ),
    },
    {
        "term": "Malignant Migrating Partial Seizures of Infancy (MMPSI) / DEE14",
        "definition": (
            "MMPSI: Continuous migration of ictal discharges from one brain region to another "
            "across both hemispheres — characteristic EEG pattern of KCNT1 DEE14. EEG shows "
            "multifocal onset seizures migrating from frontal to occipital to temporal regions "
            "continuously. Onset in the first 6 months of life. High seizure burden (>50/day). "
            "Drug-resistant. Profound neurodevelopmental impairment. Named 'malignant' for its "
            "highly refractory course. OMIM 614959."
        ),
    },
    {
        "term": "GoF (Gain-of-Function) Mutation Mechanism in KCNT1",
        "definition": (
            "GoF KCNT1 mutations lower the Na+ activation threshold of KNa1.1 → channel opens "
            "at resting intracellular Na+ concentrations → excessive constitutive K+ efflux. "
            "Paradoxical excitability results via: (1) disruption of FMRP-mediated presynaptic "
            "glutamate release modulation → uncontrolled glutamate excitation; (2) glial K+ "
            "buffering dysregulation → elevated extracellular [K+]; (3) network synchrony "
            "entrainment → MMPSI ictal migration pattern."
        ),
    },
    {
        "term": "KNa1.1 Channel Structure (1237 aa, 9q34.3)",
        "definition": (
            "N-terminal (aa 1-40) → S1-S4 voltage-sensing-like domain (aa 41-180) → S5-pore-S6 "
            "domain (aa 181-310, selectivity filter GYG motif) → large C-terminal domain (aa "
            "311-1237): RCK1 (aa 340-490, Na+ binding + NAD+ binding, GoF mutation hotspot) "
            "→ linker (aa 491-694) → RCK2 (aa 695-840, Na+ binding, NADP+ binding) → C-tail "
            "(aa 841-1237, FMRP interaction site). RCK1+RCK2 form octameric gating ring in "
            "open state. Tetramer in membrane."
        ),
    },
    {
        "term": "Quinidine Open-Channel Block of KCNT1",
        "definition": (
            "Quinidine is a Class Ia antiarrhythmic that enters the open KNa1.1 pore and "
            "physically occludes K+ conductance. ONLY effective when channel is in GoF-open "
            "state. Requires open-pore access → state-dependent block. Mechanism distinct "
            "from general K+ channel blockers. Quinidine plasma level target 2-5 ug/mL. "
            "QTc monitoring mandatory (quinidine also prolongs QTc via hERG channel "
            "inhibition). Responder rate ~30-40% in GoF MMPSI cohorts."
        ),
    },
    {
        "term": "ADNFLE2 (Autosomal Dominant Nocturnal Frontal Lobe Epilepsy type 2)",
        "definition": (
            "OMIM 615005. GoF KCNT1 mutations causing familial autosomal dominant nocturnal "
            "frontal lobe epilepsy. Childhood onset. Hypermotor nocturnal seizures from NREM "
            "sleep (tonic posturing, cycling, vocalisation). Milder cognitive trajectory than "
            "MMPSI. Autosomal dominant — 50% offspring risk. Cascade family genetic testing "
            "essential. Quinidine may help in refractory ADNFLE2."
        ),
    },
    {
        "term": "RCK Domain (Regulator of K+ Conductance)",
        "definition": (
            "RCK domains are cytoplasmic regulatory modules in large-conductance K+ channels "
            "(KCa, KNa) that act as Na+ or Ca2+ sensors to gate the pore. KCNT1 has two RCK "
            "domains: RCK1 (aa 340-490) as primary Na+ sensor (GoF mutations cluster here — "
            "R428Q, A913V, Y796H); RCK2 (aa 695-840) as secondary sensor and NADP+ binding "
            "site. RCK1+RCK2 form an octameric gating ring (two tetramers) in open state — "
            "expanding ring pulls S6 gates open."
        ),
    },
    {
        "term": "Na+-Activated K+ Channel (KNa) — Physiology",
        "definition": (
            "Class of K+ channels activated by intracellular Na+ rather than voltage or Ca2+. "
            "KNa1.1 (KCNT1/SLACK) and KNa1.2 (KCNT2/SLICK) are the two human KNa channels. "
            "Physiological role: sense Na+ load during action potential bursts → K+ efflux → "
            "membrane adaptation (after-hyperpolarisation). Expressed in cortical and hippocampal "
            "neurons, brainstem, spinal cord, glia. GoF: constitutive activation at resting Na+."
        ),
    },
    {
        "term": "Drug-Resistant Epilepsy / DEE14",
        "definition": (
            "DEE14 (Developmental and Epileptic Encephalopathy 14) is defined by KCNT1 GoF "
            "mutations. 'Drug-resistant' per ILAE 2010 definition: failure of >=2 tolerated, "
            "appropriately chosen and used AED schedules (monotherapy or polytherapy). In KCNT1 "
            "MMPSI, typically >3-5 AEDs fail before any partial response. Drug resistance is "
            "intrinsic to the channelopathy mechanism — requires mechanism-targeted treatment "
            "(quinidine) rather than empirical AED escalation."
        ),
    },
    {
        "term": "QTc Monitoring (Quinidine Safety)",
        "definition": (
            "Quinidine prolongs cardiac QTc interval via hERG (IKr) K+ channel inhibition in "
            "cardiomyocytes. QTc >450 ms → risk of Torsades de Pointes (TdP) ventricular "
            "arrhythmia. Pre-quinidine: 12-lead ECG; electrolytes (hypoK+ + hypoMg2+ "
            "increase TdP risk). QTc measured at baseline, every dose change, then monthly. "
            "HOLD quinidine if QTc >450 ms or >60 ms rise from baseline. Urgent cardiology "
            "if QTc >480 ms or arrhythmia symptoms."
        ),
    },
    {
        "term": "Migrating Ictal EEG Pattern (MMPSI signature)",
        "definition": (
            "Characteristic EEG of MMPSI/DEE14: continuous multifocal ictal discharges that "
            "sequentially 'migrate' from one hemisphere region to another — from frontal to "
            "temporal to occipital, crossing hemispheres. Clinically: multiple seizure types "
            "with different semiology in one session reflecting different cortical involvement "
            "zones. Key DDx: ATP1A3 AHC has hemiplegic episodes (not ictal EEG migration); "
            "SCN8A has tonic/clonic but not migrating; SLC25A22 may mimic but glutamate elevated."
        ),
    },
    {
        "term": "Phenocopy MMPSI (SCN1A / SCN8A / ATP1A3 DDx)",
        "definition": (
            "MMPSI-like clinical/EEG phenotype without KCNT1 variant. Key DDx: "
            "SCN1A (febrile/temperature-sensitive triggers; SMEI EEG not migrating; stiripentol "
            "+ VPA + clobazam; most common DEE gene); SCN8A (rapid progression; PHT/CBZ can "
            "help — opposite of KCNT1); ATP1A3 AHC (alternating hemiplegic episodes triggered "
            "by fever/stress — NOT ictal; flunarizine Rx; ATPase not K+ channel); TBC1D24 "
            "(DOOR syndrome; MMPSI-like + sensorineural deafness). Broad panel mandatory."
        ),
    },
    {
        "term": "KD Mechanism in KCNT1 DEE14",
        "definition": (
            "Ketogenic Diet (KD) reduces glycolytic neuronal metabolism → less acetyl-CoA "
            "from glucose → less glutamate pool excitatory neurotransmitter substrate. "
            "Ketone bodies (beta-OHB, acetoacetate) provide alternative fuel via direct TCA "
            "entry as acetyl-CoA, bypassing cytoplasmic glutamate generation. Additionally, "
            "KD reduces overall network excitability via adenosine (A1 receptor), KATP "
            "channel opening, and reduced ROS. Level B evidence (observational) in KCNT1 "
            "MMPSI. Not curative but contributes to seizure reduction in combination therapy."
        ),
    },
    {
        "term": "GoF vs LoF Distinction in KCNT1",
        "definition": (
            "GoF (gain-of-function): mutations in RCK1/RCK2 domain that lower Na+ activation "
            "threshold → constitutive channel opening → epilepsy (MMPSI/DEE14, ADNFLE2). "
            "TREATMENT: quinidine (open-channel block) is ONLY appropriate for GoF. "
            "LoF (loss-of-function): would impair adaptation current → different phenotype "
            "(intellectual disability, not primarily epilepsy). LoF is NOT the epilepsy-causing "
            "mechanism in KCNT1. Functional assessment of each variant is mandatory before "
            "quinidine initiation — GoF status cannot be assumed from variant type alone."
        ),
    },
    {
        "term": "POLG Mandatory Screen Before VPA",
        "definition": (
            "POLG (polymerase gamma) is the mitochondrial DNA polymerase. Biallelic POLG "
            "mutations → Alpers-Huttenlocher syndrome: fatal hepatic failure with VPA. "
            "Mandatory POLG sequencing before any VPA prescription in ALL developmental and "
            "epileptic encephalopathy patients. In KCNT1, VPA is not mechanistically "
            "contraindicated (no mitochondrial pathway involvement) but universal DEE "
            "POLG-before-VPA protocol must be followed. Baseline liver function tests also "
            "mandatory before VPA initiation."
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
    return round(sum(vals) / len(vals), 1)


# ── API FUNCTIONS ─────────────────────────────────────────────────────────────
def get_overview():
    pts = PATIENTS
    n = len(pts)
    etiol_dist = []
    for ec in ETIOLOGY_CATALOG:
        cat_pts = [p for p in pts if p["category"] == ec["category"]]
        etiol_dist.append({
            "etiology": ec["category"].replace("KCNT1-", "").replace("-", " "),
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
            "action": m["action"][:85] + "..." if len(m["action"]) > 85 else m["action"],
        }
        for m in MONITORING[:5]
    ]
    return {
        "gene": "KCNT1",
        "chromosome": "9q34.3",
        "omim_gene": "608042",
        "omim_disease": "614959",
        "omim_disease_adnfle": "615005",
        "protein": "KNa1.1 / SLACK / SLO2.2 — Na+-Activated K+ Channel",
        "aa_length": 1237,
        "domains": (
            "N-terminal (aa 1-40) + S1-S4 voltage-sensing-like (aa 41-180) + "
            "S5-pore-S6 (aa 181-310, GYG selectivity filter) + "
            "RCK1 (aa 340-490, Na+ binding, GoF hotspot) + "
            "RCK2 (aa 695-840, NADP+ binding) + C-tail (aa 841-1237, FMRP interaction)"
        ),
        "inheritance": "AD de novo GoF (MMPSI/DEE14) + AD familial GoF (ADNFLE2) + AR biallelic GoF (rare)",
        "disease_spectrum": "DEE14 / MMPSI (GoF de novo) → ADNFLE2 (GoF familial) → Severe DEE (GoF AR biallelic)",
        "unique_feature": (
            "No metabolic biomarker (plasma amino acids NORMAL — unlike SLC25A22). "
            "Migrating ictal EEG pattern is the hallmark. Quinidine is GoF-SPECIFIC — "
            "QTc monitoring mandatory. CBZ/OXC/PHT/LTG ABSOLUTE CI in MMPSI. "
            "POLG mandatory before VPA."
        ),
        "cohort_seed": 509,
        "kpis": {
            "n_patients": n,
            "mmpsi_pct": _pct(pts, "mmpsi"),
            "adnfle_pct": _pct(pts, "adnfle"),
            "drug_resistant_pct": _pct(pts, "drug_resistant"),
            "quinidine_tried_pct": _pct(pts, "quinidine_tried"),
            "quinidine_responded_pct": _pct(pts, "quinidine_responded"),
            "migrating_eeg_pct": _pct(pts, "eeg_migrating_pattern"),
            "profound_id_pct": _pct(pts, "profound_id"),
            "any_id_pct": _pct(pts, "any_id"),
            "acth_vgb_pct": _pct(pts, "acth_vgb_given"),
            "kd_pct": _pct(pts, "kd_tried"),
            "mean_aeds_failed": _mean(pts, "n_aeds_failed"),
            "seizure_free_pct": _pct(pts, "seizure_free"),
        },
        "etiology_distribution": etiol_dist,
        "treatments_summary": treat_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS,
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
    }


def get_breakdown():
    pts = PATIENTS
    by_cat = {}
    for p in pts:
        c = p["category"].replace("KCNT1-", "").replace("-", " ")
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(p)

    breakdown = []
    for cat, cat_pts in by_cat.items():
        breakdown.append({
            "category": cat,
            "n": len(cat_pts),
            "mmpsi_pct": _pct(cat_pts, "mmpsi"),
            "adnfle_pct": _pct(cat_pts, "adnfle"),
            "ohtahara_like_pct": _pct(cat_pts, "ohtahara_like"),
            "burst_suppression_pct": _pct(cat_pts, "burst_suppression"),
            "migrating_eeg_pct": _pct(cat_pts, "eeg_migrating_pattern"),
            "hypsarrhythmia_pct": _pct(cat_pts, "hypsarrhythmia"),
            "drug_resistant_pct": _pct(cat_pts, "drug_resistant"),
            "mean_aeds_failed": _mean(cat_pts, "n_aeds_failed"),
            "quinidine_tried_pct": _pct(cat_pts, "quinidine_tried"),
            "quinidine_responded_pct": _pct(cat_pts, "quinidine_responded"),
            "profound_id_pct": _pct(cat_pts, "profound_id"),
            "any_id_pct": _pct(cat_pts, "any_id"),
            "acth_vgb_pct": _pct(cat_pts, "acth_vgb_given"),
            "kd_pct": _pct(cat_pts, "kd_tried"),
            "seizure_free_pct": _pct(cat_pts, "seizure_free"),
        })

    etiol_details = [
        {
            "category": ec["category"].replace("KCNT1-", "").replace("-", " "),
            "typical_variant": ec["typical_variant"],
            "inheritance": ec["inheritance"],
            "functional_deficit": ec["functional_deficit"],
            "description": ec["description"],
        }
        for ec in ETIOLOGY_CATALOG
    ]

    summary = {
        "mmpsi_pct": _pct(pts, "mmpsi"),
        "adnfle_pct": _pct(pts, "adnfle"),
        "drug_resistant_pct": _pct(pts, "drug_resistant"),
        "quinidine_tried_pct": _pct(pts, "quinidine_tried"),
        "quinidine_responded_pct": _pct(pts, "quinidine_responded"),
        "migrating_eeg_pct": _pct(pts, "eeg_migrating_pattern"),
        "profound_id_pct": _pct(pts, "profound_id"),
        "any_id_pct": _pct(pts, "any_id"),
        "acth_vgb_pct": _pct(pts, "acth_vgb_given"),
        "kd_pct": _pct(pts, "kd_tried"),
        "mean_aeds_failed": _mean(pts, "n_aeds_failed"),
        "seizure_free_pct": _pct(pts, "seizure_free"),
        "polg_tested_pct": _pct(pts, "polg_tested"),
        "qtc_monitored_pct": _pct(pts, "qtc_monitored"),
    }

    return {
        "gene": "KCNT1",
        "chromosome": "9q34.3",
        "cohort_size": len(pts),
        "cohort_seed": 509,
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
        "gene": "KCNT1",
        "chromosome": "9q34.3",
        "protein": "KNa1.1 / SLACK / SLO2.2 — Na+-Activated K+ Channel (1237 aa)",
        "omim_gene": "608042",
        "omim_disease": "614959",
        "omim_disease_adnfle": "615005",
        "disease_name": "DEE14 — Developmental and Epileptic Encephalopathy 14 / MMPSI / ADNFLE2",
        "inheritance": "AD de novo GoF (MMPSI) + AD familial GoF (ADNFLE2) + AR biallelic GoF (rare); strictly GoF for epilepsy",
        "definitions": DEFINITIONS,
        "key_ddx": [
            "SCN1A Dravet (9q34.3): febrile/temperature-sensitive seizures; SMEI EEG not migrating; "
            "stiripentol + VPA + clobazam (Dravet-specific); most common DEE gene; amino acids NORMAL",
            "SLC25A22 DEE3 (11p15.5): elevated plasma glutamate >200 umol/L (pathognomonic — absent in KCNT1); "
            "bilateral BG/thalamic MRI; AR biallelic LOF (not GoF); pyridoxine trial mandatory; no quinidine",
            "ATP1A3 AHC (19q13.2): Na+/K+-ATPase alpha-3; alternating hemiplegia (triggered hemiplegic episodes, "
            "NOT ictal migration); flunarizine Rx; no migrating EEG seizure pattern; ATPase not K+ channel",
            "SCN8A DEE (12q13.13): rapid progression; tonic/clonic not MMPSI-migrating; PHT/CBZ CAN help "
            "(opposite of KCNT1); GoF SCN8A — quinidine NOT indicated; NaV1.6 not KNa1.1",
            "TBC1D24 DOOR (16p13.3): MMPSI-like + sensorineural deafness + onychodystrophy; AR; "
            "RAB GTPase signalling; no quinidine; KCNT1 panel should always include TBC1D24",
            "NKH/GLDC (9p24.1): elevated plasma GLYCINE (not glutamate); CSF:plasma glycine >0.08; "
            "hiccups pathognomonic; sodium benzoate Rx; KCNT1 MMPSI = glycine NORMAL",
        ],
        "mandatory_workup": [
            "Plasma amino acids (FASTING): should be NORMAL in KCNT1 — elevated glutamate → SLC25A22 DDx",
            "CSF amino acids: glycine ratio (NKH exclusion); glutamate (SLC25A22 DDx)",
            "KCNT1 sequencing + broad DEE panel: SCN1A/SCN8A/SCN2A/ATP1A3/TBC1D24/SLC25A22 simultaneous",
            "GoF functional assessment for KCNT1 variant: Xenopus oocyte or HEK293 electrophysiology MANDATORY before quinidine",
            "ECG (12-lead): QTc baseline BEFORE quinidine initiation; electrolytes (K+, Mg2+)",
            "Quinidine plasma level at steady state (day 5): target 2-5 ug/mL",
            "EEG STAT: migrating ictal pattern characterisation (diagnostic for MMPSI); video-EEG for semiology",
            "MRI brain (3T): initial normal → progressive cortical atrophy on serial imaging; check BG changes (DDx)",
            "POLG sequencing MANDATORY before any VPA consideration",
            "Biotinidase activity and plasma lactate/ammonia (mitochondrial DDx broad screen)",
            "VGB ophthalmology baseline if VGB used for IS component (REMS)",
            "Cascade genetic testing: de novo (low sibling recurrence <1%); ADNFLE2 family (50% offspring risk); AR biallelic (25%)",
        ],
        "standards": [
            "OMIM 614959 (DEE14 / MMPSI) — KCNT1",
            "OMIM 615005 (ADNFLE2) — KCNT1",
            "Barcia et al. (2012) Nat Genet 44:1255-1259 (original KCNT1 MMPSI paper)",
            "Milligan et al. (2014) Ann Neurol 76:826-834 (KCNT1 in 2 epilepsy phenotypes)",
            "Rizzo et al. (2016) Neurology 86:1063-1071 (quinidine KCNT1 efficacy)",
            "ILAE Gene Classification (2022): KCNT1 — definitive DEE14 gene",
            "UKISS protocol (ACTH + VGB for infantile spasms component)",
            "VGB REMS programme (visual field monitoring; max 16 weeks IS use)",
            "POLG Working Group guidelines (pre-VPA POLG screening)",
            "ClinGen KCNT1 variant curation (GoF vs LoF classification mandatory before quinidine)",
        ],
        "five_key_facts": [
            "KCNT1 MMPSI has NO metabolic biomarker — plasma amino acids (glutamate, glycine) are NORMAL "
            "(key distinction from SLC25A22 [elevated glutamate] and NKH [elevated glycine]); the diagnosis "
            "rests on EEG migrating ictal pattern + KCNT1 GoF genetic confirmation",
            "Quinidine is KCNT1 GoF-SPECIFIC (open-channel block of KNa1.1) — NEVER prescribe without "
            "confirmed GoF functional status; QTc monitoring mandatory (risk of Torsades de Pointes); "
            "hold quinidine if QTc >450 ms",
            "CBZ/OXC/PHT/LTG are ABSOLUTELY CONTRAINDICATED in KCNT1 MMPSI — Na-channel blockers "
            "worsen the migrating pattern (documented case reports of acute seizure escalation); "
            "key DDx discriminator: CBZ IS HELPFUL in KCNQ2, HARMFUL in KCNT1",
            "Three distinct phenotypes: GoF de novo = MMPSI/DEE14 (severe, neonatal, drug-resistant); "
            "GoF AD familial = ADNFLE2 (childhood nocturnal hypermotor, milder course); "
            "GoF AR biallelic = Ohtahara-like severe DEE (very rare, neonatal, most severe)",
            "POLG sequencing is MANDATORY before any VPA in all DEE patients; cascade genetic testing "
            "essential: de novo MMPSI → <1% sibling recurrence; ADNFLE2 familial → 50% offspring risk; "
            "AR biallelic → 25% sibling recurrence risk",
        ],
    }
