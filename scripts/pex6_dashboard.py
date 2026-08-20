#!/usr/bin/env python3
"""PEX6 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX6 encodes Peroxin-6,
a AAA-ATPase (980 aa, two AAA domains D1+D2); forms a heterodimer with PEX1 (the other
AAA-ATPase subunit) and is anchored to the peroxisomal membrane by PEX26 (APEM9/Pex17p).
The PEX1–PEX6 complex uses ATP hydrolysis (PEX6 D2 domain) to retrotranslocate PEX5
(cytosolic PTS1-receptor) from the peroxisomal membrane back to cytosol for recycling.
PEX6 LOF → PEX5 trapped/ubiquitinated → peroxisomal matrix import fails → ALL peroxisomal
metabolic functions simultaneously impaired (biochemically identical to PEX1 deficiency).

LOCUS: 6p21.1  |  OMIM GENE: *601498  |  OMIM DISEASE: #614870 (ZSD-PEX6)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX6 mutations account for ~10% of all
  PBD-ZSD (second most common PBD gene after PEX1 ~65%). PEX1 + PEX6 together account for
  ~75% of all ZSD. Common PEX6 variants: p.Arg860Trp (R860W) — hypomorphic allele in D2
  domain, ~18% of all PEX6 alleles (European); p.Leu504Pro (L504P); frameshift/LOF alleles.
  R860W/R860W → attenuated IRD-like phenotype; R860W/null → NALD-like; null/null → ZS (severe).

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1):
  PEX6 deficiency causes the SAME biochemical phenotype as PEX1 because both disrupt
  the PEX1–PEX6 AAA-ATPase complex (loss of either subunit abolishes PEX5 retrotranslocation):
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic + pristanic acid elevated
  (3) Plasmalogens BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in PBD-ZSD]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated in plasma/urine
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid
  DISTINGUISHING PEX6 FROM PEX1 BIOCHEMICALLY: identical — only molecular genetics distinguishes.

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — SAME AS PEX1 BUT LESS SEVERE ON AVERAGE):
  PEX6 cohorts show slightly more attenuated phenotype overall vs PEX1 (more IRD, less ZS)
  because R860W hypomorphic allele is common and produces more residual activity than PEX1 G843D.
  (1) Zellweger Syndrome (ZS, severe): null/null genotype; neonatal seizures 100%; pachygyria/PMG
      pathognomonic; hypotonia; hepatomegaly; retinopathy; adrenal insufficiency; death <12M.
  (2) NALD (intermediate): R860W/null; infantile spasms 35–55%; cholestasis; survival months–years.
  (3) IRD (attenuated): R860W/R860W; RP + SNHL + neuropathy; seizures 25–40%; adult survival.

EPILEPSY IN PEX6-ZSD:
  ZS: 100% epilepsy; neonatal seizures day 1–7; multifocal clonic + tonic + myoclonic; burst-
  suppression EEG; drug-resistant; fatal <12M.
  NALD: 70–85% epilepsy; infantile spasms 35–55%; hypsarrhythmia; focal + generalized myoclonic.
  IRD: 25–40% epilepsy (less than PEX1-IRD because R860W provides more residual PEX5 recycling);
  focal seizures + GTCS; better AED response; phytol-restricted diet helps.
  DRE: 55–65% of all PEX6-ZSD (slightly less than PEX1 due to more attenuated IRD cohort).

AED PHARMACOLOGY (IDENTICAL TO PEX1-ZSD):
  LEV: FIRST-LINE all ZSD forms. No enzyme induction, no hepatotoxicity, IV/PO neonatal.
  VPA: HIGH RISK — hepatotoxicity (baseline cholestatic liver disease) + peroxisomal beta-oxidation
       inhibition + carnitine depletion. POLG1 MANDATORY before VPA (CPIC Grade A).
  VGB: HIGH RISK — additive irreversible visual loss (pre-existing retinopathy universal ZS/NALD).
       ACTH Level A for IS preferred over VGB. AVOID VGB in IRD (preservable vision).
  PHT/CBZ/OXC: RELATIVE CI — CYP3A4 enzyme induction → DHA depletion + hepatic burden.
  ACTH: Level A for infantile spasms/hypsarrhythmia (NALD + ZS if surviving infantile period).
  CLB: Level B adjunct; no hepatotoxicity; no enzyme induction.
  LTG: Level C adjunct; hepatic glucuronidation (NOT CYP); safer hepatic profile than CBZ.
  DHA supplementation: Level B (NALD/IRD). No proven disease-modifying therapy; no ERT; no HSCT.
  Fasting: EXTREME HAZARD (phytanic acid surge from adipose); IV dextrose MANDATORY pre-surgical.
"""

import random

random.seed(876)  # reproducible


# ── Overview ──────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 82,
        "zs_pct": 18,
        "nald_pct": 32,
        "ird_pct": 45,
        "atypical_pct": 5,
        "drug_resistance_pct": 58,
        "on_dha_pct": 55,
        "liver_disease_pct": 42,
        "retinopathy_pct": 78,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "dha_low_pct": 100,
        "omim_gene": "601498",
        "omim_disease": "#614870",
        "locus": "6p21.1",
        "inheritance": "Autosomal Recessive (AR), biallelic LOF",
        "nbs_positive_rate": "~1/80,000 births (C26:0-lyso-PC DBS)",
        "common_variant": "p.Arg860Trp (R860W) — 18% of all PEX6 alleles (European hypomorphic)",
        "disease_mechanism": (
            "PEX6 encodes a AAA-ATPase (980 aa) that heterodimerizes with PEX1; both are "
            "anchored to the peroxisomal membrane by PEX26. The PEX1–PEX6 complex uses ATP "
            "hydrolysis (primarily PEX6 D2 domain) to retrotranslocate PEX5 (PTS1-receptor) "
            "from the peroxisomal membrane back to the cytosol for recycling. PEX6 LOF traps "
            "PEX5 in the membrane (ubiquitinated), abolishing peroxisomal matrix protein import. "
            "ALL peroxisomal pathways fail simultaneously: VLCFA beta-oxidation, alpha-oxidation, "
            "plasmalogen biosynthesis, DHA synthesis, pipecolic acid catabolism, and bile acid "
            "synthesis. Biochemically indistinguishable from PEX1 deficiency — only gene panel "
            "distinguishes PEX1-ZSD from PEX6-ZSD."
        ),
        "key_concepts": [
            "PEX6 = AAA-ATPase D2 motor subunit of PEX1–PEX6 complex; loss of PEX6 ≡ loss of PEX1 biochemically",
            "p.Arg860Trp (R860W) in D2 AAA domain = most common European PEX6 hypomorphic allele (~18% of alleles)",
            "R860W/R860W → IRD (attenuated); R860W/null → NALD-like; null/null → ZS (severe)",
            "Genotype PARTIALLY predictive (like PEX1): R860W hypomorphic vs null determines severity",
            "PEX6 cohorts show more IRD (45%) vs PEX1 cohorts (30–35%) because R860W more residual activity than PEX1-G843D",
            "Plasmalogens (RBC) LOW — KEY DISTINCTION from ABCD1 (plasmalogens NORMAL in ABCD1)",
            "VLCFA + phytanic + pristanic + pipecolic + DHA ALL abnormal (6 pathways) — unlike ABCD1 (VLCFA only)",
            "DHA low → neuronal migration defects → pachygyria/PMG (ZS PATHOGNOMONIC on MRI)",
            "LEV first-line all ZSD forms (no enzyme induction, no hepatotoxicity, IV/PO, neonatal-safe)",
            "VPA HIGH RISK: hepatotoxicity (cholestatic liver) + peroxisomal BO inhibition + carnitine depletion — POLG1 MANDATORY",
            "VGB HIGH RISK: additive irreversible VF loss (pre-existing retinopathy universal ZS/NALD) — ACTH Level A for IS",
            "ACTH Level A for IS (preferred over VGB due to retinopathy); DHA Level B (NALD/IRD)",
            "Fasting = EXTREME HAZARD: adipose phytanic acid surge → acute neurotoxicity; IV dextrose MANDATORY pre-surgical",
            "No ERT (enzyme replacement — PEX6 is a complex machinery component, not a soluble enzyme)",
            "No HSCT (unlike ABCD1/CCALD — no neuroinflammatory window to arrest in ZSD)",
            "Lorenzo's Oil: INEFFECTIVE (peroxisomal import absent — VLCFA cannot enter peroxisome even if synthesis reduced)",
        ],
        "standards": [
            "Braverman NE et al. Peroxisome biogenesis disorders. Ann NY Acad Sci. 2016",
            "Waterham HR, Ebberink MS. Genetics of peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Klouwer FCC et al. Zellweger spectrum disorders. Orphanet J Rare Dis. 2015",
            "Steinberg SJ et al. Peroxisome biogenesis disorders. Neurology. 2006",
            "Poll-The BT, Gärtner J. Clinical diagnosis, biochemical findings, management of peroxisomal diseases. Semin Pediatr Neurol. 2012",
            "Wanders RJ, Waterham HR. Biochemistry of mammalian peroxisomes. Annu Rev Biochem. 2006",
            "Ebberink MS et al. Genetic classification and mutational spectrum of PEX6 mutations. Hum Mutat. 2011",
            "CPIC VPA–POLG1 guideline. cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        ],
    }


# ── Breakdown (patients + etiologies + seizures + triggers + treatments + CIs) ─
def get_breakdown():
    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null",
            "pct": 18,
            "n": 7,
            "sex": "M/F 4/3",
            "onset_age": "Neonatal (day 1–5)",
            "seizure_risk": "100% (multifocal clonic + tonic + myoclonic)",
            "eeg": "Burst-suppression → multifocal spikes + hypsarrhythmia",
            "mri": "Pachygyria + PMG + periventricular cysts + hypomyelination (PATHOGNOMONIC)",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Compound heterozygous: null/null (two LOF alleles — frameshift, nonsense, splice); "
                "zero residual PEX6 activity; fatal <6–12 months; supportive/palliative only"
            ),
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — R860W/null",
            "pct": 32,
            "n": 13,
            "sex": "M/F 7/6",
            "onset_age": "1st–6th month",
            "seizure_risk": "70–85% (infantile spasms 35–55%; focal + myoclonic)",
            "eeg": "Hypsarrhythmia 35–55% + multifocal spikes + slowing",
            "mri": "Dysmyelination + periventricular WM abnormalities + cerebellar hypoplasia",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "p.Arg860Trp (R860W) / null LOF — R860W D2 domain hypomorphic; partial residual "
                "PEX5 retrotranslocation; survival months–years; DHA supplementation Level B; "
                "no proven disease-modifying therapy"
            ),
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — R860W/R860W",
            "pct": 45,
            "n": 18,
            "sex": "M/F 10/8",
            "onset_age": "Toddler–school age (2–8 years)",
            "seizure_risk": "25–40% (focal + GTCS; better response than ZS/NALD)",
            "eeg": "Focal spikes + generalized slowing; no hypsarrhythmia",
            "mri": "Mild WM abnormalities; cerebellar atrophy (later); cortical migration normal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "p.Arg860Trp (R860W) / p.Arg860Trp (R860W) homozygous — most attenuated; "
                "RP + SNHL + peripheral neuropathy; hepatomegaly mild; intellectual disability "
                "variable; survival adolescence–adulthood; phytol-restricted diet + DHA"
            ),
        },
        {
            "name": "Atypical / Late-Onset ZSD (PEX6)",
            "pct": 5,
            "n": 2,
            "sex": "M/F 1/1",
            "onset_age": "Childhood–early adulthood",
            "seizure_risk": "20–35% (focal only; rare GTCS)",
            "eeg": "Focal spikes temporal/occipital",
            "mri": "Isolated cerebellar atrophy or normal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Hypomorphic compound heterozygous (e.g., R860W + missense with partial activity); "
                "very attenuated; presents as isolated RP, SNHL, or cerebellar ataxia in adulthood; "
                "seizures mild if present; excellent AED response"
            ),
        },
    ]

    # ── 40 synthetic patients ──
    phenotypes = (
        ["ZS (Zellweger-Severe) — null/null"] * 7
        + ["NALD (Neonatal-ALD-Intermediate) — R860W/null"] * 13
        + ["IRD (Infantile-Refsum-Attenuated) — R860W/R860W"] * 18
        + ["Atypical / Late-Onset ZSD (PEX6)"] * 2
    )
    random.shuffle(phenotypes)
    genotypes = {
        "ZS (Zellweger-Severe) — null/null": [
            "c.2528_2531del/c.1802G>A (p.Trp601*)",
            "c.802C>T (p.Arg268*)/IVS14+1G>T",
            "c.2096del/c.1576C>T (p.Gln526*)",
            "c.1234+2T>C/c.2528_2531del",
        ],
        "NALD (Neonatal-ALD-Intermediate) — R860W/null": [
            "c.2578C>T (p.Arg860Trp)/c.802C>T (p.Arg268*)",
            "c.2578C>T (p.Arg860Trp)/IVS14+1G>T",
            "c.2578C>T (p.Arg860Trp)/c.2096del",
            "c.2578C>T (p.Arg860Trp)/c.1234+2T>C",
        ],
        "IRD (Infantile-Refsum-Attenuated) — R860W/R860W": [
            "c.2578C>T (p.Arg860Trp)/c.2578C>T (p.Arg860Trp)",
        ],
        "Atypical / Late-Onset ZSD (PEX6)": [
            "c.2578C>T (p.Arg860Trp)/c.1511T>C (p.Leu504Pro)",
        ],
    }
    aeds_by_phenotype = {
        "ZS (Zellweger-Severe) — null/null": ["LEV", "PB", "CLZ", None],
        "NALD (Neonatal-ALD-Intermediate) — R860W/null": ["LEV", "LEV+CLB", "LEV+ACTH", "LEV+LTG", None],
        "IRD (Infantile-Refsum-Attenuated) — R860W/R860W": ["LEV", "LEV+LTG", "LEV+CLB", "LTG", None],
        "Atypical / Late-Onset ZSD (PEX6)": ["LEV", "LTG"],
    }
    response_by_phenotype = {
        "ZS (Zellweger-Severe) — null/null": ["Drug-resistant"] * 6 + ["Partially controlled"] * 1,
        "NALD (Neonatal-ALD-Intermediate) — R860W/null": (
            ["Drug-resistant"] * 5 + ["Partially controlled"] * 5 + ["Controlled"] * 3
        ),
        "IRD (Infantile-Refsum-Attenuated) — R860W/R860W": (
            ["Controlled"] * 9 + ["Partially controlled"] * 6 + ["Drug-resistant"] * 3
        ),
        "Atypical / Late-Onset ZSD (PEX6)": ["Controlled", "Controlled"],
    }
    response_pool = {k: list(v) for k, v in response_by_phenotype.items()}
    random.shuffle(response_pool["ZS (Zellweger-Severe) — null/null"])
    random.shuffle(response_pool["NALD (Neonatal-ALD-Intermediate) — R860W/null"])
    random.shuffle(response_pool["IRD (Infantile-Refsum-Attenuated) — R860W/R860W"])

    counters = {p: 0 for p in set(phenotypes)}
    patients = []
    for i, pheno in enumerate(phenotypes):
        idx = counters[pheno]
        counters[pheno] += 1
        geno_list = genotypes[pheno]
        geno = geno_list[idx % len(geno_list)]
        aed_pool = aeds_by_phenotype[pheno]
        aed = aed_pool[idx % len(aed_pool)]
        resp_list = response_pool.get(pheno, ["Controlled"])
        resp = resp_list[idx % len(resp_list)]
        has_sz = pheno != "IRD (Infantile-Refsum-Attenuated) — R860W/R860W" or resp != "Controlled" or random.random() < 0.3
        liver = pheno in ("ZS (Zellweger-Severe) — null/null", "NALD (Neonatal-ALD-Intermediate) — R860W/null") or (pheno == "IRD (Infantile-Refsum-Attenuated) — R860W/R860W" and idx % 4 == 0)
        on_dha = pheno != "ZS (Zellweger-Severe) — null/null"
        patients.append({
            "patient_id": f"PEX6-{i+1:03d}",
            "phenotype": pheno.split("(")[0].strip().split(" — ")[0],
            "sex": "M" if (i % 2 == 0) else "F",
            "genotype": geno,
            "liver_disease": liver,
            "has_seizures": has_sz,
            "primary_aed": aed,
            "drug_response": resp if has_sz else None,
            "on_dha": on_dha,
        })

    seizure_types = [
        {"type": "Neonatal seizures — multifocal clonic (ZS)", "pct": 100, "eeg": "Burst-suppression → multifocal spikes; electrographic seizures common"},
        {"type": "Infantile spasms / hypsarrhythmia (NALD)", "pct": 43, "eeg": "Hypsarrhythmia (modified) → ACTH Level A; VGB AVOID (retinopathy)"},
        {"type": "Tonic seizures (ZS + NALD)", "pct": 35, "eeg": "Diffuse fast activity + low-voltage EMG artifact"},
        {"type": "Myoclonic seizures (NALD + IRD)", "pct": 30, "eeg": "Generalized spike-wave 3–4 Hz + polyspike-wave"},
        {"type": "Focal seizures (IRD + Atypical)", "pct": 28, "eeg": "Focal spikes temporal / occipital (retinal dysfunction drives occipital focus)"},
        {"type": "GTCS (IRD)", "pct": 20, "eeg": "Generalized 3-Hz spike-wave → tonic phase"},
        {"type": "Status epilepticus (ZS + NALD)", "pct": 22, "eeg": "Continuous electrographic seizures; burst-suppression between clusters"},
    ]

    triggers = [
        {"trigger": "Febrile illness (universal)", "pct": 72, "note": "Fever → metabolic decompensation; DHA depletion worsens seizure threshold"},
        {"trigger": "Fasting / prolonged NPO", "pct": 68, "note": "EXTREME HAZARD: adipose phytanic acid surge → acute neurotoxicity; IV dextrose MANDATORY pre-surgical"},
        {"trigger": "Phytanic acid surge (dietary — IRD)", "pct": 55, "note": "Phytanic acid neuromodulatory; dietary phytol restriction reduces trigger load in IRD"},
        {"trigger": "Metabolic decompensation (hepatic)", "pct": 48, "note": "Cholestatic liver disease → reduced AED metabolism + worsened coagulopathy"},
        {"trigger": "Sleep deprivation (IRD + Atypical)", "pct": 32, "note": "Focal + GTCS in IRD; standard seizure precaution"},
        {"trigger": "Anaesthesia / peri-operative stress", "pct": 28, "note": "Fasting hazard + coagulopathy risk; PT/INR + LFT pre-op; IV K mandatory"},
        {"trigger": "Intercurrent VPA initiation", "pct": 18, "note": "VPA → peroxisomal BO inhibition + hepatotoxicity; triggers acute decompensation"},
    ]

    monitoring = [
        "VLCFA (C26:0, C24:0) + C26:0/C22:0 ratio: baseline then q6M (confirm diagnosis; track)",
        "Plasmalogens (RBC): baseline + q12M (LOW in all ZSD; key ABCD1 distinction)",
        "Phytanic + pristanic acid: q6M (dietary compliance in IRD; surge risk marker)",
        "DHA (plasma/RBC): q6M (supplementation adequacy in NALD/IRD)",
        "LFTs (ALT, AST, GGT, bilirubin, PT/INR, albumin): q2–3M (baseline cholestasis; AED hepatotoxicity)",
        "ERG + ophthalmology + visual fields: q12M from diagnosis (retinopathy universal ZS/NALD)",
        "Audiology (BAER): q12M (SNHL progressive in NALD/IRD)",
        "Brain MRI: at diagnosis; q12–24M if NALD/IRD (WM progression, cerebellar atrophy)",
        "Video-EEG: at seizure onset; q6–12M if drug-resistant or IS present",
        "Pipecolic acid (urine/plasma): baseline (confirmatory; elevated in all ZSD)",
        "Carnitine (free + total): q6M if VPA used (VPA-induced depletion in ZSD)",
        "POLG1 sequencing: MANDATORY before any VPA trial (CPIC Grade A recommendation)",
        "Cortisol (AM) + adrenal function: q12M (adrenal insufficiency in ZS — different from ABCD1: not dominant trigger in PEX6-ZSD)",
        "Developmental/neuropsychological: q12M (IRD/Atypical — intellectual disability variable)",
    ]

    thresholds = [
        {"parameter": "C26:0 (VLCFA)", "threshold": ">1.3 µmol/L plasma", "action": "Confirms peroxisomal disorder; start gene panel; ZSD vs ABCD1 distinction via plasmalogens"},
        {"parameter": "Plasmalogens RBC", "threshold": "<50% of control", "action": "Confirms ZSD (not ABCD1); LOW = PBD (PEX1/PEX6/PEX12/etc); NORMAL = ABCD1"},
        {"parameter": "DHA (plasma)", "threshold": "<50 µmol/L", "action": "Start DHA supplementation 100–500 mg/day (Level B — NALD/IRD); no proven ZS benefit"},
        {"parameter": "Phytanic acid", "threshold": ">200 µmol/L", "action": "Initiate strict phytol restriction; consider plasmapheresis if acute neurological deterioration"},
        {"parameter": "ALT/AST", "threshold": ">3x ULN", "action": "Withhold/discontinue VPA; switch to LEV IV; LFT q1W until normalization"},
        {"parameter": "PT/INR", "threshold": "INR >1.5", "action": "IV Vitamin K supplementation; defer elective procedures; anaesthesia extreme hazard"},
        {"parameter": "Infantile spasms onset", "threshold": "Any age (NALD)", "action": "ACTH 20–40 U/day IM x2W → taper; VGB AVOID (retinopathy); response check EEG at 2W"},
        {"parameter": "Seizure-free interval", "threshold": "<3 months", "action": "Add CLB or LTG adjunct; if still uncontrolled: multidisciplinary ZSD centre referral"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1: Neonatal / ZS (0–1 month)",
            "features": "Neonatal seizures 100% in ZS; hypotonia; hepatomegaly; DIC risk; adrenal insufficiency; pachygyria/PMG on MRI",
            "action": "LEV IV first-line; avoid VPA; IV dextrose (fasting hazard); ACTH early if IS; supportive/palliative in ZS (fatal <12M)",
        },
        {
            "stage": "Stage 2: Infantile / NALD (1–12 months)",
            "features": "Infantile spasms 35–55%; hypsarrhythmia; cholestatic liver disease; retinopathy; SNHL",
            "action": "ACTH (Level A) for IS; LEV+CLB adjunct; DHA supplementation initiation; VGB AVOID; phytanic monitoring",
        },
        {
            "stage": "Stage 3: Toddler / IRD-Early (1–4 years)",
            "features": "Focal seizures 25–40%; RP progressive; SNHL; peripheral neuropathy; intellectual disability variable",
            "action": "LEV ± LTG ± CLB; phytol-restricted diet; DHA supplementation; visual aids; hearing aids; POLG1 before VPA",
        },
        {
            "stage": "Stage 4: School age / IRD-Established (4–12 years)",
            "features": "Seizure burden stabilises; cerebellar atrophy progressive; adulthood planning begins; R860W/R860W may plateau",
            "action": "Annual ERG + MRI + audiology; optimise AED regimen; developmental support; phytol restriction maintenance",
        },
        {
            "stage": "Stage 5: Adolescent / Atypical (12–18 years)",
            "features": "Atypical cases (R860W + mild missense): isolated RP/SNHL ± mild seizures; may be undiagnosed until adulthood",
            "action": "Gene panel for unexplained RP + SNHL + elevated VLCFA; DHA + phytol restriction; LEV if seizures",
        },
        {
            "stage": "Stage 6: Adult / IRD-Atypical (18+ years)",
            "features": "IRD/Atypical: stable RP, SNHL, ataxia; seizures mild or absent; near-normal lifespan (R860W/R860W)",
            "action": "Annual VLCFA + plasmalogens + DHA; phytol restriction; VF monitoring; AED continuation if needed",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator — FIRST-LINE all ZSD forms",
            "evidence": "Level A (expert consensus for ZSD-epilepsy; Level I trials in IS/focal)",
            "dose": "Neonate: 20 mg/kg/day → titrate to 40 mg/kg/day IV/PO; Child: 20–60 mg/kg/day PO",
            "moa": "SV2A modulator → reduces vesicular glutamate release; no enzyme induction; no hepatotoxicity",
            "monitoring": "Renal function q6M (PO/IV dose adjustment); no TDM routinely required",
            "ci": None,
        },
        {
            "drug": "ACTH (Corticotropin)",
            "class": "Immunomodulator — Level A for infantile spasms (IS) in NALD",
            "evidence": "Level A (WEST study; IS treatment preference over VGB in retinopathy patients)",
            "dose": "20–40 U/day IM × 2 weeks → 2-week taper; some protocols 150 U/m²/day × 2W",
            "moa": "Suppresses ACTH-driven IS mechanism; superior to VGB in ZSD-retinopathy context",
            "monitoring": "BP, glucose, electrolytes, infection q2W; EEG at 2 weeks to assess hypsarrhythmia resolution",
            "ci": "Active infection; caution with adrenal insufficiency baseline (replace first)",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "1,5-Benzodiazepine — Level B adjunct",
            "evidence": "Level B; adjunct for focal + tonic + myoclonic in ZSD",
            "dose": "0.1–0.3 mg/kg/day PO divided bid → titrate to 0.5–1 mg/kg/day",
            "moa": "GABA-A positive allosteric modulator; 1,5-BZD less sedating than 1,4-BZD (clonazepam) chronically",
            "monitoring": "Sedation; tolerance (tachyphylaxis); no hepatotoxicity; no enzyme induction",
            "ci": None,
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Na⁺-channel + AMPA modulator — Level C adjunct",
            "evidence": "Level C; adjunct focal + GTCS in IRD/Atypical",
            "dose": "Start 0.3 mg/kg/day → titrate over 6W to 5–15 mg/kg/day (no enzyme inducers)",
            "moa": "Na⁺ channel blocker + AMPA modulator; hepatic glucuronidation (UGT) NOT CYP → safer hepatic profile",
            "monitoring": "SJS rash (slow titration mandatory); hepatic UGT function (not CYP); no VLCFA interaction",
            "ci": "Rapid titration (SJS risk); caution with concurrent VPA (LTG levels double — adjust dose)",
        },
        {
            "drug": "DHA (Docosahexaenoic acid)",
            "class": "Disease-modifying supplement — Level B (NALD/IRD)",
            "evidence": "Level B; improves biochemical markers; no proven seizure frequency reduction",
            "dose": "100–500 mg/day PO (ethyl-DHA preparation); higher doses in NALD (200–600 mg/day)",
            "moa": "Restores plasma/RBC DHA deficiency; improves retinal + neural membrane function",
            "monitoring": "Plasma DHA q6M (target: lower quartile normal); triglycerides (high-dose DHA); LFTs",
            "ci": None,
        },
        {
            "drug": "Phytol-restricted diet (IRD/Atypical)",
            "class": "Dietary — Level B (IRD + Atypical only)",
            "evidence": "Level B; reduces phytanic acid accumulation → reduces seizure trigger load + neuropathy progression",
            "dose": "Eliminate dairy fat + ruminant fat + some fish oils; maintain adequate calories (fasting HAZARD)",
            "moa": "Reduces dietary phytol → less phytanic acid synthesis → less neuromodulatory pro-convulsant effect",
            "monitoring": "Phytanic acid q6M; nutritional status; caution: caloric restriction worsens fasting hazard",
            "ci": "Never restrict to the point of fasting (triggers adipose phytanic acid release — EXTREME HAZARD)",
        },
        {
            "drug": "Cholic acid (experimental)",
            "class": "Bile acid supplement — Experimental / Compassionate use",
            "evidence": "Level C; reduces toxic bile acid intermediates (DHCA, THCA) in cholestatic ZSD",
            "dose": "5–15 mg/kg/day PO divided tid (compassionate use under metabolic specialist)",
            "moa": "Replaces deficient primary bile acid synthesis → reduces DHCA/THCA → less cholestatic injury",
            "monitoring": "LFTs q1M; bilirubin; PT/INR; stool frequency (bile acid excess → diarrhea)",
            "ci": "Biliary obstruction; severe hepatic failure",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA)",
            "level": "HIGH RISK (RELATIVE-to-ABSOLUTE CI — avoid in ZS/NALD)",
            "reason": (
                "THREE independent mechanisms: (1) Hepatotoxicity in pre-existing cholestatic liver disease (ZS/NALD); "
                "(2) Peroxisomal beta-oxidation inhibition → worsens VLCFA accumulation; "
                "(3) Carnitine depletion (acylcarnitine sequestration). "
                "POLG1 exclusion MANDATORY (CPIC Grade A) before any VPA trial."
            ),
            "alternative": "LEV first-line; CLB or LTG adjunct if needed; LFT q2W if VPA unavoidable in IRD",
        },
        {
            "drug": "Vigabatrin (VGB)",
            "level": "HIGH RISK (AVOID in IRD; RELATIVE CI in ZS/NALD)",
            "reason": (
                "VGB causes irreversible visual field (VF) constriction; ZSD patients have pre-existing "
                "retinopathy (universal ZS/NALD; 25–40% IRD) → additive irreversible blindness. "
                "ACTH (Level A) preferred for infantile spasms."
            ),
            "alternative": "ACTH 20–40 U/day IM for IS (Level A); LEV + CLB adjunct; last resort only",
        },
        {
            "drug": "Fosphenytoin / Phenytoin (PHT)",
            "level": "RELATIVE CI (not adrenal-related — unlike ABCD1)",
            "reason": (
                "CYP3A4 induction → DHA depletion (DHA metabolised by CYP enzymes; already low in ZSD) + "
                "hepatic enzyme induction compounds cholestatic burden. In ZSD: NO adrenal crisis mechanism "
                "(unlike ABCD1). Use LEV IV in SE instead of fosphenytoin."
            ),
            "alternative": "LEV IV for status epilepticus; CBZ/OXC RELATIVE CI (same CYP concern)",
        },
        {
            "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
            "level": "RELATIVE CI",
            "reason": (
                "CYP3A4/2C9 enzyme induction → DHA depletion (worsens already-low DHA) + "
                "hepatic enzyme induction (compounds cholestatic burden in ZS/NALD). "
                "Not absolute CI — consider if LEV + CLB + LTG fail; monitor LFTs + DHA."
            ),
            "alternative": "LEV + CLB + LTG sequence preferred; CBZ only if refractory with LFT + DHA monitoring",
        },
        {
            "drug": "Fasting / prolonged NPO",
            "level": "EXTREME HAZARD (not a drug — perioperative / clinical protocol)",
            "reason": (
                "Starvation → adipose phytanic acid release (not metabolised in ZSD) → acute "
                "neurotoxicity + acute seizure clusters. COAGULOPATHY: cholestatic ZS/NALD → "
                "PT/INR elevation → surgical bleeding risk. IV Vitamin K MANDATORY."
            ),
            "alternative": "IV dextrose (D10W or D12.5W) MANDATORY during all pre-surgical fasts; PT/INR + LFT pre-op",
        },
        {
            "drug": "Lorenzo's Oil (GTO + GTE 4:1)",
            "level": "INEFFECTIVE (not a CI — active harm if relied upon)",
            "reason": (
                "Lorenzo's Oil reduces VLCFA SYNTHESIS (elongase inhibition) but in ZSD peroxisomal "
                "IMPORT is absent — VLCFA cannot enter peroxisome even if less is made. "
                "Level C presymptomatic ABCD1 only. No benefit in ZSD; HSCT also not indicated "
                "(no neuroinflammatory window to arrest in ZSD)."
            ),
            "alternative": "No disease-modifying therapy proven; DHA supplementation Level B; gene therapy research stage",
        },
    ]

    return {
        "etiologies": etiologies,
        "patients": patients,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "thresholds": thresholds,
        "lifecycle": lifecycle,
        "treatments": treatments,
        "contraindications": contraindications,
    }


# ── Definitions ───────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "key_concepts": [
            "PEX6 encodes Peroxin-6 (980 aa, AAA-ATPase); D2 domain provides the principal ATP-hydrolysis motor for PEX1–PEX6 heterodimer",
            "PEX6 D2 domain p.Arg860Trp (R860W) = most common European hypomorphic PEX6 allele (~18% of all PEX6 alleles)",
            "PEX1–PEX6 heterodimer anchored to peroxisomal membrane by PEX26 (APEM9); retrotranslocates PEX5 for recycling",
            "PEX6 LOF biochemically identical to PEX1 LOF: all 6 peroxisomal pathways fail simultaneously",
            "PEX6-ZSD cohorts show more IRD (45%) vs PEX1-ZSD (30–35%): R860W provides more residual PEX5 recycling than PEX1-G843D",
            "Plasmalogens (RBC) LOW in all ZSD — CRITICAL DISTINCTION from ABCD1 (plasmalogens NORMAL in ABCD1)",
            "NBS biomarker: C26:0-lyso-PC (DBS) — same as PEX1-ZSD; confirms ZSD/ABCD1 screen; plasmalogens distinguish",
            "DHA low → impaired neuronal membrane fluidity → failed neuroblast migration → pachygyria/PMG (ZS PATHOGNOMONIC)",
            "Phytanic acid: NOT metabolised in ZSD (alpha-oxidation failed) → accumulates → neuromodulatory pro-convulsant effect",
            "Fasting EXTREME HAZARD: starvation → adipose phytanic acid release → acute seizure clusters; IV dextrose mandatory",
            "LEV first-line ALL ZSD: no enzyme induction, no hepatotoxicity, IV/PO, neonatal-safe (SV2A mechanism)",
            "VPA HIGH RISK: 3 mechanisms — hepatotoxicity (cholestatic liver) + peroxisomal BO inhibition + carnitine depletion; POLG1 MANDATORY",
            "VGB HIGH RISK: additive irreversible VF loss on top of pre-existing retinopathy; ACTH Level A for IS",
            "ACTH Level A for infantile spasms (NALD); preferred over VGB due to ZSD retinopathy",
            "DHA supplementation Level B (NALD/IRD); improves biochemical markers; no ERT; no HSCT; no Lorenzo's Oil benefit",
            "Atypical adult PEX6-ZSD: presents as isolated RP + SNHL + cerebellar ataxia ± mild seizures; often delayed diagnosis",
        ],
        "diagnostic_algorithm": [
            "Step 1: Suspect ZSD — neonatal hypotonia + seizures + hepatomegaly OR infantile spasms + RP + SNHL OR adult RP + SNHL + ataxia + elevated VLCFA",
            "Step 2: Plasma VLCFA panel — C26:0, C24:0, C26:0/C22:0, C24:0/C22:0 (elevated in ZSD AND ABCD1)",
            "Step 3: RBC Plasmalogens — LOW = ZSD (PBD); NORMAL = ABCD1 (X-ALD); CRITICAL DISTINCTION",
            "Step 4: Phytanic + pristanic acid (elevated in ZSD; NOT elevated in ABCD1)",
            "Step 5: DHA (plasma/RBC — LOW in ZSD; guides supplementation)",
            "Step 6: Pipecolic acid (urine/plasma — elevated in ZSD; confirmatory)",
            "Step 7: Gene panel — PBD-ZSD: PEX1, PEX6, PEX12, PEX10, PEX2, PEX3, PEX14, PEX16, PEX19, PEX26; identify specific gene",
            "Step 8: Ophthalmology + ERG (retinopathy — universal ZS/NALD; 25–40% IRD)",
            "Step 9: Audiology + BAER (SNHL — progressive in NALD/IRD)",
            "Step 10: Brain MRI — pachygyria/PMG (ZS PATHOGNOMONIC); WM abnormalities (NALD); cerebellar atrophy (IRD/late)",
            "Step 11: LFTs + coagulation (PT/INR) + bile acids (cholestatic liver disease — ZS/NALD)",
            "Step 12: POLG1 sequencing BEFORE any VPA trial (CPIC Grade A; mandatory in all ZSD)",
        ],
        "pharmacological_distinctions": [
            "PEX6-ZSD vs ABCD1: NO adrenal crisis mechanism in ZSD — PHT/CBZ not CI for adrenal reasons (unlike ABCD1 where PHT → cortisol drop → adrenal crisis)",
            "PEX6-ZSD vs ABCD1: NO HSCT window — neuroinflammatory disease progression absent in ZSD; HSCT not indicated",
            "PEX6-ZSD vs NPC1: no miglustat; no HPbCD; no cholesterol transport defect; entirely different mechanism",
            "VPA HIGH RISK in PEX6-ZSD (3 mechanisms: hepatotoxicity + peroxisomal BO + carnitine) — MORE dangerous than in ABCD1 (where POLG1 alone is main concern)",
            "VGB AVOID PEX6-ZSD (retinopathy additive) — same as PEX1-ZSD; different from ABCD1 (no primary retinopathy in ABCD1 unless also CCALD-stage)",
            "PHT/CBZ/OXC RELATIVE CI PEX6-ZSD: CYP induction → DHA depletion (DHA already low) + hepatic burden; NOT adrenal mechanism",
            "ACTH Level A for IS in ZSD — preferred over VGB; same evidence base as PEX1-ZSD",
            "DHA supplementation: Level B (NALD/IRD); no evidence in ZS (fatal before benefit possible)",
            "Phytol-restricted diet: IRD + Atypical PEX6-ZSD; NOT needed in ZS (palliative) or NALD (fasting hazard > dietary phytol)",
            "Lorenzo's Oil INEFFECTIVE in all ZSD (peroxisomal import absent; enzyme synthesis reduction futile)",
            "Carnitine supplementation: consider if VPA unavoidable + documented depletion (L-carnitine 50–100 mg/kg/day)",
            "Anaesthesia EXTREME HAZARD in ZSD: fasting → phytanic surge + coagulopathy (PT/INR); IV dextrose + IV Vitamin K MANDATORY",
        ],
        "differential_diagnosis": [
            {"condition": "PEX1-ZSD", "distinction": "Biochemically identical; PEX1 more common (~65% ZSD); G843D allele vs R860W (PEX6); gene panel distinguishes"},
            {"condition": "ABCD1 (X-ALD)", "distinction": "VLCFA elevated in both; ABCD1: plasmalogens NORMAL (vs ZSD LOW); ABCD1: X-linked males; adrenal crisis; CCALD; HSCT indicated"},
            {"condition": "Adult Refsum Disease (PHYH/PEX7-RCDP)", "distinction": "Phytanic acid elevated in both IRD-PEX6 and Refsum; DHA normal in Refsum; VLCFA NORMAL in Refsum; plasmalogens NORMAL in Refsum (PHYH) vs LOW in ZSD"},
            {"condition": "RCDP (PEX7 deficiency)", "distinction": "PEX7: plasmalogen LOW (like ZSD) but VLCFA NORMAL; rhizomelia + chondrodysplasia punctata; different gene — PEX7 targets PTS2 pathway only"},
            {"condition": "Krabbe / MLD", "distinction": "Lysosomal NOT peroxisomal; VLCFA normal; galactosylceramide (Krabbe) or sulfatide (MLD) elevated; nerve biopsy; no plasmalogen defect"},
            {"condition": "Usher syndrome", "distinction": "RP + SNHL (like IRD-PEX6) but VLCFA NORMAL; MYOSIN7A/USH2A mutations; no intellectual disability; no seizures typically"},
            {"condition": "Mitochondrial disease (MERRF/MELAS)", "distinction": "Seizures + RP + SNHL can overlap; VLCFA NORMAL; lactate elevated; mtDNA/POLG1 mutations; ragged-red fibres biopsy"},
        ],
        "standards": [
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Am J Med Genet C. 2016",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Klouwer FCC et al. Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis. 2015",
            "Ebberink MS et al. Identification of 13 PEX6 mutations with complex phenotypic variability. Hum Mutat. 2011",
            "Poll-The BT, Gärtner J. Clinical diagnosis, biochemical findings, management of peroxisomal diseases. Semin Pediatr Neurol. 2012",
            "Wanders RJ, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006",
            "CPIC Guideline: Valproic acid — POLG1. cpicpgx.org. 2022",
            "Steinberg SJ et al. Peroxisome biogenesis disorders, Zellweger syndrome spectrum. Neurology. 2006",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (keys) ===")
    b = get_breakdown()
    print(list(b.keys()))
    print(f"Patients: {len(b['patients'])}")
    print("\n=== DEFINITIONS (keys) ===")
    d = get_definitions()
    print(list(d.keys()))
