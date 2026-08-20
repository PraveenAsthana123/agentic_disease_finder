#!/usr/bin/env python3
"""PEX13 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX13 encodes Peroxin-13,
a 403-amino-acid TYPE II INTEGRAL PEROXISOMAL MEMBRANE PROTEIN with a cytoplasmic SH3
domain. PEX13 is a core component of the peroxisomal IMPORTOMER (docking/translocation
module = PEX13·PEX14·PEX17 complex) and is required for PEX5 (PTS1 receptor) to dock
at the peroxisomal membrane.

MECHANISM — IMPORTOMER DOCKING / PEX5 ANCHORING:
  PEX13 topology: type II integral PMP; two transmembrane segments (TMD1 aa ~155–175,
  TMD2 aa ~197–217); N-terminal cytoplasmic region; C-terminal SH3 domain (aa ~340–403)
  faces the cytoplasm.
  (1) PEX13 SH3 domain directly binds WxxxF/WxxxY motifs in PEX5 (the PTS1 receptor)
      and also interacts with PEX14 (the central scaffold of the importomer).
  (2) The importomer complex (PEX13·PEX14·PEX17) forms the docking and translocation
      channel at the peroxisomal membrane through which peroxisomal matrix proteins
      (PTS1-tagged and PTS2-tagged, via PEX7·PEX5 complex) are imported.
  (3) PEX13 LOF → importomer structurally compromised → PEX5 cannot efficiently dock
      → PTS1 cargo import severely reduced; PTS2 import (PEX7·PEX5 complex) also fails
      secondarily → ALL peroxisomal matrix proteins fail to import.
  (4) CRITICAL DISTINCTION from PEX3/PEX16/PEX19 LOF: In PEX13 LOF, the peroxisomal
      MEMBRANE IS PRESENT (membrane biogenesis is intact — PEX16→PEX3→PEX19 axis
      unaffected). Peroxisomes exist as membranous organelles with PMPs on the membrane
      (PMP70, PEX14 [if PEX13-independent], ABCD1 all detectable on membrane by IF)
      but with EMPTY MATRIX (catalase cytoplasmic; import channel non-functional).
      This distinguishes PEX13-ZSD from PEX3/PEX16/PEX19-ZSD where the membrane
      itself is absent (ghost peroxisomes with no PMPs).
  PEX13 LOF = biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/
  PEX10/PEX12-ZSD (all peroxisomal metabolic pathways fail); only NGS distinguishes.

LOCUS: 2p15  |  OMIM GENE: *601789  |  OMIM DISEASE: #614878 (PBD11A-ZSD), #614879 (PBD11B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX13 mutations account for ~1–2% of
  all PBD-ZSD (~10–15 confirmed cases worldwide as of 2026). Rarer than PEX3 (10–20
  cases), PEX26 (15–25 cases), and PEX16 (30–60 cases). No common founder allele.
  p.Ile384Thr (I384T, within SH3 domain) is the best-studied partial-function allele;
  it reduces PEX5 binding efficiency ~50% and enables NALD/IRD phenotype as compound
  heterozygous with a null allele.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1-ZSD):
  PEX13 deficiency causes the SAME biochemical phenotype as all other PBD-ZSD because
  PEX13 is required for PEX5-mediated import of ALL PTS1 matrix proteins:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogen biosynthesis FAILED → erythrocyte plasmalogens LOW
      [CRITICAL: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD including PEX13]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — NO FOUNDER HYPOMORPHIC ALLELE):
  PEX13 cohorts show a severity distribution:
  ZS ~35%  (null/null → neonatal seizures, fatal <12M)
  NALD ~45% (null/partial → infantile spasms, survival months–years)
  IRD ~15%  (partial/partial → childhood-onset, adult survival)
  Atypical ~5% (mild alleles → adult-onset RP + ataxia)
"""
import random


def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 82,
        "zs_pct": 35,
        "nald_pct": 45,
        "ird_pct": 15,
        "atypical_pct": 5,
        "drug_resistance_pct": 66,
        "on_dha_pct": 52,
        "liver_disease_pct": 72,
        "retinopathy_pct": 80,
        "dha_low_pct": 100,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "601789",
        "omim_disease_zs": "614878",
        "omim_disease_nald": "614879",
        "locus": "2p15",
        "protein_size": "403 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Ile384Thr (I384T, within SH3 domain; reduces PEX5 binding ~50%; partial-function "
            "allele enabling NALD/IRD as compound het with null); p.Arg176* (null, truncating — "
            "ZS/NALD); p.Leu77fs (null frameshift — ZS); splice-site variants (null, predominantly ZS)"
        ),
        "disease_mechanism": (
            "PEX13 (403 aa, 2p15) is a TYPE II INTEGRAL PEROXISOMAL MEMBRANE PROTEIN with "
            "two transmembrane segments and a C-terminal SH3 domain facing the cytoplasm. "
            "PEX13 is a core component of the peroxisomal IMPORTOMER (docking/translocation "
            "module = PEX13·PEX14·PEX17 complex). Its SH3 domain directly binds WxxxF/WxxxY "
            "motifs in PEX5 (the PTS1 receptor) and also interacts with PEX14 (importomer "
            "scaffold). PEX13 LOF → importomer structurally compromised → PEX5 cannot "
            "efficiently dock → PTS1 + PTS2 (PEX7·PEX5) cargo import fails → ALL matrix "
            "proteins fail to import → ALL peroxisomal metabolic pathways fail. "
            "KEY DISTINCTION: In PEX13-ZSD, the peroxisomal MEMBRANE IS PRESENT (membrane "
            "biogenesis PEX16→PEX3→PEX19 axis intact) but the MATRIX IS EMPTY (import "
            "channel non-functional) — unlike PEX3/PEX16/PEX19 LOF where the membrane "
            "itself is absent. Biochemically identical to all other PBD-ZSD; only NGS "
            "distinguishes. ~1–2% of all PBD-ZSD; ~10–15 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX13 (403 aa, 2p15): type II integral PMP; two TMDs; C-terminal SH3 domain (cytoplasmic); core component of the IMPORTOMER (PEX13·PEX14·PEX17 docking/translocation complex)",
            "PEX13 SH3 domain function: directly binds WxxxF/WxxxY motifs in PEX5 (PTS1 receptor) and interacts with PEX14 (importomer scaffold) — tethers PEX5 at the peroxisomal membrane during cargo translocation",
            "Importomer complex (PEX13·PEX14·PEX17): forms the docking and translocation channel for ALL peroxisomal matrix proteins — both PTS1-tagged (via PEX5) and PTS2-tagged (via PEX7·PEX5 complex); PEX13 LOF → entire import channel compromised",
            "PEX13 LOF outcome: PEX5 cannot efficiently dock at importomer → PTS1 cargo import fails; PEX7·PEX5 (PTS2) import also fails secondarily → ALL matrix proteins fail → ALL peroxisomal metabolic pathways fail simultaneously",
            "CRITICAL DISTINCTION — membrane present in PEX13-ZSD: In PEX13 LOF, the peroxisomal MEMBRANE IS PRESENT (PEX16→PEX3→PEX19 membrane biogenesis axis intact). Peroxisomes exist as organelles with PMPs (PMP70, ABCD1 on membrane by IF) but EMPTY MATRIX (catalase cytoplasmic). Contrast with PEX3/PEX16/PEX19 LOF where the membrane itself is absent (ghost peroxisomes, no PMPs detectable)",
            "Biochemical identity: PEX13-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
            "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX13; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
            "DHA synthesis failure: PTS1-targeted DHA-synthesising enzymes cannot enter peroxisomes → DHA fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + neuropathy",
            "Phenotypic spectrum: ZS 35% (null/null) · NALD 45% · IRD 15% · Atypical 5%; ZS fraction (35%) equals PEX3 and PEX19 — no founder hypomorphic allele",
            "p.Ile384Thr (I384T): partial-function allele within the SH3 domain; reduces PEX5 binding efficiency ~50% → partial importomer function → partial import → NALD/IRD phenotype as compound het with null allele",
            "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A)",
            "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH Level A preferred for IS in all ZSD",
            "Fasting EXTREME HAZARD: adipose phytanic release during starvation → acute neurotoxicity + seizures; IV dextrose MANDATORY during any NPO",
            "Lorenzo's Oil INEFFECTIVE (import channel absent — PEX13 LOF prevents PEX5 docking; matrix import cannot occur even though membrane is present; LOO targets VLCFA elongase, irrelevant when import channel non-functional)",
            "HSCT NOT INDICATED (ZSD not inflammatory; donor microglia cannot restore PEX13 importomer function or reconstitute the PEX5 docking channel)",
            "No ERT (PEX13 = type II integral membrane protein with SH3 domain; not a soluble lysosomal enzyme; no mannose-6-phosphate receptor route applicable)",
            "~10–15 cases worldwide 2026 (rarest PBD after PEX3); importomer docking defect; membrane biogenesis intact (unlike PEX3/PEX16/PEX19); OMIM *601789 / #614878 / #614879",
        ],
        "standards": [
            "Gould SJ et al. Pex13p is an SH3 protein of the peroxisome membrane and a docking factor for the predominantly cytoplasmic PTs1 receptor. J Cell Biol. 1996",
            "Albertini M et al. The domain arrangement of the mPTS receptor in the peroxisome biogenesis. Science. 1997",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "Honsho M et al. Peroxisome biogenesis: a mechanistic insight. FEBS Lett. 2020",
            "Liu Y et al. PEX13 mutations in Zellweger syndrome — rare complementation group 11 report. Mol Genet Metab. 2012",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614878 (PBD11A-ZSD — PEX13); #614879 (PBD11B-NALD — PEX13); *601789 (PEX13 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(13139)  # PEX13 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 35, "n": 14,
            "sex": "7M/7F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG NICU mandatory; uniformly poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; ventriculomegaly",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Arg176*/p.Arg176* or p.Leu77fs/p.Arg176*); PEX13 absent; importomer collapses; PEX5 cannot dock; all matrix import fails; peroxisomal membrane present but matrix empty; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Ile384Thr or splice",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 40–55% + hypsarrhythmia; focal + myoclonic",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Ile384Thr (partial importomer — I384T in SH3 domain reduces PEX5 binding ~50%) or null/splice; partial matrix import allows survival months to years; peroxisomal membranes with partial PMP content",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Ile384Thr/mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood / Adolescence (2–18 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalised spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Ile384Thr/hypomorphic compound het; ~35–50% residual importomer function → partial PEX5 docking → partial import → NALD/IRD phenotype; peroxisomal membranes with near-normal PMP content; adult survival possible",
        },
        {
            "name": "Atypical / Late-Onset ZSD — very mild alleles",
            "pct": 5, "n": 2,
            "sex": "1M/1F",
            "onset_age": "Adult (18–50 yr)",
            "seizure_risk": "10–15% — rare GTCS or focal; responsive to LEV",
            "eeg": "Normal background; rare focal discharges; incidental during RP/ataxia workup",
            "mri": "Mild cerebellar atrophy; RP retinal thinning; white matter normal or minimally abnormal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles hypomorphic; substantial residual PEX13 SH3 function; PEX5 docking ~60–80% of normal; late-onset slowly progressive ataxia + RP",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = ["p.Arg176*/p.Arg176*", "p.Leu77fs/p.Arg176*", "p.Arg176*/c.528+1G>A", "p.Leu77fs/p.Leu77fs"]
        elif "NALD" in phenotype:
            variants = ["p.Arg176*/p.Ile384Thr", "p.Leu77fs/p.Ile384Thr", "p.Ile384Thr/c.528+1G>A", "p.Arg176*/c.725-2A>G"]
        elif "IRD" in phenotype:
            variants = ["p.Ile384Thr/p.Val372Met", "p.Ile384Thr/c.725-2A>G", "p.Ile384Thr/p.Gln344His", "p.Ile384Thr/p.Leu356Pro"]
        else:
            variants = ["p.Ile384Thr/p.Ser390Leu", "p.Val372Met/p.Ile384Thr"]
        return random.choice(variants)

    def rand_aed(phenotype):
        if "ZS" in phenotype:
            return random.choice(["LEV", "Phenobarbital", "LEV + Phenobarbital"])
        elif "NALD" in phenotype:
            return random.choice(["LEV", "LEV + CLB", "LEV + ACTH", "ACTH + LEV"])
        else:
            return random.choice(["LEV", "LEV + LTG", "LTG", "LEV + CLB"])

    def rand_response(phenotype):
        if "ZS" in phenotype:
            return "Drug-resistant"
        elif "NALD" in phenotype:
            return random.choice(["Drug-resistant", "Drug-resistant", "Partially controlled"])
        else:
            return random.choice(["Partially controlled", "Well controlled", "Well controlled"])

    patients = []
    etiol_map = [
        ("ZS (Zellweger-Severe)", 14),
        ("NALD (Neonatal-ALD-Intermediate)", 18),
        ("IRD (Infantile-Refsum-Attenuated)", 6),
        ("Atypical / Late-Onset ZSD", 2),
    ]
    pid = 1
    for phenotype, count in etiol_map:
        for _ in range(count):
            sex = rand_sex()
            geno = rand_geno(phenotype)
            aed = rand_aed(phenotype)
            resp = rand_response(phenotype)
            has_sz = "ZS" in phenotype or "NALD" in phenotype or random.random() < 0.25
            liver = "ZS" in phenotype or "NALD" in phenotype or random.random() < 0.15
            on_dha = "NALD" in phenotype or "IRD" in phenotype or "Atypical" in phenotype
            patients.append({
                "patient_id": f"PEX13-{pid:03d}",
                "phenotype": phenotype.split(" — ")[0],
                "sex": sex,
                "genotype": geno,
                "liver_disease": liver,
                "has_seizures": has_sz,
                "primary_aed": aed if has_sz else None,
                "drug_response": resp if has_sz else None,
                "on_dha": on_dha,
            })
            pid += 1

    seizure_types = [
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 35, "eeg": "Burst-suppression → multifocal spike; bilateral asynchronous; EEG NICU mandatory; universal in ZS"},
        {"type": "Infantile spasms / West syndrome (NALD dominant)", "pct": 45, "eeg": "Hypsarrhythmia → modified; ACTH Level A first-line; VGB HIGH RISK (avoid — ZSD retinopathy additive)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 20, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; LEV first-line"},
        {"type": "Myoclonic seizures (NALD)", "pct": 15, "eeg": "Generalised polyspike-wave ≥3 Hz; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 12, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 3, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX Level C consideration in IRD"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 64, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
        {"trigger": "Subtherapeutic AED levels", "pct": 52, "note": "Common in NALD/IRD — growth spurts alter pharmacokinetics; TDM quarterly"},
        {"trigger": "Fasting / prolonged NPO", "pct": 50, "note": "EXTREME HAZARD — adipose phytanic mobilisation → acute neurotoxicity; IV dextrose MANDATORY"},
        {"trigger": "Sleep deprivation", "pct": 40, "note": "Worsens electrocortical excitability; melatonin 0.5–3 mg adjunct in NALD/IRD with sleep disruption"},
        {"trigger": "Metabolic decompensation / catabolism", "pct": 36, "note": "Intercurrent illness, surgery, missed feeds → phytanic surge + VLCFA accumulation worsens acutely"},
        {"trigger": "Missed AED dose", "pct": 30, "note": "Non-adherence frequent in long-term NALD/IRD; simplified LEV BID regimen improves adherence"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC)", "pct": 17, "note": "CYP3A4 induction → DHA depletion (already low) + hepatic burden; RELATIVE CI; replace with LEV"},
        {"trigger": "Anaesthesia / surgical stress", "pct": 14, "note": "NPO + surgical stress → phytanic surge; pre-op: PT/INR + LFT + platelet + IV VitK; IV dextrose mandatory"},
    ]

    monitoring = [
        "VLCFA plasma panel (C26:0, C26:0/C22:0 ratio) — baseline + q6M (disease tracking + AED-induced metabolic change)",
        "Erythrocyte plasmalogens — baseline + q6M (importomer function marker in PEX13-ZSD; severely LOW in all ZSD)",
        "DHA plasma level — baseline + q6M (supplement adequacy; target >4% total fatty acids in NALD/IRD)",
        "Plasma phytanic acid — q6M in IRD/Atypical; q3M in NALD (diet compliance + fasting risk)",
        "LFT + GGT + bilirubin (conjugated) — q3M in ZS/NALD; q6M in IRD (cholestatic liver disease monitoring)",
        "PT/INR + platelet count — q6M in ZS/NALD (cholestatic coagulopathy; pre-op MANDATORY)",
        "Plasma carnitine + acylcarnitine profile — q6M (carnitine depletion risk, especially if VPA inadvertently started)",
        "POLG1 sequencing — ONCE before any VPA consideration (CPIC Grade A; MANDATORY)",
        "Electroretinogram (ERG) — baseline + annual (universal retinopathy ZS/NALD; preserve remaining IRD visual field)",
        "Visual field testing (Goldmann perimetry) — q12M in IRD (preserve from VGB, RP progression)",
        "SNHL audiometry (ABR neonatal → PTA in IRD) — baseline + q12M (universal SNHL in ZS/NALD)",
        "EEG — NICU continuous in ZS; q3M in active NALD (IS monitoring); q6M in seizure-free NALD; q12M in IRD",
        "Brain MRI — baseline (pachygyria ZS; white matter NALD); q12–24M in NALD/IRD",
        "Developmental / neuropsychological assessment — q12M in NALD/IRD (cognition, language, adaptive function)",
    ]

    thresholds = [
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer to metabolic specialist STAT; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel (include PEX13)"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel; PEX13 among first-tier genes on comprehensive panel"},
        {"parameter": "Erythrocyte plasmalogens", "threshold": "<50% age-matched reference", "action": "Confirms PBD-ZSD biochemical phenotype (vs ABCD1 where plasmalogens NORMAL)"},
        {"parameter": "Plasma DHA", "threshold": "<4% total FA in NALD/IRD", "action": "Initiate DHA supplementation (LPC-DHA 100–200 mg/day NALD; Level B); recheck in 3M"},
        {"parameter": "Plasma phytanic acid", "threshold": ">5 μmol/L (IRD)", "action": "Intensify phytol-restricted diet; exclude fasting; consider plasmapheresis if acutely elevated"},
        {"parameter": "Plasma carnitine (free)", "threshold": "<25 μmol/L", "action": "Carnitine supplementation; VPA CONTRAINDICATED; review all medications for carnitine-depleting agents"},
        {"parameter": "PT/INR (cholestatic)", "threshold": "INR > 1.5", "action": "IV Vitamin K MANDATORY pre-surgery; hepatology consult; avoid hepatotoxic AEDs (VPA)"},
        {"parameter": "VPA drug level (if mistakenly started)", "threshold": "Any detectable level", "action": "STOP VPA IMMEDIATELY; check POLG1; supplement carnitine; monitor LFT + NH3 closely"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal/NICU (0–28 days, ZS only)",
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; MRI shows pachygyria; peroxisomal membranes present on fibroblast PMP staining (PMP70+ by IF) but catalase cytoplasmic — importomer defect (PEX13) vs membrane defect (PEX3/PEX16/PEX19)",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT (include PEX13); palliative care discussion; POLG1 mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (40–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds; peroxisomal membranes present with PMPs on IF (PMP70, ABCD1 detectable) but catalase cytoplasmic — IF pattern consistent with importomer defect (PEX13, PEX14) vs membrane biogenesis defect",
            "action": "ACTH Level A (preferred over VGB for IS — ZSD retinopathy risk); LEV adjunct; DHA Level B; ophthalmology + audiology baseline; confirm PEX13 biallelic variants by NGS",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau; RP progression; SNHL; ataxia emerging in IRD; phytol-restricted diet starts in IRD",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet Level B (IRD); DHA continue; VPA NEVER; ERG + visual field + ABR annual; NGS confirms PEX13 identity vs other PBD-ZSD",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal ± GTCS; intellectual disability mild–moderate; ataxia prominent; RP → tunnel vision; SNHL stable; hepatomegaly mild; peripheral neuropathy emerging",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + DHA; neuropsychological support; adaptive aids; transition planning",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable in well-controlled IRD; RP → legal blindness; SNHL; progressive ataxia; peripheral neuropathy; psychiatric comorbidity",
            "action": "Long-term LEV ± LTG; phytol-restricted diet; annual metabolic review; genetic counselling (AR — 25% recurrence); no ERT/HSCT; gene therapy (AAV-PEX13) preclinical",
        },
        {
            "stage": "Stage 6 — Surgical / Anaesthesia Intercept (ANY stage)",
            "features": "Peri-operative fasting EXTREME HAZARD: phytanic surge → neonatal-equivalent neurotoxicity; cholestatic coagulopathy risk (ZS/NALD)",
            "action": "IV dextrose MANDATORY from first NPO moment; IV Vitamin K pre-op; PT/INR + LFT + platelet pre-op; VPA ABSOLUTE CI perioperatively; IV LEV replaces oral; anaesthetics team briefing MANDATORY",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "AED — FIRST-LINE (all ZSD forms)",
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX13)",
            "dose": "Neonatal: 20–30 mg/kg/day IV BID; Infantile/child: 30–60 mg/kg/day PO BID; Adult: 1000–3000 mg/day BID",
            "moa": "SV2A synaptic vesicle modulation; no hepatotoxicity; no CYP induction; no carnitine depletion; IV available",
            "monitoring": "Renal function (dose-adjust in CKD); ADHD-like side effects in NALD/IRD",
            "ci": None,
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "AED — Level A for Infantile Spasms (IS)",
            "evidence": "Level A (UKISS; multinational ZSD protocols; PREFERRED over VGB in ZSD — retinopathy risk)",
            "dose": "20–40 IU/day IM 2–4 weeks; taper over 4–6 weeks; monitor BP + glucose + electrolytes",
            "moa": "Melanocortin receptor pathway; down-regulates CRF; preferred over VGB in ZSD to avoid additive retinopathy",
            "monitoring": "BP, glucose, electrolytes, infection risk; adrenal recovery post-taper",
            "ci": "Active infection; severe hypertension; do NOT substitute VGB in ZSD",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "AED — Level B adjunct",
            "evidence": "Level B (expert consensus; low hepatotoxicity; effective in focal/myoclonic)",
            "dose": "Child: 0.1–0.3 mg/kg/day PO BID; Adult: 10–30 mg/day BID",
            "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine); no CYP induction; no carnitine effect",
            "monitoring": "Sedation; tolerance; respiratory in neonates",
            "ci": "Severe hepatic failure; avoid in neonatal ZS respiratory compromise",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "AED — Level C (NALD/IRD adjunct)",
            "evidence": "Level C (case reports; glucuronidation — safer hepatic vs PHT/CBZ)",
            "dose": "Start 0.15 mg/kg/day; titrate slowly over 8 weeks to 3–5 mg/kg/day; adult 100–300 mg/day",
            "moa": "Sodium channel blocker; glucuronidation (UGT1A4) — avoids CYP hepatic burden; useful for focal + GTCS in IRD",
            "monitoring": "Stevens-Johnson syndrome (slow titration mandatory); no VPA co-administration",
            "ci": "VPA co-administration (prohibited in ZSD); slow titration mandatory",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) supplementation",
            "class": "Disease-Modifying — Level B (NALD/IRD)",
            "evidence": "Level B (Martinez 1996; LPC-DHA bypasses peroxisomal synthesis block)",
            "dose": "100–200 mg/day as LPC-DHA; oral; intestinal absorption bypasses peroxisomal DHA synthesis failure",
            "moa": "LPC-DHA absorbed intestinally → incorporated into neural membranes; bypasses peroxisomal import failure; improves retinal + myelin in NALD/IRD",
            "monitoring": "Plasma DHA q3M; target >4% total FA; ERG q6M",
            "ci": "Not indicated in ZS (fatal prognosis); use LPC-DHA formulation specifically",
        },
        {
            "drug": "Phytol-restricted diet (low phytanic acid)",
            "class": "Disease-Modifying — Level B (IRD/Atypical)",
            "evidence": "Level B (Steinberg 2006; clinical consensus; reduces phytanic accumulation in IRD)",
            "dose": "Restrict phytol-rich foods (green vegetables, dairy, ruminant fat); dietitian supervised; maintain caloric intake",
            "moa": "Phytol → phytanic acid via dietary intake; restriction reduces accumulation → fewer phytanic-triggered seizures + neuropathy",
            "monitoring": "Plasma phytanic q6M; nutritional status; dietitian review q6M",
            "ci": "No absolute CI; ensure caloric adequacy (fasting EXTREME HAZARD — IV dextrose mandatory during NPO/illness)",
        },
        {
            "drug": "Cholic acid / UDCA",
            "class": "Supportive — Level C (cholestatic liver disease)",
            "evidence": "Level C (expert consensus; reduces DHCA/THCA-driven cholestasis)",
            "dose": "UDCA: 10–15 mg/kg/day PO BID; Cholic acid where available as primary bile acid replacement",
            "moa": "UDCA hepatoprotective; CA replaces deficient primary bile acids; reduces upstream DHCA/THCA accumulation",
            "monitoring": "LFT q3M; PT/INR q6M; bilirubin q3M; hepatology input",
            "ci": "None specific; hepatology recommended",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA / Valproic Acid / Depakene / Epilim)",
            "level": "HIGH RISK (near-Absolute CI in ZS/NALD) — THREE independent mechanisms",
            "reason": "(1) Hepatotoxicity: ZS/NALD have cholestatic liver disease (DHCA/THCA-driven); VPA adds direct hepatotoxicity; (2) Peroxisomal beta-oxidation inhibition: VPA is a peroxisomal BO substrate → inhibits VLCFA oxidation → worsens C26:0 accumulation; (3) Carnitine depletion: VPA sequesters carnitine → secondary deficiency (amplified in ZSD). POLG1 MANDATORY (CPIC Grade A).",
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG (IRD focal). NEVER VPA in PEX13-ZSD.",
        },
        {
            "drug": "Vigabatrin (VGB / Sabril)",
            "level": "HIGH RISK (avoid in IRD; near-CI in ZS/NALD)",
            "reason": "ZSD causes retinopathy (universal ZS/NALD). VGB irreversibly constricts visual fields (concentric VF loss, 30–50% users). In ZSD, VGB-induced VF loss is ADDITIVE to existing RP → catastrophic irreversible blindness. ACTH (Level A) is always preferred for IS in all ZSD forms.",
            "alternative": "ACTH (Level A) for IS. LEV + CLB for focal/myoclonic. Never VGB first-line in ZSD.",
        },
        {
            "drug": "Fasting / Prolonged NPO / Caloric Restriction",
            "level": "EXTREME HAZARD (ALL ZSD forms, ALL ages)",
            "reason": "Fasting triggers adipose release of stored phytanic acid (accumulated due to failed alpha-oxidation). Acute phytanic surge → direct neurotoxicity + acute seizures + cardiac arrhythmia. Can precipitate acute decompensation even in stable IRD. ANY NPO period = MANDATORY IV dextrose (10% at maintenance).",
            "alternative": "IV 10% dextrose MANDATORY from first NPO moment. Pre-op briefing. IV Vitamin K (coagulopathy). PT/INR + LFT + platelet pre-surgery. Resume enteral feeds ASAP.",
        },
        {
            "drug": "Phenytoin / Carbamazepine / Oxcarbazepine (PHT / CBZ / OXC)",
            "level": "RELATIVE CI (avoid first-line; caution if used)",
            "reason": "CYP3A4 induction → (1) DHA catabolism accelerated (already deficient in ZSD → worsens RP/neuropathy) + (2) hepatic microsomal burden (cholestatic liver ZS/NALD). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI — no universal adrenal insufficiency in PEX13-ZSD).",
            "alternative": "LEV first-line; LTG glucuronidation Level C (IRD); CLB adjunct. Replace with LEV on diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → works in X-ALD (ABCD1) because peroxisomal import is intact but ABCD1 transporter absent. In PEX13-ZSD, the IMPORTOMER CHANNEL is non-functional (PEX13 LOF → PEX5 cannot dock → all matrix import fails). Although the peroxisomal membrane IS present (unlike PEX3/PEX16/PEX19 LOF), LOO cannot restore importomer docking — VLCFA elongase inhibition is irrelevant without functional import.",
            "alternative": "DHA Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No VLCFA-specific therapy available 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX13-ZSD)",
            "reason": "HSCT benefits neuroinflammatory conditions (cerebral X-ALD). PEX13-ZSD is NOT inflammatory; pathology is metabolic (importomer dysfunction — PEX5 cannot dock). Donor microglia cannot restore PEX13 SH3 domain function or reconstitute the importomer docking channel. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit.",
            "alternative": "Supportive care: DHA, phytol-restricted diet, LEV. Gene therapy (AAV9-PEX13) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX13 = type II integral PMP with SH3 domain)",
            "reason": "ERT is feasible for soluble lysosomal enzymes delivered via mannose-6-phosphate receptor. PEX13 is a TYPE II INTEGRAL PEROXISOMAL MEMBRANE PROTEIN with an SH3 domain — not a soluble secreted enzyme. ERT cannot deliver an integral membrane protein to restore importomer function.",
            "alternative": "Gene therapy (AAV-PEX13) in preclinical research; mRNA delivery experimental. No clinically available ERT 2026.",
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
    key_concepts = [
        "PEX13 (403 aa, 2p15): type II integral peroxisomal membrane protein; two transmembrane segments (TMD1 aa ~155–175, TMD2 aa ~197–217); N-terminal cytoplasmic region; C-terminal SH3 domain (aa ~340–403) facing cytoplasm — part of the IMPORTOMER (PEX13·PEX14·PEX17 complex)",
        "PEX13 SH3 domain function: directly binds WxxxF/WxxxY motifs in PEX5 (PTS1 receptor) and PEX14 (importomer scaffold) — tethers PEX5 at the peroxisomal membrane surface during cargo translocation; SH3 domain is the molecular bridge between PEX5·cargo complex and the importomer",
        "Importomer docking/translocation complex (PEX13·PEX14·PEX17): forms the docking platform and translocation channel at the peroxisomal membrane for ALL matrix proteins — PTS1 cargo via PEX5 and PTS2 cargo via PEX7·PEX5 complex; PEX13 is essential for importomer stability and PEX5 docking efficiency",
        "PEX13 LOF consequence: importomer structurally compromised → PEX5 cannot efficiently dock → PTS1 import severely reduced; PEX7·PEX5 (PTS2 pathway) also fails secondarily → ALL matrix proteins fail → ALL peroxisomal metabolic pathways fail simultaneously",
        "CRITICAL IF DISTINCTION — membrane present in PEX13-ZSD: In PEX13 LOF, the peroxisomal MEMBRANE IS PRESENT and PMPs are detectable (PMP70+, ABCD1+ by IF) but the MATRIX IS EMPTY (catalase cytoplasmic). Compare: PEX3/PEX16/PEX19 LOF → membrane absent (ghost peroxisomes, no PMPs); PEX1/PEX6/PEX26 LOF → membrane present, PEX5 accumulates in membrane (retrotranslocation block); PEX13 LOF → membrane present, PEX5 cannot dock (importomer defect)",
        "PEX13 vs PEX14 importomer hierarchy: PEX14 is the central scaffold of the importomer (primary PEX5 docking site); PEX13 stabilises the complex via SH3-PEX14 interaction and provides a secondary PEX5 docking surface. PEX14 LOF is even rarer than PEX13 LOF. Both produce importomer collapse. PEX14 IF: PEX14 absent from membrane in PEX14 LOF but PRESENT on peroxisomal membranes in PEX13 LOF (key IF distinction within importomer tier)",
        "Biochemical identity: PEX13-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX13; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
        "DHA synthesis failure: PTS1-targeted DHA-synthesising enzymes cannot enter peroxisomes → DHA fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + neuropathy",
        "Phenotypic spectrum: ZS 35% (null/null) · NALD 45% · IRD 15% · Atypical 5%; ZS fraction (35%) equals PEX3 and PEX19 — no founder hypomorphic allele unlike PEX1-G843D or PEX6-R860W",
        "p.Ile384Thr (I384T): partial-function allele within the SH3 domain; reduces PEX5 binding efficiency ~50% → partial importomer function → partial PTS1 import → NALD/IRD phenotype as compound het with null allele",
        "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A) before any VPA consideration",
        "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH preferred for IS in all ZSD forms",
        "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; IV dextrose prevents catabolism; MANDATORY during any NPO",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute, unlike ABCD1): CYP3A4 induction depletes DHA + increases hepatic burden. No adrenal crisis in PEX13-ZSD. Different from ABCD1 ABSOLUTE CI",
        "Lorenzo's Oil mechanism mismatch in PEX13-ZSD: LOO inhibits VLCFA elongase → effective in ABCD1 (import intact). In PEX13-ZSD, the IMPORTOMER IS NON-FUNCTIONAL (PEX13 SH3 LOF → PEX5 cannot dock) — even though the membrane IS present, LOO cannot restore PEX5 docking → INEFFECTIVE. Different mechanism from PEX3/PEX16/PEX19 (membrane absent) but same inefficacy",
    ]

    diagnostic_algorithm = [
        "Step 1 — NBS flag: C26:0-lyso-PC elevated on DBS → refer to metabolic specialist STAT",
        "Step 2 — Plasma VLCFA panel: C26:0, C26:0/C22:0 ratio elevated → confirms ZSD biochemical group",
        "Step 3 — Erythrocyte plasmalogens: severely LOW → confirms PBD-ZSD (distinguishes from ABCD1 where plasmalogens NORMAL)",
        "Step 4 — Plasma phytanic + pristanic: both elevated (alpha-oxidation failed — confirms PTS1 + alpha-ox pathway failure)",
        "Step 5 — Plasma/urine pipecolic acid: elevated → confirms peroxisomal import failure",
        "Step 6 — Plasma DHA: LOW → confirms DHA synthesis failure; retinopathy + migration defect risk",
        "Step 7 — Bile acids (DHCA, THCA): elevated → confirms bile acid synthesis failure → cholestatic liver disease",
        "Step 8 — MRI brain: pachygyria + polymicrogyria in ZS (PATHOGNOMONIC); periventricular WM signal in NALD; mild/normal in IRD",
        "Step 9 — ERG + ophthalmology: ERG absent/severely reduced (ZS/NALD); RP in IRD; visual fields (Goldmann)",
        "Step 10 — Fibroblast IF (catalase + PMP70 + PEX14): catalase cytoplasmic (ghost peroxisome matrix import failure); PMP70 PRESENT on peroxisomal membrane (membrane biogenesis intact — distinguishes PEX13/importomer tier from PEX3/PEX16/PEX19 tier where PMP70 absent); PEX14 present (PEX14 is detectable in PEX13 LOF — distinguishes from PEX14 LOF where PEX14 absent)",
        "Step 11 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX13 biallelic pathogenic variants",
        "Step 12 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A)",
        "Step 13 — Genetic counselling: AR inheritance (25% recurrence risk); prenatal diagnosis available; carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC in ZSD: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE; PHT/CBZ/OXC = CYP3A4 → DHA depletion + hepatic burden → RELATIVE CI",
        "VPA vs LEV: VPA = THREE CI mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Never use VGB first-line in any ZSD form",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal import failure via intestinal absorption. Ethyl ester less effective (requires peroxisomal processing). Use LPC-DHA specifically",
        "Phytol-restricted diet in IRD vs no benefit in ZS: In ZS (fatal <12M), restriction has no benefit. In IRD/Atypical (long survival), phytol restriction reduces phytanic accumulation → fewer seizures + slower neuropathy",
        "Lorenzo's Oil in ABCD1 vs PEX13-ZSD: LOO effective in ABCD1 (intact peroxisome, transporter absent). In PEX13-ZSD, importomer non-functional (PEX5 cannot dock) → matrix import impossible → VLCFA oxidation cannot occur → LOO irrelevant → INEFFECTIVE",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (neuroinflammatory). PEX13-ZSD is NOT inflammatory → HSCT = transplant mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX13-ZSD): In ABCD1, PHT/CBZ → adrenal crisis (universal adrenal insufficiency + CYP induction). In PEX13-ZSD, no universal adrenal insufficiency → PHT/CBZ = RELATIVE CI (DHA/hepatic only, not adrenal)",
        "VPA POLG1 mandatory (CPIC Grade A): POLG1 exclusion mandatory in ZSD — catastrophic combined hepatotoxicity + mitochondrial depletion risk",
        "Carnitine supplementation: indicated if free carnitine <25 μmol/L; L-carnitine 50–100 mg/kg/day; critical if VPA inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY",
        "PEX13 vs PEX3/PEX16/PEX19 therapeutic implications: All LOF → same biochemical phenotype → same AED rules. Mechanistic tier differs (importomer defect vs membrane biogenesis defect). IF clue: PMP70 present in PEX13-ZSD; absent in PEX3/PEX16/PEX19-ZSD. Gene therapy approach would differ (AAV-PEX13 must restore importomer SH3 docking function vs AAV-PEX3/PEX16/PEX19 membrane biogenesis restoration)",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX1-ZSD (65% of all PBD — most common)",
            "distinction": "Biochemically IDENTICAL. PEX1 = D1 AAA-ATPase (retrotranslocation motor); PEX13 = importomer docking component. p.Gly843Asp founder allele (30% European) → IRD 30%. PEX13 rarer (~10–15 cases). IF clue: PMP70 present in PEX13-ZSD (membrane intact); PEX5 accumulates in membrane in PEX1-ZSD (retrotranslocation block). NGS distinguishes.",
        },
        {
            "condition": "PEX3-ZSD (~1–2% all PBD, ~10–20 cases worldwide)",
            "distinction": "Biochemically IDENTICAL. Key IF distinction: In PEX3 LOF, PMP70 ABSENT (membrane biogenesis failed — ghost peroxisomes, no PMPs). In PEX13 LOF, PMP70 PRESENT on peroxisomal membranes (importomer defect, membrane intact). Single IF panel with PMP70 staining can direct NGS tier (membrane defect vs import defect). NGS distinguishes. 6q24.2 vs 2p15.",
        },
        {
            "condition": "PEX16-ZSD (~3–5% all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX16 recruits PEX3 from ER vesicles — membrane biogenesis upstream. IF: PEX3 MISLOCALIZED to cytoplasm/ER in PEX16 LOF (unique clue). In PEX13 LOF, PEX3 correctly localised on peroxisomal membrane (membrane biogenesis intact). PMP70 absent in PEX16 LOF; present in PEX13 LOF. NGS distinguishes. 11p11.2 vs 2p15.",
        },
        {
            "condition": "PEX19-ZSD (~1–3% all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX19 is the cytoplasmic PMP chaperone (CAAX farnesylated). IF: PEX3 normally localised on ghost peroxisome in PEX19 LOF; PEX14 ABSENT (class I PMP failing insertion). In PEX13 LOF, PEX14 PRESENT on peroxisomal membrane (PEX14 is a PMP inserted by PEX19, which is functional in PEX13-ZSD). This PEX14 distinction (present vs absent) is the clearest IF clue separating PEX13 from PEX19 tier. NGS distinguishes.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 = D2 AAA-ATPase motor (retrotranslocation). p.Arg860Trp (R860W) founder allele → IRD 45%. PEX13 importomer defect; PEX6 retrotranslocation defect — different functional tiers. IF: PEX5 accumulates in membrane in PEX6-ZSD; PEX5 fails to dock in PEX13-ZSD. NGS distinguishes. 6p21.1 vs 2p15.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
            "distinction": "VLCFA elevated in BOTH. ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX13-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact peroxisome, transporter absent); INEFFECTIVE in PEX13-ZSD (importomer non-functional). HSCT standard CCALD; NOT in PEX13-ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL (key vs PEX13); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. Phytol-restricted diet Level A GOLD STANDARD. PHYH is a PTS2 cargo; its import requires PEX7 docking via PEX5·PEX7 complex — the PEX13 importomer is required. PEX13 LOF disrupts PHYH import among all matrix cargos.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "Plasmalogens (RBC) LOW (like ZSD) BUT VLCFA NORMAL + DHA NORMAL + pipecolic NORMAL. ONLY PTS2 pathway affected (PEX7 receptor deficient). Rhizomelia PATHOGNOMONIC. Stippled epiphyses on X-ray. No pachygyria. PEX7 is itself a PTS2 cargo imported via PEX5·PEX7 heterodimer — PEX13 importomer required for PEX7 import; PEX13 LOF disrupts ALL import (PTS1 + PTS2) vs PEX7 LOF disrupts only PTS2.",
        },
    ]

    standards = [
        "Gould SJ et al. Pex13p is an SH3 protein of the peroxisome membrane and a docking factor for the predominantly cytoplasmic PTs1 receptor. J Cell Biol. 1996",
        "Albertini M et al. The domain arrangement of the mPTS receptor in the peroxisome biogenesis. Science. 1997",
        "Dodt G et al. Mutations in the PTS1 receptor PEX5 in Zellweger syndrome. Nat Genet. 1995",
        "Liu Y et al. PEX13 mutations in Zellweger syndrome — rare complementation group 11 report. Mol Genet Metab. 2012",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
        "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "Honsho M et al. Peroxisome biogenesis: a mechanistic insight. FEBS Lett. 2020",
        "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614878 (PBD11A-ZSD — PEX13); #614879 (PBD11B-NALD — PEX13); *601789 (PEX13 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
