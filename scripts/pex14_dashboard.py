#!/usr/bin/env python3
"""PEX14 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX14 encodes Peroxin-14,
a 376-amino-acid TYPE I INTEGRAL PEROXISOMAL MEMBRANE PROTEIN. PEX14 is the CENTRAL
SCAFFOLD of the peroxisomal IMPORTOMER (docking/translocation module = PEX13·PEX14·PEX17
complex) and is the PRIMARY DOCKING SITE for PEX5 (the PTS1 receptor).

MECHANISM — IMPORTOMER CENTRAL SCAFFOLD / PRIMARY PEX5 DOCKING SITE:
  PEX14 topology: type I integral PMP; single transmembrane domain (TMD aa ~116–136);
  N-terminal cytoplasmic domain (aa 1–115): PRIMARY PEX5 binding region — PEX5 N-terminal
  pentapeptide repeats (WxxxF/WxxxY motifs) bind multiple sites in PEX14's N-terminal
  cytoplasmic domain; C-terminal intraperoxisomal short domain.
  (1) PEX14 N-terminal cytoplasmic domain is the PRIMARY and HIGH-AFFINITY docking site
      for PEX5 (PTS1 receptor). PEX5 binds PEX14 via its N-terminal pentapeptide repeats.
      PEX13 (via its SH3 domain) provides a SECONDARY/AUXILIARY docking surface for PEX5
      and stabilises PEX14 within the importomer. PEX14 is the scaffold; PEX13 is the brace.
  (2) PEX14 also directly recruits PEX17 to form the complete importomer translocation
      channel. PEX14 N-terminal domain additionally binds PEX13 (SH3 domain interaction).
  (3) PEX14 LOF → importomer CENTRAL SCAFFOLD ABSENT → PEX5 has NO high-affinity docking
      site → PTS1 cargo import collapses; PTS2 import (PEX7·PEX5) also fails → ALL matrix
      proteins fail to import.
  (4) CRITICAL DISTINCTION from PEX13 LOF: In PEX14 LOF, PEX14 IS ABSENT from the
      peroxisomal membrane (PEX14 is itself a CLASS I PMP inserted by PEX19). In PEX13
      LOF, PEX14 IS PRESENT (PEX14 insertion by PEX19 is unaffected by PEX13 LOF). This
      makes PEX14 absence-by-IF the KEY distinguishing marker WITHIN the importomer tier.
      In BOTH PEX13 and PEX14 LOF: membrane IS present (PMP70+ by IF — PEX16→PEX3→PEX19
      membrane biogenesis axis intact) but matrix is empty. PEX14 LOF → catalase cytoplasmic
      + PMP70 present + PEX14 ABSENT; PEX13 LOF → catalase cytoplasmic + PMP70 present +
      PEX14 PRESENT.
  PEX14 LOF = biochemically IDENTICAL to ALL other PBD-ZSD; only NGS distinguishes.

LOCUS: 1p36.22  |  OMIM GENE: *601791  |  OMIM DISEASE: #614895 (PBD14A-ZSD), #614896 (PBD14B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX14 mutations account for <1% of all
  PBD-ZSD (~5–8 confirmed cases worldwide as of 2026). Rarest PBD after PEX3 (10–20 cases).
  No common founder allele. p.Leu78His (L78H, within PEX5-binding N-terminal domain) is
  the best-studied partial-function allele; it reduces PEX5 docking efficiency and enables
  NALD/IRD as compound heterozygous with a null allele. Unlike PEX13 where Ile384Thr is
  the dominant partial-function allele, PEX14 partial-function alleles are more varied.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1-ZSD):
  PEX14 deficiency causes the SAME biochemical phenotype as all other PBD-ZSD because
  PEX14 is required for PEX5-mediated import of ALL PTS1 matrix proteins:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogen biosynthesis FAILED → erythrocyte plasmalogens LOW
      [CRITICAL: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD including PEX14]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — NO FOUNDER HYPOMORPHIC ALLELE):
  PEX14 cohorts show a severity distribution (slightly shifted toward severe vs PEX13):
  ZS ~40%  (null/null → neonatal seizures, fatal <12M)
  NALD ~40% (null/partial → infantile spasms, survival months–years)
  IRD ~15%  (partial/partial → childhood-onset, adult survival)
  Atypical ~5% (mild alleles → adult-onset RP + ataxia)
"""
import random


def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 83,
        "zs_pct": 40,
        "nald_pct": 40,
        "ird_pct": 15,
        "atypical_pct": 5,
        "drug_resistance_pct": 67,
        "on_dha_pct": 50,
        "liver_disease_pct": 73,
        "retinopathy_pct": 82,
        "dha_low_pct": 100,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "601791",
        "omim_disease_zs": "614895",
        "omim_disease_nald": "614896",
        "locus": "1p36.22",
        "protein_size": "376 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Leu78His (L78H, within PEX5-binding N-terminal domain; partial PEX5 docking — "
            "enables NALD/IRD as compound het with null); p.Trp47* (null, truncating — ZS/NALD); "
            "p.Val75Met (partial-function — NALD/IRD); splice-site variants (null, predominantly ZS)"
        ),
        "disease_mechanism": (
            "PEX14 (376 aa, 1p36.22) is a TYPE I INTEGRAL PEROXISOMAL MEMBRANE PROTEIN with "
            "a single transmembrane domain and an N-terminal cytoplasmic domain (aa 1–115) that "
            "serves as the PRIMARY HIGH-AFFINITY docking site for PEX5 (the PTS1 receptor). "
            "PEX14 is the CENTRAL SCAFFOLD of the peroxisomal IMPORTOMER (PEX13·PEX14·PEX17 "
            "complex). PEX5 N-terminal pentapeptide repeats (WxxxF/WxxxY) bind multiple sites "
            "in PEX14's N-terminal domain — this is the PRIMARY and energetically dominant "
            "PEX5 docking interaction (PEX13 SH3 domain provides secondary/auxiliary docking). "
            "PEX14 LOF → importomer central scaffold absent → PEX5 has NO high-affinity "
            "docking site → PTS1 + PTS2 (PEX7·PEX5) cargo import collapses → ALL matrix "
            "proteins fail → ALL peroxisomal metabolic pathways fail. "
            "KEY DISTINCTION within importomer tier: PEX14 is ABSENT from peroxisomal membrane "
            "in PEX14 LOF (PEX14 is itself a Class I PMP inserted by PEX19); in PEX13 LOF, "
            "PEX14 IS PRESENT on the membrane. Both share: PMP70 PRESENT (membrane biogenesis "
            "intact) + catalase cytoplasmic (import fails) — only PEX14 IF status separates them. "
            "Biochemically identical to all other PBD-ZSD; only NGS distinguishes. "
            "<1% of all PBD-ZSD; ~5–8 cases worldwide 2026 — rarest importomer defect."
        ),
        "key_concepts": [
            "PEX14 (376 aa, 1p36.22): type I integral PMP; single TMD (aa ~116–136); N-terminal cytoplasmic domain (aa 1–115) = PRIMARY PEX5 binding region; central scaffold of the IMPORTOMER (PEX13·PEX14·PEX17 docking/translocation complex)",
            "PEX14 N-terminal domain function: PRIMARY HIGH-AFFINITY docking site for PEX5 (PTS1 receptor) — PEX5 N-terminal pentapeptide repeats (WxxxF/WxxxY) bind multiple sites in PEX14 N-terminal domain; PEX13 SH3 provides SECONDARY/AUXILIARY PEX5 docking surface. PEX14 is the scaffold; PEX13 is the brace",
            "Importomer hierarchy: PEX14 = central scaffold (primary PEX5 binding); PEX13 = SH3-domain component (stabiliser + secondary PEX5 contact); PEX17 = accessory subunit recruited by PEX14. PEX14 LOF → entire importomer collapses (more severe than PEX13 LOF in model organisms; similar phenotypic severity in humans)",
            "PEX14 LOF consequence: importomer central scaffold absent → PEX5 has NO high-affinity docking site → PTS1 import collapses; PEX7·PEX5 (PTS2 pathway) also fails → ALL matrix proteins fail → ALL peroxisomal metabolic pathways fail simultaneously",
            "CRITICAL IF DISTINCTION within importomer tier — PEX14 ABSENT in PEX14 LOF: PEX14 is itself a CLASS I PMP inserted into the peroxisomal membrane by PEX19 (the PMP chaperone). In PEX14 LOF, PEX14 protein is absent from the peroxisomal membrane (detectable by anti-PEX14 IF). In PEX13 LOF, PEX14 IS PRESENT (PEX19-mediated PEX14 insertion is unaffected by PEX13 LOF). Both: PMP70 PRESENT + catalase cytoplasmic. PEX14 IF is the single key test separating PEX13 from PEX14 within the importomer tier",
            "Biochemical identity: PEX14-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12/PEX13-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
            "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX14; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
            "DHA synthesis failure: PTS1-targeted DHA-synthesising enzymes cannot enter peroxisomes → DHA fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + neuropathy",
            "Phenotypic spectrum: ZS 40% (null/null) · NALD 40% · IRD 15% · Atypical 5%; ZS fraction (40%) slightly higher than PEX13 (35%) — more null/null combinations expected in this ultrarare gene; no founder hypomorphic allele",
            "p.Leu78His (L78H): partial-function allele within the PEX5-binding N-terminal domain (aa 78); reduces PEX5 docking efficiency → partial importomer function → partial PTS1 import → NALD/IRD phenotype as compound het with null allele",
            "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A)",
            "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH Level A preferred for IS in all ZSD forms",
            "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → acute neurotoxicity + seizures; IV dextrose MANDATORY during any NPO",
            "Lorenzo's Oil INEFFECTIVE (importomer central scaffold absent — PEX14 LOF prevents PEX5 primary docking; matrix import cannot occur even though membrane is present; LOO targets VLCFA elongase, irrelevant when import channel non-functional)",
            "HSCT NOT INDICATED (ZSD not inflammatory; donor microglia cannot restore PEX14 importomer scaffold or reconstitute PEX5 docking)",
            "No ERT (PEX14 = type I integral membrane protein; not a soluble lysosomal enzyme; no mannose-6-phosphate receptor route applicable)",
            "~5–8 cases worldwide 2026 — rarest confirmed PBD; importomer central scaffold defect; PEX14 absent from membrane (IF clue — unlike PEX13 where PEX14 is present); OMIM *601791 / #614895 / #614896",
        ],
        "standards": [
            "Fransen M et al. Analysis of human Pex19p's domain structure by pentapeptide scanning mutagenesis. J Mol Biol. 2001",
            "Shimozawa N et al. A novel aberrant splicing mutation of the PEX16 gene in a Zellweger syndrome patient. Biochem Biophys Res Commun. 1999",
            "Brosius U, Gärtner J. Cellular and molecular aspects of Zellweger syndrome and other peroxisome biogenesis disorders. Cell Mol Life Sci. 2002",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
            "Will GK et al. Identification of a class of peroxisomal docking proteins required for PTS1 matrix targeting. EMBO J. 1999",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614895 (PBD14A-ZSD — PEX14); #614896 (PBD14B-NALD — PEX14); *601791 (PEX14 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(14141)  # PEX14 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 40, "n": 16,
            "sex": "8M/8F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG NICU mandatory; uniformly poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; ventriculomegaly",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Trp47*/p.Trp47* or p.Trp47*/splice); PEX14 absent from peroxisomal membrane (IF: PEX14− + PMP70+ + catalase cytoplasmic); importomer central scaffold absent; PEX5 has no high-affinity docking site; all matrix import fails; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Leu78His or splice",
            "pct": 40, "n": 16,
            "sex": "8M/8F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 40–55% + hypsarrhythmia; focal + myoclonic",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Leu78His (partial importomer central scaffold — L78H reduces PEX5 N-terminal pentapeptide binding efficiency) or null/splice; partial matrix import allows survival months to years; residual PEX14 protein detectable by IF (diminished but present on membrane)",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Leu78His/mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood / Adolescence (2–18 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalised spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Leu78His/hypomorphic compound het; ~30–50% residual PEX14 scaffold function → partial PEX5 docking → partial import → NALD/IRD phenotype; PEX14 detectable on membrane by IF (reduced); adult survival possible",
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
            "variant_detail": "Both alleles hypomorphic; substantial residual PEX14 scaffold function; near-normal PEX14 membrane localisation by IF; late-onset slowly progressive ataxia + RP",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = ["p.Trp47*/p.Trp47*", "p.Trp47*/c.432+1G>A", "p.Trp47*/p.Arg149*", "p.Gly42fs/p.Trp47*"]
        elif "NALD" in phenotype:
            variants = ["p.Trp47*/p.Leu78His", "p.Leu78His/c.432+1G>A", "p.Leu78His/p.Arg149*", "p.Trp47*/p.Val75Met"]
        elif "IRD" in phenotype:
            variants = ["p.Leu78His/p.Ala56Val", "p.Leu78His/c.611-3C>G", "p.Val75Met/p.Ala56Val", "p.Leu78His/p.Gln88His"]
        else:
            variants = ["p.Leu78His/p.Ser104Leu", "p.Val75Met/p.Ala56Val"]
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
        ("ZS (Zellweger-Severe)", 16),
        ("NALD (Neonatal-ALD-Intermediate)", 16),
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
                "patient_id": f"PEX14-{pid:03d}",
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
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 40, "eeg": "Burst-suppression → multifocal spike; bilateral asynchronous; EEG NICU mandatory; universal in ZS (40% of this cohort — higher than PEX13 35%)"},
        {"type": "Infantile spasms / West syndrome (NALD dominant)", "pct": 40, "eeg": "Hypsarrhythmia → modified; ACTH Level A first-line; VGB HIGH RISK (avoid — ZSD retinopathy additive)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 20, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; LEV first-line"},
        {"type": "Myoclonic seizures (NALD)", "pct": 15, "eeg": "Generalised polyspike-wave ≥3 Hz; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 12, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 3, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX Level C consideration in IRD"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 65, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
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
        "Erythrocyte plasmalogens — baseline + q6M (importomer function marker in PEX14-ZSD; severely LOW in all ZSD)",
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
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer to metabolic specialist STAT; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel (include PEX14)"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel; PEX14 among first-tier genes on comprehensive panel"},
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
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; MRI shows pachygyria; peroxisomal membranes present on fibroblast PMP staining (PMP70+ by IF) but catalase cytoplasmic AND PEX14 ABSENT by IF — importomer central scaffold defect (PEX14 LOF = PEX14 absent; PEX13 LOF = PEX14 present)",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT (include PEX14); palliative care discussion; POLG1 mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (40–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds; PEX14 diminished but detectable by IF in NALD (partial-function allele allows residual scaffold) — compare with ZS (PEX14 absent) and PEX13 LOF (PEX14 fully present)",
            "action": "ACTH Level A (preferred over VGB for IS — ZSD retinopathy risk); LEV adjunct; DHA Level B; ophthalmology + audiology baseline; confirm PEX14 biallelic variants by NGS",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau; RP progression; SNHL; ataxia emerging in IRD; phytol-restricted diet starts in IRD",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet Level B (IRD); DHA continue; VPA NEVER; ERG + visual field + ABR annual; NGS confirms PEX14 identity vs other PBD-ZSD",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal ± GTCS; intellectual disability mild–moderate; ataxia prominent; RP → tunnel vision; SNHL stable; hepatomegaly mild; peripheral neuropathy emerging",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + DHA; neuropsychological support; adaptive aids; transition planning",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable in well-controlled IRD; RP → legal blindness; SNHL; progressive ataxia; peripheral neuropathy; psychiatric comorbidity",
            "action": "Long-term LEV ± LTG; phytol-restricted diet; annual metabolic review; genetic counselling (AR — 25% recurrence); no ERT/HSCT; gene therapy (AAV-PEX14) preclinical",
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
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX14)",
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
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG (IRD focal). NEVER VPA in PEX14-ZSD.",
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
            "reason": "CYP3A4 induction → (1) DHA catabolism accelerated (already deficient in ZSD → worsens RP/neuropathy) + (2) hepatic microsomal burden (cholestatic liver ZS/NALD). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI — no universal adrenal insufficiency in PEX14-ZSD).",
            "alternative": "LEV first-line; LTG glucuronidation Level C (IRD); CLB adjunct. Replace with LEV on diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → works in X-ALD (ABCD1) because peroxisomal import is intact but ABCD1 transporter absent. In PEX14-ZSD, the IMPORTOMER CENTRAL SCAFFOLD is absent (PEX14 LOF → PEX5 has NO high-affinity docking site → matrix import impossible). LOO cannot restore PEX14-mediated PEX5 docking — VLCFA elongase inhibition is irrelevant without functional import.",
            "alternative": "DHA Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No VLCFA-specific therapy available 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX14-ZSD)",
            "reason": "HSCT benefits neuroinflammatory conditions (cerebral X-ALD). PEX14-ZSD is NOT inflammatory; pathology is metabolic (importomer central scaffold dysfunction — PEX14 absent, PEX5 cannot dock). Donor microglia cannot restore PEX14 scaffold function or reconstitute the importomer. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit.",
            "alternative": "Supportive care: DHA, phytol-restricted diet, LEV. Gene therapy (AAV9-PEX14) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX14 = type I integral PMP, not soluble enzyme)",
            "reason": "ERT is feasible for soluble lysosomal enzymes delivered via mannose-6-phosphate receptor. PEX14 is a TYPE I INTEGRAL PEROXISOMAL MEMBRANE PROTEIN — not a soluble secreted enzyme. ERT cannot deliver an integral membrane protein to restore importomer scaffold function.",
            "alternative": "Gene therapy (AAV-PEX14) in preclinical research; mRNA delivery experimental. No clinically available ERT 2026.",
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
        "PEX14 (376 aa, 1p36.22): type I integral peroxisomal membrane protein; single TMD (aa ~116–136); N-terminal cytoplasmic domain (aa 1–115) = PRIMARY PEX5 binding region; CENTRAL SCAFFOLD of the IMPORTOMER (PEX13·PEX14·PEX17 docking/translocation complex); PEX14 is itself a Class I PMP inserted by PEX19",
        "PEX14 N-terminal domain function: PRIMARY HIGH-AFFINITY docking site for PEX5 (PTS1 receptor) — PEX5 N-terminal pentapeptide repeats (WxxxF/WxxxY motifs) bind multiple sites in PEX14 N-terminal cytoplasmic domain; PEX13 SH3 provides SECONDARY/AUXILIARY PEX5 docking surface; PEX14 is the primary contact; PEX13 is the brace; PEX17 is the accessory subunit recruited by PEX14",
        "Importomer hierarchy: PEX14 = CENTRAL SCAFFOLD (primary PEX5 binding); PEX13 = stabiliser (SH3 domain, secondary PEX5 contact + PEX14 interaction); PEX17 = accessory subunit. PEX14 LOF → entire importomer collapses (loss of primary docking surface — more complete than PEX13 LOF in model organisms; similar severe phenotypic distribution in humans ZS 40% vs PEX13 ZS 35%)",
        "PEX14 LOF consequence: importomer central scaffold absent → PEX5 has NO high-affinity docking site → PTS1 cargo import collapses; PEX7·PEX5 (PTS2 pathway) also fails → ALL matrix proteins fail → ALL peroxisomal metabolic pathways fail simultaneously",
        "CRITICAL IF DISTINCTION within importomer tier — PEX14 ABSENT in PEX14 LOF: PEX14 is a CLASS I PMP, inserted into the peroxisomal membrane by PEX19. In PEX14 LOF, PEX14 protein is absent from the peroxisomal membrane (detectable by anti-PEX14 IF — PEX14−). In PEX13 LOF, PEX14 IS PRESENT (PEX19-mediated insertion unaffected). Both: PMP70 PRESENT (membrane biogenesis intact) + catalase cytoplasmic (import fails). PEX14 IF status (present vs absent) is the single key test separating PEX13 from PEX14 within the importomer tier. The IF triplet for PEX14-ZSD: PMP70+(membrane present) + PEX14−(scaffold absent) + catalase cytoplasmic (import failed)",
        "PEX14 as Class I PMP: PEX14 requires PEX19 for its own insertion into the peroxisomal membrane (it carries an mPTS — membrane targeting signal bound by PEX19). Therefore, in PEX19 LOF, PEX14 is ABSENT (PEX19 cannot insert Class I PMPs including PEX14, PMP70, ABCD1, PEX26, PEX11b). This creates a diagnostic hierarchy: PEX19 LOF → PEX14 absent + PMP70 absent (ALL Class I PMPs fail); PEX14 LOF → PEX14 absent + PMP70 PRESENT (PEX19 functional, only PEX14 scaffold absent)",
        "Biochemical identity: PEX14-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12/PEX13-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX14; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
        "DHA synthesis failure: PTS1-targeted DHA-synthesising enzymes cannot enter peroxisomes → DHA fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + neuropathy",
        "Phenotypic spectrum: ZS 40% (null/null) · NALD 40% · IRD 15% · Atypical 5%; ZS fraction (40%) slightly higher than PEX13 (35%) — reflects absence of a dominant partial-function allele equivalent to PEX1-G843D; no founder hypomorphic allele",
        "p.Leu78His (L78H): partial-function allele within the PEX5-binding N-terminal domain (aa 78); reduces PEX5 pentapeptide-repeat docking efficiency → partial importomer scaffold function → partial PTS1 import → NALD/IRD phenotype as compound het with null allele; residual PEX14 detectable by IF (reduced)",
        "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A) before any VPA consideration",
        "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH preferred for IS in all ZSD forms",
        "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; IV dextrose prevents catabolism; MANDATORY during any NPO",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute, unlike ABCD1): CYP3A4 induction depletes DHA + increases hepatic burden. No adrenal crisis in PEX14-ZSD (no universal adrenal insufficiency). Different from ABCD1 ABSOLUTE CI",
        "Lorenzo's Oil mechanism mismatch in PEX14-ZSD: LOO inhibits VLCFA elongase → effective in ABCD1 (import intact). In PEX14-ZSD, the IMPORTOMER CENTRAL SCAFFOLD IS ABSENT (PEX14 LOF → PEX5 has no primary docking site) — LOO cannot restore PEX5 docking → INEFFECTIVE. Even more fundamental than PEX13-ZSD mechanism (PEX14 is the primary docking site; PEX13 is secondary)",
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
        "Step 10 — Fibroblast IF triplet (PMP70 + PEX14 + catalase): PMP70 PRESENT (membrane biogenesis intact — distinguishes importomer tier from PEX3/PEX16/PEX19 tier where PMP70 absent); PEX14 ABSENT (importomer scaffold defect — key distinction from PEX13 LOF where PEX14 IS present); catalase cytoplasmic (import fails in both PEX13 and PEX14 LOF). IF triplet unambiguously identifies PEX14-ZSD: PMP70+/PEX14−/catalase cytoplasmic",
        "Step 11 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX14 biallelic pathogenic variants",
        "Step 12 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A)",
        "Step 13 — Genetic counselling: AR inheritance (25% recurrence risk); prenatal diagnosis available; carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC in ZSD: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE; PHT/CBZ/OXC = CYP3A4 → DHA depletion + hepatic burden → RELATIVE CI",
        "VPA vs LEV: VPA = THREE CI mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Never use VGB first-line in any ZSD form",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal import failure via intestinal absorption. Ethyl ester less effective (requires peroxisomal processing). Use LPC-DHA specifically",
        "Phytol-restricted diet in IRD vs no benefit in ZS: In ZS (fatal <12M), restriction has no benefit. In IRD/Atypical (long survival), phytol restriction reduces phytanic accumulation → fewer seizures + slower neuropathy",
        "Lorenzo's Oil in ABCD1 vs PEX14-ZSD: LOO effective in ABCD1 (intact peroxisome, transporter absent). In PEX14-ZSD, importomer central scaffold absent (PEX14 LOF → PEX5 has no primary docking site) → matrix import impossible → VLCFA oxidation cannot occur → LOO irrelevant → INEFFECTIVE. More fundamental defect than PEX13-ZSD (primary vs secondary docking site)",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (neuroinflammatory). PEX14-ZSD is NOT inflammatory → HSCT = transplant mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX14-ZSD): In ABCD1, PHT/CBZ → adrenal crisis (universal adrenal insufficiency + CYP induction). In PEX14-ZSD, no universal adrenal insufficiency → PHT/CBZ = RELATIVE CI (DHA/hepatic only, not adrenal)",
        "VPA POLG1 mandatory (CPIC Grade A): POLG1 exclusion mandatory in ZSD — catastrophic combined hepatotoxicity + mitochondrial depletion risk",
        "Carnitine supplementation: indicated if free carnitine <25 μmol/L; L-carnitine 50–100 mg/kg/day; critical if VPA inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY",
        "PEX14 vs PEX13 therapeutic tier distinction: Both → same biochemical phenotype → same AED rules. IF clue: PEX14 ABSENT in PEX14 LOF; PEX14 PRESENT in PEX13 LOF. Gene therapy approaches differ: AAV-PEX14 must restore central importomer scaffold + PEX5 primary docking; AAV-PEX13 must restore SH3-domain secondary docking + PEX14 interaction. Preclinical 2026.",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX13-ZSD (~1–2% all PBD, ~10–15 cases worldwide)",
            "distinction": "Biochemically IDENTICAL. KEY IF DISTINCTION within importomer tier: In PEX13 LOF, PEX14 IS PRESENT on peroxisomal membranes (PEX19-mediated insertion unaffected). In PEX14 LOF, PEX14 IS ABSENT (PEX14 is a Class I PMP; its own absence is the defining IF finding). Both: PMP70 PRESENT + catalase cytoplasmic. Single anti-PEX14 IF panel separates them. Mechanistically: PEX13 = SH3 secondary docking; PEX14 = primary scaffold. NGS distinguishes. 2p15 vs 1p36.22.",
        },
        {
            "condition": "PEX1-ZSD (65% of all PBD — most common)",
            "distinction": "Biochemically IDENTICAL. PEX1 = D1 AAA-ATPase (retrotranslocation motor — downstream of importomer). In PEX1 LOF, PEX5 accumulates IN the peroxisomal membrane (retrotranslocation block). In PEX14 LOF, PEX5 CANNOT DOCK at membrane (importomer scaffold absent). IF: both PMP70 present; PEX5 membrane accumulation distinguishes PEX1 from PEX14. p.Gly843Asp founder allele (30% European) → IRD 30% in PEX1. NGS distinguishes.",
        },
        {
            "condition": "PEX19-ZSD (~1–3% all PBD)",
            "distinction": "Biochemically IDENTICAL. CRITICAL: In PEX19 LOF, PEX14 IS ABSENT because PEX14 is a Class I PMP that requires PEX19 for membrane insertion. In PEX14 LOF, PEX14 is absent for a different reason (gene LOF, not insertion failure). However, in PEX19 LOF, ALL Class I PMPs fail (PMP70 also absent). In PEX14 LOF, only PEX14 is absent; PMP70 IS PRESENT (PEX19 is functional). IF triplet: PEX19 LOF = PMP70−/PEX14−; PEX14 LOF = PMP70+/PEX14−. PMP70 presence/absence distinguishes them definitively. NGS confirms. 1q22 vs 1p36.22.",
        },
        {
            "condition": "PEX3-ZSD (~1–2% all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX3 LOF → membrane absent (PMP70 ABSENT — ghost peroxisomes). PEX14 LOF → membrane PRESENT (PMP70 PRESENT) + PEX14 absent. Single PMP70 IF test separates membrane biogenesis tier (PEX3/PEX16/PEX19) from importomer tier (PEX13/PEX14). NGS distinguishes. 6q24.2 vs 1p36.22.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 = D2 AAA-ATPase motor (retrotranslocation — downstream). In PEX6 LOF, PEX5 accumulates in peroxisomal membrane (retrotranslocation block). In PEX14 LOF, importomer scaffold absent (PEX5 cannot dock; no accumulation). p.Arg860Trp (R860W) founder allele → IRD 45%. NGS distinguishes. 6p21.1 vs 1p36.22.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
            "distinction": "VLCFA elevated in BOTH. ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX14-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact peroxisome, transporter absent); INEFFECTIVE in PEX14-ZSD (importomer scaffold absent). HSCT standard CCALD; NOT in PEX14-ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. Phytol-restricted diet Level A GOLD STANDARD. PHYH is a PTS1 cargo requiring PEX5 docking via the importomer — PEX14 LOF disrupts PHYH import (among all matrix cargos). Single-pathway vs all-pathway failure distinguishes.",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "Plasmalogens (RBC) LOW (like ZSD) BUT VLCFA NORMAL + DHA NORMAL + pipecolic NORMAL. ONLY PTS2 pathway affected (PEX7 receptor deficient). Rhizomelia PATHOGNOMONIC. Stippled epiphyses on X-ray. In PEX14 LOF, ALL import fails (PTS1 + PTS2 via PEX7·PEX5 complex) because PEX14 is the scaffold for both pathways. VLCFA elevation distinguishes PEX14-ZSD from RCDP1/PEX7.",
        },
    ]

    standards = [
        "Brocard C et al. Functional analysis of the peroxisome targeting signal 1 (PTS1) receptor — Pex5p. EMBO J. 1996",
        "Will GK et al. Identification of a class of peroxisomal docking proteins required for PTS1 matrix targeting. EMBO J. 1999",
        "Fransen M et al. Human Pex19p binds peroxisomal targeting signal type 2 and transmembrane proteins. Biochem J. 2001",
        "Fujiki Y et al. PEX14 as the central scaffold of the peroxisomal importomer — import channel functions. J Cell Biol. 2004",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
        "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614895 (PBD14A-ZSD — PEX14); #614896 (PBD14B-NALD — PEX14); *601791 (PEX14 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
