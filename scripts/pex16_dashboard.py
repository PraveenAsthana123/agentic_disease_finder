#!/usr/bin/env python3
"""PEX16 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX16 encodes Peroxin-16,
a 336-amino-acid TYPE II INTEGRAL PEROXISOMAL MEMBRANE PROTEIN delivered from the ER.
PEX16 is a KEY STRUCTURAL CO-FACTOR in the PEX3-PEX16-PEX19 early membrane biogenesis
axis — it recruits PEX3 from ER-derived vesicles to the nascent peroxisomal membrane and
stabilises the PEX3-PEX19 docking platform.

MECHANISM — ER-TO-PEROXISOME DELIVERY / PEX3 RECRUITMENT FACTOR:
  PEX16 is delivered from the ER to peroxisomes via ER-derived vesicular transport
  (the ER-to-peroxisome membrane biogenesis pathway, independent of the classical
  PEX19-mediated class I PMP insertion route):
  (1) PEX16 topology: type II integral membrane protein; inserts into the peroxisomal
      membrane from ER-derived pre-peroxisomal vesicles; N-terminal cytoplasmic segment,
      single transmembrane domain, C-terminal luminal/intermembrane segment.
  (2) PEX16 function — PEX3 recruitment: PEX16 acts as a membrane-embedded receptor
      that recruits PEX3 (together with PEX19) from ER-derived vesicles to the
      peroxisomal membrane. Without PEX16, PEX3 is misrouted — PEX3 accumulates in the
      cytoplasm or ER rather than reaching the peroxisomal membrane.
  (3) PEX16 LOF → PEX3 mislocalized → PEX19 cannot dock onto the peroxisomal surface
      → ALL peroxisomal membrane proteins (PMPs) fail to insert → peroxisomal membranes
      absent or "ghost" (lipid bilayer only, no PMP content) → matrix protein import
      secondarily fails entirely → ALL peroxisomal metabolic pathways fail simultaneously.
  (4) Positional context: PEX16 acts BEFORE or CO-OPERATIVELY WITH PEX3 — PEX16 brings
      PEX3 to the membrane; PEX3 then serves as the PEX19 docking receptor. PEX16 LOF
      is therefore at the same mechanistic tier as PEX3 LOF — both are upstream of PEX26,
      PEX1-PEX6, and PEX2-PEX10-PEX12.
  PEX16 LOF = biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX2/PEX10/PEX12/PEX26-ZSD.

LOCUS: 11p11.2  |  OMIM GENE: *603360  |  OMIM DISEASE: #614876 (PBD8A-ZSD), #614861 (PBD8B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX16 mutations account for ~3–5% of all
  PBD-ZSD (~30–60 confirmed cases worldwide as of 2026). More common than PEX3 (~10–20 cases)
  and PEX26 (~15–25 cases). No common founder allele (unlike PEX1-G843D ~30% European,
  PEX6-R860W ~18% European). p.Arg176Gln (R176Q) is the best-studied partial-function allele;
  it reduces PEX16 membrane insertion efficiency and enables a milder NALD/IRD phenotype when
  compound heterozygous with a null allele.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1-ZSD):
  PEX16 deficiency causes the SAME biochemical phenotype as all other PBD-ZSD because PEX16
  is required to recruit PEX3 to the peroxisomal membrane. Without PEX16→PEX3→PEX19 axis:
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
  (3) Plasmalogens BIOSYNTHESIS FAILED → erythrocyte plasmalogens LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD including PEX16]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — NO FOUNDER HYPOMORPHIC ALLELE):
  PEX16 cohorts show a severity distribution:
  ZS ~30%  (null/null → neonatal seizures, fatal <12M)
  NALD ~50% (null/partial → infantile spasms, survival months–years)
  IRD ~15%  (partial/partial → childhood-onset, adult survival)
  Atypical ~5% (mild alleles → adult-onset RP + ataxia)
  ZS fraction (30%) is slightly less than PEX3 (35%) due to more hypomorphic alleles reaching IRD.
"""
import random


def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 82,
        "zs_pct": 30,
        "nald_pct": 50,
        "ird_pct": 15,
        "atypical_pct": 5,
        "drug_resistance_pct": 64,
        "on_dha_pct": 55,
        "liver_disease_pct": 74,
        "retinopathy_pct": 80,
        "dha_low_pct": 100,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "603360",
        "omim_disease_zs": "614876",
        "omim_disease_nald": "614861",
        "locus": "11p11.2",
        "protein_size": "336 aa",
        "nbs_positive_rate": "~98% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Arg176Gln (R176Q, within transmembrane/cytoplasmic interface; reduces PEX16 membrane "
            "stability; partial-function allele enabling NALD/IRD in compound het with null); "
            "p.Trp313* (null, truncating — ZS/NALD); p.Leu307Pro (near C-terminal; partial function); "
            "frameshift/splice-site variants (null, predominantly ZS)"
        ),
        "disease_mechanism": (
            "PEX16 (336 aa, 11p11.2) is a TYPE II INTEGRAL PEROXISOMAL MEMBRANE PROTEIN delivered "
            "from the ER via ER-to-peroxisome vesicular transport. PEX16's key function is to RECRUIT "
            "PEX3 from ER-derived vesicles to the nascent peroxisomal membrane — acting as a membrane-"
            "embedded co-factor in the PEX3-PEX16-PEX19 early membrane biogenesis axis. "
            "PEX16 LOF → PEX3 mislocalized (accumulates in cytoplasm/ER, cannot reach peroxisomal "
            "membrane) → PEX19 (cytosolic PMP chaperone) cannot dock → ALL peroxisomal membrane "
            "proteins (PMPs) fail to insert → peroxisomal membranes absent or 'ghost' (lipid only) → "
            "ALL peroxisomal metabolic pathways fail simultaneously. "
            "PEX16 acts at the SAME mechanistic tier as PEX3 — upstream of PEX26, PEX1/PEX6, and "
            "PEX2/PEX10/PEX12. Biochemically identical to all other PBD-ZSD; only NGS distinguishes. "
            "~3–5% of all PBD-ZSD; ~30–60 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX16 (336 aa, 11p11.2): type II integral peroxisomal membrane protein; delivered from ER to peroxisome via vesicular ER-to-peroxisome pathway; acts UPSTREAM of PEX19-mediated PMP insertion",
            "PEX16 function — PEX3 recruitment: PEX16 embedded in peroxisomal membrane recruits PEX3 (with PEX19) from ER-derived vesicles → PEX3 is correctly positioned as the PEX19 docking receptor on the peroxisomal surface",
            "PEX3-PEX16-PEX19 early biogenesis axis: PEX16 brings PEX3 to the membrane; PEX3 then serves as the docking receptor for PEX19; PEX19 carries new PMPs to insert into the peroxisomal membrane (class I PMP insertion pathway)",
            "PEX16 LOF outcome: PEX3 mislocalized → PEX19 cannot dock → ALL PMPs fail to insert → peroxisomes absent or 'ghost vesicles' (lipid bilayer only, no protein content) → ALL peroxisomal matrix import secondarily fails",
            "ER-to-peroxisome membrane biogenesis: PEX16 (and PEX3) travel to peroxisomes via ER-derived pre-peroxisomal vesicles — a mechanistically distinct route from the PEX19-mediated class I PMP insertion of downstream peroxins",
            "Biochemical identity: PEX16-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX2/PEX10/PEX12/PEX26-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
            "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX16; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
            "DHA synthesis failure: peroxisomal membrane absent → DHA synthesis fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + peripheral neuropathy",
            "Phenotypic spectrum: ZS 30% (null/null) · NALD 50% · IRD 15% · Atypical 5%; ZS fraction slightly lower than PEX3 (35%) — p.Arg176Gln allele enables more NALD/IRD survivors",
            "p.Arg176Gln (R176Q): partial-function allele at transmembrane/cytoplasmic interface; reduces PEX16 membrane stability → partial PEX3 recruitment → partial PEX19 docking → partial PMP insertion → NALD/IRD phenotype as compound het with null",
            "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A)",
            "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH Level A preferred for IS in all ZSD",
            "Fasting EXTREME HAZARD: adipose phytanic release during starvation → acute neurotoxicity + seizures; IV dextrose MANDATORY during any NPO",
            "Lorenzo's Oil INEFFECTIVE (peroxisomal membrane absent — PEX16 LOF prevents PEX3 recruitment; membrane cannot form; LOO targets VLCFA elongase, irrelevant when no peroxisomal membrane exists)",
            "HSCT NOT INDICATED (ZSD not inflammatory; donor microglia cannot restore PEX16 recruitment of PEX3 or reconstitute peroxisomal membrane biogenesis)",
            "No ERT (PEX16 = type II integral membrane protein delivered via ER vesicles; not a soluble enzyme amenable to mannose-6-phosphate receptor-mediated ERT)",
            "~30–60 cases worldwide 2026 (rarer than PEX1/PEX6; more common than PEX3/PEX26); PART OF PEX3-PEX16-PEX19 early membrane biogenesis axis",
        ],
        "standards": [
            "Honsho M et al. Peroxisome biogenesis: a mechanistic insight. FEBS Lett. 2020",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
            "South ST et al. Identification of the human PEX16 gene, mutated in complement group D of the peroxisome biogenesis disorders. Nat Genet. 2000",
            "Honsho M et al. Disruption of peroxisomal membrane homeostasis by mutations in Pex16p, a pleiotropic functions protein involved in peroxisome division. J Biol Chem. 1998",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "Kim PK et al. The origin and maintenance of mammalian peroxisomes involves a de novo PEX16-dependent pathway from the ER. J Cell Biol. 2006",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614876 (PBD8A-ZSD — PEX16); #614861 (PBD8B-NALD — PEX16); *603360 (PEX16 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(16163)  # PEX16 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 30, "n": 12,
            "sex": "6M/6F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression",
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG NICU mandatory; uniformly poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; ventriculomegaly",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Trp313*/p.Trp313* or p.Trp313*/frameshift/splice); PEX16 absent from membrane; PEX3 mislocalized to cytosol; PEX19 cannot dock; ghost peroxisomes; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Arg176Gln or splice",
            "pct": 50, "n": 20,
            "sex": "10M/10F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 40–55% + hypsarrhythmia; focal + myoclonic",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Arg176Gln (partial PEX16 stability) or null/splice; p.Arg176Gln at TM/cytoplasmic interface reduces PEX3 recruitment ~55%; partial PEX19 docking; survival months to years",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Arg176Gln/mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood / Adolescence (2–18 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalised spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Arg176Gln/hypomorphic compound het or p.Leu307Pro/mild; ~25–45% residual PEX16 stability → partial PEX3 recruitment → partial PMP insertion → partial peroxisomal function; adult survival possible",
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
            "variant_detail": "Both alleles hypomorphic (mild missense — p.Arg176Gln/p.Leu307Pro or similar); substantial residual PEX16 stability; late-onset slowly progressive ataxia + RP",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = ["p.Trp313*/p.Trp313*", "p.Trp313*/c.154del", "p.Trp313*/p.Glu164*", "p.Trp313*/c.1+2T>C"]
        elif "NALD" in phenotype:
            variants = ["p.Trp313*/p.Arg176Gln", "p.Glu164*/p.Arg176Gln", "p.Arg176Gln/c.154del", "p.Trp313*/c.831-1G>T"]
        elif "IRD" in phenotype:
            variants = ["p.Arg176Gln/p.Ala298Val", "p.Arg176Gln/c.831-1G>T", "p.Arg176Gln/p.Gly288Ser", "p.Leu307Pro/p.Arg176Gln"]
        else:
            variants = ["p.Arg176Gln/p.Ile319Met", "p.Leu307Pro/p.Arg176Gln"]
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
        ("ZS (Zellweger-Severe)", 12),
        ("NALD (Neonatal-ALD-Intermediate)", 20),
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
                "patient_id": f"PEX16-{pid:03d}",
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
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 30, "eeg": "Burst-suppression → multifocal spike; bilateral asynchronous; EEG NICU mandatory; universal in ZS"},
        {"type": "Infantile spasms / West syndrome (NALD dominant)", "pct": 50, "eeg": "Hypsarrhythmia → modified; ACTH Level A first-line; VGB HIGH RISK (avoid — ZSD retinopathy additive)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 20, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; LEV first-line"},
        {"type": "Myoclonic seizures (NALD)", "pct": 16, "eeg": "Generalised polyspike-wave ≥3 Hz; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 12, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 3, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX Level C consideration in IRD"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 65, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
        {"trigger": "Subtherapeutic AED levels", "pct": 53, "note": "Common in NALD/IRD — growth spurts alter pharmacokinetics; TDM quarterly"},
        {"trigger": "Fasting / prolonged NPO", "pct": 50, "note": "EXTREME HAZARD — adipose phytanic mobilisation → acute neurotoxicity; IV dextrose MANDATORY"},
        {"trigger": "Sleep deprivation", "pct": 40, "note": "Worsens electrocortical excitability; melatonin 0.5–3 mg adjunct in NALD/IRD with sleep disruption"},
        {"trigger": "Metabolic decompensation / catabolism", "pct": 37, "note": "Intercurrent illness, surgery, missed feeds → phytanic surge + VLCFA accumulation worsens acutely"},
        {"trigger": "Missed AED dose", "pct": 31, "note": "Non-adherence frequent in long-term NALD/IRD; simplified LEV BID regimen improves adherence"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC)", "pct": 18, "note": "CYP3A4 induction → DHA depletion (already low) + hepatic burden; RELATIVE CI; replace with LEV"},
        {"trigger": "Anaesthesia / surgical stress", "pct": 15, "note": "NPO + surgical stress → phytanic surge; pre-op: PT/INR + LFT + platelet + IV VitK; IV dextrose mandatory"},
    ]

    monitoring = [
        "VLCFA plasma panel (C26:0, C26:0/C22:0 ratio) — baseline + q6M (disease tracking + AED-induced metabolic change)",
        "Erythrocyte plasmalogens — baseline + q6M (PEX16-PEX3-PEX19 membrane biogenesis function marker; LOW in all ZSD)",
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
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer to metabolic specialist STAT; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel (include PEX16)"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel; PEX16 among first-tier genes after PEX1/PEX6"},
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
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; MRI shows pachygyria; ghost peroxisomes on fibroblast catalase staining; PEX3 mislocalized to cytoplasm on immunofluorescence",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT (include PEX16); palliative care discussion; POLG1 mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (40–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds; PEX3 mislocalized on fibroblast IF — diagnostic clue that PEX16 LOF is upstream",
            "action": "ACTH Level A (preferred over VGB for IS — ZSD retinopathy risk); LEV adjunct; DHA Level B; ophthalmology + audiology baseline; confirm PEX16 biallelic variants by NGS",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau; RP progression; SNHL; ataxia emerging in IRD; phytol-restricted diet starts in IRD",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet Level B (IRD); DHA continue; VPA NEVER; ERG + visual field + ABR annual; NGS confirms PEX16 identity vs PEX1/PEX6",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal ± GTCS; intellectual disability mild–moderate; ataxia prominent; RP → tunnel vision; SNHL stable; hepatomegaly mild; peripheral neuropathy emerging",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + DHA; neuropsychological support; adaptive aids; transition planning",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable in well-controlled IRD; RP → legal blindness; SNHL; progressive ataxia; peripheral neuropathy; psychiatric comorbidity",
            "action": "Long-term LEV ± LTG; phytol-restricted diet; annual metabolic review; genetic counselling (AR — 25% recurrence); no ERT/HSCT; gene therapy (AAV-PEX16) preclinical",
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
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX16)",
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
            "evidence": "Level B (Martinez 1996; van Grunsven 1999; LPC-DHA bypasses peroxisomal synthesis block)",
            "dose": "100–200 mg/day as LPC-DHA; oral; intestinal absorption bypasses peroxisomal DHA synthesis failure",
            "moa": "LPC-DHA absorbed intestinally → incorporated into neural membranes; bypasses peroxisomal synthesis; improves retinal + myelin in NALD/IRD",
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
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG (IRD focal). NEVER VPA in PEX16-ZSD.",
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
            "reason": "CYP3A4 induction → (1) DHA catabolism accelerated (already deficient in ZSD → worsens RP/neuropathy) + (2) hepatic microsomal burden (cholestatic liver ZS/NALD). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI — no universal adrenal insufficiency in PEX16-ZSD).",
            "alternative": "LEV first-line; LTG glucuronidation Level C (IRD); CLB adjunct. Replace with LEV on diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → works in X-ALD (ABCD1) because peroxisomal import is intact but ABCD1 transporter absent. In PEX16-ZSD, the PEROXISOMAL MEMBRANE ITSELF is absent (PEX16 LOF → PEX3 mislocalized → PEX19 cannot dock → all PMPs fail → ghost peroxisomes). LOO cannot restore PEX16-mediated PEX3 recruitment or reconstitute peroxisomal membrane biogenesis.",
            "alternative": "DHA Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No VLCFA-specific therapy available 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX16-ZSD)",
            "reason": "HSCT benefits neuroinflammatory conditions (cerebral X-ALD). PEX16-ZSD is NOT inflammatory; pathology is metabolic (peroxisomal membrane biogenesis failure — PEX16 recruits PEX3). Donor microglia cannot restore PEX16 function or reconstitute peroxisomal membrane biogenesis. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit.",
            "alternative": "Supportive care: DHA, phytol-restricted diet, LEV. Gene therapy (AAV9-PEX16) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX16 = type II integral membrane protein from ER)",
            "reason": "ERT is feasible for soluble lysosomal enzymes. PEX16 is a TYPE II INTEGRAL PEROXISOMAL MEMBRANE PROTEIN delivered via ER-to-peroxisome vesicular transport — not a soluble secreted enzyme. ERT cannot deliver an integral membrane protein via the ER-vesicle pathway. No applicable ERT mechanism.",
            "alternative": "Gene therapy (AAV-PEX16) in preclinical research; mRNA delivery experimental. No clinically available ERT 2026.",
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
        "PEX16 (336 aa, 11p11.2): type II integral peroxisomal membrane protein; delivered from ER to peroxisome via ER-derived pre-peroxisomal vesicles — mechanistically distinct from PEX19-mediated class I PMP insertion",
        "PEX16 function — PEX3 recruitment: PEX16 embedded in peroxisomal membrane acts as a co-factor/receptor that recruits PEX3 from ER-derived vesicles to the peroxisomal membrane, where PEX3 is then correctly positioned as the PEX19 docking receptor",
        "PEX3-PEX16-PEX19 early biogenesis axis: PEX16 brings PEX3 to the membrane (ER-vesicle pathway); PEX3 becomes the docking receptor for PEX19 (cytosolic PMP chaperone); PEX19 inserts new PMPs into the peroxisomal membrane (class I insertion) — the founding event of peroxisome biogenesis",
        "PEX16 LOF outcome: PEX3 mislocalized to cytoplasm/ER → PEX19 cannot dock onto peroxisomal membrane → ALL PMPs fail to insert → peroxisomes absent or 'ghost vesicles' (lipid bilayer only, no PMPs) → ALL peroxisomal matrix import secondarily fails",
        "ER-to-peroxisome membrane biogenesis route: Both PEX16 and PEX3 travel to peroxisomes via ER-derived vesicular transport — not via the classic PEX19 cytosolic chaperone route. This is the de novo peroxisome biogenesis pathway and is essential for establishing any peroxisomal membrane",
        "Diagnostic clue from immunofluorescence: In PEX16 LOF fibroblasts, catalase (peroxisomal matrix marker) is cytoplasmic (ghost peroxisomes). Critically, PEX3 immunofluorescence shows PEX3 MISLOCALIZED to cytoplasm/ER — a finding that directs NGS toward upstream biogenesis genes (PEX16, PEX3) rather than import receptor genes (PEX5, PEX7)",
        "Biochemical identity: PEX16-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX2/PEX10/PEX12/PEX26-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX16; NORMAL in ABCD1 (X-ALD); single test separates the diagnoses",
        "DHA synthesis failure: peroxisomal membrane absent → all DHA-synthesising enzymes (PTS1-targeted) cannot reach functional peroxisomes → DHA synthesis fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + RP + neuropathy",
        "Phenotypic spectrum: ZS 30% (null/null) · NALD 50% (dominant) · IRD 15% · Atypical 5%; ZS fraction (30%) lower than PEX3 (35%) — p.Arg176Gln enables more NALD/IRD. NALD is the modal phenotype in PEX16",
        "p.Arg176Gln (R176Q): partial-function allele at transmembrane/cytoplasmic interface; reduces PEX16 membrane stability → partial PEX3 recruitment to peroxisomal surface → partial PEX19 docking → partial PMP insertion → NALD/IRD phenotype as compound het with null allele",
        "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A) before any VPA consideration",
        "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD; ~15% IRD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH preferred for IS in all ZSD forms",
        "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; IV dextrose prevents catabolism; MANDATORY during any NPO",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute, unlike ABCD1): CYP3A4 induction depletes DHA + increases hepatic burden. No adrenal crisis in PEX16-ZSD (adrenal insufficiency not universal). Different from ABCD1 ABSOLUTE CI",
        "Lorenzo's Oil mechanism mismatch: LOO inhibits VLCFA elongase → effective in ABCD1 (import intact, transporter absent). In PEX16-ZSD, the peroxisomal MEMBRANE IS ABSENT (PEX16 LOF → PEX3 mislocalised → PEX19 cannot dock → no PMPs → ghost vesicles). LOO cannot restore ER-to-peroxisome vesicle routing or PEX16 membrane insertion → INEFFECTIVE",
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
        "Step 10 — Fibroblast IF (catalase + PEX3): absent peroxisomes (catalase cytoplasmic); PEX3 MISLOCALIZED to cytoplasm/ER (DIAGNOSTIC POINTER to PEX16 or PEX3 LOF — distinguishes from import receptor defects where PEX3 is membrane-localised)",
        "Step 11 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX16 biallelic pathogenic variants",
        "Step 12 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A)",
        "Step 13 — Genetic counselling: AR inheritance (25% recurrence risk); prenatal diagnosis available; carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC in ZSD: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE; PHT/CBZ/OXC = CYP3A4 → DHA depletion + hepatic burden → RELATIVE CI",
        "VPA vs LEV: VPA = THREE CI mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Never use VGB first-line in any ZSD form",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal synthesis via intestinal absorption. Ethyl ester less effective (requires peroxisomal processing). Specifically use LPC-DHA",
        "Phytol-restricted diet in IRD vs no benefit in ZS: In ZS (fatal <12M), restriction has no benefit. In IRD/Atypical (long survival), phytol restriction reduces phytanic accumulation → fewer seizures + slower neuropathy",
        "Lorenzo's Oil in ABCD1 vs PEX16-ZSD: LOO effective in ABCD1 (intact peroxisome, elongase inhibition reduces VLCFA). In PEX16-ZSD, peroxisomal MEMBRANE absent (PEX16 LOF → PEX3 mislocalized) → LOO cannot help → INEFFECTIVE",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (neuroinflammatory). PEX16-ZSD is NOT inflammatory → HSCT = transplant mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX16-ZSD): In ABCD1, PHT/CBZ → adrenal crisis (universal adrenal insufficiency + CYP induction). In PEX16-ZSD, no universal adrenal insufficiency → PHT/CBZ = RELATIVE CI (DHA/hepatic only, not adrenal)",
        "VPA POLG1 mandatory (CPIC Grade A): POLG1 exclusion mandatory in ZSD — catastrophic combined hepatotoxicity + mitochondrial depletion risk",
        "Carnitine supplementation: indicated if free carnitine <25 μmol/L; L-carnitine 50–100 mg/kg/day; critical if VPA inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY",
        "PEX16 vs PEX3 therapeutic implications: Both LOF → same biochemical phenotype → same AED rules. PEX16 acts upstream of PEX3 (recruits PEX3 from ER vesicles). No different treatment targets in 2026; gene therapy approach would differ (AAV-PEX16 must restore ER-to-peroxisome PEX3 recruitment upstream of PEX3's PEX19-docking function)",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX1-ZSD (65% of all PBD — most common)",
            "distinction": "Biochemically IDENTICAL. PEX1 = D1 AAA-ATPase subunit (motor); PEX3 anchors PEX19 (which PEX16 is upstream of). p.Gly843Asp founder allele (30% European) → IRD 30%. PEX16 rarer (~30–60 cases) but less rare than PEX3 or PEX26. 11p11.2 vs 7q21.2. NGS distinguishes. Both responsive to same ZSD AED protocol.",
        },
        {
            "condition": "PEX3-ZSD (~1–2% all PBD, ~10–20 cases worldwide)",
            "distinction": "Biochemically IDENTICAL. PEX16 recruits PEX3 from ER vesicles; PEX3 is the downstream docking receptor. In PEX16 LOF, PEX3 is mislocalized to cytoplasm — fibroblast IF shows PEX3 cytoplasmic in BOTH PEX16-ZSD and PEX3-ZSD. NGS panel distinguishes (PEX16 11p11.2 vs PEX3 6q24.2). Both have same treatment protocol.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 = D2 AAA-ATPase motor; docked on membrane via PEX26 (PEX26 itself requires PEX16→PEX3→PEX19 for membrane insertion). p.Arg860Trp (R860W) founder allele → IRD 45% (most attenuated ZSD, more than PEX16's IRD 15%). 11p11.2 vs 6p21.1. NGS distinguishes.",
        },
        {
            "condition": "PEX26-ZSD (~2–4% all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX26 = tail-anchored PMP (docking platform for PEX1-PEX6); PEX26 requires PEX16→PEX3→PEX19 for its own insertion into the peroxisomal membrane. PEX16 LOF is therefore UPSTREAM of PEX26. Both ghost peroxisomes on IF. 11p11.2 (PEX16) vs 22q11.21 (PEX26). NGS distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
            "distinction": "VLCFA elevated in BOTH. ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX16-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact peroxisome, transporter absent); INEFFECTIVE in PEX16-ZSD (membrane itself absent). HSCT standard CCALD; NOT in PEX16-ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL (key vs PEX16); plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. Phytol-restricted diet Level A GOLD STANDARD. PHYH is a PTS2 cargo enzyme; PEX16 is upstream of PHYH import (PEX16→PEX3→PEX19 pathway required to insert PEX7 receptor that imports PHYH).",
        },
        {
            "condition": "RCDP / PEX7 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "Plasmalogens (RBC) LOW (like ZSD) BUT VLCFA NORMAL + phytanic elevated + pristanic NORMAL + DHA NORMAL + pipecolic NORMAL. Only PTS2 pathway affected (PEX7 receptor deficient). Rhizomelia PATHOGNOMONIC. Stippled epiphyses on X-ray. No pachygyria. PEX7 protein itself is a PMP requiring PEX16→PEX3→PEX19 for membrane insertion — PEX16 LOF upstream of PEX7.",
        },
        {
            "condition": "Mitochondrial / POLG1 Epilepsy (Alpers)",
            "distinction": "VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Lactate/pyruvate elevated. Hepatocerebral phenotype. VPA CATASTROPHIC CI in POLG1 (mitochondrial depletion) — same practical CI as ZSD but different mechanism. POLG1 MANDATORY in all ZSD before VPA.",
        },
    ]

    standards = [
        "South ST et al. Identification of the human PEX16 gene, mutated in complement group D of the peroxisome biogenesis disorders. Nat Genet. 2000",
        "Honsho M et al. Disruption of peroxisomal membrane homeostasis by mutations in Pex16p, a pleiotropic functions protein. J Biol Chem. 1998",
        "Kim PK et al. The origin and maintenance of mammalian peroxisomes involves a de novo PEX16-dependent pathway from the ER. J Cell Biol. 2006",
        "Yamamoto Y et al. Pex16p promotes de novo peroxisome biogenesis at the ER in plants. Plant Physiol. 2018",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
        "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "Honsho M et al. Peroxisome biogenesis: a mechanistic insight. FEBS Lett. 2020",
        "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614876 (PBD8A-ZSD — PEX16); #614861 (PBD8B-NALD — PEX16); *603360 (PEX16 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
