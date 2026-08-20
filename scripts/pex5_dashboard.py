#!/usr/bin/env python3
"""PEX5 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX5 encodes Peroxin-5,
the CYTOSOLIC PTS1 RECEPTOR. Two isoforms: PEX5S (569 aa, short, lacks exon-8 insert)
and PEX5L (639 aa, long, contains 70-aa exon-8 insert that binds PEX7). PEX5 is the
ONLY CYTOSOLIC ZSD GENE — all other PBD-ZSD genes encode membrane proteins or
membrane-associated factors.

MECHANISM — CYTOSOLIC PTS1 RECEPTOR / PEX7 CO-RECEPTOR (PEX5L):
  PEX5 architecture (PEX5L, 639 aa):
    N-terminal domain (aa 1–110): 7 WxxxF/WxxxY pentapeptide repeats → dock to PEX14
    N-terminal cytoplasmic domain (PRIMARY high-affinity contact) and PEX13 SH3 domain
    (SECONDARY auxiliary contact). These are the SAME docking sites that PEX14 and PEX13
    present as receptors for PEX5 (i.e., PEX5 presents WxxxF; PEX14/PEX13 receive WxxxF).
    Middle insert (aa ~110–180, exon 8, PEX5L only): 70-aa region that directly binds
    PEX7 (the PTS2 receptor) → enables PEX7·PEX5L heterocomplex → PTS2 pathway.
    C-terminal TPR domain (7 tetratricopeptide repeats, aa ~300–600/639): recognises and
    binds PTS1 (C-terminal tripeptide: Ser-Lys-Leu [SKL], Ala-Lys-Leu [AKL], Ser-Arg-Leu
    [SRL] and >400 PTS1 variants). Each TPR pair forms a cargo-binding groove.
  PEX5 cycle:
    (1) Cytoplasmic PEX5 recognises PTS1 signal at C-terminus of newly synthesised
        peroxisomal matrix protein (PTS1 cargo) via TPR domain → forms cargo complex.
    (2) PEX5·cargo docks at peroxisomal membrane via N-terminal WxxxF pentapeptide
        repeats onto PEX14 N-terminal cytoplasmic domain (PRIMARY; high-affinity) +
        PEX13 SH3 domain (SECONDARY; auxiliary).
    (3) PEX5 inserts into importomer (PEX13·PEX14·PEX17 channel) → cargo translocated
        into peroxisomal matrix → cargo released into lumen.
    (4) PEX5 retrotranslocated: monoubiquitinated at Cys11 by PEX2/PEX10/PEX12 RING
        E3 ubiquitin-ligase complex → recognised by PEX1/PEX6/PEX26 AAA-ATPase motor →
        exported back to cytoplasm → deubiquitinated → recycled for next import cycle.
        Polyubiquitination at Lys → proteasomal degradation (quality control when
        PEX5 is trapped/misfolded — e.g. when retrotranslocation is blocked in PEX1/6 LOF).
  PEX5L isoform additionally:
    (5) PEX5L·PEX7 heterocomplex → brings PTS2-containing cargo (PEX7 binds PTS2 N-terminal
        nonapeptide: RLx5HL) to the importomer via the same PEX5 N-terminal docking
        mechanism → PTS2 pathway. Therefore PEX5 LOF abolishes BOTH PTS1 AND PTS2 import.
  PEX5 LOF consequences:
    ALL PTS1 import fails (no receptor to capture and deliver PTS1 cargo).
    ALL PTS2 import fails (PEX5L is required as co-receptor in PEX7·PEX5L complex).
    → ALL peroxisomal matrix proteins fail to import.
    → ALL peroxisomal metabolic pathways fail simultaneously.
  CRITICAL DISTINCTION — UNIQUE IF SIGNATURE OF PEX5-ZSD:
    PEX5 LOF is the ONLY ZSD where the peroxisomal MEMBRANE IS INTACT and the
    IMPORTOMER IS STRUCTURALLY INTACT, yet ALL matrix import fails.
    In ALL other matrix-import-defective ZSD:
      PEX13 LOF → PEX13 absent from importomer.
      PEX14 LOF → PEX14 absent from importomer.
    In PEX5 LOF: PEX13 PRESENT + PEX14 PRESENT + PMP70 PRESENT → importomer
    structurally complete but RECEPTOR IS ABSENT.
    IF pattern: PMP70 PRESENT (membrane biogenesis intact — PEX3/PEX16/PEX19 axis
    functional); PEX14 PRESENT (importomer central scaffold intact — PEX5 LOF does not
    affect PEX14 membrane insertion by PEX19); PEX13 PRESENT (importomer SH3 component
    intact); PEX5 absent from cytoplasm (null alleles) or dysfunctional; catalase
    CYTOPLASMIC (PTS1 cargo cannot enter without receptor).
  PEX5 LOF = biochemically IDENTICAL to ALL other PBD-ZSD; only NGS distinguishes.

LOCUS: 12p13.31  |  OMIM GENE: *600414  |  OMIM DISEASE: #614842 (PBD2A-ZSD), #614843 (PBD2B-NALD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX5 mutations account for ~1–2% of all
  PBD-ZSD (~15–25 confirmed cases worldwide as of 2026). No common founder allele (unlike
  PEX1 p.Gly843Asp or PEX6 p.Arg860Trp). Ultrarare; all variants private or sporadic.
  p.Trp369Arg (W369R, TPR domain) is the best-characterised partial-function allele;
  it disrupts PTS1 cargo recognition → partial TPR-domain binding → NALD/IRD spectrum.

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (IDENTICAL TO PEX1-ZSD):
  PEX5 deficiency causes the SAME biochemical phenotype as all other PBD-ZSD because
  PEX5 is required for import of ALL PTS1 matrix proteins AND (via PEX5L) ALL PTS2 matrix
  proteins (through PEX7·PEX5L complex):
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated
  (2) Alpha-oxidation FAILED → phytanic acid elevated; pristanic acid elevated
      (PHYH is a PTS2 cargo — requires PEX5L co-receptor; fails in PEX5-ZSD)
  (3) Plasmalogen biosynthesis FAILED → erythrocyte plasmalogens LOW
      [CRITICAL: plasmalogens NORMAL in ABCD1; LOW in ALL PBD-ZSD including PEX5]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated
  (5) DHA synthesis FAILED → DHA low → retinopathy + neuronal migration defects
  (6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease
  NBS BIOMARKER: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM — NO FOUNDER HYPOMORPHIC ALLELE):
  PEX5 cohorts show a severity distribution similar to PEX19 (another
  receptor-class ZSD; no dominant hypomorphic founder allele):
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
        "drug_resistance_pct": 64,
        "on_dha_pct": 52,
        "liver_disease_pct": 71,
        "retinopathy_pct": 81,
        "dha_low_pct": 100,
        "vlcfa_elevated_pct": 100,
        "plasmalogen_low_pct": 100,
        "omim_gene": "600414",
        "omim_disease_zs": "614842",
        "omim_disease_nald": "614843",
        "locus": "12p13.31",
        "protein_size_short": "569 aa (PEX5S)",
        "protein_size_long": "639 aa (PEX5L)",
        "nbs_positive_rate": "~97% sensitivity for ZS/NALD (C26:0-lyso-PC DBS)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Trp369Arg (W369R, TPR domain, disrupts PTS1 cargo recognition → partial TPR binding → "
            "enables NALD/IRD as compound het with null allele); p.Arg390* (null, truncating, deletes "
            "most of TPR domain — ZS/NALD); p.Gly28Arg (G28R, N-terminal WxxxF docking region, "
            "reduces PEX14 docking efficiency → partial receptor cycling → NALD/IRD); "
            "splice-site variants throughout (null — predominantly ZS)"
        ),
        "disease_mechanism": (
            "PEX5 (639 aa PEX5L, 569 aa PEX5S, 12p13.31) is the CYTOSOLIC PTS1 RECEPTOR — "
            "the ONLY cytosolic component among all ZSD-causing genes (all others encode membrane "
            "or membrane-associated proteins). PEX5 N-terminal domain (7 WxxxF pentapeptide repeats) "
            "docks onto PEX14 N-terminal cytoplasmic domain (PRIMARY, high-affinity) and PEX13 SH3 "
            "domain (SECONDARY, auxiliary) at the importomer. PEX5 C-terminal TPR domain (7 TPR "
            "motifs) recognises and binds PTS1 cargo in the cytoplasm. PEX5L additionally contains "
            "a 70-aa exon-8 insert that binds PEX7, enabling PTS2 import via PEX7·PEX5L complex. "
            "PEX5 LOF → no cytosolic PTS1 receptor → ALL PTS1 import fails; PEX7·PEX5L PTS2 "
            "pathway also fails → ALL matrix proteins fail → ALL peroxisomal pathways fail. "
            "UNIQUE IF SIGNATURE: In PEX5 LOF, the peroxisomal MEMBRANE IS INTACT (PMP70 PRESENT) "
            "AND the IMPORTOMER IS STRUCTURALLY INTACT (PEX14 PRESENT + PEX13 PRESENT) — yet ALL "
            "matrix import fails because the RECEPTOR is absent. This distinguishes PEX5-ZSD from "
            "PEX13 LOF (PEX13 absent) and PEX14 LOF (PEX14 absent) within the importomer tier, "
            "and from PEX3/PEX16/PEX19 LOF (PMP70 absent — membrane biogenesis tier). "
            "Biochemically identical to all other PBD-ZSD; only NGS distinguishes. "
            "~1–2% of all PBD-ZSD; ~15–25 cases worldwide 2026."
        ),
        "key_concepts": [
            "PEX5 (PEX5L 639 aa, PEX5S 569 aa, 12p13.31): CYTOSOLIC PTS1 RECEPTOR — the ONLY cytosolic ZSD-causing gene; all other PBD-ZSD encode membrane proteins or membrane-associated factors. N-terminal domain (7 WxxxF/WxxxY pentapeptide repeats) = docking arm; TPR domain (7 TPR motifs, C-terminal) = PTS1 cargo-recognition arm",
            "PEX5 docking mechanism (N-terminal pentapeptide repeats): PEX5 WxxxF/WxxxY repeats dock onto PEX14 N-terminal cytoplasmic domain (PRIMARY, high-affinity) and PEX13 SH3 domain (SECONDARY, auxiliary). This is the SAME WxxxF interaction that PEX14 presents as a docking scaffold — PEX5 is the presenter; PEX14 is the receptor. In PEX5 LOF, the docking arm is ABSENT (no receptor to dock, even though PEX14 and PEX13 ARE structurally present)",
            "PEX5 TPR domain (cargo recognition): 7 tandem tetratricopeptide repeats (C-terminal, ~aa 300–639) recognise and bind PTS1 (C-terminal tripeptide SKL/AKL/SRL and >400 PTS1 variants) on nascent peroxisomal matrix proteins in the cytoplasm. In PEX5 LOF (null), this recognition arm is absent → PTS1 cargo remains cytoplasmic",
            "PEX5L isoform — PTS2 co-receptor: PEX5L (639 aa) contains a 70-aa insert (exon 8) absent in PEX5S (569 aa). This insert directly binds PEX7 (the PTS2 receptor). PEX7·PEX5L heterocomplex → delivers PTS2 cargo to the importomer via PEX5L N-terminal docking mechanism. Therefore, PEX5 LOF abolishes BOTH PTS1 AND PTS2 import — broader than PEX7 LOF (which only affects PTS2) or a pure PEX5S defect",
            "PEX5 recycling cycle (Cys11 monoubiquitination): after cargo delivery, PEX5 is monoubiquitinated at Cys11 by the PEX2/PEX10/PEX12 RING E3 ubiquitin-ligase complex → monoubiquitinated PEX5 is recognised by the PEX1/PEX6/PEX26 AAA-ATPase motor (retrotranslocation complex) → exported from peroxisomal membrane back to cytoplasm → deubiquitinated → available for next import cycle. Polyubiquitination (Lys) → proteasomal degradation (quality control)",
            "UNIQUE IF SIGNATURE of PEX5-ZSD — all structural components present, receptor absent: In PEX5 LOF: PMP70 PRESENT (membrane biogenesis intact — PEX3/PEX16/PEX19 axis functional); PEX14 PRESENT (importomer central scaffold intact — PEX5 LOF does not affect PEX14 insertion by PEX19); PEX13 PRESENT (importomer SH3 component intact); catalase CYTOPLASMIC (import blocked — PTS1 receptor absent). This is the ONLY ZSD where membrane + importomer are structurally intact but ALL matrix import fails due to receptor absence",
            "IF TIER HIERARCHY for matrix-import-defective ZSD: (1) Membrane-biogenesis tier (PEX3/PEX16/PEX19 LOF): PMP70 ABSENT + PEX14 ABSENT + catalase cytoplasmic; (2) Importomer-scaffold tier (PEX14 LOF): PMP70 PRESENT + PEX14 ABSENT + catalase cytoplasmic; (3) Importomer-SH3 tier (PEX13 LOF): PMP70 PRESENT + PEX14 PRESENT + catalase cytoplasmic; (4) RECEPTOR TIER (PEX5 LOF): PMP70 PRESENT + PEX14 PRESENT + PEX13 PRESENT + catalase cytoplasmic — importomer structurally intact, receptor absent",
            "Biochemical identity: PEX5-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12/PEX13/PEX14-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
            "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX5; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
            "DHA synthesis failure: PTS1-targeted DHA-synthesising enzymes cannot enter peroxisomes (no PEX5 receptor) → DHA fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + neuropathy",
            "Phenotypic spectrum: ZS 35% (null/null) · NALD 45% (modal) · IRD 15% · Atypical 5%; NALD fraction highest (45%) because partial-function TPR-domain alleles (p.Trp369Arg) preserve partial PTS1 recognition; no founder hypomorphic allele",
            "p.Trp369Arg (W369R, TPR domain): partial-function allele; Trp369 in TPR-domain cargo-recognition groove; Arg substitution reduces PTS1 binding affinity → partial import → enables NALD/IRD as compound het with null allele; functional PEX5 protein present in cytoplasm but reduced PTS1 cargo capture",
            "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A)",
            "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH Level A preferred for IS in all ZSD forms",
            "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; IV dextrose MANDATORY during any NPO",
            "Lorenzo's Oil INEFFECTIVE (PEX5 receptor absent — matrix import cannot occur even though membrane and importomer are structurally intact; LOO targets VLCFA elongase, cannot restore PEX5-mediated cargo delivery to peroxisome)",
            "HSCT NOT INDICATED (ZSD not inflammatory; donor microglia cannot restore PEX5 cytosolic receptor or reconstitute PTS1/PTS2 import)",
            "No ERT (PEX5 = cytosolic receptor, not a soluble lysosomal enzyme; no mannose-6-phosphate receptor route applicable; cannot replace a cytosolic cargo receptor via ERT)",
            "~15–25 cases worldwide 2026; ~1–2% all PBD-ZSD; UNIQUE among ZSD: cytosolic receptor absent + importomer structurally intact; OMIM *600414 / #614842 / #614843",
        ],
        "standards": [
            "Dodt G et al. Mutations in the PTS1 receptor gene, PXR1 (PEX5), define complementation group 2 of the peroxisome biogenesis disorders. Nat Genet. 1995",
            "Wiemer EA et al. Human peroxisomal targeting signal type 1 receptor, Pex5p, and the intracellular shuttling of thiolase. J Biol Chem. 1995",
            "Brocard C, Hartig A. Peroxisome targeting signal 1: is it really a simple tripeptide? Biochim Biophys Acta. 2006",
            "Gatto GJ Jr et al. DOOR domains in mammalian Pex5p define docking sites for PTS1. J Cell Biol. 2000",
            "Platta HW et al. Ubiquitination of the peroxisomal import receptor Pex5p is required for its recycling. J Cell Biol. 2007",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM #614842 (PBD2A-ZSD — PEX5); #614843 (PBD2B-NALD — PEX5); *600414 (PEX5 gene)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(5505)  # PEX5 — reproducible seed

    etiologies = [
        {
            "name": "ZS (Zellweger-Severe) — null/null genotype",
            "pct": 35, "n": 14,
            "sex": "7M/7F",
            "onset_age": "Neonatal (day 1–7)",
            "seizure_risk": "100% — neonatal multifocal clonic + tonic + electrographic; EEG burst-suppression; universally fatal <12M",
            "eeg": "Burst-suppression → multifocal spike-wave; bilateral asynchronous; EEG NICU mandatory; uniformly poor prognosis",
            "mri": "Pachygyria + polymicrogyria (PATHOGNOMONIC in ZS); periventricular germinolytic cysts; hypomyelination; ventriculomegaly",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Arg390*/p.Arg390* or p.Arg390*/c.850-2A>G or p.Phe9*/null); PEX5 absent from cytoplasm (IF: PEX5− cytoplasm; importomer structurally INTACT — PMP70+ + PEX14+ + PEX13+; catalase cytoplasmic — UNIQUE IF SIGNATURE: all structural components present, receptor absent); all PTS1 + PTS2 import fails; fatal <6–12 months",
        },
        {
            "name": "NALD (Neonatal-ALD-Intermediate) — null/p.Trp369Arg or splice",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infantile (1st–12th month)",
            "seizure_risk": "80–90% — infantile spasms (IS) 38–55% + hypsarrhythmia; focal + myoclonic; ACTH Level A preferred",
            "eeg": "Hypsarrhythmia (IS period); multifocal spike-wave; modified hypsarrhythmia with burst-suppression elements",
            "mri": "Periventricular white matter signal; delayed myelination; cerebellar hypoplasia; milder gyral pattern than ZS",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Trp369Arg (partial TPR-domain PTS1 recognition — W369R reduces cargo-binding affinity in TPR groove; partial PTS1 import survives) or null/splice; partial receptor function allows months-to-years survival; PEX5 protein detectable in cytoplasm at reduced levels (missense allele); importomer structurally intact; matrix partially imported",
        },
        {
            "name": "IRD (Infantile-Refsum-Attenuated) — p.Trp369Arg/mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood / Adolescence (2–18 yr)",
            "seizure_risk": "20–30% — focal + GTCS; better AED response than ZS/NALD; phytol-restricted diet reduces burden",
            "eeg": "Focal epileptiform discharges; normal background early; generalised spike-wave with progression",
            "mri": "RP-associated retinal thinning; mild cerebellar atrophy; white matter changes mild; no pachygyria",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "p.Trp369Arg/hypomorphic compound het; ~30–50% residual PEX5 PTS1-recognition function → partial cargo capture → partial import → NALD/IRD phenotype; PEX5 protein detectable in cytoplasm; importomer structurally intact; adult survival possible",
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
            "variant_detail": "Both alleles hypomorphic; substantial residual PEX5 TPR-domain function; near-normal cytoplasmic PEX5 levels; near-normal PTS1 cargo capture; late-onset slowly progressive ataxia + RP; importomer structurally intact (PMP70+ + PEX14+ + PEX13+)",
        },
    ]

    def rand_sex():
        return random.choice(["M", "F"])

    def rand_geno(phenotype):
        if "ZS" in phenotype:
            variants = [
                "p.Arg390*/p.Arg390*",
                "p.Arg390*/c.850-2A>G",
                "p.Phe9*/p.Arg390*",
                "p.Phe9*/c.1169+1G>T",
            ]
        elif "NALD" in phenotype:
            variants = [
                "p.Arg390*/p.Trp369Arg",
                "p.Trp369Arg/c.850-2A>G",
                "p.Trp369Arg/p.Gln202*",
                "p.Arg390*/p.Gly28Arg",
            ]
        elif "IRD" in phenotype:
            variants = [
                "p.Trp369Arg/p.Tyr462Asp",
                "p.Trp369Arg/c.1408-5C>G",
                "p.Gly28Arg/p.Trp369Arg",
                "p.Trp369Arg/p.Leu411Phe",
            ]
        else:
            variants = [
                "p.Trp369Arg/p.Ser519Leu",
                "p.Gly28Arg/p.Tyr462Asp",
            ]
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
                "patient_id": f"PEX5-{pid:03d}",
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
        {"type": "Neonatal multifocal clonic (ZS)", "pct": 35, "eeg": "Burst-suppression → multifocal spike; bilateral asynchronous; EEG NICU mandatory; PEX5 LOF = 35% ZS (NALD modal at 45%)"},
        {"type": "Infantile spasms / West syndrome (NALD dominant)", "pct": 45, "eeg": "Hypsarrhythmia → modified; ACTH Level A first-line; VGB HIGH RISK (ZSD retinopathy additive — avoid)"},
        {"type": "Focal seizures bilateral spread (NALD/IRD)", "pct": 22, "eeg": "Temporal/parietal focal spike-wave; post-ictal slowing; LEV first-line"},
        {"type": "Myoclonic seizures (NALD)", "pct": 14, "eeg": "Generalised polyspike-wave ≥3 Hz; CBZ/PHT HIGH RISK (myoclonic exacerbation)"},
        {"type": "GTCS (IRD/Atypical)", "pct": 11, "eeg": "Generalised spike-wave; good post-ictal recovery in IRD; LEV + LTG effective"},
        {"type": "Absence / atypical absence (rare, IRD)", "pct": 3, "eeg": "2.5–3.5 Hz generalised spike-wave; ETX Level C consideration in IRD"},
    ]

    triggers = [
        {"trigger": "Fever / intercurrent infection", "pct": 63, "note": "Universal ZSD trigger — fever lowers seizure threshold + increases metabolic demand → phytanic mobilisation"},
        {"trigger": "Subtherapeutic AED levels", "pct": 50, "note": "Common in NALD/IRD — growth spurts alter pharmacokinetics; TDM quarterly"},
        {"trigger": "Fasting / prolonged NPO", "pct": 48, "note": "EXTREME HAZARD — adipose phytanic mobilisation → acute neurotoxicity; IV dextrose MANDATORY"},
        {"trigger": "Sleep deprivation", "pct": 39, "note": "Worsens electrocortical excitability; melatonin 0.5–3 mg adjunct in NALD/IRD with sleep disruption"},
        {"trigger": "Metabolic decompensation / catabolism", "pct": 35, "note": "Intercurrent illness, surgery, missed feeds → phytanic surge + VLCFA accumulation worsens acutely"},
        {"trigger": "Missed AED dose", "pct": 29, "note": "Non-adherence frequent in long-term NALD/IRD; simplified LEV BID regimen improves adherence"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC)", "pct": 16, "note": "CYP3A4 induction → DHA depletion (already low) + hepatic burden; RELATIVE CI; replace with LEV"},
        {"trigger": "Anaesthesia / surgical stress", "pct": 13, "note": "NPO + surgical stress → phytanic surge; pre-op: PT/INR + LFT + platelet + IV VitK; IV dextrose mandatory"},
    ]

    monitoring = [
        "VLCFA plasma panel (C26:0, C26:0/C22:0 ratio) — baseline + q6M (disease tracking + AED-induced metabolic change)",
        "Erythrocyte plasmalogens — baseline + q6M (peroxisomal import function marker; severely LOW in all ZSD including PEX5; NORMAL in ABCD1)",
        "DHA plasma level — baseline + q6M (supplement adequacy; target >4% total fatty acids in NALD/IRD)",
        "Plasma phytanic acid — q6M in IRD/Atypical; q3M in NALD (diet compliance + fasting risk)",
        "Plasma pristanic acid — q6M in IRD (alpha-oxidation status; PHYH is PTS2 cargo requiring PEX5L → elevated in PEX5-ZSD)",
        "LFT + GGT + bilirubin (conjugated) — q3M in ZS/NALD; q6M in IRD (cholestatic liver disease monitoring)",
        "PT/INR + platelet count — q6M in ZS/NALD (cholestatic coagulopathy; pre-op MANDATORY)",
        "Plasma carnitine + acylcarnitine profile — q6M (carnitine depletion risk, especially if VPA inadvertently started)",
        "POLG1 sequencing — ONCE before any VPA consideration (CPIC Grade A; MANDATORY)",
        "Electroretinogram (ERG) — baseline + annual (universal retinopathy ZS/NALD; preserve remaining IRD visual field)",
        "Visual field testing (Goldmann perimetry) — q12M in IRD (preserve from VGB, RP progression)",
        "SNHL audiometry (ABR neonatal → PTA in IRD) — baseline + q12M (universal SNHL in ZS/NALD)",
        "EEG — NICU continuous in ZS; q3M in active NALD (IS monitoring); q6M in seizure-free NALD; q12M in IRD",
        "Brain MRI — baseline (pachygyria ZS; white matter NALD); q12–24M in NALD/IRD",
    ]

    thresholds = [
        {"parameter": "C26:0-lyso-PC (NBS DBS)", "threshold": ">0.10 μmol/L", "action": "Refer to metabolic specialist STAT; confirm with plasma VLCFA + plasmalogens; NGS PEX gene panel (include PEX5)"},
        {"parameter": "Plasma C26:0/C22:0 ratio", "threshold": ">0.020", "action": "Diagnostic for ZSD biochemically; initiate NGS PEX panel (PEX5 among first-tier on comprehensive panel)"},
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
            "features": "Profound hypotonia; neonatal seizures day 1–7 (multifocal clonic + tonic); EEG burst-suppression; cholestatic jaundice; craniofacial dysmorphism; ERG absent; SNHL on ABR; MRI shows pachygyria; UNIQUE IF SIGNATURE: PMP70 PRESENT + PEX14 PRESENT + PEX13 PRESENT (entire membrane + importomer structurally intact) + PEX5 absent from cytoplasm + catalase cytoplasmic (receptor absent — unlike PEX13/PEX14 LOF where an importomer component is absent)",
            "action": "LEV IV first-line; Phenobarbital if refractory; IV dextrose continuous; VLCFA + plasmalogens STAT; NGS PEX panel STAT (include PEX5); palliative care discussion; POLG1 mandatory",
        },
        {
            "stage": "Stage 2 — Early Infantile (1–12 months, NALD dominant)",
            "features": "Infantile spasms (IS) with hypsarrhythmia (38–55%); psychomotor regression; cholestasis evolving; SNHL; visual impairment from RP; nasogastric feeds; PEX5 protein detectable in cytoplasm in NALD (partial-function TPR allele) but PTS1 cargo recognition reduced — compare with ZS (PEX5 absent) and PEX13/PEX14 LOF (importomer component absent)",
            "action": "ACTH Level A (preferred over VGB for IS — ZSD retinopathy risk); LEV adjunct; DHA Level B; ophthalmology + audiology baseline; confirm PEX5 biallelic variants by NGS",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr, NALD/IRD)",
            "features": "Seizure evolution from IS to focal + myoclonic; developmental plateau; RP progression; SNHL; ataxia emerging in IRD; phytol-restricted diet starts in IRD; pristanic elevated (PHYH is PTS2 cargo requiring PEX5L — confirms PEX5L isoform failure)",
            "action": "Optimise LEV ± CLB adjunct; phytol-restricted diet Level B (IRD); DHA continue; VPA NEVER; ERG + visual field + ABR annual; NGS confirms PEX5 identity vs other PBD-ZSD",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, IRD/Atypical)",
            "features": "Focal ± GTCS; intellectual disability mild–moderate; ataxia prominent; RP → tunnel vision; SNHL stable; peripheral neuropathy emerging; phytol-restricted diet critical",
            "action": "LEV ± LTG Level C; strict phytol-restricted diet; annual VLCFA + phytanic + pristanic + DHA; neuropsychological support; adaptive aids; transition planning",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, IRD/Atypical only)",
            "features": "Seizure frequency stable in well-controlled IRD; RP → legal blindness; SNHL; progressive ataxia; peripheral neuropathy; psychiatric comorbidity",
            "action": "Long-term LEV ± LTG; phytol-restricted diet; annual metabolic review; genetic counselling (AR — 25% recurrence); no ERT (cytosolic receptor, not lysosomal enzyme) / no HSCT; gene therapy (AAV9-PEX5) preclinical",
        },
        {
            "stage": "Stage 6 — Surgical / Anaesthesia Intercept (ANY stage)",
            "features": "Peri-operative fasting EXTREME HAZARD: phytanic surge → neonatal-equivalent neurotoxicity; cholestatic coagulopathy risk (ZS/NALD); unique PEX5-ZSD concern: both PTS1 and PTS2 cargo import fails (PHYH = PTS2 cargo → pristanic/phytanic both elevated → increased volatility during metabolic stress)",
            "action": "IV dextrose MANDATORY from first NPO moment; IV Vitamin K pre-op; PT/INR + LFT + platelet pre-op; VPA ABSOLUTE CI perioperatively; IV LEV replaces oral; anaesthetics team briefing MANDATORY",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "AED — FIRST-LINE (all ZSD forms)",
            "evidence": "Level A (expert consensus — multiple ZSD cohorts including PEX5)",
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
            "dose": "100–200 mg/day as LPC-DHA; oral; intestinal absorption bypasses peroxisomal DHA synthesis failure (PEX5 LOF → DHA-synthesising enzymes excluded from peroxisome)",
            "moa": "LPC-DHA absorbed intestinally → incorporated into neural membranes; bypasses PEX5-mediated import failure; improves retinal + myelin in NALD/IRD",
            "monitoring": "Plasma DHA q3M; target >4% total FA; ERG q6M",
            "ci": "Not indicated in ZS (fatal prognosis); use LPC-DHA formulation specifically",
        },
        {
            "drug": "Phytol-restricted diet (low phytanic acid)",
            "class": "Disease-Modifying — Level B (IRD/Atypical)",
            "evidence": "Level B (Steinberg 2006; clinical consensus; reduces phytanic + pristanic accumulation in IRD)",
            "dose": "Restrict phytol-rich foods (green vegetables, dairy, ruminant fat); dietitian supervised; note: PHYH (phytanic alpha-oxidase, PTS2 cargo via PEX5L) also fails in PEX5-ZSD → both phytanic + pristanic elevated",
            "moa": "Phytol → phytanic acid via dietary intake; restriction reduces both phytanic + pristanic accumulation → fewer phytanic/pristanic-triggered seizures + neuropathy",
            "monitoring": "Plasma phytanic + pristanic q6M; nutritional status; dietitian review q6M",
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
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG (IRD focal). NEVER VPA in PEX5-ZSD.",
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
            "reason": "Fasting triggers adipose release of stored phytanic acid (failed alpha-oxidation). In PEX5-ZSD, PHYH (phytanoyl-CoA hydroxylase, a PTS2 cargo imported via PEX7·PEX5L) also fails → both phytanic AND pristanic accumulate. Acute phytanic + pristanic surge → direct neurotoxicity + acute seizures + cardiac arrhythmia. ANY NPO period = MANDATORY IV dextrose.",
            "alternative": "IV 10% dextrose MANDATORY from first NPO moment. Pre-op briefing. IV Vitamin K (coagulopathy). PT/INR + LFT + platelet pre-surgery. Resume enteral feeds ASAP.",
        },
        {
            "drug": "Phenytoin / Carbamazepine / Oxcarbazepine (PHT / CBZ / OXC)",
            "level": "RELATIVE CI (avoid first-line; caution if used)",
            "reason": "CYP3A4 induction → (1) DHA catabolism accelerated (already deficient in ZSD → worsens RP/neuropathy) + (2) hepatic microsomal burden (cholestatic liver ZS/NALD). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI — no universal adrenal insufficiency in PEX5-ZSD).",
            "alternative": "LEV first-line; LTG glucuronidation Level C (IRD); CLB adjunct. Replace with LEV on diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "INEFFECTIVE (do NOT use in ZSD)",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → works in X-ALD (ABCD1) because peroxisomal import is intact but ABCD1 transporter absent. In PEX5-ZSD, the CYTOSOLIC PTS1 RECEPTOR IS ABSENT → matrix import impossible even though membrane and importomer are structurally intact. LOO cannot restore PEX5-mediated PTS1 cargo delivery — VLCFA elongase inhibition is irrelevant without functional PTS1 receptor.",
            "alternative": "DHA Level B (NALD/IRD). Phytol-restricted diet Level B (IRD). No VLCFA-specific therapy available 2026.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX5-ZSD)",
            "reason": "HSCT benefits neuroinflammatory conditions (cerebral X-ALD). PEX5-ZSD is NOT inflammatory; pathology is metabolic (cytosolic PTS1 receptor absent — PEX5 LOF prevents all matrix import). Donor microglia cannot restore PEX5 cytosolic receptor function or reconstitute PTS1/PTS2 import. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit.",
            "alternative": "Supportive care: DHA, phytol-restricted diet, LEV. Gene therapy (AAV9-PEX5) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX5 = cytosolic receptor, not soluble lysosomal enzyme)",
            "reason": "ERT is feasible for soluble lysosomal enzymes delivered via mannose-6-phosphate receptor. PEX5 is a CYTOSOLIC PTS1 RECEPTOR — not a secreted soluble enzyme. ERT cannot deliver a cytosolic receptor into the cytoplasm of target cells (no M6P receptor route, no endosomal escape mechanism for a cytosolic protein). Fundamentally different mechanism from lysosomal storage disease ERT.",
            "alternative": "Gene therapy (AAV-PEX5) in preclinical research; mRNA delivery experimental. No clinically available ERT 2026.",
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
        "PEX5 (PEX5L 639 aa, PEX5S 569 aa, 12p13.31): CYTOSOLIC PTS1 RECEPTOR — the ONLY cytosolic gene causing PBD-ZSD; all other ZSD-causing genes encode membrane proteins (PMPs) or membrane-associated factors. PEX5 is the dedicated carrier that shuttles PTS1 cargo from the cytoplasm to the peroxisomal importomer",
        "PEX5 architecture: N-terminal domain (aa 1–110, 7 WxxxF/WxxxY pentapeptide repeats) = docking arm onto PEX14 (primary, high-affinity) and PEX13 SH3 (secondary, auxiliary); middle 70-aa exon-8 insert (PEX5L only) = PEX7-binding region → PTS2 co-receptor function; C-terminal TPR domain (7 TPR motifs, aa ~300–639) = PTS1 cargo-recognition arm (reads C-terminal tripeptide SKL/AKL/SRL and >400 PTS1 variants)",
        "PEX5 docking mechanism: WxxxF pentapeptide repeats (N-terminal) dock onto PEX14 N-terminal cytoplasmic domain (PRIMARY — same WxxxF interaction described in PEX14 entry as the 'PEX5 docking site'; PEX5 IS the presenter of WxxxF; PEX14 IS the receiver) and PEX13 SH3 domain (SECONDARY). In PEX5 LOF, these WxxxF repeats are absent or non-functional → no docking even though PEX14 and PEX13 are structurally intact",
        "PEX5L — PTS2 co-receptor: PEX5L 70-aa exon-8 insert directly binds PEX7 (PTS2 receptor). PEX7·PEX5L heterocomplex → delivers PTS2 cargo (PHYH/phytanoyl-CoA hydroxylase, AGPS, DHAPAT, thiolase, MFP2-PTS2) to importomer via PEX5L N-terminal docking. PEX5 LOF → PEX7·PEX5L complex cannot form → PTS2 import also fails. This makes PEX5-ZSD the ONLY ZSD where BOTH PTS1 and PTS2 import fail from a single receptor deficiency (PEX7 LOF = only PTS2; PEX13/PEX14 LOF = both pathways fail at importomer docking step)",
        "PEX5 recycling (Cys11 monoubiquitination): after cargo delivery to peroxisomal matrix, PEX5 must be retrotranslocated and recycled. PEX2/PEX10/PEX12 RING E3 complex monoubiquitinates PEX5 at Cys11 → PEX1/PEX6/PEX26 AAA-ATPase extracts PEX5 from membrane → deubiquitination → cytoplasmic PEX5 recycled. Polyubiquitination at Lys → proteasomal degradation. This is why PEX1/PEX6/PEX26 LOF causes PEX5 membrane accumulation (retrotranslocation block) — different from PEX5 LOF where PEX5 receptor is absent",
        "UNIQUE IF SIGNATURE of PEX5-ZSD (distinct from all other ZSD): In PEX5 LOF: PMP70 PRESENT (membrane biogenesis intact — PEX3/PEX16/PEX19 axis functional) + PEX14 PRESENT (importomer central scaffold intact — PEX5 LOF does not affect PEX14 membrane insertion by PEX19) + PEX13 PRESENT (importomer SH3 component intact) + catalase CYTOPLASMIC (import blocked — PTS1 receptor absent). This is the ONLY ZSD where ALL structural components (membrane + importomer) are intact but ALL matrix import fails due to receptor absence",
        "IF TIER HIERARCHY for matrix-import-defective ZSD: Membrane-biogenesis tier (PEX3/PEX16/PEX19 LOF) → PMP70 ABSENT + PEX14 ABSENT + catalase cytoplasmic; Importomer-scaffold tier (PEX14 LOF) → PMP70 PRESENT + PEX14 ABSENT + catalase cytoplasmic; Importomer-SH3 tier (PEX13 LOF) → PMP70 PRESENT + PEX14 PRESENT + catalase cytoplasmic; RECEPTOR tier (PEX5 LOF) → PMP70 PRESENT + PEX14 PRESENT + PEX13 PRESENT + catalase cytoplasmic — all structural components intact, receptor absent",
        "Biochemical identity: PEX5-ZSD is biochemically IDENTICAL to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12/PEX13/PEX14-ZSD — VLCFA ↑, plasmalogens (RBC) ↓, DHA ↓, phytanic ↑, pristanic ↑, pipecolic ↑, DHCA/THCA ↑; only NGS distinguishes",
        "Plasmalogens (RBC) LOW — KEY ZSD DISTINCTION from ABCD1: erythrocyte plasmalogens severely reduced in ALL PBD-ZSD including PEX5; NORMAL in ABCD1 (X-ALD); single test separates diagnoses",
        "DHA synthesis failure: PTS1-targeted DHA-synthesising enzymes cannot enter peroxisomes (no PEX5 receptor) → DHA fails → pachygyria + polymicrogyria (ZS, MRI PATHOGNOMONIC) + retinopathy + neuropathy",
        "Phenotypic spectrum: ZS 35% (null/null) · NALD 45% (modal) · IRD 15% · Atypical 5%; NALD modal fraction (45%) reflects prevalence of partial-function TPR-domain alleles (p.Trp369Arg) that permit some PTS1 cargo recognition; no founder hypomorphic allele",
        "p.Trp369Arg (W369R, TPR domain): partial-function allele; Trp369 in the PTS1-cargo-recognition groove of the C-terminal TPR domain; Arg substitution reduces PTS1 binding affinity → partial cargo capture → reduced import efficiency → NALD/IRD as compound het with null allele; functional PEX5 protein present in cytoplasm but reduced PTS1 ligand capture",
        "VPA triple CI mechanism: (1) hepatotoxicity (cholestatic liver ZS/NALD) + (2) peroxisomal BO inhibition (worsens VLCFA) + (3) carnitine depletion. POLG1 MANDATORY (CPIC Grade A) before any VPA consideration",
        "VGB additive retinopathy: ZSD causes RP (universal ZS/NALD) + VGB irreversible VF constriction → catastrophic additive blindness; ACTH preferred for IS in all ZSD forms",
        "Fasting EXTREME HAZARD: adipose phytanic acid release during starvation → direct neurotoxicity + seizures + arrhythmia; in PEX5-ZSD, PHYH (PTS2 cargo via PEX7·PEX5L) also fails → pristanic also elevated → doubled alpha-oxidation substrate accumulation; IV dextrose MANDATORY during any NPO",
        "PHT/CBZ/OXC RELATIVE CI (NOT absolute, unlike ABCD1): CYP3A4 induction depletes DHA + increases hepatic burden. No adrenal crisis in PEX5-ZSD (no universal adrenal insufficiency). Different from ABCD1 ABSOLUTE CI",
        "Lorenzo's Oil mechanism mismatch in PEX5-ZSD: LOO inhibits VLCFA elongase → effective in ABCD1 (import intact, transporter absent). In PEX5-ZSD, the CYTOSOLIC PTS1 RECEPTOR IS ABSENT → matrix import impossible even though membrane and importomer are structurally intact. LOO cannot supply a PTS1 receptor to the cytoplasm — INEFFECTIVE. Mechanistically the most fundamental defect level among ZSD (receptor absent = no cargo delivery at all)",
    ]

    diagnostic_algorithm = [
        "Step 1 — NBS flag: C26:0-lyso-PC elevated on DBS → refer to metabolic specialist STAT",
        "Step 2 — Plasma VLCFA panel: C26:0, C26:0/C22:0 ratio elevated → confirms ZSD biochemical group",
        "Step 3 — Erythrocyte plasmalogens: severely LOW → confirms PBD-ZSD (distinguishes from ABCD1 where plasmalogens NORMAL)",
        "Step 4 — Plasma phytanic + pristanic: both elevated (alpha-oxidation + pristanic oxidation failed — in PEX5-ZSD, BOTH PHYH [PTS2 via PEX5L] AND the peroxisomal alpha-oxidation import pathway fail; pristanic elevation is a PEX5-specific hallmark vs diseases where only phytanic fails)",
        "Step 5 — Plasma/urine pipecolic acid: elevated → confirms peroxisomal import failure",
        "Step 6 — Plasma DHA: LOW → confirms DHA synthesis failure; retinopathy + migration defect risk",
        "Step 7 — Bile acids (DHCA, THCA): elevated → confirms bile acid synthesis failure → cholestatic liver disease",
        "Step 8 — MRI brain: pachygyria + polymicrogyria in ZS (PATHOGNOMONIC); periventricular WM signal in NALD; mild/normal in IRD",
        "Step 9 — ERG + ophthalmology: ERG absent/severely reduced (ZS/NALD); RP in IRD; visual fields (Goldmann)",
        "Step 10 — Fibroblast IF UNIQUE TRIPLET (PMP70 + PEX14 + catalase): PMP70 PRESENT (membrane biogenesis intact — distinguishes from PEX3/PEX16/PEX19 tier where PMP70 absent); PEX14 PRESENT (importomer central scaffold intact — distinguishes from PEX14 LOF where PEX14 absent); catalase cytoplasmic (import fails because receptor absent). UNIQUE to PEX5-ZSD among all ZSD: PMP70+/PEX14+/PEX13+/catalase cytoplasmic — structural components all present, only receptor absent",
        "Step 11 — Comprehensive NGS PEX gene panel (PEX1, PEX2, PEX3, PEX5, PEX6, PEX7, PEX10, PEX12, PEX13, PEX14, PEX16, PEX19, PEX26): identifies PEX5 biallelic pathogenic variants",
        "Step 12 — POLG1 sequencing: MANDATORY before considering any VPA (CPIC Grade A)",
        "Step 13 — Genetic counselling: AR inheritance (25% recurrence risk); prenatal diagnosis available; carrier testing for parents/siblings",
    ]

    pharmacological_distinctions = [
        "LEV vs PHT/CBZ/OXC in ZSD: LEV = no CYP induction, no hepatotoxicity, IV available → FIRST-LINE; PHT/CBZ/OXC = CYP3A4 → DHA depletion + hepatic burden → RELATIVE CI",
        "VPA vs LEV: VPA = THREE CI mechanisms (hepatotoxicity + peroxisomal BO inhibition + carnitine depletion) → HIGH RISK near-absolute CI; LEV = none → always preferred",
        "ACTH vs VGB for IS: In ZSD, ACTH Level A (preferred) vs VGB HIGH RISK (retinopathy additive). Never use VGB first-line in any ZSD form",
        "DHA supplementation: LPC-DHA (lysophosphatidylcholine-DHA) form bypasses peroxisomal import failure via intestinal absorption. Ethyl ester less effective (requires peroxisomal processing). Use LPC-DHA specifically",
        "Phytol-restricted diet in IRD vs no benefit in ZS: In ZS (fatal <12M), restriction has no benefit. In IRD/Atypical (long survival), phytol restriction reduces both phytanic and pristanic accumulation → fewer seizures + slower neuropathy. PEX5-ZSD: BOTH phytanic AND pristanic elevated (PHYH = PTS2 cargo via PEX7·PEX5L also fails)",
        "Lorenzo's Oil in ABCD1 vs PEX5-ZSD: LOO effective in ABCD1 (intact peroxisome, transporter absent). In PEX5-ZSD, the CYTOSOLIC PTS1 RECEPTOR IS ABSENT (no PEX5 to capture and dock PTS1 cargo) → matrix import impossible even though membrane and importomer are structurally intact → LOO irrelevant → INEFFECTIVE. Most fundamental defect level: receptor absent = no cargo recognition step",
        "HSCT in ABCD1 vs ZSD: HSCT standard of care in early CCALD (neuroinflammatory). PEX5-ZSD is NOT inflammatory → HSCT = transplant mortality risk with zero mechanistic benefit",
        "PHT/CBZ ABSOLUTE CI (ABCD1) vs RELATIVE CI (PEX5-ZSD): In ABCD1, PHT/CBZ → adrenal crisis (universal adrenal insufficiency + CYP induction). In PEX5-ZSD, no universal adrenal insufficiency → PHT/CBZ = RELATIVE CI (DHA/hepatic only, not adrenal)",
        "VPA POLG1 mandatory (CPIC Grade A): POLG1 exclusion mandatory in ZSD — catastrophic combined hepatotoxicity + mitochondrial depletion risk",
        "Carnitine supplementation: indicated if free carnitine <25 μmol/L; L-carnitine 50–100 mg/kg/day; critical if VPA inadvertently started",
        "IV Vitamin K perioperative: cholestatic liver (DHCA/THCA-driven) → fat-soluble vitamin K malabsorption → coagulopathy (↑PT/INR); IV Vitamin K 1–2 mg/kg (max 10 mg) pre-surgery; check PT/INR + platelet pre-op MANDATORY",
        "PEX5 vs PEX13/PEX14 therapeutic tier distinction: All → same biochemical phenotype → same AED rules. IF clue: PEX5 LOF = PMP70+/PEX14+/PEX13+/catalase cytoplasmic (receptor absent, importomer intact); PEX13 LOF = PMP70+/PEX14+/catalase cytoplasmic; PEX14 LOF = PMP70+/PEX14−/catalase cytoplasmic. Gene therapy approaches differ: AAV-PEX5 must supply functional cytosolic receptor; AAV-PEX14/PEX13 must supply integral membrane importomer components. Preclinical 2026.",
    ]

    differential_diagnosis = [
        {
            "condition": "PEX14-ZSD (~<1% all PBD, ~5–8 cases worldwide)",
            "distinction": "Biochemically IDENTICAL. KEY IF DISTINCTION: In PEX14 LOF, PEX14 IS ABSENT from peroxisomal membranes (PEX14 is a Class I PMP inserted by PEX19; absent when PEX14 gene fails). In PEX5 LOF, PEX14 IS PRESENT (importomer intact; PEX5 LOF does not affect PEX14 membrane insertion). Both: PMP70 PRESENT + catalase cytoplasmic. Anti-PEX14 IF: PEX14 absent (PEX14 LOF) vs PEX14 present (PEX5 LOF). NGS distinguishes. 1p36.22 vs 12p13.31.",
        },
        {
            "condition": "PEX13-ZSD (~1–2% all PBD, ~10–15 cases worldwide)",
            "distinction": "Biochemically IDENTICAL. KEY IF DISTINCTION: In PEX13 LOF, PEX13 IS ABSENT from membranes (integral PMP; SH3 domain). In PEX5 LOF, PEX13 IS PRESENT (importomer intact). Both: PMP70 PRESENT + PEX14 PRESENT + catalase cytoplasmic — but PEX13 absent in PEX13 LOF. Anti-PEX13 IF test separates them. NGS distinguishes. 2p15 vs 12p13.31.",
        },
        {
            "condition": "PEX1-ZSD (65% of all PBD — most common)",
            "distinction": "Biochemically IDENTICAL. PEX1 = D1 AAA-ATPase (retrotranslocation motor). In PEX1 LOF, PEX5 accumulates IN the peroxisomal membrane (retrotranslocation block — PEX5 docks normally but cannot exit). In PEX5 LOF, PEX5 is ABSENT from membrane (no receptor to dock). Anti-PEX5 IF + subcellular fractionation distinguishes: PEX1 LOF = PEX5 membrane-associated; PEX5 LOF = PEX5 absent from membrane. p.Gly843Asp founder allele in PEX1. NGS distinguishes.",
        },
        {
            "condition": "PEX19-ZSD (~1–3% all PBD)",
            "distinction": "Biochemically IDENTICAL. KEY: In PEX19 LOF, ALL Class I PMPs fail (PEX14 ABSENT + PMP70 ABSENT — ghost peroxisomes with lipid bilayer only). In PEX5 LOF, PEX14 PRESENT + PMP70 PRESENT — PEX19 is functional, inserting PMPs normally; only PEX5 receptor is absent. PMP70 status (present vs absent) is the primary IF separator: PEX19 LOF = PMP70−; PEX5 LOF = PMP70+. NGS confirms. 1q22 vs 12p13.31.",
        },
        {
            "condition": "PEX7-RCDP1 (Rhizomelic Chondrodysplasia Punctata Type 1)",
            "distinction": "PEX7 LOF: ONLY PTS2 import fails (PEX7 is the PTS2-specific receptor). PTS1 import INTACT (PEX5 is functional in RCDP1). Plasmalogens LOW (AGPS = PTS2 cargo → plasmalogen synthesis fails); phytanic elevated (PHYH = PTS2 cargo); VLCFA NORMAL (ACOX1 = PTS1 → VLCFA oxidation intact). In PEX5 LOF, BOTH pathways fail (VLCFA elevated + plasmalogens LOW + phytanic elevated). VLCFA elevation distinguishes PEX5-ZSD from RCDP1/PEX7. Rhizomelia pathognomonic in RCDP1.",
        },
        {
            "condition": "PEX6-ZSD (10% of all PBD)",
            "distinction": "Biochemically IDENTICAL. PEX6 = D2 AAA-ATPase motor (retrotranslocation). In PEX6 LOF, PEX5 accumulates in peroxisomal membrane (cannot exit after cargo delivery — retrotranslocation block). In PEX5 LOF, PEX5 is absent (no receptor to dock). PEX5 membrane accumulation (anti-PEX5 IF + membrane fraction) distinguishes PEX6 LOF from PEX5 LOF. p.Arg860Trp (R860W) founder allele in PEX6 → IRD 45%. NGS distinguishes.",
        },
        {
            "condition": "ABCD1 (X-linked Adrenoleukodystrophy)",
            "distinction": "VLCFA elevated in BOTH. ABCD1: plasmalogens (RBC) NORMAL + DHA NORMAL + adrenal insufficiency (universal, X-linked males). PHT/CBZ = ABSOLUTE CI in ABCD1 (adrenal crisis) vs RELATIVE CI in PEX5-ZSD. Lorenzo's Oil EFFECTIVE in ABCD1 (intact peroxisome with functional PEX5, transporter absent); INEFFECTIVE in PEX5-ZSD (cytosolic receptor absent — no PTS1 cargo delivery regardless of VLCFA elongase inhibition). HSCT standard CCALD; NOT in PEX5-ZSD.",
        },
        {
            "condition": "Adult Refsum Disease (PHYH deficiency)",
            "distinction": "PHYTANIC severely elevated (SOLE elevated peroxisomal metabolite); VLCFA NORMAL; plasmalogens NORMAL; DHA NORMAL. Adult-onset. No cortical migration defects. Phytol-restricted diet Level A GOLD STANDARD. Note: PHYH is a PTS2 cargo (imported via PEX7·PEX5L) — in PEX5 LOF, PHYH cannot be imported (PTS2 also fails via PEX7·PEX5L loss). In pure PHYH LOF, only phytanic/pristanic elevated (single-pathway failure). VLCFA normal vs elevated distinguishes PHYH-ARD from PEX5-ZSD.",
        },
    ]

    standards = [
        "Dodt G et al. Mutations in the PTS1 receptor gene, PXR1 (PEX5), define complementation group 2 of the peroxisome biogenesis disorders. Nat Genet. 1995",
        "Wiemer EA et al. Human peroxisomal targeting signal type 1 receptor, Pex5p, and the intracellular shuttling of thiolase. J Biol Chem. 1995",
        "Brocard C, Hartig A. Peroxisome targeting signal 1: is it really a simple tripeptide? Biochim Biophys Acta. 2006",
        "Gatto GJ Jr et al. DOOR domains in mammalian Pex5p define docking sites for PTS1. J Cell Biol. 2000",
        "Platta HW et al. Ubiquitination of the peroxisomal import receptor Pex5p is required for its recycling. J Cell Biol. 2007",
        "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
        "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
        "Fujiki Y et al. New insights into peroxisomal protein targeting pathways. Annu Rev Cell Dev Biol. 2020",
        "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
        "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
        "OMIM #614842 (PBD2A-ZSD — PEX5); #614843 (PBD2B-NALD — PEX5); *600414 (PEX5 gene)",
    ]

    return {
        "key_concepts": key_concepts,
        "diagnostic_algorithm": diagnostic_algorithm,
        "pharmacological_distinctions": pharmacological_distinctions,
        "differential_diagnosis": differential_diagnosis,
        "standards": standards,
    }
