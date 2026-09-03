#!/usr/bin/env python3
"""MT-CO2 — Leigh Syndrome / Multisystem CIV Deficiency / MELAS-CIV Overlap
   Cytochrome c Oxidase Subunit II — CuA BINUCLEAR COPPER CENTER — 227 aa / 26 kDa.

MT-CO2 (also COX2 / COII / MTCO2) is the SECOND mtDNA-encoded subunit of Complex IV
(Cytochrome c Oxidase / COX / CIV), located at mtDNA rCRS m.7586–8269 (H-strand).
227 amino acids / 26 kDa — the SMALLEST mtDNA-encoded CIV subunit; 2 transmembrane helices.
OMIM gene: *516040.

MT-CO2 CONTAINS THE CuA BINUCLEAR COPPER CENTER:
  The CuA center is a MIXED-VALENCE [Cu1.5+···Cu1.5+] BINUCLEAR CLUSTER located
  in the large hydrophilic C-terminal IMS domain of MT-CO2 (residues ~60–227).
  CuA is the PRIMARY ELECTRON ACCEPTOR from ferrocytochrome c (soluble IMS protein).
  Electron transfer chain at CIV: CuA (MT-CO2) → Heme a (MT-CO1) → Heme a3-CuB (MT-CO1) → O₂→ 2H₂O.
  CuA is the ONLY site in the entire ETC that receives electrons directly from cytochrome c.

STRUCTURE — 2 TRANSMEMBRANE HELICES (SMALLEST mtDNA CIV SUBUNIT):
  TM1 and TM2 anchor MT-CO2 in the IMM; the majority of the protein (aa ~60–227)
  forms the large IMS-facing hydrophilic domain bearing the CuA centre.
  MT-CO1 (514 aa, 12 TM — Assembly Nucleus + Heme a/a3-CuB) is synthesised first;
  MT-CO2 is the SECOND subunit added, joining the COX1-module sub-complex.
  MT-CO3 (261 aa, 7 TM — structural scaffold, no metal centres) completes the
  mtDNA-encoded trimer of the CIV core.

COPPER CHAPERONES FOR CuA (THE CLINICAL INTERSECTION):
  SCO1 (17p13.2) and SCO2 (22q13.33) — BOTH nuclear AR genes — are dedicated copper
  chaperones that deliver Cu+ ions to the CuA centre on MT-CO2.
  SCO2 additionally folds the MT-CO2 C-terminal domain via thiol-disulfide exchange.
  SCO2-deficient patients have 100% HCM — THE KEY DDx FROM MT-CO2 WHICH HAS NO HCM.
  SCO1-deficient patients have severe hepatopathy — THE KEY DDx FROM MT-CO2 (no liver).
  NDUFA4 directly contacts the COX2 IMS loop adjacent to the CuA zone (Zong 2018 CryoEM).

DISEASE PHENOTYPES:
  1. Classic CIV-Leigh infantile (m.7896G>A stop) — neonatal/infantile Leigh syndrome
  2. Multisystem CIV + peripheral neuropathy (m.8249G>A p.Arg205Trp CuA loop)
  3. MELAS-CIV overlap (m.8156T>C p.Leu128Pro TM2-IMS junction)
  4. KSS / CPEO / Pearson — large mtDNA deletion spanning CO2 region
  5. Late-onset CIV myopathy (m.7591G>A p.Val4Ile TM1 proximal) — mild/adult

Maternal inheritance (mtDNA). WES misses MT-CO2 — dedicated mtDNA sequencing required.
H-strand encoded (all 3 mtDNA CIV subunits are H-strand; unlike MT-ND6 which is L-strand).
"""

import random

GENE     = "MT-CO2"
ALIAS    = "COX2 / COII / MTCO2"
DISEASE  = "Classic CIV-Leigh / Multisystem CIV Neuropathy / MELAS-CIV Overlap / KSS-CPEO"
OMIM_G   = "516040"
OMIM_D   = "256000"   # Leigh syndrome; also 530000 (MELAS overlap)
INHERIT  = "Maternal (mtDNA, H-strand, m.7586–8269 rCRS)"
CHROM    = "mtDNA rCRS m.7586–8269 (H-strand)"
MODULE   = ("CuA Binuclear Copper Center — Primary electron acceptor from cytochrome c — "
            "2 TM helices (SMALLEST mtDNA CIV subunit); CuA→Heme a (CO1)→Heme a3-CuB→O₂")
SIZE     = "227 aa / 26 kDa (2 TM helices)"
SEED     = 747
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Classic CIV-Leigh infantile (m.7896G>A stop codon)",                  35),
    ("Multisystem CIV + peripheral neuropathy (m.8249G>A CuA loop)",        25),
    ("MELAS-CIV overlap / stroke-like episodes (m.8156T>C TM2-IMS)",        20),
    ("KSS / CPEO / Pearson (large deletion spanning CO2)",                   12),
    ("Late-onset CIV myopathy / mild (m.7591G>A TM1-proximal)",              8),
]

VARIANTS = [
    ("m.7896G>A",
     "p.Trp58Stop",
     "Premature stop at aa 58 — IMS domain entirely absent; CuA centre never formed",
     "Classic CIV-Leigh infantile",
     35,
     "Guanine-to-adenine at mtDNA position 7896 converts Trp58 UGG→UAG — a premature "
     "stop codon that eliminates the entire C-terminal IMS domain of MT-CO2, which bears "
     "the CuA binuclear copper centre. The resulting truncated peptide (57 aa) retains "
     "only TM1 and lacks the CuA-binding cysteine (Cys196, Cys200) and histidine "
     "(His161, His204) coordination residues entirely. CIV holocomplex is absent "
     "on BN-PAGE (CIV sub-complex not detected). CI/CII/CIII normal — isolated CIV "
     "fingerprint. Classic Leigh MRI: bilateral BG + brainstem T2 hyperintensity "
     "onset < 6 months. Lactic acidosis 8–20 mM. Heteroplasmy in blood "
     "underestimates muscle level by 15–30% — muscle biopsy mandatory."),
    ("m.8249G>A",
     "p.Arg205Trp",
     "C-terminal CuA coordination loop — Arg205 contacts Cu-ligand Cys200; CuA disrupted",
     "Multisystem CIV + peripheral neuropathy",
     25,
     "Guanine-to-adenine at m.8249 substitutes Arg205 (highly conserved, CuA "
     "coordination loop) with Trp. Arg205 forms a salt bridge with Glu198, stabilising "
     "the Cu-ligand Cys196–Cys200 cluster. The p.Arg205Trp substitution disrupts this "
     "bridge, causing CuA centre misassembly and CIV residual activity 5–18%. "
     "Unique phenotype: Leigh-like encephalopathy PLUS axonal peripheral neuropathy "
     "(sural nerve conduction velocity <30 m/s), sensory > motor. Peripheral neuropathy "
     "is rare in other CIV-Leigh genes — signals a specific CuA stability defect "
     "affecting peripheral axon mitochondria (high ETC demand, long axon length). "
     "BN-PAGE: CIV sub-complexes visible at reduced size; CIII2·CIV supercomplex absent."),
    ("m.8156T>C",
     "p.Leu128Pro",
     "TM2-IMS junction — helix-breaking proline disrupts TM2/IMS domain folding",
     "MELAS-CIV overlap / stroke-like episodes",
     20,
     "Thymine-to-cytosine at m.8156 converts Leu128 (at the cytoplasmic end of TM2, "
     "immediately before the IMS loop) to Pro. A proline at this position introduces a "
     "helix-breaking kink at the critical TM2/IMS boundary, destabilising the "
     "anchoring of the IMS domain. The C-terminal IMS domain collapses in part, "
     "impairing CuA centre formation. Heteroplasmy 65–85% in muscle. CIV residual "
     "8–22%. Clinical phenotype: MELAS-like stroke-like episodes (SLEs) co-occurring "
     "with CIV biochemical signature (CI often mildly reduced 60–80% due to secondary "
     "OXPHOS coupling; CIV reduced to 10–25%). DDx from MELAS (MT-TL1 m.3243A>G): "
     "CIV reduction favours MT-CO2 vs MT-TL1 which has moderate multi-complex reduction."),
    ("m.7591G>A",
     "p.Val4Ile near TM1 start",
     "TM1 proximal — Val4Ile near mature MT-CO2 N-terminus; mild CIV reduction",
     "Late-onset CIV myopathy / mild",
     8,
     "Guanine-to-adenine at m.7591 converts Val4 to Ile in the TM1 proximal region "
     "of mature MT-CO2. Val4 is conserved at the hydrophobic core of TM1; isoleucine "
     "is a conservative substitution but introduces minor steric clash with CIV "
     "nuclear subunit COX6a at the TM1 face. CIV residual 30–55% — the highest of "
     "all MT-CO2 pathogenic variants. Clinical phenotype: adult-onset proximal myopathy, "
     "exercise intolerance, elevated CK 2–4× ULN, mild lactic acidosis on exertion. "
     "No CNS involvement in pure form. COX-negative fibers 5–15% on muscle biopsy. "
     "Mother may be minimally symptomatic (exercise intolerance only). Heteroplasmy "
     "in blood often sufficient (70–90%). Mild course — best prognosis of all MT-CO2 phenotypes."),
    ("Large deletion spanning m.7586–8269 (CO2 region)",
     "Partial/complete MT-CO2 deletion",
     "Large mtDNA deletion — eliminates MT-CO2 coding; combined CI+CIV deficiency",
     "KSS / CPEO / Pearson (large deletion spanning CO2)",
     12,
     "Large-scale mtDNA deletions (typical size 4–8 kb) often span the MT-CO2 locus "
     "(m.7586–8269) as part of the common deletion breakpoint region. When CO2 is "
     "eliminated, both MT-CO1 assembly and CuA formation fail in the deleted "
     "mtDNA genome proportion. Heteroplasmy determines phenotype severity: "
     "<50% → KSS (ophthalmoplegia + retinopathy + cardiac conduction); "
     "30–60% → CPEO (progressive external ophthalmoplegia, adult onset); "
     "<30% → Pearson syndrome (sideroblastic anaemia + pancreatic exocrine failure). "
     "Combined CI + CIV deficiency on RC analysis (unlike point mutations → isolated CIV). "
     "Long-read sequencing or Southern blot required — not detected by NGS short-read. "
     "COX-negative / SDH-positive fibers proportion correlates with deletion load."),
]

SEIZURE_TYPES = [
    ("Focal seizures (occipital / temporal onset in SLEs)",      38),
    ("Generalized tonic-clonic (GTCS) breakthrough",             28),
    ("Myoclonic seizures (cortical myoclonus in MELAS overlap)", 18),
    ("Infantile spasms / West syndrome (Leigh onset)",           12),
    ("Absence seizures (mild/late-onset cases)",                   4),
]

TRIGGERS = [
    ("Febrile illness / infection → metabolic decompensation",   82),
    ("Fasting / nil-by-mouth period (CIV bottleneck crisis)",    76),
    ("Physical exertion (high ETC demand → CuA bottleneck)",     58),
    ("Perioperative stress (propofol ABSOLUTE CI + fasting)",    45),
    ("High mitochondrial toxin exposure (Metformin, VPA)",       34),
    ("Rapid growth phase (infant — mtDNA replication demand)",   28),
    ("Heat / hyperthermia (enhanced ROS at CuA-disrupted site)", 22),
    ("Subtherapeutic CoQ10 / ascorbate supply",                  16),
]

TREATMENTS = [
    ("CoQ10 / Ubiquinol",
     "Level B (CIV support)",
     "CoQH2 pool → reduced cytochrome c → CuA; ubiquinol preferred over ubiquinone; "
     "10–30 mg/kg/day; enhances residual CuA electron flux in MT-CO2-CIV deficiency"),
    ("Ascorbate (Vitamin C)",
     "Level B (cytochrome c reduction)",
     "Ascorbate maintains ferrocytochrome c (reduced) → maximises electron donation rate "
     "to CuA on MT-CO2; 500–2000 mg/day; especially useful in CIV-Leigh and MELAS-CIV"),
    ("Thiamine (B1)",
     "Mandatory empiric (treatable mimics)",
     "Empiric thiamine 100–300 mg/day; rules out SLC19A3 BTBGD and PDHC deficiency "
     "as treatable Leigh/BSN mimics; give BEFORE mtDNA result returns"),
    ("Biotin",
     "Mandatory empiric (treatable mimics)",
     "Empiric biotin 10–40 mg/day; biotinidase deficiency and BTBGD mimic CIV-Leigh; "
     "co-administer with thiamine empirically; safe, inexpensive, potentially life-saving"),
    ("Riboflavin (B2)",
     "Level C (CIV support)",
     "FADH2 supports CII bypass; limited benefit in isolated CIV vs CI deficiency; "
     "gives marginal CI-CIV supercomplex stabilisation; 100–400 mg/day"),
    ("Glucose infusion rate (GIR 6–8 mg/kg/min)",
     "Mandatory during any fasting/NBO period",
     "IV dextrose GIR 6–8 mg/kg/min; MANDATORY when nil-by-mouth; prevents "
     "β-oxidation → CIV bottleneck crisis; especially critical for perioperative care"),
    ("Levetiracetam (LEV) — preferred AED",
     "Level C",
     "LEV has no mitochondrial toxicity; preferred AED in MT-CO2-CIV; "
     "avoid valproate (VPA ABSOLUTE CI); avoid phenobarbitone if possible (CI inhibitor)"),
    ("L-Arginine IV (acute SLE / MELAS-CIV overlap)",
     "Level B (MELAS-CIV only)",
     "L-arginine 500 mg/kg IV during acute stroke-like episode; NO donor → "
     "vasodilation → improved cerebral perfusion; specific to MELAS-CIV overlap phenotype; "
     "NOT indicated in pure Leigh / neuropathy phenotype"),
]

CONTRAINDICATIONS = [
    ("Propofol",
     "ABSOLUTE CI — direct CIV inhibitor + PRIS",
     "Propofol DIRECTLY inhibits the heme a3-CuB terminal oxygen-reduction site in MT-CO1; "
     "CuA blockade proximal to this (MT-CO2) further exacerbates ETC shutdown; "
     "Propofol Infusion Syndrome (PRIS) causes fatal lactic acidosis; use SEVOFLURANE instead"),
    ("Metformin",
     "ABSOLUTE CI — Complex I inhibitor compounding CIV failure",
     "Metformin inhibits CI (ND1/ND2 subunit proton pump) → NADH backed up → "
     "electron pressure compounds CIV bottleneck at CuA (MT-CO2); "
     "risk of life-threatening lactic acidosis in any CIV deficiency; NEVER use"),
    ("Valproic acid (VPA)",
     "ABSOLUTE CI — CoA sequestration + POLG inhibition + CI inhibition",
     "VPA sequesters CoA → carnitine depletion → OXPHOS substrate deprivation; "
     "VPA is a POLG inhibitor → mtDNA depletion compounding MT-CO2 template loss; "
     "CI inhibitor directly; hepatotoxic in mt disease; use LEV instead"),
    ("Linezolid",
     "ABSOLUTE CI — blocks mt 23S rRNA → prevents MT-CO2 synthesis",
     "Linezolid inhibits the mt 70S ribosome 23S rRNA peptidyl transferase centre; "
     "blocks translation of MT-CO1, MT-CO2, and MT-CO3 — all 3 mtDNA CIV subunits; "
     "CIV holoenzyme cannot assemble; fatal CIV deficiency exacerbation"),
    ("Chloramphenicol",
     "ABSOLUTE CI — mt 50S ribosome inhibitor",
     "Chloramphenicol blocks the mt 50S (23S-equivalent) ribosome peptidyl transferase; "
     "prevents MT-CO2 translation; same mechanism as linezolid; pancytopenia risk; "
     "NEVER use in any mtDNA disease"),
    ("Ketogenic diet (KD)",
     "CONTRAINDICATED — CIV bottleneck exacerbated by FA oxidation",
     "KD → β-oxidation → excess FADH2 via CII → CoQ pool fully reduced → "
     "electron pressure at CuA site of MT-CO2 → OXPHOS crisis; "
     "KD is beneficial in CI deficiency (succinate CII-bypass) but HARMFUL in CIV deficiency"),
    ("Fasting / prolonged NBO",
     "ABSOLUTELY AVOID — triggers CIV metabolic crisis",
     "Any fasting beyond 4 hours → β-oxidation dominant → CIV bottleneck → "
     "crisis; mandatory GIR 6–8 during all NBO periods; never delay GIR while "
     "investigating fever in MT-CO2 patients"),
]

MONITORING = [
    ("Plasma lactate",                    "Baseline + q3M; target <2 mM at rest; >5 mM signals decompensation"),
    ("Respiratory chain enzymology",      "Muscle biopsy at diagnosis; CIV <30% residual confirms pathogenicity"),
    ("Brain MRI (T2/FLAIR/DWI)",         "At diagnosis + at decompensation; Leigh lesions + SLE diffusion restriction"),
    ("Echocardiography",                  "Baseline to CONFIRM NO HCM (exclude SCO2); q12–24M ongoing"),
    ("Liver function tests (LFT/GGT)",   "Baseline to confirm no hepatopathy (exclude SCO1); q12M"),
    ("EMG / NCS (nerve conduction)",      "Baseline; especially for m.8249G>A neuropathy phenotype; annual"),
    ("Ophthalmology (fundoscopy/ERG)",   "Baseline; KSS → retinopathy + KSS-pigmentary; annual"),
    ("ECG / Holter",                      "KSS → heart block; q6M in deletion phenotype"),
    ("Developmental assessment",          "q6M in infantile-onset Leigh; Bayley/Vineland scales"),
    ("mtDNA heteroplasmy in muscle",     "Quantitative qPCR at diagnosis; blood underestimates by 15–30%"),
]

REFERENCES = [
    "Jaksch M et al. 2000 Hum Mol Genet — First MT-CO2 patient series: Leigh + multisystem CIV",
    "Tsukihara T et al. 1996 Science — CIV crystal structure (bovine): CuA domain MT-CO2 residues",
    "Leary SC et al. 2007 Hum Mol Genet — SCO2/SCO1 deliver copper to MT-CO2 CuA center",
    "Weraarpachai W et al. 2009 Nat Genet — COX20 (FAM36A): MT-CO2-specific early assembly chaperone",
    "Zong S et al. 2018 Cell — CryoEM CIV structure: NDUFA4 contacts COX2 CuA IMS loop",
    "Sacconi S et al. 2005 Arch Neurol — MT-CO2 mutations: myopathy + neuropathy phenotype",
    "Gelfand JM & Rosenbaum JT 2004 — MT-CO2 MELAS-CIV overlap: SLE with CIV biochemistry",
    "Stroud DA et al. 2015 Cell Metab — CIV sub-complex survey: CO1-module precedes CO2 addition",
    "Rahman S & Thorburn D 2015 Gene Reviews — Mitochondrial complex I-IV deficiencies overview",
    "Gorman GS et al. 2016 Nat Rev Dis Primers — Mitochondrial disease epidemiology/management",
]

KEY_CONCEPTS = [
    ("CuA Binuclear Copper Center — ONLY ETC site receiving electrons from cytochrome c",
     "The CuA centre in MT-CO2 is a MIXED-VALENCE binuclear [Cu1.5+···Cu1.5+] cluster, "
     "coordinated by His161, Cys196, Cys200, His204, Met207, and Trp56. "
     "It is the UNIQUE ENTRY POINT for electrons donated by ferrocytochrome c (soluble IMS protein). "
     "Electrons flow: CuA (CO2) → Heme a (CO1) → Heme a3-CuB (CO1) → O₂. "
     "MT-CO2 mutations that disrupt CuA prevent electron entry into CIV — "
     "the entire oxygen-reduction cascade stalls, backed up to cytochrome c → CI → NADH."),
    ("SCO2 DDx — 100% HCM; MT-CO2 has NO HCM",
     "SCO2 (22q13.33) is the copper chaperone that delivers Cu+ to the CuA centre ON MT-CO2. "
     "SCO2 deficiency → CuA misloaded → CIV deficiency; PLUS SCO2 is highly expressed in myocardium. "
     "SCO2-deficient patients develop HYPERTROPHIC CARDIOMYOPATHY IN 100% OF CASES — "
     "this is the critical DDx: if CIV-Leigh + HCM, suspect SCO2 (AR nuclear) not MT-CO2 (maternal). "
     "MT-CO2 mutations cause isolated CIV WITHOUT HCM. "
     "Echocardiography to exclude HCM is MANDATORY at diagnosis of any CIV-Leigh."),
    ("SCO1 DDx — Hepatopathy; MT-CO2 has NO liver disease",
     "SCO1 (17p13.2) delivers copper to MT-CO2 CuA via a complementary thiol-based mechanism. "
     "SCO1 deficiency → severe infantile hepatopathy (transaminases >10× ULN, hyperbilirubinaemia) "
     "AND Leigh-like CIV deficiency. MT-CO2 mutations do NOT cause hepatopathy. "
     "LFTs are the first discriminator: hepatopathy → SCO1 (AR, 17p13.2); no hepatopathy → MT-CO2 or SCO2. "
     "Note: VPA is hepatotoxic and would artificially elevate LFTs — check before VPA exposure."),
    ("COX20 (FAM36A) — MT-CO2 specific early assembly chaperone",
     "COX20 (also FAM36A, AR nuclear 1p33) is an IMM integral protein that acts as the "
     "FIRST dedicated chaperone for MT-CO2 SPECIFICALLY (analogous to COA3/MITRAC for MT-CO1). "
     "COX20 stabilises newly synthesised MT-CO2 TM helices in the IMM before the C-terminal "
     "IMS domain is folded and the CuA copper is inserted by SCO1/SCO2. "
     "COX20 deficiency → AR CIV-Leigh with ataxia — distinguishable by AR inheritance pattern "
     "(unlike MT-CO2 which is maternal). COX20 is released once CuA assembly is complete."),
    ("Isolated CIV Deficiency — CI/CII/CIII NORMAL",
     "MT-CO2 point mutations (m.7896G>A, m.8249G>A, m.8156T>C, m.7591G>A) → ISOLATED CIV deficiency: "
     "CIV residual 5–55% (depending on variant severity); CI/CII/CIII biochemically NORMAL. "
     "Combined CI + CIV deficiency signals large mtDNA deletion (CO2 within deletion) "
     "or a mtDNA depletion syndrome (POLG/TWNK/DGUOK — all ETC complexes reduced). "
     "Isolated vs combined on RC enzymology is the FIRST BRANCH POINT in MT-CO2 workup."),
    ("WES MISSES MT-CO2 — Dedicated mtDNA Sequencing Required",
     "WES panels typically exclude mtDNA or have poor coverage of MT-CO2 (m.7586–8269). "
     "Pathogenic MT-CO2 point mutations (especially heteroplasmic) are NOT reliably detected "
     "by standard WES. Dedicated mtDNA sequencing (MitoSeq / high-coverage NGS) is required. "
     "Blood underestimates muscle heteroplasmy by 15–30% — MUSCLE BIOPSY is the gold standard. "
     "Large deletions (KSS/CPEO phenotype) require long-read sequencing or Southern blot. "
     "L-strand vs H-strand coverage must be verified: MT-CO2 is H-strand (verify H-strand NGS QC)."),
    ("GIR 6–8 — NEVER FAST in CIV-Leigh / MELAS-CIV overlap",
     "Glucose infusion rate 6–8 mg/kg/min IV during ANY nil-by-mouth period. "
     "Fasting → β-oxidation dominant → FADH2 excess via CII → CoQ pool reduced → "
     "electron pressure at CuA bottleneck (MT-CO2) → metabolic crisis. "
     "MT-CO2 patients decompensate acutely with febrile illness: GIR + antipyretics are FIRST-LINE. "
     "For MELAS-CIV overlap: L-arginine IV during acute SLE (vasodilation), GIR maintained."),
    ("Propofol / PRIS — ABSOLUTE CI in all CIV disorders",
     "Propofol directly inhibits the heme a3-CuB terminal oxygen reduction site in MT-CO1. "
     "In MT-CO2 deficiency, CuA electron delivery to heme a is already impaired; "
     "propofol compounds this by blocking downstream heme a3-CuB — double CIV hit. "
     "PRIS (Propofol Infusion Syndrome) causes fatal lactic acidosis via mitochondrial uncoupling. "
     "SEVOFLURANE is the safe anaesthetic alternative in all CIV disorders."),
]


def _make_patients():
    pats = []
    for i in range(1, N + 1):
        r = rng.random() * 100
        if   r < 35:  cls = PHENO_CLASSES[0][0]
        elif r < 60:  cls = PHENO_CLASSES[1][0]
        elif r < 80:  cls = PHENO_CLASSES[2][0]
        elif r < 92:  cls = PHENO_CLASSES[3][0]
        else:         cls = PHENO_CLASSES[4][0]

        is_leigh   = "Classic CIV-Leigh" in cls
        is_neuro   = "neuropathy" in cls
        is_melas   = "MELAS" in cls
        is_kss     = "KSS" in cls
        is_mild    = "Late-onset" in cls

        if is_leigh:   v = VARIANTS[0]
        elif is_neuro: v = VARIANTS[1]
        elif is_melas: v = VARIANTS[2]
        elif is_mild:  v = VARIANTS[3]
        else:          v = VARIANTS[4]   # KSS large deletion

        pats.append({
            "id":                      f"P{i:02d}",
            "phenotype":               cls,
            "variant":                 v[0],
            "cDNA":                    v[1],
            "civ_pct":                 (rng.randint(3, 15)  if is_leigh
                                        else rng.randint(5, 18) if is_neuro
                                        else rng.randint(8, 22) if is_melas
                                        else rng.randint(8, 35) if is_kss
                                        else rng.randint(30, 55)),
            "onset_mo":                (rng.randint(1, 6)   if is_leigh
                                        else rng.randint(6, 60)  if is_neuro
                                        else rng.randint(12, 120) if is_melas
                                        else rng.randint(24, 240) if is_kss
                                        else rng.randint(120, 480)),
            "has_seizure":             rng.random() < (0.68 if is_leigh or is_melas else
                                                        0.42 if is_neuro else
                                                        0.35 if is_kss else 0.12),
            "has_hypotonia":           rng.random() < (0.90 if is_leigh else
                                                        0.55 if is_neuro else
                                                        0.42 if is_melas else
                                                        0.28 if is_kss else 0.20),
            "has_lactic_acidosis":     rng.random() < (0.90 if is_leigh else
                                                        0.72 if is_neuro else
                                                        0.82 if is_melas else
                                                        0.58 if is_kss else 0.32),
            "has_leigh_mri":           rng.random() < (0.88 if is_leigh else
                                                        0.45 if is_neuro else 0.22),
            "has_sle_mri":             is_melas and rng.random() < 0.80,
            "has_neuropathy":          is_neuro or (rng.random() < 0.08),
            "has_ophthalmo":           rng.random() < (0.88 if is_kss else 0.05),
            "has_retinopathy":         rng.random() < (0.78 if is_kss else 0.03),
            "has_heart_block":         rng.random() < (0.55 if is_kss else 0.02),
            "has_hcm":                 False,   # NO HCM — KEY DDx SCO2
            "has_hepatopathy":         False,   # NO hepatopathy — KEY DDx SCO1
            "has_respiratory":         rng.random() < (0.40 if is_leigh else 0.15),
        })
    return pats


_PATIENTS = _make_patients()


def get_overview():
    pts = _PATIENTS
    n_sz   = sum(1 for p in pts if p["has_seizure"])
    n_hyp  = sum(1 for p in pts if p["has_hypotonia"])
    n_lac  = sum(1 for p in pts if p["has_lactic_acidosis"])
    n_mri  = sum(1 for p in pts if p["has_leigh_mri"])
    n_sle  = sum(1 for p in pts if p["has_sle_mri"])
    n_neu  = sum(1 for p in pts if p["has_neuropathy"])
    n_resp = sum(1 for p in pts if p["has_respiratory"])

    variant_counts: dict = {}
    for p in pts:
        variant_counts[p["variant"]] = variant_counts.get(p["variant"], 0) + 1

    phenotype_counts: dict = {}
    for p in pts:
        phenotype_counts[p["phenotype"]] = phenotype_counts.get(p["phenotype"], 0) + 1

    return {
        "gene":           GENE,
        "alias":          ALIAS,
        "disease":        DISEASE,
        "omim_gene":      OMIM_G,
        "omim_disease":   OMIM_D,
        "inheritance":    INHERIT,
        "chromosome":     CHROM,
        "module":         MODULE,
        "protein_size":   SIZE,
        "cohort_n":       N,
        "seed":           SEED,
        "kpis": {
            "seizures_pct":          round(n_sz  / N * 100),
            "hypotonia_pct":         round(n_hyp / N * 100),
            "lactic_acidosis_pct":   round(n_lac / N * 100),
            "leigh_mri_pct":         round(n_mri / N * 100),
            "sle_mri_pct":           round(n_sle / N * 100),
            "neuropathy_pct":        round(n_neu / N * 100),
            "respiratory_pct":       round(n_resp / N * 100),
            "mean_civ_pct":          round(sum(p["civ_pct"] for p in pts) / N, 1),
            "median_onset_mo":       round(
                sorted(p["onset_mo"] for p in pts)[N // 2], 1),
            "hcm_pct":               0,   # NO HCM — KEY DDx SCO2
            "hepatopathy_pct":       0,   # NO hepatopathy — KEY DDx SCO1
        },
        "phenotype_distribution": [
            {"class": pc[0], "n": phenotype_counts.get(pc[0], 0), "pct": pc[1]}
            for pc in PHENO_CLASSES
        ],
        "top_variants":  sorted(variant_counts.items(), key=lambda x: -x[1]),
        "seizure_types": [{"type": t, "pct": p} for t, p in SEIZURE_TYPES],
        "triggers":      [{"trigger": t, "pct": p} for t, p in TRIGGERS],
        "references":    REFERENCES,
        "key_concepts":  [{"concept": c[0], "detail": c[1]} for c in KEY_CONCEPTS],
    }


def get_breakdown():
    pts = _PATIENTS
    variant_rows = []
    for vname, aa, struct, phenotype_modal, freq, detail in VARIANTS:
        vpts = [p for p in pts if p["variant"] == vname]
        variant_rows.append({
            "variant":           vname,
            "amino_acid":        aa,
            "structural_impact": struct,
            "modal_phenotype":   phenotype_modal,
            "freq_pct":          freq,
            "n_in_cohort":       len(vpts),
            "detail":            detail,
        })

    return {
        "gene":      GENE,
        "cohort_n":  N,
        "variants":  variant_rows,
        "patients": [
            {
                "id":            p["id"],
                "phenotype":     p["phenotype"],
                "onset_mo":      p["onset_mo"],
                "variant":       p["variant"],
                "amino_acid":    p["cDNA"],
                "civ_pct":       p["civ_pct"],
                "seizure":       p["has_seizure"],
                "hypotonia":     p["has_hypotonia"],
                "lactic_ac":     p["has_lactic_acidosis"],
                "leigh_mri":     p["has_leigh_mri"],
                "sle_mri":       p["has_sle_mri"],
                "neuropathy":    p["has_neuropathy"],
                "ophthalmo":     p["has_ophthalmo"],
                "retinopathy":   p["has_retinopathy"],
                "heart_block":   p["has_heart_block"],
                "hcm":           p["has_hcm"],
                "hepatopathy":   p["has_hepatopathy"],
            }
            for p in pts
        ],
        "treatments":        [
            {"name": t[0], "evidence": t[1], "notes": t[2]}
            for t in TREATMENTS
        ],
        "contraindications": [
            {"drug": c[0], "class": c[1], "reason": c[2]}
            for c in CONTRAINDICATIONS
        ],
        "monitoring":        [
            {"item": m[0], "protocol": m[1]}
            for m in MONITORING
        ],
        "seizure_types":     [{"type": t, "pct": p} for t, p in SEIZURE_TYPES],
        "triggers":          [{"trigger": t, "pct": p} for t, p in TRIGGERS],
    }


def get_definitions():
    return {
        "gene":     GENE,
        "glossary": [
            {"term": "MT-CO2 / COX2 / COII",
             "definition": ("227 aa / 26 kDa mtDNA H-strand encoded Complex IV subunit 2; "
                            "SMALLEST mtDNA CIV subunit; 2 TM helices; CuA binuclear copper centre; "
                            "OMIM *516040; m.7586–8269 rCRS")},
            {"term": "CuA Binuclear Copper Center",
             "definition": ("Mixed-valence [Cu1.5+···Cu1.5+] cluster in MT-CO2 C-terminal IMS domain; "
                            "ONLY ETC site receiving electrons from cytochrome c; "
                            "coordinated by His161, Cys196, Cys200, His204, Met207, Trp56; "
                            "electrons: CuA→Heme a (CO1)→Heme a3-CuB (CO1)→O₂")},
            {"term": "SCO2 (22q13.33)",
             "definition": ("Synthesis of Cytochrome c Oxidase 2 — AR nuclear copper chaperone; "
                            "delivers Cu+ to CuA centre on MT-CO2 + folds MT-CO2 C-terminus; "
                            "SCO2 deficiency → CIV-Leigh + 100% HCM — CRITICAL DDx: "
                            "MT-CO2 mutations cause NO HCM; echocardiography is mandatory")},
            {"term": "SCO1 (17p13.2)",
             "definition": ("Synthesis of Cytochrome c Oxidase 1 — AR nuclear copper chaperone; "
                            "delivers copper to MT-CO2 CuA via complementary mechanism; "
                            "SCO1 deficiency → hepatopathy (LFTs >10× ULN) + CIV-Leigh; "
                            "MT-CO2 mutations cause NO hepatopathy — LFTs are the DDx pivot")},
            {"term": "COX20 (FAM36A, 1p33)",
             "definition": ("MT-CO2-specific IMM early assembly chaperone; first factor stabilising "
                            "nascent MT-CO2 TM helices; released after CuA copper insertion by SCO1/SCO2; "
                            "COX20 deficiency → AR CIV-Leigh + ataxia (AR inheritance vs MT-CO2 maternal)")},
            {"term": "m.7896G>A (p.Trp58Stop)",
             "definition": ("Premature stop at MT-CO2 aa 58; eliminates entire C-terminal IMS domain; "
                            "CuA centre never formed; CIV absent on BN-PAGE; neonatal/infantile Leigh; "
                            "most common MT-CO2 pathogenic variant (~35%)")},
            {"term": "m.8249G>A (p.Arg205Trp)",
             "definition": ("CuA coordination loop Arg205; disrupts Cys196-Cys200 stabilising salt bridge; "
                            "CuA misassembly; CIV 5–18% residual; unique phenotype: Leigh + axonal neuropathy; "
                            "peripheral neuropathy in CIV-Leigh favours MT-CO2 CuA domain involvement")},
            {"term": "Isolated CIV Deficiency",
             "definition": ("CIV 5–55% residual; CI/CII/CIII NORMAL — MT-CO2 point-mutation fingerprint; "
                            "combined CI+CIV signals large deletion; all-ETC-low signals depletion (POLG/TWNK); "
                            "isolated vs combined on RC enzymology is the diagnostic first branch point")},
            {"term": "MELAS-CIV Overlap",
             "definition": ("MT-CO2 m.8156T>C (p.Leu128Pro) → MELAS-like stroke-like episodes (SLEs) "
                            "with CIV biochemistry; DDx from MT-TL1 m.3243A>G MELAS: CIV reduction "
                            "favours MT-CO2 vs multi-complex reduction in MT-TL1 MELAS")},
            {"term": "Propofol / PRIS",
             "definition": ("Propofol directly inhibits heme a3-CuB in MT-CO1 + causes PRIS; "
                            "in MT-CO2 deficiency CuA→heme a electron delivery is impaired; "
                            "propofol compounds this by blocking downstream heme a3-CuB; "
                            "ABSOLUTE CI; use sevoflurane for anaesthesia")},
            {"term": "GIR 6–8",
             "definition": ("Glucose infusion rate 6–8 mg/kg/min IV dextrose during nil-by-mouth; "
                            "MANDATORY in MT-CO2-CIV; fasting → β-oxidation → CuA bottleneck crisis; "
                            "first-line management during febrile illness decompensation")},
            {"term": "H-strand (MT-CO2)",
             "definition": ("MT-CO2 encoded on mtDNA Heavy strand (m.7586–8269 rCRS); "
                            "all 3 mtDNA CIV subunits (CO1/CO2/CO3) are H-strand encoded; "
                            "H-strand NGS coverage verification required in MitoSeq QC")},
            {"term": "KSS / CPEO / Pearson",
             "definition": ("Large deletion phenotypes when deletion spans MT-CO2 locus; "
                            "combined CI+CIV deficiency (not isolated CIV); "
                            "KSS: ophthalmoplegia+retinopathy+heart block; "
                            "Pearson: sideroblastic anaemia+pancreatic failure; "
                            "Southern blot or long-read sequencing required for detection")},
        ],
        "references": REFERENCES,
    }


if __name__ == "__main__":
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first 500 chars) ===")
    print(_json.dumps(get_breakdown(), indent=2)[:500])
    print("\n=== DEFINITIONS (first 500 chars) ===")
    print(_json.dumps(get_definitions(), indent=2)[:500])
