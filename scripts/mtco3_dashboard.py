#!/usr/bin/env python3
"""MT-CO3 — Exercise-Intolerance Myopathy / CIV-Leigh / MELAS-CIV Overlap
   Cytochrome c Oxidase Subunit III — PROTON EXIT PATHWAY — 261 aa / 30 kDa.

MT-CO3 (also COX3 / COIII / MTCO3) is the THIRD mtDNA-encoded subunit of Complex IV
(Cytochrome c Oxidase / COX / CIV), located at mtDNA rCRS m.9207–9990 (H-strand).
261 amino acids / 30 kDa — 7 transmembrane helices — the MOST TM helices of the 3
mtDNA-encoded CIV subunits. OMIM gene: *516050.

MT-CO3 IS THE PROTON EXIT PATHWAY / STRUCTURAL SCAFFOLD:
  MT-CO3 contains NO metal centres (no copper, no haem).
  Its 7 TM helices form the structural perimeter of the CIV core trimer
  (MT-CO1 + MT-CO2 + MT-CO3), wrapping the back face of MT-CO1.
  MT-CO3 is required for stability of the MT-CO1/MT-CO2 catalytic dimer:
    without MT-CO3, the COX1-COX2 sub-complex is destabilised on BN-PAGE.
  Proton exit pathway: MT-CO3 TM5-TM6 cleft is part of the K-channel/D-channel
    proton exit route from the IMM matrix face to the IMS; mutations in this
    region uncouple proton pumping without abolishing electron transfer.

STRUCTURE — 7 TRANSMEMBRANE HELICES (MOST TM among 3 mtDNA CIV subunits):
  MT-CO1 (514 aa, 12 TM — Assembly Nucleus + Heme a/a3-CuB) is synthesised first;
  MT-CO2 (227 aa, 2 TM — CuA centre) joins the CO1-module next;
  MT-CO3 (261 aa, 7 TM — structural scaffold, proton exit) completes the
  mtDNA-encoded TRIMER at a late step of CIV assembly.
  Nuclear subunits COX7A, COX7B, COX8A surround MT-CO3 on the peripheral rim.
  LRPPRC stabilises all 13 mt-mRNAs including MT-CO3 mRNA — LRPPRC mutations
  cause combined CI+CIV deficiency (not isolated) — a DDx anchor.

UNIQUE PHENOTYPIC SIGNATURE OF MT-CO3 DISEASE:
  1. EXERCISE INTOLERANCE + MYOPATHY — the most common adult MT-CO3 phenotype.
     m.9438G>A (p.Gly78Asp TM4 disruption) is the classic adult myopathy variant.
  2. CIV-LEIGH (infantile) — severe frameshift or truncation mutations.
  3. MELAS-CIV OVERLAP — stroke-like episodes with CIV reduction.
  4. KSS / CPEO / Pearson — large mtDNA deletion spanning CO3.
  5. ENCEPHALOMYOPATHY — encephalopathy + myopathy combined.

Maternal inheritance (mtDNA). WES misses MT-CO3 — dedicated mtDNA sequencing required.
All 3 mtDNA CIV subunits (MT-CO1/CO2/CO3) are H-strand encoded; unlike MT-ND6 (L-strand).
NO METAL CENTRES in MT-CO3 — critical DDx from MT-CO1 (Heme a/a3-CuB) and MT-CO2 (CuA).
"""

import random

GENE     = "MT-CO3"
ALIAS    = "COX3 / COIII / MTCO3"
DISEASE  = "Exercise-Intolerance Myopathy / CIV-Leigh / MELAS-CIV Overlap / Encephalomyopathy"
OMIM_G   = "516050"
OMIM_D   = "256000"   # Leigh syndrome; also MELAS overlap
INHERIT  = "Maternal (mtDNA, H-strand, m.9207–9990 rCRS)"
CHROM    = "mtDNA rCRS m.9207–9990 (H-strand)"
MODULE   = ("Proton Exit Pathway / Structural Scaffold — NO metal centres — "
            "7 TM helices (MOST among 3 mtDNA CIV subunits); wraps MT-CO1 back face; "
            "stabilises MT-CO1/MT-CO2 catalytic dimer; K-channel/D-channel proton exit")
SIZE     = "261 aa / 30 kDa (7 TM helices — LARGEST of 3 mtDNA CIV subunits)"
SEED     = 751
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    ("Exercise intolerance / adult-onset myopathy (m.9438G>A TM4)",                  35),
    ("CIV-Leigh infantile (m.9537InsC frameshift / truncation)",                       25),
    ("MELAS-CIV overlap / stroke-like episodes (m.9957T>C TM7 junction)",             18),
    ("KSS / CPEO / Pearson (large deletion spanning CO3)",                            12),
    ("Encephalomyopathy / multisystem CIV (m.9804G>A TM6-IMS loop)",                 10),
]

VARIANTS = [
    ("m.9438G>A",
     "p.Gly78Asp (TM4 hydrophobic core)",
     "TM4 mid-helix — Gly78 Glycine flexibility lost; kink disrupts MT-CO1 back-face contact",
     "Exercise intolerance / adult-onset myopathy",
     35,
     "Guanine-to-adenine at mtDNA position 9438 converts Gly78 to Asp in the centre of TM4. "
     "Gly78 provides critical helix flexibility at the MT-CO1 contact face of MT-CO3 TM4; "
     "replacement with charged aspartate introduces a local kink and disrupts the hydrophobic "
     "packing between MT-CO3 TM4 and MT-CO1 TM10. The contact surface perturbation "
     "destabilises the MT-CO1/MT-CO2/MT-CO3 trimer interface. CIV residual 25–45% — "
     "sufficient for resting metabolism but insufficient for exercise-induced peak ATP demand. "
     "Clinical phenotype: adult-onset exercise intolerance, myalgias, proximal muscle weakness, "
     "elevated CK 2–5× ULN, exertional lactic acidosis (lactate >6 mM post-exercise), "
     "COX-negative fibres 10–20% on muscle biopsy (ragged-red fibres rare). "
     "No or minimal CNS involvement in the pure myopathy form. "
     "Heteroplasmy in blood often underestimates muscle level — muscle biopsy qPCR mandatory. "
     "EXERCISE INTOLERANCE as the leading phenotype distinguishes MT-CO3 m.9438G>A from "
     "MT-CO1/MT-CO2 (more severe, earlier onset). Best prognosis of all MT-CO3 phenotypes."),
    ("m.9537InsC",
     "Frameshift → premature stop (Leigh-severe truncation)",
     "Single C insertion after m.9537 → frameshift → premature stop → MT-CO3 truncated aa <100",
     "CIV-Leigh infantile",
     25,
     "Insertion of a single cytosine at mtDNA m.9537 causes a frameshift beginning at "
     "codon ~111, generating a premature stop codon that truncates MT-CO3 to fewer than "
     "120 amino acids. The resulting peptide lacks TM helices 4–7 entirely and cannot "
     "integrate into the CIV assembly pathway. On BN-PAGE, CIV holocomplex is absent; "
     "only a faint COX1-module intermediate is detectable. CIV residual <5% in muscle. "
     "CI/CII/CIII are biochemically normal — isolated CIV fingerprint. "
     "Clinical phenotype: classic CIV-Leigh syndrome, neonatal or early infantile onset "
     "(median 3 months), bilateral basal ganglia and brainstem T2 hyperintensities "
     "(Leigh MRI pattern), severe lactic acidosis (lactate 8–20 mM), hypotonia, "
     "psychomotor regression, seizures in 60–70%, respiratory compromise. "
     "High heteroplasmy (>80%) in muscle required for this severity. "
     "Blood heteroplasmy may be 20–40% lower than muscle — NEVER rely on blood alone."),
    ("m.9957T>C",
     "p.Leu246Pro (TM7-IMS junction — helix-breaking proline)",
     "TM7 C-terminal end — Leu246Pro kink disrupts MT-CO3/MT-CO1 lateral dimer stability",
     "MELAS-CIV overlap / stroke-like episodes",
     18,
     "Thymine-to-cytosine at m.9957 converts Leu246 at the C-terminal end of TM7 to Pro. "
     "A proline at this position introduces a helix-breaking kink at the critical MT-CO3/MT-CO1 "
     "lateral interface, destabilising the IMS-facing loop that contacts COX7A. "
     "CIV residual 10–25% with variable CI reduction (60–80%) due to secondary OXPHOS coupling "
     "impairment. Heteroplasmy 65–85% in muscle. "
     "Clinical phenotype: MELAS-like stroke-like episodes (SLEs) with headache, vomiting, "
     "cortical lesions on DWI not following vascular territories, co-occurring with "
     "CIV biochemical fingerprint. Lactate elevated 4–12 mM. "
     "DDx from MELAS due to MT-TL1 m.3243A>G: CIV reduction more prominent in MT-CO3; "
     "MT-TL1 shows multi-complex reduction (CI+CIII+CIV). "
     "L-arginine IV during acute SLE as in classic MELAS. GIR 6–8 mandatory."),
    ("Large deletion spanning m.9207–9990 (CO3 region)",
     "Partial or complete MT-CO3 deletion",
     "Large mtDNA deletion — eliminates MT-CO3 coding; combined CI+CIV biochemistry",
     "KSS / CPEO / Pearson (large deletion spanning CO3)",
     12,
     "Large mtDNA deletions of 4–8 kb that include the MT-CO3 locus (m.9207–9990) also "
     "typically encompass MT-CO1 or other mtDNA genes, giving COMBINED CI+CIV deficiency "
     "(not isolated CIV). The classic Kearns–Sayre deletion (m.8470–13446, ~5 kb) spans "
     "MT-CO3 and produces KSS: external ophthalmoplegia, ptosis, pigmentary retinopathy, "
     "plus cardiac conduction defects (heart block), cerebellar ataxia, and elevated CSF protein. "
     "Pearson marrow-pancreas syndrome (neonatal): sideroblastic anaemia + pancreatic exocrine "
     "failure due to the same deletion; survivors evolve into KSS. "
     "CPEO: isolated external ophthalmoplegia + ptosis ± myopathy (least severe). "
     "Southern blot or long-read mtDNA sequencing detects deletions missed by MitoSeq short-read. "
     "Single-deletion (sporadic) events arise de novo in oogenesis — maternal transmission rare "
     "but documented for smaller deletions. Always check cardiac Holter annually in KSS "
     "(heart block is the primary cause of sudden death — pacemaker indicated if PR >240 ms "
     "or bifascicular block present)."),
    ("m.9804G>A",
     "p.Ala184Thr (TM6-IMS loop boundary — moderate CIV reduction)",
     "TM6 C-terminus / IMS loop — Ala184Thr disturbs MT-CO3/COX7A interface; moderate CIV",
     "Encephalomyopathy / multisystem CIV",
     10,
     "Guanine-to-adenine at m.9804 converts Ala184 to Thr at the TM6/IMS loop boundary of "
     "MT-CO3 — the region that contacts the nuclear subunit COX7A on the peripheral rim. "
     "Ala184 is part of a tight hydrophobic turn; Thr introduces a hydroxyl side chain "
     "that creates a hydrogen bond network incompatible with the COX7A docking geometry. "
     "CIV residual 15–35%; CI mildly reduced (70–85%) secondary to OXPHOS coupling. "
     "Heteroplasmy 60–80% in muscle. "
     "Clinical phenotype: combined encephalopathy and myopathy — cognitive regression, "
     "myopathy with exercise intolerance, elevated CK, lactic acidosis at rest (3–8 mM). "
     "Leigh MRI in 40% of infantile/childhood cases. Seizures 38%. "
     "Multisystem involvement distinguishes this from the pure myopathy of m.9438G>A. "
     "Intermediate severity — better prognosis than frameshift Leigh, worse than adult myopathy."),
]

SEIZURE_TYPES = [
    ("Focal motor (CIV-Leigh / MELAS-CIV overlap)",   55),
    ("GTCS (generalised tonic-clonic)",                40),
    ("Myoclonic (mitochondrial myoclonus)",            28),
    ("Absence-like / NCSE",                           18),
]

TRIGGERS = [
    ("Fever / illness (metabolic decompensation)",       88),
    ("Exercise / physical exertion (myopathy, CIV)",     75),
    ("Fasting / missed meal (CIV energy crisis)",        68),
    ("Missed AED dose",                                  55),
    ("Surgical / anaesthetic stress",                    42),
    ("Sleep deprivation",                                35),
    ("Catamenial (hormonal fluctuation)",                22),
    ("High carbohydrate load (MELAS-CIV overlap)",       15),
]

TREATMENTS = [
    ("CoQ10 (Ubiquinol, 300–600 mg/day)",
     "Level C — mitochondrial cofactor support",
     "Ubiquinol (reduced CoQ10) transfers electrons between CII/CI and CIII; "
     "supplementation supports residual CIV electron pressure relief; "
     "preferred over Ubiquinone form in mitochondrial disease (better bioavailability); "
     "safe, no contraindications — use lifelong"),
    ("Ascorbate (Vitamin C, 1–3 g/day)",
     "Level C — electron donor to cytochrome c; maximises CIV substrate supply",
     "Ascorbate reduces ferricytochrome c → ferrocytochrome c in the IMS; "
     "direct substrate for MT-CO2 CuA centre (upstream of MT-CO3 scaffold); "
     "maximises electron donation to CIV despite MT-CO3 scaffold instability; "
     "particularly useful in exercise intolerance phenotype (m.9438G>A)"),
    ("Thiamine (B1, 100–300 mg/day)",
     "Level C — MANDATORY empirical supplement until SLC19A3/BTBGD excluded",
     "Biotin-thiamine-responsive basal ganglia disease (SLC19A3 / BTBGD) mimics CIV-Leigh "
     "with identical bilateral BG/brainstem MRI; thiamine IV/oral is CURATIVE in BTBGD "
     "and safe in MT-CO3 disease; always give empirically before molecular confirmation; "
     "TPP (thiamine pyrophosphate) is cofactor for pyruvate dehydrogenase — supports lactate clearance"),
    ("Biotin (10–20 mg/day)",
     "Level C — MANDATORY empirical supplement (BTD / HLCS mimics)",
     "Biotinidase deficiency (BTD) and holocarboxylase synthetase (HLCS) deficiency mimic "
     "Leigh syndrome; biotin 10–20 mg/day is curative in these treatable disorders; "
     "give empirically until molecular diagnosis confirmed; safe in MT-CO3 disease"),
    ("Riboflavin (B2, 100–200 mg/day)",
     "Level C — FAD cofactor; supports CI and CII electron flow upstream of CIV",
     "Riboflavin deficiency exacerbates OXPHOS dysfunction in mitochondrial disease; "
     "riboflavin-responsive ACAD9/ETFDH CI deficiencies are important DDx — riboflavin trial "
     "safe and excludes these; CI electron flow upstream is critical for CoQ-CIV delivery"),
    ("Levetiracetam (LEV) — AED of choice",
     "Level B — preferred AED in mitochondrial disease",
     "LEV has renal excretion with no hepatic CYP450 interactions — safe in mitochondrial disease; "
     "no POLG inhibition, no CoA sequestration, no mt-ribosome toxicity; "
     "use as first-line AED for seizures in MT-CO3 disease; "
     "avoid valproate (ABSOLUTE CI), carbamazepine/phenytoin (CYP induction reduces CoQ10)"),
    ("Glucose infusion rate 6–8 mg/kg/min (IV dextrose during NBM)",
     "Level A (emergency protocol) — MANDATORY during nil-by-mouth / febrile illness",
     "ANY fasting beyond 4 hours → β-oxidation → excess FADH2 via CII → CoQ pool reduced → "
     "electron pressure at CIV → metabolic crisis in MT-CO3 disease; "
     "GIR 6–8 mg/kg/min maintains glucose as primary substrate; "
     "never delay GIR during febrile workup in MT-CO3 patients"),
    ("L-Arginine IV (0.5 g/kg load, then 0.5 g/kg/day during acute SLE)",
     "Level B — MELAS-CIV overlap phenotype ONLY (m.9957T>C)",
     "L-Arginine is a NO precursor; NO promotes cerebrovascular vasodilation during "
     "mitochondrial stroke-like episodes (SLEs) in MELAS / MELAS-CIV overlap; "
     "use ONLY in the MELAS-CIV overlap phenotype (m.9957T>C); "
     "NOT indicated in pure Leigh or exercise myopathy phenotypes"),
]

CONTRAINDICATIONS = [
    ("Propofol",
     "ABSOLUTE CI — direct CIV heme a3-CuB inhibitor + PRIS",
     "Propofol directly inhibits the heme a3-CuB terminal O2-reduction site in MT-CO1; "
     "in MT-CO3 deficiency the CIV trimer scaffold is destabilised — "
     "propofol compound inhibition → double CIV hit; "
     "Propofol Infusion Syndrome (PRIS) causes fatal lactic acidosis; "
     "use SEVOFLURANE for anaesthesia in all CIV disorders including MT-CO3"),
    ("Metformin",
     "ABSOLUTE CI — Complex I inhibitor compounding CIV failure",
     "Metformin inhibits CI (ND1/ND2 subunit proton pump) → NADH backed up → "
     "electron pressure compounds CIV bottleneck at MT-CO3 scaffold; "
     "risk of life-threatening lactic acidosis; NEVER use"),
    ("Valproic acid (VPA)",
     "ABSOLUTE CI — CoA sequestration + POLG inhibition + CI inhibition",
     "VPA sequesters CoA → carnitine depletion → OXPHOS substrate deprivation; "
     "VPA is a POLG inhibitor → mtDNA depletion compounds MT-CO3 mRNA template loss; "
     "hepatotoxic in mitochondrial disease; use LEV instead"),
    ("Linezolid",
     "ABSOLUTE CI — blocks mt 23S rRNA → prevents MT-CO3 synthesis",
     "Linezolid inhibits the mt 70S ribosome 23S rRNA peptidyl transferase centre; "
     "blocks translation of MT-CO1, MT-CO2, AND MT-CO3 — all 3 mtDNA CIV subunits; "
     "CIV holoenzyme cannot assemble; fatal CIV deficiency exacerbation"),
    ("Chloramphenicol",
     "ABSOLUTE CI — mt 50S ribosome inhibitor",
     "Chloramphenicol blocks the mt 50S (23S-equivalent) ribosome peptidyl transferase; "
     "prevents MT-CO3 translation; same mechanism as linezolid; pancytopenia risk; "
     "NEVER use in any mtDNA disease"),
    ("Ketogenic diet (KD)",
     "CONTRAINDICATED — CIV bottleneck exacerbated by FA oxidation",
     "KD → β-oxidation → excess FADH2 via CII → CoQ pool fully reduced → "
     "electron pressure at CIV scaffold → OXPHOS crisis; "
     "KD is beneficial in CI deficiency but HARMFUL in CIV deficiency including MT-CO3"),
    ("Fasting / prolonged NBM without IV glucose",
     "ABSOLUTELY AVOID — triggers CIV metabolic crisis",
     "Fasting >4 hours → β-oxidation dominant → CIV bottleneck → crisis; "
     "GIR 6–8 mandatory during all nil-by-mouth periods; "
     "never delay GIR during febrile illness investigation in MT-CO3 patients"),
    ("Intense unmanaged exercise (without conditioning)",
     "HIGH CAUTION in exercise intolerance phenotype (m.9438G>A)",
     "Unstructured high-intensity exercise → acute exertional lactic acidosis + rhabdomyolysis risk; "
     "managed aerobic conditioning (graded 30–45 min moderate intensity) is beneficial; "
     "avoid anaerobic sprinting, heavy resistance exercise; monitor CK and lactate "
     "during exercise titration; exercise physiologist referral mandatory"),
]

MONITORING = [
    ("Plasma lactate",                      "Baseline + q3M; target <2 mM rest; >5 mM signals decompensation"),
    ("CK (creatine kinase)",               "Baseline + q3M; >2000 U/L = rhabdomyolysis risk; monitor with exercise"),
    ("Respiratory chain enzymology",        "Muscle biopsy at diagnosis; CIV <30% residual confirms pathogenicity"),
    ("Brain MRI (T2/FLAIR/DWI)",           "At diagnosis + decompensation; Leigh/SLE pattern; q12M in Leigh"),
    ("Echocardiography",                    "Baseline CONFIRM NO HCM (exclude SCO2); KSS → cardiomyopathy annual"),
    ("Cardiac Holter / ECG",               "KSS deletion phenotype: heart block risk — annual Holter; pacemaker if PR >240 ms"),
    ("Ophthalmology (fundoscopy/ERG)",     "KSS: pigmentary retinopathy + ptosis + ophthalmoplegia annual"),
    ("Liver function tests (LFT/GGT)",     "Baseline confirm no hepatopathy (exclude SCO1/POLG); q12M"),
    ("Developmental/neuropsychological",   "q6M in infantile Leigh; Bayley + VABS; school-age neuropsych annual"),
    ("mtDNA heteroplasmy — muscle",        "qPCR at diagnosis; blood underestimates muscle by 15–30%"),
]

REFERENCES = [
    "Manfredi G et al. 1995 Hum Mol Genet — m.9438G>A: first MT-CO3 adult exercise myopathy",
    "Bruno C et al. 1999 Ann Neurol — MT-CO3 mutations: encephalomyopathy and CIV-Leigh",
    "Keightley JA et al. 2000 Am J Hum Genet — MT-CO3 m.9537InsC: frameshift CIV-Leigh",
    "Tsukihara T et al. 1996 Science — CIV crystal structure (bovine): MT-CO3 7-TM scaffold",
    "Stroud DA et al. 2015 Cell Metab — CIV assembly: MT-CO3 joins late in assembly pathway",
    "Mick DU et al. 2012 Cell Metab — MITRAC: MT-CO1/MT-CO2 module; MT-CO3 late addition",
    "Gorman GS et al. 2016 Nat Rev Dis Primers — Mitochondrial disease epidemiology/management",
    "Rahman S & Thorburn D 2015 Gene Reviews — Mitochondrial complex IV deficiencies overview",
    "Sacconi S et al. 2003 Neurology — MT-CO3 MELAS-CIV overlap phenotype case series",
    "Pitceathly RD & Taanman JW 2018 JAMA Neurol — Isolated CIV deficiency: genetics + management",
]

KEY_CONCEPTS = [
    ("MT-CO3 — 7 TM Helices, NO Metal Centres — Structural Scaffold Only",
     "MT-CO3 is the LARGEST of the 3 mtDNA-encoded CIV subunits by TM helix count (7 TM). "
     "Unlike MT-CO1 (Heme a, Heme a3, CuB) and MT-CO2 (CuA binuclear copper centre), "
     "MT-CO3 contains NO metal prosthetic groups. "
     "MT-CO3 wraps the back face of MT-CO1, completing the mtDNA-encoded trimer "
     "(MT-CO1 + MT-CO2 + MT-CO3) and providing structural stability for the catalytic core. "
     "Nuclear CIV subunits (COX7A, COX7B, COX8A, COX4I1, COX5A, COX6A, COX6B1) then "
     "assemble peripherally around the MT-CO1/CO2/CO3 trimer core."),
    ("Exercise Intolerance Myopathy — Most Common Adult MT-CO3 Phenotype",
     "m.9438G>A (p.Gly78Asp, TM4) is the most common MT-CO3 pathogenic variant worldwide. "
     "It causes EXERCISE INTOLERANCE as the dominant phenotype — a distinctive feature of "
     "MT-CO3 disease relative to MT-CO1 (BSN dominant) and MT-CO2 (CIV-Leigh dominant). "
     "CIV residual 25–45% with m.9438G>A — sufficient for resting demand but insufficient "
     "for exercise-induced peak ATP: exertional lactate >6 mM, post-exercise CK elevation, "
     "myalgias. Managed aerobic conditioning is safe and beneficial. "
     "COX-negative fibres on muscle biopsy (SDH+ / COX-) confirm diagnosis."),
    ("Proton Exit Pathway — MT-CO3 TM5-TM6 K-channel/D-channel Exit",
     "CIV translocates 4 protons per O2 reduced from matrix to IMS, generating the proton "
     "motive force (ΔΨm) that drives ATP synthase. "
     "MT-CO3 TM5-TM6 region forms part of the K-channel entry from matrix and contributes to "
     "the proton exit route. Mutations in this region (e.g. m.9957T>C p.Leu246Pro) uncouple "
     "proton pumping while partially preserving electron transfer — this explains the "
     "MELAS-CIV overlap phenotype where CIV enzymatic reduction is moderate (10–25%) "
     "but lactic acidosis and SLEs are disproportionately severe."),
    ("CIV Assembly: MT-CO3 Joins LATE in the CIV Assembly Pathway",
     "CIV assembly is ordered: MT-CO1 is synthesised first at the mitoribosome (MITRAC module); "
     "COX14, COA3, PET100, SURF1 load onto nascent MT-CO1 (COX1-module / S1 intermediate). "
     "MT-CO2 joins next (COX2-module / S2 intermediate), assisted by COX20, SCO1, SCO2. "
     "MT-CO3 is added LATE (S3 intermediate), completing the mtDNA-encoded trimer core. "
     "Nuclear subunit rings are added last. MT-CO3 instability due to pathogenic variants "
     "leaves the S2 intermediate as the terminal assembly product on BN-PAGE."),
    ("DDx: NO HCM — Exclude SCO2; NO Hepatopathy — Exclude SCO1",
     "SCO2 (22q13.33) delivers copper to MT-CO2 CuA centre — SCO2 deficiency causes 100% HCM. "
     "MT-CO3 mutations cause ISOLATED CIV without HCM. "
     "Echocardiography at diagnosis is MANDATORY — HCM → suspect SCO2 (AR nuclear, not maternal). "
     "SCO1 (17p13.2) deficiency → severe neonatal hepatopathy (LFTs >10× ULN). "
     "MT-CO3 mutations cause NO hepatopathy. LFTs are the second DDx pivot."),
    ("WES Misses MT-CO3 — Dedicated mtDNA Sequencing Required",
     "WES panels typically exclude mtDNA or have poor, unreliable coverage of MT-CO3 (m.9207–9990). "
     "Heteroplasmic MT-CO3 variants (especially m.9438G>A, heteroplasmy 60–90%) may be "
     "reported as variants of uncertain significance or missed entirely. "
     "Dedicated MitoSeq (high-coverage NGS of full mitochondrial genome) is required. "
     "Blood underestimates muscle heteroplasmy by 15–30%: MUSCLE BIOPSY is mandatory for "
     "confirmation and RC enzymology. Verify H-strand coverage at m.9207–9990 in NGS QC report."),
    ("KSS Deletion: Heart Block — Annual Holter Mandatory",
     "Large mtDNA deletions spanning MT-CO3 (the classic KSS deletion m.8470–13446 includes CO3) "
     "cause KSS: ptosis, CPEO, pigmentary retinopathy, PLUS cardiac conduction defects. "
     "Heart block (PR prolongation → bifascicular → complete AV block) is the PRIMARY CAUSE "
     "of sudden death in KSS — annual Holter + ECG is mandatory. "
     "Pacemaker is indicated if PR >240 ms, bifascicular block, or AV block of any degree. "
     "Never defer cardiac screening in a deletion-phenotype MT-CO3 patient."),
    ("LRPPRC DDx — Combined CI+CIV (not isolated) if LRPPRC suspected",
     "LRPPRC (2p21) stabilises ALL 13 mt-mRNAs including MT-CO3 mRNA. "
     "LRPPRC deficiency → combined CI+CIV deficiency + Leigh syndrome French-Canadian type (LSFC). "
     "If a patient has CI+CIV reduction on RC enzymology, suspect LRPPRC (nuclear, AR) "
     "BEFORE assuming MT-CO3 point mutation (which gives ISOLATED CIV). "
     "Isolated CIV (normal CI/CII/CIII) strongly favours MT-CO3 point mutation "
     "vs LRPPRC (combined) or large deletion (combined CI+CIV)."),
]


def _make_patients():
    pats = []
    for i in range(1, N + 1):
        r = rng.random() * 100
        if   r < 35:  cls = PHENO_CLASSES[0][0]
        elif r < 60:  cls = PHENO_CLASSES[1][0]
        elif r < 78:  cls = PHENO_CLASSES[2][0]
        elif r < 90:  cls = PHENO_CLASSES[3][0]
        else:         cls = PHENO_CLASSES[4][0]

        is_myo    = "Exercise intolerance" in cls
        is_leigh  = "CIV-Leigh" in cls
        is_melas  = "MELAS" in cls
        is_kss    = "KSS" in cls
        is_enceph = "Encephalomyopathy" in cls

        if is_myo:    v = VARIANTS[0]
        elif is_leigh: v = VARIANTS[1]
        elif is_melas: v = VARIANTS[2]
        elif is_kss:   v = VARIANTS[3]
        else:          v = VARIANTS[4]

        pats.append({
            "id":                  f"P{i:02d}",
            "phenotype":           cls,
            "variant":             v[0],
            "cDNA":                v[1],
            "civ_pct":             (rng.randint(25, 45) if is_myo
                                    else rng.randint(2, 8)   if is_leigh
                                    else rng.randint(10, 25) if is_melas
                                    else rng.randint(8, 30)  if is_kss
                                    else rng.randint(15, 35)),
            "onset_mo":            (rng.randint(120, 480) if is_myo
                                    else rng.randint(1, 6)   if is_leigh
                                    else rng.randint(12, 120) if is_melas
                                    else rng.randint(24, 240) if is_kss
                                    else rng.randint(6, 72)),
            "has_seizure":         rng.random() < (0.65 if is_leigh else
                                                    0.55 if is_melas else
                                                    0.38 if is_enceph else
                                                    0.25 if is_kss else 0.08),
            "has_hypotonia":       rng.random() < (0.92 if is_leigh else
                                                    0.60 if is_enceph else
                                                    0.45 if is_melas else
                                                    0.30 if is_kss else 0.18),
            "has_lactic_acidosis": rng.random() < (0.92 if is_leigh else
                                                    0.85 if is_melas else
                                                    0.70 if is_enceph else
                                                    0.58 if is_kss else 0.40),
            "has_leigh_mri":       rng.random() < (0.90 if is_leigh else
                                                    0.40 if is_enceph else 0.10),
            "has_sle_mri":         is_melas and rng.random() < 0.82,
            "has_myopathy":        rng.random() < (1.0 if is_myo else
                                                    0.55 if is_enceph else
                                                    0.50 if is_kss else 0.20),
            "has_ophthalmo":       rng.random() < (0.92 if is_kss else 0.05),
            "has_retinopathy":     rng.random() < (0.80 if is_kss else 0.03),
            "has_heart_block":     rng.random() < (0.52 if is_kss else 0.02),
            "has_hcm":             False,   # NO HCM — KEY DDx SCO2
            "has_hepatopathy":     False,   # NO hepatopathy — KEY DDx SCO1
            "has_exercise_intol":  is_myo or (rng.random() < 0.15),
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
    n_myo  = sum(1 for p in pts if p["has_myopathy"])
    n_exer = sum(1 for p in pts if p["has_exercise_intol"])

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
            "seizures_pct":          round(n_sz   / N * 100),
            "hypotonia_pct":         round(n_hyp  / N * 100),
            "lactic_acidosis_pct":   round(n_lac  / N * 100),
            "leigh_mri_pct":         round(n_mri  / N * 100),
            "sle_mri_pct":           round(n_sle  / N * 100),
            "myopathy_pct":          round(n_myo  / N * 100),
            "exercise_intol_pct":    round(n_exer / N * 100),
            "mean_civ_pct":          round(sum(p["civ_pct"] for p in pts) / N, 1),
            "median_onset_mo":       round(sorted(p["onset_mo"] for p in pts)[N // 2], 1),
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
                "myopathy":      p["has_myopathy"],
                "ophthalmo":     p["has_ophthalmo"],
                "retinopathy":   p["has_retinopathy"],
                "heart_block":   p["has_heart_block"],
                "exercise_intol": p["has_exercise_intol"],
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
            {"term": "MT-CO3 / COX3 / COIII",
             "definition": ("261 aa / 30 kDa mtDNA H-strand encoded Complex IV subunit 3; "
                            "7 TM helices — MOST TM helices of 3 mtDNA CIV subunits; "
                            "NO metal centres; structural scaffold + proton exit pathway; "
                            "OMIM *516050; m.9207–9990 rCRS")},
            {"term": "7 TM Helices — Proton Exit Pathway",
             "definition": ("MT-CO3 wraps the back face of MT-CO1 via 7 TM helices; "
                            "TM5-TM6 contribute to K-channel/D-channel proton exit from matrix to IMS; "
                            "no metal prosthetic groups — unlike MT-CO1 (Heme a/a3/CuB) or MT-CO2 (CuA); "
                            "required for stability of the MT-CO1/MT-CO2 catalytic dimer")},
            {"term": "m.9438G>A (p.Gly78Asp)",
             "definition": ("Most common MT-CO3 pathogenic variant; TM4 hydrophobic core; "
                            "Gly78 flexibility → Asp kink disrupts MT-CO1 back-face contact; "
                            "CIV residual 25–45%; adult exercise intolerance myopathy; "
                            "BEST prognosis of all MT-CO3 phenotypes")},
            {"term": "m.9537InsC (Frameshift-Leigh)",
             "definition": ("Cytosine insertion after m.9537 → frameshift → premature stop <aa120; "
                            "TM helices 4-7 absent; CIV <5% residual; CIV-Leigh infantile; "
                            "isolated CIV fingerprint on RC enzymology (CI/CII/CIII normal)")},
            {"term": "m.9957T>C (p.Leu246Pro MELAS-CIV)",
             "definition": ("TM7-IMS junction proline; helix-breaking kink disrupts MT-CO3/COX7A interface; "
                            "CIV 10-25%; MELAS-CIV overlap with stroke-like episodes; "
                            "proton pumping uncoupled → SLEs disproportionate to CIV reduction")},
            {"term": "Isolated CIV Fingerprint",
             "definition": ("MT-CO3 point mutations → isolated CIV deficiency: CI/CII/CIII biochemically normal; "
                            "CIV <5-45% depending on variant severity; "
                            "combined CI+CIV → suggests large deletion (KSS) or LRPPRC (nuclear AR)")},
            {"term": "KSS / CPEO / Pearson (large deletion)",
             "definition": ("Large mtDNA deletions spanning MT-CO3 (m.9207-9990) → combined CI+CIV; "
                            "KSS: ophthalmoplegia+retinopathy+heart block; cardiac Holter annual mandatory; "
                            "pacemaker if PR >240ms or bifascicular block; "
                            "Pearson marrow-pancreas: sideroblastic anaemia+pancreatic failure in neonates")},
            {"term": "SCO2 DDx — 100% HCM; MT-CO3 NO HCM",
             "definition": ("SCO2 (22q13.33) is the copper chaperone for MT-CO2 CuA; SCO2 deficiency → 100% HCM; "
                            "MT-CO3 mutations NEVER cause HCM; echocardiography mandatory to exclude SCO2; "
                            "HCM → AR nuclear SCO2, not maternal MT-CO3")},
            {"term": "LRPPRC DDx — Combined CI+CIV (not isolated)",
             "definition": ("LRPPRC (2p21) stabilises ALL 13 mt-mRNAs including MT-CO3 mRNA; "
                            "LRPPRC deficiency → combined CI+CIV (LSFC, Leigh French-Canadian); "
                            "isolated CIV on enzymology strongly favours MT-CO3 point mutation; "
                            "combined CI+CIV → consider LRPPRC, POLG, large deletion first")},
            {"term": "GIR 6–8 — NEVER FAST",
             "definition": ("Glucose infusion rate 6-8 mg/kg/min IV dextrose during any nil-by-mouth; "
                            "MANDATORY: fasting → β-oxidation → FADH2 excess → CIV bottleneck → crisis; "
                            "first-line management during febrile illness + acute decompensation")},
            {"term": "Propofol / PRIS — ABSOLUTE CI in all CIV",
             "definition": ("Propofol directly inhibits heme a3-CuB in MT-CO1 + causes PRIS; "
                            "in MT-CO3 deficiency CIV trimer scaffold destabilised; "
                            "propofol double CIV hit → fatal lactic acidosis; "
                            "use SEVOFLURANE for anaesthesia in all CIV disorders")},
            {"term": "H-strand (MT-CO3)",
             "definition": ("MT-CO3 encoded on mtDNA Heavy strand (m.9207-9990 rCRS); "
                            "all 3 mtDNA CIV subunits (CO1/CO2/CO3) are H-strand encoded; "
                            "H-strand NGS coverage verification required in MitoSeq QC; "
                            "unlike MT-ND6 which is L-strand")},
        ],
        "references": REFERENCES,
    }


if __name__ == "__main__":
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first 500 chars) ===")
    print(_json.dumps(get_breakdown(), indent=2)[:500])
