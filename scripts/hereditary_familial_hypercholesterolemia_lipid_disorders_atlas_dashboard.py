#!/usr/bin/env python3
"""Hereditary-Familial-Hypercholesterolemia-Lipid-Disorders-Atlas -- Complete 8-Gene Reference
Hereditary lipid disorders: familial hypercholesterolaemia (FH), autosomal recessive
hypercholesterolaemia (ARH), familial dysbetalipoproteinaemia, familial chylomicronaemia,
hypertriglyceridaemia type V, and sitosterolaemia:

LDLR   (LDL Receptor; FH type 1; most common 60-80%; corneal arcus <45yr PATHOGNOMONIC),
APOB   (Apolipoprotein B-100; FH type 2; R3527Q/W; milder than LDLR; tendon xanthomas),
PCSK9  (Proprotein Convertase; GOF = FH type 3; D374Y PATHOGNOMONIC; LOF = natural protection),
LDLRAP1(LDL Receptor Adaptor; ARH; AR; splenomegaly PATHOGNOMONIC; biallelic required),
APOE   (Apolipoprotein E; ε2/ε2 FH type 3; palmar xanthomas PATHOGNOMONIC-unique; fibrates),
LPL    (Lipoprotein Lipase; FCS; AR; pancreatitis PATHOGNOMONIC; volanesorsen FDA-approved),
APOA5  (Apolipoprotein A-V; hypertriglyceridaemia type V; pancreatitis; fibrates + omega-3),
ABCG5  (Sterolin-1; sitosterolaemia; AR; childhood xanthomas PATHOGNOMONIC; ezetimibe first-line).

LDLR   (Low-density lipoprotein receptor; 1324aa; 19p13.2; AD LOF;
         Familial hypercholesterolaemia type 1 (FH1); most common 60-80% of FH cases;
         corneal arcus <45 yr PATHOGNOMONIC; tendon xanthomas; premature ASCVD;
         statin + ezetimibe + PCSK9i first-line; apheresis if refractory;
         seed SEED_BASE+0).
APOB   (Apolipoprotein B-100; 4563aa; 2p24.1; AD LOF;
         Familial defective ApoB (FDB) = FH type 2; R3527Q or R3527W most common;
         clinically indistinguishable from LDLR FH1 but usually milder;
         PCSK9i less effective (ApoB not degraded via PCSK9 pathway);
         seed SEED_BASE+1).
PCSK9  (Proprotein convertase subtilisin/kexin type 9; 692aa; 1p32.3; AD GOF or LOF;
         GOF mutations (D374Y PATHOGNOMONIC) = FH type 3; severe hypercholesterolaemia;
         LOF mutations (R46L, Y142X) = natural protection from ASCVD -- plasma LDL-C lower;
         PCSK9 inhibitors (evolocumab, alirocumab) reduce LDL-C 50-60% on top of statin;
         seed SEED_BASE+2).
LDLRAP1(LDL receptor adaptor protein 1; 308aa; 1p36.11; AR LOF;
         Autosomal recessive hypercholesterolaemia (ARH); biallelic required;
         splenomegaly PATHOGNOMONIC (not seen in AD FH); hepatosplenomegaly;
         responds partially to statins + PCSK9i; LDL apheresis often needed;
         seed SEED_BASE+3).
APOE   (Apolipoprotein E; 299aa; 19q13.32; codominant / AR ε2/ε2;
         Familial dysbetalipoproteinaemia (FH type 3 = Fredrickson Type III);
         ε2/ε2 homozygote required (1 in 60 people are ε2/ε2 but need secondary trigger);
         xanthoma striata palmaris (palmar crease xanthomas) PATHOGNOMONIC-UNIQUE;
         fibrates first-line (not statins alone); ApoE genotyping mandatory;
         seed SEED_BASE+4).
LPL    (Lipoprotein lipase; 475aa; 8p21.3; AR LOF;
         Familial chylomicronaemia syndrome (FCS); biallelic LPL LOF;
         recurrent acute pancreatitis PATHOGNOMONIC; eruptive xanthomas;
         fat restriction <20g/day first-line; volanesorsen (siRNA) FDA-approved 2019 (Canada);
         omega-3 and fibrates ineffective in pure LPL deficiency;
         seed SEED_BASE+5).
APOA5  (Apolipoprotein A-V; 366aa; 11q23.3; AD LOF;
         Hypertriglyceridaemia type V; severe mixed hyperlipidaemia; pancreatitis risk;
         eruptive xanthomas; GWAS-confirmed cardiovascular risk modifier;
         fibrates + omega-3 + low-fat diet; alcohol avoidance critical;
         seed SEED_BASE+6).
ABCG5  (ATP-binding cassette subfamily G member 5 / Sterolin-1; 651aa; 2p21; AR LOF;
         Sitosterolaemia (phytosterolaemia); plant sterol accumulation;
         tendon and tuberous xanthomas in childhood PATHOGNOMONIC;
         premature ASCVD; haemolytic anaemia PATHOGNOMONIC-unique;
         ezetimibe FIRST-LINE SPECIFIC (blocks intestinal sterol absorption);
         low plant-sterol diet; stanol/sterol-enriched foods paradoxically worsen;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3766-3773)
"""
import random

SEED_BASE = 3766

ATLAS_GENES = [
    {
        "gene": "LDLR",
        "protein": (
            "LDLR -- 19p13.2 AD LOF -- 1324aa -- "
            "LDL-Receptor-Familial-Hypercholesterolaemia-FH1-"
            "Most-Common-60-80pct-Corneal-Arcus-lt45yr-PATHOGNOMONIC-"
            "Statin-Ezetimibe-PCSK9i-Apheresis-"
            "LDLR-OMIM-606945"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "1324 aa / ~164 kDa / 19p13.2 LDLR LDL receptor molecular context: "
            "STRUCTURE: "
            "  1324 aa / ~164 kDa transmembrane glycoprotein; "
            "  Ligand-binding domain: 7 cysteine-rich repeats binding ApoB-100 and ApoE; "
            "  EGF-like domain: mediates pH-dependent ligand release in endosome; "
            "  Classified by functional defect: Class I (no protein), II (transport), "
            "    III (binding), IV (internalisation), V (recycling); "
            "  >2,000 LDLR pathogenic variants catalogued; "
            "FAMILIAL HYPERCHOLESTEROLAEMIA FH1: "
            "  AD LOF (haploinsufficiency); most common FH cause 60-80%; "
            "  Prevalence: 1 in 250 heterozygous (FH); 1 in 300,000 homozygous (HoFH); "
            "  Simon Broome criteria: TC >7.5 mmol/L adult + tendon xanthomas or FH in 1st-degree; "
            "  Dutch Lipid Clinic Network Score: genetic probability stratification; "
            "CLINICAL FEATURES: "
            "  Tendon xanthomas (Achilles, extensor): PATHOGNOMONIC for FH, seen in ~50%; "
            "  Corneal arcus (arcus corneae) <45 yr: PATHOGNOMONIC - seen before 45 yr is FH marker; "
            "  Xanthelasma: periorbital, less specific but supportive; "
            "  Premature ASCVD: MI in males <55 yr, females <60 yr; "
            "  LDL-C: typically 5-15 mmol/L heterozygous; >13 mmol/L homozygous; "
            "TREATMENT: "
            "  Goal: LDL-C <1.8 mmol/L (very high risk) or <2.6 mmol/L (high risk); "
            "  Statin (high-intensity): atorvastatin 40-80 mg or rosuvastatin 20-40 mg; "
            "  Ezetimibe 10 mg: add-on; reduces LDL-C by additional 15-20%; "
            "  PCSK9 inhibitors: evolocumab 140 mg Q2W or alirocumab 75-150 mg Q2W; "
            "    Reduce LDL-C 50-60% on top of statin+ezetimibe; "
            "  Inclisiran (siRNA): 300 mg Q6months -- twice-yearly injection; "
            "  LDL apheresis: Homozygous FH (HoFH) or statin-intolerant with very high LDL; "
            "  Evinacumab (anti-ANGPTL3): HoFH - null LDLR variants respond poorly to PCSK9i; "
            "  Lomitapide: HoFH; hepatotoxicity monitoring mandatory; "
            "HOMOZYGOUS FH (HoFH): "
            "  Both LDLR alleles null or severe; "
            "  LDL-C often >13 mmol/L; "
            "  Aortic stenosis in childhood; xanthomas in childhood; MI <20 yr; "
            "  Statins alone insufficient -- must add PCSK9i + apheresis + evinacumab; "
            "CASCADE: "
            "  AD 50% first-degree risk; "
            "  Cascade testing: cholesterol + genetic testing all 1st-degree relatives from childhood; "
            "  Universal paediatric lipid screening 9-11 yr recommended"
        ),
        "syndrome": "Familial-Hypercholesterolaemia-FH1 / AD-LOF-LDL-receptor / most-common-FH-60-80pct",
        "inheritance": "AD LOF (haploinsufficiency)",
        "pathognomonic": "corneal-arcus <45yr = FH PATHOGNOMONIC; tendon-xanthomas-Achilles PATHOGNOMONIC; LDL-C >4.9mmol/L adult; premature-ASCVD males <55yr; evinacumab-HoFH-LDLR-null",
        "key_drugs_ci": ["stanol-enriched-foods-worsen-coexisting-sitosterolaemia", "statins-alone-insufficient-HoFH-add-PCSK9i"],
        "surveillance": ["annual-LDL-C-lipid-panel", "echo-aortic-stenosis-HoFH", "cascade-1st-degree", "paediatric-lipid-screening-9-11yr"],
    },
    {
        "gene": "APOB",
        "protein": (
            "APOB -- 2p24.1 AD LOF -- 4563aa -- "
            "Apolipoprotein-B-100-Familial-Defective-ApoB-FDB-FH-Type-2-"
            "R3527Q-R3527W-Most-Common-Milder-LDLR-"
            "PCSK9i-Less-Effective-"
            "APOB-OMIM-107730"
        ),
        "locus": "2p24.1",
        "protein_size": (
            "4563 aa / ~550 kDa / 2p24.1 APOB Apolipoprotein B-100 molecular context: "
            "STRUCTURE: "
            "  4563 aa / ~550 kDa; one of the largest human proteins; "
            "  ApoB-100: full-length form in liver, the only form in LDL; "
            "  ApoB-48: truncated intestinal form via RNA editing; "
            "  LDL-receptor binding region: Arg3527 (p.Arg3527Gln/Trp = FDB); "
            "  LOF at Arg3527: ApoB-100 fails to bind LDLR -> LDL-C accumulation; "
            "FAMILIAL DEFECTIVE ApoB (FDB): "
            "  AD LOF (dominant-negative at receptor-binding site); "
            "  p.Arg3527Gln (c.10580G>A): most common FDB variant; "
            "  p.Arg3527Trp (c.10579C>T): second most common; "
            "  Prevalence: ~1 in 500-1000 (less common than LDLR FH); "
            "CLINICAL FEATURES: "
            "  Clinically indistinguishable from LDLR FH1 on presentation; "
            "  LDL-C elevation typically MILDER than LDLR heterozygous FH; "
            "  Tendon xanthomas: less frequent than LDLR FH (because LDL-C lower); "
            "  Corneal arcus: may be present; "
            "  Premature ASCVD: risk elevated but lower than LDLR FH; "
            "TREATMENT: "
            "  PCSK9 INHIBITORS LESS EFFECTIVE in APOB FDB: "
            "    Mechanism: PCSK9i upregulates LDLR, but LDL cannot bind LDLR efficiently in APOB FDB; "
            "    PCSK9i still reduces LDL-C but by smaller magnitude than in LDLR FH; "
            "  Statins + ezetimibe: similar response to LDLR FH; "
            "  Inclisiran: effective (reduces ApoB synthesis, not LDLR-dependent); "
            "  Lomitapide: effective (reduces ApoB-100 synthesis in liver); "
            "CASCADE: "
            "  AD 50% 1st-degree risk; genetic test distinguishes from LDLR FH clinically; "
            "  Genetic testing modifies PCSK9i dosing expectation"
        ),
        "syndrome": "Familial-Defective-ApoB-FDB-FH2 / AD-LOF-ApoB-100-Arg3527 / milder-than-LDLR",
        "inheritance": "AD LOF (dominant-negative binding site)",
        "pathognomonic": "R3527Q/W-ApoB-PATHOGNOMONIC-variant-FDB; PCSK9i-less-effective-APOB-specific; inclisiran+lomitapide-effective; LDL-C-milder-than-LDLR",
        "key_drugs_ci": ["PCSK9i-reduced-efficacy-ApoB-FDB-inform-patient", "statins-alone-may-be-sufficient-mild-cases"],
        "surveillance": ["annual-LDL-C", "ASCVD-risk-calculator-annually", "cascade-1st-degree", "genetic-test-to-guide-PCSK9i-choice"],
    },
    {
        "gene": "PCSK9",
        "protein": (
            "PCSK9 -- 1p32.3 AD GOF/LOF -- 692aa -- "
            "Proprotein-Convertase-Subtilisin-Kexin-9-"
            "GOF-D374Y-FH-Type-3-PATHOGNOMONIC-Severe-"
            "LOF-R46L-Natural-Protection-ASCVD-"
            "Evolocumab-Alirocumab-FDA2015-"
            "PCSK9-OMIM-607786"
        ),
        "locus": "1p32.3",
        "protein_size": (
            "692 aa / ~75 kDa / 1p32.3 PCSK9 proprotein convertase subtilisin/kexin type 9: "
            "STRUCTURE: "
            "  692 aa / ~75 kDa; serine protease prodomain + catalytic domain + C-terminal domain; "
            "  Prodomain undergoes autocatalytic cleavage; "
            "  PCSK9 binds extracellular domain of LDLR and traffics it to lysosome for degradation; "
            "  GOF PCSK9: LDLR degraded faster -> LDL-C elevated; "
            "  LOF PCSK9: LDLR recycled more -> LDL-C lower; "
            "GAIN-OF-FUNCTION (GOF) = FH TYPE 3: "
            "  D374Y (c.1120G>T): MOST SEVERE GOF variant; PATHOGNOMONIC for FH type 3; "
            "    D374Y binds LDLR ~25x more tightly -> massive LDLR degradation; "
            "    LDL-C levels extreme; premature ASCVD in 3rd-4th decade; "
            "  Other GOF: S127R, F216L, D129G; "
            "  Prevalence GOF: rare (<1 in 2,500 FH); "
            "LOSS-OF-FUNCTION (LOF) = NATURAL CARDIOVASCULAR PROTECTION: "
            "  R46L (c.137G>T): most common LOF in Europeans; ~2-3% allele frequency; "
            "  Y142X, C679X: common LOF in African-Americans; "
            "  LOF heterozygotes: LDL-C 15-28% lower, ASCVD risk ~47% lower (lifetime); "
            "  LOF homozygotes: LDL-C very low but no adverse effects; "
            "PCSK9 INHIBITORS (monoclonal antibodies): "
            "  Evolocumab (Repatha) 140mg Q2W or 420mg Q4W: FDA-approved 2015; "
            "  Alirocumab (Praluent) 75-150mg Q2W: FDA-approved 2015; "
            "  Mechanism: block PCSK9-LDLR interaction -> LDLR recycled; "
            "  LDL-C reduction: 50-60% on top of maximum statin+ezetimibe; "
            "  FOURIER trial (evolocumab) and ODYSSEY OUTCOMES (alirocumab): CV benefit proven; "
            "  Note: GOF PCSK9 variants respond WELL to PCSK9 inhibitors (directly target GOF protein); "
            "  Inclisiran (siRNA): 300mg Q6months; reduces PCSK9 mRNA; "
            "CASCADE: "
            "  AD 50% 1st-degree risk for GOF; "
            "  LOF: cascade testing reveals naturally protected family members -- can inform lower-intensity statin"
        ),
        "syndrome": "FH-Type-3-PCSK9-GOF-D374Y-PATHOGNOMONIC / natural-LOF-R46L-protective / PCSK9-inhibitors-FDA2015",
        "inheritance": "AD GOF (FH) or AD LOF (protective)",
        "pathognomonic": "D374Y-PCSK9-GOF-PATHOGNOMONIC-most-severe-FH; PCSK9i-directly-targets-GOF-most-responsive; LOF-R46L-natural-ASCVD-protection-47pct-reduction; inclisiran-siRNA-twice-yearly",
        "key_drugs_ci": [],
        "surveillance": ["annual-LDL-C", "PCSK9-inhibitor-dose-monitoring", "cascade-1st-degree", "ASCVD-risk-annually"],
    },
    {
        "gene": "LDLRAP1",
        "protein": (
            "LDLRAP1 -- 1p36.11 AR LOF -- 308aa -- "
            "LDL-Receptor-Adaptor-Protein-1-"
            "Autosomal-Recessive-Hypercholesterolaemia-ARH-"
            "Splenomegaly-PATHOGNOMONIC-Biallelic-Required-"
            "LDLRAP1-OMIM-605747"
        ),
        "locus": "1p36.11",
        "protein_size": (
            "308 aa / ~35 kDa / 1p36.11 LDLRAP1 LDL receptor adaptor protein 1: "
            "STRUCTURE: "
            "  308 aa / ~35 kDa; contains PTB (phosphotyrosine-binding) domain; "
            "  Adaptor protein: bridges LDLR cytoplasmic tail to clathrin-coated pits; "
            "  Liver-specific expression: explains hepatic LDLR dysfunction despite normal LDLR; "
            "  LOF LDLRAP1: LDLR present on hepatocyte surface but CANNOT be internalised; "
            "  Biochemistry: very high LDL-C despite normal LDLR function in peripheral cells; "
            "AUTOSOMAL RECESSIVE HYPERCHOLESTEROLAEMIA (ARH): "
            "  AR LOF: BOTH alleles must be mutant; "
            "  More severe than heterozygous LDLR FH but milder than HoFH; "
            "  LDL-C: typically 10-20 mmol/L; "
            "  Splenomegaly: PATHOGNOMONIC for ARH (not seen in AD FH); "
            "  Hepatomegaly: common; "
            "  Xanthomas: tendon xanthomas, tuberous xanthomas, xanthelasma; "
            "  Premature ASCVD: childhood/teen onset; MI in 2nd-3rd decade untreated; "
            "TREATMENT: "
            "  Statins: partial response (still some LDLR activity in peripheral cells); "
            "  PCSK9 inhibitors: PARTIAL response (LDLR on hepatocytes not internalised efficiently); "
            "  LDL apheresis: often required every 1-2 weeks; "
            "  Evinacumab (anti-ANGPTL3): effective regardless of LDLR status; "
            "CASCADE: "
            "  AR 25% risk both alleles affected in sibling; "
            "  Parents are obligate heterozygous carriers -- mild LDL elevation; "
            "  Test all siblings"
        ),
        "syndrome": "Autosomal-Recessive-Hypercholesterolaemia-ARH / AR-LOF-biallelic / splenomegaly-PATHOGNOMONIC",
        "inheritance": "AR LOF (biallelic required)",
        "pathognomonic": "splenomegaly-ARH-PATHOGNOMONIC (absent in AD FH); hepatomegaly; LDLR-present-but-non-functional-hepatocyte; PCSK9i-partial-response; evinacumab-ANGPTL3i-effective",
        "key_drugs_ci": ["PCSK9i-alone-insufficient-ARH-add-apheresis-evinacumab"],
        "surveillance": ["LDL-C-q3months", "spleen-size-ultrasound-annually", "LDL-apheresis-schedule", "cascade-siblings-mandatory", "evinacumab-ANGPTL3i-annual-review"],
    },
    {
        "gene": "APOE",
        "protein": (
            "APOE -- 19q13.32 codominant/AR-e2e2 -- 299aa -- "
            "Apolipoprotein-E-Familial-Dysbetalipoproteinaemia-"
            "FH-Type-3-Fredrickson-Type-III-"
            "e2e2-Genotype-Palmar-Xanthoma-PATHOGNOMONIC-"
            "Fibrates-First-Line-"
            "APOE-OMIM-107741"
        ),
        "locus": "19q13.32",
        "protein_size": (
            "299 aa / ~34 kDa / 19q13.32 APOE Apolipoprotein E molecular context: "
            "STRUCTURE: "
            "  299 aa / ~34 kDa; 4 isoforms: ε2 (Arg158Cys), ε3 (reference), ε4 (Cys112Arg); "
            "  ApoE mediates clearance of chylomicron remnants and VLDL via LDLR and LRP1; "
            "  ApoE ε2 binds LDLR ~100-fold less tightly than ε3 -> impaired remnant clearance; "
            "  ApoE ε4: risk factor for Alzheimer disease (separate phenotype); "
            "  Genotype frequencies: ε3/ε3 (60-70%), ε3/ε4 (20-25%), ε3/ε2 (10-15%), ε2/ε2 (~1%); "
            "FAMILIAL DYSBETALIPOPROTEINAEMIA (FH TYPE 3 = FREDRICKSON TYPE III): "
            "  ε2/ε2 homozygosity required (necessary but not sufficient); "
            "  Only 1-5% of ε2/ε2 individuals develop overt disease; "
            "  Secondary trigger required: obesity, DM2, hypothyroidism, alcohol, oestrogen deficiency; "
            "  Lipids: BOTH cholesterol AND triglycerides elevated (mixed hyperlipidaemia); "
            "  TC: 7-12 mmol/L; TG: 4-10 mmol/L (ratio TC:TG ~1:1); "
            "  IDL/remnant lipoprotein accumulation (broad-beta band on electrophoresis); "
            "CLINICAL FEATURES -- UNIQUE XANTHOMA PATTERNS: "
            "  Xanthoma striata palmaris (palmar crease xanthomas): PATHOGNOMONIC UNIQUE for FH type 3; "
            "    Yellow-orange deposits in palmar and digital creases; seen only in Type III; "
            "  Tuberous xanthomas: over elbows and knees; "
            "  Xanthelasma; "
            "  Premature coronary AND peripheral vascular disease; "
            "TREATMENT: "
            "  First-line: FIBRATES (fenofibrate, bezafibrate) -- superior to statins for clearing IDL; "
            "    Mechanism: upregulates LPL + ApoC-III suppression -> remnant clearance; "
            "  Diet: low-fat, low-refined-carbohydrate; treat obesity; manage DM2; "
            "  Statins: add after fibrate if LDL component remains; "
            "  Omega-3 fatty acids: adjunct for TG reduction; "
            "  PCSK9i: less studied in FH type 3 (ApoE ε2/ε2 remnant); "
            "  ApoE GENOTYPING: required for diagnosis; "
            "CASCADE: "
            "  Siblings: test ApoE genotype; "
            "  Treat underlying trigger in ε2/ε2 to prevent disease expression"
        ),
        "syndrome": "Familial-Dysbetalipoproteinaemia-FH3-Fredrickson-III / ApoE-e2e2-required / palmar-xanthoma-PATHOGNOMONIC",
        "inheritance": "Codominant / AR phenotypic expression (ε2/ε2 homozygote + trigger)",
        "pathognomonic": "palmar-crease-xanthomas-xanthoma-striata-palmaris-PATHOGNOMONIC-UNIQUE-FH-type-3; ε2/ε2-genotype-required; fibrates-first-line-not-statins; mixed-TC+TG-ratio-1:1",
        "key_drugs_ci": ["statins-alone-insufficient-FH-type-3-fibrates-first", "treat-secondary-trigger-DM2-hypothyroidism-obesity"],
        "surveillance": ["ApoE-genotyping-diagnostic", "TC-TG-ratio-monitoring", "fibrate-dose-review-q6months", "treat-triggers-DM2-TSH-weight", "cascade-siblings-ApoE"],
    },
    {
        "gene": "LPL",
        "protein": (
            "LPL -- 8p21.3 AR LOF -- 475aa -- "
            "Lipoprotein-Lipase-Familial-Chylomicronaemia-FCS-"
            "Pancreatitis-Recurrent-PATHOGNOMONIC-"
            "Eruptive-Xanthomas-"
            "Volanesorsen-FDA-Approved-2019-"
            "LPL-OMIM-238600"
        ),
        "locus": "8p21.3",
        "protein_size": (
            "475 aa / ~55 kDa (dimer) / 8p21.3 LPL Lipoprotein lipase molecular context: "
            "STRUCTURE: "
            "  475 aa monomer / active as non-covalent homodimer; "
            "  Serine lipase catalytic triad; GPI-anchored on endothelial surface via GPIHBP1; "
            "  Hydrolyses triglycerides in chylomicrons and VLDL -> fatty acids for tissues; "
            "  Activated by ApoC-II on chylomicron surface; inhibited by ApoC-III; "
            "  LOF LPL: triglycerides accumulate -> massive hyperchylomicronaemia; "
            "FAMILIAL CHYLOMICRONAEMIA SYNDROME (FCS): "
            "  AR LOF: biallelic pathogenic LPL variants; "
            "  TG typically >10 mmol/L (often >50-100 mmol/L in acute episodes); "
            "  Chylomicrons present in plasma at room temp (cream layer on top); "
            "  Serum appears lactescent (milky); "
            "CLINICAL FEATURES: "
            "  Recurrent acute pancreatitis PATHOGNOMONIC: life-threatening; TG >10 mmol/L trigger; "
            "  Eruptive xanthomas: crops of small yellow-white papules on buttocks, arms, trunk; "
            "  Lipaemia retinalis: retinal arteries appear white/cream (fundoscopy); "
            "  Hepatosplenomegaly: fat-laden macrophages; "
            "  Abdominal pain, nausea; memory issues from hyperchylomicronaemia; "
            "TREATMENT: "
            "  Fat restriction <20 g/day: cornerstone treatment; medium-chain TG allowed; "
            "  Volanesorsen (siRNA targeting ApoC-III): FDA/Health Canada approved 2019; "
            "    Reduces ApoC-III -> restores LPL-independent TG clearance pathway; "
            "    ~70% TG reduction; Q2W subcutaneous injection; monitor platelets (thrombocytopenia); "
            "  Fibrates: INEFFECTIVE in pure biallelic LPL deficiency; "
            "  Omega-3: INEFFECTIVE in pure LPL deficiency; "
            "  Glofitamab/olezarsen (next-gen ApoC-III siRNA): investigational; "
            "  Alipogene tiparvovec (AAV-LPL gene therapy): approved in EU 2012, withdrawn commercially; "
            "  Alcohol avoidance MANDATORY: markedly worsens hypertriglyceridaemia; "
            "  Oestrogen AVOID: worsens TG; "
            "CASCADE: "
            "  AR; parents obligate carriers; siblings 25% risk; "
            "  Test all siblings with TG measurement + LPL sequencing"
        ),
        "syndrome": "Familial-Chylomicronaemia-FCS / AR-LOF-biallelic-LPL / pancreatitis-PATHOGNOMONIC / volanesorsen-FDA2019",
        "inheritance": "AR LOF (biallelic required)",
        "pathognomonic": "recurrent-pancreatitis-PATHOGNOMONIC; eruptive-xanthomas-PATHOGNOMONIC; lipaemia-retinalis; chylomicron-cream-layer; fibrates+omega3-INEFFECTIVE-pure-LPL-def; volanesorsen-FDA-siRNA",
        "key_drugs_ci": ["fibrates-INEFFECTIVE-pure-LPL-deficiency", "omega3-INEFFECTIVE-pure-LPL-deficiency", "oestrogen-AVOID-worsens-TG", "alcohol-ABSOLUTELY-CI"],
        "surveillance": ["TG-q3months", "pancreatitis-risk-scoring", "volanesorsen-platelet-monitoring-monthly", "fat-diary-dietitian", "cascade-siblings-TG+LPL"],
    },
    {
        "gene": "APOA5",
        "protein": (
            "APOA5 -- 11q23.3 AD LOF -- 366aa -- "
            "Apolipoprotein-A-V-Hypertriglyceridaemia-Type-V-"
            "Pancreatitis-Risk-"
            "Eruptive-Xanthomas-"
            "Fibrates-Omega3-First-Line-"
            "APOA5-OMIM-606368"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "366 aa / ~40 kDa / 11q23.3 APOA5 Apolipoprotein A-V molecular context: "
            "STRUCTURE: "
            "  366 aa / ~40 kDa; member of the apolipoprotein A-I/C-III/A-IV/A-V gene cluster; "
            "  APOA5 activates LPL and reduces ApoC-III concentration; "
            "  Also promotes hepatic VLDL-TG uptake via proteoglycan pathway; "
            "  LOF APOA5: impaired LPL activation + TG accumulation; "
            "  Strong GWAS signal: APOA5 variants explain ~5% of TG variance in population; "
            "HYPERTRIGLYCERIDAEMIA TYPE V: "
            "  AD LOF heterozygous: moderately severe HTG (TG 5-15 mmol/L); "
            "  Compound heterozygous: more severe (TG >15 mmol/L); "
            "  Elevated VLDL + chylomicrons (Type V phenotype); "
            "  Not as severe as biallelic LPL deficiency; "
            "CLINICAL FEATURES: "
            "  Eruptive xanthomas: as TG >10 mmol/L; "
            "  Pancreatitis risk: elevated when TG >10 mmol/L (lower threshold than LPL deficiency); "
            "  Cardiovascular risk: moderately increased via remnant-rich particle atherogenicity; "
            "  Secondary triggers worsen: DM2, alcohol, hypothyroidism, oestrogen; "
            "TREATMENT: "
            "  Fibrates (fenofibrate): first-line; "
            "  Omega-3 fatty acids (icosapentaenoic acid EPA -- Vascepa/REDUCE-IT): "
            "    REDUCE-IT: icosapentaenoic acid 4g/day reduced MACE 25% on statin in HTG patients; "
            "  Low-refined-carbohydrate diet; alcohol avoidance; treat obesity; "
            "  Statins: modest TG reduction; needed for ASCVD risk in mixed dyslipidaemia; "
            "  Volanesorsen: if TG >10 mmol/L and pancreatitis risk despite fibrate+omega3; "
            "CASCADE: "
            "  AD 50% 1st-degree risk; "
            "  Screen first-degree relatives TG + APOA5 sequencing if severe HTG in family"
        ),
        "syndrome": "Hypertriglyceridaemia-Type-V-APOA5 / AD-LOF / pancreatitis-risk / REDUCE-IT-omega3",
        "inheritance": "AD LOF (variable penetrance)",
        "pathognomonic": "eruptive-xanthomas-PATHOGNOMONIC->TG>10mmol; pancreatitis-risk-TG>10mmol; REDUCE-IT-omega3-25pct-MACE-reduction; fibrates+EPA-first-line; alcohol-DM2-triggers",
        "key_drugs_ci": ["oestrogen-worsens-HTG-APOA5", "alcohol-AVOID-primary-trigger"],
        "surveillance": ["TG-q3months", "pancreatitis-risk-counselling", "REDUCE-IT-eligibility-review", "DM2-thyroid-screen-annually", "cascade-1st-degree"],
    },
    {
        "gene": "ABCG5",
        "protein": (
            "ABCG5 -- 2p21 AR LOF -- 651aa -- "
            "Sterolin-1-ABC-Transporter-Sitosterolaemia-"
            "Childhood-Xanthomas-PATHOGNOMONIC-"
            "Haemolytic-Anaemia-PATHOGNOMONIC-"
            "Ezetimibe-First-Line-Specific-"
            "ABCG5-OMIM-210250"
        ),
        "locus": "2p21",
        "protein_size": (
            "651 aa / ~75 kDa / 2p21 ABCG5 ATP-binding cassette subfamily G member 5 (Sterolin-1): "
            "STRUCTURE: "
            "  651 aa / ~75 kDa; half-transporter; functions as heterodimer with ABCG8 (Sterolin-2); "
            "  ABCG5/ABCG8 heterodimer: 1302 aa functional unit; "
            "  Expressed in intestine and liver; "
            "  Mechanism: secretes plant sterols (sitosterol, campesterol, stigmasterol) and "
            "    cholesterol back into intestinal lumen and into bile; "
            "  LOF ABCG5 (or ABCG8): plant sterols accumulate in plasma and tissues; "
            "  Normal plasma plant sterol <0.5 mmol/L; sitosterolaemia >1-2 mmol/L; "
            "SITOSTEROLAEMIA (PHYTOSTEROLAEMIA): "
            "  AR LOF: biallelic ABCG5 (or ABCG8) variants; "
            "  Plant sterols incorporated into cell membranes, arteries, tendons; "
            "CLINICAL FEATURES: "
            "  Tendon and tuberous xanthomas IN CHILDHOOD: PATHOGNOMONIC; "
            "    Develop by 3-10 yr of age -- earlier than in LDLR FH; "
            "    Key distinguishing feature from other FH; "
            "  Haemolytic anaemia: PATHOGNOMONIC; plant sterols alter red cell membrane deformability; "
            "    Stomatocytes on blood film; episodic haemolysis; "
            "  Premature ASCVD: aortic stenosis possible even in childhood; "
            "  TC elevated (both cholesterol AND plant sterols measured together); "
            "DIAGNOSIS: "
            "  Plasma phytosterol assay (sitosterol, campesterol): required; "
            "  NOT routinely measured in standard cholesterol panel; "
            "  Must specifically request PLANT STEROL measurement; "
            "  ABCG5 + ABCG8 sequencing; "
            "TREATMENT: "
            "  Ezetimibe FIRST-LINE SPECIFIC: blocks NPC1L1 -> reduces intestinal plant sterol absorption; "
            "    Best drug for sitosterolaemia (mechanism-matched); "
            "    Reduces plant sterols 20-40%; "
            "  Low plant-sterol diet: avoid nuts, seeds, vegetables oils, margarines; "
            "  STANOL/STEROL-ENRICHED FOODS: ABSOLUTELY CONTRAINDICATED -- "
            "    Recommended to reduce cholesterol in general population; "
            "    In sitosterolaemia: PARADOXICALLY INCREASE plant sterols -- CATASTROPHIC; "
            "  Statins: less effective (plant sterols not LDL-pathway dependent); "
            "  Bile acid sequestrants: adjunct; "
            "CASCADE: "
            "  AR; siblings 25% risk; parents obligate carriers; "
            "  Screen siblings with plant sterol assay + ABCG5/8 sequencing"
        ),
        "syndrome": "Sitosterolaemia-Phytosterolaemia / AR-LOF-biallelic-ABCG5 / childhood-xanthomas-PATHOGNOMONIC / haemolytic-anaemia-PATHOGNOMONIC",
        "inheritance": "AR LOF (biallelic required, ABCG5 or ABCG8)",
        "pathognomonic": "tendon-tuberous-xanthomas-in-CHILDHOOD-PATHOGNOMONIC; haemolytic-anaemia-stomatocytes-PATHOGNOMONIC; ezetimibe-first-line-mechanism-matched; stanol-enriched-foods-CATASTROPHIC-CI; plant-sterol-assay-MANDATORY",
        "key_drugs_ci": ["stanol-sterol-enriched-foods-ABSOLUTELY-CI-sitosterolaemia", "statins-less-effective-primary-mechanism-not-matched"],
        "surveillance": ["plant-sterol-assay-q6months", "ezetimibe-response-monitoring", "haemolysis-screen-CBC-LDH-annually", "aortic-stenosis-echo-childhood", "cascade-siblings-plant-sterol"],
    },
]

DEFINITIONS = {
    "FH": "Familial hypercholesterolaemia: hereditary LDL-C elevation; FH1 (LDLR), FH2 (APOB), FH3 (PCSK9 GOF); prevalence 1 in 250 heterozygous; premature ASCVD; treat with statin+ezetimibe+PCSK9i.",
    "Corneal_arcus_lt45yr": "Corneal arcus (arcus corneae) before age 45: PATHOGNOMONIC for FH; lipid deposition in peripheral corneal stroma; distinguish from benign senile arcus (>50yr) which is non-specific.",
    "Tendon_xanthomas": "Cholesterol deposits in tendons (Achilles, extensor tendons of hands): PATHOGNOMONIC for FH; seen in ~50% LDLR FH; indicates longstanding LDL-C elevation; absence does NOT exclude FH.",
    "Palmar_xanthomas": "Xanthoma striata palmaris: yellow-orange deposits in palmar and digital creases: PATHOGNOMONIC UNIQUE for familial dysbetalipoproteinaemia (FH type 3, ApoE ε2/ε2); not seen in other FH types.",
    "HoFH": "Homozygous familial hypercholesterolaemia: both LDLR alleles affected; LDL-C >13 mmol/L; aortic stenosis in childhood; MI <20yr untreated; requires apheresis + evinacumab + lomitapide.",
    "ARH": "Autosomal recessive hypercholesterolaemia: biallelic LDLRAP1 LOF; splenomegaly PATHOGNOMONIC; LDLR present but cannot be internalised by hepatocytes; partial response to statins+PCSK9i; apheresis often required.",
    "FCS": "Familial chylomicronaemia syndrome: biallelic LPL LOF; TG >10 mmol/L; recurrent pancreatitis; eruptive xanthomas; fibrates+omega-3 ineffective; volanesorsen (anti-ApoC-III siRNA) FDA-approved.",
    "Familial_dysbetalipoproteinaemia": "FH type 3 = Fredrickson type III; ApoE ε2/ε2 + secondary trigger; BOTH TC and TG elevated equally; palmar crease xanthomas PATHOGNOMONIC; fibrates first-line.",
    "Sitosterolaemia": "ABCG5/ABCG8 biallelic LOF; plant sterol accumulation; childhood xanthomas + haemolytic anaemia PATHOGNOMONIC; ezetimibe mechanism-matched first-line; stanol-enriched foods CATASTROPHIC CI.",
    "Eruptive_xanthomas": "Small yellow-white papules on buttocks, arms, trunk; appear when TG >10 mmol/L; seen in LPL deficiency, APOA5 HTG, and FH type 3; regress with TG normalisation.",
    "PCSK9_inhibitors": "Monoclonal antibodies blocking PCSK9-LDLR interaction; evolocumab (Repatha) and alirocumab (Praluent) FDA-approved 2015; 50-60% LDL-C reduction on top of statin+ezetimibe.",
    "Volanesorsen": "ApoC-III siRNA; subcutaneous Q2W injection; FDA/Health Canada approved for FCS (LPL deficiency); ~70% TG reduction; monitor platelets (thrombocytopenia side-effect).",
    "LDL_apheresis": "Extracorporeal removal of LDL particles; Q1-2W sessions; for HoFH, severe FH statin-refractory, or ARH; bridges to novel therapies (evinacumab, gene therapy).",
    "Evinacumab": "Anti-ANGPTL3 monoclonal antibody; FDA-approved for HoFH 2021; effective regardless of LDLR status (LDLR-independent mechanism); reduces LDL-C 47% in HoFH.",
    "ApoE_genotype": "Three isoforms: ε2 (Cys158), ε3 (reference), ε4 (Arg112); ε2/ε2 required for FH type 3 (1 in 60 are ε2/ε2 but 95% never develop disease without trigger); ε4 associated with Alzheimer risk.",
    "Dutch_Lipid_Clinic_Network": "FH diagnostic scoring system: family history, clinical history, physical examination (xanthomas), LDL-C level, and genetic testing; score >=8 = definite FH; standard cascade testing tool.",
    "REDUCE_IT_trial": "Icosapentaenoic acid (EPA) 4g/day vs placebo on top of statin in HTG (TG 1.5-5.6 mmol/L); 25% relative risk reduction in MACE; applicable to APOA5 and mixed HTG patients.",
    "Cascade_FH": "Systematic testing of 1st-degree relatives of FH probands; each FH patient has 50% chance of affected children/siblings/parents; childhood testing at 2 yr for HoFH risk families; cascade testing discovers 50% of FH cases.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_overview() -> dict:
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(1.0, rng.gauss(
            8.0  if g["gene"] == "ABCG5"   else  # childhood xanthomas
            15.0 if g["gene"] == "LPL"     else  # FCS childhood pancreatitis
            20.0 if g["gene"] == "LDLRAP1" else  # ARH juvenile onset
            30.0 if g["gene"] == "PCSK9"   else  # GOF severe early
            35.0 if g["gene"] == "LDLR"    else  # classical FH
            42.0 if g["gene"] == "APOB"    else  # milder FDB
            45.0 if g["gene"] == "APOE"    else  # needs trigger
            40.0,  # APOA5
            9.0)), 1)
        pct_female = round(rng.uniform(40, 55), 1)
        cohort_stats.append({
            "gene": g["gene"],
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_female,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        })

    return {
        "atlas_name": "Hereditary-Familial-Hypercholesterolaemia-Lipid-Disorders-Atlas",
        "atlas": "Hereditary-Familial-Hypercholesterolaemia-Lipid-Disorders-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference -- LDLR / APOB / PCSK9 / LDLRAP1 / "
            "APOE / LPL / APOA5 / ABCG5"
        ),
        "total_patients": 320,
        "gene_count": 8,
        "seeds": "3766-3773",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "ad_fh_genes": ["LDLR", "APOB", "PCSK9"],
            "ar_severe_fh": ["LDLRAP1"],
            "mixed_dyslipidaemia": ["APOE"],
            "chylomicronaemia_pancreatitis": ["LPL"],
            "hypertriglyceridaemia": ["APOA5"],
            "sitosterolaemia": ["ABCG5"],
            "pancreatitis_risk": ["LPL", "APOA5"],
            "childhood_onset": ["ABCG5", "LPL", "LDLRAP1"],
            "pcsk9i_responsive": ["LDLR", "PCSK9"],
            "pcsk9i_partial_or_reduced": ["APOB", "LDLRAP1"],
            "ezetimibe_first_line": ["ABCG5"],
            "fibrates_first_line": ["APOE", "APOA5"],
        },
        "key_clinical_rules": [
            "LDLR: corneal-arcus <45yr = FH PATHOGNOMONIC; tendon-xanthomas PATHOGNOMONIC; statin+ezetimibe+PCSK9i standard; evinacumab HoFH-LDLR-null; stanol-enriched-foods worsen coexisting sitosterolaemia",
            "APOB: R3527Q/W most common; PCSK9i LESS EFFECTIVE vs LDLR FH; inclisiran+lomitapide effective; milder LDL elevation; genetic test distinguishes from LDLR clinically",
            "PCSK9: D374Y-GOF PATHOGNOMONIC most severe FH3; LOF R46L natural ASCVD protection 47%; PCSK9i directly targets GOF -- most responsive; inclisiran siRNA twice-yearly option",
            "LDLRAP1: splenomegaly PATHOGNOMONIC (not in AD FH); AR biallelic required; evinacumab ANGPTL3i effective regardless of LDLR; LDL apheresis often needed",
            "APOE: ε2/ε2 required but only 1-5% develop disease -- trigger needed; palmar-crease-xanthomas PATHOGNOMONIC UNIQUE; fibrates FIRST-LINE (not statins alone); ApoE genotyping mandatory",
            "LPL: recurrent pancreatitis PATHOGNOMONIC; fibrates+omega3 INEFFECTIVE pure LPL deficiency; volanesorsen siRNA FDA-approved; alcohol ABSOLUTELY CI; fat <20g/day",
            "APOA5: eruptive-xanthomas pancreatitis risk; REDUCE-IT omega3 25% MACE reduction; fibrates+EPA first-line; alcohol+DM2+oestrogen triggers worsen",
            "ABCG5: childhood-xanthomas PATHOGNOMONIC; haemolytic-anaemia stomatocytes PATHOGNOMONIC; ezetimibe mechanism-matched FIRST-LINE; stanol-enriched-foods CATASTROPHIC CI",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(1.0, rng.gauss(
            8.0  if g["gene"] == "ABCG5"   else
            15.0 if g["gene"] == "LPL"     else
            20.0 if g["gene"] == "LDLRAP1" else
            30.0 if g["gene"] == "PCSK9"   else
            35.0 if g["gene"] == "LDLR"    else
            42.0 if g["gene"] == "APOB"    else
            45.0 if g["gene"] == "APOE"    else
            40.0,
            9.0)), 1)
        pct_f = round(rng.uniform(40, 55), 1)
        # Pancreatitis events
        pancreatitis_n = rng.randint(18, 32) if g["gene"] in ("LPL", "APOA5") else rng.randint(1, 4)
        # CV events (MI/stroke)
        cv_event_n = rng.randint(14, 26) if g["gene"] in ("LDLR", "PCSK9", "LDLRAP1") else rng.randint(4, 14)
        # Xanthoma present
        xanthoma_n = rng.randint(20, 36) if g["gene"] in ("LDLR", "ABCG5", "LDLRAP1", "APOE") else rng.randint(6, 18)
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age_dx": mean_age,
            "pct_female": pct_f,
            "pancreatitis_n": pancreatitis_n,
            "cv_event_n": cv_event_n,
            "xanthoma_n": xanthoma_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "pathognomonic": g["pathognomonic"],
        }

    return {
        "atlas": "Hereditary-Familial-Hypercholesterolaemia-Lipid-Disorders-Atlas",
        "total_patients": 320,
        "seeds": "3766-3773",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "ad_fh_three_genes": ["LDLR", "APOB", "PCSK9"],
            "ar_requires_biallelic": ["LDLRAP1", "LPL", "ABCG5"],
            "mixed_phenotype_trigger_needed": ["APOE"],
            "pancreatitis_risk_genes": ["LPL", "APOA5"],
            "childhood_onset": ["ABCG5", "LPL", "LDLRAP1"],
            "pcsk9i_most_effective": ["LDLR", "PCSK9"],
            "pcsk9i_reduced_efficacy": ["APOB", "LDLRAP1"],
            "ezetimibe_mechanism_matched": ["ABCG5"],
            "fibrates_first_line": ["APOE", "APOA5"],
            "stanol_foods_catastrophic_ci": ["ABCG5"],
            "fibrates_omega3_ineffective": ["LPL"],
            "apheresis_needed": ["LDLRAP1", "LDLR"],
            "evinacumab_indicated": ["LDLRAP1", "LDLR"],
            "cascade_all_fh_genes": ["LDLR", "APOB", "PCSK9", "LDLRAP1", "APOE", "LPL", "APOA5", "ABCG5"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Familial-Hypercholesterolaemia-Lipid-Disorders-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "LDLR-CORNEAL-ARCUS-LT45YR-PATHOGNOMONIC": (
                "Corneal arcus before age 45 = FH PATHOGNOMONIC -- "
                "distinguish from benign senile arcus >50yr; trigger LDLR genetic testing; "
                "Dutch Lipid Clinic Network Score: corneal arcus <45yr = +4 points"
            ),
            "APOB-PCSK9i-REDUCED-EFFICACY": (
                "APOB FDB (R3527Q/W): PCSK9i LESS effective vs LDLR FH -- "
                "ApoB-100 cannot bind LDLR even if LDLR upregulated; "
                "prefer inclisiran (ApoB-synthesis siRNA) or lomitapide; "
                "genetic test mandatory to guide PCSK9i dose expectation"
            ),
            "PCSK9-D374Y-GOF-MOST-SEVERE-FH": (
                "PCSK9 D374Y GOF = most severe FH type 3 -- binds LDLR 25x tighter than wild-type; "
                "PCSK9 inhibitors DIRECTLY target the D374Y protein -- most responsive to evolocumab/alirocumab; "
                "LOF R46L = natural protection: LDL-C 15-28% lower, ASCVD 47% lower lifetime"
            ),
            "LDLRAP1-SPLENOMEGALY-PATHOGNOMONIC": (
                "Splenomegaly in hypercholesterolaemia = ARH/LDLRAP1 PATHOGNOMONIC -- "
                "absent in all AD FH (LDLR, APOB, PCSK9); "
                "LDLR functional but cannot be internalised by hepatocyte without adaptor; "
                "evinacumab (ANGPTL3i) effective regardless of LDLR status"
            ),
            "APOE-PALMAR-XANTHOMA-PATHOGNOMONIC-UNIQUE": (
                "Palmar crease xanthomas (xanthoma striata palmaris) = FH type 3 PATHOGNOMONIC UNIQUE -- "
                "seen ONLY in familial dysbetalipoproteinaemia (ApoE ε2/ε2); "
                "fibrates FIRST-LINE (not statins alone); ApoE genotyping required; "
                "95% of ε2/ε2 need secondary trigger to express disease -- treat trigger first"
            ),
            "LPL-FIBRATES-OMEGA3-INEFFECTIVE": (
                "Fibrates and omega-3 INEFFECTIVE in pure LPL biallelic deficiency -- "
                "fibrates work by upregulating LPL; omega-3 works partly via LPL-activation; "
                "no LPL enzyme present = no effect; use fat restriction <20g/day + volanesorsen; "
                "alcohol ABSOLUTELY CI -- markedly worsens chylomicronaemia"
            ),
            "ABCG5-STANOL-FOODS-CATASTROPHIC-CI": (
                "Plant stanol/sterol-enriched foods (margarines, yogurts): CATASTROPHIC CI in sitosterolaemia -- "
                "recommended to lower LDL in general population but paradoxically increase plant sterols in ABCG5/8 deficiency; "
                "ezetimibe FIRST-LINE mechanism-matched (blocks intestinal NPC1L1 sterol absorption); "
                "haemolytic anaemia with stomatocytes = PATHOGNOMONIC unique feature"
            ),
            "ABCG5-CHILDHOOD-XANTHOMAS-PATHOGNOMONIC": (
                "Tendon/tuberous xanthomas developing in CHILDHOOD (3-10yr) = sitosterolaemia PATHOGNOMONIC -- "
                "xanthomas before age 10 must prompt ABCG5/ABCG8 testing and plasma phytosterol assay; "
                "standard cholesterol panel does NOT measure plant sterols -- must specifically request"
            ),
            "LPL-VOLANESORSEN-FDA-PANCREATITIS-CI": (
                "LPL deficiency FCS: recurrent pancreatitis life-threatening -- TG >10 mmol/L trigger; "
                "volanesorsen (anti-ApoC-III siRNA) FDA/Health Canada approved 2019; "
                "70% TG reduction; Q2W subcutaneous; monitor platelets for thrombocytopenia; "
                "oestrogen AVOID (worsens TG); fat restriction <20g/day cornerstone"
            ),
            "CASCADE-ALL-FH-GENES-MANDATORY": (
                "Cascade genetic testing all 1st-degree relatives MANDATORY for all 8 lipid-disorder genes -- "
                "AD genes (LDLR, APOB, PCSK9): 50% 1st-degree risk; "
                "AR genes (LDLRAP1, LPL, ABCG5): siblings 25% risk; parents are obligate carriers; "
                "paediatric cholesterol screening at 9-11yr; HoFH risk families test at 2yr"
            ),
            "HoFH-EVINACUMAB-ANGPTL3i": (
                "Homozygous FH (HoFH): evinacumab (anti-ANGPTL3) reduces LDL-C 47% REGARDLESS of LDLR status -- "
                "only therapy effective in null-null LDLR HoFH (PCSK9i require at least one functional LDLR); "
                "FDA-approved 2021 for HoFH; combine with statins+ezetimibe+lomitapide+apheresis"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Familial Hypercholesterolaemia & Lipid Disorders Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, pancreatitis={d['pancreatitis_n']}/40, CV-events={d['cv_event_n']}/40, xanthomas={d['xanthoma_n']}/40")
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
