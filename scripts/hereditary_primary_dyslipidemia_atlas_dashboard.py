#!/usr/bin/env python3
"""Hereditary-Primary-Dyslipidemia-Atlas — Complete 8-Gene Familial Hypercholesterolaemia & Dyslipidaemia Atlas
LDLR    (Low-Density Lipoprotein Receptor; 860 aa precursor, 839 aa mature; 19p13.2; AR/AD;
         FH1 — Familial Hypercholesterolaemia type 1; most common monogenic FH (~1/250 heterozygous);
         LDLR mediates hepatic LDL endocytosis; LOF -> LDL-C cannot be cleared -> marked elevation;
         tendon xanthomata + corneal arcus + premature CVD as early as 2nd-3rd decade in HeFH;
         HoFH (biallelic) = untreated LDL-C 12-30 mmol/L; aortic stenosis, fatal MI <20y;
         high-intensity statin + ezetimibe + PCSK9i first-line; LDL apheresis / lomitapide for HoFH;
         seed SEED_BASE+0) *
APOB    (Apolipoprotein B-100; 4563 aa; 2p24.1; AD;
         FDB — Familial Defective ApoB; Arg3527Gln (R3500Q in mature protein) European founder mutation;
         APOB is the sole ligand for LDLR; Arg3527Gln disrupts LDLR-binding domain ->
         LDL particle cannot be endocytosed efficiently -> LDL-C elevated (milder than FH1);
         phenotype: moderate hypercholesterolaemia, tendon xanthomata in 30-50%; corneal arcus;
         premature CVD (milder than LDLR); responds to statins + PCSK9i;
         seed SEED_BASE+1) *
PCSK9   (Proprotein Convertase Subtilisin/Kexin Type 9; 692 aa; 1p32.3; AD GOF/LOF;
         FH3 (GOF) or hypocholesterolaemia/CVD protection (LOF);
         PCSK9 binds LDLR on hepatocyte surface -> targets LDLR for lysosomal degradation;
         GOF mutations (D374Y most common): degraded LDLR -> LDL-C very high = FH3;
         LOF variants (R46L, Y142X, C679X): LDLR recycled more -> LDL-C low -> 88% ↓ CVD risk;
         Evolocumab (Repatha) + Alirocumab (Praluent) = mAb anti-PCSK9 -> LDLR recycled;
         seed SEED_BASE+2) *
LDLRAP1 (LDL Receptor Adaptor Protein 1; 308 aa; 1p36.11; AR;
         ARH — Autosomal Recessive Hypercholesterolaemia; phosphotyrosine-binding domain adaptor;
         LDLRAP1 links LDLR cytoplasmic tail to clathrin-coated pit machinery in hepatocytes;
         ARH: hepatic LDLR expressed but cannot internalise LDL -> LDL-C 10-20 mmol/L;
         DISTINGUISHES from HoFH: LDLR present on lymphocytes (lymphocyte binding NORMAL); hepatic
         internalisation absent; responds partially to statins + PCSK9i;
         seed SEED_BASE+3) *
LIPA    (Lysosomal Acid Lipase; 399 aa; 10q23.31; AR;
         Wolman disease (complete LOF, infantile) and LAL-D (partial LOF, childhood/adult);
         LAL hydrolyses cholesteryl esters and triglycerides in lysosomes; LOF ->
         cholesteryl ester accumulation in lysosomes of liver/macrophages/adrenal;
         Wolman: hepatosplenomegaly + adrenal calcification (PATHOGNOMONIC) + death <12 months;
         LAL-D (Cholesteryl Ester Storage Disease, CESD): hepatomegaly + elevated LDL-C +
         low HDL-C + elevated transaminases; Sebelipase alfa (Kanuma) = ERT, FDA/EMA 2015;
         seed SEED_BASE+4) *
ABCA1   (ATP-Binding Cassette transporter A1; 2261 aa; 9q31.1; AR;
         Tangier disease (severe biallelic) / familial hypoalphalipoproteinaemia (FHA, heterozygous);
         ABCA1 mediates cholesterol efflux from cells to lipid-poor apoA-I -> HDL biogenesis;
         Tangier disease: HDL-C near zero (PATHOGNOMONIC) + orange-yellow tonsils (PATHOGNOMONIC) +
         polyneuropathy (relapsing) + hepatosplenomegaly + corneal opacities;
         orange tonsils = lipid-laden macrophages in tonsillar tissue; classic physical sign;
         premature CVD despite low LDL-C (impaired reverse cholesterol transport);
         seed SEED_BASE+5) *
ABCG5   (ATP-Binding Cassette transporter G5 / Sterolin-1; 651 aa; 2p21; AR;
         Sitosterolaemia (also ABCG8-based); ABCG5-ABCG8 obligate heterodimer;
         ABCG5-ABCG8 heterodimer pumps plant sterols (sitosterol, campesterol) and cholesterol
         out of enterocytes and biliary epithelium back into gut lumen / bile;
         ABCG5 biallelic LOF -> failure to excrete plant sterols -> sitosterol accumulates in blood;
         plasma sitosterol >200 µmol/L (normal <10); premature atherosclerosis; haemolytic anaemia;
         xanthomata (including tendon xanthomata) in childhood even with low/normal LDL-C;
         ezetimibe (NPC1L1 inhibitor) + low plant sterol diet = first-line;
         seed SEED_BASE+6) *
ABCG8   (ATP-Binding Cassette transporter G8 / Sterolin-2; 673 aa; 2p21; AR;
         Sitosterolaemia type 2 (same phenotype as ABCG5; same gene locus 2p21);
         ABCG8 is the obligate heterodimer partner of ABCG5; either gene LOF = same disease;
         Distinguishing ABCG5 vs ABCG8: requires sequencing both genes (identical phenotype);
         sitosterol + campesterol + brassicasterol plasma levels elevated equally regardless of which
         ABC transporter gene is mutated; ABCG8 variants more common in South Asian populations;
         ezetimibe + low plant sterol diet; bile acid sequestrants second-line;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3006-3013)
"""
import random

SEED_BASE = 3006

ATLAS_GENES = [
    {
        "gene": "LDLR",
        "protein": (
            "LDLR -- 19p13.2 AR/AD -- 860aa precursor 839aa mature -- Low-Density-Lipoprotein-Receptor-"
            "95kDa-EGF-Precursor-Homology-Domain-Ligand-Binding-Repeats-1-7-"
            "FH1-Familial-Hypercholesterolaemia-Most-Common-1-in-250-HeFH-Tendon-Xanthomata-PATHOGNOMONIC-OMIM-606945"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "860 aa precursor / 839 aa mature / 95 kDa (LDLR -- Low-density lipoprotein receptor; "
            "FUNCTION: cell-surface endocytic receptor; mediates receptor-mediated endocytosis of LDL; "
            "  Domain structure: "
            "    (1) Signal peptide (aa 1-21); "
            "    (2) Ligand-binding domain (aa 22-292): 7 cysteine-rich repeats (LBD1-7); "
            "        LBD3/4/5: APOB-100 and APOE contact residues; "
            "    (3) EGF precursor homology domain (aa 293-692): contains EGF-A, EGF-B, beta-propeller, EGF-C; "
            "        beta-propeller: responsible for pH-dependent release of LDL in endosome (critical!); "
            "    (4) O-linked sugar domain (aa 693-750): extensive glycosylation; "
            "    (5) Transmembrane domain (aa 751-771): single pass; "
            "    (6) Cytoplasmic tail (aa 772-839): NPVY motif -> clathrin-coated pit targeting; "
            "LDLR CYCLE: "
            "  (1) LDL binds LDLR at cell surface (neutral pH); "
            "  (2) LDLR-LDL internalised in clathrin-coated vesicle (via LDLRAP1-NPVY interaction); "
            "  (3) Endosome acidifies (pH 5.0-5.5) -> beta-propeller folds -> LDL released; "
            "  (4) LDLR recycled to cell surface (>100 cycles per receptor); "
            "  (5) PCSK9 binds LDLR EGF-A -> blocks release -> LDLR degraded in lysosome; "
            "EXPRESSION: highest in liver (70% LDL clearance); also adrenal, ovary, lymphocytes; "
            "LDLR LOF MECHANISM: "
            "  Class 1 (synthesis): no LDLR mRNA/protein; "
            "  Class 2 (transport): LDLR misfolded -> ER retention -> degradation (most common, ~45%); "
            "  Class 3 (binding): defective LDL binding (LBD mutations); "
            "  Class 4 (internalisation): defective endocytosis (NPVY motif mutations); "
            "  Class 5 (recycling): LDLR cannot release LDL -> degraded with cargo; "
            "encoded 19p13.2; OMIM gene 606945, disease FH1 #143890"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (HeFH) / AUTOSOMAL RECESSIVE (HoFH) -- LDLR / FH1: "
            "HETEROZYGOUS FH (HeFH) -- MOST COMMON MONOGENIC DISORDER: "
            "  Prevalence: ~1 in 250 in general population (previously underestimated at 1/500); "
            "  LDL-C: 4.5-9.0 mmol/L (untreated); "
            "  Tendon xanthomata: Achilles tendons, extensor tendons dorsum of hands; "
            "    PATHOGNOMONIC for FH; palpable Achilles tenderness/thickening (>9 mm on USS); "
            "  Corneal arcus: before age 45 years = significant for FH; "
            "  Xanthelasma: periorbital; non-specific (also in normolipidaemia); "
            "  CVD: premature; MI age 40-60 males untreated; females 10-20 years later; "
            "  Dutch Lipid Clinic Network (DLCN) score >= 6 = probable FH, >= 8 = definite FH; "
            "HOMOZYGOUS FH (HoFH): "
            "  Prevalence: ~1 in 160,000-300,000; "
            "  LDL-C: 12-30 mmol/L (untreated); "
            "  Xanthomata: cutaneous (flat, tuberous, planar) from early childhood; "
            "  Aortic stenosis/aortopathy: atherosclerotic plaque in aortic root (DISTINCTIVE of HoFH); "
            "  Fatal MI possible before age 20 without treatment; "
            "  True HoFH vs compound heterozygote vs ARH: check LDLRAP1; "
            "DIAGNOSIS: "
            "  LDL-C + personal/family history of premature CVD + tendon xanthomata; "
            "  DLCN score; Simon Broome Criteria; "
            "  Genetic: LDLR sequencing + MLPA (large deletions 5-10%); "
            "TREATMENT: "
            "  HeFH: high-intensity statin (atorvastatin 40-80 mg or rosuvastatin 20-40 mg) + "
            "    ezetimibe -> ~50% LDL-C reduction; target LDL-C <1.8 mmol/L (established CVD) or <2.6 mmol/L; "
            "  PCSK9 inhibitors (Evolocumab, Alirocumab): additional 50-60% LDL-C reduction; "
            "  HoFH additional options: lomitapide (MTP inhibitor), LDL apheresis (every 2 weeks); "
            "  Inclisiran (siRNA anti-PCSK9): twice-yearly injection option; "
            "  Cascade testing: all 1st-degree relatives (50% carriers); start at birth/age 2 (HoFH)"
        ),
        "disease_category": (
            "FH1-LDLR-MOST-COMMON-MONOGENIC-FH-1-IN-250: "
            "  KEY RULE: Achilles tendon xanthomata + LDL-C >4.9 mmol/L (untreated) = PROBABLE FH -> genetic testing; "
            "  DLCN SCORE >=8: definite FH; high-intensity statin + ezetimibe IMMEDIATELY; "
            "  HoFH LDL-C 12-30: aortic stenosis in childhood; LDL apheresis; lomitapide; "
            "  LDLR CLASS 2 (misfolding): most common mutation class (~45%); ER retention; statins still helpful (upregulate normal allele in HeFH); "
            "  PCSK9i: add if LDL-C target not met on statin+ezetimibe; FDA/EMA approved; 50-60% additional reduction; "
            "  CASCADE TESTING MANDATORY: 1st-degree relatives; 50% carrier risk; start at age 2 if HoFH suspected; "
        ),
    },
    {
        "gene": "APOB",
        "protein": (
            "APOB -- 2p24.1 AD -- 4563aa -- Apolipoprotein-B-100-"
            "550kDa-Single-Pass-Amphipathic-Belt-Structure-"
            "FDB-Familial-Defective-ApoB-Arg3527Gln-European-Founder-LDL-Receptor-Ligand-OMIM-107730"
        ),
        "locus": "2p24.1",
        "protein_size": (
            "4563 aa / 550 kDa (APOB-100 -- full-length form present in LDL/IDL/VLDL; "
            "APOB-48 (2153 aa): intestinal form; present in chylomicrons; "
            "FUNCTION: "
            "  Structural scaffold of VLDL, IDL, LDL, Lp(a); one APOB-100 molecule per LDL particle; "
            "  Sole LDLR-binding ligand: domain around Arg3527 (mature Arg3500 equivalent in older nomenclature); "
            "  Synthesised in hepatocytes -> co-translationally lipidated by MTP (microsomal TG transfer protein); "
            "  After VLDL secretion: VLDL -> IDL -> LDL via lipolysis of TG by LPL and HL; "
            "APOB STRUCTURE: "
            "  NH2-terminal 1000 aa: beta-alpha1 module (amphipathic alpha-helix belt); "
            "  LDLR-binding domain: central region (~aa 3000-3600); "
            "    Arg3527 (R3527, mature numbering Arg3500 in some literature): critical for LDLR binding; "
            "    Arg3527Gln (c.10580G>A): reduces LDLR-binding affinity by 60-70%; "
            "  LPL-binding site: NH2-terminal segment; "
            "  Signal peptide: aa 1-27 (cleaved); "
            "FDB MECHANISM: "
            "  Arg3527Gln substitution: positive charge -> neutral -> LDLR-binding domain defect; "
            "  LDL particle cannot bind LDLR efficiently -> circulating LDL-C elevated; "
            "  Severity: milder than LDLR FH because some residual LDLR binding remains; "
            "  LDL-C: typically 5-8 mmol/L (HeFH range but lower); "
            "encoded 2p24.1; OMIM gene 107730, disease FDB #144010"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT -- APOB / FAMILIAL DEFECTIVE APOB (FDB): "
            "PREVALENCE: "
            "  Arg3527Gln: ~1 in 1000 in European populations (common founder variant); "
            "  Most common in: Germany, Austria, Switzerland, Northern Europe; "
            "  Accounts for ~5% of all FH phenotypes in European populations; "
            "CLINICAL FEATURES: "
            "  Phenotype clinically indistinguishable from LDLR-FH HeFH; "
            "  LDL-C: 5-8 mmol/L (slightly milder than LDLR FH); "
            "  Tendon xanthomata: present in 30-50% (less frequent than LDLR FH); "
            "  Corneal arcus: age <45; "
            "  Premature CVD: present but somewhat later onset than LDLR FH; "
            "  HDL-C and TG usually normal (distinguishes from polygenic hypercholesterolaemia); "
            "FDB vs LDLR FH DISTINCTION: "
            "  Cannot be distinguished clinically (both cause isolated elevated LDL-C); "
            "  GENETIC TESTING required to separate FDB from LDLR FH; "
            "  FDB: Arg3527Gln single variant usually; LDLR FH: >2000 known variants; "
            "  TREATMENT RESPONSE: FDB responds BETTER to statins (LDL-C reduction ~45-55%) "
            "    because LDLR itself is functional -> statin upregulates LDLR -> more clearance; "
            "    LDLR FH: statin upregulates absent/reduced LDLR -> less clearance improvement; "
            "DIAGNOSIS: "
            "  LDL-C elevation + family history; "
            "  Genetic: Arg3527Gln targeted testing (or FH gene panel including APOB); "
            "TREATMENT: "
            "  High-intensity statin (excellent response: ~50% LDL-C reduction); "
            "  Ezetimibe: additional 15-20%; "
            "  PCSK9 inhibitors: effective (LDLR functional -> PCSK9i allows more recycling); "
            "  Cascade testing: 50% offspring risk"
        ),
        "disease_category": (
            "FDB-APOB-ARG3527GLN-EUROPEAN-FOUNDER-MILDER-THAN-LDLR-FH: "
            "  KEY RULE: elevated LDL-C + European ancestry + FH phenotype + statins respond well -> consider FDB; "
            "  ARG3527GLN: single variant accounts for >95% FDB; targeted test sufficient; "
            "  STATIN RESPONSE BETTER THAN LDLR FH: LDLR functional -> statin upregulation more effective; "
            "  FDB vs LDLR FH: identical phenotype; only genetic testing distinguishes; "
            "  PCSK9i EFFECTIVE: LDLR present and functional; PCSK9i prevents degradation -> more recycling; "
        ),
    },
    {
        "gene": "PCSK9",
        "protein": (
            "PCSK9 -- 1p32.3 AD GOF/LOF -- 692aa -- Proprotein-Convertase-Subtilisin-Kexin-Type-9-"
            "74kDa-Secreted-Protease-LDLR-Degradation-Master-Regulator-"
            "FH3-GOF-D374Y-Evolocumab-Alirocumab-LOF-CVD-Protection-88pct-Risk-Reduction-OMIM-607786"
        ),
        "locus": "1p32.3",
        "protein_size": (
            "692 aa / 74 kDa secreted mature form (PCSK9 -- proprotein convertase subtilisin/kexin type 9; "
            "FUNCTION: regulator of LDLR expression/degradation; "
            "  PCSK9 BIOSYNTHESIS: "
            "    Signal peptide (aa 1-30) -> prodomain (31-152) -> catalytic domain (153-451) -> "
            "    hinge (452-507) -> C-terminal domain (508-692); "
            "    PCSK9 autocleaves its prodomain in ER -> secreted as non-covalent prodomain-catalytic complex; "
            "    Prodomain remains bound -> catalytic site BLOCKED -> PCSK9 is an INACTIVE protease once secreted; "
            "  PCSK9 MECHANISM OF ACTION (extracellular): "
            "    PCSK9 binds LDLR EGF-A domain at cell surface (pH 7.4); "
            "    LDLR-PCSK9 complex internalised; "
            "    In endosome (pH 5.5): PCSK9 prodomain rebinds more tightly at low pH -> "
            "      LDLR CANNOT release LDL -> LDLR directed to lysosomal degradation; "
            "    Net effect: PCSK9 reduces cell-surface LDLR -> reduces LDL clearance; "
            "  GOF MUTATIONS (FH3): "
            "    D374Y: ~10x increased LDLR-binding affinity -> massive LDLR degradation -> LDL-C very high; "
            "    Other: S127R, F216L; "
            "  LOF VARIANTS (protective): "
            "    R46L (common in Europeans, ~3%): reduced LDLR binding -> more LDLR -> LDL-C lower; "
            "    Y142X, C679X: truncating -> no PCSK9 secreted; "
            "    Black subjects: C679X: ~2.4% carrier frequency -> 28% lower LDL-C + 88% lower CVD risk; "
            "  PCSK9 INHIBITORS: "
            "    Evolocumab (Repatha): fully human IgG2 mAb; "
            "    Alirocumab (Praluent): humanised IgG1 mAb; "
            "    Both bind PCSK9 -> block LDLR binding -> LDLR recycled -> LDL-C ↓50-60% on top of statins; "
            "    Inclisiran: siRNA -> reduces hepatic PCSK9 mRNA -> twice-yearly dosing; "
            "encoded 1p32.3; OMIM gene 607786, disease FH3 #603776"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GOF (FH3) / AUTOSOMAL DOMINANT/RECESSIVE LOF (PROTECTIVE): "
            "GOF MUTATIONS = FH3 (Familial Hypercholesterolaemia type 3): "
            "  Prevalence: rare (~1-3% of FH cases); D374Y mainly in Norwegian families; "
            "  LDL-C: markedly elevated (similar to LDLR FH); "
            "  Phenotype: clinically indistinguishable from LDLR FH (tendon xanthomata, premature CVD); "
            "  UNIQUE: FH3 phenotype responds LESS WELL to statins (PCSK9 still abundant -> continues "
            "    to degrade the LDLR upregulated by statins); "
            "  PCSK9i SPECIFICALLY EFFECTIVE in FH3: directly neutralises the overactive PCSK9; "
            "LOF VARIANTS = CVD PROTECTION: "
            "  Natural human experiment: carriers of PCSK9 LOF have lifelong ~25% lower LDL-C; "
            "  CVD risk reduction: ~88% (R46L + Y142X/C679X in Dallas Heart Study); "
            "  Linear relationship: every 1 mmol/L LDL-C reduction over lifetime -> ~22% CVD risk reduction; "
            "  LOF carriers: health endpoints uniformly excellent; validates LDL-C causality in atherosclerosis; "
            "PCSK9 INHIBITOR CLINICAL EVIDENCE: "
            "  FOURIER trial (Evolocumab): LDL-C ↓59%; CVD events ↓15% (NNT 66 over 2 years); "
            "  ODYSSEY OUTCOMES (Alirocumab): LDL-C ↓54%; MACE ↓15%; mortality benefit in high LDL tertile; "
            "  ORION trials (Inclisiran): LDL-C ↓50% at 12 months; twice-yearly dosing; "
            "DIAGNOSIS: "
            "  FH gene panel (LDLR + APOB + PCSK9); "
            "  D374Y hotspot analysis if Norwegian/Northern European + FH3 suspected; "
            "TREATMENT: "
            "  FH3 (GOF): PCSK9i FIRST-LINE (add to statin + ezetimibe); highest benefit class; "
            "  All FH: PCSK9i as add-on to achieve LDL-C target; "
            "  LOF screening: not clinically actionable directly but validates treatment target"
        ),
        "disease_category": (
            "FH3-PCSK9-GOF-D374Y-LOF-CVD-PROTECTION-88pct: "
            "  KEY RULE: FH phenotype + FH3 diagnosis -> PCSK9i FIRST CHOICE add-on (statin less effective alone); "
            "  D374Y: ~10x LDLR affinity; Norwegian/European families; "
            "  PCSK9i FOURIER/ODYSSEY: 50-60% additional LDL-C reduction on statin; CVD events reduced 15%; "
            "  LOF Y142X/C679X: Black women; 88% lower CVD risk; natural proof of LDL-C causality; "
            "  INCLISIRAN TWICE YEARLY: practical for adherence; same LDL-C reduction as mAb; "
            "  STATIN-PCSK9i SYNERGY: statin upregulates LDLR transcription; PCSK9i prevents LDLR degradation; "
        ),
    },
    {
        "gene": "LDLRAP1",
        "protein": (
            "LDLRAP1 -- 1p36.11 AR -- 308aa -- LDL-Receptor-Adaptor-Protein-1-"
            "35kDa-PTB-Domain-Clathrin-Coated-Pit-NPVY-Motif-Hepatocyte-Specific-"
            "ARH-Autosomal-Recessive-Hypercholesterolaemia-Hepatic-LDLR-Internalisation-Defect-OMIM-605747"
        ),
        "locus": "1p36.11",
        "protein_size": (
            "308 aa / 35 kDa (LDLRAP1 -- LDL receptor adaptor protein 1; "
            "FUNCTION: clathrin-coated pit adaptor for LDLR in hepatocytes; "
            "  DOMAIN STRUCTURE: "
            "    N-terminal lipid-binding/actin-binding region; "
            "    Central region; "
            "    C-terminal PTB (phosphotyrosine-binding) domain; "
            "  LDLRAP1 MECHANISM: "
            "    PTB domain recognises and binds NPVY motif in LDLR cytoplasmic tail; "
            "    LDLRAP1 simultaneously binds clathrin-coated pit components (clathrin, adaptor protein 2); "
            "    This bridging function is ESSENTIAL for hepatic LDLR endocytosis; "
            "    Liver-specific expression pattern -> only hepatic LDL uptake affected; "
            "  ARH CONSEQUENCE: "
            "    Hepatic LDLR expressed normally on surface but CANNOT be internalised; "
            "    Lymphocyte LDLR (lymphocytes use different adaptor) -> LDL binding PRESERVED; "
            "    This distinguishes ARH from HoFH: "
            "      ARH: lymphocyte LDLR binding NORMAL; liver LDLR internalisation ABSENT; "
            "      HoFH: lymphocyte LDLR binding ABSENT (LDLR not expressed/misfolded); "
            "    PCSK9i partially effective (LDLR protein present on hepatocytes -> mAb can prevent degradation "
            "      even without full internalisation -> LDLR surface level slightly restored); "
            "  EXPRESSION: liver >> intestine; lymphocytes express different adaptor -> not affected; "
            "encoded 1p36.11; OMIM gene 605747, disease ARH #603813"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- LDLRAP1 / AUTOSOMAL RECESSIVE HYPERCHOLESTEROLAEMIA (ARH): "
            "PREVALENCE: "
            "  Rare; highest frequency in Sardinian and Lebanese founder populations; "
            "  Distinct from HoFH (true biallelic LDLR mutations); "
            "CLINICAL FEATURES: "
            "  LDL-C: 9-20 mmol/L (intermediate between HeFH and HoFH); "
            "  Tendon xanthomata: prominent; "
            "  Coronary artery disease: present but typically milder onset than HoFH; "
            "  Aortic stenosis: less common than HoFH; "
            "  Hepatic steatosis: sometimes; "
            "ARH vs HoFH DIFFERENTIAL: "
            "  CRITICAL DISTINCTION: LYMPHOCYTE LDL BINDING TEST; "
            "    ARH: lymphocyte LDLR binding NORMAL (>50% of control); "
            "    HoFH: lymphocyte LDLR binding reduced (<20% of control); "
            "  ARH: both parents heterozygous (apparently unaffected); "
            "  HoFH: parents often HeFH (both hypercholesterolaemic); "
            "  ARH genetics: biallelic LDLRAP1 mutations; single-exon deletion common in Sardinia; "
            "TREATMENT: "
            "  Statins: partially effective (LDLR present on surface -> statin upregulates LDLR mRNA -> "
            "    more LDLR protein even if internalisation impaired); "
            "  Ezetimibe: useful adjunct; "
            "  PCSK9i: partially effective (LDLR present -> PCSK9 blockade preserves some LDLR function); "
            "  LDL apheresis: required if targets not met; "
            "  Lomitapide: considered for refractory cases (MTP inhibitor, reduces hepatic VLDL secretion); "
            "DIAGNOSIS: "
            "  Severe hypercholesterolaemia + AR inheritance pattern + normal lymphocyte LDLR binding; "
            "  Genetic: LDLRAP1 sequencing (LDLR sequencing first to exclude LDLR HoFH)"
        ),
        "disease_category": (
            "ARH-LDLRAP1-HEPATIC-ENDOCYTOSIS-DEFECT-LYMPHOCYTE-LDLR-NORMAL: "
            "  KEY RULE: severe hypercholesterolaemia + AR pattern + normal lymphocyte LDLR binding = ARH (not HoFH); "
            "  LYMPHOCYTE LDLR BINDING TEST: ARH normal; HoFH reduced; key differentiating assay; "
            "  SARDINIAN FOUNDER DELETION: common single-exon deletion; Sardinian + severe FH = test LDLRAP1; "
            "  PCSK9i PARTIALLY EFFECTIVE: LDLR present (unlike HoFH with class 2/null LDLR mutations); "
            "  LDL APHERESIS: as for HoFH if target LDL-C not achieved; fortnightly regimen; "
        ),
    },
    {
        "gene": "LIPA",
        "protein": (
            "LIPA -- 10q23.31 AR -- 399aa -- Lysosomal-Acid-Lipase-"
            "45kDa-alpha-beta-Hydrolase-Fold-Lysosomal-Cholesteryl-Ester-TG-Hydrolysis-"
            "Wolman-Disease-Complete-LOF-Adrenal-Calcification-PATHOGNOMONIC-LAL-D-CESD-Partial-LOF-Sebelipase-Alfa-OMIM-613497"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "399 aa / 45 kDa lysosomal form (LIPA -- lysosomal acid lipase; "
            "FUNCTION: lysosomal enzyme; hydrolyses cholesteryl esters and TG within lysosomes; "
            "  SUBSTRATE: cholesteryl esters and triglycerides delivered to lysosomes via "
            "    LDLR-mediated endocytosis of LDL; also via phagocytosis in macrophages/foam cells; "
            "  PRODUCT: free cholesterol (substrate for membrane synthesis, steroid synthesis, bile); "
            "    Free fatty acids; "
            "  REGULATION: LAL activity essential for cholesterol homeostasis in liver and macrophages; "
            "    LAL LOF -> CE accumulation -> failed intracellular cholesterol sensing -> "
            "      LDL receptors not downregulated -> continued LDL uptake -> CE storage disease; "
            "  WOLMAN DISEASE (complete LOF): "
            "    Massive CE + TG accumulation in liver (hepatomegaly, cirrhosis) + adrenal glands; "
            "    Adrenal calcification: bilateral on imaging (X-ray/CT) -- PATHOGNOMONIC; "
            "    Onset: weeks 1-2 of life; diarrhoea + vomiting + failure to thrive; "
            "    Death: typically 3-6 months without treatment; "
            "  LAL-D/CESD (partial LOF, >1-3% residual activity): "
            "    Hepatomegaly + hepatic steatosis + cirrhosis; "
            "    Dyslipidaemia: elevated LDL-C + low HDL-C; elevated transaminases; "
            "    Onset: childhood to adulthood; underdiagnosed; "
            "    Masquerades as NAFLD; liver biopsy: CE-laden hepatocytes (birefringent CE crystals); "
            "encoded 10q23.31; OMIM gene 613497, Wolman #620151, CESD #278000"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- LIPA / WOLMAN DISEASE + LAL-D (CESD): "
            "WOLMAN DISEASE (severe): "
            "  Incidence: ~1 in 350,000; "
            "  Onset: neonatal (2-4 weeks); "
            "  Clinical features: "
            "    Hepatosplenomegaly; massive hepatomegaly; "
            "    Bilateral adrenal calcification on imaging (PATHOGNOMONIC); "
            "    Severe gastrointestinal symptoms: malabsorption, diarrhoea, vomiting; "
            "    Failure to thrive; cachexia; "
            "  Death: 3-6 months without treatment; "
            "  TREATMENT: Sebelipase alfa (Kanuma): recombinant human LAL; "
            "    FDA approved December 2015; EMA approved August 2015; "
            "    First-line for both Wolman and CESD; "
            "    Dosing Wolman: 1 mg/kg weekly (escalate to 3 mg/kg if no response); "
            "    Outcome: survival with ERT; "
            "CESD / LAL-D (partial, residual activity): "
            "  Incidence: ~1 in 40,000-300,000 (underdiagnosed); "
            "  Onset: childhood (often discovered during evaluation of hepatomegaly or dyslipidaemia); "
            "  Clinical features: "
            "    Hepatomegaly (universal); "
            "    Elevated LDL-C + low HDL-C; "
            "    Elevated ALT/AST; "
            "    Progressive hepatic fibrosis/cirrhosis; "
            "    Atherosclerosis: premature CVD; "
            "  DIAGNOSIS: "
            "    DBS (dried blood spot) LIPA enzyme activity assay: < 0.02 nmol/punch/hr; "
            "    Genetic: LIPA sequencing; splice-site mutation c.894G>A (E8SJM) common in CESD (55-80% alleles); "
            "    Liver biopsy: CE-laden hepatocytes; birefringent CE crystals under polarised light; "
            "  TREATMENT: Sebelipase alfa (Kanuma): "
            "    CESD dosing: 1 mg/kg biweekly (Q2W); "
            "    Reduces transaminases, LDL-C, hepatic fat; "
            "    Start EARLY: prevents cirrhosis; "
            "  NLSD (neutral lipid storage disease): differentiate by ATGL/ABHD5 mutations"
        ),
        "disease_category": (
            "LIPA-WOLMAN-ADRENAL-CALCIFICATION-PATHOGNOMONIC-LAL-D-SEBELIPASE-ALFA: "
            "  KEY RULE: neonatal hepatosplenomegaly + bilateral adrenal calcification on imaging = Wolman (LIPA) UNTIL PROVEN OTHERWISE; "
            "  SEBELIPASE ALFA: FDA/EMA 2015; only treatment; Wolman 1 mg/kg/wk; CESD 1 mg/kg Q2W; "
            "  CESD UNDERDIAGNOSED: hepatomegaly + LDL-C elevation + low HDL-C = request LIPA DBS enzyme assay; "
            "  E8SJM SPLICE SITE: c.894G>A; 55-80% of CESD alleles; targeted sequencing sufficient for most; "
            "  LIVER BIOPSY CE CRYSTALS: birefringent under polarised light; pathognomonic for CE storage; "
        ),
    },
    {
        "gene": "ABCA1",
        "protein": (
            "ABCA1 -- 9q31.1 AR -- 2261aa -- ATP-Binding-Cassette-Transporter-A1-"
            "220kDa-Full-Transporter-12-TM-Helices-2-NBDs-Cholesterol-Efflux-ApoA-I-Lipidation-HDL-Biogenesis-"
            "Tangier-Disease-HDL-Near-Zero-PATHOGNOMONIC-Orange-Tonsils-PATHOGNOMONIC-Polyneuropathy-OMIM-600046"
        ),
        "locus": "9q31.1",
        "protein_size": (
            "2261 aa / 220 kDa (ABCA1 -- ATP-binding cassette transporter A1; "
            "FUNCTION: master regulator of cellular cholesterol efflux; HDL biogenesis; "
            "  STRUCTURE: full ABC transporter (not half-transporter); "
            "    2 x transmembrane domain (6 TM helices each); "
            "    2 x nucleotide-binding domain (NBD); "
            "    2 x exocytoplasmic loops (large loops extracellular); "
            "  MECHANISM OF CHOLESTEROL EFFLUX: "
            "    ABCA1 expression induced by cholesterol loading -> via LXR-alpha activation; "
            "    ABCA1 at plasma membrane flips phospholipids/cholesterol to outer membrane leaflet; "
            "    Creates membrane protrusion recognised by lipid-poor apoA-I; "
            "    ApoA-I acquires phospholipid + cholesterol -> nascent HDL (pre-beta HDL disc) formed; "
            "    LCAT (lecithin-cholesterol acyltransferase) esterifies free cholesterol in HDL -> HDL matures; "
            "    Mature HDL: delivers CE to liver via SR-BI (reverse cholesterol transport); "
            "  TANGIER DISEASE MECHANISM: "
            "    ABCA1 LOF -> apoA-I cannot acquire lipid -> apoA-I rapidly catabolised; "
            "    HDL-C: near zero (<0.1 mmol/L) -- PATHOGNOMONIC; "
            "    Lipid-laden macrophages (foam cells) accumulate in tonsils, liver, spleen, peripheral nerves; "
            "    Orange-yellow discolouration of tonsils (lipid-laden macrophages) -- PATHOGNOMONIC physical sign; "
            "  HETEROZYGOUS ABCA1 (FHA): "
            "    HDL-C ~25-40% of normal -> familial hypoalphalipoproteinaemia; "
            "    Increased CVD risk; "
            "encoded 9q31.1; OMIM gene 600046, Tangier disease #205400"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (Tangier) / AUTOSOMAL DOMINANT (FHA, heterozygous): "
            "TANGIER DISEASE (biallelic ABCA1 LOF): "
            "  Prevalence: extremely rare; ~100 cases reported worldwide; "
            "  Named after Tangier Island, Virginia (original family); "
            "  CLINICAL FEATURES: "
            "    ORANGE-YELLOW TONSILS: pathognomonic; enlarged, lobulated, orange-yellow (lipid foam cells); "
            "      Even after tonsillectomy: rectal mucosa, adenoids also orange-tinged; "
            "    HDL-C near zero (<0.1 mmol/L): pathognomonic; "
            "    LDL-C: LOW to low-normal (initially counterintuitive -> less CETP activity without HDL); "
            "    Total cholesterol: often LOW; "
            "    Triglycerides: ELEVATED (reduced clearance via HDL pathway); "
            "    Hepatosplenomegaly: foam cell accumulation; "
            "    Polyneuropathy (relapsing-remitting): predominantly sensory; limb weaknesses episodic; "
            "      TANGIER POLYNEUROPATHY: unusual feature; not in ABCG5/ABCG8 sitosterolaemia; "
            "    Premature CVD: despite low LDL-C (impaired reverse cholesterol transport increases CV risk); "
            "    Corneal opacities: stromal lipid deposition (mild); "
            "    Thrombocytopenia in some patients; "
            "DIAGNOSIS: "
            "  HDL-C near zero + orange tonsils + polyneuropathy -> Tangier disease; "
            "  ABCA1 genetic sequencing (biallelic LOF); "
            "  Cell-based cholesterol efflux assay: absent/minimal efflux; "
            "TREATMENT: "
            "  No specific treatment; symptomatic management; "
            "  Low-fat diet; statin for CVD prevention; "
            "  Tonsillectomy if airway obstruction (but not curative); "
            "  Experimental: ABCA1 upregulators (niacin, fibrates: modestly increase remnant HDL); "
            "  Gene therapy: investigational; "
            "FAMILIAL HYPOALPHALIPOPROTEINAEMIA (FHA, heterozygous ABCA1): "
            "  HDL-C ~25-40% of normal; "
            "  Increased CVD risk (similar to low HDL from other causes); "
            "  Lifestyle modification; statin for LDL-C target"
        ),
        "disease_category": (
            "TANGIER-ABCA1-HDL-NEAR-ZERO-ORANGE-TONSILS-PATHOGNOMONIC-POLYNEUROPATHY: "
            "  KEY RULE: HDL-C <0.1 mmol/L + orange/yellow enlarged tonsils = TANGIER DISEASE (ABCA1); "
            "  ORANGE TONSILS: most memorable sign in all of lipid disorders; lipid-laden macrophages; "
            "  POLYNEUROPATHY: relapsing-remitting; distinguishes from other HDL disorders; "
            "  CVD DESPITE LOW LDL: impaired reverse cholesterol transport -> atherosclerosis risk; "
            "  FHA HETEROZYGOUS: HDL-C 25-40% normal; CVD risk; no tonsil sign; statins for LDL; "
        ),
    },
    {
        "gene": "ABCG5",
        "protein": (
            "ABCG5 -- 2p21 AR -- 651aa -- ATP-Binding-Cassette-Transporter-G5-Sterolin-1-"
            "75kDa-Half-Transporter-6-TM-1-NBD-ABCG5-ABCG8-Obligate-Heterodimer-"
            "Sitosterolaemia-Plant-Sterols-Elevated-Ezetimibe-NPC1L1-OMIM-605459"
        ),
        "locus": "2p21",
        "protein_size": (
            "651 aa / 75 kDa (ABCG5 -- ATP-binding cassette transporter G5, sterolin-1; "
            "FUNCTION: ABC half-transporter; obligate heterodimer with ABCG8 (sterolin-2); "
            "  STRUCTURE: ABCG family (N-terminal NBD + C-terminal TMD; reverse of ABCA/ABCB); "
            "    6 TM helices + 1 NBD; cannot function as homodimer; requires ABCG8; "
            "  ABCG5-ABCG8 HETERODIMER LOCALISATION: "
            "    Intestinal enterocytes: apical membrane; "
            "    Hepatocytes: canalicular membrane; "
            "  FUNCTION OF HETERODIMER: "
            "    Intestinal ABCG5/G8: pumps absorbed plant sterols + cholesterol back into gut lumen; "
            "      Limits plant sterol absorption (<5% of dietary sitosterol absorbed normally; >30% in disease); "
            "    Hepatic ABCG5/G8: secretes sterols into bile for excretion; "
            "  PLANT STEROLS (SUBSTRATES): "
            "    Sitosterol (most common): beta-sitosterol; from plant cell membranes; "
            "    Campesterol; brassicasterol; stigmasterol; "
            "    These cannot be used for steroid hormone synthesis (unlike cholesterol); "
            "    Accumulate in arteries, tendons, skin -> atherosclerosis + xanthomata; "
            "  SITOSTEROLAEMIA MECHANISM: "
            "    ABCG5 or ABCG8 biallelic LOF -> failure to excrete plant sterols -> "
            "      Plasma sitosterol >200 µmol/L (normal <10 µmol/L); "
            "      Premature atherosclerosis + xanthomata (tendon + tuberous) even in CHILDHOOD; "
            "      Haemolytic anaemia (plant sterols incorporate into RBC membranes); "
            "      Stomatocytes on blood film; "
            "encoded 2p21; OMIM gene 605459, sitosterolaemia #210250"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- ABCG5 (or ABCG8) / SITOSTEROLAEMIA: "
            "PREVALENCE: "
            "  Rare; ~100 reported kindreds; higher in East Asian and South Asian populations; "
            "  ABCG5 and ABCG8 mutations cause phenotypically identical disease; "
            "  ABCG5 on 2p21; ABCG8 also on 2p21 (adjacent gene, tail-to-tail orientation); "
            "CLINICAL FEATURES: "
            "  CHILDHOOD TENDON XANTHOMATA: Achilles, patellar tendons in children (even age 2-5); "
            "    DISTINCTIVE: xanthomata at very young age + normal/mildly elevated LDL-C -> "
            "      investigate plant sterols (not just LDL receptor disorders); "
            "  Tuberous xanthomata + xanthelasma; "
            "  Premature atherosclerosis: coronary artery disease even in teenagers; "
            "  Haemolytic anaemia: variable severity; stomatocytes on blood film; "
            "  Elevated LDL-C: usually mild-moderate (not as high as LDLR FH); "
            "DIAGNOSIS: "
            "  PLASMA PLANT STEROLS: sitosterol >200 µmol/L (normal <10); campesterol elevated; "
            "    This test DISTINGUISHES sitosterolaemia from LDLR FH (LDL-C may overlap); "
            "    GC-MS plant sterol panel; "
            "  Genetic: ABCG5 + ABCG8 sequencing (adjacent genes, same panel); "
            "  Stomatocytes on blood film: clue to haemolytic component; "
            "TREATMENT: "
            "  EZETIMIBE: FIRST-LINE; NPC1L1 inhibitor -> blocks absorption of all sterols "
            "    (cholesterol AND plant sterols) from enterocyte; dramatic reduction in plant sterol levels; "
            "    Plasma sitosterol reduction: 50-90% on ezetimibe; "
            "  Low plant sterol diet: "
            "    Avoid: vegetable oils, margarine, nuts, seeds (high sitosterol sources); "
            "    Avoid: shellfish (contain campesterol + brassicasterol); "
            "  Bile acid sequestrants (cholestyramine): second-line; increase biliary sterol excretion; "
            "  AVOID: sitosterolaemia-misdiagnosed-as-FH pitfall; "
            "    Statins: less effective (plant sterols are substrate, not just cholesterol); "
            "  Regular CVD surveillance from childhood"
        ),
        "disease_category": (
            "SITOSTEROLAEMIA-ABCG5-CHILDHOOD-XANTHOMATA-PLASMA-PLANT-STEROLS-EZETIMIBE: "
            "  KEY RULE: childhood tendon xanthomata + mild LDL-C elevation -> measure plasma plant sterols; "
            "  PLASMA SITOSTEROL >200 µmol/L: diagnostic of sitosterolaemia; "
            "  EZETIMIBE FIRST-LINE: blocks NPC1L1; reduces plant sterol absorption 50-90%; "
            "  HAEMOLYTIC ANAEMIA + STOMATOCYTES: clue to RBC membrane plant sterol incorporation; "
            "  ABCG5 AND ABCG8 IDENTICAL PHENOTYPE: always sequence both (adjacent genes, 2p21); "
        ),
    },
    {
        "gene": "ABCG8",
        "protein": (
            "ABCG8 -- 2p21 AR -- 673aa -- ATP-Binding-Cassette-Transporter-G8-Sterolin-2-"
            "77kDa-Half-Transporter-6-TM-1-NBD-ABCG5-ABCG8-Obligate-Heterodimer-Partner-"
            "Sitosterolaemia-Type2-South-Asian-Enriched-ABCG8-D19H-GallStones-Common-Variant-OMIM-605460"
        ),
        "locus": "2p21",
        "protein_size": (
            "673 aa / 77 kDa (ABCG8 -- ATP-binding cassette transporter G8, sterolin-2; "
            "FUNCTION: ABC half-transporter; obligate heterodimer with ABCG5 (sterolin-1); "
            "  GENE LOCATION: 2p21; adjacent to ABCG5 (head-to-head orientation with shared promoter region); "
            "    Co-expressed; co-regulated by LXR-alpha (oxysterol nuclear receptor); "
            "  ABCG8 STRUCTURE: same topology as ABCG5; "
            "    6 TM helices + 1 NBD (N-terminal); "
            "    Walker A + Walker B motifs in NBD; "
            "    ATP hydrolysis provides energy for sterol flipping; "
            "  HETERODIMER FUNCTION: "
            "    Cannot function without partner: ABCG5 alone does not traffic to apical membrane; "
            "    ABCG8 alone: same issue; both required for ER exit and apical trafficking; "
            "    Active transporter only when heterodimer formed; "
            "  CLINICAL NOTES: "
            "    ABCG8 LOF: phenotype identical to ABCG5 sitosterolaemia; "
            "    ABCG8 D19H (p.Asp19His): very common variant (minor allele ~7% Europeans); "
            "      GALLSTONE RISK ASSOCIATION: D19H heterozygotes have increased cholesterol gallstone risk; "
            "      Gallstone association: increased cholesterol secretion into bile; "
            "      NOT causing frank sitosterolaemia (heterozygous carriers: normal plant sterol levels); "
            "    SOUTH ASIAN ABCG8 FOUNDER MUTATIONS: higher frequency of biallelic sitosterolaemia in Indians; "
            "  INHERITED TOGETHER WITH ABCG5: "
            "    Compound heterozygotes possible: one allele ABCG5 + one allele ABCG8 LOF; "
            "    Rare digenic compound het: two different gene mutations; still causes disease; "
            "encoded 2p21; OMIM gene 605460, sitosterolaemia #210250"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE -- ABCG8 / SITOSTEROLAEMIA TYPE 2 (identical to ABCG5 disease): "
            "RELATIONSHIP TO ABCG5: "
            "  Same phenotype; require genetic testing to identify which gene is mutated; "
            "  Treatment identical; "
            "  ABCG8 MORE COMMON IN SOUTH ASIAN POPULATIONS (Indian subcontinent); "
            "  Co-regulated with ABCG5 by LXR-alpha -> both upregulated by dietary cholesterol; "
            "SITOSTEROLAEMIA PHENOTYPE (same as ABCG5): "
            "  Tendon xanthomata in childhood; "
            "  Premature atherosclerosis; "
            "  Haemolytic anaemia; "
            "  Plasma sitosterol >200 µmol/L; campesterol elevated; "
            "  LDL-C mildly-moderately elevated; "
            "ABCG8 D19H COMMON VARIANT AND GALLSTONES: "
            "  D19H: common missense; heterozygotes have mildly increased cholesterol secretion into bile; "
            "  Cholesterol gallstone risk increased ~1.5x in D19H heterozygotes; "
            "  No frank sitosterolaemia; plant sterols normal in heterozygotes; "
            "  Not clinically actionable for lipid management; "
            "DIGENIC SITOSTEROLAEMIA: "
            "  One biallelic hit in ABCG5 AND/OR ABCG8 sufficient (not compound het between genes); "
            "  Rare cases: compound heterozygous with one ABCG5 + one ABCG8 allele mutated -> "
            "    Phenotype may be milder (some residual heterodimer function); "
            "DIAGNOSIS: "
            "  Plasma plant sterol panel (sitosterol, campesterol) + ABCG5 + ABCG8 sequencing; "
            "  Both genes must be sequenced; "
            "TREATMENT: "
            "  Identical to ABCG5 sitosterolaemia: "
            "    Ezetimibe FIRST-LINE; low plant sterol diet; "
            "    Bile acid sequestrants second-line; "
            "    CVD surveillance"
        ),
        "disease_category": (
            "SITOSTEROLAEMIA-ABCG8-IDENTICAL-ABCG5-SOUTH-ASIAN-D19H-GALLSTONES: "
            "  KEY RULE: sitosterolaemia diagnosed -> sequence BOTH ABCG5 and ABCG8 (adjacent 2p21, identical phenotype); "
            "  ABCG8 SOUTH ASIAN ENRICHMENT: higher frequency biallelic mutations in Indian populations; "
            "  D19H COMMON VARIANT: gallstone risk only; no clinical lipid management change needed; "
            "  TREATMENT IDENTICAL TO ABCG5: ezetimibe + low plant sterol diet; "
            "  BOTH GENES CO-REGULATED BY LXR-ALPHA: dietary cholesterol loading induces both; "
        ),
    },
]


def _make_patients(gene_info: dict, n: int, seed: int) -> list:
    """Generate n synthetic patient records for the given gene."""
    rng = random.Random(seed)

    gene = gene_info["gene"]

    # Gene-specific ranges
    ranges = {
        "LDLR":    dict(ldl_lo=4.5,  ldl_hi=12.0, hdl_lo=0.8,  hdl_hi=1.6,  tg_lo=0.8,  tg_hi=3.5, age_lo=12, age_hi=55, cv_pct=0.60),
        "APOB":    dict(ldl_lo=4.2,  ldl_hi=8.0,  hdl_lo=0.9,  hdl_hi=1.8,  tg_lo=0.7,  tg_hi=2.5, age_lo=20, age_hi=60, cv_pct=0.35),
        "PCSK9":   dict(ldl_lo=4.8,  ldl_hi=11.0, hdl_lo=0.8,  hdl_hi=1.5,  tg_lo=0.9,  tg_hi=3.0, age_lo=15, age_hi=55, cv_pct=0.55),
        "LDLRAP1": dict(ldl_lo=9.0,  ldl_hi=20.0, hdl_lo=0.6,  hdl_hi=1.4,  tg_lo=1.0,  tg_hi=3.5, age_lo=5,  age_hi=40, cv_pct=0.65),
        "LIPA":    dict(ldl_lo=3.5,  ldl_hi=8.0,  hdl_lo=0.3,  hdl_hi=1.0,  tg_lo=1.5,  tg_hi=5.0, age_lo=0,  age_hi=20, cv_pct=0.20),
        "ABCA1":   dict(ldl_lo=1.5,  ldl_hi=4.5,  hdl_lo=0.03, hdl_hi=0.15, tg_lo=1.5,  tg_hi=4.5, age_lo=10, age_hi=50, cv_pct=0.45),
        "ABCG5":   dict(ldl_lo=3.0,  ldl_hi=7.0,  hdl_lo=0.8,  hdl_hi=1.6,  tg_lo=0.8,  tg_hi=2.5, age_lo=2,  age_hi=35, cv_pct=0.30),
        "ABCG8":   dict(ldl_lo=3.0,  ldl_hi=7.0,  hdl_lo=0.8,  hdl_hi=1.6,  tg_lo=0.8,  tg_hi=2.5, age_lo=2,  age_hi=35, cv_pct=0.30),
    }
    r = ranges[gene]

    patients = []
    for i in range(n):
        ldl_c = round(rng.uniform(r["ldl_lo"], r["ldl_hi"]), 2)
        hdl_c = round(rng.uniform(r["hdl_lo"], r["hdl_hi"]), 2)
        tg    = round(rng.uniform(r["tg_lo"],  r["tg_hi"]),  2)
        age   = rng.randint(r["age_lo"], r["age_hi"])
        cv    = rng.random() < r["cv_pct"]

        # Gene-specific features
        xanthomata       = rng.random() < (0.55 if gene in ("LDLR","LDLRAP1") else 0.35 if gene == "APOB" else 0.40 if gene in ("ABCG5","ABCG8") else 0.15)
        corneal_arcus    = rng.random() < (0.45 if gene in ("LDLR","APOB","PCSK9") else 0.10)
        orange_tonsils   = (gene == "ABCA1") and (rng.random() < 0.85)
        polyneuropathy   = (gene == "ABCA1") and (rng.random() < 0.55)
        adrenal_calc     = (gene == "LIPA") and (rng.random() < 0.75)
        hepatomegaly     = (gene in ("LIPA","ABCA1")) and (rng.random() < 0.70)
        haemolytic_anaem = (gene in ("ABCG5","ABCG8")) and (rng.random() < 0.40)
        on_pcsk9i        = gene in ("LDLR","PCSK9","LDLRAP1") and rng.random() < 0.40
        on_ezetimibe     = gene in ("ABCG5","ABCG8","LDLR","APOB") and rng.random() < 0.55
        on_sebelipase    = (gene == "LIPA") and (rng.random() < 0.70)

        patients.append({
            "patient_id":           f"{gene}-{SEED_BASE + list(ranges.keys()).index(gene):04d}-{i+1:03d}",
            "gene":                 gene,
            "age_at_diagnosis":     age,
            "ldl_c_mmol_L":         ldl_c,
            "hdl_c_mmol_L":         hdl_c,
            "triglycerides_mmol_L": tg,
            "premature_cvd":        cv,
            "tendon_xanthomata":    xanthomata,
            "corneal_arcus":        corneal_arcus,
            "orange_tonsils":       orange_tonsils,
            "polyneuropathy":       polyneuropathy,
            "adrenal_calcification":adrenal_calc,
            "hepatomegaly":         hepatomegaly,
            "haemolytic_anaemia":   haemolytic_anaem,
            "on_pcsk9_inhibitor":   on_pcsk9i,
            "on_ezetimibe":         on_ezetimibe,
            "on_sebelipase_alfa":   on_sebelipase,
        })
    return patients


def generate_overview() -> dict:
    """Atlas overview: gene summary table, key clinical pearls, epidemiology."""
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        all_patients.extend(_make_patients(g, 40, SEED_BASE + idx))

    n = len(all_patients)
    mean_ldl   = round(sum(p["ldl_c_mmol_L"] for p in all_patients) / n, 2)
    mean_hdl   = round(sum(p["hdl_c_mmol_L"] for p in all_patients) / n, 2)
    mean_tg    = round(sum(p["triglycerides_mmol_L"] for p in all_patients) / n, 2)
    cv_pct     = round(100 * sum(1 for p in all_patients if p["premature_cvd"]) / n, 1)
    xanth_pct  = round(100 * sum(1 for p in all_patients if p["tendon_xanthomata"]) / n, 1)
    ezetimibe_pct = round(100 * sum(1 for p in all_patients if p["on_ezetimibe"]) / n, 1)
    pcsk9i_pct    = round(100 * sum(1 for p in all_patients if p["on_pcsk9_inhibitor"]) / n, 1)

    return {
        "atlas":            "Hereditary-Primary-Dyslipidemia-Atlas",
        "subtitle":         "Complete 8-Gene Familial Hypercholesterolaemia & Dyslipidaemia Reference",
        "genes":            [g["gene"] for g in ATLAS_GENES],
        "gene_count":       len(ATLAS_GENES),
        "total_patients":   n,
        "seeds":            f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "aggregate_stats": {
            "mean_ldl_c_mmol_L":        mean_ldl,
            "mean_hdl_c_mmol_L":        mean_hdl,
            "mean_triglycerides_mmol_L":mean_tg,
            "premature_cvd_pct":        cv_pct,
            "tendon_xanthomata_pct":    xanth_pct,
            "on_ezetimibe_pct":         ezetimibe_pct,
            "on_pcsk9i_pct":            pcsk9i_pct,
        },
        "key_clinical_pearls": [
            "LDLR FH1: most common monogenic disorder (1/250); Achilles tendon xanthomata PATHOGNOMONIC; cascade testing all 1st-degree relatives",
            "APOB FDB: Arg3527Gln European founder; identical phenotype to LDLR FH; responds BETTER to statins (LDLR functional)",
            "PCSK9 GOF (FH3): D374Y statin-resistant; PCSK9i FIRST CHOICE; LOF variants protect against CVD (88% risk reduction Y142X/C679X)",
            "LDLRAP1 ARH: severe FH + AR inheritance + normal lymphocyte LDLR binding -> ARH (not HoFH); partially responds to PCSK9i",
            "LIPA Wolman: bilateral adrenal calcification + neonatal hepatosplenomegaly PATHOGNOMONIC; Sebelipase alfa FDA 2015",
            "ABCA1 Tangier: HDL-C near zero + orange-yellow tonsils PATHOGNOMONIC + relapsing polyneuropathy",
            "ABCG5/ABCG8 sitosterolaemia: childhood tendon xanthomata + elevated plasma sitosterol >200 µmol/L; ezetimibe FIRST-LINE",
            "ABCG5 and ABCG8: adjacent genes 2p21, obligate heterodimer; sequence BOTH when sitosterolaemia suspected",
        ],
        "inheritance_spectrum": {
            "AD_GOF_LOF":             ["PCSK9"],
            "AD_haploinsufficiency":  ["LDLR", "APOB"],
            "AR_complete":            ["LDLRAP1", "LIPA", "ABCA1", "ABCG5", "ABCG8"],
        },
        "treatment_highlights": {
            "PCSK9_inhibitors":       ["LDLR", "APOB", "PCSK9", "LDLRAP1"],
            "Ezetimibe":              ["ABCG5", "ABCG8", "LDLR", "APOB"],
            "Sebelipase_alfa_ERT":    ["LIPA"],
            "LDL_apheresis":          ["LDLR_HoFH", "LDLRAP1"],
            "No_specific_treatment":  ["ABCA1"],
        },
    }


def generate_breakdown() -> dict:
    """Per-gene aggregate statistics (40 patients each)."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts  = _make_patients(g, 40, SEED_BASE + idx)
        n    = len(pts)
        gene = g["gene"]

        mean_ldl     = round(sum(p["ldl_c_mmol_L"] for p in pts) / n, 2)
        mean_hdl     = round(sum(p["hdl_c_mmol_L"] for p in pts) / n, 2)
        mean_tg      = round(sum(p["triglycerides_mmol_L"] for p in pts) / n, 2)
        cv_pct       = round(100 * sum(1 for p in pts if p["premature_cvd"]) / n, 1)
        xanth_pct    = round(100 * sum(1 for p in pts if p["tendon_xanthomata"]) / n, 1)
        corneal_pct  = round(100 * sum(1 for p in pts if p["corneal_arcus"]) / n, 1)
        pcsk9i_pct   = round(100 * sum(1 for p in pts if p["on_pcsk9_inhibitor"]) / n, 1)
        ezet_pct     = round(100 * sum(1 for p in pts if p["on_ezetimibe"]) / n, 1)
        mean_age_dx  = round(sum(p["age_at_diagnosis"] for p in pts) / n, 1)

        extra = {}
        if gene == "ABCA1":
            extra["orange_tonsils_pct"]  = round(100 * sum(1 for p in pts if p["orange_tonsils"]) / n, 1)
            extra["polyneuropathy_pct"]  = round(100 * sum(1 for p in pts if p["polyneuropathy"]) / n, 1)
        if gene == "LIPA":
            extra["adrenal_calc_pct"]    = round(100 * sum(1 for p in pts if p["adrenal_calcification"]) / n, 1)
            extra["hepatomegaly_pct"]    = round(100 * sum(1 for p in pts if p["hepatomegaly"]) / n, 1)
            extra["sebelipase_pct"]      = round(100 * sum(1 for p in pts if p["on_sebelipase_alfa"]) / n, 1)
        if gene in ("ABCG5", "ABCG8"):
            extra["haemolytic_anaemia_pct"] = round(100 * sum(1 for p in pts if p["haemolytic_anaemia"]) / n, 1)

        genes_data.append({
            "gene":                   gene,
            "locus":                  g["locus"],
            "n_patients":             n,
            "mean_age_dx":            mean_age_dx,
            "mean_ldl_c_mmol_L":      mean_ldl,
            "mean_hdl_c_mmol_L":      mean_hdl,
            "mean_triglycerides_mmol_L": mean_tg,
            "premature_cvd_pct":      cv_pct,
            "tendon_xanthomata_pct":  xanth_pct,
            "corneal_arcus_pct":      corneal_pct,
            "on_pcsk9i_pct":          pcsk9i_pct,
            "on_ezetimibe_pct":       ezet_pct,
            "protein_size":           g["protein_size"][:120],
            "inheritance":            g["inheritance"][:120],
            "disease_category":       g["disease_category"][:120],
            **extra,
        })

    return {
        "atlas":   "Hereditary-Primary-Dyslipidemia-Atlas",
        "count":   len(genes_data),
        "genes":   genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions: key concepts, differentials, treatment principles."""
    definitions = [
        {
            "term": "LDLR Class 1-5 Mutation Classification -- FH Allele Functional Consequence",
            "genes": ["LDLR"],
            "definition": (
                "LDLR MUTATION CLASSES (FH ALLELES): "
                "CLASS 1 -- NULL (NO SYNTHESIS): "
                "  Nonsense, frameshift, large deletion -> no LDLR mRNA or truncated; "
                "  No protein produced; "
                "  Statins: minimal benefit on affected allele; "
                "CLASS 2 -- TRANSPORT DEFECTIVE (MOST COMMON ~45%): "
                "  Missense -> misfolded protein -> retained in ER by quality control; "
                "  LDLR degraded before reaching cell surface; "
                "  Statins: upregulate LDLR transcription on WT allele (HeFH); partial benefit; "
                "CLASS 3 -- BINDING DEFECTIVE: "
                "  Ligand-binding domain mutations (LBD repeats 3/4/5); "
                "  LDLR reaches surface but cannot bind LDLR; "
                "  Statins: intermediate benefit; "
                "CLASS 4 -- INTERNALISATION DEFECTIVE: "
                "  NPVY motif mutations in cytoplasmic tail; "
                "  LDLR on surface, binds LDL, but clathrin-coated pit entry impaired; "
                "  LDLRAP1 also required for normal class; "
                "CLASS 5 -- RECYCLING DEFECTIVE: "
                "  EGF precursor domain mutations (beta-propeller region); "
                "  LDLR cannot release LDL in endosome -> degraded with cargo; "
                "  Statins: limited benefit (LDLR internalised but not recycled); "
                "PRACTICAL SIGNIFICANCE: "
                "  Class 2 = ER retention -> BiP/GRP78 molecular chaperone binding; "
                "  Large deletions (MLPA): account for 5-10% of LDLR FH alleles; always perform MLPA; "
                "  >2000 LDLR variants known; ClinVar/FH-registry classification required for VUS management."
            ),
        },
        {
            "term": "PCSK9 Inhibitors Mechanism and Clinical Evidence (FOURIER, ODYSSEY, ORION)",
            "genes": ["PCSK9", "LDLR", "APOB", "LDLRAP1"],
            "definition": (
                "PCSK9 INHIBITORS -- MECHANISM AND EVIDENCE: "
                "MECHANISM: "
                "  PCSK9 normally binds LDLR EGF-A domain at cell surface -> "
                "    LDLR-PCSK9 internalised -> endosome pH 5.0: PCSK9 binds more tightly -> "
                "    LDLR directed to lysosomal degradation rather than recycling; "
                "  PCSK9 inhibitors (mAb): bind circulating PCSK9 -> prevent LDLR binding -> "
                "    LDLR recycled -> more LDLR surface expression -> more LDL clearance; "
                "  LDL-C reduction: 50-60% on background of maximum-tolerated statin; "
                "STATIN + PCSK9i SYNERGY: "
                "  Statin (HMG-CoA reductase inhibitor) -> ↓ intrahepatic cholesterol -> "
                "    Upregulates SREBP2 -> more LDLR transcription + more PCSK9 transcription; "
                "  More LDLR available -> PCSK9i prevents their degradation -> additive effect; "
                "  This is why statin + PCSK9i >> either alone; "
                "FOURIER TRIAL (Evolocumab, n=27,564): "
                "  Background: max statin + ezetimibe; prior CVD; "
                "  LDL-C reduction: 59% (median LDL 0.78 mmol/L achieved); "
                "  Primary endpoint (MACE): 15% relative risk reduction; "
                "  Absolute risk reduction: 1.5% over 2.2 years (NNT ~67); "
                "ODYSSEY OUTCOMES (Alirocumab, n=18,924): "
                "  ACS (recent MI) population; "
                "  LDL-C reduction: 54%; "
                "  MACE: 15% RRR; mortality benefit in highest LDL tertile; "
                "ORION-1/3/10 (Inclisiran siRNA): "
                "  Mechanism: siRNA -> RISC complex -> cleaves PCSK9 mRNA in hepatocyte; "
                "  LDL-C reduction: ~50% sustained at 12 months; "
                "  Dosing: loading dose, 3 months, then every 6 months (2x/year); "
                "  FDA approved 2021; EMA approved 2020; "
                "FH-SPECIFIC USE: "
                "  HeFH: add PCSK9i if LDL-C target not met on statin+ezetimibe; "
                "  FH3 (PCSK9 GOF): PCSK9i specifically neutralises the overactive PCSK9; "
                "  LDLRAP1 (ARH): partially effective (LDLR present on hepatocytes); "
                "  HoFH: evolocumab approved; less effective in null LDLR mutations."
            ),
        },
        {
            "term": "Tangier Disease (ABCA1) -- Orange Tonsils, Near-Zero HDL, Polyneuropathy",
            "genes": ["ABCA1"],
            "definition": (
                "TANGIER DISEASE (ABCA1 BIALLELIC LOF): "
                "PATHOGNOMONIC FEATURES (both required for high specificity): "
                "  1. ORANGE-YELLOW ENLARGED TONSILS: "
                "     Lipid-laden macrophages (foam cells) accumulate in tonsillar tissue; "
                "     Tonsils appear orange-yellow ('butterscotch') on examination; "
                "     Even post-tonsillectomy: rectal mucosa, colorectal mucosa appears orange (sigmoidoscopy); "
                "  2. HDL-C NEAR ZERO (<0.1 mmol/L, often undetectable): "
                "     apoA-I cannot acquire lipid -> rapidly catabolised (t1/2 ~2 days vs 5 days normal); "
                "     apoA-I level <1 mg/dL (near zero); "
                "POLYNEUROPATHY (TANGIER NEUROPATHY): "
                "  Mechanism: lipid accumulation in Schwann cells; "
                "  Pattern 1: multifocal asymmetric neuropathy (relapsing-remitting limb weakness); "
                "  Pattern 2: symmetrical polyneuropathy (sensory > motor); "
                "  DISTINGUISHES from other HDL disorders (CETP deficiency, ApoA-I deficiency - no neuropathy); "
                "  Ptosis + facial weakness in some cases; "
                "LIPID PROFILE (counterintuitive): "
                "  HDL-C: near zero; "
                "  LDL-C: LOW to low-normal (30-80 mg/dL); "
                "    Mechanism: reduced CETP activity (CETP shuttles CE from HDL to LDL; no HDL -> less LDL-CE); "
                "  Total cholesterol: OFTEN LOW; "
                "  Triglycerides: ELEVATED (reduced LPL activation; reduced HDL-mediated TG clearance); "
                "  This lipid pattern is unique: total cholesterol low + TG elevated + HDL=0; "
                "HEPATOSPLENOMEGALY: "
                "  Foam cell accumulation in liver/spleen; "
                "  Mild jaundice possible; "
                "PREMATURE CVD: "
                "  Despite low LDL-C: impaired reverse cholesterol transport -> foam cells in arteries; "
                "  Paradox: atherogenic despite 'favourable' LDL-C; "
                "DIAGNOSIS: "
                "  HDL near zero + orange tonsils -> ABCA1 sequencing; "
                "  Cell cholesterol efflux assay: absent efflux to apoA-I; "
                "TREATMENT: "
                "  No specific therapy; lipid management supportive; "
                "  Low-fat diet; omega-3 for TG; statin for CVD primary prevention (despite low LDL)."
            ),
        },
        {
            "term": "Sitosterolaemia (ABCG5/ABCG8) -- Childhood Xanthomata, Haemolysis, Ezetimibe",
            "genes": ["ABCG5", "ABCG8"],
            "definition": (
                "SITOSTEROLAEMIA (ABCG5 OR ABCG8 BIALLELIC LOF): "
                "PATHOPHYSIOLOGY: "
                "  Normal: dietary plant sterols absorbed ~5%; ABCG5/G8 pumps >95% back into gut/bile; "
                "  Sitosterolaemia: absorption increased 30-50%; excretion abolished -> accumulation; "
                "  Plant sterols (sitosterol, campesterol, brassicasterol): "
                "    Incorporated into cell membranes (partially replace cholesterol); "
                "    Incorporated into atherosclerotic plaques; "
                "    Incorporated into tendons -> xanthomata; "
                "    Incorporated into RBC membranes -> stomatocytes -> haemolysis; "
                "CLINICAL FEATURES: "
                "  CHILDHOOD TENDON XANTHOMATA: "
                "    KEY DIAGNOSTIC CLUE: xanthomata at age 2-10 years with mild LDL-C elevation; "
                "    Achilles tendons + patellar + extensor tendons; "
                "    IMPORTANT DDx: FH (LDLR/APOB) with HoFH; but LDL-C usually lower in sitosterolaemia; "
                "    Measure plasma plant sterols to differentiate; "
                "  PREMATURE ATHEROSCLEROSIS: coronary disease in teens-20s; "
                "  HAEMOLYTIC ANAEMIA: "
                "    Stomatocytes on peripheral blood film; "
                "    Haemoglobin variable; reticulocytosis; "
                "    Splenomegaly (haemolysis); "
                "  LDL-C: mild-moderate elevation (4-8 mmol/L) -- NOT as high as LDLR HoFH; "
                "  Arthralgia (joint accumulation of sterols); "
                "PLASMA PLANT STEROLS (KEY DIAGNOSTIC TEST): "
                "  Sitosterol: >200 µmol/L (normal <10 µmol/L); "
                "  Campesterol: elevated; "
                "  GC-MS or GC-FID method; "
                "  Distinguishes from all LDLR/APOB/PCSK9 FH variants; "
                "GENETIC TESTING: "
                "  Both ABCG5 and ABCG8 must be sequenced (adjacent 2p21; identical phenotype); "
                "TREATMENT: "
                "  EZETIMIBE (FIRST-LINE): "
                "    NPC1L1 inhibitor in enterocyte brush border; "
                "    Blocks absorption of cholesterol AND plant sterols; "
                "    Plasma sitosterol reduction: 50-90% on ezetimibe; "
                "    Dramatic clinical response: xanthomata regress; LDL-C falls; haemolysis reduces; "
                "  LOW PLANT STEROL DIET: "
                "    Avoid: sunflower/soya/corn oils (very high sitosterol); nuts; seeds; "
                "    Avoid: shellfish (campesterol/brassicasterol); "
                "  Bile acid sequestrants: increase biliary sterol excretion (second-line); "
                "STATINS: less effective (primarily reduce cholesterol synthesis; plant sterol pathology separate)."
            ),
        },
        {
            "term": "LIPA (Wolman Disease vs LAL-D/CESD) -- Adrenal Calcification, Sebelipase",
            "genes": ["LIPA"],
            "definition": (
                "LIPA DEFICIENCY SPECTRUM: WOLMAN DISEASE (SEVERE) vs LAL-D/CESD (PARTIAL): "
                "WOLMAN DISEASE (COMPLETE LIPA LOF, <1% RESIDUAL ACTIVITY): "
                "  ONSET: neonatal (weeks 1-4); "
                "  PATHOGNOMONIC: BILATERAL ADRENAL CALCIFICATION: "
                "    CE accumulation in adrenal cortex -> calcification (readily visible on AXR/USS/CT); "
                "    Adrenal insufficiency may occur; "
                "  Clinical: hepatosplenomegaly (massive) + malabsorption + vomiting + diarrhoea; "
                "    Failure to thrive; cachexia; anaemia; "
                "  Death: 3-6 months without ERT; "
                "LAL-D / CESD (PARTIAL LIPA LOF, >1-3% RESIDUAL ACTIVITY): "
                "  ONSET: childhood to adulthood; highly variable; "
                "  Commonest mutation: c.894G>A (E8SJM, exon 8 splice junction mutation); "
                "    Leads to exon 8 skipping + partial read-through -> residual activity; "
                "    Accounts for 55-80% of CESD alleles in European populations; "
                "  Clinical features: "
                "    HEPATOMEGALY (universal): may be the only finding in childhood; "
                "    Hepatic steatosis -> fibrosis -> cirrhosis (late); "
                "    Elevated ALT/AST (often marked); "
                "    Dyslipidaemia: LDL-C elevated + HDL-C LOW; "
                "    Premature atherosclerosis; "
                "    Liver biopsy: CE-laden hepatocytes (birefringent crystals under polarised light); "
                "  MASQUERADES AS NAFLD: major underdiagnosis issue; "
                "    Key clue: hepatomegaly + elevated LDL + low HDL + elevated AST/ALT in child -> test LIPA; "
                "DIAGNOSIS: "
                "  DBS LAL enzyme activity: <0.02 nmol/punch/hr (severely reduced in both); "
                "  Genetic: LIPA sequencing (E8SJM targeted first); "
                "  Liver biopsy (CESD): CE crystals, foamy macrophages, periportal fibrosis; "
                "SEBELIPASE ALFA (KANUMA): "
                "  Recombinant human LAL (Kanuma, Alexion); "
                "  FDA approved December 2015; EMA approved August 2015; "
                "  Wolman: 1 mg/kg IV weekly; escalate to 3 mg/kg if insufficient response; "
                "  CESD: 1 mg/kg IV every 2 weeks; "
                "  Clinical outcomes: "
                "    Wolman: life-saving; survival with ERT vs ~100% mortality untreated by 6 months; "
                "    CESD: normalises transaminases (>80%); reduces hepatic fat + LDL-C + TG; improves HDL; "
                "  Start ERT EARLY in CESD: prevents progression to cirrhosis."
            ),
        },
        {
            "term": "ARH (LDLRAP1) -- Lymphocyte LDLR Binding Test Differentiates from True HoFH",
            "genes": ["LDLRAP1"],
            "definition": (
                "ARH (AUTOSOMAL RECESSIVE HYPERCHOLESTEROLAEMIA) DIFFERENTIAL FROM HoFH: "
                "THE CRITICAL DIAGNOSTIC TEST: LYMPHOCYTE LDL BINDING ASSAY: "
                "  ARH: lymphocyte LDLR binding NORMAL (>50% of normal control); "
                "    WHY: lymphocytes use a DIFFERENT clathrin adaptor (AP2, ARH2) not LDLRAP1; "
                "    LDLRAP1 function is HEPATOCYTE-SPECIFIC for LDLR endocytosis; "
                "    Lymphocyte LDLR is expressed and functional; can bind and internalise LDL; "
                "  HoFH: lymphocyte LDLR binding REDUCED (<20% of normal); "
                "    WHY: LDLR itself is absent/misfolded/absent-from-surface; "
                "    Affects ALL cell types; "
                "CLINICAL DIFFERENTIATION: "
                "  ARH: "
                "    LDL-C: 9-20 mmol/L (between HeFH and true HoFH); "
                "    AR inheritance: both parents phenotypically NORMAL LDL-C (heterozygous LDLRAP1 carriers); "
                "    DISTINGUISHING FEATURE: parents unaffected (unlike HoFH where both parents are HeFH); "
                "    CVD: less severe than true HoFH; aortic stenosis less common; "
                "  True HoFH: "
                "    LDL-C: typically 12-30 mmol/L (higher than ARH); "
                "    Both parents typically have HeFH phenotype (elevated LDL-C); "
                "    Cutaneous xanthomata from very early childhood; "
                "    Aortic root atherosclerosis with stenosis; "
                "TREATMENT RESPONSE DIFFERENCE: "
                "  ARH: STATINS + EZETIMIBE + PCSK9i = PARTIAL RESPONSE; "
                "    LDLR present on hepatocyte surface -> statin upregulates LDLR mRNA; "
                "    Even though endocytosis impaired, some TG-rich lipoprotein uptake occurs via other receptors; "
                "  True HoFH: "
                "    Null mutations: statins minimally effective; "
                "    Residual function mutations: statins help more; "
                "    Lomitapide (MTP inhibitor): FDA approved HoFH; reduces hepatic VLDL secretion; "
                "    LDL apheresis: mainstay for both ARH and HoFH if targets unmet; "
                "SARDINIAN FOUNDER MUTATION: "
                "  ARH is especially prevalent in Sardinia (founder single-exon deletion in LDLRAP1); "
                "  Any Sardinian patient with severe FH + AR pattern -> sequence LDLRAP1 early."
            ),
        },
        {
            "term": "8-Gene Hereditary Primary Dyslipidemia Differential Guide",
            "genes": ["LDLR", "APOB", "PCSK9", "LDLRAP1", "LIPA", "ABCA1", "ABCG5", "ABCG8"],
            "definition": (
                "8-GENE HEREDITARY PRIMARY DYSLIPIDEMIA DIFFERENTIAL: "
                "BY PRIMARY LIPID ABNORMALITY: "
                "  VERY HIGH LDL-C (>9 mmol/L) + AD: LDLR HoFH, PCSK9 GOF D374Y; "
                "  VERY HIGH LDL-C (>9 mmol/L) + AR: LDLRAP1 (ARH), LDLR HoFH biallelic; "
                "  HIGH LDL-C (5-9 mmol/L) + AD: LDLR HeFH (~1/250), APOB FDB, PCSK9 GOF; "
                "  MILD-MODERATE LDL-C (3-7 mmol/L) + childhood xanthomata: ABCG5, ABCG8 (measure plant sterols!); "
                "  LOW LDL-C + NEAR ZERO HDL-C: ABCA1 Tangier disease; "
                "  ELEVATED LDL-C + LOW HDL-C + HEPATOMEGALY + CHILD: LIPA (LAL-D/CESD); "
                "BY KEY PHYSICAL SIGN: "
                "  Achilles tendon xanthomata (FH range LDL-C): LDLR, APOB, PCSK9 GOF, LDLRAP1; "
                "  Childhood tendon xanthomata (milder LDL-C): ABCG5, ABCG8 (measure sitosterol!); "
                "  ORANGE TONSILS: ABCA1 (PATHOGNOMONIC); "
                "  Bilateral adrenal calcification on imaging: LIPA Wolman (PATHOGNOMONIC); "
                "  Corneal arcus <45 years: LDLR, APOB, PCSK9; "
                "  Hepatomegaly in infant/child: LIPA (Wolman/LAL-D); ABCA1 (Tangier); "
                "BY INHERITANCE: "
                "  AD GOF (elevated LDL): PCSK9 D374Y; "
                "  AD LOF: LDLR, APOB, PCSK9; "
                "  AR: LDLRAP1, LIPA, ABCA1, ABCG5, ABCG8; "
                "BY KEY DIAGNOSTIC TEST: "
                "  Lymphocyte LDLR binding: ARH normal, HoFH reduced (differentiates LDLRAP1 from LDLR biallelic); "
                "  Plasma plant sterols (sitosterol >200): ABCG5, ABCG8 (distinguishes from all FH types); "
                "  HDL near zero: ABCA1 (and apoA-I deficiency, LCAT deficiency as differential); "
                "  DBS LAL enzyme activity: LIPA (Wolman/CESD); "
                "BY FIRST-LINE TREATMENT: "
                "  PCSK9i FIRST: PCSK9 GOF FH3; also key for LDLR HeFH/HoFH + LDLRAP1; "
                "  EZETIMIBE FIRST: ABCG5, ABCG8 (dramatic response); also add-on for LDLR; "
                "  SEBELIPASE ALFA: LIPA (Wolman + CESD); only ERT; "
                "  LDL APHERESIS: LDLR HoFH + LDLRAP1 ARH (if targets unmet); "
                "  HIGH-INTENSITY STATIN: LDLR HeFH (backbone); APOB FDB (excellent response); "
                "  NO SPECIFIC TREATMENT: ABCA1 Tangier disease; supportive only; "
                "PRACTICAL ALGORITHM: "
                "  Step 1: Lipid panel + family history; "
                "  Step 2: LDL >4.9 + FH features -> DLCN score; gene panel (LDLR+APOB+PCSK9); "
                "  Step 3: Severe FH + AR pattern -> add LDLRAP1; lymphocyte LDLR binding test; "
                "  Step 4: Childhood xanthomata + LDL-C mild -> plasma plant sterols first; "
                "  Step 5: Orange tonsils + HDL near zero -> ABCA1; "
                "  Step 6: Child + hepatomegaly + elevated LDL + low HDL -> DBS LAL activity; LIPA sequencing."
            ),
        },
        {
            "term": "FH Diagnostic Criteria (DLCN Score, Simon Broome) and Cascade Testing",
            "genes": ["LDLR", "APOB", "PCSK9"],
            "definition": (
                "FH DIAGNOSTIC CRITERIA: "
                "DUTCH LIPID CLINIC NETWORK (DLCN) SCORE: "
                "  Family history: "
                "    1st-degree relative with premature CVD (<55M, <60F) or tendon xanthomata: 1 pt; "
                "    1st-degree relative with LDL-C >95th centile: 1 pt; "
                "    Child <18 with LDL-C >95th centile: 2 pts; "
                "  Clinical history: "
                "    Premature coronary artery disease (<55M, <60F): 2 pts; "
                "    Premature cerebral/peripheral vascular disease (<55M, <60F): 1 pt; "
                "  Physical examination: "
                "    Tendon xanthomata: 6 pts; "
                "    Corneal arcus <45 years: 4 pts; "
                "  LDL-C (untreated): "
                "    >8.5 mmol/L: 8 pts; "
                "    6.5-8.4: 5 pts; "
                "    5.0-6.4: 3 pts; "
                "    4.0-4.9: 1 pt; "
                "  DNA analysis: "
                "    Causative mutation in LDLR/APOB/PCSK9: 8 pts; "
                "  INTERPRETATION: >=8 = definite FH; 6-7 = probable FH; 3-5 = possible FH; <3 = unlikely FH; "
                "SIMON BROOME CRITERIA (UK): "
                "  DEFINITE FH: LDL-C >4.9 mmol/L (adult) + tendon xanthomata or 1st-degree relative with these; "
                "  OR LDL-C elevation + DNA mutation (LDLR/APOB/PCSK9); "
                "CASCADE TESTING (MANDATORY): "
                "  Index case identified -> trace 1st, 2nd degree relatives; "
                "  Genetic-first cascade: identify mutation in proband -> test relatives for SAME mutation; "
                "  Yield: 50% of 1st-degree relatives will be carriers; "
                "  Age: test children from age 2 (if HoFH risk) or from puberty (HeFH); "
                "  Cost-effectiveness: genetic cascade > cholesterol-first cascade (fewer false negatives); "
                "PHARMACOLOGICAL TARGETS: "
                "  EAS/ESC 2019 targets: "
                "    Established ASCVD: LDL-C <1.4 mmol/L AND >=50% reduction; "
                "    High risk (FH with no CVD): LDL-C <1.8 mmol/L; "
                "    Very high risk (FH + CVD): LDL-C <1.4 mmol/L; "
                "STATIN ADHERENCE: "
                "  Major challenge: myalgia in 5-10%; CK >10x ULN = true myopathy (stop statin); "
                "  Statin intolerance: try lower dose, alternate days, different statin, or rosuvastatin; "
                "  Ezetimibe + PCSK9i if statin truly intolerable."
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Primary-Dyslipidemia-Atlas",
        "count":       len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:1200])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    for g in bd["genes"]:
        print(
            f"  {g['gene']:8s}: n={g['n_patients']}, "
            f"LDL-C={g['mean_ldl_c_mmol_L']} mmol/L, "
            f"HDL-C={g['mean_hdl_c_mmol_L']} mmol/L, "
            f"TG={g['mean_triglycerides_mmol_L']} mmol/L, "
            f"CVD={g['premature_cvd_pct']}%, "
            f"xanth={g['tendon_xanthomata_pct']}%, "
            f"mean_age_dx={g['mean_age_dx']}"
        )
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
    for d in df["definitions"]:
        print(f"  - {d['term'][:80]}")
