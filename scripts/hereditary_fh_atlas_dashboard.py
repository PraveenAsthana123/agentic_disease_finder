"""Hereditary Familial Hypercholesterolemia Atlas — 8-Gene Reference
LDLR-APOB-PCSK9-LDLRAP1-ABCG5-ABCG8-LIPA-LPA
320 patients (8 x 40), seeds 2766-2773.
Endpoints: /api/hereditary-fh-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "LDLR",
        "protein": (
            "LDLR -- 19p13.2 AD -- 860aa -- Low-Density-Lipoprotein-Receptor-95kDa-Type-I-Transmembrane-"
            "Glycoprotein-Clathrin-Coated-Pit-Endocytosis-LDL-Binding-EGF-Precursor-Homology-Domain-"
            "OMIM-Gene-606945-Disease-FH1-143890"
        ),
        "locus": "19p13.2",
        "protein_size": "860 aa / 95 kDa (type-I transmembrane glycoprotein; EGF-precursor homology domain; clathrin-coated pit endocytosis)",
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous LOF → HeFH; biallelic → HoFH or compound heterozygous FH; "
            "LDLR (low-density lipoprotein receptor) is the primary clearance receptor for LDL and VLDL remnants; "
            "LDLR binds APOB-100 on LDL surface → clathrin-coated pit endocytosis → lysosomal CE hydrolysis → "
            "LDLR recycled to surface (~100-200 cycles per receptor lifetime); "
            "LDLR LOF → LDL cannot be cleared from plasma → LDL-C elevation; "
            "PREVALENCE: MOST COMMON MONOGENIC FH — 1:200-500 heterozygous globally; "
            "HoFH prevalence: 1:160,000-1,000,000 (depending on consanguinity); "
            "MUTATION CLASSES: "
            "  Class 1 (null, no protein): most severe LDL elevation; "
            "  Class 2 (transport-defective, retained in ER): most common class; "
            "  Class 3 (binding-defective, reaches surface but cannot bind LDL): LDL-C intermediate; "
            "  Class 4 (internalisation-defective, binds LDL but cannot endocytose): intermediate; "
            "  Class 5 (recycling-defective, cannot release LDL in endosome): intermediate; "
            "MUTATION CATALOGUE: >2600 pathogenic LDLR variants documented; "
            "FOUNDER MUTATIONS: p.Trp66Gly (Lebanese founder); p.Asp200Glu (South African Afrikaner founder 1:70); "
            "  p.Val408Met (Finnish founder); del exon 1-8 (French-Canadian founder); "
            "CHOLESTEROL BIOLOGY: LDLR expression SUPPRESSED by high intracellular cholesterol (SREBP2 pathway); "
            "  STATINS: inhibit HMG-CoA reductase → intracellular cholesterol falls → SREBP2 activates → "
            "  LDLR expression UPREGULATED 2-3x → basis of statin mechanism; "
            "  PCSK9 INHIBITORS: prevent PCSK9-mediated LDLR degradation → more LDLR recycled → LDL cleared; "
            "  LDLR-NULL patients: statins/PCSK9i have MINIMAL effect (no receptor to upregulate/recycle)"
        ),
        "disease_category": (
            "FAMILIAL HYPERCHOLESTEROLEMIA TYPE 1 (FH1) — OMIM 143890; "
            "BIOCHEMICAL HALLMARKS: "
            "  HeFH: LDL-C 250-450 mg/dL (adults); LDL-C >160 mg/dL in children; "
            "  HoFH: LDL-C 500-1000+ mg/dL; untreated → childhood CHD (MI in teens); "
            "  TG NORMAL (distinguish from familial hypertriglyceridaemia); "
            "  HDL-C often modestly reduced; "
            "CLINICAL FEATURES: "
            "  TENDON XANTHOMAS (Achilles + extensor tendons of hands/feet) — PATHOGNOMONIC when present; "
            "    presence confirms FH diagnosis without genetic testing; "
            "  CORNEAL ARCUS before age 45 — significant in HeFH; "
            "  XANTHELASMA palpebrarum (periorbital lipid deposits); "
            "  PREMATURE CHD: HeFH — CHD risk 4-10x population; "
            "  HoFH: aortic stenosis (calcific, from childhood) + CHD before age 20; "
            "DUTCH LIPID CLINIC NETWORK SCORE: "
            "  ≥8 points = DEFINITE FH (no genetic test required); "
            "  6-7 = probable FH; 3-5 = possible FH; "
            "  Points: family history (1-2) + clinical history CHD (2) + physical signs (4-6) + LDL-C level (1-8) + DNA (8); "
            "SIMON BROOME CRITERIA (UK): "
            "  Definite FH: TC >7.5 mmol/L (adult) + tendon xanthomas in patient or first/second-degree relative; "
            "  Probable FH: TC >7.5 mmol/L + family history CHD <60 years"
        ),
        "disease_pathway": (
            "LDLR PATHWAY — LDL CLEARANCE CYCLE: "
            "NORMAL: "
            "  1. LDLR synthesised in ER → maturation in Golgi; "
            "  2. LDLR travels to cell surface coated pit via LDLRAP1 (ARH adaptor); "
            "  3. APOB-100 on LDL binds LDLR at plasma pH → coated pit internalisation; "
            "  4. Endosome acidification → LDLR releases LDL → LDLR recycles to surface; "
            "  5. Lysosome: CE hydrolysed → free cholesterol for cell; "
            "  6. PCSK9 (in endosome) binds LDLR → directs to lysosomal degradation (limits LDLR recycling); "
            "LDLR DEFICIENCY: "
            "  LDL remains in plasma → oxidised LDL taken up by macrophages → foam cells → atheromatous plaque; "
            "  STATIN MECHANISM: HMG-CoA reductase inhibition → intracellular cholesterol falls → "
            "    SREBP2 nuclear translocation → LDLR transcription upregulated → more surface LDLR → LDL cleared; "
            "    REQUIRES FUNCTIONAL LDLR — statins ineffective in LDLR-null HoFH; "
            "EZETIMIBE MECHANISM: "
            "  Blocks NPC1L1 intestinal cholesterol absorption → less dietary cholesterol → "
            "  hepatic intracellular cholesterol falls → LDLR upregulated → LDL cleared; "
            "PCSK9i MECHANISM: "
            "  Blocks PCSK9-LDLR interaction → LDLR recycled instead of degraded → "
            "  more surface LDLR → 50-60% additional LDL reduction; "
            "  REQUIRES SOME LDLR — less effective in LDLR-null than Class 3/4/5 mutations; "
            "EVINACUMAB MECHANISM (HoFH): "
            "  Anti-ANGPTL3 mAb → LPL + HL more active → faster VLDL clearance → less IDL→LDL conversion; "
            "  LDLR-INDEPENDENT → works even in LDLR-null; "
            "LDL APHERESIS: "
            "  Physical removal of LDL from plasma every 1-2 weeks; "
            "  reduces LDL 50-70% acutely; HoFH standard of care when medications insufficient"
        ),
        "pathognomonic": (
            "TENDON XANTHOMAS — PATHOGNOMONIC FOR FH: "
            "  Achilles tendons (palpable thickening; confirm with ultrasound — >8mm = abnormal); "
            "  extensor tendons of hands (knuckle xanthomas); "
            "  patellar tendon; Achilles tendon US >8mm bilateral = CONFIRMS FH without genetic test; "
            "  PRESENT IN ~50% HeFH patients (higher in older, more severe, untreated); "
            "  PATHOLOGY: LDL-laden macrophage foam cells in tendon collagen; "
            "CORNEAL ARCUS <45 YEARS: "
            "  white/grey arc around corneal periphery (arcus lipoides); "
            "  in young patient = strong FH indicator; "
            "  NOT pathognomonic alone (common in elderly, African descent); "
            "XANTHELASMA: "
            "  yellowish periorbital plaques; "
            "  associated with FH but also seen in normal-LDL individuals; "
            "  less specific than tendon xanthomas; "
            "DUTCH LIPID CLINIC NETWORK SCORE ≥8 = DEFINITE FH: "
            "  clinical diagnosis without genetic test; "
            "  sensitivity ~80% for LDLR mutations when score ≥8; "
            "HoFH PATHOGNOMONIC: "
            "  LDL >500 mg/dL + childhood tendon xanthomas + aortic valve calcification = HoFH until proven otherwise"
        ),
        "treatment": (
            "HeFH: "
            "1. HIGH-INTENSITY STATIN (first-line): rosuvastatin 20-40mg or atorvastatin 40-80mg/day; "
            "   LDL reduction 40-60%; TARGET: LDL <70 mg/dL (high CVD risk) or <55 mg/dL (very high risk); "
            "2. EZETIMIBE (add-on): + 20% additional LDL reduction; "
            "3. PCSK9 INHIBITORS: evolocumab (Repatha 140mg SC q2w or 420mg monthly) or alirocumab (Praluent); "
            "   FDA 2015; + 50-60% additional LDL reduction on statin; "
            "4. INCLISIRAN (Leqvio FDA2021): siRNA against PCSK9 mRNA; 284mg SC q6 months; "
            "   LDL reduction 50%; hepatic GalNAc delivery; twice-yearly dosing = adherence advantage; "
            "HoFH: "
            "5. EVINACUMAB (Evkeeza FDA2021): anti-ANGPTL3 15mg/kg IV monthly; "
            "   LDLR-INDEPENDENT mechanism; ~47% additional LDL reduction even in LDLR-null; "
            "6. LOMITAPIDE (Juxtapid FDA2012): MTP inhibitor; blocks VLDL assembly in liver; "
            "   HoFH only; hepatotoxicity risk; monthly LFT monitoring; fat-soluble vitamin supplement; "
            "7. LDL APHERESIS: every 1-2 weeks; standard for HoFH; "
            "   reduces LDL 50-70% per session; available in specialised centres; "
            "8. LIVER TRANSPLANT (historical HoFH): replaced by evinacumab/lomitapide era; "
            "CASCADE SCREENING: screen all first-degree relatives; "
            "PEDIATRIC FH: statins from age 8-10 (HeFH); earlier if HoFH; "
            "STATIN INTOLERANCE: switch statin + ezetimibe + PCSK9i; "
            "PREGNANCY: stop statins (teratogenic); cholestyramine (bile acid sequestrant) safe in pregnancy"
        ),
        "key_facts": [
            "MOST COMMON MONOGENIC FH: 1:200-500 heterozygous globally",
            "DUTCH LIPID CLINIC SCORE ≥8 = DEFINITE FH without genetic test",
            "TENDON XANTHOMAS (Achilles) PATHOGNOMONIC for FH",
            "HoFH: LDL >500 mg/dL + childhood CHD + aortic valve calcification",
            ">2600 pathogenic LDLR variants; Class 2 (transport-defective) most common",
            "STATINS INEFFECTIVE in LDLR-null HoFH (no receptor to upregulate)",
            "EVINACUMAB FDA2021: LDLR-independent LDL reduction for HoFH",
            "INCLISIRAN (siRNA PCSK9 inhibitor): dosing every 6 months",
        ],
        "seed": 2766,
    },
    {
        "gene": "APOB",
        "protein": (
            "APOB -- 2p24.1 AD -- 4563aa -- Apolipoprotein-B100-550kDa-LDL-VLDL-Structural-Protein-"
            "LDLR-Ligand-Domain-R3500-Familial-Defective-ApoB-FDB-"
            "OMIM-Gene-107730-Disease-FDB"
        ),
        "locus": "2p24.1",
        "protein_size": "4563 aa / 550 kDa (structural protein of LDL and VLDL; LDLR-ligand binding domain at Arg3500)",
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous mutations in LDLR-ligand domain; "
            "APOB (apolipoprotein B-100) is the obligate structural protein of VLDL, IDL, and LDL; "
            "APOB-100 (liver) vs APOB-48 (intestine, chylomicrons — truncated by APOBEC1 editing); "
            "LDLR-LIGAND DOMAIN: residues 3354-3369 (arginine-rich region near Arg3500); "
            "FDB MUTATIONS: "
            "  p.Arg3500Gln (R3500Q) — MOST COMMON; European founder; prevalence 1:700-1000 in Europeans; "
            "    R3500Q alters LDLR-binding domain → LDL cannot bind LDLR → LDL accumulates; "
            "    1-2% of all clinically diagnosed FH presentations; "
            "  p.Arg3500Trp (R3500W) — more severe; R3500W rarer; "
            "  p.Arg3531Cys — milder effect on LDLR binding; "
            "WHY FDB IS MILDER THAN LDLR-FH: "
            "  HEPATIC LDLR IS INTACT → hepatic LDLR removes VLDL-derived particles normally; "
            "  only LDL particles (bearing defective APOB) are poorly cleared; "
            "  VLDL (bearing intact APOE for LDLR binding) cleared normally; "
            "  → LDL-C in FDB: 250-350 mg/dL (lower than LDLR-FH 350-450 mg/dL); "
            "PCSK9i IN FDB: VERY EFFECTIVE "
            "  PCSK9i increases LDLR recycling → even with defective APOB-LDL, "
            "  higher LDLR density compensates partially; plus normal VLDL/IDL cleared better; "
            "  LDL-C reduction with PCSK9i in FDB often exceeds reduction in LDLR-FH"
        ),
        "disease_category": (
            "FAMILIAL DEFECTIVE ApoB-100 (FDB) — LDLR-ligand domain defect; "
            "BIOCHEMICAL HALLMARKS: "
            "  LDL-C 250-350 mg/dL (adults); milder than LDLR-FH; "
            "  TG NORMAL; HDL-C normal to mildly reduced; "
            "  FUNCTIONAL ASSAY: LDL-LDLR binding affinity REDUCED (20-30% of normal); "
            "    measured by competitive binding of patient LDL vs normal LDL; "
            "  GENETIC TEST: p.Arg3500Gln on targeted sequencing or FH gene panel; "
            "CLINICAL FEATURES: "
            "  TENDON XANTHOMAS LESS FREQUENT than LDLR-FH (lower LDL levels); "
            "  corneal arcus before age 45; xanthelasma; "
            "  PREMATURE CHD: risk elevated but lower than LDLR-FH due to lower LDL-C; "
            "  phenotype may overlap with polygenic hypercholesterolaemia; "
            "DDx FROM LDLR-FH: "
            "  statins less effective in LDLR-null LDLR-FH than in FDB (LDLR intact in FDB); "
            "  PCSK9i MORE effective in FDB (intact LDLR recycled more efficiently); "
            "  genetic test definitively distinguishes; "
            "  functional LDLR assay (fibroblast) normal in FDB"
        ),
        "disease_pathway": (
            "APOB-100 STRUCTURE AND LDL CLEARANCE FAILURE IN FDB: "
            "NORMAL: "
            "  Liver assembles VLDL: ApoB-100 + TG + CE + phospholipids → VLDL secretion; "
            "  VLDL lipolysis by LPL → IDL → further LDLR/HL-mediated → LDL; "
            "  LDL APOB-100 Arg3500 region binds LDLR → endocytosis → LDL cleared; "
            "FDB: "
            "  VLDL assembled normally; lipolysis to IDL/LDL proceeds normally; "
            "  LDL-APOB-100 R3500Q → LDLR-ligand domain altered → poor LDLR binding; "
            "  LDL accumulates in plasma (cannot be cleared efficiently); "
            "  BUT: VLDL and IDL (which use APOE for LDLR binding, not APOB-R3500) cleared NORMALLY; "
            "  → selective accumulation of LDL (not IDL/VLDL); "
            "PCSK9 INTERACTION: "
            "  PCSK9 degrades LDLR in lysosome (reduces LDLR recycling); "
            "  PCSK9i blocks this → more LDLR at surface; "
            "  In FDB: more LDLR + defective APOB-LDL → even partially-defective binding can clear some LDL; "
            "  therefore PCSK9i particularly effective in FDB (functional receptor recycled, higher density compensates); "
            "APOB TRUNCATION MUTATIONS: "
            "  Familial hypobetalipoproteinemia (FHBL): truncating APOB mutations → low LDL; "
            "  Complete APOB deficiency (abetalipoproteinaemia): MTTP mutations (not APOB); "
            "  FDB is distinct (LDLR-ligand domain missense, not truncation)"
        ),
        "pathognomonic": (
            "p.Arg3500Gln (R3500Q) ON APOB SEQUENCING — FOUNDATIONAL DIAGNOSTIC: "
            "  R3500Q is detectable on targeted APOB sequencing or commercial FH gene panels; "
            "  European population frequency ~1:700 (relatively common for a FH mutation); "
            "  CLINICAL CLUE: DUTCH LIPID CLINIC SCORE 6-7 (probable FH, not definite) "
            "    → LDL-C slightly lower than expected for LDLR-FH → FDB should be in DDx; "
            "LDL-LDLR BINDING ASSAY (research): "
            "  Patient LDL competes with fluorescent normal LDL for LDLR binding → "
            "  FDB LDL: 20-30% normal binding affinity; "
            "  LDLR-FH (null): LDL binding NORMAL (problem is LDLR, not LDL); "
            "PCSK9i SUPERIOR RESPONSE IN FDB: "
            "  LDL-C falls >50% on PCSK9i in FDB vs typically 40-50% in LDLR-FH; "
            "  superior response + genetic test = confirms FDB; "
            "TENDON XANTHOMA LESS LIKELY: "
            "  FDB: LDL 250-350 → tendon xanthomas in ~30% (vs ~50% in LDLR-HeFH); "
            "  absence of tendon xanthomas with moderate LDL elevation → consider FDB"
        ),
        "treatment": (
            "1. HIGH-INTENSITY STATIN (first-line): rosuvastatin 20-40mg or atorvastatin 40-80mg; "
            "   LDL-C reduction 40-55%; MORE EFFECTIVE than in LDLR-null FH (LDLR intact in FDB → statin upregulates it); "
            "2. EZETIMIBE: additional 20% LDL-C reduction; "
            "3. PCSK9 INHIBITORS — PARTICULARLY EFFECTIVE IN FDB: "
            "   evolocumab 140mg SC q2w or 420mg monthly; alirocumab; "
            "   LDL-C reduction often 50-65% on PCSK9i; "
            "   mechanism: more LDLR recycling → partially compensates for defective APOB-LDL binding; "
            "4. INCLISIRAN: siRNA against PCSK9; 284mg SC q6 months; "
            "5. LDL APHERESIS: rarely needed (LDL usually lower than HoFH); "
            "   consider if LDL >300 mg/dL refractory + high CVD risk; "
            "6. CASCADE SCREENING: screen all first-degree relatives for R3500Q; "
            "7. TARGET LDL-C: <70 mg/dL (high CVD risk) or <55 mg/dL (established CVD); "
            "8. STATIN SAFETY: LFT + CK at baseline; rhabdomyolysis risk <0.1%; "
            "PREGNANCY: statins teratogenic (stop at conception) → cholestyramine safe alternative; "
            "CHILDREN: consider statin from age 8-10 if LDL persistently >190 mg/dL"
        ),
        "key_facts": [
            "FDB: p.Arg3500Gln (R3500Q) European founder 1:700-1000",
            "FDB MILDER than LDLR-FH: LDL 250-350 mg/dL (LDLR intact, clears VLDL normally)",
            "PCSK9i HIGHLY EFFECTIVE in FDB: intact LDLR recycled → compensates defective APOB",
            "Tendon xanthomas LESS FREQUENT in FDB than LDLR-FH",
            "Genetic test distinguishes FDB from LDLR-FH — important for PCSK9i decision",
            "APOB-48 (intestinal, chylomicrons) NOT affected in FDB — hepatic APOB-100 only",
            "Statins MORE effective in FDB than LDLR-null (functional LDLR upregulated by statin)",
        ],
        "seed": 2767,
    },
    {
        "gene": "PCSK9",
        "protein": (
            "PCSK9 -- 1p32.3 AD-GOF(FH3) / LOF-protective -- 692aa -- Proprotein-Convertase-Subtilisin-Kexin-9-"
            "72kDa-Serine-Protease-LDLR-Degradation-Target-Evolocumab-Alirocumab-Inclisiran-"
            "OMIM-Gene-607786-Disease-FH3-603776"
        ),
        "locus": "1p32.3",
        "protein_size": "692 aa / 72 kDa (serine protease; autocatalytic cleavage; prodomain-bound mature form secreted)",
        "inheritance": (
            "AD GAIN-OF-FUNCTION → FH3 (FH type 3); "
            "HETEROZYGOUS LOF → PROTECTIVE (reduced CVD); "
            "PCSK9 MECHANISM: "
            "  PCSK9 secreted by liver → binds LDLR at cell surface (EGF-A domain interaction) → "
            "  PCSK9-LDLR complex endocytosed → in acidic endosome, PCSK9 retains LDLR binding → "
            "  directs LDLR to lysosomal degradation (instead of recycling); "
            "  NORMAL PCSK9 therefore LIMITS LDLR recycling (reduces LDLR density); "
            "GOF MUTATIONS — FH3: "
            "  D374Y (Norwegian founder) — MOST SEVERE; dramatically increased LDLR binding; "
            "    LDL-C 400-800 mg/dL in heterozygotes; aortic stenosis in childhood; "
            "  S127R — severe; "
            "  F216L — moderate; "
            "  R218S — milder; "
            "PCSK9 LOF MUTATIONS — PROTECTIVE: "
            "  R46L (European, ~2%): LDL 15-28% lower; CVD risk 28% lower; "
            "  Y142X + C679X (African-American): LDL 40% lower; CVD risk 88% lower; "
            "  COMPLETE LOF (rare homozygous): LDL 50-100 mg/dL; essentially zero CVD; "
            "  VALIDATES PCSK9 INHIBITION as safe therapeutic target; "
            "THERAPEUTIC PCSK9 INHIBITORS: "
            "  Evolocumab (Repatha FDA2015): mAb; 140mg SC q2w or 420mg monthly; "
            "  Alirocumab (Praluent FDA2015): mAb; 75-150mg SC q2w; "
            "  Inclisiran (Leqvio FDA2021): siRNA against PCSK9 mRNA; 284mg SC q6 months; "
            "    GalNAc-conjugated → hepatic NTLA delivery; suppresses PCSK9 mRNA → no PCSK9 secreted; "
            "    twice-yearly dosing is major adherence advantage"
        ),
        "disease_category": (
            "FAMILIAL HYPERCHOLESTEROLEMIA TYPE 3 (FH3) — GOF mutations; OMIM 603776; "
            "BIOCHEMICAL HALLMARKS: "
            "  LDL-C 300-500+ mg/dL in GOF heterozygotes; "
            "  D374Y heterozygotes: LDL 400-800 mg/dL (approaches HoFH levels); "
            "  D374Y HOMOZYGOUS: LDL >900 mg/dL — most severe known PCSK9 GOF phenotype; "
            "  TG NORMAL; HDL normal to low; "
            "CLINICAL FEATURES: "
            "  TENDON XANTHOMAS: present in severe GOF (especially D374Y); "
            "  AORTIC VALVE STENOSIS: D374Y → calcific aortic stenosis in childhood/teens; "
            "  PREMATURE CHD: especially with D374Y; "
            "  CORNEAL ARCUS, XANTHELASMA: as in LDLR-FH; "
            "PCSK9 GOF vs LDLR-FH DISTINCTION: "
            "  LDLR expression and structure NORMAL in PCSK9 GOF; "
            "  fibroblast LDLR activity: REDUCED (GOF PCSK9 degrades it) but not absent; "
            "  PCSK9i VERY EFFECTIVE in GOF (blocks the pathological PCSK9 → LDLR restored); "
            "  STATINS EFFECTIVE (LDLR intact — statin upregulates LDLR → GOF PCSK9 degrades it, "
            "    but net effect still some LDL-C reduction; add PCSK9i for full benefit)"
        ),
        "disease_pathway": (
            "PCSK9 — LDLR DEGRADATION PATHWAY: "
            "NORMAL PCSK9 CYCLE: "
            "  1. PCSK9 synthesised as 75kDa precursor in ER; "
            "  2. Autocatalytic cleavage of prodomain in ER → mature PCSK9-prodomain complex secreted; "
            "  3. Circulating PCSK9 binds LDLR EGF-A domain (extracellular) at neutral pH; "
            "  4. PCSK9-LDLR endocytosed in clathrin-coated pit; "
            "  5. Endosomal acidification: NORMAL LDLR releases ligand → LDLR recycles; "
            "     PCSK9 BOUND: PCSK9 retains LDLR binding at acid pH → LDLR cannot disengage → "
            "     directed to lysosomal degradation; "
            "  6. PCSK9 acts as a recyclable LDLR destabiliser (PCSK9 itself may recycle); "
            "PCSK9 GOF: "
            "  D374Y: Asp374→Tyr in EGF-A domain binding → tighter LDLR binding → "
            "    more LDLR directed to lysosome → LDLR density at surface falls dramatically; "
            "PCSK9 LOF (NATURAL HUMAN EXPERIMENT): "
            "  R46L, Y142X, C679X → PCSK9 absent or inactive → LDLR not degraded → "
            "  more LDLR recycled → LDL cleared more efficiently → lower LDL and CVD; "
            "INCLISIRAN MECHANISM: "
            "  siRNA delivered to hepatocytes (GalNAc-NTLA) → cleaves PCSK9 mRNA → "
            "  no PCSK9 synthesised → LDLR constitutively recycled → sustained LDL reduction; "
            "  duration: 6 months per dose (PCSK9 mRNA continuously degraded); "
            "EVOLOCUMAB/ALIROCUMAB MECHANISM: "
            "  Monoclonal IgG binds circulating PCSK9 → PCSK9 cannot bind LDLR → "
            "  LDLR recycled normally → surface LDLR density increases 2-3x → LDL cleared"
        ),
        "pathognomonic": (
            "D374Y NORWEGIAN FOUNDER — MOST SEVERE PCSK9 GOF: "
            "  LDL-C 400-800 mg/dL in heterozygotes; aortic stenosis in teens; "
            "  Norwegian/Scandinavian ancestry clue; "
            "  genetic confirmation (PCSK9 sequencing); "
            "DRAMATIC PCSK9i RESPONSE IN FH3: "
            "  Evolocumab/alirocumab: LDL falls 60-70% in PCSK9 GOF (more than in LDLR-FH) "
            "  because the causal mutation is directly blocked by PCSK9i; "
            "  → PCSK9i is MECHANISM-SPECIFIC TREATMENT for FH3; "
            "PCSK9 SERUM LEVEL (research/clinical trial use): "
            "  PCSK9 measurable in blood; normal 100-300 ng/mL; "
            "  GOF: PCSK9 may be elevated or normal (depends on variant); "
            "  LOF: PCSK9 low or absent; "
            "  After inclisiran: PCSK9 undetectable (mRNA suppressed); "
            "PCSK9 LOF — CLINICAL PEARL: "
            "  Patient with LDL unexpectedly low (<70 mg/dL untreated) + normal APOB + no medication → "
            "  consider PCSK9 LOF variant; protective; no treatment needed; "
            "  R46L (European) detectable on lipid gene panels"
        ),
        "treatment": (
            "FH3 (PCSK9 GOF): "
            "1. PCSK9 INHIBITORS — MECHANISM-SPECIFIC, HIGHLY EFFECTIVE: "
            "   evolocumab 140mg SC q2w or 420mg monthly; alirocumab 75-150mg SC q2w; "
            "   LDL-C reduction 60-70% in PCSK9 GOF (blocks the causal molecular defect); "
            "2. INCLISIRAN (siRNA): 284mg SC at day 1, month 3, then every 6 months; "
            "   hepatic GalNAc delivery → suppresses PCSK9 mRNA; "
            "   LDL reduction 50-60%; twice-yearly adherence advantage; "
            "3. HIGH-INTENSITY STATIN: rosuvastatin 20-40mg; atorvastatin 40-80mg; "
            "   upregulates LDLR (but GOF PCSK9 degrades upregulated LDLR → statin less effective alone); "
            "   ADD PCSK9i to statin for synergy; "
            "4. EZETIMIBE: +15-20% additional LDL reduction; "
            "5. LDL APHERESIS: severe GOF (D374Y HoFH) + inadequate drug response; "
            "6. EVINACUMAB: HoFH context if PCSK9i + statins + apheresis insufficient; "
            "PCSK9 LOF (protective variants): "
            "   No treatment needed; beneficial phenotype; inform patient of protective status; "
            "CASCADE SCREENING: screen first-degree relatives for GOF mutations; "
            "TARGET LDL-C: <70 mg/dL (high CVD risk); <55 mg/dL (very high/established CVD)"
        ),
        "key_facts": [
            "FH3: PCSK9 GOF mutations; D374Y Norwegian founder most severe",
            "PCSK9 LOF (R46L, Y142X, C679X) PROTECTIVE: lower LDL + 28-88% less CVD",
            "PCSK9i (evolocumab/alirocumab FDA2015) mechanism-specific for FH3",
            "INCLISIRAN (siRNA FDA2021): twice-yearly dosing; suppresses PCSK9 mRNA",
            "D374Y: LDL 400-800 mg/dL heterozygous; aortic stenosis in childhood",
            "PCSK9 inhibition idea came from human LOF genetics — natural Mendelian experiment",
            "Statins upregulate LDLR but GOF PCSK9 degrades it — add PCSK9i for full effect",
        ],
        "seed": 2768,
    },
    {
        "gene": "LDLRAP1",
        "protein": (
            "LDLRAP1 -- 1p36.11 AR -- 308aa -- LDL-Receptor-Adaptor-Protein-1-34kDa-"
            "PTB-Domain-Clathrin-Coated-Pit-LDLR-Internalisation-Hepatocyte-Specific-"
            "OMIM-Gene-605747-Disease-ARH-603813"
        ),
        "locus": "1p36.11",
        "protein_size": "308 aa / 34 kDa (phosphotyrosine-binding domain; clathrin-coated pit adaptor; hepatocyte-specific expression)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "LDLRAP1 (LDL receptor adaptor protein 1, also called ARH) links the LDLR cytoplasmic tail "
            "to clathrin-coated pit machinery for endocytosis; "
            "LDLRAP1 MECHANISM: "
            "  LDLRAP1 PTB domain binds NPVY motif on LDLR cytoplasmic tail; "
            "  LDLRAP1 also binds clathrin heavy chain + AP-2 adaptor complex; "
            "  Bridge function: LDLR → LDLRAP1 → clathrin-coated pit → endocytosis; "
            "  WITHOUT LDLRAP1: LDLR is expressed at cell surface but CANNOT be internalised efficiently; "
            "CRITICAL FEATURE: HEPATOCYTE-SPECIFIC REQUIREMENT: "
            "  LDLRAP1 is required for LDLR internalisation in HEPATOCYTES specifically; "
            "  In FIBROBLASTS: LDLR internalisation proceeds via LDL ADAPTOR PROTEIN β (β2-adaptin) → "
            "    LDLR functional in fibroblasts despite LDLRAP1 absence; "
            "  In HEPATOCYTES: β2-adaptin cannot substitute → LDLR non-functional; "
            "  THEREFORE: fibroblast LDLR assay NORMAL in ARH → falsely reassuring; "
            "FOUNDER MUTATIONS: "
            "  p.Trp22Ter (Sardinian founder) — most characterised; "
            "  p.Tyr72Ter (Saudi Arabian); "
            "  PREVALENCE: rare — <100 families described; most in consanguineous populations; "
            "PHENOTYPE SEVERITY: "
            "  ARH: LDL 400-700 mg/dL in homozygotes — SIMILAR TO HoFH but MILDER; "
            "  statins PARTIALLY EFFECTIVE (unlike LDLR-null HoFH where statins nearly useless); "
            "  mechanism: statins upregulate LDLR → surface LDLR present → some alternative internalisation; "
            "  PCSK9i SUBOPTIMAL (LDLR not cycling normally even without PCSK9 degrading it)"
        ),
        "disease_category": (
            "AUTOSOMAL RECESSIVE HYPERCHOLESTEROLEMIA (ARH) — OMIM 603813; "
            "BIOCHEMICAL HALLMARKS: "
            "  LDL-C 400-700 mg/dL (homozygotes); similar to HoFH but typically milder; "
            "  TG normal; HDL normal to mildly reduced; "
            "  LDLR EXPRESSION NORMAL (LDLR protein present on cell surface — not degraded); "
            "  LDLR FUNCTION IMPAIRED SELECTIVELY IN HEPATOCYTES (not fibroblasts); "
            "CLINICAL FEATURES: "
            "  TENDON XANTHOMAS: present (Achilles, extensor); "
            "  premature CHD: typically in 3rd-4th decade (later than LDLR-null HoFH); "
            "  aortic valve involvement possible but less severe than LDLR-null HoFH; "
            "  xanthelasma; corneal arcus; "
            "KEY DDx FROM LDLR-HoFH: "
            "  ARH statins PARTIALLY EFFECTIVE → suggests residual LDLR cycling (implies LDLRAP1 not LDLR defect); "
            "  Fibroblast LDLR FUNCTIONAL → rules out LDLR LOF mutation; "
            "  LDL 400-700 mg/dL (not 500-1000+ of severe LDLR-null HoFH); "
            "  parents heterozygous (AR pattern) — not dominant FH; "
            "  genetic test confirms LDLRAP1 biallelic mutations"
        ),
        "disease_pathway": (
            "LDLRAP1 — LDLR INTERNALISATION PATHWAY IN HEPATOCYTES: "
            "NORMAL HEPATIC LDL CLEARANCE: "
            "  1. LDL in plasma → APOB-100 binds hepatic LDLR; "
            "  2. LDLR cytoplasmic NPVY motif → LDLRAP1 PTB domain binding; "
            "  3. LDLRAP1 recruits clathrin + AP-2 → coated pit formation → LDLR+LDL internalised; "
            "  4. Endosome acidification → LDL released → LDLR recycled to surface; "
            "ARH (LDLRAP1 DEFICIENCY): "
            "  LDLR at surface: present and can bind LDL; "
            "  Clathrin-coated pit connection severed (LDLRAP1 missing); "
            "  LDL-LDLR complex trapped at surface → LDLR accumulates at surface without internalising; "
            "  → plasma LDL cannot be cleared (hepatic LDLR non-functional in vivo); "
            "  FIBROBLAST EXCEPTION: β2-adaptin (AP-2 subunit) can substitute for LDLRAP1 in fibroblasts; "
            "    → fibroblast LDLR WORKS despite ARH → fibroblast assay MISLEADINGLY NORMAL; "
            "STATIN PARTIAL EFFECTIVENESS: "
            "  Statins → LDLR transcription increased; "
            "  More surface LDLR → even with poor LDLRAP1-independent internalisation, "
            "    some LDL clearance occurs via alternative endocytosis routes; "
            "  Result: statins reduce LDL 20-30% in ARH (vs near-zero in LDLR-null HoFH); "
            "PCSK9i IN ARH: "
            "  PCSK9i prevents LDLR lysosomal degradation → more LDLR recycled; "
            "  BUT: LDLR internalisation itself impaired (LDLRAP1 absent) → benefit blunted; "
            "  Partial LDL-C reduction with PCSK9i in ARH"
        ),
        "pathognomonic": (
            "FIBROBLAST LDLR FUNCTIONAL + HIGH LDL-C IN HoFH-LIKE PRESENTATION: "
            "  Fibroblast LDLR assay (gold standard for LDLR function) → NORMAL in ARH; "
            "  clinical phenotype resembles HoFH → fibroblast assay normal → suspect ARH; "
            "  NEXT STEP: LDLRAP1 sequencing; "
            "STATINS PARTIALLY EFFECTIVE (unlike LDLR-null HoFH): "
            "  ARH patients: LDL falls 20-35% on high-intensity statin; "
            "  LDLR-null HoFH: LDL barely changes with statin; "
            "  partial statin response + HoFH-range LDL = ARH clue; "
            "AUTOSOMAL RECESSIVE INHERITANCE: "
            "  both parents have NORMAL LDL-C (heterozygous LDLRAP1 carriers: LDL normal or marginally elevated); "
            "  contrast LDLR-HoFH or PCSK9 GOF (compound heterozygous parents have elevated LDL); "
            "  consanguineous kindred (Sardinian, Saudi, Lebanese) with severe early childhood hypercholesterolaemia → ARH; "
            "SARDINIAN FOUNDER p.Trp22Ter: "
            "  most documented ARH families from Sardinia; "
            "  sequence exon 1 of LDLRAP1 for p.Trp22Ter as first targeted test"
        ),
        "treatment": (
            "1. HIGH-INTENSITY STATIN: partially effective (20-35% LDL reduction); "
            "   rosuvastatin 40mg or atorvastatin 80mg — FIRST LINE; "
            "   mechanism: upregulates LDLR → some alternative internalisation; "
            "2. EZETIMIBE: +15% additional LDL-C reduction; "
            "3. PCSK9 INHIBITORS (SUBOPTIMAL but clinically used): "
            "   evolocumab or alirocumab; LDL reduction ~30-40% (blunted vs LDLR-intact FH); "
            "   add to statin + ezetimibe for combination; "
            "4. LDL APHERESIS — OFTEN REQUIRED: "
            "   LDL 400-700 mg/dL not adequately controlled by drugs; "
            "   every 1-2 weeks; effective; reduces LDL 50-70% per session; "
            "5. LOMITAPIDE (HoFH-like indication in severe ARH): "
            "   MTP inhibitor; reduces VLDL assembly → less LDL production; "
            "   hepatotoxicity monitoring; "
            "6. EVINACUMAB (anti-ANGPTL3): "
            "   LDLR-independent mechanism → effective in ARH (bypasses LDLRAP1 defect); "
            "7. LIVER TRANSPLANT: curative in principle (restores hepatic LDLRAP1); "
            "   now superseded by apheresis + novel therapies; "
            "CASCADE SCREENING: AR — screen siblings (25% risk of homozygosity in consanguineous); "
            "   screen parents as obligate heterozygous carriers"
        ),
        "key_facts": [
            "ARH: LDLR EXPRESSED NORMALLY but cannot be internalised in hepatocytes",
            "FIBROBLAST LDLR NORMAL — misleadingly normal in ARH (LDLRAP1 not needed in fibroblasts)",
            "STATINS PARTIALLY EFFECTIVE in ARH (unlike LDLR-null HoFH where statins nearly useless)",
            "PCSK9i SUBOPTIMAL in ARH (LDLR not cycling normally)",
            "Sardinian founder p.Trp22Ter; Saudi p.Tyr72Ter",
            "AR inheritance: parents have NORMAL LDL (contrast dominant FH)",
            "LDL apheresis usually required; evinacumab LDLR-independent → effective",
        ],
        "seed": 2769,
    },
    {
        "gene": "ABCG5",
        "protein": (
            "ABCG5 -- 2p21 AR -- 651aa -- ATP-Binding-Cassette-Sub-Family-G-Member-5-Sterolin-1-75kDa-"
            "Obligate-Heterodimer-ABCG8-Intestinal-Apical-Hepatic-Canalicular-Plant-Sterol-Efflux-"
            "OMIM-Gene-605459-Disease-Sitosterolemia-210250"
        ),
        "locus": "2p21",
        "protein_size": "651 aa / 75 kDa (half-transporter; obligate heterodimer with ABCG8; 6 TM domains)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in ABCG5 (Sterolin-1); "
            "ABCG5 MECHANISM: "
            "  ABCG5 forms obligate heterodimer with ABCG8 (both required for functional transporter); "
            "  ABCG5/ABCG8 heterodimer localised at: "
            "    (1) intestinal apical brush border membrane — effluxes absorbed plant sterols BACK into lumen; "
            "    (2) hepatic canalicular membrane — effluxes plant sterols into bile for faecal excretion; "
            "  NORMAL FUNCTION: plant sterols absorbed from diet → ABCG5/G8 rapidly effluxes them → "
            "    net absorption <5% of dietary plant sterols; "
            "  ABCG5 LOF → plant sterol efflux blocked → "
            "    sitosterol, campesterol, stigmasterol accumulate in blood + tissues; "
            "BIOCHEMICAL HALLMARKS OF SITOSTEROLEMIA: "
            "  plasma sitosterol: NORMAL <0.5 mg/dL → SITOSTEROLEMIA 15-40 mg/dL (>30-fold); "
            "  plasma campesterol elevated; total cholesterol may be normal, elevated, or paradoxically normal; "
            "PREVALENCE: very rare — ~100-200 families described; consanguineous populations; "
            "FOUNDER VARIANTS: "
            "  ABCG5 p.Gln604Ter (Q604X) — Amish founder; "
            "  various missense and truncating mutations across gene; "
            "IMPORTANT: ABCG5 and ABCG8 GENES are head-to-head on chromosome 2p21 with shared promoter; "
            "  ABCG5 encodes Sterolin-1; ABCG8 encodes Sterolin-2; "
            "  both must be present for any functional ABCG5/G8 transporter at either location"
        ),
        "disease_category": (
            "SITOSTEROLEMIA (PHYTOSTEROLEMIA) TYPE A — OMIM 210250; "
            "BIOCHEMICAL HALLMARKS: "
            "  PLASMA SITOSTEROL MARKEDLY ELEVATED (15-40 mg/dL; normal <0.5 mg/dL); "
            "  PLASMA CAMPESTEROL ELEVATED; "
            "  Total LDL-C: may be NORMAL or only modestly elevated — KEY DDx PITFALL; "
            "  (Sitosterol deposits in tissues despite normal/near-normal LDL-C); "
            "CLINICAL FEATURES: "
            "  TENDON XANTHOMAS in CHILDHOOD — KEY DDx FROM FH: "
            "    FH xanthomas: require LDL >300+ mg/dL sustained; adult onset typical; "
            "    Sitosterolemia xanthomas: can occur in FIRST DECADE with NORMAL LDL-C; "
            "  TUBEROUS XANTHOMAS (over joints, elbows, knees); "
            "  PREMATURE ATHEROSCLEROSIS: plant sterols in macrophages → foam cells → plaques; "
            "  MACROTHROMBOCYTOPENIA: large platelets + thrombocytopenia (stomatocytes on blood film); "
            "    mechanism: plant sterols alter platelet membrane structure; "
            "  HAEMOLYTIC ANAEMIA: precipitated by infection/stress (stomatocytosis); "
            "DIAGNOSIS CLUE: "
            "  XANTHOMAS IN CHILD WITH NON-SEVERELY ELEVATED LDL-C → MEASURE PLANT STEROLS by GC-MS; "
            "  incorrectly diagnosed as FH is common error; "
            "  FH panels may return negative if only LDLR/APOB/PCSK9 screened"
        ),
        "disease_pathway": (
            "ABCG5/ABCG8 — STEROL EFFLUX TRANSPORTER PATHWAY: "
            "NORMAL DIETARY STEROL HANDLING: "
            "  Diet: contains cholesterol + plant sterols (sitosterol, campesterol, stigmasterol, brassicasterol); "
            "  Intestinal NPC1L1: absorbs ALL sterols non-selectively; "
            "  Intestinal ABCG5/G8: effluxes plant sterols and some cholesterol BACK into lumen → "
            "    net plant sterol absorption <5%; net cholesterol absorption 50%; "
            "  Hepatic ABCG5/G8: secretes accumulated plant sterols into bile → faecal excretion; "
            "SITOSTEROLEMIA: "
            "  ABCG5 LOF → intestinal efflux of plant sterols abolished → absorption 60-80% of dietary plant sterols; "
            "  Hepatic efflux also abolished → plant sterols accumulate in liver + blood + tissues; "
            "  Sitosterol + campesterol in LDL particle → LDL taken up by macrophages (less efficiently than cholesterol); "
            "    → xanthoma formation with sitosterol-laden foam cells; "
            "  Atherosclerotic plaques contain sitosterol deposits; "
            "EZETIMIBE MECHANISM IN SITOSTEROLEMIA: "
            "  Ezetimibe blocks NPC1L1 at intestinal brush border → prevents both cholesterol AND plant sterol absorption; "
            "  DRAMATICALLY EFFECTIVE in sitosterolemia: "
            "    plasma sitosterol falls 50-70% with ezetimibe; "
            "    xanthoma regression documented; "
            "  EZETIMIBE IS FIRST-LINE TREATMENT (before cholestyramine, before diet alone); "
            "CHOLESTYRAMINE (bile acid sequestrant): "
            "  reduces intestinal absorption by sequestering bile acids → increases plant sterol cycling; "
            "  modest additional benefit after ezetimibe"
        ),
        "pathognomonic": (
            "TENDON + TUBEROUS XANTHOMAS IN CHILDHOOD WITH NORMAL OR MODESTLY ELEVATED LDL-C: "
            "  FH diagnostic criteria not met (LDL not severely elevated); "
            "  xanthomas present (foam cells with plant sterol content); "
            "  → MEASURE PLASMA PLANT STEROLS (sitosterol, campesterol) by GC-MS; "
            "  SITOSTEROL >5 mg/dL = diagnostic threshold; >15 mg/dL = unequivocal sitosterolemia; "
            "MACROTHROMBOCYTOPENIA ON FULL BLOOD COUNT: "
            "  large platelets (MPV >12 fL) + thrombocytopenia (<150,000/µL); "
            "  STOMATOCYTES on blood film (elliptical red cell membrane changes); "
            "  this combination in a child with xanthomas → sitosterolemia workup; "
            "HAEMOLYTIC ANAEMIA EPISODES: "
            "  triggered by infections or increased dietary plant sterol intake; "
            "  Coombs negative (not antibody-mediated); "
            "EZETIMIBE DRAMATIC RESPONSE: "
            "  sitosterol falls >50% within 4-8 weeks of ezetimibe initiation; "
            "  xanthomabegin regressing; "
            "  dramatic response confirms diagnosis before genetic confirmation"
        ),
        "treatment": (
            "1. EZETIMIBE (FIRST-LINE) — HIGHLY EFFECTIVE: "
            "   10mg daily; blocks NPC1L1 → prevents plant sterol absorption; "
            "   SITOSTEROL FALLS 50-70%; XANTHOMAS REGRESS; "
            "   mechanism-targeted treatment (prevents the primary absorption defect); "
            "2. LOW PHYTOSTEROL DIET: "
            "   AVOID: vegetable oils (high sitosterol), nuts, seeds, shellfish; "
            "   AVOID: foods fortified with plant sterols (margarine, yoghurt with added sterols — DANGEROUS in sitosterolemia); "
            "   low cholesterol diet NOT necessary if LDL-C not elevated; "
            "3. BILE ACID SEQUESTRANTS (cholestyramine): "
            "   modest additional plant sterol reduction; combine with ezetimibe; "
            "4. STATINS: NOT EFFECTIVE FOR XANTHOMA REGRESSION "
            "   (xanthomas reflect plant sterol, not LDL, accumulation); "
            "   may add if LDL-C also elevated (some patients have dual pathology); "
            "5. PCSK9 INHIBITORS: NOT HELPFUL (problem is sterol efflux, not LDLR cycling); "
            "6. MONITORING: "
            "   plasma sitosterol + campesterol (every 6-12 months on treatment); "
            "   platelet count (macrothrombocytopenia monitoring); "
            "   CBC for haemolytic anaemia; "
            "7. AVOID PLANT-STEROL-ENRICHED FOODS (actively harmful in sitosterolemia — worsen disease); "
            "CASCADE SCREENING: siblings at 25% risk (AR); "
            "PROGNOSIS: excellent with ezetimibe + diet — xanthomas regress, CVD risk falls"
        ),
        "key_facts": [
            "SITOSTEROLEMIA type A: plasma sitosterol 15-40 mg/dL (normal <0.5 mg/dL)",
            "XANTHOMAS IN CHILDHOOD WITH NORMAL LDL-C — key DDx from FH",
            "MACROTHROMBOCYTOPENIA + STOMATOCYTES on blood film — key diagnostic clue",
            "EZETIMIBE HIGHLY EFFECTIVE: blocks NPC1L1 → reduces plant sterol absorption 50-70%",
            "AVOID plant-sterol-enriched foods (harmful — worsens disease)",
            "STATINS and PCSK9i NOT EFFECTIVE for xanthoma regression",
            "GC-MS plant sterol assay required — not on standard lipid panel",
        ],
        "seed": 2770,
    },
    {
        "gene": "ABCG8",
        "protein": (
            "ABCG8 -- 2p21 AR -- 673aa -- ATP-Binding-Cassette-Sub-Family-G-Member-8-Sterolin-2-75kDa-"
            "Obligate-Heterodimer-ABCG5-Head-to-Head-Gene-2p21-Shared-Promoter-"
            "OMIM-Gene-605460-Disease-Sitosterolemia-210250"
        ),
        "locus": "2p21",
        "protein_size": "673 aa / 75 kDa (half-transporter; obligate heterodimer with ABCG5; 6 TM domains)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations in ABCG8 (Sterolin-2); "
            "ABCG8 STRUCTURE + GENE ORGANISATION: "
            "  ABCG5 and ABCG8 genes are HEAD-TO-HEAD on chromosome 2p21 with SHARED BIDIRECTIONAL PROMOTER; "
            "  transcribed in opposite directions; "
            "  both genes regulated together (facilitates co-expression); "
            "  ABCG8 encodes Sterolin-2 (673 aa); ABCG5 encodes Sterolin-1 (651 aa); "
            "  BOTH proteins required for functional transporter — a homodimer of either alone is non-functional; "
            "ABCG8 LOF → identical clinical phenotype to ABCG5 LOF (sitosterolemia type B vs type A); "
            "ABCG8 FOUNDER MUTATIONS: "
            "  p.Asp19His (D19H) — COMMON ASIAN FOUNDER (Japanese, Chinese, Korean); "
            "  p.Arg263Gln — European; "
            "  various truncating mutations; "
            "ABCG8 COMMON VARIANT (population genetics): "
            "  ABCG8 p.Asp19His also exists as common variant in Asian populations → modestly elevated plant sterols; "
            "  not sufficient alone for full sitosterolemia (needs second allele); "
            "GALLSTONE GENETICS: "
            "  ABCG8 variants associated with gallstone risk (biliary cholesterol secretion reduced); "
            "  distinct from sitosterolemia phenotype; low-penetrance association; "
            "PREVALENCE: similar to ABCG5 sitosterolemia (combined ABCG5+ABCG8 sitosterolemia ~1:200,000)"
        ),
        "disease_category": (
            "SITOSTEROLEMIA TYPE B — OMIM 210250 (same OMIM as ABCG5 — same disease, different gene); "
            "IDENTICAL CLINICAL PHENOTYPE TO ABCG5 SITOSTEROLEMIA: "
            "  PLASMA SITOSTEROL 15-40 mg/dL (normal <0.5 mg/dL); "
            "  PLASMA CAMPESTEROL ELEVATED; "
            "  LDL-C: normal to modestly elevated; "
            "CLINICAL FEATURES (identical to ABCG5): "
            "  TENDON + TUBEROUS XANTHOMAS in childhood; "
            "  PREMATURE ATHEROSCLEROSIS; "
            "  MACROTHROMBOCYTOPENIA: large platelets + thrombocytopenia; "
            "  HAEMOLYTIC ANAEMIA precipitated by infections; "
            "  STOMATOCYTES ON BLOOD FILM — key clue in paediatric xanthoma workup; "
            "DISTINGUISHING ABCG5 vs ABCG8 SITOSTEROLEMIA: "
            "  clinical phenotype IDENTICAL; only genetic sequencing distinguishes; "
            "  ABCG5 mutations → sitosterolemia type A; "
            "  ABCG8 mutations → sitosterolemia type B; "
            "  treatment is IDENTICAL regardless of which gene; "
            "IMPORTANT NOTE ON PCSK9i: "
            "  PCSK9 INHIBITORS DO NOT HELP — problem is sterol efflux from intestine/liver, not LDLR cycling; "
            "  do not mistake elevated LDL (if present) as target — treat plant sterols with ezetimibe first"
        ),
        "disease_pathway": (
            "ABCG8 — STEROL EFFLUX (IDENTICAL PATHWAY TO ABCG5): "
            "ABCG8 FUNCTION: "
            "  Forms heterodimer with ABCG5 (Sterolin-1); "
            "  ABCG8 NBD (nucleotide-binding domain) provides ATPase activity for sterol transport; "
            "  ABCG5 provides the second NBD; sterol transport requires both NBDs to hydrolyse ATP; "
            "  Localised at intestinal apical membrane + hepatic canalicular membrane; "
            "ABCG8 LOF: "
            "  ABCG5/G8 heterodimer non-functional (ABCG8 NBD absent → no sterol efflux ATPase); "
            "  intestinal plant sterol absorption uninhibited → sitosterol accumulates; "
            "  hepatic secretion abolished → sitosterol accumulates in liver; "
            "ASIAN D19H FOUNDER VARIANT: "
            "  p.Asp19His at N-terminus — disrupts ABCG8 protein processing or dimerisation; "
            "  homozygous or compound heterozygous D19H → sitosterolemia; "
            "  heterozygous D19H: modestly elevated plant sterols (not disease); "
            "MACROTHROMBOCYTOPENIA MECHANISM: "
            "  Plant sterols intercalate into platelet membrane lipid bilayer → alters membrane fluidity; "
            "  megakaryocytes produce fewer, larger platelets (ineffective thrombopoiesis); "
            "  stomatocyte formation: sitosterol alters red cell membrane → stomatocytic RBC; "
            "HAEMOLYSIS MECHANISM: "
            "  Plant sterols in RBC membrane → reduced deformability → haemolysis; "
            "  precipitated by infections (oxidative stress) or excess dietary plant sterols"
        ),
        "pathognomonic": (
            "STOMATOCYTES ON BLOOD FILM — KEY CLUE IN PAEDIATRIC XANTHOMA WORKUP: "
            "  stomatocytes = cup-shaped RBCs (single pale area instead of biconcave); "
            "  ANY child with xanthomas + stomatocytes → sitosterolemia workup MANDATORY; "
            "  MACROTHROMBOCYTOPENIA (large platelets, low count) on CBC adds specificity; "
            "HAEMOLYTIC ANAEMIA TRIGGERED BY INFECTION: "
            "  child with recurrent mild haemolytic anaemia + xanthomas + Coombs NEGATIVE → "
            "  sitosterolemia (Coombs positive would suggest immune haemolysis); "
            "PLASMA SITOSTEROL + CAMPESTEROL by GC-MS: "
            "  diagnostic; available at specialist lipid/metabolic laboratories; "
            "  ROUTINE LIPID PANEL DOES NOT INCLUDE PLANT STEROLS; "
            "EZETIMIBE RESPONSE (diagnostic + therapeutic): "
            "  sitosterol falls >50% within 8 weeks; "
            "  xanthomas begin regressing; platelets may improve partially; "
            "ABCG8 D19H CLUE: "
            "  East Asian ancestry + xanthomas in childhood + normal LDL → target ABCG8 D19H first; "
            "  Asian FH gene panels should include ABCG8 sequencing"
        ),
        "treatment": (
            "IDENTICAL TO ABCG5 SITOSTEROLEMIA: "
            "1. EZETIMIBE (FIRST-LINE): 10mg daily; "
            "   blocks NPC1L1 → prevents plant sterol absorption; "
            "   sitosterol falls 50-70%; xanthomas regress; "
            "2. LOW PHYTOSTEROL DIET: "
            "   avoid vegetable oils (sitosterol-rich), nuts, seeds, shellfish; "
            "   ABSOLUTELY AVOID plant-sterol-enriched foods (spreads/yoghurts); "
            "3. BILE ACID SEQUESTRANTS: cholestyramine 4-16g/day; modest additive effect; "
            "4. STATINS: NOT EFFECTIVE for xanthoma regression (plant sterol xanthomas, not LDL); "
            "5. PCSK9 INHIBITORS: NOT EFFECTIVE (problem is sterol efflux not LDLR); "
            "6. MONITORING: "
            "   plasma sitosterol + campesterol (q6-12 months); "
            "   CBC + platelet count; "
            "   echocardiography if aortic valve involvement suspected; "
            "7. HAEMOLYTIC CRISIS: "
            "   supportive care; reduce dietary plant sterols aggressively; "
            "   transfusion if haemoglobin <7 g/dL; "
            "8. CASCADE SCREENING: siblings 25% risk; consanguineous families highest risk; "
            "PROGNOSIS: excellent with ezetimibe — normalises plant sterol levels if adherent"
        ),
        "key_facts": [
            "SITOSTEROLEMIA type B: identical phenotype to ABCG5 type A",
            "ABCG5/ABCG8 head-to-head on 2p21, shared bidirectional promoter",
            "D19H ASIAN FOUNDER (Japanese/Chinese): commonest ABCG8 mutation in East Asia",
            "STOMATOCYTES on blood film + macrothrombocytopenia = key diagnostic clue",
            "PCSK9i DO NOT HELP (sterol efflux defect, not LDLR cycling)",
            "STATINS NOT EFFECTIVE for xanthoma regression",
            "Plant-sterol-enriched foods ACTIVELY HARMFUL — must be avoided",
        ],
        "seed": 2771,
    },
    {
        "gene": "LIPA",
        "protein": (
            "LIPA -- 10q23.31 AR -- 399aa -- Lysosomal-Acid-Lipase-A-45kDa-"
            "Lysosomal-Cholesteryl-Ester-TG-Hydrolase-Wolman-CESD-Sebelipase-Alfa-Kanuma-"
            "OMIM-Gene-613497-Disease-Wolman-278000-CESD-278000"
        ),
        "locus": "10q23.31",
        "protein_size": "399 aa / 45 kDa (lysosomal enzyme; N-glycosylated; mannose-6-phosphate receptor targeting)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "LIPA (lysosomal acid lipase A) is the lysosomal enzyme responsible for hydrolysing: "
            "  (1) cholesteryl esters (CE) from LDL-derived endosomes → free cholesterol; "
            "  (2) triglycerides in lysosomes → fatty acids; "
            "LIPA MECHANISM: "
            "  LDL endocytosed via LDLR → endolysosome; "
            "  LIPA hydrolyses LDL-CE → free cholesterol → intracellular regulatory pool; "
            "  Without LIPA: CE + TG accumulate in lysosomes of liver, adrenal, intestine, macrophages; "
            "TWO PHENOTYPES (severity-dependent on residual enzyme activity): "
            "  WOLMAN DISEASE (complete/near-zero activity; <1% residual): "
            "    infantile onset (<3 months); fulminant; fatal without treatment by 6-12 months; "
            "  CESD (>1% residual activity; hypomorphic mutations): "
            "    childhood hepatomegaly → adult cirrhosis; premature atherosclerosis; FH-like LDL elevation; "
            "KEY MUTATIONS: "
            "  WOLMAN: p.Gln298Ter, various null mutations; "
            "  CESD: c.894G>A (E8SJM — exon 8 splice junction mutation; most common CESD; p.delGlu254); "
            "    c.894G>A: 1-3% residual LIPA activity (enough to avoid Wolman phenotype); "
            "PREVALENCE: Wolman 1:350,000-500,000; CESD 1:40,000-300,000 (underdiagnosed); "
            "SEBELIPASE ALFA (Kanuma): recombinant human LAL; FDA2015; "
            "  for both Wolman (weekly 1 mg/kg IV then escalate) and CESD (1 mg/kg IV q2w)"
        ),
        "disease_category": (
            "TWO PHENOTYPES FROM SAME LIPA GENE: "
            "WOLMAN DISEASE (complete LIPA deficiency): "
            "  INFANTILE ONSET: vomiting, diarrhoea, failure to thrive by age 3 months; "
            "  HEPATOSPLENOMEGALY: massive; progressive hepatic failure; "
            "  BILATERAL ADRENAL CALCIFICATION: seen on plain X-ray — PATHOGNOMONIC for Wolman; "
            "    calcification of adrenal cortex (CE accumulation → calcification); "
            "  MALABSORPTION: CE accumulation in intestinal epithelium; "
            "  RAPIDLY FATAL: death by 6-12 months WITHOUT treatment; "
            "  SEBELIPASE ALFA (Kanuma FDA2015): TRANSFORMS FATAL DISEASE; "
            "    1 mg/kg IV weekly initially → 3 mg/kg weekly if severe; reduces CE accumulation; "
            "CESD (Cholesteryl Ester Storage Disease — partial LIPA deficiency): "
            "  CHILDHOOD TO ADULT PRESENTATION; insidious; often missed; "
            "  HEPATOMEGALY (hepatic CE accumulation → microvesicular steatosis → cirrhosis); "
            "  ELEVATED LDL-C (MIMICS FH): CE accumulation → reduced intracellular free cholesterol → "
            "    LDLR upregulation suppressed → LDL-C accumulates (FH-like); "
            "  PREMATURE ATHEROSCLEROSIS: macrophage CE foam cells in vessel walls; "
            "  ADRENAL CALCIFICATION: RARE in CESD (contrast Wolman where pathognomonic); "
            "  SEBELIPASE ALFA (Kanuma FDA2015): approved for CESD; reduces CE; improves LFTs + LDL"
        ),
        "disease_pathway": (
            "LIPA — LYSOSOMAL CE/TG HYDROLYSIS PATHWAY: "
            "NORMAL: "
            "  1. LDL binds LDLR → clathrin-coated pit endocytosis; "
            "  2. Early endosome → late endosome/lysosome; "
            "  3. LIPA (lysosomal acid lipase) in lysosome → hydrolyses LDL-CE → free cholesterol; "
            "  4. Free cholesterol exported from lysosome via NPC1/NPC2; "
            "  5. Free cholesterol → cell membrane, esterification by ACAT, or regulatory functions; "
            "  6. High intracellular cholesterol → suppresses LDLR (SREBP2 pathway) → LDL not overaccumulated; "
            "LIPA DEFICIENCY (WOLMAN/CESD): "
            "  LDL endocytosed normally; "
            "  Lysosomal CE/TG NOT hydrolysed → CE + TG accumulate in lysosomes; "
            "  Free cholesterol NOT released → intracellular regulatory pool depleted; "
            "  LDLR NOT suppressed (thinks cell is cholesterol-starved) → LDLR UPREGULATED; "
            "    → MORE LDL endocytosed → more CE accumulates → lysosomal storage overload; "
            "FH-LIKE LDL ELEVATION MECHANISM IN CESD: "
            "  CE trapped in lysosomes → not available for SREBP2 suppression → LDLR stays high; "
            "  MORE LDL cleared from plasma initially BUT → lysosomes saturated → CE spills into cytoplasm; "
            "  Net effect: LDL-C elevated (similar to FH); "
            "SEBELIPASE ALFA MECHANISM: "
            "  Recombinant human LAL → IV infused → taken up by liver/adrenal/macrophages via M6P receptor; "
            "  Lysosomal LAL activity restored → CE hydrolysed → CE accumulation reversed; "
            "ADRENAL CALCIFICATION: "
            "  Adrenal cortex most CE-rich tissue in body (uses cholesterol for steroidogenesis); "
            "  CE accumulation → dystrophic calcification in adrenal cortex; "
            "  WOLMAN: bilateral calcification visible on plain X-ray in infants"
        ),
        "pathognomonic": (
            "WOLMAN: BILATERAL ADRENAL CALCIFICATION ON ABDOMINAL X-RAY in INFANT: "
            "  eggshell/stippled calcification of both adrenals; "
            "  ANY infant with hepatosplenomegaly + failure to thrive → abdominal X-ray or CT; "
            "  adrenal calcification + hepatosplenomegaly in infant = Wolman until proven otherwise; "
            "  RARE: also seen in neuroblastoma, neonatal haemorrhage — context distinguishes; "
            "LAL ENZYME ACTIVITY ASSAY (DRY BLOOD SPOT): "
            "  LIPA activity on DBS or leukocytes → near-zero in Wolman; 1-10% in CESD; "
            "  NBS LAL DBS assay available in some jurisdictions; "
            "  DEFINITIVE DIAGNOSIS: LAL activity on DBS + LIPA sequencing; "
            "FH-LIKE LDL ELEVATION + HEPATOMEGALY → CHECK LAL ACTIVITY: "
            "  CESD in adult: unexplained hepatomegaly + elevated LDL-C + liver biopsy showing "
            "    microvesicular steatosis (CE deposits) → CESD; "
            "  LIVER BIOPSY: orange/yellow CE crystals in hepatocytes (under polarised light — birefringent); "
            "  LAL ACTIVITY ON DBS: simple non-invasive confirmation; "
            "c.894G>A SPLICE MUTATION (CESD): "
            "  MOST COMMON CESD MUTATION globally; detectable on targeted sequencing; "
            "  combined with another LOF allele → CESD phenotype"
        ),
        "treatment": (
            "WOLMAN DISEASE: "
            "1. SEBELIPASE ALFA (Kanuma FDA2015, EMA2015) — LIFE-SAVING: "
            "   1 mg/kg IV weekly; escalate to 3 mg/kg weekly if severely affected; "
            "   reduces CE accumulation in liver, adrenal, intestine; "
            "   improves survival dramatically (untreated: fatal <12 months; treated: long-term survival possible); "
            "2. NUTRITIONAL SUPPORT: nasogastric/parenteral nutrition for malabsorption; "
            "3. HAEMATOPOIETIC STEM CELL TRANSPLANT: prior to sebelipase era; now rarely needed; "
            "4. ADRENAL INSUFFICIENCY: hydrocortisone replacement if adrenal function impaired; "
            "CESD: "
            "5. SEBELIPASE ALFA (Kanuma FDA2015 for CESD): 1 mg/kg IV q2 weeks; "
            "   reduces liver CE; improves LFTs; reduces LDL-C; slows fibrosis progression; "
            "6. STATINS: reduce LDL-C (partially effective — LDLR upregulated by statin); "
            "   BUT does not treat hepatic CE accumulation (sebelipase alfa required); "
            "7. EZETIMIBE: additional LDL-C reduction; "
            "8. LIVER DISEASE MANAGEMENT: avoid hepatotoxins; monitor LFTs; fibroscan; "
            "9. CARDIOVASCULAR RISK REDUCTION: statins for atherosclerosis prevention in CESD adults; "
            "CASCADE SCREENING: siblings at 25% risk; LAL DBS assay (simple, non-invasive); "
            "NEWBORN SCREENING: LAL DBS assay available — some jurisdictions; detect Wolman before symptoms"
        ),
        "key_facts": [
            "WOLMAN: bilateral adrenal calcification on X-ray PATHOGNOMONIC in infants",
            "CESD: FH-like LDL elevation + hepatomegaly — LAL activity assay distinguishes",
            "SEBELIPASE ALFA (Kanuma FDA2015): transforms fatal Wolman to survivable disease",
            "c.894G>A splice mutation: commonest CESD mutation globally",
            "LAL DBS assay: simple non-invasive diagnosis (dried blood spot)",
            "Statins reduce LDL-C in CESD but do NOT treat hepatic CE accumulation",
            "CESD often misdiagnosed as NAFLD or cryptogenic hepatomegaly",
        ],
        "seed": 2772,
    },
    {
        "gene": "LPA",
        "protein": (
            "LPA -- 6q27 AD-dose-effect -- 5927aa -- Apolipoprotein-a-500-700kDa-"
            "KIV2-Kringle-Repeat-Variable-Number-Disulfide-Bond-ApoB100-Lp(a)-Particle-"
            "OMIM-Gene-152200-Disease-Elevated-Lp(a)-Hyperlipoproteinemia"
        ),
        "locus": "6q27",
        "protein_size": "5927 aa / 500-700 kDa (size varies by KIV-2 kringle repeat number; disulfide-bonded to ApoB-100 on LDL)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (additive/semi-dominant) — Lp(a) levels determined 70-90% by LPA gene; "
            "LPA GENE STRUCTURE: "
            "  KIV-2 KRINGLE REPEAT NUMBER: LPA gene contains variable number (2-40+) tandem repeats of KIV-2 kringle domain; "
            "  FEWER KIV-2 REPEATS → SHORTER Apo(a) → HIGHER Lp(a) levels (inverse relationship); "
            "  MORE KIV-2 REPEATS → LONGER Apo(a) → LOWER Lp(a) levels; "
            "  KIV-2 repeat number is the MAIN DETERMINANT of Lp(a) level; "
            "ETHNIC VARIATION: "
            "  Lp(a) levels 2-3x HIGHER in individuals of African descent vs European; "
            "  Europeans with Lp(a) >50 mg/dL: ~20% of population; "
            "  African-Americans with Lp(a) >50 mg/dL: ~40% of population; "
            "LPA vs LDLR INTERACTION: "
            "  Lp(a) can bind LDLR (via ApoB-100 component) but clearance of Lp(a) is INEFFICIENT; "
            "  PCSK9i: reduces Lp(a) 20-30% (increases LDLR → modestly improves Lp(a) clearance); "
            "  STATINS: may INCREASE Lp(a) 5-15% (unclear mechanism — possible upregulation of Lp(a) secretion); "
            "Lp(a) CARDIOVASCULAR RISK: "
            "  Mendelian randomisation studies confirm CAUSAL role of Lp(a) in CVD and AORTIC VALVE CALCIFICATION; "
            "  independent of LDL-C"
        ),
        "disease_category": (
            "FAMILIAL Lp(a) HYPERLIPOPROTEINEMIA / ELEVATED Lp(a) — OMIM 152200; "
            "BIOCHEMICAL HALLMARKS: "
            "  Lp(a) >50 mg/dL (>125 nmol/L) = HIGH CARDIOVASCULAR RISK threshold; "
            "  Lp(a) >150 mg/dL ≈ FH-equivalent cardiovascular risk (similar to HeFH); "
            "  LDL-C often normal or separately elevated; "
            "  Lp(a)-cholesterol content: approximately 30% of Lp(a) mass is cholesterol; "
            "  POPULATION PREVALENCE: ~1:5 persons have Lp(a) >50 mg/dL; "
            "CARDIOVASCULAR RISK: "
            "  Lp(a) >50 mg/dL: ~2x MI risk vs low Lp(a); "
            "  Lp(a) >150 mg/dL: 3-4x MI risk; approaches HeFH-LDLR risk; "
            "  AORTIC VALVE CALCIFICATION: Lp(a) independently linked (KIV-2 repeat analysis confirms causality); "
            "  STROKE risk mildly elevated; "
            "PROTHROMBOTIC MECHANISM: "
            "  Apo(a) has structural homology to PLASMINOGEN; "
            "  Apo(a) COMPETES WITH PLASMINOGEN for fibrin binding → impairs fibrinolysis → "
            "  prothrombotic state (in addition to atherogenic LDL component); "
            "MEASUREMENT CONSIDERATIONS: "
            "  Lp(a) by mass (mg/dL) OR molar concentration (nmol/L): "
            "    mg/dL and nmol/L NOT interchangeable by simple conversion (size varies); "
            "  ISOFORM-INSENSITIVE ASSAY preferred (avoids KIV-2 size artifact); "
            "  FASTING NOT REQUIRED (Lp(a) stable regardless of fasting); "
            "  MEASURE ONCE IN LIFETIME (Lp(a) stable over lifetime; not affected by diet or most drugs)"
        ),
        "disease_pathway": (
            "Lp(a) — STRUCTURE AND DUAL ATHEROGENIC + PROTHROMBOTIC MECHANISM: "
            "Lp(a) PARTICLE STRUCTURE: "
            "  Core: LDL-like particle (APOB-100 + CE + PL + TG); "
            "  APOB-100 disulfide-bonded to Apo(a) protein → Lp(a) particle; "
            "  Apo(a): multiple kringle IV repeats (KIV-1 to KIV-10, KV) + serine protease domain (INACTIVE); "
            "  KIV-2 repeats: variable number → determines size and Lp(a) level; "
            "ATHEROGENIC MECHANISM 1 — LDL-LIKE: "
            "  Lp(a) enters arterial intima → taken up by macrophages → foam cells; "
            "  Lp(a) less efficiently cleared than LDL → longer plasma residence time; "
            "ATHEROGENIC MECHANISM 2 — AORTIC VALVE: "
            "  Lp(a) binds OxPL (oxidised phospholipids on Lp(a)) → promotes calcification; "
            "  OxPL content is the atherogenic mediator (not just the particle); "
            "  explains aortic valve calcific stenosis association; "
            "PROTHROMBOTIC MECHANISM — PLASMINOGEN COMPETITION: "
            "  Apo(a) kringle IV-10 binds fibrin (mimics plasminogen) → "
            "  blocks tissue plasminogen activator (tPA) activation → impaired clot lysis; "
            "EMERGING THERAPIES: "
            "  PELACARSEN (TQJ230): GalNAc-antisense LPA mRNA; reduces Lp(a) 80-90%; Phase 3 HORIZON trial; "
            "  OLPASIRAN (AMG 890): siRNA against LPA mRNA; Phase 3; LPA Lp(a) reduction >90%; "
            "  MUVALAPLIN: small molecule; disrupts Apo(a)-ApoB disulfide bond formation → less Lp(a) assembled; "
            "  NIACIN (historical): reduces Lp(a) 25-40% but no CV outcome benefit (AIM-HIGH, HPS2-THRIVE negative); "
            "  PCSK9i: reduces Lp(a) 20-30% (insufficient to reach <50 mg/dL if starting very high)"
        ),
        "pathognomonic": (
            "Lp(a) LEVEL >50 mg/dL ON STANDARD TESTING (not on routine lipid panel): "
            "  Lp(a) NOT on standard lipid panel — must be specifically ordered; "
            "  measure once in lifetime (stable); fasting not required; "
            "  isoform-insensitive assay (important for accuracy across KIV-2 sizes); "
            "CONCURRENT HIGH LDL-C + HIGH Lp(a) = EXTREMELY HIGH CVD RISK: "
            "  Lp(a) adds INDEPENDENT risk on top of LDL-C; "
            "  DUTCH LIPID CLINIC SCORE may underestimate risk if Lp(a) not considered; "
            "  Lp(a) >150 mg/dL ≈ equivalent of LDLR-HeFH for CVD purposes; "
            "RECURRENT PREMATURE MI IN TREATED FH: "
            "  Patient on maximal statin + PCSK9i, LDL well-controlled, still has MI → "
            "  CHECK Lp(a): residual CVD risk despite LDL control may be Lp(a)-driven; "
            "AORTIC STENOSIS UNDER AGE 60: "
            "  Calcific aortic valve disease at young age → check Lp(a); "
            "  Lp(a) independently doubles risk of aortic valve calcification; "
            "CASCADE TESTING INDICATION: "
            "  Measure Lp(a) in ALL first-degree relatives if proband Lp(a) >100 mg/dL; "
            "  single lifetime measurement sufficient"
        ),
        "treatment": (
            "CURRENT STANDARD OF CARE (limited options): "
            "1. AGGRESSIVE LDL-C LOWERING (indirectly protective): "
            "   high-intensity statin + ezetimibe + PCSK9i; "
            "   STATINS: may increase Lp(a) modestly (5-15%) — do NOT withhold (LDL benefit outweighs); "
            "   PCSK9i (evolocumab/alirocumab): REDUCE Lp(a) 20-30%; "
            "     insufficient if Lp(a) >200 mg/dL but meaningful in moderate elevation; "
            "2. NIACIN: reduces Lp(a) 25-40% but NO CV outcome benefit (AIM-HIGH, HPS2-THRIVE negative trials); "
            "   rarely used now due to side effects + neutral outcomes; "
            "3. LDL APHERESIS: removes both LDL and Lp(a); some guidelines support for Lp(a) >100+ mg/dL; "
            "   q1-2 weeks; reduces Lp(a) 50-70% per session; "
            "EMERGING (HIGHLY PROMISING): "
            "4. PELACARSEN (TQJ230): GalNAc-ASO against LPA mRNA; "
            "   Phase 3 HORIZON trial (75mg monthly SC); Lp(a) reduction 80-90%; CV outcomes pending; "
            "5. OLPASIRAN (AMG 890): siRNA against LPA mRNA; Phase 3; "
            "   Lp(a) reduction >90%; quarterly SC dosing; "
            "6. MUVALAPLIN: oral small molecule; disrupts Apo(a)-ApoB disulfide linkage; Phase 2; "
            "CARDIOVASCULAR RISK MANAGEMENT (until Lp(a)-specific therapies available): "
            "   treat modifiable risk factors aggressively (BP, smoking, diabetes, LDL); "
            "   consider aspirin in high Lp(a) + high CVD risk (antithrombotic); "
            "   avoid progesterone-only contraceptives (may increase Lp(a)); "
            "PATIENT COUNSELLING: "
            "   measure ONCE; stable over lifetime; screen family members if proband >100 mg/dL; "
            "   inform about emerging therapies in development"
        ),
        "key_facts": [
            "Lp(a) NOT on routine lipid panel — must be specifically ordered; measure once in lifetime",
            "Lp(a) >50 mg/dL: ~1:5 persons; 2x MI risk; >150 mg/dL ≈ HeFH-LDLR equivalent risk",
            "STATINS may INCREASE Lp(a) 5-15% (do not withhold — LDL benefit outweighs)",
            "PCSK9i reduce Lp(a) 20-30% — only current meaningful Lp(a) drug therapy",
            "PELACARSEN (Phase 3 HORIZON): GalNAc-ASO; reduces Lp(a) 80-90%",
            "OLPASIRAN (siRNA, Phase 3): >90% Lp(a) reduction; quarterly dosing",
            "Apo(a) competes with plasminogen for fibrin → prothrombotic mechanism",
            "Aortic valve calcification independently caused by Lp(a) (OxPL mechanism)",
        ],
        "seed": 2773,
    },
]


def _generate_patients(gene_idx: int, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    gene_data = ATLAS_GENES[gene_idx]
    gene = gene_data["gene"]

    gene_params = {
        "LDLR":    {"ldl_range": (250, 1000), "has_tendon_xanth": 0.50, "has_cvd": 0.55,
                    "has_aortic_stenosis": 0.20, "has_corneal_arcus": 0.45, "statin_response": 0.75,
                    "pcsk9i_candidate": 0.80, "onset_range": (10, 45), "apheresis_needed": 0.20},
        "APOB":    {"ldl_range": (200, 380), "has_tendon_xanth": 0.28, "has_cvd": 0.38,
                    "has_aortic_stenosis": 0.05, "has_corneal_arcus": 0.35, "statin_response": 0.85,
                    "pcsk9i_candidate": 0.70, "onset_range": (20, 55), "apheresis_needed": 0.05},
        "PCSK9":   {"ldl_range": (280, 900), "has_tendon_xanth": 0.45, "has_cvd": 0.60,
                    "has_aortic_stenosis": 0.30, "has_corneal_arcus": 0.50, "statin_response": 0.55,
                    "pcsk9i_candidate": 0.90, "onset_range": (8, 40), "apheresis_needed": 0.25},
        "LDLRAP1": {"ldl_range": (350, 720), "has_tendon_xanth": 0.55, "has_cvd": 0.48,
                    "has_aortic_stenosis": 0.15, "has_corneal_arcus": 0.50, "statin_response": 0.45,
                    "pcsk9i_candidate": 0.60, "onset_range": (5, 25), "apheresis_needed": 0.55},
        "ABCG5":   {"ldl_range": (100, 280), "has_tendon_xanth": 0.72, "has_cvd": 0.35,
                    "has_aortic_stenosis": 0.08, "has_corneal_arcus": 0.15, "statin_response": 0.10,
                    "pcsk9i_candidate": 0.05, "onset_range": (2, 20), "apheresis_needed": 0.02},
        "ABCG8":   {"ldl_range": (90, 270), "has_tendon_xanth": 0.68, "has_cvd": 0.32,
                    "has_aortic_stenosis": 0.06, "has_corneal_arcus": 0.12, "statin_response": 0.10,
                    "pcsk9i_candidate": 0.04, "onset_range": (2, 18), "apheresis_needed": 0.02},
        "LIPA":    {"ldl_range": (140, 380), "has_tendon_xanth": 0.08, "has_cvd": 0.40,
                    "has_aortic_stenosis": 0.05, "has_corneal_arcus": 0.10, "statin_response": 0.50,
                    "pcsk9i_candidate": 0.20, "onset_range": (0, 40), "apheresis_needed": 0.05},
        "LPA":     {"ldl_range": (80, 220), "has_tendon_xanth": 0.05, "has_cvd": 0.50,
                    "has_aortic_stenosis": 0.30, "has_corneal_arcus": 0.15, "statin_response": 0.60,
                    "pcsk9i_candidate": 0.55, "onset_range": (35, 65), "apheresis_needed": 0.12},
    }

    params = gene_params[gene]
    patients = []

    for i in range(n):
        ldl = rng.randint(*params["ldl_range"])
        onset = rng.randint(*params["onset_range"])
        hdl = rng.randint(25, 55) if ldl > 300 else rng.randint(35, 70)
        tg = rng.randint(60, 200)  # FH: TG usually normal

        # Lp(a) level: high for LPA gene, variable for others
        if gene == "LPA":
            lpa_mg_dl = rng.randint(80, 300)
        else:
            lpa_mg_dl = rng.randint(5, 60)

        patients.append({
            "patient_id": f"{gene}-{seed:04d}-{i+1:02d}",
            "gene": gene,
            "ldl_c_mg_dL": ldl,
            "hdl_c_mg_dL": hdl,
            "tg_mg_dL": tg,
            "lpa_mg_dL": lpa_mg_dl,
            "tendon_xanthoma": rng.random() < params["has_tendon_xanth"],
            "cvd_event": rng.random() < params["has_cvd"],
            "aortic_stenosis": rng.random() < params["has_aortic_stenosis"],
            "corneal_arcus": rng.random() < params["has_corneal_arcus"],
            "statin_responder": rng.random() < params["statin_response"],
            "pcsk9i_candidate": rng.random() < params["pcsk9i_candidate"],
            "apheresis_needed": rng.random() < params["apheresis_needed"],
            "onset_years": onset,
            "dutch_score_ge8": ldl > 330 or (rng.random() < params["has_tendon_xanth"]),
        })

    return patients


def generate_overview():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2766 + idx))

    summary = {}
    for p in all_patients:
        g = p["gene"]
        if g not in summary:
            summary[g] = {
                "gene": g, "n": 0,
                "tendon_xanth_n": 0, "cvd_n": 0, "aortic_stenosis_n": 0,
                "corneal_n": 0, "pcsk9i_n": 0, "apheresis_n": 0,
                "dutch_ge8_n": 0,
                "mean_ldl": 0.0, "mean_onset": 0.0, "mean_lpa": 0.0,
            }
        s = summary[g]
        s["n"] += 1
        s["tendon_xanth_n"] += int(p["tendon_xanthoma"])
        s["cvd_n"] += int(p["cvd_event"])
        s["aortic_stenosis_n"] += int(p["aortic_stenosis"])
        s["corneal_n"] += int(p["corneal_arcus"])
        s["pcsk9i_n"] += int(p["pcsk9i_candidate"])
        s["apheresis_n"] += int(p["apheresis_needed"])
        s["dutch_ge8_n"] += int(p["dutch_score_ge8"])
        s["mean_ldl"] += p["ldl_c_mg_dL"]
        s["mean_onset"] += p["onset_years"]
        s["mean_lpa"] += p["lpa_mg_dL"]

    gene_summaries = []
    for g, s in summary.items():
        n = s["n"]
        gene_summaries.append({
            "gene": g,
            "n": n,
            "tendon_xanthoma_pct": round(100 * s["tendon_xanth_n"] / n, 1),
            "cvd_event_pct": round(100 * s["cvd_n"] / n, 1),
            "aortic_stenosis_pct": round(100 * s["aortic_stenosis_n"] / n, 1),
            "corneal_arcus_pct": round(100 * s["corneal_n"] / n, 1),
            "pcsk9i_candidate_pct": round(100 * s["pcsk9i_n"] / n, 1),
            "apheresis_needed_pct": round(100 * s["apheresis_n"] / n, 1),
            "dutch_score_ge8_pct": round(100 * s["dutch_ge8_n"] / n, 1),
            "mean_ldl_c_mg_dL": round(s["mean_ldl"] / n, 1),
            "mean_onset_years": round(s["mean_onset"] / n, 1),
            "mean_lpa_mg_dL": round(s["mean_lpa"] / n, 1),
        })

    return {
        "atlas": "Hereditary Familial Hypercholesterolemia Atlas",
        "atlas_subtitle": "Complete 8-Gene FH Reference — LDLR · APOB · PCSK9 · LDLRAP1 · ABCG5 · ABCG8 · LIPA · LPA",
        "total_patients": len(all_patients),
        "gene_count": 8,
        "genes_covered": [g["gene"] for g in ATLAS_GENES],
        "seed_range": "2766-2773",
        "gene_summaries": gene_summaries,
        "fh_classification": [
            "FH TYPE 1 (LDLR) — most common monogenic FH; 1:200-500 heterozygous; LDL 250-450 mg/dL HeFH; "
            "LDL 500-1000+ HoFH; TENDON XANTHOMAS PATHOGNOMONIC; Dutch score ≥8 = definite FH; "
            "statins first-line; PCSK9i add-on; evinacumab FDA2021 for HoFH LDLR-independent",
            "FDB (APOB) — p.Arg3500Gln European founder 1:700; LDL 250-350 (MILDER than LDLR-FH); "
            "LDLR intact in FDB; PCSK9i highly effective (intact LDLR recycled); "
            "tendon xanthomas less frequent; statins more effective than in LDLR-null",
            "FH TYPE 3 (PCSK9 GOF) — D374Y Norwegian founder most severe; LDL 300-800 mg/dL; "
            "PCSK9i mechanism-specific treatment; inclisiran siRNA FDA2021 twice-yearly; "
            "PCSK9 LOF (R46L, Y142X) PROTECTIVE — validated PCSK9 as therapeutic target",
            "ARH (LDLRAP1) — AR; LDL 400-700 mg/dL; LDLR EXPRESSED NORMALLY but not internalised in hepatocytes; "
            "fibroblast LDLR NORMAL (misleadingly reassuring); statins PARTIALLY effective; "
            "apheresis usually required; evinacumab LDLR-independent → effective",
            "SITOSTEROLEMIA TYPE A (ABCG5) — AR; plant sterols 15-40 mg/dL; "
            "XANTHOMAS in CHILDHOOD with NORMAL LDL-C (key DDx from FH); "
            "macrothrombocytopenia; ezetimibe HIGHLY EFFECTIVE (first-line); "
            "statins and PCSK9i NOT effective for xanthoma regression",
            "SITOSTEROLEMIA TYPE B (ABCG8) — AR; identical to ABCG5; D19H Asian founder; "
            "stomatocytes on blood film; macrothrombocytopenia; ezetimibe first-line; "
            "plant-sterol-enriched foods ACTIVELY HARMFUL",
            "LAL-D (LIPA) — AR; TWO PHENOTYPES: Wolman (complete — fatal infant; adrenal calcification PATHOGNOMONIC) "
            "and CESD (partial — hepatomegaly + FH-like LDL); sebelipase alfa (Kanuma FDA2015) life-saving; "
            "LAL DBS assay diagnostic",
            "ELEVATED Lp(a) (LPA) — AD dose-effect; KIV-2 repeats determine Lp(a); "
            "Lp(a) >50 mg/dL ~1:5 persons; 2x MI risk; STATINS may INCREASE Lp(a); "
            "PCSK9i reduce Lp(a) 20-30%; pelacarsen Phase 3 (80% reduction); measure ONCE lifetime",
        ],
        "critical_distinctions": [
            "TENDON XANTHOMAS IN CHILDHOOD + NORMAL LDL-C → SITOSTEROLEMIA not FH (measure plant sterols by GC-MS)",
            "LDLR-NULL HoFH: statins and PCSK9i minimally effective — need evinacumab + apheresis + lomitapide",
            "ARH (LDLRAP1): fibroblast LDLR NORMAL — do NOT rule out severe hypercholesterolaemia; check LDLRAP1",
            "FDB (APOB R3500Q): MILDER LDL than LDLR-FH; PCSK9i MOST EFFECTIVE (intact LDLR recycled)",
            "PCSK9 GOF (FH3): inclisiran/evolocumab mechanism-specific — blocks the causal protein directly",
            "CESD (LIPA): FH-like LDL elevation + hepatomegaly → LAL DBS assay before assuming FH",
            "Lp(a) NOT on routine lipid panel — must order separately; STATINS may RAISE Lp(a)",
            "PCSK9i reduce Lp(a) 20-30% only — pelacarsen/olpasiran needed for high Lp(a) control",
            "WOLMAN: bilateral adrenal calcification on X-ray in infant + hepatosplenomegaly = emergency (sebelipase alfa)",
            "STOMATOCYTES + MACROTHROMBOCYTOPENIA + XANTHOMAS → ABCG8 sitosterolemia (not FH, not ITP)",
        ],
        "key_drug_classes": [
            "Statins (HMG-CoA reductase inhibitors): upregulate LDLR; first-line FH; ineffective in LDLR-null",
            "Ezetimibe (NPC1L1 inhibitor): +20% LDL reduction; CURATIVE in sitosterolemia (blocks plant sterol absorption)",
            "PCSK9 inhibitors (evolocumab/alirocumab mAb FDA2015): +50-60% LDL on statin; reduce Lp(a) 20-30%",
            "Inclisiran (siRNA PCSK9 mRNA FDA2021): twice-yearly SC; 50-60% LDL reduction; hepatic GalNAc delivery",
            "Evinacumab (anti-ANGPTL3 FDA2021): LDLR-independent; +47% LDL reduction in HoFH LDLR-null",
            "Lomitapide (MTP inhibitor FDA2012): HoFH only; blocks VLDL assembly; hepatotoxicity monitoring",
            "Sebelipase alfa (Kanuma FDA2015): recombinant LAL; Wolman life-saving; CESD hepatic CE reduction",
            "Pelacarsen (GalNAc-ASO LPA mRNA, Phase 3): 80-90% Lp(a) reduction; HORIZON trial ongoing",
            "Olpasiran (siRNA LPA mRNA, Phase 3): >90% Lp(a) reduction; quarterly dosing",
            "LDL apheresis: physical LDL removal q1-2 weeks; HoFH standard; also removes Lp(a)",
        ],
    }


def generate_breakdown():
    breakdown = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2766 + idx)
        n = len(patients)
        tendon_n = sum(1 for p in patients if p["tendon_xanthoma"])
        cvd_n = sum(1 for p in patients if p["cvd_event"])
        aortic_n = sum(1 for p in patients if p["aortic_stenosis"])
        corneal_n = sum(1 for p in patients if p["corneal_arcus"])
        pcsk9i_n = sum(1 for p in patients if p["pcsk9i_candidate"])
        apheresis_n = sum(1 for p in patients if p["apheresis_needed"])
        dutch_n = sum(1 for p in patients if p["dutch_score_ge8"])
        mean_ldl = sum(p["ldl_c_mg_dL"] for p in patients) / n
        mean_hdl = sum(p["hdl_c_mg_dL"] for p in patients) / n
        mean_tg = sum(p["tg_mg_dL"] for p in patients) / n
        mean_lpa = sum(p["lpa_mg_dL"] for p in patients) / n

        breakdown.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"][:350],
            "disease_category": gene_data["disease_category"][:450],
            "pathognomonic": gene_data["pathognomonic"][:400],
            "treatment_summary": gene_data["treatment"][:400],
            "key_facts": gene_data["key_facts"],
            "patients_n": n,
            "tendon_xanthoma_pct": round(100 * tendon_n / n, 1),
            "cvd_event_pct": round(100 * cvd_n / n, 1),
            "aortic_stenosis_pct": round(100 * aortic_n / n, 1),
            "corneal_arcus_pct": round(100 * corneal_n / n, 1),
            "pcsk9i_candidate_pct": round(100 * pcsk9i_n / n, 1),
            "apheresis_needed_pct": round(100 * apheresis_n / n, 1),
            "dutch_score_ge8_pct": round(100 * dutch_n / n, 1),
            "mean_ldl_c_mg_dL": round(mean_ldl, 1),
            "mean_hdl_c_mg_dL": round(mean_hdl, 1),
            "mean_tg_mg_dL": round(mean_tg, 1),
            "mean_lpa_mg_dL": round(mean_lpa, 1),
            "sample_patients": patients[:5],
        })

    return {"atlas": "Hereditary Familial Hypercholesterolemia Atlas", "breakdown": breakdown}


def generate_definitions():
    glossary = {
        "Familial Hypercholesterolemia (FH)": (
            "Monogenic hypercholesterolaemia; most common: LDLR mutations (FH1, 1:200-500 HeFH); "
            "also APOB FDB (R3500Q), PCSK9 GOF (FH3), LDLRAP1 ARH, and FH-mimics (ABCG5/8, LIPA, LPA); "
            "LDL-C >190 mg/dL adult + family history + clinical signs; "
            "Dutch Lipid Clinic Network Score ≥8 = definite FH (no genetic test needed); "
            "HoFH: LDL >500 mg/dL; childhood CHD; aortic valve stenosis; requires apheresis ± evinacumab"
        ),
        "Dutch Lipid Clinic Network Score": (
            "Clinical FH scoring system; ≥8 = DEFINITE FH; 6-7 = probable; 3-5 = possible; "
            "points from: family history CHD/FH, personal CHD history, tendon xanthomas (6pts), "
            "corneal arcus <45 years (4pts), LDL-C level (1-8pts), DNA positive (8pts); "
            "sensitivity ~80% for LDLR mutations at ≥8 threshold"
        ),
        "Tendon Xanthomas": (
            "Pathognomonic for FH (when present) — LDL-laden foam cells in tendon collagen; "
            "Achilles tendon thickening (normal <8mm; >8mm on ultrasound = abnormal); "
            "extensor tendons of hands (knuckle); patellar tendon; "
            "present in ~50% HeFH LDLR patients; ~30% FDB patients; higher in older/untreated; "
            "DISTINCT from tuberous xanthomas (FH) and eruptive xanthomas (hypertriglyceridemia); "
            "CHILDHOOD XANTHOMAS + NORMAL LDL-C → sitosterolemia (not FH) — measure plant sterols"
        ),
        "PCSK9 Inhibitors (Evolocumab, Alirocumab, Inclisiran)": (
            "Evolocumab (Repatha FDA2015): anti-PCSK9 mAb; 140mg SC q2w or 420mg monthly; LDL -55-60%; "
            "Alirocumab (Praluent FDA2015): anti-PCSK9 mAb; 75-150mg SC q2w; LDL -50-60%; "
            "Inclisiran (Leqvio FDA2021): siRNA against PCSK9 mRNA; 284mg SC at 0, 3 months then q6 months; "
            "  hepatic GalNAc delivery; LDL -50%; twice-yearly adherence advantage; "
            "All reduce Lp(a) 20-30% additionally; require some functional LDLR for full effect; "
            "PCSK9 LOF (R46L, Y142X) naturally validates: 28-88% less CVD in carriers"
        ),
        "Evinacumab (Evkeeza)": (
            "Anti-ANGPTL3 monoclonal IgG4; FDA 2021 for homozygous FH ≥12 years; "
            "15 mg/kg IV monthly; LDL reduction ~47% additional on maximal therapy; "
            "UNIQUE: LDLR-INDEPENDENT mechanism → works even in LDLR-null HoFH; "
            "mechanism: ANGPTL3 inhibition → LPL + HL more active → faster VLDL clearance → "
            "less IDL→LDL conversion → LDL production reduced without LDLR; "
            "indicated when statins + ezetimibe + PCSK9i + LDL apheresis insufficient"
        ),
        "Sitosterolemia (ABCG5/ABCG8)": (
            "AR biallelic LOF of ABCG5 (Sterolin-1, type A) or ABCG8 (Sterolin-2, type B); "
            "plasma sitosterol 15-40 mg/dL (normal <0.5 mg/dL); campesterol elevated; "
            "LDL-C may be NORMAL — child with xanthomas + normal LDL = sitosterolemia not FH; "
            "MACROTHROMBOCYTOPENIA + STOMATOCYTES on blood film = key diagnostic clue; "
            "EZETIMIBE HIGHLY EFFECTIVE (blocks NPC1L1 → reduces plant sterol absorption 50-70%); "
            "AVOID plant-sterol-enriched foods (harmful in this condition); "
            "statins + PCSK9i NOT effective for xanthoma regression"
        ),
        "Lysosomal Acid Lipase Deficiency (LAL-D)": (
            "AR biallelic LIPA mutations; two phenotypes by residual LAL activity: "
            "WOLMAN DISEASE (<1% activity): infantile (<3 months); hepatosplenomegaly; "
            "BILATERAL ADRENAL CALCIFICATION on X-ray PATHOGNOMONIC; fatal <12 months untreated; "
            "SEBELIPASE ALFA (Kanuma FDA2015) life-saving (1-3 mg/kg IV weekly); "
            "CESD (>1% activity): childhood-adult hepatomegaly + FH-like LDL elevation + premature CVD; "
            "c.894G>A splice mutation most common CESD; LAL DBS assay diagnostic (simple, non-invasive)"
        ),
        "Lp(a) and LPA Gene": (
            "Lp(a) = LDL-like particle with Apo(a) disulfide-bonded to ApoB-100; "
            "KIV-2 kringle repeats determine Apo(a) size: fewer repeats = smaller Apo(a) = higher Lp(a); "
            "Lp(a) >50 mg/dL: ~1:5 persons; 2x MI risk; >150 mg/dL ≈ HeFH risk; "
            "MEASURE ONCE LIFETIME (stable; fasting not required; isoform-insensitive assay preferred); "
            "NOT on routine lipid panel — order specifically; "
            "STATINS may increase Lp(a) 5-15%; PCSK9i reduce Lp(a) 20-30%; "
            "PELACARSEN (Phase 3): 80-90% Lp(a) reduction; OLPASIRAN (Phase 3): >90%"
        ),
        "ARH (Autosomal Recessive Hypercholesterolemia — LDLRAP1)": (
            "AR biallelic LDLRAP1 LOF; LDLR expressed normally but CANNOT be internalised in hepatocytes; "
            "FIBROBLAST LDLR FUNCTIONAL (misleading — LDLRAP1 not needed in fibroblasts); "
            "LDL 400-700 mg/dL (similar to HoFH but milder); "
            "STATINS PARTIALLY EFFECTIVE (unlike LDLR-null); "
            "PCSK9i suboptimal (LDLR not cycling normally even without PCSK9 degrading it); "
            "LDL apheresis usually required; evinacumab (LDLR-independent) effective; "
            "AR pattern: parents have normal LDL (contrast dominant FH)"
        ),
        "Familial Defective ApoB (FDB — APOB R3500Q)": (
            "AD APOB missense in LDLR-ligand domain; p.Arg3500Gln European founder 1:700-1000; "
            "LDL 250-350 mg/dL (milder than LDLR-FH — LDLR intact in FDB; VLDL cleared normally); "
            "PCSK9i MOST EFFECTIVE (intact LDLR recycled; higher LDLR density compensates defective ApoB binding); "
            "statins more effective in FDB than LDLR-null (functional LDLR upregulated by statin); "
            "tendon xanthomas less frequent than LDLR-FH (lower LDL); "
            "genetic test distinguishes FDB from LDLR-FH — important for therapeutic decisions"
        ),
        "LDL Apheresis": (
            "Physical removal of LDL from plasma; indicated for HoFH, severe HeFH (LDL >300 mg/dL + CHD), "
            "statin-intolerant HeFH, high Lp(a) (some guidelines >100 mg/dL); "
            "reduces LDL 50-70% per session; Lp(a) also removed; "
            "frequency: q1-2 weeks (LDL rebounds between sessions); "
            "available at specialised lipid centres; requires central or good peripheral venous access; "
            "adsorption columns (DEXTRAN SULFATE, HEPARIN EXTRACORPOREAL PRECIPITATION, CASCADE FILTRATION)"
        ),
        "Inclisiran — siRNA PCSK9 Inhibitor": (
            "GalNAc-conjugated siRNA targeting PCSK9 mRNA in hepatocytes; FDA 2021; "
            "administration: 284mg SC at day 1, month 3, then every 6 months; "
            "mechanism: siRNA cleaves PCSK9 mRNA in hepatocyte RISC complex → no PCSK9 secreted; "
            "LDLR constitutively recycled → sustained LDL reduction 50% between doses; "
            "ADVANTAGES: twice-yearly dosing (adherence); no neutralising antibody concern (vs mAb PCSK9i); "
            "SIMILAR EFFICACY to evolocumab/alirocumab; can be combined with either"
        ),
    }

    return {
        "atlas": "Hereditary Familial Hypercholesterolemia Atlas",
        "gene_entries": {g["gene"]: {
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_facts": g["key_facts"],
        } for g in ATLAS_GENES},
        "glossary": glossary,
    }
