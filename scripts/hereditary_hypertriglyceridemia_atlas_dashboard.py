"""Hereditary Hypertriglyceridemia & Familial Chylomicronemia Atlas — 8-Gene Reference
LPL-APOC2-APOA5-GPIHBP1-LMF1-APOC3-ANGPTL3-LIPC
320 patients (8 x 40), seeds 2686-2693.
Endpoints: /api/hereditary-hypertriglyceridemia-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "LPL",
        "protein": (
            "LPL -- 8p21.3 AR -- 475aa -- Lipoprotein-Lipase-53kDa-Homodimer-GPI-Anchored-via-GPIHBP1-"
            "Capillary-Endothelial-Lumen-Hydrolyses-TG-Core-of-Chylomicrons-and-VLDL-"
            "OMIM-Gene-609708-Disease-FCS-238600"
        ),
        "locus": "8p21.3",
        "protein_size": "475 aa / 53 kDa (homodimer, GPI-anchored to capillary endothelium via GPIHBP1)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "LPL (lipoprotein lipase) is the rate-limiting enzyme for TG hydrolysis in peripheral capillaries; "
            "LPL requires APOC2 as obligate cofactor (APOC2 on TRL surface activates LPL catalytic site); "
            "LPL is transported from pericytes/smooth muscle to capillary lumen by GPIHBP1 (GPI-anchored shuttle); "
            "LPL LOF → chylomicrons + VLDL accumulate → severe hypertriglyceridaemia (TG >2000 mg/dL); "
            "MOST COMMON CAUSE of Familial Chylomicronemia Syndrome (FCS): ~70% of genetically confirmed FCS; "
            "PREVALENCE: FCS 1:1,000,000; heterozygous pathogenic LPL variants 1:500 → moderate hypertriglyceridaemia susceptibility; "
            "KEY MUTATIONS: p.Trp86Arg, p.Asp9Asn, p.Asn291Ser (reduced activity), p.Gly188Glu (null); "
            "FOUNDER MUTATIONS: p.Pro207Leu (Dutch); c.3-del/ins9bp in exon 6 (French-Canadian, 1:49 carrier); "
            "POSTHEPARIN LPL ACTIVITY: near-zero in FCS (<10% normal) — postheparin plasma assay diagnostic"
        ),
        "disease_category": (
            "FAMILIAL CHYLOMICRONEMIA SYNDROME (FCS) TYPE I — OMIM 238600; "
            "BIOCHEMICAL HALLMARKS: "
            "  TG >2000 mg/dL (often 5000-20,000 mg/dL) — cream/lactescent plasma; "
            "  STANDING PLASMA TEST: chylomicron cream layer at top of refrigerated plasma — PATHOGNOMONIC; "
            "  LDL-C normal to low (chylomicrons dilute LDL; LDL may be unmeasurably low by Friedewald); "
            "  HDL-C very low (<20 mg/dL in acute attack); "
            "  apoB-48 ELEVATED (chylomicron marker) — diagnostic; "
            "CLINICAL FEATURES: "
            "  ACUTE PANCREATITIS — most feared complication; TG >1000 mg/dL = significant risk; TG >2000 mg/dL = high risk; "
            "  eruptive xanthomas (papular yellow lesions on buttocks/shoulders/elbows); "
            "  lipaemia retinalis (fundoscopic finding — pale retinal vessels); "
            "  hepatosplenomegaly (TG storage in Kupffer cells + spleen macrophages); "
            "  recurrent abdominal pain (even without overt pancreatitis); "
            "  cognitive/mood effects (fatigue, brain fog — mechanism unclear); "
            "  NO INCREASED ATHEROSCLEROTIC CVD RISK (chylomicrons too large to enter arterial wall); "
            "ONSET: typically childhood to young adult; "
            "NBS: NOT detected by standard NBS; can be detected by fasting TG in infancy but not universal; "
            "DIAGNOSIS: severe hypertriglyceridaemia + postheparin LPL activity + LPL sequencing"
        ),
        "disease_pathway": (
            "LIPOPROTEIN LIPASE PATHWAY — TRIGLYCERIDE CLEARANCE: "
            "Dietary fat → intestinal chylomicrons (apoB-48) → lymph → blood; "
            "Liver VLDL (apoB-100) → blood; "
            "NORMAL: LPL (+ APOC2 cofactor, on GPIHBP1 platform at capillary endothelium) "
            "  → hydrolyses TG core → FFA released to muscle/adipose → chylomicron remnants taken up by liver; "
            "LPL DEFICIENCY: "
            "  chylomicrons + VLDL accumulate (cannot be hydrolysed); "
            "  TG >2000 mg/dL → pancreatitis risk (lipase-mediated pancreatic inflammation from FFA release in situ); "
            "TREATMENT RATIONALE: "
            "  ULTRA-LOW FAT DIET (<20g fat/day, <15% calories from fat) — reduces chylomicron substrate; "
            "  medium-chain triglycerides (MCT oil) — absorbed directly without chylomicron formation; "
            "  alipogene tiparvovec (Glybera): gene therapy (AAV1-LPL-S447X); EMA approved 2012, withdrawn 2017 (commercial); "
            "  VOLANESORSEN (Waylivra): antisense APOC3 inhibitor — reduces APOC3 (LPL inhibitor) → partially restores TRL clearance; "
            "    FDA rejected for FCS (bleeding risk); EMA approved 2019 for FCS; platelet monitoring mandatory; "
            "  PEGYLATED APOC2 MIMETIC PEPTIDE (investigational); "
            "  FITUSIRAN (ANGTPL3/4 inhibitor — investigational); "
            "  ALCOHOL AVOID (exacerbates hypertriglyceridaemia); "
            "  FFP TRANSFUSION: provides APOC2 during acute pancreatitis attack (emergency measure, <48h); "
            "  HEPARIN: releases LPL from endothelium transiently — postheparin plasma for LPL assay (diagnostic use only); "
            "  INSULIN (IV): activates LPL in adipose/muscle — useful in DKA-associated hypertriglyceridaemia; "
            "  PLASMAPHERESIS: emergency TG removal for TG >5000 mg/dL + severe pancreatitis"
        ),
        "pathognomonic": (
            "CREAM LAYER (CHYLOMICRONS) ON STANDING PLASMA: "
            "  tube of plasma refrigerated overnight → cream layer floating on top = chylomicrons; "
            "  turbid/lactescent base = VLDL; "
            "  PATHOGNOMONIC for severe chylomicronaemia regardless of cause; "
            "ERUPTIVE XANTHOMAS: "
            "  papular yellow-orange lesions on buttocks/shoulders/elbows; "
            "  appear rapidly with severe hypertriglyceridaemia; "
            "  regress with TG control; "
            "  DISTINCT from tendon xanthomas of FH (which reflect LDL accumulation, not TG); "
            "LIPAEMIA RETINALIS: "
            "  fundoscopy → pale/cream retinal arteries and veins; "
            "  present when TG >3000-4000 mg/dL; "
            "  completely reversible with TG reduction; "
            "POSTHEPARIN LPL ACTIVITY NEAR-ZERO: "
            "  IV heparin 100 U/kg → collect plasma 15 min later → measure LPL activity; "
            "  normal: >150 mU/mL; FCS: <10 mU/mL (often undetectable)"
        ),
        "treatment": (
            "1. ULTRA-LOW FAT DIET: <20g total fat/day (<15% calories) — cornerstone of FCS management; "
            "2. MCT OIL SUPPLEMENT: provides calories without chylomicron formation (medium-chain FA absorbed directly to portal vein); "
            "3. VOLANESORSEN (EMA2019): 285 mg SC weekly → reduces APOC3 (LPL inhibitor) → TG reduction ~70%; "
            "   MANDATORY PLATELET MONITORING: thrombocytopenia in up to 40%; hold if platelets <75,000; "
            "4. FIBRATES (modestly effective in heterozygous LPL, NOT in homozygous FCS where LPL absent); "
            "5. ALCOHOL STRICT AVOIDANCE; "
            "6. ACUTE PANCREATITIS: "
            "   FFP 2-4 units (provides APOC2); insulin infusion (activates residual LPL in heterozygotes); "
            "   plasmapheresis if TG >5000 + organ failure (reduces TG 40-60% per session); "
            "7. PAIN MANAGEMENT: opioids in acute pancreatitis (NSAID avoid in pancreatitis); "
            "8. GENE THERAPY (alipogene tiparvovec — historical, market withdrawn 2017; next-gen AAV trials ongoing)"
        ),
        "seed_base": 2686,
    },
    {
        "gene": "APOC2",
        "protein": (
            "APOC2 -- 19q13.32 AR -- 101aa -- Apolipoprotein-C-II-9kDa-LPL-Obligate-Cofactor-"
            "Exchangeable-Apolipoprotein-TRL-Surface-N-terminal-LPL-Activation-Domain-"
            "OMIM-Gene-608083-Disease-FCS-207750"
        ),
        "locus": "19q13.32",
        "protein_size": "101 aa / 9 kDa (exchangeable apolipoprotein, HDL shuttle, TRL surface during lipolysis)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "APOC2 (apolipoprotein C-II) is the obligate cofactor for LPL activation at the capillary endothelium; "
            "APOC2 resides on HDL in fasting state → transfers to nascent chylomicrons/VLDL in postprandial state; "
            "APOC2 C-terminal domain binds LPL catalytic domain → 10-fold activation of LPL TG hydrolysis; "
            "APOC2 LOF → LPL PRESENT BUT INACTIVE → phenotype identical to LPL deficiency; "
            "DISTINCTIVE DIAGNOSTIC TEST: addition of exogenous APOC2 to postheparin plasma → LPL activity RESTORES; "
            "  (in LPL deficiency: exogenous APOC2 does NOT restore activity — LPL enzyme absent/non-functional); "
            "PREVALENCE: rarest FCS cause; ~<100 families worldwide; "
            "GENE CLUSTER: APOC2 resides in same gene cluster as APOC1, APOE, APOC4 on 19q13.32; "
            "NOTE: APOC3 (on 11q23.3) is a SEPARATE gene and an LPL INHIBITOR (opposite function to APOC2)"
        ),
        "disease_category": (
            "FAMILIAL CHYLOMICRONEMIA SYNDROME TYPE Ib / HYPERLIPOPROTEINEMIA TYPE I — OMIM 207750; "
            "BIOCHEMICAL HALLMARKS: "
            "  TG >2000 mg/dL; cream layer on standing plasma; "
            "  LPL ACTIVITY LOW IN POSTHEPARIN PLASMA (LPL enzyme present but inactive without APOC2); "
            "  APOC2 CORRECTION TEST PATHOGNOMONIC: add exogenous APOC2 → LPL activity normalises; "
            "  APOC2 unmeasurably low or absent on apolipoprotein electrophoresis; "
            "CLINICAL FEATURES (identical to LPL FCS): "
            "  acute pancreatitis; eruptive xanthomas; lipaemia retinalis; hepatosplenomegaly; "
            "  recurrent abdominal pain; cognitive effects; no excess CVD risk; "
            "  childhood onset; "
            "ACUTE ATTACK TREATMENT PEARL: "
            "  FFP TRANSFUSION EFFECTIVE (provides functional APOC2) — more predictably effective than in LPL FCS; "
            "  2-4 units FFP rapidly lowers TG during acute attack; "
            "DISTINGUISHING FCS CAUSES: "
            "  LPL deficiency: postheparin LPL activity low; APOC2 correction test NEGATIVE (LPL non-functional); "
            "  APOC2 deficiency: postheparin LPL activity low; APOC2 correction test POSITIVE (LPL can be activated); "
            "  GPIHBP1 deficiency: postheparin LPL NORMAL (LPL released by heparin); but LPL not stable on endothelium; "
            "  LMF1 deficiency: multiple lipases reduced (LPL + HL); postheparin activities low"
        ),
        "disease_pathway": (
            "LPL ACTIVATION PATHWAY — APOC2 MECHANISM: "
            "Fasting: APOC2 resides on HDL particles (HDL acts as apolipoprotein reservoir); "
            "Postprandial: APOC2 transfers from HDL to nascent chylomicron surface; "
            "LPL mechanism: "
            "  APOC2 C-terminal amphipathic helix (residues 58-78) = LPL activation domain; "
            "  APOC2 binding to LPL → conformational change → opens TG substrate binding site; "
            "  APOC2 LOF: LPL substrate binding site remains closed → TG hydrolysis blocked; "
            "APOC3 COUNTERBALANCE (separate gene, 19q13.32 cluster): "
            "  APOC3 inhibits LPL (independent of APOC2); "
            "  APOC3:APOC2 ratio determines LPL activity in vivo; "
            "  HIGH APOC3 (most common cause of moderate-severe hypertriglyceridaemia in general population); "
            "TREATMENT RATIONALE for APOC2 FCS: "
            "  FFP (fresh frozen plasma): provides circulating APOC2 → transiently activates endogenous LPL; "
            "  APOC2 mimetic peptide (ISIS-APOCIIIRX — investigational): replaces APOC2 function; "
            "  Ultra-low fat diet (same as LPL FCS); "
            "  VOLANESORSEN: suppresses APOC3 → partial LPL activation (APOC3 relief-of-inhibition); "
            "    LESS EFFECTIVE than in LPL FCS (APOC2 still absent → LPL still unactivatable); "
            "NOTE: FIBRATES, OMEGA-3, statins generally INEFFECTIVE for homozygous APOC2 FCS"
        ),
        "pathognomonic": (
            "APOC2 CORRECTION TEST — PATHOGNOMONIC FOR APOC2 FCS: "
            "  Step 1: postheparin plasma from patient → LPL activity near-zero (same as LPL FCS); "
            "  Step 2: add exogenous normal APOC2 (or normal serum) to patient postheparin plasma; "
            "  Step 3: LPL activity RESTORES to normal → CONFIRMS APOC2 deficiency (not LPL deficiency); "
            "  This is the DEFINITIVE FUNCTIONAL DISTINCTION between LPL vs APOC2 FCS; "
            "APOLIPOPROTEIN ELECTROPHORESIS: APOC2 BAND ABSENT; "
            "APOLIPOPROTEIN QUANTIFICATION: APOC2 undetectable by ELISA; "
            "FFP RESPONSE: "
            "  TG falls dramatically (50-90%) within 24-48h of FFP transfusion; "
            "  more reliable TG lowering than in LPL FCS (because LPL is intact and waits for its cofactor)"
        ),
        "treatment": (
            "1. ULTRA-LOW FAT DIET: <20g/day — as for LPL FCS; MCT supplements for calories; "
            "2. ACUTE ATTACK — FFP INFUSION FIRST-LINE (2-4 units): "
            "   FFP provides APOC2 → activates existing LPL → TG hydrolysis resumes; "
            "   repeat every 24-48h until TG < 500 mg/dL; "
            "   more reliably effective than in LPL FCS; "
            "3. VOLANESORSEN (APOC3 antisense): partial TG reduction by removing APOC3-mediated LPL inhibition; "
            "   not as effective as in LPL FCS (APOC2 still absent); "
            "4. PLASMAPHERESIS for severe acute pancreatitis; "
            "5. APOC2 MIMETIC PEPTIDES (investigational); "
            "6. GENE THERAPY (investigational); "
            "7. INSULIN INFUSION (modest benefit — some APOC2-independent LPL activation); "
            "8. FIBRATES, OMEGA-3, STATINS: generally INEFFECTIVE for homozygous APOC2 FCS"
        ),
        "seed_base": 2687,
    },
    {
        "gene": "APOA5",
        "protein": (
            "APOA5 -- 11q23.3 AR(severe)/AD-susceptibility -- 366aa -- Apolipoprotein-A-V-41kDa-"
            "Liver-Secreted-Heparan-Proteoglycan-Binding-LPL-Enhancer-"
            "OMIM-Gene-606368-Disease-HLPV-144650"
        ),
        "locus": "11q23.3",
        "protein_size": "366 aa / 41 kDa (liver-secreted, heparan sulphate proteoglycan-binding, TRL surface)",
        "inheritance": (
            "BIALLELIC LOF → severe FCS/Hyperlipoproteinemia Type V; "
            "HETEROZYGOUS LOF → common hypertriglyceridaemia susceptibility (1.7-fold TG increase per allele); "
            "APOA5 has DUAL MECHANISM OF ACTION: "
            "  (1) activates LPL (similar to APOC2 but less potent; independent binding site on LPL); "
            "  (2) binds heparan sulphate proteoglycans (HSPG) on hepatocyte surface → promotes TRL remnant uptake; "
            "PLASMA CONCENTRATION: only 10-15 µg/mL (extremely low, contrast APOA1 ~100 mg/dL); "
            "  trace amounts suffice for physiological TG regulation; "
            "HIGH-IMPACT COMMON VARIANT: "
            "  APOA5 -1131T>C (rs662799, promoter) — 4-fold higher TG in carriers; 5-10% European frequency; "
            "  c.56G>C (p.Gly19Arg, rs2075291) — severe effect; 4-fold TG increase; common in East Asian (5%); "
            "GENE CLUSTER: APOA5 is embedded in APOA1/C3/A4/A5 gene cluster on 11q23.3; "
            "APOA5 is co-regulated with APOC3 by PPARA (fibrates upregulate APOA5 → explains some fibrate effect); "
            "COMPOUND HETEROZYGOSITY: APOA5 rare + common variant combinations can produce FCS phenotype"
        ),
        "disease_category": (
            "HYPERLIPOPROTEINEMIA TYPE V / FCS TYPE (BIALLELIC) — OMIM 144650; "
            "HETEROZYGOUS: MODERATE HYPERTRIGLYCERIDAEMIA SUSCEPTIBILITY (TG 200-1000 mg/dL with secondary factors); "
            "BIOCHEMICAL HALLMARKS (biallelic): "
            "  TG >2000 mg/dL; VLDL + chylomicrons elevated (Type V lipoprotein pattern = Type I + Type IV); "
            "  Mixed hyperlipidaemia: TG very high + LDL-C low + HDL-C very low; "
            "  TYPE V PHENOTYPE: both chylomicrons (fasting) AND VLDL elevated (distinct from pure Type I = only chylomicrons); "
            "CLINICAL FEATURES: "
            "  acute pancreatitis (high risk at TG >2000); "
            "  eruptive xanthomas + lipaemia retinalis; "
            "  hepatosplenomegaly; abdominal pain; "
            "  secondary factors PRECIPITATE ATTACKS: alcohol, high fat diet, oestrogens (OCP), pregnancy, hypothyroidism, T2DM; "
            "IMPORTANT: Heterozygous APOA5 LOF alone rarely causes FCS — needs secondary precipitants; "
            "POSTHEPARIN PLASMA: LPL activity REDUCED but NOT abolished (unlike biallelic LPL or APOC2); "
            "APOC2 CORRECTION TEST: partial improvement only (not full normalisation); "
            "DIAGNOSIS: TG + APOA5 sequencing; APOA5 level not routinely measured"
        ),
        "disease_pathway": (
            "APOA5 DUAL FUNCTION IN TRL METABOLISM: "
            "1. LPL ACTIVATION: "
            "   APOA5 binds to TRL surface → recruits LPL → enhances TG hydrolysis rate (complementary to APOC2); "
            "   APOA5 binds different site on LPL than APOC2 (can cooperate); "
            "   APOA5 LOF → LPL activation reduced (but APOC2 still present → partial LPL function); "
            "2. HSPG-MEDIATED HEPATIC UPTAKE: "
            "   APOA5 bridges TRL remnants to heparan sulphate proteoglycans on hepatocyte surface; "
            "   promotes LDL receptor-independent TRL remnant clearance; "
            "   APOA5 LOF → remnant accumulation → Type V pattern; "
            "SECONDARY HYPERTRIGLYCERIDAEMIA + APOA5 HETEROZYGOSITY: "
            "  T2DM: insulin resistance → VLDL overproduction → overwhelms residual APOA5 capacity; "
            "  Alcohol: VLDL overproduction + LPL inhibition; "
            "  Oestrogens: increase VLDL secretion (critical during pregnancy/OCP in APOA5 carriers); "
            "  Hypothyroidism: LPL expression reduced; "
            "TREATMENT RATIONALE: "
            "  Fibrates: upregulate APOA5 (PPARα agonist) → partially compensate for APOA5 deficiency; "
            "  Omega-3 (4g/day): reduce VLDL secretion (independent of APOA5); "
            "  Volanesorsen: suppress APOC3 (relieves APOC3 inhibition of LPL); "
            "  Treat secondary factors (OCP switch, alcohol cessation, thyroid replacement)"
        ),
        "pathognomonic": (
            "TYPE V LIPOPROTEIN PHENOTYPE (biallelic APOA5): "
            "  BOTH chylomicrons (cream layer) AND VLDL (turbid base) elevated in fasting plasma; "
            "  Electrophoresis: Type V = chylomicron band + VLDL (pre-beta) band both elevated; "
            "  Contrast Type I (pure chylomicronaemia, cream only, turbid base clears): Type V = BOTH elevated; "
            "APOA5 COMMON VARIANTS — POPULATION SCREENING: "
            "  rs662799 (-1131T>C): detectable on standard lipid gene panels; "
            "  rs2075291 (p.Gly19Arg): elevated TG in East Asian; "
            "  compound heterozygotes (rare + common) → FCS-like TG; "
            "FIBRATE RESPONSE: typically better TG lowering than in biallelic LPL/APOC2 FCS "
            "  (because fibrates upregulate residual APOA5 and enhance LPL expression); "
            "PREGNANCY RISK: "
            "  APOA5 heterozygous women + pregnancy → severe gestational hypertriglyceridaemia; "
            "  pancreatitis of pregnancy (third trimester) — APOA5 carrier diagnosis frequently made here"
        ),
        "treatment": (
            "1. ULTRA-LOW FAT DIET: primary intervention; MCT substitution; "
            "2. FIBRATES (FIRST-LINE for maintenance in APOA5 deficiency): "
            "   fenofibrate 160mg/day or gemfibrozil 600mg BD; "
            "   upregulate APOA5 expression (PPARα target gene) + increase LPL + reduce VLDL; "
            "   MORE EFFECTIVE than in LPL/APOC2 FCS (some residual APOA5 function); "
            "3. OMEGA-3 FATTY ACIDS (4g/day icosapentaenoic acid/EPA or DHA): "
            "   reduce VLDL synthesis (independent mechanism); combine with fibrates; "
            "4. VOLANESORSEN (APOC3 ASO): reduces APOC3-mediated LPL inhibition; "
            "5. TREAT SECONDARY FACTORS: "
            "   DM control (metformin/GLP-1 reduce VLDL); alcohol cessation; hypothyroidism treatment; "
            "   OCP switch (avoid ethinylestradiol → use progestogen-only or IUD); "
            "6. PREGNANCY MANAGEMENT: ultra-low fat diet + insulin (reduces VLDL) + plasma exchange if needed; "
            "7. STATINS: NOT effective for TG component (but add if LDL-C elevated); "
            "8. ACUTE PANCREATITIS: FFP (partial APOC2 support) + plasmapheresis for severe"
        ),
        "seed_base": 2688,
    },
    {
        "gene": "GPIHBP1",
        "protein": (
            "GPIHBP1 -- 8q24.13 AR -- 184aa -- GPI-Anchored-HDL-Binding-Protein-1-22kDa-"
            "Ly6-Domain-LPL-Shuttle-Interstitium-to-Capillary-Lumen-Heparan-Proteoglycan-Binding-"
            "OMIM-Gene-612757-Disease-FCS-615947"
        ),
        "locus": "8q24.13",
        "protein_size": "184 aa / 22 kDa (GPI-anchored to endothelial cells; Ly6/uPAR domain structure)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "GPIHBP1 (GPI-anchored HDL-binding protein 1) is the endothelial platform for LPL anchoring; "
            "GPIHBP1 mechanism: "
            "  GPIHBP1 on capillary endothelial lumen → binds LPL secreted by pericytes/adipocytes → "
            "  transcytoses LPL across endothelium → presents LPL at capillary lumen for TRL lipolysis; "
            "GPIHBP1 LOF → LPL cannot reach capillary lumen → TG-rich lipoproteins cannot be hydrolysed; "
            "DISTINCTIVE FINDING: "
            "  POSTHEPARIN PLASMA LPL ACTIVITY: LPL is RELEASED (heparin displaces LPL from HSPGs) "
            "    and measured NORMALLY in vitro — BUT LPL cannot function in vivo (cannot be transported to lumen); "
            "  (Contrast LPL deficiency: postheparin LPL activity LOW); "
            "  → GPIHBP1 deficiency has NORMAL POSTHEPARIN LPL ACTIVITY with severe in vivo hypertriglyceridaemia; "
            "ANTI-GPIHBP1 AUTOANTIBODIES: "
            "  Acquired FCS caused by IgG anti-GPIHBP1 antibodies (block GPIHBP1-LPL interaction); "
            "  PARANEOPLASTIC or autoimmune aetiology; "
            "  Distinct from hereditary (biallelic) GPIHBP1 deficiency; "
            "  Responds to immunosuppression; plasmapheresis; "
            "PREVALENCE: rare; <100 genetically confirmed hereditary GPIHBP1 FCS families"
        ),
        "disease_category": (
            "FAMILIAL CHYLOMICRONEMIA SYNDROME (FCS) TYPE — OMIM 615947; "
            "BIOCHEMICAL HALLMARKS: "
            "  TG >2000 mg/dL; cream layer; chylomicronaemia; "
            "  POSTHEPARIN LPL ACTIVITY NORMAL (key distinguishing feature from LPL and APOC2 FCS); "
            "  LPL ANTIGEN LEVELS NORMAL in postheparin plasma; "
            "  in vivo LPL function absent (cannot be measured directly without postheparin); "
            "CLINICAL FEATURES: "
            "  identical to LPL FCS: pancreatitis, eruptive xanthomas, lipaemia retinalis, hepatosplenomegaly; "
            "  onset in childhood; "
            "ACQUIRED GPIHBP1 DEFICIENCY (autoantibody): "
            "  DISTINGUISHING FEATURES: adult onset; can occur with lymphoma, solid tumours, or autoimmune disease; "
            "  IgG autoantibodies against GPIHBP1 detected by ELISA or competition assay; "
            "  TREATABLE: immunosuppression (steroids ± rituximab) → autoantibody clearance → TG normalises; "
            "  RAPID DIAGNOSIS IMPORTANT: life-threatening pancreatitis until diagnosed and treated; "
            "TRIGLYCERIDE PARADOX: "
            "  postheparin LPL activity appears normal (LPL released and measured in plasma) → "
            "    lab report seems reassuring → diagnosis missed; "
            "  clinical picture: severe chylomicronaemia → GPIHBP1 sequencing/antibody test"
        ),
        "disease_pathway": (
            "GPIHBP1 — LPL TRANSPORT PLATFORM: "
            "NORMAL LIPOLYSIS CYCLE: "
            "  1. LPL synthesised in pericytes, smooth muscle cells, adipocytes, myocytes; "
            "  2. LPL secreted into interstitial space (pericyte side); "
            "  3. GPIHBP1 (abluminal endothelial surface) captures LPL via Ly6 domain; "
            "  4. GPIHBP1-LPL complex transcytoses to luminal surface; "
            "  5. GPIHBP1 (luminal endothelial surface) presents LPL to passing TRL; "
            "  6. APOC2 on TRL activates LPL → TG hydrolysis → FFA to underlying cells; "
            "  7. LPL dissociates after lipolysis → GPIHBP1 recycles; "
            "GPIHBP1 DEFICIENCY: "
            "  LPL secreted normally by pericytes but cannot be transported to lumen; "
            "  LPL present in interstitium (heparin releases it → postheparin LPL normal); "
            "  TRL pass through capillary lumen without being hydrolysed → chylomicronaemia; "
            "GPIHBP1 Ly6 DOMAIN: "
            "  critical finger-loop structure for LPL binding; "
            "  mutations in Ly6 loops disrupt LPL binding → loss of transport; "
            "ACQUIRED ANTI-GPIHBP1: "
            "  IgG blocks GPIHBP1-LPL interaction → same phenotype as hereditary; "
            "  mechanism: antibody-bound GPIHBP1 cannot bind LPL → LPL stranded in interstitium"
        ),
        "pathognomonic": (
            "NORMAL POSTHEPARIN LPL ACTIVITY + SEVERE CHYLOMICRONAEMIA = PATHOGNOMONIC for GPIHBP1 deficiency: "
            "  TG >2000 mg/dL with cream plasma → expect LPL activity to be low → "
            "  postheparin LPL activity NORMAL → GPIHBP1 cause (not LPL or APOC2); "
            "GPIHBP1 IMMUNOSTAINING OF CAPILLARY BIOPSY (research): "
            "  skeletal muscle biopsy → anti-GPIHBP1 immunostaining → absent or mislocalised GPIHBP1; "
            "ANTI-GPIHBP1 ANTIBODY ELISA: "
            "  positive in acquired (autoimmune) GPIHBP1 FCS; "
            "  negative in hereditary biallelic GPIHBP1 FCS; "
            "RESPONSE TO IMMUNOSUPPRESSION: "
            "  acquired GPIHBP1 → TG normalises within weeks of steroids/rituximab → "
            "  CONFIRMS autoimmune mechanism; "
            "  hereditary GPIHBP1 → no response to immunosuppression"
        ),
        "treatment": (
            "HEREDITARY GPIHBP1 FCS: "
            "1. ULTRA-LOW FAT DIET (<20g/day) — cornerstone; "
            "2. VOLANESORSEN (EMA 2019 for FCS): APOC3 ASO → reduces APOC3 inhibition → partial TRL clearance; "
            "   partially effective (some bypass lipolysis possible via LPL-independent routes or non-GPIHBP1-dependent LPL); "
            "3. FIBRATES (partial effect — upregulate LPL and APOA5 but LPL cannot reach lumen); "
            "4. PLASMAPHERESIS / FFP for acute pancreatitis; "
            "5. MCT oil; alcohol avoidance; treat secondary factors; "
            "ACQUIRED (AUTOIMMUNE) GPIHBP1 FCS: "
            "1. IMMUNOSUPPRESSION FIRST-LINE: "
            "   oral prednisolone 1mg/kg/day → autoantibody suppression → TG normalises; "
            "   RITUXIMAB (anti-CD20): for refractory/steroid-dependent cases; "
            "2. PLASMAPHERESIS: removes autoantibodies acutely during pancreatitis; "
            "3. TREAT UNDERLYING MALIGNANCY if paraneoplastic; "
            "4. MONITOR ANTIBODY TITRES: correlate with TG and clinical status"
        ),
        "seed_base": 2689,
    },
    {
        "gene": "LMF1",
        "protein": (
            "LMF1 -- 16p13.3 AR -- 567aa -- Lipase-Maturation-Factor-1-65kDa-Multipass-ER-Membrane-"
            "Chaperone-for-LPL-and-Hepatic-Lipase-HL-and-Endothelial-Lipase-EL-Maturation-"
            "OMIM-Gene-611761-Disease-Combined-Lipase-Deficiency-246650"
        ),
        "locus": "16p13.3",
        "protein_size": "567 aa / 65 kDa (multi-pass ER transmembrane chaperone protein)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "LMF1 (lipase maturation factor 1) is an endoplasmic reticulum chaperone required for "
            "  correct folding and secretion of all three members of the lipase gene family: "
            "  LPL (lipoprotein lipase), HL (hepatic lipase / LIPC), EL (endothelial lipase / LIPG); "
            "LMF1 LOF → all three lipases misfolded in ER → retained/degraded → not secreted; "
            "COMBINED LIPASE DEFICIENCY — unique phenotype: "
            "  LPL + HL both deficient → TG very high + remnant accumulation + reduced HDL; "
            "  DISTINCT from LPL-only deficiency (pure Type I) → LMF1 also affects HL → remnant-type dyslipidaemia overlaps; "
            "POSTHEPARIN PLASMA: "
            "  LPL activity LOW (as in LPL FCS); "
            "  HL ACTIVITY ALSO LOW — KEY DISTINGUISHING FEATURE; "
            "  (In pure LPL FCS: HL activity is NORMAL postheparin); "
            "ORIGINAL DESCRIPTION: cld (combined lipase deficiency) mouse model identified LMF1 function; "
            "PREVALENCE: very rare; a few dozen families described; "
            "KNOWN MUTATIONS: p.Tyr439* (premature stop, severe); missense mutations throughout gene"
        ),
        "disease_category": (
            "COMBINED LIPASE DEFICIENCY — OMIM 246650; "
            "BIOCHEMICAL HALLMARKS: "
            "  TG very high (>1000-5000 mg/dL); "
            "  POSTHEPARIN LPL + HL BOTH LOW — PATHOGNOMONIC for LMF1 (contrast pure LPL FCS where HL normal); "
            "  HDL-C low (HL normally processes HDL remodelling; HL deficiency → HDL rises paradoxically in isolated HL def); "
            "  Remnant lipoproteins (IDL, chylomicron remnants) elevated (HL also clears remnants); "
            "  MIXED PHENOTYPE: Type I (chylomicronaemia) + Type III (remnant-like) features; "
            "CLINICAL FEATURES: "
            "  acute pancreatitis (TG driven); "
            "  eruptive xanthomas; lipaemia retinalis; hepatosplenomegaly; "
            "  ADDITIONAL HL DEFICIENCY FEATURES: "
            "    premature CVD (remnant particles are highly atherogenic); "
            "    intermediate density lipoprotein (IDL) elevation; "
            "    HDL-C paradoxically ELEVATED in isolated HL deficiency (but in LMF1: LPL deficiency keeps HDL low); "
            "PANCREATITIS RISK: "
            "  HIGH (driven by chylomicronaemia from LPL deficiency component); "
            "  management similar to other FCS causes"
        ),
        "disease_pathway": (
            "LMF1 — LIPASE CHAPERONE FUNCTION: "
            "NORMAL: "
            "  Lipase precursors synthesised in ER; "
            "  LMF1 (ER transmembrane protein) associates with nascent lipase polypeptides; "
            "  LMF1-mediated folding → correctly folded, catalytically active lipase; "
            "  Secretion to post-Golgi and endothelial surface; "
            "LMF1 DEFICIENCY: "
            "  Lipase precursors misfolded in ER → ERAD (ER-associated degradation) → no secretion; "
            "  Affects ALL LMF1-dependent lipases simultaneously: LPL + HL + EL; "
            "THREE LIPASES AFFECTED: "
            "  LPL (major TG hydrolysis at capillaries) → chylomicronaemia; "
            "  HL (hepatic: clears IDL, chylomicron remnants, remodels HDL) → remnant accumulation; "
            "  EL (endothelial lipase: remodels HDL) → HDL composition altered; "
            "POSTHEPARIN ASSAY UTILITY: "
            "  Heparin releases both LPL and HL from HSPGs into plasma; "
            "  LMF1 deficiency: BOTH LPL and HL activities low postheparin; "
            "  Pure LPL FCS: LPL low, HL NORMAL postheparin → key diagnostic discriminator"
        ),
        "pathognomonic": (
            "COMBINED LOW POSTHEPARIN LPL + HL ACTIVITIES — PATHOGNOMONIC for LMF1 deficiency: "
            "  Step 1: collect postheparin plasma (IV heparin 100U/kg, sample at 15 min); "
            "  Step 2: measure LPL activity → LOW; "
            "  Step 3: measure HL activity (trioleate substrate ± 1M NaCl for salt-resistant HL vs salt-sensitive LPL); "
            "    HL activity also LOW in LMF1 deficiency; "
            "    HL activity NORMAL in pure LPL FCS, APOC2 FCS, GPIHBP1 FCS; "
            "  → Combined LPL + HL deficiency = LMF1 until proven otherwise; "
            "POSTHEPARIN HL ASSAY: "
            "  HL = salt-resistant lipase (active at 1M NaCl); LPL = salt-sensitive; "
            "  differential inhibition by salt separates HL from LPL activity in same postheparin plasma; "
            "REMNANT LIPOPROTEIN ELEVATION: "
            "  IDL and chylomicron remnants elevated (from HL deficiency component); "
            "  LPL + HL combined = more severe dyslipidaemia than pure LPL deficiency"
        ),
        "treatment": (
            "1. ULTRA-LOW FAT DIET: primary intervention (reduces chylomicron substrate); "
            "2. MCT OIL: caloric supplement without chylomicron formation; "
            "3. VOLANESORSEN: APOC3 ASO → partial TG reduction; "
            "4. FIBRATES: upregulate LPL and HL gene expression (but LMF1 absent → lipases still misfolded); "
            "   LIMITED BENEFIT in LMF1 FCS; "
            "5. STATIN + OMEGA-3: address remnant accumulation (HL deficiency component) + reduce VLDL; "
            "6. ACUTE PANCREATITIS: plasmapheresis; FFP (limited benefit — LPL enzyme misfolded, not absent cofactor); "
            "7. INVESTIGATIONAL: "
            "   LMF1 gene therapy (in development); "
            "   HL replacement (investigational); "
            "   pharmacological chaperones (to facilitate LMF1-independent lipase folding — research stage)"
        ),
        "seed_base": 2690,
    },
    {
        "gene": "APOC3",
        "protein": (
            "APOC3 -- 19q13.32 AD-LOF-protective/dominant-effect -- 99aa -- Apolipoprotein-C-III-9kDa-"
            "LPL-Inhibitor-Hepatic-TRL-Uptake-Inhibitor-Target-Volanesorsen-Apelimod-"
            "OMIM-Gene-107720-Disease-FHTG-145750"
        ),
        "locus": "19q13.32",
        "protein_size": "99 aa / 9 kDa (exchangeable apolipoprotein; HDL, TRL surface)",
        "inheritance": (
            "NO SINGLE-GENE RECESSIVE PATTERN — complex dominant-like dosage effect; "
            "PATHOLOGICAL HIGH-FUNCTION: common variants and overexpression → elevated APOC3 → severe HTG; "
            "PROTECTIVE LOF: rare heterozygous LOF mutations → reduced CVD risk; "
            "APOC3 DUAL LPL INHIBITORY MECHANISM: "
            "  (1) direct LPL inhibition (binds LPL catalytic domain → reduces TG hydrolysis rate); "
            "  (2) hepatic TRL uptake inhibition (competes with APOE/APOA5-mediated receptor binding); "
            "  HIGH APOC3 → both inhibitory effects → TG accumulates; "
            "APOC3 OVEREXPRESSION CAUSES: "
            "  insulin resistance (APOC3 transcription suppressed by insulin; insulin resistance → high APOC3); "
            "  alcohol (direct transcriptional upregulation of APOC3); "
            "  liver disease (impaired APOC3 clearance); "
            "  APOC3 promoter variants (e.g. 3238C>G, 455T>C, -482C>T — insulin response element disruption); "
            "FAMILIAL COMBINED HYPERLIPIDAEMIA (FCH) LINK: "
            "  APOC3 overexpression is a major contributor to FCH (VLDL + LDL + TG all elevated); "
            "LOF NATURAL EXPERIMENT: "
            "  APOC3 R19X LOF (Amish founder): carriers have TG 40% lower + CVD events 40% lower (Science 2008); "
            "  heterozygous APOC3 LOF = THERAPEUTIC TARGET (volanesorsen mimics this); "
            "GENE CLUSTER: APOC3 on 11q23.3 in APOA1/APOC3/APOA4/APOA5 cluster"
        ),
        "disease_category": (
            "FAMILIAL HYPERTRIGLYCERIDAEMIA (FHTG) — OMIM 145750; SEVERE HTG / FCS MODIFIER; "
            "BIOCHEMICAL HALLMARKS: "
            "  TG moderately-severely elevated (500-2000 mg/dL range typical; can reach FCS levels with secondary triggers); "
            "  VLDL ELEVATED (Type IV lipoprotein pattern most common; Type V if chylomicronaemia also); "
            "  LDL-C normal to low (dilution effect); HDL-C low (APOC3 inhibits HDL remodelling via HL); "
            "  APOC3 LEVELS ELEVATED (measurable by ELISA; normal <10 mg/dL; high in FHTG >15 mg/dL); "
            "CLINICAL SIGNIFICANCE: "
            "  pancreatitis risk at TG >1000 mg/dL; "
            "  ATHEROSCLEROTIC CVD RISK: VLDL remnants ARE atherogenic (unlike chylomicrons); "
            "    APOC3 elevation independently predicts CVD risk (Mendelian randomisation studies); "
            "  most common SEVERE HTG aetiology (combined genetic + secondary factors); "
            "SECONDARY TRIGGERS (compound with APOC3 overexpression): "
            "  insulin resistance / T2DM; alcohol; oestrogens; nephrotic syndrome; hypothyroidism; glucocorticoids"
        ),
        "disease_pathway": (
            "APOC3 — LPL INHIBITOR AND TRL CLEARANCE BLOCKER: "
            "MECHANISM 1 — LPL INHIBITION: "
            "  APOC3 on TRL surface → directly interacts with LPL → reduces TG hydrolysis Vmax; "
            "  Competes with APOC2 (activator) vs APOC3 (inhibitor) — ratio determines LPL activity in vivo; "
            "  High APOC3:APOC2 ratio → net LPL inhibition → TG accumulates; "
            "MECHANISM 2 — HEPATIC UPTAKE BLOCK: "
            "  APOC3 on TRL surface → blocks APOE recognition by LDLR and LDLR-related protein (LRP1); "
            "  APOC3 blocks APOA5-mediated HSPG binding; "
            "  → remnant particle hepatic clearance reduced → IDL and VLDL remnants accumulate; "
            "INSULIN REGULATION: "
            "  Insulin SUPPRESSES APOC3 transcription via FoxO1 → stimulates TRL clearance; "
            "  INSULIN RESISTANCE: FoxO1 constitutively active → APOC3 overexpressed → TG high; "
            "  explains why T2DM / metabolic syndrome → hypertriglyceridaemia; "
            "VOLANESORSEN MECHANISM: "
            "  antisense oligonucleotide (ASO) targeting APOC3 mRNA in liver → reduces APOC3 production → "
            "  LPL inhibition relieved + hepatic TRL uptake restored → TG reduced 70-80% in FCS; "
            "APELIMOD / ISIS-APOCIIIRX: "
            "  second-generation ASO; subcutaneous; monthly dosing (vs weekly volanesorsen); "
            "  REDUCE-IT trial context: EPA independently reduces APOC3"
        ),
        "pathognomonic": (
            "ELEVATED PLASMA APOC3 LEVEL (>15 mg/dL) WITH SEVERE HTG: "
            "  APOC3 measured by nephelometry or ELISA; "
            "  normal: 6-12 mg/dL; FCS/severe FHTG: often 20-40 mg/dL; "
            "VOLANESORSEN RESPONSE: "
            "  dramatic TG reduction (70-80%) within 4-8 weeks = confirms APOC3-driven mechanism; "
            "  superior to fibrates (which modestly reduce APOC3 via PPARα); "
            "PLATELET COUNT FALL WITH VOLANESORSEN: "
            "  APOC3 ASO therapy → immune-mediated thrombocytopenia; "
            "  MANDATORY monitoring: platelets before each injection; "
            "  STOP if platelets <75,000/µL; ABSOLUTE STOP if <50,000/µL; "
            "APOC3 R19X NATURAL LOF (Amish): "
            "  heterozygous carriers have significantly lower TG + HDL higher + CVD reduced; "
            "  natural human proof-of-concept for APOC3 as therapeutic target; "
            "GENETIC TESTING: "
            "  APOC3 promoter variants (insulin response element mutations) in familial FHTG; "
            "  not a simple single-gene disorder — polygenic + secondary factors"
        ),
        "treatment": (
            "1. VOLANESORSEN (Waylivra, EMA 2019 for FCS; FDA rejected for FCS — approved for FHTG with pancreatitis): "
            "   285 mg SC weekly (prefilled syringe); TG reduction 70-80%; "
            "   PLATELET MONITORING MANDATORY (baseline, then every 2 weeks initially); "
            "   CONTRAINDICATED: platelets <140,000/µL at baseline; "
            "2. FIBRATES (fenofibrate, gemfibrozil): PPARα agonists → suppress APOC3 + enhance LPL/APOA5; "
            "   TG reduction 30-50%; first-line oral therapy; "
            "3. OMEGA-3 (EPA: icosapentaenoic acid 4g/day): "
            "   VASCEPA (pure EPA): reduces APOC3 + VLDL secretion; CV benefit (REDUCE-IT trial); "
            "   LOVAZA (EPA+DHA): TG reduction 30-50%; "
            "4. STATIN: reduces cardiovascular risk if LDL-C/remnant cholesterol elevated; "
            "5. ALCOHOL CESSATION (major TG-elevating factor via APOC3 upregulation); "
            "6. GLYCAEMIC CONTROL (metformin/GLP-1/SGLT-2 reduce VLDL + APOC3 expression); "
            "7. SECOND-GEN ASO (apelimod — trials): monthly dosing, fewer platelet effects; "
            "8. DIET: Mediterranean / low carbohydrate (reduce substrate for VLDL secretion)"
        ),
        "seed_base": 2691,
    },
    {
        "gene": "ANGPTL3",
        "protein": (
            "ANGPTL3 -- 1p31.3 AR-LOF-FamCombinedHypolipidemia/AD-GOF-rare -- 460aa -- "
            "Angiopoietin-Like-Protein-3-54kDa-Liver-Secreted-Pan-Lipase-Inhibitor-"
            "Target-Evinacumab-FDA2021-HoFH-"
            "OMIM-Gene-604774-Disease-FamCombinedHypolipidemia-605019"
        ),
        "locus": "1p31.3",
        "protein_size": "460 aa / 54 kDa (liver-secreted; N-terminal coiled-coil, C-terminal fibrinogen-like domain)",
        "inheritance": (
            "AR LOF → Familial Combined Hypolipidaemia (all lipid fractions low); "
            "HETEROZYGOUS LOF → intermediate TG + CVD protection; "
            "ANGPTL3 LOF: pan-lipase inhibitor is abolished → LPL + HL + EL all more active → "
            "  TG low, LDL-C low, HDL-C low (all lipoproteins reduced); "
            "ANGPTL3 GOF (theoretical): would cause hypertriglyceridaemia (inhibits LPL) but rare pathological GOF not well-characterised; "
            "NATURAL HUMAN EXPERIMENT: "
            "  Cohort study (Stitziel 2017 NEJM): ANGPTL3 LOF carriers → TG 27% lower, LDL 9% lower, CVD 34% lower; "
            "  genome-wide association confirms ANGPTL3 locus as major TG-determining gene; "
            "ANGPTL3 INHIBITION AS THERAPY: "
            "  EVINACUMAB (anti-ANGPTL3 monoclonal IgG4): FDA 2021 for HoFH; "
            "    LDL-C reduction ~47% on top of maximal therapy including LDL apheresis; "
            "    MECHANISM UNIQUE: LDLR-independent LDL lowering → works even in LDLR-null HoFH patients; "
            "  VUPANORSEN (ANGPTL3 ASO): TG + LDL reduction; "
            "  ELEBSIRAN (ANGPTL3 siRNA, liver-targeted, monthly dosing): Phase 3 trials; "
            "ANGPTL3 LOF VARIANTS: p.Arg59Ter, p.Ser17Ter (exon 1 truncations); various LOF missense"
        ),
        "disease_category": (
            "FAMILIAL COMBINED HYPOLIPIDAEMIA — OMIM 605019 (biallelic ANGPTL3 LOF); "
            "THERAPEUTIC TARGET (evinacumab) for HOMOZYGOUS FH; "
            "HYPOLIPIDAEMIA PHENOTYPE (biallelic ANGPTL3 LOF): "
            "  TG very low (<40 mg/dL); LDL-C very low (<50 mg/dL); HDL-C low (<25 mg/dL); "
            "  apoB very low; apoA-I low; "
            "  individuals are generally HEALTHY (no clinical disease); "
            "  validates that ANGPTL3 is a safe therapeutic target; "
            "EVINACUMAB FOR HoFH: "
            "  mechanism: evinacumab blocks ANGPTL3 → LPL + HL fully active → VLDL faster clearance → "
            "    LDL precursor VLDL/IDL consumed → LDL production reduced via LDLR-independent route; "
            "  UNIQUE: works in patients with ZERO LDLR ACTIVITY (classic FH maximal therapy + apheresis); "
            "  LDL reduction: ~47% additional (on top of max lipid-lowering therapy); "
            "  FDA approved 12+ years age, monthly IV infusion; "
            "ANGPTL3 HIGH → HYPERTRIGLYCERIDAEMIA LINK: "
            "  elevated ANGPTL3 levels seen in metabolic syndrome, T2DM, hypothyroidism; "
            "  ANGPTL3 inhibits LPL + HL → TG rises with ANGPTL3 overactivity; "
            "  ANGPTL3 pathway therefore relevant to COMMON hypertriglyceridaemia pathogenesis"
        ),
        "disease_pathway": (
            "ANGPTL3 — PAN-LIPASE INHIBITOR: "
            "MECHANISM: "
            "  ANGPTL3 secreted by liver → circulates in blood; "
            "  Inhibits LPL (extrahepatic TG clearance) — independent of APOC3; "
            "  Inhibits HL (hepatic TRL remnant clearance + HDL remodelling); "
            "  Inhibits EL (endothelial lipase; HDL phospholipid hydrolysis); "
            "  Net: inhibits ALL three lipases simultaneously → TG + HDL metabolism both regulated; "
            "ANGPTL3 vs ANGPTL4 vs ANGPTL8: "
            "  ANGPTL3 (liver) + ANGPTL8 (liver/adipose) form complex → potent LPL inhibitor in fasting state; "
            "  ANGPTL4 (adipose, fasting) → inhibits LPL specifically in adipose during fasting; "
            "  Tandem regulation: fasting → ANGPTL3+8 active → conserve TG; feeding → ANGPTL3+8 decrease → LPL active; "
            "EVINACUMAB LDLR-INDEPENDENT LDL LOWERING: "
            "  ANGPTL3 inhibition → enhanced LPL activity → faster VLDL hydrolysis → "
            "  less IDL → less IDL-to-LDL conversion → LDL production reduced; "
            "  AND: ANGPTL3 block → enhanced HL → faster IDL→LDL receptor-independent clearance; "
            "  THEREFORE: works even when LDLR completely absent (HoFH LDLR-null patients); "
            "  This is fundamentally different from statins/PCSK9i (which require some LDLR)"
        ),
        "pathognomonic": (
            "FAMILIAL COMBINED HYPOLIPIDAEMIA (biallelic ANGPTL3 LOF) — ALL LIPIDS LOW: "
            "  TG <40 mg/dL + LDL <50 mg/dL + HDL <25 mg/dL + apoB very low + no CVD = diagnostic signature; "
            "  often found incidentally; benign phenotype; "
            "EVINACUMAB RESPONSE MARKER: "
            "  LDL-C reduction >40% on evinacumab confirms biologically active ANGPTL3 inhibition; "
            "  anti-ANGPTL3 antibody titers monitored (ADA can reduce efficacy); "
            "ANGPTL3 SERUM LEVEL: "
            "  measurable by ELISA (normal 200-400 ng/mL); "
            "  low in LOF carriers; elevated in metabolic syndrome; "
            "ANGPTL3 IN HoFH CONTEXT: "
            "  LDL-C >400 mg/dL + tendon xanthomas + family history + LDLR null variants → "
            "  evinacumab indicated when PCSK9i + apheresis insufficient; "
            "  LDL response confirms absence of ANGPTL3 inhibition prior to treatment"
        ),
        "treatment": (
            "ANGPTL3 LOF (FAMILIAL COMBINED HYPOLIPIDAEMIA): "
            "1. No treatment needed (phenotype is benign very low lipids); "
            "2. Monitor for fat-soluble vitamin deficiency if TG extremely low; "
            "3. No pancreatitis risk (TG very low); "
            "EVINACUMAB FOR HoFH (ANGPTL3 INHIBITION AS THERAPY): "
            "1. EVINACUMAB (Evkeeza, FDA 2021): 15 mg/kg IV monthly; "
            "   INDICATION: HoFH aged ≥12 years as add-on to maximally tolerated lipid-lowering therapy; "
            "   LDL reduction: ~47% additional on top of statins + ezetimibe + PCSK9i; "
            "   UNIQUE BENEFIT: LDL reduction PRESERVED in LDLR-null patients; "
            "2. VUPANORSEN (ANGPTL3 ASO): SC monthly; TG reduction 50-70%; investigational for FHTG; "
            "3. ELEBSIRAN (ANGPTL3 siRNA, GalNAc-conjugated): Phase 3; quarterly dosing; "
            "4. IN HYPERTRIGLYCERIDAEMIA CONTEXT: "
            "   ANGPTL3 inhibitors reduce TG + LDL simultaneously → attractive for mixed dyslipidaemia; "
            "   fibrates + omega-3 partially reduce ANGPTL3 activity indirectly"
        ),
        "seed_base": 2692,
    },
    {
        "gene": "LIPC",
        "protein": (
            "LIPC -- 15q22.1 AR -- 499aa -- Hepatic-Lipase-53kDa-GPI-Anchored-Liver-Sinusoidal-Endothelium-"
            "Hydrolyses-IDL-Remnants-HDL-Phospholipids-TG-in-IDL-and-LDL-"
            "OMIM-Gene-151670-Disease-HepaticLipaseDeficiency-614025"
        ),
        "locus": "15q22.1",
        "protein_size": "499 aa / 53 kDa (GPI-anchored to liver sinusoidal endothelium; salt-resistant lipase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF mutations; "
            "LIPC (hepatic lipase / HL) is expressed exclusively in liver parenchyma/sinusoidal endothelium; "
            "HL has DUAL FUNCTION: "
            "  (1) TRIGLYCERIDASE: hydrolyses TG in IDL (chylomicron remnants + VLDL remnants) → LDL conversion; "
            "  (2) PHOSPHOLIPASE: remodels HDL2 → HDL3 (HDL maturation cycle); "
            "HL LOF → IDL/remnant accumulation + large buoyant HDL2 (not converted to HDL3); "
            "POSTHEPARIN PLASMA: "
            "  HL activity LOW or absent; LPL activity NORMAL (distinguishes from LMF1 combined deficiency); "
            "  SALT-RESISTANT LIPASE ASSAY: HL activity measured at 1M NaCl (HL active, LPL inhibited by salt); "
            "  HEPATIC LIPASE ACTIVITY < 10% normal in biallelic LIPC deficiency; "
            "COMMON LIPC PROMOTER VARIANT: "
            "  LIPC -514C>T (rs1800588): reduces HL activity 20-30%; "
            "  homozygous TT: HDL-C elevated; remnant accumulation susceptibility; "
            "  NOT a disease-causing variant alone; modulates TG/HDL in population"
        ),
        "disease_category": (
            "HEPATIC LIPASE DEFICIENCY — OMIM 614025; "
            "BIOCHEMICAL HALLMARKS: "
            "  TG MODERATELY elevated (300-2000 mg/dL — generally less severe than LPL FCS, pancreatitis rarer); "
            "  IDL/REMNANT LIPOPROTEINS ELEVATED — ATHEROGENIC; "
            "  HDL-C ELEVATED (large HDL2 — paradoxically high because HL normally converts HDL2→HDL3); "
            "  LDL-C normal to elevated (buoyant large LDL predominates); "
            "  apoE-rich HDL particles (apoE-HDL intermediate) — characteristic; "
            "  'MIXED HYPERLIPIDAEMIA' — TG + LDL-C both elevated; "
            "  LIPOPROTEIN ELECTROPHORESIS: Type III-like (remnant accumulation) or Type IV/V; "
            "CLINICAL FEATURES: "
            "  PREMATURE ATHEROSCLEROSIS / CVD — major concern (IDL/remnants are atherogenic); "
            "  xanthelasma (not typical tendon xanthomas — IDL deposits vs LDL deposits); "
            "  palmar xanthomas (IDL type — planar orange discolouration in palm creases); "
            "  pancreatitis risk LESS than LPL FCS (TG usually lower); "
            "  corneal arcus if young"
        ),
        "disease_pathway": (
            "HEPATIC LIPASE — IDL CLEARANCE AND HDL REMODELLING: "
            "NORMAL HL FUNCTION: "
            "  1. IDL CLEARANCE: HL at liver sinusoidal endothelium → hydrolyses TG/phospholipids in IDL → "
            "     IDL converted to LDL (LDLR-mediated uptake) OR directly taken up by LRP1; "
            "  2. HDL REMODELLING: HL hydrolyses HDL2-phospholipids → smaller HDL3; "
            "     HDL cycle: HDL3 (nascent ApoA-I) → acquires cholesterol → HDL2 (large) → "
            "     HL converts HDL2→HDL3 → cycle continues; "
            "     CETP (cholesteryl ester transfer protein) also remodels HDL → VLDL transfer; "
            "HL DEFICIENCY: "
            "  IDL NOT converted to LDL → IDL accumulates (Type III-like remnant hyperlipidaemia); "
            "  HDL2 NOT converted to HDL3 → HDL-C rises paradoxically; "
            "  apoE-rich large HDL particles accumulate; "
            "DISTINGUISHING FROM FH (Type IIb/IIa): "
            "  FH: LDL-C very high; remnants present if also FCH; tendon xanthomas; LDLR mutations; "
            "  HL deficiency: IDL elevated + large HDL; palmar xanthomas; LIPC mutations; normal LPL; "
            "DISTINGUISHING FROM TYPE III (APOE ε2/ε2): "
            "  Type III: APOE2 homozygous; palmar and tuberoeruptive xanthomas; beta-VLDL by electrophoresis; "
            "  HL deficiency: APOE genotype normal; HL activity low; LIPC mutations; "
            "TREATMENT: "
            "  fibrates (reduce VLDL/IDL production + enhance LPL to partially compensate); "
            "  statins (reduce LDL and IDL precursor production); "
            "  omega-3 (reduce VLDL); "
            "  HL gene therapy (investigational)"
        ),
        "pathognomonic": (
            "LOW POSTHEPARIN HL ACTIVITY + NORMAL LPL ACTIVITY — PATHOGNOMONIC for isolated HL deficiency: "
            "  SALT-RESISTANT LIPASE (HL at 1M NaCl) LOW or absent; "
            "  SALT-SENSITIVE LIPASE (LPL at 0M NaCl) NORMAL (distinguishes from LMF1 where BOTH low); "
            "ELEVATED LARGE BUOYANT HDL2 + IDL: "
            "  HDL-C paradoxically HIGH (often >70-80 mg/dL) despite TG elevation; "
            "  IDL band on lipoprotein electrophoresis (pre-beta VLDL + intermediate zone); "
            "  apoE-rich HDL on 2D gel electrophoresis; "
            "PALMAR XANTHOMAS: "
            "  orange-yellow planar lipid deposits in palm creases; "
            "  same finding as Type III (APOE2/2) — inspect palms in all mixed hyperlipidaemia; "
            "  APOE genotype distinguishes: APOE ε2/ε2 in Type III; normal APOE in HL deficiency; "
            "POSTHEPARIN HL ASSAY: "
            "  HL activity < 10% normal + normal LPL = isolated HL deficiency = LIPC sequencing indicated"
        ),
        "treatment": (
            "1. FIBRATES (fenofibrate/gemfibrozil): "
            "   reduce VLDL/IDL production (PPARα) + upregulate LPL (partial compensation for absent HL); "
            "   TG reduction 30-50%; first-line for TG-driven symptoms; "
            "2. STATINS (FIRST-LINE for cardiovascular risk reduction): "
            "   reduce LDL precursors (VLDL/IDL) + LDL-C → atheroprotective; "
            "   atorvastatin/rosuvastatin + fibrate combination (monitor CK — rhabdomyolysis risk with gemfibrozil+statin); "
            "   PREFER: fenofibrate + statin (lower rhabdomyolysis risk vs gemfibrozil + statin); "
            "3. OMEGA-3 (4g/day): reduce VLDL production; complementary to fibrate + statin; "
            "4. EZETIMIBE: blocks intestinal cholesterol absorption; reduce LDL/IDL substrate; "
            "5. LDL APHERESIS (if severely elevated IDL + refractory CVD despite maximal therapy); "
            "6. PALMAR/XANTHELASMA: improve with TG + IDL control; laser/surgery rarely needed; "
            "7. DIET: low saturated fat + low refined carbohydrate + low alcohol; "
            "8. CARDIOVASCULAR RISK MANAGEMENT: aspirin, BP control (IDL-driven atherosclerosis); "
            "9. HL REPLACEMENT / GENE THERAPY (investigational)"
        ),
        "seed_base": 2693,
    },
]


def _generate_patients(gene_idx: int, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    gene_data = ATLAS_GENES[gene_idx]
    gene = gene_data["gene"]

    # Gene-specific clinical parameters
    gene_params = {
        "LPL":     {"tg_range": (2000, 18000), "lpl_low": True, "hl_low": False, "has_pancreatitis": 0.75,
                    "has_xanthoma": 0.65, "has_cvd": 0.08, "onset_range": (5, 25), "ffp_response": 0.45},
        "APOC2":   {"tg_range": (1800, 12000), "lpl_low": True, "hl_low": False, "has_pancreatitis": 0.70,
                    "has_xanthoma": 0.60, "has_cvd": 0.08, "onset_range": (8, 30), "ffp_response": 0.90},
        "APOA5":   {"tg_range": (500, 8000), "lpl_low": False, "hl_low": False, "has_pancreatitis": 0.50,
                    "has_xanthoma": 0.40, "has_cvd": 0.22, "onset_range": (15, 50), "ffp_response": 0.30},
        "GPIHBP1": {"tg_range": (1500, 10000), "lpl_low": False, "hl_low": False, "has_pancreatitis": 0.72,
                    "has_xanthoma": 0.62, "has_cvd": 0.09, "onset_range": (5, 30), "ffp_response": 0.35},
        "LMF1":    {"tg_range": (1200, 9000), "lpl_low": True, "hl_low": True, "has_pancreatitis": 0.68,
                    "has_xanthoma": 0.55, "has_cvd": 0.25, "onset_range": (5, 25), "ffp_response": 0.30},
        "APOC3":   {"tg_range": (400, 6000), "lpl_low": False, "hl_low": False, "has_pancreatitis": 0.35,
                    "has_xanthoma": 0.25, "has_cvd": 0.42, "onset_range": (20, 60), "ffp_response": 0.15},
        "ANGPTL3": {"tg_range": (20, 80), "lpl_low": False, "hl_low": False, "has_pancreatitis": 0.00,
                    "has_xanthoma": 0.00, "has_cvd": 0.03, "onset_range": (0, 0), "ffp_response": 0.00},
        "LIPC":    {"tg_range": (200, 2000), "lpl_low": False, "hl_low": True, "has_pancreatitis": 0.20,
                    "has_xanthoma": 0.30, "has_cvd": 0.60, "onset_range": (30, 65), "ffp_response": 0.10},
    }

    params = gene_params[gene]
    patients = []

    for i in range(n):
        tg = rng.randint(*params["tg_range"])
        onset = rng.randint(*params["onset_range"]) if params["onset_range"][1] > 0 else 0
        hdl = rng.randint(10, 30) if tg > 500 else rng.randint(35, 90)
        ldl = rng.randint(30, 80) if tg > 1500 else rng.randint(70, 180)

        patients.append({
            "patient_id": f"{gene}-{seed:04d}-{i+1:02d}",
            "gene": gene,
            "tg_mg_dL": tg,
            "hdl_c_mg_dL": hdl,
            "ldl_c_mg_dL": ldl,
            "postheparin_lpl_low": params["lpl_low"],
            "postheparin_hl_low": params["hl_low"],
            "pancreatitis_episode": rng.random() < params["has_pancreatitis"],
            "eruptive_xanthoma": rng.random() < params["has_xanthoma"],
            "lipaemia_retinalis": tg > 3000 and rng.random() < 0.55,
            "cvd_event": rng.random() < params["has_cvd"],
            "ffp_response": rng.random() < params["ffp_response"],
            "onset_years": onset,
            "volanesorsen_candidate": gene in ("LPL", "APOC2", "APOC3", "APOA5", "GPIHBP1", "LMF1"),
            "diet_controlled": tg < 1000 and rng.random() < 0.60,
        })

    return patients


def generate_overview():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2686 + idx))

    summary = {}
    for p in all_patients:
        g = p["gene"]
        if g not in summary:
            summary[g] = {
                "gene": g, "n": 0,
                "pancreatitis_n": 0, "xanthoma_n": 0, "cvd_n": 0,
                "lipaemia_n": 0, "lpl_low_n": 0, "hl_low_n": 0,
                "mean_tg": 0.0, "mean_onset": 0.0,
                "postheparin_lpl_low": p["postheparin_lpl_low"],
                "postheparin_hl_low": p["postheparin_hl_low"],
            }
        s = summary[g]
        s["n"] += 1
        s["pancreatitis_n"] += int(p["pancreatitis_episode"])
        s["xanthoma_n"] += int(p["eruptive_xanthoma"])
        s["cvd_n"] += int(p["cvd_event"])
        s["lipaemia_n"] += int(p["lipaemia_retinalis"])
        s["lpl_low_n"] += int(p["postheparin_lpl_low"])
        s["hl_low_n"] += int(p["postheparin_hl_low"])
        s["mean_tg"] += p["tg_mg_dL"]
        s["mean_onset"] += p["onset_years"]

    gene_summaries = []
    for g, s in summary.items():
        n = s["n"]
        gene_summaries.append({
            "gene": g,
            "n": n,
            "pancreatitis_pct": round(100 * s["pancreatitis_n"] / n, 1),
            "eruptive_xanthoma_pct": round(100 * s["xanthoma_n"] / n, 1),
            "cvd_event_pct": round(100 * s["cvd_n"] / n, 1),
            "lipaemia_retinalis_pct": round(100 * s["lipaemia_n"] / n, 1),
            "postheparin_lpl_low": s["postheparin_lpl_low"],
            "postheparin_hl_low": s["postheparin_hl_low"],
            "mean_tg_mg_dL": round(s["mean_tg"] / n, 1),
            "mean_onset_years": round(s["mean_onset"] / n, 1),
        })

    return {
        "atlas": "Hereditary Hypertriglyceridemia & Familial Chylomicronemia Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": len(all_patients),
        "seeds": list(range(2686, 2694)),
        "gene_summaries": gene_summaries,
        "fcs_classification": [
            "FCS TYPE I — LPL deficiency (most common FCS, ~70%): postheparin LPL low; APOC2 correction negative; "
            "cream plasma; pancreatitis risk very high; NO excess CVD",
            "FCS TYPE Ib — APOC2 deficiency: postheparin LPL low; APOC2 CORRECTION TEST POSITIVE; "
            "FFP acutely effective; cream plasma; pancreatitis risk very high",
            "FCS — APOA5 biallelic: postheparin LPL reduced not absent; Type V phenotype (VLDL+chylomicrons); "
            "fibrates more effective; secondary triggers amplify severity",
            "FCS — GPIHBP1 deficiency: postheparin LPL NORMAL (LPL released by heparin but cannot function in vivo); "
            "DISTINGUISHING PEARL; acquired autoimmune form with anti-GPIHBP1 Ab",
            "COMBINED LIPASE DEFICIENCY — LMF1: postheparin LPL + HL BOTH low; "
            "concurrent IDL accumulation from HL deficiency; more CVD risk than pure FCS",
            "FAMILIAL HYPERTRIGLYCERIDAEMIA — APOC3: common; TG 500-2000; CVD risk (VLDL remnants atherogenic); "
            "volanesorsen first ASO approved for FCS/severe FHTG; platelet monitoring mandatory",
            "FAMILIAL COMBINED HYPOLIPIDAEMIA — ANGPTL3 LOF: TG + LDL + HDL all very low; benign; "
            "validates ANGPTL3 as therapeutic target; evinacumab FDA2021 for HoFH (LDLR-independent mechanism)",
            "HEPATIC LIPASE DEFICIENCY — LIPC: postheparin HL low + LPL normal; IDL + large HDL2 elevated; "
            "CVD risk high (IDL atherogenic); palmar xanthomas; distinguish from Type III (APOE genotype normal)",
        ],
        "critical_distinctions": [
            "POSTHEPARIN LPL LOW: LPL, APOC2, LMF1 — NOT GPIHBP1",
            "POSTHEPARIN HL LOW: LMF1 (both LPL + HL) — NOT LPL, APOC2, APOA5, GPIHBP1",
            "APOC2 CORRECTION TEST POSITIVE: APOC2 deficiency specifically",
            "POSTHEPARIN LPL NORMAL + CHYLOMICRONAEMIA: GPIHBP1 deficiency",
            "FFP MOST EFFECTIVE: APOC2 deficiency (LPL intact, just needs its cofactor)",
            "NO EXCESS CVD: LPL, APOC2, APOA5 FCS (chylomicrons too large to be atherogenic)",
            "HIGH CVD RISK: LIPC, ANGPTL3-target HoFH, APOC3 (IDL/remnants are atherogenic)",
            "VOLANESORSEN PLATELET MONITORING MANDATORY: thrombocytopenia in up to 40%",
        ],
    }


def generate_breakdown():
    breakdown = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2686 + idx)
        n = len(patients)
        panc_n = sum(1 for p in patients if p["pancreatitis_episode"])
        xanth_n = sum(1 for p in patients if p["eruptive_xanthoma"])
        cvd_n = sum(1 for p in patients if p["cvd_event"])
        lipaem_n = sum(1 for p in patients if p["lipaemia_retinalis"])
        ffp_n = sum(1 for p in patients if p["ffp_response"])
        vol_cand_n = sum(1 for p in patients if p["volanesorsen_candidate"])
        mean_tg = sum(p["tg_mg_dL"] for p in patients) / n
        mean_hdl = sum(p["hdl_c_mg_dL"] for p in patients) / n
        mean_ldl = sum(p["ldl_c_mg_dL"] for p in patients) / n

        breakdown.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"][:300],
            "disease_category": gene_data["disease_category"][:400],
            "pathognomonic": gene_data["pathognomonic"][:400],
            "treatment_summary": gene_data["treatment"][:400],
            "patients_n": n,
            "pancreatitis_pct": round(100 * panc_n / n, 1),
            "eruptive_xanthoma_pct": round(100 * xanth_n / n, 1),
            "cvd_event_pct": round(100 * cvd_n / n, 1),
            "lipaemia_retinalis_pct": round(100 * lipaem_n / n, 1),
            "ffp_response_pct": round(100 * ffp_n / n, 1),
            "volanesorsen_candidate_pct": round(100 * vol_cand_n / n, 1),
            "mean_tg_mg_dL": round(mean_tg, 1),
            "mean_hdl_c_mg_dL": round(mean_hdl, 1),
            "mean_ldl_c_mg_dL": round(mean_ldl, 1),
            "postheparin_lpl_low": gene_data["seed_base"] in (2686, 2687, 2690),
            "postheparin_hl_low": gene_data["seed_base"] == 2690 or gene_data["seed_base"] == 2693,
            "sample_patients": patients[:5],
        })

    return {"atlas": "Hereditary Hypertriglyceridemia & Familial Chylomicronemia Atlas", "breakdown": breakdown}


def generate_definitions():
    glossary = {
        "Familial Chylomicronemia Syndrome (FCS)": (
            "Severe monogenic hypertriglyceridaemia (TG >2000 mg/dL) due to inability to hydrolyse TRL; "
            "causes: LPL (most common), APOC2, APOA5 (biallelic), GPIHBP1, LMF1; "
            "cream layer plasma; pancreatitis risk; NO excess CVD (chylomicrons too large for artery wall); "
            "treatment: ultra-low fat diet + volanesorsen (EMA2019) ± plasmapheresis"
        ),
        "Postheparin LPL Activity": (
            "IV heparin 100U/kg → plasma collected 15 min later → LPL activity measured (trioleate hydrolysis); "
            "normal >150 mU/mL; FCS (LPL, APOC2, LMF1): <10 mU/mL; "
            "GPIHBP1 FCS: NORMAL postheparin (LPL released into plasma but cannot function in vivo); "
            "APOC2 correction test: add normal APOC2 → if LPL restores = APOC2 deficiency; if not = LPL deficiency"
        ),
        "APOC2 Correction Test": (
            "Diagnostic: patient postheparin plasma + exogenous APOC2 (or normal serum) → measure LPL activity; "
            "POSITIVE (LPL restores) = APOC2 deficiency — LPL enzyme is normal but lacks its cofactor; "
            "NEGATIVE (LPL does not restore) = LPL deficiency — enzyme itself is absent/non-functional; "
            "key step before gene sequencing to identify cause of FCS"
        ),
        "Volanesorsen (Waylivra)": (
            "Antisense oligonucleotide (ASO) targeting APOC3 mRNA in liver → reduces APOC3 production 70-80%; "
            "EMA 2019 approved for FCS (TG >880 mg/dL + dietary restriction insufficient); "
            "FDA rejected (bleeding risk from thrombocytopenia); "
            "285 mg SC weekly; "
            "MANDATORY: platelet monitoring (weekly x 6 months, then monthly); "
            "HOLD if platelets <75,000; STOP if <50,000; "
            "reduces TG 70-80% and pancreatitis episodes"
        ),
        "Evinacumab (Evkeeza)": (
            "Monoclonal IgG4 antibody against ANGPTL3; FDA 2021 for homozygous FH (HoFH); "
            "15 mg/kg IV monthly; LDL-C reduction ~47% additional; "
            "UNIQUE MECHANISM: LDLR-independent LDL lowering — works even in LDLR-null patients; "
            "rationale: ANGPTL3 inhibition → enhanced LPL + HL → faster VLDL/IDL clearance → less LDL production; "
            "for patients with HoFH failing statins + PCSK9i + LDL apheresis"
        ),
        "Alipogene Tiparvovec (Glybera)": (
            "First approved gene therapy in Western world (EMA 2012, for LPL FCS); "
            "AAV1 vector expressing LPL-S447X (gain-of-function LPL variant); "
            "single IM injection series; transient TG reduction (6-12 months peak effect); "
            "withdrawn from market 2017 (commercial reasons, not safety); "
            "price: ~1 million EUR/treatment; next-gen LPL gene therapies in development"
        ),
        "Lipoprotein Type I / Type V Phenotype": (
            "Type I (Fredrickson): chylomicrons only elevated; TG >2000 mg/dL; "
            "cream layer at top of standing plasma; turbid base CLEARS (no VLDL); "
            "LPL, APOC2, GPIHBP1 FCS; "
            "Type V: chylomicrons AND VLDL both elevated; "
            "cream layer at top + turbid base (DOES NOT CLEAR); "
            "APOA5 biallelic FCS or severe secondary HTG with FCS; "
            "electrophoresis: Type V = chylomicron band + pre-beta (VLDL) band both present"
        ),
        "GPIHBP1 Autoantibody (Acquired FCS)": (
            "IgG autoantibodies against GPIHBP1 prevent GPIHBP1-LPL interaction → "
            "  same phenotype as hereditary GPIHBP1 deficiency; "
            "ACQUIRED FCS: adult onset; associated with lymphoma, solid tumours, or autoimmune disease; "
            "diagnosis: GPIHBP1 antibody ELISA (positive in acquired, negative in hereditary); "
            "treatment: immunosuppression (steroids ± rituximab) → autoantibody clearance → TG normalises; "
            "plasmapheresis for acute pancreatitis; treat underlying malignancy"
        ),
        "Eruptive Xanthomas": (
            "Small papular yellow-orange skin lesions on buttocks, shoulders, elbows; "
            "appear suddenly with severe hypertriglyceridaemia (TG >2000 mg/dL); "
            "foam cells laden with TG-rich lipoproteins in dermis; "
            "regress rapidly with TG control; "
            "DISTINCT from tendon xanthomas (FH — LDL-laden); "
            "DISTINCT from palmar xanthomas (Type III / HL deficiency — IDL-laden, orange palm creases)"
        ),
        "Hepatic Lipase (HL) — Salt-Resistant Assay": (
            "Postheparin plasma contains both LPL (salt-sensitive) and HL (salt-resistant); "
            "HL assay: measure lipase activity at 1M NaCl (LPL inhibited, HL active) → HL activity; "
            "LPL assay: total activity at 0M NaCl minus activity at 1M NaCl = LPL-specific activity; "
            "LMF1 deficiency: BOTH HL and LPL low postheparin — combined lipase defect; "
            "Pure HL deficiency (LIPC): HL low + LPL NORMAL postheparin"
        ),
        "Familial Combined Hypolipidaemia (ANGPTL3 LOF)": (
            "Biallelic ANGPTL3 LOF → all lipid fractions very low (TG <40, LDL <50, HDL <25 mg/dL); "
            "clinically benign; no pancreatitis (TG very low); no CVD (LDL very low); "
            "VALIDATES ANGPTL3 as safe therapeutic target; "
            "evinacumab (anti-ANGPTL3 mAb) mimics this state pharmacologically for HoFH treatment; "
            "ANGPTL3 LOF carriers (heterozygous): TG 27% lower, CVD 34% lower (Mendelian randomisation evidence)"
        ),
        "Palmar Xanthomas": (
            "Orange-yellow planar lipid deposits in palm creases; "
            "pathognomonic for IDL/remnant accumulation; "
            "causes: (1) Type III hyperlipidaemia (APOE ε2/ε2) — most common; "
            "  (2) Hepatic lipase deficiency (LIPC) — differentiate by APOE genotype; "
            "  (3) Sitosterolaemia (ABCG5/8) — phytosterol deposits; "
            "inspect palms in ALL patients with mixed hyperlipidaemia (TG + LDL both elevated)"
        ),
        "FFP (Fresh Frozen Plasma) in Hypertriglyceridaemia": (
            "FFP provides functional APOC2 (cofactor for LPL) and other apolipoproteins; "
            "MOST EFFECTIVE in APOC2 FCS: provides missing cofactor → LPL activates → TG falls 50-90% in 24-48h; "
            "USEFUL (less dramatic) in LPL FCS: provides TRL-processing apolipoproteins; "
            "DOSE: 2-4 units repeated every 24-48h until TG < 500 mg/dL; "
            "EMERGENCY MEASURE for acute pancreatitis; "
            "volumes limit use (up to 20 mL/kg per infusion); "
            "superseded by plasmapheresis for very high TG in severe pancreatitis"
        ),
        "APOC3 vs APOC2 — Opposing Functions": (
            "APOC2 (19q13.32, 101 aa): ACTIVATES LPL — obligate cofactor; biallelic LOF → FCS; "
            "APOC3 (11q23.3, 99 aa): INHIBITS LPL + blocks hepatic TRL uptake — LPL antagonist; "
            "  high APOC3 → severe HTG (most common severe HTG mechanism); "
            "  LOF APOC3 → low TG + protective CVD (natural Mendelian randomisation); "
            "APOC3:APOC2 RATIO determines net LPL activity in vivo; "
            "therapeutic targeting: APOC3 inhibition (volanesorsen ASO) = relief from inhibition = net LPL activation"
        ),
    }

    return {
        "atlas": "Hereditary Hypertriglyceridemia & Familial Chylomicronemia Atlas",
        "gene_entries": {g["gene"]: {
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
        } for g in ATLAS_GENES},
        "glossary": glossary,
    }
