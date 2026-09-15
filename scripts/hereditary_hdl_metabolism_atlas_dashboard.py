"""Hereditary HDL-Metabolism Atlas — 8-Gene Reference
ABCA1-APOA1-LCAT-LIPC-CETP-APOE-SCARB1-LIPG
320 patients (8 x 40), seeds 2774-2781.
Endpoints: /api/hereditary-hdl-metabolism-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ABCA1",
        "protein": (
            "ABCA1 -- 9q31.1 AR/AD -- 2058aa -- ATP-Binding-Cassette-Transporter-A1-254kDa-"
            "Cholesterol-Phospholipid-Efflux-to-ApoA-I-HDL-Biogenesis-Rate-Limiting-Step-"
            "OMIM-Gene-600046-Disease-Tangier-205400-FHA-604091"
        ),
        "locus": "9q31.1",
        "protein_size": "2058 aa / 254 kDa (12-TM ABC transporter; two NBDs; flippase cholesterol+phospholipid from inner to outer leaflet; rate-limiting for nascent HDL disc formation)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (Tangier disease — biallelic null/severe LOF) or "
            "AUTOSOMAL DOMINANT (Familial hypoalphalipoproteinemia / FHA — monoallelic LOF; HDL-C 10th-percentile); "
            "ABCA1 (ATP-binding cassette transporter A1) is the MASTER RATE-LIMITING STEP for HDL biogenesis; "
            "ABCA1 transfers free cholesterol + phospholipids from cell inner leaflet to lipid-poor ApoA-I → forms nascent discoidal HDL (pre-β-HDL); "
            "Without ABCA1, ApoA-I cannot acquire lipid → ApoA-I is rapidly cleared by kidneys → plasma HDL-C → near zero; "
            "TANGIER DISEASE (biallelic): HDL-C <5 mg/dL; ApoA-I <3 mg/dL; "
            "  cholesteryl ester accumulates in macrophages (reticuloendothelial system) of tonsils, intestinal mucosa, peripheral nerves, liver, spleen; "
            "FHA (monoallelic): HDL-C 10-40 mg/dL; 2-4x premature CAD risk; no Tangier-specific signs; "
            "EXPRESSION: ubiquitous; HIGHEST in liver and macrophages; liver is key for RCT efflux; "
            "REGULATION: LXR (liver X receptor) + PPAR-γ transcriptional activation; statins may upregulate ABCA1 via LXR pathway"
        ),
        "disease_category": (
            "TANGIER DISEASE (biallelic) — OMIM 205400; "
            "BIOCHEMICAL HALLMARKS: "
            "  HDL-C <5 mg/dL (often undetectable); ApoA-I <3 mg/dL (should be 100-180 mg/dL); "
            "  TG mildly elevated (impaired HDL-TG exchange); LDL-C LOW or normal (reduced CETP activity); "
            "CLINICAL FEATURES — PATHOGNOMONIC: "
            "  ORANGE/YELLOW TONSILS — cholesteryl ester-laden macrophage foam cells; PATHOGNOMONIC for Tangier disease; "
            "  enlarged orange tonsils visible on inspection; may require tonsillectomy for obstruction; "
            "  PERIPHERAL NEUROPATHY: demyelinating + axonal; relapsing-remitting or progressive; "
            "    mononeuropathy multiplex pattern OR syringomyelia-like (dissociated sensory loss); "
            "    nerve biopsy: lipid-laden macrophages in endoneurium (DIAGNOSTIC); "
            "  HEPATOSPLENOMEGALY: foam cell accumulation; usually mild; "
            "  PREMATURE CAD: moderate risk (less than anticipated given very low HDL due to low LDL-C); "
            "  CORNEAL INFILTRATES: lipid deposits (less prominent than LCAT deficiency); "
            "  RECTAL MUCOSA: orange-streaked (foam cell infiltration — diagnostic on sigmoidoscopy); "
            "FAMILIAL HYPOALPHALIPOPROTEINEMIA (monoallelic): "
            "  HDL-C low but not absent; 2-4x premature CAD; no peripheral neuropathy; no orange tonsils; "
            "DIAGNOSTIC PATHWAY: "
            "  1. HDL-C <5 mg/dL → suspect Tangier; "
            "  2. Tonsil appearance (orange/yellow if enlarged) + peripheral neuropathy → strong clinical diagnosis; "
            "  3. ABCA1 sequencing (2 pathogenic alleles = Tangier; 1 = FHA); "
            "  4. Cholesterol efflux assay (fibroblasts/macrophages from patient → minimal efflux to ApoA-I)"
        ),
        "disease_pathway": (
            "REVERSE CHOLESTEROL TRANSPORT (RCT) — ABCA1 INITIATION STEP: "
            "NORMAL RCT: "
            "  1. Macrophage/hepatocyte ABCA1 activated by LXR (oxysterols accumulate when cholesterol excess); "
            "  2. ABCA1 flips phospholipids (PC, SM) + free cholesterol to outer leaflet → membrane blebbing; "
            "  3. Lipid-poor ApoA-I binds membrane bleb → acquires lipid → forms nascent discoidal pre-β-HDL; "
            "  4. LCAT esterifies free cholesterol on nascent HDL → cholesteryl ester → HDL matures to spherical α-HDL; "
            "  5. Mature HDL acquires more cholesterol via ABCG1 (macrophage) or SR-BI (bidirectional); "
            "  6. SR-BI on hepatocyte selectively uptakes CE from HDL → liver for bile/redistribution; "
            "  7. CETP transfers CE from HDL to LDL/VLDL (exchanges for TG) → LDL pathway to liver; "
            "ABCA1 DEFICIENCY: "
            "  ApoA-I cannot acquire lipid → lipid-poor ApoA-I cleared by renal filtration (t½ <1hr vs normal 5 days); "
            "  No HDL formed → CE accumulates in macrophages (no RCT) → foam cells in tonsils/nerves/liver/spleen; "
            "  LDL-C paradoxically LOW (less CETP-mediated CE transfer to LDL); "
            "TREATMENT: no FDA-approved disease-modifying therapy; "
            "  niacin historically used (modest HDL raise) but CVD benefit not proven; "
            "  CSL112 (reconstituted ApoA-I infusion) — Phase III trials for acute coronary syndrome; "
            "  Dietary modification (low fat) reduces intestinal foam cell burden; "
            "  Tonsillectomy if airway obstruction from foam-cell-laden tonsils; "
            "CASCADE TESTING: HDL-C in first-degree relatives; monoallelic carriers = FHA risk"
        ),
        "pathognomonic": (
            "ORANGE/YELLOW TONSILS — PATHOGNOMONIC FOR TANGIER DISEASE: "
            "  Caused by cholesteryl ester-laden macrophage foam cells in tonsillar tissue; "
            "  Classic appearance: orange or yellow-orange discolouration of enlarged palatine tonsils; "
            "  Visible on oropharyngeal inspection; NOT present in other hypoalphalipoproteinemia causes; "
            "  Even post-tonsillectomy patients — orange appearance on remaining tissue/adenoids; "
            "RECTAL MUCOSA FOAM CELLS: "
            "  Orange-streaked rectal mucosa on sigmoidoscopy — diagnostic; "
            "  Only disease with both orange tonsils + rectal foam cells + <5 mg/dL HDL + neuropathy; "
            "KEY DISTINGUISHING FEATURES: "
            "  Tangier vs LCAT deficiency: Tangier = orange tonsils + neuropathy; LCAT = corneal haze + anemia + proteinuria; "
            "  Tangier vs ApoA-I deficiency: Tangier = orange tonsils + AR; ApoA-I = AD + severe corneal + xanthoma; "
            "  Tangier vs Fish-eye disease: Fish-eye = PARTIAL LCAT deficiency (lecithin-cholesterol esterification in HDL absent, VLDL/LDL normal)"
        ),
        "key_facts": [
            "ABCA1-ORANGE-TONSILS-PATHOGNOMONIC",
            "ABCA1-HDL-NEAR-ZERO-TANGIER",
            "ABCA1-NEUROPATHY-RELAPSING",
            "ABCA1-LDL-LOW-PARADOX",
            "ABCA1-FHA-MONOALLELIC-CAD",
            "ABCA1-LXR-REGULATED",
            "ABCA1-RCT-MASTER-INITIATOR",
            "ABCA1-CHOLESTEROL-EFFLUX-ASSAY",
        ],
        "treatment": (
            "Tangier disease: no disease-modifying FDA-approved therapy; "
            "niacin (modest HDL-C increase, no proven CVD benefit); "
            "CSL112 reconstituted ApoA-I infusion (Phase III acute coronary syndrome trials); "
            "Dietary: low saturated fat reduces foam-cell burden; "
            "Tonsillectomy for obstructive foam-cell tonsils; "
            "Neuropathy: supportive (physiotherapy, pain management); "
            "CASCADE: HDL-C in all first-degree relatives; monoallelic → FHA counselling"
        ),
        "seed_base": 2774,
        "n_patients": 40,
    },
    {
        "gene": "APOA1",
        "protein": (
            "APOA1 -- 11q23.3 AD -- 267aa -- Apolipoprotein-A-I-28kDa-Major-HDL-Structural-Protein-"
            "LCAT-Activator-RCT-Acceptor-Lipid-Poor-ApoA-I-ABCA1-Substrate-"
            "OMIM-Gene-107680-Disease-ApoA-I-Deficiency-107680"
        ),
        "locus": "11q23.3",
        "protein_size": "267 aa / 28 kDa (exchangeable apolipoprotein; amphipathic helices; LCAT activator; primary structural protein of HDL; ABCA1 substrate for nascent HDL)",
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous loss-of-function (ApoA-I deficiency/hypoalphalipoproteinemia); "
            "biallelic null → complete ApoA-I deficiency (severe, rare); "
            "APOA1 (Apolipoprotein A-I) is the MAJOR HDL STRUCTURAL PROTEIN (70-80% of HDL protein mass); "
            "ApoA-I is the obligate substrate for ABCA1-mediated lipid efflux — lipid-poor ApoA-I accepts phospholipid + cholesterol; "
            "ApoA-I is the COFACTOR/ACTIVATOR for LCAT (lecithin-cholesterol acyltransferase); "
            "Without ApoA-I: no HDL formation; no LCAT activation; no RCT; "
            "GENE CLUSTER: APOA1-APOC3-APOA4-APOA5 cluster on 11q23.3; "
            "TANDEM MUTATIONS: APOA1 inversions/deletions can simultaneously knock out APOC3 → "
            "  paradoxical TG reduction (APOC3 LOF) + HDL-C absent (APOA1 LOF); "
            "HETEROZYGOUS: HDL-C 20-40 mg/dL; 2-4x premature CAD risk; "
            "BIALLELIC: HDL-C <5 mg/dL; severe premature CAD; corneal opacities; planar xanthomas"
        ),
        "disease_category": (
            "APOA-I DEFICIENCY — OMIM 107680 (same locus, deficiency = rare AR subtype); "
            "BIOCHEMICAL HALLMARKS: "
            "  Heterozygous: HDL-C 20-40 mg/dL; ApoA-I 40-80 mg/dL; "
            "  Biallelic null: HDL-C <5 mg/dL; ApoA-I undetectable; "
            "  TG may be elevated (impaired TG-rich lipoprotein clearance via APOC2/ApoA-V); "
            "CLINICAL FEATURES (biallelic): "
            "  PREMATURE SEVERE CAD: coronary artery disease in 3rd-4th decade; "
            "  CORNEAL OPACITIES: arcus corneae + diffuse stromal haziness; "
            "  PLANAR XANTHOMAS: flat yellowish skin plaques (foam cell deposits); NOT tendon xanthomas; "
            "    xanthomatosis involves xanthomas at pressure points + eyelids + creases; "
            "  HEPATOSPLENOMEGALY: mild, from RES foam cell accumulation; "
            "  NO peripheral neuropathy (distinguishes from Tangier/ABCA1 deficiency); "
            "  NO orange tonsils (distinguishes from Tangier disease); "
            "VARIANT: APOA1 MILANO (Arg173Cys) — Italian founder mutation; "
            "  heterozygotes: very low HDL-C yet NO excess CAD (paradox); "
            "  cysteine allows homodimer formation → may enhance cholesterol efflux differently; "
            "  ETC-216 (recombinant ApoA-I Milano) — Phase II trials showed plaque regression; "
            "  ApoA-I Milano paradox: demonstrates HDL function > HDL quantity for cardioprotection"
        ),
        "disease_pathway": (
            "APOA1 AS RCT SCAFFOLD AND LCAT ACTIVATOR: "
            "NORMAL: "
            "  1. Liver + intestine secrete lipid-poor ApoA-I (pre-β-HDL); "
            "  2. ApoA-I binds ABCA1 on macrophages → acquires free cholesterol + phospholipid → nascent discoidal HDL; "
            "  3. ApoA-I activates LCAT: LCAT esterifies free cholesterol → CE → buried in HDL core → spherical HDL-3 → HDL-2; "
            "  4. ApoA-I also binds ABCG1 (macrophage) and SR-BI (bidirectional) for additional lipid loading; "
            "  5. Mature HDL delivers CE to liver via SR-BI selective uptake; "
            "  6. ApoA-I recycled after CE delivery → resecretion for further RCT cycles; "
            "APOA1 DEFICIENCY: "
            "  No ApoA-I → LCAT has no substrate → no CE formation → no mature HDL; "
            "  Free cholesterol accumulates in plasma → deposits in cornea (lipid keratopathy) and skin (xanthomas); "
            "  APOA1 MILANO PARADOX: "
            "    Arg173Cys → Cys-Cys homodimer formation via disulfide bond; "
            "    Homodimer has FASTER LIPID EFFLUX from macrophages despite lower steady-state HDL-C; "
            "    Demonstrates that RCT FLUX (cholesterol efflux capacity) > HDL-C concentration for protection; "
            "THERAPEUTIC IMPLICATION: "
            "  HDL mimetics: synthetic ApoA-I peptides (4F, 5A) enhance cholesterol efflux; "
            "  CSL112: full-length recombinant ApoA-I (Phase III AEGIS-II trial); "
            "  Reconstituted HDL infusions: acute plaque stabilisation in ACS"
        ),
        "pathognomonic": (
            "APOA-I MILANO PARADOX — KEY TEACHING POINT: "
            "  Italian village of Limone sul Garda: 40 carriers, Arg173Cys founder mutation; "
            "  HDL-C extremely low (10-20 mg/dL) but NO excess cardiovascular events vs population; "
            "  Demonstrates: HDL FUNCTION (efflux capacity) not HDL-C level determines protection; "
            "  KEY DDx APOA-I DEFICIENCY vs TANGIER DISEASE: "
            "  ApoA-I deficiency: AD inheritance; corneal haze; xanthomas; NO orange tonsils; NO neuropathy; "
            "  Tangier disease: AR; orange tonsils PATHOGNOMONIC; neuropathy; NO xanthomas; "
            "  KEY DDx vs LCAT DEFICIENCY: "
            "  LCAT def: corneal haze + anemia + proteinuria; ApoA-I def: corneal haze + xanthomas + no anemia; "
            "GENE CLUSTER NOTE: "
            "  APOA1 deletion inversions simultaneously eliminate APOC3 (hyperTG suppressor) → "
            "  low HDL + normal/low TG (APOC3 LOF counteracts) = diagnostic clue"
        ),
        "key_facts": [
            "APOA1-LCAT-ACTIVATOR-OBLIGATE",
            "APOA1-MILANO-PARADOX-LOW-HDL-NO-CAD",
            "APOA1-CORNEAL-OPACITIES-XANTHOMAS",
            "APOA1-NO-ORANGE-TONSILS-DDX-TANGIER",
            "APOA1-ABCA1-SUBSTRATE",
            "APOA1-GENE-CLUSTER-APOC3-APOA4-APOA5",
            "APOA1-BIALLELIC-SEVERE-CAD",
            "APOA1-CSL112-PHASE-III",
        ],
        "treatment": (
            "No disease-modifying FDA-approved therapy for ApoA-I deficiency; "
            "CSL112 (reconstituted ApoA-I, Phase III AEGIS-II trial); "
            "HDL mimetic peptides (4F, 5A) in preclinical/early clinical; "
            "Strict cardiovascular risk reduction: statins, aspirin, smoking cessation; "
            "Corneal opacities: ophthalmology monitoring; "
            "CASCADE: HDL-C + ApoA-I in first-degree relatives"
        ),
        "seed_base": 2775,
        "n_patients": 40,
    },
    {
        "gene": "LCAT",
        "protein": (
            "LCAT -- 16q22.1 AR -- 440aa -- Lecithin-Cholesterol-Acyltransferase-50kDa-"
            "Esterifies-Free-Cholesterol-on-HDL-ApoA-I-Activated-CE-Core-Formation-"
            "OMIM-Gene-606967-Disease-FLD-245900-FED-136120"
        ),
        "locus": "16q22.1",
        "protein_size": "440 aa / 50 kDa (serine esterase; ApoA-I-activated; transfers sn-2 fatty acid from PC to free cholesterol → CE; buried in HDL core driving HDL maturation)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "LCAT (lecithin-cholesterol acyltransferase) is the SOLE ENZYME THAT ESTERIFIES FREE CHOLESTEROL ON HDL; "
            "LCAT mechanism: activated by ApoA-I on HDL surface; transfers sn-2 acyl from phosphatidylcholine (lecithin) to "
            "  free cholesterol → cholesteryl ester (CE) → CE buried in hydrophobic core → HDL disc → sphere; "
            "Without LCAT: free cholesterol accumulates on HDL surface → unstable nascent discoidal HDL → rapid clearance; "
            "  HDL-C → very low; plasma free cholesterol ELEVATED (abnormal distribution); "
            "TWO PHENOTYPES depending on residual activity: "
            "  FAMILIAL LCAT DEFICIENCY (FLD): complete LCAT deficiency — affects both HDL and VLDL/LDL esterification; "
            "  FISH-EYE DISEASE (FED): partial LCAT deficiency — esterifies VLDL/LDL but NOT HDL; "
            "  FLD = severe; FED = milder (corneal only, no renal disease); "
            "PREVALENCE: very rare; <200 FLD cases reported worldwide; FED: similarly rare"
        ),
        "disease_category": (
            "FAMILIAL LCAT DEFICIENCY (FLD) — OMIM 245900; "
            "FISH-EYE DISEASE (FED) — OMIM 136120; "
            "BIOCHEMICAL HALLMARKS: "
            "  HDL-C very low (<10 mg/dL in FLD; <20 mg/dL in FED); "
            "  ApoA-I low (rapid turnover without CE core); "
            "  FREE CHOLESTEROL elevated in plasma (unesterified); "
            "  LIPOPROTEIN-X (Lp-X): abnormal vesicular lipoprotein — phospholipid-rich, CE-poor; present in FLD; "
            "  TG elevated (impaired VLDL remodelling); "
            "CLINICAL FEATURES — FLD (complete): "
            "  CORNEAL HAZE: diffuse grayish stromal opacity; arcus-like at periphery; starts in infancy/childhood; "
            "    bilateral, symmetric; slit-lamp: fine greyish dots in stroma (PATHOGNOMONIC appearance); "
            "  HEMOLYTIC ANEMIA: normochromic normocytic; red cell membranes enriched in unesterified cholesterol → "
            "    abnormal membrane fluidity → haemolysis; target cells + stomatocytes on smear; "
            "  PROTEINURIA → RENAL FAILURE: Lp-X and free cholesterol deposit in glomerular mesangium → "
            "    foam cell glomerulonephritis → progressive CKD → ESRD (if untreated); "
            "    renal biopsy: foam cells in mesangium + subendothelial deposits; "
            "  SPLENOMEGALY: mild foam cell accumulation; "
            "FISH-EYE DISEASE (partial): "
            "  CORNEAL HAZE ONLY (no anemia, no renal disease — partial esterification preserved for VLDL/LDL); "
            "  Name: from the opaque, whitish-grey corneal appearance resembling a cooked fish eye"
        ),
        "disease_pathway": (
            "LCAT IN HDL MATURATION AND RCT: "
            "NORMAL: "
            "  1. Nascent discoidal HDL (from ABCA1+ApoA-I) has free cholesterol on outer leaflet; "
            "  2. LCAT (circulates in plasma; docks on HDL via ApoA-I helix 6) is activated; "
            "  3. LCAT: phosphatidylcholine → sn-2 fatty acid cleaved → esterifies free cholesterol on HDL; "
            "  4. CE formed → hydrophobic → migrates to HDL core → HDL disc becomes sphere; "
            "  5. Spherical HDL-3 → HDL-2 (mature) → CE delivered to liver via SR-BI; "
            "  6. LCAT also active on VLDL/LDL (using ApoC-I activation) → systemic cholesterol esterification; "
            "LCAT DEFICIENCY: "
            "  Free cholesterol on lipoprotein surface cannot be esterified; "
            "  Unstable discoidal HDL is rapidly cleared → very low HDL-C; "
            "  Free cholesterol accumulates in cell membranes (red cells → haemolysis; glomerulus → Lp-X deposits); "
            "  Lp-X formation: free cholesterol + phospholipid form abnormal vesicular structure in plasma → "
            "    nephrotoxic; deposits in mesangium; corneal deposition; "
            "TREATMENT: "
            "  LCAT ENZYME REPLACEMENT THERAPY (ERT): recombinant LCAT (MEDI6012 / rLCAT): "
            "    Phase II trials: normalises cholesterol esterification; reduces corneal opacity; "
            "    reduces Lp-X (renal protection); "
            "  Renal transplant: Lp-X redeposits in transplanted kidney without ERT — concurrent ERT required; "
            "  Plasma infusion: temporary LCAT replacement; "
            "  Dietary: low fat reduces free cholesterol load"
        ),
        "pathognomonic": (
            "CORNEAL HAZE + HEMOLYTIC ANEMIA + PROTEINURIA — TRIAD OF FLD: "
            "  All three together strongly suggest complete LCAT deficiency; "
            "  CORNEAL HAZE: slit-lamp shows fine gray dots in entire stroma (vs ABCA1/ApoA-I = arcus-like peripheral); "
            "    FISH-EYE DISEASE: dense central whitish haze; name literally from fish-eye appearance; "
            "  HEMOLYTIC ANEMIA: target cells + stomatocytes + spherocytes on smear; "
            "    DAT negative (not immune-mediated) — distinguishes from autoimmune haemolysis; "
            "  LPX ON LIPOPROTEIN ELECTROPHORESIS: abnormal Lp-X band at cathode (very slow migrating); "
            "    Lp-X = phospholipid vesicles (albumin inside) — PATHOGNOMONIC for LCAT deficiency + cholestasis; "
            "  FLD vs FED: "
            "    FLD = cornea + anemia + proteinuria (all three); FED = cornea ONLY (partial enzyme preserves HDL-E); "
            "  KEY DDx: "
            "    LCAT def vs Tangier: Tangier = orange tonsils + neuropathy; LCAT = anemia + proteinuria + Lp-X; "
            "    LCAT def vs ApoA-I def: ApoA-I = xanthomas (no anemia/proteinuria); LCAT = no xanthomas"
        ),
        "key_facts": [
            "LCAT-SOLE-HDL-CE-ESTERIFICATION-ENZYME",
            "LCAT-FLD-CORNEA-ANEMIA-PROTEINURIA-TRIAD",
            "LCAT-FISH-EYE-DISEASE-PARTIAL-CORNEA-ONLY",
            "LCAT-LPX-PATHOGNOMONIC",
            "LCAT-RENAL-FOAM-CELL-GLOMERULONEPHRITIS",
            "LCAT-ERT-MEDI6012-PHASE-II",
            "LCAT-APOA1-ACTIVATED",
            "LCAT-DAT-NEGATIVE-HEMOLYSIS",
        ],
        "treatment": (
            "MEDI6012 recombinant LCAT enzyme replacement (Phase II trials); "
            "Renal transplant + concurrent rLCAT (Lp-X redeposits in transplanted kidney without ERT); "
            "Plasma infusion (temporary LCAT replacement); "
            "Dietary low saturated fat (reduce free cholesterol load); "
            "Corneal grafting (for visual impairment); "
            "Anaemia: folic acid supplementation; transfusion if severe"
        ),
        "seed_base": 2776,
        "n_patients": 40,
    },
    {
        "gene": "LIPC",
        "protein": (
            "LIPC -- 15q21.3 AR -- 476aa -- Hepatic-Lipase-53kDa-Liver-Anchored-Phospholipase-Triglyceride-Lipase-"
            "HDL-Remodelling-IDL-Remnant-Clearance-ApoE-Interaction-"
            "OMIM-Gene-151670-Disease-HL-Deficiency-614025"
        ),
        "locus": "15q21.3",
        "protein_size": "476 aa / 53 kDa (member of triglyceride lipase family; GPI-anchored on hepatocyte surface; phospholipase + TG-lipase activity; remodels HDL-2 to HDL-3; clears IDL remnants)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; rare; "
            "LIPC (hepatic lipase / HL) is a LIVER-ANCHORED LIPOLYTIC ENZYME with dual function: "
            "  1. PHOSPHOLIPASE A1: hydrolyses phospholipids on HDL-2 → converts HDL-2 (large, buoyant) → HDL-3 (small, dense); "
            "  2. TRIGLYCERIDE LIPASE: hydrolyses TG in IDL/remnant particles → facilitates remnant-receptor clearance; "
            "LIPC DEFICIENCY: "
            "  HDL-C markedly elevated (cannot catabolise HDL-2 → accumulates); "
            "  IDL/VLDL remnants elevated (cannot clear TG core → IDL accumulates); "
            "  LDL phenotype: large, buoyant LDL (unusual); "
            "  TRIGLYCERIDE ELEVATED: from IDL/remnant accumulation; "
            "EXPRESSION: liver-specific; "
            "INTERACTION: HL cooperates with LPL (endothelial-anchored, peripheral tissues) and EL (LIPG, endothelial); "
            "GENETIC VARIANTS: common promoter variant -514C→T (reduced HL activity, race-specific allele frequency); "
            "  -514T allele: higher HDL-C; found in populations with 30-50% lower HL activity"
        ),
        "disease_category": (
            "HEPATIC LIPASE DEFICIENCY — OMIM 614025; "
            "BIOCHEMICAL HALLMARKS: "
            "  HDL-C markedly elevated (HDL-2 accumulates — large buoyant particles): HDL-C 60-100+ mg/dL; "
            "  ApoA-I elevated (carried on accumulated large HDL-2); "
            "  TG elevated (IDL remnant accumulation); "
            "  IDL elevated on lipoprotein electrophoresis (broad beta band); "
            "  LDL: large, buoyant, cholesterol-rich; "
            "CLINICAL FEATURES: "
            "  PREMATURE CORONARY ARTERY DISEASE: despite elevated HDL-C (dysfunctional large HDL-2 cannot complete RCT cycle); "
            "    HIGH HDL-C does NOT protect if HDL is enlarged/dysfunctional; "
            "  XANTHOMAS: tuberous or eruptive xanthomas (from IDL/remnant accumulation); "
            "  PANCREATITIS: if TG very elevated (>1000 mg/dL); "
            "  NO orange tonsils; NO peripheral neuropathy; NO corneal haze; "
            "  HEPATIC LIPASE ACTIVITY ASSAY: plasma PHLA (post-heparin lipolytic activity) — HL-specific component absent; "
            "    Post-heparin plasma: heparin releases HL from hepatocyte surface → measurable in plasma; "
            "    LIPC-deficient patients: normal LPL activity post-heparin but absent HL component; "
            "PARADOX: high HDL-C with CAD — demonstrates that HDL-2 accumulation (dysfunctional) is not protective; "
            "  HDL must be functional (able to deliver CE to liver via RCT) not just abundant"
        ),
        "disease_pathway": (
            "HEPATIC LIPASE IN HDL REMODELLING AND REMNANT CLEARANCE: "
            "NORMAL HDL REMODELLING: "
            "  1. Peripheral tissues: ABCA1/ABCG1 efflux CE to nascent HDL (ApoA-I) → HDL matures; "
            "  2. CETP: transfers CE from HDL to TG-rich lipoproteins (VLDL/IDL) in exchange for TG; "
            "  3. HDL-2 (CE-rich, TG-enriched from CETP) → hepatic lipase action: "
            "     a. HL phospholipase: removes phospholipid corona from HDL-2; "
            "     b. HL TG-lipase: removes CETP-acquired TG from HDL core; "
            "     c. Result: HDL-2 → HDL-3 (smaller, denser) → released for another RCT cycle; "
            "     d. SR-BI simultaneously extracts CE from HDL to liver; "
            "NORMAL IDL CLEARANCE: "
            "  VLDL → IDL (LPL action removes TG core); "
            "  HL removes remaining TG + phospholipid from IDL → facilitates ApoE-mediated LDLR binding → clearance; "
            "  LIPC deficiency: IDL cannot be cleared → IDL/remnant accumulation → cardiovascular risk; "
            "LIPC DEFICIENCY: "
            "  HDL-2 cannot be remodelled to HDL-3 → large HDL-2 accumulates → high HDL-C but dysfunctional; "
            "  IDL remnants cannot be cleared → TG elevated; "
            "  No FDA-approved therapy; treat cardiovascular risk factors aggressively; "
            "  Fibrates (reduce VLDL production → less IDL formation); "
            "  Post-heparin PHLA assay: diagnostic — absent HL activity with normal LPL"
        ),
        "pathognomonic": (
            "HIGH HDL-C + HIGH TG + IDL ELEVATION + PREMATURE CAD — LIPC DEFICIENCY PATTERN: "
            "  High HDL-C paradoxically NOT protective (dysfunctional, enlarged HDL-2 cannot complete RCT); "
            "  POST-HEPARIN PLASMA DIAGNOSTIC: "
            "    Heparin IV → releases HL from liver heparan sulfate proteoglycans into plasma; "
            "    Measure TG-lipase activity: LPL component (inhibited by 1M NaCl) vs HL component (salt-stable); "
            "    LIPC deficiency: HL-specific lipolytic activity absent; LPL normal; "
            "  LIPOPROTEIN ELECTROPHORESIS: "
            "    Broad beta band (IDL accumulation) — similar to Type III HLP (ApoE2/E2) "
            "    but LIPC patients have HIGH HDL while ApoE2/E2 have LOW HDL; "
            "  KEY DDx: "
            "    LIPC def vs CETP def: CETP = very high HDL but normal TG; LIPC = high HDL + high TG + IDL; "
            "    LIPC def vs Type III HLP: Type III = low HDL; LIPC = HIGH HDL"
        ),
        "key_facts": [
            "LIPC-HIGH-HDL-HIGH-TG-DYSFUNCTIONAL",
            "LIPC-HDL2-CANNOT-BE-REMODELLED",
            "LIPC-IDL-REMNANT-ACCUMULATION",
            "LIPC-POST-HEPARIN-PHLA-DIAGNOSTIC",
            "LIPC-PREMATURE-CAD-HIGH-HDL-PARADOX",
            "LIPC-TG-LIPASE-PHOSPHOLIPASE-DUAL",
            "LIPC-IDL-BROAD-BETA-ELECTROPHORESIS",
            "LIPC-FIBRATES-REDUCE-IDL",
        ],
        "treatment": (
            "No FDA-approved disease-modifying therapy; "
            "Fibrates (reduce VLDL production → less IDL accumulation); "
            "Omega-3 fatty acids (reduce TG); "
            "Statins for LDL-associated risk; "
            "Low-fat diet (reduces IDL substrate); "
            "Aggressive CVD risk factor management; "
            "Post-heparin PHLA for diagnostic confirmation"
        ),
        "seed_base": 2777,
        "n_patients": 40,
    },
    {
        "gene": "CETP",
        "protein": (
            "CETP -- 16q21 AD -- 493aa -- Cholesteryl-Ester-Transfer-Protein-53kDa-"
            "Transfers-CE-from-HDL-to-VLDL-IDL-LDL-Exchanges-for-TG-HDL-CE-Catabolism-"
            "OMIM-Gene-118470-Disease-CETP-Deficiency-143470"
        ),
        "locus": "16q21",
        "protein_size": "493 aa / 53 kDa (lipid transfer protein; boomerang structure; transfers CE from HDL to ApoB-containing lipoproteins in exchange for TG; central mediator of HDL-VLDL crosstalk)",
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous (partial CETP deficiency) or biallelic LOF (complete CETP deficiency); "
            "CETP (cholesteryl ester transfer protein) transfers CE from HDL → VLDL/IDL/LDL in exchange for TG; "
            "CETP is the MAIN CATABOLISM PATHWAY for HDL-CE: ~40-70% of HDL-CE transferred to ApoB lipoproteins; "
            "CETP LOF: "
            "  CE cannot be transferred out of HDL → CE accumulates in HDL → HDL-C VERY HIGH; "
            "  Less CE in LDL → LDL-C may be modestly low; "
            "  CE-enriched, TG-depleted HDL → large buoyant HDL-1 / α-HDL particles; "
            "JAPANESE FOUNDER MUTATION: Asp442Gly (exon 15) — most common CETP mutation worldwide; "
            "  allele frequency 7% in Japanese population; ~1:200 homozygotes; "
            "  heterozygotes: HDL-C 65-75 mg/dL; homozygotes: HDL-C >100 mg/dL; "
            "  hyperalphalipoproteinemia (HALP) = CETP deficiency phenotype; "
            "POPULATION GENETICS: CETP deficiency most common in Japan; rare in other populations; "
            "CETP INHIBITORS (therapeutic): anacetrapib, evacetrapib, dalcetrapib, torcetrapib — "
            "  all raised HDL-C substantially; torcetrapib increased mortality (off-target aldosterone); "
            "  Dalcetrapib: no CVD benefit despite HDL-C raise; Anacetrapib: marginal benefit (REVEAL trial)"
        ),
        "disease_category": (
            "CETP DEFICIENCY / HYPERALPHALIPOPROTEINEMIA (HALP) — OMIM 143470; "
            "BIOCHEMICAL HALLMARKS: "
            "  HDL-C VERY HIGH: heterozygote 65-100 mg/dL; homozygote >100-200 mg/dL; "
            "  ApoA-I markedly elevated; "
            "  TG normal or slightly reduced (less TG-for-CE exchange); "
            "  LDL-C low-normal; "
            "  CE-enriched, TG-poor large HDL-1 particles on electrophoresis; "
            "CLINICAL FEATURES: "
            "  GENERALLY BENIGN: most CETP-deficient individuals are healthy; longevity association in Japanese cohorts; "
            "  CARDIOVASCULAR: debated — some studies show protection; heterozygotes have LOWER CAD risk; "
            "  NO xanthomas; NO corneal haze; NO neuropathy; NO orange tonsils; "
            "  CENTENARY LONGEVITY ASSOCIATION: Ashkenazi Jewish centenarian cohort: "
            "    higher CETP deficiency prevalence → large HDL associated with longevity (Albert Einstein LLFS study); "
            "  JAPANESE LONGEVITY: Okinawan centenarian studies: CETP Asp442Gly heterozygotes overrepresented; "
            "CETP INHIBITOR LESSONS: "
            "  Torcetrapib: CETP inhibition raises HDL-C +72% but increased mortality (off-target: aldosterone stimulation + hypertension); "
            "  Dalcetrapib: 30% HDL-C raise, no CVD benefit (DALCOR trial failed); "
            "  Anacetrapib: 138% HDL-C raise; modest 9% CVD event reduction (REVEAL trial — very long follow-up); "
            "  CONCLUSION: HDL-C level alone is an inadequate therapeutic target; HDL FUNCTION matters more"
        ),
        "disease_pathway": (
            "CETP IN HDL REMODELLING AND CE REDISTRIBUTION: "
            "NORMAL: "
            "  1. Mature HDL accumulates CE via LCAT (ApoA-I activated); "
            "  2. CETP: CE transferred from HDL → VLDL/IDL/LDL (exchange for TG); "
            "     This routes CE from peripheral tissues → VLDL/LDL pathway → hepatic LDLR uptake; "
            "  3. HDL simultaneously receives TG from VLDL (via CETP); "
            "  4. HL + EL remove TG from HDL → smaller HDL-3 for another RCT cycle; "
            "  5. SR-BI directly extracts CE from HDL to liver (non-CETP pathway — ~30-60% HDL-CE catabolism); "
            "CETP DEFICIENCY: "
            "  CE cannot leave HDL via CETP → CE accumulates → HDL-C very high; "
            "  Less CE delivered to LDL (lower LDL-CE) → LDL smaller; "
            "  SR-BI pathway still active → some CE delivery to liver; "
            "  Whether CETP deficiency is protective debated: natural LOF (Asp442Gly) heterozygotes show lower CAD; "
            "  CETP inhibitors (pharmacological): raise HDL-C more than natural LOF but CVD benefit inconsistent; "
            "  DALCETRAPIB PHARMACOGENOMICS (dal-GenE trial): ADCY9 gene variant rs1967309 predicts dalcetrapib response; "
            "    AA homozygotes at rs1967309: significant CVD event reduction with dalcetrapib; "
            "    Example of precision medicine in lipid pharmacogenomics"
        ),
        "pathognomonic": (
            "VERY HIGH HDL-C (>100 mg/dL) IN HEALTHY JAPANESE INDIVIDUAL — CETP DEFICIENCY: "
            "  HDL-C >100 mg/dL with normal TG and no secondary cause (no alcohol, no medications) → "
            "  CETP deficiency most likely; "
            "  MOST COMMON CAUSE OF HYPERALPHALIPOPROTEINEMIA (HALP) in Japanese population; "
            "  Asp442Gly (exon 15) = Japanese founder mutation; 7% allele frequency; "
            "  CENTENARY LONGEVITY ASSOCIATION: "
            "    CETP-deficient individuals in Ashkenazi Jewish + Japanese cohorts: "
            "    overrepresented among centenarians; large protective HDL-1 particles; "
            "  KEY DDx HIGH HDL-C: "
            "    CETP deficiency: very high HDL, normal TG, Japanese/Ashkenazi ancestry, benign; "
            "    LIPC deficiency: high HDL + HIGH TG + IDL (premature CAD); "
            "    ApoA-I Milano: low HDL (opposite); "
            "    Secondary (alcohol, exercise, oestrogens): exclude by history; "
            "  CETP INHIBITOR LESSON: "
            "    raising HDL-C pharmacologically ≠ natural protective CETP deficiency"
        ),
        "key_facts": [
            "CETP-VERY-HIGH-HDL-HALP",
            "CETP-ASP442GLY-JAPANESE-FOUNDER-7PCT",
            "CETP-LONGEVITY-CENTENARIANS",
            "CETP-DALCETRAPIB-ADCY9-PHARMACOGENOMICS",
            "CETP-TORCETRAPIB-OFF-TARGET-ALDOSTERONE",
            "CETP-CE-TRANSFER-HDL-TO-VLDL",
            "CETP-HDL-FUNCTION-NOT-LEVEL",
            "CETP-BENIGN-MOSTLY-NO-XANTHOMAS",
        ],
        "treatment": (
            "CETP deficiency: generally no treatment required (benign, often longevity-associated); "
            "Cardiovascular monitoring for very high HDL (rule out secondary causes); "
            "If LIPC deficiency coexists: treat IDL/TG component; "
            "CETP inhibitors (pharmacological): not in clinical use (CVD benefit inconsistent); "
            "dalcetrapib pharmacogenomics (ADCY9 rs1967309) — precision medicine approach"
        ),
        "seed_base": 2778,
        "n_patients": 40,
    },
    {
        "gene": "APOE",
        "protein": (
            "APOE -- 19q13.32 AD -- 317aa -- Apolipoprotein-E-36kDa-Receptor-Binding-Domain-"
            "LDLR-LRP1-Ligand-TRL-Remnant-Clearance-ε2-ε3-ε4-Isoforms-Type-III-HLP-"
            "OMIM-Gene-107741-Disease-HLPIII-107741"
        ),
        "locus": "19q13.32",
        "protein_size": "317 aa / 36 kDa (3 common isoforms: ε2/ε3/ε4 by Cys/Arg at aa 112+158; receptor-binding domain aa 136-150; lipid-binding C-terminal; ligand for LDLR, LRP1, HSPG)",
        "inheritance": (
            "AUTOSOMAL DOMINANT — Type III HLP is ε2/ε2 homozygous with second hit; "
            "COMPLEX GENETICS: ApoE has 3 common isoforms (ε2, ε3, ε4); "
            "  ε3 (Cys112/Arg158): most common (~77%); normal LDLR binding; "
            "  ε4 (Arg112/Arg158): strong LDLR binding; clears remnants fast; ALZHEIMER'S RISK; "
            "  ε2 (Cys112/Cys158): WEAK LDLR binding (100x less than ε3); delayed remnant clearance; "
            "TYPE III HYPERLIPOPROTEINEMIA (HLP-III/Dysbetalipoproteinemia): "
            "  Requires HOMOZYGOUS ε2/ε2 (frequency 1:100) PLUS a second hit: "
            "    hypothyroidism, obesity, T2DM, estrogen deficiency, or another lipid disorder; "
            "  ε2/ε2 alone (without second hit): often normal lipids; "
            "  ε2/ε2 + second hit: remnants (IDL/VLDL remnants) accumulate → mixed hyperlipidemia; "
            "  Prevalence of Type III HLP: 1:5,000-10,000; "
            "ALZHEIMER'S CONNECTION: "
            "  ε4 allele: #1 genetic risk factor for late-onset Alzheimer's disease; "
            "  ε4/ε4: 14x relative risk; ε3/ε4: 3x; ε2/ε2: PROTECTIVE vs ε3/ε3"
        ),
        "disease_category": (
            "TYPE III HYPERLIPOPROTEINEMIA (HLP-III) / DYSBETALIPOPROTEINEMIA — OMIM 107741; "
            "BIOCHEMICAL HALLMARKS: "
            "  Both TC and TG elevated (mixed hyperlipidemia): TC 350-600 mg/dL; TG 400-2000 mg/dL; "
            "  VLDL-CHOLESTEROL elevated (VLDL is CE-rich, cholesterol-laden remnants not TG-laden); "
            "  β-VLDL: abnormal lipoprotein — VLDL with floating beta electrophoretic mobility + high CE:TG ratio; "
            "  VLDL-C/TG ratio >0.3 (normal <0.2): diagnostic of Type III HLP; "
            "  IDL elevated (broad beta band on electrophoresis); "
            "  LDL-C paradoxically LOW (IDL → LDL conversion impaired by ε2); "
            "CLINICAL FEATURES: "
            "  PALMAR XANTHOMAS (XANTHOMA STRIATA PALMARIS) — PATHOGNOMONIC FOR TYPE III HLP: "
            "    yellow lipid deposits in creases of palms and fingers; "
            "    ABSENT in FH (which has tendon xanthomas); present ONLY in Type III HLP and rarely cholestasis; "
            "  TUBEROUS / TUBEROERUPTIVE XANTHOMAS: "
            "    at pressure points (elbows, knees); yellowish-orange nodular deposits; "
            "  PREMATURE PERIPHERAL VASCULAR DISEASE: claudication, femoral bruits (more than coronary); "
            "  PREMATURE CAD: coronary risk elevated (less than FH but significant); "
            "DIAGNOSIS: "
            "  ApoE genotype (confirm ε2/ε2); "
            "  VLDL-C/TG ratio >0.3 on ultracentrifugation; "
            "  β-VLDL on electrophoresis (broad beta band); "
            "  Exclude secondary causes (check TFTs, glucose, BMI)"
        ),
        "disease_pathway": (
            "APOE IN REMNANT CLEARANCE AND RECEPTOR BINDING: "
            "NORMAL: "
            "  1. Chylomicrons (intestinal) and VLDL (hepatic) are secreted into circulation; "
            "  2. LPL removes TG core → chylomicron remnant + IDL formed; "
            "  3. ApoE on remnant surface binds LDLR (liver) + LRP1 + HSPG → rapid hepatic uptake; "
            "  4. IDL: hepatic lipase removes remaining TG → LDL (ApoB-100 only, no ApoE); "
            "  5. LDL cleared via LDLR (ApoB-100/LDLR interaction); "
            "ApoE ISOFORMS AND RECEPTOR BINDING: "
            "  ε3: optimal LDLR binding; normal remnant clearance; "
            "  ε4: slightly better LDLR binding; faster remnant + chylomicron clearance; "
            "    faster CE delivery to liver → reduced LDLR expression (cholesterol feedback) → higher LDL-C; "
            "    ε4 carriers: higher LDL-C (1-5 mg/dL per allele); "
            "  ε2: Arg158Cys → cannot engage LDLR positively charged binding site; "
            "    100-fold reduced LDLR binding → delayed remnant clearance; "
            "    β-VLDL accumulation → CE-enriched remnants → peripheral deposition; "
            "TYPE III HLP MECHANISM: "
            "  ε2/ε2 alone → slow remnant clearance but usually compensated; "
            "  Second hit (hypothyroidism/T2DM/obesity/another lipid gene) → "
            "    overwhelms residual clearance → β-VLDL accumulates → Type III HLP; "
            "TREATMENT: "
            "  FIBRATES: first-line for Type III HLP (activate LPL + reduce VLDL production); "
            "  STATINS: effective (upregulate LDLR/LRP1); "
            "  ADDRESS SECOND HIT: treat hypothyroidism (TSH normalisation alone may resolve HLP III); "
            "  Omega-3 fatty acids; weight loss"
        ),
        "pathognomonic": (
            "PALMAR XANTHOMAS (XANTHOMA STRIATA PALMARIS) — PATHOGNOMONIC FOR TYPE III HLP: "
            "  Yellow-orange lipid deposits specifically in PALMAR and DIGITAL CREASES; "
            "  UNIQUE TO TYPE III HLP (and rarely cholestasis — context distinguishes); "
            "  ABSENT in FH (which has TENDON xanthomas at Achilles and extensor tendons); "
            "  ABSENT in Tangier (which has orange tonsils — no xanthomas); "
            "  ABSENT in LCAT deficiency (which has corneal haze — no xanthomas); "
            "  VLDL-C/TG RATIO >0.3: "
            "    Calculated from ultracentrifugation: if VLDL-C/TG >0.3 → β-VLDL present → Type III HLP; "
            "    Normal: VLDL-C/TG <0.2 (VLDL is TG-rich not CE-rich); "
            "  ε2/ε2 GENOTYPE: found in 1:100 — only 2-5% develop HLP-III (require second hit); "
            "    SCREEN for secondary causes in ALL ε2/ε2 found on genotyping; "
            "  BROAD BETA BAND on lipoprotein electrophoresis: IDL in beta region; "
            "    'Broad beta' = Type II b vs III distinction; Type III = broad single beta; "
            "  HYPOTHYROIDISM TREATMENT: normalise TSH → Type III HLP may fully resolve"
        ),
        "key_facts": [
            "APOE-PALMAR-XANTHOMAS-PATHOGNOMONIC-TYPE-III",
            "APOE-E2-E2-REQUIRES-SECOND-HIT",
            "APOE-VLDL-C-TG-RATIO-GT03",
            "APOE-BROAD-BETA-BAND-ELECTROPHORESIS",
            "APOE-FIBRATES-FIRST-LINE",
            "APOE-HYPOTHYROIDISM-TREAT-RESOLVES-HLP",
            "APOE-E4-ALZHEIMER-RISK-14X",
            "APOE-E2-PROTECTIVE-ALZHEIMER",
        ],
        "treatment": (
            "Type III HLP: "
            "Fibrates (first-line: bezafibrate, fenofibrate — reduce VLDL + activate LPL); "
            "Treat second hit: hypothyroidism (TSH normalisation may fully resolve HLP-III); "
            "Weight loss + low fat + low refined carb diet; "
            "Statins (effective but not first-line alone for Type III); "
            "Omega-3 fatty acids (reduce VLDL-TG); "
            "PCSK9 inhibitors (LRP1 pathway upregulation may help β-VLDL clearance)"
        ),
        "seed_base": 2779,
        "n_patients": 40,
    },
    {
        "gene": "SCARB1",
        "protein": (
            "SCARB1 -- 12q24.31 AR -- 509aa -- Scavenger-Receptor-Class-B-Type-I-57kDa-"
            "Selective-CE-Uptake-from-HDL-to-Liver-Bidirectional-Cholesterol-Transfer-"
            "RCT-Final-Step-Hepatic-OMIM-Gene-601040-Disease-SCARB1-Deficiency"
        ),
        "locus": "12q24.31",
        "protein_size": "509 aa / 57 kDa (class B scavenger receptor; two TM domains; large extracellular loop; selective CE uptake — takes CE from HDL WITHOUT endocytosing the whole particle; bidirectional cholesterol transfer)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF (rare, < 20 cases reported); "
            "MONOALLELIC (Exon 1 c.1A>C variant): more common; mild hyperalphalipoproteinemia; "
            "SCARB1 (SR-BI) is the HEPATIC HDL RECEPTOR for SELECTIVE CE UPTAKE: "
            "  SR-BI binds mature HDL → selectively extracts CE from HDL → CE enters liver → "
            "  HDL particle depleted of CE but retains ApoA-I shell → ApoA-I returns to plasma for another RCT cycle; "
            "  This is the FINAL STEP of RCT — delivering CE to liver for excretion as bile; "
            "WITHOUT SR-BI: "
            "  CE cannot be selectively extracted from HDL → CE accumulates in HDL → HDL-C very high; "
            "  PARADOX: very high HDL-C + PREMATURE CAD (RCT blocked at final step); "
            "  Erythrocytes: SR-BI also maintains RBC membrane cholesterol content; "
            "    SR-BI deficiency → erythrocytes have excess free cholesterol → large, stomatocytic RBCs; "
            "    PLATELET DYSFUNCTION: SR-BI maintains platelet membrane; LOF → platelet activation abnormality; "
            "  FERTILITY: SR-BI critical for adrenal CE uptake for steroidogenesis; "
            "    female SCARB1 LOF mice: infertile (impaired adrenal cholesterol supply); "
            "    human cases: adrenal insufficiency risk in extreme physiological stress"
        ),
        "disease_category": (
            "SCARB1 DEFICIENCY (SR-BI DEFICIENCY) — no specific OMIM disease number (rare); "
            "BIOCHEMICAL HALLMARKS: "
            "  HDL-C markedly elevated (CE accumulates in HDL): HDL-C 80-150+ mg/dL; "
            "  ApoA-I elevated; "
            "  TG: normal or mildly elevated; "
            "  LDL-C: may be elevated (less CETP-mediated CE back-transfer to LDL); "
            "  RBC MORPHOLOGY ABNORMAL: large stomatocytic erythrocytes (excess FC in membrane); "
            "CLINICAL FEATURES: "
            "  PREMATURE CORONARY ARTERY DISEASE: despite very high HDL-C; "
            "    RCT is BLOCKED at final delivery step → CE cannot reach liver → accumulates → atherogenic; "
            "  PLATELET DYSFUNCTION: mild platelet activation abnormality (SR-BI on platelets); "
            "  ANEMIA: mild, from stomatocytic RBC haemolysis; "
            "  ADRENAL INSUFFICIENCY RISK: "
            "    Adrenal cortex relies on SR-BI for CE uptake from HDL for steroidogenesis; "
            "    Under physiological stress (illness, surgery) → adrenal steroidogenesis may be inadequate; "
            "    Consider stress-dose steroids in SCARB1-deficient patients during illness/surgery; "
            "  NO xanthomas; NO corneal haze; NO orange tonsils; NO peripheral neuropathy; "
            "MONOALLELIC (Exon 1 p.Pro2 variant): "
            "  Common in some populations; mild hyperalphalipoproteinemia; modest CAD risk increase"
        ),
        "disease_pathway": (
            "SR-BI AS RCT FINAL STEP AND STEROIDOGENESIS SUPPLIER: "
            "NORMAL: "
            "  1. Mature HDL (CE-rich) circulates; "
            "  2. SR-BI on hepatocyte binds HDL via extracellular loop interaction with ApoA-I and phospholipid; "
            "  3. SR-BI selectively extracts CE from HDL → CE enters hepatocyte cytoplasm → "
            "     CE for bile acid synthesis (CYP7A1 rate-limiting) or VLDL-CE assembly; "
            "  4. ApoA-I-containing HDL shell returns to plasma (smaller, CE-depleted) → available for RCT again; "
            "  5. Adrenal cortex: SR-BI provides CE from HDL for steroidogenesis (cortisol, aldosterone, DHEA); "
            "  6. Ovarian granulosa cells: SR-BI for progesterone + oestradiol synthesis from HDL-CE; "
            "SCARB1 DEFICIENCY: "
            "  HDL cannot deliver CE to liver → RCT halted at final step; "
            "  CE-engorged HDL accumulates → very high HDL-C (dysfunctional); "
            "  Liver: less CE available → increased de novo cholesterol synthesis + increased LDL-CE uptake; "
            "  Adrenal: CE supply impaired → basal steroidogenesis adequate (via LDL) but stress response blunted; "
            "  RBC: excess free cholesterol in membrane → stomatocytes + shortened RBC survival; "
            "THERAPEUTIC TARGET: "
            "  No approved therapy; "
            "  SR-BI AGONISTS being investigated (would restore hepatic CE uptake); "
            "  Statins increase SCARB1 expression (some benefit); "
            "  Adrenal crisis prevention: stress-dose hydrocortisone during illness/surgery"
        ),
        "pathognomonic": (
            "VERY HIGH HDL-C + PREMATURE CAD + STOMATOCYTIC RBCs — SCARB1 DEFICIENCY: "
            "  Very high HDL-C (>80 mg/dL) with premature CAD = RCT FINAL STEP BLOCKED; "
            "  Stomatocytic erythrocytes on blood smear: large bowl-shaped RBCs (excess membrane FC); "
            "    Distinguishes SCARB1 from CETP deficiency (CETP = normal RBCs); "
            "    Distinguishes from LIPC (LIPC = some stomatocytes but less pronounced + high TG); "
            "  ADRENAL CRISIS RISK: "
            "    Under stress, adrenal steroidogenesis blunted; "
            "    SCARB1-deficient surgical patients should receive stress-dose steroids; "
            "  KEY DDx VERY HIGH HDL-C: "
            "    CETP deficiency: normal RBCs; benign (usually); Japanese founder; "
            "    SCARB1 deficiency: stomatocytes; premature CAD; adrenal risk; "
            "    LIPC deficiency: high TG + IDL + post-heparin absent HL; "
            "    Secondary (alcohol): exclude by history + LFTs; "
            "  RCT PARADOX: "
            "    All four genes (ABCA1-initiator, APOA1-scaffold, LCAT-esterification, SCARB1-delivery) = "
            "    if any step blocked → premature CAD despite potentially high HDL-C"
        ),
        "key_facts": [
            "SCARB1-VERY-HIGH-HDL-PREMATURE-CAD",
            "SCARB1-RCT-FINAL-STEP-BLOCKED",
            "SCARB1-STOMATOCYTES-RBC-EXCESS-FC",
            "SCARB1-ADRENAL-CRISIS-STRESS-STEROIDS",
            "SCARB1-SELECTIVE-CE-UPTAKE-NO-ENDOCYTOSIS",
            "SCARB1-PLATELET-DYSFUNCTION",
            "SCARB1-FEMALE-INFERTILITY-MICE",
            "SCARB1-BIDIRECTIONAL-CHOLESTEROL",
        ],
        "treatment": (
            "No FDA-approved disease-modifying therapy; "
            "Statins (increase SCARB1 expression; reduce LDL-C); "
            "Aggressive CVD risk factor management; "
            "Adrenal crisis prevention: stress-dose hydrocortisone during illness/surgery/anaesthesia; "
            "SR-BI agonists (investigational); "
            "Monitor: periodic adrenal function assessment under physiological stress"
        ),
        "seed_base": 2780,
        "n_patients": 40,
    },
    {
        "gene": "LIPG",
        "protein": (
            "LIPG -- 18q21.1 AR -- 500aa -- Endothelial-Lipase-68kDa-Vascular-Endothelium-Anchored-"
            "Phospholipase-A1-Preferentially-Hydrolyses-HDL-Phospholipids-"
            "OMIM-Gene-605512-Disease-EL-Deficiency-Hyperalphalipoproteinemia"
        ),
        "locus": "18q21.1",
        "protein_size": "500 aa / 68 kDa (lipase subfamily; GPI-anchored on vascular endothelial surface; preferential phospholipase A1 activity on HDL — hydrolyses sn-1 fatty acid from HDL phospholipids; reduces HDL size + raises ApoA-I turnover)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → elevated HDL-C); "
            "LIPG (endothelial lipase / EL) is a VASCULAR ENDOTHELIUM-EXPRESSED LIPASE: "
            "  Synthesised by endothelial cells; GPI-anchored; released by heparin; "
            "  PREFERENTIAL PHOSPHOLIPASE A1 activity on HDL (vs LPL=TG-lipase, HL=both); "
            "  EL removes phospholipids from HDL surface → HDL shrinks → ApoA-I dissociates → catabolised; "
            "  EL is major CATABOLIC enzyme for HDL phospholipids; "
            "EL DEFICIENCY: "
            "  HDL phospholipids cannot be hydrolysed → HDL persists longer → HDL-C elevated; "
            "  ApoA-I catabolism reduced → ApoA-I elevated; "
            "EXPRESSION: endothelium, macrophages, hepatocytes; INDUCED by pro-inflammatory cytokines (IL-1β, TNF-α, LPS); "
            "  INFLAMMATION LINK: acute inflammation → increased EL expression → HDL catabolism accelerates → "
            "  HDL-C drops in acute illness (part of acute phase HDL-C fall); "
            "COMMON VARIANTS: Thr111Ile (rs2000813) + Asn396Ser — functional SNPs; "
            "  associated with HDL-C levels in population studies (GWAS); "
            "GENETIC EPIDEMIOLOGY: "
            "  Biallelic LOF → LIPG-deficiency hyperalphalipoproteinemia (rare, few cases); "
            "  Common variant Asn396Ser: associated with 5% higher HDL-C per allele in large GWAS"
        ),
        "disease_category": (
            "ENDOTHELIAL LIPASE DEFICIENCY / LIPG-HYPERALPHALIPOPROTEINEMIA (rare); "
            "BIOCHEMICAL HALLMARKS: "
            "  HDL-C elevated: 60-90+ mg/dL (less pronounced than CETP deficiency); "
            "  ApoA-I elevated (slower catabolism); "
            "  TG: normal; "
            "  LDL-C: normal; "
            "  POST-HEPARIN PLASMA: absent endothelial lipase activity (distinct from HL); "
            "CLINICAL FEATURES: "
            "  GENERALLY BENIGN: most LIPG-deficient individuals healthy; "
            "  CARDIOVASCULAR EFFECT DEBATED: "
            "    EL LOF: some epidemiological data suggest protective; "
            "    EL overexpression (e.g. in inflammation) → HDL catabolism accelerates → HDL-C drops → pro-atherogenic; "
            "  INFLAMMATION CONNECTION: "
            "    Acute illness → TNF-α/IL-1β → EL expression surges → HDL-C falls acutely; "
            "    This is why HDL-C DROPS in sepsis/acute inflammation — EL-mediated catabolism; "
            "    LIPG-deficient patients: blunted HDL-C drop in inflammation; "
            "  NO xanthomas; NO corneal haze; NO orange tonsils; NO neuropathy; NO adrenal risk; "
            "THERAPEUTIC IMPLICATIONS: "
            "  EL INHIBITORS: under investigation as HDL-raising strategy; "
            "  More selective for HDL than CETP inhibitors (does not affect LDL pathway); "
            "  Challenge: EL inhibition may reduce HDL FUNCTION not just raise HDL-C (same concern as CETP inhibitors)"
        ),
        "disease_pathway": (
            "ENDOTHELIAL LIPASE IN HDL CATABOLISM AND INFLAMMATION: "
            "NORMAL: "
            "  1. Vascular endothelium secretes EL → anchored to luminal surface via heparan sulfate; "
            "  2. Circulating HDL encounters EL on endothelial surface; "
            "  3. EL hydrolyses phospholipids from HDL surface (preferentially sn-1 position); "
            "     Phospholipids removed → HDL surface area shrinks; "
            "  4. ApoA-I becomes unstable on smaller HDL → dissociates → excreted by kidneys or reused; "
            "  5. Phospholipid hydrolysis products (lysophosphatidylcholine = LPC) released → "
            "     LPC is pro-inflammatory (PAFR activation) — thus EL products have direct vascular effects; "
            "INFLAMMATORY REGULATION: "
            "  IL-1β, TNF-α, LPS → strongly induce EL expression in endothelium; "
            "  Acute phase: HDL-C falls (EL surge); "
            "  PPARα agonists (fibrates) → inhibit EL transcription → modest HDL-C rise partly via EL suppression; "
            "EL DEFICIENCY: "
            "  HDL phospholipids not hydrolysed → HDL particles persist → HDL-C elevated; "
            "  Paradox of very high HDL may not translate to protection: "
            "    Need to assess cholesterol efflux capacity, not just HDL-C; "
            "LIPG vs LIPC vs LPL: "
            "  LIPG (EL): endothelial; phospholipase-A1 preferential; HDL substrate; "
            "  LIPC (HL): hepatocyte; TG-lipase + phospholipase; HDL-2+IDL substrate; "
            "  LPL: endothelial (capillary); TG-lipase; VLDL+chylomicron substrate"
        ),
        "pathognomonic": (
            "MILDLY ELEVATED HDL-C + INFLAMMATION-LINKED HDL DROP — LIPG CONTEXT: "
            "  No single pathognomonic feature (biallelic deficiency extremely rare); "
            "  LIPG DEFICIENCY suspected when: "
            "    Elevated HDL-C + normal TG + no CETP/LIPC/ABCA1 mutations + absent EL post-heparin activity; "
            "  INFLAMMATORY HDL-C DROP: "
            "    Acutely ill patient: HDL-C falls from 60 → 20 mg/dL within 24-48h of sepsis; "
            "    Mechanism: IL-1β/TNF-α → EL surge → HDL catabolism; "
            "    EL as mediator of HDL deficiency in acute illness = clinically relevant (NOT genetic disease); "
            "  KEY LIPG DISTINGUISHING FEATURES: "
            "    EL deficiency: HDL elevated, normal TG, no xanthomas; "
            "    EL overactivity (inflammatory): HDL drops, acute illness context; "
            "  POST-HEPARIN PLASMA: "
            "    Heparin releases EL + HL from endothelium; "
            "    Measure lipolytic activity: EL activity can be measured (salt-sensitive component); "
            "    LIPG deficiency: EL component absent, HL normal; "
            "  LIPG vs LIPC DIFFERENTIATION: "
            "    LIPC: high HDL + HIGH TG + IDL; LIPG: high HDL + normal TG + no IDL"
        ),
        "key_facts": [
            "LIPG-ENDOTHELIAL-PHOSPHOLIPASE-A1",
            "LIPG-INFLAMMATION-SURGE-HDL-DROP",
            "LIPG-HDL-CATABOLISM-ACUTE-ILLNESS",
            "LIPG-POST-HEPARIN-EL-ACTIVITY",
            "LIPG-EL-INHIBITORS-INVESTIGATIONAL",
            "LIPG-LPC-PRO-INFLAMMATORY-PRODUCT",
            "LIPG-NORMAL-TG-VS-LIPC",
            "LIPG-PPAR-FIBRATES-SUPPRESS-EL",
        ],
        "treatment": (
            "LIPG biallelic deficiency: no specific treatment required (generally benign); "
            "Cardiovascular risk assessment; "
            "No xanthoma or organ-specific complications; "
            "EL inhibitors (investigational — no approved agents); "
            "Fibrates partially suppress EL transcription (modest HDL-C benefit); "
            "For acute illness: recognise EL-mediated HDL-C drop is physiological, not a chronic defect"
        ),
        "seed_base": 2781,
        "n_patients": 40,
    },
]


def _patient_data(gene_info: dict) -> list:
    seed = gene_info["seed_base"]
    rng = random.Random(seed)
    n = gene_info["n_patients"]
    gene = gene_info["gene"]
    rows = []
    for i in range(n):
        age = rng.randint(22, 78)
        sex = rng.choice(["M", "F"])
        rows.append({
            "patient_id": f"HDL-{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "hdl_c": round(rng.uniform(2, 180), 1),
            "ldl_c": round(rng.uniform(40, 350), 1),
            "tg": round(rng.uniform(60, 800), 1),
            "total_cholesterol": round(rng.uniform(100, 700), 1),
            "apoa1_mgdl": round(rng.uniform(10, 320), 1),
            "inheritance": gene_info["inheritance"].split("AUTOSOMAL")[1].split(";")[0].strip() if "AUTOSOMAL" in gene_info["inheritance"] else "complex",
            "key_phenotype": gene_info["key_facts"][rng.randint(0, len(gene_info["key_facts"]) - 1)],
        })
    return rows


def generate_overview() -> dict:
    all_patients = []
    for g in ATLAS_GENES:
        all_patients.extend(_patient_data(g))

    total = len(all_patients)
    genes = [g["gene"] for g in ATLAS_GENES]

    avg_hdl = round(sum(p["hdl_c"] for p in all_patients) / total, 1)
    avg_ldl = round(sum(p["ldl_c"] for p in all_patients) / total, 1)
    avg_tg = round(sum(p["tg"] for p in all_patients) / total, 1)

    per_gene = {}
    for g in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == g["gene"]]
        per_gene[g["gene"]] = {
            "n": len(pts),
            "avg_hdl": round(sum(p["hdl_c"] for p in pts) / len(pts), 1),
            "avg_ldl": round(sum(p["ldl_c"] for p in pts) / len(pts), 1),
            "avg_tg": round(sum(p["tg"] for p in pts) / len(pts), 1),
            "locus": g["locus"],
            "protein_size": g["protein_size"].split("(")[0].strip(),
        }

    return {
        "atlas": "Hereditary-HDL-Metabolism-Atlas",
        "subtitle": "Complete 8-Gene HDL Metabolism & Reverse Cholesterol Transport Reference",
        "genes": genes,
        "total_patients": total,
        "seeds": "2774-2781",
        "summary": {
            "avg_hdl_c_mgdl": avg_hdl,
            "avg_ldl_c_mgdl": avg_ldl,
            "avg_tg_mgdl": avg_tg,
            "per_gene": per_gene,
        },
        "rct_pathway_steps": {
            "step1_initiation": "ABCA1 — efflux FC+PL from macrophage to lipid-poor ApoA-I → nascent discoidal HDL",
            "step2_scaffold": "APOA1 — structural scaffold; LCAT activator; ABCA1 substrate; RCT acceptor",
            "step3_esterification": "LCAT — esterifies FC on HDL → CE core → nascent → mature spherical HDL",
            "step4_remodelling": "LIPC (HL) — remodels HDL-2 → HDL-3; clears IDL remnants via ApoE-LDLR",
            "step5_redistribution": "CETP — transfers CE from HDL → VLDL/LDL (exchange for TG); HDL-CE catabolism",
            "step6_final_delivery": "SCARB1 (SR-BI) — selective CE uptake from HDL → liver (RCT final step; bile formation)",
            "step7_phospholipid_catabolism": "LIPG (EL) — hydrolyses HDL phospholipids; HDL turnover; inflammation-regulated",
            "step0_remnant_clearance": "APOE — LDLR/LRP1 ligand on remnants; Type III HLP when ε2/ε2 + second hit",
        },
        "pathognomonic_signs": {
            "ABCA1": "Orange/yellow tonsils (Tangier disease) — PATHOGNOMONIC",
            "APOA1": "ApoA-I Milano paradox (low HDL, no CAD) — HDL function > HDL quantity",
            "LCAT": "Corneal haze + hemolytic anemia + proteinuria + Lp-X (FLD triad)",
            "LIPC": "High HDL + high TG + IDL + post-heparin absent HL activity + premature CAD",
            "CETP": "Very high HDL (>100 mg/dL) in healthy Japanese individual; Asp442Gly founder",
            "APOE": "Palmar xanthomas (xanthoma striata palmaris) — PATHOGNOMONIC for Type III HLP",
            "SCARB1": "Very high HDL + premature CAD + stomatocytic RBCs + adrenal crisis risk",
            "LIPG": "Inflammation-mediated HDL-C drop (EL surge); elevated HDL in biallelic deficiency",
        },
        "cascade_testing": "HDL-C in all first-degree relatives of index cases; ApoA-I for ABCA1/APOA1; ApoE genotype if mixed hyperlipidemia + palmar xanthomas; cholesterol efflux assay for functional assessment",
        "registered": "2026-09-15",
    }


def generate_breakdown() -> dict:
    result = {}
    for g in ATLAS_GENES:
        pts = _patient_data(g)
        result[g["gene"]] = {
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"][:400],
            "disease_category": g["disease_category"][:600],
            "disease_pathway": g["disease_pathway"][:600],
            "pathognomonic": g["pathognomonic"][:500],
            "key_facts": g["key_facts"],
            "treatment": g["treatment"],
            "n_patients": len(pts),
            "seed": g["seed_base"],
            "sample_patients": pts[:5],
        }
    return result


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-HDL-Metabolism-Atlas Glossary",
        "terms": {
            "RCT": "Reverse Cholesterol Transport — pathway moving excess cholesterol from peripheral macrophages to liver for excretion; ABCA1→ApoA-I→LCAT→SCARB1 sequential steps",
            "ABCA1": "ATP-binding cassette transporter A1; rate-limiting for HDL biogenesis; master initiator of RCT; mutated in Tangier disease (AR) and familial hypoalphalipoproteinemia (AD)",
            "Tangier_Disease": "Biallelic ABCA1 LOF; HDL-C <5 mg/dL; orange/yellow tonsils PATHOGNOMONIC; peripheral neuropathy (relapsing-remitting); rectal foam cells on sigmoidoscopy",
            "APOA1": "Apolipoprotein A-I; major HDL structural protein (70-80% HDL protein); LCAT obligate activator; ABCA1 substrate; ApoA-I Milano (Arg173Cys) paradox = low HDL + no CAD",
            "ApoA1_Milano": "Arg173Cys APOA1 founder mutation; Italian village Limone sul Garda; very low HDL-C but NO excess CAD; demonstrates HDL function > HDL-C level",
            "LCAT": "Lecithin-cholesterol acyltransferase; sole enzyme esterifying FC on HDL; ApoA-I-activated; FLD (complete LOF) = corneal haze + anemia + proteinuria; FED (partial) = corneal only",
            "FLD": "Familial LCAT Deficiency (complete); corneal haze + hemolytic anemia + proteinuria triad; Lp-X (abnormal vesicular lipoprotein) pathognomonic on electrophoresis",
            "Fish_Eye_Disease": "Partial LCAT deficiency; esterifies VLDL/LDL but NOT HDL; corneal opacity only (no anemia/renal disease); named for cloudy whitish corneal appearance",
            "LpX": "Lipoprotein-X; phospholipid-rich CE-poor vesicular lipoprotein; forms in LCAT deficiency + obstructive cholestasis; nephrotoxic; absent CE esterification → unesterified FC in vesicles",
            "LIPC": "Hepatic lipase; liver-anchored; TG-lipase + phospholipase A1; remodels HDL-2→HDL-3; clears IDL remnants; deficiency → high HDL + high TG + premature CAD (dysfunctional HDL-2)",
            "PHLA": "Post-Heparin Lipolytic Activity; plasma assay after IV heparin releases HL + LPL; salt-resistant fraction = HL; LIPC deficiency → absent salt-resistant HL component; LIPG deficiency → absent EL component",
            "CETP": "Cholesteryl ester transfer protein; transfers CE from HDL → VLDL/LDL (exchange for TG); LOF = very high HDL (HALP); Asp442Gly = Japanese founder (7% allele frequency); centenarian longevity association",
            "HALP": "Hyperalphalipoproteinemia; HDL-C >75th percentile; causes: CETP deficiency (most common genetic cause), LIPC deficiency, SCARB1 deficiency, LIPG deficiency, secondary (alcohol, oestrogen, exercise)",
            "CETP_Inhibitors": "Torcetrapib (off-target aldosterone → mortality); dalcetrapib (no CVD benefit; ADCY9 pharmacogenomics subset benefit); anacetrapib (marginal 9% CVD reduction, REVEAL trial); evacuetrapib (terminated); lesson: HDL-C level ≠ protection",
            "APOE": "Apolipoprotein E; 3 isoforms (ε2/ε3/ε4); LDLR + LRP1 + HSPG ligand for remnant clearance; ε2/ε2 + second hit → Type III HLP; ε4 = Alzheimer's risk (ε4/ε4 = 14x); ε2 = Alzheimer's protective",
            "Type_III_HLP": "Type III hyperlipoproteinemia / dysbetalipoproteinemia; ε2/ε2 + second hit; mixed TC+TG elevation; β-VLDL; palmar xanthomas PATHOGNOMONIC; VLDL-C/TG ratio >0.3",
            "Palmar_Xanthomas": "Xanthoma striata palmaris; yellow lipid deposits in palmar/digital creases; PATHOGNOMONIC for Type III HLP; absent in FH (Achilles tendon xanthomas) and Tangier (orange tonsils)",
            "SCARB1": "Scavenger receptor class B type I (SR-BI); hepatic HDL receptor for selective CE uptake; RCT final step; LOF → very high HDL + premature CAD + stomatocytic RBCs + adrenal crisis risk",
            "SR_BI_Selective_Uptake": "CE extracted from HDL without internalising whole particle; ApoA-I returns to plasma; contrasts with LDLR (endocytoses entire LDL particle); hepatic + adrenal + gonadal expression",
            "LIPG": "Endothelial lipase (EL); vascular endothelium anchored; phospholipase A1 on HDL; HDL catabolism; inflammation-induced (TNF-α, IL-1β) → acute-phase HDL-C drop in sepsis; LOF → elevated HDL",
            "HDL_Function": "Cholesterol efflux capacity from macrophages (gold standard measure); more predictive of CAD risk than HDL-C level; measures RCT flux not steady-state HDL-C",
            "RCT_Paradox": "High HDL-C can coexist with premature CAD if RCT is blocked (ABCA1-APOA1-LCAT-LIPC-SCARB1 deficiencies all demonstrate this); HDL function must accompany HDL quantity",
            "Cascade_Testing": "HDL-C in first-degree relatives of all HDL-metabolism disorder index cases; ApoE genotype if palmar xanthomas + mixed hyperlipidemia; cholesterol efflux assay for functional HDL assessment",
        }
    }
