"""Hereditary Kidney Disease Atlas — 8-Gene Reference
PKD1-PKD2-PKHD1-COL4A5-COL4A3-UMOD-HNF1B-NPHS2
320 patients (8 x 40), seeds 2598-2605.
Endpoints: /api/hereditary-kidney-disease-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "PKD1",
        "protein": (
            "PKD1 -- 16p13.3 AD -- 4303aa -- Polycystin-1-462kDa-Receptor-Like-Membrane-Protein-"
            "ADPKD1-Most-Common-AD -- OMIM-Gene-601313-Disease-ADPKD1-173900"
        ),
        "locus": "16p13.3",
        "protein_size": "4303 aa / 462 kDa",
        "inheritance": (
            "AD (haploinsufficiency + somatic second-hit 'two-hit' model); "
            "ADPKD type 1 — ~85% of all ADPKD; "
            "Most common life-threatening monogenic disorder 1:400-1:1000 births; "
            "ESRD at median ~54 years (earlier/more severe than PKD2); "
            "PKD1 truncating variants more severe than missense; "
            "de novo mutations account for ~10% (no family history); "
            "somatic second hit in tubular cells triggers cystogenesis"
        ),
        "disease_category": (
            "Autosomal Dominant Polycystic Kidney Disease type 1 (ADPKD1); "
            "bilateral progressive renal cysts → progressive kidney enlargement → CKD → ESRD; "
            "Tolvaptan (V2R antagonist) FDA 2018 — first disease-modifying therapy; slows TKV increase; "
            "Hypertension early and universal — ACEi/ARB first-line; "
            "Intracranial aneurysm (ICA) 10% — screen if family history ICA or prior rupture (MR angiography); "
            "Liver cysts 80% (usually asymptomatic); cardiac valve abnormalities 25%"
        ),
        "disease_pathway": (
            "Polycystin-1 (PC1, PKD1) is a large receptor-like transmembrane protein. "
            "It forms a functional receptor-channel complex with Polycystin-2 (PC2, PKD2) "
            "at the primary cilium and lateralised cell membrane. "
            "PC1/PC2 complex regulates intracellular Ca2+ and mTOR/cAMP signalling. "
            "Loss of PC1 function (haploinsufficiency + somatic second-hit in tubular cells) → "
            "↓ intracellular Ca2+, ↑ cAMP → activation of MAPK/ERK and mTOR pathways → "
            "epithelial cell proliferation + fluid secretion into cyst lumen → cyst expansion. "
            "Tolvaptan blocks V2R-mediated cAMP production in collecting duct cells → "
            "reduces cyst fluid secretion and epithelial proliferation, slowing total kidney volume (TKV) growth."
        ),
        "pathognomonic": (
            "BILATERAL RENAL CYSTS on ultrasound — Pei-Ravine criteria: if family history, "
            "age 15-39: ≥3 cysts (total bilateral); age 40-59: ≥2 cysts per kidney; age ≥60: ≥4 cysts per kidney; "
            "TOTAL KIDNEY VOLUME (TKV) by MRI — most accurate prognostic marker; "
            "TKV >600 ml or height-adjusted TKV (htTKV) >600 ml/m = rapidly progressive (Class 1C-1E — tolvaptan eligible); "
            "MAYO CLINIC CLASSIFICATION: Class 1A-1E based on htTKV growth rate (A = slowest, E = fastest); "
            "LIVER CYSTS (80%) — multiple bilateral hepatic cysts without biochemical liver dysfunction (DDx PCLD); "
            "INTRACRANIAL ANEURYSM on MR angiography — 10%; berry aneurysm; rupture risk increases with size >7mm"
        ),
        "treatment": (
            "BLOOD PRESSURE: ACEi (enalapril, ramipril) or ARB (losartan, valsartan) — first-line; "
            "target BP <110/75 mmHg in young patients (HALT-PKD trial: low BP target slows TKV growth); "
            "TOLVAPTAN (Jynarque/Samsca) — FDA 2018; V2R antagonist; reduces htTKV growth by ~50%; "
            "Indications: 18-55yr, CKD stage 1-3, rapidly progressive (htTKV >750 ml/m or Mayo 1C/D/E); "
            "CI: liver disease, hyponatraemia risk; Monitoring: LFTs monthly × 18m then every 3m (FDA REMS); "
            "Aquaresis: polyuria/nocturia — take in AM; avoid fluid restriction; "
            "PAIN: opioid-sparing; avoid NSAIDs (nephrotoxic in CKD); cyst aspiration or laparoscopic fenestration for severe pain; "
            "INTRACRANIAL ANEURYSM: MR angiography screening if family history ICA/SAH or occupational risk; "
            "aneurysm >7mm or rapid growth → neurosurgical/neuro-interventional review; "
            "DIALYSIS/TRANSPLANT: kidney transplant preferred over haemodialysis; living donor evaluation; "
            "bilateral nephrectomy occasionally needed pre-transplant for massive kidneys (volume >4 litres)"
        ),
        "key_features": [
            "PKD1: ~85% of ADPKD; most common life-threatening monogenic disorder 1:400-1:1000; ESRD ~54yr",
            "Two-hit model: germline PKD1 loss + somatic second hit in tubular cells → cystogenesis (explains bilateral despite AD)",
            "Tolvaptan FDA 2018 — first disease-modifying therapy; slows TKV; LFT monitoring mandatory (REMS programme)",
            "Mayo Class 1C-1E (htTKV >750 ml/m or >8%/yr growth) = rapidly progressive → tolvaptan eligible",
            "Intracranial aneurysm 10% — screen with MR angiography if family ICA, prior SAH, or high-risk occupation",
            "Low BP target (<110/75 in young) with ACEi/ARB slows TKV growth and GFR decline (HALT-PKD)",
            "Bilateral renal cysts + liver cysts (80%) + family history = diagnostic triad; genetic testing only if uncertain",
            "Truncating PKD1 variants more severe than missense; PKD2 significantly milder (ESRD ~74yr vs ~54yr)",
        ],
        "key_ddx": [
            "PKD2 ADPKD — same bilateral cyst phenotype but milder; ESRD 20yr later; PKD2 gene on 4q22.1",
            "Von Hippel-Lindau (VHL) — bilateral renal cysts+clear cell RCC; retinal+cerebellar haemangioblastomas",
            "Tuberous sclerosis (TSH1/TSC2) — bilateral angiomyolipomas; skin lesions; brain tubers; epilepsy",
            "Simple renal cysts — incidental in elderly; unilateral; no hypertension; Bosnian criteria for malignancy risk",
        ],
        "esrd_age": 54,
        "tkv_growth_pct": 7,
        "liver_cyst_pct": 80,
        "intracranial_aneurysm_pct": 10,
        "tolvaptan_eligible_pct": 45,
        "hypertension_pct": 85,
        "hematuria_pct": 50,
    },
    {
        "gene": "PKD2",
        "protein": (
            "PKD2 -- 4q22.1 AD -- 968aa -- Polycystin-2-110kDa-TRP-Family-Ca2+-Channel-"
            "ADPKD2-Milder-Later-Onset-AD -- OMIM-Gene-173910-Disease-ADPKD2-613095"
        ),
        "locus": "4q22.1",
        "protein_size": "968 aa / 110 kDa",
        "inheritance": (
            "AD (haploinsufficiency + two-hit); "
            "ADPKD type 2 — ~15% of all ADPKD; "
            "Significantly MILDER than PKD1; ESRD at median ~74 years; "
            "many PKD2 patients die of other causes before reaching ESRD; "
            "PKD2 mutations uniformly truncating (missense rarely pathogenic); "
            "complete loss of PC2 channel function in cyst-lining cells"
        ),
        "disease_category": (
            "Autosomal Dominant Polycystic Kidney Disease type 2 (ADPKD2); "
            "same bilateral progressive cysts as PKD1 but 20yr later ESRD; "
            "Tolvaptan also used in PKD2 if rapidly progressive (same Mayo criteria); "
            "Hypertension — still common but onset somewhat later than PKD1; "
            "Liver cysts — less frequent than PKD1; "
            "ICA — slightly lower frequency than PKD1 but still ~5-7%"
        ),
        "disease_pathway": (
            "Polycystin-2 (PC2, TRPP2, PKD2) is a TRP-family Ca2+-permeable cation channel "
            "located at the primary cilium, endoplasmic reticulum, and basolateral membrane. "
            "PC2 forms a heteromeric channel with PC1 — PC1 acts as the mechanosensor/receptor, "
            "PC2 provides the ion channel activity for Ca2+ influx in response to flow/bending of the cilium. "
            "Loss of PC2 → impaired ciliary Ca2+ signalling → ↑ cAMP → identical downstream pathway to PKD1 "
            "(MAPK/ERK, mTOR activation → proliferation + secretion → cystogenesis). "
            "Milder phenotype than PKD1 because PC2 loss is 'pure channel' loss without disrupting the large "
            "extracellular signalling scaffold contributed by PC1's N-terminal domains."
        ),
        "pathognomonic": (
            "BILATERAL RENAL CYSTS — same Pei-Ravine criteria as PKD1 (use same age-adjusted criteria); "
            "GENETICALLY MILDER: patients often diagnosed later in life; TKV growth slower; "
            "ESRD ~74yr — most PKD2 patients do NOT reach ESRD before age 70; "
            "GENETIC TESTING — essential to distinguish PKD2 from PKD1 for prognosis counselling; "
            "PKD2 MISSENSE VARIANTS: most are benign/VUS; truncating variants = pathogenic; "
            "FAMILY HISTORY: PKD2 pedigrees often show milder progression than PKD1 families"
        ),
        "treatment": (
            "Same framework as PKD1: "
            "ACEi/ARB — first-line for hypertension; target BP <130/80 (relaxed from PKD1 110/75 given milder course); "
            "TOLVAPTAN — same eligibility criteria as PKD1 (htTKV >750 ml/m or Mayo 1C/D/E); "
            "less likely to reach eligibility criteria given slower progression; "
            "LFT MONITORING mandatory with tolvaptan regardless of PKD1 vs PKD2; "
            "GENETIC COUNSELLING: 50% offspring risk; distinguish PKD2 from PKD1 for prognosis; "
            "TRANSPLANT: same indications as CKD stage 5; smaller kidneys at transplant than PKD1; "
            "ICA screening: family history ICA/SAH or occupational risk; same MR angiography protocol; "
            "PAIN: same approach — opioid-sparing; avoid NSAIDs"
        ),
        "key_features": [
            "PKD2: ~15% of ADPKD; TRP-family Ca2+ channel; ESRD ~74yr — 20yr later than PKD1",
            "Many PKD2 patients die of unrelated causes before ESRD — mortality pattern differs from PKD1",
            "PKD2 missense variants usually benign/VUS — truncating variants pathogenic (unlike PKD1 where missense also pathogenic)",
            "Tolvaptan eligible if Mayo 1C-1E; less commonly needed given slower TKV growth",
            "Genetic testing essential: PKD2 vs PKD1 distinction critical for prognosis and insurance/employment discussions",
            "Same bilateral cyst phenotype as PKD1 on imaging — cannot be distinguished by ultrasound alone",
            "ICA risk ~5-7% — slightly lower than PKD1; same MR angiography screening indications",
            "Liver cysts less frequent than PKD1 (40-60%); same dietary/BP management principles",
        ],
        "key_ddx": [
            "PKD1 ADPKD — same cyst phenotype but earlier/more severe; PKD1 gene on 16p13.3; ESRD 20yr earlier",
            "Isolated liver cysts with unilateral/few renal cysts — consider PRKCSH/SEC63 (ADPLD); no renal impairment",
            "Acquired cystic kidney disease (dialysis patients) — bilateral cysts but in CKD context; no family history",
            "Birt-Hogg-Dubé (FLCN) — skin fibrofolliculomas; lung cysts + spontaneous pneumothorax; renal hybrid tumours",
        ],
        "esrd_age": 74,
        "tkv_growth_pct": 4,
        "liver_cyst_pct": 50,
        "intracranial_aneurysm_pct": 6,
        "tolvaptan_eligible_pct": 20,
        "hypertension_pct": 70,
        "hematuria_pct": 35,
    },
    {
        "gene": "PKHD1",
        "protein": (
            "PKHD1 -- 6p12.2 AR -- 4074aa -- Fibrocystin-Polyductin-447kDa-Ciliary-Receptor-"
            "ARPKD-Hepatic-Fibrosis-AR -- OMIM-Gene-606702-Disease-ARPKD-263200"
        ),
        "locus": "6p12.2",
        "protein_size": "4074 aa / 447 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "Autosomal Recessive Polycystic Kidney Disease (ARPKD); "
            "1:20,000 births; carrier frequency 1:70; "
            "NEONATAL/INFANTILE presentation in severe forms (Potter sequence); "
            "null/null (biallelic truncating) genotypes = most severe → oligohydramnios/pulmonary hypoplasia/stillbirth; "
            "missense/missense genotypes often milder — may survive to childhood/adulthood; "
            "compound heterozygous most common in Western populations"
        ),
        "disease_category": (
            "Autosomal Recessive Polycystic Kidney Disease (ARPKD) with CONGENITAL HEPATIC FIBROSIS; "
            "MASSIVE bilateral kidneys (macroscopically → fusiform collecting duct ectasia, NOT discrete cysts); "
            "HEPATIC FIBROSIS + PORTAL HYPERTENSION — defining hepatic involvement distinguishes from ADPKD; "
            "Neonatal severe: Potter sequence (oligohydramnios → pulmonary hypoplasia → respiratory failure → stillbirth/neonatal death); "
            "Survivors: CKD progression + portal hypertension dominates long-term course; "
            "No disease-modifying therapy; management: hypertension, CKD, portal hypertension complications"
        ),
        "disease_pathway": (
            "Fibrocystin/Polyductin (FPC, PKHD1) is a large single-pass transmembrane receptor "
            "localised to primary cilia, basal bodies, and lateral membranes of tubular epithelial cells. "
            "FPC is expressed in renal collecting ducts, biliary epithelium (cholangiocytes), and pancreatic ducts. "
            "Loss of FPC → impaired ciliary signalling → dysregulated cell polarity, proliferation, and planar cell polarity (PCP) → "
            "fusiform dilation of collecting ducts (NOT spherical cysts as in ADPKD) → massive kidney enlargement. "
            "Biliary involvement: FPC loss in cholangiocytes → ductal plate malformation → "
            "congenital hepatic fibrosis (CHF) + Caroli disease → progressive portal hypertension → "
            "variceal bleeding, hypersplenism, and cholangitis. "
            "Severity correlates with genotype: biallelic null/null → maximal loss → Potter; "
            "compound het (null + missense) → partial function preserved → milder/later phenotype."
        ),
        "pathognomonic": (
            "MASSIVE BILATERAL ECHOGENIC KIDNEYS on prenatal/neonatal ultrasound — "
            "cortex-medulla differentiation lost; fusiform collecting duct ectasia (NOT discrete cysts); "
            "OLIGOHYDRAMNIOS in severe prenatal cases → Potter sequence (pulmonary hypoplasia + characteristic facies + limb deformities); "
            "CONGENITAL HEPATIC FIBROSIS on biopsy — proliferated bile ducts in portal tracts; "
            "PORTAL HYPERTENSION: splenomegaly + oesophageal varices + hypersplenism WITHOUT cirrhosis; "
            "NORMAL LIVER SYNTHETIC FUNCTION — CHF does NOT cause cirrhosis (hepatocellular function preserved); "
            "CAROLI DISEASE — saccular biliary dilatation; risk recurrent cholangitis"
        ),
        "treatment": (
            "NEONATAL CRITICAL CARE: mechanical ventilation for pulmonary hypoplasia; "
            "respiratory support titration; surfactant if premature; "
            "HYPERTENSION: ACEi/ARB or calcium channel blocker — universal; critical for CKD progression; "
            "PORTAL HYPERTENSION: non-selective beta-blocker (propranolol/carvedilol) for variceal prophylaxis; "
            "VARICES: endoscopic band ligation or sclerotherapy if large varices or bleeding; "
            "HYPERSPLENISM: platelet count monitoring; splenectomy or partial splenic embolisation if severe; "
            "CHOLANGITIS: antibiotics (cephalosporins/ciprofloxacin); ursodeoxycholic acid for biliary symptoms; "
            "CKD MANAGEMENT: standard CKD diet, anaemia management (EPO), phosphate control, renal replacement planning; "
            "TRANSPLANTATION: combined liver-kidney transplant preferred if both advanced; "
            "isolated kidney transplant if hepatic fibrosis stable/compensated; "
            "GENETIC COUNSELLING: AR — both parents carriers; 25% recurrence risk; prenatal testing available"
        ),
        "key_features": [
            "ARPKD: AR; biallelic PKHD1 loss; massive echogenic kidneys + congenital hepatic fibrosis (NOT discrete cysts)",
            "Potter sequence (null/null): oligohydramnios → pulmonary hypoplasia → stillbirth/neonatal death",
            "Normal liver SYNTHETIC function despite CHF and portal hypertension — hepatocellular function preserved",
            "Portal hypertension WITHOUT cirrhosis — oesophageal varices + hypersplenism + splenomegaly",
            "Caroli disease (saccular biliary ectasia) — risk cholangitis; ursodeoxycholic acid + antibiotics",
            "Compound het (null+missense) milder than null/null — survival to childhood/adulthood possible",
            "Combined liver-kidney transplant if both organs fail; isolated kidney transplant if liver compensated",
            "Carrier frequency 1:70; 25% recurrence risk in siblings; prenatal ultrasound + PKHD1 testing available",
        ],
        "key_ddx": [
            "ADPKD (PKD1/PKD2) — AD, parental involvement, discrete cysts (NOT fusiform), liver cysts (NOT CHF), later onset",
            "Nephronophthisis (NPHP genes) — AR, medullary cysts, normal-sized kidneys, retinal dystrophy, adolescent ESRD",
            "Congenital hepatic fibrosis without renal involvement — POLR3A, DCDC2; no bilateral renal cysts",
            "Jeune syndrome (asphyxiating thoracic dystrophy) — skeletal dysplasia + kidney disease; thoracic constriction",
        ],
        "esrd_age": 30,
        "tkv_growth_pct": 15,
        "liver_cyst_pct": 100,
        "portal_htn_pct": 70,
        "cholangitis_pct": 30,
        "hypertension_pct": 90,
        "hematuria_pct": 20,
    },
    {
        "gene": "COL4A5",
        "protein": (
            "COL4A5 -- Xq22.3 XLD/XLR -- 1454aa -- Collagen-Alpha5-IV-161kDa-GBM-Type-IV-"
            "Collagen-Network-Alport-XL-Most-Common -- OMIM-Gene-303630-Disease-Alport-XL-301050"
        ),
        "locus": "Xq22.3",
        "protein_size": "1454 aa / 161 kDa",
        "inheritance": (
            "X-linked dominant (XLD in females: heterozygous; XLR in males: hemizygous); "
            "X-linked Alport syndrome — ~80% of all Alport syndrome; "
            "Males: hemizygous → classic Alport; microscopic haematuria in infancy → proteinuria → ESRD ~25yr (range 15-35); "
            "Females: heterozygous → mild disease usually (haematuria, proteinuria); ESRD in ~12% by age 40; "
            "truncating variants in males = most severe; Glycine substitutions in triple-helix = severe; "
            "no genotype-phenotype correlation for females"
        ),
        "disease_category": (
            "X-linked Alport syndrome (XLAS) — most common Alport type (~80%); "
            "Progressive glomerulopathy: microscopic haematuria → proteinuria → CKD → ESRD; "
            "GBM thinning → splitting/lamellation (seen on EM) → thickening: PATHOGNOMONIC on EM; "
            "SENSORINEURAL HEARING LOSS (SNHL) — high-frequency, bilateral, 90% of males by ESRD; "
            "ANTERIOR LENTICONUS — bilateral lens conical protrusion; PATHOGNOMONIC for Alport; "
            "MACULAR FLECKS (dot-and-fleck retinopathy) — seen in ~70% of males; "
            "ACEi/ARB slows progression; ESRD males ~25yr WITHOUT treatment, later WITH early ACEi"
        ),
        "disease_pathway": (
            "Type IV collagen network in the glomerular basement membrane (GBM) consists of "
            "α3α4α5(IV) heterotrimer exclusively in mature GBM. "
            "This heterotrimer is uniquely resistant to metalloprotease degradation and provides "
            "structural integrity and charge selectivity of the GBM filtration barrier. "
            "COL4A5 mutation → absent or abnormal α5(IV) → the α3α4α5(IV) heterotrimer cannot assemble → "
            "GBM retains the fetal α1α2(IV) composition → thinner, more susceptible to mechanical stress → "
            "progressive GBM splitting, lamellation, thinning (visible on EM as the 'basket weave' pattern) → "
            "podocyte foot process effacement → proteinuria → mesangial expansion → glomerulosclerosis → CKD. "
            "Cochlear and lens basement membranes also contain α3α4α5(IV) → SNHL and anterior lenticonus. "
            "ACEi reduces intra-glomerular pressure and podocyte stress, slowing GFR decline."
        ),
        "pathognomonic": (
            "ANTERIOR LENTICONUS (bilateral) — conical/oil-droplet protrusion of anterior lens; PATHOGNOMONIC for Alport; "
            "requires slit-lamp examination; ophthalmology referral for ALL Alport patients; "
            "GBM EM: irregular thinning + dense lamellation ('basket weave'/'worm-eaten' appearance) — DIAGNOSTIC; "
            "ABSENT α5(IV) on GBM immunofluorescence (indirect staining) — males; "
            "MICROSCOPIC HAEMATURIA from infancy — non-visible (dipstick positive); RBC casts indicate glomerulonephritis; "
            "HIGH-FREQUENCY SENSORINEURAL HEARING LOSS — bilateral, 2000-8000Hz range; progressive; "
            "AUDIOGRAM: bilateral sloping high-frequency loss; confirm with pure tone audiometry"
        ),
        "treatment": (
            "ACEi (ramipril, enalapril) — FIRST LINE; START EARLY (even pre-proteinuria in males with confirmed XLAS); "
            "reduces intraglomerular pressure → slows GFR decline; ESRD delayed by ~10yr in males starting ACEi early; "
            "ARB (losartan, irbesartan) — alternative if ACEi intolerant; COMBINATION not recommended (↑ adverse effects); "
            "TARGET: BP <130/80 (or lower if tolerated); reduce proteinuria to <0.5g/day; "
            "HEARING AIDS — for progressive SNHL; audiological review annually; "
            "ANTERIOR LENTICONUS: no treatment; avoid contact sports (lens rupture risk); cataract surgery if needed; "
            "MACULAR FLECKS: benign; no treatment; annual ophthalmology review; "
            "DIALYSIS/TRANSPLANT: ESRD males ~25yr; kidney transplant CURATIVE; "
            "NOTE: post-transplant anti-GBM antibody disease (anti-GBM nephritis in allograft) — rare complication in Alport; "
            "occurs because males have never seen normal α3α4α5(IV) → immunological naive → antibody response; "
            "WOMEN CARRIERS: annual urine dipstick; GFR monitoring; ACEi if proteinuria >0.5g/day; "
            "GENETIC COUNSELLING: X-linked; maternal carrier → 50% sons affected, 50% daughters carrier"
        ),
        "key_features": [
            "COL4A5 X-linked Alport: ~80% of Alport; males hemizygous → classic ESRD ~25yr; females heterozygous → variable",
            "Anterior lenticonus PATHOGNOMONIC — slit-lamp mandatory; bilateral conical lens protrusion; no other disease causes this",
            "GBM EM: basket-weave/worm-eaten lamellation pathognomonic; α5(IV) absent on immunofluorescence",
            "ACEi EARLY (even pre-proteinuria) delays ESRD ~10yr in males; start at diagnosis of haematuria/proteinuria",
            "SNHL bilateral high-frequency: 2000-8000Hz range; 90% of males by ESRD; annual audiogram",
            "Post-transplant anti-GBM nephritis: rare but critical — Alport males immunologically naive to normal α3α4α5(IV) GBM",
            "Females heterozygous: usually haematuria only; ESRD in ~12% by age 40; annual monitoring mandatory",
            "No disease-specific therapy beyond ACEi/ARB; gene therapy in early preclinical studies",
        ],
        "key_ddx": [
            "COL4A3/COL4A4 AR Alport — same GBM EM findings; both sexes equally severe; no X-linked pedigree",
            "Thin Basement Membrane Nephropathy (TBMN, COL4A3 AD het) — isolated microscopic haematuria; GBM thin NOT lamellated; benign in most",
            "IgA nephropathy — mesangial IgA deposits on immunofluorescence; episodic macroscopic haematuria with upper respiratory tract infections; no SNHL/lenticonus",
            "Fabry disease (GLA) — lysosomal storage; Zebra bodies on EM; low alpha-Gal A; skin angiokeratoma; XLR",
        ],
        "esrd_age": 25,
        "snhl_pct": 90,
        "anterior_lenticonus_pct": 60,
        "macular_fleck_pct": 70,
        "proteinuria_pct": 95,
        "acei_eligible_pct": 100,
        "hypertension_pct": 80,
        "hematuria_pct": 100,
    },
    {
        "gene": "COL4A3",
        "protein": (
            "COL4A3 -- 2q36.3 AR/AD -- 1670aa -- Collagen-Alpha3-IV-186kDa-GBM-Type-IV-"
            "Collagen-Heterotrimer-Alport-AR-TBMN-AD -- OMIM-Gene-120070-Disease-Alport-AR-203780-TBMN-141200"
        ),
        "locus": "2q36.3",
        "protein_size": "1670 aa / 186 kDa",
        "inheritance": (
            "AR (biallelic) → Autosomal Recessive Alport (ARAS) — both sexes equally severe; ESRD ~25yr; "
            "AD (heterozygous) → Thin Basement Membrane Nephropathy (TBMN) — 1% of general population; "
            "mostly isolated microscopic haematuria; 20-25% progress to CKD over decades; "
            "COL4A3 and COL4A4 on chromosome 2q36 — same dual AR/AD phenotype spectrum; "
            "de novo and familial; compound het with COL4A4 rare but reported (digenic Alport)"
        ),
        "disease_category": (
            "AR Alport syndrome — same severity as X-linked Alport (males/females equally severe); "
            "ESRD ~25yr without treatment; SNHL and anterior lenticonus as X-linked Alport; "
            "OR "
            "Thin Basement Membrane Nephropathy (TBMN) — heterozygous COL4A3/COL4A4; "
            "1% of general population; most common cause of persistent microscopic haematuria in young people; "
            "GBM uniformly thin (<150nm) on EM — no lamellation (DDx Alport); "
            "mostly benign but ~20% develop CKD/proteinuria over 40+ years; "
            "ACEi if proteinuria develops"
        ),
        "disease_pathway": (
            "Same final pathway as COL4A5 — disruption of α3α4α5(IV) heterotrimer in mature GBM. "
            "COL4A3 encodes the α3(IV) chain, which is the structural core of the mature heterotrimer. "
            "Biallelic (AR) COL4A3 loss: complete absence of α3(IV) → no α3α4α5 heterotrimer → "
            "GBM retains fetal α1α2(IV) → same progressive lamellation/splitting seen in X-linked Alport → ESRD. "
            "Heterozygous (AD) COL4A3 loss: 50% of α3(IV) produced → enough for thin but structurally "
            "near-normal GBM (thin basement membrane, <150nm) → mechanical vulnerability under glomerular "
            "hydrostatic pressure → low-level haematuria but usually stable GFR for decades (TBMN phenotype). "
            "TBMN progression risk: CKD3+ develops in ~20% after age 50 — monitoring lifelong warranted."
        ),
        "pathognomonic": (
            "AR ALPORT: same as COL4A5 Alport — GBM EM basket-weave lamellation; SNHL; anterior lenticonus; "
            "ABSENT α3(IV) on GBM immunofluorescence — diagnostic (COL4A3 biallelic); "
            "both sexes equally severely affected (unlike X-linked); "
            "TBMN: GBM UNIFORMLY THIN (<150nm on EM) — NO lamellation (distinguishes from Alport); "
            "α3(IV) REDUCED (not absent) on IF — heterozygous haploinsufficiency; "
            "PERSISTENT MICROSCOPIC HAEMATURIA — 100% of TBMN; non-visible; no casts; "
            "FAMILY HISTORY: autosomal dominant pedigree (TBMN); most relatives have haematuria; "
            "NO SNHL, NO LENTICONUS in pure TBMN (absence distinguishes from Alport)"
        ),
        "treatment": (
            "AR ALPORT: same as COL4A5 — early ACEi; SNHL monitoring; ophthalmology; transplant at ESRD; "
            "anti-GBM post-transplant risk same as X-linked males; "
            "TBMN: "
            "Annual urine dipstick + protein:creatinine ratio (PCR) monitoring; "
            "Annual BP check and GFR every 2-3 years; "
            "ACEi/ARB — if proteinuria >0.5g/day or hypertension develops; "
            "LIFESTYLE: maintain healthy BP; avoid NSAIDs; no specific dietary restriction; "
            "PROGNOSIS COUNSELLING: ~80% of TBMN patients maintain normal GFR lifelong; "
            "risk factors for progression: male sex, proteinuria >500mg/day, hypertension, co-existing genes; "
            "GENETIC COUNSELLING: TBMN — het COL4A3 carrier; 50% offspring risk of haematuria; "
            "rare risk of having 2 het carriers as parents → AR Alport child (25% risk)"
        ),
        "key_features": [
            "COL4A3 dual phenotype: AR biallelic → full Alport (ESRD ~25yr, SNHL, lenticonus); AD het → TBMN (mostly benign haematuria)",
            "TBMN is the most common cause of persistent microscopic haematuria in young people (1% population)",
            "GBM EM: TBMN shows uniform thinning (<150nm) — no lamellation; distinguishes from Alport (lamellation = basket-weave)",
            "AR Alport: both sexes equally severe; same ESRD age as X-linked males (~25yr without ACEi)",
            "TBMN 20-25% progress to CKD over decades — lifelong monitoring of proteinuria, GFR, BP mandatory",
            "Absent α3(IV) on GBM IF = AR Alport; reduced (not absent) = TBMN/X-linked female",
            "Post-transplant anti-GBM nephritis risk in AR Alport (same as X-linked males; immunologically naive to normal GBM)",
            "Digenic Alport: compound het COL4A3+COL4A4 reported; COL4A3 and COL4A4 are contiguous on 2q36",
        ],
        "key_ddx": [
            "COL4A5 X-linked Alport — X-linked pedigree; males far more severe than females; same GBM EM",
            "IgA Nephropathy — episodic machaematuria with URTI; mesangial IgA IF; no GBM thinning/lamellation",
            "Thin Basement Membrane Disease (TBMN) with CKD progression — check proteinuria; recheck GFR; ACEi if proteinuria",
            "Benign familial haematuria — clinical term for TBMN before genetic testing; must monitor for progression",
        ],
        "esrd_age": 25,
        "tbmn_progression_pct": 20,
        "snhl_pct": 85,
        "anterior_lenticonus_pct": 55,
        "proteinuria_pct": 90,
        "acei_eligible_pct": 100,
        "hypertension_pct": 75,
        "hematuria_pct": 100,
    },
    {
        "gene": "UMOD",
        "protein": (
            "UMOD -- 16p12.3 AD -- 640aa -- Uromodulin-Tamm-Horsfall-Glycoprotein-85kDa-"
            "Thick-Ascending-Limb-ADTKD-UMOD-Hyperuricaemia -- OMIM-Gene-191845-Disease-ADTKD-UMOD-162000"
        ),
        "locus": "16p12.3",
        "protein_size": "640 aa / 85 kDa",
        "inheritance": (
            "AD; autosomal dominant tubulointerstitial kidney disease type UMOD (ADTKD-UMOD); "
            "formerly called Medullary Cystic Kidney Disease type 2 (MCKD2) or Familial Juvenile Hyperuricaemic Nephropathy (FJHN); "
            "100% penetrance; ESRD age 20-70yr (average ~50yr); "
            "all pathogenic UMOD variants affect conserved cysteine residues → ER misfolding → ER retention"
        ),
        "disease_category": (
            "Autosomal Dominant Tubulointerstitial Kidney Disease — UMOD type (ADTKD-UMOD); "
            "YOUNG-ONSET GOUT in context of SLOWLY PROGRESSIVE CKD = classic triad; "
            "HYPERURICAEMIA — reduced fractional excretion of urate (FEUA <6% = diagnostic); "
            "MEDULLARY CYSTS — small (often not visible on standard ultrasound); "
            "INTERSTITIAL FIBROSIS + TUBULAR ATROPHY on biopsy — tubular uromodulin deposits; "
            "Allopurinol/febuxostat for gout; no disease-modifying therapy for CKD; "
            "transplant at ESRD; UMOD misfolding is druggable (ER quality control restoration — experimental)"
        ),
        "disease_pathway": (
            "Uromodulin (Tamm-Horsfall protein, UMOD) is the most abundant urinary protein and is "
            "expressed exclusively in thick ascending limb (TAL) cells of the loop of Henle. "
            "UMOD is a GPI-anchored protein that polymerises into filamentous structures in the tubular lumen — "
            "forming a protective anti-biofilm barrier and modulating ion transport (NaCl reabsorption via NKCC2). "
            "All pathogenic UMOD variants substitute conserved cysteine residues in the D8C/EGF domains → "
            "unpaired cysteines → protein misfolding → ER retention → ER stress → "
            "UPR (unfolded protein response) activation → TAL cell damage and death → "
            "progressive tubulointerstitial nephritis/fibrosis + medullary cyst formation. "
            "Parallel effect: reduced UMOD secretion → impaired renal urate handling → hyperuricaemia → gout."
        ),
        "pathognomonic": (
            "YOUNG-ONSET GOUT (<35yr) + FAMILY HISTORY OF CKD — cardinal diagnostic clue; "
            "HYPERURICAEMIA WITH LOW FEUA (<6% on spot urine) — tubular secretion defect NOT overproduction; "
            "CKD with SLOWLY PROGRESSIVE course — no haematuria, no proteinuria initially; "
            "MEDULLARY CYSTS on MRI (not CT; cysts are small 1-10mm) — corticomedullary junction; "
            "standard abdominal ultrasound often MISSES these cysts — MRI required; "
            "INTERSTITIAL FIBROSIS on renal biopsy + uromodulin protein aggregates in tubular cells (special staining); "
            "FAMILY HISTORY CKD + GOUT in autosomal dominant pattern"
        ),
        "treatment": (
            "GOUT: Allopurinol (XO inhibitor) or Febuxostat — urate-lowering therapy; "
            "target serum urate <6 mg/dL (360 μmol/L); "
            "Colchicine for acute gout flares (avoid NSAIDs — nephrotoxic); "
            "AVOID probenecid (uricosuric — competes with residual renal urate secretion); "
            "CKD MANAGEMENT: ACEi/ARB if proteinuria develops; BP control; avoid nephrotoxins; "
            "anaemia management (EPO); CKD diet as stage advances; "
            "AVOID: allopurinol allergy — check HLA-B*5801 (risk of Stevens-Johnson syndrome) in high-risk ethnicities; "
            "NO DISEASE-MODIFYING THERAPY for kidney fibrosis at present; "
            "ER quality-control restoration approaches (small molecule chaperones, 4-PBA) — experimental only; "
            "TRANSPLANT: ESRD ~50yr; kidney transplant preferred; UMOD expressed only in native kidney → "
            "DISEASE DOES NOT RECUR in transplant (UMOD not expressed in donor kidney); "
            "GENETIC COUNSELLING: AD; 50% offspring risk; genetic testing available (NGS panel)"
        ),
        "key_features": [
            "UMOD ADTKD: young-onset gout (<35yr) + slowly progressive CKD in AD pedigree = classic triad",
            "Low FEUA (<6%) — reduced tubular urate secretion (NOT overproduction); distinguishes from primary gout",
            "Medullary cysts: small (1-10mm); MRI required — standard USS frequently misses them",
            "All pathogenic variants affect conserved cysteines → ER misfolding → ER retention → tubular cell ER stress",
            "Allopurinol/febuxostat for gout; AVOID NSAIDs (nephrotoxic); AVOID probenecid in CKD",
            "Kidney transplant CURATIVE for renal disease — UMOD disease DOES NOT RECUR post-transplant",
            "No haematuria (unlike glomerular diseases); no proteinuria early — 'silent' CKD until late stage",
            "Interstitial fibrosis + tubular UMOD aggregates on biopsy — pathological hallmark",
        ],
        "key_ddx": [
            "Primary gout (overproduction) — high FEUA (>8%); no family CKD; no medullary cysts; responds to allopurinol",
            "HNF1B ADTKD — same family CKD + cysts; but pancreatic atrophy/agenesis + MODY5 + mullerian anomalies",
            "Nephronophthisis (NPHP) — AR; younger ESRD; same medullary cysts; retinal dystrophy; corticocalyceal junction cysts",
            "Secondary hyperuricaemia (diuretics, cyclosporin, ESRD) — identifiable cause; normal FEUA correction on withdrawal",
        ],
        "esrd_age": 50,
        "gout_pct": 90,
        "medullary_cyst_pct": 80,
        "hyperuricemia_pct": 95,
        "low_feua_pct": 95,
        "hypertension_pct": 70,
        "hematuria_pct": 5,
    },
    {
        "gene": "HNF1B",
        "protein": (
            "HNF1B -- 17q12 AD -- 557aa -- Hepatocyte-Nuclear-Factor-1-Beta-68kDa-Transcription-"
            "Factor-ADTKD-HNF1B-RCAD-MODY5 -- OMIM-Gene-189907-Disease-RCAD-137920"
        ),
        "locus": "17q12",
        "protein_size": "557 aa / 68 kDa",
        "inheritance": (
            "AD; 17q12 deletion (WHOLE GENE DELETION most common ~50%) or point mutations; "
            "ADTKD-HNF1B / Renal Cysts And Diabetes (RCAD) syndrome; "
            "de novo in ~50% (no family history); "
            "MODY5 (maturity-onset diabetes of the young type 5) — diabetes in young adults; "
            "multi-organ transcription factor — renal development, pancreas, liver, mullerian ducts"
        ),
        "disease_category": (
            "Autosomal Dominant Tubulointerstitial Kidney Disease — HNF1B type (ADTKD-HNF1B); "
            "RENAL CYSTS (varied morphology — not typical ADPKD) + DIABETES (MODY5) = RCAD syndrome; "
            "PANCREATIC ATROPHY/AGENESIS — exocrine pancreatic insufficiency; DM often insulin-requiring from onset (exocrine+endocrine); "
            "MULLERIAN DUCT ANOMALIES in females — bicornuate/unicornuate uterus, uterine agenesis, vaginal aplasia; "
            "LIVER TRANSAMINASE ELEVATION — hepatic cysts or cholestasis; "
            "HYPOMAGNESAEMIA — tubular magnesium wasting; "
            "GOUT/HYPERURICAEMIA — similar to UMOD-ADTKD (overlap)"
        ),
        "disease_pathway": (
            "HNF1B (Hepatocyte Nuclear Factor 1 Beta) is a homeodomain transcription factor "
            "expressed during embryonic development in kidney, pancreas, liver, and mullerian ducts. "
            "In the kidney, HNF1B regulates the expression of genes required for tubular differentiation "
            "and cyst-suppression including PKHD1 (fibrocystin), PKD2, and UMOD. "
            "Loss of HNF1B function → dysregulated tubular development → cyst formation of variable morphology "
            "(glomerulocysts, medullary cysts, hypoplastic/dysplastic kidneys — NOT the typical ADPKD bilateral cysts). "
            "In the pancreas: HNF1B regulates PDX1 and beta-cell mass → HNF1B loss → pancreatic hypoplasia/atrophy → "
            "MODY5 (non-autoimmune diabetes, often insulin-requiring). "
            "The 17q12 deletion (~1.4 Mb) removes HNF1B and LHX1 (lateral plate mesoderm gene) — "
            "accounting for additional neurodevelopmental features in some deletion carriers."
        ),
        "pathognomonic": (
            "RENAL CYSTS (varied — NOT typical bilateral ADPKD): can be glomerulocysts, medullary cysts, "
            "renal hypoplasia, horseshoe kidney, or small dysplastic kidney with cysts; "
            "MODY5 DIABETES — young-onset (<35yr), non-obese, family history, insulin-requiring from onset; "
            "NEGATIVE autoimmune antibodies (GAD, IA-2, ZnT8) — distinguishes from Type 1 DM; "
            "PANCREATIC ATROPHY on imaging (MRI abdomen) — reduced pancreatic volume; body-tail most affected; "
            "ELEVATED LIVER TRANSAMINASES without cirrhosis — hepatic cysts or cholestasis; "
            "HYPOMAGNESAEMIA — tubular Mg wasting; fasting Mg <0.7 mmol/L; "
            "MULLERIAN ANOMALIES in females — bicornuate uterus; identified on pelvic MRI; "
            "17q12 DELETION on chromosomal microarray — most common genotype"
        ),
        "treatment": (
            "DIABETES (MODY5): "
            "Insulin therapy — often required from diagnosis (pancreatic hypoplasia + insulin secretion defect); "
            "SULPHONYLUREAS and GLP-1 agonists are LESS effective than in HNF1A-MODY (HNF1B has different biology); "
            "Exocrine pancreatic enzyme replacement (CREON) if steatorrhoea present; "
            "KIDNEY DISEASE: CKD management; ACEi/ARB if proteinuria; "
            "Renal replacement if ESRD (transplant preferred); "
            "HYPOMAGNESAEMIA: oral magnesium supplementation (magnesium glycerophosphate — better absorbed than MgOH); "
            "HYPERURICAEMIA/GOUT: allopurinol/febuxostat; colchicine for flares; "
            "MULLERIAN ANOMALIES: gynecological review for reproductive planning; hysteroscopy/laparoscopy if indicated; "
            "17q12 DELETION: chromosomal microarray — may have additional neurodevelopmental features (LHX1 deletion); "
            "GENETIC COUNSELLING: 50% offspring risk; de novo in ~50%; whole exome or panel including HNF1B + chromosomal microarray"
        ),
        "key_features": [
            "HNF1B ADTKD: renal cysts (variable morphology — NOT ADPKD pattern) + MODY5 + pancreatic atrophy = RCAD syndrome",
            "17q12 deletion most common (~50%): chromosomal microarray required — sequencing alone MISSES deletions",
            "MODY5 diabetes: negative autoimmune antibodies; young non-obese; insulin-requiring; sulphonylureas LESS effective",
            "Pancreatic atrophy on MRI: body-tail affected; exocrine insufficiency → malabsorption; enzyme replacement needed",
            "Hypomagnesaemia: tubular Mg wasting; supplement with magnesium glycerophosphate (better absorbed)",
            "Mullerian anomalies in females: bicornuate/unicornuate uterus; pelvic MRI mandatory in all affected females",
            "De novo in ~50% — family history absent; diagnosis often unexpected finding; microarray in all unexplained renal cysts + diabetes",
            "HNF1B regulates PKHD1 and PKD2 expression — explains why HNF1B loss causes cysts similar to but distinct from ADPKD",
        ],
        "key_ddx": [
            "HNF1A-MODY3 — same MODY phenotype; sulphonylureas HIGHLY effective; no renal cysts; no pancreatic atrophy",
            "Type 1 Diabetes — positive autoimmune antibodies; acute onset; DKA; no renal cysts or pancreatic atrophy",
            "ADPKD (PKD1/PKD2) — symmetric bilateral discrete cysts; no diabetes; no pancreatic atrophy; no mullerian anomalies",
            "UMOD ADTKD — overlapping CKD+gout+medullary cysts; no diabetes, no pancreatic atrophy, no mullerian anomalies",
        ],
        "esrd_age": 45,
        "diabetes_pct": 85,
        "pancreatic_atrophy_pct": 75,
        "hypomagnesaemia_pct": 50,
        "mullerian_anomaly_pct": 40,
        "hypertension_pct": 65,
        "hematuria_pct": 10,
    },
    {
        "gene": "NPHS2",
        "protein": (
            "NPHS2 -- 1q25.2 AR -- 383aa -- Podocin-42kDa-Slit-Diaphragm-Lipid-Raft-"
            "Scaffolding-SRNS2-FSGS-AR -- OMIM-Gene-604766-Disease-SRNS2-600995"
        ),
        "locus": "1q25.2",
        "protein_size": "383 aa / 42 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "Steroid-Resistant Nephrotic Syndrome type 2 (SRNS2); "
            "~20% of childhood steroid-resistant nephrotic syndrome globally; "
            "p.R138Q — most common pathogenic variant in European populations (founder); "
            "CHILDHOOD ONSET: 1-5yr usually; some adolescent/adult onset; "
            "compound heterozygous (p.R138Q + other variant) — very common; "
            "digenic interactions with NPHS1 (nephrin) reported"
        ),
        "disease_category": (
            "Steroid-Resistant Nephrotic Syndrome type 2 (SRNS2) / Focal Segmental Glomerulosclerosis (FSGS); "
            "STEROIDS DO NOT WORK — DEFINING FEATURE; immunosuppression largely ineffective; "
            "nephrotic syndrome: heavy proteinuria (>3.5g/day), hypoalbuminaemia, oedema, hyperlipidaemia; "
            "FSGS on biopsy (focal = some glomeruli; segmental = portion of tuft); "
            "ESRD in childhood/young adult (median ~12yr from diagnosis); "
            "KIDNEY TRANSPLANT IS CURATIVE — disease does NOT recur post-transplant (unlike NPHS1 minimal change); "
            "NO HLA typing abnormality; NO circulating permeability factor"
        ),
        "disease_pathway": (
            "Podocin (NPHS2) is a stomatin-family protein localised exclusively to the slit diaphragm "
            "of glomerular podocytes — forming a critical scaffolding complex at the lipid raft microdomains "
            "of the slit diaphragm. It interacts with Nephrin (NPHS1), CD2AP, and TRPC6 within the slit diaphragm complex. "
            "Podocin's hairpin membrane topology anchors Nephrin to lipid rafts, concentrating signalling "
            "molecules (PI3K/Akt pathway) that regulate podocyte survival and slit diaphragm architecture. "
            "Loss of Podocin → Nephrin mislocalised (cytoplasmic redistribution) → slit diaphragm assembly failure → "
            "podocyte foot process effacement (FPE) → disrupted filtration barrier → massive proteinuria → "
            "podocyte detachment and depletion → scarring (segmental glomerulosclerosis) → progressive GFR loss. "
            "Unlike minimal change disease (also FPE), NPHS2 SRNS has no immune mechanism → "
            "STEROIDS AND CALCINEURIN INHIBITORS DO NOT WORK; no circulating permeability factor (unlike NPHS1)."
        ),
        "pathognomonic": (
            "STEROID-RESISTANT NEPHROTIC SYNDROME — failure to remit after 8 weeks of high-dose prednisolone; "
            "FSGS on biopsy — focal (<50% glomeruli) and segmental (<50% of tuft) sclerosis; "
            "NPHS2 BIALLELIC PATHOGENIC VARIANTS on genetic panel — DIAGNOSTIC; "
            "p.R138Q — most common European variant; may be compound het with p.V290M, p.A284V, p.R229Q; "
            "p.R229Q in compound with p.A284V — adult-onset variant; "
            "PODOCIN ABSENT on GBM immunofluorescence — confirmed with anti-Podocin antibody staining; "
            "NORMAL COMPLEMENT, NEGATIVE ANA, NEGATIVE ANCA, NEGATIVE HEPATITIS B/C — excludes secondary causes; "
            "GENETIC TESTING priority in ALL children with SRNS before starting immunosuppression"
        ),
        "treatment": (
            "STEROIDS: NOT EFFECTIVE — do not start long-course steroids; "
            "SHORT TRIAL of prednisolone (4 weeks) to confirm resistance then STOP to avoid steroid toxicity; "
            "CALCINEURIN INHIBITORS (ciclosporin, tacrolimus): limited/no response in NPHS2-SRNS; "
            "may try short trial if genetic diagnosis uncertain; stop if no partial remission at 6 months; "
            "SUPPORTIVE CARE: "
            "ACEi/ARB — MANDATORY; reduces proteinuria and intraglomerular pressure; "
            "Low sodium diet; fluid restriction if oedema severe; "
            "Diuretics (furosemide) for oedema; albumin infusion if severe hypoalbuminaemia (<15 g/L); "
            "STATIN — dyslipidaemia management (hypercholesterolaemia in nephrotic syndrome); "
            "ANTICOAGULATION: anticoagulate if serum albumin <20 g/L (high VTE risk — renal vein thrombosis); "
            "TRANSPLANT: ESRD ~12yr from onset; kidney transplant CURATIVE; "
            "NOTE: NPHS2-SRNS DOES NOT RECUR post-transplant (unlike FSGS with circulating permeability factor — 20-40% recurrence); "
            "GENETIC COUNSELLING: AR; 25% recurrence; prenatal testing available; siblings: 25% affected"
        ),
        "key_features": [
            "NPHS2 SRNS2: AR podocin deficiency; steroid-RESISTANT is the defining feature — test genetics before prolonged steroids",
            "p.R138Q European founder: most common NPHS2 variant; frequently compound het with second variant",
            "Genetic testing MANDATORY in all childhood SRNS before immunosuppression — avoids steroid toxicity in non-responders",
            "Kidney transplant CURATIVE — disease DOES NOT RECUR (unlike FSGS with circulating factor: 20-40% recurrence)",
            "ACEi/ARB mandatory: reduces proteinuria even without remission; delays CKD progression",
            "Anticoagulate if albumin <20 g/L: renal vein thrombosis risk; pulmonary embolism risk",
            "FSGS on biopsy is the histological pattern — not a diagnosis; genetic/immune/secondary causes all cause FSGS",
            "Podocin anchors nephrin to lipid rafts — NPHS2 loss mislocalises nephrin → slit diaphragm failure → FPE",
        ],
        "key_ddx": [
            "Minimal Change Disease (MCD) — steroid-SENSITIVE; complete remission with prednisolone; FPE on EM but no FSGS",
            "NPHS1-SRNS (nephrin) — congenital nephrotic syndrome; massive proteinuria at birth; Finnish-type nephrotic syndrome",
            "Secondary FSGS (obesity, reflux, solitary kidney, drugs) — no genetic variant; identifiable cause; partial steroid response possible",
            "Primary FSGS with circulating permeability factor — recurs post-transplant (20-40%); plasma exchange reduces recurrence",
        ],
        "esrd_age": 12,
        "steroid_resistance_pct": 100,
        "fsgs_on_biopsy_pct": 90,
        "proteinuria_pct": 100,
        "albumin_below_20_pct": 60,
        "transplant_no_recurrence_pct": 95,
        "hypertension_pct": 60,
        "hematuria_pct": 30,
    },
]

SEEDS = [2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605]


def _simulate_cohort(gene: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pts = []
    g = gene["gene"]
    for _ in range(40):
        age = rng.randint(8, 72)
        sex = rng.choice(["M", "F"])

        if g == "PKD1":
            esrd_age = gene["esrd_age"] + rng.randint(-12, 12)
            tkv_ml = rng.randint(600, 3200)
            gfr = max(5, 90 - max(0, (age - 35)) * 2.5 + rng.gauss(0, 8))
            liver_cysts = rng.random() < 0.82
            ica = rng.random() < 0.11
            tolvaptan = (age < 56) and (tkv_ml > 800) and (gfr > 25)
            hypertension = rng.random() < 0.87
            hematuria = rng.random() < 0.52
            pts.append({
                "age": age, "sex": sex, "esrd_age": esrd_age,
                "tkv_ml": tkv_ml, "gfr": round(gfr, 1),
                "liver_cysts": int(liver_cysts), "ica": int(ica),
                "tolvaptan": int(tolvaptan), "hypertension": int(hypertension),
                "hematuria": int(hematuria), "gene_variant": rng.choice(["Truncating","Missense","Splice"]),
                "mayo_class": rng.choice(["1B","1C","1D","1E"]) if tkv_ml > 750 else "1A",
                "seed": seed
            })
        elif g == "PKD2":
            esrd_age = gene["esrd_age"] + rng.randint(-10, 10)
            tkv_ml = rng.randint(300, 1500)
            gfr = max(5, 90 - max(0, (age - 50)) * 1.5 + rng.gauss(0, 8))
            liver_cysts = rng.random() < 0.52
            ica = rng.random() < 0.07
            tolvaptan = (age < 56) and (tkv_ml > 750) and (gfr > 25)
            hypertension = rng.random() < 0.72
            pts.append({
                "age": age, "sex": sex, "esrd_age": esrd_age,
                "tkv_ml": tkv_ml, "gfr": round(gfr, 1),
                "liver_cysts": int(liver_cysts), "ica": int(ica),
                "tolvaptan": int(tolvaptan), "hypertension": int(hypertension),
                "hematuria": int(rng.random() < 0.36),
                "gene_variant": rng.choice(["Truncating","Frameshift"]),
                "mayo_class": rng.choice(["1C","1D","1E"]) if tkv_ml > 750 else rng.choice(["1A","1B"]),
                "seed": seed
            })
        elif g == "PKHD1":
            age = rng.randint(0, 30)
            gfr = max(5, 80 - age * 2.0 + rng.gauss(0, 10))
            portal_htn = rng.random() < 0.72
            cholangitis = rng.random() < 0.32
            neonatal_death = (age == 0) and (rng.random() < 0.3)
            pts.append({
                "age": age, "sex": sex, "gfr": round(gfr, 1),
                "portal_htn": int(portal_htn), "cholangitis": int(cholangitis),
                "neonatal_death": int(neonatal_death), "esrd_age": 20 + rng.randint(-8, 8),
                "liver_cysts": 1, "ica": 0, "tolvaptan": 0,
                "hypertension": int(rng.random() < 0.90),
                "hematuria": int(rng.random() < 0.20),
                "gene_variant": rng.choice(["Compound het (null+missense)","Null/Null","Missense/Missense"]),
                "seed": seed
            })
        elif g == "COL4A5":
            sex = "M" if rng.random() < 0.6 else "F"
            esrd_age = (25 + rng.randint(-6, 8)) if sex == "M" else (45 + rng.randint(-10, 20))
            snhl = (rng.random() < 0.92) if sex == "M" else (rng.random() < 0.55)
            lenticonus = (rng.random() < 0.62) if sex == "M" else (rng.random() < 0.22)
            proteinuria_g = round(rng.uniform(0.5, 8.0), 1) if age > 12 else round(rng.uniform(0, 1.0), 1)
            gfr = max(5, 85 - max(0, (age - 15)) * 3.0 + rng.gauss(0, 10)) if sex == "M" else max(20, 90 - max(0, (age - 25)) * 1.2 + rng.gauss(0, 8))
            pts.append({
                "age": age, "sex": sex, "esrd_age": esrd_age, "gfr": round(gfr, 1),
                "snhl": int(snhl), "lenticonus": int(lenticonus),
                "proteinuria_g": proteinuria_g, "acei": 1,
                "hypertension": int(rng.random() < 0.82),
                "hematuria": 1,
                "gene_variant": rng.choice(["Truncating","Glycine substitution","Missense"]),
                "seed": seed
            })
        elif g == "COL4A3":
            is_ar_alport = rng.random() < 0.35
            snhl = int(is_ar_alport and rng.random() < 0.85)
            lenticonus = int(is_ar_alport and rng.random() < 0.55)
            gfr = max(5, 90 - (age - 20) * (2.5 if is_ar_alport else 0.5) + rng.gauss(0, 8)) if age > 20 else 85 + rng.gauss(0, 5)
            pts.append({
                "age": age, "sex": sex, "gfr": round(max(5, gfr), 1),
                "is_ar_alport": int(is_ar_alport), "snhl": snhl, "lenticonus": lenticonus,
                "esrd_age": (25 + rng.randint(-5, 8)) if is_ar_alport else (65 + rng.randint(-15, 15)),
                "proteinuria_g": round(rng.uniform(0.2, 5.0), 1) if is_ar_alport else round(rng.uniform(0, 0.3), 2),
                "acei": int(is_ar_alport or rng.random() < 0.3),
                "hypertension": int(is_ar_alport and rng.random() < 0.75),
                "hematuria": 1,
                "gene_variant": rng.choice(["Biallelic truncating","Het missense","Het truncating (TBMN)"]),
                "seed": seed
            })
        elif g == "UMOD":
            age = rng.randint(15, 70)
            gout_age = rng.randint(15, 35) if rng.random() < 0.90 else None
            gfr = max(5, 80 - max(0, (age - 30)) * 1.5 + rng.gauss(0, 10))
            pts.append({
                "age": age, "sex": sex, "gfr": round(gfr, 1),
                "gout": int(gout_age is not None), "gout_onset_age": gout_age or 0,
                "hyperuricemia_mgdl": round(rng.uniform(7.5, 12.0), 1),
                "feua_pct": round(rng.uniform(2.5, 5.8), 1),
                "medullary_cysts_mri": int(rng.random() < 0.82),
                "esrd_age": 50 + rng.randint(-15, 20),
                "allopurinol": int(rng.random() < 0.80),
                "hypertension": int(rng.random() < 0.72),
                "hematuria": int(rng.random() < 0.06),
                "gene_variant": rng.choice(["Cys-substitution EGF","Cys-substitution D8C","Truncating (rare)"]),
                "seed": seed
            })
        elif g == "HNF1B":
            has_diabetes = rng.random() < 0.87
            diabetes_age = rng.randint(15, 40) if has_diabetes else None
            pancreas_atrophy = rng.random() < 0.77
            hypoMg = rng.random() < 0.52
            mullerian = int((sex == "F") and (rng.random() < 0.42))
            renal_cysts = rng.random() < 0.90
            gfr = max(5, 75 - max(0, (age - 30)) * 1.3 + rng.gauss(0, 10))
            pts.append({
                "age": age, "sex": sex, "gfr": round(gfr, 1),
                "diabetes": int(has_diabetes), "diabetes_onset_age": diabetes_age or 0,
                "pancreas_atrophy": int(pancreas_atrophy), "hypoMg": int(hypoMg),
                "mullerian_anomaly": mullerian, "renal_cysts": int(renal_cysts),
                "esrd_age": 45 + rng.randint(-15, 15),
                "hypertension": int(rng.random() < 0.67),
                "hematuria": int(rng.random() < 0.12),
                "gene_variant": rng.choice(["17q12 deletion","Frameshift","Missense homeodomain"]),
                "seed": seed
            })
        elif g == "NPHS2":
            age = rng.randint(1, 30)
            steroid_resistant = True
            proteinuria_g = round(rng.uniform(4.0, 18.0), 1)
            albumin = round(rng.uniform(8, 28), 1)
            fsgs_biopsy = rng.random() < 0.92
            esrd_age = 12 + rng.randint(-4, 8)
            anticoagulated = int(albumin < 20)
            pts.append({
                "age": age, "sex": sex, "gfr": round(max(5, 90 - age * 3.5 + rng.gauss(0, 10)), 1),
                "steroid_resistant": 1, "proteinuria_g": proteinuria_g,
                "albumin_gdl": albumin, "fsgs_biopsy": int(fsgs_biopsy),
                "esrd_age": esrd_age, "anticoagulated": anticoagulated,
                "transplant_no_recurrence": int(rng.random() < 0.96),
                "hypertension": int(rng.random() < 0.62),
                "hematuria": int(rng.random() < 0.32),
                "gene_variant": rng.choice(["p.R138Q/other","Biallelic p.R138Q","Compound het (p.R229Q+p.A284V)"]),
                "seed": seed
            })
    return pts


def generate_overview() -> dict:
    summary_by_gene = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        n = len(pts)
        avg_gfr = round(sum(p.get("gfr", 60) for p in pts) / n, 1)
        avg_esrd = round(sum(p.get("esrd_age", 50) for p in pts) / n, 1)
        htn_pct = round(sum(p.get("hypertension", 0) for p in pts) / n * 100, 1)
        hematuria_pct = round(sum(p.get("hematuria", 0) for p in pts) / n * 100, 1)
        summary_by_gene.append({
            "gene": gene["gene"], "locus": gene["locus"],
            "n_patients": n, "avg_gfr": avg_gfr,
            "avg_esrd_age": avg_esrd, "hypertension_pct": htn_pct,
            "hematuria_pct": hematuria_pct,
        })

    all_pts = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        all_pts.extend(_simulate_cohort(gene, seed))
    total = len(all_pts)
    overall_avg_gfr = round(sum(p.get("gfr", 60) for p in all_pts) / total, 1)
    overall_htn = round(sum(p.get("hypertension", 0) for p in all_pts) / total * 100, 1)

    return {
        "atlas": "Hereditary-Kidney-Disease-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "total_patients": total,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "aggregate_metrics": {
            "overall_avg_gfr": overall_avg_gfr,
            "overall_hypertension_pct": overall_htn,
        },
        "gene_summaries": summary_by_gene,
        "disease_classes": [
            f"{g['gene']} — {g['disease_category'].split(';')[0].strip()}"
            for g in ATLAS_GENES
        ],
        "clinical_pearls": [
            "PKD1 (85% of ADPKD): tolvaptan FDA 2018 slows TKV growth; LFT monitoring mandatory; low BP <110/75 with ACEi/ARB",
            "PKD2 (15% of ADPKD): ESRD 20yr later than PKD1 (~74yr); many die of unrelated causes; genotyping essential for prognosis",
            "PKHD1 ARPKD: massive echogenic kidneys + CHF + portal HTN; null/null → Potter sequence; normal liver synthesis despite portal HTN",
            "COL4A5 X-linked Alport (~80%): anterior lenticonus PATHOGNOMONIC (slit-lamp mandatory); ACEi early delays ESRD ~10yr; SNHL 90% males",
            "COL4A3 AR/AD: biallelic → full Alport (ESRD ~25yr); heterozygous → TBMN (1% population; mostly benign haematuria; 20% progress)",
            "UMOD ADTKD: young gout (<35yr) + CKD + low FEUA (<6%) + medullary cysts on MRI = diagnostic tetrad; allopurinol for gout; AVOID NSAIDs",
            "HNF1B RCAD: renal cysts + MODY5 + pancreatic atrophy + mullerian anomalies; 17q12 deletion (chromosomal microarray, not sequencing alone)",
            "NPHS2 SRNS2: steroid-RESISTANT — STEROIDS DON'T WORK; genetic testing before immunosuppression; kidney transplant CURATIVE (no recurrence)",
            "Anti-GBM post-transplant: Alport males (COL4A5 hemizygous) and AR Alport (COL4A3/COL4A4 biallelic) — immunologically naive to normal GBM",
            "TBMN (COL4A3/COL4A4 het): 1% general population; most common cause of unexplained microscopic haematuria; GBM thin NOT lamellated on EM",
        ],
    }


def generate_breakdown() -> dict:
    gene_breakdowns = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        n = len(pts)
        avg_gfr = round(sum(p.get("gfr", 60) for p in pts) / n, 1)
        avg_esrd = round(sum(p.get("esrd_age", 50) for p in pts) / n, 1)
        htn_pct = round(sum(p.get("hypertension", 0) for p in pts) / n * 100, 1)
        hematuria_pct = round(sum(p.get("hematuria", 0) for p in pts) / n * 100, 1)
        liver_pct = round(sum(p.get("liver_cysts", 0) for p in pts) / n * 100, 1) if any("liver_cysts" in p for p in pts) else None

        entry = {
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"],
            "disease_category": gene["disease_category"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "n_patients": n,
            "avg_gfr": avg_gfr,
            "avg_esrd_age": avg_esrd,
            "hypertension_pct": htn_pct,
            "hematuria_pct": hematuria_pct,
        }
        if liver_pct is not None:
            entry["liver_cyst_pct"] = liver_pct

        # Gene-specific metrics
        if gene["gene"] == "PKD1":
            entry["avg_tkv_ml"] = round(sum(p.get("tkv_ml", 1200) for p in pts) / n)
            entry["ica_pct"] = round(sum(p.get("ica", 0) for p in pts) / n * 100, 1)
            entry["tolvaptan_eligible_pct"] = round(sum(p.get("tolvaptan", 0) for p in pts) / n * 100, 1)
        elif gene["gene"] == "PKD2":
            entry["avg_tkv_ml"] = round(sum(p.get("tkv_ml", 700) for p in pts) / n)
            entry["ica_pct"] = round(sum(p.get("ica", 0) for p in pts) / n * 100, 1)
            entry["tolvaptan_eligible_pct"] = round(sum(p.get("tolvaptan", 0) for p in pts) / n * 100, 1)
        elif gene["gene"] == "PKHD1":
            entry["portal_htn_pct"] = round(sum(p.get("portal_htn", 0) for p in pts) / n * 100, 1)
            entry["cholangitis_pct"] = round(sum(p.get("cholangitis", 0) for p in pts) / n * 100, 1)
            entry["neonatal_death_pct"] = round(sum(p.get("neonatal_death", 0) for p in pts) / n * 100, 1)
        elif gene["gene"] in ("COL4A5", "COL4A3"):
            entry["snhl_pct"] = round(sum(p.get("snhl", 0) for p in pts) / n * 100, 1)
            entry["anterior_lenticonus_pct"] = round(sum(p.get("lenticonus", 0) for p in pts) / n * 100, 1)
            entry["avg_proteinuria_g"] = round(sum(p.get("proteinuria_g", 1.0) for p in pts) / n, 2)
            if gene["gene"] == "COL4A3":
                entry["ar_alport_pct"] = round(sum(p.get("is_ar_alport", 0) for p in pts) / n * 100, 1)
        elif gene["gene"] == "UMOD":
            entry["gout_pct"] = round(sum(p.get("gout", 0) for p in pts) / n * 100, 1)
            entry["avg_uric_acid_mgdl"] = round(sum(p.get("hyperuricemia_mgdl", 9.0) for p in pts) / n, 1)
            entry["avg_feua_pct"] = round(sum(p.get("feua_pct", 4.5) for p in pts) / n, 1)
            entry["medullary_cysts_mri_pct"] = round(sum(p.get("medullary_cysts_mri", 0) for p in pts) / n * 100, 1)
        elif gene["gene"] == "HNF1B":
            entry["diabetes_pct"] = round(sum(p.get("diabetes", 0) for p in pts) / n * 100, 1)
            entry["pancreas_atrophy_pct"] = round(sum(p.get("pancreas_atrophy", 0) for p in pts) / n * 100, 1)
            entry["hypoMg_pct"] = round(sum(p.get("hypoMg", 0) for p in pts) / n * 100, 1)
            entry["mullerian_anomaly_pct"] = round(sum(p.get("mullerian_anomaly", 0) for p in pts) / n * 100, 1)
        elif gene["gene"] == "NPHS2":
            entry["steroid_resistance_pct"] = 100.0
            entry["fsgs_biopsy_pct"] = round(sum(p.get("fsgs_biopsy", 0) for p in pts) / n * 100, 1)
            entry["avg_proteinuria_g"] = round(sum(p.get("proteinuria_g", 8.0) for p in pts) / n, 1)
            entry["anticoagulation_pct"] = round(sum(p.get("anticoagulated", 0) for p in pts) / n * 100, 1)
            entry["transplant_no_recurrence_pct"] = round(sum(p.get("transplant_no_recurrence", 0) for p in pts) / n * 100, 1)

        gene_breakdowns.append(entry)
    return {"gene_breakdowns": gene_breakdowns}


def generate_definitions() -> dict:
    gene_entries = {}
    for gene in ATLAS_GENES:
        gene_entries[gene["gene"]] = {
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip(),
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment_summary": gene["treatment"].split(";")[0].strip() + "...",
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
        }

    return {
        "gene_entries": gene_entries,
        "kidney_disease_glossary": {
            "ADPKD — Autosomal Dominant Polycystic Kidney Disease": (
                "ADPKD is the most common life-threatening monogenic kidney disease (~1:400-1:1000 births). "
                "Bilateral progressive renal cyst expansion → CKD → ESRD. "
                "PKD1 (~85%): ESRD ~54yr; PKD2 (~15%): ESRD ~74yr. "
                "Tolvaptan (V2R antagonist) FDA 2018 — first disease-modifying therapy; slows TKV growth; mandatory LFT monitoring. "
                "Extrarenal features: liver cysts (80%), intracranial aneurysm (10%), cardiac valve abnormalities (25%). "
                "Mayo Clinic Classification (1A-1E) based on htTKV growth rate — Class 1C-1E eligible for tolvaptan. "
                "ACEi/ARB low BP target (<110/75 in young patients) slows TKV growth and GFR decline (HALT-PKD trial)."
            ),
            "Alport Syndrome — Type IV Collagen GBM Disease": (
                "Alport syndrome is caused by mutations in COL4A3, COL4A4, or COL4A5 — encoding the α3α4α5(IV) collagen heterotrimer "
                "exclusive to mature glomerular, cochlear, and ocular basement membranes. "
                "X-linked (COL4A5, ~80%): males ESRD ~25yr; females variable. "
                "Autosomal Recessive (COL4A3/COL4A4 biallelic, ~15%): both sexes ESRD ~25yr. "
                "Autosomal Dominant (COL4A3/COL4A4 het) = TBMN. "
                "Classic triad: microscopic haematuria + progressive CKD + sensorineural hearing loss. "
                "Pathognomonic: ANTERIOR LENTICONUS on slit-lamp. "
                "GBM EM: basket-weave lamellation (Alport) vs uniform thinning <150nm (TBMN). "
                "ACEi EARLY delays ESRD ~10yr. Transplant curative; anti-GBM post-transplant risk in males/AR Alport."
            ),
            "ADTKD — Autosomal Dominant Tubulointerstitial Kidney Disease": (
                "ADTKD is a clinico-pathological diagnosis of progressive CKD (tubulointerstitial fibrosis) "
                "with AD inheritance. Four main genetic subtypes: "
                "ADTKD-UMOD (uromodulin, 16p12.3): young gout + low FEUA + medullary cysts; "
                "ADTKD-HNF1B (17q12): renal cysts + MODY5 + pancreatic atrophy + mullerian anomalies; "
                "ADTKD-MUC1: cytosine duplication in GC-rich region — not detectable by standard NGS; "
                "ADTKD-REN (renin, 1q32): hyperkalaemia + hyperuricaemia + anaemia without EPO response. "
                "Shared features: CKD without haematuria/proteinuria; medullary/corticomedullary cysts often <1cm; "
                "normal urinalysis early; interstitial fibrosis on biopsy."
            ),
            "ARPKD — Autosomal Recessive Polycystic Kidney Disease (PKHD1)": (
                "ARPKD (1:20,000 births) presents in neonates/infants with massively enlarged echogenic kidneys "
                "and congenital hepatic fibrosis (CHF). "
                "Fibrocystin/polyductin (PKHD1) localised to primary cilia of collecting duct cells and cholangiocytes. "
                "Null/null genotype → Potter sequence: oligohydramnios → pulmonary hypoplasia → neonatal death. "
                "Survivors: slowly progressive CKD + portal hypertension from CHF. "
                "KEY POINT: Liver SYNTHETIC function is NORMAL despite CHF and portal hypertension — "
                "biliary fibrosis without hepatocellular damage. "
                "Combined liver-kidney transplant if both advanced."
            ),
            "NPHS2 SRNS — Podocin-Deficiency Steroid-Resistant Nephrotic Syndrome": (
                "NPHS2 encodes Podocin — a stomatin-family protein critical for slit diaphragm integrity at lipid rafts. "
                "Loss of Podocin → Nephrin (NPHS1) mislocalised from slit diaphragm → foot process effacement → "
                "massive proteinuria → FSGS → ESRD ~12yr. "
                "STEROIDS DO NOT WORK — steroid resistance defines the condition. "
                "Calcineurin inhibitors also largely ineffective. "
                "ACEi/ARB mandatory to reduce proteinuria. "
                "p.R138Q European founder — most common variant; frequently compound heterozygous. "
                "Kidney transplant is CURATIVE — disease does NOT recur post-transplant "
                "(unlike primary FSGS with circulating factor which recurs in 20-40% of allografts)."
            ),
            "Tolvaptan (Jynarque) — V2R Antagonist for ADPKD": (
                "Tolvaptan is a selective vasopressin V2-receptor antagonist that reduces cAMP-mediated fluid secretion "
                "into renal cysts and epithelial proliferation in collecting duct cells. "
                "FDA approved 2018 for adults with ADPKD at risk of rapid progression. "
                "Eligibility: age 18-55, CKD stage 1-3 (eGFR >25), rapidly progressive disease "
                "(htTKV >750 ml/m or htTKV 1C-1E per Mayo classification, or htTKV annual growth >5%). "
                "Aquaresis side effects: polyuria, nocturia, polydipsia — take first dose in morning. "
                "HEPATOTOXICITY: rare but serious — FDA REMS programme; LFTs monthly for first 18 months, "
                "every 3 months thereafter; stop if ALT/AST >3×ULN. "
                "CI: liver disease, hyponatraemia risk, inability to access water. "
                "Slows TKV growth by ~50% and delays eGFR decline."
            ),
            "GBM Electron Microscopy — Alport vs TBMN": (
                "The glomerular basement membrane (GBM) on electron microscopy (EM) is critical for Alport/TBMN diagnosis. "
                "NORMAL GBM: uniform electron-dense lamina densa, 320-340nm thick. "
                "ALPORT GBM: irregular thinning alternating with thickening; lamellation/splitting of lamina densa "
                "creating 'basket-weave' or 'worm-eaten' appearance — PATHOGNOMONIC; "
                "must include ALL three changes (thinning, thickening, lamellation) for Alport diagnosis. "
                "TBMN GBM: UNIFORMLY THIN (<150nm); NO lamellation; only thinning; "
                "GBM thin throughout (uniform diffuse) — absence of lamellation distinguishes from Alport. "
                "Alpha chain immunofluorescence staining: α5(IV) absent = X-linked Alport males; "
                "α3(IV) absent = AR Alport; reduced α3(IV)/α5(IV) = TBMN/het females."
            ),
            "Mayo Clinic Classification of ADPKD Progression": (
                "Mayo Clinic image classification stratifies ADPKD patients by TKV growth rate for tolvaptan eligibility. "
                "Uses height-adjusted TKV (htTKV) and patient age to classify into 5 subclasses: "
                "Class 1A: htTKV <200 ml/m (growth <1.5%/yr) — slowest; tolvaptan NOT indicated; "
                "Class 1B: htTKV 200-350 ml/m (~1.5-2.8%/yr) — mild; watchful waiting; "
                "Class 1C: htTKV 350-500 ml/m (~2.8-4.5%/yr) — moderate; TOLVAPTAN ELIGIBLE; "
                "Class 1D: htTKV 500-750 ml/m (~4.5-6.0%/yr) — rapid; tolvaptan eligible; "
                "Class 1E: htTKV >750 ml/m (>6%/yr) — most rapid; strong tolvaptan indication. "
                "Kidney length >17cm on ultrasound correlates with Class 1C-1E in most patients."
            ),
        },
    }


# ── DB population ──────────────────────────────────────────────────────────────
def populate_db():
    import sqlite3, os
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")
    db_path = os.path.normpath(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS hereditary_kidney_disease_atlas")
    cur.execute("""
        CREATE TABLE hereditary_kidney_disease_atlas (
            id INTEGER PRIMARY KEY,
            gene TEXT,
            protein TEXT,
            aa_length INTEGER,
            chromosome TEXT,
            inheritance TEXT,
            disease_category TEXT,
            patient_age INTEGER,
            sex TEXT,
            gfr REAL,
            esrd_age INTEGER,
            hypertension INTEGER,
            hematuria INTEGER,
            gene_variant TEXT,
            seed INTEGER
        )
    """)
    row_id = 1
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        aa_length = int(gene["protein_size"].split(" ")[0])
        for p in pts:
            cur.execute("""
                INSERT INTO hereditary_kidney_disease_atlas VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                row_id, gene["gene"], gene["protein"][:120], aa_length, gene["locus"],
                gene["inheritance"].split(";")[0].strip(),
                gene["disease_category"].split(";")[0].strip(),
                p.get("age", 0), p.get("sex", "M"),
                p.get("gfr", 60.0), p.get("esrd_age", 50),
                p.get("hypertension", 0), p.get("hematuria", 0),
                p.get("gene_variant", "Unknown"), seed
            ))
            row_id += 1
    conn.commit()
    conn.close()
    print(f"Populated hereditary_kidney_disease_atlas: {row_id - 1} rows in {db_path}")


if __name__ == "__main__":
    populate_db()
    ov = generate_overview()
    print(f"Overview: {ov['total_patients']} patients, {ov['n_genes']} genes, seeds {ov['seeds']}")
    bd = generate_breakdown()
    print(f"Breakdown: {len(bd['gene_breakdowns'])} gene entries")
    df = generate_definitions()
    print(f"Definitions: {len(df['gene_entries'])} gene entries, {len(df['kidney_disease_glossary'])} glossary terms")
