#!/usr/bin/env python3
"""Hereditary-MSUD-BCAA-Disorder-Atlas -- Complete 8-Gene Reference
Maple Syrup Urine Disease and Branched-Chain Amino Acid Catabolism Disorders
(BCKDHA/MSUD-Ia-Mennonite-Alloisoleucine-PATHOGNOMONIC-Leucine-Toxic-Thiamine-Responsive-LiTx-Curative,
 BCKDHB/MSUD-Ib-E1beta-HeteroTetramer-SamePhenotype-E1alpha,
 DBT/MSUD-II-E2-Scaffold-IntermittentMilder,
 DLD/MSUD-III-COMBINED-PDH-KGDH-BCKDH-LacticAcidosis-AshkenaziJewish-PATHOGNOMONIC,
 BCKDK/BCAA-LOW-Syndrome-OPPOSITE-MSUD-Autism-Epilepsy-BCAA-Supp-NBS-MISSES-PATHOGNOMONIC,
 PPM1K/MSUD-V-Intermittent-FebrileRegression-BCKDH-Phosphatase,
 HIBCH/VCatabolism-MethacrylylCoA-CysteineToxicity-LeighLike-PATHOGNOMONIC,
 MCCC1/3MCC-LeucineCatabolism-BENIGN-MostCommon-NBS-C5OH-PATHOGNOMONIC-MaternalNBS):

BCKDHA (Branched-chain alpha-keto acid dehydrogenase E1α subunit; 445aa; 19q13.2; AR-LOF;
         BCKDHA encodes the E1α catalytic subunit of the mitochondrial BCKDH (Branched-Chain
           Keto acid Dehydrogenase) complex; BCKDH is a multi-enzyme complex (E1α₂β₂ + E2 + E3)
           analogous to PDH and alpha-KGDH; BCKDH catalyses oxidative decarboxylation of the
           branched-chain alpha-keto acids (BCKAs): alpha-ketoisocaproate (KIC, from leucine),
           alpha-keto-beta-methylvalerate (KMV, from isoleucine), and alpha-ketoisovalerate
           (KIV, from valine) → corresponding acyl-CoAs → eventually acetyl-CoA / propionyl-CoA;
         BCKDHA LOF → complete BCKDH failure → BCKAs and BCCAs accumulate systemically;
         MAPLE SYRUP URINE DISEASE (MSUD) TYPE IA:
         PATHOGNOMONIC BIOCHEMICAL MARKERS:
           (1) ALLOISOLEUCINE: formed non-enzymatically from the alpha-keto acid of isoleucine;
               detected in plasma amino acids by ion-exchange chromatography; ALLOISOLEUCINE
               IS PATHOGNOMONIC FOR MSUD — absent in all other conditions;
               even 1-5 µmol/L alloisoleucine = confirmed MSUD;
           (2) ELEVATED BCCAs: leucine (primary toxin) >> isoleucine > valine elevated;
               leucine >400 µmol/L = encephalopathy risk; leucine >1000 µmol/L = coma/death;
           (3) MAPLE SYRUP ODOR: urine and cerumen (earwax) maple-syrup/fenugreek smell;
               from condensed BCKAs (particularly sotolon = 4,5-dimethyl-3-hydroxy-2(5H)-furanone);
         NEONATAL PRESENTATION (Classic MSUD): normal birth → days 4-7 encephalopathy develops;
           poor feeding, altered tone (hypertonicity/opisthotonus alternating with hypotonia),
           cerebral oedema on MRI (reversible with treatment), respiratory failure, death if untreated;
         NBS POSITIVE: elevated plasma leucine (>200 µmol/L cut-off in most programs);
         THIAMINE-RESPONSIVE VARIANT: ~15-20% of BCKDHA mutations have partial thiamine
           (vitamin B1) responsiveness; thiamine 100-300 mg/day can reduce BCKA burden;
           ALL classic MSUD should receive thiamine trial before ruling out;
         MENNONITE FOUNDER MUTATION: p.Y393N homozygous — carrier rate 1:10 in Old Order
           Mennonite; MSUD prevalence 1:358 Mennonite (vs 1:185,000 general population);
           NBS in Pennsylvania Mennonite community transformed outcomes;
         TREATMENT: LEUCINE RESTRICTION diet; maintain plasma leucine 75-300 µmol/L;
           ACUTE CRISIS: IV glucose/insulin (anabolism), dialysis (if leucine >1500), BCAA-free formula;
           LIVER TRANSPLANTATION (LiTx): curative biochemically — donor liver provides BCKDH activity;
           transplanted patients can eat liberally without diet restriction; however does NOT reverse
           existing neurological damage; best outcomes with early transplant at 6-12 months;
         seed SEED_BASE+0).

BCKDHB (Branched-chain alpha-keto acid dehydrogenase E1β subunit; 392aa; 6q14.1; AR-LOF;
         BCKDHB encodes the E1β regulatory subunit of the BCKDH complex; the E1 component forms
           an α₂β₂ heterotetramer — BCKDHA (2 subunits) + BCKDHB (2 subunits); E1β does NOT
           contain the thiamine pyrophosphate (TPP) catalytic site (that is on E1α) but is
           essential for proper E1 heterotetrameric folding, stability, and activity;
         BCKDHB LOF → BCKDH complex destabilized → identical biochemical phenotype to BCKDHA;
         MSUD TYPE IB: clinically identical to Type Ia — same alloisoleucine elevation, same
           leucine toxicity, same neonatal encephalopathy, same maple syrup odor;
         FREQUENCY: 25-35% of all MSUD cases (BCKDHA = 35-45%, BCKDHB = 25-35%, others smaller);
         THIAMINE RESPONSIVENESS: BCKDHB mutations are LESS likely to be thiamine-responsive
           (thiamine primarily helps stabilise E1α thiamine-pyrophosphate binding); BCKDHB
           mutations should still receive thiamine trial but response rate lower (~5-10%);
         GENOTYPE-PHENOTYPE: similar range to BCKDHA — classic/intermediate/intermittent forms
           depending on residual BCKDH complex activity from BCKDHB variant pathogenicity;
         DDx from BCKDHA: biochemically identical; distinguish only by molecular gene panel;
         TREATMENT: identical to BCKDHA: leucine restriction; acute management same protocol;
           liver transplant curative — donor BCKDHB-sufficient liver normalises complex;
         seed SEED_BASE+1).

DBT    (Dihydrolipoamide branched-chain acyltransferase / BCKDH E2 subunit; 482aa; 1p21.2; AR-LOF;
         DBT encodes the E2 (dihydrolipoamide acetyltransferase) core scaffold subunit of the
           BCKDH multi-enzyme complex; E2 forms a 24-mer core to which E1 (α₂β₂) and E3 (LL)
           associate; DBT LOF → BCKDH E2 core absent → entire complex non-functional;
         MSUD TYPE II: ~5% of MSUD cases; DBT deficiency tends to have more RESIDUAL BCKDH
           activity than BCKDHA/B deficiency, resulting in a broader clinical spectrum;
         CLINICAL SPECTRUM:
           (1) Classic MSUD (severe DBT LOF mutations): same neonatal presentation as Types Ia/Ib;
           (2) INTERMEDIATE MSUD: onset weeks to months; developmental delay; acute crises during
               febrile illness; alloisoleucine present; leucine 300-500 µmol/L baseline;
           (3) INTERMITTENT MSUD: near-normal plasma BCCAs between illnesses; acute metabolic
               crises ONLY with catabolic stress (illness, surgery, fasting); normal NBS possible;
         INTERMITTENT MSUD DIAGNOSTIC TRAP: NBS may be NORMAL between crises; alloisoleucine
           only detectable during acute crises; repeat NBS during illness recommended if suspected;
         DBT: note that E2 of BCKDH (DBT) is structurally homologous but biochemically distinct
           from E2 of PDH (DLAT) and E2 of alpha-KGDH (DLST) — loss of DBT ONLY affects BCKDH;
         TREATMENT: leucine restriction calibrated to plasma leucine (more relaxed in intermediate/
           intermittent); avoid prolonged fasting; emergency protocol for febrile illness;
           liver transplant effective for classic/severe forms; intermediate may not require LiTx;
         seed SEED_BASE+2).

DLD    (Dihydrolipoamide dehydrogenase / BCKDH-PDH-KGDH shared E3 subunit; 474aa; 7q31.1; AR-LOF;
         DLD encodes the E3 subunit (dihydrolipoamide dehydrogenase / lipoamide dehydrogenase) that
           is SHARED among THREE mitochondrial alpha-keto acid dehydrogenase complexes:
             (1) Pyruvate Dehydrogenase (PDH): E3 component
             (2) Alpha-Ketoglutarate Dehydrogenase (alpha-KGDH / alpha-KGD): E3 component
             (3) Branched-Chain Keto Acid Dehydrogenase (BCKDH): E3 component
           E3 oxidises dihydrolipoamide back to lipoamide using FAD → NADH; if E3 absent, ALL
           three complexes are simultaneously non-functional;
         MSUD TYPE III / DLD DEFICIENCY (LIPOAMIDE DEHYDROGENASE DEFICIENCY):
         PATHOGNOMONIC COMBINED ORGANIC ACIDEMIA:
           (1) LACTIC ACIDOSIS: PDH failure → pyruvate cannot enter TCA → lactic acidosis;
           (2) ALPHA-KETOGLUTARIC ACIDURIA: alpha-KGDH failure → alpha-ketoglutarate accumulates;
           (3) BCAA ELEVATION + ALLOISOLEUCINE: BCKDH failure → BCKAs/BCCAs accumulate;
           TRIAD of lactic acidosis + alpha-ketoglutaric aciduria + branched-chain aminoaciduria
           = PATHOGNOMONIC for DLD deficiency; no other single enzyme deficiency causes all three;
         CLINICAL: severe neonatal acidosis; hypoglycemia; hypotonia; hepatomegaly; cardiomyopathy;
           often fatal in neonatal period without aggressive treatment; survivors have intellectual
           disability + ataxia + Leigh-like neuropathology;
         ASHKENAZI JEWISH FOUNDER: p.G229C (most common in Ashkenazi Jewish) + p.D479V second;
           carrier rate ~1:100 in Ashkenazi Jewish population; prenatal testing important;
         TREATMENT: thiamine (E3 cofactor FAD) + riboflavin; dichloroacetate (DCA) to activate PDH;
           leucine restriction for BCKDH component; LIVER TRANSPLANT NOT FULLY CURATIVE (alpha-KGDH
           and PDH E3 deficiency persist in non-hepatic tissues after LiTx → continued lactic acidosis);
         seed SEED_BASE+3).

BCKDK  (Branched-chain alpha-keto acid dehydrogenase kinase; 382aa; 16q22.1; AR-LOF;
         BCKDK encodes BCKDH kinase, the mitochondrial kinase that PHOSPHORYLATES and INACTIVATES
           the BCKDH complex; BCKDK is the key regulatory enzyme that TURNS OFF BCKDH activity by
           phosphorylating E1α Ser293 → prevents leucine catabolism → conserves plasma BCAAs;
         BCKDK LOSS OF FUNCTION = INVERSE MSUD / BCAA-DEFICIENCY SYNDROME:
           LOF → BCKDH constitutively ACTIVE (never phosphorylated → never inactivated) →
           BCCAs CONTINUOUSLY AND EXCESSIVELY DEGRADED → plasma BCCAs VERY LOW;
         PATHOGNOMONIC BIOCHEMICAL PROFILE:
           PLASMA BCCAs ALL VERY LOW: leucine often <50 µmol/L (normal 100-200 µmol/L);
           isoleucine and valine also very low; ALLOISOLEUCINE ABSENT; NO MAPLE SYRUP ODOR;
           LOW BCAA PROFILE IS PATHOGNOMONIC — opposite direction from MSUD and all other
           BCAA catabolism disorders; if NBS measures elevated BCCAs (for MSUD), BCKDK
           deficiency is MISSED because BCCAs are low (NBS specifically looks for HIGH values);
         NBS MISSES BCKDK DEFICIENCY COMPLETELY;
         CLINICAL PHENOTYPE:
           AUTISM SPECTRUM DISORDER (ASD): prominent; non-verbal or minimally verbal autism;
           INTELLECTUAL DISABILITY: moderate-severe; expressive language most affected;
           SEIZURES/EPILEPSY: various seizure types; may be medication-refractory;
           Onset: infancy-early childhood; often delayed diagnosis (autism/epilepsy workup first);
         KEY CLINICAL CLUE: autistic/epileptic child + low plasma BCAAs → BCKDK;
         TREATMENT: ORAL BCAA SUPPLEMENTATION (leucine + isoleucine + valine);
           reports of dramatic improvement in autistic features, language, and seizure control
           in some patients with prompt supplementation; normalises plasma BCAAs;
           BCAA supplementation dose: leucine 100-200 mg/kg/day in divided doses;
           IMPORTANT: AVOID dietary protein restriction (makes BCCAs even lower); HIGH-protein diet
           + BCAA supplements; branched-chain amino acid infusions during acute illness;
         seed SEED_BASE+4).

PPM1K  (Protein phosphatase 2C mitochondrial / PP2Cm; 359aa; 4q22.1; AR-LOF;
         PPM1K encodes PP2Cm (mitochondrial-targeted protein phosphatase 2C), the Mg²⁺-dependent
           serine phosphatase that DEPHOSPHORYLATES and ACTIVATES the BCKDH complex by removing
           the inhibitory phosphate from E1α Ser293 placed there by BCKDK;
         BCKDK (kinase) INACTIVATES BCKDH; PPM1K (phosphatase) REACTIVATES BCKDH;
         PPM1K LOF → BCKDH cannot be reactivated → BCCAs accumulate = intermittent MSUD;
         MSUD TYPE V (PPM1K DEFICIENCY):
         CLINICAL PATTERN:
           (1) INTERMITTENT MSUD: plasma BCCAs near-normal between catabolic events;
               during illness/fasting/catabolism: BCCAs rise rapidly → acute metabolic crises;
           (2) NEUROLOGICAL REGRESSION: episodes of Leigh-like regression correlating with crises;
               MRI: reversible diffusion restriction in basal ganglia, brainstem during crises;
           (3) MILDER BASELINE: between crises, children may be developmentally normal or mildly delayed;
           (4) LEUCINE RISES WITH STRESS: distinguish from dietary protein excess; crises triggered
               by catabolic stress (fever, surgery, prolonged fasting), NOT by protein overload;
         NBS VARIABLE: may be normal or mildly elevated; intermittent pattern = risk of NBS miss;
         DIAGNOSTIC: plasma amino acids during febrile illness + plasma BCCAs + alloisoleucine;
           PPM1K molecular testing; fibroblast BCKDH enzyme assay (chronically inhibited low activity);
         TREATMENT: leucine restriction ONLY during illness/crises; avoid prolonged fasting;
           emergency protocol: IV glucose + BCAA-free formula during catabolic stress;
           dichloroacetate: may help by lowering pyruvate (does not directly affect BCKDH-PPM1K axis);
         seed SEED_BASE+5).

HIBCH  (3-Hydroxyisobutyryl-CoA hydrolase; 378aa; 2q32.2; AR-LOF;
         HIBCH encodes 3-hydroxyisobutyryl-CoA hydrolase, a mitochondrial enzyme in the VALINE-
           SPECIFIC catabolism pathway; valine catabolism proceeds: valine → isobutyryl-CoA →
           methacrylyl-CoA → 3-hydroxyisobutyryl-CoA → [HIBCH here] → 3-hydroxyisobutyrate →
           methylmalonate semialdehyde → methylmalonyl-CoA → succinyl-CoA (TCA entry);
         HIBCH LOF → 3-hydroxyisobutyryl-CoA accumulates → upstream methacrylyl-CoA ALSO accumulates;
         METHACRYLYL-CoA MECHANISM OF TOXICITY:
           METHACRYLYL-CoA is an α,β-UNSATURATED THIOESTER — highly electrophilic; reacts with
           free cysteine residues on proteins and free glutathione → CYSTEINE DEPLETION;
           depletes mitochondrial thiol buffers → OXIDATIVE STRESS → mitochondrial complex dysfunction;
         PATHOGNOMONIC FEATURE:
           LEIGH-LIKE PROGRESSIVE NEUROLOGICAL DISEASE: regression, brainstem + basal ganglia lesions
           on MRI (T2 high signal) resembling Leigh syndrome BUT WITHOUT typical Leigh gene mutations;
           PLASMA BCCA NOT SIGNIFICANTLY ELEVATED (valine catabolism DISTAL to BCKDH complex);
           NO ALLOISOLEUCINE (no BCKDH involvement); NO MAPLE SYRUP ODOR;
           Leigh-like + low/normal BCCAs + absence of BCKDH complex organic acids = HIBCH DDx;
         BIOCHEMICAL CLUE: urine 3-hydroxyisobutyrate elevated; methylmalonate may be mildly elevated;
           acylcarnitine: C4OH (3-hydroxybutyrylcarnitine) may be elevated on NBS (non-specific);
           methacrylyl-CoA-modified proteins (cysteine adducts) not routinely measured;
         CLINICAL: progressive; onset infancy-early childhood; hypotonia; regression; MRI Leigh-like;
           episodes of metabolic decompensation with febrile illness; elevated lactate;
         TREATMENT: NO ESTABLISHED TARGETED THERAPY;
           VALINE RESTRICTION: reduces methacrylyl-CoA substrate burden (controversial, theoretically);
           N-ACETYLCYSTEINE (NAC): replenishes cysteine/glutathione; may mitigate methacrylyl-CoA toxicity;
           carnitine supplementation; riboflavin (mitochondrial support); antioxidants;
           avoid prolonged fasting and catabolic stress;
         seed SEED_BASE+6).

MCCC1  (3-Methylcrotonyl-CoA carboxylase 1 alpha subunit; 725aa; 3q27.1; AR-LOF;
         MCCC1 encodes the α-subunit (biotin-binding/carboxyl-transferase subunit) of 3-MCC
           (3-methylcrotonyl-CoA carboxylase), a biotin-dependent mitochondrial enzyme in the
           LEUCINE-SPECIFIC catabolism pathway: leucine → acetoacetyl-CoA → HMG-CoA → acetyl-CoA;
         3-MCC STEP: 3-methylcrotonyl-CoA → [MCCC1+MCCC2 + biotin] → 3-methylglutaconyl-CoA;
         MCCC1 LOF → 3-methylcrotonyl-CoA accumulates → metabolised to 3-hydroxyisovalerylcarnitine
           (C5-OH) and 3-methylcrotonylglycine (urine organic acid);
         3-MCC DEFICIENCY / 3-METHYLCROTONYLGLYCINURIA:
         MOST IMPORTANT CLINICAL PEARL: MAJORITY OF CASES ARE BENIGN (CLINICALLY SILENT);
           3-MCC deficiency is one of the MOST COMMON ORGANIC ACIDURIAS detected by NBS
           (1:36,000-50,000 in US; up to 1:20,000 in some European populations);
           up to 90% of NBS-detected 3-MCC cases are ASYMPTOMATIC THROUGHOUT LIFE;
           AVOID OVER-TREATMENT: many patients treated with carnitine and leucine restriction
           who would have been healthy without intervention;
         NBS BIOCHEMICAL MARKER: elevated C5-OH (3-hydroxyisovalerylcarnitine) = PATHOGNOMONIC
           metabolite for 3-MCC deficiency on tandem MS NBS; urine 3-methylcrotonylglycine elevated;
           MATERNAL 3-MCC: mother with 3-MCC deficiency may have elevated C5-OH → newborn NBS
           elevates C5-OH via transplacental transfer → positive NBS in ASYMPTOMATIC MOTHER;
           maternal 3-MCC is MOST COMMON CAUSE of positive NBS for 3-MCC deficiency — test the MOTHER;
         SYMPTOMATIC MINORITY: ~5-10% of MCCC1 cases; hypoglycemia + metabolic encephalopathy
           during febrile illness; Reye-like episodes; hypoketotic hypoglycemia;
           elevated liver transaminases during crises; neonatal form rarely: neonatal acute;
         DISTINCTION FROM BIOTINIDASE DEFICIENCY (BTD) AND HOLOCARBOXYLASE SYNTHETASE (HLCS):
           BTD/HLCS deficiency → MULTIPLE carboxylase deficiency (3-MCC + PC + PCC + ACC affected);
           3-MCC isolated deficiency: ONLY 3-MCC affected; biotin treatment NOT indicated for isolated;
           urinary organic acids: 3-MCC alone + lactic acid + propionic acid in BTD/HLCS deficiency;
         TREATMENT: ONLY symptomatic patients require intervention;
           carnitine supplementation (repletes secondary carnitine deficiency from C5-OH excretion);
           avoid prolonged fasting in symptomatic cases;
           leucine restriction ONLY for symptomatic patients during crises (not chronically);
           biotin NOT useful for isolated MCCC1 deficiency (biotin works for BTD/HLCS, not isolated 3-MCC);
         seed SEED_BASE+7).
"""
import random

SEED_BASE = 4182
GENES = ["BCKDHA", "BCKDHB", "DBT", "DLD", "BCKDK", "PPM1K", "HIBCH", "MCCC1"]

GENE_META = {
    "BCKDHA": {
        "full_name": "Branched-chain alpha-keto acid dehydrogenase E1α subunit (BCKDHA)",
        "aa": 445, "kda": 50, "locus": "19q13.2", "inheritance": "AR",
        "disease": "MSUD Type Ia (Classic / Intermediate / Thiamine-Responsive)",
        "omim_gene": "608348", "omim_disease": "248600",
        "nbs": True,
        "pathognomonic": "ALLOISOLEUCINE in plasma PATHOGNOMONIC for all MSUD; maple syrup odor cerumen/urine; neonatal encephalopathy days 4-7; leucine >400 µmol/L toxic; Mennonite founder p.Y393N (1:10 carrier); thiamine trial (B1) 100-300 mg/day in ALL; LiTx curative; avoid leucine >150-300 µmol/L",
        "key_drug": "Leucine restriction diet (keep leucine 75-300 µmol/L); acute: IV glucose/insulin/dialysis; thiamine 100-300 mg/day (15-20% responsive); liver transplant curative at 6-12 months",
        "precision_therapy": "Alloisoleucine monitoring; LiTx (donor liver provides BCKDH, patient eats liberally post-LiTx); thiamine-responsive subgroup: confirm with BCKDH enzyme assay; BCAA-free formula for acute management",
        "avoid": "High protein during crises; prolonged fasting (catabolism releases BCCAs); delay in leucine restriction (irreversible brain damage); BCAA supplements (leucine rises further)",
    },
    "BCKDHB": {
        "full_name": "Branched-chain alpha-keto acid dehydrogenase E1β subunit (BCKDHB)",
        "aa": 392, "kda": 44, "locus": "6q14.1", "inheritance": "AR",
        "disease": "MSUD Type Ib (same clinical as Type Ia, lower thiamine responsiveness)",
        "omim_gene": "248611", "omim_disease": "248600",
        "nbs": True,
        "pathognomonic": "ALLOISOLEUCINE present (same as Type Ia); maple syrup odor; E1α/E1β heterotetramer — loss of either → same complex failure; BCKDHB less likely thiamine-responsive (~5-10%); distinguish from BCKDHA only by molecular testing",
        "key_drug": "Leucine restriction (identical protocol to BCKDHA); thiamine trial (lower expectation of response); liver transplant curative",
        "precision_therapy": "BCKDH enzyme assay in fibroblasts; molecular MSUD gene panel (BCKDHA+BCKDHB+DBT); thiamine loading test; LiTx for classic severe forms",
        "avoid": "Same avoidances as BCKDHA: leucine spike during fever/illness; prolonged fasting; over-restriction of isoleucine/valine (must balance all 3 BCCAs)",
    },
    "DBT": {
        "full_name": "Dihydrolipoamide branched-chain acyltransferase / BCKDH E2 (DBT)",
        "aa": 482, "kda": 53, "locus": "1p21.2", "inheritance": "AR",
        "disease": "MSUD Type II (Intermediate / Classic / Intermittent spectrum)",
        "omim_gene": "248610", "omim_disease": "248600",
        "nbs": True,
        "pathognomonic": "ALLOISOLEUCINE when elevated BCCAs; NBS MAY BE NORMAL in intermittent form; residual BCKDH activity from DBT determines severity spectrum; Intermittent MSUD: normal plasma BCCAs between crises — diagnostic trap for NBS programs",
        "key_drug": "Leucine restriction calibrated to phenotype severity; emergency illness protocol; LiTx for classic forms; monitoring with plasma amino acids",
        "precision_therapy": "BCKDH enzyme activity measurement (E2 component measured separately); repeat NBS/amino acids DURING febrile illness if intermittent suspected; BCAA-free formula emergency supply",
        "avoid": "Missing intermittent MSUD (NBS normal between crises); failure to provide emergency protocol card; high-protein diet during febrile illness in intermittent cases",
    },
    "DLD": {
        "full_name": "Dihydrolipoamide dehydrogenase / shared E3 subunit (DLD)",
        "aa": 474, "kda": 54, "locus": "7q31.1", "inheritance": "AR",
        "disease": "MSUD Type III / DLD Deficiency (combined PDH + alpha-KGDH + BCKDH failure)",
        "omim_gene": "238331", "omim_disease": "246900",
        "nbs": True,
        "pathognomonic": "COMBINED TRIAD: LACTIC ACIDOSIS (PDH↓) + ALPHA-KETOGLUTARIC ACIDURIA (KGDH↓) + ALLOISOLEUCINE/BCAA↑ (BCKDH↓) PATHOGNOMONIC — NO OTHER SINGLE ENZYME CAUSES ALL THREE; Ashkenazi Jewish founders p.G229C+p.D479V; liver transplant does NOT fully correct (KGDH/PDH persist in non-hepatic tissues)",
        "key_drug": "Thiamine (B1) + riboflavin (B2) as E3 cofactors; dichloroacetate (DCA) to activate PDH; leucine restriction; aggressive lactic acidosis management; carnitine",
        "precision_therapy": "Measure PDH + alpha-KGDH + BCKDH enzyme activities (all three reduced in DLD deficiency); plasma amino acids + urine organic acids + lactate/pyruvate; molecular DLD analysis; liver transplant NOT fully curative",
        "avoid": "Liver transplant alone (DLD deficiency in non-hepatic tissues — KGDH/PDH deficiency persists post-LiTx); missing combined diagnosis (urine organic acids essential: check alpha-ketoglutarate AND BCKAs AND lactate together)",
    },
    "BCKDK": {
        "full_name": "Branched-chain alpha-keto acid dehydrogenase kinase (BCKDK)",
        "aa": 382, "kda": 43, "locus": "16q22.1", "inheritance": "AR",
        "disease": "BCKDK Deficiency / BCAA-Low Syndrome (OPPOSITE of MSUD: BCCAs are VERY LOW)",
        "omim_gene": "614712", "omim_disease": "614923",
        "nbs": False,
        "pathognomonic": "BCAA-LOW PATHOGNOMONIC: plasma leucine often <50 µmol/L (normal 100-200); alloisoleucine ABSENT; NO maple syrup odor; NBS MISSES (NBS looks for HIGH BCCAs not LOW); autistic/epileptic child + low plasma BCCAs = BCKDK until proven; BCAA supplementation — dramatic improvement in some",
        "key_drug": "ORAL BCAA SUPPLEMENTATION (leucine + isoleucine + valine); high-protein diet; leucine 100-200 mg/kg/day; avoid dietary protein restriction",
        "precision_therapy": "Plasma amino acids (all show low BCCAs); BCKDK molecular testing (can target therapies); BCKDH enzyme assay fibroblasts (constitutively active, non-phosphorylatable); protein-rich diet + BCAA supplement optimisation; autism/epilepsy management co-management",
        "avoid": "BCAA restriction (worsens BCAA depletion — OPPOSITE of MSUD management); prolonged fasting; low-protein diet; missing diagnosis by relying on NBS alone (NBS is NEGATIVE in BCKDK)",
    },
    "PPM1K": {
        "full_name": "Protein phosphatase 2C mitochondrial / PP2Cm (PPM1K)",
        "aa": 359, "kda": 41, "locus": "4q22.1", "inheritance": "AR",
        "disease": "MSUD Type V / PPM1K Deficiency (Intermittent MSUD, catabolic crises)",
        "omim_gene": "611065", "omim_disease": "615135",
        "nbs": False,
        "pathognomonic": "INTERMITTENT: near-normal BCCAs between crises; BCKDH cannot be reactivated (PPM1K phosphatase absent); alloisoleucine ONLY during crises; Leigh-like MRI during crises (reversible); CATABOLIC-STRESS TRIGGERED (not dietary); NBS may be NORMAL or borderline",
        "key_drug": "Emergency illness protocol: IV glucose + BCAA-free formula during catabolic stress; leucine restriction ONLY during illness; avoid prolonged fasting",
        "precision_therapy": "Plasma amino acids during AND between crises (to show intermittent pattern); BCKDH enzyme assay (shows constitutively inactive complex); molecular PPM1K testing; BCAA-free formula for illness management kit",
        "avoid": "Chronic leucine restriction in intermittent form (unnecessary; affects growth/development); failing to provide emergency protocol card; missing diagnosis because NBS and between-crisis amino acids are normal",
    },
    "HIBCH": {
        "full_name": "3-Hydroxyisobutyryl-CoA hydrolase (HIBCH)",
        "aa": 378, "kda": 43, "locus": "2q32.2", "inheritance": "AR",
        "disease": "HIBCH Deficiency / Valine Catabolism Disorder (Leigh-Like, Methacrylyl-CoA Toxicity)",
        "omim_gene": "610690", "omim_disease": "250620",
        "nbs": False,
        "pathognomonic": "LEIGH-LIKE PATHOGNOMONIC without BCKDH complex involvement: BCCAs NOT elevated; alloisoleucine ABSENT; NO maple syrup odor; METHACRYLYL-CoA depletes cysteine/glutathione → mitochondrial dysfunction → Leigh-like MRI (basal ganglia/brainstem T2 hyperintensity); urine 3-hydroxyisobutyrate elevated; DDx Leigh: absent BCKDHA/B/DBT mutations + normal BCCAs",
        "key_drug": "N-acetylcysteine (NAC) to replenish cysteine/glutathione; valine restriction (reduces methacrylyl-CoA substrate); carnitine supplementation; antioxidants; avoid catabolism",
        "precision_therapy": "Urine organic acids (3-hydroxyisobutyrate); HIBCH molecular testing; whole-exome sequencing in Leigh-like without standard Leigh mutations; NAC dose: 70-100 mg/kg/day; valine restriction effect variable",
        "avoid": "BCAA restriction broadly (only valine is specifically relevant); missing diagnosis in Leigh-like without elevated BCCAs (standard MSUD panel misses HIBCH); surgery/prolonged fasting without metabolic support",
    },
    "MCCC1": {
        "full_name": "3-Methylcrotonyl-CoA carboxylase 1 alpha-subunit (MCCC1)",
        "aa": 725, "kda": 82, "locus": "3q27.1", "inheritance": "AR",
        "disease": "3-MCC Deficiency / 3-Methylcrotonylglycinuria (MOST CASES BENIGN; Leucine catabolism)",
        "omim_gene": "609010", "omim_disease": "210200",
        "nbs": True,
        "pathognomonic": "C5-OH (3-HYDROXYISOVALERYLCARNITINE) ELEVATED on NBS PATHOGNOMONIC marker; MOST CASES CLINICALLY SILENT (NBS over-diagnoses); MATERNAL 3-MCC: MOST COMMON CAUSE of positive NBS — check MOTHER first; distinct from BTD/HLCS (isolated 3-MCC; biotin NOT indicated; biotin for BTD/HLCS only); symptomatic minority: hypoketotic hypoglycemia during illness",
        "key_drug": "SYMPTOMATIC PATIENTS ONLY: carnitine supplement (secondary carnitine deficiency); avoid prolonged fasting; leucine restriction only during crises; NO biotin (biotin for BTD/HLCS NOT isolated 3-MCC)",
        "precision_therapy": "ALWAYS test mother: if C5-OH elevated in asymptomatic mother → maternal 3-MCC, newborn NBS is normal (maternofetal transfer); MCCC1+MCCC2 molecular testing; confirm with urine 3-methylcrotonylglycine; enzyme assay (leukocytes/fibroblasts)",
        "avoid": "OVER-TREATMENT: most asymptomatic 3-MCC patients do NOT need leucine restriction or carnitine; biotin supplementation (ineffective for isolated 3-MCC; may mask BTD/HLCS if misused); missing maternal 3-MCC diagnosis (tests the mother, not just infant)",
    },
}


def _make_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = GENE_META[gene]

    n = 40
    # Onset age distributions (months)
    onset_map = {
        "BCKDHA": (0, 0), "BCKDHB": (0, 0), "DBT": (1, 2),
        "DLD": (0, 0), "BCKDK": (18, 12), "PPM1K": (8, 6),
        "HIBCH": (6, 4), "MCCC1": (0, 1),  # NBS detection neonatal
    }
    om, os_ = onset_map.get(gene, (2, 2))
    onset_months = max(0, int(rng.gauss(om, os_)))

    # Sex distribution (mostly equal for AR)
    frac = 0.50 + rng.gauss(0, 0.03)
    male_n = int(round(n * max(0.35, min(0.65, frac))))
    female_n = n - male_n

    # NBS detection rate
    nbs_detected_map = {
        "BCKDHA": 0.90, "BCKDHB": 0.88, "DBT": 0.65,
        "DLD": 0.72, "BCKDK": 0.00, "PPM1K": 0.15,
        "HIBCH": 0.05, "MCCC1": 0.99,
    }
    nbs_pct = nbs_detected_map.get(gene, 0.5) + rng.uniform(-0.03, 0.03)
    nbs_pct = max(0.0, min(1.0, nbs_pct))

    # Liver transplant rate (only for classic MSUD types)
    litx_map = {
        "BCKDHA": 0.35, "BCKDHB": 0.30, "DBT": 0.12,
        "DLD": 0.08, "BCKDK": 0.00, "PPM1K": 0.03,
        "HIBCH": 0.00, "MCCC1": 0.00,
    }
    litx_pct = litx_map.get(gene, 0.0) + rng.uniform(-0.03, 0.03)
    litx_pct = max(0.0, min(1.0, litx_pct))

    # Thiamine responsiveness
    thiamine_resp_map = {
        "BCKDHA": 0.18, "BCKDHB": 0.07, "DBT": 0.10,
        "DLD": 0.12, "BCKDK": 0.00, "PPM1K": 0.00,
        "HIBCH": 0.00, "MCCC1": 0.00,
    }
    thiamine_pct = thiamine_resp_map.get(gene, 0.0) + rng.uniform(-0.02, 0.02)
    thiamine_pct = max(0.0, min(1.0, thiamine_pct))

    # Alloisoleucine present
    alloisoleucine_map = {
        "BCKDHA": True, "BCKDHB": True, "DBT": True,
        "DLD": True, "BCKDK": False, "PPM1K": False,  # only during crises
        "HIBCH": False, "MCCC1": False,
    }

    bcaa_direction_map = {
        "BCKDHA": "HIGH", "BCKDHB": "HIGH", "DBT": "HIGH-to-normal",
        "DLD": "HIGH + lactic acidosis", "BCKDK": "LOW (opposite MSUD)",
        "PPM1K": "HIGH only during crises", "HIBCH": "Normal (valine-specific)",
        "MCCC1": "Normal (leucine catabolism only)",
    }

    return {
        "gene": gene,
        "n": n,
        "male_n": male_n,
        "female_n": female_n,
        "median_onset_months": onset_months,
        "nbs_detection_pct": round(nbs_pct, 3),
        "liver_transplant_pct": round(litx_pct, 3),
        "thiamine_responsive_pct": round(thiamine_pct, 3),
        "alloisoleucine_present": alloisoleucine_map.get(gene, False),
        "bcaa_direction": bcaa_direction_map.get(gene, "variable"),
        "pathognomonic": g["pathognomonic"],
        "key_drug": g["key_drug"],
        "disease": g["disease"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-MSUD-BCAA-Disorder-Atlas",
        "total_patients": 320,
        "genes": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "theme": "Branched-chain amino acid catabolism: BCKDH complex subunit defects, regulatory kinase/phosphatase disorders, peripheral BCAA catabolism enzymes",
        "pathognomonic_highlights": [
            "BCKDHA/MSUD-Ia: ALLOISOLEUCINE PATHOGNOMONIC for all MSUD; neonatal days 4-7 encephalopathy; Mennonite founder p.Y393N (1:10 carrier); leucine primary toxin; thiamine trial ALL; LiTx curative",
            "BCKDHB/MSUD-Ib: clinically IDENTICAL to Ia; E1α+E1β heterotetramer — loss of either → same failure; BCKDHB less thiamine-responsive (~5-10% vs ~18%); distinguish by molecular only",
            "DBT/MSUD-II: INTERMITTENT FORM NBS TRAP — NBS NORMAL between crises; alloisoleucine only during illness; repeat NBS during fever; residual E2 activity determines severity",
            "DLD/MSUD-III: COMBINED TRIAD PATHOGNOMONIC: lactic acidosis (PDH↓) + alpha-ketoglutaric aciduria (KGDH↓) + alloisoleucine (BCKDH↓) SIMULTANEOUSLY; Ashkenazi Jewish founders; LiTx NOT fully curative",
            "BCKDK/BCAA-LOW: OPPOSITE OF MSUD — BCCAs VERY LOW; NBS MISSES ENTIRELY; autistic/epileptic + low BCCAs = BCKDK; BCAA supplementation (NOT restriction); do NOT protein-restrict",
            "PPM1K/MSUD-V: INTERMITTENT; near-normal BCCAs between crises; catabolic-stress triggered (not dietary); Leigh-like reversible MRI during crises; NBS may be normal",
            "HIBCH/Valine: LEIGH-LIKE without BCKDH involvement; BCCAs NORMAL; methacrylyl-CoA depletes cysteine; NAC supplementation logical; valine restriction reduces substrate",
            "MCCC1/3-MCC: MOSTLY BENIGN (90% asymptomatic); MOST COMMON CAUSE positive NBS C5-OH = MATERNAL 3-MCC (test the MOTHER first); biotin NOT indicated (biotin = BTD/HLCS, not isolated 3-MCC)",
        ],
        "gene_cohorts": cohorts,
    }


def generate_breakdown() -> dict:
    cohorts = {g: _make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)}
    return {
        "atlas": "Hereditary-MSUD-BCAA-Disorder-Atlas",
        "pathway_groups": {
            "bckdh_core_complex": {
                "genes": ["BCKDHA", "BCKDHB", "DBT", "DLD"],
                "mechanism": "BCKDH complex (E1α₂β₂ / E2 / E3 multienzyme complex) catalyses oxidative decarboxylation of branched-chain alpha-keto acids; BCKDHA/BCKDHB form the α₂β₂ E1 heterotetrameric catalytic unit; DBT forms the E2 24-mer scaffold; DLD is the shared E3 FAD-dependent oxidoreductase also shared with PDH and alpha-KGDH",
                "shared_feature": "ALLOISOLEUCINE ELEVATION (pathognomonic for all BCKDH complex defects); maple syrup odor; elevated plasma leucine/isoleucine/valine; neonatal encephalopathy in classic forms; NBS positive (except intermittent forms between crises)",
                "key_ddx": "BCKDHA vs BCKDHB: identical phenotype, molecular panel only; DBT: wider spectrum (intermittent may miss NBS); DLD: COMBINED triad (lactic acidosis + alpha-ketoglutarate + alloisoleucine) distinguishes from BCKDHA/B/DBT (BCAA elevation only)",
            },
            "bckdh_regulatory_kinase_phosphatase": {
                "genes": ["BCKDK", "PPM1K"],
                "mechanism": "BCKDK kinase INACTIVATES BCKDH by phosphorylating E1α Ser293; PPM1K phosphatase ACTIVATES BCKDH by removing the phosphate; balanced kinase/phosphatase activity maintains appropriate BCKDH flux. BCKDK LOF → constitutively active BCKDH → BCCAs continuously degraded → BCAA-LOW syndrome; PPM1K LOF → BCKDH cannot be reactivated → BCCAs accumulate under catabolism → intermittent MSUD",
                "shared_feature": "Neither has maple syrup odor in most presentations; both have alloisoleucine only in crisis (PPM1K) or absent entirely (BCKDK); NBS often misses both (BCKDK: BCCAs too LOW; PPM1K: BCCAs between normal range between crises)",
                "key_ddx": "BCKDK: BCCA-LOW (opposite direction); autism + epilepsy + low plasma BCCAs; supplement BCCAs. PPM1K: BCCAs intermittently HIGH (only during catabolic stress); Leigh-like MRI during crises; restrict leucine only during illness",
            },
            "peripheral_bcaa_catabolism": {
                "genes": ["HIBCH", "MCCC1"],
                "mechanism": "HIBCH acts in the VALINE-specific catabolism pathway (downstream of BCKDH); MCCC1/MCCC2 acts in LEUCINE-specific catabolism (also downstream of BCKDH); neither enzyme is part of the BCKDH complex itself, so BCCAs at the BCKDH step are NOT elevated — these are downstream single-BCAA catabolism defects",
                "shared_feature": "BCCAs NOT elevated (no BCKDH involvement); alloisoleucine ABSENT; no maple syrup odor; NBS may detect MCCC1 via C5-OH but often maternal; HIBCH often missed on NBS",
                "key_ddx": "HIBCH vs Leigh syndrome: BCCAs normal + urine 3-hydroxyisobutyrate elevated + methacrylyl-CoA cysteine depletion → HIBCH; MCCC1: most cases benign; NBS C5-OH → check mother for maternal 3-MCC first; BTD/HLCS: multiple carboxylase deficiency (propionic + lactic acidosis ALSO elevated in BTD/HLCS, NOT in isolated 3-MCC)",
            },
        },
        "bcaa_direction_cascade": [
            "Step 1: Plasma amino acids (leucine, isoleucine, valine + alloisoleucine)",
            "Step 2: BCCAs HIGH + alloisoleucine → BCKDH complex defect (BCKDHA/BCKDHB/DBT/DLD)",
            "Step 3: BCCAs HIGH + lactic acidosis + alpha-ketoglutaric aciduria → DLD deficiency (COMBINED triad)",
            "Step 4: BCCAs HIGH + lactic acidosis + BCKAs but NO alpha-ketoglutarate → BCKDHA/BCKDHB/DBT",
            "Step 5: BCCAs LOW (leucine <50 µmol/L) + autism/epilepsy → BCKDK deficiency; supplement BCCAs",
            "Step 6: BCCAs INTERMITTENTLY HIGH only during fever/catabolism → PPM1K or DBT intermittent",
            "Step 7: BCCAs NORMAL + Leigh-like MRI + urine 3-hydroxyisobutyrate → HIBCH",
            "Step 8: C5-OH on NBS + normal BCCAs → 3-MCC deficiency; test MOTHER first (maternal 3-MCC most common)",
        ],
        "treatment_pathways": [
            {
                "gene": "BCKDHA",
                "disease": "MSUD Type Ia",
                "first_line": "Leucine restriction (target plasma leucine 75-300 µmol/L); BCAA-free metabolic formula",
                "precision_therapy": "Thiamine (B1) 100-300 mg/day trial in ALL classic/intermediate MSUD; liver transplant at 6-12 months for classic severe; post-LiTx diet liberalisation",
                "avoid": "High protein during illness; fasting (catabolism); leucine >400 µmol/L (encephalopathy); isoleucine/valine over-restriction (causes secondary deficiencies)",
                "monitoring": "Plasma amino acids (alloisoleucine, leucine) weekly in infancy; MRI for brain oedema in acute crisis",
            },
            {
                "gene": "BCKDHB",
                "disease": "MSUD Type Ib",
                "first_line": "Leucine restriction (identical to Type Ia); BCAA-free formula; thiamine trial (lower expectation)",
                "precision_therapy": "Molecular MSUD gene panel (distinguish BCKDHA/B); LiTx curative for classic; BCKDH enzyme assay for thiamine responsiveness testing",
                "avoid": "Same as BCKDHA; expecting high thiamine response rate (BCKDHB less responsive)",
                "monitoring": "Plasma amino acids; branched-chain keto acids in urine; leucine monitoring",
            },
            {
                "gene": "DBT",
                "disease": "MSUD Type II",
                "first_line": "Leucine restriction calibrated to severity; emergency illness protocol; BCAA-free formula for illness",
                "precision_therapy": "Amino acids during AND between crises (to characterise intermittent vs classic); LiTx for classic/severe DBT",
                "avoid": "Assuming NBS normal = disease excluded (intermittent DBT has normal NBS between crises)",
                "monitoring": "Amino acids during every febrile illness; alloisoleucine as crisis marker",
            },
            {
                "gene": "DLD",
                "disease": "DLD Deficiency / MSUD Type III",
                "first_line": "Thiamine (B1) + riboflavin (B2) + lipoic acid + leucine restriction + dichloroacetate (DCA)",
                "precision_therapy": "Measure ALL THREE complex activities (PDH + alpha-KGDH + BCKDH) in fibroblasts; LiTx NOT fully curative; DCA for lactic acidosis; Ashkenazi Jewish prenatal testing",
                "avoid": "Liver transplant as sole treatment (non-hepatic PDH/KGDH persist); missing combined diagnosis by checking only BCAA panel (must also check lactate/pyruvate + urine organic acids for alpha-ketoglutarate)",
                "monitoring": "Lactate/pyruvate ratio; plasma amino acids; urine organic acids (alpha-ketoglutarate + BCKAs)",
            },
            {
                "gene": "BCKDK",
                "disease": "BCKDK Deficiency / BCAA-Low Syndrome",
                "first_line": "Oral BCAA supplementation (leucine + isoleucine + valine 100-200 mg/kg/day); high-protein diet",
                "precision_therapy": "Plasma amino acids to confirm low BCCAs; BCKDK molecular testing; BCAA supplement dose titration to plasma leucine 100-200 µmol/L; autism/epilepsy co-management",
                "avoid": "BCAA restriction (worsens depletion); low-protein diet; NBS reliance (NBS misses BCKDK entirely)",
                "monitoring": "Plasma amino acids (leucine target 100-200 µmol/L); autism/developmental assessments; seizure diary",
            },
            {
                "gene": "PPM1K",
                "disease": "MSUD Type V / PPM1K Deficiency",
                "first_line": "Emergency illness protocol (IV glucose + BCAA-free formula during catabolic stress); avoid fasting",
                "precision_therapy": "Leucine restriction ONLY during crises (not chronically); BCAA-free formula illness kit; molecular PPM1K testing",
                "avoid": "Chronic leucine restriction between crises (unnecessary; stunts growth); missing diagnosis (normal NBS + normal between-crisis amino acids = not MSUD by NBS)",
                "monitoring": "Amino acids during every febrile illness; brain MRI during crises (reversible Leigh-like changes)",
            },
            {
                "gene": "HIBCH",
                "disease": "HIBCH Deficiency / Valine Catabolism Disorder",
                "first_line": "N-acetylcysteine (NAC) 70-100 mg/kg/day; valine restriction; carnitine supplement; avoid prolonged fasting",
                "precision_therapy": "Urine organic acids (3-hydroxyisobutyrate); HIBCH molecular testing or WES; valine restriction degree empirical; NAC cysteine replenishment logical",
                "avoid": "Broad BCAA restriction (only valine relevant); missing diagnosis in Leigh-like without standard mitochondrial mutations",
                "monitoring": "Urine 3-hydroxyisobutyrate; neuroimaging MRI during regression; mitochondrial function tests",
            },
            {
                "gene": "MCCC1",
                "disease": "3-MCC Deficiency (mostly benign; check mother first)",
                "first_line": "SYMPTOMATIC ONLY: carnitine supplement; avoid prolonged fasting; NO biotin; leucine restriction only during crises",
                "precision_therapy": "ALWAYS check maternal plasma/urine (maternal 3-MCC most common cause of positive NBS); confirm with urine 3-methylcrotonylglycine; MCCC1+MCCC2 molecular panel",
                "avoid": "OVER-TREATMENT of asymptomatic NBS-detected 3-MCC; biotin supplementation (ineffective for isolated 3-MCC; biotin = BTD/HLCS deficiency); leucine restriction in asymptomatic patients",
                "monitoring": "Urine 3-methylcrotonylglycine; plasma C5-OH; clinical follow-up only for symptomatic patients; maternal testing",
            },
        ],
        "hlh_risk_table": {
            "BCKDHA": "Brain oedema risk HIGH in acute crises (leucine >600 µmol/L); metabolic stroke risk; LiTx prevents long-term dietary crises",
            "BCKDHB": "Same cerebral oedema risk as BCKDHA; lower thiamine responsiveness (less intervention option)",
            "DBT": "Brain oedema during intermittent crises; MISSED DIAGNOSES risk (normal NBS between crises = false negative)",
            "DLD": "MOST SEVERE: combined PDH+KGDH+BCKDH failure → severe neonatal lactic acidosis + encephalopathy; often fatal without aggressive management",
            "BCKDK": "Autism/epilepsy chronicity; missed diagnosis (NBS normal); response to BCAA supplements can be dramatic if diagnosed early",
            "PPM1K": "Reversible Leigh-like crises; risk of permanent brain damage if crises repeated without treatment; good prognosis if identified",
            "HIBCH": "Progressive Leigh-like neurodegeneration; methacrylyl-CoA cysteine depletion mechanism; NAC may slow progression",
            "MCCC1": "MOSTLY BENIGN; symptomatic minority: hypoketotic hypoglycemia crises; over-treatment harms asymptomatic patients (false positive NBS risk)",
        },
    }


def generate_definitions() -> dict:
    return {
        "key_distinguishers": {
            "alloisoleucine_rule": "ALLOISOLEUCINE in plasma = MSUD confirmed (pathognomonic); absent in BCKDK, PPM1K (between crises), HIBCH, MCCC1. Alloisoleucine forms from alpha-keto-beta-methylvalerate (isoleucine keto acid) by non-enzymatic transamination — only accumulates when BCKDH complex is defective.",
            "bcaa_direction_rule": "BCAA-HIGH + alloisoleucine → BCKDH complex defect (BCKDHA/B/DBT/DLD). BCAA-LOW (leucine <50) + autism/epilepsy → BCKDK deficiency (supplement, do NOT restrict). BCAA-NORMAL + Leigh-like MRI → HIBCH or PPM1K (intermittent)",
            "nbs_miss_alert": "NBS misses: (1) BCKDK (BCCAs LOW — NBS only detects HIGH leucine); (2) PPM1K intermittent (BCCAs normal between crises); (3) HIBCH (no BCKDH metabolites). Always repeat amino acids DURING febrile illness if MSUD suspected.",
            "combined_triad_dld": "DLD deficiency COMBINED TRIAD: lactic acidosis (PDH E3 failure) + alpha-ketoglutaric aciduria (KGDH E3 failure) + alloisoleucine/elevated BCCAs (BCKDH E3 failure). NO other single enzyme deficiency causes all three simultaneously. If all three present → DLD molecular testing mandatory.",
            "maternal_3mcc": "Positive NBS C5-OH in asymptomatic neonate → TEST THE MOTHER first. Maternal 3-MCC deficiency is the MOST COMMON cause of NBS C5-OH positive (transplacental C5-OH transfer). Mother may be asymptomatic lifelong. If mother positive → infant NBS is maternally derived, not infant disease.",
        },
        "treatment_matrix": {
            "leucine_restriction_indications": "MSUD Types Ia/Ib/II/III (BCKDHA/B/DBT/DLD): chronic leucine restriction to plasma target 75-300 µmol/L. PPM1K: leucine restriction ONLY during catabolic crises. MCCC1: restriction only for symptomatic patients during crises. HIBCH: valine restriction (not leucine). BCKDK: CONTRAINDICATED (worsens BCAA depletion).",
            "thiamine_trial": "ALL classic/intermediate MSUD (BCKDHA/B/DBT/DLD): thiamine (vitamin B1) 100-300 mg/day trial for 3 months. BCKDHA most likely to respond (~18%); BCKDHB less (~5-10%); DLD via FAD-dependent E3 mechanism. Document response with plasma leucine and BCAA-keto acid levels.",
            "liver_transplant": "Curative for BCKDHA/BCKDHB/DBT classic MSUD — donor liver provides BCKDH activity, BCCAs normalise, diet liberalised. NOT recommended for BCKDK (no BCKDH defect). NOT fully curative for DLD (PDH+KGDH deficiency persist in non-hepatic tissues). Optimal timing: 6-12 months before neurological damage.",
            "bcaa_supplementation": "BCKDK deficiency ONLY: oral leucine + isoleucine + valine supplementation (100-200 mg/kg/day each). Target plasma leucine 100-200 µmol/L. Monitor autism/seizure response. Use high-protein diet concurrently. Never use BCAA supplements in classic MSUD (raises leucine further).",
            "biotin_use": "Biotin ONLY for: BTD (biotinidase deficiency) and HLCS (holocarboxylase synthetase) deficiency → multiple carboxylase deficiency. Biotin NOT indicated for ISOLATED 3-MCC deficiency (MCCC1/MCCC2). 3-MCC enzyme is biotin-dependent but isolated deficiency is enzyme-structural, not biotin-responsive.",
        },
        "definitions": [
            {
                "term": "ALLOISOLEUCINE",
                "definition": "Stereoisomer of isoleucine formed non-enzymatically from the alpha-keto acid of isoleucine (alpha-keto-beta-methylvalerate) when BCKDH complex is defective. Present in plasma of ALL patients with BCKDH complex defects (Types Ia/Ib/II/III) and pathognomonic for MSUD. Even 1-5 µmol/L is diagnostic. Completely absent in BCKDK deficiency (BCCAs too low), HIBCH, MCCC1.",
            },
            {
                "term": "BCKDH COMPLEX",
                "definition": "Branched-Chain Keto-Acid Dehydrogenase — mitochondrial multienzyme complex (analogous to PDH) catalysing oxidative decarboxylation of branched-chain alpha-keto acids. Three subunits: E1 (α₂β₂ heterotetramer: BCKDHA+BCKDHB), E2 (24-mer: DBT), E3 (shared with PDH and KGDH: DLD). Regulated by BCKDK kinase (inactivates) and PPM1K phosphatase (activates).",
            },
            {
                "term": "BCAA-LOW SYNDROME (BCKDK DEFICIENCY)",
                "definition": "BCKDK kinase normally phosphorylates E1α-Ser293 to INACTIVATE BCKDH. BCKDK LOF → constitutively active BCKDH → BCCAs continuously catabolised → plasma BCCAs very low (leucine often <50 µmol/L, normal 100-200). Presents with autism/intellectual disability/epilepsy. NBS entirely misses (NBS detects HIGH leucine, not LOW). Treat with BCAA supplementation (opposite of MSUD).",
            },
            {
                "term": "METHACRYLYL-CoA TOXICITY (HIBCH DEFICIENCY)",
                "definition": "HIBCH deficiency → 3-hydroxyisobutyryl-CoA accumulates → upstream methacrylyl-CoA (alpha,beta-unsaturated thioester) also accumulates. Methacrylyl-CoA is highly electrophilic, alkylates free cysteine residues and glutathione → cysteine/GSH depletion → mitochondrial oxidative stress → Leigh-like neurodegeneration. BCCAs are NORMAL (no BCKDH involvement). N-acetylcysteine replenishes cysteine pool.",
            },
            {
                "term": "3-MCC DEFICIENCY / MATERNAL NBS TRAP",
                "definition": "3-Methylcrotonyl-CoA carboxylase deficiency (MCCC1 or MCCC2): leucine catabolism defect. Elevated C5-OH (3-hydroxyisovalerylcarnitine) on NBS. CRITICAL: majority of NBS-detected 3-MCC cases are BENIGN. Most common cause of positive C5-OH NBS = MATERNAL 3-MCC (asymptomatic mother with unrecognised 3-MCC transfers C5-OH transplacentally). Always test mother before treating infant.",
            },
            {
                "term": "MENNONITE FOUNDER MUTATION (BCKDHA p.Y393N)",
                "definition": "p.Y393N homozygous BCKDHA mutation: carrier frequency 1:10 in Old Order Mennonite (Lancaster County, PA; Ontario). MSUD prevalence in Mennonite communities 1:358 vs 1:185,000 general population. Universal NBS in Mennonite communities transformed outcomes. Mandatory screening and genetic counselling in Mennonite populations.",
            },
            {
                "term": "DLD DEFICIENCY ASHKENAZI JEWISH FOUNDER",
                "definition": "p.G229C is the Ashkenazi Jewish founder mutation for DLD deficiency (carrier rate ~1:100 in Ashkenazi Jewish). p.D479V is a second common Ashkenazi variant. Combined carrier rate means DLD deficiency risk ~1:40,000 in Ashkenazi Jewish couples. Prenatal DLD testing recommended for at-risk Ashkenazi couples.",
            },
            {
                "term": "CASCADE: BCAA Plasma Interpretation Algorithm",
                "definition": "Step 1: BCCAs elevated + alloisoleucine → BCKDH complex defect (BCKDHA/B/DBT/DLD panel). Step 2: BCCAs elevated + lactic acidosis + alpha-ketoglutaric aciduria → DLD deficiency. Step 3: BCCAs VERY LOW (leucine <75) → BCKDK deficiency (supplement BCCAs). Step 4: BCCAs intermittently elevated only during fever → DBT intermittent or PPM1K. Step 5: BCCAs normal + Leigh-like MRI → HIBCH. Step 6: C5-OH on NBS + normal BCCAs → 3-MCC (check mother first).",
            },
        ],
    }
