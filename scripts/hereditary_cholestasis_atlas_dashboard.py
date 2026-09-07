#!/usr/bin/env python3
"""Hereditary-Cholestasis-Atlas — Complete 8-Gene Atlas
ATP8B1  (FIC1 / CDC36; 1251 aa; 18q21.31; AR;
         PFIC1 — Byler disease / BRIC1;
         Low GGT PATHOGNOMONIC — canalicular function intact, hepatocyte export pump absent;
         extrahepatic: chronic diarrhea, pancreatic insufficiency, hearing loss;
         post-LTx diarrhoea WORSENS — FIC1 expressed in intestine;
         seed SEED_BASE+0) .
ABCB11  (BSEP / SPGP; 1321 aa; 2q31.1; AR;
         PFIC2 — BSEP deficiency — most common severe PFIC;
         Low GGT PATHOGNOMONIC;
         HCC WITHOUT cirrhosis — MANDATORY surveillance from birth;
         Odevixibat (Bylvay) FDA2021 IBAT inhibitor first-line pharmacotherapy;
         E297G missense → residual BSEP → best odevixibat response;
         seed SEED_BASE+1) .
ABCB4   (MDR3; 1279 aa; 7q21.12; AR/AD;
         PFIC3 (AR) / LPAC (AD) — LOW phospholipid bile;
         HIGH GGT PATHOGNOMONIC — contrast to PFIC1/2/4/5/6 (all LOW GGT);
         UDCA FIRST-LINE — phospholipid-depleted bile stabilised by UDCA;
         LPAC: intrahepatic stones + young age + recurrence after cholecystectomy;
         seed SEED_BASE+2) .
TJP2    (ZO-2; 1221 aa; 9q21.11; AR;
         PFIC4 — tight junction protein 2 deficiency;
         Low GGT;
         HCC WITHOUT cirrhosis (as in PFIC2) — surveillance MANDATORY;
         neurological features in subset (extra-hepatic TJP2 expression);
         seed SEED_BASE+3) .
NR1H4   (FXR; 472 aa; 12q23.1; AR;
         PFIC5 — FXR (farnesoid X receptor) deficiency;
         Low GGT; most severe neonatal presentation; combined BSEP + MDR3 defect;
         FXR normally upregulates ABCB11 and ABCB4;
         FXR agonists (obeticholic acid) CANNOT work — receptor itself absent;
         seed SEED_BASE+4) .
MYO5B   (Myosin Vb; 1852 aa; 18q21.1; AR;
         PFIC6 / Microvillus Inclusion Disease (MVID);
         Low GGT; apical membrane recycling defect;
         MVID: life-threatening neonatal secretory diarrhoea (up to 300 mL/kg/day);
         TPN-dependent; combined intestinal + liver transplant for MVID+PFIC6;
         seed SEED_BASE+5) .
JAG1    (JAGGED1; 1218 aa; 20p12.2; AD;
         Alagille Syndrome Type 1 (ALGS1); NOTCH ligand;
         bile duct paucity on biopsy PATHOGNOMONIC (<0.5 ducts/portal tract);
         butterfly vertebrae on X-ray PATHOGNOMONIC (posterior arch fusion defect);
         peripheral pulmonary stenosis most common cardiac manifestation;
         94% penetrance + VARIABLE expressivity — asymptomatic parent COMMON;
         seed SEED_BASE+6) .
NOTCH2  (NOTCH2; 2471 aa; 1p12; AD;
         Alagille Syndrome Type 2 (ALGS2); NOTCH2 receptor for JAG1;
         renal anomalies MORE prominent than ALGS1 (dysplasia/RTA);
         hepatic disease GENERALLY milder than ALGS1;
         Hajdu-Cheney syndrome (GOF): distinct — acroosteolysis, osteoporosis;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1894–1901)
"""

import random

SEED_BASE = 1894

CHOLESTASIS_GENES = [
    # -- ATP8B1 -- PFIC1 / Byler disease / BRIC1 ----------------------------------
    {
        "gene": "ATP8B1",
        "protein": (
            "ATP8B1 -- 18q21.31 AR -- FIC1-1251aa -- "
            "PFIC1-Byler-Disease-BRIC1-Progressive-Familial-Intrahepatic-Cholestasis-Type1 -- "
            "Low-GGT-PATHOGNOMONIC-Canalicular-Secretion-Intact-Phospholipid-Flippase-Defect -- "
            "Extrahepatic-Diarrhea-Pancreatitis-Hearing-Loss-Short-Stature -- "
            "Post-LTx-Diarrhea-WORSENS-FIC1-Expressed-Intestine-NOT-Just-Liver"
        ),
        "alias": (
            "ATP8B1 (ATPase phospholipid transporting 8B1 / FIC1 / CDC36); OMIM gene 602397; "
            "PFIC1 (Byler disease) OMIM 211600 / BRIC1 (benign recurrent intrahepatic cholestasis type 1) OMIM 243300. "
            "18q21.31; 1251 aa; ~140 kDa; type IV P-type ATPase; autosomal recessive. "
            "FUNCTION: ATP8B1 encodes FIC1 (familial intrahepatic cholestasis 1), a phospholipid flippase expressed at the canalicular membrane. "
            "FIC1 maintains phospholipid asymmetry in the canalicular membrane by flipping "
            "phosphatidylserine and phosphatidylethanolamine from the outer to inner membrane leaflet. "
            "Loss of FIC1 → membrane phospholipid asymmetry disrupted → membrane fragility → "
            "altered bile acid transport signalling (FXR signalling impaired via reduced phospholipid scaffold). "
            "The bile salt export pump (ABCB11/BSEP) is present and functional — "
            "hence GGT is NORMAL/LOW (canalicular secretion of GGT requires gamma-glutamyl transferase solubilisation by bile acids/phospholipids; "
            "when phospholipids are absent GGT is not solubilised into bile → serum GGT low). "
            "CLINICAL PRESENTATION: "
            "Neonatal/infantile onset: jaundice, pruritus, poor growth, fat-soluble vitamin deficiency; "
            "Severe pruritus — often more severe than in PFIC2 despite lower bile acid levels; "
            "Extrahepatic features: chronic watery diarrhoea (FIC1 expressed in intestinal epithelium), "
            "pancreatic insufficiency, sensorineural hearing loss, short stature; "
            "Episodic form: BRIC1 — self-limiting episodes of cholestasis lasting weeks to months; "
            "BRIC1 and PFIC1 are allelic — severity correlates with residual FIC1 activity. "
            "CRITICAL WARNING — POST-LIVER-TRANSPLANT: "
            "Liver transplant cures hepatic disease BUT intestinal FIC1 remains absent; "
            "Post-transplant diarrhoea WORSENS (not improves) — FIC1 intestinal function not restored; "
            "Hepatic steatosis develops post-transplant (NAFLD phenotype); "
            "Counsel family: LTx corrects liver disease but extrahepatic features persist/worsen. "
            "DIAGNOSIS: "
            "Liver biopsy: intracanalicular bile deposits; electron microscopy — Byler bile (granular, amorphous); "
            "Serum GGT: LOW/NORMAL PATHOGNOMONIC in context of cholestasis; "
            "Bile acid levels: ELEVATED in serum; "
            "ATP8B1 gene sequencing: G308V (Amish founder), I661T; "
            "Extrahepatic features confirm. "
            "TREATMENT: "
            "Pharmacological: rifampicin (enzyme inducer, anti-pruritic), cholestyramine, UDCA (limited efficacy); "
            "Surgical: partial biliary diversion (PEBD) — can arrest disease progression if < stage 3 fibrosis; "
            "Odevixibat/Maralixibat: IBAT inhibitors — moderate efficacy in PFIC1 (less than PFIC2); "
            "Liver transplant: curative for liver disease BUT extrahepatic disease persists/worsens (see above). "
            "KEY CLINICAL FACTS: "
            "LOW GGT IN CHOLESTASIS = PFIC1/2/4/5/6 OR ALGS — must exclude canalicular pump defects; "
            "HIGH GGT IN CHOLESTASIS = PFIC3 (MDR3 defect) until proven otherwise; "
            "POST-LTx DIARRHEA WORSE = ATP8B1 pathognomonic clue — no other PFIC has this; "
            "EXTRAHEPATIC = ATP8B1 — diarrhoea, pancreatitis, hearing loss absent in PFIC2/3/4/5."
        ),
        "age_of_onset": "Neonatal/early infantile; BRIC1 episodic from childhood/adolescence",
        "inheritance": "AR",
        "locus": "18q21.31",
        "protein_size": "1251 aa",
        "key_biomarker": "Low/normal GGT PATHOGNOMONIC in cholestasis; elevated serum bile acids; Byler bile on EM",
        "pathognomonic": "PFIC1: low GGT cholestasis + extrahepatic features (diarrhoea/pancreatitis) + post-LTx diarrhoea WORSENS",
        "treatment": "Rifampicin; PEBD; odevixibat (limited PFIC1 efficacy); LTx (cures liver; extrahepatic persists)",
        "critical_flags": [
            "LOW-GGT-PATHOGNOMONIC — GGT low/normal in context of jaundice/pruritus = canalicular pump defect (PFIC1/2/4/5/6)",
            "POST-LTx-DIARRHEA-WORSENS — FIC1 expressed in intestine; LTx removes hepatic disease but diarrhoea persists/worsens; counsel pre-transplant",
            "EXTRAHEPATIC-FEATURES-ATP8B1-ONLY — diarrhoea + pancreatitis + SNHL absent in PFIC2/3/4/5; presence of these = ATP8B1",
            "PEBD-BEFORE-LTx — partial external biliary diversion can arrest progression if < stage 3 fibrosis; try before LTx",
            "BRIC1-vs-PFIC1-ALLELIC — same gene, same mutations possible; severity = residual FIC1 function; BRIC1 → PFIC1 transition possible",
            "HEPATIC-STEATOSIS-POST-LTx — NAFLD phenotype after LTx; monitor liver function / USS annually post-transplant",
            "VITAMIN-DEFICIENCY — fat-soluble vitamins A/D/E/K; monitor levels 6-monthly; supplement",
        ],
    },
    # -- ABCB11 -- PFIC2 / BSEP deficiency ----------------------------------------
    {
        "gene": "ABCB11",
        "protein": (
            "ABCB11 -- 2q31.1 AR -- BSEP-1321aa -- "
            "PFIC2-BSEP-Deficiency-Most-Common-Severe-PFIC-Progressive-Familial-Intrahepatic-Cholestasis-Type2 -- "
            "Low-GGT-PATHOGNOMONIC-Bile-Salt-Export-Pump-Absent -- "
            "HCC-WITHOUT-Cirrhosis-MANDATORY-Surveillance-from-Birth -- "
            "Odevixibat-Bylvay-FDA2021-IBAT-Inhibitor-First-Line-E297G-Residual-BSEP-Best-Response"
        ),
        "alias": (
            "ABCB11 (ATP-binding cassette sub-family B member 11 / BSEP / SPGP); OMIM gene 603201; "
            "PFIC2 OMIM 601847 / BRIC2 OMIM 605479. "
            "2q31.1; 1321 aa; ~140 kDa; ATP-dependent canalicular bile salt export pump; autosomal recessive. "
            "FUNCTION: ABCB11 encodes BSEP (bile salt export pump), the primary ATP-dependent transporter "
            "for conjugated bile salts (taurochenodeoxycholate, taurocholate) across the hepatocyte canalicular membrane. "
            "BSEP drives the majority of bile salt-dependent bile flow. "
            "Loss of BSEP → bile salts accumulate inside hepatocytes → hepatocellular injury → progressive liver disease. "
            "Unlike ATP8B1/PFIC1: PFIC2 is purely hepatic — no extrahepatic features. "
            "GGT: LOW/NORMAL — bile salt transport defect, not phospholipid/gamma-glutamyl defect. "
            "MOST IMPORTANT COMPLICATION: "
            "HCC (hepatocellular carcinoma) develops WITHOUT cirrhosis — "
            "ABCB11/BSEP null mutations → hepatocellular accumulation of hydrophobic bile salts → direct hepatocyte DNA damage; "
            "HCC can occur as early as 1 year of age; "
            "Surveillance: hepatic MRI + AFP every 6 months from diagnosis (from birth in null mutations). "
            "GENOTYPE-PHENOTYPE: "
            "Null (nonsense/frameshift) mutations → complete PFIC2; early HCC risk; poor pharmacological response; "
            "E297G missense → residual BSEP at canalicular membrane → BRIC2 phenotype or mild PFIC2; "
            "E297G (and other missense with residual BSEP) → BEST RESPONSE to odevixibat. "
            "CLINICAL PRESENTATION: "
            "Neonatal/infantile jaundice, severe pruritus, poor growth, fat-soluble vitamin deficiency; "
            "NO diarrhoea, NO pancreatitis (purely hepatic unlike ATP8B1); "
            "Liver biopsy: giant cell hepatitis (infantile); later: biliary cirrhosis; "
            "Progressive if untreated: biliary cirrhosis → LTx need by 2-10 years in null mutations. "
            "TREATMENT: "
            "Odevixibat (Bylvay, IPSEN) — FDA/EMA approved 2021/2021 — IBAT (ileal bile acid transporter) inhibitor; "
            "reduces entero-hepatic circulation of bile salts → reduces hepatocyte bile salt burden; "
            "best efficacy in residual BSEP (E297G, missense); "
            "Maralixibat (Livmarli, Mirum) — also IBAT inhibitor, approved for PFIC; "
            "Partial external biliary diversion (PEBD): effective if < stage 3; "
            "Liver transplant: CURATIVE — corrects BSEP deficiency; NO recurrence unless de novo anti-BSEP antibodies; "
            "Anti-BSEP antibodies: develop post-LTx in null mutation patients (alloimmune reaction to neo-antigen BSEP); "
            "anti-BSEP antibody recurrence post-LTx: treat with plasmapheresis + rituximab. "
            "KEY CLINICAL FACTS: "
            "HCC SURVEILLANCE FROM BIRTH — MRI + AFP every 6 months; HCC without cirrhosis is the hallmark; "
            "ODEVIXIBAT REQUIRES RESIDUAL BSEP — null mutations (frameshift/nonsense) respond poorly; genotype before prescribing; "
            "NO EXTRAHEPATIC FEATURES — purely hepatic; presence of diarrhoea/pancreatitis = think ATP8B1 (PFIC1); "
            "ANTI-BSEP ANTIBODIES POST-LTx — null mutation patients; can cause recurrent disease in transplanted liver."
        ),
        "age_of_onset": "Neonatal/early infantile; HCC may occur from age 1 year",
        "inheritance": "AR",
        "locus": "2q31.1",
        "protein_size": "1321 aa",
        "key_biomarker": "Low/normal GGT; elevated serum bile acids; HCC on MRI/AFP; BSEP absent on immunostaining",
        "pathognomonic": "Low GGT cholestasis + HCC WITHOUT cirrhosis; pure hepatic (no extrahepatic features); null = early severe + HCC",
        "treatment": "Odevixibat (IBAT inhibitor; best in E297G/missense); PEBD; LTx curative; HCC surveillance MRI+AFP 6-monthly",
        "critical_flags": [
            "HCC-WITHOUT-CIRRHOSIS-MANDATORY-SURVEILLANCE — MRI + AFP every 6 months from diagnosis; HCC age 1+ year possible",
            "ODEVIXIBAT-REQUIRES-RESIDUAL-BSEP — null mutations (nonsense/frameshift) respond poorly; E297G/missense respond best; genotype FIRST",
            "NO-EXTRAHEPATIC-FEATURES — purely hepatic; diarrhoea/pancreatitis ABSENT; their presence = think ATP8B1 not ABCB11",
            "ANTI-BSEP-ANTIBODIES-POST-LTx — null mutation patients: neo-antigen BSEP in donor liver → antibody-mediated rejection; treat rituximab + plasmapheresis",
            "LOW-GGT-PATHOGNOMONIC — in context of cholestasis; HIGH GGT = PFIC3 (ABCB4) until proven otherwise",
            "LIVER-TRANSPLANT-CURATIVE — no extrahepatic disease; recurrence only via anti-BSEP antibodies (manageable)",
            "BRIC2-ALLELIC — E297G → episodic mild cholestasis; may progress; monitor annually",
        ],
    },
    # -- ABCB4 -- PFIC3 / LPAC -------------------------------------------------------
    {
        "gene": "ABCB4",
        "protein": (
            "ABCB4 -- 7q21.12 AR/AD -- MDR3-1279aa -- "
            "PFIC3-MDR3-Deficiency-Progressive-Familial-Intrahepatic-Cholestasis-Type3 -- "
            "HIGH-GGT-PATHOGNOMONIC-Contrast-to-PFIC1-2-4-5-6-All-Low-GGT -- "
            "UDCA-FIRST-LINE-Phospholipid-Depleted-Bile-Bile-Duct-Epithelium-Protection -- "
            "LPAC-AD-Heterozygous-Intrahepatic-Lithiasis-Young-Age-Recurrence-After-Cholecystectomy"
        ),
        "alias": (
            "ABCB4 (ATP-binding cassette sub-family B member 4 / MDR3 / PFIC3); OMIM gene 171060; "
            "PFIC3 OMIM 602347 / LPAC (low-phospholipid-associated cholelithiasis). "
            "7q21.12; 1279 aa; ~140 kDa; canalicular phospholipid translocase (floppase); autosomal recessive (PFIC3) or dominant (LPAC). "
            "FUNCTION: ABCB4 encodes MDR3 (multidrug resistance protein 3), a floppase on the canalicular membrane "
            "that translocates phosphatidylcholine (PC) from the inner to the outer membrane leaflet into bile. "
            "PC is essential to form mixed micelles with bile salts — without PC, bile salts are not micellarised "
            "and remain free (monomeric), highly toxic to biliary epithelium. "
            "Loss of MDR3 → phosphatidylcholine absent from bile → toxic free bile salts → "
            "bile duct epithelial injury → cholangiopathy → portal inflammation → biliary cirrhosis. "
            "GGT: HIGH PATHOGNOMONIC — GGT is solubilised from biliary epithelium by bile salts; "
            "biliary epithelial damage releases GGT; HIGH GGT distinguishes PFIC3 from ALL other PFIC types (PFIC1/2/4/5/6 all LOW GGT). "
            "GENOTYPE: "
            "Biallelic loss-of-function → PFIC3 (progressive, childhood-onset biliary cirrhosis); "
            "Heterozygous (single pathogenic allele) → LPAC (low phospholipid associated cholelithiasis): "
            "intrahepatic biliary stones in young adults (< 40 years), recurrent biliary pain, "
            "recurrence after cholecystectomy (intrahepatic stones NOT removed by cholecystectomy). "
            "CLINICAL PRESENTATION (PFIC3): "
            "Later onset than PFIC1/2 — often childhood/adolescence (rarely neonatal); "
            "Jaundice, pruritus (usually milder than PFIC1/2), high GGT; "
            "Progressive biliary cirrhosis; "
            "Liver biopsy: bile ductular proliferation, portal fibrosis, biliary cirrhosis. "
            "CLINICAL PRESENTATION (LPAC): "
            "Young adult (< 40 years) with intrahepatic biliary stones; "
            "Pain after cholecystectomy (stones remain in intrahepatic bile ducts); "
            "Cholestasis of pregnancy; recurrent choledocholithiasis. "
            "TREATMENT: "
            "UDCA (ursodeoxycholic acid) FIRST-LINE for PFIC3 and LPAC: "
            "provides exogenous bile phospholipid-like protective effect; reduces bile salt toxicity; "
            "most effective of all PFIC types — can prevent/slow progression; "
            "Dose: 15-20 mg/kg/day; lifelong in PFIC3; during pregnancy in LPAC; "
            "PEBD: less effective than in PFIC1/2 (cholangiopathy already ongoing); "
            "Odevixibat/Maralixibat: limited evidence in PFIC3; "
            "Liver transplant: curative; no recurrence (MDR3 expressed in donor liver). "
            "KEY CLINICAL FACTS: "
            "HIGH GGT = PFIC3 PATHOGNOMONIC in context of familial cholestasis — if GGT is HIGH, think ABCB4 first; "
            "UDCA IS MOST EFFECTIVE IN PFIC3 — of all PFIC types; do NOT skip UDCA trial; "
            "LPAC: if young adult with intrahepatic stones — sequence ABCB4 MANDATORY; lifelong UDCA prevents recurrence; "
            "CHOLECYSTECTOMY IN LPAC DOES NOT CURE — intrahepatic stones remain; UDCA prevents new stone formation."
        ),
        "age_of_onset": "PFIC3: childhood/adolescence (later than PFIC1/2); LPAC: young adults (< 40 years)",
        "inheritance": "AR (PFIC3) / AD-heterozygous (LPAC)",
        "locus": "7q21.12",
        "protein_size": "1279 aa",
        "key_biomarker": "HIGH GGT PATHOGNOMONIC (distinguishes from all other PFIC); intrahepatic stones on USS (LPAC)",
        "pathognomonic": "HIGH GGT cholestasis + intrahepatic stones in young adult (LPAC) OR progressive biliary cirrhosis (PFIC3) — UDCA responsive",
        "treatment": "UDCA 15-20 mg/kg/day FIRST-LINE (most effective of all PFIC types); LTx curative; PEBD less effective",
        "critical_flags": [
            "HIGH-GGT-PATHOGNOMONIC — HIGH GGT in familial cholestasis = PFIC3 (ABCB4) until proven otherwise; ALL other PFIC types have LOW GGT",
            "UDCA-MOST-EFFECTIVE-OF-ALL-PFIC — responsive to UDCA unlike PFIC1/2; must NOT omit UDCA trial; lifelong in PFIC3",
            "LPAC-CHOLECYSTECTOMY-NOT-CURATIVE — intrahepatic stones remain; recurrence after cholecystectomy = ABCB4 heterozygote; sequence MANDATORY",
            "LPAC-PREGNANCY — cholestasis of pregnancy may be first manifestation; UDCA safe in pregnancy; stop at delivery; restart postpartum",
            "PFIC3-LATER-ONSET — childhood/adolescence; not neonatal; milder pruritus than PFIC1/2; biliary cirrhosis predominates",
            "BILIARY-EPITHELIUM-INJURY — mechanism: PC-free bile → toxic free bile salts → cholangiopathy; explains HIGH GGT (biliary origin) vs LOW GGT (hepatocyte origin) distinction",
            "UDCA-LIFELONG-PREVENTION — prevents stone recurrence in LPAC; prevents progression in PFIC3 mild cases",
        ],
    },
    # -- TJP2 -- PFIC4 ---------------------------------------------------------------
    {
        "gene": "TJP2",
        "protein": (
            "TJP2 -- 9q21.11 AR -- ZO2-1221aa -- "
            "PFIC4-Tight-Junction-Protein-2-Deficiency-Progressive-Familial-Intrahepatic-Cholestasis-Type4 -- "
            "Low-GGT-Canalicular-Tight-Junction-Integrity-Defect -- "
            "HCC-WITHOUT-Cirrhosis-Surveillance-MANDATORY-Similar-to-PFIC2 -- "
            "Neurological-Features-Subset-Extrahepatic-TJP2-Expression"
        ),
        "alias": (
            "TJP2 (tight junction protein 2 / ZO-2 / ZO2); OMIM gene 607709; "
            "PFIC4 OMIM 615878. "
            "9q21.11; 1221 aa; ~160 kDa; tight junction scaffold protein; autosomal recessive. "
            "FUNCTION: TJP2 encodes ZO-2 (zona occludens 2), a tight junction scaffolding protein. "
            "ZO-2 is expressed at the apical junctional complex of hepatocytes (canalicular tight junctions) and at other epithelia. "
            "In the liver: canalicular tight junctions maintain paracellular impermeability — "
            "preventing bile components from leaking back from the canalicular lumen into sinusoidal space. "
            "Loss of TJP2 → canalicular tight junctions disrupted → bile acid paracellular leak → "
            "hepatocyte exposure to concentrated bile → hepatocellular injury → progressive liver disease. "
            "GGT: LOW/NORMAL — canalicular secretion pumps (BSEP, MDR3) intact; tight junction defect; "
            "hence GGT not released into bile abnormally (biliary epithelium intact). "
            "HCC WITHOUT CIRRHOSIS: "
            "TJP2-null mutations → HCC reported WITHOUT established cirrhosis; "
            "mechanism: bile acid accumulation in hepatocytes → direct mutagenic/carcinogenic effect; "
            "HCC surveillance: MRI + AFP every 6 months from diagnosis — MANDATORY (as in PFIC2). "
            "EXTRAHEPATIC FEATURES: "
            "TJP2 is also expressed in the nervous system and other epithelia; "
            "Neurological features in a subset of patients: intellectual disability, hearing loss (variable); "
            "Alopecia reported in some patients (follicular epithelium expression). "
            "CLINICAL PRESENTATION: "
            "Neonatal/infantile cholestasis, low GGT; variable severity; "
            "Severe variants: rapidly progressive to biliary cirrhosis; "
            "Mild variants: episodic cholestasis (BRIC-like); "
            "Liver biopsy: canalicular changes, giant cell hepatitis (infantile), progressive fibrosis. "
            "TREATMENT: "
            "Rifampicin (anti-pruritic); UDCA (limited); "
            "Odevixibat/Maralixibat: data limited for PFIC4; "
            "PEBD: can help if < stage 3; "
            "Liver transplant: curative for liver disease; neurological features may persist post-LTx; "
            "HCC: standard HCC protocols (resection/transplant/ablation) — "
            "LTx for HCC also addresses underlying PFIC4. "
            "KEY CLINICAL FACTS: "
            "TJP2 IS DISTINCT FROM PFIC1/2 — tight junction defect, not phospholipid flippase/export pump defect; "
            "HCC SURVEILLANCE MANDATORY — as in PFIC2 (ABCB11); HCC without cirrhosis is a shared feature; "
            "LOW GGT DISCRIMINATES FROM PFIC3 (ABCB4) — ABCB4 always HIGH GGT; TJP2 always LOW GGT; "
            "NEUROLOGICAL FEATURES HINT AT TJP2 — absent in PFIC1/2/3/5/6 + Alagille; presence suggests TJP2 or MVID."
        ),
        "age_of_onset": "Neonatal/early infantile (cholestasis); HCC from childhood/young adulthood",
        "inheritance": "AR",
        "locus": "9q21.11",
        "protein_size": "1221 aa",
        "key_biomarker": "Low/normal GGT; HCC on MRI (without cirrhosis); neurological features in subset",
        "pathognomonic": "Low GGT cholestasis + HCC WITHOUT cirrhosis ± neurological features; distinguished from PFIC2 by TJP2 gene sequencing",
        "treatment": "PEBD; LTx curative (liver); neurological features may persist; HCC surveillance MRI+AFP 6-monthly",
        "critical_flags": [
            "HCC-WITHOUT-CIRRHOSIS-MANDATORY-SURVEILLANCE — MRI + AFP every 6 months; shared feature with PFIC2 (ABCB11); different mechanism",
            "LOW-GGT-DISTINGUISHES-FROM-PFIC3 — TJP2 (PFIC4) always LOW GGT; ABCB4 (PFIC3) always HIGH GGT; one lab value separates",
            "NEUROLOGICAL-FEATURES-SUBSET — intellectual disability/hearing loss if TJP2; absent in PFIC1/2/3/5/6; tip-off for TJP2 diagnosis",
            "LTx-DOES-NOT-CURE-NEUROLOGICAL — liver transplant corrects liver; neurological features (extrahepatic TJP2) persist post-LTx; counsel family",
            "TIGHT-JUNCTION-MECHANISM — not pump defect; paracellular bile leak; explains why IBAT inhibitors have limited evidence (different pathway)",
            "ALOPECIA-REPORTED — follicular epithelial TJP2 expression; variable penetrance; inspect scalp/hair clinically",
            "PEBD-EARLY — attempt partial external biliary diversion before LTx if < stage 3 fibrosis",
        ],
    },
    # -- NR1H4 -- PFIC5 / FXR deficiency -------------------------------------------
    {
        "gene": "NR1H4",
        "protein": (
            "NR1H4 -- 12q23.1 AR -- FXR-472aa -- "
            "PFIC5-FXR-Farnesoid-X-Receptor-Deficiency-Most-Severe-Neonatal-Cholestasis -- "
            "Low-GGT-Combined-BSEP-Plus-MDR3-Deficiency-FXR-Regulates-Both -- "
            "FXR-Agonists-Obeticholic-Acid-CANNOT-Work-Receptor-Absent -- "
            "Alpha-Fetoprotein-MARKEDLY-Elevated-Neonatal-HCC-Risk-Highest-All-PFIC"
        ),
        "alias": (
            "NR1H4 (nuclear receptor subfamily 1 group H member 4 / FXR / BAR); OMIM gene 603826; "
            "PFIC5 OMIM 617049. "
            "12q23.1; 472 aa; ~55 kDa; nuclear hormone receptor / transcription factor; autosomal recessive. "
            "FUNCTION: NR1H4 encodes FXR (farnesoid X receptor), the primary bile acid nuclear sensor. "
            "FXR is activated by primary bile acids (chenodeoxycholic acid > cholic acid) in the ileum and liver. "
            "FXR transcriptional targets include: "
            "(1) ABCB11 (BSEP) — bile salt export pump; FXR UPREGULATES BSEP; "
            "(2) ABCB4 (MDR3) — phospholipid translocase; FXR UPREGULATES MDR3; "
            "(3) SLC51A (OSTα) — bile acid export from hepatocyte; "
            "(4) SHP (NR0B2) — represses CYP7A1 (bile acid synthesis). "
            "Loss of FXR → "
            "COMBINED DEFICIENCY of BSEP + MDR3 (and other targets) → "
            "most severe of all PFIC phenotypes — "
            "low GGT cholestasis (BSEP effect) + bile duct injury (MDR3 effect) simultaneously. "
            "HCC RISK: "
            "Highest HCC risk of all PFIC types — FXR also regulates hepatocyte proliferation/apoptosis; "
            "Alpha-fetoprotein MARKEDLY elevated even in early infancy (not just liver injury — FXR loss disrupts AFP suppression); "
            "AFP elevation in PFIC5 is DISPROPORTIONATE to liver disease severity — use AFP as PFIC5 marker. "
            "FXR AGONISTS: "
            "Obeticholic acid (Ocaliva) and other FXR agonists target FXR; "
            "In PFIC5: FXR RECEPTOR IS ABSENT → FXR agonists have NO EFFECT; "
            "Prescribing obeticholic acid in PFIC5 = pharmacologically impossible to work; "
            "This is a critical prescribing error to avoid. "
            "CLINICAL PRESENTATION: "
            "Most severe neonatal presentation — severe jaundice from day 1 of life; "
            "Rapidly progressive liver failure; AFP disproportionately elevated; "
            "Low GGT (combined BSEP defect component); "
            "No extrahepatic features (purely hepatic nuclear receptor defect). "
            "TREATMENT: "
            "Liver transplant EARLIEST possible — fastest-progressing PFIC; "
            "FXR agonists (obeticholic acid): CONTRAINDICATED (receptor absent — no target); "
            "IBAT inhibitors (odevixibat/maralixibat): limited evidence; "
            "Rifampicin: anti-pruritic only; "
            "PEBD: limited data; severe disease may preclude. "
            "KEY CLINICAL FACTS: "
            "FXR AGONISTS DO NOT WORK IN PFIC5 — this is a critical treatment error; receptor is absent; "
            "AFP DISPROPORTIONATELY HIGH — PFIC5 marker; AFP >> what liver disease severity would predict; "
            "MOST SEVERE PFIC — combined BSEP+MDR3 deficiency; earliest LTx listing mandatory; "
            "LOW GGT + SEVERE NEONATAL DISEASE + HIGH AFP = PFIC5 until proven otherwise."
        ),
        "age_of_onset": "Neonatal — most severe; day 1 of life presentation",
        "inheritance": "AR",
        "locus": "12q23.1",
        "protein_size": "472 aa",
        "key_biomarker": "Low/normal GGT; markedly elevated AFP (disproportionate); severe neonatal cholestasis from day 1",
        "pathognomonic": "Severe neonatal cholestasis + low GGT + disproportionately HIGH AFP; FXR agonists have no target (receptor absent)",
        "treatment": "URGENT LTx (fastest-progressing); FXR agonists CONTRAINDICATED; IBAT inhibitors limited evidence",
        "critical_flags": [
            "FXR-AGONISTS-CONTRAINDICATED — obeticholic acid and all FXR agonists CANNOT work; receptor absent; prescribing is pharmacologically impossible",
            "AFP-DISPROPORTIONATELY-HIGH — AFP elevation exceeds what liver disease predicts; marker of PFIC5 (FXR regulates AFP expression)",
            "MOST-SEVERE-PFIC-EARLIEST-LTx — combined BSEP+MDR3 deficiency; list for transplant at diagnosis; fastest-progressing",
            "LOW-GGT-DESPITE-SEVERITY — GGT low even in severe disease; combined BSEP/MDR3 defect without biliary epithelial GGT release",
            "NO-FXR-TARGET-FOR-ANY-AGONIST — NR1H4 null = no FXR protein; all FXR-targeting drugs are futile; document in record",
            "NEONATAL-DAY-1-PRESENTATION — unlike PFIC3 (childhood); unlike PFIC1/2 (early infantile); PFIC5 = MOST SEVERE + EARLIEST",
            "HCC-RISK-HIGHEST-ALL-PFIC — FXR regulates hepatocyte proliferation; surveillance mandatory; AFP already elevated as disease marker",
        ],
    },
    # -- MYO5B -- PFIC6 / MVID -------------------------------------------------------
    {
        "gene": "MYO5B",
        "protein": (
            "MYO5B -- 18q21.1 AR -- MyosinVb-1852aa -- "
            "PFIC6-Microvillus-Inclusion-Disease-MVID-Apical-Membrane-Recycling-Defect -- "
            "Low-GGT-Myosin-Vb-Rab8a-Rab11a-Dependent-Canalicular-and-Intestinal-Apical-Trafficking -- "
            "MVID-Neonatal-Secretory-Diarrhea-Life-Threatening-300mL-per-kg-per-day-TPN-Dependent -- "
            "Combined-Intestinal-Plus-Liver-Transplant-for-MVID-Plus-PFIC6"
        ),
        "alias": (
            "MYO5B (myosin Vb); OMIM gene 606540; "
            "PFIC6 (cholestasis) / MVID (microvillus inclusion disease) OMIM 251850. "
            "18q21.1; 1852 aa; ~215 kDa; unconventional myosin motor protein; autosomal recessive. "
            "FUNCTION: MYO5B encodes myosin Vb, an actin-based motor involved in Rab8a/Rab11a-dependent "
            "recycling endosome trafficking to the apical membrane in polarised epithelial cells. "
            "In intestinal enterocytes: MYO5B required for correct apical membrane recycling; "
            "Loss of MYO5B → "
            "microvilli (brush border) fail to form correctly at enterocyte surface; "
            "microvilli-containing membrane INTERNALISED into the cell as inclusion bodies; "
            "MICROVILLUS INCLUSION DISEASE (MVID) — "
            "PAS-positive secretory granules containing brush border enzymes seen intracellularly; "
            "intestinal biopsy: microvillus inclusions on EM PATHOGNOMONIC; "
            "Defective absorption → massive secretory diarrhoea up to 200-300 mL/kg/day — "
            "life-threatening watery diarrhoea within hours of birth. "
            "In hepatocytes: MYO5B required for canalicular membrane recycling; "
            "Loss → canalicular transport defect → cholestasis (PFIC6). "
            "GGT: LOW/NORMAL (canalicular pump intact at gene level; trafficking defect only). "
            "CLINICAL SEVERITY SPECTRUM: "
            "Severe MVID (null mutations): neonatal diarrhoea within hours of birth; "
            "Adult/attenuated MVID (missense): onset later; less severe diarrhoea; "
            "PFIC6 alone: cholestasis without severe intestinal disease (rare MYO5B variants). "
            "DIAGNOSIS: "
            "Intestinal biopsy: EM — microvillus inclusions PATHOGNOMONIC; "
            "PAS stain + alkaline phosphatase IHC: intracellular (subapical) rather than apical → PATHOGNOMONIC; "
            "MYO5B gene sequencing: confirms; "
            "Cholestasis + low GGT confirms PFIC6 component. "
            "TREATMENT: "
            "MVID: Total parenteral nutrition (TPN) — ALWAYS required; "
            "NO oral feeds possible without severe diarrhoea; "
            "Gut rehabilitation trials: marginal; "
            "Intestinal transplant ± liver transplant: only curative option for severe MVID + PFIC6; "
            "Combined intestinal-liver transplant technically complex; high mortality; "
            "Isolated liver transplant for PFIC6 alone (without MVID): curative for liver disease. "
            "KEY CLINICAL FACTS: "
            "MVID = LIFE-THREATENING NEONATAL DIARRHOEA — 200-300 mL/kg/day secretory; immediate TPN; "
            "INTESTINAL BIOPSY MANDATORY — EM microvillus inclusions PATHOGNOMONIC; PAS/ALP IHC screening; "
            "COMBINED LIVER+INTESTINAL TRANSPLANT — for MVID+PFIC6; isolated intestinal Tx alone insufficient if PFIC6 present; "
            "MYO5B-PFIC6-LOW-GGT — confirms canalicular involvement; without intestinal features think TJP2 or other PFIC."
        ),
        "age_of_onset": "Neonatal (MVID: hours-days after birth); PFIC6 alone: early infantile",
        "inheritance": "AR",
        "locus": "18q21.1",
        "protein_size": "1852 aa",
        "key_biomarker": "Intestinal biopsy EM — microvillus inclusions PATHOGNOMONIC; low/normal GGT; massive secretory diarrhoea",
        "pathognomonic": "Neonatal secretory diarrhoea (200-300 mL/kg/day) + low GGT cholestasis + microvillus inclusions on EM",
        "treatment": "TPN (always); combined intestinal+liver Tx for MVID+PFIC6; isolated LTx for PFIC6 alone",
        "critical_flags": [
            "MVID-LIFE-THREATENING-DIARRHEA — 200-300 mL/kg/day secretory from hours of birth; immediate TPN; no oral feeds possible",
            "INTESTINAL-BIOPSY-EM-MANDATORY — microvillus inclusions PATHOGNOMONIC; PAS/ALP IHC intracellular = screening; EM = confirmation",
            "COMBINED-LIVER-INTESTINE-Tx — MVID + PFIC6 requires both organs; isolated intestinal Tx leaves cholestasis; isolated LTx leaves diarrhoea",
            "LOW-GGT-IN-CONTEXT-OF-CHOLESTASIS — MYO5B causes trafficking not pump defect; GGT low as in PFIC1/2/4/5",
            "TPN-FOREVER-WITHOUT-Tx — gut rehabilitation marginal; committed lifeline until transplant",
            "PFIC6-WITHOUT-MVID — rare MYO5B variants cause isolated cholestasis; isolated LTx curative; EM still recommended to exclude MVID",
            "ADULT-ATTENUATED-MVID — missense MYO5B; later onset; lower volume diarrhoea; may tolerate limited enteral feeding",
        ],
    },
    # -- JAG1 -- Alagille Syndrome Type 1 -------------------------------------------
    {
        "gene": "JAG1",
        "protein": (
            "JAG1 -- 20p12.2 AD -- JAGGED1-1218aa -- "
            "Alagille-Syndrome-Type1-ALGS1-NOTCH-Signalling-Ligand -- "
            "Bile-Duct-Paucity-Liver-Biopsy-PATHOGNOMONIC-Less-Than-0.5-Ducts-Per-Portal-Tract -- "
            "Butterfly-Vertebrae-X-Ray-PATHOGNOMONIC-Posterior-Arch-Fusion-Defect -- "
            "Peripheral-Pulmonary-Stenosis-Most-Common-Cardiac-Manifestation -- "
            "94pct-Penetrance-VARIABLE-Expressivity-Asymptomatic-Parent-COMMON"
        ),
        "alias": (
            "JAG1 (jagged canonical NOTCH ligand 1); OMIM gene 601920; "
            "Alagille syndrome type 1 (ALGS1) OMIM 118450. "
            "20p12.2; 1218 aa; ~134 kDa; NOTCH pathway transmembrane ligand; autosomal dominant (haploinsufficiency). "
            "FUNCTION: JAG1 encodes Jagged-1, a canonical NOTCH signalling ligand. "
            "JAG1 binds NOTCH receptors (especially NOTCH2) on adjacent cells, activating NOTCH signalling. "
            "During embryogenesis: JAG1-NOTCH2 signalling required for: "
            "(1) Intrahepatic bile duct development — specification of cholangiocyte fate; "
            "(2) Cardiovascular development — outflow tract, pulmonary arteries; "
            "(3) Vertebral segmentation — posterior vertebral arch; "
            "(4) Renal development; (5) Ocular development; (6) Facial morphology. "
            "Loss of one JAG1 allele (haploinsufficiency) → insufficient NOTCH signalling → "
            "BILE DUCT PAUCITY: reduced number of intrahepatic bile duct branches — "
            "< 0.5 intrahepatic bile ducts per portal tract (normal > 0.9) PATHOGNOMONIC. "
            "CLINICAL FEATURES (5 major): "
            "(1) CHOLESTASIS: bile duct paucity → intrahepatic cholestasis; GGT elevated (biliary origin); "
            "variable severity — may spontaneously improve in childhood; cirrhosis in ~15-20%; "
            "(2) CARDIAC: peripheral pulmonary stenosis (PPS) MOST COMMON — bilateral branch PA stenosis; "
            "tetralogy of Fallot in ~13%; complex CHD; cardiac cause of mortality; "
            "(3) VERTEBRAL: butterfly vertebrae — anterior clefting of vertebral body (posterior arch fusion defect); "
            "X-ray chest/spine: butterfly configuration PATHOGNOMONIC; no functional significance; "
            "(4) OCULAR: posterior embryotoxon (prominent Schwalbe ring) — slit lamp; 78% of ALGS1; "
            "not pathognomonic alone (seen in 8-15% of general population); "
            "(5) FACIAL: characteristic — prominent forehead, deep-set eyes, hypertelorism, "
            "pointed chin, straight nose with bulbous tip; characteristic from childhood; "
            "(6) RENAL: renal anomalies in ~40%; vesicoureteral reflux, renal dysplasia, RTA. "
            "VARIABLE EXPRESSIVITY: "
            "94% penetrance but EXTREMELY variable — asymptomatic/minimally affected parent is COMMON; "
            "Parent may only have posterior embryotoxon or butterfly vertebrae; "
            "DO NOT exclude JAG1 mutation in parent without full evaluation; "
            "Family cascade testing MANDATORY (ECG, echo, liver USS, slit lamp, spinal X-ray). "
            "DIAGNOSIS: "
            "Liver biopsy: bile duct paucity PATHOGNOMONIC (may be absent in first 6 months); "
            "Chest X-ray: butterfly vertebrae; Echocardiogram: PPS/CHD; Slit lamp: posterior embryotoxon; "
            "JAG1 gene sequencing: 94% of ALGS; whole gene deletion/duplication (MLPA). "
            "TREATMENT: "
            "Pruritus: rifampicin; cholestyramine; odevixibat/maralixibat (FDA approved ALGS1); "
            "Odevixibat (Bylvay): specifically FDA-approved for ALGS pruritus 2023; "
            "Biliary disease: UDCA (limited efficacy); "
            "Cardiac: PPS may require balloon valvuloplasty/surgery if severe; "
            "Liver transplant: 15-20% require LTx; biliary cirrhosis indication; NO recurrence post-LTx; "
            "Fat-soluble vitamins: A/D/E/K monitoring 6-monthly. "
            "KEY CLINICAL FACTS: "
            "BUTTERFLY VERTEBRAE PATHOGNOMONIC — standard chest X-ray shows it; pursue JAG1 immediately; "
            "PRURITUS IS SEVERE — often the most disabling symptom; odevixibat approved specifically for ALGS pruritus; "
            "CARDIAC ASSESSMENT MANDATORY — PPS, Fallot; cardiac cause of mortality in ALGS; echo at diagnosis; "
            "VARIABLE EXPRESSIVITY = PARENT MAY BE 'NORMAL' — asymptomatic parents still carry mutation; full family assessment mandatory."
        ),
        "age_of_onset": "Neonatal/infantile (cholestasis); cardiac manifestations may be immediate",
        "inheritance": "AD (haploinsufficiency)",
        "locus": "20p12.2",
        "protein_size": "1218 aa",
        "key_biomarker": "Bile duct paucity on biopsy (<0.5/portal tract); butterfly vertebrae on X-ray; posterior embryotoxon on slit lamp",
        "pathognomonic": "Butterfly vertebrae on X-ray PATHOGNOMONIC; bile duct paucity on biopsy PATHOGNOMONIC; peripheral pulmonary stenosis",
        "treatment": "Odevixibat FDA2023 for ALGS pruritus; rifampicin; cardiac Rx for PPS/CHD; LTx 15-20%; vitamins A/D/E/K",
        "critical_flags": [
            "BUTTERFLY-VERTEBRAE-PATHOGNOMONIC — chest X-ray; posterior arch fusion defect; diagnose immediately; sequence JAG1",
            "ODEVIXIBAT-FDA-APPROVED-ALGS — Bylvay FDA approved 2023 specifically for ALGS1 pruritus; first-line pharmacotherapy for pruritus",
            "CARDIAC-ASSESSMENT-MANDATORY — echo at diagnosis; PPS bilateral; TOF 13%; cardiac = leading cause of mortality; cardiology co-management",
            "VARIABLE-EXPRESSIVITY-94pct-PENETRANCE — parent may have only butterfly vertebrae / embryotoxon; full family evaluation not just 'looks normal'; MLPA for deletions",
            "BILE-DUCT-PAUCITY-FALSE-NEGATIVE-INFANTS — biopsy in first 6 months may show normal duct number; repeat at 6-12 months if clinical suspicion",
            "GGT-ELEVATED-IN-ALGS — biliary origin (ductal injury); CONTRAST to PFIC1/2/4/5/6 where GGT is LOW; ALGS has HIGH GGT like PFIC3",
            "FAT-SOLUBLE-VITAMINS — A/D/E/K monitoring 6-monthly; rickets and coagulopathy major complications; supplement aggressively",
        ],
    },
    # -- NOTCH2 -- Alagille Syndrome Type 2 / Hajdu-Cheney -------------------------
    {
        "gene": "NOTCH2",
        "protein": (
            "NOTCH2 -- 1p12 AD -- NOTCH2-2471aa -- "
            "Alagille-Syndrome-Type2-ALGS2-NOTCH2-Receptor-JAG1-Ligand-Partner -- "
            "Renal-Anomalies-MORE-Prominent-Than-ALGS1-Dysplasia-RTA -- "
            "Hepatic-Disease-GENERALLY-Milder-Than-ALGS1 -- "
            "Hajdu-Cheney-Syndrome-GOF-NOTCH2-Distinct-Acroosteolysis-Osteoporosis-Short-Stature"
        ),
        "alias": (
            "NOTCH2 (neurogenic locus notch homolog protein 2); OMIM gene 600275; "
            "Alagille syndrome type 2 (ALGS2) OMIM 610205 / Hajdu-Cheney syndrome OMIM 102500. "
            "1p12; 2471 aa; ~265 kDa; NOTCH family transmembrane receptor; autosomal dominant. "
            "FUNCTION: NOTCH2 encodes the NOTCH2 transmembrane receptor. "
            "NOTCH2 is the primary receptor for JAG1 signalling in hepatic ductal plate formation. "
            "ALGS1 (JAG1) and ALGS2 (NOTCH2) represent two sides of the same signalling axis — "
            "JAG1 (ligand) → NOTCH2 (receptor) → intrahepatic bile duct specification. "
            "LOF mutations in NOTCH2 → deficient JAG1-NOTCH2 signalling → "
            "bile duct paucity (as in ALGS1 but generally milder hepatic disease). "
            "ALGS2 COMPARED TO ALGS1: "
            "Only ~6% of all Alagille syndrome (JAG1 = 94%); "
            "RENAL ANOMALIES MORE PROMINENT: renal dysplasia, renal cysts, chronic kidney disease, "
            "renal tubular acidosis more frequent and severe than in ALGS1; "
            "HEPATIC DISEASE GENERALLY MILDER — bile duct paucity present but liver failure less frequent; "
            "Same cardiac (PPS), vertebral (butterfly vertebrae), ocular (posterior embryotoxon), facial features as ALGS1; "
            "Cannot be distinguished from ALGS1 clinically — requires gene sequencing. "
            "GOF MUTATIONS — HAJDU-CHENEY SYNDROME (DISTINCT): "
            "Gain-of-function NOTCH2 mutations (truncating exon 34 → stable NOTCH2 ICD accumulation) → "
            "Hajdu-Cheney syndrome: "
            "acroosteolysis (dissolution of distal phalanges) PATHOGNOMONIC; "
            "severe osteoporosis + multiple fractures; "
            "short stature, distinctive face; "
            "NO cholestasis in Hajdu-Cheney (GOF); "
            "DO NOT confuse with ALGS2 (LOF). "
            "CLINICAL FEATURES (ALGS2): "
            "Bile duct paucity on liver biopsy (same diagnostic criterion as ALGS1); "
            "Peripheral pulmonary stenosis (PPS); "
            "Butterfly vertebrae; Posterior embryotoxon; Characteristic face; "
            "Renal: renal dysplasia / cysts / RTA — more prominent than ALGS1; "
            "Renal failure may develop before liver failure in ALGS2. "
            "DIAGNOSIS: "
            "Liver biopsy: bile duct paucity; "
            "Renal USS + GFR: mandatory (renal anomalies more prominent); "
            "NOTCH2 gene sequencing: confirms; distinguished from JAG1 only by sequencing; "
            "Hajdu-Cheney exclusion: look for acroosteolysis on hand X-rays. "
            "TREATMENT: "
            "Same principles as ALGS1: odevixibat for pruritus; vitamins A/D/E/K; cardiac management; "
            "RENAL MONITORING: more rigorous in ALGS2 — GFR 6-monthly; USS annually; "
            "Avoid nephrotoxic drugs (aminoglycosides, NSAIDs) — reduced renal reserve; "
            "Liver transplant: fewer ALGS2 patients require LTx (milder hepatic disease); "
            "If concurrent renal failure: combined liver-kidney transplant consideration. "
            "KEY CLINICAL FACTS: "
            "ALGS2 = JAG1-NOTCH2 AXIS — same pathway, same liver/cardiac/vertebral/ocular features; sequencing distinguishes; "
            "RENAL IS THE DISTINGUISHING FEATURE — more renal disease in ALGS2; check GFR at every visit; "
            "HAJDU-CHENEY IS GOF NOT LOF — completely different phenotype; acroosteolysis key; no overlap with ALGS2 clinically; "
            "AVOID NEPHROTOXINS — renal reserve reduced; aminoglycosides/contrast/NSAIDs managed carefully."
        ),
        "age_of_onset": "Neonatal/infantile (ALGS2 bile duct paucity); renal disease may manifest in childhood/adolescence",
        "inheritance": "AD (LOF → ALGS2; GOF → Hajdu-Cheney)",
        "locus": "1p12",
        "protein_size": "2471 aa",
        "key_biomarker": "Bile duct paucity on biopsy; renal anomalies on USS (more prominent than ALGS1); GFR reduction",
        "pathognomonic": "ALGS2: bile duct paucity + renal dysplasia/CKD more prominent than ALGS1; Hajdu-Cheney (GOF): acroosteolysis PATHOGNOMONIC",
        "treatment": "Odevixibat; vitamins A/D/E/K; GFR monitoring 6-monthly; avoid nephrotoxins; combined liver-kidney Tx if both fail",
        "critical_flags": [
            "RENAL-MORE-PROMINENT-THAN-ALGS1 — GFR monitoring 6-monthly mandatory in ALGS2; renal failure may precede liver failure",
            "HAJDU-CHENEY-GOF-DISTINCT — acroosteolysis (distal phalangeal osteolysis) + osteoporosis; no cholestasis; DO NOT confuse with ALGS2 LOF",
            "SEQUENCING-REQUIRED-TO-DISTINGUISH-FROM-JAG1 — ALGS1 and ALGS2 clinically indistinguishable; gene panel mandatory",
            "AVOID-NEPHROTOXINS — aminoglycosides/contrast/NSAIDs; renal reserve reduced; document in medication allergy/caution record",
            "COMBINED-LIVER-KIDNEY-Tx — if both organs fail concurrently; plan early with transplant team; milder hepatic disease means less common",
            "GFR-DECLINE-SILENT — renal dysplasia causes gradual GFR decline without symptoms; active monitoring not symptom-triggered",
            "SAME-PATHWAY-AS-JAG1 — JAG1 (ligand) NOTCH2 (receptor); both cause bile duct paucity; shared mechanism, different gene; same NOTCH pathway targeted",
        ],
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    ages = [round(rng.gauss(1.8, 2.5), 1) for _ in range(n)]
    ages = [max(0.0, min(18.0, a)) for a in ages]
    sexes = [rng.choice(["M", "F"]) for _ in range(n)]
    severities = [rng.choice(["mild", "moderate", "severe"]) for _ in range(n)]
    # NR1H4 (PFIC5): all severe (most severe PFIC)
    if gene_entry["gene"] == "NR1H4":
        severities = ["severe"] * n
    # MYO5B (MVID): predominantly severe
    if gene_entry["gene"] == "MYO5B":
        severities = [rng.choices(["moderate", "severe"], weights=[1, 3])[0] for _ in range(n)]
    # ABCB4 LPAC: mixed; adults later onset — adjust ages
    if gene_entry["gene"] == "ABCB4":
        ages = [round(rng.gauss(8.0, 6.0), 1) for _ in range(n)]
        ages = [max(0.0, min(18.0, a)) for a in ages]
    return [
        {
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "age_at_diagnosis_yr": ages[i],
            "sex": sexes[i],
            "severity": severities[i],
            "inheritance": gene_entry["inheritance"],
            "locus": gene_entry["locus"],
        }
        for i in range(n)
    ]


def overview() -> dict:
    all_patients = []
    for idx, g in enumerate(CHOLESTASIS_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        all_patients.extend(cohort)

    total = len(all_patients)
    gene_counts = {}
    for p in all_patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    age_vals = [p["age_at_diagnosis_yr"] for p in all_patients]
    avg_age = round(sum(age_vals) / len(age_vals), 1)
    severe_count = sum(1 for p in all_patients if p["severity"] == "severe")

    gene_summary = []
    for g in CHOLESTASIS_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": gene_counts.get(g["gene"], 0),
            "critical_flags": g["critical_flags"],
        })

    return {
        "atlas": "Hereditary-Cholestasis-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Cholestatic Liver Disease Atlas — "
            "ATP8B1-1251aa-18q21.31-AR-PFIC1-FIC1-Low-GGT-Extrahepatic-Diarrhea-Post-LTx-Worsens | "
            "ABCB11-1321aa-2q31.1-AR-PFIC2-BSEP-Low-GGT-HCC-Without-Cirrhosis-Odevixibat-FDA2021 | "
            "ABCB4-1279aa-7q21.12-AR/AD-PFIC3/LPAC-MDR3-HIGH-GGT-PATHOGNOMONIC-UDCA-First-Line | "
            "TJP2-1221aa-9q21.11-AR-PFIC4-ZO2-Low-GGT-HCC-Without-Cirrhosis-Neurological-Subset | "
            "NR1H4-472aa-12q23.1-AR-PFIC5-FXR-Low-GGT-Most-Severe-FXR-Agonists-CANNOT-Work | "
            "MYO5B-1852aa-18q21.1-AR-PFIC6/MVID-Low-GGT-Neonatal-Diarrhea-300mL/kg/day-TPN | "
            "JAG1-1218aa-20p12.2-AD-Alagille1-Bile-Duct-Paucity-Butterfly-Vertebrae-PATHOGNOMONIC | "
            "NOTCH2-2471aa-1p12-AD-Alagille2-Renal-More-Prominent-Hajdu-Cheney-GOF-DISTINCT | "
            "320-Patient-Aggregate-8x40-seeds-1894-1901"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(CHOLESTASIS_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(CHOLESTASIS_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "GGT-HIGH-vs-LOW: PFIC3 (ABCB4) = HIGH GGT PATHOGNOMONIC; all other PFIC types (ATP8B1/ABCB11/TJP2/NR1H4/MYO5B) = LOW GGT; Alagille (JAG1/NOTCH2) = HIGH GGT (biliary origin)",
            "HCC-WITHOUT-CIRRHOSIS: PFIC2 (ABCB11) + PFIC4 (TJP2) — both cause HCC without cirrhosis; MANDATORY surveillance MRI+AFP 6-monthly from birth",
            "FXR-AGONISTS-CI-PFIC5: obeticholic acid and all FXR agonists CANNOT work in NR1H4 (PFIC5) — receptor absent; prescribing error",
            "POST-LTx-DIARRHEA-WORSE: ATP8B1 (PFIC1) only — FIC1 expressed in intestine; LTx corrects liver but extrahepatic diarrhea worsens; unique to PFIC1",
            "UDCA-MOST-EFFECTIVE-PFIC3: ABCB4 (PFIC3/LPAC) is the most UDCA-responsive PFIC type; trial mandatory; LPAC → lifelong UDCA prevents stone recurrence",
            "BUTTERFLY-VERTEBRAE-PATHOGNOMONIC: JAG1 (Alagille1) — X-ray; diagnose immediately; pursue JAG1 gene sequencing; check NOTCH2 if negative",
            "MVID-TPN-DEPENDENT: MYO5B (PFIC6/MVID) — massive secretory diarrhoea; no oral feeds; TPN immediately; combined liver+intestine Tx only curative option",
            "ODEVIXIBAT-FDA-APPROVED-ALGS: JAG1 Alagille syndrome — odevixibat (Bylvay) FDA 2023 specifically for ALGS pruritus; best pharmacological evidence in ALGS",
            "ALGS2-RENAL-PROMINENCE: NOTCH2 (ALGS2) — renal dysplasia/CKD more prominent than ALGS1; GFR monitoring 6-monthly; avoid nephrotoxins",
            "PEBD-BEFORE-LTx: partial external biliary diversion can arrest PFIC1/2/4 progression if < stage 3 fibrosis; attempt before LTx listing",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(CHOLESTASIS_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(CHOLESTASIS_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Cholestasis-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in CHOLESTASIS_GENES
        ],
        "glossary": {
            "PFIC": "Progressive familial intrahepatic cholestasis — hereditary cholestasis types 1-6; all AR; classified by GGT (low: PFIC1/2/4/5/6; high: PFIC3)",
            "BRIC": "Benign recurrent intrahepatic cholestasis — allelic to PFIC; episodic cholestasis; self-limiting; ATP8B1 (BRIC1) or ABCB11 (BRIC2)",
            "ALGS": "Alagille syndrome — NOTCH signalling defect; JAG1 (ALGS1, 94%) or NOTCH2 (ALGS2, 6%); bile duct paucity + cardiac + vertebral + ocular + renal + facial features",
            "Bile duct paucity": "< 0.5 intrahepatic bile ducts per portal tract (normal > 0.9); pathognomonic for ALGS on liver biopsy; may be absent in infants < 6 months",
            "BSEP": "Bile salt export pump (ABCB11) — primary canalicular bile salt transporter; ATP-dependent; deficient in PFIC2; odevixibat reduces entero-hepatic bile salt recirculation",
            "Butterfly vertebrae": "Posterior vertebral arch fusion defect — X-ray appearance of bifid vertebral body; pathognomonic for Alagille syndrome; JAG1 or NOTCH2; no functional significance",
            "FIC1": "ATP8B1 product — phospholipid flippase; canalicular membrane asymmetry; deficient in PFIC1/BRIC1; extrahepatic expression = post-LTx diarrhea signature",
            "FXR": "Farnesoid X receptor (NR1H4) — bile acid nuclear sensor; upregulates BSEP + MDR3; absent in PFIC5; FXR agonists (obeticholic acid) CANNOT work in PFIC5",
            "GGT": "Gamma-glutamyl transferase — marker of biliary epithelial damage; HIGH GGT = PFIC3 (MDR3/ABCB4) or Alagille; LOW GGT = PFIC1/2/4/5/6; critical discriminator",
            "HCC without cirrhosis": "Hepatocellular carcinoma developing without established cirrhosis — PFIC2 (ABCB11) and PFIC4 (TJP2) signature; bile salt-mediated direct hepatocyte carcinogenesis",
            "IBAT inhibitor": "Ileal bile acid transporter inhibitor — odevixibat (Bylvay) / maralixibat (Livmarli); interrupts entero-hepatic bile salt recycling; reduces hepatocyte bile salt burden; FDA-approved for PFIC and ALGS",
            "LPAC": "Low-phospholipid-associated cholelithiasis — heterozygous ABCB4 (MDR3) mutation; intrahepatic biliary stones; young adult; recurrence after cholecystectomy; UDCA prevents recurrence",
            "MDR3": "Multidrug resistance protein 3 (ABCB4) — canalicular phospholipid translocase; floppase for phosphatidylcholine; deficient in PFIC3/LPAC; absent PC → toxic free bile salts → biliary epithelial injury → HIGH GGT",
            "Microvillus inclusion disease (MVID)": "MYO5B deficiency — apical membrane recycling defect in intestinal enterocytes; microvillus inclusions on EM; neonatal secretory diarrhea 200-300 mL/kg/day; TPN-dependent; combined intestine+liver Tx",
            "Odevixibat (Bylvay)": "IBAT inhibitor — FDA 2021 (PFIC) + 2023 (ALGS); reduces ileal bile acid reabsorption → reduces hepatocyte bile acid load; best efficacy PFIC2 (residual BSEP: E297G) + ALGS1",
            "PEBD": "Partial external biliary diversion — surgical; catheter from gallbladder to abdominal skin; diverts bile externally → reduces entero-hepatic bile salt cycling; effective in PFIC1/2 before stage 3 fibrosis",
            "PPS": "Peripheral pulmonary stenosis — bilateral branch pulmonary artery stenosis; most common cardiac finding in Alagille syndrome (JAG1/NOTCH2); may require balloon valvuloplasty",
            "Posterior embryotoxon": "Prominent Schwalbe ring — slit lamp; anterior segment anomaly; 78% of ALGS1; not pathognomonic alone (8-15% general population); part of ALGS diagnostic criteria",
            "UDCA": "Ursodeoxycholic acid — protective hydrophilic bile acid; replaces toxic hydrophobic bile salts; most effective in PFIC3 (ABCB4); also used in PFIC1/2 and Alagille; dose 15-20 mg/kg/day",
            "ZO-2": "Zona occludens 2 (TJP2) — tight junction scaffold; canalicular junction integrity; deficient in PFIC4; paracellular bile leak mechanism; HCC without cirrhosis; neurological features in subset",
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-CHOLESTASIS-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = breakdown()
    print(json.dumps(bd["genes"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
