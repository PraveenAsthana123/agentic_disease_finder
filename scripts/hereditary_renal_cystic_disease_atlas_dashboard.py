#!/usr/bin/env python3
"""Hereditary-Renal-Cystic-Disease-Atlas — Complete 8-Gene Atlas
PKD1    (Polycystin-1; 4304 aa; 16p13.3; AD;
         Autosomal Dominant Polycystic Kidney Disease type 1 (ADPKD1);
         most common monogenic renal disease (1:400–1,000);
         PC1 = large integral membrane protein; TRP-channel partner (PC2);
         bilateral renal cysts expanding over decades → ESRD median age 54 yr;
         intracranial aneurysms 4× general population; liver cysts 80% women;
         tolvaptan (Jynarque) V2R antagonist FDA2018 — FIRST approved ADPKD therapy;
         seed SEED_BASE+0) .
PKD2    (Polycystin-2 / TRPP2; 968 aa; 4q22.1; AD;
         Autosomal Dominant Polycystic Kidney Disease type 2 (ADPKD2);
         15% ADPKD; milder — ESRD median age 74 yr (20-yr later than PKD1);
         tolvaptan FDA-approved same indication;
         seed SEED_BASE+1) .
PKHD1   (Fibrocystin / Polyductin; 4074 aa; 6p21.2; AR;
         Autosomal Recessive Polycystic Kidney Disease (ARPKD);
         most common inherited renal cystic disease in children (1:20,000);
         collecting-duct cysts (ectatic, not balloon cysts) + CHF mandatory;
         oligohydramnios → Potter sequence → neonatal respiratory failure;
         liver fibrosis ALWAYS present; may dominate over renal in older children;
         seed SEED_BASE+2) .
MUC1    (Mucin-1; 1255 aa; 1q22; AD;
         Autosomal Dominant TubuloInterstitial Kidney Disease — MUC1 (ADTKD-MUC1);
         cytosine insertion in GC-rich VNTR — STANDARD NGS MISSES — specific long-read or VNTR assay required;
         tubulointerstitial nephritis without cysts; slow ESRD 5th–6th decade;
         frameshift produces toxic MUC1-fs protein → ER stress → tubular cell death;
         no specific therapy — ACE-i for proteinuria; avoid NSAIDs + contrast;
         seed SEED_BASE+3) .
UMOD    (Uromodulin / Tamm-Horsfall protein; 640 aa; 16p12.3; AD;
         Autosomal Dominant TubuloInterstitial Kidney Disease — UMOD (ADTKD-UMOD);
         hyperuricemia + gout in teens/20s PATHOGNOMONIC — screen UMOD first;
         uromodulin = most abundant urinary protein; mutant UMOD misfolds → ER;
         allopurinol for hyperuricemia/gout prevention; febuxostat alternative;
         medullary cysts (not always visible on ultrasound — MRI better);
         seed SEED_BASE+4) .
REN     (Renin; 406 aa; 1q32.1; AD;
         Autosomal Dominant TubuloInterstitial Kidney Disease — REN (ADTKD-REN);
         childhood anemia + childhood hyperkalemia + borderline-LOW BP before ESRD;
         LOF → low-renin hypoaldosteronism → tubular dysfunction;
         childhood anemia + elevated creatinine in family member without hypertension = screen REN;
         no specific treatment — ESA for anemia; avoid ACE-i/ARB (exacerbate hyperkalemia);
         seed SEED_BASE+5) .
HNF1B   (Hepatocyte nuclear factor 1-beta; 557 aa; 17q12; AD;
         Renal Cysts And Diabetes syndrome (RCAD) / MODY5;
         MLPA MANDATORY — 50% large deletions MISSED by Sanger/NGS panel;
         renal cysts often prenatal; MODY5 DM usually preceded by cysts;
         pancreatic hypoplasia → exocrine insufficiency (PERT); hypomagnesaemia;
         uterine anomalies (bicornuate 30%), gout/hyperuricemia, abnormal LFTs;
         sulfonylurea INEFFECTIVE — insulin required; multiorgan surveillance mandatory;
         seed SEED_BASE+6) .
DNAJB11 (DnaJ heat shock protein family member B11; 354 aa; 3q27.3; AD;
         Autosomal Dominant TubuloInterstitial Kidney Disease — DNAJB11 (ADTKD-DNAJB11);
         newest ADTKD gene (2018); atypical polycystic pattern mimicking PKD1/PKD2;
         DNAJB11 = ER co-chaperone; mutant → ER stress → tubular + cyst cell death;
         PKD1/PKD2 negative atypical polycystic → sequence DNAJB11;
         cysts variable; slow progression ESRD 6th–7th decade;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1918–1925)
"""

import random

SEED_BASE = 1918

RENAL_CYSTIC_GENES = [
    # -- PKD1 — ADPKD type 1 / Polycystin-1 ------------------------------------------
    {
        "gene": "PKD1",
        "alt_name": "Polycystin-1",
        "protein": (
            "PKD1 -- 16p13.3 AD -- Polycystin1-4304aa -- "
            "ADPKD1-Most-Common-85pct-ADPKD-ESRD-Median-54yr -- "
            "Tolvaptan-Jynarque-V2R-Antagonist-FDA2018-FIRST-ADPKD-Therapy -- "
            "Intracranial-Aneurysm-4x-General-Population-Screen-Positive-Family-History -- "
            "Bilateral-Renal-Cysts-Liver-Cysts-80pct-Women"
        ),
        "locus": "16p13.3",
        "protein_size": "4304 aa",
        "inheritance": "AD",
        "age_of_onset": "Cysts form in utero; renal enlargement childhood; ESRD median 54 yr (PKD1 vs 74 yr PKD2)",
        "key_biomarker": (
            "Total Kidney Volume (TKV) by MRI — Mayo Imaging Classification (1A–1E): 1C–1E = rapid progressors → tolvaptan eligible; "
            "eGFR decline >2.5 mL/min/yr = rapid; urine osmolality <280 mOsm/kg (V2R effect of tolvaptan); "
            "ALT/AST before tolvaptan (hepatotoxicity monitoring); urine albumin:creatinine ratio; "
            "serum sodium (tolvaptan aquaresis — maintain adequate hydration)"
        ),
        "pathognomonic": (
            "Bilateral multiple renal cysts + family history (AD pattern) + age-appropriate cyst count: "
            "Ravine criteria age <30 yr: ≥2 cysts (either kidney); age 30–59: ≥2 cysts per kidney; age ≥60: ≥4 cysts per kidney; "
            "massive bilateral kidney enlargement (TKV >1,500 mL) with preserved function = PKD1/2 until proven otherwise; "
            "NO hematuria + NO hypertension + NO flank pain in young with bilateral cysts = presymptomatic PKD"
        ),
        "treatment": (
            "Tolvaptan (Jynarque, V2R antagonist, FDA2018) — FIRST approved ADPKD therapy; "
            "reduces TKV growth rate 49% and eGFR decline 26% (TEMPO 3:4, REPRISE trials); "
            "eligible: age 18–55, eGFR ≥25, Mayo 1C–1E or equivalent rapid progression evidence; "
            "HEPATOTOXIC: LFTs monthly ×18 months then every 3 months; STOP if ALT >2×ULN; "
            "AVOID: low-fluid intake — must drink ≥3 L/day; grapefruit juice (CYP3A4); "
            "ACE-i/ARB for hypertension (30–50% have HTN); avoid NSAIDs (nephrotoxic); "
            "intracranial aneurysm: screen with MRA if first-degree relative with ICA rupture or high-risk occupation; "
            "pain: analgesics → cyst aspiration → laparoscopic fenestration → nephrectomy; "
            "renal replacement: transplant preferred over dialysis (excellent outcomes)"
        ),
        "critical_flags": [
            "PKD1-TOLVAPTAN-FDA2018-FIRST-APPROVED: V2R antagonist; FIRST disease-modifying ADPKD approval; rapid progressors only (Mayo 1C-1E)",
            "PKD1-VS-PKD2-ESRD: PKD1 ESRD median 54 yr; PKD2 ESRD median 74 yr; same presentation, 20-yr prognosis difference",
            "PKD1-ICA-4x-RISK: intracranial aneurysm 4–12% vs 1–2% general; screen MRA if positive family history of ICA or high-risk job",
            "PKD1-TOLVAPTAN-HEPATOTOXICITY: ALT monitoring mandatory; idiosyncratic; STOP if ALT >2×ULN; fatal cases reported",
            "PKD1-MAYO-CLASSIFICATION: MRI TKV mandatory for tolvaptan eligibility; ultrasound insufficient for classification",
            "PKD1-HTN-FIRST-SIGN: hypertension common age 20–34 before renal impairment; ACE-i/ARB first-line",
            "PKD1-LIVER-CYSTS-WOMEN: liver cysts 80% women, 40% men; PCLD (polycystic liver disease) — portal hypertension rare",
            "PKD1-GENE-SIZE: largest gene causing ADPKD (4304 aa); 15% de novo; 70% of all ADPKD cases",
        ],
        "alias": (
            "PKD1 (polycystic kidney disease 1; 4304 aa; 16p13.3) encodes polycystin-1 (PC1), a large integral "
            "membrane protein with extracellular leucine-rich repeats and PKD domains. "
            "PC1 forms a receptor-channel complex with PC2 (PKD2, TRPP2); "
            "LOF disrupts mechanosensation in renal tubular primary cilia → mTOR + cAMP dysregulation → cystogenesis. "
            "ADPKD1 (OMIM #173900) affects 1:400–1,000 persons; PKD1 accounts for 70–85% of ADPKD. "
            "Bilateral multiple renal cysts with Ravine age-adjusted criteria diagnose >99% of at-risk individuals. "
            "Total Kidney Volume (TKV) by MRI — Mayo classification 1A–1E — identifies rapid progressors (1C–1E) for tolvaptan. "
            "Tolvaptan (Jynarque, V2R antagonist, FDA 2018): FIRST approved ADPKD disease-modifying therapy; "
            "reduces TKV growth 49% and eGFR decline 26% (TEMPO 3:4; REPRISE); hepatotoxic — monthly LFT monitoring. "
            "Extrarenal manifestations: intracranial aneurysms 4–12% (MRA screen if +ve family history of rupture), "
            "liver cysts (80% women), pancreatic cysts, cardiac valve disease (MVP 25%). "
            "Hypertension affects 50–70% by age 30 (intrarenal renin-angiotensin activation — ACE-i/ARB first-line). "
            "ESRD at median 54 yr (PKD1) vs 74 yr (PKD2); transplant outcomes excellent."
        ),
    },

    # -- PKD2 — ADPKD type 2 / Polycystin-2 / TRPP2 -----------------------------------
    {
        "gene": "PKD2",
        "alt_name": "Polycystin-2 / TRPP2",
        "protein": (
            "PKD2 -- 4q22.1 AD -- Polycystin2-968aa -- "
            "ADPKD2-15pct-ADPKD-MILDER-ESRD-Median-74yr-20yr-Later-Than-PKD1 -- "
            "Tolvaptan-Same-Indication-If-Rapid-Progression-Mayo-1C-1E -- "
            "TRP-Channel-PC1-PC2-Heterotetrameric-Complex-Cilia-Mechanosensation -- "
            "ICA-Risk-Lower-Than-PKD1-But-Still-Screen-Positive-Family-History"
        ),
        "locus": "4q22.1",
        "protein_size": "968 aa",
        "inheritance": "AD",
        "age_of_onset": "Cysts form in 3rd–4th decade (later than PKD1); ESRD median 74 yr; many never reach ESRD",
        "key_biomarker": (
            "TKV by MRI (Mayo 1A–1E) — same as PKD1; tolvaptan eligibility identical criteria; "
            "genetic testing distinguishes PKD1 vs PKD2 (important for prognosis counselling); "
            "eGFR trajectory; urine osmolality post-tolvaptan; "
            "PKD2 patients overall fewer cysts per kidney, later onset, lower TKV at same age"
        ),
        "pathognomonic": (
            "Bilateral multiple renal cysts with Ravine criteria + family history; "
            "PKD2 cysts indistinguishable from PKD1 on imaging — genetic testing required for PKD1/PKD2 differentiation; "
            "ESRD age >60 yr in PKD2 patient without comorbidities = strongly suggests PKD2 (not PKD1); "
            "atypical ADPKD with later onset and milder progression → consider PKD2 first"
        ),
        "treatment": (
            "Tolvaptan (Jynarque, FDA2018) — same eligibility as PKD1 (Mayo 1C–1E or rapid progressor evidence); "
            "many PKD2 patients are 1A–1B (slow) and do NOT need tolvaptan; "
            "ACE-i/ARB for hypertension; avoid NSAIDs; "
            "ICA screen if positive family history of rupture; "
            "genetic counselling: 50% offspring risk; "
            "transplant preferred over dialysis for ESRD"
        ),
        "critical_flags": [
            "PKD2-MILDER-ESRD-74yr: 20 years later than PKD1 (54 yr); critical prognosis counselling point",
            "PKD2-SAME-TOLVAPTAN-IF-RAPID: if Mayo 1C-1E or rapid eGFR decline — tolvaptan eligible regardless of PKD1/2",
            "PKD2-GENETIC-NEEDED: imaging cannot differentiate PKD1 vs PKD2; genetic testing essential for prognosis",
            "PKD2-MANY-NEVER-ESRD: PKD2 patients often die of cardiovascular disease before reaching ESRD",
            "PKD2-ICA-LOWER-RISK: ICA still elevated vs general population but lower than PKD1; screen if +ve FHx rupture",
            "PKD2-TRP-CHANNEL: PKD2 encodes TRPP2 (TRP channel) — Ca²⁺-permeable; PC1-PC2 heterotetrameric complex",
            "PKD2-LIVER-CYSTS-LESS: fewer liver cysts than PKD1; less likely PCLD",
            "PKD2-HTN-ALSO-COMMON: hypertension 50-70% at presentation, same as PKD1 — ACE-i/ARB first-line",
        ],
        "alias": (
            "PKD2 (polycystic kidney disease 2; 968 aa; 4q22.1) encodes polycystin-2 (PC2), a transient receptor "
            "potential (TRPP2) calcium channel. PC2 requires PC1 for trafficking to primary cilia; "
            "together they form a mechanosensory complex that regulates intracellular Ca²⁺ and mTOR/cAMP signalling. "
            "ADPKD2 (OMIM #613095) accounts for 15% of ADPKD; 1:5,000–1:10,000 persons. "
            "PKD2 is clinically milder than PKD1: ESRD at median 74 yr (vs 54 yr PKD1); "
            "many PKD2 patients die of cardiovascular disease before reaching ESRD. "
            "Diagnosis: Ravine ultrasound criteria (same as PKD1); genetic testing differentiates PKD1/2. "
            "Treatment: tolvaptan (Jynarque, FDA 2018) if rapid progressor (Mayo 1C–1E); "
            "many PKD2 patients are 1A–1B and do NOT qualify. "
            "ACE-i/ARB for hypertension; avoid NSAIDs; ICA screening if positive family history. "
            "Genetic counselling: AD 50% transmission; de novo PKD2 mutations exist but less common than PKD1."
        ),
    },

    # -- PKHD1 — ARPKD / Fibrocystin / Polyductin ------------------------------------
    {
        "gene": "PKHD1",
        "alt_name": "Fibrocystin / Polyductin",
        "protein": (
            "PKHD1 -- 6p21.2 AR -- Fibrocystin-4074aa -- "
            "ARPKD-Most-Common-Inherited-Renal-Cystic-Disease-Children-1:20000 -- "
            "Collecting-Duct-Ectasia-NOT-Balloon-Cysts-Sunburst-Pattern-MRI -- "
            "Congenital-Hepatic-Fibrosis-CHF-ALWAYS-Present-Portal-Hypertension -- "
            "Oligohydramnios-Potter-Sequence-Neonatal-Respiratory-Failure-30pct-Mortality"
        ),
        "locus": "6p21.2",
        "protein_size": "4074 aa",
        "inheritance": "AR",
        "age_of_onset": "Perinatal/neonatal (severe) to childhood/adolescence (milder); biallelic mutations; one truncating = severe",
        "key_biomarker": (
            "Renal ultrasound: echogenic kidneys with sunburst/striped pattern (ectatic collecting ducts — NOT discrete cysts); "
            "liver biopsy: congenital hepatic fibrosis (CHF) — always present; "
            "portal hypertension: splenomegaly + varices (Doppler ultrasound); "
            "fibrocystin urine levels (research); "
            "genetic testing: PKHD1 is the largest gene in the human genome (67 exons); "
            "one truncating + one missense = intermediate; two truncating = perinatal lethal"
        ),
        "pathognomonic": (
            "Echogenic kidneys with ectatic collecting ducts (sunburst/corticomedullary stripes) + "
            "congenital hepatic fibrosis (CHF) on biopsy — ARPKD confirmed; "
            "ARPKD kidneys: fusiform tubular cysts (not spherical); preserve kidney shape (vs ADPKD balloons); "
            "Porter sequence (oligohydramnios → hypoplastic lungs → Potter facies) — perinatal lethal; "
            "Caroli disease/syndrome (intrahepatic bile duct dilatation) with CHF = ARPKD association"
        ),
        "treatment": (
            "No approved targeted therapy; supportive: "
            "neonatal respiratory: ventilation (PPV/CPAP); bilateral nephrectomy + transplant allows lung growth; "
            "hypertension: ACE-i/ARB (intrarenal RAS activation); "
            "portal hypertension: beta-blocker (propranolol/nadolol) for varices; band ligation/TIPS; "
            "cholangitis (recurrent biliary tract infections): antibiotics prophylaxis; "
            "liver disease dominant: isolated liver transplant; combined liver-kidney transplant; "
            "renal transplant: excellent outcomes — liver fibrosis usually manageable; "
            "ursodeoxycholic acid (UDCA) for biliary complications (limited evidence)"
        ),
        "critical_flags": [
            "PKHD1-CHF-ALWAYS-PRESENT: congenital hepatic fibrosis in ALL ARPKD — liver fibrosis mandates biliary + portal evaluation",
            "PKHD1-COLLECTING-DUCT-ECTASIA-NOT-DISCRETE-CYSTS: echogenic kidneys with stripes, not balloon cysts; sunburst US pattern",
            "PKHD1-POTTER-SEQUENCE: oligohydramnios → pulmonary hypoplasia → respiratory failure at birth (30% neonatal mortality)",
            "PKHD1-LIVER-MAY-DOMINATE: in surviving children liver disease (portal HTN, cholangitis) often overtakes renal disease",
            "PKHD1-TWO-TRUNCATING-LETHAL: biallelic truncating mutations = perinatal lethal; one truncating + one missense = intermediate phenotype",
            "PKHD1-CAROLI-ASSOCIATION: Caroli syndrome (intrahepatic bile duct dilatation) + CHF = ARPKD until proven otherwise",
            "PKHD1-BP-ACE-I: hypertension nearly universal; ACE-i/ARB first-line from infancy",
            "PKHD1-COMBINED-TRANSPLANT: combined liver-kidney transplant for severe bimodal disease; liver alone if renal preserved",
        ],
        "alias": (
            "PKHD1 (polycystic kidney and hepatic disease 1; 4074 aa; 6p21.2) encodes fibrocystin/polyductin, "
            "a receptor-like protein localised to primary cilia, centrosomes, and basal bodies of renal collecting ducts "
            "and intrahepatic bile ducts. LOF → ectasia of renal collecting ducts and hepatic bile ducts. "
            "ARPKD (OMIM #263200): 1:20,000 births; biallelic mutations required; AR inheritance. "
            "Genotype-phenotype: two truncating mutations → perinatal lethal (pulmonary hypoplasia from oligohydramnios); "
            "one truncating + one missense → survival with variable renal/liver disease; two missense → mildest. "
            "Renal: echogenic kidneys with fusiform ectatic collecting ducts (sunburst/striped pattern on US — not spherical cysts); "
            "kidneys may be massively enlarged but maintain reniform shape; "
            "ESRD in 30–40% by adolescence, >50% by 20 yr. "
            "Liver: congenital hepatic fibrosis (CHF) — universally present; portal hypertension → varices, splenomegaly; "
            "Caroli syndrome (intrahepatic bile duct dilatation) in ~25%; recurrent cholangitis risk. "
            "Treatment: supportive — ACE-i/ARB (HTN), beta-blockers (varices), UDCA (biliary), antibiotics (cholangitis); "
            "bilateral nephrectomy + transplant for neonatal lung survival; combined liver-kidney transplant for severe bimodal disease."
        ),
    },

    # -- MUC1 — ADTKD-MUC1 / Mucin-1 / Frameshift VNTR --------------------------------
    {
        "gene": "MUC1",
        "alt_name": "Mucin-1 / ADTKD-MUC1",
        "protein": (
            "MUC1 -- 1q22 AD -- Mucin1-1255aa-Variable-VNTR -- "
            "ADTKD-MUC1-Frameshift-VNTR-STANDARD-NGS-MISSES-Long-Read-Assay-Required -- "
            "Tubulointerstitial-Nephritis-NO-Cysts-Slow-ESRD-5th-6th-Decade -- "
            "MUC1-fs-Toxic-Protein-ER-Stress-Tubular-Cell-Death -- "
            "Hyperuricemia-Gout-NOT-Prominent-DDx-UMOD"
        ),
        "locus": "1q22",
        "protein_size": "1255 aa (VNTR-dependent; varies)",
        "inheritance": "AD",
        "age_of_onset": "Slow progression; ESRD 5th–6th decade (range 30–80 yr); childhood gout absent (DDx vs UMOD)",
        "key_biomarker": (
            "Renal biopsy: tubulointerstitial fibrosis (non-specific) — no diagnostic markers; "
            "genetic testing: STANDARD NGS/exome MISSES MUC1 frameshift in VNTR (GC-rich repeat region); "
            "specific long-read sequencing (PacBio/Oxford Nanopore) or VNTR assay required; "
            "immunohistochemistry: anti-MUC1-fs antibody detects truncated protein in tubular cells (research tool); "
            "urinary MUC1-fs (research); "
            "family history essential: AD pattern, multiple family members with ESRD without cause"
        ),
        "pathognomonic": (
            "Adult-onset chronic kidney disease + tubulointerstitial pattern on biopsy + AD family history + "
            "STANDARD NGS NEGATIVE — suspect ADTKD-MUC1; "
            "no diagnostic imaging finding (cysts absent or minimal); "
            "hyperuricemia NOT prominent (DDx from ADTKD-UMOD where gout is hallmark); "
            "MUC1-fs protein on IHC of tubular cells = diagnostic (if biopsy obtained)"
        ),
        "treatment": (
            "No approved targeted therapy; supportive: "
            "ACE-i/ARB for proteinuria/hypertension (mild to moderate); "
            "AVOID NSAIDs + contrast agents + nephrotoxins; "
            "allopurinol only if overt hyperuricemia/gout (less common than UMOD); "
            "genetic counselling: 50% offspring risk; "
            "renal replacement: transplant (excellent outcomes; recurrence NOT reported post-transplant); "
            "investigational: BRD4 inhibitors (reduce MUC1-fs expression — preclinical)"
        ),
        "critical_flags": [
            "MUC1-STANDARD-NGS-MISSES: cytosine insertion in GC-rich VNTR — exome/panel NGS FAILS; specific long-read or VNTR assay required",
            "MUC1-TUBULOINTERSTITIAL-NO-CYSTS: tubulointerstitial fibrosis without cysts on biopsy; imaging unhelpful",
            "MUC1-fs-TOXIC-PROTEIN: frameshift generates MUC1-fs gain-of-toxic-function; ER stress → tubular death",
            "MUC1-SLOW-PROGRESSION: ESRD 5th-6th decade; decades-long slow decline; early diagnosis important for counselling",
            "MUC1-DDx-UMOD: MUC1 = NO childhood gout; UMOD = childhood gout + hyperuricemia PATHOGNOMONIC",
            "MUC1-AD-FAMILY-HISTORY: multiple family members with ESRD + negative workup = ADTKD — send VNTR assay",
            "MUC1-TRANSPLANT-SAFE: MUC1-fs not re-expressed in donor kidney; transplant outcomes excellent",
            "MUC1-INVESTIGATIONAL: BRD4 inhibitors reduce MUC1-fs in tubular cells — phase 1/2 trials anticipated",
        ],
        "alias": (
            "MUC1 (mucin-1; 1255 aa; 1q22) encodes mucin-1, a large glycoprotein with a central variable number tandem repeat (VNTR). "
            "ADTKD-MUC1 (OMIM #174000) is caused by a single cytosine insertion within the GC-rich VNTR — "
            "a frameshift generating a toxic gain-of-function protein (MUC1-fs) that accumulates in the ER "
            "of distal tubular cells, triggering ER stress, apoptosis, and progressive tubulointerstitial fibrosis. "
            "CRITICAL: standard short-read NGS/exome sequencing MISSES this mutation due to the repetitive GC-rich VNTR; "
            "diagnosis requires long-read sequencing (PacBio/Oxford Nanopore) or a dedicated VNTR-specific PCR assay. "
            "Clinical: AD inheritance; chronic progressive CKD without cysts; ESRD 5th–6th decade (range 30–80 yr); "
            "tubulointerstitial pattern on renal biopsy; hyperuricemia/gout NOT prominent (DDx from ADTKD-UMOD); "
            "no extrarenal manifestations. "
            "Treatment: supportive only; ACE-i/ARB; avoid nephrotoxins; "
            "renal transplant excellent outcomes; MUC1-fs NOT expressed in donor kidney — no recurrence. "
            "BRD4 inhibitors (epigenetic suppression of MUC1-fs) — preclinical stage, trials anticipated."
        ),
    },

    # -- UMOD — ADTKD-UMOD / Uromodulin / Tamm-Horsfall protein ----------------------
    {
        "gene": "UMOD",
        "alt_name": "Uromodulin / Tamm-Horsfall Protein",
        "protein": (
            "UMOD -- 16p12.3 AD -- Uromodulin-640aa -- "
            "ADTKD-UMOD-Hyperuricemia-Gout-Teens-20s-PATHOGNOMONIC-Screen-UMOD-First -- "
            "Most-Abundant-Urinary-Protein-TH-Tubular-Protective -- "
            "Medullary-Cysts-MRI-Better-Ultrasound-Variable -- "
            "Allopurinol-Febuxostat-Gout-Prevention-NOT-Curative"
        ),
        "locus": "16p12.3",
        "protein_size": "640 aa",
        "inheritance": "AD",
        "age_of_onset": "Gout/hyperuricemia teens–30s; CKD 3rd–5th decade; ESRD 5th–6th decade (range 40–70 yr)",
        "key_biomarker": (
            "Serum uric acid (elevated) — gout crystals on joint aspiration; "
            "renal MRI: medullary cysts (variable presence, MRI better than US); "
            "urine UMOD protein (reduced — mutant UMOD retained in ER, not secreted); "
            "genetic testing: UMOD pathogenic variants in zona pellucida (ZP) domain — cysteine residues most common; "
            "family history: multiple members with gout + CKD without another cause"
        ),
        "pathognomonic": (
            "Gout/hyperuricemia onset in teens or 20s in family with CKD = screen UMOD FIRST; "
            "gout without obvious cause (not alcohol, not diuretics, not obesity) in young adult with family history CKD; "
            "medullary cysts on MRI + hyperuricemia + AD family history = ADTKD-UMOD; "
            "reduced urinary uromodulin (uUMOD <25th percentile) + AD CKD family history"
        ),
        "treatment": (
            "Allopurinol (XOI) — first-line for gout prevention and urate-lowering; start early to prevent gouty arthritis; "
            "febuxostat (XOI) — alternative if allopurinol intolerant; "
            "AVOID: loop/thiazide diuretics (worsen hyperuricemia); low-dose aspirin (raises uric acid); "
            "colchicine for acute gouty attacks; "
            "ACE-i/ARB for CKD/hypertension; "
            "TRANSPLANT: excellent outcomes; uUMOD normalises post-transplant (donor organ secretes normal UMOD); "
            "no targeted therapy for the tubular/fibrosis component"
        ),
        "critical_flags": [
            "UMOD-GOUT-TEENS-PATHOGNOMONIC: hyperuricemia + gout in teens/20s + family history CKD = screen UMOD first",
            "UMOD-GOUT-WITHOUT-DIURETICS: gout in young non-obese without diuretics or alcohol = ADTKD-UMOD until proven otherwise",
            "UMOD-ALLOPURINOL-EARLY: start allopurinol at diagnosis to prevent gouty arthritis; does NOT stop CKD progression",
            "UMOD-MEDULLARY-CYSTS-MRI: cysts may be absent or tiny on US; MRI (coronal T2) more sensitive for medullary cysts",
            "UMOD-URINE-UMOD-LOW: urinary uromodulin reduced (mutant UMOD retained in ER); research test but diagnostically useful",
            "UMOD-ZP-DOMAIN-CYSTEINE: most pathogenic variants affect cysteine residues in ZP domain → disulfide bond disruption → ER retention",
            "UMOD-TRANSPLANT-EXCELLENT: donor kidney secretes normal UMOD; transplant cures ESRD; recurrence not observed",
            "UMOD-AVOID-DIURETICS: loop/thiazide diuretics worsen hyperuricemia significantly — avoid or minimise",
        ],
        "alias": (
            "UMOD (uromodulin / Tamm-Horsfall protein; 640 aa; 16p12.3) encodes uromodulin, "
            "the most abundant protein in normal urine. Secreted exclusively by the thick ascending limb (TAL) of Henle; "
            "functions in urinary tract innate immunity and tubular protection. "
            "ADTKD-UMOD (OMIM #162000): AD inheritance; pathogenic variants (mostly cysteine substitutions in ZP domain) → "
            "misfolded UMOD retained in ER → ER stress → tubular cell death → tubulointerstitial fibrosis. "
            "Clinical hallmark: HYPERURICEMIA + GOUT in teenagers/20s — PATHOGNOMONIC when combined with AD family history of CKD. "
            "Mechanism: mutant UMOD retained in ER + impaired NKCC2 regulation → reduced uric acid excretion → hyperuricemia. "
            "Medullary cysts: present in 40–60% on MRI (often absent/small on ultrasound). "
            "Urinary uromodulin: significantly reduced (<25th percentile) — ER retention prevents secretion. "
            "CKD progression: ESRD 5th–6th decade (range 40–70 yr); rate varies by variant. "
            "Treatment: allopurinol/febuxostat (gout prevention — first-line, start early); "
            "ACE-i/ARB (CKD); avoid diuretics and low-dose aspirin (both raise uric acid); "
            "renal transplant (excellent outcomes; donor UMOD normalises; no recurrence)."
        ),
    },

    # -- REN — ADTKD-REN / Renin -------------------------------------------------------
    {
        "gene": "REN",
        "alt_name": "Renin / ADTKD-REN",
        "protein": (
            "REN -- 1q32.1 AD -- Renin-406aa -- "
            "ADTKD-REN-Childhood-Anemia-Hyperkalemia-Low-BP-Before-ESRD -- "
            "LOF-Low-Renin-Hypoaldosteronism-Tubular-Dysfunction -- "
            "NO-ACE-I-ARB-Hyperkalemia-WORSENS -- "
            "Rarest-ADTKD-Anemia-ESA-Required-Childhood-Clue"
        ),
        "locus": "1q32.1",
        "protein_size": "406 aa",
        "inheritance": "AD",
        "age_of_onset": "Childhood anemia + hyperkalemia (presymptomatic CKD); ESRD 4th–6th decade",
        "key_biomarker": (
            "Plasma renin activity (PRA): very low — LOF of renin → low-renin state; "
            "serum potassium: elevated (hyperkalemia without acidosis in childhood); "
            "hemoglobin: low from childhood (hypo-erythropoietic anemia — intrarenal EPO precursor reduced); "
            "BP: borderline-LOW or normal (not hypertensive) — unusual for CKD; "
            "genetic testing: REN gene sequencing (rare gene — panel may omit); "
            "AD family history of CKD + childhood anemia + no obvious cause = screen REN"
        ),
        "pathognomonic": (
            "Childhood normochromic/normocytic anemia + hyperkalemia + borderline-LOW or normal BP + family history CKD = ADTKD-REN; "
            "childhood onset of anemia requiring ESA WITHOUT typical cause (not iron, not B12, not folate) + AD family history = screen REN; "
            "LOW renin with hyperkalemia + CKD in young patient = REN first; "
            "absence of hypertension despite CKD in family cluster = strongly suggests ADTKD-REN"
        ),
        "treatment": (
            "Erythropoiesis-stimulating agents (ESA): darbepoetin/erythropoietin for anemia (start childhood); "
            "potassium management: low-potassium diet; kayexalate (sodium polystyrene); patiromer; ZS-9 (zirconium cyclosilicate); "
            "AVOID ACE-i/ARB: worsen hyperkalemia significantly in ADTKD-REN (no benefit from RAS modulation — renin already low); "
            "AVOID potassium-sparing diuretics (spironolactone, amiloride, eplerenone); "
            "BP: loop diuretics if fluid overload develops; "
            "renal transplant: excellent outcomes"
        ),
        "critical_flags": [
            "REN-CHILDHOOD-ANEMIA-FIRST-CLUE: anemia in child with AD family history of CKD = screen REN before assuming thalassaemia/iron",
            "REN-HYPERKALEMIA-CHILDHOOD: hyperkalemia without acidosis in childhood + CKD + family history = REN",
            "REN-LOW-BP-NOT-HIGH: normal or borderline LOW BP — unusual for CKD family (DDx from ADTKD-UMOD where HTN common)",
            "REN-NO-ACE-I-ARB: renin already very low; ACE-i/ARB worsen hyperkalemia WITHOUT blood pressure benefit — AVOID",
            "REN-RAREST-ADTKD: rarest ADTKD gene; often missed — not on all NGS panels; screen if AD CKD + childhood anemia",
            "REN-LOW-RENIN-STATE: mechanism is LOF of renin → aldosterone deficiency → hyperkalemia + tubular impairment",
            "REN-ESA-EARLY: start ESA for anemia from diagnosis; delays need for transfusion; improves quality of life",
            "REN-TRANSPLANT-EXCELLENT: transplant cures ESRD; ESA may still be needed early post-transplant",
        ],
        "alias": (
            "REN (renin; 406 aa; 1q32.1) encodes renin, the rate-limiting enzyme of the renin-angiotensin-aldosterone system (RAAS). "
            "ADTKD-REN (OMIM #179820): AD inheritance; LOF mutations in signal peptide or catalytic domain → "
            "misfolded renin accumulates in ER of juxtaglomerular cells → JGA cell destruction → "
            "progressive tubulointerstitial fibrosis + very low renin activity → low aldosterone → hyperkalemia. "
            "Clinical triad: (1) childhood-onset normochromic anemia (impaired intrarenal EPO precursor maturation), "
            "(2) childhood hyperkalemia (low-renin hypoaldosteronism), "
            "(3) borderline-LOW or normal BP (unusual for CKD — absence of hypertension is a clue). "
            "ADTKD-REN is the rarest ADTKD gene (only a few large pedigrees described); "
            "often not included on standard renal NGS panels — specifically request if clinical picture fits. "
            "CRITICAL: AVOID ACE inhibitors/ARBs — renin is already very low; these drugs have no BP benefit "
            "and significantly worsen hyperkalemia. AVOID potassium-sparing diuretics. "
            "Treatment: ESA for anemia (start in childhood); patiromer/ZS-9 for hyperkalemia; renal transplant (excellent outcomes)."
        ),
    },

    # -- HNF1B — RCAD / Renal Cysts And Diabetes / MODY5 ----------------------------
    {
        "gene": "HNF1B",
        "alt_name": "HNF1β / RCAD / MODY5",
        "protein": (
            "HNF1B -- 17q12 AD -- HNF1beta-557aa -- "
            "RCAD-Renal-Cysts-AND-Diabetes-MODY5-Multiorgan -- "
            "MLPA-MANDATORY-50pct-Large-Deletions-Missed-NGS-Sanger -- "
            "Pancreatic-Hypoplasia-Exocrine-Insufficiency-PERT-Required -- "
            "Sulfonylurea-INEFFECTIVE-Insulin-Required-HNF1B-DM"
        ),
        "locus": "17q12",
        "protein_size": "557 aa",
        "inheritance": "AD",
        "age_of_onset": "Renal cysts often prenatal/neonatal; DM5 onset typically 20s–30s; multiorgan progression",
        "key_biomarker": (
            "Fetal/neonatal ultrasound: renal cysts (often bilateral, hyperechogenic kidneys) + structural anomalies; "
            "fasting glucose/HbA1c: MODY5 DM (usually DM precedes or coincides with renal diagnosis); "
            "serum magnesium: hypomagnesaemia (renal tubular wasting — PATHOGNOMONIC when combined with DM + renal cysts); "
            "LFTs: abnormal in ~50% (no clinical liver disease); "
            "MLPA: detects 17q12 deletions (50% HNF1B cases — missed by sequencing); "
            "pancreatic imaging (MRI): pancreatic hypoplasia/atrophy; "
            "uterine MRI/USS: bicornuate uterus ~30%"
        ),
        "pathognomonic": (
            "Bilateral renal cysts in fetus/neonate + prenatal or early-onset DM (MODY5 pattern, insulin-dependent) = HNF1B; "
            "renal cysts + MODY5 + hypomagnesaemia TRIAD = HNF1B near-diagnostic; "
            "hyperechogenic kidneys at fetal USS + DM in parent = screen HNF1B; "
            "DM requiring insulin from onset with negative autoantibodies + renal cysts + FHx = HNF1B (not type 1 DM)"
        ),
        "treatment": (
            "Diabetes: INSULIN required (sulfonylurea INEFFECTIVE in HNF1B — HNF1β is required for SUR1/Kir6.2 expression); "
            "DO NOT try sulfonylurea (unlike HNF1A/MODY3 where SU works 98%); "
            "pancreatic exocrine insufficiency: PERT (pancreatic enzyme replacement therapy — creon/pancreaze); "
            "hypomagnesaemia: oral Mg supplements (magnesium oxide/glycinate); IV Mg for severe; "
            "renal: ACE-i/ARB for proteinuria; avoid nephrotoxins; "
            "MLPA negative → sequence HNF1B; sequence negative + typical phenotype → MLPA (order BOTH); "
            "uterine anomalies: fertility counselling; gynaecology referral; "
            "multiorgan surveillance: annual renal USS, HbA1c, Mg, LFTs, pancreatic function"
        ),
        "critical_flags": [
            "HNF1B-MLPA-MANDATORY: 50% HNF1B cases = 17q12 deletion MISSED by NGS/Sanger — MLPA must be ordered alongside sequencing",
            "HNF1B-SULFONYLUREA-INEFFECTIVE: HNF1β drives SUR1/Kir6.2 — SU cannot work; INSULIN mandatory from diagnosis",
            "HNF1B-RENAL-CYSTS-PRENATAL: renal cysts often antenatal on USS — DM parent + fetal renal cysts = screen HNF1B",
            "HNF1B-HYPOMAGNESAEMIA-TRIAD: renal cysts + DM5 + low Mg = HNF1B; Mg wasting = renal tubular effect",
            "HNF1B-PANCREATIC-ATROPHY: pancreatic hypoplasia → exocrine insufficiency → PERT mandatory if steatorrhoea/malabsorption",
            "HNF1B-BICORNUATE-UTERUS-30pct: uterine anomalies in women → fertility counselling mandatory",
            "HNF1B-LFT-ABNORMAL-50pct: elevated LFTs without clinical liver disease — monitor but usually benign",
            "HNF1B-GOUT-HYPERURICEMIA: like ADTKD-UMOD, HNF1B also associates with hyperuricemia/gout — but DM + cysts + Mg distinguish",
        ],
        "alias": (
            "HNF1B (hepatocyte nuclear factor 1-beta; 557 aa; 17q12) encodes HNF1β, a homeodomain transcription factor "
            "regulating development of kidney, pancreas, liver, and reproductive tract. "
            "ADTKD-HNF1B / RCAD (Renal Cysts And Diabetes; OMIM #137920): AD; "
            "50% caused by large 17q12 chromosomal deletions MISSED by standard sequencing — MLPA mandatory. "
            "Clinical pentad: (1) bilateral renal cysts (antenatal to early childhood), "
            "(2) MODY5 diabetes (insulin-dependent — sulfonylurea INEFFECTIVE as HNF1β is required for K-ATP channel expression), "
            "(3) hypomagnesaemia (renal tubular Mg wasting), "
            "(4) pancreatic hypoplasia ± exocrine insufficiency (PERT required if symptomatic), "
            "(5) uterine anomalies (bicornuate 30%). "
            "Additional: abnormal LFTs (50%, usually benign), hyperuricemia/gout, epididymal cysts (males). "
            "Renal: collecting system abnalomies, dysplastic kidneys, ESRD variable (30–50% by 5th decade). "
            "Diabetes: insulin from diagnosis; SU trial is futile and should be avoided. "
            "Diagnostic approach: sequence HNF1B AND order MLPA simultaneously (50% deletions; 50% point mutations). "
            "Management: insulin, PERT (exocrine insufficiency), Mg supplements, ACE-i/ARB, multiorgan surveillance annually."
        ),
    },

    # -- DNAJB11 — ADTKD-DNAJB11 / Newest ADTKD gene --------------------------------
    {
        "gene": "DNAJB11",
        "alt_name": "DnaJ Hsp40 B11 / ADTKD-DNAJB11",
        "protein": (
            "DNAJB11 -- 3q27.3 AD -- DNAJB11-354aa -- "
            "ADTKD-DNAJB11-Newest-ADTKD-Gene-2018-Atypical-Polycystic-Mimics-PKD1-PKD2 -- "
            "ER-Co-Chaperone-DNAJ-Domain-Assists-BiP-GRP78 -- "
            "PKD1-PKD2-Negative-Atypical-Polycystic-Sequence-DNAJB11-Next -- "
            "Slow-Progression-ESRD-6th-7th-Decade-Cysts-Variable"
        ),
        "locus": "3q27.3",
        "protein_size": "354 aa",
        "inheritance": "AD",
        "age_of_onset": "Adulthood; ESRD 6th–7th decade; atypical polycystic pattern with variable cyst burden",
        "key_biomarker": (
            "Renal MRI/CT: cysts (variable — from few to many; tubular + spherical pattern); "
            "renal biopsy: tubulointerstitial fibrosis + cystic dilations; "
            "genetic testing: DNAJB11 sequencing — pathogenic missense in DNAJ domain most common; "
            "PKD1/PKD2 negative atypical polycystic + AD family history = next step: ADTKD panel including DNAJB11, MUC1, UMOD, REN; "
            "no serum/urine specific biomarker yet identified"
        ),
        "pathognomonic": (
            "Atypical ADPKD-like phenotype (bilateral cysts) + PKD1/PKD2 NEGATIVE sequencing + "
            "AD family history of CKD = sequence DNAJB11 + ADTKD genes; "
            "DNAJB11 is the newest ADTKD gene (discovered 2018) — not yet on all NGS panels; "
            "polycystic kidney pattern more variable than classic ADPKD (fewer, irregular, tubular component); "
            "ER stress signature on renal biopsy (research)"
        ),
        "treatment": (
            "No specific therapy; supportive: "
            "ACE-i/ARB for proteinuria/hypertension; "
            "avoid NSAIDs + nephrotoxins + contrast agents; "
            "tolvaptan NOT proven effective (unlike PKD1/2 — cyst mechanism differs); "
            "renal transplant: standard care for ESRD (no recurrence post-transplant expected); "
            "genetic counselling: AD 50% transmission; clinical phenotype variable even within families"
        ),
        "critical_flags": [
            "DNAJB11-NEWEST-ADTKD-2018: discovered 2018; not on all NGS panels — specifically request if PKD1/2 negative atypical polycystic",
            "DNAJB11-MIMICS-PKD1-PKD2: atypical polycystic pattern; PKD1/PKD2 negative + AD FHx = DNAJB11 + ADTKD genes next",
            "DNAJB11-ER-COCHAPERONE: assists BiP/GRP78 in ER protein quality control; LOF → ER stress → tubular cell death",
            "DNAJB11-CYSTS-VARIABLE: fewer and more irregular cysts than classic ADPKD; tubular component on biopsy",
            "DNAJB11-TOLVAPTAN-NOT-PROVEN: tolvaptan targets V2R-cAMP pathway (ADPKD mechanism); ADTKD-DNAJB11 mechanism different",
            "DNAJB11-SLOW-ESRD: ESRD 6th-7th decade; slower than PKD1; comparable to PKD2/other ADTKD",
            "DNAJB11-COMPLETE-ADTKD-PANEL: if PKD1/2 negative: order ADTKD panel (MUC1-VNTR assay, UMOD, REN, HNF1B/MLPA, DNAJB11)",
            "DNAJB11-NO-RECURRENCE: post-transplant donor kidney provides normal DNAJB11 — disease does not recur",
        ],
        "alias": (
            "DNAJB11 (DnaJ heat shock protein family member B11; 354 aa; 3q27.3) encodes an ER-resident Hsp40 co-chaperone "
            "that assists BiP/GRP78 in protein folding and quality control. "
            "ADTKD-DNAJB11 (OMIM #617570): the newest ADTKD gene, described in 2018 (Senum et al., Kidney Int 2019). "
            "Pathogenic missense variants in the DNAJ domain impair co-chaperone function → "
            "ER stress in renal tubular cells → tubulointerstitial fibrosis with cystic changes. "
            "Clinical: AD inheritance; bilateral renal cysts (variable — from few irregular cysts to many spherical cysts) "
            "mimicking ADPKD; PKD1/PKD2 sequencing negative; ESRD 6th–7th decade. "
            "CRITICAL diagnostic algorithm: atypical polycystic kidneys + AD family history + PKD1/PKD2 negative → "
            "order comprehensive ADTKD panel: MUC1-VNTR assay (long-read), UMOD sequencing, REN sequencing, "
            "HNF1B sequencing + MLPA, DNAJB11 sequencing. "
            "Treatment: supportive (ACE-i/ARB; avoid nephrotoxins); "
            "tolvaptan not applicable (V2R-cAMP pathway ≠ mechanism here); "
            "renal transplant for ESRD (excellent outcomes; no disease recurrence in donor kidney)."
        ),
    },
]


# ---------- patient cohort generator -----------------------------------------

def _make_cohort(gene_data: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    cohort = []
    for i in range(n):
        age = rng.randint(18, 78)
        severity = rng.choices(
            ["mild", "moderate", "severe"],
            weights=[35, 40, 25],
            k=1,
        )[0]
        # gene-specific adjustments
        if gene == "PKD1":
            esrd_age = rng.randint(42, 70)
            feature = rng.choice(["bilateral cysts", "hypertension", "flank pain", "hematuria", "ICA screen", "liver cysts"])
            therapy = "tolvaptan" if severity in ("moderate", "severe") and age >= 18 else "ACE-i/ARB"
        elif gene == "PKD2":
            esrd_age = rng.randint(60, 88)
            feature = rng.choice(["bilateral cysts (milder)", "hypertension", "hematuria", "liver cysts (fewer)", "ICA screen"])
            therapy = "tolvaptan (if 1C-1E)" if severity == "severe" else "ACE-i/ARB"
        elif gene == "PKHD1":
            esrd_age = rng.randint(5, 35)
            feature = rng.choice(["echogenic kidneys", "congenital hepatic fibrosis", "portal hypertension", "cholangitis", "Caroli syndrome"])
            therapy = rng.choice(["ACE-i", "beta-blocker (varices)", "UDCA + antibiotics", "combined Tx"])
        elif gene == "MUC1":
            esrd_age = rng.randint(45, 75)
            feature = rng.choice(["tubulointerstitial fibrosis", "no cysts on imaging", "NGS-negative (VNTR missed)", "family ESRD"])
            therapy = "supportive (ACE-i; avoid nephrotoxins)"
        elif gene == "UMOD":
            esrd_age = rng.randint(40, 68)
            feature = rng.choice(["gout age 20s", "hyperuricemia", "medullary cysts (MRI)", "low urinary UMOD", "family gout+CKD"])
            therapy = "allopurinol + ACE-i/ARB"
        elif gene == "REN":
            esrd_age = rng.randint(35, 65)
            feature = rng.choice(["childhood anemia", "hyperkalemia without acidosis", "low renin", "normal/low BP + CKD family"])
            therapy = "ESA + K-binders (NO ACE-i)"
        elif gene == "HNF1B":
            esrd_age = rng.randint(30, 60)
            feature = rng.choice(["prenatal renal cysts", "MODY5 (insulin-required)", "hypomagnesaemia", "pancreatic atrophy", "bicornuate uterus"])
            therapy = "insulin + PERT + Mg supplements"
        elif gene == "DNAJB11":
            esrd_age = rng.randint(50, 75)
            feature = rng.choice(["atypical polycystic (PKD1/2 neg)", "tubulointerstitial cysts (biopsy)", "AD family ESRD", "slow progression"])
            therapy = "supportive (ACE-i; avoid nephrotoxins)"
        else:
            esrd_age = rng.randint(40, 70)
            feature = "CKD"
            therapy = "supportive"

        cohort.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "age": age,
            "gene": gene,
            "severity": severity,
            "key_feature": feature,
            "projected_esrd_age": esrd_age,
            "current_therapy": therapy,
        })
    return cohort


# ---------- API endpoint functions -------------------------------------------

def overview() -> dict:
    total = 0
    severe_count = 0
    avg_age_sum = 0
    gene_summary = []
    for idx, g in enumerate(RENAL_CYSTIC_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        total += len(cohort)
        severe_count += sum(1 for p in cohort if p["severity"] == "severe")
        avg_age_sum += sum(p["age"] for p in cohort)
        gene_summary.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "n_patients": len(cohort),
        })
    avg_age = round(avg_age_sum / total, 1)
    return {
        "atlas": "Hereditary-Renal-Cystic-Disease-Atlas",
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(RENAL_CYSTIC_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(RENAL_CYSTIC_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "PKD1-VS-PKD2-ESRD-20yr: PKD1 ESRD median 54 yr vs PKD2 74 yr — 20-yr difference; same imaging, critical prognosis counselling",
            "PKHD1-CHF-ALWAYS: congenital hepatic fibrosis in ALL ARPKD — liver disease may dominate over renal in surviving children",
            "MUC1-STANDARD-NGS-MISSES: cytosine insertion in VNTR missed by all short-read NGS — long-read/VNTR-assay required",
            "UMOD-GOUT-TEENS-PATHOGNOMONIC: gout + hyperuricemia in teens/20s + AD family CKD = screen UMOD first",
            "REN-NO-ACE-I-ARB: low renin state — ACE-i/ARB worsen hyperkalemia without BP benefit; AVOID",
            "HNF1B-MLPA-MANDATORY: 50% HNF1B = 17q12 deletion; missed by sequencing — MLPA must be ordered; SU INEFFECTIVE in MODY5",
            "DNAJB11-PKD1-PKD2-NEGATIVE: atypical polycystic + PKD1/2 negative = sequence DNAJB11 + full ADTKD panel",
            "PKHD1-POTTER-SEQUENCE: oligohydramnios → pulmonary hypoplasia → 30% neonatal mortality; bilateral nephrectomy + transplant for lung growth",
            "TOLVAPTAN-RAPID-PROGRESSORS-ONLY: FDA2018 for ADPKD Mayo 1C-1E; hepatotoxicity monitoring mandatory; NOT for slow progressors",
            "REN-CHILDHOOD-ANEMIA-CLUE: anemia in child + AD family history CKD + no obvious cause = REN before haematology workup",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(RENAL_CYSTIC_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(RENAL_CYSTIC_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Renal-Cystic-Disease-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "alt_name": g.get("alt_name", ""),
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in RENAL_CYSTIC_GENES
        ],
        "glossary": {
            "ADPKD (Autosomal Dominant PKD)": "Most common monogenic renal disease (1:400–1,000); PKD1 85% (ESRD 54 yr) + PKD2 15% (ESRD 74 yr); bilateral cysts; tolvaptan FDA2018 for rapid progressors (Mayo 1C-1E)",
            "ARPKD (Autosomal Recessive PKD)": "PKHD1 mutations; 1:20,000 births; collecting duct ectasia (NOT balloon cysts); CHF universally present; Potter sequence in severe neonatal form",
            "ADTKD (Autosomal Dominant TubuloInterstitial Kidney Disease)": "Four genes: MUC1, UMOD, REN, HNF1B (RCAD), DNAJB11; adult ESRD; AD family history; tubulointerstitial nephritis without prominent cysts (except HNF1B and DNAJB11)",
            "Tolvaptan (Jynarque)": "V2R antagonist; FDA2018 for ADPKD; reduces TKV growth 49% + eGFR decline 26%; rapid progressors only (Mayo 1C-1E); hepatotoxic — monthly LFT monitoring mandatory",
            "Total Kidney Volume (TKV)": "Measured by MRI; Mayo classification 1A-1E for ADPKD prognosis; 1C-1E = rapid progressors → tolvaptan eligible; 1A-1B = slow progressors → no tolvaptan",
            "Mayo Imaging Classification": "1A (slowest) to 1E (fastest); based on height-adjusted TKV vs age; 1C-1E qualifies for tolvaptan; predicts ESRD timing",
            "Congenital Hepatic Fibrosis (CHF)": "Portal fibrosis with bile duct proliferation; universally present in ARPKD (PKHD1); portal hypertension → varices; Caroli disease association; biopsy diagnostic",
            "Potter Sequence": "Oligohydramnios → pulmonary hypoplasia → Potter facies + limb deformities; seen in severe ARPKD (biallelic truncating PKHD1); 30% neonatal mortality from respiratory failure",
            "ADTKD-MUC1 (VNTR frameshift)": "Cytosine insertion in GC-rich VNTR → MUC1-fs toxic protein → ER stress → tubular death; STANDARD NGS MISSES — long-read/VNTR assay required; tubulointerstitial nephritis without cysts",
            "ADTKD-UMOD": "Uromodulin mutations; hyperuricemia + gout in teens/20s PATHOGNOMONIC; most abundant urinary protein; cysteine mutations in ZP domain → ER retention; allopurinol first-line",
            "ADTKD-REN": "Renin LOF → low-renin hypoaldosteronism; childhood anemia + hyperkalemia + low/normal BP; RAREST ADTKD; ACE-i/ARB CONTRAINDICATED (worsen hyperkalemia); ESA for anemia",
            "RCAD (Renal Cysts And Diabetes)": "HNF1B disease; renal cysts + MODY5 (insulin-required; SU INEFFECTIVE) + hypomagnesaemia + pancreatic hypoplasia; 50% large 17q12 deletions → MLPA mandatory",
            "ADTKD-DNAJB11": "Newest ADTKD gene (2018); ER co-chaperone; atypical polycystic pattern mimicking PKD1/2; PKD1/2 negative → sequence DNAJB11; ESRD 6th-7th decade; no tolvaptan indication",
            "Fibrocystin / Polyductin (PKHD1)": "Receptor-like ciliary protein in collecting ducts and intrahepatic bile ducts; largest gene body in ARPKD; 4074 aa; biallelic LOF → collecting duct ectasia + CHF; 67 exons; compound heterozygous most common",
            "Intracranial Aneurysm (ICA) in PKD": "4–12% ADPKD vs 1–2% general population; screen with MRA if first-degree relative with rupture or high-risk occupation; subarachnoid haemorrhage is leading cause of ADPKD death in young patients",
            "Uromodulin (Tamm-Horsfall protein)": "Most abundant urinary protein; secreted by TAL of Henle; innate immunity + tubular protection; urinary uromodulin REDUCED in ADTKD-UMOD (ER retention of mutant protein); uUMOD <25th centile + AD CKD = screen UMOD",
            "MODY5 / HNF1B diabetes": "Insulin-dependent from onset; sulfonylurea INEFFECTIVE (HNF1β required for K-ATP channel expression in β-cells); autoantibodies negative; distinguish from type 1 DM by family history + renal cysts + hypoMg",
            "Polycystin complex (PC1-PC2)": "PC1 (PKD1, 4304 aa) + PC2 (PKD2/TRPP2, 968 aa) form mechanosensory heterotetrameric complex in primary cilia; Ca²⁺ influx regulates mTOR/cAMP; LOF → cAMP accumulation → tubular cell proliferation → cysts",
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-RENAL-CYSTIC-DISEASE-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (UMOD — gout gene) ===")
    bd = breakdown()
    umod = next(g for g in bd["genes"] if g["gene"] == "UMOD")
    print(json.dumps(umod, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
