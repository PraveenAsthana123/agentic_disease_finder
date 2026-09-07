#!/usr/bin/env python3
"""Hereditary-Vascular-Malformation-Atlas — Complete 8-Gene Atlas (Hereditary Vascular Malformations)
ENG     (Endoglin; 658 aa; 9q34.11; AD;
         Hereditary Hemorrhagic Telangiectasia type 1 (HHT1 / Osler-Weber-Rendu);
         Pulmonary AVM most common (60–80%) — highest paradoxical embolism / stroke risk;
         Epistaxis UNIVERSAL first manifestation — begin in 1st decade;
         seed SEED_BASE+0) .
ACVRL1  (Activin A Receptor Like Type 1 / ALK1; 503 aa; 12q13.13; AD;
         Hereditary Hemorrhagic Telangiectasia type 2 (HHT2);
         Hepatic AVM dominant (>80%) — liver shunting → high-output cardiac failure;
         GI bleeding most prominent; milder pulmonary AVM than HHT1;
         seed SEED_BASE+1) .
SMAD4   (SMAD Family Member 4; 552 aa; 18q21.2; AD;
         Juvenile Polyposis / HHT combined syndrome (JPHT);
         ONLY gene causing BOTH HHT and Juvenile Polyposis — colon cancer mandatory surveillance;
         ALL SMAD4 patients need annual colonoscopy from age 15 regardless of HHT severity;
         seed SEED_BASE+2) .
KRIT1   (Krev Interaction Trapped 1 / CCM1; 736 aa; 7q21.2; AD;
         Cerebral Cavernous Malformation type 1 (CCM1 / OMIM #116860);
         Most common hereditary CCM; p.Q455X Mexican-American founder — 40% of CCM1 alleles;
         Multiple cavernomas on MRI; seizures most common presentation (40–70%);
         seed SEED_BASE+3) .
CCM2    (Malcavernin / CCM2; 444 aa; 7p13; AD;
         Cerebral Cavernous Malformation type 2 (CCM2 / OMIM #603284);
         Less aggressive than CCM1 and CCM3; scaffold protein bridging KRIT1 and PDCD10;
         de novo mutations more common than CCM1; fewer total lesions than CCM3;
         seed SEED_BASE+4) .
PDCD10  (Programmed Cell Death 10 / CCM3; 212 aa; 3q26.1; AD;
         Cerebral Cavernous Malformation type 3 (CCM3 / OMIM #603285) — MOST AGGRESSIVE CCM;
         Youngest age of onset (infantile/childhood); most cavernomas; fastest progression;
         Spinal cord cavernomas DISTINCTIVE; meningioma association;
         seed SEED_BASE+5) .
RASA1   (RAS p21 Protein Activator 1 / p120-RasGAP; 1047 aa; 5q14.3; AD;
         Capillary Malformation-AVM syndrome type 1 (CM-AVM1 / OMIM #608354);
         Multifocal capillary malformations (port-wine stain-like) + fast-flow AVMs;
         Parkes Weber syndrome = limb hypertrophy + cutaneous CM + soft-tissue AVMs;
         seed SEED_BASE+6) .
TEK     (TEK Receptor Tyrosine Kinase / TIE2; 1124 aa; 9p21.2; AD + Somatic;
         Multiple Cutaneous and Mucosal Venous Malformations (VMCM / OMIM #600195);
         Most common somatic vascular malformation gene;
         Germline = multiple blue compressible venous malformations; Somatic = sporadic VM;
         Rapamycin (mTOR inhibition) emerging treatment — reduces lesion size and pain;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1966–1973)
"""

import random

SEED_BASE = 1966

VASCULAR_MALFORMATION_GENES = [
    # -- ENG — Endoglin / HHT1 --------------------------------------------------
    {
        "gene": "ENG",
        "alt_name": "ENG (HHT1 / Hereditary Hemorrhagic Telangiectasia type 1 / Endoglin)",
        "protein": (
            "ENG -- 9q34.11 AD -- ENG-658aa -- "
            "HHT1-Most-Common-HHT-Pulmonary-AVM-60-80pct -- "
            "Epistaxis-UNIVERSAL-First-Decade-Almost-100pct -- "
            "Paradoxical-Embolism-Stroke-Brain-Abscess-Highest-Risk-All-HHT -- "
            "Bevacizumab-Anti-VEGF-Emerging-Nasal-IV-Epistaxis -- "
            "TGF-Beta-Co-Receptor-Endothelial-Cells"
        ),
        "locus": "9q34.11",
        "protein_size": "658 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Epistaxis: onset in first decade in ~50%; virtually universal by 3rd decade; "
            "progressive frequency and severity with age; "
            "Pulmonary AVM: congenital; may present acutely (haemoptysis, haemothorax, paradoxical embolism); "
            "Stroke/brain abscess: first presentation in some patients (paradoxical embolism via PAVM); "
            "Telangiectasias: lips/tongue/hands — visible from 2nd–3rd decade; "
            "Cerebral AVM: 10% of HHT1; onset any age; haemorrhage risk 0.5–1%/yr"
        ),
        "key_biomarker": (
            "CT pulmonary angiography (CTPA): PAVMs — feeding artery ≥3mm = treatment threshold; "
            "bubble echocardiography: right-to-left shunt screen (bubbles appear in left heart <3 beats); "
            "SpO2: resting desaturation in large PAVM; orthodeoxia (SpO2 drops on standing); "
            "brain MRI with contrast: cerebral AVM + telangiectasias in 10% HHT1; "
            "molecular: ENG pathogenic variant (nonsense/frameshift most common — haploinsufficiency); "
            "ferritin / Hgb: iron deficiency anaemia from chronic epistaxis — very common; "
            "liver Doppler US: hepatic AVM assessment (milder in HHT1 than HHT2)"
        ),
        "pathognomonic": (
            "Curacao criteria (3/4 = definite HHT): (1) spontaneous/recurrent epistaxis; "
            "(2) mucocutaneous telangiectasias (lips, oral mucosa, fingers, nose); "
            "(3) visceral AVM (pulmonary, hepatic, cerebral, spinal); "
            "(4) first-degree relative with HHT; "
            "HHT1 (ENG) vs HHT2 (ACVRL1): HHT1 = more PAVM + cerebral AVM; HHT2 = more hepatic AVM; "
            "DISTINGUISH sporadic epistaxis: no family history; no telangiectasias; no visceral AVM; "
            "DISTINGUISH JPHT (SMAD4): SMAD4 = HHT phenotype PLUS juvenile colonic polyps — colonoscopy mandatory"
        ),
        "treatment": (
            "Epistaxis: humidification + moisturisers; laser cautery; bevacizumab intranasal spray (Level B); "
            "IV bevacizumab for severe epistaxis/anaemia (Level B); thalidomide low-dose (Level C); "
            "septodermoplasty/Young's procedure for refractory; "
            "PAVM: embolisation if feeding artery ≥3mm (interventional radiology); "
            "AVOID: ASA/NSAIDs/anticoagulants unless mandatory; "
            "Dental/surgical: antibiotic prophylaxis for all PAVM procedures (brain abscess risk); "
            "Cerebral AVM: treat if accessible + symptomatic (neurosurgery/radiosurgery/embolisation); "
            "Genetic counselling: AD; 50% offspring risk; HHT-expert centre mandatory; "
            "Iron replacement: oral or IV for epistaxis-related anaemia; transfusion threshold Hgb <70 g/L"
        ),
        "critical_flags": [
            "ENG-PAVM-PARADOXICAL-EMBOLISM: PAVMs bypass pulmonary filter — bacteria, air, clot reach systemic circulation directly; brain abscess + stroke are sentinel events; ALL HHT1 patients need CTPA at diagnosis; embolise all PAVMs with feeding artery ≥3mm regardless of symptoms",
            "ENG-ANTIBIOTIC-PROPHYLAXIS-MANDATORY: any procedure in PAVM patient without antibiotic cover = brain abscess risk; IV amoxicillin 2g pre-dental/surgical; dentists must be informed; 'silent' PAVMs post-embolisation can still harbour shunts",
            "ENG-BEVACIZUMAB-ANTI-VEGF: IV bevacizumab (5mg/kg q2wk × 6 cycles) is Level B evidence for severe epistaxis refractory to local measures; reduces epistaxis score >50% in 70% of patients; also reduces transfusion requirements; teratogenic — contraception mandatory",
            "ENG-CEREBRAL-AVM-10PCT: cerebral AVMs in 10% HHT1 (vs 1% HHT2); baseline brain MRI mandatory at diagnosis; haemorrhage risk 0.5–1%/yr per AVM; treatment decision requires multidisciplinary neurovascular team",
            "ENG-PREGNANCY-HIGH-RISK: PAVM grows during pregnancy (oestrogen effect on endoglin); new PAVMs can form; SpO2 monitoring mandatory; CTPA before conception if possible; haemoptysis or desaturation in pregnancy = emergency",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- ACVRL1 — ALK1 / HHT2 ---------------------------------------------------
    {
        "gene": "ACVRL1",
        "alt_name": "ACVRL1 / ALK1 (HHT2 / Hereditary Hemorrhagic Telangiectasia type 2)",
        "protein": (
            "ACVRL1 -- 12q13.13 AD -- ACVRL1-503aa -- "
            "HHT2-Hepatic-AVM-Dominant-Over-80pct -- "
            "GI-Bleeding-Most-Prominent-Feature -- "
            "High-Output-Cardiac-Failure-Hepatic-Shunting -- "
            "Milder-Pulmonary-AVM-Than-HHT1 -- "
            "TGF-Beta-Receptor-ALK1-Endothelial-BMP9-BMP10-Signalling"
        ),
        "locus": "12q13.13",
        "protein_size": "503 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Epistaxis: later onset than HHT1; milder; adult presentation typical; "
            "GI bleeding (telangiectasias of stomach/small bowel): 3rd–5th decade; iron deficiency dominant; "
            "Hepatic AVM: clinically overt 4th–6th decade; high-output cardiac failure, biliary ischaemia, portal hypertension; "
            "PAVM: less common than HHT1 (~30%); feeding artery smaller; paradoxical embolism risk lower but real; "
            "Telangiectasias: later and less prominent than HHT1"
        ),
        "key_biomarker": (
            "Liver Doppler ultrasound + contrast-enhanced CT: hepatic AVM — arteriovenous, arterioportal, portovenous shunts; "
            "echocardiography: high cardiac output (>8 L/min), elevated LA pressure, pulmonary hypertension; "
            "bubble echo: right-to-left shunt (less common than HHT1); "
            "GI endoscopy: gastric/duodenal/small bowel telangiectasias with active bleeding; "
            "ferritin + Hgb: profound iron deficiency from GI blood loss; "
            "molecular: ACVRL1 pathogenic variant; haploinsufficiency mechanism (like ENG); "
            "NT-proBNP: elevated if high-output failure from hepatic AVM"
        ),
        "pathognomonic": (
            "HHT2 vs HHT1: HHT2 more hepatic AVM + GI bleeding + milder epistaxis; HHT1 more PAVM + cerebral AVM; "
            "Hepatic AVM high-output failure: warm peripheries + bounding pulse + LV enlargement + elevated CO = hepatic AVM until proven; "
            "Curacao criteria apply: ≥3/4 = definite HHT (same as HHT1); "
            "DISTINGUISH from portal hypertension: hepatic AVM = normal LFTs usually; high CO; Doppler shows shunting; "
            "DISTINGUISH from cirrhosis: cirrhosis = stigmata; low albumin; thrombocytopenia; HHT2 = normal liver synthetic function"
        ),
        "treatment": (
            "Epistaxis: same as HHT1 (humidification, laser, bevacizumab); "
            "GI bleeding: argon plasma coagulation (endoscopic); IV bevacizumab reduces GI blood loss (Level B); "
            "Hepatic AVM: bevacizumab IV — first-line for high-output cardiac failure (Level B); "
            "beta-blockade for rate control if AF from high-output; "
            "diuretics for fluid overload; "
            "liver transplantation for refractory hepatic AVM with failure (Level A — curative); "
            "AVOID hepatic artery embolisation: high complication rate (biliary ischaemia) in hepatic HHT; "
            "PAVM: embolise if feeding artery ≥3mm; antibiotic prophylaxis mandatory; "
            "Iron: IV iron preferred (high GI loss; oral iron poorly tolerated)"
        ),
        "critical_flags": [
            "ACVRL1-HEPATIC-EMBOLISATION-CI: hepatic artery embolisation is CONTRAINDICATED in HHT hepatic AVMs (unlike non-HHT liver AVMs); it causes biliary ischaemia and hepatic necrosis; bevacizumab + liver transplant are the correct escalation pathway",
            "ACVRL1-HIGH-OUTPUT-FAILURE: hepatic arteriovenous shunting → CO > 8 L/min → dilated cardiomyopathy pattern; any HHT patient with high-output heart failure = look for hepatic AVM first; do NOT treat as primary DCM before ruling out HHT shunting",
            "ACVRL1-GI-SURVEILLANCE-MANDATORY: gastric/duodenal/small bowel telangiectasias cause chronic occult GI loss; annual Hgb/ferritin; upper GI endoscopy if anaemia; video capsule for small bowel; argon plasma coagulation most effective hemostatic endoscopic technique",
            "ACVRL1-BMP9-BMP10-PATHWAY: ACVRL1 (ALK1) is the receptor for BMP9 and BMP10 on endothelial cells; loss of function → abnormal arteriovenous specification; anti-BMP9 antibodies investigational; understanding pathway critical for emerging targeted therapies (luspatercept)",
            "ACVRL1-PAVM-NOT-ZERO: 30% of HHT2 have PAVMs (vs 70% HHT1); CTPA at diagnosis in all HHT2; do NOT assume hepatic-dominant HHT2 has no PAVM; paradoxical embolism risk requires the same antibiotic prophylaxis as HHT1",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- SMAD4 — SMAD Family Member 4 / JPHT ------------------------------------
    {
        "gene": "SMAD4",
        "alt_name": "SMAD4 (JPHT / Juvenile Polyposis-HHT Combined Syndrome / SMAD4)",
        "protein": (
            "SMAD4 -- 18q21.2 AD -- SMAD4-552aa -- "
            "JPHT-ONLY-Gene-Causing-HHT-PLUS-Juvenile-Polyposis -- "
            "Colon-Cancer-Risk-Mandatory-Annual-Colonoscopy-from-Age-15 -- "
            "HHT-Phenotype-ALL-Curacao-Features -- "
            "Gastric-Polyps-Gastric-Cancer-Risk-Distinctive -- "
            "TGF-Beta-SMAD-Signalling-Tumour-Suppressor"
        ),
        "locus": "18q21.2",
        "protein_size": "552 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Juvenile polyposis: GI polyps in childhood; rectal bleeding + anaemia + protein-losing enteropathy; "
            "HHT manifestations: epistaxis + telangiectasias + visceral AVM — same as HHT1/HHT2; "
            "Colorectal cancer: lifetime risk 39–68%; onset possible from 3rd decade if polyps uncontrolled; "
            "Gastric polyps: gastric cancer risk — less well defined but real; upper GI surveillance essential; "
            "Cardiac: aortic dilation reported (distinct from ENG/ACVRL1 HHT)"
        ),
        "key_biomarker": (
            "Colonoscopy: multiple juvenile polyps (hamartomatous with lamina propria expansion); "
            "upper GI endoscopy: gastric polyps — carpet-like in severe cases; "
            "molecular: SMAD4 pathogenic variant — confirms JPHT (vs isolated JP = BMPR1A; vs isolated HHT = ENG/ACVRL1); "
            "Hgb/albumin/protein: hypoproteinaemia from protein-losing enteropathy in heavy polyposis; "
            "Bubble echo + CTPA: PAVM assessment — same as other HHT; "
            "liver Doppler: hepatic AVM (intermediate frequency between HHT1 and HHT2)"
        ),
        "pathognomonic": (
            "HHT + juvenile polyps = SMAD4 until proven; "
            "Isolated HHT = ENG or ACVRL1; isolated JP = BMPR1A (~20%) or SMAD4; "
            "SMAD4 JPHT: ALWAYS perform full Curacao assessment + colonoscopy; "
            "DISTINGUISH from Peutz-Jeghers syndrome: PJS = STK11; perioral/mucocutaneous melanotic spots; hamartomas with smooth muscle core; no HHT; "
            "DISTINGUISH from BRRS (PTEN): PTEN = macrocephaly; thyroid hamartomas; no visceral AVM; "
            "Aortic root dilation in SMAD4 JPHT: screen with echocardiography — mechanism TGF-β pathway overlap with Marfan"
        ),
        "treatment": (
            "Colonoscopy: annual from age 15 (or at diagnosis if later); polypectomy of all accessible polyps; "
            "Colectomy: if polyp burden uncontrollable or dysplasia identified; "
            "Upper GI endoscopy: every 2–3 years from age 25; gastric polypectomy; "
            "HHT: same management as ENG/ACVRL1 (epistaxis, PAVM embolisation, bevacizumab); "
            "Aortic echo: baseline at diagnosis + repeat if dilated; "
            "Genetic counselling: SMAD4 AD; 50% risk offspring; all first-degree relatives must be offered testing; "
            "AVOID NSAIDs (promote polyp growth); aspirin controversial — potential polyposis benefit but GI risk; "
            "Multidisciplinary: gastroenterology + HHT centre co-management mandatory"
        ),
        "critical_flags": [
            "SMAD4-COLONOSCOPY-MANDATORY: ALL SMAD4 patients — colonoscopy annual from age 15; colorectal cancer lifetime risk 39–68%; polyp clearance + surveillance is life-saving; do NOT manage only the HHT component and miss the polyposis",
            "SMAD4-JPHT-NOT-JUST-HHT: diagnosing HHT in a SMAD4 patient without recognising the juvenile polyposis component is a critical error; every HHT patient should have SMAD4 tested and if positive, immediate GI surveillance initiated",
            "SMAD4-GASTRIC-CANCER: gastric polyps can undergo adenomatous change; upper GI endoscopy every 2–3 years from age 25; gastric cancer risk is real and under-recognised in JPHT",
            "SMAD4-AORTIC-DILATION: aortic root dilation occurs in SMAD4 JPHT (TGF-β pathway); baseline echocardiogram at diagnosis; annual surveillance if >40mm; surgical threshold lower than Marfan (discuss at 45mm); distinct from ENG/ACVRL1",
            "SMAD4-PROTEIN-LOSING-ENTEROPATHY: heavy gastric polyposis → protein loss → hypoalbuminaemia, oedema, growth failure in children; albumin monitoring + nutritional support; consider proton pump inhibitor to reduce mucosal inflammation and protein loss",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- KRIT1 — Cerebral Cavernous Malformation 1 ------------------------------
    {
        "gene": "KRIT1",
        "alt_name": "KRIT1 / CCM1 (Cerebral Cavernous Malformation type 1 — Most Common Hereditary CCM)",
        "protein": (
            "KRIT1 -- 7q21.2 AD -- KRIT1-736aa -- "
            "CCM1-Most-Common-Hereditary-CCM -- "
            "p.Q455X-Mexican-American-Founder-40pct-CCM1-Alleles -- "
            "Multiple-Cavernomas-MRI-Gradient-Echo-T2-Star -- "
            "Seizures-40-70pct-Most-Common-Presentation -- "
            "RAP1-RAL-MAPK-Signalling-Junctional-Integrity"
        ),
        "locus": "7q21.2",
        "protein_size": "736 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Seizures: most common (40–70%); onset 2nd–4th decade; focal or secondarily generalised; "
            "Haemorrhage: 0.5–1%/lesion/yr in general; higher in brainstem cavernomas; "
            "Focal deficits: depend on lesion location (brainstem = cranial nerve palsies, ataxia); "
            "Headache: common but non-specific; "
            "Silent radiological: up to 50% of carriers have only incidentally found lesions; "
            "Paediatric: childhood onset in p.Q455X Mexican-American founder — more aggressive"
        ),
        "key_biomarker": (
            "MRI brain (gradient echo T2* or SWI): CCMs — 'popcorn' appearance with haemosiderin rim; "
            "SWI (susceptibility-weighted imaging): most sensitive — detects microhaemorrhages and small lesions; "
            "contrast MRI: CCMs do NOT enhance (no blood-brain barrier disruption at rest); "
            "lesion count: familial CCM = multiple lesions (>5 on SWI typical); sporadic CCM = 1–2 lesions; "
            "molecular: KRIT1 pathogenic variant; p.Q455X founder (c.1363C>T) — Mexican-American; "
            "EEG: interictal epileptiform discharges if seizures; video-EEG for surgical evaluation"
        ),
        "pathognomonic": (
            "Multiple CCMs on SWI MRI + AD family history = hereditary CCM until proven; "
            "p.Q455X (KRIT1) in Mexican-American = highest frequency familial CCM; "
            "DISTINGUISH from AVM: AVM — fast-flow; no haemosiderin rim; angio shows nidus; CCM — no-flow; haemosiderin; no nidus on angio; "
            "DISTINGUISH from CCM2 (CCM2/malcavernin) and CCM3 (PDCD10): only by genetic testing; "
            "CCM3 (PDCD10) most aggressive — youngest onset, most lesions, spinal cavernomas — DISTINGUISH urgently; "
            "Sporadic CCM (somatic 2-hit in CCM1/2/3): single lesion; no family history; no germline variant"
        ),
        "treatment": (
            "Seizures: antiseizure medication (levetiracetam, lacosamide); surgical resection if drug-resistant + accessible; "
            "Surgical resection: indicated for symptomatic accessible lesion (haemorrhage + deficit); "
            "Brainstem CCM: surgery only if repeated haemorrhage + life-threatening (high-risk approach); "
            "Radiosurgery: NOT recommended for CCM (increases radiation-induced bleeding; Level III evidence only); "
            "Sirolimus (mTOR inhibitor): clinical trials — reduces lesion haemorrhage; "
            "Propranolol: Phase 2 trials (CARE-CCM); may stabilise CCM haemorrhage; "
            "Surveillance MRI: SWI every 3–5 years in asymptomatic familial CCM; annually if prior haemorrhage; "
            "AVOID: anticoagulation unless mandatory (increases haemorrhage risk); contact sports with brainstem CCM; "
            "Genetic counselling: AD; 50% offspring risk; first-degree relatives offered MRI + testing"
        ),
        "critical_flags": [
            "KRIT1-SWI-MRI-MANDATORY: gradient echo T2* or SWI is essential — conventional T1/T2 misses up to 50% of CCMs; all suspected hereditary CCM and all first-degree relatives require SWI MRI; 'normal MRI' without SWI does NOT exclude CCM",
            "KRIT1-Q455X-MEXICAN-AMERICAN-FOUNDER: p.Q455X (c.1363C>T) is a common founder mutation in Mexican-American families — up to 1 in 200 in affected communities; seizure onset in paediatric age; test FIRST before full panel; highly penetrant",
            "KRIT1-RADIOSURGERY-NOT-RECOMMENDED: stereotactic radiosurgery is NOT recommended for CCM (unlike AVM); radiation increases haemorrhage risk and causes radiation necrosis without clear benefit; Level III evidence only; avoid unless extraordinary circumstances",
            "KRIT1-BRAINSTEM-CCM-HIGH-RISK: brainstem CCMs have higher haemorrhage rate and devastating neurological consequences; manage conservatively unless life-threatening haemorrhage; refer to high-volume CCM surgical centre; discuss risks explicitly",
            "KRIT1-ANTICOAGULATION-CAUTION: anticoagulants significantly increase CCM haemorrhage risk; if anticoagulation unavoidable (AF, DVT, prosthetic valve), use lowest effective dose and monitor MRI; shared decision-making essential; document rationale",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- CCM2 — Malcavernin / Cerebral Cavernous Malformation 2 -----------------
    {
        "gene": "CCM2",
        "alt_name": "CCM2 / Malcavernin (Cerebral Cavernous Malformation type 2)",
        "protein": (
            "CCM2 -- 7p13 AD -- CCM2-444aa -- "
            "CCM2-Less-Aggressive-Than-CCM1-and-CCM3 -- "
            "Scaffold-Protein-KRIT1-PDCD10-Bridging -- "
            "De-Novo-Mutations-More-Common-CCM2-Than-CCM1 -- "
            "Fewer-Total-Lesions-Than-CCM3 -- "
            "Intermediate-Phenotype-CCM-Spectrum"
        ),
        "locus": "7p13",
        "protein_size": "444 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Seizures: 30–60% of symptomatic CCM2; onset 2nd–5th decade; "
            "Haemorrhage: lower rate than CCM3; similar to CCM1; "
            "Fewer lesions than CCM3 (typically <10 vs >20+ in CCM3); "
            "De novo mutations: ~20% CCM2 are de novo (vs ~5% CCM1); family history absent in de novo; "
            "Paediatric presentation: less common than CCM3 but reported"
        ),
        "key_biomarker": (
            "MRI SWI/gradient echo T2*: multiple CCMs — popcorn lesions with haemosiderin rim; "
            "lesion count lower than CCM3 on average; "
            "molecular: CCM2 pathogenic variant — all exons sequenced (no founder variant); "
            "NGS panel: concurrent CCM1 (KRIT1) and CCM3 (PDCD10) testing to assign genotype; "
            "EEG if seizures; "
            "ophthalmological exam: retinal cavernous malformations in CCM2 (less common than CCM1)"
        ),
        "pathognomonic": (
            "Multiple CCMs + AD family history: hereditary CCM; CCM1/CCM2/CCM3 differentiation requires molecular testing; "
            "CCM2 intermediate phenotype: less aggressive than CCM3 (most aggressive) but more lesion burden than sporadic CCM; "
            "De novo CCM2: no family history; single patient; thorough CCM1/2/3 panel testing essential — sporadic appearance ≠ sporadic genetics; "
            "DISTINGUISH from CCM3 (PDCD10): CCM3 = more lesions + younger onset + spinal cord + meningioma; "
            "DISTINGUISH sporadic single CCM: no germline variant on panel; somatic 2-hit mechanism"
        ),
        "treatment": (
            "Seizures: antiseizure medication (levetiracetam first-line); surgical resection if drug-resistant + accessible; "
            "Surgical resection: symptomatic accessible cavernoma with haemorrhage or drug-resistant epilepsy; "
            "Surveillance MRI SWI: every 3–5 years in asymptomatic; annually if prior haemorrhage; "
            "Emerging: sirolimus + propranolol trials apply equally to CCM2; "
            "Genetic counselling: AD; 50% offspring risk; all 1st-degree relatives offered SWI MRI; "
            "AVOID: anticoagulation without compelling indication; radiosurgery; "
            "Retinal CCM: fundoscopy + OCT at diagnosis; treat if vitreous haemorrhage (photocoagulation)"
        ),
        "critical_flags": [
            "CCM2-DE-NOVO-MUTATIONS: ~20% of CCM2 are de novo germline mutations; patients without family history of CCM can still have hereditary CCM2; ALL multiple CCM patients (familial appearance OR de novo) require full CCM1/2/3 panel sequencing",
            "CCM2-SCAFFOLD-ROLE: CCM2 protein (malcavernin) is the scaffold bridging KRIT1 (CCM1) and PDCD10 (CCM3) in the CCM signalling complex; loss of CCM2 destabilises the entire complex; explains phenotypic overlap between all three CCM types",
            "CCM2-RETINAL-CAVERNOMAS: retinal cavernous malformations occur in ~5% of hereditary CCM (all three types); fundoscopy at diagnosis; vitreous haemorrhage from retinal CCM requires urgent ophthalmological assessment",
            "CCM2-PANEL-TESTING-NOT-SINGLE-GENE: clinical and radiological features cannot reliably distinguish CCM1, CCM2, and CCM3; ALWAYS use multi-gene panel covering all three; single-gene sequential testing misses the correct diagnosis in 30–40% of cases",
            "CCM2-PREGNANCY-CONSIDERATION: new and enlarging CCMs reported during pregnancy; enhanced surveillance in pregnant CCM patients; avoid anticoagulation in first trimester if possible; neuro-anaesthesia review mandatory for labour planning",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- PDCD10 — Programmed Cell Death 10 / CCM3 / Most Aggressive -------------
    {
        "gene": "PDCD10",
        "alt_name": "PDCD10 / CCM3 (Cerebral Cavernous Malformation type 3 — MOST AGGRESSIVE)",
        "protein": (
            "PDCD10 -- 3q26.1 AD -- PDCD10-212aa -- "
            "CCM3-MOST-AGGRESSIVE-CCM-Youngest-Onset -- "
            "Spinal-Cord-Cavernomas-DISTINCTIVE-Not-in-CCM1-CCM2 -- "
            "Meningioma-Association-UNIQUE -- "
            "Most-Lesions-Most-Rapid-Progression -- "
            "PDCD10-Apoptosis-Signalling-STK24-STK25-STRIPAK-Complex"
        ),
        "locus": "3q26.1",
        "protein_size": "212 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Youngest onset of all hereditary CCM types: infantile/childhood (1st decade) in many; "
            "Most lesions: typically >20–50 lesions on SWI (vs <10 in CCM1/2); "
            "Haemorrhage: more frequent than CCM1/2; progressive disability; "
            "Spinal cord CCM: DISTINCTIVE for CCM3 (rare in CCM1/2); paraplegia risk; "
            "Meningioma: co-occurrence in ~10% of CCM3 families (unique among CCM types); "
            "Progression: fastest of all CCM types — new lesions appear on serial MRI"
        ),
        "key_biomarker": (
            "MRI brain SWI: numerous CCMs ('miliary' pattern); >20 lesions suggests CCM3 vs CCM1/2; "
            "spinal cord MRI (spine MRI): spinal cavernomas — PATHOGNOMONIC for CCM3; "
            "gadolinium MRI brain: look for co-existing meningioma (T1 enhancing dural-based lesion); "
            "molecular: PDCD10 pathogenic variant; no common founder; full gene sequencing; "
            "neuro-ophthalmology: optic nerve cavernoma in CCM3 (rare); "
            "surveillance MRI: annual (more aggressive than CCM1/2 — more dynamic disease)"
        ),
        "pathognomonic": (
            "Multiple CCMs (>10 on SWI) + paediatric onset + spinal cord cavernomas = CCM3 (PDCD10) until proven; "
            "Spinal cord CCM: almost exclusively CCM3; presence STRONGLY suggests PDCD10 germline variant; "
            "Meningioma + CCM: combination highly suggestive of CCM3; "
            "DISTINGUISH from CCM1 (KRIT1): CCM1 fewer lesions; adult onset more typical; no spinal; no meningioma; "
            "DISTINGUISH from CCM2: CCM2 intermediate lesion burden; no spinal CCM; no meningioma; "
            "Miliary CCM pattern (dozens of tiny lesions): most common in CCM3"
        ),
        "treatment": (
            "Seizures: aggressive ASM therapy; surgical resection for accessible drug-resistant lesions; "
            "Haemorrhage: surgical resection for symptomatic accessible lesions; "
            "Brainstem CCM3: highest risk; conservative unless repeated haemorrhage; high-volume centre; "
            "Spinal CCM3: surgical if progressive deficit or recurrent haemorrhage; "
            "Meningioma: manage independently (surgery/radiosurgery per standard meningioma guidelines); "
            "Surveillance: annual MRI brain SWI + spinal MRI (not every 3–5yr as in CCM1/2 — more aggressive); "
            "Emerging: sirolimus + propranolol trials (same as CCM1/2); PDCD10-STRIPAK pathway inhibitors preclinical; "
            "Genetic counselling: AD; 50% offspring; URGENT — paediatric presentation requires immediate family cascade testing"
        ),
        "critical_flags": [
            "PDCD10-MOST-AGGRESSIVE-CCM: CCM3 (PDCD10) has youngest onset, most lesions, fastest progression, and unique extra-cerebral features (spinal CCM + meningioma); if a CCM patient has paediatric onset or >20 lesions on SWI, presume CCM3 and test PDCD10 FIRST",
            "PDCD10-SPINAL-CORD-PATHOGNOMONIC: spinal cord cavernomas are almost exclusive to CCM3 in hereditary CCM; ALL CCM3 patients need baseline spinal MRI; new back pain or limb symptoms = emergency spinal MRI to rule out haemorrhage",
            "PDCD10-MENINGIOMA-SCREEN: co-existing meningioma occurs in ~10% CCM3 families; gadolinium brain MRI at diagnosis; incidental meningioma may not require treatment but changes surveillance approach",
            "PDCD10-PAEDIATRIC-ONSET: infantile/childhood seizures or haemorrhage in a patient with multiple brain lesions = urgent PDCD10 panel testing; paediatric CCM3 is progressive and requires more intensive surveillance than adult-onset CCM",
            "PDCD10-ANNUAL-MRI-NOT-TRIENNIAL: unlike CCM1/2 where 3–5 year surveillance is standard, CCM3 requires ANNUAL MRI SWI due to rapid lesion evolution; quarterly MRI if prior haemorrhage or new neurological symptoms",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- RASA1 — RAS p21 Protein Activator 1 / CM-AVM1 -------------------------
    {
        "gene": "RASA1",
        "alt_name": "RASA1 / p120-RasGAP (CM-AVM1 / Capillary Malformation-AVM Syndrome / Parkes Weber)",
        "protein": (
            "RASA1 -- 5q14.3 AD -- RASA1-1047aa -- "
            "CM-AVM1-Multifocal-Capillary-Malformations-Port-Wine-Stain-Like -- "
            "Fast-Flow-AVMs-Brain-Spine-Skin-Viscera -- "
            "Parkes-Weber-Syndrome-Limb-Hypertrophy-CM-Soft-Tissue-AVM -- "
            "High-Flow-Shunting-Cardiac-Complications -- "
            "RAS-GAP-Ras-GTPase-Activating-Protein-MAPK-Pathway"
        ),
        "locus": "5q14.3",
        "protein_size": "1047 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Capillary malformations: present at birth (cutaneous); multifocal (>3 lesions = pathognomonic); "
            "Parkes Weber syndrome: neonatal/infancy — limb enlargement + warm cutaneous CM; "
            "AVM: brain AVMs 20% CM-AVM1; spinal AVMs; "
            "Haemorrhage from brain AVM: onset any age; higher annual risk than sporadic AVM; "
            "High-output cardiac failure: infancy/childhood from high-flow visceral AVMs; "
            "Adult: new CMs may appear with age; AVM progression slower after childhood"
        ),
        "key_biomarker": (
            "Skin examination: multiple (>3) small pinkish-red CMs with pale halo — characteristic of CM-AVM; "
            "MRI brain + MR angiography: brain AVM — fast-flow nidus; flow voids; angiography for grading; "
            "spinal MRI: spinal AVM/AVF — flow voids; cord signal change; "
            "Doppler US limb: Parkes Weber — arteriovenous shunting in affected limb; "
            "echocardiography: high cardiac output if multiple AVMs with shunting; "
            "molecular: RASA1 pathogenic variant — loss of RasGAP activity; "
            "MRI body: visceral AVMs in liver/lung if suspected"
        ),
        "pathognomonic": (
            "Multifocal CMs (>3) with pale halo + any fast-flow AVM = CM-AVM1 (RASA1) until proven; "
            "Parkes Weber syndrome: one limb enlarged at birth + warm + CM + Doppler shunting = RASA1 until proven; "
            "DISTINGUISH from isolated port-wine stain (Sturge-Weber): unilateral; no halo; GNAQ somatic; no AVM; "
            "DISTINGUISH from HHT: HHT = epistaxis + telangiectasias + visceral AVM; no cutaneous CM halos; "
            "DISTINGUISH from CM-AVM2 (EPHB4): identical phenotype; differentiation only by molecular"
        ),
        "treatment": (
            "Cutaneous CM: pulsed dye laser (578nm) for cosmesis; multiple sessions required; "
            "Brain AVM: treat if accessible + unruptured high-risk or ruptured (surgery/embolisation/radiosurgery); "
            "Spinal AVM: endovascular embolisation; neurosurgery; "
            "Parkes Weber limb: compressive stockings; avoid trauma; orthotic footwear; "
            "Limb-length discrepancy: epiphysiodesis if >2cm predicted at skeletal maturity; "
            "High-output cardiac failure: diuretics; beta-blockade; "
            "AVOID: sclerotherapy of CM in CM-AVM (risk of AVM shunting augmentation); "
            "Genetic counselling: AD; 50% offspring risk; screening all 1st-degree relatives for CMs; "
            "Annual surveillance MRI brain if brain AVM; annual Doppler limb if Parkes Weber"
        ),
        "critical_flags": [
            "RASA1-MULTIFOCAL-CM-NOT-SPORADIC: >3 small CMs with pale halo = hereditary CM-AVM syndrome (RASA1 or EPHB4) NOT sporadic port-wine stain; full RASA1/EPHB4 molecular testing mandatory; screen for brain/spinal AVM in all confirmed CM-AVM patients",
            "RASA1-BRAIN-AVM-20PCT: 20% of CM-AVM1 patients have brain AVMs; higher haemorrhage rate than sporadic brain AVMs due to underlying RasGAP dysregulation; baseline brain MRI + MRA at diagnosis in ALL RASA1 carriers",
            "RASA1-PARKES-WEBER-CARDIAC: Parkes Weber syndrome (RASA1) with high-flow limb shunting can cause high-output heart failure in infancy; echocardiogram at diagnosis; cardiological management if CO elevated; limb AVMs grow with age requiring staged embolisation",
            "RASA1-SCLEROTHERAPY-RISK: sclerotherapy of CMs in CM-AVM is hazardous — risks augmenting underlying AVM flow and causing paradoxical embolism; consult RASA1 vascular malformation expert before any CM treatment",
            "RASA1-EPHB4-DDX: CM-AVM2 (EPHB4 mutations) is clinically identical to CM-AVM1 (RASA1); differentiation ONLY by molecular testing; EPHB4 is associated with RASA1-negative CM-AVM families; always test both genes",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- TEK — TIE2 / Venous Malformation (Germline + Somatic) ------------------
    {
        "gene": "TEK",
        "alt_name": "TEK / TIE2 (Multiple Cutaneous and Mucosal Venous Malformations / VMCM + Sporadic VM)",
        "protein": (
            "TEK -- 9p21.2 AD+Somatic -- TEK-1124aa -- "
            "VMCM-Germline-Multiple-Blue-Compressible-Cutaneous-Mucosal-Venous-Malformations -- "
            "Sporadic-VM-Somatic-TEK-Mutations-Most-Common-Somatic-Vascular-Malformation-Gene -- "
            "Rapamycin-mTOR-Inhibition-Emerging-Treatment-Reduces-Lesion-Size-Pain -- "
            "Chronic-Thrombosis-Phleboliths-Painful -- "
            "TIE2-Angiopoietin-Receptor-Endothelial-AKT-PI3K-mTOR-Signalling"
        ),
        "locus": "9p21.2",
        "protein_size": "1124 aa",
        "inheritance": "AD (germline VMCM) + Somatic (sporadic VM)",
        "age_of_onset": (
            "Germline VMCM: present at birth or early childhood; "
            "multiple blue soft compressible lesions on skin/mucosa; "
            "grow slowly with age and pregnancy; "
            "Somatic sporadic VM: typically single lesion; any age; most common in childhood/young adult; "
            "Pain: phleboliths (dystrophic calcification from thrombosis) — common from 2nd decade; "
            "Complications: swelling, phlebothrombosis, Kasabach-Merritt (localized intravascular coagulation), rarely PE"
        ),
        "key_biomarker": (
            "MRI (T2 with fat suppression): hyperintense venous malformation; fluid-fluid levels from thrombosis; phleboliths (signal void); "
            "D-dimer: elevated in large VMs — reflects chronic localised intravascular coagulation; "
            "plain X-ray/CT: phleboliths (characteristic round calcifications within VM); "
            "Doppler US: slow-flow venous malformation (contrast to AVM fast-flow); "
            "molecular: TEK germline variant (VMCM) vs somatic TEK hotspot (R849W most common sporadic) — tissue biopsy for somatic; "
            "blood sampling: localised consumptive coagulopathy — low fibrinogen, elevated D-dimer in large VMs"
        ),
        "pathognomonic": (
            "Blue compressible soft lesion + expansion with Valsalva/dependency + slow-flow on Doppler = venous malformation; "
            "Phleboliths on plain X-ray within a soft lesion = VM (pathognomonic for chronic slow-flow malformation); "
            "VMCM (germline TEK): multiple lesions; family history; lips/tongue/hands/feet; AD; "
            "Sporadic VM (somatic TEK): single lesion; no family history; often deep/intramuscular; "
            "DISTINGUISH from AVM: AVM = fast-flow; pulsatile; warm; no Valsalva compressibility; "
            "DISTINGUISH from lymphatic malformation: LM = non-compressible; fluid aspirate is chylous/serous not bloody; MRI = macrocystic or microcystic"
        ),
        "treatment": (
            "Sclerotherapy: first-line for symptomatic VM (ethanol or sodium tetradecyl sulphate); "
            "Rapamycin (sirolimus) oral: Level B evidence — reduces VM size + pain + D-dimer; "
            "start 0.8mg/m2 BID; target trough 10–15 ng/mL; "
            "Anticoagulation: LMWH peri-procedure + pre-operatively; long-term in recurrent thrombophlebitis; "
            "Surgery: resection if accessible + focal + refractory; "
            "Phleboliths + pain: NSAIDS short-term; LMWH reduces new thrombosis and pain; "
            "Compression: elastic compression for extremity VMs; "
            "AVOID: aspiration without treatment plan (rapid refill); "
            "Genetic counselling: germline VMCM — AD; 50% offspring; somatic VM = not heritable; "
            "Pregnancy: VMs enlarge; plan management before conception; avoid hormonal therapy (oestrogen expands VM)"
        ),
        "critical_flags": [
            "TEK-RAPAMYCIN-EMERGING: oral sirolimus (rapamycin) is now Level B evidence for venous malformations; reduces lesion size, pain, D-dimer, and bleeding episodes; do NOT withhold from patients with symptomatic extensive or progressive VMs pending surgery or sclerotherapy",
            "TEK-LOCALIZED-INTRAVASCULAR-COAGULATION: large VMs cause chronic LIC (consumptive coagulopathy — low fibrinogen, elevated D-dimer, low platelets); pre-surgical LMWH normalises coagulation; surgeons/anaesthetists must be informed; PE risk is real in large untreated VMs",
            "TEK-PHLEBOLITHS-DIAGNOSTIC: phleboliths (round calcified foci within a soft tissue lesion on plain X-ray or CT) are pathognomonic of chronic venous malformation; their presence confirms low-flow malformation and guides sclerotherapy planning",
            "TEK-SOMATIC-VS-GERMLINE: sporadic VM (single lesion, no family history) = somatic TEK mutation (not heritable); germline VMCM (multiple lesions, AD family history) = inherited TEK variant; somatic testing requires lesion biopsy not blood NGS; management is same but genetic counselling differs",
            "TEK-PREGNANCY-EXPANSION: venous malformations dramatically enlarge during pregnancy due to oestrogen-mediated TEK receptor upregulation; plan intervention before conception; LMWH throughout pregnancy for large VMs; avoid OCP/HRT (same mechanism)",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry: dict) -> list:
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    pts = []
    for i in range(40):
        age_dx = rng.randint(0, 50)
        htt_genes = ("ENG", "ACVRL1", "SMAD4")
        ccm_genes = ("KRIT1", "CCM2", "PDCD10")
        is_htt = gene in htt_genes
        is_ccm = gene in ccm_genes
        is_rasa1 = gene == "RASA1"
        is_tek = gene == "TEK"

        # HHT-specific phenotype
        pavm = rng.random() < (0.7 if gene == "ENG" else 0.35 if gene == "ACVRL1" else 0.4) if is_htt else False
        hepatic_avm = rng.random() < (0.4 if gene == "ENG" else 0.82 if gene == "ACVRL1" else 0.6) if is_htt else False
        gi_polyps = gene == "SMAD4" and rng.random() < 0.95
        epistaxis = is_htt and rng.random() < (0.95 if gene == "ENG" else 0.85 if gene == "ACVRL1" else 0.80)

        # CCM-specific phenotype
        seizures = rng.random() < (0.55 if gene == "KRIT1" else 0.45 if gene == "CCM2" else 0.70) if is_ccm else False
        haemorrhage = rng.random() < (0.25 if gene == "KRIT1" else 0.20 if gene == "CCM2" else 0.40) if is_ccm else False
        spinal_ccm = gene == "PDCD10" and rng.random() < 0.28
        meningioma = gene == "PDCD10" and rng.random() < 0.10
        lesion_count = int(rng.gauss(8, 4)) if gene == "KRIT1" else int(rng.gauss(6, 3)) if gene == "CCM2" else int(rng.gauss(22, 8)) if gene == "PDCD10" else 0

        # RASA1-specific
        brain_avm = rng.random() < 0.22 if is_rasa1 else False
        parkes_weber = rng.random() < 0.35 if is_rasa1 else False

        # TEK-specific
        phleboliths = rng.random() < 0.55 if is_tek else False
        on_rapamycin = rng.random() < 0.30 if is_tek else False

        pts.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "age_at_diagnosis": max(0, age_dx),
            "sex": rng.choice(["M", "F"]),
            # HHT fields
            "pavm": pavm,
            "hepatic_avm": hepatic_avm,
            "gi_polyps": gi_polyps,
            "epistaxis": epistaxis,
            # CCM fields
            "seizures": seizures,
            "haemorrhage": haemorrhage,
            "spinal_ccm": spinal_ccm,
            "meningioma": meningioma,
            "ccm_lesion_count": max(0, lesion_count),
            # RASA1 fields
            "brain_avm": brain_avm,
            "parkes_weber": parkes_weber,
            # TEK fields
            "phleboliths": phleboliths,
            "on_rapamycin": on_rapamycin,
        })
    return pts


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in VASCULAR_MALFORMATION_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]
    pavm_n = sum(1 for p in all_pts if p["pavm"])
    hepatic_n = sum(1 for p in all_pts if p["hepatic_avm"])
    seizure_n = sum(1 for p in all_pts if p["seizures"])
    haem_n = sum(1 for p in all_pts if p["haemorrhage"])
    polyp_n = sum(1 for p in all_pts if p["gi_polyps"])
    gene_counts = {g["gene"]: len(_generate_cohort(g)) for g in VASCULAR_MALFORMATION_GENES}
    return {
        "atlas": "Hereditary-Vascular-Malformation-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Vascular Malformation Atlas",
        "genes": [g["gene"] for g in VASCULAR_MALFORMATION_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE+7}",
        "pavm_patients": pavm_n,
        "hepatic_avm_patients": hepatic_n,
        "ccm_seizure_patients": seizure_n,
        "ccm_haemorrhage_patients": haem_n,
        "gi_polyps_patients": polyp_n,
        "gene_patient_counts": gene_counts,
        "pathway": (
            "HHT (ENG/ACVRL1/SMAD4): TGF-β / BMP9-BMP10 / ALK1-Endoglin signalling — "
            "haploinsufficiency → AVM formation at arteriovenous junctions. "
            "CCM (KRIT1/CCM2/PDCD10): CCM signalling complex → junctional integrity maintenance — "
            "loss → cavernoma formation. "
            "RASA1: RasGAP → Ras pathway — loss → AVM + CM via MAPK hyperactivation. "
            "TEK (TIE2): Angiopoietin receptor → PI3K/AKT/mTOR → endothelial survival and remodelling — "
            "GOF somatic/germline → venous ectasia."
        ),
        "key_clinical_insight": (
            "HHT1 (ENG): pulmonary AVM highest risk — antibiotic prophylaxis mandatory (brain abscess). "
            "HHT2 (ACVRL1): hepatic AVM — AVOID hepatic embolisation (fatal biliary ischaemia). "
            "SMAD4: ONLY gene causing HHT + Juvenile Polyposis — annual colonoscopy mandatory from age 15. "
            "CCM3 (PDCD10): most aggressive CCM — youngest onset, spinal CCM, meningioma — annual MRI. "
            "TEK: rapamycin is Level B evidence — treat symptomatic venous malformations."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in VASCULAR_MALFORMATION_GENES:
        cohort = _generate_cohort(gene_entry)
        pavm_pct = round(100 * sum(1 for p in cohort if p["pavm"]) / len(cohort))
        hepatic_pct = round(100 * sum(1 for p in cohort if p["hepatic_avm"]) / len(cohort))
        seizure_pct = round(100 * sum(1 for p in cohort if p["seizures"]) / len(cohort))
        haem_pct = round(100 * sum(1 for p in cohort if p["haemorrhage"]) / len(cohort))
        polyp_pct = round(100 * sum(1 for p in cohort if p["gi_polyps"]) / len(cohort))
        result[gene_entry["gene"]] = {
            "gene": gene_entry["gene"],
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "pavm_pct": pavm_pct,
            "hepatic_avm_pct": hepatic_pct,
            "seizure_pct": seizure_pct,
            "haemorrhage_pct": haem_pct,
            "gi_polyps_pct": polyp_pct,
            "age_of_onset": gene_entry["age_of_onset"],
            "key_biomarker": gene_entry["key_biomarker"],
            "pathognomonic": gene_entry["pathognomonic"],
            "treatment": gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed": gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Vascular-Malformation-Atlas",
        "gene_definitions": {
            g["gene"]: {
                "protein": g["protein"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
            }
            for g in VASCULAR_MALFORMATION_GENES
        },
        "glossary": {
            "Hereditary Hemorrhagic Telangiectasia (HHT)": "Autosomal dominant vascular dysplasia; telangiectasias + AVMs in multiple organs; caused by ENG, ACVRL1, or SMAD4; Curacao criteria for diagnosis",
            "PAVM (Pulmonary Arteriovenous Malformation)": "Direct arteriovenous connection in pulmonary circulation; bypasses pulmonary capillary filter; paradoxical embolism → stroke or brain abscess; treat if feeding artery ≥3mm",
            "Curacao criteria": "Diagnostic criteria for HHT: ≥3/4 = definite (1. epistaxis, 2. mucocutaneous telangiectasias, 3. visceral AVM, 4. first-degree relative with HHT); 2/4 = suspected",
            "Cerebral Cavernous Malformation (CCM)": "Low-flow vascular lesion of CNS; 'popcorn' appearance on MRI SWI/T2*; haemosiderin-lined sinusoidal spaces; seizures, haemorrhage, focal deficits",
            "SWI (Susceptibility-Weighted Imaging)": "Most sensitive MRI sequence for CCM detection; detects haemosiderin from microhaemorrhages; essential for CCM surveillance; conventional T2 misses up to 50% of lesions",
            "CCM signalling complex": "KRIT1 (CCM1) + CCM2 (malcavernin) + PDCD10 (CCM3) form a ternary complex; maintains endothelial junctional integrity; loss of any component destabilises all three",
            "CM-AVM syndrome (Capillary Malformation-AVM)": "RASA1 (CM-AVM1) or EPHB4 (CM-AVM2); multifocal CMs + fast-flow AVMs; Parkes Weber syndrome = limb hypertrophy variant",
            "Parkes Weber syndrome": "RASA1-associated; unilateral limb hypertrophy + warm cutaneous CM + soft-tissue AVMs; high-flow shunting → cardiac overload",
            "Venous Malformation (VM)": "Low-flow vascular malformation; blue compressible lesion; expands with Valsalva; phleboliths (dystrophic calcification) pathognomonic; TEK somatic mutations in sporadic VM",
            "phleboliths": "Dystrophic calcifications within venous malformations; round/oval on X-ray; result from chronic thrombosis; pathognomonic for low-flow VM; visible on plain X-ray",
            "Rapamycin / Sirolimus": "mTOR inhibitor; Level B evidence for venous malformations (TEK); reduces lesion size, pain, D-dimer; target trough 10–15 ng/mL; reduces PI3K/AKT/mTOR signalling activated by TIE2 GOF mutations",
            "Bevacizumab": "Anti-VEGF monoclonal antibody; Level B evidence for HHT epistaxis and hepatic AVM; inhibits VEGF-driven AVM formation; teratogenic — contraception mandatory; reduces epistaxis frequency >50%",
            "Paradoxical embolism": "Right-to-left shunting via PAVM bypasses pulmonary filter; any systemic embolism (air, clot, bacteria) reaches systemic circulation; brain abscess + stroke principal risks; antibiotic prophylaxis mandatory for all procedures",
            "Juvenile Polyposis": "Hamartomatous GI polyposis; SMAD4 (JPHT) or BMPR1A; colorectal cancer risk 39–68%; annual colonoscopy from age 15; distinguished from Peutz-Jeghers (STK11) and FAP (APC) by histology and gene",
            "Localised Intravascular Coagulation (LIC)": "Chronic consumptive coagulopathy within large VMs; low fibrinogen, elevated D-dimer, thrombocytopenia; pre-surgical LMWH mandatory; PE risk if untreated",
        },
        "clinical_pearls": [
            "HHT1 (ENG): ALL patients need CTPA at diagnosis — 60–80% have PAVM; embolise feeding artery ≥3mm; antibiotic prophylaxis for all PAVM procedures (brain abscess)",
            "HHT2 (ACVRL1): hepatic artery embolisation is ABSOLUTELY CONTRAINDICATED — use bevacizumab IV + liver transplant pathway for high-output failure",
            "SMAD4: every SMAD4 HHT patient MUST have annual colonoscopy from age 15 — colon cancer risk 39–68%; GI surveillance saves lives beyond the HHT management",
            "CCM3 (PDCD10): youngest onset, most lesions, spinal CCM, meningioma — annual MRI (not triennial); suspect CCM3 if >20 lesions or paediatric CCM; URGENT cascade testing",
            "Sporadic VM + phleboliths on X-ray: somatic TEK mutation; sclerotherapy first-line; offer sirolimus if extensive or painful; test for germline TEK if multiple lesions or family history",
            "KRIT1 Q455X: Mexican-American population; 1 in 200 frequency; test FIRST before full panel; paediatric onset; SWI MRI mandatory for all carriers",
            "Radiosurgery in CCM: NOT recommended — increases haemorrhage risk; radiosurgery is for AVM, not CCM",
            "Bubble echo: screens for right-to-left shunting in HHT; bubbles appearing in left heart <3 heartbeats = shunt; quantify if positive; CTPA if shunt confirmed",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = breakdown()
    first_gene = list(bd.keys())[0]
    print(json.dumps(bd[first_gene], indent=2, default=str)[:1500])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:4], indent=2))
