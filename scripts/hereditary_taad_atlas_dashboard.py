#!/usr/bin/env python3
"""Hereditary-TAAD-Atlas -- Complete 8-Gene Reference
Hereditary Thoracic Aortic Aneurysm-Dissection (TAAD) and related aortopathies:
FBN1  (Marfan-Syndrome-aortic-root-ectopia-lentis-tall-stature-PATHOGNOMONIC-triad),
TGFBR1 (Loeys-Dietz-LDS1A-bifid-uvula-arterial-tortuosity-PATHOGNOMONIC-aggressive-4cm),
TGFBR2 (Loeys-Dietz-LDS1B-small-aorta-dissection-PATHOGNOMONIC-4cm-threshold),
ACTA2  (TAAD2-Moyamoya-livedo-reticularis-iris-flocculi-PATHOGNOMONIC-cerebrovascular),
MYH11  (TAAD4-PDA-patent-ductus-PATHOGNOMONIC-SM-MHC-bicuspid-aortic-valve),
SMAD3  (LDS3-AOS-early-osteoarthritis-PATHOGNOMONIC-aneurysm-osteoarthritis-syndrome),
COL3A1 (vEDS-spontaneous-rupture-ANY-diameter-surgery-EMERGENCY-ONLY-ABSOLUTELY-CI),
LOX    (TAAD-FTAAD-collagen-elastin-crosslinking-bicuspid-aortic-valve-recently-described).

FBN1   (Fibrillin-1; 2871aa; 15q21.1; AD LOF;
         Marfan syndrome MFS1; aortic root aneurysm + ectopia lentis + tall stature PATHOGNOMONIC;
         beta-blockers/ARBs (losartan); surgery at 4.5-5.0 cm or 4.5cm+risk factors;
         pregnancy: ABSOLUTE-CI if aortic root >4.0cm;
         seed SEED_BASE+0).
TGFBR1 (TGF-beta receptor type 1; 503aa; 9q22.33; AD LOF;
         Loeys-Dietz syndrome type 1A (LDS1A); bifid uvula/cleft palate PATHOGNOMONIC;
         arterial tortuosity + hypertelorism; more aggressive than Marfan at SAME aortic size;
         surgery threshold: 4.0cm (not 4.5cm); cerebrovascular imaging mandatory;
         seed SEED_BASE+1).
TGFBR2 (TGF-beta receptor type 2; 567aa; 3p24.1; AD LOF;
         Loeys-Dietz syndrome type 1B (LDS1B); bifid uvula + tortuosity PATHOGNOMONIC;
         dissection at small diameters (<4cm) PATHOGNOMONIC; branch vessel aneurysms;
         surgery threshold: 4.0cm or even earlier with rapid growth;
         seed SEED_BASE+2).
ACTA2  (Smooth muscle alpha-actin; 377aa; 10q23.31; AD GOF missense;
         TAAD2; livedo reticularis + iris flocculi PATHOGNOMONIC; Moyamoya cerebrovascular disease;
         R179H variant: Moyamoya + TAAD + early stroke PATHOGNOMONIC triad;
         cerebral MRA mandatory; stroke risk even before aortic events;
         seed SEED_BASE+3).
MYH11  (Smooth muscle myosin heavy chain; 1972aa; 16p13.11; AD LOF;
         TAAD4; patent ductus arteriosus (PDA) + TAAD PATHOGNOMONIC pair;
         bicuspid aortic valve (BAV) association; reduced penetrance;
         PDA in adult family member + TAAD -> MYH11 panel;
         seed SEED_BASE+4).
SMAD3  (SMAD3; 425aa; 15q22.33; AD LOF;
         Loeys-Dietz syndrome type 3 (LDS3) = Aneurysm-Osteoarthritis Syndrome (AOS);
         early-onset osteoarthritis PATHOGNOMONIC (distinguishes from other LDS);
         surgery threshold 4.5cm; whole-aortic + branch imaging mandatory;
         seed SEED_BASE+5).
COL3A1 (Collagen type III alpha 1; 1466aa; 2q32.2; AD LOF dominant-negative;
         vascular Ehlers-Danlos syndrome (vEDS); spontaneous arterial rupture AT ANY AORTIC DIAMETER;
         SURGERY ABSOLUTELY CONTRAINDICATED except emergency (vessel fragility = intraoperative death);
         celiprolol reduces event rate; avoid needle procedures in large vessels;
         seed SEED_BASE+6).
LOX    (Lysyl oxidase; 417aa; 5q23.1; AD LOF;
         Hereditary TAAD (FTAAD); collagen and elastin crosslinking impairment;
         bicuspid aortic valve (BAV) enriched; recently described gene (2018);
         broader vascular surveillance needed; fewer natural history data;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3758-3765)
"""
import random

SEED_BASE = 3758

ATLAS_GENES = [
    {
        "gene": "FBN1",
        "protein": (
            "FBN1 -- 15q21.1 AD LOF -- 2871aa -- "
            "Fibrillin-1-Marfan-Syndrome-MFS1-Aortic-Root-Ectopia-Lentis-Tall-Stature-"
            "PATHOGNOMONIC-Triad-Beta-Blocker-ARB-Losartan-Surgery-4.5cm-"
            "FBN1-OMIM-134797"
        ),
        "locus": "15q21.1",
        "protein_size": (
            "2871 aa / ~320 kDa / 15q21.1 FBN1 Fibrillin-1 molecular context: "
            "STRUCTURE: "
            "  2871 aa / ~320 kDa; extracellular matrix glycoprotein; "
            "  47 calcium-binding EGF-like (cbEGF) domains; 8 TB (TGF-beta binding) domains; "
            "  Assembles into microfibrils (10-12 nm); scaffolds elastic fibres; "
            "  LOF FBN1 -> impaired microfibril assembly + increased TGF-beta signalling; "
            "MARFAN SYNDROME MFS1: "
            "  AD LOF (haploinsufficiency + dominant-negative); "
            "  DIAGNOSTIC TRIAD: aortic root aneurysm + ectopia lentis + skeletal (tall stature, arachnodactyly, scoliosis); "
            "  Revised Ghent nosology 2010: aortic root Z-score >=2 + ectopia lentis = DEFINITIVE Marfan; "
            "  Aortic root aneurysm: progressive dilation -> dissection Type A (aortic root -> arch); "
            "  Ectopia lentis: lens subluxation (superior-temporal) PATHOGNOMONIC; "
            "AORTIC MANAGEMENT: "
            "  Surveillance: transthoracic echo annually (or every 6-12m if rapid growth); "
            "  CT/MRI aorta: annual from root to iliacs; "
            "  Surgery THRESHOLD: "
            "    Standard: 4.5-5.0 cm aortic root diameter (Valsalva sinuses); "
            "    Earlier (4.5 cm) if: family history of dissection, rapid growth (>0.3 cm/yr), "
            "      severe aortic regurgitation, planned pregnancy; "
            "  Pregnancy: aortic root >4.0 cm = PREGNANCY CONTRAINDICATED; "
            "    If proceeding: high-risk obstetric + cardiac surgery on-standby; "
            "    Vaginal delivery preferred if root <4 cm with epidural; "
            "PHARMACOLOGY: "
            "  Beta-blocker (atenolol/metoprolol): first-line; reduces aortic growth rate; "
            "  ARB (losartan): MARFAN-SPECIFIC benefit beyond beta-blocker (COMPARE trial); "
            "    Losartan reduces TGF-beta signalling in FBN1-LOF; "
            "  AVOID: fluoroquinolones (aortic aneurysm risk); vigorous sport; contact sport; "
            "SKELETAL / OCULAR: "
            "  Annual ophthalmology (ectopia lentis + glaucoma + retinal detachment); "
            "  Physiotherapy for scoliosis; "
            "  Pectus excavatum: surgery if severe cardiopulmonary compromise; "
            "CASCADE: "
            "  Autosomal dominant: 50% first-degree relatives at risk; "
            "  Clinical evaluation all 1st-degree relatives; genetic testing if variant identified"
        ),
        "syndrome": "Marfan-Syndrome-MFS1 / aortic-root-aneurysm / ectopia-lentis",
        "inheritance": "AD LOF (haploinsufficiency + dominant-negative)",
        "pathognomonic": "aortic-root-aneurysm + ectopia-lentis + tall-stature = Marfan PATHOGNOMONIC triad; lens dislocation superior-temporal; losartan FBN1-specific benefit beyond beta-blocker",
        "key_drugs_ci": ["fluoroquinolones-AVOID-aortic-risk", "vigorous-sport-CI", "pregnancy-CI-root-gt4.0cm"],
        "surveillance": ["annual-echo-CT-MRI-aorta-root-to-iliacs", "annual-ophthalmology", "cascade-1st-degree", "sport-restriction"],
    },
    {
        "gene": "TGFBR1",
        "protein": (
            "TGFBR1 -- 9q22.33 AD LOF -- 503aa -- "
            "TGF-beta-Receptor-Type-1-Loeys-Dietz-LDS1A-"
            "Bifid-Uvula-Cleft-Palate-Arterial-Tortuosity-PATHOGNOMONIC-"
            "Surgery-4cm-More-Aggressive-Than-Marfan-"
            "TGFBR1-OMIM-190181"
        ),
        "locus": "9q22.33",
        "protein_size": (
            "503 aa / ~55 kDa / 9q22.33 TGFBR1 TGF-beta receptor type 1 molecular context: "
            "STRUCTURE: "
            "  503 aa / ~55 kDa; type I TGF-beta receptor serine/threonine kinase; "
            "  GS domain (Gly-Ser repeat): regulated by TGFBR2 phosphorylation; "
            "  Kinase domain: phosphorylates SMAD2/SMAD3 on activation; "
            "  LOF -> paradoxical TGF-beta pathway upregulation in aortic wall (canonical + non-canonical); "
            "LOEYS-DIETZ SYNDROME TYPE 1A (LDS1A): "
            "  AD LOF (heterozygous); "
            "  PATHOGNOMONIC CRANIOFACIAL TRIAD: bifid uvula OR cleft palate + arterial tortuosity + hypertelorism; "
            "  Bifid uvula: cleft/split uvula = SOFT PALATE MARKER; must inspect palate in all TAAD patients; "
            "  Arterial tortuosity: snake-like elongation/kinking of arteries; "
            "  AGGRESSIVE phenotype: dissection at SMALLER aortic diameters than Marfan; "
            "AORTIC MANAGEMENT -- LDS IS MORE AGGRESSIVE THAN MARFAN: "
            "  Surgery THRESHOLD: 4.0 cm (vs 4.5 cm in Marfan); "
            "    Rationale: LDS patients dissect at smaller diameters; "
            "    Some centres: 4.0 cm OR rapid growth >0.5 cm/yr -> surgery; "
            "  Surveillance: CT/MRI entire aorta AND branch vessels (celiac, renal, iliac, cerebrovascular); "
            "    Annual total-body MRA from brain to pelvis; "
            "    Brain MRA: dural ectasia + intracranial aneurysms in LDS; "
            "PHARMACOLOGY: "
            "  Beta-blocker + ARB (losartan): LDS standard; "
            "  ARB reduces TGF-beta signalling (same FBN1-pathway mechanism); "
            "CASCADE AND GENETIC: "
            "  TGFBR1 and TGFBR2 together = LDS type 1 (previously called type 1A/1B); "
            "  More severe connective tissue features than Marfan; "
            "  Uvula inspection + echocardiography all 1st-degree relatives"
        ),
        "syndrome": "Loeys-Dietz-LDS1A / bifid-uvula-tortuosity-hypertelorism-PATHOGNOMONIC / aggressive-4cm-threshold",
        "inheritance": "AD LOF",
        "pathognomonic": "bifid-uvula/cleft-palate + arterial-tortuosity + hypertelorism = LDS PATHOGNOMONIC triad; surgery at 4.0cm (not 4.5cm as in Marfan); total-body-MRA-mandatory",
        "key_drugs_ci": ["delay-surgery-to-4.5cm-NOT-applicable-TGFBR1-LDS"],
        "surveillance": ["annual-total-body-MRA-brain-to-pelvis", "branch-vessel-surveillance", "cascade-1st-degree", "uvula-inspection-all-TAAD"],
    },
    {
        "gene": "TGFBR2",
        "protein": (
            "TGFBR2 -- 3p24.1 AD LOF -- 567aa -- "
            "TGF-beta-Receptor-Type-2-Loeys-Dietz-LDS1B-"
            "Dissection-Small-Diameters-lt4cm-PATHOGNOMONIC-"
            "Branch-Vessel-Aneurysms-Surgery-4cm-Aggressive-"
            "TGFBR2-OMIM-190182"
        ),
        "locus": "3p24.1",
        "protein_size": (
            "567 aa / ~65 kDa / 3p24.1 TGFBR2 TGF-beta receptor type 2 molecular context: "
            "STRUCTURE: "
            "  567 aa / ~65 kDa; type II TGF-beta receptor serine/threonine kinase; "
            "  Constitutively active kinase; recruits TGFBR1 on TGF-beta binding; "
            "  TGFBR2 phosphorylates GS domain of TGFBR1 -> SMAD2/3 downstream signalling; "
            "  LOF TGFBR2 -> paradoxical TGF-beta upregulation via non-canonical ERK/p38 MAPK; "
            "LOEYS-DIETZ SYNDROME TYPE 1B (LDS1B): "
            "  Phenotypically overlapping with TGFBR1 (LDS1A); "
            "  DISSECTION AT SMALL DIAMETERS PATHOGNOMONIC: "
            "    LDS patients can dissect at <4 cm aortic root diameter; "
            "    Diameter alone insufficient to predict risk in LDS; "
            "    Growth rate 0.3-0.5 cm/yr also triggers surgery consideration; "
            "  BRANCH VESSEL ANEURYSMS: celiac, renal, splenic, mesenteric arteries aneurysmal; "
            "  Bicuspid aortic valve: found in some TGFBR2 LDS patients; "
            "SEVERITY: "
            "  TGFBR2 LDS can be more severe than TGFBR1 in some series; "
            "  Aggressive surveillance required; "
            "SURGERY THRESHOLD: "
            "  4.0 cm or symptomatic or rapid growth; "
            "  DO NOT wait for 5.0 cm (Marfan threshold) in LDS; "
            "PHARMACOLOGY: "
            "  Beta-blocker + ARB; "
            "  Angiotensin II AT1-receptor antagonist: losartan, irbesartan; "
            "SURVEILLANCE: "
            "  Annual total-body MRA (all arteries head to pelvis); "
            "  6-monthly echo if rapid growth; "
            "  Serial cervical spine imaging (instability in some LDS); "
            "  Dural ectasia can cause spinal symptoms"
        ),
        "syndrome": "Loeys-Dietz-LDS1B / dissection-at-small-diameters-lt4cm-PATHOGNOMONIC / branch-vessel-aneurysms",
        "inheritance": "AD LOF",
        "pathognomonic": "aortic-dissection-at-lt4cm-diameter = LDS PATHOGNOMONIC (contrast Marfan 4.5cm threshold); branch-vessel-aneurysms; serial-total-body-MRA-mandatory",
        "key_drugs_ci": ["wait-for-5cm-standard-threshold-NOT-applicable-LDS", "fluoroquinolones-AVOID"],
        "surveillance": ["annual-total-body-MRA", "6-monthly-echo-rapid-growth", "cervical-spine-imaging", "cascade-1st-degree"],
    },
    {
        "gene": "ACTA2",
        "protein": (
            "ACTA2 -- 10q23.31 AD GOF missense -- 377aa -- "
            "Smooth-Muscle-Alpha-Actin-TAAD2-Moyamoya-"
            "Livedo-Reticularis-Iris-Flocculi-PATHOGNOMONIC-"
            "Cerebrovascular-Stroke-MRA-MANDATORY-"
            "ACTA2-OMIM-102620"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "377 aa / ~42 kDa / 10q23.31 ACTA2 smooth muscle alpha-actin molecular context: "
            "STRUCTURE: "
            "  377 aa / ~42 kDa; smooth muscle-specific actin isoform; "
            "  Cytoskeletal thin filament component of smooth muscle cells; "
            "  Expressed in vascular smooth muscle, visceral smooth muscle; "
            "  Pathogenic missense -> gain-of-function stiffening of smooth muscle cytoskeleton; "
            "ACTA2-TAAD (TAAD2): "
            "  AD GOF missense variants cause smooth muscle cell dysfunction; "
            "  TAAD + widespread smooth muscle arteriopathy; "
            "  Most common familial TAAD gene after FBN1/TGFBR1/TGFBR2; "
            "SKIN / OCULAR PATHOGNOMONIC: "
            "  LIVEDO RETICULARIS: mottled reticular skin pattern on extremities/torso PATHOGNOMONIC; "
            "  IRIS FLOCCULI (iris pigment clumps): PATHOGNOMONIC for ACTA2 R179H and similar; "
            "    Slit-lamp examination mandatory in all suspected ACTA2 carriers; "
            "CEREBROVASCULAR DISEASE -- CRITICAL DISTINGUISHER: "
            "  Moyamoya-pattern cerebrovascular disease: stenosis + collaterals at circle of Willis; "
            "  Stroke risk INDEPENDENT of aortic diameter; "
            "  R179H variant: MOST SEVERE -- Moyamoya + TAAD + patent ductus + fixed mydriasis; "
            "  Cerebral MRA: MANDATORY at diagnosis and every 2yr; "
            "  Neurosurgical evaluation: revascularisation (EDAS/STA-MCA) if Moyamoya symptomatic; "
            "AORTIC MANAGEMENT: "
            "  Surgery threshold: 4.5-5.0 cm; "
            "  Cerebral events may precede aortic events in R179H; "
            "  Cardiovascular + neurology co-management mandatory; "
            "PHARMACOLOGY: "
            "  Beta-blocker; "
            "  Statin: some evidence for cerebrovascular protection in Moyamoya; "
            "  Antiplatelet: for Moyamoya stroke prevention (ASA after revascularisation); "
            "SURVEILLANCE: "
            "  Annual echo + CT/MRI aorta; "
            "  Brain MRA every 2 yr (Moyamoya surveillance); "
            "  Ophthalmology: slit-lamp iris flocculi; "
            "  Dermatology: livedo reticularis documentation"
        ),
        "syndrome": "TAAD2 / Moyamoya-cerebrovascular / livedo-reticularis-iris-flocculi-PATHOGNOMONIC",
        "inheritance": "AD GOF missense",
        "pathognomonic": "livedo-reticularis + iris-flocculi = ACTA2 PATHOGNOMONIC; Moyamoya-disease + TAAD; R179H = most-severe-triad-Moyamoya-TAAD-PDA; cerebral-MRA-MANDATORY",
        "key_drugs_ci": ["ignore-cerebrovascular-risk-ACTA2-stroke-before-aortic-event"],
        "surveillance": ["annual-echo-CT-MRI-aorta", "brain-MRA-every-2yr-Moyamoya", "slit-lamp-iris-flocculi", "livedo-reticularis-skin"],
    },
    {
        "gene": "MYH11",
        "protein": (
            "MYH11 -- 16p13.11 AD LOF -- 1972aa -- "
            "Smooth-Muscle-Myosin-Heavy-Chain-TAAD4-"
            "PDA-Patent-Ductus-Arteriosus-PATHOGNOMONIC-Pair-"
            "Bicuspid-Aortic-Valve-Reduced-Penetrance-"
            "MYH11-OMIM-160745"
        ),
        "locus": "16p13.11",
        "protein_size": (
            "1972 aa / ~227 kDa / 16p13.11 MYH11 smooth muscle myosin heavy chain molecular context: "
            "STRUCTURE: "
            "  1972 aa / ~227 kDa; smooth muscle-specific myosin heavy chain; "
            "  N-terminal motor domain; coiled-coil rod domain; "
            "  Forms smooth muscle thick filament with MYL9/MYL12 light chains; "
            "  LOF -> impaired vascular smooth muscle contractility and signalling; "
            "TAAD4 / PDA SYNDROME: "
            "  AD LOF with REDUCED PENETRANCE; "
            "  PDA (patent ductus arteriosus) PATHOGNOMONIC PAIRING with TAAD: "
            "    MYH11 is the only TAAD gene with consistent PDA association; "
            "    Adult patient with repaired PDA + family TAAD -> MYH11 gene panel; "
            "    PDA in affected family member = diagnostic clue; "
            "  BICUSPID AORTIC VALVE (BAV): MYH11 enriched in BAV + TAAD; "
            "  Reduced penetrance: many MYH11 variant carriers are asymptomatic; "
            "  TAAD phenotype often presents in 3rd-5th decade; "
            "AORTIC MANAGEMENT: "
            "  Surgery threshold: 5.0 cm (standard TAAD threshold); "
            "  BAV-related ascending aorta: 4.5-5.0 cm if BAV + MYH11 variant; "
            "  Surveillance: echo annually; CT/MRI aorta every 3-5yr or when growing; "
            "PHARMACOLOGY: "
            "  Beta-blocker: first-line aortic growth reduction; "
            "  ARB: second agent; "
            "CASCADE: "
            "  Penetrance ~50% in families; "
            "  Screen for PDA in relatives (paediatric cardiology review); "
            "  Echocardiography + genetic testing 1st-degree relatives"
        ),
        "syndrome": "TAAD4 / PDA-TAAD-PATHOGNOMONIC-Pair / Bicuspid-Aortic-Valve",
        "inheritance": "AD LOF (reduced penetrance ~50%)",
        "pathognomonic": "PDA + TAAD in same patient/family = MYH11 PATHOGNOMONIC pairing; BAV enriched; PDA-history in adult TAAD = order MYH11 panel; reduced penetrance",
        "key_drugs_ci": [],
        "surveillance": ["annual-echo", "CT-MRI-aorta-3-5yr", "PDA-screen-relatives-paediatric", "BAV-surveillance"],
    },
    {
        "gene": "SMAD3",
        "protein": (
            "SMAD3 -- 15q22.33 AD LOF -- 425aa -- "
            "SMAD3-Loeys-Dietz-LDS3-Aneurysm-Osteoarthritis-Syndrome-AOS-"
            "Early-Onset-Osteoarthritis-PATHOGNOMONIC-"
            "Surgery-4.5cm-Whole-Aortic-Branch-Imaging-"
            "SMAD3-OMIM-603109"
        ),
        "locus": "15q22.33",
        "protein_size": (
            "425 aa / ~48 kDa / 15q22.33 SMAD3 signal transducer molecular context: "
            "STRUCTURE: "
            "  425 aa / ~48 kDa; receptor-regulated SMAD; downstream TGF-beta signalling; "
            "  MH1 domain (N-terminal): DNA binding; "
            "  MH2 domain (C-terminal): receptor interaction, SMAD4 binding, transcriptional activation; "
            "  Linker region: phosphorylation by ERK/CDK; regulates SMAD3 activity; "
            "  SMAD3 LOF -> reduced TGF-beta canonical signalling -> paradoxical aortic TGF-beta upregulation via non-canonical; "
            "LOEYS-DIETZ SYNDROME TYPE 3 (LDS3) = ANEURYSM-OSTEOARTHRITIS SYNDROME (AOS): "
            "  AD LOF; "
            "  EARLY-ONSET OSTEOARTHRITIS PATHOGNOMONIC: "
            "    Distinguishes LDS3 from LDS1 (TGFBR1/2) and Marfan (FBN1); "
            "    OA in 3rd-4th decade (premature, large joints: spine, hips, knees); "
            "    OA preceding aortic events in some families; "
            "    Young adult OA + family aortic aneurysm -> SMAD3 panel; "
            "  Aortic aneurysm: root + ascending + descending + arch; "
            "  Branch vessel aneurysms (renal, celiac, splenic); "
            "  Bifid uvula less common than in TGFBR1/2 but may be present; "
            "  Mildly dysmorphic facies in some; "
            "AORTIC MANAGEMENT: "
            "  Surgery threshold: 4.5 cm (intermediate between Marfan 4.5 and LDS1 4.0); "
            "  Annual CT/MRI: whole aorta + branch vessels; "
            "  Cerebrovascular imaging included; "
            "PHARMACOLOGY: "
            "  Beta-blocker + ARB (losartan); "
            "  NSAIDs for OA: CAUTION (some NSAIDs may affect aortic remodelling); "
            "ORTHOPAEDIC: "
            "  OA management: physiotherapy, weight management; "
            "  Total joint replacement if severe OA; "
            "SURVEILLANCE: "
            "  Annual echo + CT/MRI aorta to pelvis; "
            "  Orthopaedic follow-up for OA; "
            "  Ophthalmology (lens dislocation rare but reported); "
            "  Dermatology: striae, skin translucency"
        ),
        "syndrome": "LDS3 / AOS-Aneurysm-Osteoarthritis / early-OA-PATHOGNOMONIC",
        "inheritance": "AD LOF",
        "pathognomonic": "early-onset-OA + aortic-aneurysm = SMAD3/AOS PATHOGNOMONIC distinguisher from LDS1 and Marfan; premature-OA-3rd-4th-decade-PATHOGNOMONIC; whole-aorta-branch-imaging-mandatory",
        "key_drugs_ci": ["NSAIDs-CAUTION-aortic-remodelling-concern"],
        "surveillance": ["annual-echo-CT-MRI-whole-aorta-to-pelvis", "orthopaedic-OA-follow-up", "cascade-1st-degree", "cerebrovascular-imaging"],
    },
    {
        "gene": "COL3A1",
        "protein": (
            "COL3A1 -- 2q32.2 AD LOF dominant-negative -- 1466aa -- "
            "Collagen-Type-III-Alpha-1-vEDS-Vascular-Ehlers-Danlos-"
            "Spontaneous-Arterial-Rupture-ANY-Diameter-"
            "SURGERY-ABSOLUTELY-CONTRAINDICATED-except-Emergency-"
            "Celiprolol-Reduces-Events-Needle-Vessels-AVOID-"
            "COL3A1-OMIM-120180"
        ),
        "locus": "2q32.2",
        "protein_size": (
            "1466 aa / ~138 kDa / 2q32.2 COL3A1 Collagen type III alpha 1 molecular context: "
            "STRUCTURE: "
            "  1466 aa / ~138 kDa; fibrillar collagen; pro-alpha1(III) chain; "
            "  Gly-X-Y repeats in triple-helix domain; N/C-terminal propeptides cleaved post-secretion; "
            "  Homotrimer [alpha1(III)]3; major component of blood vessels, hollow organs, skin; "
            "  LOF (haploinsufficiency or dominant-negative Gly substitution): "
            "    Haploinsufficiency: less severe (classical vEDS); "
            "    Glycine substitution (dominant-negative): severe phenotype; abnormal triple helix disrupts all chains; "
            "VASCULAR EDS (vEDS) -- THE MOST DANGEROUS EDS TYPE: "
            "  SPONTANEOUS ARTERIAL RUPTURE PATHOGNOMONIC: "
            "    Rupture WITHOUT prior aneurysmal dilation detectable on imaging; "
            "    Arteries can rupture at ANY diameter (4cm, 3cm, even normal-calibre vessels); "
            "    'Aortic diameter threshold' concept does NOT apply to vEDS; "
            "  Sites of rupture: celiac, renal, splenic, mesenteric (most common), "
            "    aorta, carotid, vertebral; "
            "  Median age first vascular event: 26-29 yr (often without prior warning); "
            "  Uterine rupture in pregnancy: ABSOLUTE CI for pregnancy in severe vEDS; "
            "SURGERY: CONTRAINDICATED EXCEPT EMERGENCY: "
            "  Surgical/endovascular repair = EXTREMELY HIGH RISK: "
            "    Vascular tissue fragility: sutures tear through; vessels tear during manipulation; "
            "    Intraoperative death rate high; "
            "  CONTRAINDICATION: elective surgical/endovascular repair of aneurysm in vEDS; "
            "  EMERGENCY only: exsanguinating haemorrhage; immediate life threat; "
            "  Endovascular (EVAR/TEVAR): preferred over open surgery if intervention unavoidable; "
            "PHARMACOLOGY -- CELIPROLOL: "
            "  Celiprolol (beta1-blocker with beta2-agonist/vasodilatory effect): "
            "    ONLY drug with RCT evidence reducing vascular events in vEDS (Ong 2010); "
            "    400-800 mg/day; "
            "    Mechanism: reduces vascular wall stress + may have collagen-stabilising effect; "
            "PROCEDURES: "
            "  Avoid needle procedures near major vessels (arteriography, central lines): "
            "    Arterial wall fragility -> haematoma, pseudoaneurysm, rupture; "
            "  Arterial line placement: radial preferred; femoral AVOID; "
            "  NO arterial angiography without haemorrhage indication; "
            "SURVEILLANCE: "
            "  Annual MRI/CT (avoid contrast injection near femoral/brachial: use peripheral IV); "
            "  Monitor for bowel/uterine rupture (high index of suspicion); "
            "  Genetic counselling: 50% AD risk; vEDS most severe for reproductive decisions"
        ),
        "syndrome": "vEDS-Vascular-Ehlers-Danlos / spontaneous-arterial-rupture-ANY-diameter-PATHOGNOMONIC / surgery-CONTRAINDICATED-except-emergency",
        "inheritance": "AD LOF dominant-negative (Gly substitution = severe) or haploinsufficiency (milder)",
        "pathognomonic": "spontaneous-arterial-rupture-at-ANY-diameter-including-normal-calibre = vEDS PATHOGNOMONIC; surgery-ABSOLUTELY-CI-except-emergency; celiprolol-only-RCT-evidence; avoid-needle-near-vessels",
        "key_drugs_ci": ["elective-surgery-ABSOLUTELY-CI-vEDS", "femoral-arterial-procedures-AVOID", "angiography-AVOID-without-haemorrhage-indication"],
        "surveillance": ["annual-MRI-CT-avoid-contrast-femoral", "high-index-suspicion-bowel-uterine-rupture", "genetic-counselling-reproductive-decisions", "celiprolol-400-800mg"],
    },
    {
        "gene": "LOX",
        "protein": (
            "LOX -- 5q23.1 AD LOF -- 417aa -- "
            "Lysyl-Oxidase-Hereditary-TAAD-FTAAD-"
            "Collagen-Elastin-Crosslinking-Impairment-"
            "Bicuspid-Aortic-Valve-BAV-Enriched-Recently-Described-2018-"
            "LOX-OMIM-153455"
        ),
        "locus": "5q23.1",
        "protein_size": (
            "417 aa / ~48 kDa / 5q23.1 LOX Lysyl oxidase molecular context: "
            "STRUCTURE: "
            "  417 aa / ~48 kDa; copper-dependent amine oxidase; "
            "  Signal peptide (secreted); propeptide (regulatory); catalytic domain (LTQ cofactor); "
            "  Catalyses lysyl oxidation: cross-linking of collagen and elastin in ECM; "
            "  LOX LOF -> reduced collagen/elastin crosslinking -> weakened vessel wall; "
            "  Secreted into extracellular space; acts on fibrillar collagen and elastin; "
            "HEREDITARY TAAD / FTAAD: "
            "  First described as TAAD gene 2018 (Lee et al.); "
            "  AD LOF causing familial thoracic aortic aneurysm; "
            "  BICUSPID AORTIC VALVE (BAV) ENRICHED: "
            "    LOX LOF enriched in BAV + TAAD cohorts; "
            "    BAV is the most common congenital heart defect (1-2% population); "
            "    LOX + BAV -> accelerated aortic dilation; "
            "  Less natural history data than FBN1/TGFBR1 due to recent discovery; "
            "  Penetrance and expressivity still being characterised; "
            "AORTIC MANAGEMENT: "
            "  Surgery threshold: 5.0 cm (standard; insufficient LOX-specific data); "
            "  BAV + LOX: some centres use 4.5 cm given dual risk; "
            "  Broad aortic surveillance: root + ascending + whole aorta; "
            "PHARMACOLOGY: "
            "  Beta-blocker first-line; "
            "  ARB second agent; "
            "  Copper supplementation: theoretical (LOX requires copper); no established clinical role yet; "
            "CASCADE: "
            "  Recently identified gene -- family history important; "
            "  Echo + genetic testing all 1st-degree relatives; "
            "SURVEILLANCE: "
            "  Annual echo; CT/MRI aorta every 1-3 yr; "
            "  BAV follow-up per BAV guidelines; "
            "  Broader vascular surveillance evolving as natural history described"
        ),
        "syndrome": "FTAAD-Familial-TAAD / BAV-enriched / collagen-elastin-crosslinking",
        "inheritance": "AD LOF",
        "pathognomonic": "LOX + BAV + TAAD family = lysyl-oxidase-FTAAD-triad; recently-described-2018-less-natural-history-data; copper-dependent-crosslinking-impairment",
        "key_drugs_ci": [],
        "surveillance": ["annual-echo", "CT-MRI-aorta-1-3yr", "BAV-guideline-surveillance", "cascade-1st-degree", "copper-status-evolving"],
    },
]

DEFINITIONS = {
    "TAAD": "Thoracic Aortic Aneurysm and Dissection: dilation/rupture of thoracic aorta; Type A (ascending) and Type B (descending); heritable in 15-20% of cases.",
    "Marfan_Syndrome": "FBN1 AD LOF; Revised Ghent 2010: aortic root Z-score >=2 + ectopia lentis = DEFINITIVE; surgery 4.5-5.0 cm; losartan + beta-blocker; fluoroquinolones AVOID.",
    "Loeys_Dietz_LDS": "TGFBR1/TGFBR2 AD LOF; bifid uvula + arterial tortuosity + hypertelorism PATHOGNOMONIC; more aggressive than Marfan; surgery threshold 4.0 cm; total-body MRA annually.",
    "vEDS": "Vascular Ehlers-Danlos syndrome; COL3A1 AD LOF; spontaneous arterial rupture at ANY diameter; surgery ABSOLUTELY CI except emergency; celiprolol reduces events (Ong 2010 RCT).",
    "Moyamoya": "Cerebrovascular pattern in ACTA2: progressive stenosis of internal carotid arteries + formation of collateral vessels (smoke pattern on angiography); stroke risk independent of aortic diameter.",
    "PDA": "Patent ductus arteriosus: failure of ductus arteriosus to close after birth; MYH11 TAAD4 has specific PDA + TAAD pairing PATHOGNOMONIC; adult with repaired PDA + family TAAD -> MYH11.",
    "AOS": "Aneurysm-Osteoarthritis Syndrome = Loeys-Dietz type 3; SMAD3 AD LOF; early-onset OA + aortic aneurysm PATHOGNOMONIC; OA distinguishes LDS3 from LDS1/Marfan.",
    "Ectopia_lentis": "Lens subluxation/dislocation; FBN1 Marfan: superior-temporal displacement PATHOGNOMONIC; annual ophthalmology mandatory; glaucoma + retinal detachment risk.",
    "Bifid_uvula": "Split or cleft uvula; PATHOGNOMONIC soft-palate marker for Loeys-Dietz (TGFBR1/TGFBR2); inspect palate in all TAAD patients; associated with cleft palate spectrum.",
    "Iris_flocculi": "Iris pigment clumps on slit-lamp exam; PATHOGNOMONIC for ACTA2 particularly R179H variant; mandates ACTA2 genetic testing in TAAD patients; ophthalmology referral.",
    "Livedo_reticularis": "Mottled reticular skin discolouration on extremities; PATHOGNOMONIC for ACTA2 TAAD2; caused by smooth muscle dysfunction in skin vessels; inspect skin in all TAAD patients.",
    "Celiprolol": "Beta1-blocker with beta2-agonist/vasodilatory effect; ONLY drug with RCT evidence (Ong 2010) reducing vascular events in vEDS/COL3A1; 400-800 mg/day; mechanism: reduces pulsatile stress.",
    "Surgery_thresholds_comparison": "Marfan/FBN1: 4.5-5.0 cm; LDS/TGFBR1/2: 4.0 cm; ACTA2: 4.5-5.0 cm; MYH11: 5.0 cm; SMAD3: 4.5 cm; vEDS/COL3A1: NO elective surgery (spontaneous rupture at any diameter); LOX: 5.0 cm.",
    "Fluoroquinolone_TAAD_risk": "Fluoroquinolones (ciprofloxacin, levofloxacin) increase aortic aneurysm/dissection risk in population studies; AVOID in all hereditary TAAD patients; use alternative antibiotics.",
    "Revised_Ghent_Marfan": "2010 Ghent nosology: aortic root Z-score >=2 AND ectopia lentis = DEFINITIVE Marfan regardless of family history; FBN1 pathogenic variant alone with aortic root Z-score >=2 = DEFINITIVE.",
    "Total_body_MRA": "MRA from brain to pelvis: mandatory in LDS (TGFBR1/TGFBR2/SMAD3) annually; detects branch vessel aneurysms (celiac, renal, splenic), cerebral aneurysms, dural ectasia.",
    "BAV": "Bicuspid aortic valve: most common congenital heart defect (1-2%); associated with TAAD (aortopathy); enriched in MYH11 and LOX TAAD families; BAV aortopathy may need earlier surgical threshold (4.5 cm).",
    "Cascade_TAAD": "Cascade genetic testing all 1st-degree relatives in all hereditary TAAD genes; 50% AD inheritance risk; pre-symptomatic identification allows surveillance before first aortic event.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_overview() -> dict:
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(1.0, rng.gauss(
            27.0 if g["gene"] == "COL3A1" else   # vEDS -- youngest rupture
            32.0 if g["gene"] in ("TGFBR1", "TGFBR2") else  # LDS aggressive young
            38.0 if g["gene"] == "ACTA2" else   # ACTA2 TAAD2
            35.0 if g["gene"] == "SMAD3" else   # AOS -- OA in 30s, aneurysm 40s
            40.0 if g["gene"] == "MYH11" else
            43.0 if g["gene"] == "LOX" else
            45.0,  # FBN1 Marfan
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
        "atlas_name": "Hereditary-TAAD-Atlas",
        "atlas": "Hereditary-TAAD-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference -- FBN1 / TGFBR1 / TGFBR2 / ACTA2 / "
            "MYH11 / SMAD3 / COL3A1 / LOX"
        ),
        "total_patients": 320,
        "gene_count": 8,
        "seeds": "3758-3765",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "marfan_spectrum": ["FBN1"],
            "loeys_dietz_spectrum": ["TGFBR1", "TGFBR2", "SMAD3"],
            "smooth_muscle_genes": ["ACTA2", "MYH11"],
            "collagen_ecm_genes": ["COL3A1", "LOX"],
            "surgery_ci_veds": ["COL3A1"],
            "aggressive_4cm_threshold": ["TGFBR1", "TGFBR2"],
            "cerebrovascular_risk": ["ACTA2"],
            "bav_associated": ["MYH11", "LOX"],
        },
        "key_clinical_rules": [
            "FBN1: aortic-root + ectopia-lentis + tall-stature = Marfan PATHOGNOMONIC triad; losartan + beta-blocker; surgery 4.5-5.0 cm; fluoroquinolones AVOID",
            "TGFBR1: bifid-uvula + arterial-tortuosity = LDS PATHOGNOMONIC; MORE AGGRESSIVE than Marfan -- surgery at 4.0 cm (NOT 4.5 cm); total-body MRA annually",
            "TGFBR2: dissection at lt4cm diameter PATHOGNOMONIC for LDS; branch-vessel aneurysms; total-body MRA annually; surgery threshold 4.0 cm",
            "ACTA2: livedo-reticularis + iris-flocculi = ACTA2 PATHOGNOMONIC; Moyamoya cerebrovascular disease; cerebral MRA MANDATORY; stroke risk before aortic events",
            "MYH11: PDA + TAAD in family = MYH11 PATHOGNOMONIC pairing; BAV associated; reduced penetrance ~50%; adult with repaired PDA + family TAAD -> MYH11",
            "SMAD3: early-onset OA + aortic aneurysm = LDS3/AOS PATHOGNOMONIC distinguisher; OA in 30s preceding aortic events; whole-aorta branch imaging mandatory",
            "COL3A1: vEDS -- spontaneous arterial rupture at ANY diameter PATHOGNOMONIC; surgery ABSOLUTELY CI except emergency; celiprolol RCT evidence; avoid needle near major vessels",
            "LOX: recently-described-2018 TAAD gene; BAV enriched; collagen/elastin crosslinking impairment; insufficient natural-history data -- standard 5.0 cm threshold",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(1.0, rng.gauss(
            27.0 if g["gene"] == "COL3A1" else
            32.0 if g["gene"] in ("TGFBR1", "TGFBR2") else
            38.0 if g["gene"] == "ACTA2" else
            35.0 if g["gene"] == "SMAD3" else
            40.0 if g["gene"] == "MYH11" else
            43.0 if g["gene"] == "LOX" else
            45.0,
            9.0)), 1)
        pct_f = round(rng.uniform(40, 55), 1)
        # Dissection/rupture events
        dissection_n = rng.randint(18, 32) if g["gene"] in ("COL3A1", "TGFBR1", "TGFBR2") else rng.randint(6, 18)
        surgery_n = 0 if g["gene"] == "COL3A1" else rng.randint(12, 28)
        stroke_n = rng.randint(8, 18) if g["gene"] == "ACTA2" else rng.randint(1, 5)
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age_dx": mean_age,
            "pct_female": pct_f,
            "dissection_rupture_n": dissection_n,
            "surgery_n": surgery_n,
            "stroke_n": stroke_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "pathognomonic": g["pathognomonic"],
        }

    return {
        "atlas": "Hereditary-TAAD-Atlas",
        "total_patients": 320,
        "seeds": "3758-3765",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "marfan_spectrum_genes": ["FBN1"],
            "loeys_dietz_spectrum_genes": ["TGFBR1", "TGFBR2", "SMAD3"],
            "smooth_muscle_genes": ["ACTA2", "MYH11"],
            "ecm_crosslinking_genes": ["COL3A1", "LOX"],
            "surgery_contraindicated_elective": ["COL3A1"],
            "aggressive_4cm_surgery_threshold": ["TGFBR1", "TGFBR2"],
            "cerebrovascular_moyamoya_risk": ["ACTA2"],
            "bav_associated_genes": ["MYH11", "LOX"],
            "fluoroquinolones_avoid_all": ["FBN1", "TGFBR1", "TGFBR2", "ACTA2", "MYH11", "SMAD3", "COL3A1", "LOX"],
            "total_body_mra_mandatory": ["TGFBR1", "TGFBR2", "SMAD3"],
            "celiprolol_rct_evidence": ["COL3A1"],
            "losartan_specific_benefit": ["FBN1", "TGFBR1", "TGFBR2", "SMAD3"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-TAAD-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "FBN1-MARFAN-PATHOGNOMONIC-TRIAD": (
                "FBN1 Marfan: aortic root Z-score >=2 + ectopia lentis = DEFINITIVE Marfan (Ghent 2010) -- "
                "inspect lens; beta-blocker + losartan; surgery 4.5-5.0 cm; AVOID fluoroquinolones"
            ),
            "TGFBR1-TGFBR2-LDS-4CM-THRESHOLD": (
                "LDS (TGFBR1/TGFBR2): surgery threshold 4.0 cm -- NOT 4.5 cm as in Marfan -- "
                "LDS dissects at smaller diameters; total-body MRA annually; bifid uvula inspection in all TAAD"
            ),
            "TGFBR2-DISSECTION-SMALL-DIAMETER-PATHOGNOMONIC": (
                "Aortic dissection at lt4cm = LDS/TGFBR1/TGFBR2 PATHOGNOMONIC -- "
                "do NOT reassure with 'normal aortic size'; branch vessel aneurysms common"
            ),
            "ACTA2-MOYAMOYA-CEREBRAL-MRA-MANDATORY": (
                "ACTA2: cerebral MRA MANDATORY at diagnosis and every 2yr -- "
                "Moyamoya stroke can precede aortic events; livedo reticularis + iris flocculi = ACTA2 PATHOGNOMONIC"
            ),
            "COL3A1-VEDS-SURGERY-ABSOLUTELY-CI": (
                "vEDS/COL3A1: elective surgical/endovascular repair ABSOLUTELY CONTRAINDICATED -- "
                "vessel fragility causes intraoperative death; emergency surgery only; celiprolol 400-800mg/day (only RCT evidence)"
            ),
            "COL3A1-RUPTURE-ANY-DIAMETER-PATHOGNOMONIC": (
                "vEDS: spontaneous arterial rupture at ANY aortic diameter PATHOGNOMONIC -- "
                "'aortic diameter threshold' concept does NOT apply; avoid needle procedures near major vessels"
            ),
            "MYH11-PDA-TAAD-PATHOGNOMONIC-PAIRING": (
                "MYH11: PDA + TAAD in patient or family = PATHOGNOMONIC pairing -- "
                "adult with repaired PDA + family TAAD: order MYH11 panel; BAV enriched; reduced penetrance ~50%"
            ),
            "SMAD3-AOS-OA-PATHOGNOMONIC-DISTINGUISHER": (
                "SMAD3/AOS/LDS3: early-onset OA (30s-40s) + aortic aneurysm = PATHOGNOMONIC for LDS3 -- "
                "distinguishes from LDS1 (TGFBR1/2) and Marfan (FBN1); young adult OA + aortic family Hx -> SMAD3"
            ),
            "FLUOROQUINOLONE-AVOID-ALL-TAAD-GENES": (
                "Fluoroquinolones (ciprofloxacin, levofloxacin): AVOID in ALL hereditary TAAD genes -- "
                "aortic aneurysm/dissection risk in population studies; use alternative antibiotics"
            ),
            "LOSARTAN-FBN1-LDS-SPECIFIC-BENEFIT": (
                "Losartan (ARB): TAAD-gene-specific benefit beyond beta-blocker in FBN1/TGFBR1/TGFBR2/SMAD3 -- "
                "reduces TGF-beta pathway activation in aortic wall; add to beta-blocker in all Marfan/LDS patients"
            ),
            "TOTAL-BODY-MRA-LDS-ANNUALLY": (
                "LDS (TGFBR1/TGFBR2/SMAD3): total-body MRA brain to pelvis ANNUALLY -- "
                "detects intracranial, branch vessel (celiac/renal/splenic), and aortic aneurysms; dural ectasia"
            ),
            "ALL-TAAD-GENES-CASCADE-MANDATORY": (
                "Cascade genetic testing all 1st-degree relatives MANDATORY for all 8 TAAD genes -- "
                "50% AD inheritance risk; pre-symptomatic surveillance prevents first aortic/vascular event"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary TAAD Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, dissection/rupture={d['dissection_rupture_n']}/40, surgery={d['surgery_n']}/40, stroke={d['stroke_n']}/40")
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
