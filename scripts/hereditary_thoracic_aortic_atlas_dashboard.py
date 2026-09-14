#!/usr/bin/env python3
"""Hereditary-Thoracic-Aortic-Atlas — Complete 8-Gene Hereditary Thoracic Aortic Disease Atlas
(FBN1 · TGFBR1 · TGFBR2 · SMAD3 · ACTA2 · MYH11 · COL3A1 · SLC2A10).

FBN1     (Fibrillin-1; 2871 aa; 350 kDa; 15q21.1; AD;
          Marfan syndrome (MFS); most common HTAD gene (~65% of heritable TAAD);
          aortic root aneurysm + ectopia lentis + tall stature = classical triad;
          prophylactic aortic root replacement at 50 mm (45 mm for rapid progression);
          beta-blocker + losartan (ARB) standard medical therapy;
          seed SEED_BASE+0).
TGFBR1   (TGF-β receptor type I serine/threonine kinase; 503 aa; 56 kDa; 9q22.33; AD;
          Loeys-Dietz syndrome type 1 (LDS1); bifid uvula / cleft palate PATHOGNOMONIC;
          AGGRESSIVE — dissection at smaller aortic diameters than MFS;
          wide arterial aneurysm burden (beyond aortic root);
          prophylactic surgery threshold LOWER (45 mm or 42 mm with risk factors);
          seed SEED_BASE+1).
TGFBR2   (TGF-β receptor type II; 567 aa; 70 kDa; 3p24.1; AD;
          Loeys-Dietz syndrome type 2 (LDS2); hypertelorism + bifid uvula;
          most aggressive LDS gene — dissection earliest (mean 26 years);
          p.Arg460Cys most common — kinase domain missense;
          MANDATORY full-body MRI/MRA every 6-12 months — subclavian/iliac aneurysms common;
          seed SEED_BASE+2).
SMAD3    (SMAD family member 3; 425 aa; 48 kDa; 15q22.33; AD;
          Aneurysm-Osteoarthritis syndrome (AOS) / LDS type 3;
          OSTEOARTHRITIS onset < 30 years PATHOGNOMONIC — distinguishes from other LDS;
          aortic + branch vessel aneurysms; mild craniofacial features;
          TGF-β signalling pathway — downstream of TGFBR1/TGFBR2;
          seed SEED_BASE+3).
ACTA2    (Actin alpha 2 smooth muscle; 375 aa; 42 kDa; 10q23.31; AD;
          Multisystemic smooth muscle dysfunction syndrome (MSMDS) / familial TAAD4;
          p.Arg179His: MSMDS triad — TAAD + stroke + fixed dilated pupils PATHOGNOMONIC;
          premature coronary artery disease + Moyamoya disease;
          other ACTA2 variants: familial TAAD only (milder);
          seed SEED_BASE+4).
MYH11    (Myosin heavy chain 11 smooth muscle; 1972 aa; 227 kDa; 16p13.11; AD;
          Familial TAAD with patent ductus arteriosus (TAAD + PDA) PATHOGNOMONIC pairing;
          rare gene (~2% HTAD); smooth muscle myosin heavy chain isoform;
          aortic diameter smaller at dissection than FBN1;
          echocardiographic surveillance every 1-2 years;
          seed SEED_BASE+5).
COL3A1   (Collagen type III alpha 1; 1466 aa; 139 kDa; 2q32.2; AD;
          Vascular Ehlers-Danlos syndrome (vEDS) — most lethal connective tissue disorder;
          SPONTANEOUS ARTERIAL RUPTURE without aneurysm PATHOGNOMONIC;
          SURGERY CONTRAINDICATED (extreme vessel friability — mortality > 50%);
          celiprolol 400 mg/day reduces events (Level A RCT evidence);
          bowel/uterine rupture also pathognomonic (hollow organ);
          seed SEED_BASE+6).
SLC2A10  (Solute carrier family 2 member 10 / GLUT10; 541 aa; 57 kDa; 20q13.12; AR;
          Arterial tortuosity syndrome (ATS); 100% arterial tortuosity DIAGNOSTIC;
          stenosis + aneurysm of major arteries (pulmonary, aorta, coronary);
          AR — BOTH SEXES; neonatal/infantile presentation;
          GLUT10 is a dehydroascorbic acid (DHA) transporter → ascorbate deficiency;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2566-2573).
"""

import random

SEED_BASE = 2566

HTAD_GENES = [
    # -- FBN1 -- Marfan syndrome ---------------------------------------------------------------
    {
        "gene": "FBN1",
        "alt_name": (
            "FBN1 (FBN1-2871aa-15q21.1 / AD -- "
            "MARFAN-SYNDROME-MOST-COMMON-HTAD-65pct -- "
            "AORTIC-ROOT-ANEURYSM-ECTOPIA-LENTIS-TALL-STATURE-TRIAD -- "
            "PROPHYLACTIC-ROOT-REPLACEMENT-50mm-45mm-RAPID-PROGRESSION -- "
            "BETA-BLOCKER-LOSARTAN-STANDARD-MEDICAL-THERAPY)"
        ),
        "protein": (
            "FBN1 -- 15q21.1 AD -- FBN1-2871aa -- "
            "Fibrillin-1-350kDa-Extracellular-Matrix-Glycoprotein-Calcium-Binding-EGF-Modules -- "
            "Microfibril-Scaffold-TGF-beta-Sequestration-Regulatory-Role -- "
            "Aortic-Wall-Elastic-Lamella-Architecture -- "
            "OMIM-Gene-134797-Disease-Marfan-154700"
        ),
        "locus": "15q21.1",
        "protein_size": "2871 aa / 350 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous LOF or dominant-negative missense); "
            "de novo mutations in 25-30% (no family history); "
            "most common heritable thoracic aortic disease gene (~65% of heritable TAAD); "
            "FBN1 mutations cause Marfan syndrome (MFS) — systemic connective tissue disorder; "
            "CLASSIC TRIAD: aortic root aneurysm + ectopia lentis + tall stature with disproportion; "
            "penetrance near complete but expressivity highly variable within families; "
            "p.Cys mutations (EGF domain disulphide bonds) often more severe"
        ),
        "disease_category": (
            "Marfan syndrome — systemic fibrillinopathy; "
            "primary risk: aortic root aneurysm → type A dissection (most lethal); "
            "ocular: ectopia lentis (60%), myopia, retinal detachment; "
            "skeletal: arachnodactyly, pectus deformities, scoliosis, tall stature; "
            "dural ectasia (~70%) — LP complications; "
            "PROPHYLACTIC aortic root replacement at 50 mm (45 mm in rapid growth >3 mm/year or family history of dissection)"
        ),
        "disease_pathway": (
            "Fibrillin-1 forms the backbone of extracellular matrix microfibrils, "
            "which provide structural support to the aortic wall elastic laminae. "
            "FBN1 loss-of-function → reduced microfibril scaffolding → increased TGF-β bioavailability "
            "(fibrillin normally sequesters TGF-β in the matrix). "
            "Elevated TGF-β signalling → aortic smooth muscle cell apoptosis + MMP upregulation "
            "→ elastic lamella degradation → progressive aortic dilation. "
            "Losartan (ARB) blocks AT1R → reduces TGF-β signalling → slows aortic growth (supported by evidence). "
            "Beta-blockers reduce aortic wall stress (dP/dt reduction) → additive benefit. "
            "Both therapies are lifelong — neither halts, only slows progression."
        ),
        "pathognomonic": (
            "AORTIC ROOT ANEURYSM + ECTOPIA LENTIS + ARACHNODACTYLY IN A TALL PATIENT: "
            "pathognomonic for Marfan syndrome (2010 revised Ghent nosology). "
            "Thumb sign (Steinberg): distal thumb phalanx protrudes beyond ulnar border of fist. "
            "Wrist sign (Walker-Murdoch): thumb + 5th digit overlap when encircling opposite wrist. "
            "Slit-lamp: ectopia lentis (superotemporal displacement — opposite to homocystinuria inferotemporal). "
            "Dural ectasia on MRI (sacral): highly specific for MFS (70%). "
            "Revised Ghent criteria score ≥ 7 systemic features OR FBN1 pathogenic variant: diagnostic."
        ),
        "treatment": (
            "BETA-BLOCKER (atenolol 1-2 mg/kg/day OR propranolol): reduces aortic growth rate, lifelong; "
            "LOSARTAN (0.6-1.4 mg/kg/day): ARB — blocks AT1R, reduces TGF-β — synergistic with beta-blocker; "
            "PROPHYLACTIC AORTIC ROOT REPLACEMENT: at 50 mm (45 mm if rapid growth >3 mm/year, family history); "
            "Valve-sparing aortic root replacement (David/Yacoub) preferred in young patients; "
            "Bentall procedure (composite graft with mechanical valve): if valve incompetent; "
            "Ocular: annual slit-lamp + refraction; retinal detachment surgery if occurs; "
            "Avoid strenuous isometric exercise, contact sports, competitive athletics; "
            "Pregnancy: high risk — prophylactic surgery recommended before pregnancy if root >40 mm; "
            "Dural ectasia: MRI surveillance, avoid LP at L4-5 (dural sac enlarged)."
        ),
        "key_features": [
            "Arachnodactyly — positive thumb sign (Steinberg) + wrist sign (Walker-Murdoch)",
            "Ectopia lentis — superotemporal (OPPOSITE to homocystinuria which is inferonasal)",
            "Aortic root Z-score ≥ 2 at sinuses of Valsalva — first site of dilation",
            "Tall stature with disproportion — arm span/height > 1.05, reduced upper/lower segment ratio",
            "Pectus excavatum/carinatum — frequent skeletal manifestation",
            "Spontaneous pneumothorax — apical bullae (up to 5% of MFS patients)",
            "Dural ectasia — L3-S1 sacral expansion on MRI (70% — highly specific)",
            "Beta-blocker + losartan lifelong — reduces aortic growth, does NOT halt it"
        ],
        "key_ddx": [
            "TGFBR1/TGFBR2 (LDS) — bifid uvula, hypertelorism, more aggressive; no ectopia lentis",
            "Homocystinuria (CBS) — ectopia lentis INFERONASAL; thromboembolism; AR; elevated homocysteine",
            "Loeys-Dietz syndrome — overlapping skeletal features; bifid uvula pathognomonic",
            "MASS phenotype (FBN1 missense) — myopia, MVP, aortic dilation minimal, skin striae; not full MFS"
        ],
        "onset_age_years_median": 25,
        "aortic_event_pct": 40,
        "beta_blocker_pct": 90,
        "losartan_pct": 72,
        "surgery_required_pct": 45,
        "arterial_tortuosity_pct": 30,
        "vascular_rupture_risk_pct": 25,
        "bilateral_aneurysm_pct": 35,
        "nbs_indicated": False,
    },
    # -- TGFBR1 -- Loeys-Dietz syndrome type 1 ------------------------------------------------
    {
        "gene": "TGFBR1",
        "alt_name": (
            "TGFBR1 (TGFBR1-503aa-9q22.33 / AD -- "
            "LOEYS-DIETZ-SYNDROME-TYPE-1-LDS1 -- "
            "BIFID-UVULA-CLEFT-PALATE-HYPERTELORISM-PATHOGNOMONIC -- "
            "AGGRESSIVE-DISSECTION-SMALLER-DIAMETER-THAN-MFS -- "
            "PROPHYLACTIC-THRESHOLD-45mm-42mm-RISK-FACTORS)"
        ),
        "protein": (
            "TGFBR1 -- 9q22.33 AD -- TGFBR1-503aa -- "
            "TGF-beta-Receptor-Type-I-56kDa-Serine-Threonine-Kinase-ALK5 -- "
            "GS-Domain-LXXL-Motif-Kinase-Domain-SMAD2-SMAD3-Phosphorylation -- "
            "TGF-beta-Signalling-Canonical-Smad-Pathway -- "
            "OMIM-Gene-190181-Disease-LDS1-609192"
        ),
        "locus": "9q22.33",
        "protein_size": "503 aa / 56 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous GOF or LOF variants); "
            "Loeys-Dietz syndrome type 1 (LDS1); "
            "CRANIOFACIAL features pathognomonic: bifid uvula, cleft palate, hypertelorism, craniosynostosis; "
            "AGGRESSIVE: dissection occurs at smaller aortic diameters (mean 40-42 mm) vs FBN1 (50+ mm); "
            "WIDESPREAD ARTERIAL INVOLVEMENT: subclavian, renal, iliac, coronary aneurysms; "
            "full-body MRI/MRA every 6-12 months mandatory (beyond aortic root)"
        ),
        "disease_category": (
            "Loeys-Dietz syndrome type 1 — TGF-β receptor serine/threonine kinase; "
            "aggressive aortic + multivessel aneurysm; "
            "BIFID UVULA: simple finding but distinguishes LDS from MFS (uvula normal in MFS); "
            "lower surgical threshold than MFS (45 mm, 42 mm with risk factors); "
            "TGF-β paradox: LOF receptor mutations → INCREASED TGF-β signalling (through non-canonical pathways); "
            "losartan first-line (ARB) — same TGF-β pathway rationale as MFS"
        ),
        "disease_pathway": (
            "TGFBR1 (ALK5) receives TGF-β signal from TGFBR2 via heterodimeric receptor complex. "
            "Paradoxically, LOF TGFBR1 mutations → ELEVATED, not reduced, TGF-β signalling. "
            "This 'TGF-β paradox' occurs because LOF → upregulation of SMAD-independent pathways "
            "(ERK, JNK, p38 MAPK) that drive MMP production + elastic lamella destruction. "
            "Aortic wall: elevated TGF-β → myofibroblast activation → MMP-9/MMP-2 → ECM degradation. "
            "Losartan (AT1R blockade) → downstream TGF-β pathway attenuation. "
            "Cervical spine instability from joint laxity can cause cord injury — MRI spine mandatory."
        ),
        "pathognomonic": (
            "BIFID UVULA + AORTIC ROOT ANEURYSM: pathognomonic for Loeys-Dietz syndrome. "
            "Cleft palate (hard or soft) in combination with TAAD — LDS until proven otherwise. "
            "Hypertelorism (wide-set eyes, interpupillary distance > 2 SD): present in LDS1. "
            "Craniosynostosis: premature fusion, particularly coronal — LDS1 more than LDS2. "
            "Tortuous arteries on CTA (carotid, subclavian, renal) before major aneurysm development. "
            "Full-body MRA: multiple aneurysms at different arterial sites simultaneously — diagnostic."
        ),
        "treatment": (
            "LOSARTAN (0.6-1.4 mg/kg/day): first-line — TGF-β pathway suppression; "
            "BETA-BLOCKER (atenolol): additive benefit; "
            "PROPHYLACTIC AORTIC ROOT REPLACEMENT: at 45 mm (42 mm with risk factors); "
            "FULL-BODY MRI/MRA: every 6-12 months (aorta + entire arterial tree — abdominal/pelvic arteries); "
            "CERVICAL SPINE MRI: baseline + when symptomatic (atlantoaxial instability); "
            "Valve-sparing preferred if valve anatomy permits; "
            "Branch artery aneurysms (subclavian, renal, iliac): endovascular/surgical if symptomatic or growing; "
            "Pregnancy: very high risk — complete arterial imaging before each pregnancy; "
            "Avoid contact sports, isometric exercise; gentle aerobic exercise (swimming) acceptable."
        ),
        "key_features": [
            "BIFID UVULA — simple clinical finding, PATHOGNOMONIC for LDS; examine uvula in ALL TAAD patients",
            "Hypertelorism — wide-set eyes; interpupillary distance > 2 SD above mean",
            "Craniosynostosis — premature suture fusion; macrocephaly or abnormal head shape",
            "Aortic dissection at SMALLER diameters than MFS (mean ~40-42 mm) — lower threshold",
            "Widespread arterial involvement: subclavian, renal, iliac, coronary aneurysms simultaneously",
            "Cervical spine instability (atlantoaxial) — MRI spine mandatory at diagnosis",
            "Club feet, camptodactyly (congenital joint contractures at fingers)",
            "Full-body MRI/MRA every 6-12 months — NOT just echocardiography"
        ],
        "key_ddx": [
            "TGFBR2 (LDS2) — identical features; most aggressive; molecular test separates",
            "FBN1 (MFS) — NO bifid uvula; ectopia lentis in MFS; larger aneurysm at dissection",
            "SMAD3 (LDS3/AOS) — early-onset osteoarthritis PATHOGNOMONIC; milder craniofacial",
            "Shprintzen-Goldberg syndrome (SKI) — craniosynostosis + Marfanoid + intellectual disability"
        ],
        "onset_age_years_median": 12,
        "aortic_event_pct": 55,
        "beta_blocker_pct": 95,
        "losartan_pct": 90,
        "surgery_required_pct": 60,
        "arterial_tortuosity_pct": 80,
        "vascular_rupture_risk_pct": 40,
        "bilateral_aneurysm_pct": 70,
        "nbs_indicated": False,
    },
    # -- TGFBR2 -- Loeys-Dietz syndrome type 2 ------------------------------------------------
    {
        "gene": "TGFBR2",
        "alt_name": (
            "TGFBR2 (TGFBR2-567aa-3p24.1 / AD -- "
            "LOEYS-DIETZ-SYNDROME-TYPE-2-LDS2-MOST-AGGRESSIVE -- "
            "DISSECTION-MEAN-AGE-26-YEARS-SMALLEST-DIAMETER -- "
            "HYPERTELORISM-BIFID-UVULA-FULL-BODY-MRA-MANDATORY -- "
            "p.Arg460Cys-KINASE-DOMAIN-MOST-COMMON)"
        ),
        "protein": (
            "TGFBR2 -- 3p24.1 AD -- TGFBR2-567aa -- "
            "TGF-beta-Receptor-Type-II-70kDa-Constitutively-Active-Serine-Threonine-Kinase -- "
            "Extracellular-Ligand-Binding-Domain-Transmembrane-Helix-Kinase-Domain -- "
            "Transphosphorylates-TGFBR1-ALK5-on-GS-Domain -- "
            "OMIM-Gene-190182-Disease-LDS2-610168"
        ),
        "locus": "3p24.1",
        "protein_size": "567 aa / 70 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous missense/LOF); "
            "Loeys-Dietz syndrome type 2 (LDS2) — most aggressive LDS subtype; "
            "mean age at dissection: 26 years (earlier than FBN1 and TGFBR1); "
            "p.Arg460Cys (kinase domain): most common TGFBR2 variant worldwide; "
            "phenotype OVERLAPS with MFS/LDS1 but typically MORE SEVERE; "
            "FULL-BODY MRA every 6 months (shorter interval than TGFBR1)"
        ),
        "disease_category": (
            "LDS type 2 — most aggressive LDS; earlier dissection, more extensive aneurysm burden; "
            "BIFID UVULA + HYPERTELORISM (same as LDS1); "
            "osteoarthritis also present (overlap with LDS3/SMAD3); "
            "aortic dissection occurs at smaller diameters AND earlier ages than FBN1; "
            "MANDATORY full-body MRI/MRA — extracranial/intracranial aneurysms also described; "
            "losartan first-line; prophylactic surgery at lower diameter threshold"
        ),
        "disease_pathway": (
            "TGFBR2 binds TGF-β ligand with high affinity and recruits + phosphorylates TGFBR1 (ALK5). "
            "TGFBR2 is constitutively kinase-active (unlike TGFBR1 which requires TGFBR2 activation). "
            "LOF TGFBR2 mutations → same TGF-β paradox as TGFBR1: ELEVATED non-canonical TGF-β signalling "
            "(ERK1/2, JNK, p38) despite LOF. "
            "Aortic smooth muscle cells: TGF-β paradox → MMP-2/MMP-9 overexpression → elastic lamella "
            "fragmentation → aneurysm formation and dissection at smaller diameters. "
            "TGFBR2 splice-site and kinase domain mutations tend to be more severe than missense; "
            "p.Arg460Cys kinase domain: disrupts TGFBR1 substrate phosphorylation — severe phenotype."
        ),
        "pathognomonic": (
            "BIFID UVULA + HYPERTELORISM + AORTIC DISSECTION AT AGE < 30: pathognomonic triad for LDS2. "
            "Full-body MRA showing MULTIPLE aneurysms (aorta + branch + cervical arteries): LDS until proven otherwise. "
            "p.Arg460Cys on sequencing in a young patient with aortic event: diagnostic. "
            "Cervical/intracranial arterial tortuosity + early dissection: distinguishes LDS2 from FBN1."
        ),
        "treatment": (
            "LOSARTAN 1.4 mg/kg/day: TGF-β suppression — first-line; "
            "ATENOLOL 1-2 mg/kg/day: beta-blocker — additive; "
            "PROPHYLACTIC AORTIC ROOT REPLACEMENT: 42-45 mm (LOWER threshold than FBN1); "
            "FULL-BODY MRI/MRA: every 6 months (TGFBR2 — shorter interval than 12 months); "
            "Intracranial imaging: baseline MRA brain; "
            "Cervical spine MRI: baseline (atlantoaxial instability); "
            "Pregnancy: ABSOLUTE HIGH RISK — prophylactic surgery mandatory before pregnancy if root >40 mm; "
            "Celiprolol not studied in LDS; standard beta-blocker preferred; "
            "Avoid all strenuous isometric exercise; no contact sports."
        ),
        "key_features": [
            "Most aggressive LDS gene — EARLIEST dissection (mean 26 years) at SMALLEST diameter",
            "p.Arg460Cys: most common TGFBR2 variant; kinase domain; severe phenotype",
            "Bifid uvula + hypertelorism same as LDS1 (same phenotypic group)",
            "Full-body MRI/MRA every 6 months (SHORTER interval than other HTAD genes)",
            "Extracranial cervical artery tortuosity + intracranial aneurysms possible",
            "Early-onset osteoarthritis (overlap with SMAD3/LDS3)",
            "Lower surgical threshold (42-45 mm) — do NOT wait for 50 mm FBN1 threshold",
            "Pregnancy: highest-risk HTAD gene for maternal mortality from dissection"
        ],
        "key_ddx": [
            "TGFBR1 (LDS1) — clinically identical; craniosynostosis more in TGFBR1; molecular test",
            "FBN1 (MFS) — ectopia lentis; larger diameter at dissection; normal uvula",
            "SMAD3 (AOS/LDS3) — early osteoarthritis; less aggressive arterial disease",
            "Thoracic aortic aneurysm without syndrome — sporadic; no craniofacial features"
        ],
        "onset_age_years_median": 10,
        "aortic_event_pct": 60,
        "beta_blocker_pct": 95,
        "losartan_pct": 92,
        "surgery_required_pct": 65,
        "arterial_tortuosity_pct": 85,
        "vascular_rupture_risk_pct": 45,
        "bilateral_aneurysm_pct": 75,
        "nbs_indicated": False,
    },
    # -- SMAD3 -- Aneurysm-Osteoarthritis syndrome (LDS3) ------------------------------------
    {
        "gene": "SMAD3",
        "alt_name": (
            "SMAD3 (SMAD3-425aa-15q22.33 / AD -- "
            "ANEURYSM-OSTEOARTHRITIS-SYNDROME-AOS-LDS3 -- "
            "EARLY-ONSET-OSTEOARTHRITIS-<30-YEARS-PATHOGNOMONIC -- "
            "TGF-BETA-DOWNSTREAM-SMAD-SIGNALLING -- "
            "AORTIC-PLUS-BRANCH-ANEURYSMS-MILD-CRANIOFACIAL)"
        ),
        "protein": (
            "SMAD3 -- 15q22.33 AD -- SMAD3-425aa -- "
            "SMAD-Family-Member-3-48kDa-TGF-beta-Signalling-Transcription-Factor -- "
            "MH1-Linker-MH2-Domains-Phosphorylation-by-TGFBR1-ALK5 -- "
            "Nuclear-Translocation-SMAD4-Complex-Target-Gene-Expression -- "
            "OMIM-Gene-600993-Disease-AOS-613795"
        ),
        "locus": "15q22.33",
        "protein_size": "425 aa / 48 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous LOF); "
            "Aneurysm-Osteoarthritis syndrome (AOS) / Loeys-Dietz syndrome type 3; "
            "UNIQUE HALLMARK: early-onset osteoarthritis (< 30 years) — pathognomonic; "
            "SMAD3 is downstream of TGFBR1/TGFBR2 in canonical TGF-β signalling cascade; "
            "aortic + peripheral arterial aneurysms (less aggressive than TGFBR1/TGFBR2); "
            "MILD craniofacial features (not prominent bifid uvula/craniosynostosis)"
        ),
        "disease_category": (
            "AOS/LDS3 — SMAD3 downstream TGF-β signalling; "
            "EARLY OSTEOARTHRITIS hallmark distinguishes from all other HTAD genes; "
            "aortic + peripheral aneurysms (subclavian, iliac, renal); "
            "mild bifid uvula (30-40%), hypertelorism (30%); "
            "typically less aggressive than TGFBR1/TGFBR2 — dissection at wider diameters; "
            "TGF-β pathway same downstream logic — losartan reasonable but less evidence"
        ),
        "disease_pathway": (
            "SMAD3 is the canonical downstream transcription factor for TGF-β1/2/3. "
            "TGF-β → TGFBR2 → TGFBR1 (ALK5) → phosphorylates SMAD3 MH2 domain → "
            "SMAD3/SMAD4 heterocomplex → nucleus → activates target genes (collagen, fibronectin, MMP inhibitors). "
            "SMAD3 LOF → paradoxically INCREASED TGF-β target gene expression (same paradox as TGFBR1/2 LOF). "
            "Cartilage: SMAD3 LOF → loss of chondrocyte homeostasis → early-onset OA (< 30 years). "
            "Vasculature: SMAD3 LOF → reduced elastic matrix expression + MMP dysregulation → aortic aneurysm. "
            "This dual phenotype (OA + TAAD) makes SMAD3 mutations the most tissue-specific HTAD gene."
        ),
        "pathognomonic": (
            "AORTIC ANEURYSM + EARLY-ONSET OSTEOARTHRITIS (< 30 YEARS): pathognomonic combination. "
            "Young adult with severe hip/knee OA requiring joint replacement + family history of aortic events. "
            "Bifid uvula present in ~35% (vs ~75% in TGFBR1/TGFBR2) — less reliable sign. "
            "Peripheral arterial aneurysms (abdominal aorta, subclavian, iliac) alongside root aneurysm. "
            "Full-body MRA in a patient with premature OA + TAAD family history — screening rationale."
        ),
        "treatment": (
            "LOSARTAN (0.6-1.4 mg/kg/day): TGF-β suppression — first-line (same rationale as TGFBR1/2); "
            "BETA-BLOCKER (atenolol): additive benefit on aortic wall stress; "
            "PROPHYLACTIC AORTIC SURGERY: at 48-50 mm (intermediate threshold between MFS and LDS1/2); "
            "FULL-BODY MRI/MRA: annual (peripheral aneurysms common); "
            "ORTHOPAEDICS: early OA management — physiotherapy, joint protection, analgesia; "
            "joint replacement for severe OA (hips/knees); orthopaedic review from diagnosis; "
            "Ophthalmology: myopia surveillance; "
            "Avoid isometric exercise; gentle swimming/cycling; "
            "Genetic cascade testing: OA in relatives of TAAD patients → check SMAD3."
        ),
        "key_features": [
            "EARLY OSTEOARTHRITIS < 30 years — pathognomonic hallmark; joint replacement often needed < 40 years",
            "Aortic + peripheral aneurysms (subclavian, renal, iliac) — full-body MRA annually",
            "Mild bifid uvula (~35%) — less prominent than TGFBR1/TGFBR2",
            "Less aggressive arterial disease than LDS1/2 — but still requires lower threshold than FBN1",
            "Intervertebral disc disease + scoliosis common (spinal involvement)",
            "Mild hypertelorism (~30%); camptodactyly; joint laxity",
            "FAMILY HISTORY: premature OA + TAAD in same pedigree — cardinal clue to SMAD3",
            "TGF-β paradox same as TGFBR1/TGFBR2 — downstream transcription factor LOF"
        ],
        "key_ddx": [
            "TGFBR1/TGFBR2 (LDS1/2) — bifid uvula more prominent; NO early OA (OA specific to SMAD3)",
            "FBN1 (MFS) — ectopia lentis; no early OA; Ghent criteria dominant",
            "Familial TAAD (no syndrome) — no OA; no craniofacial; MYH11/ACTA2/MYLK/PRKG1",
            "Juvenile idiopathic arthritis — inflammatory, elevated ESR/CRP; different mechanism"
        ],
        "onset_age_years_median": 30,
        "aortic_event_pct": 50,
        "beta_blocker_pct": 88,
        "losartan_pct": 85,
        "surgery_required_pct": 52,
        "arterial_tortuosity_pct": 68,
        "vascular_rupture_risk_pct": 35,
        "bilateral_aneurysm_pct": 60,
        "nbs_indicated": False,
    },
    # -- ACTA2 -- Multisystemic smooth muscle dysfunction ------------------------------------
    {
        "gene": "ACTA2",
        "alt_name": (
            "ACTA2 (ACTA2-375aa-10q23.31 / AD -- "
            "MULTISYSTEMIC-SMOOTH-MUSCLE-DYSFUNCTION-MSMDS -- "
            "p.Arg179His-TAAD-STROKE-FIXED-DILATED-PUPILS-TRIAD-PATHOGNOMONIC -- "
            "PREMATURE-CORONARY-ARTERY-DISEASE-MOYAMOYA -- "
            "FAMILIAL-TAAD4-OTHER-ACTA2-VARIANTS-MILDER)"
        ),
        "protein": (
            "ACTA2 -- 10q23.31 AD -- ACTA2-375aa -- "
            "Smooth-Muscle-Actin-Alpha2-42kDa-Cytoskeletal-Contractile-Protein -- "
            "G-Actin-Monomer-F-Actin-Filament-Cross-Linked-Myosin-Contraction -- "
            "Vascular-Smooth-Muscle-Cell-Aortic-Media-Dominant-Actin-Isoform -- "
            "OMIM-Gene-102620-Disease-MSMDS-613834-TAAD4-132900"
        ),
        "locus": "10q23.31",
        "protein_size": "375 aa / 42 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous missense); "
            "TWO DISTINCT PHENOTYPES based on mutation site: "
            "(1) p.Arg179His: MULTISYSTEMIC SMOOTH MUSCLE DYSFUNCTION SYNDROME (MSMDS) — "
            "TAAD + stroke + fixed dilated pupils (mydriasis) + pulmonary hypertension + Moyamoya + bowel hypoperistalsis; "
            "(2) Other ACTA2 variants: familial TAAD only — aorta-specific, less severe; "
            "penetrance high but expressivity variable; "
            "ACTA2 is the dominant smooth muscle actin isoform in the aortic media"
        ),
        "disease_category": (
            "ACTA2-related HTAD — heterogeneous; p.Arg179His is the key distinguishing variant; "
            "MSMDS (p.Arg179His): systemic smooth muscle dysfunction — aortic + cerebrovascular + ocular + GI; "
            "premature CAD (by age 40): CORONARY STENTING HIGH-RISK — smooth muscle dysfunction; "
            "Moyamoya disease (cerebral vasculopathy) — carotid artery stenosis; "
            "Familial TAAD4 (other ACTA2): predominantly aortic, milder, longer natural history"
        ),
        "disease_pathway": (
            "ACTA2 encodes vascular smooth muscle α2-actin — the dominant actin isoform in aortic media. "
            "Missense ACTA2 mutations → abnormal actin polymerisation → disorganised F-actin filaments "
            "→ smooth muscle cell dysfunction + inability to maintain vessel wall tension. "
            "p.Arg179His (Arg→His at ATP-binding site): disrupts actin ATPase activity → "
            "smooth muscle cells throughout the body fail to function → multisystem dysfunction. "
            "Aortic media: VSMC dysfunction → reduced tensile strength → progressive dilation + dissection. "
            "Cerebral arteries: VSMC dysfunction → intimal hyperplasia + stenosis (Moyamoya pattern). "
            "Pupils: iris smooth muscle (dilator pupillae) dysfunction → fixed mydriasis. "
            "GI: enteric smooth muscle dysfunction → hypoperistalsis, constipation, pseudo-obstruction."
        ),
        "pathognomonic": (
            "TAAD + STROKE (YOUNG ADULT) + FIXED DILATED PUPILS: pathognomonic MSMDS triad (p.Arg179His). "
            "Fixed mydriasis (dilated pupils not reactive to light) in infancy + family history of aortic events. "
            "Moyamoya pattern on MRA (bilateral carotid/MCA stenosis) in HTAD patient → ACTA2 p.Arg179His. "
            "Straight aorta (loss of normal curvature) on CTA — aortic smooth muscle dysfunction. "
            "Premature CAD (STEMI < 40 years) in Marfanoid patient without FBN1 → ACTA2 screening."
        ),
        "treatment": (
            "MSMDS (p.Arg179His): "
            "ANTIHYPERTENSIVE therapy (beta-blockers — atenolol preferred, ARBs); "
            "Aortic surgery threshold: 45 mm (lower than FBN1; smooth muscle dysfunction accelerates progression); "
            "NEUROSURGICAL: Moyamoya revascularisation (EC-IC bypass) if cerebral ischaemia; "
            "OPHTHALMOLOGY: fixed mydriasis — glare protection, dilated fundal exam annually; "
            "CARDIOLOGY: early coronary artery screening (calcium score CT from age 30); "
            "Avoid vasopressors intraoperatively (VSMC dysfunction → unpredictable response); "
            "GI: osmotic laxatives; prokinetics; NG tube/parenteral nutrition in pseudo-obstruction; "
            "Familial TAAD4 (other ACTA2): annual echo + CT/MRI; same beta-blocker/ARB therapy."
        ),
        "key_features": [
            "p.Arg179His MSMDS: TAAD + stroke + fixed dilated pupils = pathognomonic triad",
            "Premature CAD (< 40 years) — coronary VSMC dysfunction; stenting HIGH-RISK",
            "Moyamoya disease — carotid/MCA stenosis + intracranial collateralisation on MRA",
            "Fixed mydriasis (dilated, non-reactive pupils) from iris smooth muscle dysfunction",
            "Hypoperistalsis / intestinal pseudo-obstruction — GI smooth muscle dysfunction",
            "Pulmonary hypertension — pulmonary artery smooth muscle dysfunction",
            "Straight aorta (loss of curvature on CTA) — vascular smooth muscle sign",
            "Other ACTA2 variants: aorta-predominant; milder; similar surveillance to FBN1"
        ],
        "key_ddx": [
            "FBN1 (MFS) — ectopia lentis; no stroke/Moyamoya; no fixed mydriasis",
            "TGFBR1/TGFBR2 (LDS) — bifid uvula; no premature CAD; no mydriasis",
            "Moyamoya disease (RNF213) — no aortic involvement; no smooth muscle dysfunction systemically",
            "Marfan-like phenotype without FBN1 mutation — screen ACTA2 + MYH11 + TGFBR2"
        ],
        "onset_age_years_median": 40,
        "aortic_event_pct": 45,
        "beta_blocker_pct": 85,
        "losartan_pct": 68,
        "surgery_required_pct": 42,
        "arterial_tortuosity_pct": 35,
        "vascular_rupture_risk_pct": 30,
        "bilateral_aneurysm_pct": 40,
        "nbs_indicated": False,
    },
    # -- MYH11 -- Familial TAAD + PDA --------------------------------------------------------
    {
        "gene": "MYH11",
        "alt_name": (
            "MYH11 (MYH11-1972aa-16p13.11 / AD -- "
            "FAMILIAL-TAAD-PLUS-PATENT-DUCTUS-ARTERIOSUS-PATHOGNOMONIC -- "
            "SMOOTH-MUSCLE-MYOSIN-HEAVY-CHAIN-AORTIC-MEDIA -- "
            "RARE-2pct-HTAD-ECHOCARDIOGRAPHY-ANNUAL -- "
            "AORTIC-SMALLER-DIAMETER-AT-DISSECTION-THAN-FBN1)"
        ),
        "protein": (
            "MYH11 -- 16p13.11 AD -- MYH11-1972aa -- "
            "Smooth-Muscle-Myosin-Heavy-Chain-227kDa-Motor-Domain-Lever-Arm-Coiled-Coil -- "
            "Acto-Myosin-Cross-Bridge-Cycle-ATP-Hydrolysis-Smooth-Muscle-Contraction -- "
            "Aorta-Smooth-Muscle-Cell-Dominant-Myosin-Isoform -- "
            "OMIM-Gene-160745-Disease-TAAD4-132900"
        ),
        "locus": "16p13.11",
        "protein_size": "1972 aa / 227 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous missense/LOF); "
            "FAMILIAL TAAD4 — specifically aortic aneurysm/dissection; "
            "PATENT DUCTUS ARTERIOSUS (PDA) in the same pedigree: PATHOGNOMONIC pairing; "
            "rare gene — ~2% of heritable TAAD families; "
            "MYH11 encodes smooth muscle myosin heavy chain (SM-MHC); "
            "smooth muscle cell contraction impaired → reduced aortic wall tensile force; "
            "aortic diameter at dissection: smaller than FBN1 (closer to TGFBR1)"
        ),
        "disease_category": (
            "MYH11-related TAAD — smooth muscle myosin heavy chain; "
            "TAAD + PDA combination in pedigree: high specificity for MYH11; "
            "isolated TAAD (no PDA) also possible but less specific; "
            "rare — often underdiagnosed because PDA corrected in childhood, history not sought; "
            "smooth muscle cell dysfunction: same mechanism as ACTA2 but myosin not actin; "
            "echocardiographic surveillance annually + CT/MRI for branch vessels"
        ),
        "disease_pathway": (
            "MYH11 (smooth muscle myosin heavy chain II) pairs with smooth muscle actin (ACTA2) "
            "to form the contractile apparatus of vascular smooth muscle cells. "
            "MYH11 LOF → impaired cross-bridge cycling (ATP hydrolysis + actin binding) → "
            "reduced VSMC contractility → reduced aortic wall tension during systole → "
            "progressive dilation under pulsatile pressure. "
            "PDA: ductus arteriosus closure depends on smooth muscle contraction of the ductal wall after birth; "
            "MYH11 dysfunction → insufficient VSMC contraction → ductal obliteration fails → PDA. "
            "The simultaneous PDA (fixed in childhood) + aortic aneurysm (manifests in adulthood) "
            "explains why MYH11 is underdiagnosed — the PDA clue is lost."
        ),
        "pathognomonic": (
            "FAMILY HISTORY OF PDA + TAAD IN SAME PEDIGREE: pathognomonic for MYH11. "
            "Adult presenting with TAAD + personal or family history of PDA closure in childhood: "
            "screen MYH11 immediately. "
            "Aortic dissection at relatively small diameter (< 50 mm) in patient without MFS features: "
            "MYH11 in differential. "
            "Concurrent PDA in an adult with TAAD (before correction): pathognomonic."
        ),
        "treatment": (
            "BETA-BLOCKER (atenolol): reduces aortic wall stress; "
            "LOSARTAN / ARB: reasonable by analogy with ACTA2/FBN1; "
            "PROPHYLACTIC AORTIC ROOT REPLACEMENT: 45 mm threshold (not 50 mm); "
            "ECHOCARDIOGRAPHY: annually; "
            "CT AORTA: every 2-3 years (or sooner if growth detected on echo); "
            "FAMILY SCREENING: specifically ask about PDA history in relatives — critical clue; "
            "PDA repair in at-risk family members (detected before spontaneous closure); "
            "Avoid strenuous isometric exercise; "
            "Pregnancy: increase surveillance frequency; "
            "Annual cardiology review throughout life."
        ),
        "key_features": [
            "PDA + TAAD pedigree: pathognomonic — the PDA clue is often missed if corrected in childhood",
            "Rare gene (~2% heritable TAAD) — specifically seek family history of PDA in TAAD patients",
            "Smooth muscle myosin heavy chain — contractile apparatus partner of ACTA2",
            "Aortic dissection at smaller diameter than FBN1 (~ 46 mm mean in reported cases)",
            "No systemic features (no Marfanoid habitus, no ocular involvement)",
            "Annual echocardiography from diagnosis",
            "16p13.11 — same chromosome arm as CREBBP (Rubinstein-Taybi); distinct loci",
            "Heterozygous missense in motor domain or coiled-coil: dominant negative mechanism likely"
        ],
        "key_ddx": [
            "ACTA2 — also smooth muscle protein; MSMDS (p.Arg179His) has systemic features; no PDA association",
            "FBN1 (MFS) — ectopia lentis; Marfanoid; dissection at larger diameter; no PDA",
            "Sporadic PDA (non-genetic) — no TAAD family history; no MYH11 mutation",
            "Turner syndrome — aortic coarctation + bicuspid aortic valve; 45,X karyotype"
        ],
        "onset_age_years_median": 35,
        "aortic_event_pct": 40,
        "beta_blocker_pct": 83,
        "losartan_pct": 62,
        "surgery_required_pct": 38,
        "arterial_tortuosity_pct": 22,
        "vascular_rupture_risk_pct": 25,
        "bilateral_aneurysm_pct": 28,
        "nbs_indicated": False,
    },
    # -- COL3A1 -- Vascular Ehlers-Danlos syndrome (vEDS) ------------------------------------
    {
        "gene": "COL3A1",
        "alt_name": (
            "COL3A1 (COL3A1-1466aa-2q32.2 / AD -- "
            "VASCULAR-EDS-vEDS-MOST-LETHAL-CONNECTIVE-TISSUE-DISORDER -- "
            "SPONTANEOUS-ARTERIAL-RUPTURE-WITHOUT-ANEURYSM-PATHOGNOMONIC -- "
            "SURGERY-CONTRAINDICATED-VESSEL-FRIABILITY-MORTALITY->50pct -- "
            "CELIPROLOL-400mg-REDUCES-EVENTS-LEVEL-A-RCT)"
        ),
        "protein": (
            "COL3A1 -- 2q32.2 AD -- COL3A1-1466aa -- "
            "Collagen-Type-III-Alpha-1-Chain-139kDa-Gly-X-Y-Triple-Helix -- "
            "Type-III-Procollagen-Homotrimer-Fibril-Cross-Linked-ECM-Scaffold -- "
            "Vessel-Wall-Skin-Hollow-Organ-Dominant-Collagen-Isoform -- "
            "OMIM-Gene-120180-Disease-vEDS-130050"
        ),
        "locus": "2q32.2",
        "protein_size": "1466 aa / 139 kDa",
        "inheritance": (
            "AD (autosomal dominant — heterozygous Gly missense or splice LOF dominant-negative); "
            "Vascular EDS (vEDS) — most lethal form of Ehlers-Danlos syndrome; "
            "SPONTANEOUS ARTERIAL RUPTURE is the hallmark — often WITHOUT pre-existing aneurysm; "
            "SURGERY IS CONTRAINDICATED (vessels extremely friable — any manipulation causes lethal haemorrhage); "
            "BOWEL RUPTURE + UTERINE RUPTURE in pregnancy also pathognomonic; "
            "mean age at first complication: 29 years (range 6-70 years); "
            "Gly→Arg or Gly→Val substitutions in triple helix → dominant negative trimer disruption"
        ),
        "disease_category": (
            "vEDS — type III collagen deficiency; most dangerous connective tissue disorder; "
            "SPONTANEOUS RUPTURE without warning: aorta, medium arteries (splenic, hepatic, renal), "
            "hollow organs (colon, uterus); "
            "SKIN features: thin, translucent (veins visible), easy bruising, velvety texture; "
            "FACIAL features: thin lips, small chin, pinched nose, prominent eyes — characteristic; "
            "CELIPROLOL 400 mg/day: reduces arterial events (BBEST trial — Level A RCT evidence); "
            "vascular surgery AVOIDED unless absolutely life-saving (extreme operative mortality)"
        ),
        "disease_pathway": (
            "Type III collagen (COL3A1 homotrimer) is the dominant collagen of arterial walls, skin, "
            "and hollow organs. Gly-X-Y triple helix disruption (dominant-negative missense) → "
            "abnormal type III procollagen that POISONS normal chain incorporation → "
            "net severe reduction in functional type III collagen in all tissues. "
            "Arterial wall: type III collagen provides tensile strength to adventitia + media; "
            "without it → vessels rupture spontaneously under normal blood pressure. "
            "Hollow organs: bowel wall + uterus rely on type III collagen → spontaneous perforation. "
            "Unlike TAAD where vessels dilate gradually before rupture, vEDS vessels rupture without dilation — "
            "this is why surgery is so dangerous: every suture line tears."
        ),
        "pathognomonic": (
            "SPONTANEOUS ARTERIAL RUPTURE IN A YOUNG ADULT WITHOUT ANEURYSM: pathognomonic vEDS. "
            "Thin, translucent skin + visible veins + characteristic facies (thin lips/pinched nose) + "
            "family history of early vascular death → vEDS until proven otherwise. "
            "Spontaneous bowel perforation (sigmoid most common) in young adult — colonoscopy CONTRAINDICATED. "
            "Uterine rupture during pregnancy in a young woman with family history. "
            "Type III collagen immunostaining reduced on skin biopsy (fibroblast culture): diagnostic. "
            "Biochemistry: absent type III collagen bands on electrophoresis of fibroblast culture."
        ),
        "treatment": (
            "CELIPROLOL 400 mg/day: cardioselective beta-blocker with vasodilatory activity; "
            "REDUCES ARTERIAL EVENTS 36% vs placebo (BBEST RCT, Ong 2010); "
            "CONTRAINDICATIONS: vascular SURGERY (extreme operative mortality > 50%); "
            "colonoscopy AVOIDED (hollow organ perforation risk); "
            "CT angiography PREFERRED over invasive angiography; "
            "EMERGENCY HAEMORRHAGE: IR (interventional radiology) endovascular approach preferred; "
            "Pregnancy: ABSOLUTE HIGH RISK — maternal mortality 10-15% per pregnancy; "
            "advise against pregnancy; if pregnant, plan caesarean section at 34-36 weeks; "
            "Avoid NSAIDs (antiplatelet effect); "
            "Medical alert bracelet (surgical teams must know vEDS diagnosis BEFORE any operation); "
            "Annual MRI/CTA for surveillance (if stable — not standard in all centres)."
        ),
        "key_features": [
            "SPONTANEOUS ARTERIAL RUPTURE without aneurysm — NO warning, NO dilation before rupture",
            "SURGERY CONTRAINDICATED — operative mortality > 50% from vessel friability",
            "CELIPROLOL 400 mg/day: ONLY evidence-based treatment (Level A RCT) — reduces events 36%",
            "Thin, translucent skin — veins visible through skin; characteristic facies",
            "Bowel perforation (sigmoid colon) — spontaneous; colonoscopy AVOIDED",
            "Uterine rupture during pregnancy — maternal mortality 10-15%; advise against pregnancy",
            "Type III collagen absent/abnormal on fibroblast culture — biochemical confirmation",
            "Medical alert bracelet MANDATORY — surgical team must know vEDS before any procedure"
        ],
        "key_ddx": [
            "FBN1 (MFS) — aortic root aneurysm; ectopia lentis; surgery NOT contraindicated",
            "Classic EDS (COL5A1/COL5A2) — hypermobility; less vascular catastrophe; surgery safe",
            "Kyphoscoliotic EDS (PLOD1) — scoliosis + ocular fragility; AR; rupture less common",
            "Spontaneous aortic dissection (sporadic) — no skin/facies features; check COL3A1"
        ],
        "onset_age_years_median": 30,
        "aortic_event_pct": 35,
        "beta_blocker_pct": 92,
        "losartan_pct": 38,
        "surgery_required_pct": 18,
        "arterial_tortuosity_pct": 40,
        "vascular_rupture_risk_pct": 75,
        "bilateral_aneurysm_pct": 45,
        "nbs_indicated": False,
    },
    # -- SLC2A10 -- Arterial tortuosity syndrome (ATS, AR) ------------------------------------
    {
        "gene": "SLC2A10",
        "alt_name": (
            "SLC2A10 (SLC2A10-541aa-20q13.12 / AR -- "
            "ARTERIAL-TORTUOSITY-SYNDROME-ATS -- "
            "100pct-ARTERIAL-TORTUOSITY-UNIVERSAL-DIAGNOSTIC -- "
            "GLUT10-DEHYDROASCORBIC-ACID-TRANSPORTER-MITOCHONDRIAL -- "
            "NEONATAL-INFANTILE-PRESENTATION-BOTH-SEXES-EQUAL)"
        ),
        "protein": (
            "SLC2A10 -- 20q13.12 AR -- SLC2A10-541aa -- "
            "GLUT10-Glucose-Transporter-10-57kDa-12-TM-Helix-Major-Facilitator-Superfamily -- "
            "Dehydroascorbic-Acid-DHA-Transporter-Mitochondrial-Inner-Membrane -- "
            "Ascorbate-Recycling-Collagen-Hydroxylation-TGF-beta-Suppression -- "
            "OMIM-Gene-606145-Disease-ATS-208050"
        ),
        "locus": "20q13.12",
        "protein_size": "541 aa / 57 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF); "
            "BOTH SEXES equally affected; "
            "Arterial Tortuosity Syndrome (ATS) — neonatal/infantile presentation; "
            "100% ARTERIAL TORTUOSITY: universal, diagnostic — aorta, pulmonary, carotid, renal, coronary; "
            "stenosis + aneurysm of major arteries; "
            "skin hyperextensibility + joint hypermobility (milder EDS-like features); "
            "GLUT10 transports dehydroascorbic acid (DHA = oxidised ascorbate) into mitochondria; "
            "DHA deficiency → impaired collagen hydroxylation + TGF-β overactivity"
        ),
        "disease_category": (
            "Arterial Tortuosity Syndrome — AR HTAD; "
            "UNIVERSAL ARTERIAL TORTUOSITY: 100% affected arteries throughout the body; "
            "PULMONARY ARTERY STENOSIS in ~50% (may cause RV failure in infancy); "
            "facial features: elongated face, high palate, micrognathia, blue sclerae; "
            "lax skin + joint hypermobility (EDS-like); "
            "DHA → mitochondrial ascorbate recycling → maintains prolyl and lysyl hydroxylase activity "
            "→ collagen crosslinking; GLUT10 LOF → reduced mitochondrial DHA → collagen/elastin defect"
        ),
        "disease_pathway": (
            "GLUT10 (SLC2A10) transports dehydroascorbic acid (DHA) across the mitochondrial inner membrane. "
            "Inside mitochondria, DHA is reduced back to ascorbate (vitamin C) by glutaredoxins/thioredoxins. "
            "Mitochondrial ascorbate is required as cofactor for: "
            "(1) Prolyl-4-hydroxylase: hydroxylates collagen Gly-X-Pro → Gly-X-Hyp (triple helix stability); "
            "(2) Lysyl hydroxylase: hydroxylates collagen Lys → Hyl (crosslink precursor); "
            "GLUT10 LOF → mitochondrial ascorbate deficiency → reduced collagen hydroxylation → "
            "unstable triple helices + reduced crosslinks → weak arterial walls. "
            "Additionally: GLUT10 LOF → elevated TGF-β activity (ascorbate normally suppresses TGF-β signalling) "
            "→ MMP upregulation → arterial matrix degradation → tortuosity and stenosis."
        ),
        "pathognomonic": (
            "UNIVERSAL ARTERIAL TORTUOSITY (aorta + all major vessels) IN A NEONATE/INFANT: pathognomonic. "
            "Pulmonary artery stenosis on echocardiography + aortic tortuosity: ATS until excluded. "
            "Facial gestalt (elongated face, blue sclerae, micrognathia) + tortuosity. "
            "EDS-like skin laxity + blue sclerae in a neonate with cardiovascular compromise. "
            "Molecular confirmation: biallelic SLC2A10 pathogenic variants."
        ),
        "treatment": (
            "No disease-specific therapy approved (ascorbate supplementation theoretical — limited evidence); "
            "HIGH-DOSE ASCORBATE (vitamin C 0.5-1 g/day): compensates for DHA deficiency — used empirically; "
            "LOSARTAN: TGF-β pathway suppression — used by analogy with TGFBR/FBN1; limited evidence; "
            "PULMONARY ARTERY STENOSIS: balloon angioplasty/stenting for severe RV obstruction; "
            "AORTIC ANEURYSM: surgical/endovascular repair — higher-risk than standard HTAD surgery (vessel tortuosity); "
            "CARDIOLOGICAL surveillance: echocardiography + CT/MRI annually; "
            "OPHTHALMOLOGY: ectopia lentis and retinal detachment can occur; "
            "Skin protection: sun cream, trauma avoidance (easy bruising); "
            "Genetic counselling: AR inheritance — carrier parents, sibling risk 25%."
        ),
        "key_features": [
            "UNIVERSAL ARTERIAL TORTUOSITY — 100% of arteries affected; aorta + pulmonary + renal + carotid",
            "AR inheritance — BOTH SEXES equally affected (contrast with X-linked HTAD genes)",
            "Neonatal/infantile onset — pulmonary stenosis may cause RV failure in infancy",
            "Pulmonary artery stenosis (~50%) — balloon angioplasty/stenting for severe cases",
            "GLUT10 = DHA transporter → mitochondrial ascorbate → collagen hydroxylation",
            "Facial gestalt: elongated face, blue sclerae, micrognathia, high palate",
            "EDS-like skin laxity + joint hypermobility — connective tissue weakness",
            "High-dose ascorbate supplementation: empirical treatment based on mechanism"
        ],
        "key_ddx": [
            "FBN1 (MFS) — tortuosity present but NOT universal; ectopia lentis; AD",
            "TGFBR1/TGFBR2 (LDS) — tortuosity + bifid uvula; AD; no universal stenosis",
            "Cutis laxa (ELN/FBLN4/FBLN5) — skin laxity dominant; less severe arterial tortuosity",
            "Congenital rubella syndrome — TORCH serology; no genetic cause; not heritable"
        ],
        "onset_age_years_median": 5,
        "aortic_event_pct": 30,
        "beta_blocker_pct": 68,
        "losartan_pct": 55,
        "surgery_required_pct": 35,
        "arterial_tortuosity_pct": 100,
        "vascular_rupture_risk_pct": 20,
        "bilateral_aneurysm_pct": 55,
        "nbs_indicated": False,
    },
]


def _make_patients(gene_entry, n=40, seed=None):
    rng = random.Random(seed)
    patients = []
    for i in range(n):
        onset_age = max(0, int(rng.gauss(
            gene_entry["onset_age_years_median"] * 12,
            gene_entry["onset_age_years_median"] * 12 * 0.4 + 6
        )))  # in months
        had_event = rng.random() < gene_entry["aortic_event_pct"] / 100
        patients.append({
            "patient_id": f"{gene_entry['gene']}-{seed}-{i:03d}",
            "gene": gene_entry["gene"],
            "onset_age_months": onset_age,
            "aortic_event": had_event,
            "on_beta_blocker": rng.random() < gene_entry["beta_blocker_pct"] / 100,
            "on_losartan": rng.random() < gene_entry["losartan_pct"] / 100,
            "required_surgery": rng.random() < gene_entry["surgery_required_pct"] / 100,
            "arterial_tortuosity": rng.random() < gene_entry["arterial_tortuosity_pct"] / 100,
            "vascular_rupture": rng.random() < gene_entry["vascular_rupture_risk_pct"] / 100,
            "bilateral_aneurysm": rng.random() < gene_entry["bilateral_aneurysm_pct"] / 100,
            "aortic_diameter_mm": round(rng.uniform(38, 58) if had_event else rng.uniform(30, 50), 1),
            "outcome": "deceased" if (had_event and rng.random() < 0.12) else rng.choice(
                ["stable", "post-surgery", "active-surveillance", "dissection-survived"]
            ),
        })
    return patients


def generate_overview():
    all_patients = []
    gene_summaries = []
    for idx, entry in enumerate(HTAD_GENES):
        pts = _make_patients(entry, n=40, seed=SEED_BASE + idx)
        all_patients.extend(pts)
        gene_summaries.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_name": entry["disease_category"].split(" —")[0].strip()[:80],
            "pathognomonic_short": entry["pathognomonic"].split(". ")[0][:120],
            "onset_age_years_median": entry["onset_age_years_median"],
            "aortic_event_pct": entry["aortic_event_pct"],
            "beta_blocker_pct": entry["beta_blocker_pct"],
            "losartan_pct": entry["losartan_pct"],
            "surgery_required_pct": entry["surgery_required_pct"],
            "arterial_tortuosity_pct": entry["arterial_tortuosity_pct"],
            "vascular_rupture_risk_pct": entry["vascular_rupture_risk_pct"],
            "avg_aortic_diameter": round(
                sum(p["aortic_diameter_mm"] for p in pts if p["aortic_event"]) /
                max(1, sum(1 for p in pts if p["aortic_event"])), 1
            ),
            "surgery_actual_pct": round(sum(1 for p in pts if p["required_surgery"]) / 40 * 100, 1),
        })

    agg = {
        "aortic_event_pct": round(sum(1 for p in all_patients if p["aortic_event"]) / 320 * 100, 1),
        "surgery_pct": round(sum(1 for p in all_patients if p["required_surgery"]) / 320 * 100, 1),
        "on_beta_blocker_pct": round(sum(1 for p in all_patients if p["on_beta_blocker"]) / 320 * 100, 1),
        "on_losartan_pct": round(sum(1 for p in all_patients if p["on_losartan"]) / 320 * 100, 1),
        "arterial_tortuosity_pct": round(sum(1 for p in all_patients if p["arterial_tortuosity"]) / 320 * 100, 1),
        "vascular_rupture_pct": round(sum(1 for p in all_patients if p["vascular_rupture"]) / 320 * 100, 1),
        "bilateral_aneurysm_pct": round(sum(1 for p in all_patients if p["bilateral_aneurysm"]) / 320 * 100, 1),
    }

    return {
        "title": "Hereditary Thoracic Aortic Disease Atlas",
        "subtitle": "Complete 8-Gene Hereditary Thoracic Aortic Disease & Connective Tissue Aortopathy Reference",
        "genes": [e["gene"] for e in HTAD_GENES],
        "n_genes": 8,
        "total_patients": 320,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "disease_classes": [
            "FBN1 — Marfan syndrome; aortic root + ectopia lentis + tall stature triad; most common HTAD gene",
            "TGFBR1 — LDS1; bifid uvula + craniosynostosis; aggressive; lower surgery threshold",
            "TGFBR2 — LDS2; most aggressive; dissection mean age 26 years; full-body MRA every 6 months",
            "SMAD3 — LDS3/AOS; early-onset osteoarthritis pathognomonic; aortic + branch vessel aneurysms",
            "ACTA2 — MSMDS (p.Arg179His): TAAD + stroke + fixed dilated pupils; premature CAD + Moyamoya",
            "MYH11 — Familial TAAD + patent ductus arteriosus; rare; smooth muscle myosin heavy chain",
            "COL3A1 — vascular EDS; spontaneous rupture WITHOUT aneurysm; surgery CONTRAINDICATED; celiprolol",
            "SLC2A10 — arterial tortuosity syndrome; AR; 100% tortuosity; neonatal onset; DHA transporter",
        ],
        "gene_summary": gene_summaries,
        "aggregate_metrics": agg,
        "clinical_pearls": [
            "FBN1 (MFS): prophylactic aortic root replacement at 50 mm (45 mm if rapid growth or family history); "
            "beta-blocker + losartan LIFELONG — slows but does NOT halt progression; ectopia lentis superotemporal "
            "(opposite to homocystinuria).",
            "TGFBR1/TGFBR2 (LDS1/2): EXAMINE THE UVULA in ALL TAAD patients — bifid uvula is a 30-second finding "
            "that CHANGES management (lower surgical threshold + full-body MRA). Never miss it.",
            "TGFBR2 (LDS2): MOST AGGRESSIVE HTAD gene — dissection at mean 26 years; full-body MRI/MRA every "
            "6 MONTHS (not annually); prophylactic surgery at 42-45 mm (NOT 50 mm).",
            "SMAD3 (LDS3/AOS): YOUNG ADULT WITH OSTEOARTHRITIS NEEDING JOINT REPLACEMENT + AORTIC ANEURYSM "
            "in same pedigree — screen SMAD3; early OA is the PATHOGNOMONIC hallmark.",
            "ACTA2 p.Arg179His: MSMDS TRIAD — TAAD + stroke (young adult) + FIXED DILATED PUPILS = "
            "pathognomonic; Moyamoya + premature CAD also part of syndrome; pupil exam in all HTAD patients.",
            "MYH11: ASK ABOUT PATENT DUCTUS ARTERIOSUS — PDA corrected in childhood is the lost clue; "
            "MYH11 TAAD + PDA pedigree is pathognomonic; rare (~2% heritable TAAD).",
            "COL3A1 (vEDS): SURGERY IS CONTRAINDICATED (> 50% operative mortality from vessel friability); "
            "CELIPROLOL 400 mg/day is the ONLY evidence-based treatment (Level A RCT — BBEST trial); "
            "COLONOSCOPY AVOIDED; MEDICAL ALERT BRACELET MANDATORY.",
            "SLC2A10 (ATS): ONLY AR gene in this panel; 100% ARTERIAL TORTUOSITY universal; neonatal onset; "
            "pulmonary artery stenosis in 50%; high-dose ascorbate (DHA transporter defect) — empirical therapy.",
            "TGF-β PARADOX applies to FBN1/TGFBR1/TGFBR2/SMAD3: LOF mutations INCREASE TGF-β activity "
            "(paradox) via non-canonical pathways → LOSARTAN (ARB) blocks AT1R → reduces TGF-β signalling "
            "→ slows aortic growth in all four genes.",
            "FULL-BODY MRI/MRA mandatory for TGFBR1, TGFBR2, SMAD3 — aortic root echo ALONE misses "
            "subclavian, renal, iliac, mesenteric aneurysms that can dissect independently of the aortic root.",
        ],
    }


def generate_breakdown():
    breakdowns = []
    for idx, entry in enumerate(HTAD_GENES):
        pts = _make_patients(entry, n=40, seed=SEED_BASE + idx)
        breakdowns.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "n_patients": 40,
            "aortic_event_pct": round(sum(1 for p in pts if p["aortic_event"]) / 40 * 100, 1),
            "surgery_pct": round(sum(1 for p in pts if p["required_surgery"]) / 40 * 100, 1),
            "on_beta_blocker_pct": round(sum(1 for p in pts if p["on_beta_blocker"]) / 40 * 100, 1),
            "on_losartan_pct": round(sum(1 for p in pts if p["on_losartan"]) / 40 * 100, 1),
            "arterial_tortuosity_pct": round(sum(1 for p in pts if p["arterial_tortuosity"]) / 40 * 100, 1),
            "vascular_rupture_pct": round(sum(1 for p in pts if p["vascular_rupture"]) / 40 * 100, 1),
            "avg_aortic_diameter_mm": round(sum(p["aortic_diameter_mm"] for p in pts) / 40, 1),
            "avg_onset_age_months": round(sum(p["onset_age_months"] for p in pts) / 40, 1),
        })
    return {"gene_breakdowns": breakdowns}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["alt_name"].split(" (")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:600],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "aortic_event_pct": entry["aortic_event_pct"],
                "vascular_rupture_risk_pct": entry["vascular_rupture_risk_pct"],
                "surgery_required_pct": entry["surgery_required_pct"],
                "nbs_indicated": entry["nbs_indicated"],
            }
            for entry in HTAD_GENES
        },
        "htad_glossary": {
            "TGF-β Paradox in Hereditary TAAD": (
                "LOF mutations in TGF-β pathway genes (FBN1, TGFBR1, TGFBR2, SMAD3) INCREASE TGF-β signalling "
                "rather than reduce it. Mechanism: canonical SMAD2/3 pathway is reduced, but non-canonical "
                "pathways (ERK1/2, JNK, p38 MAPK) are UPREGULATED. These non-canonical pathways drive MMP "
                "expression, SMC apoptosis, and elastic lamella destruction. "
                "LOSARTAN (AT1R blocker) inhibits angiotensin II signalling → reduces TGF-β release and activity "
                "→ attenuates non-canonical pathway upregulation → slows aortic growth. "
                "This is why ARBs are first-line in FBN1/LDS/AOS despite the LOF paradox."
            ),
            "LDS — Loeys-Dietz Syndrome Types and Surgery Thresholds": (
                "LDS1 (TGFBR1): prophylactic surgery at 45 mm (42 mm with risk factors). "
                "LDS2 (TGFBR2): prophylactic surgery at 42-45 mm; full-body MRA every 6 months. "
                "LDS3 (SMAD3): surgery at 48-50 mm; early OA pathognomonic. "
                "MFS (FBN1): surgery at 50 mm (45 mm rapid growth). "
                "NEVER apply FBN1 thresholds to LDS — dissection occurs at SMALLER diameters in LDS. "
                "Examine the uvula in ALL TAAD patients: bifid uvula = LDS until proven otherwise. "
                "Full-body MRI/MRA mandatory in ALL LDS subtypes — isolated aortic surveillance is INSUFFICIENT."
            ),
            "vEDS (COL3A1) — Surgical Contraindication and Celiprolol": (
                "Vascular EDS is the ONLY HTAD where vascular surgery is CONTRAINDICATED. "
                "Type III collagen (COL3A1) provides arterial wall tensile strength — its absence → "
                "every suture tears, every clamp causes haemorrhage → operative mortality > 50%. "
                "CELIPROLOL (beta-1/2 agonist + beta-1 antagonist): BBEST RCT (Ong 2010) showed "
                "36% reduction in arterial events vs placebo — only evidence-based treatment. "
                "Mechanism: celiprolol → reduced wall stress + possible direct collagen-stabilising effect. "
                "Standard beta-blockers (atenolol) NOT shown to benefit in vEDS. "
                "COLONOSCOPY and INVASIVE angiography AVOIDED — hollow organ/arterial perforation risk."
            ),
            "ACTA2 p.Arg179His — MSMDS Triad and Moyamoya": (
                "ACTA2 mutations cause TWO distinct syndromes: "
                "(1) p.Arg179His → MSMDS: TAAD + stroke + fixed dilated pupils + premature CAD + Moyamoya; "
                "(2) Other ACTA2 variants → familial TAAD only (no systemic features). "
                "p.Arg179 is the ATP-binding site of smooth muscle alpha-actin — disrupts ALL smooth muscle cells. "
                "FIXED DILATED PUPILS (mydriasis): iris dilator smooth muscle dysfunction — examine pupils in all TAAD. "
                "MOYAMOYA: carotid/MCA intimal hyperplasia → progressive occlusion → collateral formation → "
                "stroke risk; EC-IC bypass surgery if ischaemic symptoms. "
                "PREMATURE CAD (< 40 years): coronary artery stenting high-risk in MSMDS (VSMC dysfunction)."
            ),
            "MYH11 and Patent Ductus Arteriosus — The Lost Clue": (
                "MYH11 (smooth muscle myosin heavy chain) mutations cause familial TAAD + PDA. "
                "PDA is the PATHOGNOMONIC pairing — but it is often corrected in childhood, "
                "removing the clinical clue before the TAAD manifests in adulthood. "
                "CLINICAL STRATEGY: in every TAAD patient without FBN1/LDS features, "
                "SPECIFICALLY ASK about PDA repair in childhood (self or family members). "
                "A family history of PDA repair + TAAD events → screen MYH11. "
                "Smooth muscle myosin pairs with ACTA2 — functionally related; together account for ~4-6% of heritable TAAD."
            ),
            "Arterial Tortuosity Syndrome (SLC2A10) — DHA and Ascorbate": (
                "GLUT10 (SLC2A10) is a dehydroascorbic acid (DHA = oxidised vitamin C) transporter. "
                "GLUT10 transports DHA into mitochondria where it is reduced back to ascorbate. "
                "Mitochondrial ascorbate is a cofactor for prolyl-4-hydroxylase and lysyl hydroxylase — "
                "essential for collagen triple-helix stability and crosslinking. "
                "GLUT10 LOF → mitochondrial ascorbate deficiency → undermethylated collagen → weak ECM. "
                "Additionally, GLUT10 LOF → elevated TGF-β activity → MMP-driven arterial remodelling → tortuosity. "
                "100% of arteries are tortuous in ATS — a universal sign (unlike FBN1/LDS where tortuosity is partial). "
                "Empirical ascorbate supplementation: basis is the DHA transporter defect; formal RCT lacking."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (FBN1) ===")
    defn = generate_definitions()
    print(json.dumps(defn["gene_entries"]["FBN1"], indent=2)[:1500])
