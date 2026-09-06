#!/usr/bin/env python3
"""Hereditary-Connective-Tissue-Atlas — Complete 8-Gene Hereditary Connective Tissue Disorder Atlas
FBN1    (Fibrillin-1; 2871 aa; ~350 kDa; 15q21.1; AD;
         OMIM gene 134797; Marfan syndrome OMIM 154700;
         most common hereditary connective tissue disorder; 1:5,000 births;
         aortic root aneurysm + ectopia lentis + skeletal features (arm span > height);
         beta-blocker + losartan; elective aortic root replacement ≥50 mm;
         NO contact sports / isometric exercise; Ghent criteria 2010;
         seed SEED_BASE+0) ·
TGFBR1  (TGF-β Receptor Type 1 / ALK5; 503 aa; ~59 kDa; 9q22.33; AD;
         OMIM gene 190181; Loeys-Dietz Syndrome 1 OMIM 609192;
         aggressive aneurysm at smaller diameters — prophylactic surgery ≥42 mm (not ≥50 mm);
         triad: bifid uvula + hypertelorism + arterial tortuosity; head-to-pelvis MRA mandatory;
         losartan primary; beta-blocker adjunct;
         seed SEED_BASE+1) ·
TGFBR2  (TGF-β Receptor Type 2; 567 aa; ~66 kDa; 3p24.1; AD;
         OMIM gene 190182; Loeys-Dietz Syndrome 2 OMIM 610168;
         similar to LDS1; club foot in some patients; skin findings (velvety skin, bruising);
         aggressive surgical threshold ≥42 mm; losartan + beta-blocker;
         seed SEED_BASE+2) ·
COL3A1  (Procollagen type III α1; 1466 aa; ~138 kDa; 2q32.2; AD;
         OMIM gene 120180; Vascular EDS (vEDS) OMIM 130050;
         most lethal EDS — 25% major complication by age 25; median survival 48 years;
         spontaneous arterial rupture (celiac/mesenteric/renal), sigmoid perforation, uterine rupture;
         NO elective surgery; NO colonoscopy except emergency; celiprolol beta-blocker RCT evidence;
         seed SEED_BASE+3) ·
COL5A1  (Procollagen type V α1; 1838 aa; ~184 kDa; 9q34.3; AD;
         OMIM gene 120215; Classical EDS OMIM 130000;
         haploinsufficiency most common; skin hyperextensibility + atrophic scarring + joint hypermobility;
         skin fragility; recurrent joint dislocations; physiotherapy + bracing; NOT vascular danger;
         seed SEED_BASE+4) ·
ACTA2   (Smooth muscle α-actin; 377 aa; ~42 kDa; 10q23.31; AD;
         OMIM gene 102620; HTAD OMIM 611788;
         livedo reticularis + iris flocculi PATHOGNOMONIC (esp. Arg179 variants);
         aortic root aneurysm + premature CAD + Moyamoya (Arg179 variants);
         surgical threshold ≥45–50 mm; annual TTE; statin + beta-blocker for CAD prevention;
         seed SEED_BASE+5) ·
MYH11   (Smooth muscle myosin heavy chain; 1972 aa; ~227 kDa; 16p13.11; AD;
         OMIM gene 160745; HTAD OMIM 132900;
         key diagnostic triad: aortic aneurysm + patent ductus arteriosus + MVP;
         PDA often previously ligated in childhood (ask specifically in history);
         surgical threshold ≥50 mm; annual TTE + 5-yearly CTA/MRA;
         seed SEED_BASE+6) ·
SMAD3   (SMAD family member 3; 425 aa; ~48 kDa; 15q22.33; AD;
         OMIM gene 603109; LDS3 / Aneurysm-Osteoarthritis Syndrome OMIM 613795;
         PATHOGNOMONIC: early-onset OA (20s–30s, small joints + knees) combined with aortic aneurysm;
         bifid uvula + arterial tortuosity; aggressive surgical threshold ≥42 mm;
         losartan + beta-blocker; OA management NSAIDs + physio;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1542–1549)
"""

import random

SEED_BASE = 1542

CONNECTIVE_TISSUE_GENES = [
    # ── FBN1 — Marfan Syndrome ──
    {
        "gene": "FBN1",
        "protein": "Fibrillin-1 — Marfan Syndrome, Aortic Root Aneurysm, Ectopia Lentis, Arm Span > Height, Beta-Blocker + Losartan, Elective Root ≥50 mm",
        "alias": (
            "FBN1; OMIM gene 134797; Marfan syndrome OMIM 154700; 15q21.1; 2871 aa; ~350 kDa; "
            "FBN1 encodes fibrillin-1, a large glycoprotein that forms the structural backbone "
            "of extracellular matrix microfibrils. Fibrillin-1 microfibrils serve two critical "
            "functions: (1) structural scaffolding in elastic tissues — aorta, lens zonules, "
            "periosteum, skin; (2) sequestration and regulated release of TGF-β family ligands "
            "(TGF-β1/2/3, BMPs) bound as large latent complexes (LLC) tethered via LTBP to "
            "fibrillin-1 microfibrils. Loss-of-function (haploinsufficiency) or dominant-negative "
            "FBN1 variants reduce microfibril integrity → structural weakening of the aortic wall "
            "AND excess TGF-β signalling (reduced microfibril tethering releases TGF-β). "
            "Pathological consequence: progressive aortic root dilation (sinuses of Valsalva) → "
            "type A aortic dissection if untreated. Lens zonule failure → superotemporal ectopia "
            "lentis (upward lens dislocation, in contrast to homocystinuria's infero-nasal "
            "dislocation). Skeletal overgrowth (tall stature, arachnodactyly, arm span > height, "
            "pectus deformities, scoliosis) due to enhanced TGF-β signalling in periosteum. "
            "Revised Ghent nosology 2010: major features — aortic root Z-score ≥2 OR aortic "
            "dissection, ectopia lentis; systemic score ≥7 (includes wrist+thumb sign, pectus "
            "carinatum, hindfoot deformity, pneumothorax, dural ectasia, scoliosis, reduced "
            "upper-to-lower body ratio, arachnodactyly). Systemic score ≥7 alone insufficient "
            "without aortic OR ectopia lentis features for definitive diagnosis. Annual "
            "echocardiography; elective prophylactic aortic root repair when root ≥50 mm "
            "(Bentall or valve-sparing David/Yacoub); repair at 45 mm with rapid progression "
            "(>3 mm/year), family history of dissection, or planned pregnancy. Beta-blocker "
            "(propranolol/atenolol/metoprolol) reduces rate of aortic growth and dissection — "
            "start at diagnosis. Losartan (TGF-β antagonist via AT1R blockade) adjunct — "
            "COMPARE trial confirmed benefit; combine with beta-blocker. NO contact sports, "
            "competitive sports, weight-lifting, isometric exercise; low-intensity aerobic "
            "activity permitted (swimming, walking). Pregnancy: HIGH RISK — aortic dissection "
            "risk increases during third trimester/delivery; prophylactic repair recommended "
            "if root ≥45 mm before conception; beta-blocker maintained throughout pregnancy."
        ),
        "aa": "2871 aa",
        "kDa": "~350 kDa",
        "locus": "15q21.1",
        "omim_gene": 134797,
        "omim_disease": 154700,
        "inheritance": "AD — autosomal dominant; haploinsufficiency or dominant-negative; de novo ~25%; 1:5,000 prevalence",
        "gene_class": (
            "FBN1 (fibrillin-1) is a 2871-amino acid cysteine-rich extracellular matrix "
            "glycoprotein. Domain architecture: 47 calcium-binding EGF-like (cbEGF) domains "
            "— calcium binding stabilises the rod-like structure and protects against "
            "proteolysis; 7 transforming growth factor-β binding protein-like (TB/8-cys) "
            "domains — mediate TGF-β large latent complex binding; hybrid domains; "
            "N-terminal and C-terminal unique domains for fibrillin-fibrillin head-to-tail "
            "polymerisation forming 10–12 nm microfibrils. Fibrillin-1 microfibrils form "
            "the elastic fibre scaffold in the aortic media and ciliary zonules. "
            "Pathogenic variants span the entire gene; missense variants in cbEGF domains "
            "disrupting calcium coordination are particularly pathogenic; cysteine-affecting "
            "variants create non-native disulfide bonds with dominant-negative effects. "
            "FBN2 (fibrillin-2, 5q23.2) causes congenital contractural arachnodactyly "
            "(CCA/Beals syndrome) — a milder phenotype without aortic involvement typically; "
            "rare FBN1/FBN2 overlap cases exist. The fibrillin-TGF-β axis is therapeutic "
            "target: losartan blocks AT1R → reduces TGF-β signalling → slows aortic root "
            "dilation in Marfan mouse models and clinical trials."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 0,
        "etiologies": [
            ("FBN1 missense cbEGF domain AD — haploinsufficiency/dominant-negative, classic Marfan aortic + lens + skeletal", 0.40),
            ("FBN1 frameshift/truncating AD — haploinsufficiency, variable expressivity, aortic root predominant", 0.30),
            ("FBN1 splice-site AD — reduced normal transcript, Marfan spectrum, often milder skeletal involvement", 0.20),
            ("FBN1 large exon deletion AD — severe early-onset Marfan, neonatal/infantile presentation, FBN2 mimic excluded", 0.10),
        ],
        "key_alerts": [
            "FBN1-AORTIC-ROOT-SURGERY-50MM: Elective prophylactic aortic root repair at ≥50 mm (Bentall or valve-sparing David/Yacoub) — DO NOT wait for dissection; advance threshold to ≥45 mm with rapid progression >3 mm/year, family history of dissection, planned pregnancy, or severe aortic regurgitation; annual TTE mandatory from diagnosis",
            "FBN1-BETA-BLOCKER-START-AT-DIAGNOSIS: Beta-blocker (propranolol/atenolol/metoprolol) should be started at diagnosis — reduces rate of aortic root growth and risk of dissection; titrate to heart rate <60 bpm at rest and <100 bpm during moderate exercise; combine with losartan (AT1R blocker) per COMPARE trial evidence",
            "FBN1-NO-CONTACT-SPORTS-LIFELONG: NO contact sports, competitive athletics, weight-lifting, isometric exercise (Valsalva manoeuvre dramatically increases aortic wall stress) — LIFELONG restriction; low-intensity aerobic activity permitted (walking, swimming at moderate pace); written sport restriction certificate required",
            "FBN1-PREGNANCY-HIGH-RISK-ROOT-45MM: Pregnancy in Marfan is HIGH RISK — elective aortic root repair recommended before conception if root ≥45 mm; beta-blocker maintained throughout pregnancy (atenolol preferred; monitor fetal growth); delivery in tertiary centre with cardiac surgery on standby; highest dissection risk in third trimester and 6 weeks postpartum",
            "FBN1-ECTOPIA-LENTIS-SUPEROTEMPORAL: Ectopia lentis in Marfan is SUPEROTEMPORAL (upward-outward) — in contrast to homocystinuria where lens dislocates INFERONASAL (downward-inward); annual ophthalmology review; slit-lamp examination ESSENTIAL as ectopia lentis may be subclinical without visual symptoms",
            "FBN1-GHENT-CRITERIA-2010-SYSTEMIC-SCORE: Apply revised Ghent nosology 2010 — systemic score: wrist+thumb sign (3), wrist OR thumb sign (1), pectus carinatum (2), pectus excavatum/chest asymmetry (1), hindfoot deformity (2), plain pes planus (1), spontaneous pneumothorax (2), dural ectasia (2), protrusio acetabulae (2), reduced upper-to-lower segment ratio AND arm span-to-height >1.05 (1), scoliosis/kyphosis (1), reduced elbow extension (1), facial features (1), striae (1), myopia >3 dioptres (1); systemic score ≥7 supports diagnosis",
            "FBN1-DURAL-ECTASIA-SCREEN: Dural ectasia (expansion of dural sac, particularly lumbosacral region) occurs in 60-92% of Marfan patients — causes low back pain, headache, proximal leg pain, bladder symptoms; MRI lumbosacral spine if symptomatic; counts 2 points in systemic Ghent score; neurosurgical referral only if severe neurological deficit",
        ],
    },
    # ── TGFBR1 — Loeys-Dietz Syndrome 1 ──
    {
        "gene": "TGFBR1",
        "protein": "TGF-β Receptor 1 / ALK5 — LDS1, Bifid Uvula + Hypertelorism + Arterial Tortuosity TRIAD, Aggressive Surgery ≥42 mm NOT ≥50 mm, Head-to-Pelvis MRA",
        "alias": (
            "TGFBR1; OMIM gene 190181; Loeys-Dietz Syndrome 1 OMIM 609192; 9q22.33; 503 aa; ~59 kDa; "
            "TGFBR1 encodes TGF-β receptor type 1 (also called activin receptor-like kinase 5, "
            "ALK5), the signal-transducing serine/threonine kinase receptor that is activated "
            "by TGF-β ligand binding to the TGFBR2 extracellular domain. Loss-of-function "
            "variants in TGFBR1 paradoxically INCREASE downstream TGF-β signalling in the "
            "vessel wall (via non-canonical Smad-independent pathways and compensatory upregulation "
            "of TGF-β in a complex feedback loop). Loeys-Dietz syndrome 1 (LDS1) was first "
            "described by Loeys and Dietz in 2005. "
            "KEY CLINICAL DISTINCTION from Marfan syndrome: LDS aortic aneurysms rupture/dissect "
            "at SMALLER diameters than MFS — prophylactic surgery recommended at 40–42 mm "
            "root diameter (vs ≥50 mm in MFS); DO NOT use Marfan surgical thresholds in LDS. "
            "LDS triad: (1) bifid uvula or cleft palate — bifid uvula is more common; "
            "(2) hypertelorism (increased inter-canthal distance); (3) generalised arterial "
            "tortuosity — the entire arterial tree from intracranial to iliac arteries is "
            "tortuous, aneurysmal, and prone to rupture; head-to-pelvis MRA is mandatory "
            "at diagnosis and every 2 years. Additional features: widely spaced eyes, "
            "craniosynostosis (10–15%), blue sclerae, pectus deformities, scoliosis, "
            "joint laxity, skin translucency/striae, club foot (talipes equinovarus). "
            "Losartan is the primary pharmacotherapy (TGF-β antagonism via AT1R blockade); "
            "beta-blocker as adjunct. Pregnancy management as per Marfan but with lower "
            "surgical threshold (consider prophylactic repair at ≥40 mm before conception)."
        ),
        "aa": "503 aa",
        "kDa": "~59 kDa",
        "locus": "9q22.33",
        "omim_gene": 190181,
        "omim_disease": 609192,
        "inheritance": "AD — autosomal dominant loss-of-function; de novo ~50%; heterozygous missense or truncating variants in kinase domain",
        "gene_class": (
            "TGFBR1 is a 503-amino acid type I transmembrane serine/threonine kinase receptor. "
            "Domains: signal peptide (aa 1-22); cysteine-rich extracellular domain (aa 23-101) "
            "— binds preformed TGF-β/TGFBR2 complex; transmembrane domain (aa 102-122); "
            "intracellular GS domain (aa 123-195, Gly-Ser rich) — phosphorylated by TGFBR2 "
            "kinase at Ser165, Ser167 to activate TGFBR1; intracellular kinase domain "
            "(aa 196-503) — serine/threonine kinase; activation loop (DFG motif Asp289); "
            "SSXS motif (aa 495-499) contacts Smad2/3 MH2 domain. Signalling: "
            "TGF-β binds TGFBR2 → recruits TGFBR1 → TGFBR2 phosphorylates TGFBR1 GS domain "
            "→ TGFBR1 kinase activated → phosphorylates Smad2/Smad3 at C-terminal SSXS → "
            "pSmad2/3 complex with Smad4 → nuclear translocation → TGF-β target gene "
            "transcription (CTGF, PAI1, fibronectin, collagens). LDS kinase-dead variants "
            "block canonical signalling but paradoxically increase non-canonical TGF-β output "
            "in the aortic wall. The difference from MFS is mechanistic: FBN1 LOF releases "
            "tethered TGF-β (increases extracellular TGF-β availability); TGFBR1 LOF "
            "blocks canonical intracellular signalling but triggers compensatory TGF-β "
            "overproduction and non-canonical pathway activation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("TGFBR1 kinase domain missense AD — LDS1, bifid uvula + hypertelorism + tortuosity, aggressive aneurysm ≥42 mm", 0.45),
            ("TGFBR1 truncating/frameshift AD — LDS1 haploinsufficiency, craniosynostosis variant, severe multi-vessel tortuosity", 0.25),
            ("TGFBR1 GS-domain missense AD — LDS1, prominent joint laxity + skin findings, moderate aneurysm progression", 0.20),
            ("TGFBR1 splice-site AD — LDS1 spectrum, mild triad features, aneurysm dominant, bifid uvula partial expression", 0.10),
        ],
        "key_alerts": [
            "TGFBR1-SURGERY-42MM-NOT-50MM: LDS surgical threshold ≥42 mm (NOT ≥50 mm as in Marfan) — LDS aneurysms dissect/rupture at SMALLER diameters; applying Marfan thresholds in LDS is DANGEROUS; some centres use ≥40 mm for TGFBR1/TGFBR2; discuss with specialist; DO NOT wait for 50 mm in LDS patients",
            "TGFBR1-HEAD-TO-PELVIS-MRA-MANDATORY: Whole-arterial-tree imaging from intracranial vessels to iliac arteries MANDATORY at diagnosis and every 2 years — intracranial aneurysms (5–12%), vertebral artery tortuosity/aneurysm, celiac/mesenteric/renal aneurysms occur in LDS; CT/MRA head-to-pelvis annually if any aneurysm found outside the aortic root",
            "TGFBR1-BIFID-UVULA-TRIAD-CHECK: Actively examine uvula (bifid vs normal) and measure inter-canthal distance for hypertelorism at first visit — bifid uvula is pathognomonic for LDS when combined with arterial findings; triad (bifid uvula + hypertelorism + tortuosity) is 100% specific for LDS; check family members for triad",
            "TGFBR1-LOSARTAN-PRIMARY-THERAPY: Losartan is the primary pharmacotherapy for LDS (TGF-β antagonism via AT1R blockade) — target dose 1.4 mg/kg/day (children) or 100 mg/day (adults); beta-blocker added as adjunct; COMPARATOR trial evidence; AVOID stopping losartan suddenly in LDS; irbesartan is alternative AT1R blocker if losartan intolerant",
            "TGFBR1-CRANIOSYNOSTOSIS-SCREEN: Craniosynostosis in LDS1 (10–15%) — premature skull suture fusion; screen with cranial CT at diagnosis; neurosurgical referral if present; associated with more severe systemic phenotype; hypertelorism may be secondary to craniosynostosis pattern",
            "TGFBR1-PREGNANCY-40MM-THRESHOLD: Prophylactic aortic root repair recommended before conception if root ≥40 mm in LDS (more conservative than MFS 45 mm threshold) — LDS pregnancy risk mirrors MFS but at smaller diameter; losartan MUST be stopped at conception (teratogenic, category X); switch to beta-blocker for duration of pregnancy",
        ],
    },
    # ── TGFBR2 — Loeys-Dietz Syndrome 2 ──
    {
        "gene": "TGFBR2",
        "protein": "TGF-β Receptor 2 — LDS2, Aggressive Aneurysm ≥42 mm, Club Foot + Skin Findings, Losartan + Beta-Blocker, Head-to-Pelvis MRA",
        "alias": (
            "TGFBR2; OMIM gene 190182; Loeys-Dietz Syndrome 2 OMIM 610168; 3p24.1; 567 aa; ~66 kDa; "
            "TGFBR2 encodes TGF-β receptor type 2, the constitutively active serine/threonine "
            "kinase that is the primary ligand-binding receptor in the TGF-β receptor complex. "
            "Unlike TGFBR1 (ALK5) which is activated by phosphorylation, TGFBR2 is "
            "constitutively kinase-active and directly binds TGF-β1/2/3 dimers. "
            "Loeys-Dietz Syndrome 2 (LDS2, due to TGFBR2 LOF) is clinically nearly "
            "identical to LDS1 (TGFBR1 LOF) — both feature the LDS triad (bifid uvula, "
            "hypertelorism, arterial tortuosity), aggressive aortic aneurysm progression, "
            "and multi-vessel involvement. "
            "TGFBR2-specific features that may help distinguish LDS2 from LDS1: "
            "club foot (talipes equinovarus) is more common in LDS2; more prominent "
            "skin findings — velvety skin texture, easy bruising, striae; "
            "TGFBR2 missense variants involving the kinase domain (Arg528Cys is historically "
            "significant — one of the first described LDS variants) cause LDS2. "
            "Clinical management is IDENTICAL to LDS1: prophylactic aortic surgery ≥42 mm, "
            "losartan primary pharmacotherapy, head-to-pelvis MRA annually, "
            "no contact sports. Differential: hereditary vascular EDS (COL3A1) — "
            "vEDS has arterial RUPTURE not aneurysm as primary feature; skin is translucent "
            "with visible veins; NO bifid uvula triad; LDS has tortuosity + triad. "
            "Lynch syndrome (TGFBR2 frameshift in microsatellite-unstable colorectal cancer) "
            "is a distinct somatic tumour suppressor mechanism — DO NOT confuse with "
            "germline TGFBR2 LOF causing LDS2."
        ),
        "aa": "567 aa",
        "kDa": "~66 kDa",
        "locus": "3p24.1",
        "omim_gene": 190182,
        "omim_disease": 610168,
        "inheritance": "AD — autosomal dominant loss-of-function; de novo ~50%; kinase domain missense most common",
        "gene_class": (
            "TGFBR2 is a 567-amino acid type II transmembrane receptor. Domains: "
            "signal peptide (aa 1-23); cysteine-rich extracellular domain (aa 24-159) — "
            "direct TGF-β dimer binding site; transmembrane domain (aa 160-180); "
            "intracellular juxtamembrane region (aa 181-222) — contains serine/threonine "
            "residues for TGFBR1 recruitment and phosphorylation; kinase domain "
            "(aa 223-567) — constitutively active; activation loop (DFG motif Asp444); "
            "TGFBR2 phosphorylates TGFBR1 GS domain → activates TGFBR1 kinase → Smad2/3 "
            "phosphorylation → canonical TGF-β signalling. TGFBR2 also directly "
            "phosphorylates non-Smad substrates (ShcA, Par6, LIMK1) for "
            "non-canonical signalling. TGFBR2 LOF variants in LDS2: same paradoxical "
            "increase in TGF-β output as TGFBR1 LOF in LDS1 — vessel wall TGF-β "
            "activity is elevated despite receptor LOF (compensatory upregulation, "
            "enhanced bioavailability, non-canonical pathway activation). "
            "Somatic TGFBR2 frameshift mutations in microsatellite-unstable colorectal "
            "cancer (MLH1/MSH2 deficiency → microsatellite instability → frameshift in "
            "TGFBR2 poly-A tract → somatic tumour suppressor function loss) are a "
            "COMPLETELY DIFFERENT mechanism from germline AD LDS2."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("TGFBR2 kinase domain missense AD — LDS2, classic triad, club foot present, skin findings, aggressive root ≥42 mm", 0.40),
            ("TGFBR2 extracellular domain missense AD — LDS2 spectrum, ligand-binding impaired, moderate tortuosity, triad", 0.25),
            ("TGFBR2 truncating/frameshift AD — LDS2 haploinsufficiency, severe phenotype, multi-vessel aneurysm, craniosynostosis", 0.20),
            ("TGFBR2 splice-site AD — LDS2 spectrum, variable penetrance, skin laxity dominant, triad partial expression", 0.15),
        ],
        "key_alerts": [
            "TGFBR2-CLUB-FOOT-DIAGNOSTIC-CLUE: Club foot (talipes equinovarus) is more common in LDS2 (TGFBR2) than LDS1 (TGFBR1) — in any child/adult with club foot AND aortic aneurysm/tortuosity, prioritise TGFBR2 sequencing; club foot corrected in infancy may be absent from active history — ask specifically about birth foot deformity",
            "TGFBR2-SURGERY-42MM-NOT-50MM: LDS2 surgical threshold ≥42 mm aortic root (same as LDS1, NOT Marfan's ≥50 mm) — TGFBR2 aneurysms are as prone to rupture at small sizes as TGFBR1 aneurysms; Marfan thresholds are UNSAFE in LDS; discuss with aortic surgery specialist at first visit",
            "TGFBR2-SKIN-FINDINGS-DISTINGUISH-VEDS: LDS2 skin findings (velvety skin, easy bruising, striae) may overlap with vascular EDS — KEY distinction: vEDS (COL3A1) has NO bifid uvula triad; vEDS arteries RUPTURE spontaneously without prior aneurysm; LDS arteries are TORTUOUS + ANEURYSMAL; COL3A1 and TGFBR2 gene panel resolves ambiguity",
            "TGFBR2-LYNCH-SOMATIC-CONFUSION: TGFBR2 frameshift in colorectal cancer = somatic tumour suppressor event in Lynch syndrome — COMPLETELY DIFFERENT from germline TGFBR2 LOF causing LDS2; do NOT label a cancer patient with somatic TGFBR2 frameshift as having LDS; confirm germline status with blood/saliva DNA",
            "TGFBR2-MRA-HEAD-TO-PELVIS: Head-to-pelvis MRA at diagnosis and every 2 years — intracranial, vertebral, celiac, mesenteric, renal artery aneurysms identified in TGFBR2 families; intracranial aneurysm screening with CTA/MRA brain annually if prior intracranial lesion found",
            "TGFBR2-LOSARTAN-STOP-IN-PREGNANCY: Losartan (and all AT1R blockers, ACEi) MUST be stopped immediately at conception confirmation — teratogenic (fetal renal dysgenesis, oligohydramnios, fetal death); switch to beta-blocker; restart losartan immediately postpartum; contraception counselling MANDATORY before losartan initiation in women of reproductive age",
        ],
    },
    # ── COL3A1 — Vascular EDS ──
    {
        "gene": "COL3A1",
        "protein": "Type III Collagen α1 — Vascular EDS, Most Lethal EDS, Spontaneous Arterial Rupture, Bowel/Uterine Perforation, NO Elective Surgery, Celiprolol RCT Evidence",
        "alias": (
            "COL3A1; OMIM gene 120180; Vascular EDS (vEDS) OMIM 130050; 2q32.2; 1466 aa; ~138 kDa; "
            "COL3A1 encodes the procollagen type III α1 chain. Three identical α1(III) "
            "chains assemble into the type III collagen homotrimer. Type III collagen is the "
            "predominant structural collagen in blood vessel walls (tunica media), skin dermis, "
            "and hollow organ walls (bowel, uterus). "
            "COL3A1 haploinsufficiency (truncating variants, large deletions) or "
            "dominant-negative missense variants (especially glycine substitutions in the "
            "Gly-X-Y repeating tripeptide backbone — Gly→X disrupts triple-helix formation) "
            "cause vascular EDS (vEDS), the MOST LETHAL form of EDS. "
            "Natural history: 25% have a major complication (arterial dissection/rupture, "
            "sigmoid colon perforation, uterine rupture) by age 25; 80% by age 40; "
            "median survival approximately 48 years in older series (improving with specialist care). "
            "Typical rupture sites: celiac artery, superior/inferior mesenteric artery, "
            "renal artery, iliac artery; aortic root aneurysm is UNCOMMON (unlike MFS/LDS); "
            "small-to-medium artery spontaneous rupture is the hallmark. "
            "Bowel perforation: sigmoid colon is the most common site; presents as acute "
            "abdomen without prior inflammatory bowel disease. Uterine rupture during pregnancy. "
            "CRITICAL SAFETY RULE: NO elective surgery, invasive procedures, or colonoscopy "
            "(except emergency) — surgical complications (anastomotic dehiscence, arterial "
            "injury, poor wound healing) exceed the benefit of elective intervention; "
            "if emergency surgery is unavoidable, approach via minimal-access techniques "
            "with experienced vascular team available. "
            "Celiprolol (beta-1 selective adrenergic blocker): BBEST RCT (Ong 2010 NEJM) "
            "showed celiprolol 200–400 mg/day significantly reduced major vascular events "
            "in vEDS patients over 5 years; currently recommended as primary pharmacotherapy."
        ),
        "aa": "1466 aa",
        "kDa": "~138 kDa",
        "locus": "2q32.2",
        "omim_gene": 120180,
        "omim_disease": 130050,
        "inheritance": "AD — autosomal dominant; glycine-substitution missense (dominant-negative) OR haploinsufficiency (truncating); de novo ~50%; biallelic AR form causes extremely severe perinatal vEDS (extremely rare)",
        "gene_class": (
            "COL3A1 encodes the 1466-amino acid procollagen III α1 precursor. Structure: "
            "signal peptide (aa 1-25); N-propeptide (aa 26-103) — cleaved after secretion; "
            "triple-helical domain (aa 104-1378) — contains 425 uninterrupted Gly-X-Y repeats; "
            "C-propeptide (aa 1379-1466) — initiates triple helix formation, cleaved "
            "extracellularly. The triple helix requires strictly glycine at every third position "
            "(Gly-X-Y): glycine is the smallest amino acid and must fit inside the tight "
            "helical core; any substitution (Gly→Ser/Arg/Asp/Cys/Val/Glu) disrupts "
            "triple-helix folding, retaining misfolded procollagen III in the ER → "
            "dominant-negative effect (normal chains trapped with abnormal). "
            "Glycine-substitution missense variants are the most common and pathogenic "
            "vEDS variants (dominant-negative severity > haploinsufficiency severity). "
            "The resulting structurally deficient type III collagen matrix causes: "
            "fragile, thin-walled medium arteries with absent medial smooth muscle support; "
            "connective tissue of the bowel wall fails to maintain luminal pressure → "
            "spontaneous sigmoid perforation; thin, friable uterine wall → intrapartum rupture. "
            "Skin: translucent skin with visible subcutaneous veins, easy bruising, "
            "poor wound healing — but skin hyperextensibility is MINIMAL (distinguishes "
            "vEDS from classical/hypermobile EDS). Biochemical diagnosis: type III "
            "procollagen secretion defect on cultured fibroblast assay (reduced/absent "
            "type III collagen on gel electrophoresis) prior to molecular era; now "
            "COL3A1 sequencing is first-line diagnostic test."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("COL3A1 glycine-substitution missense AD — dominant-negative, most lethal vEDS, arterial rupture high risk", 0.50),
            ("COL3A1 truncating/haploinsufficiency AD — milder vEDS, bowel perforation risk, skin fragility, variable expressivity", 0.25),
            ("COL3A1 splice-site AD — exon skipping causing in-frame deletion of Gly-X-Y repeats, intermediate severity", 0.15),
            ("COL3A1 large exon deletion AD — variable haploinsufficiency, skin + bowel predominant, arterial risk moderate", 0.10),
        ],
        "key_alerts": [
            "COL3A1-NO-ELECTIVE-SURGERY-ABSOLUTE: NO elective surgery or invasive procedures in vEDS — anastomotic dehiscence, uncontrolled arterial bleeding, and poor wound healing make elective surgery risk >> benefit; if emergency surgery unavoidable: minimal-access approach, experienced vascular surgeon on standby, avoid bowel anastomosis if possible (prefer stoma), ICU postoperative monitoring mandatory",
            "COL3A1-NO-COLONOSCOPY-EMERGENCY-ONLY: Colonoscopy is CONTRAINDICATED in vEDS except emergency (suspected perforation/life-threatening bleed) — sigmoid perforation risk from colonoscope insufflation and insertion; CT colonography (virtual colonoscopy) is the safer alternative for routine colorectal screening if indicated",
            "COL3A1-CELIPROLOL-RCT-EVIDENCE: Celiprolol (beta-1 selective, partial beta-2 agonist) 200–400 mg/day is the only treatment with RCT evidence (BBEST trial, Ong 2010 NEJM) showing 5-year reduction in major vascular events in vEDS — start at 200 mg/day, uptitrate to 400 mg/day if tolerated; monitor for bradycardia; maintain lifelong",
            "COL3A1-ARTERIAL-RUPTURE-EMERGENCY-ENDOVASCULAR: Acute arterial dissection/rupture in vEDS — endovascular (stent-graft) approach preferred over open surgery when feasible (lower mortality); celiac/mesenteric/renal ruptures: emergency interventional radiology; DO NOT delay for haematological optimisation if bleeding is life-threatening; vascular/IR team must know vEDS diagnosis before any access",
            "COL3A1-PREGNANCY-HIGH-RISK-UTERINE-RUPTURE: Pregnancy in vEDS carries up to 12% maternal mortality — uterine rupture (especially at term/delivery), postpartum arterial rupture; planned elective caesarean section at 36–37 weeks gestation in tertiary centre with vascular surgery and ICU backup; counsel at diagnosis regarding pregnancy risk BEFORE conception",
            "COL3A1-SKIN-TRANSLUCENCY-BRUISING: vEDS skin findings: translucent skin (visible subcutaneous veins especially chest/upper arm), easy bruising (minor trauma → large haematomas), poor wound healing (scars widened but NOT hyperextensible); skin hyperextensibility is MINIMAL in vEDS — if marked skin stretch present, reconsider classical EDS (COL5A1); skin findings DO NOT predict arterial rupture risk",
            "COL3A1-DISTINGUISH-FROM-LDS: vEDS vs LDS differentiation: vEDS — arterial RUPTURE without prior aneurysm; NO bifid uvula triad; skin translucent; bowel + uterine perforation; LDS — arterial TORTUOSITY + ANEURYSM; bifid uvula + hypertelorism; aortic root enlarged; COL3A1 + LDS panel resolves all ambiguous cases",
        ],
    },
    # ── COL5A1 — Classical EDS ──
    {
        "gene": "COL5A1",
        "protein": "Type V Collagen α1 — Classical EDS, Skin Hyperextensibility + Atrophic Scarring + Joint Hypermobility, NOT Vascular Danger, Physiotherapy + Bracing",
        "alias": (
            "COL5A1; OMIM gene 120215; Classical EDS OMIM 130000; 9q34.3; 1838 aa; ~184 kDa; "
            "COL5A1 encodes the procollagen type V α1 chain. Type V collagen forms heterotrimers "
            "([α1(V)]₂α2(V) or α1(V)α2(V)α3(V)) that function as a regulatory collagen — "
            "type V collagen controls the DIAMETER of type I collagen fibrils by co-polymerising "
            "with type I collagen during fibril assembly; reduced type V collagen (haploinsufficiency "
            "from COL5A1 LOF) results in irregular, large-diameter type I collagen fibrils with "
            "reduced mechanical strength. Classical EDS (cEDS) is the most common genetically "
            "confirmed EDS subtype; prevalence ~1:20,000–40,000. "
            "Clinical triad: (1) Skin hyperextensibility — skin stretches >1.5 cm beyond normal; "
            "tested on dorsum of hand/forearm; soft, doughy skin texture; (2) Widened atrophic "
            "scarring (cigarette-paper scars) — scars heal widely, thin, and discoloured; "
            "even minor wounds → disproportionate scarring; molluscoid pseudotumours at "
            "pressure sites; (3) Joint hypermobility — Beighton score ≥5/9 in adults; "
            "recurrent dislocations (shoulder, knee, patella, hip, ankle). "
            "IMPORTANT DISTINCTION from vascular EDS: cEDS does NOT carry the same risk of "
            "spontaneous arterial rupture or bowel perforation; vascular complications are "
            "uncommon and mild; surgical risk is increased (poor wound healing, tissue fragility) "
            "but NOT the life-threatening risk seen in vEDS. "
            "Management: primarily supportive — physiotherapy for joint stability, proprioception, "
            "and muscle strengthening; joint bracing/splinting for unstable joints; "
            "wound care (Steri-Strips, paper tape to prevent wound widening); avoid contact sports, "
            "heavy lifting, activities with sudden joint impact."
        ),
        "aa": "1838 aa",
        "kDa": "~184 kDa",
        "locus": "9q34.3",
        "omim_gene": 120215,
        "omim_disease": 130000,
        "inheritance": "AD — autosomal dominant haploinsufficiency; truncating variants most common (nonsense, frameshift, splice-site); COL5A2 (2q32.2) haploinsufficiency causes classical EDS type 2",
        "gene_class": (
            "COL5A1 is a 1838-amino acid procollagen V α1 chain. Domain structure: "
            "signal peptide; N-propeptide (PCOLCE-N) — unique to type V collagen; "
            "contains thrombospondin-1-like and variable (TSPN) domains; extends to the "
            "fibril surface and regulates fibril-to-fibril spacing; triple-helical domain "
            "with Gly-X-Y repeats; short C-propeptide. Type V collagen is obligatorily "
            "co-polymerised with type I collagen in skin dermis, tendons, ligaments, and "
            "cornea. The molecular ratio of type V to type I collagen in a fibril is "
            "approximately 1:8; even haploinsufficiency (50% reduction) of type V collagen "
            "results in significantly enlarged, mechanically weaker fibrils. "
            "Haploinsufficiency is the predominant mechanism in cEDS: nonsense, frameshift, "
            "and splice-site variants in COL5A1 reduce functional α1(V) chain output by ~50%; "
            "unlike COL3A1 glycine-substitution missense variants (dominant-negative, severe), "
            "COL5A1 haploinsufficiency (not dominant-negative) results in a milder, non-"
            "vascular phenotype. COL5A2 (chromosome 2q32.2) encodes the α2(V) chain; "
            "COL5A2 haploinsufficiency causes the less common cEDS type 2 with overlapping "
            "features. COL5A3 encodes α3(V); its role in human disease is less established. "
            "Approximately 50% of clinically diagnosed cEDS patients have a detectable "
            "COL5A1/COL5A2 variant; the remainder may have haploinsufficiency variants "
            "missed by sequencing (large deletions) or may represent hypermobile EDS "
            "mis-classified as classical."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("COL5A1 nonsense/frameshift haploinsufficiency AD — classic cEDS skin + scarring + hypermobility triad, mild course", 0.45),
            ("COL5A1 splice-site haploinsufficiency AD — cEDS, variable expressivity, skin laxity dominant, scarring mild", 0.25),
            ("COL5A1 large exon deletion haploinsufficiency AD — cEDS, scarring + hypermobility prominent, skin stretch moderate", 0.15),
            ("COL5A2 haploinsufficiency AD — cEDS type 2, phenotypically identical, less common, confirm by COL5A2 sequencing", 0.15),
        ],
        "key_alerts": [
            "COL5A1-NOT-VASCULAR-DANGER: Classical EDS (COL5A1/COL5A2) does NOT carry the life-threatening vascular rupture risk of vEDS (COL3A1) — counsel patients that their EDS subtype is NOT vascular; avoid unnecessary vascular screening causing anxiety; prognosis is good with joint management and wound care",
            "COL5A1-WOUND-CARE-MANDATORY: All wounds in cEDS should be closed with Steri-Strips or paper tape in addition to sutures — sutures alone pull through the fragile dermis; sutures removed late (week 3–4 instead of week 1–2 to allow prolonged healing); advise patients to carry wound closure strips at all times; surgical wounds require extra support dressings for 4–6 weeks",
            "COL5A1-PHYSIOTHERAPY-FIRST-LINE: Physiotherapy is the primary treatment for joint instability in cEDS — proprioceptive neuromuscular facilitation, joint stabilisation exercises, hydrotherapy; bracing for recurrent dislocations (shoulder, knee, patella); swimming and cycling preferred over high-impact exercise; avoid sudden joint loading and extreme ranges of motion",
            "COL5A1-BEIGHTON-SCORE-DOCUMENT: Document Beighton score at each visit (0–9 points; ≥5 = hypermobility in adults); assess: bilateral little finger metacarpophalangeal dorsiflexion ≥90° (2 pts), bilateral thumb-to-forearm opposition (2 pts), bilateral elbow hyperextension ≥10° (2 pts), bilateral knee hyperextension ≥10° (2 pts), palms flat on floor with knees straight (1 pt); score does NOT correlate with pain severity",
            "COL5A1-SKIN-TEST-DORSUM-HAND: Skin hyperextensibility test: pull skin on dorsum of hand to ≥1.5 cm — positive test; also test forearm; avoid testing at neck (skin naturally extensible) or palmar skin (not representative); document as positive/negative with measured extent in notes",
            "COL5A1-DISTINGUISH-HEDS: Most patients with hypermobile EDS (hEDS) do NOT have a COL5A1/COL5A2 variant — hEDS remains genetically unresolved; cEDS is diagnosed by: skin hyperextensibility + atrophic scarring + COL5A1/COL5A2 pathogenic variant; hEDS diagnosis requires 2018 international criteria with NO confirmed genetic cause; gene panel distinguishes the two",
        ],
    },
    # ── ACTA2 — Hereditary Thoracic Aortic Disease ──
    {
        "gene": "ACTA2",
        "protein": "Smooth Muscle α-Actin — HTAD, Livedo Reticularis + Iris Flocculi PATHOGNOMONIC (Arg179), Premature CAD, Moyamoya, Aortic Root ≥45–50 mm",
        "alias": (
            "ACTA2; OMIM gene 102620; Hereditary Thoracic Aortic Disease OMIM 611788; 10q23.31; 377 aa; ~42 kDa; "
            "ACTA2 encodes smooth muscle α-actin (α-SMA, ACTA2), the predominant actin isoform "
            "of vascular smooth muscle cells (VSMCs). ACTA2 is the major contractile protein "
            "responsible for VSMC contraction and vessel wall tone. Heterozygous ACTA2 "
            "loss-of-function or dominant-negative variants impair VSMC cytoskeletal integrity "
            "and contractile function → aortic wall weakening and aneurysm formation. "
            "ACTA2-related HTAD is phenotypically distinct from MFS and LDS: "
            "PATHOGNOMONIC findings in ACTA2 Arg179 variants (Arg179His, Arg179Cys — "
            "most severe variant class): (1) livedo reticularis — mottled, net-like skin "
            "discolouration of extremities due to VSMC dysfunction in small cutaneous vessels; "
            "present in childhood; (2) iris flocculi — pigment clumps on the anterior iris "
            "surface (iris neovascular tufts); visible on slit-lamp examination; "
            "combination of livedo + iris flocculi in a young person = DIAGNOSTIC for ACTA2 "
            "Arg179 variant without molecular testing needed. Additional ACTA2 Arg179 features: "
            "Moyamoya-like cerebrovascular disease (bilateral intracranial ICA stenosis + "
            "collateral vessel formation); patent ductus arteriosus (PDA); bicuspid aortic "
            "valve; premature coronary artery disease (CAD onset in 4th–5th decade, decades "
            "before typical CAD in population). "
            "Non-Arg179 ACTA2 variants: predominantly thoracic aortic aneurysm/dissection "
            "without the severe cutaneous/cerebrovascular/cardiac features of Arg179; "
            "aortic root + ascending aorta aneurysm; root dissection risk at younger ages."
        ),
        "aa": "377 aa",
        "kDa": "~42 kDa",
        "locus": "10q23.31",
        "omim_gene": 102620,
        "omim_disease": 611788,
        "inheritance": "AD — autosomal dominant; Arg179 variants (most severe) OR other missense/truncating variants; de novo common; familial HTAD also from MYH11, SMAD3, TGFBR1/2, FBN1",
        "gene_class": (
            "ACTA2 is a 377-amino acid actin isoform. Actin is a 42 kDa globular protein (G-actin) "
            "that polymerises into 7 nm helical filaments (F-actin). In VSMC: ACTA2 F-actin "
            "filaments form contractile stress fibres cross-linked by myosin II (MYH11) → "
            "VSMC contraction. ACTA2 domains: DNase I-binding loop (DB-loop, residues 40–50) "
            "— important for filament-filament contacts; hydrophobic plug (residues 262–274); "
            "nucleotide binding cleft (ATP/ADP binding, residues 1–11, 156–172, 203–213, "
            "285–290); D-loop and H-plug regulate inter-strand filament contacts. "
            "Residue Arg179 (Arg179His or Arg179Cys): located in subdomain 3 (residues "
            "137–337); Arg179 contacts Glu72 of the adjacent actin protomer in the filament; "
            "substitution to His or Cys (both smaller/less basic) disrupts the inter-strand "
            "salt bridge → destabilised F-actin → VSMC stiffness reduction → vascular smooth "
            "muscle contractile failure in small vessels (livedo) and large arteries (aortic). "
            "The severe Arg179 phenotype (livedo + iris flocculi + Moyamoya + PDA + CAD) "
            "reflects the critical dependence of multiple organ systems on ACTA2-mediated "
            "VSMC contractility: cutaneous arterioles → livedo; ciliary body vessels → "
            "iris flocculi (neovascular tufts); intracranial ICA smooth muscle → Moyamoya; "
            "ductus arteriosus VSMC → PDA; coronary VSMC → premature CAD."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("ACTA2 Arg179His/Cys severe variant AD — livedo reticularis + iris flocculi + Moyamoya + PDA + premature CAD + aortic", 0.35),
            ("ACTA2 non-Arg179 missense AD — HTAD aortic root aneurysm predominant, no livedo, premature CAD risk, aortic root ≥45 mm", 0.40),
            ("ACTA2 truncating/splice-site AD — HTAD haploinsufficiency, ascending aorta predominant, moderate risk, family screening", 0.15),
            ("ACTA2 de novo severe missense AD — childhood-onset HTAD, livedo present, bicuspid aortic valve, Moyamoya screen required", 0.10),
        ],
        "key_alerts": [
            "ACTA2-LIVEDO-IRIS-FLOCCULI-PATHOGNOMONIC: Livedo reticularis + iris flocculi in a young person = PATHOGNOMONIC for ACTA2 Arg179 variant — clinical diagnosis possible without awaiting molecular result; request urgent ACTA2 sequencing + echocardiogram + cerebrovascular MRA + ophthalmology slit-lamp examination simultaneously",
            "ACTA2-MOYAMOYA-SCREEN-ARG179: All ACTA2 Arg179 variant carriers require cerebrovascular MRA screening at diagnosis — Moyamoya-like ICA stenosis (bilateral) with collateral vessel formation; risk of ischaemic stroke in childhood/young adulthood; neurosurgery referral for revascularisation (encephaloduroarteriosynangiosis, EDAS) if flow-limiting stenosis identified",
            "ACTA2-PREMATURE-CAD-STATIN-EARLY: ACTA2 patients (especially Arg179) have premature CAD risk — start statin therapy in 3rd decade even without traditional cardiovascular risk factors; aggressive risk factor modification; annual ECG; coronary CT angiography in 4th decade or earlier if symptomatic; beta-blocker for both CAD prevention and aortic growth rate reduction",
            "ACTA2-AORTIC-THRESHOLD-45-50MM: ACTA2 aortic root surgical threshold ≥45–50 mm (intermediate between MFS ≥50 mm and LDS ≥42 mm) — discuss individually with specialist; annual TTE; advance threshold to ≥40 mm with rapid progression, strong family history of dissection, or planned pregnancy; CT/MRA for ascending aorta surveillance beyond root",
            "ACTA2-BICUSPID-AORTIC-VALVE-SCREEN: Bicuspid aortic valve (BAV) occurs in some ACTA2 variant carriers — TTE at diagnosis; BAV accelerates aortic root dilation and is an independent risk factor for dissection; BAV + ACTA2 → lower surgical threshold; serial 6-monthly TTE if significant aortic stenosis/regurgitation present",
            "ACTA2-PDA-HISTORY: Patent ductus arteriosus (PDA) is a feature of ACTA2 Arg179 variants — often surgically ligated in infancy without cardiac genetics referral; ask specifically 'Were you told your heart had a defect at birth or in childhood?' and review birth records; prior PDA ligation does not eliminate HTAD risk",
        ],
    },
    # ── MYH11 — HTAD + PDA ──
    {
        "gene": "MYH11",
        "protein": "Smooth Muscle Myosin Heavy Chain — HTAD + Patent Ductus Arteriosus + MVP Triad, Diagnostic Clue PDA in Adult with Aortic Aneurysm, Annual TTE",
        "alias": (
            "MYH11; OMIM gene 160745; Hereditary Thoracic Aortic Disease OMIM 132900; 16p13.11; 1972 aa; ~227 kDa; "
            "MYH11 encodes smooth muscle myosin heavy chain (SMyHC), the motor protein that "
            "powers VSMC contraction by interacting with ACTA2 actin filaments in an "
            "ATP-dependent cross-bridge cycling mechanism. MYH11 mutations in HTAD were "
            "identified in families with thoracic aortic aneurysm combined with patent "
            "ductus arteriosus (PDA). "
            "The combination of thoracic aortic aneurysm/dissection + patent ductus "
            "arteriosus (PDA) in the same patient or family is the DIAGNOSTIC CLUE for "
            "MYH11-related HTAD. In clinical practice: adult patients may have had PDA "
            "surgically ligated in infancy without any cardiac genetics evaluation; "
            "a thorough history ('Were you told you had a heart defect as a baby?') "
            "or review of birth/neonatal records is essential. Mitral valve prolapse (MVP) "
            "also occurs at increased frequency in MYH11 families — the combination of "
            "MVP + PDA (even if repaired) + ascending aortic aneurysm strongly suggests MYH11. "
            "Phenotype: predominantly ascending aorta and aortic root aneurysm; "
            "aortic dissection risk at relatively young ages (3rd–5th decade); "
            "no bifid uvula triad (distinguishes from LDS); no livedo/iris flocculi "
            "(distinguishes from ACTA2 Arg179); no ectopia lentis (distinguishes from MFS). "
            "Medical management: beta-blocker + annual TTE; CT/MRA of entire thoracic aorta "
            "and descending aorta for extent assessment; surgical threshold ≥50 mm "
            "(similar to MFS; less aggressive than LDS). Family cascade screening for both "
            "aortic disease AND PDA/MVP in all first-degree relatives."
        ),
        "aa": "1972 aa",
        "kDa": "~227 kDa",
        "locus": "16p13.11",
        "omim_gene": 160745,
        "omim_disease": 132900,
        "inheritance": "AD — autosomal dominant; missense and truncating variants; inter-familial variable expressivity; de novo cases reported",
        "gene_class": (
            "MYH11 (smooth muscle myosin heavy chain, SMyHC) is a 1972-amino acid class II "
            "myosin motor protein. Domain architecture: N-terminal motor domain (head, aa 1–780) "
            "— contains ATP-binding P-loop (GXXXXGKS/T, aa 179–186) and actin-binding "
            "surface loop (loop 1, aa 204–216, and loop 2, aa 626–646); converter domain "
            "(aa 701–780) — transmits conformational change; lever arm (aa 781–850) — "
            "amplifies power stroke; IQ motifs (aa 830–850) — bind essential light chains "
            "(ELC, MYL9) and regulatory light chain (RLC, MRLC); coiled-coil rod domain "
            "(aa 851–1972) — mediates filament assembly into bipolar thick filaments. "
            "MYH11 and ACTA2 form the contractile apparatus of VSMC: MYH11 thick filaments "
            "interact with ACTA2 thin filaments; phosphorylation of the regulatory light chain "
            "(RLC) by myosin light-chain kinase (MLCK) activates the ATPase cycle; "
            "hydrolysis of ATP drives power strokes generating contractile force. "
            "MYH11 mutations in HTAD: most variants are missense in the motor or rod domain, "
            "disrupting either motor function (force generation) or filament assembly; "
            "reduced contractile force → aortic wall structural weakness → aneurysm formation. "
            "The association with PDA suggests a critical role for MYH11 in ductal smooth "
            "muscle contraction at birth (ductal closure is driven by VSMC constriction "
            "triggered by oxygenation; MYH11 deficiency impairs this process → persistent PDA)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("MYH11 motor-domain missense AD — HTAD aortic root + ascending, PDA history, MVP, annual TTE, beta-blocker", 0.40),
            ("MYH11 rod-domain missense AD — HTAD predominantly ascending aorta, family history PDA/MVP, moderate risk", 0.30),
            ("MYH11 truncating/frameshift AD — HTAD haploinsufficiency, ascending + descending aorta, broader surveillance", 0.20),
            ("MYH11 splice-site AD — HTAD spectrum, PDA in family, moderate aortic risk, cascade screening mandatory", 0.10),
        ],
        "key_alerts": [
            "MYH11-PDA-DIAGNOSTIC-TRIAD: Thoracic aortic aneurysm + patent ductus arteriosus (even if repaired in infancy) + MVP = DIAGNOSTIC TRIAD for MYH11 — ALWAYS ask about childhood heart surgery or 'hole in the heart' in thoracic aneurysm patients; review neonatal/paediatric records; this triad is SPECIFIC for MYH11 among HTAD genes",
            "MYH11-PDA-HISTORY-MISSED: PDA in MYH11 is often surgically ligated in infancy and not documented in adult records — ask 'Were you operated on the heart as a child?' or 'Did you have a heart murmur at birth?'; birth records or childhood cardiac surgery notes confirm prior PDA; absence of PDA in an adult does NOT exclude MYH11 (PDA repaired)",
            "MYH11-FAMILY-CASCADE-PDA-MVP: First-degree relatives of MYH11 probands should be screened for BOTH aortic aneurysm (TTE) AND PDA/MVP (paediatric cardiology review if <18 years, adult cardiology TTE if adult) — family screening identifies PDA in children who may not yet have aortic dilation; early intervention for PDA prevents pulmonary hypertension",
            "MYH11-SURGICAL-THRESHOLD-50MM: MYH11 aortic root surgical threshold ≥50 mm (similar to MFS, less aggressive than LDS ≥42 mm) — annual TTE; CT/MRA of full thoracic aorta at diagnosis and every 3–5 years; advance to ≥45 mm with rapid progression or family dissection history; discuss with specialist at dedicated HTAD clinic",
            "MYH11-BETA-BLOCKER-LIFELONG: Beta-blocker (bisoprolol/atenolol/metoprolol) should be started at diagnosis and maintained lifelong — reduces aortic wall stress and growth rate; target resting HR 55–65 bpm; losartan adjunct may be added (by analogy with MFS/LDS evidence, though MYH11-specific trial data are limited)",
            "MYH11-DISTINGUISH-ACTA2-LDS-MFS: MYH11 vs ACTA2 — no livedo reticularis or iris flocculi in MYH11; MYH11 has PDA + MVP triad; MYH11 vs LDS — no bifid uvula in MYH11; LDS has head-to-pelvis tortuosity + smaller surgical thresholds; MYH11 vs MFS — no ectopia lentis, no Marfan skeletal features; HTAD multigene panel resolves all ambiguous cases",
        ],
    },
    # ── SMAD3 — Loeys-Dietz Syndrome 3 / Aneurysm-Osteoarthritis Syndrome ──
    {
        "gene": "SMAD3",
        "protein": "SMAD3 — LDS3/AOS, PATHOGNOMONIC Early Osteoarthritis in 20s + Aortic Aneurysm, Aggressive Surgery ≥42 mm, Losartan + Beta-Blocker, OA Management NSAIDs",
        "alias": (
            "SMAD3; OMIM gene 603109; Loeys-Dietz Syndrome 3 / Aneurysm-Osteoarthritis Syndrome OMIM 613795; "
            "15q22.33; 425 aa; ~48 kDa; "
            "SMAD3 encodes SMAD family member 3, a signal transducer in the TGF-β canonical "
            "signalling pathway. SMAD3 is an R-Smad (receptor-activated Smad) phosphorylated "
            "by TGFBR1 (ALK5) at its C-terminal SSXS motif → pSMAD3 complexes with SMAD4 "
            "→ nuclear translocation → TGF-β-responsive gene transcription. "
            "SMAD3 loss-of-function variants cause Loeys-Dietz Syndrome type 3 (LDS3), "
            "also historically termed Aneurysm-Osteoarthritis Syndrome (AOS). "
            "PATHOGNOMONIC FEATURE: early-onset osteoarthritis (OA) of SMALL JOINTS "
            "(proximal interphalangeal joints, distal interphalangeal joints, thumb carpometacarpal) "
            "AND KNEES, diagnosed in the 2nd–3rd decade of life — combined with aortic aneurysm "
            "in the same patient or family. OA at age 20–35 in combination with any aortic "
            "aneurysm should trigger SMAD3 sequencing immediately. "
            "Additional LDS3/AOS features: bifid uvula (less common than in LDS1/2), "
            "arterial tortuosity (intracranial + thoracic + abdominal), mildly dysmorphic "
            "facial features (hypertelorism, depressed nasal bridge), joint hypermobility "
            "(Beighton ≥5), skin abnormalities, variable spinal deformity. "
            "Aggressive aortic surgery: threshold ≥42 mm (same as LDS1/2); prophylactic "
            "repair before 42 mm with rapid progression or family dissection. "
            "Medical therapy: losartan + beta-blocker (by analogy with LDS1/2; no LDS3 "
            "specific RCT). OA management: NSAIDs, physiotherapy, joint protection; "
            "avoid early joint replacement if possible (infection/healing risk in CTD)."
        ),
        "aa": "425 aa",
        "kDa": "~48 kDa",
        "locus": "15q22.33",
        "omim_gene": 603109,
        "omim_disease": 613795,
        "inheritance": "AD — autosomal dominant loss-of-function; heterozygous missense and truncating variants; de novo ~40%; familial AOS with dominant inheritance across generations",
        "gene_class": (
            "SMAD3 is a 425-amino acid intracellular signal transducer. Domain architecture: "
            "N-terminal MH1 domain (aa 1–145) — binds SMAD-binding elements (SBE, GTCT motif) "
            "in target gene promoters; β-hairpin structure inserts into the major groove of "
            "DNA; L1 loop (aa 37–53) contacts DNA backbone; also binds importin-β3 for "
            "nuclear translocation; C-terminal MH2 domain (aa 235–425) — the signal "
            "transduction domain; three α-helices (H1–H3) and five β-sheets form a flat "
            "surface (L3 loop + helix H1 + β8) for SMAD4 heterodimerisation; phosphorylated "
            "SSXS motif (Ser423/Ser425) by TGFBR1 → pSMAD3; phosphorylation promotes "
            "pSMAD3-SMAD4 complex formation; 'saddle surface' of MH2 interacts with "
            "transcriptional co-activators (CBP/p300, ARC105/MED15); linker region (aa 146–234) "
            "— connects MH1 and MH2; multiple serine/threonine residues phosphorylated by "
            "MAPK, CDK, GSK3β (inhibitory phosphorylation reduces SMAD3 activity in "
            "physiological contexts). SMAD3 LOF paradoxically enhances TGF-β pathway "
            "activity in connective tissues via compensatory feedback — same paradox as "
            "TGFBR1/TGFBR2 LOF in LDS1/2. The OA in SMAD3 LOF may reflect loss of "
            "SMAD3-mediated cartilage homeostasis signalling: SMAD3 normally maintains "
            "chondrocyte differentiation state; LOF → premature chondrocyte hypertrophy → "
            "cartilage degeneration → early OA. SMAD3 is distinct from SMAD2 "
            "(15q22.33, same chromosomal band — confirm gene identity by OMIM/RefSeq)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("SMAD3 MH2 domain missense AD — LDS3/AOS, early OA 20s–30s + aortic aneurysm, arterial tortuosity, aggressive surgery ≥42 mm", 0.40),
            ("SMAD3 MH1 domain missense AD — LDS3 spectrum, OA prominent, mild aortic root, variable facial features, joint laxity", 0.25),
            ("SMAD3 truncating/frameshift AD — LDS3 haploinsufficiency, severe OA + aneurysm, bifid uvula, multi-vessel tortuosity", 0.25),
            ("SMAD3 linker-region missense AD — AOS mild, early OA diagnostic clue, aortic aneurysm moderate, cascade essential", 0.10),
        ],
        "key_alerts": [
            "SMAD3-EARLY-OA-AORTIC-ANEURYSM-PATHOGNOMONIC: Early-onset OA (age 20–35) in SMALL JOINTS (PIP/DIP/thumb CMC) or knees COMBINED with any aortic aneurysm in the SAME patient or family = PATHOGNOMONIC for SMAD3/LDS3 — order SMAD3 sequencing immediately; do NOT dismiss as 'normal wear and tear' OA in a young person with aneurysm",
            "SMAD3-SURGERY-42MM-AGGRESSIVE: SMAD3 aortic surgery threshold ≥42 mm (same as LDS1/2, more aggressive than MFS ≥50 mm) — head-to-pelvis MRA at diagnosis for whole-arterial-tree aneurysm mapping; intracranial, vertebral, celiac, mesenteric aneurysms occur; surgery 42 mm OR lower with rapid progression/family dissection history",
            "SMAD3-OA-NSAIDs-JOINT-PROTECTION: OA management in SMAD3: NSAIDs (with PPI gastroprotection), physiotherapy, joint protection (avoid high-impact exercise, heavy manual work); steroid injections for acute OA flares; refer to rheumatology for OA management alongside cardiology for aortic surveillance; delay joint replacement if possible (infection and wound risk higher in CTD)",
            "SMAD3-LDS3-SAME-AS-LDS1-2-PROTOCOL: SMAD3 LDS3 management follows SAME protocol as TGFBR1/TGFBR2 (LDS1/2) — losartan primary + beta-blocker adjunct; head-to-pelvis MRA every 2 years; aggressive surgical threshold ≥42 mm; NO contact sports; cascade screen first-degree relatives; pregnancy higher risk at smaller aortic diameters",
            "SMAD3-BIFID-UVULA-LESS-COMMON: Bifid uvula is present in ~50% of LDS3/AOS cases (less common than in LDS1/2 where it is nearly universal) — absence of bifid uvula does NOT exclude SMAD3; the EARLY OA + aneurysm combination is the more specific diagnostic clue for SMAD3 vs TGFBR1/TGFBR2",
            "SMAD3-CASCADE-SCREEN-OA-AND-AORTA: First-degree relatives of SMAD3 probands must be screened for BOTH early OA (clinical examination + X-rays of hands/knees) AND aortic aneurysm (TTE) — a relative with OA alone but no aneurysm yet may be SMAD3 carrier; relatives with aneurysm alone may lack clinical OA; BOTH phenotypes must be assessed at cascade screening visits",
        ],
    },
]


def _make_cohort(gd):
    r = random.Random(gd["seed"])
    gene = gd["gene"]
    pts = []
    etiols = gd["etiologies"]
    weights = [e[1] for e in etiols]
    labels = [e[0] for e in etiols]

    for i in range(gd["n_patients"]):
        roll = r.random()
        cumul = 0.0
        etiol = labels[-1]
        for lbl, wt in zip(labels, weights):
            cumul += wt
            if roll < cumul:
                etiol = lbl
                break

        sex = r.choice(["M", "F"])

        # Age at diagnosis
        if gene == "FBN1":
            age_dx = r.gauss(22, 12)      # often diagnosed in teens/20s; range 0–50
        elif gene in ("TGFBR1", "TGFBR2"):
            age_dx = r.gauss(18, 12)
        elif gene == "COL3A1":
            age_dx = r.gauss(28, 10)      # often after first major event
        elif gene == "COL5A1":
            age_dx = r.gauss(25, 10)
        elif gene == "ACTA2":
            age_dx = r.gauss(20, 12)
        elif gene == "MYH11":
            age_dx = r.gauss(30, 12)
        elif gene == "SMAD3":
            age_dx = r.gauss(30, 8)       # OA triggers investigation in 30s
        else:
            age_dx = r.gauss(25, 10)
        age_dx = max(0.5, round(age_dx, 1))

        # Dx delay
        if gene in ("TGFBR1", "TGFBR2", "SMAD3"):
            dx_delay = r.gauss(36, 24)    # LDS often misdiagnosed as MFS
        elif gene == "COL3A1":
            dx_delay = r.gauss(24, 18)    # vEDS may be diagnosed after first event
        elif gene == "ACTA2":
            dx_delay = r.gauss(48, 24)    # livedo/iris flocculi not always recognised
        elif gene == "MYH11":
            dx_delay = r.gauss(42, 24)
        else:
            dx_delay = r.gauss(24, 18)
        dx_delay = max(0.0, round(dx_delay, 0))

        # Aortic root diameter at diagnosis (mm)
        if gene == "FBN1":
            aortic_root_mm = r.gauss(42, 6)
        elif gene in ("TGFBR1", "TGFBR2", "SMAD3"):
            aortic_root_mm = r.gauss(38, 5)    # LDS diagnosed at smaller sizes
        elif gene == "COL3A1":
            aortic_root_mm = r.gauss(35, 5)    # vEDS — root not always enlarged
        elif gene == "COL5A1":
            aortic_root_mm = r.gauss(34, 4)    # cEDS — mild/no aortic involvement
        elif gene == "ACTA2":
            aortic_root_mm = r.gauss(40, 6)
        elif gene == "MYH11":
            aortic_root_mm = r.gauss(41, 5)
        else:
            aortic_root_mm = r.gauss(38, 5)
        aortic_root_mm = round(max(25.0, min(70.0, aortic_root_mm)), 1)

        # Gene-specific features
        beta_blocker = r.random() < (0.85 if gene in ("FBN1","TGFBR1","TGFBR2","ACTA2","MYH11","SMAD3") else 0.40)
        losartan = r.random() < (0.80 if gene in ("TGFBR1","TGFBR2","FBN1","SMAD3") else 0.35)
        aortic_surgery = False
        if gene == "FBN1":
            aortic_surgery = aortic_root_mm >= 50 and r.random() < 0.80
        elif gene in ("TGFBR1", "TGFBR2", "SMAD3"):
            aortic_surgery = aortic_root_mm >= 42 and r.random() < 0.80
        elif gene == "ACTA2":
            aortic_surgery = aortic_root_mm >= 47 and r.random() < 0.75
        elif gene == "MYH11":
            aortic_surgery = aortic_root_mm >= 50 and r.random() < 0.75
        aortic_dissection = r.random() < (0.12 if gene in ("FBN1","TGFBR1","TGFBR2","SMAD3") else 0.08 if gene in ("ACTA2","MYH11") else 0.05)
        ectopia_lentis = r.random() < (0.65 if gene == "FBN1" else 0.02)
        livedo_reticularis = r.random() < (0.85 if gene == "ACTA2" and "Arg179" in etiol else 0.05)
        iris_flocculi = r.random() < (0.75 if gene == "ACTA2" and "Arg179" in etiol else 0.02)
        moyamoya = r.random() < (0.40 if gene == "ACTA2" and "Arg179" in etiol else 0.02)
        pda_history = r.random() < (0.65 if gene == "MYH11" else 0.05)
        mvp = r.random() < (0.40 if gene == "MYH11" else 0.10 if gene == "FBN1" else 0.07)
        bifid_uvula = r.random() < (0.85 if gene in ("TGFBR1","TGFBR2") else 0.50 if gene == "SMAD3" else 0.04)
        hypertelorism = r.random() < (0.75 if gene in ("TGFBR1","TGFBR2","SMAD3") else 0.04)
        arterial_tortuosity = r.random() < (0.90 if gene in ("TGFBR1","TGFBR2","SMAD3") else 0.15 if gene in ("FBN1","ACTA2","MYH11") else 0.05)
        skin_hyperextensibility = r.random() < (0.90 if gene == "COL5A1" else 0.20 if gene in ("TGFBR1","TGFBR2") else 0.15)
        atrophic_scarring = r.random() < (0.85 if gene == "COL5A1" else 0.05)
        joint_hypermobility = r.random() < (0.85 if gene in ("COL5A1","TGFBR1","TGFBR2","SMAD3") else 0.30 if gene == "FBN1" else 0.15)
        early_oa = r.random() < (0.85 if gene == "SMAD3" else 0.10)
        arterial_rupture = r.random() < (0.35 if gene == "COL3A1" else 0.02)
        bowel_perforation = r.random() < (0.18 if gene == "COL3A1" else 0.01)
        celiprolol = r.random() < (0.75 if gene == "COL3A1" else 0.03)
        sport_restriction = r.random() < (0.90 if gene in ("FBN1","TGFBR1","TGFBR2","ACTA2","MYH11","SMAD3") else 0.60)
        cascade_tested = r.random() < 0.70
        de_novo = r.random() < (0.45 if gene in ("TGFBR1","TGFBR2","SMAD3","ACTA2") else 0.25)

        pts.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_at_diagnosis": age_dx,
            "dx_delay_months": dx_delay,
            "aortic_root_mm": aortic_root_mm,
            "beta_blocker": beta_blocker,
            "losartan": losartan,
            "aortic_surgery": aortic_surgery,
            "aortic_dissection": aortic_dissection,
            "ectopia_lentis": ectopia_lentis,
            "livedo_reticularis": livedo_reticularis,
            "iris_flocculi": iris_flocculi,
            "moyamoya": moyamoya,
            "pda_history": pda_history,
            "mvp": mvp,
            "bifid_uvula": bifid_uvula,
            "hypertelorism": hypertelorism,
            "arterial_tortuosity": arterial_tortuosity,
            "skin_hyperextensibility": skin_hyperextensibility,
            "atrophic_scarring": atrophic_scarring,
            "joint_hypermobility": joint_hypermobility,
            "early_oa": early_oa,
            "arterial_rupture": arterial_rupture,
            "bowel_perforation": bowel_perforation,
            "celiprolol": celiprolol,
            "sport_restriction": sport_restriction,
            "cascade_tested": cascade_tested,
            "de_novo": de_novo,
        })
    return pts


def _pct(lst, key, val=True):
    if not lst:
        return 0.0
    return round(100.0 * sum(1 for x in lst if x.get(key) == val) / len(lst), 1)


def get_overview():
    all_pts = []
    for gd in CONNECTIVE_TISSUE_GENES:
        all_pts.extend(_make_cohort(gd))

    gene_pts = {}
    for gd in CONNECTIVE_TISSUE_GENES:
        gene_pts[gd["gene"]] = _make_cohort(gd)

    fbn1 = gene_pts["FBN1"]
    tgfbr1 = gene_pts["TGFBR1"]
    tgfbr2 = gene_pts["TGFBR2"]
    col3a1 = gene_pts["COL3A1"]
    col5a1 = gene_pts["COL5A1"]
    acta2 = gene_pts["ACTA2"]
    myh11 = gene_pts["MYH11"]
    smad3 = gene_pts["SMAD3"]

    s = {
        "total_patients": len(all_pts),
        "mean_dx_age_years": round(sum(p["age_at_diagnosis"] for p in all_pts) / len(all_pts), 1),
        "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in all_pts) / len(all_pts), 0),
        "aortic_surgery_pct": _pct(all_pts, "aortic_surgery"),
        "aortic_dissection_pct": _pct(all_pts, "aortic_dissection"),
        "beta_blocker_pct": _pct(all_pts, "beta_blocker"),
        "losartan_pct": _pct(all_pts, "losartan"),
        "sport_restriction_pct": _pct(all_pts, "sport_restriction"),
        "cascade_tested_pct": _pct(all_pts, "cascade_tested"),
        "de_novo_pct": _pct(all_pts, "de_novo"),
        # FBN1 Marfan
        "fbn1_ectopia_lentis_pct": _pct(fbn1, "ectopia_lentis"),
        "fbn1_aortic_surgery_pct": _pct(fbn1, "aortic_surgery"),
        "fbn1_beta_blocker_pct": _pct(fbn1, "beta_blocker"),
        "fbn1_losartan_pct": _pct(fbn1, "losartan"),
        "fbn1_sport_restriction_pct": _pct(fbn1, "sport_restriction"),
        "fbn1_joint_hypermobility_pct": _pct(fbn1, "joint_hypermobility"),
        # TGFBR1 LDS1
        "tgfbr1_bifid_uvula_pct": _pct(tgfbr1, "bifid_uvula"),
        "tgfbr1_hypertelorism_pct": _pct(tgfbr1, "hypertelorism"),
        "tgfbr1_arterial_tortuosity_pct": _pct(tgfbr1, "arterial_tortuosity"),
        "tgfbr1_aortic_surgery_pct": _pct(tgfbr1, "aortic_surgery"),
        "tgfbr1_losartan_pct": _pct(tgfbr1, "losartan"),
        # TGFBR2 LDS2
        "tgfbr2_bifid_uvula_pct": _pct(tgfbr2, "bifid_uvula"),
        "tgfbr2_arterial_tortuosity_pct": _pct(tgfbr2, "arterial_tortuosity"),
        "tgfbr2_aortic_surgery_pct": _pct(tgfbr2, "aortic_surgery"),
        "tgfbr2_joint_hypermobility_pct": _pct(tgfbr2, "joint_hypermobility"),
        # COL3A1 vEDS
        "col3a1_arterial_rupture_pct": _pct(col3a1, "arterial_rupture"),
        "col3a1_bowel_perforation_pct": _pct(col3a1, "bowel_perforation"),
        "col3a1_celiprolol_pct": _pct(col3a1, "celiprolol"),
        "col3a1_aortic_dissection_pct": _pct(col3a1, "aortic_dissection"),
        "col3a1_skin_hyperextensibility_pct": _pct(col3a1, "skin_hyperextensibility"),
        # COL5A1 cEDS
        "col5a1_skin_hyperextensibility_pct": _pct(col5a1, "skin_hyperextensibility"),
        "col5a1_atrophic_scarring_pct": _pct(col5a1, "atrophic_scarring"),
        "col5a1_joint_hypermobility_pct": _pct(col5a1, "joint_hypermobility"),
        "col5a1_aortic_surgery_pct": _pct(col5a1, "aortic_surgery"),
        # ACTA2 HTAD
        "acta2_livedo_reticularis_pct": _pct(acta2, "livedo_reticularis"),
        "acta2_iris_flocculi_pct": _pct(acta2, "iris_flocculi"),
        "acta2_moyamoya_pct": _pct(acta2, "moyamoya"),
        "acta2_aortic_surgery_pct": _pct(acta2, "aortic_surgery"),
        "acta2_pda_history_pct": _pct(acta2, "pda_history"),
        # MYH11 HTAD+PDA
        "myh11_pda_history_pct": _pct(myh11, "pda_history"),
        "myh11_mvp_pct": _pct(myh11, "mvp"),
        "myh11_aortic_surgery_pct": _pct(myh11, "aortic_surgery"),
        "myh11_beta_blocker_pct": _pct(myh11, "beta_blocker"),
        # SMAD3 LDS3/AOS
        "smad3_early_oa_pct": _pct(smad3, "early_oa"),
        "smad3_bifid_uvula_pct": _pct(smad3, "bifid_uvula"),
        "smad3_arterial_tortuosity_pct": _pct(smad3, "arterial_tortuosity"),
        "smad3_aortic_surgery_pct": _pct(smad3, "aortic_surgery"),
        "smad3_losartan_pct": _pct(smad3, "losartan"),
    }

    genes_out = []
    for gd in CONNECTIVE_TISSUE_GENES:
        pts = gene_pts[gd["gene"]]
        genes_out.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "key_alerts": gd["key_alerts"],
            "mean_dx_age": round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
            "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in pts) / len(pts), 0),
            "mean_aortic_root_mm": round(sum(p["aortic_root_mm"] for p in pts) / len(pts), 1),
            "aortic_surgery_pct": _pct(pts, "aortic_surgery"),
            "aortic_dissection_pct": _pct(pts, "aortic_dissection"),
        })

    top_alerts = []
    for gd in CONNECTIVE_TISSUE_GENES:
        for alert in gd["key_alerts"][:2]:
            top_alerts.append({"gene": gd["gene"], "alert": alert})

    return {
        "dashboard": "Hereditary Connective Tissue Disorders Atlas",
        "subtitle": "Complete 8-Gene Hereditary Connective Tissue Disorder Reference — FBN1/TGFBR1/TGFBR2/COL3A1/COL5A1/ACTA2/MYH11/SMAD3",
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
        "aggregate_stats": s,
        "top_alerts": top_alerts,
        "genes": genes_out,
    }


def get_breakdown():
    out = {}
    for gd in CONNECTIVE_TISSUE_GENES:
        pts = _make_cohort(gd)
        gene = gd["gene"]

        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        age_buckets = {"<10": 0, "10–19": 0, "20–29": 0, "30–39": 0, "40–49": 0, "≥50": 0}
        for p in pts:
            a = p["age_at_diagnosis"]
            if a < 10:
                age_buckets["<10"] += 1
            elif a < 20:
                age_buckets["10–19"] += 1
            elif a < 30:
                age_buckets["20–29"] += 1
            elif a < 40:
                age_buckets["30–39"] += 1
            elif a < 50:
                age_buckets["40–49"] += 1
            else:
                age_buckets["≥50"] += 1

        root_buckets = {"<35": 0, "35–39": 0, "40–44": 0, "45–49": 0, "50–54": 0, "≥55": 0}
        for p in pts:
            rm = p["aortic_root_mm"]
            if rm < 35:
                root_buckets["<35"] += 1
            elif rm < 40:
                root_buckets["35–39"] += 1
            elif rm < 45:
                root_buckets["40–44"] += 1
            elif rm < 50:
                root_buckets["45–49"] += 1
            elif rm < 55:
                root_buckets["50–54"] += 1
            else:
                root_buckets["≥55"] += 1

        out[gene] = {
            "gene": gene,
            "protein": gd["protein"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "n_patients": len(pts),
            "etiologies": etiol_counts,
            "age_at_diagnosis_distribution": age_buckets,
            "aortic_root_distribution_mm": root_buckets,
            "stats": {
                "beta_blocker_pct": _pct(pts, "beta_blocker"),
                "losartan_pct": _pct(pts, "losartan"),
                "aortic_surgery_pct": _pct(pts, "aortic_surgery"),
                "aortic_dissection_pct": _pct(pts, "aortic_dissection"),
                "ectopia_lentis_pct": _pct(pts, "ectopia_lentis"),
                "livedo_reticularis_pct": _pct(pts, "livedo_reticularis"),
                "iris_flocculi_pct": _pct(pts, "iris_flocculi"),
                "moyamoya_pct": _pct(pts, "moyamoya"),
                "pda_history_pct": _pct(pts, "pda_history"),
                "mvp_pct": _pct(pts, "mvp"),
                "bifid_uvula_pct": _pct(pts, "bifid_uvula"),
                "hypertelorism_pct": _pct(pts, "hypertelorism"),
                "arterial_tortuosity_pct": _pct(pts, "arterial_tortuosity"),
                "skin_hyperextensibility_pct": _pct(pts, "skin_hyperextensibility"),
                "atrophic_scarring_pct": _pct(pts, "atrophic_scarring"),
                "joint_hypermobility_pct": _pct(pts, "joint_hypermobility"),
                "early_oa_pct": _pct(pts, "early_oa"),
                "arterial_rupture_pct": _pct(pts, "arterial_rupture"),
                "bowel_perforation_pct": _pct(pts, "bowel_perforation"),
                "celiprolol_pct": _pct(pts, "celiprolol"),
                "sport_restriction_pct": _pct(pts, "sport_restriction"),
                "cascade_tested_pct": _pct(pts, "cascade_tested"),
                "de_novo_pct": _pct(pts, "de_novo"),
                "mean_dx_age": round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
                "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in pts) / len(pts), 0),
                "mean_aortic_root_mm": round(sum(p["aortic_root_mm"] for p in pts) / len(pts), 1),
            },
            "key_alerts": gd["key_alerts"],
            "patients": pts[:10],
        }
    return out


def get_definitions():
    return {
        "atlas": "Hereditary Connective Tissue Disorders Atlas — Complete 8-Gene Reference",
        "genes_covered": [gd["gene"] for gd in CONNECTIVE_TISSUE_GENES],
        "concepts": {
            "Marfan_Syndrome": (
                "Autosomal dominant connective tissue disorder (FBN1, 15q21.1); prevalence 1:5,000; "
                "aortic root aneurysm + ectopia lentis + skeletal features. Revised Ghent criteria 2010: "
                "aortic root Z-score ≥2 OR dissection + ectopia lentis = definitive; systemic score ≥7 "
                "supports diagnosis. Arm span > height (>1.05 ratio) + upper:lower body ratio <0.85. "
                "Treatment: beta-blocker + losartan; surgery ≥50 mm root; no contact sports lifelong."
            ),
            "Loeys_Dietz_Syndrome": (
                "AD connective tissue disorder (TGFBR1/TGFBR2/SMAD3 LOF or TGFBR1/TGFBR2 rarely GOF); "
                "triad: bifid uvula/cleft palate + hypertelorism + arterial tortuosity; aneurysm at smaller "
                "diameters — prophylactic surgery ≥42 mm (NOT ≥50 mm as in MFS); head-to-pelvis MRA mandatory; "
                "losartan primary pharmacotherapy. LDS3 (SMAD3) has PATHOGNOMONIC early OA + aneurysm combination."
            ),
            "Vascular_EDS": (
                "AD disorder (COL3A1 — type III collagen); most lethal EDS; spontaneous arterial rupture "
                "(celiac/mesenteric/renal), sigmoid colon perforation, uterine rupture; 25% major complication "
                "by age 25; median survival 48 years. NO elective surgery; celiprolol RCT evidence (BBEST trial). "
                "Skin translucent with visible veins; minimal skin hyperextensibility."
            ),
            "Classical_EDS": (
                "AD disorder (COL5A1/COL5A2 haploinsufficiency — type V collagen); triad: skin hyperextensibility "
                "+ widened atrophic scarring (cigarette-paper scars) + joint hypermobility (Beighton ≥5). "
                "NOT vascular danger; management supportive — physiotherapy, bracing, wound care. Prevalence 1:20,000–40,000."
            ),
            "HTAD": (
                "Hereditary Thoracic Aortic Disease — familial aneurysm-dissection due to SMC contractile "
                "protein variants (ACTA2 — α-SMA; MYH11 — SMyHC). ACTA2 Arg179 variants: PATHOGNOMONIC livedo "
                "reticularis + iris flocculi + Moyamoya + PDA. MYH11: diagnostic triad aortic aneurysm + PDA + MVP."
            ),
            "Ghent_Criteria_2010": (
                "Revised Marfan diagnostic criteria: aortic root Z-score + ectopia lentis + systemic score (0–20 points) "
                "+ FBN1 variant. Definitive MFS: aortic root ≥2 SD + ectopia lentis; OR aortic root ≥2 SD + FBN1 "
                "pathogenic variant; OR aortic root ≥2 SD + systemic score ≥7; OR ectopia lentis + FBN1 variant "
                "with known aortic association."
            ),
            "Celiprolol_vEDS": (
                "Beta-1 selective adrenergic blocker with partial beta-2 agonism; only treatment with RCT evidence "
                "for vEDS (COL3A1) — BBEST trial (Ong 2010 NEJM) showed significant reduction in major vascular "
                "events over 5 years at 200–400 mg/day. Standard pharmacotherapy for vEDS; start at 200 mg/day."
            ),
            "Losartan_CTD": (
                "Angiotensin II type-1 receptor blocker (AT1R); blocks TGF-β signalling pathway indirectly via "
                "AT1R; COMPARE trial confirmed aortic root growth reduction in MFS; used in LDS1/2/3 as primary "
                "TGF-β antagonist. Teratogenic — MUST be stopped at conception; switch to beta-blocker for pregnancy. "
                "Target dose: 1.4 mg/kg/day in children; 100 mg/day in adults."
            ),
            "Livedo_Reticularis": (
                "Mottled, reticulate (net-like) skin discolouration of extremities due to vascular smooth muscle "
                "dysfunction in small cutaneous arterioles. In ACTA2 Arg179 variants: present from childhood; "
                "combined with iris flocculi = PATHOGNOMONIC for ACTA2 Arg179. Distinguished from physiological "
                "cutis marmorata (transient, cold-induced, symmetric, non-fixed) — livedo in ACTA2 is fixed and "
                "persistent regardless of temperature."
            ),
            "Iris_Flocculi": (
                "Iris neovascular tufts (pigment clumps on anterior iris surface); appear as irregular, "
                "brown-red elevated lesions on slit-lamp examination. PATHOGNOMONIC for ACTA2 Arg179 variants "
                "when combined with livedo reticularis. Caused by vascular smooth muscle dysfunction in ciliary "
                "body vasculature. Not the same as iris heterochromia or iris nevi."
            ),
            "Moyamoya": (
                "Progressive bilateral stenosis of the terminal internal carotid arteries (ICAs) with "
                "compensatory basal collateral vessel formation resembling 'puff of smoke' on angiography "
                "(moya = smoke in Japanese). In ACTA2 Arg179 variants: VSMC dysfunction → ICA smooth muscle "
                "stenosis → collateral formation. Risk of ischaemic stroke (especially in childhood). "
                "Treatment: neurosurgical revascularisation (EDAS — encephaloduroarteriosynangiosis) for "
                "flow-limiting stenosis."
            ),
            "Aortic_Root_Z_score": (
                "Number of standard deviations above the age/BSA-adjusted mean aortic root diameter. "
                "Z-score ≥2 at the level of the sinuses of Valsalva = aortic root aneurysm by Ghent criteria. "
                "Z-score calculation requires echocardiographic measurement at aortic root (sinus level) and "
                "BSA (body surface area) normalisation. Marfan surgical threshold: absolute root ≥50 mm OR "
                "Z-score ≥10 in children."
            ),
            "LDS_Surgical_Threshold": (
                "Loeys-Dietz Syndrome (TGFBR1/TGFBR2/SMAD3) requires prophylactic aortic surgery at ≥42 mm "
                "(some centres use ≥40 mm) — NOT ≥50 mm as in Marfan syndrome. LDS aneurysms dissect/rupture "
                "at smaller diameters than Marfan aneurysms. This is the single most critical clinical decision "
                "point distinguishing LDS from MFS management."
            ),
            "EDS_Subtypes": (
                "Ehlers-Danlos Syndrome: 13 subtypes by 2017 international classification. Classical EDS "
                "(COL5A1/COL5A2 — skin + scarring + hypermobility, not vascular); Vascular EDS "
                "(COL3A1 — lethal, arterial rupture); Kyphoscoliotic (PLOD1/FKBP14 — scoliosis + muscle "
                "hypotonia); Arthrochalasia (COL1A1/1A2 — severe joint hypermobility); Dermatosparaxis "
                "(ADAMTS2 — skin fragility); Hypermobile EDS (hEDS — no known gene, most common clinically). "
                "Gene panel separates subtypes by molecular confirmation."
            ),
            "Beighton_Score": (
                "9-point joint hypermobility score. Points: bilateral little finger dorsiflexion ≥90° (2); "
                "bilateral thumb-to-forearm opposition (2); bilateral elbow hyperextension ≥10° (2); "
                "bilateral knee hyperextension ≥10° (2); palms flat on floor with knees straight (1). "
                "≥5/9 = generalised joint hypermobility in adults (≥4/9 in children <16 years or adults >50 years)."
            ),
            "Cascade_Screening": (
                "First-degree relative (parent, sibling, child) genetic + clinical screening of index case "
                "(proband) with confirmed pathogenic variant. All HCTD genes require cascade screening: "
                "FBN1 — TTE + ophthalmology slit-lamp; TGFBR1/2 — TTE + head-to-pelvis MRA; "
                "COL3A1 — COL3A1 molecular + clinical exam + celiprolol start; COL5A1 — clinical exam "
                "(skin + joints) + Beighton score; ACTA2 — TTE + skin exam (livedo) + ophthalmology; "
                "MYH11 — TTE + childhood cardiac records; SMAD3 — TTE + joint exam + X-rays of hands/knees."
            ),
        },
        "key_standards": [
            "Loeys BL et al. Nature Genetics 2005 — first description of LDS (TGFBR1/TGFBR2)",
            "Ong KT et al. NEJM 2010 — celiprolol for vascular EDS (BBEST trial)",
            "Radonic T et al. PLOS ONE 2011 — losartan in Marfan syndrome",
            "Lacro RV et al. NEJM 2014 — COMPARE trial: losartan vs atenolol in MFS",
            "Malfait F et al. AJMG 2017 — 2017 international EDS classification",
            "Loeys BL & Dietz HC GeneReviews 2022 — Loeys-Dietz Syndrome comprehensive review",
            "Pyeritz RE GeneReviews 2021 — Marfan syndrome comprehensive review",
            "Braverman AC et al. JACC 2018 — HTAD genetics (ACTA2/MYH11/SMAD3)",
            "Meester JA et al. Genetics in Medicine 2017 — SMAD3 AOS/LDS3 natural history",
            "Pepin MG et al. GeneReviews 2022 — vascular EDS (COL3A1) comprehensive review",
        ],
        "pharmacological_distinctions": [
            "FBN1-MARFAN: Beta-blocker (reduces aortic growth rate) + losartan (TGF-β antagonism, COMPARE trial) — surgery ≥50 mm; losartan STOP at conception",
            "TGFBR1/TGFBR2-LDS: Losartan PRIMARY (not just adjunct) + beta-blocker — surgery ≥42 mm NOT ≥50 mm; losartan STOP at conception (teratogenic)",
            "COL3A1-vEDS: Celiprolol (beta-1 selective + partial beta-2 agonism) — ONLY treatment with RCT evidence; NO losartan first-line; celiprolol maintained lifelong",
            "COL5A1-cEDS: NO pharmacotherapy for vascular prevention — management is purely supportive (physiotherapy, bracing, wound care); wound closure strips mandatory",
            "ACTA2-HTAD: Beta-blocker + statin (premature CAD prevention) — losartan by analogy with MFS/LDS; surgery ≥45–50 mm; cerebrovascular MRA for Arg179 Moyamoya",
            "MYH11-HTAD: Beta-blocker + losartan (by analogy) — surgery ≥50 mm; cascade screen for PDA + MVP in children",
            "SMAD3-LDS3: Losartan PRIMARY + beta-blocker (same as LDS1/2) + NSAIDs/physiotherapy for OA — surgery ≥42 mm; losartan STOP at conception",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Total patients: {ov['aggregate_stats']['total_patients']}")
    print(f"Genes: {[g['gene'] for g in ov['genes']]}")
    print(f"Seeds: {ov['seeds']}")
    print("\n=== BREAKDOWN (gene list) ===")
    bd = get_breakdown()
    for gene, info in bd.items():
        print(f"  {gene}: {info['n_patients']} pts, mean root {info['stats']['mean_aortic_root_mm']} mm, surgery {info['stats']['aortic_surgery_pct']}%")
    print("\n=== DEFINITIONS ===")
    df = get_definitions()
    print(f"Concepts: {len(df['concepts'])}")
    print(f"Standards: {len(df['key_standards'])}")
    print("OK")
