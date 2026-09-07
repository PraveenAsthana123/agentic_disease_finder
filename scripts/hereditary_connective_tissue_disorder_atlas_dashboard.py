#!/usr/bin/env python3
"""Hereditary-Connective-Tissue-Disorder-Atlas — Complete 8-Gene Atlas (Hereditary Connective Tissue Disorders)
FBN1    (Fibrillin-1; 2871 aa; 15q21.1; AD;
         Marfan Syndrome; aortic root dilation + ectopia lentis + skeletal;
         losartan/ARB reduces aortic growth; elective root repair at 4.5-5.0 cm;
         beta-blockers also standard; ectopia lentis in 60%;
         seed SEED_BASE+0) .
COL3A1  (Collagen III alpha 1; 1466 aa; 2q32.2; AD;
         Vascular EDS (vEDS); arterial/bowel/uterine rupture WITHOUT warning;
         celiprolol Level B — ONLY proven drug; NO elective surgery;
         most lethal EDS subtype; median survival 48 yr untreated;
         seed SEED_BASE+1) .
TGFBR2  (TGF-beta Receptor 2; 592 aa; 3p24.1; AD;
         Loeys-Dietz Syndrome type 2 (LDS2); aggressive aortic at SMALLER diameters;
         surgery threshold 4.0 cm (vs 4.5-5.0 Marfan); hypertelorism + bifid uvula;
         ALL aortic segments must be imaged (not just root);
         seed SEED_BASE+2) .
COL1A1  (Collagen I alpha 1; 1464 aa; 17q21.33; AD;
         Osteogenesis Imperfecta (OI) type I / II / III / IV;
         bone fragility; bisphosphonates reduce fracture rate 40-50%;
         blue sclerae = OI type I; white sclerae in severe forms;
         hearing loss 50% by age 50;
         seed SEED_BASE+3) .
ELN     (Elastin; 786 aa; 7q11.23; AD / 7q11.23 deletion;
         Supravalvular Aortic Stenosis (SVAS) + Autosomal Dominant Cutis Laxa (ADCL);
         Williams-Beuren syndrome = contiguous 7q11.23 deletion (ELN + 25 other genes);
         isolated SVAS from point mutations; infantile hypercalcemia in WBS;
         SVAS: fixed subarterial obstruction — surgical relief;
         seed SEED_BASE+4) .
ABCC6   (ABC transporter C6; 1503 aa; 16p13.1; AR;
         Pseudoxanthoma Elasticum (PXE); elastic fibre calcification of skin/eyes/arteries;
         angioid streaks on fundoscopy PATHOGNOMONIC; Bruch membrane calcification;
         premature peripheral arterial disease + gastrointestinal bleeding;
         vitamin K2 (MK-7) supplementation; anti-VEGF for choroidal neovascularization;
         seed SEED_BASE+5) .
COL5A1  (Collagen V alpha 1; 2836 aa; 9q34.3; AD;
         Classical EDS (cEDS); skin hyperextensibility + atrophic scarring (Gorlin sign);
         joint hypermobility; molluscoid pseudotumours over pressure points;
         NO curative therapy; 50% de novo mutations;
         seed SEED_BASE+6) .
TNXB    (Tenascin-XB; 4243 aa; 6p21.3; AR / AD (haploinsufficiency);
         Tenascin-X Deficiency (TNX-EDS); haploinsufficiency → hypermobility EDS;
         homozygous/compound het → severe EDS + adrenal insufficiency (CAH-X contiguous CYP21A2 deletion);
         tenascin-X is the ONLY validated ligand for collagen fibril spacing;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1942–1949)
"""

import random

SEED_BASE = 1942

CONNECTIVE_TISSUE_GENES = [
    # -- FBN1 — Fibrillin-1 / Marfan Syndrome -----------------------------------
    {
        "gene": "FBN1",
        "alt_name": "Fibrillin-1 (Marfan)",
        "protein": (
            "FBN1 -- 15q21.1 AD -- Fibrillin-1-2871aa -- "
            "Marfan-Syndrome-Aortic-Root-Dilation-Ectopia-Lentis-Skeletal -- "
            "Losartan-ARB-Reduce-Aortic-Growth-Level-B -- "
            "Elective-Root-Repair-4.5-5.0cm-Threshold -- "
            "Ectopia-Lentis-60pct-Upward-Temporal -- "
            "Thumb-Wrist-Sign-Arachnodactyly-Dolichostenomelia"
        ),
        "locus": "15q21.1",
        "protein_size": "2871 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Variable: aortic complications any age (neonatal Marfan most severe, rapid dilation in infancy); "
            "ectopia lentis often detected in childhood; skeletal features apparent in childhood/adolescence; "
            "aortic dissection risk rises steeply in 3rd-4th decade if untreated"
        ),
        "key_biomarker": (
            "Echocardiogram: aortic root Z-score ≥2 (age/BSA adjusted); aortic diameter absolute; "
            "slit-lamp: upward temporal ectopia lentis (vs downward in homocystinuria); "
            "skeletal: arm span > height, reduced upper:lower segment ratio, pectus excavatum/carinatum, scoliosis; "
            "thumb sign (Steinberg), wrist sign (Walker-Murdoch); "
            "molecular: FBN1 pathogenic variant; fibrillin-1 microfibril deficiency; "
            "Revised Ghent criteria 2010: aortic Z-score + ectopia lentis ± FBN1 variant ± family history"
        ),
        "pathognomonic": (
            "Aortic root Z-score ≥2 + ectopia lentis (upward/temporal) + tall stature + arachnodactyly = Marfan; "
            "Revised Ghent 2010: aortic root Z-score ≥2 + FBN1 = Marfan regardless of skeletal/lens; "
            "DISTINGUISH from homocystinuria: downward lens dislocation + intellectual disability + elevated homocysteine; "
            "DISTINGUISH from LDS: no ectopia lentis; more aggressive aortic (surgery at smaller diameter); hypertelorism; "
            "DISTINGUISH from MASS/MVP syndrome: Z-score <2; no ectopia lentis; milder course"
        ),
        "treatment": (
            "Beta-blockers (atenolol/propranolol) — first-line: reduce aortic wall stress; "
            "ARBs (losartan 0.6-1.4 mg/kg/day): reduce TGFβ signalling → aortic growth reduction; Level B; "
            "ACE-i can substitute ARB; avoid strenuous isometric exercise, contact sports, scuba diving; "
            "Elective prophylactic aortic root replacement: Z-score ≥5 in children, absolute ≥4.5 cm adults (≥5.0 cm low-risk); "
            "Ectopia lentis: aphakic/phakic contact lens or lens extraction — avoid ocular trauma; "
            "Annual echocardiogram: stable; 6-monthly if rapid growth (>0.5 cm/yr); "
            "Regular ophthalmology, orthopaedic, dental surveillance; genetic counselling"
        ),
        "critical_flags": [
            "FBN1-AORTIC-DISSECTION-EMERGENCY: acute onset back/chest pain = aortic dissection until proven otherwise; CT angiography STAT; systolic target <120 mmHg acutely; surgery ≤6hr from presentation",
            "FBN1-SURGERY-THRESHOLD-MATTERS: elective root repair at 4.5-5.0 cm prevents dissection; NEVER wait for symptoms; dissection at smaller diameters in family history of dissection — lower threshold 4.0-4.5 cm",
            "FBN1-ECTOPIA-LENTIS-UPWARD: lens dislocation is UPWARD-TEMPORAL in Marfan; downward dislocation = homocystinuria (different disorder); slit-lamp mandatory before Marfan diagnosis",
            "FBN1-GHENT-2010: Revised Ghent criteria (not 1996) required; ectopia lentis + FBN1 = Marfan even without Z-score ≥2; aortic Z-score ≥2 + FBN1 = Marfan even without ectopia lentis",
            "FBN1-NEONATAL-MARFAN: FBN1 mutations in exons 24-32 → most severe neonatal form; rapid aortic dilation; mitral regurgitation; respiratory failure; distinct from classical adult Marfan",
            "FBN1-EXERCISE-RESTRICTION: isometric exercise + contact sports + competitive athletics + scuba diving FORBIDDEN; aerobic low-impact exercise (swimming, walking) is encouraged",
            "FBN1-PREGNANCY-RISK: aortic root >4.0 cm at conception = high risk; prophylactic repair before pregnancy; beta-blockers/ARBs throughout; monthly echo in 3rd trimester",
            "FBN1-LOSARTAN-NOT-CURE: losartan reduces aortic growth rate but does NOT prevent surgery; continues indefinitely; does NOT replace surgery threshold monitoring",
            "FBN1-THUMB-WRIST-SIGNS: Steinberg (thumb projects beyond ulnar border) + Walker-Murdoch (thumbs overlap wrist) = arachnodactyly screening; both positive = high suspicion",
            "FBN1-DURA-ECTASIA: lumbar dural ectasia on MRI in 63-92% Marfan; low back/leg pain; NOT treated directly but confirms diagnosis; avoid epidural if emergency",
        ],
        "alias": (
            "FBN1 (Fibrillin-1) — AD — 15q21.1 — Marfan Syndrome — OMIM #154700 — "
            "2871 aa extracellular glycoprotein — microfibril scaffold — "
            "TGFβ sequestration impaired (LOF FBN1 → excess free TGFβ) — "
            "Revised Ghent 2010 criteria required — aortic/ocular/skeletal triad"
        ),
        "seed": SEED_BASE + 0,
    },

    # -- COL3A1 — Collagen III / Vascular EDS -----------------------------------
    {
        "gene": "COL3A1",
        "alt_name": "Collagen III alpha 1 (vEDS)",
        "protein": (
            "COL3A1 -- 2q32.2 AD -- CollagenIII-alpha1-1466aa -- "
            "Vascular-EDS-Arterial-Bowel-Uterine-Rupture-WITHOUT-Warning -- "
            "Celiprolol-Level-B-ONLY-Proven-Drug -- "
            "NO-Elective-Surgery-Absolute-CI -- "
            "Most-Lethal-EDS-Median-Survival-48yr -- "
            "Spontaneous-Pneumothorax-Carotid-Cavernous-Fistula"
        ),
        "locus": "2q32.2",
        "protein_size": "1466 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Major complications usually in 3rd-4th decade (range 1st-6th); "
            "spontaneous arterial rupture can occur in 20s-30s; "
            "uterine rupture risk in pregnancy (any trimester + delivery); "
            "50% have had a major complication by age 40"
        ),
        "key_biomarker": (
            "Molecular: COL3A1 pathogenic variant (haploinsufficiency OR dominant-negative) — dominant-negative worse; "
            "Skin biopsy: collagen III fibroblast culture + collagen gel assay (outdated but used historically); "
            "Type III collagen peptide (COL3-related fragments) in plasma — research; "
            "Skin features: thin translucent skin (visible superficial veins), easy bruising, characteristic facial features; "
            "NO specific blood test; diagnosis primarily molecular + clinical 2017 EDS nosology"
        ),
        "pathognomonic": (
            "Spontaneous arterial rupture (particularly medium-sized arteries: celiac, splenic, renal, mesenteric) in young adult + thin translucent skin + family history of arterial rupture; "
            "COL3A1 pathogenic variant confirms vEDS; "
            "DISTINGUISH from other EDS: vEDS has THIN translucent skin (NOT hyperextensible); "
            "classical EDS has hyperextensible skin; hypermobile EDS has no molecular test; "
            "Carotid-cavernous fistula in young adult without trauma = COL3A1 until proven otherwise"
        ),
        "treatment": (
            "Celiprolol (beta-1 blocker / partial beta-2 agonist) 400-800 mg/day: ONLY Level B evidence drug; "
            "reduces arterial events by 36% (Ong 2010 Lancet); mechanism: reduces aortic wall stiffness; "
            "NO elective surgery: operative mortality markedly elevated (vessel friability, poor wound healing); "
            "Emergency surgery ONLY when arterial rupture unavoidable; "
            "Blood pressure target: systolic <120 mmHg; avoid Valsalva manoeuvres, contact sports, heavy lifting; "
            "Pregnancy: HIGH RISK; uterine rupture any time; planned Caesarean; collagen-experienced team; "
            "Surveillance: annual CT/MRI angiography of aorta + branches; avoid invasive procedures; "
            "Genetic counselling: each child 50% risk"
        ),
        "critical_flags": [
            "COL3A1-NO-ELECTIVE-SURGERY-ABSOLUTE: elective surgery ABSOLUTELY CONTRAINDICATED — vessel friability means catastrophic intraoperative hemorrhage; even diagnostic angiography carries high risk; manage conservatively",
            "COL3A1-CELIPROLOL-ONLY-PROVEN: celiprolol is the ONLY Level B evidence drug for vEDS (Ong 2010 Lancet); losartan/ARB NOT proven effective in vEDS; beta-blockers alone (atenolol) suboptimal vs celiprolol",
            "COL3A1-ARTERIAL-RUPTURE-WITHOUT-WARNING: vEDS arterial rupture occurs WITHOUT prodromal symptoms; no sentinel event; young adult with sudden severe abdominal/chest pain = arterial rupture emergency",
            "COL3A1-TRANSLUCENT-SKIN-NOT-HYPEREXTENSIBLE: vEDS skin is thin and translucent (NOT hyperextensible); visible subcutaneous veins on chest/abdomen; easy bruising; DIFFERENT from classical EDS hyperextensibility",
            "COL3A1-PREGNANCY-LETHAL-RISK: uterine rupture can occur ANY trimester including first; maternal mortality ~12% per pregnancy; plan pregnancy with vascular/obstetric team; pre-natal diagnosis available",
            "COL3A1-CCF-PATHOGNOMONIC: carotid-cavernous fistula in young adult without major head trauma = COL3A1/vEDS until proven otherwise; pulsatile exophthalmos + bruit = emergency vascular referral",
            "COL3A1-DOMINANT-NEGATIVE-WORSE: glycine substitution mutations (dominant-negative) have worse prognosis than haploinsufficiency (splice site/frameshift); genotype-phenotype correlation meaningful",
            "COL3A1-SPONTANEOUS-PNEUMOTHORAX: bilateral or recurrent spontaneous pneumothorax in young adult = COL3A1 testing; pleurodesis may be needed but with caution (vascular friability around lung)",
        ],
        "alias": (
            "COL3A1 (Collagen III alpha 1) — AD — 2q32.2 — Vascular EDS (vEDS) — OMIM #130050 — "
            "1466 aa type III procollagen chain — fibrillar collagen of blood vessels/bowel/skin — "
            "haploinsufficiency + dominant-negative mutations — "
            "most severe EDS subtype — 2017 EDS nosology required"
        ),
        "seed": SEED_BASE + 1,
    },

    # -- TGFBR2 — TGF-beta Receptor 2 / Loeys-Dietz Syndrome -------------------
    {
        "gene": "TGFBR2",
        "alt_name": "TGF-β Receptor 2 (Loeys-Dietz LDS2)",
        "protein": (
            "TGFBR2 -- 3p24.1 AD -- TGFbetaReceptor2-592aa -- "
            "Loeys-Dietz-Syndrome-Type2-LDS2-Aggressive-Aortic -- "
            "Surgery-Threshold-4.0cm-NOT-4.5-5.0cm-Marfan -- "
            "ALL-Aortic-Segments-Imaged-Not-Root-Only -- "
            "Hypertelorism-Bifid-Uvula-Cleft-Palate-LDS-Triad -- "
            "Tortuous-Arteries-Throughout-Body"
        ),
        "locus": "3p24.1",
        "protein_size": "592 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Variable; aortic complications earlier than Marfan; "
            "arterial tortuosity and aneurysms may be congenital; "
            "craniofacial features apparent from birth (hypertelorism, bifid uvula); "
            "mean age of death without treatment historically 26 yr (Loeys 2006); "
            "outcome greatly improved with aggressive surgical management"
        ),
        "key_biomarker": (
            "Echocardiogram: aortic root Z-score (but cannot be only imaging); "
            "CT angiography HEAD-TO-PELVIS: must image entire aorta + branch vessels + cerebral; "
            "craniofacial: hypertelorism (wide-set eyes), bifid uvula/cleft palate, craniosynostosis; "
            "skeletal: pectus deformity, scoliosis, joint laxity (more than Marfan); "
            "skin: translucent, easy bruising (more than Marfan, less than vEDS); "
            "molecular: TGFBR1 or TGFBR2 pathogenic variant; SMAD2/3 also LDS"
        ),
        "pathognomonic": (
            "Hypertelorism + bifid uvula + aortic root dilation = Loeys-Dietz triad (highly specific); "
            "arterial tortuosity throughout body (carotid, vertebral, iliac) = LDS characteristic; "
            "aortic dissection at smaller diameters (3.0-4.0 cm) compared to Marfan (>5 cm); "
            "DISTINGUISH from Marfan: no ectopia lentis in LDS; hypertelorism rare in Marfan; "
            "bifid uvula/cleft palate NOT in Marfan; surgery threshold lower in LDS"
        ),
        "treatment": (
            "Beta-blockers (atenolol): standard cardiovascular protection; "
            "ARBs (losartan): TGFβ pathway inhibition; some evidence of benefit; "
            "Elective aortic root/ascending replacement at 4.0 cm (vs 4.5-5.0 in Marfan) — lower threshold; "
            "Aggressive surgical surveillance: repair ALL significant aneurysms (not just root); "
            "Head-to-pelvis imaging annually with MRI/CT (entire vascular tree); "
            "Cerebral aneurysm surveillance included; "
            "Pregnancy: very high risk — repair root BEFORE pregnancy if possible; "
            "Genetic testing of first-degree relatives mandatory"
        ),
        "critical_flags": [
            "TGFBR2-SURGERY-AT-4.0cm-NOT-4.5cm: LDS surgical threshold is 4.0 cm (some centres 3.8-4.0 cm); NEVER apply Marfan 4.5-5.0 cm threshold to LDS — dissection occurs at smaller diameters",
            "TGFBR2-WHOLE-BODY-IMAGING-MANDATORY: ALL aortic segments + branch vessels + cerebral must be imaged; aortic root repair does NOT eliminate risk — other segments can rupture; head-to-pelvis CT/MRI annually",
            "TGFBR2-BIFID-UVULA-CLUE: bifid uvula in a young patient with aortic dilation = LDS first; absent in Marfan; simple inspection of uvula + hard palate = screening step before echo",
            "TGFBR2-HYPERTELORISM-CRANIOSYNOSTOSIS: hypertelorism (wide-spaced eyes) ± craniosynostosis ± cleft palate = craniofacial LDS triad; examine face before pursuing FBN1 workup in young aortic dilation",
            "TGFBR2-ARTERIAL-TORTUOSITY: tortuous arteries (carotid, vertebral, subclavian, coronary) throughout body = LDS hallmark; absent in Marfan; detectable on CTA or MRA",
            "TGFBR2-SKIN-JOINT-LAXITY: LDS skin is translucent + velvety; joint laxity often more pronounced than Marfan; malar hypoplasia/micrognathia common; club foot; cervical spine instability",
            "TGFBR2-SMAD2-SMAD3-LDS: SMAD2 (LDS5) and SMAD3 (LDS3/Aneurysm-Osteoarthritis syndrome) also cause LDS; combined osteoarthritis + aneurysm = SMAD3 first; same management principles",
        ],
        "alias": (
            "TGFBR2 (TGF-beta Receptor 2) — AD — 3p24.1 — Loeys-Dietz Syndrome type 2 (LDS2) — OMIM #610168 — "
            "592 aa serine/threonine kinase receptor — TGFβ signalling pathway — "
            "TGFBR1 (LDS1), TGFBR2 (LDS2), SMAD3 (LDS3), TGFB2 (LDS4), SMAD2 (LDS5) — "
            "aggressive aortic disease at small diameters — hypertelorism/bifid uvula/aortic triad"
        ),
        "seed": SEED_BASE + 2,
    },

    # -- COL1A1 — Collagen I alpha 1 / Osteogenesis Imperfecta ------------------
    {
        "gene": "COL1A1",
        "alt_name": "Collagen I alpha 1 (OI)",
        "protein": (
            "COL1A1 -- 17q21.33 AD -- CollagenI-alpha1-1464aa -- "
            "Osteogenesis-Imperfecta-Bone-Fragility-Bisphosphonates -- "
            "Blue-Sclerae-Type-I-Hearing-Loss-50pct -- "
            "Dentinogenesis-Imperfecta-Types-III-IV -- "
            "Type-II-Lethal-Perinatally-Ribosomes-Crushed -- "
            "Haploinsufficiency-Mild-vs-Glycine-Substitution-Severe"
        ),
        "locus": "17q21.33",
        "protein_size": "1464 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Type I (mild): fractures with minor trauma from childhood; blue sclerae; hearing loss by age 50; "
            "Type II (lethal): stillborn or death within weeks; multiple fractures in utero; "
            "Type III (progressive deforming): fractures at birth; progressive deformity; wheelchair; "
            "Type IV (moderate): intermediate severity; white sclerae; fractures childhood; "
            "all types: de novo in ~25-40% (especially severe types)"
        ),
        "key_biomarker": (
            "Molecular: COL1A1 or COL1A2 pathogenic variant; haploinsufficiency (splice/frameshift) → OI type I; "
            "glycine substitution (dominant-negative) → type II/III/IV; "
            "X-ray: wormian bones (intersutural bones, >10 = OI specific); "
            "bone mineral density Z-score low; "
            "fracture history (disproportionate to trauma); "
            "blue sclerae (thin scleral collagen → uveal pigment visible) in types I/III; "
            "DXA for bisphosphonate monitoring; urine N-telopeptide cross-links pre/post treatment"
        ),
        "pathognomonic": (
            "Childhood fractures with minor/no trauma + blue sclerae + family history = OI type I; "
            "wormian bones on skull X-ray + OI context = highly specific; "
            "dentinogenesis imperfecta (translucent, grey-brown teeth) in OI types III/IV = OI-specific dental feature; "
            "DISTINGUISH from non-accidental injury (NAI): OI has blue sclerae, wormian bones, low DXA, family history, molecular confirmation; "
            "DISTINGUISH from hypophosphatasia: low ALP + OI-like = ALP enzyme activity + ALPL gene"
        ),
        "treatment": (
            "Bisphosphonates (pamidronate IV 3-yearly cycles OR zoledronic acid IV 6-monthly): "
            "FIRST-LINE; reduce fracture rate 40-50% in types III/IV; less benefit type I; "
            "start from infancy in severe types; monitor bone density + growth; "
            "Romosozumab/teriparatide: emerging; adults with low bone density; "
            "Surgical: rodding of long bones (telescoping Fassier-Duval rods) in type III; "
            "Hearing aids/cochlear implants for conductive/sensorineural hearing loss; "
            "Physiotherapy: weight-bearing as tolerated; aquatic therapy; "
            "Dental: protective measures, composite bonding for dentinogenesis imperfecta; "
            "Avoid contact sports and high-impact activities; "
            "Genetic counselling: 50% recurrence; de novo in severe types"
        ),
        "critical_flags": [
            "COL1A1-HAPLOINSUFFICIENCY-VS-GLYCINE: haploinsufficiency (null allele: splice/frameshift/nonsense) → type I (mild); glycine substitution in triple helix → type II/III/IV (dominant-negative = severe); genotype predicts severity",
            "COL1A1-NAI-CONFUSION: OI fractures can be mistaken for non-accidental injury; ALWAYS check for blue sclerae + wormian bones + DXA + molecular testing before NAI diagnosis in child with unexplained fractures",
            "COL1A1-BISPHOSPHONATE-TIMING: bisphosphonates most effective in childhood (growing bone); start early in severe types; adults benefit less; monitor DXA and urine bone markers; drug holiday after 5 years in mild types",
            "COL1A1-DENTINOGENESIS-IMPERFECTA: grey-brown translucent teeth ± chipping = dentinogenesis imperfecta (DI); present in OI types III/IV (not type I usually); DI is COL1-specific; dental protective care from tooth eruption",
            "COL1A1-WORMIAN-BONES: >10 intersutural bones on skull X-ray = wormian bones = highly specific for OI; also seen in cleidocranial dysplasia and pyknodysostosis; always take skull X-ray in OI workup",
            "COL1A1-HEARING-LOSS-50pct: progressive conductive + sensorineural hearing loss in 50% by age 50 (type I); caused by ossicle fracture + stapes fixation; early audiological monitoring + hearing aid fitting",
            "COL1A1-PREGNANCY: fracture risk during delivery; vaginal delivery generally safe if no severe deformity; epidural caution (vertebral fracture risk); neonatal exam for fractures",
            "COL1A1-TYPE-II-LETHAL-DE-NOVO: type II OI (lethal perinatal) almost always de novo; recurrence risk low for parents but not zero (gonadal mosaicism 6%); offer molecular testing of parents",
        ],
        "alias": (
            "COL1A1 (Collagen I alpha 1) — AD — 17q21.33 — Osteogenesis Imperfecta (OI) — OMIM #166200/#166210 — "
            "1464 aa pro-alpha1(I) collagen chain — fibrillar collagen most abundant in bone/skin/tendon — "
            "haploinsufficiency (type I) vs dominant-negative glycine substitution (types II/III/IV) — "
            "bisphosphonate backbone of treatment"
        ),
        "seed": SEED_BASE + 3,
    },

    # -- ELN — Elastin / SVAS + Cutis Laxa ---------------------------------------
    {
        "gene": "ELN",
        "alt_name": "Elastin (SVAS/Williams)",
        "protein": (
            "ELN -- 7q11.23 AD/deletion -- Elastin-786aa -- "
            "Supravalvular-Aortic-Stenosis-SVAS-Williams-Beuren-Syndrome -- "
            "7q11.23-Deletion-25-Genes-WBS-ELN-Plus -- "
            "Isolated-SVAS-Point-Mutation-Only-ELN -- "
            "Autosomal-Dominant-Cutis-Laxa-ADCL-Skin-Laxity -- "
            "Infantile-Hypercalcemia-Elfin-Facies-Cocktail-Personality"
        ),
        "locus": "7q11.23",
        "protein_size": "786 aa",
        "inheritance": "AD / 7q11.23 deletion (Williams-Beuren)",
        "age_of_onset": (
            "SVAS: murmur detectable from birth; diagnosis in infancy; "
            "Williams-Beuren syndrome: all features present from birth (hypercalcemia neonatal, elfin facies, cardiac); "
            "ADCL (cutis laxa): skin laxity apparent from birth; "
            "Isolated SVAS from point mutations: cardiac features without WBS extracardiac features"
        ),
        "key_biomarker": (
            "SVAS: echocardiogram with Doppler gradient across aortic valve (supravalvular = sinotubular junction); "
            "cardiac catheterisation for peripheral pulmonary stenosis severity; "
            "Williams-Beuren: FISH or microarray for 7q11.23 deletion (≥98% of WBS); "
            "serum calcium (hypercalcemia in infancy — 5-15% WBS); "
            "urinary calcium:creatinine ratio; "
            "neurocognitive assessment (IQ typically 50-70; relative strength = auditory memory, social); "
            "renal ultrasound (nephrocalcinosis 10-20%)"
        ),
        "pathognomonic": (
            "Elfin facies (wide forehead, stellate iris, puffed cheeks, full lips, small chin) + "
            "supravalvular aortic stenosis + intellectual disability + hypercalcemia = Williams-Beuren syndrome; "
            "SVAS alone: hourglass narrowing at aortic root-ascending junction on echo; "
            "DISTINGUISH isolated SVAS (ELN point mutation) from WBS (deletion — multiple features): "
            "isolated SVAS has normal cognition, no elfin facies, no hypercalcemia; "
            "ADCL: skin hangs in folds with normal healing (unlike EDS); prematurely aged appearance"
        ),
        "treatment": (
            "SVAS surgery: patch aortoplasty or Doty procedure when gradient >50 mmHg or symptomatic; "
            "monitor for restenosis (lifelong); peripheral pulmonary stenosis may resolve with growth; "
            "WBS hypercalcemia: low-calcium diet, calcitonin acutely; sun avoidance (vitamin D excess); "
            "Neurocognitive: special education, speech therapy, occupational therapy; "
            "ADCL: NO curative therapy; sunscreen; loose-fitting clothes; plastic surgical intervention cosmetic only; "
            "Cardiology follow-up lifelong; aortic coarctation and coronary stenosis also possible in WBS; "
            "Hypertension surveillance (renovascular); renal ultrasound periodically"
        ),
        "critical_flags": [
            "ELN-WBS-CARDIAC-SUDDEN-DEATH: supravalvular aortic stenosis + bilateral coronary ostial stenosis in WBS → sudden death during general anaesthesia (fixed coronary narrowing + LVH + arrhythmia); ANAESTHESIA HIGH RISK — notify team",
            "ELN-7q11.23-DELETION-NOT-POINT-MUTATION: Williams-Beuren syndrome = contiguous 7q11.23 deletion; isolated SVAS = ELN point mutation; DIFFERENT genetics; FISH/microarray for WBS; sequencing for isolated SVAS; do NOT conflate",
            "ELN-HYPERCALCEMIA-NEONATAL: neonatal hypercalcemia in WBS → irritability, feeding difficulty, constipation; can be severe; low-calcium formula + calcitonin; monitor calcium quarterly in infancy",
            "ELN-PERIPHERAL-PULMONARY-STENOSIS: WBS has peripheral pulmonary arterial stenosis in ~50%; may require catheter-based intervention or surgery; monitor with echo; often improves with age",
            "ELN-COGNITIVE-SOCIAL-PROFILE: WBS IQ 50-70 BUT exceptional social skills ('cocktail party' personality) and auditory memory; relative strength in face recognition; require supervised living as adults; employment possible with support",
            "ELN-STELLATE-IRIS: stellate/lacy iris pattern visible on direct eye examination = WBS feature; not pathognomonic alone but distinctive; also strabismus and refractive error common",
            "ELN-ADCL-VS-EDS: ADCL (elastin cutis laxa) skin hangs in loose folds WITHOUT hyperextensibility; EDS skin is hyperextensible but springs back; cutis laxa skin does NOT spring back; joint laxity absent in ADCL",
        ],
        "alias": (
            "ELN (Elastin) — AD / 7q11.23 deletion — 7q11.23 — Supravalvular Aortic Stenosis / Williams-Beuren Syndrome / ADCL — OMIM #130160/#194050 — "
            "786 aa tropoelastin → cross-linked elastin — elastic fibres (skin/vessels/lung) — "
            "7q11.23 deletion = 25-gene WBS; isolated SVAS = ELN point mutation — "
            "cognitive + cardiac + metabolic triad in WBS"
        ),
        "seed": SEED_BASE + 4,
    },

    # -- ABCC6 — ABC transporter C6 / Pseudoxanthoma Elasticum -------------------
    {
        "gene": "ABCC6",
        "alt_name": "ABC transporter C6 (PXE)",
        "protein": (
            "ABCC6 -- 16p13.1 AR -- ABCtransporterC6-1503aa -- "
            "Pseudoxanthoma-Elasticum-PXE-Elastic-Fibre-Calcification -- "
            "Angioid-Streaks-PATHOGNOMONIC-Fundoscopy -- "
            "Bruch-Membrane-Calcification-Choroidal-Neovascularization-anti-VEGF -- "
            "Premature-Peripheral-Arterial-Disease-GI-Bleeding -- "
            "VitaminK2-MK7-Supplementation-Only-Available-Treatment"
        ),
        "locus": "16p13.1",
        "protein_size": "1503 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Skin findings: papular lesions ('plucked chicken skin') in neck/flexural areas — usually 2nd decade; "
            "Ocular: angioid streaks typically 3rd-4th decade; choroidal neovascularization (CNV) later; "
            "Cardiovascular: peripheral arterial disease typically 4th-6th decade; "
            "GI bleeding: any age; "
            "Subclinical Bruch membrane calcification may precede clinical features by decades"
        ),
        "key_biomarker": (
            "Fundoscopy / OCT: angioid streaks (crack-like extension from optic disc through Bruch membrane) PATHOGNOMONIC; "
            "peau d'orange pattern on fundoscopy (macular drusen-like deposits); "
            "skin biopsy: calcified elastic fibres in mid-dermis (von Kossa stain); "
            "serum PPi (pyrophosphate): reduced in PXE; "
            "echocardiogram: mitral valve prolapse; "
            "ABI (ankle-brachial index): peripheral arterial calcification; "
            "molecular: ABCC6 biallelic pathogenic variants"
        ),
        "pathognomonic": (
            "Angioid streaks on fundoscopy = PATHOGNOMONIC FOR PXE (when combined with skin/family history); "
            "yellowish papules in neck/axilla/groin folds ('plucked chicken skin') + angioid streaks + calcified elastic fibres on skin biopsy = PXE triad; "
            "DISTINGUISH: angioid streaks also in sickle cell, Paget's disease, acromegaly — but ABCC6 + skin + CV calcification = PXE; "
            "AVOID fundal trauma: angioid streaks extend with blunt ocular trauma → acute vision loss"
        ),
        "treatment": (
            "Vitamin K2 (menaquinone-7, MK-7) supplementation: reduces ectopic calcification (ABCC6 normally exports ATP → PP_i → inhibits calcification; LOF → low PP_i → calcification); "
            "Anti-VEGF (ranibizumab/bevacizumab) intravitreal for choroidal neovascularization: FIRST-LINE, same as AMD; "
            "Avoid ocular trauma: NO contact sports, protective eyewear, NO Valsalva manoeuvres; "
            "Cardiovascular: standard risk factor management; aspirin for peripheral arterial disease; "
            "GI bleeding: PPI prophylaxis; iron supplementation; endoscopy surveillance; "
            "Monitor: annual fundoscopy + OCT + ABI + echo; "
            "Carrier (heterozygous) ABCC6: usually asymptomatic but mild features reported"
        ),
        "critical_flags": [
            "ABCC6-ANGIOID-STREAKS-PATHOGNOMONIC: angioid streaks on fundoscopy = Bruch membrane breaks from elastic fibre calcification = highly specific for PXE; ALL PXE patients should have baseline fundoscopy + OCT at diagnosis",
            "ABCC6-AVOID-OCULAR-TRAUMA-ABSOLUTE: blunt ocular trauma → angioid streak extension → acute macular hemorrhage → sudden visual loss; NO boxing/martial arts/ball sports; protective eyewear mandatory; NO Valsalva heavy lifting",
            "ABCC6-VITAMIN-K2-MECHANISM: ABCC6 normally exports ATP → extracellular ATP → PP_i (pyrophosphate, calcification inhibitor) + adenosine; ABCC6 LOF → low serum PP_i → ectopic calcification; vitamin K2 activates matrix Gla protein → inhibits calcification independently",
            "ABCC6-CNV-ANTI-VEGF: choroidal neovascularization (CNV) occurs in ~70% PXE by age 50; managed with intravitreal anti-VEGF (same as AMD); early treatment preserves vision; OCT-A detects CNV before symptoms",
            "ABCC6-GI-BLEEDING-GASTRIC: GI bleeding from submucosal vessel calcification; mimics peptic ulcer; endoscopy shows yellowish submucosal deposits; PPI + iron; major bleeds require intervention but operative risk is elevated",
            "ABCC6-PREMATURE-CAD: PXE patients have premature coronary artery disease + peripheral arterial disease from calcified elastic fibres; ABI annually + lipid management + smoking cessation essential",
            "ABCC6-P.ARG1141X-FOUNDER: p.Arg1141X is most common ABCC6 pathogenic variant (30-50% of alleles in European PXE); compound heterozygosity common; carrier frequency ~1:300 Europeans",
        ],
        "alias": (
            "ABCC6 (ATP-binding cassette transporter C6) — AR — 16p13.1 — Pseudoxanthoma Elasticum (PXE) — OMIM #264800 — "
            "1503 aa ABC-type membrane transporter — hepatic PP_i export for systemic calcification inhibition — "
            "skin / eye / cardiovascular triple-organ calcification — "
            "angioid streaks PATHOGNOMONIC — anti-VEGF + vitamin K2 treatment"
        ),
        "seed": SEED_BASE + 5,
    },

    # -- COL5A1 — Collagen V alpha 1 / Classical EDS ----------------------------
    {
        "gene": "COL5A1",
        "alt_name": "Collagen V alpha 1 (Classical EDS)",
        "protein": (
            "COL5A1 -- 9q34.3 AD -- CollagenV-alpha1-2836aa -- "
            "Classical-EDS-cEDS-Skin-Hyperextensibility-Atrophic-Scarring -- "
            "Gorlin-Sign-Tongue-Nose-Touch-Skin-Hyperextensibility -- "
            "Molluscoid-Pseudotumours-Pressure-Points -- "
            "Joint-Hypermobility-Recurrent-Dislocations -- "
            "50pct-De-Novo-No-Curative-Therapy"
        ),
        "locus": "9q34.3",
        "protein_size": "2836 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "Skin features apparent from birth/infancy (soft, doughy, hyperextensible); "
            "atrophic scars from earliest wounds (disproportionately thin/wide scars); "
            "joint dislocations from infancy/childhood; "
            "molluscoid pseudotumours develop over pressure points in childhood/adolescence; "
            "chronic pain may worsen in adolescence/adulthood"
        ),
        "key_biomarker": (
            "Clinical: skin hyperextensibility ≥1.5 cm forearm/dorsal hand (Beighton >0.9 cm per revised criteria); "
            "Beighton score ≥5/9 for joint hypermobility; "
            "atrophic scarring at ≥2 sites; "
            "molluscoid pseudotumours over pressure points (elbow/knee/heel); "
            "Gorlin sign positive (tongue touches tip of nose); "
            "molecular: COL5A1 or COL5A2 pathogenic variant confirms cEDS (~90% of cEDS); "
            "collagen electrophoresis on skin fibroblasts (historically used; molecular now preferred)"
        ),
        "pathognomonic": (
            "Skin hyperextensibility (>1.5 cm forearm) + atrophic scars at ≥2 sites + molluscoid pseudotumours = cEDS triad (highly specific); "
            "Gorlin sign (tongue to nose) = visible skin hyperextensibility proxy; "
            "DISTINGUISH from hypermobile EDS (hEDS): hEDS has NO atrophic scars, NO skin hyperextensibility criterion met; hEDS = clinical diagnosis (no gene); "
            "DISTINGUISH from vEDS: cEDS skin IS hyperextensible; vEDS skin is thin translucent (NOT hyperextensible); "
            "DISTINGUISH from cutis laxa: EDS springs back; cutis laxa does NOT recoil"
        ),
        "treatment": (
            "NO curative therapy; symptomatic management only; "
            "Joint protection: low-impact exercise (swimming, cycling), proprioceptive physiotherapy, orthotics/bracing for hypermobile joints; "
            "Wound care: paper tape strips for wound closure; slow suture removal (2-3× normal time); layered wound closure technique; "
            "Pain management: regular paracetamol + NSAIDs; neuropathic agents (amitriptyline/gabapentin) for chronic pain; avoid opioids long-term; "
            "Anaesthesia caution: skin may be fragile; IV access difficult; joint dislocation during positioning; "
            "Avoid contact sports, heavy lifting, repetitive joint-loading activities; "
            "Genetic counselling: 50% recurrence; de novo in 50%"
        ),
        "critical_flags": [
            "COL5A1-ATROPHIC-SCARS-DIAGNOSTIC: atrophic ('cigarette paper', thin, wide, stretched) scars at ≥2 sites = REQUIRED for cEDS diagnosis (2017 criteria); absent in hEDS; probe scar history at every wound/surgical site",
            "COL5A1-WOUND-CLOSURE-TECHNIQUE: standard wound closure = WIDE atrophic scars in cEDS; use layered closure + paper tape + prolonged suture (2-3× normal removal time); advise surgical teams proactively",
            "COL5A1-BEIGHTON-SCORE: Beighton score ≥5/9 required for joint hypermobility criterion; wrists/fingers/elbows/knees/lumbar spine (9 joints); score < 5 does NOT exclude cEDS — skin and scar criteria can still be met",
            "COL5A1-MOLLUSCOID-PSEUDOTUMOURS: soft, compressible subcutaneous nodules over bony pressure points (elbows, knees, heels) = molluscoid pseudotumours; PATHOGNOMONIC for cEDS when combined with skin features; NOT seen in hEDS",
            "COL5A1-50pct-DE-NOVO: ~50% of cEDS cases are de novo; no family history does NOT exclude cEDS; molecular testing important for diagnosis AND family risk assessment (each child 50% risk once parent identified)",
            "COL5A1-NO-AORTIC-RISK: cEDS does NOT carry the severe aortic/vascular risk of vEDS or Marfan; no routine aortic imaging required unless other features suggest vascular EDS overlap; distinguish clearly for patient reassurance",
            "COL5A1-CHRONIC-PAIN: chronic musculoskeletal pain develops in ~80% cEDS by adulthood; multidisciplinary pain team involvement; physiotherapy + hydrotherapy + occupational therapy + psychology; avoid opioid escalation",
        ],
        "alias": (
            "COL5A1 (Collagen V alpha 1) — AD — 9q34.3 — Classical EDS (cEDS) — OMIM #130000 — "
            "2836 aa fibrillar collagen chain — type V collagen regulates fibril diameter in skin/tendons — "
            "haploinsufficiency mechanism — 2017 EDS nosology criteria — "
            "skin hyperextensibility + atrophic scars + molluscoid pseudotumours triad"
        ),
        "seed": SEED_BASE + 6,
    },

    # -- TNXB — Tenascin-XB / TNX-EDS -------------------------------------------
    {
        "gene": "TNXB",
        "alt_name": "Tenascin-XB (TNX-EDS)",
        "protein": (
            "TNXB -- 6p21.3 AR/AD(haplo) -- Tenascin-XB-4243aa -- "
            "Tenascin-X-Deficiency-TNX-EDS-Hypermobility -- "
            "Haploinsufficiency-cEDS-Like-Hypermobility-AD -- "
            "Homozygous-LOF-Severe-EDS-PLUS-Adrenal-Insufficiency-CAH-X -- "
            "TNXB-CYP21A2-Contiguous-6p21.3-CAH-X-Deletion -- "
            "Only-EDS-Gene-With-Known-Ligand-Collagen-Fibril-Spacing"
        ),
        "locus": "6p21.3",
        "protein_size": "4243 aa",
        "inheritance": "AR (homozygous/compound het) / AD haploinsufficiency",
        "age_of_onset": (
            "Haploinsufficiency (AD-like): hypermobility + joint pain from childhood; "
            "Homozygous TNX deficiency: severe EDS from birth; "
            "CAH-X (contiguous deletion TNXB + CYP21A2): classical CAH features at birth PLUS EDS; "
            "typically diagnosed in childhood after EDS + CAH co-diagnosis triggers CYP21A2 workup"
        ),
        "key_biomarker": (
            "Serum tenascin-X (plasma TNX level): ZERO in homozygous LOF; 50% in haploinsufficiency; "
            "ELISA for tenascin-X — ONLY EDS subtype with a specific protein biomarker blood test; "
            "molecular: TNXB pathogenic variant (deletion, nonsense, frameshift); "
            "CAH-X screen: 17-OHP (17-hydroxyprogesterone) for congenital adrenal hyperplasia; "
            "ACTH stimulation test if adrenal insufficiency suspected; "
            "CYP21A2 deletion analysis (if TNXB deletion present, check contiguous CYP21A2)"
        ),
        "pathognomonic": (
            "EDS features (hypermobility + easy bruising + soft skin) + ZERO serum tenascin-X = homozygous TNXB LOF; "
            "EDS features + 50% serum tenascin-X + TNXB heterozygous variant = TNXB haploinsufficiency; "
            "EDS + adrenal insufficiency (CAH features) = CAH-X contiguous deletion — TNXB + CYP21A2 deleted together; "
            "DISTINGUISH: TNXB haploinsufficiency resembles hEDS phenotypically; serum tenascin-X is the discriminating test; "
            "TNXB is the ONLY EDS cause with a reliable serum protein assay for diagnosis"
        ),
        "treatment": (
            "NO curative therapy; symptomatic as per hEDS/cEDS management; "
            "Haploinsufficiency: joint protection + physiotherapy + pain management; "
            "Homozygous TNXB LOF: more severe — earlier wheelchair use; pain management; specialist EDS centre; "
            "CAH-X: hydrocortisone/mineralocorticoid replacement for adrenal insufficiency (same as classical CAH); "
            "stress dosing protocol mandatory for illness/surgery in CAH-X; "
            "Monitor: adrenal function annually in CAH-X; "
            "Genetic counselling: AR inheritance for homozygous (25% recurrence); haploinsufficiency de novo or inherited"
        ),
        "critical_flags": [
            "TNXB-SERUM-TNX-BIOMARKER: tenascin-X serum ELISA is the ONLY EDS subtype with a blood test biomarker; ZERO tenascin-X = homozygous TNXB LOF; 50% normal = haploinsufficiency; use this test before labelling as hEDS (no test)",
            "TNXB-CAH-X-ADRENAL-CRISIS: contiguous TNXB + CYP21A2 deletion = CAH-X; adrenal crisis is life-threatening; any EDS patient with features of adrenal insufficiency (hypotension, hyponatremia, hypoglycemia) needs urgent cortisol + ACTH test",
            "TNXB-CYP21A2-PROXIMITY: TNXB and CYP21A2 are adjacent at 6p21.3; large TNXB deletions delete CYP21A2 simultaneously (CAH-X); ALWAYS check CYP21A2 when large TNXB deletion found — do NOT miss treatable adrenal insufficiency",
            "TNXB-COLLAGEN-FIBRIL-SPACING: tenascin-X is the ONLY validated extracellular matrix protein that directly regulates collagen fibril D-spacing; TNX-null fibroblasts have disorganised collagen fibrils — mechanistic basis of EDS phenotype",
            "TNXB-HAPLOINSUFFICIENCY-VS-hEDS: TNXB haploinsufficiency phenotype = clinically indistinguishable from hypermobile EDS (hEDS); serum tenascin-X discriminates; ~5% of clinically-labelled hEDS have detectable TNXB haploinsufficiency on testing",
            "TNXB-PSEUDOGENE-TENXB: TNXA is a pseudogene adjacent to TNXB; TNXA/TNXB recombination creates chimeric TNXA-TNXB alleles; standard sequencing may miss these; long-read sequencing or targeted deletion panels required for comprehensive testing",
            "TNXB-STRESS-DOSING-CAH-X: CAH-X patients on hydrocortisone need sick-day rules and surgical stress dosing (10× maintenance); carry hydrocortisone injection kit; emergency card; register with EDS + CAH specialist teams jointly",
        ],
        "alias": (
            "TNXB (Tenascin-XB) — AR/AD — 6p21.3 — Tenascin-X Deficiency EDS (TNX-EDS) — OMIM #606408 — "
            "4243 aa extracellular matrix glycoprotein — collagen fibril D-spacing regulator — "
            "contiguous CYP21A2 deletion → CAH-X — serum TNX ELISA only EDS blood test — "
            "only EDS gene adjacent to a classical CAH gene"
        ),
        "seed": SEED_BASE + 7,
    },
]


# ---------- Cohort generation --------------------------------------------------

def _make_cohort(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    cohort = []
    for i in range(40):
        severity = rng.choice(["mild", "moderate", "severe"])

        if gene == "FBN1":
            age = rng.randint(6, 55)
            feature = rng.choice([
                "aortic root Z-score ≥3 + ectopia lentis upward",
                "aortic root 4.8 cm — prophylactic repair", "marfanoid habitus + arachnodactyly",
                "spontaneous pneumothorax", "mitral valve prolapse + regurgitation",
                "thumb/wrist sign positive", "scoliosis + pectus excavatum",
            ])
            therapy = rng.choice([
                "atenolol + losartan", "ARB (losartan) + surveillance",
                "aortic root repair (Bentall)", "annual echo surveillance",
            ])
        elif gene == "COL3A1":
            age = rng.randint(15, 60)
            feature = rng.choice([
                "spontaneous celiac artery rupture", "uterine rupture in pregnancy",
                "spontaneous pneumothorax bilateral", "carotid-cavernous fistula",
                "thin translucent skin + easy bruising", "bowel perforation without trauma",
                "splenic artery aneurysm rupture",
            ])
            therapy = rng.choice([
                "celiprolol 400 mg/day", "celiprolol 800 mg/day + conservative management",
                "emergency surgical repair", "palliative/supportive",
            ])
        elif gene == "TGFBR2":
            age = rng.randint(2, 55)
            feature = rng.choice([
                "bifid uvula + aortic root Z-score ≥3", "hypertelorism + arterial tortuosity",
                "aortic root 4.2 cm — early surgery planned", "cervical spine instability",
                "craniosynostosis + aortic dilation", "vertebral artery aneurysm",
                "pectus deformity + scoliosis + joint laxity",
            ])
            therapy = rng.choice([
                "ARB + beta-blocker; elective surgery at 4.0 cm",
                "prophylactic aortic root replacement (4.0 cm)", "annual head-to-pelvis MRA",
            ])
        elif gene == "COL1A1":
            age = rng.randint(0, 55)
            feature = rng.choice([
                "blue sclerae + diaphyseal fracture", "hearing loss + bone fragility OI type I",
                "wormian bones on skull X-ray", "dentinogenesis imperfecta OI type IV",
                "femoral rod + pamidronate therapy", "scoliosis + vertebral compression fractures",
                "non-accidental injury concern (NAI excluded)", "type II lethal — perinatal",
            ])
            therapy = rng.choice([
                "IV pamidronate 3-yearly cycles", "zoledronic acid 6-monthly",
                "Fassier-Duval telescoping rodding + bisphosphonates",
                "hearing aids + bisphosphonates",
            ])
        elif gene == "ELN":
            age = rng.randint(0, 45)
            feature = rng.choice([
                "SVAS gradient 65 mmHg — surgical repair", "7q11.23 deletion (Williams-Beuren syndrome)",
                "elfin facies + hypercalcemia neonatal", "peripheral pulmonary stenosis",
                "intellectual disability + cocktail personality", "isolated SVAS point mutation (no WBS features)",
                "skin laxity + prematurely aged appearance (ADCL)",
            ])
            therapy = rng.choice([
                "surgical aortoplasty (Doty) SVAS",
                "low-calcium formula + calcitonin neonatal",
                "special education + speech therapy (WBS)",
                "surveillance (mild SVAS gradient <40 mmHg)",
            ])
        elif gene == "ABCC6":
            age = rng.randint(10, 65)
            feature = rng.choice([
                "angioid streaks fundoscopy both eyes", "choroidal neovascularization anti-VEGF started",
                "plucked chicken skin neck/axilla", "premature peripheral arterial disease age 45",
                "gastrointestinal bleeding submucosal calcification", "visual loss from macular hemorrhage (trauma)",
                "skin biopsy: calcified elastic fibres (von Kossa +)", "Bruch membrane calcification OCT",
            ])
            therapy = rng.choice([
                "anti-VEGF (ranibizumab) intravitreal + vitamin K2",
                "vitamin K2 (MK-7) 360 mcg/day + annual fundoscopy",
                "anti-VEGF monthly loading then PRN",
                "aspirin + cardiovascular risk management",
            ])
        elif gene == "COL5A1":
            age = rng.randint(2, 55)
            feature = rng.choice([
                "forearm skin stretches >1.5 cm + atrophic scars",
                "molluscoid pseudotumours bilateral elbows",
                "Gorlin sign positive + joint hypermobility Beighton 7/9",
                "recurrent shoulder dislocations + atrophic scars",
                "wound dehiscence + wide atrophic scar (post-surgical)",
                "chronic pain + fatigue (EDS-related)",
                "pelvic floor dysfunction + dysautonomia",
            ])
            therapy = rng.choice([
                "physiotherapy + joint protection programme",
                "paper tape wounds + layered closure + pain management",
                "orthotics + hydrotherapy",
                "multidisciplinary pain clinic",
            ])
        elif gene == "TNXB":
            age = rng.randint(2, 50)
            feature = rng.choice([
                "serum tenascin-X ZERO + severe EDS + adrenal insufficiency (CAH-X)",
                "serum tenascin-X 50% + haploinsufficiency + hypermobility",
                "CAH-X: CYP21A2 + TNXB contiguous deletion",
                "hypermobility + easy bruising + TNX low",
                "recurrent dislocations + serum TNX 50%",
                "adrenal crisis + EDS features (CAH-X)",
                "TNXA/TNXB chimeric allele on long-read sequencing",
            ])
            therapy = rng.choice([
                "hydrocortisone + fludrocortisone (CAH-X) + EDS programme",
                "joint protection + physiotherapy (haploinsufficiency)",
                "stress-dose hydrocortisone sick-day rules + EDS management",
                "pain management + EDS physiotherapy",
            ])
        else:
            feature = "connective tissue disorder"
            therapy = "symptomatic"

        cohort.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "age": age,
            "gene": gene,
            "severity": severity,
            "key_feature": feature,
            "current_therapy": therapy,
        })
    return cohort


# ---------- API endpoint functions -------------------------------------------

def overview() -> dict:
    total = 0
    severe_count = 0
    avg_age_sum = 0
    gene_summary = []
    for idx, g in enumerate(CONNECTIVE_TISSUE_GENES):
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
        "atlas": "Hereditary-Connective-Tissue-Disorder-Atlas",
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(CONNECTIVE_TISSUE_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(CONNECTIVE_TISSUE_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "FBN1-GHENT-2010: aortic root Z-score ≥2 + FBN1 = Marfan (no ectopia lentis needed); ectopia lentis = UPWARD-TEMPORAL (not downward like homocystinuria); losartan + beta-blocker reduce aortic growth; surgery threshold 4.5-5.0 cm",
            "COL3A1-NO-ELECTIVE-SURGERY: vEDS is the only EDS where elective surgery is ABSOLUTELY CONTRAINDICATED; celiprolol is the ONLY Level B drug; arterial rupture occurs WITHOUT warning; most lethal EDS",
            "TGFBR2-SURGERY-4.0cm-NOT-4.5cm: Loeys-Dietz aortic surgery threshold is 4.0 cm (not Marfan's 4.5-5.0); ALL aortic segments must be imaged (not just root); hypertelorism + bifid uvula = LDS triad",
            "COL1A1-HAPLOINSUFFICIENCY-VS-GLYCINE: null allele → OI type I (mild, blue sclerae); glycine substitution → type II/III/IV (dominant-negative, severe); bisphosphonates reduce fracture 40-50%; wormian bones on skull X-ray",
            "ELN-7q11.23-DELETION-WBS-ANAESTHESIA-RISK: WBS = 25-gene deletion; SVAS + bilateral coronary ostial stenosis → anaesthetic sudden death risk; notify anaesthetist; isolated SVAS = ELN point mutation only",
            "ABCC6-ANGIOID-STREAKS-PATHOGNOMONIC: angioid streaks on fundoscopy = PXE PATHOGNOMONIC; avoid ALL ocular trauma (blunt trauma extends streaks → macular hemorrhage); anti-VEGF for CNV; vitamin K2 supplementation",
            "COL5A1-ATROPHIC-SCARS-REQUIRED: atrophic scars at ≥2 sites required for cEDS diagnosis (2017 criteria); skin hyperextensibility + atrophic scars + molluscoid pseudotumours = cEDS triad; no aortic risk (unlike vEDS/Marfan)",
            "TNXB-SERUM-TNX-ONLY-EDS-BIOMARKER: serum tenascin-X = the ONLY specific blood test for any EDS subtype; ZERO = homozygous TNXB LOF; 50% = haploinsufficiency; CAH-X = contiguous CYP21A2 + TNXB deletion → adrenal insufficiency",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(CONNECTIVE_TISSUE_GENES):
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
    return {"genes": result, "total_genes": len(CONNECTIVE_TISSUE_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Connective-Tissue-Disorder-Atlas",
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
            for g in CONNECTIVE_TISSUE_GENES
        ],
        "glossary": {
            "Marfan Syndrome (FBN1)": (
                "FBN1 dominant LOF (haploinsufficiency); fibrillin-1 microfibril defect → excess free TGFβ; "
                "aortic root Z-score ≥2 + ectopia lentis (upward) + skeletal = Revised Ghent 2010; "
                "losartan + beta-blocker + elective root repair 4.5-5.0 cm; "
                "NEVER wait for symptoms; dissection risk peaks 3rd-4th decade without treatment"
            ),
            "Vascular EDS (COL3A1)": (
                "COL3A1 dominant LOF; type III collagen deficiency in vessels/bowel/uterus; "
                "arterial rupture WITHOUT warning = most dangerous EDS; "
                "celiprolol = ONLY Level B drug; NO elective surgery (vessel friability); "
                "thin translucent skin (NOT hyperextensible); median survival 48 yr untreated"
            ),
            "Loeys-Dietz Syndrome (TGFBR1/2)": (
                "TGFBR1/2 dominant LOF → excess free TGFβ (same pathway as Marfan but more aggressive); "
                "surgery threshold 4.0 cm (vs Marfan 4.5 cm); ALL vascular segments imaged; "
                "hypertelorism + bifid uvula + aortic = LDS triad; arterial tortuosity throughout body; "
                "SMAD2/SMAD3/TGFB2 also cause LDS (5 types)"
            ),
            "Osteogenesis Imperfecta (COL1A1/COL1A2)": (
                "COL1A1/A2 dominant; haploinsufficiency = type I (mild, blue sclerae); "
                "glycine substitution = type II/III/IV (dominant-negative = severe); "
                "bisphosphonates reduce fracture 40-50%; wormian bones on skull X-ray; "
                "dentinogenesis imperfecta in types III/IV; hearing loss 50% by age 50"
            ),
            "Williams-Beuren Syndrome / SVAS (ELN)": (
                "7q11.23 deletion (~1.5 Mb, 25 genes) = WBS; isolated point mutation = SVAS only; "
                "ELN haploinsufficiency → SVAS; SVAS + bilateral coronary stenosis → anaesthetic death risk; "
                "elfin facies + hypercalcemia + intellectual disability + cocktail personality in WBS; "
                "ADCL from ELN point mutations: skin laxity, prematurely aged"
            ),
            "Pseudoxanthoma Elasticum (ABCC6)": (
                "ABCC6 biallelic LOF → reduced hepatic PP_i export → systemic elastic fibre calcification; "
                "angioid streaks (Bruch membrane cracks) on fundoscopy PATHOGNOMONIC; "
                "skin: peau d'orange/plucked chicken neck/axilla; premature peripheral arterial disease; "
                "anti-VEGF for CNV; vitamin K2 reduces calcification; avoid all ocular trauma"
            ),
            "Classical EDS (COL5A1/COL5A2)": (
                "COL5A1 haploinsufficiency; type V collagen regulates fibril diameter; "
                "skin hyperextensibility ≥1.5 cm + atrophic scars ≥2 sites + molluscoid pseudotumours = triad; "
                "Gorlin sign (tongue to nose); 50% de novo; NO aortic risk; "
                "wound care critical (layered closure, delayed suture removal)"
            ),
            "Tenascin-X Deficiency EDS (TNXB)": (
                "TNXB biallelic = homozygous severe EDS; haploinsufficiency = hEDS-like phenotype; "
                "serum tenascin-X ELISA = ONLY EDS-specific blood biomarker (ZERO or 50%); "
                "contiguous 6p21.3 deletion TNXB + CYP21A2 = CAH-X (EDS + adrenal insufficiency); "
                "TNXB is the only known extracellular regulator of collagen fibril D-spacing"
            ),
            "Revised Ghent Criteria 2010 (Marfan)": (
                "Aortic root Z-score ≥2 + ectopia lentis = Marfan (no FBN1 needed); "
                "aortic root Z-score ≥2 + FBN1 pathogenic variant = Marfan (no ectopia lentis needed); "
                "ectopia lentis + FBN1 pathogenic variant = Marfan (no aortic Z-score needed); "
                "1996 criteria were overly inclusive (MASS syndrome misdiagnosed as Marfan)"
            ),
            "Angioid Streaks (PXE)": (
                "Crack-like breaks radiating from optic disc through Bruch membrane; "
                "caused by elastic fibre calcification → Bruch membrane brittleness; "
                "appear dark red/grey on fundoscopy; OCT shows subretinal hyporeflective lines; "
                "blunt ocular trauma → acute extension → macular hemorrhage → sudden visual loss; "
                "also seen in sickle cell, Paget's, acromegaly — but ABCC6 + skin = PXE-specific"
            ),
            "CAH-X (TNXB + CYP21A2 contiguous deletion)": (
                "TNXB and CYP21A2 are adjacent at 6p21.3; large deletions remove both genes; "
                "classical CAH phenotype (21-hydroxylase deficiency) + EDS phenotype = CAH-X; "
                "salt-wasting or simple-virilizing CAH features + hypermobility/EDS = test CYP21A2 + TNXB together; "
                "adrenal replacement (hydrocortisone + fludrocortisone) + EDS management; stress dosing mandatory"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-CONNECTIVE-TISSUE-DISORDER-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (COL3A1 — vEDS no surgery) ===")
    bd = breakdown()
    col3 = next(g for g in bd["genes"] if g["gene"] == "COL3A1")
    print(json.dumps(col3, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary: Angioid Streaks) ===")
    df = definitions()
    print(json.dumps({"angioid_streaks": df["glossary"]["Angioid Streaks (PXE)"]}, indent=2))
