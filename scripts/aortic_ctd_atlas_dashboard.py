#!/usr/bin/env python3
"""Aortic-CTD-Atlas — Complete 8-Gene Hereditary Aortic & Connective Tissue Disorders Atlas
FBN1   (Fibrillin-1; ~2871 aa; 15q21.1; Marfan Syndrome MFS1; AD; ~35% heritable TAAD; aortic root; β-blocker + losartan; repair ≥4.5 cm) ·
TGFBR2 (TGF-β Receptor II; ~567 aa; 3p24.1; Loeys-Dietz Syndrome 2 LDS2; AD; AGGRESSIVE — repair ≥4.0 cm; bifid uvula; arterial tortuosity) ·
TGFBR1 (TGF-β Receptor I; ~503 aa; 9q22.33; Loeys-Dietz Syndrome 1 LDS1; AD; hypertelorism; cleft palate; aggressive surgical threshold) ·
COL3A1 (Collagen III α1; ~1466 aa; 2q32.2; Vascular EDS vEDS; AD; spontaneous arterial rupture; NO elective vascular surgery; celiprolol) ·
ACTA2  (Smooth Muscle α-Actin; ~377 aa; 10q23.31; FTAAD + cerebrovascular; AD; Moya-Moya; straight aorta; ~15% HTAD) ·
SMAD3  (SMAD Family Member 3; ~425 aa; 15q22.33; LDS3/AOS; AD; osteoarthritis + aneurysm; repair ≥4.0 cm; TGF-β signalling) ·
MYH11  (Smooth Muscle Myosin Heavy Chain; ~1972 aa; 16p13.11; FTAAD + PDA; AD; patent ductus arteriosus; aortic media hypocellularity) ·
SKI    (SKI Proto-Oncogene; ~728 aa; 1p36.33; Shprintzen-Goldberg Syndrome SGS; AD; de novo; craniosynostosis + Marfanoid + ID)
320-patient aggregate cohort (8 × 40, seeds 1110–1117)
"""

import random

SEED_BASE = 1110

AORTIC_CTD_GENES = [
    # ── FBN1 — Marfan Syndrome MFS1 ──────────────────────────────────────────
    {
        "gene": "FBN1",
        "protein": "Fibrillin-1 (FBN1)",
        "alias": "FBN1; OMIM gene 134797; 15q21.1; ~2871 aa; Marfan Syndrome MFS1 (OMIM #154700) + MFS2 alleles; AD; ~35% heritable TAAD; extracellular matrix glycoprotein",
        "aa": "~2871 aa",
        "kDa": "~350 kDa",
        "mechanism": (
            "FBN1 encodes fibrillin-1, the major structural component of extracellular matrix "
            "microfibrils that form a scaffold for elastin deposition in the aortic wall, zonular "
            "fibres of the lens, and periosteum. "
            "NORMAL FUNCTION: fibrillin-1 microfibrils provide tensile strength and elastic recoil "
            "to large elastic arteries; also sequesters latent TGF-β in the matrix via LTBP binding. "
            "PATHOMECHANISM: pathogenic FBN1 variants (predominantly missense in cbEGF domains) → "
            "haploinsufficiency or dominant-negative effect → structurally weakened microfibrils → "
            "reduced matrix sequestration of TGF-β → increased free TGF-β signalling (SMAD2/3 pathway) → "
            "smooth muscle cell apoptosis + elastin fragmentation + progressive aortic root dilatation. "
            "SECONDARY TGF-β UPREGULATION explains why losartan (AT1R blockade, reduces TGF-β signalling) "
            "is beneficial; this was the key mechanistic insight from mouse fibrillin-1 knockout models. "
            "LENS ECTOPIA: zonular fibre insufficiency → superior-temporal lens dislocation (opposite "
            "to homocystinuria which causes inferior-nasal dislocation). "
            "SKELETAL: microfibrils constrain longitudinal bone growth → excess growth in long bones, "
            "vertebrae, ribs when fibrillin-1 is deficient → dolichostenomelia, arachnodactyly, "
            "pectus deformity, scoliosis."
        ),
        "disease_type": (
            "Marfan Syndrome MFS1 (OMIM #154700); AD; ~35% of heritable TAAD; "
            "extracellular matrix; systemic connective tissue disorder; aortic root aneurysm; "
            "β-blocker + losartan; prophylactic aortic root repair ≥4.5 cm (or 4.0–4.2 cm "
            "if rapid growth >3 mm/year, family history dissection, planning pregnancy)"
        ),
        "locus": "15q21.1",
        "omim_gene": 134797,
        "omim_disease": 154700,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants. "
            "PENETRANCE: ~100% but variable expressivity — same variant within a family can produce "
            "mild vs. severe phenotype; modifier genes and sex contribute. "
            "DE NOVO RATE: ~25% of MFS1 cases arise de novo (no family history) — cannot use family "
            "history to exclude diagnosis. "
            "FAMILY SCREENING: echocardiogram + slit-lamp eye exam + skeletal assessment + "
            "FBN1 genetic testing for all first-degree relatives. "
            "GHENT NOSOLOGY 2010: diagnostic criteria — major criterion in ≥2 different organ systems "
            "OR systemic score ≥7 + aortic root Z-score ≥2 + ectopia lentis; FBN1 pathogenic variant "
            "with aortic root dilation OR ectopia lentis = confirmed MFS regardless of systemic score. "
            "VARIANT SPECTRUM: missense ~65%; premature stop/frameshift ~25%; splice ~10%; "
            "most severe phenotype (neonatal MFS — fatal infantile form) from variants in exons 24–32 "
            "(cbEGF-like domain region) — these disrupt Ca2+-binding sites critical for microfibril assembly."
        ),
        "phenotype": (
            "CARDIOVASCULAR: aortic root aneurysm at sinuses of Valsalva (aortic root Z-score ≥2); "
            "progressive dilatation → risk of aortic dissection Type A (most lethal, surgical emergency) "
            "or Type B; mitral valve prolapse ~50% (myxomatous); tricuspid valve prolapse; "
            "aortic regurgitation (from root dilation); ectasia of the dural sac (lumbosacral). "
            "OCULAR: ectopia lentis (lens dislocation) ~60% — superior-temporal direction "
            "(PATHOGNOMONIC; distinguishes from homocystinuria = inferior-nasal); myopia (often severe); "
            "retinal detachment risk (large axial globe length); early glaucoma. "
            "SKELETAL: dolichostenomelia (disproportionately long limbs); arachnodactyly; "
            "positive thumb sign (Steinberg — thumb tip protrudes beyond ulnar border of fist) + "
            "wrist sign (Walker-Murdoch — thumb and 5th finger overlap when encircling wrist); "
            "pectus carinatum (pigeon chest) or pectus excavatum; scoliosis; pes planus; "
            "reduced upper-to-lower body segment ratio; arm span-to-height ratio >1.05. "
            "DURAL ECTASIA: lumbosacral dural sac widening → low back pain, radiculopathy, headache "
            "(diagnosed by MRI lumbosacral spine — often missed clinically)."
        ),
        "treatment_options": [
            "β-Blocker (atenolol 25-100 mg OD or propranolol) — MANDATORY first-line: reduces aortic "
            "root growth rate (dP/dt reduction); titrate to resting HR ≤60 bpm; "
            "start from diagnosis; do not stop abruptly (rebound tachycardia)",
            "Losartan (25-100 mg OD) or Irbesartan — angiotensin II receptor blocker reduces TGF-β "
            "signalling via AT1R blockade; additive benefit to β-blocker in paediatric/young adults "
            "(PEDIATRIC HEART NETWORK trial — modest aortic root growth reduction); "
            "first-choice when β-blocker not tolerated or in paediatrics",
            "Prophylactic aortic root replacement (Bentall procedure or valve-sparing root "
            "replacement — David/Yacoub): ELECTIVE surgery threshold ≥4.5 cm aortic root diameter "
            "(lower threshold 4.0–4.2 cm if: rapid growth >3 mm/year; family history aortic "
            "dissection at smaller diameter; planning pregnancy; severe AR); valve-sparing preferred "
            "to avoid lifelong anticoagulation (tissue valve = anticoagulant-free; mechanical = warfarin)",
            "AVOID: intense static exercise (weight-lifting, isometric); contact sports; "
            "competitive team sports with collision risk; high-impact activities (trampoline); "
            "scuba diving; drugs that increase HR/BP (stimulants, decongestants, caffeine excess); "
            "fluoroquinolone antibiotics (damage fibrillin-1 microfibrils — AVOID if possible); "
            "AVOID: elective cosmetic surgery unless essential (anaesthesia + BP fluctuation risk)",
            "Annual echocardiography (aortic root + ascending aorta Z-score); "
            "Cardiac MRI/CT whole aorta every 2-3 years (entire aortic tree including arch + descending); "
            "Ophthalmology slit-lamp annual (ectopia lentis progression, retinal detachment); "
            "Orthopaedic review for scoliosis (Cobb angle monitoring); dural ectasia MRI if symptoms",
            "Pregnancy: HIGH RISK — aortic root ≥4.0 cm = contraindication to pregnancy without "
            "prior root repair; if pregnant: β-blocker throughout (atenolol avoid in 1st trimester → "
            "switch to metoprolol); monthly echo 3rd trimester; elective caesarean section "
            "recommended if aortic root >4.0 cm or rapid growth during pregnancy",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "FLUOROQUINOLONES — AVOID in Marfan",
                "body": (
                    "Fluoroquinolone antibiotics (ciprofloxacin, levofloxacin, moxifloxacin) "
                    "chelate Mg²⁺ required for fibrillin-1 microfibril integrity and inhibit "
                    "matrix metalloproteinases → accelerated aortic root expansion reported "
                    "in case series. Use alternative antibiotics if possible. FDA black box "
                    "warning for aortic aneurysm risk applies to all fluoroquinolones."
                ),
            },
            {
                "type": "danger",
                "title": "AORTIC ROOT ≥4.5 cm — ELECTIVE SURGERY THRESHOLD",
                "body": (
                    "Prophylactic root repair indicated ≥4.5 cm. "
                    "LOWER threshold (4.0–4.2 cm) if: rapid growth >3 mm/year; "
                    "family history of dissection at smaller size; planning pregnancy; "
                    "severe aortic regurgitation. Do NOT wait for symptoms — dissection "
                    "is often the presenting event."
                ),
            },
        ],
        "clinical_rules": [
            "LENS DISLOCATION DIRECTION: superior-temporal = MFS/FBN1 (OPPOSITE to homocystinuria = inferior-nasal); never conflate",
            "DE NOVO ~25%: negative family history does NOT exclude Marfan — always assess index patient by Ghent 2010 criteria",
            "FLUOROQUINOLONE AVOIDANCE: document allergy-equivalent in chart; use alternative antibiotics",
            "PREGNANCY THRESHOLD: aortic root ≥4.0 cm = relative contraindication; ≥4.5 cm = strong contraindication without prior repair",
            "DURAL ECTASIA: screen with lumbosacral MRI in all FBN1 patients with low back pain/radiculopathy — often missed",
        ],
        "key_distinguishing": "SUPERIOR-TEMPORAL lens dislocation + MARFANOID habitus + AORTIC ROOT dilation = FBN1/MFS1 (not TGFBR1/2 which rarely have lens ectopia)",
        "severity_weights": {"Mild": 0.25, "Moderate": 0.40, "Severe": 0.35},
        "prevalence_per_100k": 2.0,
        "surgery_rate_pct": 55,
        "dissection_lifetime_risk_pct": 30,
        "lens_ectopia_pct": 60,
        "skeletal_pct": 85,
        "de_novo_pct": 25,
    },

    # ── TGFBR2 — Loeys-Dietz Syndrome 2 ─────────────────────────────────────
    {
        "gene": "TGFBR2",
        "protein": "TGF-β Receptor Type II (TGFBR2)",
        "alias": "TGFBR2; OMIM gene 190182; 3p24.1; ~567 aa; Loeys-Dietz Syndrome 2 LDS2 (OMIM #610168); AD; AGGRESSIVE aortic phenotype",
        "aa": "~567 aa",
        "kDa": "~70 kDa",
        "mechanism": (
            "TGFBR2 encodes the type II serine/threonine kinase receptor of the TGF-β signalling "
            "pathway. TGF-β binds TGFBR2 constitutive dimer → recruits TGFBR1 → "
            "TGFBR2 transphosphorylates TGFBR1 kinase domain → TGFBR1 phosphorylates "
            "cytoplasmic SMAD2/3 → SMAD2/3 + SMAD4 complex translocates to nucleus → "
            "transcription of extracellular matrix genes (collagen, fibronectin), cell cycle "
            "arrest, and apoptosis genes. "
            "PARADOX: haploinsufficient TGFBR2 variants INCREASE rather than decrease TGF-β "
            "pathway signalling in the aortic wall — compensatory upregulation through alternative "
            "signalling routes (ERK/MAPK non-canonical pathway) → paradoxical TGF-β upregulation "
            "in aortic tissue — this is the 'TGF-β paradox' of LDS. "
            "CONSEQUENCE: paradoxical SMAD2/3 activation → excessive MMP production → elastin "
            "fragmentation → structurally weakened aortic media → aggressive dilatation and "
            "tendency to dissect at SMALLER diameters than Marfan Syndrome."
        ),
        "disease_type": (
            "Loeys-Dietz Syndrome 2 (LDS2; OMIM #610168); AD; AGGRESSIVE aortic phenotype — "
            "aortic dissection reported at root diameter <4.0 cm; surgical repair threshold ≥4.0 cm "
            "(LOWER than MFS ≥4.5 cm); widespread arterial tortuosity; bifid uvula; hypertelorism; "
            "TGF-β pathway upregulation paradox"
        ),
        "locus": "3p24.1",
        "omim_gene": 190182,
        "omim_disease": 610168,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous variants (missense in kinase domain or haploinsufficiency). "
            "HIGH PENETRANCE: ~100% penetrance for arterial disease if genotype confirmed. "
            "DE NOVO FRACTION: ~50-75% of LDS2 cases are de novo — family history frequently absent. "
            "FAMILY SCREENING: complete vascular imaging (whole-body MRA or CT angiogram) + "
            "FBN1/TGFBR1/2/SMAD3/SKI/COL3A1/ACTA2/MYH11 gene panel; "
            "must image entire arterial tree (not just aorta — renal, mesenteric, intracranial vessels also dilate). "
            "FOLLOW-UP: MRA whole aorta + branch vessels every 6-12 months given aggressive progression."
        ),
        "phenotype": (
            "CARDIOVASCULAR: aortic root aneurysm (sinuses of Valsalva); WIDESPREAD ARTERIAL TORTUOSITY "
            "— hallmark of LDS; dissection reported at smaller aortic diameters vs MFS — this is the "
            "KEY CLINICAL DANGER; mitral valve prolapse; bicuspid aortic valve; PDA (patent ductus "
            "arteriosus) in ~25%; aneurysms involving celiac, renal, pulmonary, intracranial arteries. "
            "CRANIOFACIAL: BIFID UVULA (split uvula) — PATHOGNOMONIC marker for LDS (not seen in MFS); "
            "hypertelorism (widely spaced eyes); cleft palate; craniosynostosis in some. "
            "SKELETAL: joint hypermobility; club foot (talipes equinovarus); cervical spine instability "
            "— atlantoaxial instability; scoliosis; pectus deformity. "
            "CUTANEOUS: velvety/translucent skin; easy bruising; atrophic scarring (less severe than vEDS). "
            "ABSENCE: ectopia lentis (lens dislocation NOT a feature — distinguishes from Marfan). "
            "PROGNOSIS: median age of first arterial event 26 years without surgical intervention; "
            "WITHOUT imaging surveillance most patients die before 50."
        ),
        "treatment_options": [
            "Aggressive surgical threshold ≥4.0 cm aortic root (LOWER than MFS 4.5 cm): "
            "prophylactic Bentall or valve-sparing root repair; "
            "many centres repair at 4.0 cm or even 3.5 cm if rapid growth; "
            "do not use MFS thresholds for LDS — aortic dissection well-documented at <4.0 cm",
            "β-Blocker (atenolol/metoprolol) + ARB (losartan) dual therapy — same rationale as MFS; "
            "addresses paradoxical TGF-β upregulation; titrate HR <60 bpm resting",
            "Whole-body MRA (magnetic resonance angiography) annual surveillance — entire arterial tree "
            "(aorta, arch, descending, abdominal, renal, mesenteric, iliac, intracranial); "
            "dissection can occur anywhere not just aortic root; CT with contrast if MRI unavailable",
            "Pregnancy: VERY HIGH RISK — aortic repair recommended before planned pregnancy; "
            "aortic root ≥4.0 cm = contraindication; monthly imaging in 3rd trimester; "
            "caesarean section preferred",
            "Cervical spine MRI: screen for atlantoaxial instability before any general anaesthesia "
            "(may require spinal precautions during intubation)",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "DO NOT USE MFS SURGICAL THRESHOLDS for TGFBR2/LDS2",
                "body": (
                    "Aortic dissection documented at root <4.0 cm in LDS2. "
                    "Surgical repair threshold is 4.0 cm (NOT 4.5 cm used for MFS/FBN1). "
                    "Applying Marfan thresholds to LDS patients has caused preventable deaths — "
                    "this is the single most dangerous clinical error in aortopathy management."
                ),
            },
            {
                "type": "warning",
                "title": "BIFID UVULA — Screen for LDS in ALL patients with aortic root dilation",
                "body": (
                    "Bifid uvula is PATHOGNOMONIC for LDS — examine uvula in every patient "
                    "with aortic root aneurysm. LDS patients are frequently misdiagnosed as "
                    "Marfan syndrome; the distinction is critical because surgical thresholds differ."
                ),
            },
        ],
        "clinical_rules": [
            "SURGICAL THRESHOLD 4.0 cm (not 4.5 cm): applying Marfan thresholds to LDS2 is a sentinel clinical error",
            "BIFID UVULA PATHOGNOMONIC: examine uvula in every aortic aneurysm patient — if bifid → full LDS workup + gene panel",
            "NO LENS ECTOPIA: absence of lens dislocation distinguishes LDS from MFS/FBN1",
            "WHOLE-BODY MRA MANDATORY: LDS arteries dilate throughout; celiac/renal/intracranial aneurysms must be screened",
            "DE NOVO ~50-75%: negative family history common — do not exclude on history alone",
        ],
        "key_distinguishing": "BIFID UVULA + HYPERTELORISM + aggressive dissection at SMALL diameters + arterial TORTUOSITY = LDS2/TGFBR2",
        "severity_weights": {"Mild": 0.15, "Moderate": 0.35, "Severe": 0.50},
        "prevalence_per_100k": 0.5,
        "surgery_rate_pct": 70,
        "dissection_lifetime_risk_pct": 55,
        "lens_ectopia_pct": 2,
        "skeletal_pct": 75,
        "de_novo_pct": 60,
    },

    # ── TGFBR1 — Loeys-Dietz Syndrome 1 ─────────────────────────────────────
    {
        "gene": "TGFBR1",
        "protein": "TGF-β Receptor Type I (ALK5, TGFBR1)",
        "alias": "TGFBR1; OMIM gene 190181; 9q22.33; ~503 aa; Loeys-Dietz Syndrome 1 LDS1 (OMIM #609192); AD; more prominent craniofacial features vs LDS2",
        "aa": "~503 aa",
        "kDa": "~55 kDa",
        "mechanism": (
            "TGFBR1 (also called ALK5, Activin receptor-Like Kinase 5) encodes the type I TGF-β "
            "receptor, a serine/threonine kinase. It forms a heterotetrameric complex with TGFBR2 "
            "upon TGF-β ligand binding. TGFBR2 transphosphorylates the GS (glycine-serine rich) "
            "domain of TGFBR1 → activates TGFBR1 kinase → phosphorylates SMAD2/3 → "
            "SMAD2/3-SMAD4 complex enters nucleus → TGF-β target gene transcription. "
            "PATHOMECHANISM: haploinsufficiency or dominant-negative TGFBR1 variants → paradoxical "
            "TGF-β pathway upregulation in the aortic wall via ERK/MAPK non-canonical signalling "
            "(identical paradox to TGFBR2); nuclear pSmad2 elevated in aortic tissue despite "
            "receptor haploinsufficiency; MMP upregulation → elastin fragmentation → progressive "
            "aneurysm and dissection tendency. "
            "MOLECULAR DIFFERENCE FROM TGFBR2: LDS1 (TGFBR1) may have slightly more prominent "
            "craniofacial features (craniosynostosis more common) vs LDS2; arterial tortuosity "
            "is similarly widespread in both."
        ),
        "disease_type": (
            "Loeys-Dietz Syndrome 1 (LDS1; OMIM #609192); AD; aggressive aortic phenotype; "
            "craniosynostosis more prominent than LDS2; bifid uvula; hypertelorism; "
            "surgical threshold ≥4.0 cm same as LDS2; widespread arterial tortuosity"
        ),
        "locus": "9q22.33",
        "omim_gene": 190181,
        "omim_disease": 609192,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous variants. "
            "HIGH PENETRANCE: ~100% for arterial disease in gene carriers. "
            "DE NOVO FRACTION: ~50% of cases. "
            "FAMILY SCREENING: identical to LDS2 — complete vascular imaging + genetic cascade; "
            "first-degree relatives of all LDS1 index patients require full workup. "
            "GENETIC TESTING: LDS gene panel (TGFBR1, TGFBR2, SMAD3, SMAD2, TGFB2, SKI, ELN4) "
            "preferred over sequential single-gene testing given clinical overlap."
        ),
        "phenotype": (
            "CARDIOVASCULAR: aortic root aneurysm; arterial tortuosity (HALLMARK — tortuous, "
            "elongated arteries throughout body); aortic dissection at smaller diameters than MFS; "
            "aneurysms in branch arteries (renal, celiac, pulmonary, intracranial); PDA. "
            "CRANIOFACIAL: CRANIOSYNOSTOSIS (premature fusion of skull sutures) more prominent "
            "than in LDS2 — DISTINGUISHING feature vs TGFBR2; hypertelorism; BIFID UVULA "
            "(PATHOGNOMONIC as in LDS2); cleft palate; broad nasal root. "
            "SKELETAL: arachnodactyly; joint hypermobility; cervical spine instability (C1-C2); "
            "scoliosis; club foot; pectus deformity. "
            "CUTANEOUS: velvety/translucent skin; easy bruising. "
            "PROGNOSIS: similar aggressive course to LDS2; median aortic event mid-20s to 30s "
            "without prophylactic repair."
        ),
        "treatment_options": [
            "Prophylactic aortic root repair ≥4.0 cm (same threshold as LDS2 — NOT 4.5 cm MFS threshold)",
            "β-Blocker + ARB combination (losartan) as for LDS2",
            "Whole-body MRA annually — entire vascular tree",
            "Neurosurgical/craniofacial referral early: craniosynostosis may require cranial vault remodelling surgery in infancy",
            "Cervical spine stability assessment (CT/MRI C-spine) before any general anaesthesia",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "CRANIOSYNOSTOSIS — Neurosurgical Emergency in Neonates",
                "body": (
                    "LDS1 (TGFBR1) has higher craniosynostosis rate than LDS2. "
                    "Early referral to craniofacial neurosurgery is mandatory — "
                    "untreated craniosynostosis → raised intracranial pressure → "
                    "visual loss and cognitive impairment. Surgical repair in first year of life."
                ),
            },
            {
                "type": "warning",
                "title": "CERVICAL INSTABILITY before Intubation",
                "body": (
                    "C1-C2 atlantoaxial instability in LDS1/2 — assess cervical MRI "
                    "before elective general anaesthesia. Communicate to anaesthetic team; "
                    "may require awake fibreoptic intubation or spinal precautions."
                ),
            },
        ],
        "clinical_rules": [
            "LDS1 vs LDS2: craniosynostosis MORE prominent in LDS1/TGFBR1; otherwise very similar phenotype",
            "SURGICAL THRESHOLD 4.0 cm: identical to LDS2 — do not use MFS 4.5 cm threshold",
            "CRANIOSYNOSTOSIS URGENT: refer to craniofacial neurosurgery in first weeks of life",
            "BIFID UVULA also PATHOGNOMONIC in LDS1 — same as LDS2",
            "CERVICAL MRI pre-anaesthesia: mandatory for C-spine instability assessment",
        ],
        "key_distinguishing": "CRANIOSYNOSTOSIS (more prominent than LDS2) + BIFID UVULA + ARTERIAL TORTUOSITY = LDS1/TGFBR1",
        "severity_weights": {"Mild": 0.15, "Moderate": 0.40, "Severe": 0.45},
        "prevalence_per_100k": 0.4,
        "surgery_rate_pct": 65,
        "dissection_lifetime_risk_pct": 50,
        "lens_ectopia_pct": 2,
        "skeletal_pct": 78,
        "de_novo_pct": 55,
    },

    # ── COL3A1 — Vascular EDS ────────────────────────────────────────────────
    {
        "gene": "COL3A1",
        "protein": "Collagen Type III Alpha-1 Chain (COL3A1)",
        "alias": "COL3A1; OMIM gene 120180; 2q32.2; ~1466 aa; Vascular Ehlers-Danlos Syndrome vEDS (OMIM #130050); AD; MOST LETHAL connective tissue disorder; spontaneous arterial rupture",
        "aa": "~1466 aa",
        "kDa": "~138 kDa",
        "mechanism": (
            "COL3A1 encodes the alpha-1 chain of type III collagen, which forms homotrimeric "
            "type III collagen (three identical alpha-1 chains in a characteristic triple helix). "
            "Type III collagen is the predominant collagen of compliant hollow organs and "
            "large elastic arteries: aorta, bowel wall, uterus, skin. "
            "NORMAL FUNCTION: type III collagen forms a fibrillar meshwork that provides tensile "
            "strength while permitting distension; it is co-assembled with type I collagen fibres "
            "in most tissues. "
            "PATHOMECHANISM: most pathogenic COL3A1 variants are glycine substitutions in the "
            "Gly-X-Y repeat motif of the triple helix → structurally abnormal pro-alpha1(III) chains "
            "→ dominant-negative disruption of triple helix assembly (2-3 of every 4 heterotrimers "
            "contain at least one mutant chain) → secreted or intracellularly degraded → "
            "insufficient functional type III collagen → structurally weakened arterial media + "
            "bowel wall + uterine wall → spontaneous rupture WITHOUT preceding aneurysm in many cases. "
            "HAPLOINSUFFICIENCY VARIANTS: frameshift/PTC variants → haploinsufficiency → less severe "
            "phenotype (MILDER vEDS) — these patients retain 50% of normal collagen III and have "
            "better prognosis than glycine-substitution dominant-negative variants. "
            "KEY: unlike MFS/LDS, there is often NO ANEURYSM before arterial rupture — the artery "
            "RUPTURES without preceding dilation → this is why imaging surveillance is less effective "
            "and surgery is dangerous."
        ),
        "disease_type": (
            "Vascular Ehlers-Danlos Syndrome vEDS (OMIM #130050); AD; MOST LETHAL CTD; "
            "spontaneous arterial rupture without preceding aneurysm; "
            "bowel perforation; uterine rupture in pregnancy; "
            "NO elective vascular surgery (surgery often MORE LETHAL than the lesion); "
            "celiprolol reduces event rate (only RCT evidence)"
        ),
        "locus": "2q32.2",
        "omim_gene": 120180,
        "omim_disease": 130050,
        "inheritance": (
            "AUTOSOMAL DOMINANT: predominantly heterozygous. "
            "DOMINANT-NEGATIVE glycine substitutions: most severe — encode structurally abnormal "
            "collagen that poisons triple helix assembly. "
            "HAPLOINSUFFICIENCY variants (PTC, splice): encode for less severe phenotype — "
            "these should be classified separately as they have significantly different prognosis. "
            "PENETRANCE: high; 80% have had a major complication (arterial rupture/surgery/bowel "
            "perforation) by age 40. "
            "DE NOVO FRACTION: ~50% of vEDS. "
            "FAMILY SCREENING: genetic testing + CT angiogram full aorta + branch vessels; "
            "consider annual whole-body MRA in affected gene carriers but note aneurysm often absent "
            "before rupture — limits effectiveness of surveillance. "
            "GENOTYPE-PHENOTYPE: haploinsufficiency variants → milder vEDS; "
            "glycine substitutions → severe classical vEDS with highest rupture risk."
        ),
        "phenotype": (
            "ARTERIAL: spontaneous rupture of medium and large arteries (celiac, mesenteric, renal, "
            "splenic, iliac, carotid — NOT predominantly aortic root like MFS/LDS) — often WITHOUT "
            "preceding aneurysm; presents as acute abdomen, haemodynamic collapse, or sudden death. "
            "BOWEL: spontaneous perforation of sigmoid colon — SECOND MOST COMMON life-threatening "
            "complication (10-15% of vEDS patients); presents as acute abdomen/peritonitis. "
            "UTERINE: uterine rupture in pregnancy (CATASTROPHIC — high maternal mortality); "
            "pregnancy is HIGH RISK in vEDS; discuss before conception. "
            "CUTANEOUS: translucent skin (subcutaneous vessels visible), easy bruising — "
            "BUT these features are LESS PROMINENT than classic EDS; skin hyperextensibility ABSENT "
            "(a key distinguishing feature — classic EDS has skin hyperextensibility; vEDS does NOT). "
            "FACIAL: characteristic vEDS facies — thin vermilion border of lips, small lobeless ears, "
            "narrow nose, prominent eyes — present in many but not all. "
            "PROGNOSIS: 25% have first vascular complication by age 20; 80% by age 40; "
            "median survival 48-51 years (classic series)."
        ),
        "treatment_options": [
            "Celiprolol (β1-selective blocker with β2-agonism): ONLY intervention with RCT evidence "
            "(Ong et al. 2010, Lancet) — reduced arterial event rate by 36% vs placebo; "
            "start 200 mg BD titrating to 400 mg BD; mechanism: reduces wall shear stress + "
            "vasodilatory β2 effect reduces arterial mechanical stress; "
            "DOES NOT cure vEDS but meaningfully delays arterial events",
            "AVOID elective vascular surgery: arterial fragility makes operative mortality of "
            "ELECTIVE procedures as high or HIGHER than the natural history of the lesion; "
            "endovascular approaches (EVAR/TEVAR) preferred over open surgery when intervention "
            "is unavoidable for haemodynamically significant lesions; "
            "emergency surgery (ruptured artery) is unavoidable but has very high mortality",
            "Conservative management of arterial lesions: bed rest, IV analgesia, close monitoring "
            "for haemodynamically stable patients with non-ruptured arterial injuries; "
            "many vEDS arterial dissections/haematomas resolve without intervention",
            "Pregnancy counselling: HIGH RISK — uterine rupture risk; discuss before conception; "
            "if pregnant: obstetric/maternal-fetal medicine + vascular surgery team co-management; "
            "early hospitalisation near term; caesarean preferred but carries its own risks; "
            "pregnancy avoidance may be appropriate for some patients (individual decision)",
            "Annual CT/MRA surveillance of abdomen and pelvis (where spontaneous ruptures occur most): "
            "note that normal imaging does NOT guarantee safety as rupture often occurs without "
            "preceding aneurysm; patient education on emergency presentation (emergency department "
            "immediately for acute abdominal pain, chest pain, or collapse)",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "NO ELECTIVE VASCULAR SURGERY — COL3A1/vEDS",
                "body": (
                    "Elective vascular surgery in vEDS carries operative mortality "
                    "often EXCEEDING natural history risk. Arterial walls do not hold sutures "
                    "normally; anastomotic rupture and catastrophic bleeding are frequent. "
                    "Endovascular approaches preferred when intervention unavoidable. "
                    "This is the most dangerous clinical error in vEDS management."
                ),
            },
            {
                "type": "danger",
                "title": "PREGNANCY — UTERINE RUPTURE RISK",
                "body": (
                    "Uterine rupture carries very high maternal mortality in COL3A1 vEDS. "
                    "Counsel ALL women with vEDS before pregnancy. "
                    "If pregnant: specialist obstetric + vascular surgery co-management mandatory "
                    "from conception. Do not manage as low-risk obstetric patient."
                ),
            },
            {
                "type": "warning",
                "title": "CELIPROLOL — Only RCT-proven therapy",
                "body": (
                    "Celiprolol 200-400 mg BD is the only therapy with randomised evidence "
                    "for reducing arterial events in vEDS. Start early after diagnosis. "
                    "Standard β-blockers (atenolol, metoprolol) do NOT have this RCT evidence "
                    "in vEDS specifically."
                ),
            },
        ],
        "clinical_rules": [
            "NO ELECTIVE SURGERY: document in capital letters in chart; alert surgical teams; this is a lifesaving clinical rule",
            "RUPTURE WITHOUT ANEURYSM: normal vascular imaging does NOT exclude imminent rupture in vEDS — patient education critical",
            "CELIPROLOL NOT STANDARD β-BLOCKER: specifically celiprolol 200-400 mg BD (RCT evidence); atenolol/metoprolol not equivalent for vEDS",
            "HAPLOINSUFFICIENCY BETTER PROGNOSIS: PTC/splice variants have milder vEDS than glycine-substitution dominant-negative variants — counsel accordingly",
            "PREGNANCY DISCUSSION MANDATORY: counsel all women of reproductive age at diagnosis; do not wait for planned pregnancy",
        ],
        "key_distinguishing": "SPONTANEOUS ARTERIAL RUPTURE without preceding aneurysm + NO skin hyperextensibility (unlike classic EDS) + BOWEL perforation risk = COL3A1/vEDS",
        "severity_weights": {"Mild": 0.20, "Moderate": 0.30, "Severe": 0.50},
        "prevalence_per_100k": 0.8,
        "surgery_rate_pct": 35,
        "dissection_lifetime_risk_pct": 70,
        "lens_ectopia_pct": 0,
        "skeletal_pct": 30,
        "de_novo_pct": 50,
    },

    # ── ACTA2 — FTAAD + Cerebrovascular ─────────────────────────────────────
    {
        "gene": "ACTA2",
        "protein": "Smooth Muscle Alpha-Actin (ACTA2, α-SMA)",
        "alias": "ACTA2; OMIM gene 102620; 10q23.31; ~377 aa; Familial TAAD (OMIM #611788); AD; ~15% heritable TAAD; cerebrovascular disease; Moya-Moya pattern",
        "aa": "~377 aa",
        "kDa": "~42 kDa",
        "mechanism": (
            "ACTA2 encodes smooth muscle cell alpha-actin (α-SMA), the predominant actin isoform "
            "in vascular smooth muscle cells (VSMCs). α-SMA assembles into contractile thin "
            "filaments that interact with smooth muscle myosin (MYH11) to generate the contractile "
            "force required for arterial vasoconstriction, blood pressure regulation, and aortic "
            "mechanical stabilisation. "
            "NORMAL FUNCTION: ACTA2 forms the thin filament backbone of VSMC contractile apparatus; "
            "α-SMA expression is a marker of differentiated (contractile-phenotype) VSMCs; "
            "dedifferentiation → loss of α-SMA expression → synthetic phenotype → matrix remodelling. "
            "PATHOMECHANISM: pathogenic ACTA2 missense variants (most common: p.Arg179His, p.Arg258Cys, "
            "p.Arg149Cys) → abnormal α-SMA → dysfunctional contractile thin filaments → VSMCs adopt "
            "synthetic/migratory phenotype → deficient aortic media → progressive thoracic aortic "
            "aneurysm and dissection. "
            "CEREBROVASCULAR: p.Arg179His specifically → hyperplastic smooth muscle occlusion of "
            "small cerebral arteries → MOYA-MOYA disease pattern (progressive stenosis of "
            "intracranial ICA → stroke in childhood or young adults); "
            "also bilateral fusiform intracranial aneurysms (rare); straight (non-tortuous) aorta "
            "on imaging is a CHARACTERISTIC FEATURE (opposite of LDS tortuosity)."
        ),
        "disease_type": (
            "Familial Thoracic Aortic Aneurysm and Dissection FTAAD (OMIM #611788); AD; "
            "~15% of heritable TAAD families; cerebrovascular disease (Moya-Moya, stroke); "
            "STRAIGHT aorta on imaging; livedo reticularis; iris flocculi; "
            "p.Arg179His = most severe — Moya-Moya + PDA + iris flocculi"
        ),
        "locus": "10q23.31",
        "omim_gene": 102620,
        "omim_disease": 611788,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants; predominantly missense. "
            "PENETRANCE: high (~90%) for aortic disease; cerebrovascular complications depend "
            "strongly on specific variant. "
            "KEY VARIANT: p.Arg179His — most severe: Moya-Moya cerebrovascular disease, "
            "patent ductus arteriosus, iris flocculi (diagnostic marker); high stroke risk; "
            "other ACTA2 variants: lower/absent cerebrovascular risk. "
            "DE NOVO FRACTION: ~25% of ACTA2 cases. "
            "FAMILY SCREENING: echocardiogram + CT angiogram thorax + brain MRI/MRA "
            "(cerebrovascular assessment mandatory in all ACTA2 carriers); "
            "additional specific surveillance for Arg179His: paediatric neurology + ophthalmology "
            "(iris flocculi exam under slit lamp)."
        ),
        "phenotype": (
            "AORTIC: progressive thoracic aortic aneurysm (ascending aorta predominant); "
            "aortic dissection Type A or B; mean age at first dissection 43 years; "
            "STRAIGHT AORTA characteristic on CT/MRI (unlike LDS which has arterial tortuosity). "
            "CEREBROVASCULAR (especially p.Arg179His): bilateral Moya-Moya pattern — progressive "
            "stenosis of supraclinoid internal carotid arteries → basal ganglia collateral network "
            "(puff of smoke angiographic appearance); stroke in childhood or young adulthood; "
            "TIA/recurrent stroke; rare bilateral fusiform intracranial aneurysms. "
            "IRIS FLOCCULI (p.Arg179His): small whitish ciliary body nodules visible on "
            "slit-lamp exam — PATHOGNOMONIC for p.Arg179His ACTA2 variant; "
            "clinically silent but diagnostically definitive. "
            "PDA (p.Arg179His): patent ductus arteriosus — frequently identified in neonatal period. "
            "LIVEDO RETICULARIS: mottled skin vasculature pattern — seen in 25% of ACTA2 patients; "
            "not seen in MFS/LDS. "
            "CORONARY ARTERY DISEASE: early onset CAD reported in some pedigrees (VSMC dysfunction)."
        ),
        "treatment_options": [
            "Prophylactic aortic repair ≥4.5 cm (similar threshold to MFS); "
            "some centres use lower threshold for Arg179His given cerebrovascular risk; "
            "ascending aorta involved more than root in some ACTA2 families",
            "β-Blocker + ARB as for other HTADs",
            "Cerebrovascular surveillance: baseline brain MRI/MRA + annual follow-up; "
            "neurology referral for Moya-Moya if p.Arg179His; "
            "antithrombotic therapy for Moya-Moya (aspirin); revascularisation surgery "
            "(STA-MCA bypass) for symptomatic Moya-Moya",
            "Ophthalmology slit-lamp: iris flocculi examination for p.Arg179His carriers; "
            "diagnostically useful when genetic result uncertain",
            "Avoid high-dose stimulants, decongestants, caffeine excess; no competitive contact sports",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "p.Arg179His — CEREBROVASCULAR (Moya-Moya) + STROKE RISK",
                "body": (
                    "ACTA2 p.Arg179His is associated with Moya-Moya cerebrovascular disease "
                    "causing stroke in children and young adults. "
                    "Brain MRI/MRA MANDATORY at diagnosis and annually. "
                    "Neurology referral mandatory. Iris flocculi on slit-lamp = "
                    "pathognomonic for this specific variant."
                ),
            },
            {
                "type": "warning",
                "title": "STRAIGHT AORTA — Not All HTAAD Look Tortuous",
                "body": (
                    "ACTA2-FTAAD shows a STRAIGHT (non-tortuous) aorta — "
                    "opposite of LDS arterial tortuosity. "
                    "Do not be falsely reassured by absence of LDS-like tortuosity "
                    "when evaluating ACTA2 patients."
                ),
            },
        ],
        "clinical_rules": [
            "p.Arg179His SPECIFIC RISKS: Moya-Moya + PDA + iris flocculi — these do NOT apply to all ACTA2 variants",
            "BRAIN MRI/MRA MANDATORY in ALL ACTA2 carriers (not just Arg179His) given variable cerebrovascular risk",
            "IRIS FLOCCULI on slit-lamp = PATHOGNOMONIC for p.Arg179His ACTA2 — request specifically when suspected",
            "STRAIGHT AORTA characteristic: LDS shows tortuosity; ACTA2 shows straight aorta — distinguishing imaging feature",
            "LIVEDO RETICULARIS: present in ~25% of ACTA2 — not seen in MFS/LDS; useful clinical clue",
        ],
        "key_distinguishing": "STRAIGHT AORTA + MOYA-MOYA cerebrovascular + IRIS FLOCCULI (Arg179His) + LIVEDO RETICULARIS = ACTA2-FTAAD",
        "severity_weights": {"Mild": 0.20, "Moderate": 0.40, "Severe": 0.40},
        "prevalence_per_100k": 0.6,
        "surgery_rate_pct": 50,
        "dissection_lifetime_risk_pct": 45,
        "lens_ectopia_pct": 0,
        "skeletal_pct": 15,
        "de_novo_pct": 25,
    },

    # ── SMAD3 — Loeys-Dietz Syndrome 3 / AOS ────────────────────────────────
    {
        "gene": "SMAD3",
        "protein": "SMAD Family Member 3 (SMAD3)",
        "alias": "SMAD3; OMIM gene 603109; 15q22.33; ~425 aa; Loeys-Dietz Syndrome 3 / Aneurysm-Osteoarthritis Syndrome LDS3/AOS (OMIM #613795); AD; EARLY OSTEOARTHRITIS + aortic aneurysm",
        "aa": "~425 aa",
        "kDa": "~48 kDa",
        "mechanism": (
            "SMAD3 encodes a receptor-regulated SMAD (R-SMAD) transcription factor that mediates "
            "canonical TGF-β signalling downstream of TGFBR1/TGFBR2. "
            "SIGNALLING PATHWAY: TGF-β → TGFBR2:TGFBR1 heterocomplex → "
            "TGFBR1 phosphorylates SMAD3 at its C-terminal SSXS motif → "
            "phosphorylated SMAD3 forms heterotrimers with SMAD4 → "
            "nuclear translocation → TGF-β target gene activation "
            "(ECM genes: collagen, fibronectin; cell cycle arrest genes: p21; "
            "cartilage genes: SOX9, aggrecan, collagen II; bone remodelling). "
            "PATHOMECHANISM: haploinsufficiency or dominant-negative SMAD3 variants → "
            "deficient canonical TGF-β signalling → paradoxical upregulation of non-canonical "
            "TGF-β (ERK/JNK) + compensatory upregulation through alternative SMAD proteins → "
            "dysregulated ECM in aortic wall + articular cartilage → "
            "aortic aneurysm (via MMP upregulation + elastin degradation) + "
            "EARLY OSTEOARTHRITIS (via deficient cartilage matrix maintenance). "
            "UNIQUE DUAL PHENOTYPE: SMAD3 pathogenic variants uniquely combine VASCULAR disease "
            "(aortic aneurysm/dissection) with MUSCULOSKELETAL disease (early osteoarthritis) "
            "— this combination is the KEY distinguishing feature of LDS3/AOS."
        ),
        "disease_type": (
            "Loeys-Dietz Syndrome 3 / Aneurysm-Osteoarthritis Syndrome LDS3/AOS (OMIM #613795); "
            "AD; DUAL PHENOTYPE: early-onset osteoarthritis (joints) + aortic aneurysm (vascular); "
            "surgical threshold ≥4.0 cm; TGF-β pathway; often MISDIAGNOSED as isolated OA"
        ),
        "locus": "15q22.33",
        "omim_gene": 603109,
        "omim_disease": 613795,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants. "
            "HIGH PENETRANCE for both OA and aortic disease: >90%. "
            "DE NOVO FRACTION: ~30% of LDS3/AOS cases. "
            "FAMILY SCREENING: echocardiogram + full joint assessment + CT angiogram; "
            "young adults with early-onset OA (especially hips, knees, elbows) should be "
            "screened for aortic aneurysm and SMAD3 tested. "
            "IMPORTANT: LDS3 patients are frequently misdiagnosed by rheumatologists as "
            "primary osteoarthritis and NEVER have aortic imaging — leading to preventable "
            "deaths from unsuspected aortic dissection. "
            "SCREENING RULE: any patient <40 years with OA (especially multiple joints) "
            "should be screened for heritable aortopathy."
        ),
        "phenotype": (
            "MUSCULOSKELETAL (DISTINGUISHING): EARLY-ONSET OSTEOARTHRITIS — present in nearly all "
            "affected individuals; joint involvement often begins in 20s-30s; large joints "
            "(hips, knees) + small joints (hands, elbows) — widespread, progressive; "
            "often leads to joint replacement in 30s-40s. "
            "CARDIOVASCULAR: aortic root aneurysm ± ascending aorta; dissection at smaller "
            "diameters than MFS (similar to LDS1/LDS2); arterial aneurysms in other locations "
            "(celiac, renal, intracranial); mitral valve prolapse. "
            "CRANIOFACIAL: bifid uvula (present in some but less consistently than LDS1/2); "
            "hypertelorism; mild craniofacial features. "
            "SKELETAL: joint hypermobility (contrasting with stiffness of OA — dual picture); "
            "scoliosis; cervical spine instability. "
            "INTEGUMENT: mild skin features (velvety texture, bruising — less prominent than vEDS)."
        ),
        "treatment_options": [
            "Prophylactic aortic repair ≥4.0 cm (LDS threshold — NOT MFS 4.5 cm threshold): "
            "aggressive surgical approach warranted given dissection risk at small sizes",
            "β-Blocker + ARB (losartan) for aortic root growth rate reduction",
            "Whole-body MRA annually (branch vessel aneurysms in addition to aortic root)",
            "Rheumatology co-management for OA: physiotherapy; joint-preserving surgery early; "
            "total joint replacement when OA advanced; AVOID NSAIDs chronically if possible "
            "(cardiovascular risk considerations)",
            "Genetic counselling: emphasise that isolated early-onset OA without aortic imaging "
            "is a sentinel missed diagnosis risk",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "EARLY OA + AORTIC DISEASE — DO NOT MANAGE AS ISOLATED OA",
                "body": (
                    "SMAD3/LDS3 patients frequently reach rheumatologists for OA without "
                    "aortic imaging. Aortic dissection is the leading cause of death. "
                    "ALL young patients (<40y) with OA should be screened for aortic aneurysm. "
                    "This missed diagnosis is a common preventable fatality in vascular genetics."
                ),
            },
            {
                "type": "warning",
                "title": "JOINT HYPERMOBILITY + OA COEXIST — Both Features Present",
                "body": (
                    "LDS3/SMAD3 is unusual in having both joint hypermobility (from connective "
                    "tissue laxity) AND osteoarthritis (from deficient cartilage maintenance). "
                    "The combination of hypermobility + early OA in a young person should "
                    "immediately prompt heritable aortopathy genetic panel testing."
                ),
            },
        ],
        "clinical_rules": [
            "EARLY OA <40y = mandatory aortic screening: SMAD3/LDS3 most important missed diagnosis in rheumatology",
            "SURGICAL THRESHOLD 4.0 cm: same as LDS1/2 — aggressive; do not use MFS 4.5 cm threshold",
            "BIFID UVULA less consistent than LDS1/2: absence of bifid uvula does NOT exclude LDS3",
            "DUAL PHENOTYPE KEY: any patient with OA + family history of aortic aneurysm → urgent genetic panel",
            "JOINT REPLACEMENT early but coordinate with vascular team for perioperative aortic surveillance",
        ],
        "key_distinguishing": "EARLY-ONSET OSTEOARTHRITIS + AORTIC ANEURYSM in same patient = LDS3/SMAD3 (AOS — Aneurysm-Osteoarthritis Syndrome)",
        "severity_weights": {"Mild": 0.20, "Moderate": 0.45, "Severe": 0.35},
        "prevalence_per_100k": 0.3,
        "surgery_rate_pct": 60,
        "dissection_lifetime_risk_pct": 45,
        "lens_ectopia_pct": 1,
        "skeletal_pct": 95,
        "de_novo_pct": 30,
    },

    # ── MYH11 — FTAAD + PDA ─────────────────────────────────────────────────
    {
        "gene": "MYH11",
        "protein": "Smooth Muscle Myosin Heavy Chain (MYH11, SMMHC)",
        "alias": "MYH11; OMIM gene 160745; 16p13.11; ~1972 aa; Familial TAAD + PDA (OMIM #132900); AD; aortic media hypocellularity; smooth muscle myosin",
        "aa": "~1972 aa",
        "kDa": "~227 kDa",
        "mechanism": (
            "MYH11 encodes the smooth muscle myosin heavy chain (SMMHC), the motor protein of "
            "vascular smooth muscle cells. Together with ACTA2 (α-SMA thin filament), MYH11 "
            "forms the contractile machinery of VSMCs: "
            "ACTA2 (thin filament) + MYH11 (thick filament) → cross-bridge cycling → force generation. "
            "NORMAL FUNCTION: MYH11 is required for arterial smooth muscle contraction, "
            "blood pressure regulation, and structural integrity of the aortic media "
            "(SMCs maintain lamellar organisation of the aortic wall). "
            "PATHOMECHANISM: haploinsufficiency or dominant-negative MYH11 variants → "
            "dysfunctional contractile apparatus → VSMC dedifferentiation → "
            "reduced VSMC density in the aortic media (aortic media HYPOCELLULARITY — "
            "DISTINCTIVE histological finding in MYH11-TAAD) → "
            "structurally weakened aortic wall → progressive thoracic aortic aneurysm and dissection. "
            "PDA MECHANISM: patent ductus arteriosus requires smooth muscle-mediated vasoconstriction "
            "for postnatal closure; MYH11 LOF → failure of ductal smooth muscle contraction → "
            "PDA persists after birth — PDA + TAAD in same family is a PATHOGNOMONIC combination "
            "for MYH11 mutation."
        ),
        "disease_type": (
            "Familial TAAD + Patent Ductus Arteriosus (OMIM #132900); AD; "
            "PDA + thoracic aortic aneurysm/dissection = PATHOGNOMONIC combination for MYH11; "
            "aortic media hypocellularity on histology; VSMC contractile apparatus dysfunction; "
            "~2-5% of heritable TAAD"
        ),
        "locus": "16p13.11",
        "omim_gene": 160745,
        "omim_disease": 132900,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants (missense or PTC). "
            "PENETRANCE: high (~85-90%) for aortic disease by age 50-60; "
            "PDA penetrance variable — ~50% of MYH11 families have PDA history. "
            "DE NOVO FRACTION: ~20-30% of MYH11-TAAD. "
            "FAMILY SCREENING: echocardiogram + CT angiogram + cardiac history review for PDA; "
            "neonatal cardiac screening (echo) in at-risk newborns for PDA identification. "
            "KEY DIAGNOSTIC CLUE: family history combining aortic aneurysm/dissection with "
            "history of PDA (in the index patient or family members) should prompt MYH11 testing."
        ),
        "phenotype": (
            "AORTIC: progressive thoracic aortic aneurysm predominantly involving ascending aorta; "
            "aortic dissection Type A or B; mean age at dissection ~40-50 years; "
            "risk similar to ACTA2/MYH11 rather than the more aggressive LDS1/2. "
            "PDA (PATENT DUCTUS ARTERIOSUS): failure of postnatal ductal closure → left-to-right "
            "shunt → pulmonary over-circulation; clinically silent small PDA or symptomatic large "
            "PDA (machine murmur, bounding pulses, pulmonary hypertension if unrepaired). "
            "HISTOLOGY: aortic media hypocellularity (reduced SMC density) — DISTINCTIVE "
            "histological finding on surgical specimen that suggests MYH11 diagnosis even "
            "before genetic confirmation. "
            "AORTIC ARCH: arch aneurysms reported in some MYH11 families (unlike MFS where root "
            "predominantly involved). "
            "CORONARY: early coronary artery disease in some families."
        ),
        "treatment_options": [
            "Prophylactic aortic repair ≥4.5 cm (similar threshold to MFS/FBN1; "
            "some centres use 4.0-4.5 cm given the young age of dissection events); "
            "include arch imaging given arch involvement in some families",
            "β-Blocker + ARB standard combination",
            "PDA repair: if haemodynamically significant PDA — device closure (transcatheter) "
            "preferred over surgical ligation; small haemodynamically silent PDA may be observed "
            "but cardiologist follow-up required",
            "Annual echocardiogram + CT/MRI aorta; whole aortic imaging important given "
            "arch and descending involvement",
            "Histology review: if patient undergoes aortic surgery — send specimen for histology; "
            "aortic media hypocellularity should prompt MYH11 genetic testing if not yet done",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "PDA + AORTIC ANEURYSM COMBINATION — PATHOGNOMONIC for MYH11",
                "body": (
                    "Any family history combining PDA with thoracic aortic aneurysm/dissection "
                    "is PATHOGNOMONIC for MYH11 mutation. Send MYH11 genetic testing. "
                    "Many of these families are misclassified as isolated TAAD without PDA "
                    "history being specifically elicited — always ask about PDA in all HTAAD families."
                ),
            },
        ],
        "clinical_rules": [
            "PDA + TAAD FAMILY HISTORY = MYH11 until proven otherwise — specifically ask about PDA in every HTAAD pedigree",
            "AORTIC MEDIA HYPOCELLULARITY on surgical specimen → MYH11 testing if not already done",
            "ARCH IMAGING: include full aortic arch in surveillance (not just aortic root as in MFS)",
            "PDA REPAIR: transcatheter closure preferred over surgical ligation in MYH11 patients due to arterial fragility",
            "NEONATAL SCREEN: at-risk MYH11 carrier newborns → neonatal echo for PDA in first week of life",
        ],
        "key_distinguishing": "PDA + THORACIC AORTIC ANEURYSM in same family = MYH11 (aortic media hypocellularity on histology)",
        "severity_weights": {"Mild": 0.25, "Moderate": 0.45, "Severe": 0.30},
        "prevalence_per_100k": 0.3,
        "surgery_rate_pct": 45,
        "dissection_lifetime_risk_pct": 40,
        "lens_ectopia_pct": 0,
        "skeletal_pct": 10,
        "de_novo_pct": 25,
    },

    # ── SKI — Shprintzen-Goldberg Syndrome ───────────────────────────────────
    {
        "gene": "SKI",
        "protein": "SKI Proto-Oncogene (SKI)",
        "alias": "SKI; OMIM gene 164780; 1p36.33; ~728 aa; Shprintzen-Goldberg Syndrome SGS (OMIM #182212); AD; de novo; craniosynostosis + Marfanoid + intellectual disability",
        "aa": "~728 aa",
        "kDa": "~84 kDa",
        "mechanism": (
            "SKI encodes the SKI proto-oncogene, a transcriptional co-repressor of the TGF-β "
            "canonical signalling pathway. "
            "NORMAL FUNCTION: SKI binds SMAD2/3 and SMAD4 in the nucleus, recruiting histone "
            "deacetylase (HDAC) complexes → represses TGF-β target gene transcription; "
            "SKI acts as a BRAKE on TGF-β signalling, preventing excessive pathway activation; "
            "it is also involved in other transcriptional programmes (Wnt, Hedgehog). "
            "PATHOMECHANISM: haploinsufficiency or dominant-negative SKI variants → "
            "loss of SMAD2/3-SMAD4 transcriptional repression → paradoxical upregulation of "
            "TGF-β target genes in aortic wall + cranial sutures + other tissues → "
            "aortic aneurysm (same TGF-β upregulation mechanism as LDS1/2 but from DOWNSTREAM "
            "pathway repressor loss rather than upstream receptor LOF) + "
            "craniosynostosis (premature suture fusion from dysregulated cranial SMAD signalling) + "
            "intellectual disability (CNS TGF-β pathway dysregulation). "
            "SKI mutations are predominantly de novo — gain-of-function with respect to TGF-β "
            "pathway activation despite being loss-of-function for the SKI repressor itself."
        ),
        "disease_type": (
            "Shprintzen-Goldberg Syndrome SGS (OMIM #182212); AD; predominantly DE NOVO; "
            "craniosynostosis + Marfanoid habitus + intellectual disability + aortic aneurysm; "
            "TGF-β pathway (downstream repressor loss); "
            "most patients require multidisciplinary team from birth"
        ),
        "locus": "1p36.33",
        "omim_gene": 164780,
        "omim_disease": 182212,
        "inheritance": (
            "AUTOSOMAL DOMINANT: predominantly DE NOVO heterozygous variants "
            "(virtually all reported cases are de novo — familial recurrence extremely rare). "
            "PENETRANCE: 100% — all pathogenic SKI variants cause SGS. "
            "RECURRENCE RISK FOR PARENTS: ~1% (gonadal mosaicism) — siblings of de novo cases "
            "have low but non-negligible recurrence risk. "
            "FAMILY SCREENING: genetic testing of parents required to confirm de novo status; "
            "if confirmed de novo → parents' recurrence risk ~1%; "
            "affected individual → 50% risk to offspring (but severe phenotype — most do not "
            "reproduce). "
            "VARIANT SPECTRUM: clustering in exon 1 (N-terminal SMAD-binding domain); "
            "most are missense variants affecting the SMAD2/3 interaction interface — "
            "these disrupt SKI's ability to repress SMAD3-driven transcription."
        ),
        "phenotype": (
            "CRANIOFACIAL: CRANIOSYNOSTOSIS — premature fusion of cranial sutures (coronal, "
            "sagittal, or multiple sutures); results in abnormal head shape; raised intracranial "
            "pressure; orbital abnormalities; downslanted palpebral fissures; "
            "HYPERTELORISM; high-arched palate; prominent eyes. "
            "MARFANOID HABITUS: tall stature; ARACHNODACTYLY (long spider fingers); "
            "pectus excavatum; SCOLIOSIS; hypermobile joints; pes planus; "
            "RESEMBLES MARFAN SYNDROME on external examination (hence the 'Marfanoid' descriptor). "
            "INTELLECTUAL DISABILITY: cognitive impairment variable (mild-moderate); "
            "DISTINGUISHES SGS from Marfan Syndrome (MFS has normal intelligence). "
            "AORTIC: aortic root dilation (sinuses of Valsalva) — less aggressive progression than "
            "LDS1/2 in most cases but still requires surveillance and prophylactic surgery when "
            "threshold reached; mitral valve prolapse. "
            "ABSENT: ectopia lentis (differentiates from MFS/FBN1); absence of severe arterial "
            "tortuosity (distinguishes from LDS1/2). "
            "BRAIN ABNORMALITIES: Chiari malformation in some; enlarged cisterna magna; "
            "various structural brain anomalies on MRI."
        ),
        "treatment_options": [
            "MULTIDISCIPLINARY TEAM from birth: craniofacial neurosurgery + cardiology + "
            "ophthalmology + paediatric genetics + developmental paediatrics + orthopaedics",
            "Craniosynostosis surgery: early cranial vault remodelling in infancy (typically <1 year "
            "of age) to relieve raised ICP and allow brain growth; timing depends on sutures involved",
            "Cardiac surveillance: echocardiogram every 6-12 months; CT/MRA when echo suboptimal; "
            "prophylactic aortic root repair when ≥4.0-4.5 cm (use LDS-like threshold given "
            "TGF-β pathway involvement)",
            "β-Blocker + ARB (evidence extrapolated from LDS, not specific SGS trials)",
            "Scoliosis monitoring and bracing/surgery as needed; ophthalmic surveillance; "
            "developmental support, special education; speech and language therapy",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "INTELLECTUAL DISABILITY — Distinguishes SGS from Marfan",
                "body": (
                    "SKI/SGS patients have intellectual disability (absent in MFS/FBN1). "
                    "A Marfanoid patient with cognitive impairment MUST trigger SKI testing. "
                    "Misdiagnosis as MFS leads to delayed craniofacial care and underappreciation "
                    "of neurodevelopmental support needs."
                ),
            },
            {
                "type": "warning",
                "title": "CRANIOSYNOSTOSIS SURGICAL URGENCY in Neonates/Infants",
                "body": (
                    "Untreated craniosynostosis → rising ICP → visual loss, cognitive worsening. "
                    "Refer to craniofacial neurosurgery within first weeks of life. "
                    "Do NOT delay for genetic confirmation — treat craniofacial urgently, "
                    "then confirm genetics."
                ),
            },
        ],
        "clinical_rules": [
            "MARFANOID + INTELLECTUAL DISABILITY = SKI/SGS not MFS/FBN1 — cognitive impairment is the KEY differentiating sign",
            "CRANIOSYNOSTOSIS URGENT: refer to craniofacial surgery in first weeks; do not wait for genetic results",
            "DE NOVO: counsel parents on ~1% sibling recurrence (gonadal mosaicism); low but document",
            "AORTIC THRESHOLD 4.0-4.5 cm: use LDS-like thresholds given TGF-β pathway involvement",
            "BRAIN MRI: structural anomalies common — baseline MRI in all SGS patients; screen for Chiari",
        ],
        "key_distinguishing": "MARFANOID HABITUS + CRANIOSYNOSTOSIS + INTELLECTUAL DISABILITY (absent in MFS) + DE NOVO = SKI/Shprintzen-Goldberg Syndrome",
        "severity_weights": {"Mild": 0.15, "Moderate": 0.45, "Severe": 0.40},
        "prevalence_per_100k": 0.1,
        "surgery_rate_pct": 60,
        "dissection_lifetime_risk_pct": 25,
        "lens_ectopia_pct": 0,
        "skeletal_pct": 90,
        "de_novo_pct": 95,
    },
]


# ── Patient Simulation ───────────────────────────────────────────────────────

def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    patients = []
    weights = gene_data["severity_weights"]
    sev_choices = list(weights.keys())
    sev_probs = list(weights.values())

    for i in range(40):
        sev = rng.choices(sev_choices, sev_probs)[0]
        onset = rng.randint(5, 50) if sev == "Severe" else rng.randint(10, 65)
        dx_delay = rng.randint(1, 12)
        surgery = rng.random() < gene_data["surgery_rate_pct"] / 100
        dissection = rng.random() < gene_data["dissection_lifetime_risk_pct"] / 100
        lens = rng.random() < gene_data["lens_ectopia_pct"] / 100
        skeletal = rng.random() < gene_data["skeletal_pct"] / 100
        de_novo = rng.random() < gene_data["de_novo_pct"] / 100
        drug_error = rng.random() < 0.12  # 12% drug/surgical threshold error rate across cohort

        patients.append({
            "gene": gene_data["gene"],
            "severity": sev,
            "onset_age_y": onset,
            "diagnosis_age_y": onset + dx_delay,
            "aortic_surgery": surgery,
            "aortic_dissection": dissection,
            "lens_ectopia": lens,
            "skeletal_features": skeletal,
            "de_novo": de_novo,
            "drug_threshold_error": drug_error,
            "sex": rng.choice(["M", "F"]),
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gd in enumerate(AORTIC_CTD_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients_for_gene(gd, seed))
    return all_pts


# ── API functions ────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    surgery_n = sum(1 for p in patients if p["aortic_surgery"])
    dissection_n = sum(1 for p in patients if p["aortic_dissection"])
    lens_n = sum(1 for p in patients if p["lens_ectopia"])
    skeletal_n = sum(1 for p in patients if p["skeletal_features"])
    de_novo_n = sum(1 for p in patients if p["de_novo"])
    error_n = sum(1 for p in patients if p["drug_threshold_error"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_dx = round(sum(p["diagnosis_age_y"] for p in patients) / n, 1)

    gene_surgery_pct = {}
    for gd in AORTIC_CTD_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_surgery_pct[gd["gene"]] = round(
            100 * sum(1 for p in gpts if p["aortic_surgery"]) / len(gpts), 1
        )

    return {
        "atlas": "Aortic-CTD-Atlas",
        "full_name": "Complete 8-Gene Hereditary Aortic & Connective Tissue Disorders Atlas",
        "subtitle": (
            "FBN1·TGFBR2·TGFBR1·COL3A1·ACTA2·SMAD3·MYH11·SKI — "
            "320 patients (8×40, seeds 1110–1117)"
        ),
        "description": (
            "Comprehensive atlas of 8 major genetic hereditary aortic and connective tissue disorders: "
            "MFS1/FBN1 (AD; ~35% heritable TAAD; extracellular matrix; aortic root; β-blocker + losartan; "
            "fluoroquinolone AVOID; surgical ≥4.5 cm; lens ectopia superior-temporal PATHOGNOMONIC); "
            "LDS2/TGFBR2 (AD; AGGRESSIVE — dissects at <4.0 cm; bifid uvula PATHOGNOMONIC; "
            "arterial tortuosity; repair ≥4.0 cm; whole-body MRA mandatory); "
            "LDS1/TGFBR1 (AD; craniosynostosis MORE PROMINENT than LDS2; bifid uvula; repair ≥4.0 cm; "
            "cervical spine instability; same threshold as LDS2); "
            "vEDS/COL3A1 (AD; MOST LETHAL — spontaneous arterial rupture WITHOUT preceding aneurysm; "
            "NO elective surgery; celiprolol ONLY RCT-proven therapy; pregnancy = uterine rupture risk); "
            "FTAAD/ACTA2 (AD; ~15% heritable TAAD; Moya-Moya cerebrovascular; iris flocculi Arg179His; "
            "straight aorta; livedo reticularis; brain MRA mandatory); "
            "LDS3/AOS/SMAD3 (AD; early osteoarthritis + aortic aneurysm = DUAL PHENOTYPE; "
            "MISDIAGNOSED as isolated OA; repair ≥4.0 cm; OA <40y → MANDATORY aortic screen); "
            "FTAAD+PDA/MYH11 (AD; PDA + TAAD combination PATHOGNOMONIC; aortic media "
            "hypocellularity on histology; neonatal echo for at-risk newborns; "
            "~2-5% heritable TAAD); "
            "SGS/SKI (AD; de novo ~95%; MARFANOID + CRANIOSYNOSTOSIS + intellectual disability; "
            "distinguishes from MFS by cognitive impairment; multidisciplinary care from birth)."
        ),
        "n": n,
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": "8"},
            {"label": "Aortic Surgery Rate", "value": f"{round(100*surgery_n/n, 1)}%"},
            {"label": "Dissection Events", "value": f"{round(100*dissection_n/n, 1)}%"},
            {"label": "Lens Ectopia (FBN1)", "value": f"{round(100*lens_n/n, 1)}%"},
            {"label": "De Novo Variants", "value": f"{round(100*de_novo_n/n, 1)}%"},
            {"label": "Mean Onset Age", "value": f"{mean_onset}y"},
            {"label": "Mean Dx Age", "value": f"{mean_dx}y"},
            {"label": "Dx Delay (mean)", "value": f"{round(mean_dx - mean_onset, 1)}y"},
            {"label": "Threshold/Drug Error", "value": f"{round(100*error_n/n, 1)}%"},
        ],
        "severity": {k: round(100 * v / n, 1) for k, v in sev.items()},
        "severity_prevalence": {k: round(100 * v / n, 1) for k, v in sev.items()},
        "disease_category_breakdown": {
            "Heritable TAAD (FBN1/MFS1)": round(100 * 40 / n, 1),
            "Loeys-Dietz Syndrome (TGFBR1/2/SMAD3)": round(100 * 120 / n, 1),
            "Vascular EDS (COL3A1)": round(100 * 40 / n, 1),
            "FTAAD + Cerebrovascular (ACTA2)": round(100 * 40 / n, 1),
            "FTAAD + PDA (MYH11)": round(100 * 40 / n, 1),
            "Shprintzen-Goldberg (SKI)": round(100 * 40 / n, 1),
        },
        "clinical_features_prevalence": {
            "Aortic Aneurysm": 100,
            "Skeletal Features": round(100 * skeletal_n / n, 1),
            "Aortic Surgery": round(100 * surgery_n / n, 1),
            "Aortic Dissection": round(100 * dissection_n / n, 1),
            "De Novo Variant": round(100 * de_novo_n / n, 1),
            "Lens Ectopia": round(100 * lens_n / n, 1),
        },
        "gene_surgery_pct": gene_surgery_pct,
        "drug_alerts": [
            {
                "type": "danger",
                "title": "COL3A1/vEDS — NO ELECTIVE VASCULAR SURGERY",
                "body": (
                    "Elective vascular surgery in vEDS/COL3A1 carries mortality often EXCEEDING "
                    "natural history risk. Arterial walls do not hold sutures. Only celiprolol "
                    "has RCT evidence. Endovascular preferred when unavoidable."
                ),
            },
            {
                "type": "danger",
                "title": "TGFBR1/2/SMAD3 — SURGICAL THRESHOLD 4.0 cm (NOT 4.5 cm)",
                "body": (
                    "LDS1/LDS2/LDS3 aortic dissection documented at <4.0 cm. "
                    "Applying Marfan 4.5 cm threshold to LDS patients is a sentinel error — "
                    "it has caused preventable dissection deaths."
                ),
            },
            {
                "type": "danger",
                "title": "FBN1/MFS — FLUOROQUINOLONE AVOIDANCE",
                "body": (
                    "Fluoroquinolone antibiotics accelerate aortic root expansion in Marfan. "
                    "Document as allergy-equivalent. Use alternative antibiotics."
                ),
            },
            {
                "type": "warning",
                "title": "ACTA2 p.Arg179His — CEREBROVASCULAR (Moya-Moya) MANDATORY Brain MRA",
                "body": (
                    "ACTA2 p.Arg179His causes Moya-Moya stroke in children. "
                    "Brain MRI/MRA mandatory at diagnosis and annually. "
                    "Iris flocculi on slit-lamp is pathognomonic."
                ),
            },
            {
                "type": "warning",
                "title": "MYH11 — PDA + TAAD = PATHOGNOMONIC Combination",
                "body": (
                    "Any family combining PDA with thoracic aortic aneurysm/dissection = "
                    "MYH11 until proven otherwise. Always ask about PDA history in all HTAAD families."
                ),
            },
        ],
        "critical_rules": [
            "FBN1: FLUOROQUINOLONE AVOID; superior-temporal lens ectopia (not inferior-nasal = homocystinuria)",
            "TGFBR1/2/SMAD3 (LDS): surgical threshold 4.0 cm NOT 4.5 cm — applying MFS threshold is a lethal error",
            "COL3A1 (vEDS): NO ELECTIVE SURGERY; celiprolol 200-400 mg BD = ONLY RCT-proven therapy",
            "ACTA2 p.Arg179His: Moya-Moya + PDA + iris flocculi; brain MRA mandatory ALL ACTA2 carriers",
            "SMAD3 (AOS): early OA <40y = MANDATORY aortic screen — most common missed diagnosis in rheumatology referrals",
            "MYH11: PDA + TAAD family history = pathognomonic combination; aortic media hypocellularity on histology",
            "SKI: MARFANOID + intellectual disability (MFS has NORMAL intelligence) = SKI/SGS; craniosynostosis urgent neurosurgery",
        ],
    }


def get_breakdown() -> list:
    patients = _gen_cohort()
    result = []
    for gd in AORTIC_CTD_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        n = len(gpts)
        surgery_pct = round(100 * sum(1 for p in gpts if p["aortic_surgery"]) / n, 1)
        dissect_pct = round(100 * sum(1 for p in gpts if p["aortic_dissection"]) / n, 1)
        lens_pct = round(100 * sum(1 for p in gpts if p["lens_ectopia"]) / n, 1)
        skel_pct = round(100 * sum(1 for p in gpts if p["skeletal_features"]) / n, 1)
        de_novo_pct = round(100 * sum(1 for p in gpts if p["de_novo"]) / n, 1)
        onset_ages = [p["onset_age_y"] for p in gpts]
        mean_onset = round(sum(onset_ages) / n, 1)

        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "disease_type": gd["disease_type"],
            "inheritance": "AD",
            "n_patients": n,
            "mean_onset_age_y": mean_onset,
            "severity_weights": gd["severity_weights"],
            "aortic_surgery_pct": surgery_pct,
            "dissection_pct": dissect_pct,
            "lens_ectopia_pct": lens_pct,
            "skeletal_features_pct": skel_pct,
            "de_novo_pct": de_novo_pct,
            "prevalence_per_100k": gd["prevalence_per_100k"],
            "key_distinguishing": gd["key_distinguishing"],
            "treatment_options": gd["treatment_options"],
            "drug_alerts": gd["drug_alerts"],
            "clinical_rules": gd["clinical_rules"],
        })
    return result


def get_definitions() -> list:
    return [
        {
            "term": "HTAAD",
            "definition": (
                "Heritable Thoracic Aortic Aneurysm and Dissection — genetic conditions causing "
                "aneurysm formation and/or dissection of the thoracic aorta (ascending, arch, "
                "descending) due to pathogenic variants in connective tissue or smooth muscle genes "
                "(FBN1, TGFBR1/2, COL3A1, ACTA2, SMAD3, MYH11, MYLK, PRKG1, LOX, ELN, etc.). "
                "~20% of thoracic aortic aneurysms are heritable."
            ),
        },
        {
            "term": "Marfan Syndrome (MFS1)",
            "definition": (
                "FBN1-related systemic connective tissue disorder (OMIM #154700); AD; ~35% of "
                "heritable TAAD; diagnostic Ghent 2010 criteria; aortic root aneurysm + "
                "lens ectopia + skeletal features; β-blocker + losartan; fluoroquinolone AVOID; "
                "surgical ≥4.5 cm (lower if rapid growth, family history dissection, or pregnancy)."
            ),
        },
        {
            "term": "Loeys-Dietz Syndrome (LDS)",
            "definition": (
                "TGF-β pathway aortopathy (TGFBR1=LDS1, TGFBR2=LDS2, SMAD3=LDS3, SMAD2=LDS4, "
                "TGFB2=LDS5, TGFB3=LDS6, SKI=LDS7); AD; AGGRESSIVE — dissect at smaller "
                "diameters than Marfan; repair threshold ≥4.0 cm (NOT 4.5 cm); "
                "bifid uvula PATHOGNOMONIC; arterial tortuosity hallmark; whole-body MRA mandatory."
            ),
        },
        {
            "term": "Bifid Uvula",
            "definition": (
                "Split (forked) uvula — PATHOGNOMONIC for Loeys-Dietz Syndrome (LDS1 and LDS2). "
                "Examine uvula in ALL patients with aortic root dilation. "
                "LDS patients are frequently misdiagnosed as Marfan syndrome; "
                "bifid uvula is the KEY distinguishing clinical sign."
            ),
        },
        {
            "term": "Vascular EDS (vEDS)",
            "definition": (
                "COL3A1-related vascular connective tissue disorder (OMIM #130050); AD; "
                "MOST LETHAL CTD; spontaneous arterial rupture often WITHOUT preceding aneurysm; "
                "NO elective vascular surgery; celiprolol 200-400 mg BD = only RCT evidence; "
                "bowel perforation (sigmoid colon); uterine rupture in pregnancy; "
                "median survival 48-51 years."
            ),
        },
        {
            "term": "TGF-β Paradox (LDS)",
            "definition": (
                "Loss-of-function variants in TGF-β receptors (TGFBR1, TGFBR2) or downstream "
                "effectors (SMAD3) paradoxically INCREASE rather than decrease TGF-β signalling "
                "in the aortic wall — via compensatory upregulation through non-canonical "
                "ERK/MAPK pathway. This is the mechanistic basis for LDS and explains why "
                "TGF-β target genes are overexpressed despite receptor LOF."
            ),
        },
        {
            "term": "Celiprolol",
            "definition": (
                "Cardioselective β1-blocker with partial β2-agonism (vasodilatory). "
                "The ONLY therapy with randomised controlled trial evidence for reducing "
                "arterial events in vascular EDS (COL3A1) — Ong et al. 2010, Lancet. "
                "Dose: 200 mg BD titrating to 400 mg BD. "
                "Mechanism: reduces aortic wall shear stress + β2 vasodilation reduces "
                "mechanical stress on fragile arterial walls. "
                "STANDARD β-blockers (atenolol, metoprolol) do NOT have equivalent vEDS evidence."
            ),
        },
        {
            "term": "Moya-Moya (ACTA2)",
            "definition": (
                "Progressive bilateral stenosis/occlusion of supraclinoid internal carotid "
                "arteries with basal ganglia collateral vessel formation (puff-of-smoke "
                "angiographic appearance). Seen predominantly in ACTA2 p.Arg179His variant — "
                "causes stroke in children and young adults. Brain MRI/MRA mandatory. "
                "Antithrombotic (aspirin) and revascularisation surgery (STA-MCA bypass) "
                "for symptomatic cases."
            ),
        },
        {
            "term": "Iris Flocculi",
            "definition": (
                "Small whitish nodules on the ciliary body visible on slit-lamp examination. "
                "PATHOGNOMONIC for ACTA2 p.Arg179His variant. Clinically silent but "
                "diagnostically definitive — request slit-lamp specifically when this variant "
                "is suspected. Not a feature of MFS or LDS."
            ),
        },
        {
            "term": "Aneurysm-Osteoarthritis Syndrome (AOS/LDS3)",
            "definition": (
                "SMAD3-related dual phenotype syndrome combining early-onset osteoarthritis "
                "(joints; beginning in 20s-30s — hips, knees, elbows) with aortic aneurysm "
                "and dissection. Frequently MISDIAGNOSED as isolated primary OA by rheumatologists "
                "→ aortic imaging never obtained → preventable death from dissection. "
                "Key rule: OA <40 years = mandatory aortic screen."
            ),
        },
        {
            "term": "Shprintzen-Goldberg Syndrome (SGS)",
            "definition": (
                "SKI-related syndrome (OMIM #182212); AD; de novo ~95%; "
                "Marfanoid habitus + CRANIOSYNOSTOSIS + intellectual disability + aortic dilation. "
                "KEY DISTINGUISHING: intellectual disability (ABSENT in MFS/FBN1 which has normal "
                "intelligence). Urgent craniofacial neurosurgery in first year of life for "
                "craniosynostosis; multidisciplinary team from birth."
            ),
        },
        {
            "term": "PDA + TAAD (MYH11)",
            "definition": (
                "Patent Ductus Arteriosus (failure of postnatal ductal closure) combined with "
                "familial thoracic aortic aneurysm and dissection — PATHOGNOMONIC combination "
                "for MYH11 (smooth muscle myosin heavy chain) mutation. "
                "Always elicit PDA history when taking pedigree for heritable TAAD. "
                "Aortic media hypocellularity on surgical histology also suggests MYH11."
            ),
        },
        {
            "term": "Ectopia Lentis",
            "definition": (
                "Subluxation (partial dislocation) of the crystalline lens of the eye. "
                "SUPERIOR-TEMPORAL direction = FBN1/Marfan Syndrome (PATHOGNOMONIC). "
                "INFERIOR-NASAL direction = Homocystinuria (CBS). "
                "ABSENT in LDS, ACTA2, MYH11, SMAD3, SKI — distinguishes MFS from other CTDs. "
                "Requires slit-lamp examination for diagnosis; clinical flashlight exam may miss."
            ),
        },
        {
            "term": "Ghent Nosology 2010",
            "definition": (
                "Revised diagnostic criteria for Marfan Syndrome (De Paepe 2010). "
                "Major categories: aortic root Z-score ≥2; ectopia lentis; pathogenic FBN1 variant; "
                "systemic score ≥7. For MFS diagnosis: FBN1 + aortic root dilation OR ectopia lentis "
                "= confirmed MFS. Without FBN1: aortic dilation + ectopia lentis = MFS. "
                "Also enables diagnoses of MASS phenotype, Marfan-related disorders."
            ),
        },
        {
            "term": "Craniosynostosis",
            "definition": (
                "Premature fusion of one or more cranial sutures (normally fuse in adulthood). "
                "Seen in LDS1/TGFBR1 and SKI/SGS (TGF-β pathway disorders). "
                "Consequences: abnormal skull shape; raised intracranial pressure → visual loss "
                "and cognitive impairment. "
                "Requires urgent craniofacial neurosurgery in first year of life. "
                "Screen with head circumference + clinical assessment + skull X-ray/CT."
            ),
        },
        {
            "term": "Aortic Media Hypocellularity",
            "definition": (
                "Histological finding of reduced smooth muscle cell density in the aortic wall — "
                "DISTINCTIVE for MYH11 pathogenic variants. "
                "Found on surgical aortic tissue specimen. If identified → request MYH11 genetic "
                "testing if not already performed. Reflects VSMC dysfunction from loss of "
                "smooth muscle myosin contractile apparatus."
            ),
        },
    ]
