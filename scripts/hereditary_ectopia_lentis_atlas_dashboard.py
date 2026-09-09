#!/usr/bin/env python3
"""Hereditary-Ectopia-Lentis-Atlas — Complete 8-Gene Atlas
(FBN1 · CBS · ADAMTSL4 · ADAMTS10 · ADAMTS17 · LTBP2 · SUOX · FBN2).

FBN1    (Fibrillin-1; 2871 aa; ~350 kDa; 15q21.1; AD;
          Marfan syndrome — BILATERAL SUPEROTEMPORAL ECTOPIA LENTIS PATHOGNOMONIC;
          aortic root dilation (Z-score >2), tall marfanoid habitus, arachnodactyly;
          seed SEED_BASE+0).
CBS     (Cystathionine beta-synthase; 551 aa; ~63 kDa; 21q22.3; AR;
          Classic homocystinuria — BILATERAL INFERONASAL ECTOPIA LENTIS PATHOGNOMONIC;
          OPPOSITE direction to Marfan; thromboembolism risk; B6-responsiveness test;
          seed SEED_BASE+1).
ADAMTSL4 (ADAMTS-like protein 4; 1212 aa; ~138 kDa; 2q36.1; AR;
           Isolated ectopia lentis / ectopia lentis et pupillae (ELP) — NO SYSTEMIC;
           seed SEED_BASE+2).
ADAMTS10 (A Disintegrin And Metalloproteinase with Thrombospondin Motifs 10; 1103 aa;
           ~125 kDa; 19p13.2; AR;
           Weill-Marchesani syndrome type 2 (WMS2) — MICROSPHEROPHAKIA + ANTERIOR
           LENS SUBLUXATION + SHORT STATURE + BRACHYDACTYLY (INVERSE MARFAN);
           seed SEED_BASE+3).
ADAMTS17 (A Disintegrin And Metalloproteinase with Thrombospondin Motifs 17; 1221 aa;
           ~139 kDa; 15q26.3; AR;
           Weill-Marchesani syndrome type 4 (WMS4) — similar to WMS2, milder;
           seed SEED_BASE+4).
LTBP2   (Latent TGF-β Binding Protein 2; 1821 aa; ~200 kDa; 14q24.3; AR;
          Microspherophakia with secondary glaucoma — SPHERICAL SUBLUXATED LENS
          PATHOGNOMONIC; AVOID miotics (pupillary block risk); Gulf Arab founder;
          seed SEED_BASE+5).
SUOX    (Sulfite oxidase; 545 aa; ~60 kDa; 12q13.2; AR;
          Isolated sulfite oxidase deficiency — ECTOPIA LENTIS + NEONATAL REFRACTORY
          SEIZURES + SULFITE-POSITIVE URINE DIPSTICK PATHOGNOMONIC;
          KEY DDx homocystinuria — no plasma homocysteine elevation;
          seed SEED_BASE+6).
FBN2    (Fibrillin-2; 2832 aa; ~330 kDa; 5q23.3; AD;
          Congenital contractural arachnodactyly (CCA / Beals-Hecht syndrome) —
          CRUMPLED EAR PINNA + CONGENITAL JOINT CONTRACTURES PATHOGNOMONIC;
          KEY DDx Marfan (FBN1): NO aortic root dilation in CCA; occasional EL;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2398-2405).
"""

import random

SEED_BASE = 2398

EL_GENES = [
    # -- FBN1 -- Marfan Syndrome --------------------------------------------------------------
    {
        "gene": "FBN1",
        "alt_name": (
            "FBN1 (FBN1-2871aa-15q21.1 / AD -- "
            "MARFAN-SYNDROME-MFS-BILATERAL-SUPEROTEMPORAL-ECTOPIA-LENTIS-PATHOGNOMONIC -- "
            "AORTIC-ROOT-DILATION-Z-SCORE>2-ASCENDING-AORTA-DISSECTION-RISK -- "
            "TALL-MARFANOID-HABITUS-ARACHNODACTYLY-PECTUS-SCOLIOSIS -- "
            "LOSARTAN-ARB-AORTIC-PROTECTION-BETA-BLOCKER-LIFELONG)"
        ),
        "protein": (
            "FBN1 -- 15q21.1 AD -- FBN1-2871aa -- "
            "Fibrillin-1-350kDa-Extracellular-Matrix-Glycoprotein -- "
            "47-EGF-Like-Domains-7-TB-Calcium-Binding-EGF-Like-Domains-RGD-Integrin-Binding -- "
            "Structural-Scaffold-For-Elastic-Fibres-Sequesters-Latent-TGF-Beta-LLC -- "
            "Expressed-Connective-Tissue-Lens-Zonules-Aorta-Skin-Bone-Periosteum -- "
            "OMIM-Gene-134797-Disease-MFS-154700"
        ),
        "locus": "15q21.1",
        "protein_size": "2871 aa / 350 kDa",
        "inheritance": (
            "AD — haploinsufficiency + dominant-negative mechanisms; "
            "FBN1 encodes the principal component of zonular fibres suspending the crystalline lens — "
            "zonular FBN1 haploinsufficiency → structurally weak zonules → lens dislocation (superotemporal, bilateral); "
            "FBN1 also sequesters TGF-β in the LLC (large latent complex) — FBN1 LOF → excess active TGF-β → "
            "tissue inflammation, bone overgrowth, aortic smooth muscle dysregulation; "
            "pathogenic variants: missense (most common, >1800 described), frameshift, nonsense, splice; "
            "de novo: ~25-30% of cases; familial: autosomal dominant with near-complete penetrance; "
            "genotype-phenotype: early-onset severe neonatal Marfan (OMIM 154700): missense in exons 24-32; "
            "isolated lens dislocation (no cardiac): possible with mild FBN1 variants (MASS phenotype); "
            "no homozygous LOF in humans — likely lethal"
        ),
        "disease_category": "Marfan syndrome (MFS) — fibrillinopathy with ectopia lentis + aortic root dilation + marfanoid skeletal features",
        "disease_pathway": (
            "FBN1 (Fibrillin-1) is the primary structural scaffold protein of extracellular matrix microfibrils. "
            "In the eye, FBN1 forms the ZONULAR FIBRES (suspensory ligament) that suspend the crystalline lens from the ciliary body. "
            "ECTOPIA LENTIS MECHANISM: FBN1 haploinsufficiency → structurally weakened zonular fibres → "
            "progressive fibre breakage → lens displacement superiorly and temporally (SUPEROTEMPORAL). "
            "Superotemporal displacement is characteristic because the inferior zonules bear greater mechanical load; "
            "when FBN1-deficient upper zonules fail first → lens shifts up and temporally. "
            "AORTIC MECHANISM: FBN1 microfibrils in the aortic wall sequester the large latent TGF-β complex (LLC). "
            "FBN1 LOF → LLC destabilisation → excess active TGF-β → "
            "smooth muscle cell apoptosis + matrix metalloproteinase activation → "
            "AORTIC ROOT DILATION (measured as Z-score at sinus of Valsalva) → "
            "risk of aortic dissection (Type A, catastrophic if untreated). "
            "ARB (losartan) + beta-blocker reduce TGF-β signalling and aortic wall stress — "
            "primary medical prevention. "
            "SKELETAL: excess TGF-β + FBN1 loss → bone overgrowth → tall stature, long limbs, arachnodactyly, scoliosis, pectus. "
            "LENS DISLOCATION: seen in ~60-80% of Marfan patients; may be partial (subluxation) or complete (luxation into vitreous or anterior chamber)."
        ),
        "pathognomonic": (
            "BILATERAL SUPEROTEMPORAL ECTOPIA LENTIS (lens displaced superiorly and temporally) + "
            "AORTIC ROOT DILATION (echocardiographic Z-score >2 at sinus of Valsalva) + "
            "TALL MARFANOID HABITUS (arm span > height, upper:lower segment ratio abnormal) = MARFAN TRIAD PATHOGNOMONIC. "
            "SUPEROTEMPORAL DIRECTION IS DIAGNOSTIC: superior temporal zonular failure first — "
            "distinguishes from CBS homocystinuria (INFERONASAL) and WMS/LTBP2 (anterior spherical). "
            "WRIST SIGN (Walker-Murdoch): opposite thumb overlaps little fingernail when wrapping opposite wrist. "
            "THUMB SIGN (Steinberg): thumb protrudes beyond ulnar border when fingers flex around closed fist. "
            "ECTOPIA LENTIS ON SLIT LAMP: iridodonesis (iris tremor when eye moves) — zonular instability. "
            "AORTIC DISSECTION RISK: prophylactic aortic root surgery when diameter >50mm (or >45mm with family history of dissection). "
            "BETA-BLOCKER + LOSARTAN: mandatory from diagnosis — reduces aortic root expansion rate. "
            "Revised Ghent criteria 2010: systemic score ≥7 + EL + FBN1 mutation OR aortic root Z ≥2 = Marfan diagnosis."
        ),
        "treatment": (
            "AORTIC PROTECTION (MANDATORY, LIFELONG): "
            "Beta-blocker (atenolol/propranolol) — reduce aortic wall stress; "
            "ARB (losartan 1.4mg/kg/day paediatric) — TGF-β pathway antagonism + aortic protection; "
            "Annual echocardiography — aortic root diameter monitoring; "
            "PROPHYLACTIC AORTIC ROOT SURGERY: elective Bentall procedure or valve-sparing root replacement "
            "(David/Yacoub) when root >50mm (or >45mm high-risk); "
            "ECTOPIA LENTIS MANAGEMENT: "
            "Aphakic spectacles/contact lenses if lens tilted (partial subluxation with useful vision); "
            "LENS EXTRACTION (lensectomy ± vitrectomy): if visual acuity poor or pupillary block glaucoma risk; "
            "iris-claw or sutured IOL for aphakia post-extraction; "
            "AVOID contact sports, isometric exercise, Valsalva manoeuvre (aortic stress); "
            "ENDOCARDITIS PROPHYLAXIS: per cardiology if MV disease; "
            "SKELETAL: physiotherapy for scoliosis; orthopaedic referral for severe curves; "
            "GENETIC COUNSELLING: 50% AD; prenatal/preimplantation diagnosis available; "
            "CASCADE FAMILY TESTING: echo + ophthalmology for all first-degree relatives."
        ),
        "key_features": [
            "Bilateral superotemporal ectopia lentis (60-80%) — zonular FBN1 failure",
            "Aortic root dilation (Z-score >2) — annual echo mandatory",
            "Tall marfanoid habitus — arm span > height, long fingers/toes",
            "Wrist sign + thumb sign — clinical diagnostic clues",
            "Iridodonesis — iris tremor on slit lamp (zonular instability)",
            "Losartan + beta-blocker mandatory — TGF-β reduction + aortic protection",
            "Prophylactic aortic root surgery >50mm (>45mm high-risk)",
            "Revised Ghent criteria 2010 — systematic scoring for diagnosis",
        ],
        "key_ddx": (
            "CBS (homocystinuria): EL is INFERONASAL not superotemporal; SHORT not tall; "
            "THROMBOEMBOLISM risk; elevated plasma homocysteine; B6 test mandatory; "
            "FBN2 (CCA/Beals): CRUMPLED EAR + contractures + NO aortic root dilation; no thromboembolism; "
            "ADAMTSL4 (isolated EL): NO systemic features — no tall stature, no aortic root; "
            "ADAMTS10/17 (WMS): SHORT stature + brachydactyly — INVERSE Marfan body habitus; microspherophakia; "
            "LTBP2 (microspherophakia): spherical lens anteriorly displaced; Gulf Arab; no tall stature; "
            "Lens subluxation from trauma: unilateral; history; no systemic features."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital zonular weakness; EL apparent by childhood; aortic dilation progressive",
        "surgical_urgency": "Urgent cardiac surveillance annual; lens surgery when VA or pupillary block",
        "gene_family": "Fibrillin (FBN1 — EGF-like domain extracellular matrix protein)",
        "morphology": "Bilateral superotemporal lens dislocation + marfanoid habitus + aortic root dilation",
    },

    # -- CBS -- Classic Homocystinuria ---------------------------------------------------------
    {
        "gene": "CBS",
        "alt_name": (
            "CBS (CBS-551aa-21q22.3 / AR -- "
            "CLASSIC-HOMOCYSTINURIA-HCU-BILATERAL-INFERONASAL-ECTOPIA-LENTIS-PATHOGNOMONIC -- "
            "THROMBOEMBOLISM-RISK-PULMONARY-EMBOLISM-DVT-STROKE-MORTALITY -- "
            "B6-PYRIDOXINE-RESPONSIVE-50pct-MUST-TEST-B6-FIRST -- "
            "INFERONASAL-DIRECTION-KEY-DDx-MARFAN-SUPEROTEMPORAL)"
        ),
        "protein": (
            "CBS -- 21q22.3 AR -- CBS-551aa -- "
            "Cystathionine-Beta-Synthase-63kDa-PLP-Dependent-Pyridoxal-5-Phosphate-Enzyme -- "
            "Converts-Homocysteine+Serine-To-Cystathionine-Transsulfuration-Pathway -- "
            "Heme-Domain-Catalytic-Domain-CBS-Domains-C-Terminal-Regulatory -- "
            "Expressed-Liver-Brain-Kidney-Intestine-Lens-Vascular-Endothelium -- "
            "OMIM-Gene-613381-Disease-HCU-236200"
        ),
        "locus": "21q22.3",
        "protein_size": "551 aa / 63 kDa",
        "inheritance": (
            "AR — biallelic LOF; "
            "CBS catalyses the first step of the transsulfuration pathway (homocysteine → cystathionine → cysteine); "
            "CBS LOF → homocysteine accumulates → elevated plasma total homocysteine (tHcy usually >100 µmol/L in untreated classic HCU); "
            "PYRIDOXINE-RESPONSIVE (B6-responsive): ~50% of patients respond to pyridoxine 200-1000mg/day — "
            "B6 acts as cofactor to PLP-dependent CBS → residual enzyme activity; B6-responsive patients have milder phenotype; "
            "B6-NON-RESPONSIVE: require methionine-restricted diet + betaine supplementation + cysteine supplementation; "
            "NEONATAL SCREENING: elevated methionine on Guthrie/tandem-MS screen — but false-negatives in B6-responsive; "
            "common variants: p.Ile278Thr (Irish founder, B6-non-responsive), p.Gly307Ser (Irish/European, B6-responsive)"
        ),
        "disease_category": "Classic homocystinuria (HCU) — transsulfuration defect with ectopia lentis + thromboembolism + marfanoid features + intellectual disability",
        "disease_pathway": (
            "CBS (Cystathionine Beta-Synthase) is the pyridoxal-5-phosphate (PLP)-dependent enzyme that condenses "
            "homocysteine with serine to form cystathionine, the first step of the transsulfuration pathway "
            "(homocysteine → cystathionine → cysteine → glutathione). "
            "CBS LOF → HOMOCYSTEINE ACCUMULATION in plasma and tissues: "
            "ECTOPIA LENTIS: homocysteine disrupts disulfide bonds in fibrillin-1 (FBN1) and fibrillin-2 — "
            "impairs zonular fibre integrity → progressive INFERONASAL lens dislocation (lower zonules fail first — "
            "OPPOSITE of Marfan superotemporal). "
            "VASCULAR TOXICITY: homocysteine damages endothelium → platelet aggregation + oxidative stress + "
            "fibrin cross-linking disruption → THROMBOEMBOLIC RISK across all vessel beds "
            "(deep vein thrombosis, pulmonary embolism, stroke, MI) — a major cause of morbidity and mortality. "
            "SKELETAL: homocysteine interferes with collagen cross-linking → marfanoid skeletal features "
            "(tall stature, arachnodactyly, scoliosis, pectus) — SIMILAR TO FBN1 MARFAN in body habitus. "
            "INTELLECTUAL DISABILITY: homocysteine is neurotoxic — interference with N-methyl-D-aspartate (NMDA) receptors, "
            "myelin synthesis disruption → cognitive impairment (variable, worse in B6-non-responsive). "
            "PYRIDOXINE RESPONSE: B6-cofactor boosts residual CBS activity in B6-responsive mutations — "
            "dramatically reduces plasma homocysteine and EL risk."
        ),
        "pathognomonic": (
            "BILATERAL INFERONASAL ECTOPIA LENTIS (lens displaced inferiorly and nasally) + "
            "ELEVATED PLASMA TOTAL HOMOCYSTEINE (tHcy >100 µmol/L untreated) + "
            "THROMBOEMBOLIC EVENTS (DVT, PE, stroke — any age, risk from childhood) = HCU TRIAD PATHOGNOMONIC. "
            "INFERONASAL DIRECTION IS KEY DDx FROM MARFAN: "
            "FBN1 Marfan → SUPEROTEMPORAL; CBS HCU → INFERONASAL — direction is diagnostic before genetics. "
            "B6 RESPONSIVENESS TEST (MANDATORY BEFORE DIAGNOSIS IS COMPLETE): "
            "pyridoxine 200-500 mg/day × 3-4 weeks → measure plasma tHcy — "
            "normalisation = B6-responsive (better prognosis, milder EL, less thrombosis risk); "
            "no response = B6-non-responsive (strict methionine diet + betaine + cysteine mandatory). "
            "URINE SODIUM NITROPRUSSIDE TEST (silver cyanide reaction): detects disulfide compounds including homocystine — "
            "rapid bedside screen; "
            "NEONATAL SCREENING: elevated methionine on tandem-MS Guthrie — screen all positive for CBS. "
            "ANTI-COAGULATION: lifelong prophylactic anticoagulation (aspirin or LMWH) in high-risk cases "
            "— anaesthesia poses HIGH THROMBOEMBOLISM RISK (dehydration + prothrombotic state)."
        ),
        "treatment": (
            "B6-RESPONSIVE (50%): "
            "Pyridoxine 200-1000mg/day (under supervision) — normalise plasma tHcy < 50 µmol/L; "
            "methionine restriction mild; folic acid 5mg/day supplementation; "
            "BETAINE: 6g/day (in adults) — alternative remethylation pathway donor; "
            "B6-NON-RESPONSIVE: "
            "Methionine-restricted diet (<50-100mg/day methionine) — phenylalanine-free amino acid formula; "
            "BETAINE (trimethylglycine) 6-20g/day — lowers tHcy via remethylation; "
            "CYSTEINE supplementation (becomes essential amino acid in HCU); "
            "FOLIC ACID + VITAMIN B12 — co-factors for remethylation; "
            "THROMBOEMBOLISM PREVENTION: "
            "Pre-operative LMWH + hydration — HIGH RISK with GENERAL ANAESTHESIA; "
            "lifelong aspirin in high-risk; anticoagulation for established thrombosis; "
            "ECTOPIA LENTIS: "
            "Aphakic spectacles/contact lenses if partial subluxation; "
            "Lensectomy + vitrectomy if pupillary block or VA poor; "
            "caution with general anaesthesia for eye surgery (thromboembolism); "
            "INTELLECTUAL DISABILITY: early intervention, metabolic control optimises cognitive outcome; "
            "NEONATAL SCREENING + CASCADE: tandem-MS neonatal screen + family screening."
        ),
        "key_features": [
            "Bilateral inferonasal ectopia lentis — OPPOSITE direction to Marfan",
            "Elevated plasma total homocysteine (tHcy >100 µmol/L untreated)",
            "Thromboembolism risk — DVT/PE/stroke from childhood (major mortality cause)",
            "B6-responsiveness test MANDATORY — 50% respond (better prognosis)",
            "Marfanoid habitus (tall, arachnodactyly) — but INFERONASAL EL distinguishes from FBN1",
            "Intellectual disability (variable, worse B6-non-responsive)",
            "Anaesthesia: HIGH thromboembolism risk — LMWH + hydration pre-op",
            "Neonatal screen: elevated methionine on tandem-MS Guthrie",
        ],
        "key_ddx": (
            "FBN1 (Marfan): EL is SUPEROTEMPORAL not inferonasal; normal plasma homocysteine; "
            "no thromboembolism; aortic root dilation; "
            "SUOX (sulfite oxidase): EL + neonatal seizures + POSITIVE URINE SULFITE — NO elevated homocysteine; "
            "MTR/MTRR (remethylation defect): low methionine + elevated homocysteine — responds to B12 not B6; "
            "MTHFR (thermolabile): mild hyperhomocysteinaemia only — not classic HCU range; "
            "Isolated EL (ADAMTSL4): no systemic features at all; normal homocysteine; "
            "Weill-Marchesani (ADAMTS10/17/LTBP2): SHORT stature, brachydactyly, microspherophakia — not marfanoid."
        ),
        "systemic_involvement": True,
        "onset_age": "EL presents 2-10 years (progressive); thromboembolism risk from childhood",
        "surgical_urgency": "High — thromboembolic risk with any surgery; lens surgery with LMWH cover",
        "gene_family": "PLP-dependent transsulfuration enzyme (CBS — cystathionine beta-synthase)",
        "morphology": "Bilateral inferonasal lens dislocation + marfanoid habitus + thrombophilic state",
    },

    # -- ADAMTSL4 -- Isolated Ectopia Lentis / Ectopia Lentis et Pupillae --------------------
    {
        "gene": "ADAMTSL4",
        "alt_name": (
            "ADAMTSL4 (ADAMTSL4-1212aa-2q36.1 / AR -- "
            "ISOLATED-ECTOPIA-LENTIS-IREL-ECTOPIA-LENTIS-ET-PUPILLAE-ELP -- "
            "NO-SYSTEMIC-FEATURES-PURE-OCULAR-PRESENTATION -- "
            "BILATERAL-EL-PLUS-OR-MINUS-PUPIL-ECTOPIA-ELP-PATHOGNOMONIC -- "
            "KEY-DDx-MARFAN-HOMOCYSTINURIA-EXCLUDED-BY-NORMAL-SYSTEMIC)"
        ),
        "protein": (
            "ADAMTSL4 -- 2q36.1 AR -- ADAMTSL4-1212aa -- "
            "ADAMTS-Like-Protein-4-138kDa-Secreted-ECM-Protein -- "
            "N-Terminal-Signal-Peptide-Thrombospondin-Type-1-Repeats-TSR-PLAC-Module -- "
            "Expressed-Lens-Zonules-Ciliary-Body-Anterior-Segment -- "
            "Regulates-Fibrillin-1-Microfibril-Assembly-In-Zonular-Fibres -- "
            "OMIM-Gene-610113-Disease-IREL-225100"
        ),
        "locus": "2q36.1",
        "protein_size": "1212 aa / 138 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "ADAMTSL4 is a member of the ADAMTS superfamily secreted into the extracellular matrix; "
            "expressed specifically in the ciliary body and zonular fibres of the eye; "
            "ADAMTSL4 interacts with fibrillin-1 and fibrillin-2 microfibrils — critical for proper "
            "zonular fibre assembly and maintenance; "
            "biallelic LOF → zonular fibre weakness → ectopia lentis without systemic involvement; "
            "ELP (ectopia lentis et pupillae — displaced lens + displaced pupil in opposite directions): "
            "ADAMTSL4 is the commonest known genetic cause of ELP; "
            "heterozygous carriers: clinically unaffected (haploinsufficiency insufficient for EL); "
            "mutations: frameshift, missense, splice — diverse alleles; no common founder variant in all populations"
        ),
        "disease_category": "Isolated ectopia lentis (IREL) / ectopia lentis et pupillae (ELP) — pure ocular fibrillinopathy (ADAMTSL4-associated)",
        "disease_pathway": (
            "ADAMTSL4 belongs to the ADAMTS-like (ADAMTSL) subfamily — secreted ECM proteins lacking the "
            "metalloprotease domain of ADAMTS enzymes. "
            "ADAMTSL4 is expressed in the ciliary body and secreted into the periciliary space, "
            "where it associates with fibrillin-1 and fibrillin-2 microfibrils forming the ZONULAR FIBRES. "
            "PATHOMECHANISM: ADAMTSL4 LOF → impaired fibrillin microfibril assembly at the ciliary-zonular interface → "
            "structurally abnormal zonular fibres (reduced mechanical strength) → "
            "progressive zonular failure → BILATERAL ECTOPIA LENTIS (direction variable — usually superotemporal or nasal, "
            "less directionally predictable than FBN1 Marfan). "
            "ECTOPIA LENTIS ET PUPILLAE (ELP): In ELP, the lens and pupil are displaced in OPPOSITE DIRECTIONS — "
            "ADAMTSL4 biallelic LOF is the most common identified cause; "
            "pupil ectopia results from concurrent developmental abnormality of the iris sphincter/dilator attachment zone "
            "at the ciliary body where ADAMTSL4 is expressed. "
            "NO SYSTEMIC FEATURES: ADAMTSL4 expression is highly tissue-specific (eye > other tissues); "
            "biallelic LOF does not cause marfanoid features, aortic disease, skeletal changes, "
            "or homocysteine elevation — PURE OCULAR DISEASE."
        ),
        "pathognomonic": (
            "BILATERAL ECTOPIA LENTIS (direction variable, often superotemporal or superior) WITH NORMAL SYSTEMIC EXAM "
            "(no marfanoid features, no tall stature, no aortic dilation, normal plasma homocysteine) = IREL PATHOGNOMONIC. "
            "ECTOPIA LENTIS ET PUPILLAE (ELP): lens displaced in ONE direction + pupil displaced in OPPOSITE direction "
            "(pupil ectopia opposite to lens — displaced pupil visible as oval/eccentric pupil on slit lamp); "
            "ELP is PATHOGNOMONIC for ADAMTSL4 as the most common cause, though not exclusive. "
            "SLIT LAMP: iridodonesis (iris tremor) — unstable zonules; "
            "bilateral EL at young age (typically detected 1st-2nd decade) in an otherwise normal child → "
            "HIGH SUSPICION for ADAMTSL4 biallelic LOF. "
            "INVESTIGATIONS TO EXCLUDE: "
            "plasma total homocysteine (normal — excludes CBS HCU); "
            "echocardiography (normal aortic root — excludes FBN1 Marfan); "
            "urine sulfite dipstick (normal — excludes SUOX deficiency); "
            "GENE PANEL: FBN1 + CBS + ADAMTSL4 + ADAMTS10/17 + LTBP2 + SUOX + FBN2 — all EL genes."
        ),
        "treatment": (
            "ECTOPIA LENTIS MANAGEMENT: "
            "Optical refraction first — aphakic/astigmatic correction with spectacles or contact lenses; "
            "if partial subluxation with useful vision through intact zonular sector: "
            "large-pupil contact lens to maximise aperture through clearest lens area; "
            "LENSECTOMY (pars plana) + VITRECTOMY: when lens pupillary margin, VA poor, or pupillary block; "
            "APHAKIC REHABILITATION: iris-claw ACIOL or sutured posterior chamber IOL post-lensectomy; "
            "scleral-fixated IOL in adults; "
            "GLAUCOMA SURVEILLANCE: pupillary block glaucoma risk if lens migrates anteriorly — "
            "laser iridotomy prophylactic if lens approaching pupil plane; "
            "AMBLYOPIA: childhood onset EL → risk of meridional amblyopia — patching + optical correction essential; "
            "GENETIC COUNSELLING: AR — 25% recurrence; sibling testing essential; "
            "NO SYSTEMIC WORKUP NEEDED beyond exclusion panel at diagnosis; "
            "ANNUAL: dilated fundus exam (RD risk if lens luxated into vitreous) + IOP + VA."
        ),
        "key_features": [
            "Bilateral ectopia lentis — pure ocular, no systemic features (key distinguisher)",
            "Ectopia lentis et pupillae (ELP): lens + pupil displaced in opposite directions",
            "ADAMTSL4 most common cause of ELP",
            "Normal plasma homocysteine, normal aortic root, normal stature",
            "Zonular FBN1-microfibril assembly defect — ciliary body expressed",
            "AR biallelic LOF — diverse alleles, no single founder",
            "Amblyopia risk in children — early optical correction + patching",
            "Lensectomy + vitrectomy when VA poor or pupillary block risk",
        ],
        "key_ddx": (
            "FBN1 (Marfan): SUPEROTEMPORAL + tall stature + aortic root dilation — systemic features present; "
            "CBS (HCU): INFERONASAL + elevated homocysteine + thromboembolism + marfanoid; "
            "ADAMTS10/17 (WMS): SHORT stature + brachydactyly + microspherophakia — systemic; "
            "LTBP2 (microspherophakia): spherical lens anteriorly displaced + Gulf Arab founder; "
            "SUOX deficiency: EL + neonatal seizures + urine sulfite positive — neurological; "
            "Traumatic EL: unilateral, history; "
            "Acquired EL (pseudoexfoliation): elderly, anterior direction, PXE findings."
        ),
        "systemic_involvement": False,
        "onset_age": "1st-2nd decade (progressive zonular failure)",
        "surgical_urgency": "Moderate — lensectomy when VA poor or pupillary block risk",
        "gene_family": "ADAMTS-like protein (ADAMTSL4 — ECM fibrillin-interacting protein)",
        "morphology": "Bilateral ectopia lentis ± ectopia lentis et pupillae — no systemic disease",
    },

    # -- ADAMTS10 -- Weill-Marchesani Syndrome Type 2 ----------------------------------------
    {
        "gene": "ADAMTS10",
        "alt_name": (
            "ADAMTS10 (ADAMTS10-1103aa-19p13.2 / AR -- "
            "WEILL-MARCHESANI-SYNDROME-TYPE-2-WMS2-MICROSPHEROPHAKIA-PATHOGNOMONIC -- "
            "ANTERIOR-LENS-SUBLUXATION-SECONDARY-GLAUCOMA-PUPILLARY-BLOCK -- "
            "SHORT-STATURE-BRACHYDACTYLY-JOINT-STIFFNESS-INVERSE-MARFAN-PHENOTYPE -- "
            "AVOID-MIOTICS-PUPILLARY-BLOCK-RISK-LASER-PI-PROPHYLACTIC)"
        ),
        "protein": (
            "ADAMTS10 -- 19p13.2 AR -- ADAMTS10-1103aa -- "
            "A-Disintegrin-And-Metalloproteinase-With-Thrombospondin-Motifs-10-125kDa -- "
            "Signal-Furin-Propeptide-Metalloprotease-Disintegrin-TSP-Cysteine-Rich-Spacer-PLAC -- "
            "Expressed-Ciliary-Body-Zonules-Connective-Tissue-Fibroblasts -- "
            "Cleaves-Aggrecan-Versican-ECM-Proteoglycans-Regulates-Fibrillin-1-Zonule-Assembly -- "
            "OMIM-Gene-608990-Disease-WMS2-608328"
        ),
        "locus": "19p13.2",
        "protein_size": "1103 aa / 125 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "ADAMTS10 is a zinc-dependent metalloprotease expressed in the ciliary body and connective tissue; "
            "cleaves ECM proteoglycans (aggrecan, versican) and regulates fibrillin-1 microfibril assembly; "
            "biallelic LOF → abnormal fibrillin-1 incorporation into zonules + ECM dysregulation → "
            "microspherophakia (abnormally small, spherical lens) + brachydactyly + short stature; "
            "WMS2 (AR-ADAMTS10) is the most common AR form of Weill-Marchesani syndrome; "
            "note: WMS can also be AD (FBN1 heterozygous missense — WMS1/ECTOPIA LENTIS); "
            "AD WMS (FBN1): less brachydactyly; milder lens disease; "
            "heterozygous ADAMTS10 carriers: usually unaffected but may show mild EL features"
        ),
        "disease_category": "Weill-Marchesani syndrome type 2 (WMS2) — fibrillinopathy with microspherophakia + short stature + brachydactyly (inverse Marfan)",
        "disease_pathway": (
            "ADAMTS10 is a zinc metalloprotease expressed in the ciliary body, lens, and connective tissue fibroblasts. "
            "ADAMTS10 interacts with fibrillin-1 microfibrils — the structural backbone of zonular fibres and connective tissue. "
            "PATHOMECHANISM: ADAMTS10 LOF → impaired fibrillin-1 microfibril assembly → "
            "two separate consequences: "
            "(1) OCULAR: abnormal zonular architecture → MICROSPHEROPHAKIA (lens is abnormally SMALL and SPHERICAL) — "
            "the zonules are abnormally short + stiff rather than long + elastic, "
            "causing the lens to assume a spherical shape under equal circumferential tension; "
            "spherical microspherophakic lens → high refractive myopia (steeper lens curvature) → "
            "ANTERIOR SUBLUXATION into anterior chamber → ACUTE PUPILLARY BLOCK GLAUCOMA (medical emergency). "
            "(2) SYSTEMIC: ADAMTS10 LOF in fibroblasts and growth plate cartilage → "
            "impaired proteoglycan/ECM turnover → SHORT STATURE (proportionate, below 3rd percentile) + "
            "BRACHYDACTYLY (short broad fingers/toes) + JOINT STIFFNESS. "
            "INVERSE MARFAN PHENOTYPE: WMS patients are SHORT with SHORT fingers/brachydactyly — "
            "the exact OPPOSITE of Marfan (tall with long arachnodactylous fingers) — "
            "despite both affecting FBN1-related pathways; both conditions can include EL."
        ),
        "pathognomonic": (
            "MICROSPHEROPHAKIA (small spherical lens — measured by keratometry of posterior lens surface on UBM) + "
            "SHORT STATURE (proportionate, height below 3rd centile) + "
            "BRACHYDACTYLY (short, broad distal phalanges — radiograph confirms) = WMS TRIAD PATHOGNOMONIC. "
            "ANTERIOR LENS SUBLUXATION: WMS microspherophakic lens displaced ANTERIORLY — "
            "opposite to FBN1 Marfan (superotemporal) and CBS (inferonasal); "
            "anterior displacement → ACUTE ANGLE CLOSURE GLAUCOMA from pupillary block — "
            "a medical emergency: IOP may exceed 50-60 mmHg; "
            "AVOID MIOTICS (pilocarpine/carbachol): ABSOLUTE CONTRAINDICATION in WMS — "
            "miotics increase pupillary block by constricting iris around already anterior spherical lens. "
            "LASER IRIDOTOMY (LPI): prophylactic to break pupillary block cycle. "
            "JOINT STIFFNESS: reduced range of motion in wrists, elbows, hips — "
            "distinguishes from Marfan hypermobility (WMS has STIFFNESS, Marfan has LAXITY). "
            "WMS2 (ADAMTS10-AR) vs WMS1 (FBN1-AD): "
            "WMS2 has more severe brachydactyly; WMS1 more mild systemic features."
        ),
        "treatment": (
            "PUPILLARY BLOCK GLAUCOMA (EMERGENCY): "
            "LASER IRIDOTOMY (LPI): IMMEDIATE — opens communication between posterior and anterior chambers; "
            "AVOID miotics (pilocarpine): ABSOLUTE CI — worsens block; "
            "AVOID sympathomimetics that dilate pupil (also risky) — cycloplegics may be safer; "
            "IOP lowering: IV acetazolamide + topical beta-blocker + CAI; "
            "PROPHYLACTIC LPI: before acute crisis if microspherophakia detected on UBM; "
            "LENS EXTRACTION: lensectomy if repeated pupillary block despite LPI; "
            "MYOPIA MANAGEMENT: high myopia (due to spherical lens) — spectacles/contact lenses; "
            "aphakic correction post-lensectomy; "
            "SYSTEMIC: no growth hormone for short stature (WMS short stature does not respond); "
            "physiotherapy for joint stiffness; "
            "orthopaedic referral for functional impairment; "
            "GENETIC COUNSELLING: AR 25% recurrence; "
            "CASCADE SCREENING: echo (FBN1-WMS1: mild cardiac) + ophthalmology siblings; "
            "ANNUAL: IOP + lens position on UBM + VA monitoring."
        ),
        "key_features": [
            "Microspherophakia — small, spherical lens (UBM diagnostic)",
            "Short stature + brachydactyly — INVERSE Marfan body habitus",
            "Anterior lens subluxation → acute pupillary block glaucoma (emergency)",
            "AVOID miotics (pilocarpine) — ABSOLUTE CI in WMS",
            "Laser iridotomy prophylactic for pupillary block",
            "Joint stiffness (vs Marfan hypermobility — WMS is stiff)",
            "High myopia from spherical microspherophakic lens",
            "WMS2 (ADAMTS10 AR) more severe brachydactyly vs WMS1 (FBN1 AD)",
        ],
        "key_ddx": (
            "FBN1 (Marfan): TALL + SUPEROTEMPORAL EL + HYPERMOBILE joints — OPPOSITE habitus; aortic dilation; "
            "CBS (HCU): TALL + marfanoid + INFERONASAL EL + thromboembolism + elevated homocysteine; "
            "LTBP2 (microspherophakia): spherical lens + Gulf Arab founder + no brachydactyly; "
            "ADAMTS17 (WMS4): SAME WMS phenotype but milder — genetic confirmation needed; "
            "ADAMTSL4 (IREL): pure ocular EL — no brachydactyly, no short stature, normal lens size; "
            "Acute angle closure glaucoma (primary): elderly, no EL, no microspherophakia; "
            "Pilocarpine-induced pupillary block: iatrogenic in WMS patient — NEVER give miotics to WMS."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital microspherophakia; EL and glaucoma from childhood to adulthood",
        "surgical_urgency": "Urgent — acute pupillary block glaucoma emergency; LPI prophylactic",
        "gene_family": "ADAMTS metalloprotease (zinc-dependent ECM remodelling enzyme)",
        "morphology": "Microspherophakia + short stature + brachydactyly (inverse Marfan)",
    },

    # -- ADAMTS17 -- Weill-Marchesani Syndrome Type 4 ----------------------------------------
    {
        "gene": "ADAMTS17",
        "alt_name": (
            "ADAMTS17 (ADAMTS17-1221aa-15q26.3 / AR -- "
            "WEILL-MARCHESANI-SYNDROME-TYPE-4-WMS4-MILDER-THAN-WMS2 -- "
            "MICROSPHEROPHAKIA+SHORT-STATURE+BRACHYDACTYLY-SIMILAR-TO-WMS2 -- "
            "AVOID-MIOTICS-PUPILLARY-BLOCK-SAME-AS-ADAMTS10-WMS2 -- "
            "JOINT-STIFFNESS-MILD-SKIN-LAXITY-SUBSET)"
        ),
        "protein": (
            "ADAMTS17 -- 15q26.3 AR -- ADAMTS17-1221aa -- "
            "A-Disintegrin-And-Metalloproteinase-With-Thrombospondin-Motifs-17-139kDa -- "
            "Signal-Propeptide-Metalloprotease-Disintegrin-TSP-Cysteine-Rich-Spacer-GON-Domain -- "
            "Expressed-Ciliary-Body-Connective-Tissue-Cartilage -- "
            "Regulates-Fibrillin-1-Microfibril-Assembly-Zonular-Fibres-ECM -- "
            "OMIM-Gene-607511-Disease-WMS4-613195"
        ),
        "locus": "15q26.3",
        "protein_size": "1221 aa / 139 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "ADAMTS17 is a closely related paralogue to ADAMTS10 — same ADAMTS metalloprotease family; "
            "expressed in ciliary body, connective tissue, and cartilage; "
            "regulates fibrillin-1 microfibril assembly in zonular fibres (same pathway as ADAMTS10); "
            "WMS4 (ADAMTS17-AR) is generally MILDER than WMS2 (ADAMTS10-AR): "
            "less severe brachydactyly, less prominent short stature; "
            "microspherophakia and EL are similar severity between WMS2 and WMS4; "
            "some WMS4 patients: mild skin laxity (cutis laxa-like features); "
            "mutations: frameshift, splice, missense in metalloprotease domain; "
            "first described in 2010 (Morales et al.) — small number of reported families"
        ),
        "disease_category": "Weill-Marchesani syndrome type 4 (WMS4) — milder WMS variant with microspherophakia + short stature + brachydactyly (ADAMTS17-associated)",
        "disease_pathway": (
            "ADAMTS17 is a zinc-dependent metalloprotease in the same ADAMTS family as ADAMTS10. "
            "ADAMTS17 regulates fibrillin-1 microfibril assembly in the zonular fibres and connective tissue ECM. "
            "PATHOMECHANISM: ADAMTS17 biallelic LOF → impaired fibrillin-1 incorporation into zonular fibres → "
            "MICROSPHEROPHAKIA (same mechanism as ADAMTS10) — small spherical lens with forward displacement tendency. "
            "SYSTEMIC: ADAMTS17 LOF in cartilage and connective tissue fibroblasts → "
            "impaired proteoglycan/ECM turnover → SHORT STATURE (milder than WMS2) + "
            "BRACHYDACTYLY (less severe than WMS2). "
            "WMS4 is generally MILDER than WMS2 because ADAMTS10 and ADAMTS17 have partially overlapping functions — "
            "ADAMTS10 LOF cannot be compensated, while ADAMTS17 LOF is partially rescued by ADAMTS10 activity. "
            "PUPILLARY BLOCK RISK: same mechanism as WMS2 — microspherophakic lens migrates anteriorly → "
            "pupillary block → acute angle closure glaucoma → MIOTICS ABSOLUTELY CONTRAINDICATED. "
            "MILD SKIN LAXITY: some WMS4 patients show mild cutis laxa-like skin — distinguishing from WMS2."
        ),
        "pathognomonic": (
            "MICROSPHEROPHAKIA + SHORT STATURE + BRACHYDACTYLY (MILDER than WMS2) + OPTIONAL MILD SKIN LAXITY = WMS4. "
            "GENETICALLY CONFIRMED: ADAMTS17 biallelic variants required to distinguish WMS4 from WMS2 (ADAMTS10); "
            "clinical phenotype of WMS4 and WMS2 OVERLAP — genetic panel essential. "
            "MICROSPHEROPHAKIA: UBM confirms small spherical lens; "
            "high myopia from spherical curvature; "
            "AVOID MIOTICS (same absolute CI as WMS2 — pupillary block risk identical). "
            "LASER IRIDOTOMY: prophylactic if microspherophakia confirmed. "
            "SKIN LAXITY: mild, does not require intervention — distinguishes from ADAMTS10-WMS2 (no skin laxity). "
            "JOINT STIFFNESS: milder than WMS2 — some WMS4 patients have near-normal range of motion. "
            "BRACHYDACTYLY: metacarpal and phalangeal shortening — milder than WMS2."
        ),
        "treatment": (
            "Same principles as WMS2 (ADAMTS10): "
            "PUPILLARY BLOCK GLAUCOMA PREVENTION: "
            "LASER IRIDOTOMY (LPI): prophylactic once microspherophakia confirmed — prevents acute crisis; "
            "AVOID MIOTICS (pilocarpine/carbachol/echothiophate): ABSOLUTE CI; "
            "ACUTE CRISIS: IV acetazolamide + topical beta-blocker + CAI + LPI emergency; "
            "LENSECTOMY: if repeated pupillary block despite LPI or VA poor; "
            "MYOPIA CORRECTION: spectacles/contact lenses for high spherical myopia; "
            "SKIN LAXITY (WMS4 subset): mild — emollient; plastic surgery rarely needed; "
            "JOINT STIFFNESS: physiotherapy — milder requirement than WMS2; "
            "GENETIC COUNSELLING: AR 25% recurrence; "
            "DISTINCTION FROM WMS2 (ADAMTS10): requires ADAMTS10 + ADAMTS17 combined panel; "
            "ANNUAL: UBM for lens position + IOP + VA."
        ),
        "key_features": [
            "Microspherophakia — same as WMS2 but genetically ADAMTS17 biallelic",
            "Short stature + brachydactyly — MILDER than WMS2",
            "Mild skin laxity in subset — distinguishes from WMS2",
            "AVOID miotics — same absolute CI as WMS2",
            "Laser iridotomy prophylactic — same indication as WMS2",
            "Genetic panel (ADAMTS10 + ADAMTS17) required to distinguish WMS2 from WMS4",
            "Joint stiffness milder than WMS2",
            "Less prominent systemic features than WMS2 overall",
        ],
        "key_ddx": (
            "ADAMTS10 (WMS2): same phenotype but MORE SEVERE brachydactyly + NO skin laxity — "
            "ADAMTS17 panel confirmation mandatory; "
            "FBN1 (Marfan/WMS1): TALL + EL superotemporal + aortic root — opposite habitus; "
            "LTBP2 (microspherophakia): no brachydactyly, no short stature — Gulf Arab founder; "
            "ADAMTSL4 (IREL): pure ocular — no brachydactyly, no short stature; "
            "CBS (HCU): inferonasal EL + TALL marfanoid + thromboembolism + elevated homocysteine; "
            "Cutis laxa syndromes (ELN, FBLN5): severe skin laxity + systemic — more pronounced than WMS4."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital microspherophakia; EL from childhood to adulthood",
        "surgical_urgency": "Moderate — same pupillary block protocol as WMS2; LPI prophylactic",
        "gene_family": "ADAMTS metalloprotease (ADAMTS17 — ECM remodelling, paralogue of ADAMTS10)",
        "morphology": "Microspherophakia + mild short stature + brachydactyly ± skin laxity (milder WMS2)",
    },

    # -- LTBP2 -- Microspherophakia with Secondary Glaucoma / Gulf Arab -----------------------
    {
        "gene": "LTBP2",
        "alt_name": (
            "LTBP2 (LTBP2-1821aa-14q24.3 / AR -- "
            "MICROSPHEROPHAKIA-SECONDARY-GLAUCOMA-SPHERICAL-SUBLUXATED-LENS-PATHOGNOMONIC -- "
            "GULF-ARAB-FOUNDER-pARG299CYS-PAKISTANI-CONSANGUINEOUS -- "
            "AVOID-MIOTICS-PUPILLARY-BLOCK-LASER-PI-EMERGENCY -- "
            "PCG-PRIMARY-CONGENITAL-GLAUCOMA-LTBP2-VARIANT)"
        ),
        "protein": (
            "LTBP2 -- 14q24.3 AR -- LTBP2-1821aa -- "
            "Latent-TGF-Beta-Binding-Protein-2-LTBP2-200kDa-ECM-Glycoprotein -- "
            "Calcium-Binding-EGF-Like-Repeats-8-Cysteine-TB-Motifs-LTBP-Unique-Regions -- "
            "Binds-Fibrillin-1-Microfibrils-Expressed-Ciliary-Body-Zonule-TM-ECM -- "
            "Regulates-TGF-Beta-Sequestration-Zonular-Fibre-Integrity-Trabecular-Meshwork-ECM -- "
            "OMIM-Gene-602091-Disease-Microspherophakia-251750-PCG-GLC3F-615066"
        ),
        "locus": "14q24.3",
        "protein_size": "1821 aa / 200 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "LTBP2 is a member of the LTBP family — secreted ECM proteins that bind fibrillin-1 microfibrils "
            "and regulate TGF-β latency in the extracellular space; "
            "expressed in ciliary body, zonular fibres, trabecular meshwork, and lens; "
            "GULF ARAB FOUNDER: p.Arg299Cys mutation is the most common pathogenic variant — "
            "found in consanguineous Arab families from Qatar, Saudi Arabia, UAE, Kuwait (founder effect); "
            "Pakistani consanguineous families: also frequent LTBP2 biallelic LOF; "
            "LTBP2 causes TWO distinct phenotypes: "
            "(1) MICROSPHEROPHAKIA ± secondary glaucoma (most common LTBP2 presentation); "
            "(2) PRIMARY CONGENITAL GLAUCOMA (PCG-GLC3F) without microspherophakia (trabecular meshwork LOF); "
            "heterozygous carriers: mildly elevated IOP (incomplete penetrance)"
        ),
        "disease_category": "Microspherophakia with secondary glaucoma / Primary congenital glaucoma (GLC3F) — LTBP2-associated fibrillinopathy",
        "disease_pathway": (
            "LTBP2 (Latent TGF-β Binding Protein 2) is secreted into the extracellular space, where it binds to "
            "fibrillin-1 microfibrils and anchors the large latent TGF-β complex (LLC) to the ECM. "
            "LTBP2 is expressed at high levels in: "
            "(1) CILIARY BODY / ZONULAR FIBRES: maintains structural integrity of zonular ECM; "
            "(2) TRABECULAR MESHWORK: regulates aqueous outflow pathway ECM. "
            "PATHOMECHANISM (MICROSPHEROPHAKIA): "
            "LTBP2 LOF → disrupted fibrillin-1 microfibril assembly in ciliary body → "
            "abnormal zonular architecture → spherical small lens (MICROSPHEROPHAKIA) — "
            "same pathway as ADAMTS10/17 WMS but LTBP2 acts through TGF-β pathway regulation; "
            "microspherophakic lens → anterior migration → ACUTE PUPILLARY BLOCK GLAUCOMA. "
            "PATHOMECHANISM (PCG variant): "
            "LTBP2 LOF in trabecular meshwork → dysregulated TGF-β signalling → impaired TM ECM → "
            "elevated IOP from birth (PCG phenotype — no microspherophakia in some families). "
            "DIGENIC: CYP1B1 + LTBP2 compound heterozygotes → more severe PCG (synergistic)."
        ),
        "pathognomonic": (
            "SPHERICAL SUBLUXATED LENS (microspherophakia) + SECONDARY ANGLE CLOSURE GLAUCOMA FROM PUPILLARY BLOCK "
            "IN GULF ARAB / PAKISTANI CONSANGUINEOUS PATIENT = LTBP2 PATHOGNOMONIC. "
            "UBM (ULTRASOUND BIOMICROSCOPY): confirms microspherophakia — spherical lens with reduced AP diameter; "
            "anterior displacement of spherical lens against iris/pupil → pupillary block; "
            "peripheral anterior synechiae + angle closure on gonioscopy. "
            "AVOID MIOTICS: ABSOLUTE CONTRAINDICATION — pilocarpine constricts iris ONTO spherical lens → "
            "worsens pupillary block → IOP crisis; "
            "LASER PERIPHERAL IRIDOTOMY (LPI): EMERGENCY for acute angle closure; prophylactic once microspherophakia detected. "
            "PCG PHENOTYPE (LTBP2 subset): "
            "buphthalmos + Haab striae + elevated IOP from birth — same as CYP1B1 PCG; "
            "diagnosis: LTBP2 sequencing in PCG panels (especially Gulf Arab/Pakistani). "
            "p.Arg299Cys VARIANT: targeted sequencing in Gulf Arab families before full panel."
        ),
        "treatment": (
            "MICROSPHEROPHAKIA + ACUTE PUPILLARY BLOCK (EMERGENCY): "
            "LASER PERIPHERAL IRIDOTOMY (LPI): IMMEDIATE — relieves pupillary block; "
            "AVOID miotics (pilocarpine): ABSOLUTE CI; "
            "MEDICAL: IV acetazolamide + topical beta-blocker + CAI for IOP crisis; "
            "CYCLOPLEGIA (atropine 1%): may help by moving lens posteriorly (cycloplegic widens ciliary ring); "
            "LENSECTOMY (pars plana): definitive treatment for recurrent block; "
            "aphakic rehabilitation: scleral-fixated IOL / iris-claw IOL; "
            "PCG PHENOTYPE: "
            "goniotomy or trabeculotomy as for CYP1B1 PCG — same angle surgery protocol; "
            "MYOPIA MANAGEMENT: high myopia from spherical microspherophakia — spectacles/contact lenses; "
            "GENETIC COUNSELLING: AR; p.Arg299Cys targeted in Gulf Arab families first; "
            "ANNUAL: IOP + lens position UBM + VA; "
            "FAMILY SCREENING: consanguineous families — targeted sequencing."
        ),
        "key_features": [
            "Microspherophakia — spherical lens with anterior subluxation",
            "Secondary angle closure glaucoma from pupillary block",
            "AVOID miotics (pilocarpine) — ABSOLUTE CI (worsens block)",
            "Laser iridotomy EMERGENCY for acute angle closure",
            "Gulf Arab founder: p.Arg299Cys (Qatar, Saudi, UAE, Kuwait)",
            "Pakistani consanguineous families also common",
            "PCG variant (no microspherophakia) — trabecular meshwork LOF",
            "LTBP2 regulates fibrillin-1 + TGF-β in zonule + trabecular meshwork",
        ],
        "key_ddx": (
            "ADAMTS10 (WMS2): same microspherophakia + brachydactyly + short stature — no Gulf Arab founder; "
            "ADAMTS17 (WMS4): same WMS phenotype — milder systemic; "
            "CYP1B1 (PCG): PCG phenotype similar — CYP1B1 more common worldwide; LTBP2 in consanguineous Arab/Pakistani; "
            "FBN1 (Marfan): superotemporal EL — NOT spherical lens; tall not short; "
            "Primary angle closure (POAG/PACD): elderly, no microspherophakia, no EL; "
            "Pilocarpine miosis in any EL patient: always check for microspherophakia before prescribing miotics."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital microspherophakia; glaucoma from childhood to adulthood",
        "surgical_urgency": "Urgent — acute pupillary block emergency; LPI prophylactic",
        "gene_family": "Latent TGF-β Binding Protein (LTBP2 — fibrillin-bound ECM TGF-β regulator)",
        "morphology": "Spherical lens (microspherophakia) with anterior subluxation + secondary glaucoma",
    },

    # -- SUOX -- Isolated Sulfite Oxidase Deficiency -------------------------------------------
    {
        "gene": "SUOX",
        "alt_name": (
            "SUOX (SUOX-545aa-12q13.2 / AR -- "
            "ISOLATED-SULFITE-OXIDASE-DEFICIENCY-ECTOPIA-LENTIS+NEONATAL-SEIZURES-PATHOGNOMONIC -- "
            "URINE-SULFITE-DIPSTICK-POSITIVE-RAPID-BEDSIDE-TEST -- "
            "KEY-DDx-HOMOCYSTINURIA-NO-ELEVATED-PLASMA-HOMOCYSTEINE -- "
            "NO-TREATMENT-SEVERE-NEONATAL-NEURO-DEVASTATION)"
        ),
        "protein": (
            "SUOX -- 12q13.2 AR -- SUOX-545aa -- "
            "Sulfite-Oxidase-60kDa-Molybdopterin-Containing-Mitochondrial-Intermembrane-Space -- "
            "N-Terminal-Heme-Cytochrome-B5-Domain-Molybdopterin-Domain-Dimerisation-Domain -- "
            "Oxidises-Sulfite-To-Sulfate-Final-Step-Sulfur-Amino-Acid-Catabolism -- "
            "Receives-Molybdenum-Cofactor-MoCo-For-Catalysis-Sulphur-Oxidation -- "
            "OMIM-Gene-606887-Disease-SUOX-Deficiency-272300"
        ),
        "locus": "12q13.2",
        "protein_size": "545 aa / 60 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "isolated sulfite oxidase deficiency (ISOD): SUOX gene specifically mutated — "
            "molybdenum cofactor (MoCo) is intact; "
            "to be distinguished from MOLYBDENUM COFACTOR DEFICIENCY (MoCoD): MOCS1/MOCS2/GPHN mutations → "
            "MoCo absent → SUOX + xanthine oxidase + aldehyde oxidase ALL deficient (combined MoCoD); "
            "clinical presentation IDENTICAL in ISOD and MoCoD type A (early onset); "
            "KEY DISTINGUISHING TEST: urine xanthine/uric acid — "
            "ISOD: normal xanthine (only SUOX absent → normal XO) + elevated sulfite; "
            "MoCoD: elevated xanthine + absent uric acid (XO also absent) + elevated sulfite; "
            "mutations: missense, frameshift, splice — diverse; consanguineous families most common; "
            "prenatal diagnosis: molecular or urine sulfite testing (elevated in amniocytes)"
        ),
        "disease_category": "Isolated sulfite oxidase deficiency (ISOD) — ectopia lentis + severe neonatal seizures + neurodegeneration",
        "disease_pathway": (
            "SUOX (sulfite oxidase) is a molybdenum-dependent enzyme in the mitochondrial intermembrane space "
            "that catalyses the final step of sulfur amino acid catabolism: "
            "SULFITE → SULFATE (requires MoCo as cofactor). "
            "CBS and methionine catabolism generate sulfite as an intermediate; without SUOX, sulfite accumulates. "
            "PATHOMECHANISM: SUOX LOF → sulfite + S-sulfocysteine accumulate → "
            "(1) OCULAR: ECTOPIA LENTIS — sulfite disrupts disulfide bonds in fibrillin-1 zonular fibres "
            "(same mechanism as homocysteine in CBS deficiency) → "
            "bilateral lens dislocation (direction variable — inferonasal common but less directionally consistent than CBS); "
            "(2) NEUROLOGICAL (DEVASTATING): sulfite + S-sulfocysteine are potent NMDA receptor AGONISTS → "
            "neonatal refractory seizures + severe encephalopathy + cystic leukomalacia + "
            "developmental arrest → most patients have profound intellectual disability + spastic quadriplegia; "
            "SULFOCYSTEINE: S-sulfocysteine (cysteine + sulfite adduct) is the dominant urinary biomarker. "
            "NO EFFECTIVE TREATMENT: unlike homocystinuria (B6-responsive), ISOD has no proven treatment — "
            "low sulfur diet experimental but unproven; "
            "prognosis: severe — most patients die in infancy or survive with profound neurological impairment."
        ),
        "pathognomonic": (
            "ECTOPIA LENTIS + NEONATAL REFRACTORY SEIZURES (onset 1st week of life) + "
            "URINE SULFITE DIPSTICK POSITIVE (freshly voided urine — must test FRESH as sulfite auto-oxidises) = "
            "ISOLATED SULFITE OXIDASE DEFICIENCY PATHOGNOMONIC. "
            "URINE SULFITE DIPSTICK: specific sulfite test strips (e.g. Merckoquant sulfite test) — "
            "MUST USE FRESH URINE (sulfite oxidises to sulfate spontaneously — delay gives false negative); "
            "KEY DDx FROM CBS HOMOCYSTINURIA: "
            "SUOX DEFICIENCY: positive urine sulfite + NORMAL plasma homocysteine + neonatal onset seizures; "
            "CBS HCU: negative urine sulfite + ELEVATED plasma homocysteine + no neonatal seizures; "
            "KEY DDx FROM MoCoD (MOCS1/MOCS2): "
            "ISOD (SUOX): urine xanthine NORMAL + uric acid NORMAL + sulfite POSITIVE; "
            "MoCoD: urine xanthine ELEVATED + uric acid ABSENT (XO absent) + sulfite POSITIVE; "
            "BRAIN MRI: cystic leukomalacia + cortical necrosis — severe in most neonatal cases; "
            "EEG: burst suppression or multifocal epileptiform discharges — neonatal; "
            "METABOLIC WORK-UP: plasma amino acids (elevated taurine, cystine low), urine organic acids."
        ),
        "treatment": (
            "NO PROVEN EFFECTIVE TREATMENT (unlike homocystinuria which responds to B6): "
            "LOW SULFUR (LOW METHIONINE / LOW CYSTEINE) DIET: reduces sulfite substrate — "
            "theoretically beneficial but evidence weak; practically difficult in neonates; "
            "PYRIDOXINE: no benefit in SUOX deficiency (unlike CBS HCU — B6 test negative); "
            "CYCLIC PYRANOPTERIN MONOPHOSPHATE (cPMP): effective for MoCoD type A (MOCS1) — "
            "NOT effective for ISOD (SUOX intact pathway, MoCo normal); "
            "SEIZURE MANAGEMENT: benzodiazepines (clonazepam), levetiracetam, phenobarbitone — "
            "typically refractory to standard AEDs; "
            "PALLIATIVE/SUPPORTIVE: gastrostomy for feeding; physiotherapy; "
            "most patients die in infancy or survive with profound disability; "
            "GENETIC COUNSELLING: AR 25% recurrence; SUOX vs MoCoD distinction critical for recurrence risk; "
            "prenatal diagnosis: molecular testing (SUOX biallelic) or fetal urine sulfite; "
            "NEONATAL METABOLIC SCREEN: NOT currently part of standard NBS in most countries — "
            "clinical diagnosis on symptom onset; metabolic testing on suspicion."
        ),
        "key_features": [
            "Ectopia lentis (bilateral) + neonatal refractory seizures — combined presentation pathognomonic",
            "Urine sulfite dipstick POSITIVE (FRESH urine only — auto-oxidises) — rapid bedside test",
            "Normal plasma homocysteine — KEY DDx from CBS homocystinuria",
            "Normal urine xanthine + uric acid — KEY DDx from MoCoD (MOCS1/MOCS2)",
            "Severe neurological devastation — cystic leukomalacia, spastic quadriplegia",
            "No proven effective treatment — no B6 responsiveness unlike CBS HCU",
            "SUOX = sulfite oxidase — molybdenum-dependent, final sulfur catabolism step",
            "Brain MRI: cystic leukomalacia + cortical necrosis in severe neonatal form",
        ],
        "key_ddx": (
            "CBS (HCU): ELEVATED plasma homocysteine + negative urine sulfite + LATE onset EL (2-10yr) + "
            "B6-responsive subset — NO neonatal seizures; thromboembolism; "
            "MOCS1/MOCS2 (MoCoD): IDENTICAL SUOX-like phenotype BUT elevated xanthine + absent uric acid — "
            "cPMP therapy available for MoCoD type A (MOCS1); ISOD has no MoCoD therapy; "
            "GPHN (MoCoD type C): same as MoCoD; GPHN also involved in glycine receptor clustering → "
            "hyperekplexia features possible; "
            "FBN1 (Marfan): EL only — no seizures, no sulfite; "
            "Pyridoxine-dependent epilepsy (ALDH7A1): neonatal seizures + responds to B6 — "
            "no EL, plasma pipecolic acid elevated; "
            "Neonatal seizures from other causes: hypoxic-ischemic, hypoglycaemia, Na abnormality — "
            "urine sulfite + EL distinguishes SUOX."
        ),
        "systemic_involvement": True,
        "onset_age": "Neonatal (first week of life — seizures); EL present at birth or early infancy",
        "surgical_urgency": "Urgent — neonatal seizure management; EL surgery deferred due to neurological severity",
        "gene_family": "Molybdenum-dependent oxidase (SUOX — sulfite oxidase, mitochondrial IMM space)",
        "morphology": "Bilateral ectopia lentis + severe neonatal encephalopathy + cystic leukomalacia",
    },

    # -- FBN2 -- Congenital Contractural Arachnodactyly / Beals-Hecht Syndrome ----------------
    {
        "gene": "FBN2",
        "alt_name": (
            "FBN2 (FBN2-2832aa-5q23.3 / AD -- "
            "CONGENITAL-CONTRACTURAL-ARACHNODACTYLY-CCA-BEALS-HECHT-SYNDROME -- "
            "CRUMPLED-EAR-PINNA+CONGENITAL-JOINT-CONTRACTURES-PATHOGNOMONIC -- "
            "KEY-DDx-MARFAN-FBN1-NO-AORTIC-ROOT-DILATION-IN-CCA -- "
            "OCCASIONAL-ECTOPIA-LENTIS-CARDIOVASCULAR-RARE)"
        ),
        "protein": (
            "FBN2 -- 5q23.3 AD -- FBN2-2832aa -- "
            "Fibrillin-2-330kDa-Extracellular-Matrix-Glycoprotein -- "
            "47-EGF-Like-Domains-7-TB-Calcium-Binding-EGF-Like-Domains-RGD-Integrin-Binding -- "
            "Structural-Scaffold-Fetal-Connective-Tissue-Elastic-Fibres-Joints-Zonules -- "
            "Predominantly-Expressed-Fetal-Embryonic-Connective-Tissue-Less-Adult-vs-FBN1 -- "
            "OMIM-Gene-612570-Disease-CCA-121050"
        ),
        "locus": "5q23.3",
        "protein_size": "2832 aa / 330 kDa",
        "inheritance": (
            "AD — haploinsufficiency; FBN2 is the fetal homologue of FBN1 — "
            "FBN2 microfibrils are abundant in fetal connective tissue and are gradually replaced by FBN1 microfibrils postnatally; "
            "FBN2 haploinsufficiency → defective fetal fibrillin-2 microfibril assembly → "
            "CONGENITAL JOINT CONTRACTURES (develop in utero) + crumpled ear pinna + arachnodactyly; "
            "FBN2 is expressed in the lens zonules (at lower levels than FBN1) — "
            "occasional ectopia lentis reported in CCA (less common than Marfan — ~20% of CCA); "
            "cardiovascular: FBN2 variants rarely cause aortic root dilation (unlike FBN1 Marfan); "
            "exceptions: some FBN2 mutations may cause mild MVP or aortic root dilation in a minority; "
            "de novo: ~50% of CCA cases; familial AD with variable expressivity (contractures may resolve with age)"
        ),
        "disease_category": "Congenital contractural arachnodactyly (CCA / Beals-Hecht syndrome) — fibrillin-2 haploinsufficiency with congenital joint contractures + crumpled ear + occasional ectopia lentis",
        "disease_pathway": (
            "FBN2 (Fibrillin-2) is the fetal-predominant fibrillin paralogue expressed in developing connective tissues "
            "during embryogenesis. FBN2 microfibrils provide scaffold for elastic fibre assembly in joints, cartilage, "
            "and connective tissue sheaths that develop in utero. "
            "PATHOMECHANISM: FBN2 haploinsufficiency → disrupted microfibril scaffold in FETAL connective tissue → "
            "(1) JOINT CONTRACTURES (CONGENITAL): absent/reduced elastic fibre scaffold in joint capsules and tendons → "
            "joints fixed in flexed position at birth (CONGENITAL — present from birth, unlike Marfan); "
            "most severely affected: fingers (camptodactyly — permanently flexed), wrists, elbows, hips, knees, feet; "
            "contractures may IMPROVE with physiotherapy over years (FBN1 replaces FBN2 postnatally → partial compensation). "
            "(2) CRUMPLED EAR PINNA: FBN2 expressed in auricular cartilage → "
            "dysplastic cartilage → crumpled, folded, or 'cauliflower' pinna appearance at birth — PATHOGNOMONIC FOR CCA. "
            "(3) ARACHNODACTYLY: tall marfanoid build + long fingers/toes — similar to FBN1 Marfan but WITHOUT aortic root dilation. "
            "(4) ECTOPIA LENTIS (occasional ~20%): FBN2 expressed in lens zonules at lower level — "
            "EL less common, less severe than FBN1 Marfan; superotemporal or superior direction."
        ),
        "pathognomonic": (
            "CRUMPLED PINNA (dysplastic, folded auricular cartilage — congenital at birth) + "
            "CONGENITAL JOINT CONTRACTURES (camptodactyly, elbow/knee/hip contractures present at birth) = "
            "CCA/BEALS-HECHT PATHOGNOMONIC. "
            "CRUMPLED EAR: key distinguishing feature from FBN1 Marfan — "
            "FBN1 Marfan does NOT have crumpled ear; FBN2 CCA DOES (found in ~50-75% of CCA patients). "
            "NO AORTIC ROOT DILATION: KEY DDx from FBN1 Marfan — "
            "echocardiogram is NORMAL (or only trivially abnormal) in CCA; "
            "aortic root Z-score normal in most CCA — no prophylactic beta-blocker or ARB required. "
            "CONTRACTURES IMPROVE: congenital contractures often partially resolve with physiotherapy "
            "in the first years of life (FBN1 replaces FBN2 postnatally). "
            "ECTOPIA LENTIS (~20%): superotemporal (same direction as FBN1 Marfan) but less severe; "
            "SCOLIOSIS: same as FBN1 Marfan — kyphoscoliosis common (40-60%). "
            "KEY CLINICAL TEACHING: CCA = CONTRACTURAL (stiff joints) vs FBN1 MARFAN = LOOSE joints (hypermobile) — "
            "despite both causing tall marfanoid habitus."
        ),
        "treatment": (
            "JOINT CONTRACTURES: "
            "PHYSIOTHERAPY from birth — passive stretching of contracted joints; "
            "serial casting for foot contractures (clubfoot); "
            "contractures often partially resolve — functional improvement with therapy; "
            "orthopaedic surgery for severe persistent contractures (tendon release, splinting); "
            "EAR: no treatment needed for crumpled pinna; cosmetic surgery if requested (adult); "
            "ECTOPIA LENTIS (if present — ~20%): "
            "ophthalmology surveillance + refraction; "
            "lensectomy if VA poor or pupillary block risk (same as FBN1-Marfan EL protocol); "
            "CARDIAC MONITORING: echo at diagnosis — aortic root normal in most; "
            "if aortic root Z>2 found: treat as FBN1 Marfan (beta-blocker + ARB, annual echo); "
            "SCOLIOSIS: physiotherapy + bracing + orthopaedic surgical fusion for severe curves; "
            "GENETIC COUNSELLING: AD 50% recurrence; FBN2 panel testing; "
            "distinguish CCA from FBN1 Marfan clinically (crumpled ear + contractures + no aortic dilation = CCA); "
            "ANNUAL: ophthalmology (EL surveillance) + echo (aortic monitoring) + scoliosis."
        ),
        "key_features": [
            "Crumpled pinna (dysplastic auricular cartilage) — PATHOGNOMONIC for CCA, absent in Marfan",
            "Congenital joint contractures (camptodactyly, elbows, hips) — present at birth",
            "NO aortic root dilation — KEY DDx from FBN1 Marfan (no beta-blocker/ARB needed)",
            "Tall marfanoid habitus + arachnodactyly (similar to FBN1 but no aortic risk)",
            "Ectopia lentis ~20% (less common than FBN1 Marfan ~60-80%)",
            "Contractures IMPROVE with physiotherapy over years (FBN1 replaces FBN2 postnatally)",
            "FBN2 is fetal fibrillin paralogue — predominantly expressed in utero",
            "CCA STIFF joints vs FBN1-Marfan LOOSE (hypermobile) joints — key clinical teaching",
        ],
        "key_ddx": (
            "FBN1 (Marfan): CRUMPLED EAR ABSENT + LOOSE joints (hypermobile) + AORTIC ROOT DILATION — "
            "all opposite to CCA/FBN2; FBN1 has no congenital contractures; "
            "HCCS (MIDAS): contractures in some + ear anomalies — but LINEAR SKIN DEFECTS + microphthalmia + XLD females; "
            "Arthrogryposis multiplex congenita (AMC): multiple congenital contractures — diverse etiologies, "
            "no crumpled ear, no marfanoid; "
            "Stiff skin syndrome (FBN1 p.Cys1679): scleroderma-like stiff skin + contractures — "
            "different FBN1 variant class; "
            "Ehlers-Danlos kyphoscoliotic (PLOD1): kyphoscoliosis + joint laxity — NOT contractures; "
            "ADAMTS10/17 (WMS): SHORT stature + brachydactyly + STIFF joints — but no crumpled ear, no marfanoid."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital (contractures + crumpled ear at birth); EL if present from childhood",
        "surgical_urgency": "Low cardiac urgency (no aortic dilation); physiotherapy urgent for contractures",
        "gene_family": "Fibrillin (FBN2 — fetal fibrillin-2, EGF-like domain ECM protein)",
        "morphology": "Crumpled pinna + congenital joint contractures + occasional EL (no aortic dilation)",
    },
]


def _make_cohort(entry, seed):
    rng = random.Random(seed)
    gene = entry["gene"]
    patients = []
    for i in range(40):
        # Ectopia lentis present
        if gene == "FBN1":
            el_present = rng.random() < 0.72  # 60-80%
        elif gene == "CBS":
            el_present = rng.random() < 0.78  # most HCU patients develop EL by adulthood
        elif gene == "ADAMTSL4":
            el_present = True  # defining feature
        elif gene == "ADAMTS10":
            el_present = rng.random() < 0.85  # microspherophakia → EL in most
        elif gene == "ADAMTS17":
            el_present = rng.random() < 0.80
        elif gene == "LTBP2":
            el_present = rng.random() < 0.88  # microspherophakia nearly always
        elif gene == "SUOX":
            el_present = rng.random() < 0.70  # present but may not be first presenting feature
        else:  # FBN2
            el_present = rng.random() < 0.22  # ~20% of CCA have EL

        # Microspherophakia (WMS/LTBP2 only)
        microspherophakia = False
        if gene in ("ADAMTS10", "ADAMTS17"):
            microspherophakia = rng.random() < 0.90
        elif gene == "LTBP2":
            microspherophakia = rng.random() < 0.92

        # Glaucoma / pupillary block
        if gene in ("ADAMTS10", "ADAMTS17"):
            glaucoma = microspherophakia and rng.random() < 0.55
        elif gene == "LTBP2":
            glaucoma = rng.random() < 0.65  # high rate (microspherophakia + PCG variant)
        elif gene == "FBN1":
            glaucoma = rng.random() < 0.08  # uncommon direct glaucoma
        elif gene == "CBS":
            glaucoma = rng.random() < 0.05
        elif gene == "ADAMTSL4":
            glaucoma = rng.random() < 0.12  # pupillary block if lens migrates
        elif gene == "SUOX":
            glaucoma = rng.random() < 0.08
        else:  # FBN2
            glaucoma = rng.random() < 0.05

        # Systemic / aortic involvement
        if gene == "FBN1":
            aortic = rng.random() < 0.90  # near-universal aortic root dilation
        elif gene == "FBN2":
            aortic = rng.random() < 0.08  # rare in CCA
        else:
            aortic = False

        # Thromboembolism (CBS only)
        thromboembolism = False
        if gene == "CBS":
            thromboembolism = rng.random() < 0.35  # ~30-40% lifetime event untreated

        # B6-responsive (CBS only)
        b6_responsive = False
        if gene == "CBS":
            b6_responsive = rng.random() < 0.50

        # Seizures (SUOX only)
        neonatal_seizures = False
        if gene == "SUOX":
            neonatal_seizures = rng.random() < 0.95

        # Lens surgery performed
        if gene in ("ADAMTS10", "ADAMTS17", "LTBP2") and glaucoma:
            surgery = rng.random() < 0.72
        elif gene == "FBN1" and el_present:
            surgery = rng.random() < 0.45
        elif gene == "CBS" and el_present:
            surgery = rng.random() < 0.52
        elif gene == "ADAMTSL4":
            surgery = el_present and rng.random() < 0.38
        elif gene == "SUOX":
            surgery = el_present and rng.random() < 0.20  # neurological priority
        else:  # FBN2
            surgery = el_present and rng.random() < 0.35

        # VA poor
        if gene == "SUOX":
            va_poor = rng.random() < 0.95  # neurological devastation
        elif gene in ("ADAMTS10", "ADAMTS17", "LTBP2") and glaucoma:
            va_poor = rng.random() < 0.55
        elif gene in ("FBN1", "CBS") and el_present and not surgery:
            va_poor = rng.random() < 0.40
        elif gene == "ADAMTSL4" and el_present:
            va_poor = rng.random() < 0.30
        elif gene == "FBN2":
            va_poor = rng.random() < 0.12
        else:
            va_poor = rng.random() < 0.08

        # Consanguineous
        if gene in ("CBS", "ADAMTSL4", "ADAMTS10", "ADAMTS17", "SUOX"):
            consanguineous = rng.random() < 0.42
        elif gene == "LTBP2":
            consanguineous = rng.random() < 0.80  # Gulf Arab/Pakistani families
        else:  # FBN1, FBN2 are AD
            consanguineous = rng.random() < 0.06

        # Direction of EL
        if gene == "FBN1":
            el_direction = "superotemporal" if el_present else "none"
        elif gene == "CBS":
            el_direction = "inferonasal" if el_present else "none"
        elif gene in ("ADAMTS10", "ADAMTS17", "LTBP2"):
            el_direction = "anterior-spherical" if el_present else "none"
        elif gene in ("ADAMTSL4",):
            el_direction = rng.choice(["superotemporal", "superior", "nasal"]) if el_present else "none"
        elif gene == "SUOX":
            el_direction = "inferonasal" if el_present else "none"
        else:  # FBN2
            el_direction = "superotemporal" if el_present else "none"

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "el_present": el_present,
            "el_direction": el_direction,
            "microspherophakia": microspherophakia,
            "glaucoma": glaucoma,
            "aortic_involvement": aortic,
            "thromboembolism": thromboembolism,
            "b6_responsive": b6_responsive,
            "neonatal_seizures": neonatal_seizures,
            "surgery_performed": surgery,
            "va_poor": va_poor,
            "consanguineous": consanguineous,
            "inheritance": entry["inheritance"].split(";")[0].strip(),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(EL_GENES):
        all_patients.extend(_make_cohort(entry, SEED_BASE + idx))

    total = len(all_patients)
    el_count = sum(1 for p in all_patients if p["el_present"])
    micro_count = sum(1 for p in all_patients if p["microspherophakia"])
    glaucoma_count = sum(1 for p in all_patients if p["glaucoma"])
    aortic_count = sum(1 for p in all_patients if p["aortic_involvement"])
    thrombosis_count = sum(1 for p in all_patients if p["thromboembolism"])
    surgery_count = sum(1 for p in all_patients if p["surgery_performed"])
    va_poor_count = sum(1 for p in all_patients if p["va_poor"])
    consanguineous_count = sum(1 for p in all_patients if p["consanguineous"])

    gene_summary = {}
    for idx, entry in enumerate(EL_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "morphology": entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age": entry["onset_age"],
            "surgical_urgency": entry["surgical_urgency"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "el_pct": round(100 * sum(1 for p in cohort if p["el_present"]) / len(cohort), 1),
            "microspherophakia_pct": round(100 * sum(1 for p in cohort if p["microspherophakia"]) / len(cohort), 1),
            "glaucoma_pct": round(100 * sum(1 for p in cohort if p["glaucoma"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgery_performed"]) / len(cohort), 1),
            "va_poor_pct": round(100 * sum(1 for p in cohort if p["va_poor"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Ectopia-Lentis-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Ectopia Lentis Reference -- FBN1/CBS/ADAMTSL4/ADAMTS10/ADAMTS17/LTBP2/SUOX/FBN2",
        "genes_covered": [e["gene"] for e in EL_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "el_present_pct": round(100 * el_count / total, 1),
            "microspherophakia_pct": round(100 * micro_count / total, 1),
            "glaucoma_pct": round(100 * glaucoma_count / total, 1),
            "aortic_involvement_pct": round(100 * aortic_count / total, 1),
            "thromboembolism_pct": round(100 * thrombosis_count / total, 1),
            "surgery_performed_pct": round(100 * surgery_count / total, 1),
            "va_worse_than_6_18_pct": round(100 * va_poor_count / total, 1),
            "consanguineous_family_pct": round(100 * consanguineous_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(EL_GENES):
        cohort = _make_cohort(entry, SEED_BASE + idx)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "morphology": entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age": entry["onset_age"],
            "surgical_urgency": entry["surgical_urgency"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "el_pct": round(100 * sum(1 for p in cohort if p["el_present"]) / len(cohort), 1),
            "microspherophakia_pct": round(100 * sum(1 for p in cohort if p["microspherophakia"]) / len(cohort), 1),
            "glaucoma_pct": round(100 * sum(1 for p in cohort if p["glaucoma"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgery_performed"]) / len(cohort), 1),
            "va_poor_pct": round(100 * sum(1 for p in cohort if p["va_poor"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:500],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "morphology": entry["morphology"],
                "systemic_involvement": entry["systemic_involvement"],
                "onset_age": entry["onset_age"],
                "surgical_urgency": entry["surgical_urgency"],
                "gene_family": entry["gene_family"],
            }
            for entry in EL_GENES
        },
        "el_glossary": {
            "Ectopia Lentis — Direction as Diagnostic Clue": (
                "The DIRECTION of lens displacement is clinically diagnostic and should be recorded at every slit-lamp examination. "
                "SUPEROTEMPORAL (up and outward): FBN1 Marfan syndrome (and FBN2 CCA occasionally) — "
                "superior zonules fail first in fibrillin haploinsufficiency; "
                "INFERONASAL (down and inward): CBS homocystinuria — "
                "homocysteine disrupts inferior zonular fibrillin-1 disulfide bonds; "
                "ANTERIOR (spherical lens migrating forward): WMS (ADAMTS10/17) and microspherophakia (LTBP2) — "
                "small spherical lens moves anteriorly into pupillary plane — PUPILLARY BLOCK EMERGENCY; "
                "VARIABLE / HORIZONTAL / SUPERIOR: ADAMTSL4 isolated ectopia lentis — "
                "less predictable direction; "
                "CLINICAL TOOL: direction alone can differentiate FBN1 from CBS before genetic testing; "
                "NEXT STEP AFTER DIRECTION: plasma total homocysteine (to exclude CBS) + echo (to exclude FBN1 aortic root) + UBM (to detect microspherophakia)."
            ),
            "Microspherophakia — Miotics ABSOLUTELY Contraindicated": (
                "MICROSPHEROPHAKIA = abnormally small, spherical lens — seen in WMS (ADAMTS10/17) and LTBP2-associated microspherophakia. "
                "NORMAL LENS: biconvex, 9-10mm diameter, elongated equatorially. "
                "MICROSPHEROPHAKIC LENS: <7mm diameter, spherical, small equatorial diameter. "
                "CLINICAL CONSEQUENCE: spherical lens under equal circumferential zonular tension floats anteriorly — "
                "migrates toward pupil → PUPILLARY BLOCK (iris blocks posterior-to-anterior chamber communication) → "
                "ACUTE ANGLE CLOSURE GLAUCOMA (IOP may exceed 60 mmHg — ophthalmic emergency). "
                "UBM (ULTRASOUND BIOMICROSCOPY) DIAGNOSTIC: measures lens diameters and anterior position. "
                "MIOTICS (PILOCARPINE / CARBACHOL / ECHOTHIOPHATE): ABSOLUTELY CONTRAINDICATED — "
                "miotic constricts iris sphincter → iris grips spherical lens tighter → worsens pupillary block → "
                "IOP crisis intensifies; "
                "STANDARD TREATMENT: "
                "1. Laser peripheral iridotomy (LPI) — opens posterior-to-anterior flow, breaks block; "
                "2. Cycloplegics (atropine 1%) — relaxes ciliary ring → lens moves posteriorly; "
                "3. Lensectomy if recurrent — definitive."
            ),
            "Marfan vs Homocystinuria — Critical Clinical DDx Table": (
                "TWO CONDITIONS THAT BOTH CAUSE TALL MARFANOID HABITUS + ECTOPIA LENTIS: "
                "FBN1 MARFAN: superotemporal EL + aortic root dilation + JOINT HYPERMOBILITY + "
                "normal plasma homocysteine + NO thromboembolism + AD (50% family risk); "
                "CBS HOMOCYSTINURIA: INFERONASAL EL + NO aortic root dilation + JOINT STIFFNESS (rare laxity) + "
                "ELEVATED plasma homocysteine (>100 µmol/L untreated) + THROMBOEMBOLISM risk + "
                "AR (25% recurrence) + B6-responsiveness test mandatory; "
                "BODY HABITUS DIFFERENCE: Marfan HYPERMOBILE joints; HCU joints NORMAL or STIFF; "
                "ANAESTHESIA RISK: HCU = HIGH THROMBOEMBOLISM risk (LMWH + hydration mandatory pre-op); "
                "Marfan = aortic dissection risk (echo + careful BP monitoring); "
                "RAPID BEDSIDE DIFFERENTIATION: plasma tHcy + urine sodium nitroprusside test; "
                "both confirmed by molecular panel."
            ),
            "Weill-Marchesani Syndrome — Inverse Marfan and Emergency Protocol": (
                "WEILL-MARCHESANI SYNDROME (WMS): the 'INVERSE MARFAN' — short, brachydactylous patient with lens disease. "
                "PHENOTYPE COMPARISON: "
                "FBN1 Marfan: TALL + long fingers + hypermobile + EL superotemporal + aortic root; "
                "WMS (ADAMTS10/17/LTBP2): SHORT + short/broad fingers (brachydactyly) + stiff joints + "
                "MICROSPHEROPHAKIA with ANTERIOR subluxation. "
                "IMPORTANT: WMS can also be caused by FBN1 heterozygous missense variants (WMS1/AD) — "
                "milder systemic features than ADAMTS10-WMS2. "
                "ACUTE ANGLE CLOSURE EMERGENCY in WMS: "
                "1. Identify microspherophakia (UBM if cornea hazy); "
                "2. DO NOT GIVE MIOTICS; "
                "3. Laser peripheral iridotomy IMMEDIATELY; "
                "4. Acetazolamide IV + topical beta-blocker + CAI; "
                "5. Cycloplegia (atropine 1%) to push lens posteriorly; "
                "PROPHYLAXIS: all WMS patients with confirmed microspherophakia should receive PROPHYLACTIC LPI."
            ),
            "Sulfite Oxidase Deficiency — Fresh Urine Dipstick Protocol": (
                "SULFITE OXIDASE DEFICIENCY (SUOX): neonatal EL + seizures + sulfite-positive urine — "
                "a rare but important differential in any neonate with ectopia lentis + seizures. "
                "KEY TEST: URINE SULFITE DIPSTICK (specific sulfite test strip, e.g. Merckoquant). "
                "CRITICAL RULE: MUST USE FRESHLY VOIDED URINE (within 1-2 minutes of collection) — "
                "sulfite SPONTANEOUSLY OXIDISES to sulfate at room temperature → "
                "delay of 10-15 minutes can give false-negative result. "
                "PROTOCOL: void into container → dip strip IMMEDIATELY → read at 30 seconds; "
                "POSITIVE = purple-violet colour change (scale 0-80 mg/L or higher); "
                "NEGATIVE (falsely) if urine stored or refrigerated before testing — REPEAT WITH FRESH SAMPLE. "
                "DDx SUOX vs MoCoD: "
                "SUOX deficiency: sulfite POSITIVE + xanthine NORMAL + uric acid NORMAL; "
                "MoCoD (MOCS1/MOCS2/GPHN): sulfite POSITIVE + xanthine ELEVATED + uric acid ABSENT/LOW. "
                "DDx SUOX vs CBS HCU: "
                "SUOX: sulfite POSITIVE + homocysteine NORMAL; "
                "CBS HCU: sulfite NEGATIVE + homocysteine ELEVATED."
            ),
        },
    }
