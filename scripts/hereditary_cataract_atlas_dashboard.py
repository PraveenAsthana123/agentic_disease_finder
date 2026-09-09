#!/usr/bin/env python3
"""Hereditary-Cataract-Atlas — Complete 8-Gene Hereditary Cataract Atlas
(CRYAA · CRYAB · GJA8 · GJA3 · MIP · EPHA2 · NHS · FYCO1).

CRYAA   (αA-Crystallin / HspB4; 173 aa; ~20 kDa; 21q22.3; AD/AR;
          Cataract-9 — MOST COMMON crystallin cataract gene worldwide;
          Posterior subcapsular or zonular-lamellar; R116C most common AD variant;
          AR biallelic: congenital cataract + microphthalmia + iris coloboma;
          seed SEED_BASE+0).
CRYAB   (αB-Crystallin / HspB5; 175 aa; ~20 kDa; 11q23.1; AD/AR;
          Cataract-16 + Myofibrillar Myopathy/Dilated Cardiomyopathy;
          R120G pathognomonic for desmin-related cardiomyopathy with cataract;
          FIBRILLAR PROTEIN AGGREGATES IN CARDIAC/SKELETAL MUSCLE PATHOGNOMONIC;
          seed SEED_BASE+1).
GJA8    (Connexin 50 / Cx50; 440 aa; ~50 kDa; 1q21.1; AD;
          Cataract-1 — NUCLEAR PULVERULENT / TOTAL NUCLEAR most common morphology;
          Lens fiber cell gap junction — lens cellular coupling;
          W45S, P88S common pathogenic variants; seed SEED_BASE+2).
GJA3    (Connexin 46 / Cx46; 435 aa; ~46 kDa; 13q12.11; AD;
          Cataract-14 — NUCLEAR TOTAL / CERULEAN morphology;
          Lens fiber cell gap junction — critical for ion/water transport;
          GJA3 + GJA8 double knockout = total nuclear lens opacity in mice;
          seed SEED_BASE+3).
MIP     (Major Intrinsic Protein / Aquaporin-0 / AQP0; 263 aa; ~28 kDa; 12q13.3; AD;
          Cataract-15 — LAMELLAR / ZONULAR most common morphology;
          Most abundant lens fiber cell membrane protein (~45% of total);
          T138R, R33C, E134G common AD variants; seed SEED_BASE+4).
EPHA2   (Ephrin Type-A Receptor 2; 976 aa; ~108 kDa; 1p36.13; AD/AR;
          Cataract-6 — CORTICAL / POSTERIOR SUBCAPSULAR most common;
          Receptor tyrosine kinase; common risk factor for age-related PSC cataract;
          R721Q most prevalent European AD variant; seed SEED_BASE+5).
NHS     (Nance-Horan Syndrome Protein; 1630 aa; ~177 kDa; Xp22.13; XLR;
          Nance-Horan Syndrome — DENSE NUCLEAR CATARACT IN HEMIZYGOUS MALES PATHOGNOMONIC;
          Carrier females: posterior sutural opacities only (NON-DENSE);
          DENTAL ANOMALIES: supplemental maxillary incisors + screwdriver-shaped teeth;
          seed SEED_BASE+6).
FYCO1   (FYVE and Coiled-Coil Domain Autophagy Adaptor 1; 1478 aa; ~167 kDa; 3p21.31; AR;
          Cataract-18 — AR founder mutations in Middle Eastern/South Asian/Chinese consanguineous families;
          Autophagy adaptor — disrupts lens crystallin clearance;
          Total nuclear cataract; NO systemic involvement;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2382-2389).
"""

import random

SEED_BASE = 2382

CATARACT_GENES = [
    # -- CRYAA -- αA-Crystallin / Cataract-9 ------------------------------------------------
    {
        "gene": "CRYAA",
        "alt_name": (
            "CRYAA (CRYAA-173aa-21q22.3 / AD/AR -- "
            "CATARACT-9-MOST-COMMON-CRYSTALLIN-CATARACT-GENE -- "
            "POSTERIOR-SUBCAPSULAR-OR-ZONULAR-LAMELLAR-AD -- "
            "AR-BIALLELIC-CATARACT+MICROPHTHALMIA+IRIS-COLOBOMA -- "
            "R116C-MOST-COMMON-AD-VARIANT-DISRUPTS-CHAPERONE-CORE)"
        ),
        "protein": (
            "CRYAA -- 21q22.3 AD/AR -- CRYAA-173aa -- "
            "Alpha-A-Crystallin-HspB4-20kDa-Small-Heat-Shock-Protein-Eye-Lens-Specific -- "
            "N-Terminal-ARM-Domain-Alpha-Crystallin-Domain-C-Terminal-Extension -- "
            "Forms-Large-Polydisperse-Oligomers-24-mer-Average-with-CRYAB-HspB5 -- "
            "Major-Structural-Protein-Lens-Epithelium-Fiber-Cells-35pct-Lens-Protein -- "
            "Chaperone-Prevents-Crystallin-Aggregation-Maintains-Lens-Transparency -- "
            "OMIM-Gene-123580-Disease-CATARACT9-604219"
        ),
        "locus": "21q22.3",
        "protein_size": "173 aa / 20 kDa",
        "inheritance": (
            "AD (dominant-negative or haploinsufficiency) for posterior subcapsular / zonular cataract; "
            "AR (biallelic LOF) for congenital cataract + microphthalmia + iris coloboma (more severe); "
            "CRYAA is one of the most common crystallin genes causing hereditary cataract; "
            "R116C variant: most common AD pathogenic variant, disrupts alpha-crystallin domain hydrophobic core, "
            "reduces chaperone function by 50%, dominant-negative; "
            "incomplete penetrance and variable expressivity in some AD families; "
            "compound heterozygosity described for AR cases"
        ),
        "disease_category": "Cataract-9 (CRYAA-associated hereditary cataract) — posterior subcapsular AD; congenital cataract + microphthalmia + coloboma AR",
        "disease_pathway": (
            "CRYAA encodes αA-crystallin (HspB4), a small heat shock protein that constitutes ~35% of total lens protein. "
            "αA-crystallin forms large polydisperse oligomers (average 24-mer) with αB-crystallin (CRYAB/HspB5). "
            "PRIMARY FUNCTION: molecular chaperone — prevents aggregation of denatured lens proteins (βB2-crystallin, γD-crystallin) "
            "and maintains lens transparency by sequestering misfolded crystallins before they aggregate. "
            "PATHOMECHANISM (AD): Missense variants (R116C most common) disrupt the α-crystallin domain hydrophobic core → "
            "reduced chaperone function → abnormal partner crystallin aggregation → opacification in posterior subcapsular or zonular pattern. "
            "R116C causes dominant-negative oligomer dysfunction — mutant CRYAA incorporates into wild-type oligomers and impairs their chaperone activity. "
            "PATHOMECHANISM (AR): Biallelic LOF → complete absence of CRYAA chaperone activity → "
            "severe misfolding of lens proteins → dense nuclear cataract + microphthalmia + iris coloboma (loss of CRYAA during lens development). "
            "Secondary: CRYAA mutations may also disrupt lens fiber cell membrane interactions (MIP/AQP0 chaperone activity). "
            "Mouse knockout: CRYAA-/- mice develop nuclear cataracts at 6 months confirming its essential lens transparency role."
        ),
        "pathognomonic": (
            "POSTERIOR SUBCAPSULAR OR ZONULAR-LAMELLAR CATARACT at slit-lamp for AD cases. "
            "Nuclear cataract at birth for AR biallelic LOF cases — combined with MICROPHTHALMIA (small eye, axial length <19mm) and IRIS COLOBOMA. "
            "R116C VARIANT: most common pathogenic CRYAA variant — reported in multiple unrelated families worldwide; "
            "cataract morphology: nuclear → posterior subcapsular → total; "
            "onset: childhood to young adulthood in AD cases. "
            "R49C, R49S variants: congenital nuclear cataract — earlier onset, denser opacity. "
            "AR biallelic cases: CATARACT + MICROPHTHALMIA + IRIS COLOBOMA TRIAD PATHOGNOMONIC for biallelic CRYAA. "
            "Posterior subcapsular cataract (PSC): glare, reduced BCVA for near, visual symptoms out of proportion to slit-lamp opacity size. "
            "Zonular-lamellar: discrete shell of opacity within a clear lens — bilateral, symmetric. "
            "R116C on one allele — always check second allele for compound het in younger/more severe cases."
        ),
        "treatment": (
            "Surgical: Phacoemulsification + posterior chamber intraocular lens implant (PC-IOL) when visually significant — "
            "the definitive curative treatment; no medical therapy exists. "
            "Timing: adult AD cases — when BCVA degrades or glare impacts quality of life; "
            "congenital AR cases — urgent surgery within weeks of birth (critical period for visual development amblyopia prevention). "
            "IOL power calculation: standard biometry; target emmetropia. "
            "Paediatric considerations (AR/congenital cases): contact lens or glasses for optical correction post-surgery; "
            "amblyopia treatment (patching/atropine penalization of fellow eye); regular orthoptic review; "
            "microphthalmia may limit IOL options — consult specialist paediatric ophthalmologist. "
            "Genetic counselling: AD cases — 50% risk per offspring; AR — 25%; "
            "screen first-degree relatives slit-lamp annually. "
            "Posterior capsule opacification (PCO): more frequent in paediatric cases — "
            "primary posterior capsulectomy + anterior vitrectomy at index surgery in children <7 years."
        ),
        "key_features": [
            "Most common crystallin gene causing hereditary cataract worldwide",
            "AD: posterior subcapsular or zonular-lamellar — R116C most common variant",
            "AR biallelic: CATARACT + MICROPHTHALMIA + IRIS COLOBOMA triad pathognomonic",
            "αA-crystallin chaperone maintains lens transparency — 35% of total lens protein",
            "Paediatric cases: urgent surgery + contact lens + patching (amblyopia prevention)",
        ],
        "key_ddx": [
            "CRYAA vs CRYAB: CRYAB R120G uniquely associated with cardiomyopathy/myopathy; CRYAA cardiac involvement rare",
            "Posterior subcapsular cataract (PSC): CRYAA vs EPHA2 — EPHA2 more cortical/PSC in adults, CRYAA earlier onset",
            "AR CRYAA microphthalmia vs nanophthalmos: nanophthalmos axial length <20mm but normal lens/cornea ratio; CRYAA has coloboma",
            "Iris coloboma from CRYAA vs PAX6 aniridia: PAX6 = absent iris (aniridia, not coloboma); CRYAA = sectoral coloboma inferior",
            "Zonular cataract: CRYAA vs MIP (AQP0) — MIP gives lamellar/zonular also; molecular testing distinguishes",
        ],
        "morphology": "Posterior subcapsular / Zonular-lamellar (AD); Dense nuclear + microphthalmia + iris coloboma (AR)",
        "systemic_involvement": False,
        "onset_age": "Childhood–young adult (AD); Congenital (AR)",
        "surgical_urgency": "Urgent in AR/congenital (amblyopia); Elective in AD adult-onset",
        "gene_family": "Small Heat Shock Protein / αA-Crystallin",
    },
    # -- CRYAB -- αB-Crystallin / Cataract-16 + Myofibrillar Myopathy -----------------------
    {
        "gene": "CRYAB",
        "alt_name": (
            "CRYAB (CRYAB-175aa-11q23.1 / AD/AR -- "
            "CATARACT-16-PLUS-MYOFIBRILLAR-MYOPATHY-OR-DILATED-CARDIOMYOPATHY -- "
            "R120G-PATHOGNOMONIC-DESMIN-RELATED-MYOPATHY-CARDIAC+CATARACT -- "
            "FIBRILLAR-PROTEIN-AGGREGATES-CARDIAC-SKELETAL-MUSCLE-BIOPSY-PATHOGNOMONIC)"
        ),
        "protein": (
            "CRYAB -- 11q23.1 AD/AR -- CRYAB-175aa -- "
            "Alpha-B-Crystallin-HspB5-20kDa-Small-Heat-Shock-Protein-Ubiquitous-Expression -- "
            "N-Terminal-ARM-Domain-Alpha-Crystallin-Domain-C-Terminal-IPV-Motif -- "
            "Forms-Oligomers-with-CRYAA-HspB4-Also-Homomers -- "
            "Ubiquitously-Expressed-Lens-Cardiac-Skeletal-Muscle-Brain-High-Stress-Response -- "
            "Chaperone-Desmin-IF-Client-Prevents-Desmin-Aggregation-in-Cardiomyocytes -- "
            "OMIM-Gene-123590-Disease-CATARACT16-613763-MFM2-608810"
        ),
        "locus": "11q23.1",
        "protein_size": "175 aa / 20 kDa",
        "inheritance": (
            "AD (dominant-negative) for Cataract-16 + Myofibrillar Myopathy Type 2 (MFM2) / Dilated Cardiomyopathy; "
            "AR (biallelic LOF) for congenital cataract without major systemic disease in some reports; "
            "R120G: MOST PATHOGENIC VARIANT — complete dominant-negative disruption of CRYAB chaperone function; "
            "causes all three: posterior cataract + cardiomyopathy + myopathy simultaneously; "
            "464delCT frameshift: AR LOF — congenital cataract without significant myopathy; "
            "D109H variant: cardiomyopathy predominant, minimal cataract; "
            "variable expressivity — some families cataract-dominant, others cardiac-dominant"
        ),
        "disease_category": "Cataract-16 (CRYAB AD) / Myofibrillar Myopathy Type 2 (MFM2) / Dilated Cardiomyopathy — posterior cataract + desmin-related myopathy",
        "disease_pathway": (
            "CRYAB encodes αB-crystallin (HspB5), a small heat shock protein expressed in lens, cardiac muscle, skeletal muscle, and brain. "
            "Unlike CRYAA (lens-restricted), CRYAB is ubiquitous. "
            "PRIMARY FUNCTION: molecular chaperone for desmin intermediate filaments (IFs) in cardiac and skeletal muscle. "
            "Desmin forms the IF network that maintains Z-disc structural integrity in sarcomeres. "
            "PATHOMECHANISM (R120G): Arg120→Gly substitution disrupts the hydrophobic core of the alpha-crystallin domain → "
            "CRYAB cannot fold properly → cannot chaperone desmin IFs → "
            "DESMIN AGGREGATION in Z-discs → Z-disc disintegration → sarcomere fragmentation → "
            "cardiomyopathy (dilated or hypertrophic) + skeletal myopathy (proximal limb weakness). "
            "In the lens: R120G disrupts chaperone activity → beta/gamma crystallin aggregation → posterior subcapsular cataract. "
            "HISTOPATHOLOGY (pathognomonic): Gömöri trichrome-positive FIBRILLAR INCLUSIONS in cardiac and skeletal muscle biopsy; "
            "congophilic (amyloid-like) deposits; desmin immunoreactivity around and within inclusions. "
            "CRYAB itself colocalizes within inclusions — diagnostic immunohistochemistry. "
            "In extreme cases: CRYAB forms large intracellular inclusion bodies visible by EM as granulofilamentous material."
        ),
        "pathognomonic": (
            "R120G VARIANT: posterior subcapsular/cortical cataract + skeletal muscle weakness (proximal > distal) + "
            "cardiomyopathy (dilated most common) — TRIAD PATHOGNOMONIC for CRYAB R120G. "
            "MUSCLE BIOPSY: Gömöri trichrome-positive fibrillar inclusions PATHOGNOMONIC for MFM; "
            "desmin immunostaining: perinuclear and subsarcolemmal deposits + within inclusions; "
            "CRYAB immunostaining: colocalises in inclusions — diagnostic. "
            "Electron microscopy: 10-15nm granulofilamentous material = hallmark EM finding. "
            "CARDIAC: dilated cardiomyopathy most common; also hypertrophic; restrictive pattern; "
            "heart failure; arrhythmias; sudden cardiac death — cardiac MRI shows late gadolinium enhancement (midwall fibrosis). "
            "EMG: myopathic pattern (short-duration, polyphasic); "
            "CK: mildly elevated (~2-5x ULN) — not dramatically elevated (unlike dystrophinopathy). "
            "Cataract: posterior subcapsular or nuclear; bilateral; onset 3rd–5th decade in R120G; "
            "may precede or follow muscle/cardiac symptoms — ophthalmological diagnosis may be first clue."
        ),
        "treatment": (
            "Cataract: phacoemulsification + PC-IOL — standard surgical management when visually significant. "
            "Cardiomyopathy: guideline-directed heart failure therapy (ACEi/ARBi + beta-blocker + diuretics + SGLT2i); "
            "ICD implantation for primary prevention of sudden cardiac death in DCM with reduced EF < 35%; "
            "cardiac transplantation for end-stage; regular cardiology review every 6 months. "
            "Myopathy: physiotherapy; occupational therapy; ankle-foot orthoses for foot drop; "
            "respiratory assessment (FVC, MIP, MEP) annually — respiratory failure uncommon but described; "
            "nutritional support for dysphagia (MFM with oropharyngeal involvement). "
            "Genetic screening: all first-degree relatives — cardiac and ophthalmological assessment; "
            "cascade testing with CRYAB sequencing; "
            "prenatal/preimplantation genetic diagnosis available for severe R120G families. "
            "Experimental: proteasome activators (to clear aggregates) in pre-clinical stage."
        ),
        "key_features": [
            "R120G pathognomonic: posterior cataract + myopathy + cardiomyopathy triad",
            "Desmin-related myopathy (MFM2): Gömöri trichrome fibrillar inclusions on biopsy pathognomonic",
            "Ubiquitous expression vs CRYAA (lens-restricted) — hence systemic involvement",
            "Sudden cardiac death risk: ICD indicated when EF <35%",
            "Cataract may be the first clinical presentation before cardiac/muscle disease",
        ],
        "key_ddx": [
            "CRYAB vs DES (desmin): DES mutations give identical MFM histopathology; distinguish by gene testing",
            "CRYAB vs CRYAA: CRYAA has no cardiac/muscle disease; CRYAB R120G gives triad",
            "DCM from CRYAB vs LMNA: LMNA gives conduction disease early + arrhythmia; CRYAB gives systolic failure + inclusions",
            "MFM vs polymyositis: MFM has inclusions on biopsy, no inflammatory infiltrate; CK mildly elevated",
            "CRYAB cataract vs age-related PSC: family history + systemic features + age <50 → genetic testing",
        ],
        "morphology": "Posterior subcapsular / cortical (AD R120G); Nuclear (AR biallelic)",
        "systemic_involvement": True,
        "onset_age": "3rd–5th decade AD; Congenital AR",
        "surgical_urgency": "Elective (adult onset)",
        "gene_family": "Small Heat Shock Protein / αB-Crystallin",
    },
    # -- GJA8 -- Connexin 50 / Cataract-1 ---------------------------------------------------
    {
        "gene": "GJA8",
        "alt_name": (
            "GJA8 (GJA8-440aa-1q21.1 / AD -- "
            "CATARACT-1-NUCLEAR-PULVERULENT-TOTAL-NUCLEAR -- "
            "CONNEXIN-50-CX50-LENS-FIBER-CELL-GAP-JUNCTION -- "
            "W45S-P88S-COMMON-AD-PATHOGENIC-VARIANTS -- "
            "LENS-ZONULAR-FIBERS-ZONE-OF-DISCONTINUITY-OPACIFICATION)"
        ),
        "protein": (
            "GJA8 -- 1q21.1 AD -- GJA8-440aa -- "
            "Connexin-50-Cx50-50kDa-Gap-Junction-Protein-Alpha-8 -- "
            "4-TM-Domains-2-Extracellular-Loops-1-Cytoplasmic-Loop-N-C-Terminal-Cytoplasmic -- "
            "Forms-Gap-Junction-Channels-Between-Lens-Fiber-Cells-Hexameric-Connexons -- "
            "Lens-Fiber-Cell-Specific-Expression-Also-in-Ganglion-Cells-Minor -- "
            "Ion-Metabolite-Water-Transport-Between-Avascular-Lens-Fibers -- "
            "OMIM-Gene-600897-Disease-CATARACT1-116200"
        ),
        "locus": "1q21.1",
        "protein_size": "440 aa / 50 kDa",
        "inheritance": (
            "AD (dominant-negative or gain-of-abnormal-function); "
            "rare AR biallelic LOF reported but most pathogenic variants are AD; "
            "W45S (Trp45→Ser): most common pathogenic variant — disrupts first extracellular loop, "
            "prevents hemichannel docking, dominant-negative effects on wild-type Cx50; "
            "P88S (Pro88→Ser): second most common — disrupts transmembrane helix 2; "
            "multiple other missense variants throughout gene; "
            "GJA8 at 1q21.1 — same chromosomal region as GJA3 (13q12.11 nearby but different chromosome); "
            "extensive genetic heterogeneity within cataract-1 locus families"
        ),
        "disease_category": "Cataract-1 — nuclear pulverulent / total nuclear / lamellar congenital hereditary cataract; lens fiber cell gap junction defect",
        "disease_pathway": (
            "GJA8 encodes Connexin 50 (Cx50), a lens fiber cell gap junction protein. "
            "The lens is avascular — it depends entirely on gap junction networks (Cx50 + Cx46/GJA3) for nutrition, "
            "ion homeostasis, and waste removal. "
            "Lens fiber cells lose organelles (including mitochondria) as they mature → completely dependent on gap junction transport. "
            "FUNCTION: Cx50 forms hexameric hemichannels (connexons) that dock with connexons on adjacent fiber cells "
            "to create intercellular gap junction channels. "
            "Channels allow passage of: ions (Na+, K+, Ca2+), small metabolites (glucose, glutathione), water. "
            "PATHOMECHANISM: GJA8 missense variants (W45S, P88S) → "
            "misfolded connexin cannot form functional hemichannels → "
            "dominant-negative incorporation into wild-type Cx50/Cx46 channels → "
            "disrupted lens fiber cell communication → "
            "ion gradient collapse → oxidative stress → crystallin aggregation → "
            "NUCLEAR PULVERULENT CATARACT (powdery nuclear opacification). "
            "In severe variants (W45S homozygous): total nuclear cataract + microphthalmos (mouse GJA8-/- confirms). "
            "Cx50 also required for lens fiber cell elongation — loss causes shorter fiber cells → smaller lens."
        ),
        "pathognomonic": (
            "NUCLEAR PULVERULENT CATARACT: fine dusty powdery nuclear opacification PATHOGNOMONIC for GJA8-associated cataract. "
            "Bilateral, symmetric nuclear opacification with fine granular appearance on slit-lamp retroillumination. "
            "Total nuclear cataract in severe cases (W45S homozygous or severe missense). "
            "LAMELLAR morphology in some families. "
            "Onset: CONGENITAL or early infantile (birth to 6 months) — nystagmus and visual deprivation amblyopia if not treated early. "
            "Microphthalmos may accompany dense nuclear cataracts (similar to GJA8-/- mouse). "
            "NO SYSTEMIC INVOLVEMENT — pure ocular. "
            "Family history: AD — 50% of offspring affected; often multiple generations; "
            "variable intra-familial severity (some members mild, others dense — same W45S variant). "
            "W45S: most commonly reported in South Asian families (Pakistani, Indian); "
            "P88S: European families predominantly. "
            "Lens proteomics: elevated gamma-crystallin oligomers confirming chaperone failure secondary to transport disruption."
        ),
        "treatment": (
            "Phacoemulsification + PC-IOL for adult-onset or less severe cases. "
            "CONGENITAL CATARACT (most GJA8 cases): lens aspiration (lensectomy) + anterior vitrectomy + "
            "primary posterior capsulotomy in children <7 years — prevents posterior capsule opacification. "
            "IOL: primary IOL insertion at ≥6 months (specialist centres); "
            "aphakic contact lens + glasses for infants <6 months; "
            "IOL exchange with growth if performed very early. "
            "AMBLYOPIA MANAGEMENT: critical — aggressive patching of better eye (2-6 hours/day); "
            "atropine penalization of better eye alternately; "
            "contact lens fitting for optical rehabilitation post-surgery. "
            "Strabismus: evaluate post-operatively — early surgical correction if esotropia develops. "
            "Genetic counselling: 50% AD offspring risk; "
            "screen neonates of affected parent by slit-lamp at birth. "
            "Paediatric ophthalmology follow-up: 3-6 monthly minimum until age 8 (amblyopia period)."
        ),
        "key_features": [
            "Nuclear pulverulent (fine powdery nuclear) cataract — pathognomonic morphology",
            "Connexin 50 lens fiber gap junction — avascular lens dependent on Cx50 + Cx46 for nutrition",
            "Congenital onset in most — urgent surgery required to prevent amblyopia",
            "W45S most common variant (South Asian); P88S European — dominant-negative",
            "No systemic involvement — pure ocular lens defect",
        ],
        "key_ddx": [
            "GJA8 vs GJA3 nuclear cataract: identical morphology; GJA8 at 1q21.1, GJA3 at 13q12.11 — gene testing only",
            "Nuclear pulverulent from GJA8 vs FYCO1: FYCO1 is AR (consanguineous families); GJA8 is AD",
            "Congenital cataract: rule out metabolic (galactosaemia: reducing substances in urine), "
                "intrauterine infection (TORCH titres), trisomy 21",
            "Total nuclear GJA8 vs CRYAA AR: CRYAA AR has microphthalmia + iris coloboma; GJA8 AD has neither",
            "GJA8 with microophthalmos vs FOXE3 Peters anomaly: Peters has corneal opacification centrally",
        ],
        "morphology": "Nuclear pulverulent / Total nuclear / Lamellar",
        "systemic_involvement": False,
        "onset_age": "Congenital / early infantile",
        "surgical_urgency": "Urgent (congenital — amblyopia risk)",
        "gene_family": "Connexin / Gap Junction Protein",
    },
    # -- GJA3 -- Connexin 46 / Cataract-14 ---------------------------------------------------
    {
        "gene": "GJA3",
        "alt_name": (
            "GJA3 (GJA3-435aa-13q12.11 / AD -- "
            "CATARACT-14-NUCLEAR-TOTAL-CERULEAN -- "
            "CONNEXIN-46-CX46-LENS-FIBER-CELL-GAP-JUNCTION -- "
            "N188T-P187L-COMMON-AD-PATHOGENIC-VARIANTS -- "
            "GJA3+GJA8-DOUBLE-KO-MOUSE-COMPLETE-NUCLEAR-OPACITY)"
        ),
        "protein": (
            "GJA3 -- 13q12.11 AD -- GJA3-435aa -- "
            "Connexin-46-Cx46-46kDa-Gap-Junction-Protein-Alpha-3 -- "
            "4-TM-Domains-2-Extracellular-Loops-Cytoplasmic-Loop-N-C-Terminal-Cytoplasmic -- "
            "Lens-Fiber-Cell-Specific-Not-Detected-in-Lens-Epithelium-Only-Fiber-Cells -- "
            "Forms-Heteromeric-Channels-with-Cx50-GJA8-Also-Homomeric-Cx46-Channels -- "
            "Critical-for-Intracellular-Calcium-Ion-Homeostasis-in-Inner-Cortical-Fiber-Cells -- "
            "OMIM-Gene-121015-Disease-CATARACT14-601885"
        ),
        "locus": "13q12.11",
        "protein_size": "435 aa / 46 kDa",
        "inheritance": (
            "AD (dominant-negative); most GJA3 pathogenic variants are missense causing dominant-negative effects; "
            "N188T: most frequently reported AD variant — asparagine in extracellular loop 2, disrupts hemichannel docking; "
            "P187L: adjacent proline, same extracellular loop 2 region; "
            "rare AR biallelic variants described causing more severe/earlier cataract; "
            "GJA3 at 13q12.11 — distinct chromosomal location from GJA8 (1q21.1) despite being in same connexin family; "
            "GJA3 and GJA8 sometimes co-mutated in consanguineous families (digenic) — more severe phenotype; "
            "extensive allelic heterogeneity with >50 pathogenic variants described"
        ),
        "disease_category": "Cataract-14 — nuclear total / cerulean / multiple punctate congenital hereditary cataract; lens fiber cell Cx46 gap junction defect",
        "disease_pathway": (
            "GJA3 encodes Connexin 46 (Cx46), expressed exclusively in mature lens fiber cells (not in epithelium). "
            "FUNCTION: Cx46 forms gap junction channels (homo- or heteromeric with Cx50/GJA8) between inner cortical fiber cells. "
            "Critical roles of Cx46: "
            "1) Calcium homeostasis — regulates Ca2+ flux in inner fiber cells that lack ER; "
            "2) Metabolite delivery — ensures glucose/glutathione reach central fiber cells; "
            "3) Ion balance — maintains K+/Na+ equilibrium preventing osmotic swelling. "
            "PATHOMECHANISM (N188T, P187L): disruption of extracellular loop 2 → "
            "inability to dock with apposing connexon → hemichannel remains undocked → "
            "two consequences: (a) dominant-negative incorporation into wild-type hexamers → "
            "loss of gap junction function in inner cortex; "
            "(b) unopposed hemichannel activity → abnormal Ca2+ influx → calpain activation → crystallin proteolysis. "
            "GJA3-/- mouse: develop nuclear cataracts by 2 months despite Cx50 intact → "
            "confirms Cx46 essential and non-redundant in inner fiber cell Ca2+ control. "
            "GJA3+GJA8 double knockout: complete nuclear opacity by birth."
        ),
        "pathognomonic": (
            "NUCLEAR TOTAL or CERULEAN cataract PATHOGNOMONIC for GJA3 variants. "
            "CERULEAN (blue-dot) morphology: small blue-white punctate opacities in peripheral cortex and nucleus "
            "on slit-lamp retroillumination — classic for GJA3 (also seen in CYP51A1 and some other rare genes). "
            "Total nuclear opacity: dense white nuclear cataract in more severe variants. "
            "Bilateral, symmetric; onset: congenital or first year of life. "
            "NO SYSTEMIC FEATURES — pure lens disease (critical DDx from CRYAB). "
            "N188T variant: cerulean + sutural opacities in some families. "
            "P187L variant: bilateral congenital nuclear cataract, denser than N188T. "
            "Anterior subcapsular changes in some adult-onset milder variants. "
            "Slit-lamp pearl: retroillumination superior to direct illumination for identifying cerulean dots. "
            "Molecular diagnosis: panel sequencing (inherited cataract panel including GJA3, GJA8, CRYAA, CRYAB, MIP, EPHA2). "
            "Nystagmus: present in dense congenital cases (visual deprivation → latent/manifest nystagmus). "
            "Strabismus: esotropia common after congenital cataract surgery (late diagnosis)."
        ),
        "treatment": (
            "Same principles as GJA8 (congenital cataract). "
            "Dense nuclear/total cataract: lens aspiration + anterior vitrectomy + posterior capsulotomy + IOL "
            "(primary implantation if ≥6 months; aphakia + contact lens if <6 months). "
            "Cerulean mild cataracts: observation until visually significant; "
            "monitor BCVA, refraction, visual evoked potentials in infants. "
            "AMBLYOPIA: critical — early aggressive management. "
            "If surgery before amblyopiogenic period (<10 years) and rehabilitation prompt: good visual prognosis. "
            "Regular refraction: high myopia may develop post-surgery (nuclear cataract → myopic shift); "
            "glasses + low-vision aids if myopia not fully correctable. "
            "Nystagmus: usually improves post-cataract surgery but may persist; prism glasses if head posture. "
            "Genetic counselling: AD — 50% risk per offspring; "
            "prenatal diagnosis/PGT available if severe pathogenic variant identified."
        ),
        "key_features": [
            "Cerulean (blue-dot) or total nuclear cataract — pathognomonic morphologies for GJA3",
            "Connexin 46 exclusively in lens fiber cells — critical Ca2+ homeostasis",
            "GJA3 + GJA8 double knockout = complete nuclear opacity: functional synergy essential",
            "No systemic involvement — pure ocular lens disease",
            "Congenital onset — urgent surgical rehabilitation with amblyopia prevention",
        ],
        "key_ddx": [
            "GJA3 vs GJA8 nuclear cataract: cerulean morphology more GJA3; pulverulent more GJA8; gene testing definitive",
            "Cerulean cataract: GJA3 vs CYP51A1 (hypocholesterolaemia + cerulean) — CYP51A1 has low plasma cholesterol",
            "Congenital nuclear: GJA3 vs FYCO1 (AR — consanguinity history); GJA3 is AD",
            "GJA3 vs CRYAB: CRYAB has cardiomyopathy + myopathy; GJA3 pure lens",
            "Nystagmus from GJA3 vs idiopathic: look for cerulean/nuclear opacities on dilated slit-lamp",
        ],
        "morphology": "Nuclear total / Cerulean (blue-dot) / Sutural",
        "systemic_involvement": False,
        "onset_age": "Congenital / early infantile",
        "surgical_urgency": "Urgent if dense (congenital amblyopia risk); Observation if cerulean mild",
        "gene_family": "Connexin / Gap Junction Protein",
    },
    # -- MIP -- Aquaporin-0 / Major Intrinsic Protein / Cataract-15 -------------------------
    {
        "gene": "MIP",
        "alt_name": (
            "MIP (MIP-263aa-12q13.3 / AD -- "
            "CATARACT-15-LAMELLAR-ZONULAR -- "
            "AQUAPORIN-0-AQP0-MAJOR-INTRINSIC-PROTEIN-LENS-FIBER -- "
            "MOST-ABUNDANT-LENS-FIBER-CELL-MEMBRANE-PROTEIN-45pct -- "
            "T138R-R33C-E134G-COMMON-AD-PATHOGENIC-VARIANTS)"
        ),
        "protein": (
            "MIP -- 12q13.3 AD -- MIP-263aa -- "
            "Major-Intrinsic-Protein-Aquaporin-0-AQP0-28kDa-Water-Channel -- "
            "6-TM-Domains-2-NPA-Motifs-Classic-Aquaporin-Fold-Tetrameric-Assembly -- "
            "Most-Abundant-Lens-Fiber-Cell-Membrane-Protein-45pct-Total-Lens-Protein -- "
            "Dual-Function-Water-Channel-AND-Cell-Cell-Adhesion-Junction-Former -- "
            "Long-Ranged-Ordered-Junction-Arrays-Between-Lens-Fiber-Cells -- "
            "OMIM-Gene-154050-Disease-CATARACT15-615274"
        ),
        "locus": "12q13.3",
        "protein_size": "263 aa / 28 kDa",
        "inheritance": (
            "AD (dominant-negative haploinsufficiency); "
            "T138R (Thr138→Arg): most common pathogenic variant — within transmembrane domain 4, "
            "disrupts water-channel pore selectivity filter, dominant-negative oligomer effects; "
            "R33C: disrupts N-terminal cytoplasmic domain, affects MIP trafficking to membrane; "
            "E134G: transmembrane domain 3, disrupts pore architecture; "
            "rare AR biallelic cases described with more severe lamellar cataract; "
            "MIP at 12q13.3; widely separated from GJA8 (1q21.1) and GJA3 (13q12.11); "
            "incomplete penetrance described in some MIP families"
        ),
        "disease_category": "Cataract-15 — lamellar (zonular) cataract; lens fiber cell membrane protein / water channel defect",
        "disease_pathway": (
            "MIP (AQP0) encodes Aquaporin-0, the most abundant protein in the lens fiber cell plasma membrane (~45% of total membrane protein). "
            "DUAL FUNCTION — unique among aquaporins: "
            "1) WATER CHANNEL: facilitates passive water transport through lens fiber cell membranes; "
            "maintains lens hydration equilibrium (microcirculation system); "
            "2) CELL-CELL ADHESION: forms ordered junction arrays (thin junctions) between apposing lens fiber cells — "
            "structural role in maintaining fiber cell alignment and lens architecture. "
            "PATHOMECHANISM: MIP missense variants (T138R, E134G) disrupt water channel pore geometry → "
            "reduced water permeability in fiber cells → hydration imbalance in specific lens zones → "
            "LAMELLAR (ZONULAR) OPACITY: affects a specific concentric shell of lens fibers "
            "(corresponding to fiber cells present during a critical developmental window). "
            "Dominant-negative: mutant MIP incorporates into tetramers → poisons wild-type AQP0 channels. "
            "R33C: disrupts trafficking → MIP retained in ER → reduced surface expression → haplo-insufficiency. "
            "Lamellar pattern = discrete zone of opacity within a clear lens (unlike nuclear which affects entire nucleus). "
            "Mouse Mip-/- knockout: dense nuclear cataract at birth — confirms indispensable role."
        ),
        "pathognomonic": (
            "LAMELLAR (ZONULAR) CATARACT: discrete shell or ring of opacity within a clear central nucleus and clear cortex — "
            "PATHOGNOMONIC for MIP-associated hereditary cataract. "
            "Slit-lamp: sharply demarcated lamellar opacity (like a disk within the lens); "
            "retroillumination: annular/ring opacity visible; "
            "direct: diffuse gray opacification in a zone. "
            "BILATERAL and SYMMETRIC — helps distinguish from acquired lamellar cataract. "
            "Onset: CONGENITAL or early childhood — nystagmus present in dense cases. "
            "Mild cases: lamellar opacity with good visual function (zone of discontinuity only); "
            "dense cases: visual axis opacity → visual deprivation. "
            "T138R variant: lamellar + faint nuclear opacity in some families. "
            "NO SYSTEMIC INVOLVEMENT — pure lens opacity. "
            "Family history: AD — multiple generations with lamellar cataract. "
            "Incomplete penetrance: some obligate carriers have normal lenses (important for genetic counselling). "
            "Sutural opacities may also be present in some MIP variant families."
        ),
        "treatment": (
            "Dense lamellar cataract involving visual axis: lens aspiration/phacoemulsification + PC-IOL. "
            "Mild lamellar cataract not involving visual axis: conservative management — "
            "regular BCVA monitoring + refraction; cycloplegic refraction in children; "
            "consider surgery only when BCVA <6/12 sustained or risk of amblyopia. "
            "Contact lens: trial of aphakic contact lens if infant/young child — aids refraction assessment before IOL. "
            "Amblyopia prevention: patching or atropine penalization of fellow eye if asymmetric (one eye denser). "
            "IOL calculation: axial length measurement + keratometry; "
            "target slight hypermetropia in paediatric IOL (expected myopic shift during growth). "
            "Genetic counselling: AD with incomplete penetrance — counsel about 50% risk per offspring "
            "but penetrance data important for variant-specific prognosis. "
            "Screening: examine neonates of affected parents at birth + 3, 6 months with dilated slit-lamp."
        ),
        "key_features": [
            "Lamellar (zonular) cataract: discrete shell of opacity within clear lens — classic MIP morphology",
            "Most abundant lens fiber cell membrane protein (45% total) — water channel + adhesion dual function",
            "T138R most common variant — disrupts pore selectivity filter, dominant-negative",
            "Incomplete penetrance in some families — counselling requires variant-specific data",
            "No systemic involvement — pure lens disease",
        ],
        "key_ddx": [
            "Lamellar cataract: MIP vs CRYAA — CRYAA more PSC/nuclear; MIP gives discrete lamellar shell",
            "MIP lamellar vs acquired (cortisone-induced lamellar) — acquired: unilateral, steroid history, sutural changes",
            "Hereditary lamellar: MIP vs NHS (XLR nuclear in males); NHS gives nuclear not lamellar",
            "MIP with sutural opacity vs GJA3 (sutural + cerulean) — GJA3 has cerulean dots in addition",
            "Incomplete penetrance in MIP vs non-penetrant AD vs AR carrier: full pedigree analysis required",
        ],
        "morphology": "Lamellar / Zonular / Sutural",
        "systemic_involvement": False,
        "onset_age": "Congenital / early childhood",
        "surgical_urgency": "Elective if mild; Urgent if visual axis opacification",
        "gene_family": "Aquaporin / Major Intrinsic Protein",
    },
    # -- EPHA2 -- Ephrin Receptor A2 / Cataract-6 / PSC / Age-related cortical ---------------
    {
        "gene": "EPHA2",
        "alt_name": (
            "EPHA2 (EPHA2-976aa-1p36.13 / AD/AR -- "
            "CATARACT-6-CORTICAL-POSTERIOR-SUBCAPSULAR -- "
            "EPHRIN-TYPE-A-RECEPTOR-2-RTK-LENS-EPITHELIUM-FIBER-DIFFERENTIATION -- "
            "R721Q-MOST-PREVALENT-EUROPEAN-AD-VARIANT -- "
            "COMMON-RISK-GENE-FOR-AGE-RELATED-CORTICAL-AND-PSC-CATARACT)"
        ),
        "protein": (
            "EPHA2 -- 1p36.13 AD/AR -- EPHA2-976aa -- "
            "Ephrin-Type-A-Receptor-2-108kDa-Receptor-Tyrosine-Kinase -- "
            "Extracellular-Ephrin-Binding-Domain-Cysteine-Rich-EGF-Fibronectin-Domains -- "
            "Transmembrane-Domain-Juxtamembrane-Kinase-SAM-PDZ-Binding-Cytoplasmic -- "
            "Expressed-Lens-Epithelial-Cells-Equatorial-Zone-Fiber-Differentiation -- "
            "Regulates-Lens-Epithelial-to-Fiber-Differentiation-EphA2-Ephrin-A5-Signaling -- "
            "OMIM-Gene-176946-Disease-CATARACT6-116600"
        ),
        "locus": "1p36.13",
        "protein_size": "976 aa / 108 kDa",
        "inheritance": (
            "AD: missense variants causing gain-of-abnormal-function or dominant-negative; "
            "AR: biallelic LOF variants causing autosomal recessive cataract; "
            "R721Q (Arg721→Gln): most common AD variant in European populations — in kinase domain, "
            "reduces autophosphorylation; common risk haplotype in age-related cataract GWAS; "
            "P620S: AR LOF variant common in Chinese and Middle Eastern consanguineous families; "
            "EPHA2 is the most common gene associated with age-related cortical cataract in GWAS (OR ~1.4-1.6); "
            "hereditary family cases (AD) distinct from common age-related EPHA2 risk variants; "
            "incomplete penetrance described for R721Q AD families"
        ),
        "disease_category": "Cataract-6 — cortical / posterior subcapsular hereditary cataract; Ephrin receptor tyrosine kinase defect; also common genetic risk for age-related cortical cataract",
        "disease_pathway": (
            "EPHA2 encodes Ephrin type-A receptor 2, a receptor tyrosine kinase (RTK) expressed in lens epithelial cells "
            "of the equatorial zone (germinative zone) where epithelial cells differentiate into lens fiber cells. "
            "NORMAL FUNCTION: EphA2-EphrinA5 signaling regulates: "
            "1) Epithelial cell polarity and hexagonal packing; "
            "2) Epithelial-to-fiber cell differentiation at the equatorial bow; "
            "3) Adherens junction integrity in epithelial cells; "
            "4) N-cadherin-mediated cell-cell adhesion during fiber elongation. "
            "PATHOMECHANISM (R721Q): Reduced kinase autophosphorylation → "
            "disrupted EphA2 downstream signaling (Rho GTPase regulation) → "
            "aberrant epithelial polarity → abnormal fiber cell differentiation → "
            "CORTICAL AND POSTERIOR SUBCAPSULAR OPACIFICATION (posterior cortex = last-differentiated fibers). "
            "PATHOMECHANISM (AR LOF, P620S): Complete loss → no EphA2 signaling → "
            "severe disruption of epithelial packing → dense posterior cortical cataract from birth/early life. "
            "Age-related link: common EPHA2 haplotype variants mildly reduce signaling → cumulative oxidative stress → "
            "cortical spoke and wedge opacities in 6th-7th decade."
        ),
        "pathognomonic": (
            "CORTICAL WEDGE OR SPOKE OPACITIES: radially oriented cortical spokes from peripheral cortex toward visual axis — "
            "classic age-related pattern also seen in hereditary EPHA2 cases. "
            "POSTERIOR SUBCAPSULAR CATARACT (PSC): granular opacification just anterior to posterior lens capsule — "
            "the other main EPHA2-associated morphology. "
            "PSC symptoms: severe glare, halos, reduced BCVA for near (PSC proximity to nodal point). "
            "Hereditary EPHA2 cases: earlier onset (2nd–4th decade) vs age-related (6th+). "
            "R721Q AD variant: posterior subcapsular + cortical; bilateral; onset 30-50 years. "
            "AR EPHA2 (P620S): dense nuclear + cortical cataract congenital/childhood — more severe. "
            "NO SYSTEMIC FEATURES — pure lens disease. "
            "Retroillumination: PSC shows granular posterior opacity; "
            "cortical: wedge/spoke radial opacities from equator. "
            "Age-related risk: EPHA2 R721Q haplotype + smoking + UV exposure = additive cortical cataract risk. "
            "GWAS-confirmed common variant (rs3754334) associated with cortical cataract in large population studies."
        ),
        "treatment": (
            "Phacoemulsification + PC-IOL: definitive treatment when BCVA impaired or glare debilitating. "
            "PSC cataract: symptoms often severe relative to slit-lamp size — operate when functional impairment (glare on driving, difficulty with near work). "
            "Adults with hereditary EPHA2: anticipate earlier surgery (30-50 years vs age-related 65+). "
            "AR biallelic EPHA2 congenital cases: same urgent paediatric protocol as GJA8/GJA3. "
            "ANTI-GLARE: polarised / anti-reflective spectacle lenses as temporary measure for mild PSC. "
            "Pupillary dilation drops (phenylephrine 2.5%) to dilate past PSC for near — short-term measure only. "
            "IOL selection: blue-light filtering IOL; multifocal IOL generally avoided if glare pre-operatively. "
            "Genetic counselling: hereditary AD cases — 50% offspring risk; "
            "general population EPHA2 R721Q heterozygosity: modest cataract risk — reassurance, eye protection (sunglasses + UV400)."
        ),
        "key_features": [
            "Cortical / PSC cataract — most common hereditary morphologies",
            "Most common GWAS gene for age-related cortical cataract (common risk variant) AND hereditary cataract",
            "Receptor tyrosine kinase — regulates lens epithelial-to-fiber differentiation",
            "PSC: symptoms severe relative to opacity size (glare, near difficulty)",
            "R721Q: most common European hereditary variant (AD); P620S: AR in consanguineous families",
        ],
        "key_ddx": [
            "PSC from EPHA2 vs steroid-induced PSC: steroid history (systemic or inhaled); EPHA2 = family history + younger",
            "PSC EPHA2 vs diabetic PSC: HbA1c, fasting glucose; EPHA2 = normoglycaemic",
            "EPHA2 cortical vs age-related cortical: family history + onset <55 years + no other risk factors → EPHA2",
            "AR EPHA2 congenital vs FYCO1 AR congenital: clinical identical — gene panel distinguishes",
            "EPHA2 PSC vs posterior capsule fibrosis post-trauma: monocular, trauma history; EPHA2 bilateral, no trauma",
        ],
        "morphology": "Cortical (spokes/wedges) / Posterior subcapsular (PSC)",
        "systemic_involvement": False,
        "onset_age": "30–50 years AD; Congenital AR",
        "surgical_urgency": "Elective (adult onset); Urgent for AR congenital cases",
        "gene_family": "Ephrin Receptor / Receptor Tyrosine Kinase",
    },
    # -- NHS -- Nance-Horan Syndrome / XLR Dense Nuclear Cataract + Dental Anomalies --------
    {
        "gene": "NHS",
        "alt_name": (
            "NHS (NHS-1630aa-Xp22.13 / XLR -- "
            "NANCE-HORAN-SYNDROME-DENSE-NUCLEAR-CATARACT-HEMIZYGOUS-MALES-PATHOGNOMONIC -- "
            "DENTAL-ANOMALIES-SUPPLEMENTAL-MAXILLARY-INCISORS+SCREWDRIVER-TEETH -- "
            "CARRIER-FEMALES-POSTERIOR-SUTURAL-OPACITIES-ONLY-NON-DENSE -- "
            "INTELLECTUAL-DISABILITY-30-50pct)"
        ),
        "protein": (
            "NHS -- Xp22.13 XLR -- NHS-1630aa -- "
            "Nance-Horan-Syndrome-Protein-177kDa-Actin-Cytoskeleton-Regulatory -- "
            "WH2-Domain-Actin-Binding-WAVE-Homology-Domain-N-Terminal -- "
            "Expressed-Lens-Epithelium-Neurons-Kidney-Retina-Multiple-Tissues -- "
            "Regulates-Actin-Cytoskeleton-Dynamics-Lens-Epithelial-Cell-Morphology -- "
            "Interacts-WAVE1-WAVE2-Complex-Rac1-CDC42-Downstream-Effects -- "
            "OMIM-Gene-300457-Disease-NANCE-HORAN-302350"
        ),
        "locus": "Xp22.13",
        "protein_size": "1630 aa / 177 kDa",
        "inheritance": (
            "X-LINKED RECESSIVE (XLR): only hemizygous males severely affected; "
            "carrier females have POSTERIOR SUTURAL OPACITIES (non-dense, non-visually-significant in most); "
            "pathogenic variants: frameshift/nonsense (LOF) most common in severely affected males; "
            "missense variants associated with milder male phenotype; "
            "c.2023C>T, c.382C>T, large exon deletions common in NHS; "
            "de novo mutations described; "
            "maternal grandfather affected: classical X-linked pedigree; "
            "affected male → all daughters obligate carriers, no sons affected; "
            "rare manifesting carrier females with significant lens opacities described (X-inactivation skewing)"
        ),
        "disease_category": "Nance-Horan Syndrome — dense nuclear cataract + dental anomalies + intellectual disability (30-50%) in hemizygous males; XLR",
        "disease_pathway": (
            "NHS encodes the Nance-Horan syndrome protein, a large actin-regulatory protein with a WH2 (WASP Homology 2) domain "
            "that binds monomeric actin and a WAVE homology domain that interacts with the WAVE regulatory complex. "
            "FUNCTION: NHS regulates actin cytoskeleton dynamics in lens epithelial cells, "
            "required for maintaining epithelial cell morphology, adherens junction integrity, "
            "and lens epithelial-to-fiber cell transition. "
            "PATHOMECHANISM: LOF of NHS → disrupted actin polymerization in lens epithelium → "
            "lens epithelial cells become disorganized → failure to maintain normal epithelial monolayer → "
            "abnormal fiber cell differentiation → DENSE NUCLEAR CATARACT accumulates in developing fetal lens. "
            "DENTAL ANOMALIES: NHS expressed in dental epithelium (enamel organ) → "
            "loss causes abnormal dental morphogenesis → supplemental maxillary incisors + screwdriver-shaped incisors. "
            "INTELLECTUAL DISABILITY: NHS expressed in neurons → loss affects dendritic spine density and synaptic maturation → "
            "cognitive impairment in 30-50% of affected males (not universal). "
            "Carrier females (X-inactivation mosaic): sutural opacities = mild NHS LOF in cells with unfavourable X-inactivation."
        ),
        "pathognomonic": (
            "DENSE TOTAL NUCLEAR CATARACT IN HEMIZYGOUS MALES AT BIRTH PATHOGNOMONIC for Nance-Horan Syndrome. "
            "Nuclear cataract: white dense nuclear opacity visible on fundoscopy (absent red reflex); "
            "bilateral, symmetric; often diagnose by absent red reflex in newborn screen. "
            "DENTAL ANOMALIES (X-linked tetrad): "
            "1) SUPPLEMENTAL MAXILLARY INCISORS: extra teeth (mesiodens/supplemental incisors) between or behind upper front teeth; "
            "2) SCREWDRIVER-SHAPED INCISORS: narrow, peg-shaped, tapered upper central incisors (similar to Hutchinson incisors of congenital syphilis — but NHS not syphilis); "
            "3) Diastema (gaps between teeth); "
            "4) Poorly formed cusp tips. "
            "CARRIER FEMALES: POSTERIOR SUTURAL (Y-suture) OPACITIES — fine, non-dense sutural opacities on dilated slit-lamp; "
            "typically do NOT cause significant visual impairment; seen in ~70% of carrier females. "
            "INTELLECTUAL DISABILITY: mild-to-moderate in 30-50% of males — NOT universal; "
            "some males have normal intellect despite severe cataract. "
            "X-LINKED PEDIGREE: maternal uncle/maternal grandfather affected; no male-to-male transmission. "
            "DDx from Norrie disease (NDP): Norrie = blind neonates (retinal/vitreous, not lens); "
            "NHS = clear vitreous + dense lens nucleus."
        ),
        "treatment": (
            "CATARACT: urgent bilateral lens aspiration in neonatal period — "
            "critical to prevent visual deprivation amblyopia (dense bilateral nuclear cataract = severe amblyopia risk). "
            "Surgery timing: within weeks of birth in bilateral dense cases; "
            "paediatric ophthalmologist + paediatric anaesthesiology required. "
            "Optical rehabilitation: aphakic contact lenses immediately post-surgery; "
            "primary IOL insertion at ≥3 months if surgeon/centre experienced with infant IOL. "
            "Amblyopia: aggressive — bilateral dense cataract means risk in BOTH eyes; "
            "ensure refractive correction in both eyes simultaneously; "
            "patching or atropine if asymmetric visual development. "
            "DENTAL: paediatric dentistry + orthodontist consultation — supplemental teeth extraction/management; "
            "orthodontic correction of spacing and bite. "
            "INTELLECTUAL DISABILITY: developmental paediatrics + special educational support if present; "
            "formal neuropsychological assessment by age 3-4 years. "
            "GENETICS: maternal carrier testing (posterior sutural opacity on slit-lamp + NHS sequencing); "
            "prenatal diagnosis / preimplantation genetic testing available for NHS LOF families."
        ),
        "key_features": [
            "Dense nuclear cataract in hemizygous males at birth — pathognomonic for NHS",
            "X-LINKED RECESSIVE: carrier females have posterior sutural opacities (non-dense)",
            "DENTAL TETRAD: supplemental maxillary incisors + screwdriver-shaped teeth — clinical clue",
            "Intellectual disability 30-50% males — not universal; formal neuropsychological assessment needed",
            "X-linked pedigree: maternal uncle/grandfather blind from birth with dental anomalies",
        ],
        "key_ddx": [
            "NHS dense nuclear in males vs Norrie disease (NDP): NHS = lens opacity (red reflex absent); NDP = vitreous/retinal pseudoglioma",
            "NHS carrier posterior sutural vs normal lens: 70% carriers have subtle sutural opacities — slit-lamp examination mandatory",
            "NHS vs galactosaemia nuclear cataract: galactosaemia = urine reducing substances positive; NHS = XLR pedigree + dental",
            "Supplemental teeth in NHS vs hypodontia vs cleidocranial dysplasia: CCD = absent clavicles + multiple supernumerary teeth everywhere",
            "NHS intellectual disability vs X-linked intellectual disability syndromes: NHS = cataract + dental = distinguishing features",
        ],
        "morphology": "Dense total nuclear (hemizygous males); Posterior sutural (carrier females)",
        "systemic_involvement": True,
        "onset_age": "Congenital (birth) in hemizygous males",
        "surgical_urgency": "Extremely urgent (neonatal — bilateral dense nuclear in males)",
        "gene_family": "Actin Cytoskeleton Regulator / WH2 Domain Protein",
    },
    # -- FYCO1 -- FYVE+Coiled-Coil Autophagy Adaptor / Cataract-18 / AR -------------------
    {
        "gene": "FYCO1",
        "alt_name": (
            "FYCO1 (FYCO1-1478aa-3p21.31 / AR -- "
            "CATARACT-18-AR-FOUNDER-MUTATIONS-MIDDLE-EASTERN-SOUTH-ASIAN-CHINESE -- "
            "FYVE-COILED-COIL-DOMAIN-AUTOPHAGY-ADAPTOR -- "
            "TOTAL-NUCLEAR-CONGENITAL-NO-SYSTEMIC -- "
            "DISRUPTS-LENS-CRYSTALLIN-AUTOPHAGY-CLEARANCE)"
        ),
        "protein": (
            "FYCO1 -- 3p21.31 AR -- FYCO1-1478aa -- "
            "FYVE-and-Coiled-Coil-Domain-Autophagy-Adaptor-1-167kDa -- "
            "N-Terminal-FYVE-Domain-PI3P-Binding-Endosome/Lysosome-Targeting -- "
            "RUN-Domain-Small-GTPase-Interaction -- "
            "Coiled-Coil-Domain-Protein-Protein-Interactions-Kinesin-Recruitment -- "
            "C-Terminal-LIR-Motif-LC3-Interaction-Region-Autophagosome-Tethering -- "
            "Links-Autophagosomes-to-Microtubule-Motor-Kinesin-for-Anterograde-Transport -- "
            "OMIM-Gene-607182-Disease-CATARACT18-615763"
        ),
        "locus": "3p21.31",
        "protein_size": "1478 aa / 167 kDa",
        "inheritance": (
            "AR (autosomal recessive biallelic LOF); "
            "p.Gln762Ter (c.2284C>T): FOUNDER MUTATION in Pakistani/Punjabi consanguineous families (one of the most common AR hereditary cataract variants globally); "
            "p.Arg644Ter: founder mutation in Iranian/Middle Eastern consanguineous families; "
            "p.Arg1163Ter: East Asian (Chinese/Korean) consanguineous families; "
            "other nonsense/frameshift variants throughout gene in South Asian, Arab, Turkish consanguineous pedigrees; "
            "compound heterozygotes described in non-consanguineous European families; "
            "homozygosity mapping in consanguineous families has identified FYCO1 as one of the most common AR cataract genes worldwide; "
            "NO AD variants described — purely AR"
        ),
        "disease_category": "Cataract-18 — AR total nuclear congenital/early childhood cataract; FYCO1 autophagy adaptor defect; prominent in consanguineous Middle Eastern, South Asian, East Asian populations",
        "disease_pathway": (
            "FYCO1 encodes FYVE and Coiled-Coil Domain Autophagy Adaptor 1, a large scaffolding protein that links "
            "autophagosomes to microtubule-based kinesin motor proteins for anterograde transport. "
            "DOMAIN ARCHITECTURE: "
            "FYVE domain → binds phosphatidylinositol-3-phosphate (PI3P) on endosomal/autophagosomal membranes; "
            "RUN domain → interacts with Rab7 GTPase; "
            "Coiled-coil domain → recruits kinesin-1/kinesin-2 motor proteins; "
            "LIR (LC3-Interacting Region) motif → tethers to autophagosome coat protein LC3/GABARAP. "
            "FUNCTION IN LENS: "
            "Lens fiber cells accumulate crystallins throughout life; during terminal differentiation, "
            "fiber cells extrude organelles (mitochondria, ER, nucleus) via autophagy. "
            "FYCO1 is required for anterograde transport of autophagosomes containing damaged/aggregated crystallins "
            "toward cell periphery for degradation. "
            "PATHOMECHANISM: FYCO1 LOF → autophagosomes cannot be transported along microtubules → "
            "accumulation of damaged/aggregated crystallins in lens nucleus → NUCLEAR CATARACT. "
            "In mouse Fyco1 knockdown: nuclear opacity within 3 months confirming requirement for crystallin turnover. "
            "Unique mechanism: autophagy-mediated crystallin clearance — distinct from chaperone (CRYAA/CRYAB) or channel (GJA3/GJA8) defects."
        ),
        "pathognomonic": (
            "TOTAL NUCLEAR CATARACT in childhood/congenital — bilateral, dense, white nuclear opacity. "
            "NO SYSTEMIC INVOLVEMENT — pure lens disease (critical DDx from CRYAB which has cardiomyopathy). "
            "CONSANGUINITY: almost universal in affected families — parental consanguinity (first cousins) + "
            "multiple affected siblings → AR inheritance pattern. "
            "p.Gln762Ter HOMOZYGOUS: the most common AR cataract variant seen in Pakistani clinics — "
            "diagnose with a single Sanger or panel test in appropriate ethnic background. "
            "p.Arg644Ter: second most common — Middle Eastern consanguineous families. "
            "Onset: CONGENITAL or first year of life — dense white nuclear opacity; "
            "nystagmus from early visual deprivation; absent red reflex on ophthalmoscopy. "
            "No associated features: no intellectual disability (DDx NHS), no microphthalmia (DDx CRYAA AR), "
            "no dental anomalies (DDx NHS), no systemic disease. "
            "Fundal view blocked by dense nuclear opacity — B-scan ultrasound to exclude posterior segment pathology. "
            "PKU/metabolic screen: galactosaemia excluded (reducing substances urine negative). "
            "Molecular diagnosis: included in most inherited cataract gene panels; "
            "consanguineous South Asian family with nuclear cataract → FYCO1 first gene to test."
        ),
        "treatment": (
            "URGENT bilateral lens aspiration (lensectomy) + anterior vitrectomy + posterior capsulotomy — "
            "dense nuclear cataract in infants requires neonatal/early infantile surgery. "
            "Timing: within 4-6 weeks of birth for dense bilateral nuclear — "
            "delays beyond 10 weeks risk irreversible amblyopia from stimulus deprivation. "
            "IOL insertion: primary IOL at ≥3 months preferred if ≥7 dioptre cataract; "
            "aphakic contact lens + glasses first if <3 months. "
            "Optical correction: silicone/soft contact lenses (aphakia correction) + spectacles for near immediately post-surgery. "
            "Amblyopia: bilateral dense cataract — both eyes deprived equally; "
            "post-correction, if asymmetry develops: patching/atropine of better eye. "
            "Posterior capsule management: primary posterior capsulotomy + anterior vitrectomy in children <5 years "
            "(prevents PCO which is near-universal in paediatric cases without it). "
            "Genetic counselling: AR — 25% risk per subsequent pregnancy; "
            "prenatal diagnosis by CVS or amniocentesis available for families with known FYCO1 variant; "
            "preimplantation genetic testing (PGT-M) if available; "
            "screen all newborn siblings immediately after birth (dilated fundoscopy + red reflex)."
        ),
        "key_features": [
            "Most common AR hereditary cataract gene in consanguineous South Asian/Middle Eastern families",
            "p.Gln762Ter (c.2284C>T): Pakistani/Punjabi founder mutation — single test in appropriate ethnic background",
            "Autophagy adaptor: links autophagosomes to kinesin — unique mechanism (crystallin clearance, not chaperone/channel)",
            "Purely AR: no AD disease; consanguinity history almost universal",
            "No systemic features — pure nuclear lens opacity (critical DDx from CRYAB/NHS)",
        ],
        "key_ddx": [
            "FYCO1 AR nuclear vs NHS XLR nuclear: NHS = males only + dental + carrier female sutural; FYCO1 = both sexes + consanguinity",
            "FYCO1 vs GJA8 nuclear: GJA8 = AD (one parent affected); FYCO1 = AR (both parents normal carriers, consanguinity)",
            "FYCO1 vs galactosaemia: galactosaemia = reducing substances in urine + jaundice + hepatomegaly; FYCO1 = metabolically normal",
            "FYCO1 vs CRYAA AR: CRYAA AR has microphthalmia + iris coloboma; FYCO1 = normal eye size, no coloboma",
            "FYCO1 vs CRYAB: CRYAB = cardiomyopathy + myopathy; FYCO1 = no systemic disease",
        ],
        "morphology": "Total nuclear / Dense nuclear",
        "systemic_involvement": False,
        "onset_age": "Congenital / early infantile",
        "surgical_urgency": "Extremely urgent (bilateral dense nuclear — severe amblyopia risk)",
        "gene_family": "Autophagy Adaptor / FYVE Domain Protein",
    },
]


def _make_cohort(entry, seed):
    """Generate 40 deterministic synthetic patients for one hereditary cataract gene."""
    rng = random.Random(seed)
    gene = entry["gene"]
    patients = []

    for i in range(40):
        # Age at diagnosis (years)
        if gene in ("NHS", "FYCO1", "GJA8", "GJA3"):
            age_dx = rng.randint(0, 1)
        elif gene in ("CRYAA",):
            # AD cases childhood-adult; AR congenital — mix
            if rng.random() < 0.3:
                age_dx = rng.randint(0, 1)   # AR/congenital subset
            else:
                age_dx = rng.randint(5, 40)
        elif gene == "CRYAB":
            age_dx = rng.randint(15, 55)   # adult onset AD predominant
        elif gene == "MIP":
            age_dx = rng.randint(0, 10)    # congenital / early childhood
        elif gene == "EPHA2":
            if rng.random() < 0.15:
                age_dx = rng.randint(0, 2)  # AR biallelic congenital subset
            else:
                age_dx = rng.randint(25, 55)  # adult AD
        else:
            age_dx = rng.randint(0, 30)

        # Cataract morphology (encoded as primary feature)
        if gene == "CRYAA":
            morphology = rng.choice(["posterior_subcapsular", "zonular_lamellar", "nuclear"])
        elif gene == "CRYAB":
            morphology = rng.choice(["posterior_subcapsular", "cortical", "nuclear"])
        elif gene in ("GJA8",):
            morphology = rng.choice(["nuclear_pulverulent", "total_nuclear", "lamellar"])
        elif gene == "GJA3":
            morphology = rng.choice(["cerulean", "total_nuclear", "sutural"])
        elif gene == "MIP":
            morphology = rng.choice(["lamellar", "zonular", "sutural"])
        elif gene == "EPHA2":
            morphology = rng.choice(["cortical", "posterior_subcapsular", "cortical"])
        elif gene == "NHS":
            morphology = "total_nuclear"
        elif gene == "FYCO1":
            morphology = "total_nuclear"
        else:
            morphology = "nuclear"

        # Dense/visually significant cataract
        if gene in ("NHS", "FYCO1", "GJA8"):
            dense = rng.random() < 0.92
        elif gene == "GJA3":
            dense = rng.random() < 0.80
        elif gene == "MIP":
            dense = rng.random() < 0.55
        elif gene == "CRYAA":
            dense = rng.random() < 0.72
        elif gene == "CRYAB":
            dense = rng.random() < 0.68
        elif gene == "EPHA2":
            dense = rng.random() < 0.60
        else:
            dense = rng.random() < 0.70

        # Systemic involvement
        if gene == "CRYAB":
            cardiomyopathy = rng.random() < 0.60
            myopathy = rng.random() < 0.55
            intellectual_disability = False
        elif gene == "NHS":
            cardiomyopathy = False
            myopathy = False
            intellectual_disability = rng.random() < 0.40
        else:
            cardiomyopathy = False
            myopathy = False
            intellectual_disability = False

        # Dental anomalies (NHS)
        dental_anomalies = (gene == "NHS" and rng.random() < 0.88)

        # Surgery performed
        if gene in ("NHS", "FYCO1", "GJA8"):
            surgery = rng.random() < 0.95
        elif gene == "GJA3":
            surgery = rng.random() < 0.82
        elif gene == "CRYAB":
            surgery = dense and rng.random() < 0.78
        elif gene == "MIP":
            surgery = dense and rng.random() < 0.72
        else:
            surgery = dense and rng.random() < 0.80

        # Amblyopia post-treatment
        if gene in ("NHS", "FYCO1", "GJA8", "GJA3"):
            amblyopia = rng.random() < 0.35  # despite treatment
        else:
            amblyopia = rng.random() < 0.10

        # BCVA worse than 6/60 (not adequately correctable)
        if not surgery and dense:
            bcva_poor = rng.random() < 0.75
        elif surgery and amblyopia:
            bcva_poor = rng.random() < 0.22
        elif surgery and not amblyopia:
            bcva_poor = rng.random() < 0.05
        elif gene == "CRYAB" and (cardiomyopathy or myopathy):
            bcva_poor = rng.random() < 0.15
        else:
            bcva_poor = rng.random() < 0.08

        # Consanguinity (strong for FYCO1; rare others)
        consanguineous = (gene == "FYCO1" and rng.random() < 0.88) or \
                         (gene == "GJA3" and rng.random() < 0.12) or \
                         (gene == "CRYAA" and rng.random() < 0.08)

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis_years": age_dx,
            "morphology": morphology,
            "dense_cataract": dense,
            "surgery_performed": surgery,
            "amblyopia": amblyopia,
            "bcva_worse_than_6_60": bcva_poor,
            "cardiomyopathy": cardiomyopathy,
            "myopathy": myopathy,
            "intellectual_disability": intellectual_disability,
            "dental_anomalies": dental_anomalies,
            "consanguineous": consanguineous,
            "systemic_involvement": entry["systemic_involvement"],
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(CATARACT_GENES):
        all_patients.extend(_make_cohort(entry, SEED_BASE + idx))

    total = len(all_patients)
    dense_count = sum(1 for p in all_patients if p["dense_cataract"])
    surgery_count = sum(1 for p in all_patients if p["surgery_performed"])
    amblyopia_count = sum(1 for p in all_patients if p["amblyopia"])
    systemic_count = sum(1 for p in all_patients if p["systemic_involvement"])
    bcva_poor_count = sum(1 for p in all_patients if p["bcva_worse_than_6_60"])
    consanguineous_count = sum(1 for p in all_patients if p["consanguineous"])

    gene_summary = {}
    for idx, entry in enumerate(CATARACT_GENES):
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
            "dense_pct": round(100 * sum(1 for p in cohort if p["dense_cataract"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgery_performed"]) / len(cohort), 1),
            "amblyopia_pct": round(100 * sum(1 for p in cohort if p["amblyopia"]) / len(cohort), 1),
            "bcva_poor_pct": round(100 * sum(1 for p in cohort if p["bcva_worse_than_6_60"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Cataract-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Cataract Reference -- CRYAA/CRYAB/GJA8/GJA3/MIP/EPHA2/NHS/FYCO1",
        "genes_covered": [e["gene"] for e in CATARACT_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "dense_cataract_pct": round(100 * dense_count / total, 1),
            "surgery_performed_pct": round(100 * surgery_count / total, 1),
            "amblyopia_pct": round(100 * amblyopia_count / total, 1),
            "systemic_involvement_pct": round(100 * systemic_count / total, 1),
            "bcva_worse_than_6_60_pct": round(100 * bcva_poor_count / total, 1),
            "consanguineous_family_pct": round(100 * consanguineous_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(CATARACT_GENES):
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
            "dense_pct": round(100 * sum(1 for p in cohort if p["dense_cataract"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgery_performed"]) / len(cohort), 1),
            "amblyopia_pct": round(100 * sum(1 for p in cohort if p["amblyopia"]) / len(cohort), 1),
            "bcva_poor_pct": round(100 * sum(1 for p in cohort if p["bcva_worse_than_6_60"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
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
            for entry in CATARACT_GENES
        },
        "cataract_glossary": {
            "Congenital Cataract — Urgent vs Elective Surgical Timing": (
                "CONGENITAL CATARACT represents an ophthalmological emergency when dense and bilateral. "
                "VISUAL DEVELOPMENT CRITICAL PERIODS: "
                "0-2 months: most critical — form deprivation amblyopia develops within days-weeks if bilateral dense; "
                "2-12 months: high risk — early surgery essential; "
                "1-7 years: sensitive period — amblyopia still develops but slower; "
                "SURGICAL TIMING: "
                "Dense BILATERAL congenital cataract (NHS, FYCO1, GJA8 severe): surgery within 4-6 weeks of birth; "
                "Dense UNILATERAL: within 4-6 weeks if fellow eye at risk; "
                "Mild lamellar (MIP mild, EPHA2 mild): observe with refraction; "
                "OPTICAL REHABILITATION: "
                "Contact lens within days of surgery — silicone aphakic contact lens; "
                "Glasses overcorrection for near (+3.00 above distance); "
                "PRIMARY IOL: centre-dependent — ≥3 months for bilateral, ≥4 weeks for unilateral at experienced centres. "
                "POSTERIOR CAPSULOTOMY + VITRECTOMY: mandatory in children <7 years — "
                "100% PCO rate without it; posterior capsule fibrosis blocks visual rehabilitation."
            ),
            "Lens Anatomy — Zones of Hereditary Cataract": (
                "The lens is organized in concentric zones corresponding to developmental age of fiber cells: "
                "EMBRYONIC NUCLEUS (EY suture fetal): first formed, innermost — cataracts here present at birth (FYCO1, NHS, GJA8); "
                "FETAL NUCLEUS: Y suture (posterior) / upright Y (anterior) formed 7th-8th week gestation; "
                "INFANTILE/JUVENILE NUCLEUS: formed postnatally; "
                "ADULT CORTEX: continuously added from epithelium throughout life (site of age-related cortical spokes — EPHA2); "
                "POSTERIOR SUBCAPSULAR ZONE (PSC): just anterior to posterior capsule — last-differentiated fibers; "
                "most metabolically active zone; site of PSC from CRYAA, CRYAB, EPHA2. "
                "CLINICAL IMPLICATION: "
                "The zone of opacity = the developmental window when the gene was most critical; "
                "LAMELLAR (MIP) = specific cortical shell = postnatal period window; "
                "NUCLEAR (FYCO1) = embryonic nucleus = fetal development; "
                "PSC (EPHA2, CRYAA) = adult cortex differentiation ongoing issue."
            ),
            "Connexins in the Lens — Why GJA8 and GJA3 Are Both Essential": (
                "The lens is completely AVASCULAR — it relies on gap junction networks for nutrition, ion homeostasis, "
                "and waste transport (the 'microcirculation system'). "
                "TWO CONNEXINS co-expressed in lens fiber cells: "
                "GJA8 (Connexin 50, Cx50) — expressed in both epithelium and fiber cells; "
                "GJA3 (Connexin 46, Cx46) — expressed ONLY in fiber cells (not epithelium). "
                "REDUNDANCY vs SPECIALIZATION: "
                "GJA8-/- mouse: small cataract + microphthalmos (fiber cells shorter); "
                "GJA3-/- mouse: nuclear cataract by 2 months; "
                "GJA8-/-/GJA3-/- double knockout: complete nuclear opacity at birth — far worse than either alone. "
                "FUNCTIONAL SPECIALIZATION: "
                "Cx50 (GJA8): larger channels, ion metabolite transport to outermost fibers; "
                "Cx46 (GJA3): critical for Ca2+ homeostasis in inner cortical/nuclear fibers; "
                "Cx46 controls hemichannel-mediated Ca2+ flux — key for preventing calpain activation. "
                "HUMAN GENETICS: GJA8 and GJA3 variants phenocopy each other (both nuclear) — "
                "clinical distinction impossible without molecular testing."
            ),
            "Crystallin Chaperone System — CRYAA vs CRYAB": (
                "Alpha-crystallins (CRYAA and CRYAB) are the principal molecular chaperones of the lens, "
                "preventing aggregation of other crystallins and maintaining lens transparency. "
                "CRYAA (HspB4, αA-crystallin): "
                "LENS-SPECIFIC — highest expression in lens epithelium and fiber cells; "
                "~35% of total lens protein; "
                "forms large dynamic oligomers (24-mer average); "
                "chaperones beta and gamma crystallins; "
                "LOF → cataract only (no systemic disease); "
                "AR biallelic: adds microphthalmia + iris coloboma. "
                "CRYAB (HspB5, αB-crystallin): "
                "UBIQUITOUS — expressed in cardiac muscle, skeletal muscle, brain, lens; "
                "molecular chaperone for desmin IFs in cardiomyocytes; "
                "R120G dominant-negative: disrupts desmin chaperone → desmin aggregation → MFM2/DCM; "
                "cataract + cardiomyopathy + myopathy triad (R120G); "
                "LENS vs HEART distinction: CRYAA = lens only; CRYAB = lens + heart + muscle. "
                "PATHOGNOMONIC DISTINCTION: "
                "Any hereditary cataract + cardiomyopathy + myopathy → CRYAB first; "
                "Hereditary cataract + microphthalmia + iris coloboma → CRYAA AR first."
            ),
            "X-Linked Cataract — NHS vs Other XLR Eye Disease": (
                "Nance-Horan Syndrome (NHS gene, Xp22.13) is the most important XLR hereditary cataract. "
                "XLR CATARACT PEDIGREE RECOGNITION: "
                "Hemizygous males (XY): severely affected — dense nuclear cataract at birth; "
                "Carrier females (XX): mild posterior sutural opacities (non-dense); "
                "No male-to-male transmission (sons of affected male are unaffected); "
                "Daughters of affected males are ALL obligate carriers. "
                "NHS DISTINGUISHING FEATURES vs other XLR eye disease: "
                "vs Norrie Disease (NDP): Norrie = bilateral vitreous pseudoglioma (retinal/vitreous, NOT lens), "
                "NHS = lens opacity with clear vitreous; "
                "vs X-linked FEVR (NDP): similar — look at structure; NHS has lens, NDP has retinal dysplasia; "
                "vs X-linked RP (RPGR/RP2): RPGR = progressive retinal dystrophy, normal lens at birth; "
                "vs Fabry disease (GLA): corneal verticillata (not cataract), systemic angiokeratomas; "
                "vs OA1 (GPR143): ocular albinism, macromelanosomes, not cataract. "
                "NHS-SPECIFIC DDx CLUE: DENTAL ANOMALIES (supplemental incisors + screwdriver teeth) in "
                "NHS males — this combination of cataract + specific dental morphology = NHS until proven otherwise."
            ),
            "FYCO1 — Autophagy in the Lens and Founder Mutations": (
                "FYCO1 represents an emerging major cause of AR hereditary cataract in consanguineous populations. "
                "AUTOPHAGY IN LENS DEVELOPMENT: "
                "During terminal lens fiber cell differentiation, cells extrude all organelles "
                "(mitochondria, ribosomes, nuclei) via macroautophagy → this produces the transparent, "
                "organelle-free 'organelle-free zone' (OFZ) of the mature lens. "
                "FYCO1's role: links autophagosomes (containing damaged crystallins/organelles) to kinesin "
                "motors for anterograde microtubule transport → ensures autophagosomes reach lysosomes for degradation. "
                "LOF → crystallin accumulation → nuclear opacity. "
                "FOUNDER MUTATIONS (ethnicity-specific): "
                "c.2284C>T (p.Gln762Ter): Pakistani/Punjabi/South Asian — the most common variant; "
                "c.1930C>T (p.Arg644Ter): Iranian, Arab, Middle Eastern; "
                "c.3487C>T (p.Arg1163Ter): Chinese, Korean, East Asian; "
                "Turkish c.4075C>T (p.Arg1359Ter): Turkey and related populations. "
                "CLINICAL RULE: AR nuclear cataract + consanguinity + South Asian/Middle Eastern/East Asian background "
                "→ FYCO1 is the highest-priority gene — likely homozygous founder mutation. "
                "CARRIER TESTING: obligate carrier parents appear normal (no lens opacity) — "
                "confirm carrier status by molecular testing only."
            ),
            "Amblyopia Prevention in Congenital Cataract — The Race Against Visual Development": (
                "Amblyopia from congenital cataract is the most devastating preventable visual impairment in childhood. "
                "MECHANISM: dense lens opacity → form deprivation → "
                "failure of cortical visual neuron maturation → permanently reduced visual acuity even after cataract removed. "
                "CRITICAL WINDOW PRINCIPLES: "
                "Sensitive period: birth to ~10 years; most critical: birth to 18 months; "
                "BILATERAL dense cataract: more urgent than unilateral (both eyes at risk simultaneously); "
                "Deadline: dense bilateral cataract must be cleared within 4-6 weeks of birth or permanent amblyopia ensues. "
                "POST-SURGERY AMBLYOPIA MANAGEMENT: "
                "Contact lens immediately (within 48h of wound healing) — aphakic silicone lens; "
                "Glasses: +12.00–+14.00 for distance in infant aphakia; "
                "PATCHING: for asymmetric cases — patch better eye ≥2h/day in active period; "
                "ATROPINE penalization: 1% atropine to better eye 1-2x/week (for mild asymmetry); "
                "TARGET BCVA: 6/6 to 6/9 if surgery early + aggressive amblyopia treatment; "
                "Late-treated dense cataract: 6/18–6/60 at best; "
                "OUTCOME DETERMINANT: timing of surgery + optical rehabilitation + amblyopia compliance — "
                "genetics of underlying cause (NHS, FYCO1, GJA8) secondary to treatment speed."
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
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
