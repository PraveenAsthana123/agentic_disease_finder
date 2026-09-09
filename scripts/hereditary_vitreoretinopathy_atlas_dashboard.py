#!/usr/bin/env python3
"""Hereditary-Vitreoretinopathy-Atlas — Complete 8-Gene Hereditary Vitreoretinopathy Atlas
(COL2A1 · COL11A1 · VCAN · FZD4 · NDP · LRP5 · TSPAN12 · ZNF408).

COL2A1   (Collagen Type II Alpha 1 Chain; 1487 aa; ~141 kDa; 12q13.11; AD;
          Stickler Syndrome Type 1 (STL1) — MOST COMMON inherited vitreoretinopathy;
          TYPE 1 OPTICALLY EMPTY VITREOUS PATHOGNOMONIC on biomicroscopy;
          systemic: midface hypoplasia + cleft palate + sensorineural/conductive deafness + arthropathy;
          retinal detachment risk 30-70% lifetime; prophylactic 360° retinopexy;
          seed SEED_BASE+0).
COL11A1  (Collagen Type XI Alpha 1 Chain; 1837 aa; ~186 kDa; 1p21.1; AD;
          Stickler Syndrome Type 2 (STL2);
          TYPE 2 FIBRILLAR BEADED VITREOUS PATHOGNOMONIC — distinct from COL2A1 optically empty;
          more pronounced sensorineural hearing loss than Type 1; Marshall syndrome overlap;
          seed SEED_BASE+1).
VCAN     (Versican; 3396 aa; ~370 kDa; 5q14.2; AD;
          Wagner Syndrome / Wagner Vitreoretinopathy;
          VITREOUS SYNCHYSIS + FIBROVASCULAR VEILS PATHOGNOMONIC;
          NO SYSTEMIC INVOLVEMENT — pure ocular phenotype (DDx Stickler: no deafness, no arthropathy);
          progressive choroidal atrophy + pigmentary retinopathy;
          seed SEED_BASE+2).
FZD4     (Frizzled Class Receptor 4; 537 aa; ~57 kDa; 11q14.2; AD;
          FEVR Type 1 (EVR1) — MOST COMMON FEVR GENE;
          AVASCULAR PERIPHERAL RETINA ON FFA PATHOGNOMONIC;
          Wnt/beta-catenin retinal vascularization; DDx ROP (term babies, family history);
          Stage 1-5 spectrum from avascularity to total tractional RD;
          seed SEED_BASE+3).
NDP      (Norrin; 133 aa; ~15 kDa; Xp11.4; XLR;
          Norrie Disease / FEVR Type 2 (EVR2);
          CONGENITAL BILATERAL LEUKOCORIA IN BOYS PATHOGNOMONIC — born blind;
          sensorineural deafness 35%; intellectual disability / psychiatric 25-35%;
          most severe vitreoretinopathy in atlas;
          seed SEED_BASE+4).
LRP5     (LDL Receptor-Related Protein 5; 1615 aa; ~179 kDa; 11q13.2; AD/AR;
          FEVR Type 4 (EVR4) AD / Osteoporosis-Pseudoglioma Syndrome (OPPG) AR;
          PSEUDOGLIOMA + LOW BONE MASS COMBINATION PATHOGNOMONIC for AR biallelic LOF;
          AD heterozygous LOF → FEVR4; AD GOF → high bone density NO eye disease;
          seed SEED_BASE+5).
TSPAN12  (Tetraspanin 12; 305 aa; ~32 kDa; 7q31.31; AD;
          FEVR Type 5 (EVR5);
          INCOMPLETE PENETRANCE ~50% — carrier can be unaffected;
          Norrin-FZD4-LRP5-TSPAN12 complex; milder FEVR phenotype;
          avascular peripheral retina; fibrovascular proliferation in subset;
          seed SEED_BASE+6).
ZNF408   (Zinc Finger Protein 408; 720 aa; ~81 kDa; 11p11.2; AD;
          FEVR Type 6 (EVR6) / Persistent Fetal Vasculature (PFV/PHPV);
          PERSISTENT FETAL VASCULATURE + AVASCULAR PERIPHERAL RETINA PATHOGNOMONIC;
          PFV: failure of hyaloid vasculature regression → white pupil + lens abnormalities;
          most phenotypically variable FEVR gene;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2374-2381).
"""

import random

SEED_BASE = 2374

VITREORETINOPATHY_GENES = [
    # -- COL2A1 -- Stickler Syndrome Type 1 -----------------------------------------------
    {
        "gene": "COL2A1",
        "alt_name": (
            "COL2A1 (COL2A1-1487aa-12q13.11 / AD -- "
            "STICKLER-SYNDROME-TYPE-1-MOST-COMMON-INHERITED-VITREORETINOPATHY -- "
            "TYPE-1-OPTICALLY-EMPTY-VITREOUS-PATHOGNOMONIC -- "
            "MIDFACE-HYPOPLASIA+CLEFT-PALATE+DEAFNESS+ARTHROPATHY-SYSTEMIC-TRIAD -- "
            "RETINAL-DETACHMENT-30-70pct-LIFETIME-PROPHYLACTIC-RETINOPEXY)"
        ),
        "protein": (
            "COL2A1 -- 12q13.11 AD -- COL2A1-1487aa -- "
            "Collagen-Type-II-Alpha-1-Chain-141kDa-Major-Hyaline-Cartilage-Vitreous-Collagen -- "
            "Triple-Helix-Gly-X-Y-Repeat-Domain-1400aa-N-Terminal-Propeptide-C-Terminal-Propeptide -- "
            "Expressed-in-Vitreous-Articular-Cartilage-Nucleus-Pulposus-Cochlear-Matrix-Fetal-Chondrocyte -- "
            "Dominant-Negative-Missense-Variants-Disrupt-Triple-Helix-or-Early-Truncating-LOF -- "
            "Vitreous-Scaffold-Degraded-by-Pathogenic-COL2A1-Collagen-II-Absent-Vitreous-Gel -- "
            "OMIM-Gene-120140-Disease-STL1-108300"
        ),
        "locus": "12q13.11",
        "protein_size": "1487 aa / 141 kDa",
        "inheritance": (
            "AD (autosomal dominant); dominant-negative or haploinsufficiency; COL2A1 most common Stickler gene; "
            "~80% of Stickler syndrome is COL2A1; variable expressivity within families; "
            "rare AR Stickler reported (biallelic COL2A1) — more severe; "
            "splice-site variants affecting exon 2 (alternatively spliced) associated with non-ocular Stickler (no vitreous phenotype but systemic present)"
        ),
        "disease_category": "Stickler Syndrome Type 1 (STL1) — Hereditary Progressive Arthro-ophthalmopathy; most common inherited vitreoretinopathy worldwide",
        "disease_pathway": (
            "COL2A1 encodes the alpha-1 chain of type II collagen, the principal structural collagen of hyaline cartilage and the vitreous gel. "
            "In the vitreous, collagen II forms a scaffold with collagen IX and XI and hyaluronan that maintains the gel state. "
            "Pathogenic variants cause dominant-negative interference with triple-helix assembly → collagen II-depleted vitreous → "
            "TYPE 1 VITREOUS PHENOTYPE: complete gel collapse (synchysis) leaving an OPTICALLY EMPTY CENTRAL VITREOUS space surrounded by a membrane. "
            "The collagen-depleted gel cannot support the retina → vitreoretinal traction → giant retinal tears → rhegmatogenous retinal detachment. "
            "Systemic manifestations arise from the same COL2A1 protein in cartilage (epiphyseal dysplasia, spondyloepiphyseal features), "
            "sensorineural and conductive hearing loss (cochlear collagen II), and midface structures (cleft palate, micrognathia, flat midface). "
            "Robin sequence (micrognathia + tongue base obstruction + cleft palate) is the classic neonatal presentation."
        ),
        "pathognomonic": (
            "TYPE 1 OPTICALLY EMPTY VITREOUS on slit-lamp or wide-field biomicroscopy PATHOGNOMONIC for COL2A1 Stickler. "
            "The central vitreous cavity appears clear/empty with a pre-retinal membrane at the vitreoretinal interface. "
            "SYSTEMIC TRIAD (midface hypoplasia + cleft palate/submucous cleft + sensorineural or mixed deafness) present in ~90%. "
            "JOINT HYPERMOBILITY in childhood followed by premature osteoarthritis in adults. "
            "High myopia (often >-6D) from birth; vitreous base condensation. "
            "RETINAL DETACHMENT (30-70% lifetime): giant retinal tears most common type; "
            "often bilateral; inferior detachments from inferior vitreous base pathology. "
            "Robin sequence in neonates: micrognathia + glossoptosis + U-shaped cleft palate (feeding difficulty, "
            "airway compromise at birth — multidisciplinary neonatal management mandatory). "
            "Hearing: mixed conductive (ossicular hypoplasia) + sensorineural — audiometry mandatory at diagnosis."
        ),
        "treatment": (
            "Prophylactic retinal laser photocoagulation (360° retinopexy around vitreous base condensations and retinal breaks) "
            "is strongly recommended — reduces lifetime retinal detachment risk; should be performed early and repeated at each new lesion. "
            "Annual ophthalmological review for retinal breaks from childhood. "
            "Retinal detachment surgery: scleral buckle + vitrectomy + silicone oil or C3F8 tamponade; "
            "giant retinal tears require vitrectomy + heavy liquid + long-term tamponade. "
            "Myopia: spectacles/contact lenses; avoid contact sports. "
            "Hearing: hearing aids; ENT review; cochlear implant if severe SNHL. "
            "Cleft palate: palatoplasty within first year of life; speech therapy. "
            "Arthropathy: low-impact exercise, physiotherapy, joint replacement in adulthood. "
            "Multidisciplinary team: ophthalmology, audiology, ENT, oral surgery, rheumatology, genetics."
        ),
        "key_features": [
            "Most common inherited vitreoretinopathy — ~80% of Stickler due to COL2A1",
            "Type 1 optically empty vitreous: pathognomonic on biomicroscopy",
            "Retinal detachment 30-70% lifetime: prophylactic retinopexy mandatory",
            "Robin sequence neonates: micrognathia + glossoptosis + cleft palate — airway emergency",
            "Multidisciplinary: ophthalmology + ENT + audiology + rheumatology + genetics",
        ],
        "key_ddx": [
            "COL2A1 vs COL11A1 Stickler: Type 1 optically empty vitreous (COL2A1) vs Type 2 fibrillar beaded vitreous (COL11A1)",
            "Stickler vs Wagner: Stickler has systemic features (deafness, arthropathy, midface); Wagner pure ocular",
            "Stickler vs Marfan: Marfan = fibrillin-1, aortic dilation, tall; Stickler = collagen II, myopia dominant, midface hypoplasia",
            "Giant retinal tear from COL2A1 vs trauma: bilateral family history in Stickler",
            "Stickler vs Marshall syndrome: Marshall has more severe craniofacial, calcifications; both COL11A1",
        ],
        "vitreous_phenotype": "Type 1 — optically empty central vitreous",
        "systemic_involvement": True,
        "retinal_detachment_risk_pct": "30–70%",
        "hearing_loss_type": "Mixed (sensorineural + conductive)",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "Congenital (myopia, vitreous) + progressive; RD in 2nd–4th decade",
        "gene_family": "Fibrillar Collagen",
    },
    # -- COL11A1 -- Stickler Syndrome Type 2 / Marshall Syndrome ----------------------------
    {
        "gene": "COL11A1",
        "alt_name": (
            "COL11A1 (COL11A1-1837aa-1p21.1 / AD -- "
            "STICKLER-SYNDROME-TYPE-2 -- "
            "TYPE-2-FIBRILLAR-BEADED-VITREOUS-PATHOGNOMONIC-DDx-COL2A1 -- "
            "EARLIER-MORE-SEVERE-SNHL-THAN-TYPE-1 -- "
            "MARSHALL-SYNDROME-OVERLAP-COL11A1)"
        ),
        "protein": (
            "COL11A1 -- 1p21.1 AD -- COL11A1-1837aa -- "
            "Collagen-Type-XI-Alpha-1-Chain-186kDa-Heterotrimeric-Collagen-with-COL11A2-COL2A1 -- "
            "Expressed-Vitreous-Cartilage-Cochlea-Nucleus-Pulposus -- "
            "Controls-Fibril-Diameter-of-Collagen-II-in-Vitreous-and-Cartilage -- "
            "Dominant-Negative-Missense-Disrupts-Triple-Helix-Type-2-Vitreous-Phenotype -- "
            "OMIM-Gene-120280-Disease-STL2-604841"
        ),
        "locus": "1p21.1",
        "protein_size": "1837 aa / 186 kDa",
        "inheritance": (
            "AD (autosomal dominant); dominant-negative; Type 2 Stickler ~10-15% of all Stickler; "
            "Marshall syndrome (more severe craniofacial phenotype + intracranial calcifications) also caused by COL11A1 splice variants; "
            "COL11A2 (same chromosome 6p21.32) causes non-ocular Stickler (no vitreous because COL11A2 not expressed in vitreous); "
            "COL9A1/A2/A3 biallelic → AR Stickler without typical vitreous phenotype"
        ),
        "disease_category": "Stickler Syndrome Type 2 (STL2) — fibrillar vitreous type; Marshall Syndrome (COL11A1 splice variants with more severe craniofacial)",
        "disease_pathway": (
            "COL11A1 encodes the alpha-1 chain of type XI collagen, which forms a heterotrimeric collagen [alpha1(XI), alpha2(XI), alpha1(II)] "
            "in cartilage and [alpha1(XI), alpha1(XI), alpha1(II)] in vitreous. "
            "Type XI collagen is a regulatory collagen — it controls fibril diameter of type II collagen in cartilage and vitreous. "
            "Pathogenic COL11A1 variants → TYPE 2 VITREOUS PHENOTYPE: "
            "vitreous retains fibrous/beaded architecture (not optically empty as in COL2A1) → "
            "sparse fibrillar collagen remnants visible on biomicroscopy as a 'beaded' or 'membranous but fibrillar' pattern. "
            "Cochlear collagen XI deficiency → more prominent sensorineural hearing loss than STL1. "
            "Marshall syndrome (same gene): specific COL11A1 splice variants cause exon skipping → "
            "more severe midface hypoplasia, intracranial calcifications, ectodermal dysplasia features."
        ),
        "pathognomonic": (
            "TYPE 2 FIBRILLAR BEADED VITREOUS on biomicroscopy PATHOGNOMONIC — "
            "distinct sparse fibrillar remnants visible (DDx COL2A1 Type 1 optically empty vitreous). "
            "SENSORINEURAL HEARING LOSS: typically more severe and earlier onset than Type 1 Stickler; "
            "profound SNHL possible in childhood; mandatory audiometry. "
            "Systemic: midface hypoplasia, flat nasal bridge, micrognathia, cleft palate (Robin sequence possible), joint laxity, premature arthritis. "
            "High myopia from birth; lattice degeneration and retinal tears. "
            "MARSHALL SYNDROME variant: hypohidrotic ectodermal features + intracranial calcifications + "
            "more severe midface — Hutchinson-Gilford progeroid facial appearance in some; "
            "COL11A1 splice variants causing exon 50/52 skipping typical. "
            "Cataracts: cortical and perinuclear more common than STL1. "
            "Retinal detachment risk similar to COL2A1 — prophylactic laser mandatory."
        ),
        "treatment": (
            "Same vitreoretinal management as STL1: prophylactic 360° retinopexy for vitreous base condensations and retinal breaks. "
            "Annual retinal surveillance from childhood. "
            "Hearing: early hearing aids critical — cochlear implant for profound SNHL; "
            "hearing loss may be more rapidly progressive than STL1 — 6-monthly audiometry. "
            "Cleft palate / Robin sequence: same multidisciplinary neonatal management. "
            "Arthropathy management. "
            "Marshall syndrome: additional monitoring for intracranial calcifications, "
            "ectodermal features (dental anomalies, hypohidrosis) — dermatology + neurology."
        ),
        "key_features": [
            "Type 2 fibrillar beaded vitreous: visible on biomicroscopy (vs COL2A1 optically empty)",
            "More severe/earlier sensorineural hearing loss than Stickler Type 1",
            "Marshall syndrome: same gene, more severe craniofacial + intracranial calcifications",
            "COL11A2 (6p21) causes non-ocular Stickler — no vitreous disease (COL11A2 not in vitreous)",
            "Prophylactic retinopexy mandatory — same RD risk as Type 1",
        ],
        "key_ddx": [
            "COL11A1 vs COL2A1 Stickler: fibrillar vitreous (COL11A1) vs optically empty (COL2A1)",
            "Marshall vs Stickler Type 2: same COL11A1 gene, exon-skip splice variants → ectodermal features + calcifications",
            "COL11A1 vs COL11A2: COL11A2 Stickler has NO vitreous/ocular phenotype (COL11A2 not expressed in vitreous)",
            "COL11A1 hearing loss vs connexin 26 (GJB2): STL2 has ocular + joint + midface features",
        ],
        "vitreous_phenotype": "Type 2 — fibrillar beaded vitreous",
        "systemic_involvement": True,
        "retinal_detachment_risk_pct": "25–60%",
        "hearing_loss_type": "Sensorineural (typically more severe than STL1)",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "Congenital high myopia; progressive vitreous; RD 2nd–4th decade",
        "gene_family": "Fibrillar Collagen",
    },
    # -- VCAN -- Wagner Syndrome -------------------------------------------------------------
    {
        "gene": "VCAN",
        "alt_name": (
            "VCAN (VCAN-3396aa-5q14.2 / AD -- "
            "WAGNER-SYNDROME-PURE-OCULAR-VITREORETINOPATHY -- "
            "VITREOUS-SYNCHYSIS+FIBROVASCULAR-VEILS-PATHOGNOMONIC -- "
            "NO-SYSTEMIC-INVOLVEMENT-KEY-DDx-FROM-STICKLER -- "
            "PROGRESSIVE-CHOROIDAL-ATROPHY-PIGMENTARY-RETINOPATHY)"
        ),
        "protein": (
            "VCAN -- 5q14.2 AD -- VCAN-3396aa -- "
            "Versican-Chondroitin-Sulfate-Proteoglycan-370kDa-Core-Protein-EGF-Lectins-Hyaluronan-Binding -- "
            "Expressed-in-Vitreous-Neural-Retina-Cornea-Not-in-Cartilage-Cochlea -- "
            "Splicing-Variants-Intronic-Donor-Acceptor-Cause-Exon-Skipping -- "
            "VCAN-V0-V1-V2-V3-Isoforms-Vitreous-V0-V1-V2 -- "
            "Aberrant-Versican-Disrupts-Vitreous-Matrix-Organization-Collagen-Fibril-Spacing -- "
            "OMIM-Gene-118661-Disease-Wagner-143200"
        ),
        "locus": "5q14.2",
        "protein_size": "3396 aa / 370 kDa (core protein; with GAG chains ~1000 kDa)",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "most variants are splice-site (intron 7/8 donor/acceptor) → VCAN exon skipping → isoform imbalance; "
            "variable expressivity: same variant → severe tractional RD in one family member, mild avascularity in another; "
            "incomplete penetrance documented in some families"
        ),
        "disease_category": "Wagner Syndrome (Autosomal Dominant Vitreoretinal Degeneration) — pure ocular; no systemic features (DDx from Stickler)",
        "disease_pathway": (
            "VCAN encodes versican, a large extracellular matrix proteoglycan found in the vitreous as isoforms V0/V1/V2. "
            "Versican interacts with hyaluronan, fibronectin, and collagen fibrils to organize vitreous architecture. "
            "Pathogenic splice variants alter the ratio of versican isoforms → disorganized vitreous matrix → "
            "vitreous liquefaction (synchysis) followed by condensation into fibrovascular veils. "
            "Unlike Stickler (systemic collagen II), VCAN is NOT expressed in cartilage or cochlea → "
            "Wagner is a PURE OCULAR disease. "
            "Progressive natural history: vitreous synchysis → fibrovascular pre-retinal membranes (veils) → "
            "vitreoretinal traction → tractional retinal detachment OR rhegmatogenous from tears at vitreous base. "
            "Choroid: progressive pericentral choroidal atrophy → relative scotomata; pigmentary changes. "
            "Cataract: nuclear sclerosis, perinuclear cortical opacities develop in adulthood."
        ),
        "pathognomonic": (
            "VITREOUS SYNCHYSIS (complete vitreous liquefaction with collapse) AND "
            "PRE-RETINAL FIBROVASCULAR VEILS (condensed fibrovascular membranes within liquefied vitreous) PATHOGNOMONIC on biomicroscopy. "
            "NO SYSTEMIC FEATURES — critical DDx from Stickler: "
            "Wagner patients have NORMAL hearing, NORMAL joints/arthropathy, NORMAL palate, NORMAL face. "
            "PROGRESSIVE CHOROIDAL ATROPHY: pericentral distribution → irregular RPE atrophy, "
            "choroidal vessel visibility, relative scotomata; not RP-like peripheral ring scotoma. "
            "PIGMENTARY RETINOPATHY: bone-spicule pigment deposits at equator and posterior pole. "
            "Cataract: nuclear sclerosis + perinuclear cortical opacities — earlier than general population. "
            "Retinal detachment (20-50% lifetime): tractional (pre-retinal membranes) and rhegmatogenous. "
            "Myopia: mild to moderate; less severe than Stickler. "
            "Electroretinogram (ERG): reduced amplitudes — photoreceptor dysfunction."
        ),
        "treatment": (
            "Vitreoretinal surgery for symptomatic tractional or rhegmatogenous retinal detachment: "
            "vitrectomy with membrane peeling for tractional component; combined scleral buckle + vitrectomy for complex cases. "
            "Preventive laser: photocoagulate retinal breaks and thin areas but fibrovascular veils can tether retina. "
            "Cataract surgery when visually significant; combined phaco-vitrectomy if vitreoretinal procedure also needed. "
            "Low vision: magnifiers, eccentric viewing training for choroidal atrophy-related scotomata. "
            "Annual surveillance: retina, ERG, visual fields for choroidal atrophy progression. "
            "No disease-modifying therapy; anti-VEGF not indicated (fibrovascular veils are not neo-vascularisation in same sense as DR/AMD). "
            "Genetic counselling: 50% offspring risk; molecular confirmation by VCAN splice-site sequencing."
        ),
        "key_features": [
            "Pure ocular disease — no systemic features (DDx key from Stickler)",
            "Vitreous synchysis + fibrovascular veils: pathognomonic biomicroscopy findings",
            "Progressive choroidal atrophy + pigmentary retinopathy → reduced ERG amplitudes",
            "Most variants are splice-site (intron 7/8) → VCAN isoform imbalance",
            "Cataract earlier than general population: nuclear sclerosis + perinuclear cortical",
        ],
        "key_ddx": [
            "Wagner vs Stickler: no systemic (no deafness, no arthropathy, no midface) in Wagner — KEY DDx",
            "Wagner vs enhanced S-cone syndrome (NR2E3): ERG pattern (supernormal S-cone in NR2E3)",
            "Wagner vs Goldmann-Favre syndrome (NR2E3): similar but more severe, schisis-like changes",
            "Wagner VCAN vs FEVR (FZD4): FEVR has avascular peripheral retina on FFA; Wagner has veils",
            "Wagner vs PVR (proliferative vitreoretinopathy): PVR post-traumatic or post-RRD, not hereditary",
        ],
        "vitreous_phenotype": "Synchysis (complete liquefaction) + fibrovascular veils",
        "systemic_involvement": False,
        "retinal_detachment_risk_pct": "20–50%",
        "hearing_loss_type": "None (pure ocular)",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "2nd–4th decade for vitreous changes; choroidal atrophy progressive lifetime",
        "gene_family": "Extracellular Matrix Proteoglycan",
    },
    # -- FZD4 -- FEVR Type 1 -----------------------------------------------------------------
    {
        "gene": "FZD4",
        "alt_name": (
            "FZD4 (FZD4-537aa-11q14.2 / AD -- "
            "FEVR-TYPE-1-EVR1-MOST-COMMON-FEVR-GENE -- "
            "AVASCULAR-PERIPHERAL-RETINA-ON-FFA-PATHOGNOMONIC -- "
            "WNT-BETA-CATENIN-RETINAL-VASCULARIZATION -- "
            "STAGE-1-AVASCULARITY-TO-STAGE-5-TOTAL-RD-SPECTRUM)"
        ),
        "protein": (
            "FZD4 -- 11q14.2 AD -- FZD4-537aa -- "
            "Frizzled-Class-Receptor-4-57kDa-7-TM-GPCR-Wnt-Pathway -- "
            "Cysteine-Rich-Domain-CRD-N-Terminal-Norrin-Binding-Domain -- "
            "Norrin-High-Affinity-Ligand-NDP-Gene-Product-Activates-FZD4 -- "
            "FZD4-LRP5-TSPAN12-Complex-Canonical-Wnt-Signaling-Retinal-Vasculogenesis -- "
            "LOF-or-Dominant-Negative-Impairs-Retinal-Vascularization-Peripheral-Avascularity -- "
            "OMIM-Gene-604579-Disease-EVR1-133780"
        ),
        "locus": "11q14.2",
        "protein_size": "537 aa / 57 kDa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency or dominant-negative; "
            "most common FEVR gene — ~40-50% of all FEVR families; "
            "variable expressivity: same variant → Stage 5 total RD in one sibling, "
            "asymptomatic avascularity on FFA in another; "
            "AR FZD4 biallelic: more severe, early-onset bilateral tractional RD; "
            "penetrance near complete but expressivity wide"
        ),
        "disease_category": "FEVR Type 1 (EVR1) — Familial Exudative Vitreoretinopathy type 1; most common FEVR gene",
        "disease_pathway": (
            "FZD4 is a Wnt receptor critical for canonical Wnt/beta-catenin signaling in retinal vascular development. "
            "During fetal development, the Norrin (NDP)-FZD4-LRP5-TSPAN12 receptor complex activates Wnt/beta-catenin signaling "
            "to drive angiogenic sprouting into the peripheral retina. "
            "Loss-of-function FZD4 variants → Wnt signaling deficiency → "
            "peripheral retinal vessels fail to reach the ora serrata → "
            "AVASCULAR PERIPHERAL RETINA (temporal > nasal, unilateral or bilateral). "
            "The avascular zone develops abnormal fibrovascular proliferation from the vascular-avascular junction → "
            "exudation (like ROP) → traction → falciform folds → retinal detachment. "
            "Spectrum (Stages 1-5): Stage 1 = avascularity only; Stage 2 = extraretinal fibrovascular proliferation; "
            "Stage 3 = ridge + plus disease; Stage 4 = subtotal RD (exudative or tractional); "
            "Stage 5 = total RD + phthisis."
        ),
        "pathognomonic": (
            "AVASCULAR PERIPHERAL RETINA ON FUNDUS FLUORESCEIN ANGIOGRAPHY (FFA) PATHOGNOMONIC — "
            "temporal retina most severely affected; avascular zone extends from temporal periphery; "
            "the vascular-avascular boundary is sharp on FFA with leakage at the junction. "
            "DDx from ROP: FEVR occurs in TERM BABIES without prematurity history; bilateral family history. "
            "STAGE-SPECIFIC FEATURES: "
            "Stage 1: avascular periphery only, asymptomatic; "
            "Stage 2A: exudative fibrovascular proliferation without RD; "
            "Stage 2B: same with RD risk zone; "
            "Stage 3: extraretinal fibrovascular proliferation (ridge) + plus disease; "
            "Stage 4A: subtotal tractional RD, macula on; "
            "Stage 4B: subtotal tractional RD, macula off; "
            "Stage 5: total RD, funnel configuration; phthisis bulbi end-stage. "
            "ASYMPTOMATIC CARRIERS: FFA reveals avascularity in family members with normal visual acuity — "
            "screen all first-degree relatives. "
            "Ectopia lentis not present (DDx Marfan). "
            "Dragged disc / straightened retinal vessels toward temporal periphery on fundoscopy."
        ),
        "treatment": (
            "Stage 1-2: laser photocoagulation of avascular peripheral retina to prevent fibrovascular progression; "
            "same principle as ROP laser. "
            "Stage 3: laser + anti-VEGF injection (bevacizumab/ranibizumab off-label) to regress neovascularization. "
            "Stage 4: vitreoretinal surgery — vitrectomy + membrane peeling ± scleral buckle for tractional RD; "
            "silicone oil tamponade for severe cases. "
            "Stage 5: very limited prognosis; surgery may restore some light perception. "
            "Family screening: FFA of ALL first-degree relatives — asymptomatic avascularity identified → "
            "prophylactic laser prevents progression to tractional RD. "
            "Anti-VEGF as adjunct, NOT as monotherapy. "
            "No regenerative retinal vascularization therapy available yet. "
            "Genetic counselling: 50% risk to offspring."
        ),
        "key_features": [
            "Most common FEVR gene (~40-50% of FEVR families)",
            "Avascular peripheral retina on FFA pathognomonic: term baby DDx from ROP",
            "Same variant → wide expressivity: asymptomatic to total RD in siblings",
            "Screening all first-degree relatives by FFA mandatory — asymptomatic carriers treated",
            "Norrin-FZD4-LRP5-TSPAN12: four-gene Wnt complex, all cause FEVR",
        ],
        "key_ddx": [
            "FEVR vs ROP: FEVR in term babies, family history, bilateral (not necessarily symmetric); ROP premature + O2",
            "FEVR FZD4 vs NDP (Norrie): NDP = X-linked boys, born blind (leukocoria); FZD4 = AD, variable onset",
            "FEVR vs persistent fetal vasculature (PFV/ZNF408): PFV has persistent hyaloid, lens changes, unilateral often",
            "FEVR vs Coats disease: Coats unilateral, no family history, telangiectatic vessels",
            "FEVR Stage 4-5 vs retinoblastoma: FEVR family history, bilateral often; Rb genetic testing",
        ],
        "vitreous_phenotype": "Vitreous traction from fibrovascular proliferation at vascular-avascular junction",
        "systemic_involvement": False,
        "retinal_detachment_risk_pct": "10–40% (stage-dependent)",
        "hearing_loss_type": "None",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "Congenital to infancy; severe cases present neonatal; mild may be found on family screening in adulthood",
        "gene_family": "Wnt Receptor / Frizzled",
    },
    # -- NDP -- Norrie Disease / FEVR Type 2 ------------------------------------------------
    {
        "gene": "NDP",
        "alt_name": (
            "NDP (NDP-133aa-Xp11.4 / XLR -- "
            "NORRIE-DISEASE-FEVR-TYPE-2-EVR2 -- "
            "CONGENITAL-BILATERAL-LEUKOCORIA-BOYS-BIRTH-PATHOGNOMONIC -- "
            "SENSORINEURAL-DEAFNESS-35pct -- "
            "INTELLECTUAL-DISABILITY-PSYCHIATRIC-25-35pct)"
        ),
        "protein": (
            "NDP -- Xp11.4 XLR -- NDP-133aa -- "
            "Norrin-15kDa-Cystine-Knot-Motif-Wnt-Pathway-Ligand -- "
            "High-Affinity-FZD4-Binding-Norrin-FZD4-LRP5-Complex -- "
            "Expressed-Retinal-Müller-Cells-Inner-Nuclear-Layer -- "
            "Norrin-Activates-Canonical-Wnt-Signaling-via-FZD4-not-via-Wnt-Proteins -- "
            "LOF-Complete-Absence-of-Retinal-Vascularization-Pseudoglioma -- "
            "OMIM-Gene-310600-Disease-ND-310600"
        ),
        "locus": "Xp11.4",
        "protein_size": "133 aa / 15 kDa",
        "inheritance": (
            "XLR (X-linked recessive); hemizygous males severely affected; "
            "carrier females: usually unaffected but may have mild FEVR-like FFA changes (peripheral avascularity); "
            "de novo NDP mutations account for 10-15% (no family history); "
            "severe phenotype: total blindness from birth in affected males; "
            "FEVR Type 2 (EVR2): carrier females with FEVR-like findings; some hemizygous males with milder FEVR rather than classic Norrie"
        ),
        "disease_category": "Norrie Disease (ND) — X-linked congenital blindness with systemic features; FEVR Type 2 (EVR2) in carrier females and mild-variant hemizygous males",
        "disease_pathway": (
            "Norrin is a 133-amino acid secreted cystine-knot protein expressed by Müller glia in the retina. "
            "Norrin acts as the HIGH-AFFINITY LIGAND for FZD4, activating canonical Wnt/beta-catenin signaling — "
            "the same pathway as FZD4, LRP5, and TSPAN12 but upstream (ligand vs receptor). "
            "NDP LOF → Norrin absent → FZD4 receptor not activated → complete absence of retinal vascularization → "
            "retinal ischemia → fibrovascular proliferation WITHIN the vitreous (not pre-retinal) → "
            "pseudogliomatous vitreous mass behind the lens (leukocoria). "
            "Additionally: cochlear vascularization requires Norrin-FZD4 signaling → "
            "NDP LOF → cochlear stria vascularis degeneration → progressive SNHL. "
            "Central nervous system: Norrin expressed in brain → cognitive/psychiatric phenotype in subset."
        ),
        "pathognomonic": (
            "CONGENITAL BILATERAL LEUKOCORIA (WHITE PUPIL REFLEX) IN MALE INFANTS PATHOGNOMONIC for Norrie Disease. "
            "Unlike retinoblastoma (Rb): Norrie is BILATERAL, family history present (X-linked), NO calcification on US. "
            "Ophthalmoscopy: both globes contain retrolental fibrovascular mass (pseudoglioma) filling vitreous cavity; "
            "retina folded into central mass; total retinal detachment present at birth. "
            "BORN BLIND: unlike FEVR which presents with mild peripheral avascularity, Norrie = congenital total blindness. "
            "SENSORINEURAL DEAFNESS: progressive, affecting 35% of hemizygous males; "
            "audiometry from infancy; may worsen in adulthood. "
            "INTELLECTUAL DISABILITY AND PSYCHIATRIC FEATURES (25-35%): ASD traits, psychosis, aggression; "
            "not universal — phenotypic variability by NDP variant. "
            "CARRIER FEMALES: FFA may reveal peripheral retinal avascularity (FEVR-like); visual acuity usually normal. "
            "Progressive phthisis: globes shrink over years; enucleation often performed for cosmesis."
        ),
        "treatment": (
            "Ophthalmology: unfortunately, retinal surgery rarely restores useful vision in classic Norrie Disease — "
            "globes are typically phthisical by adulthood. "
            "Vitreoretinal surgery (vitrectomy + membrane peel) in MILD NDP variants occasionally preserves light perception. "
            "Low vision services and blind rehabilitation from birth. "
            "Prosthetic eyes: custom ocular prostheses for enucleated/phthisical globes. "
            "Hearing: hearing aids from infancy for sensorineural deafness; cochlear implant in severe SNHL — "
            "dramatically improves quality of life even in blind + deaf patients. "
            "Neurological/cognitive: educational support, behavioural therapy, neuropsychiatric assessment. "
            "Carrier female screening: FFA + ophthalmological review; laser if avascular zones identified. "
            "Genetic counselling: X-linked — 50% of sons affected, 50% of daughters carrier."
        ),
        "key_features": [
            "Congenital bilateral leukocoria in boys: most severe vitreoretinopathy in this atlas",
            "Born blind: pseudogliomatous fibrovascular mass fills vitreous at birth",
            "X-linked: carrier females have mild FEVR-like FFA changes, usually normal VA",
            "Sensorineural deafness (35%) + intellectual disability/psychiatric (25-35%) — triad possible",
            "Norrin = upstream FZD4 ligand: NDP/FZD4/LRP5/TSPAN12 = same Wnt retinal pathway",
        ],
        "key_ddx": [
            "Norrie vs retinoblastoma: bilateral Norrie has X-linked family history, no US calcification",
            "Norrie vs FEVR FZD4: Norrie congenital total blindness; FEVR variable/mild in term babies",
            "Norrie vs persistent fetal vasculature (PFV): PFV usually unilateral, not XLR",
            "Norrie vs congenital cataract: leukocoria — ophthalmoscopy distinguishes (lens vs vitreous)",
            "NDP carrier vs ROP: carrier female FFA avascularity in term babies with no prematurity",
        ],
        "vitreous_phenotype": "Total fibrovascular obliteration — pseudoglioma (retrolental mass)",
        "systemic_involvement": True,
        "retinal_detachment_risk_pct": "~100% (total RD at birth in classic Norrie)",
        "hearing_loss_type": "Sensorineural (35% of hemizygous males, progressive)",
        "prophylactic_retinopexy_indicated": False,
        "onset_age": "Congenital — bilateral total RD at birth",
        "gene_family": "Wnt Ligand (Norrin)",
    },
    # -- LRP5 -- FEVR Type 4 / OPPG ----------------------------------------------------------
    {
        "gene": "LRP5",
        "alt_name": (
            "LRP5 (LRP5-1615aa-11q13.2 / AD-LOF-FEVR4 / AR-LOF-OPPG / AD-GOF-HIGH-BONE -- "
            "FEVR-TYPE-4-EVR4-AD-HETEROZYGOUS-LOF -- "
            "OSTEOPOROSIS-PSEUDOGLIOMA-OPPG-AR-BIALLELIC-LOF -- "
            "PSEUDOGLIOMA+LOW-BONE-MASS-COMBINATION-PATHOGNOMONIC-OPPG -- "
            "AD-GOF-HIGH-BONE-DENSITY-NO-EYE-DISEASE)"
        ),
        "protein": (
            "LRP5 -- 11q13.2 AD/AR -- LRP5-1615aa -- "
            "LDL-Receptor-Related-Protein-5-179kDa-Wnt-Co-Receptor -- "
            "FZD4-LRP5-Complex-Binds-Norrin-and-Wnt-Ligands-Co-Receptor -- "
            "4-YWTD-Propeller-Domains-3-EGF-Domains-Transmembrane -- "
            "LOF-Impairs-Retinal-Vascularization-AND-Bone-Formation -- "
            "GOF-A214V-G171V-Activating-Wnt-Signaling-High-Bone-Density-Van-Buchem -- "
            "OMIM-Gene-603506-Disease-EVR4-601813-OPPG-259770"
        ),
        "locus": "11q13.2",
        "protein_size": "1615 aa / 179 kDa",
        "inheritance": (
            "AD LOF heterozygous → FEVR Type 4 (EVR4): autosomal dominant FEVR; "
            "AR biallelic LOF → OPPG (Osteoporosis-Pseudoglioma Syndrome): severe; "
            "AD GOF (activating missense: A214V, G171V, D111Y) → HIGH BONE DENSITY (HBM): no retinal disease; "
            "OPPG heterozygous parents (carriers): usually MILD low bone density, "
            "may have very mild FEVR-like FFA changes — haploinsufficiency effect; "
            "Note: same gene, three opposite phenotypes depending on variant type"
        ),
        "disease_category": "FEVR Type 4 (EVR4) — AD heterozygous LOF; Osteoporosis-Pseudoglioma Syndrome (OPPG) — AR biallelic LOF; High Bone Density syndrome — AD GOF (no retinal disease)",
        "disease_pathway": (
            "LRP5 is the Wnt co-receptor that partners with FZD4 to form the Norrin/Wnt signaling receptor complex. "
            "LRP5 is expressed in retinal vasculature AND in osteoblasts. "
            "Heterozygous LOF (FEVR4): reduced LRP5 signaling in retina → similar to FZD4 FEVR but typically milder; "
            "avascular peripheral retina ± fibrovascular proliferation. "
            "Biallelic LOF (OPPG): no LRP5 signaling → "
            "(1) RETINAL: complete absence of retinal vascularization → pseudoglioma (retrolental fibrovascular mass) similar to NDP/Norrie, "
            "but AUTOSOMAL RECESSIVE (not X-linked) → affects both boys and girls; "
            "(2) BONE: osteoblast Wnt signaling absent → severe juvenile osteoporosis → fractures + vertebral collapse. "
            "The COMBINATION OF PSEUDOGLIOMA + SEVERE EARLY-ONSET OSTEOPOROSIS is pathognomonic for OPPG. "
            "GOF variants: constitutive Wnt activation in osteoblasts → high bone mass (HBM); retina unaffected because GOF does not disrupt retinal vascular patterning."
        ),
        "pathognomonic": (
            "OPPG: COMBINATION OF PSEUDOGLIOMA (bilateral fibrovascular retrolental mass + congenital or early infantile blindness) "
            "AND SEVERE JUVENILE OSTEOPOROSIS (fractures in first decade, vertebral compression) PATHOGNOMONIC. "
            "Unlike Norrie Disease (X-linked, boys): OPPG is AR and affects BOTH SEXES equally. "
            "BONE PHENOTYPE DOMINANT: spontaneous fractures from minor trauma, vertebral compression fractures in childhood → "
            "severe kyphoscoliosis; DEXA Z-score << -2.5 in childhood. "
            "RETINAL PHENOTYPE: pseudogliomatous mass or tractional RD depending on severity of biallelic LOF; "
            "congenital blindness in severe OPPG; milder biallelic variants → FEVR-like peripheral avascularity. "
            "FEVR Type 4 (AD heterozygous): similar to FZD4 FEVR — avascular peripheral retina ± traction; "
            "no bone phenotype (adequate haploinsufficiency compensation in osteoblasts). "
            "HBM (AD GOF): high bone density on DEXA; increased fracture resistance; NO retinal disease. "
            "FFA of OPPG heterozygous parents: mild FEVR-like avascularity in some."
        ),
        "treatment": (
            "OPPG retinal: vision rehabilitation; low vision services; "
            "vitreoretinal surgery rarely restores vision in severe OPPG. "
            "OPPG bone: bisphosphonates (pamidronate, zoledronic acid) IV to improve bone density; "
            "anabolic therapy (teriparatide, romosozumab) being investigated; "
            "orthopaedic support for fractures; spinal bracing for vertebral collapse. "
            "OPPG: multidisciplinary: ophthalmology + endocrinology/metabolic bone disease + orthopaedics. "
            "FEVR Type 4 AD: same management as FZD4 FEVR — laser for avascular zones, vitreoretinal surgery for RD. "
            "Carrier screening in OPPG families: FFA of parents/siblings for FEVR-like changes. "
            "Genetic counselling: AR → 25% recurrence if both parents are carriers (often incidentally found)."
        ),
        "key_features": [
            "Same gene: AD LOF→FEVR4, AR LOF→OPPG, AD GOF→high bone density — three opposite phenotypes",
            "OPPG: pseudoglioma + severe juvenile osteoporosis — both genders (vs Norrie Disease boys only)",
            "Bone phenotype: DEXA Z-score << -2.5 in childhood; spontaneous fractures first decade",
            "LRP5 co-receptor for FZD4: Norrin-FZD4-LRP5-TSPAN12 Wnt retinal vascular signaling complex",
            "OPPG parents (carriers): usually mild low bone density ± mild FEVR-like FFA changes",
        ],
        "key_ddx": [
            "OPPG vs Norrie Disease: OPPG AR (both sexes) + bone disease; Norrie XLR + deafness/cognitive",
            "OPPG vs juvenile osteoporosis: OPPG has pseudogliomatous retinal mass in addition",
            "LRP5 FEVR4 vs FZD4 FEVR1: clinically similar; molecular testing distinguishes",
            "LRP5 GOF (HBM) vs van Buchem disease (SOST LOF): both high bone density; molecular testing",
            "OPPG eye vs retinoblastoma: OPPG bilateral, AR family history, no calcification",
        ],
        "vitreous_phenotype": "OPPG: pseudoglioma (retrolental mass); FEVR4: vitreous traction at vascular-avascular junction",
        "systemic_involvement": True,
        "retinal_detachment_risk_pct": "OPPG: ~80-100% (congenital/infantile); FEVR4: 5–30%",
        "hearing_loss_type": "None",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "OPPG: congenital/infantile; FEVR4: variable childhood-adult",
        "gene_family": "Wnt Co-Receptor / LDL Receptor-Related",
    },
    # -- TSPAN12 -- FEVR Type 5 --------------------------------------------------------------
    {
        "gene": "TSPAN12",
        "alt_name": (
            "TSPAN12 (TSPAN12-305aa-7q31.31 / AD -- "
            "FEVR-TYPE-5-EVR5 -- "
            "INCOMPLETE-PENETRANCE-50pct-CARRIERS-UNAFFECTED -- "
            "NORRIN-FZD4-LRP5-TSPAN12-WNT-COMPLEX -- "
            "MILDER-FEVR-PHENOTYPE-THAN-FZD4-NDP)"
        ),
        "protein": (
            "TSPAN12 -- 7q31.31 AD -- TSPAN12-305aa -- "
            "Tetraspanin-12-32kDa-4-Transmembrane-Spans-EC2-Large-Extracellular-Loop -- "
            "Expressed-Retinal-Vasculature-Müller-Cells-Not-in-Bone -- "
            "TSPAN12-Organizes-Norrin-FZD4-LRP5-Signaling-Complex-at-Plasma-Membrane -- "
            "TSPAN12-Enhances-Norrin-FZD4-Binding-Affinity-10-Fold -- "
            "LOF-Reduces-Wnt-Signaling-at-Vascular-Front-Incomplete-Peripheral-Vascularization -- "
            "OMIM-Gene-613138-Disease-EVR5-613310"
        ),
        "locus": "7q31.31",
        "protein_size": "305 aa / 32 kDa",
        "inheritance": (
            "AD (autosomal dominant); dominant-negative or haploinsufficiency; "
            "INCOMPLETE PENETRANCE (~50%): obligate carriers (parents with affected child) may have normal FFA; "
            "this incomplete penetrance is KEY for genetic counselling (not all 50% at-risk offspring will be affected, "
            "and phenotypically normal parents may still pass on the variant); "
            "typically milder FEVR phenotype than FZD4 or NDP; "
            "some families described with more severe tractional RD"
        ),
        "disease_category": "FEVR Type 5 (EVR5) — Familial Exudative Vitreoretinopathy type 5; incomplete penetrance; milder typical phenotype",
        "disease_pathway": (
            "TSPAN12 is a tetraspanin transmembrane protein that physically organizes the Norrin-FZD4-LRP5 receptor signaling complex "
            "at the plasma membrane of retinal vascular endothelial cells. "
            "TSPAN12 interaction with FZD4 increases Norrin-FZD4 binding affinity ~10-fold — "
            "it is an essential scaffold for high-efficiency Wnt/beta-catenin signaling. "
            "TSPAN12 LOF → reduced Wnt signaling amplitude at the vascular growth front → "
            "incomplete peripheral retinal vascularization, but typically less severe than FZD4 LOF "
            "because some residual Norrin-FZD4-LRP5 signaling persists without TSPAN12 scaffolding. "
            "Hence: avascular peripheral retina with a LOWER RISK of tractional RD compared to FZD4 FEVR. "
            "TSPAN12 is not expressed in bone → no bone phenotype (unlike LRP5)."
        ),
        "pathognomonic": (
            "AVASCULAR PERIPHERAL RETINA ON FFA (same pattern as FZD4 FEVR but typically less extensive). "
            "INCOMPLETE PENETRANCE: obligate carriers may have NORMAL FFA — key counselling point. "
            "Phenotypic spectrum: "
            "Mild: peripheral avascularity without fibrovascular proliferation, asymptomatic; "
            "Moderate: temporal dragged disc, straightened retinal vessels; "
            "Severe (minority): tractional RD, sub-retinal exudates. "
            "Because incomplete penetrance, penetrating family members may be skipped → "
            "apparent sporadic case with family history negative → always test parents by FFA. "
            "FFA at vascular-avascular junction: leakage + neovascular tuft in proliferative stages. "
            "Macular: macular dragging, ectopic fovea in severe cases; "
            "macular exudates (exudative FEVR) uncommon but reported. "
            "Cataract and vitreous changes rare in mild TSPAN12 FEVR."
        ),
        "treatment": (
            "Same laser approach as FZD4 FEVR: laser photocoagulation of avascular peripheral retina. "
            "Milder phenotype: many TSPAN12 FEVR patients require only observation or minimal laser. "
            "Anti-VEGF: bevacizumab for active fibrovascular proliferation, adjunct to laser. "
            "Vitreoretinal surgery: reserved for tractional RD (Stage 4-5) — less frequently needed than FZD4. "
            "Family screening: FFA of all first-degree relatives — including phenotypically unaffected parents "
            "(incomplete penetrance → may find avascular zones in visually normal parent). "
            "Genetic counselling: AD with incomplete penetrance — penetrance stated as ~50%; "
            "offspring who inherit the variant have ~50% chance of showing retinal disease themselves. "
            "No bone disease: multidisciplinary team not required unless co-morbidities."
        ),
        "key_features": [
            "Incomplete penetrance ~50%: obligate carrier parents may have NORMAL FFA",
            "Milder typical FEVR phenotype than FZD4 or NDP",
            "TSPAN12 scaffold: increases Norrin-FZD4 binding affinity 10-fold",
            "Norrin-FZD4-LRP5-TSPAN12 = four-gene retinal Wnt complex: variants in each cause FEVR",
            "Family FFA essential: phenotypically normal parent may still carry and transmit variant",
        ],
        "key_ddx": [
            "TSPAN12 FEVR vs FZD4 FEVR: TSPAN12 typically milder + incomplete penetrance; molecular testing",
            "Incomplete penetrance TSPAN12 vs de novo variant: test both parents by FFA + genetics",
            "TSPAN12 vs LRP5 FEVR4: LRP5 has potential bone association in OPPG; TSPAN12 no bone",
            "TSPAN12 FEVR vs Coats: Coats unilateral, telangiectatic dilated vessels, no family history",
        ],
        "vitreous_phenotype": "Mild vitreous traction at vascular-avascular junction; less severe than FZD4",
        "systemic_involvement": False,
        "retinal_detachment_risk_pct": "5–20%",
        "hearing_loss_type": "None",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "Variable: childhood to adulthood; often detected on family screening",
        "gene_family": "Tetraspanin / Wnt Complex Scaffold",
    },
    # -- ZNF408 -- FEVR Type 6 / PFV --------------------------------------------------------
    {
        "gene": "ZNF408",
        "alt_name": (
            "ZNF408 (ZNF408-720aa-11p11.2 / AD -- "
            "FEVR-TYPE-6-EVR6 -- "
            "PERSISTENT-FETAL-VASCULATURE-PFV-PHPV -- "
            "PFV+AVASCULAR-PERIPHERAL-RETINA-COMBINATION-PATHOGNOMONIC -- "
            "MOST-PHENOTYPICALLY-VARIABLE-FEVR-GENE)"
        ),
        "protein": (
            "ZNF408 -- 11p11.2 AD -- ZNF408-720aa -- "
            "Zinc-Finger-Protein-408-81kDa-C2H2-Zinc-Finger-Domain -- "
            "Nuclear-Transcription-Factor -- "
            "Expressed-Retinal-Vasculature-Hyaloid-Vasculature-During-Development -- "
            "Regulates-Retinal-Angiogenesis-Gene-Expression-Program -- "
            "LOF-Impairs-Peripheral-Retinal-Vascularization-AND-Hyaloid-Regression -- "
            "OMIM-Gene-616454-Disease-EVR6-616468"
        ),
        "locus": "11p11.2",
        "protein_size": "720 aa / 81 kDa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency; "
            "MOST PHENOTYPICALLY VARIABLE FEVR gene — ranges from classic FEVR (avascular peripheral retina) "
            "to persistent fetal vasculature (PFV/PHPV); "
            "rare AR biallelic ZNF408 reported with severe early-onset disease; "
            "incomplete penetrance documented"
        ),
        "disease_category": "FEVR Type 6 (EVR6) — Familial Exudative Vitreoretinopathy type 6; Persistent Fetal Vasculature (PFV/PHPV) overlap; most variable FEVR gene",
        "disease_pathway": (
            "ZNF408 is a C2H2 zinc-finger transcription factor expressed in retinal vascular endothelial cells "
            "and in the developing hyaloid vasculature. "
            "During normal development, the hyaloid vascular system (fetal vasculature supplying lens and vitreous in utero) "
            "undergoes programmed regression in the third trimester and postnatal period. "
            "ZNF408 haploinsufficiency → (1) impaired peripheral retinal angiogenesis (FEVR component) "
            "AND (2) incomplete hyaloid vasculature regression → PERSISTENT FETAL VASCULATURE (PFV/PHPV). "
            "PFV presents as a fibrovascular stalk running from optic disc to lens (Cloquet's canal remnant) "
            "and/or a retrolental fibrovascular plaque — causing lens abnormalities and traction on the retina. "
            "The COMBINATION of FEVR-like peripheral avascularity + PFV is characteristic of ZNF408."
        ),
        "pathognomonic": (
            "COMBINATION OF PERSISTENT FETAL VASCULATURE (PFV) AND AVASCULAR PERIPHERAL RETINA PATHOGNOMONIC for ZNF408. "
            "PFV features: "
            "Retrolental fibrovascular stalk or plaque visible behind lens on slit-lamp; "
            "Cloquet's canal persistence (hyaloid canal remnant through vitreous from disc to lens); "
            "Elongated ciliary processes dragged posteriorly by the fibrovascular stalk; "
            "Lens abnormalities: posterior capsule plaque, posterior polar cataract, lens notch; "
            "Leukocoria possible (DDx retinoblastoma). "
            "PFV is often UNILATERAL (one eye more severely affected) — key distinction from bilateral FEVR. "
            "FEVR component: avascular peripheral retina on FFA (may be bilateral, even when PFV is unilateral). "
            "ERG: reduced in eyes with severe fibrovascular proliferation. "
            "Microphthalmia in some: severe PFV → small globe due to traction."
        ),
        "treatment": (
            "PFV management: "
            "Mild PFV (no significant fibrovascular traction, clear lens): observation; "
            "Moderate PFV (posterior lens plaque + mild traction): cataract surgery with posterior capsulotomy + anterior vitrectomy "
            "to remove the retrolental fibrovascular stalk; amblyopia treatment critical post-operatively. "
            "Severe PFV (extensive stalk, RD, microphthalmia): vitreoretinal surgery; prognosis guarded. "
            "FEVR component: laser photocoagulation of avascular peripheral retina. "
            "Amblyopia: critical management — unilateral involvement → patching of fellow eye; "
            "contact lens for optical correction post-cataract. "
            "Anti-VEGF: off-label for active neovascularization component. "
            "Family screening: FFA of all relatives — PFV parent may have only FEVR-like peripheral avascularity. "
            "Genetic counselling: AD with incomplete penetrance; ZNF408 sequencing to confirm; "
            "phenotypic variability within family limits prognosis."
        ),
        "key_features": [
            "FEVR Type 6: most phenotypically variable FEVR gene",
            "PFV/PHPV component: persistent hyaloid vasculature failure to regress",
            "Combination of PFV + avascular peripheral retina pathognomonic",
            "Unilateral PFV + bilateral FEVR-like avascularity pattern typical",
            "ZNF408 = transcription factor (distinct from Wnt pathway genes FZD4/NDP/LRP5/TSPAN12)",
        ],
        "key_ddx": [
            "ZNF408 PFV vs sporadic PFV: family history, bilateral FEVR component on FFA, molecular testing",
            "ZNF408 leukocoria vs retinoblastoma: ZNF408 family history + FFA bilateral avascularity; Rb calcification on US",
            "ZNF408 vs FZD4 FEVR: ZNF408 has PFV component; FZD4 typically no PFV",
            "ZNF408 PFV vs congenital cataract: PFV has retrolental stalk not just lens opacity",
            "ZNF408 FEVR vs Norrie Disease: Norrie XLR boys, bilateral total RD at birth; ZNF408 AD, variable",
        ],
        "vitreous_phenotype": "Persistent fetal vasculature (Cloquet's canal / retrolental stalk) + FEVR traction",
        "systemic_involvement": False,
        "retinal_detachment_risk_pct": "10–35% (PFV traction-related)",
        "hearing_loss_type": "None",
        "prophylactic_retinopexy_indicated": True,
        "onset_age": "Congenital (PFV); FEVR peripheral avascularity from birth; tractional RD childhood-adult",
        "gene_family": "Zinc-Finger Transcription Factor",
    },
]


def _make_cohort(entry, seed):
    """Generate 40 deterministic synthetic patients for one gene."""
    rng = random.Random(seed)
    gene = entry["gene"]
    patients = []

    for i in range(40):
        # Age at diagnosis
        if gene in ("NDP", "LRP5"):      # congenital / infantile
            age_dx = rng.randint(0, 2)
        elif gene == "ZNF408":           # neonatal to infant (PFV)
            age_dx = rng.randint(0, 5)
        elif gene in ("COL2A1", "COL11A1"):
            age_dx = rng.randint(2, 45)
        elif gene == "VCAN":
            age_dx = rng.randint(10, 55)
        else:  # FZD4, TSPAN12
            age_dx = rng.randint(0, 40)

        # Retinal detachment
        if gene == "NDP":
            rd = rng.random() < 0.95
        elif gene == "LRP5":
            rd = rng.random() < 0.70
        elif gene in ("COL2A1", "COL11A1"):
            rd = rng.random() < 0.50
        elif gene == "VCAN":
            rd = rng.random() < 0.38
        elif gene == "FZD4":
            rd = rng.random() < 0.28
        elif gene == "ZNF408":
            rd = rng.random() < 0.25
        elif gene == "TSPAN12":
            rd = rng.random() < 0.12
        else:
            rd = rng.random() < 0.20

        # Avascular peripheral retina on FFA
        if gene in ("FZD4", "TSPAN12", "NDP", "LRP5", "ZNF408"):
            avascular = True
        else:
            avascular = rng.random() < 0.20  # may see in severe Stickler/Wagner

        # Persistent fetal vasculature
        pfv = gene == "ZNF408" and rng.random() < 0.55

        # Hearing loss
        if gene == "NDP":
            hearing_loss = rng.random() < 0.35
        elif gene in ("COL2A1", "COL11A1"):
            hearing_loss = rng.random() < (0.60 if gene == "COL11A1" else 0.45)
        else:
            hearing_loss = False

        # Bone disease (OPPG LRP5)
        bone_disease = gene == "LRP5" and rng.random() < 0.30  # OPPG severe patients

        # Prophylactic laser
        if gene in ("FZD4", "TSPAN12", "LRP5", "ZNF408"):
            laser = rng.random() < 0.58
        elif gene in ("COL2A1", "COL11A1"):
            laser = rng.random() < 0.50
        else:
            laser = rng.random() < 0.30

        # Surgical intervention
        if gene == "NDP":
            surgery = rng.random() < 0.40  # often too advanced for useful surgery
        elif gene == "LRP5":
            surgery = rd and rng.random() < 0.65
        elif gene in ("COL2A1", "COL11A1"):
            surgery = rd and rng.random() < 0.80
        elif gene == "VCAN":
            surgery = rd and rng.random() < 0.75
        else:
            surgery = rd and rng.random() < 0.70

        # Vitreous synchysis (mainly Wagner / Stickler)
        if gene == "VCAN":
            synchysis = rng.random() < 0.90
        elif gene in ("COL2A1", "COL11A1"):
            synchysis = rng.random() < 0.75
        else:
            synchysis = rng.random() < 0.15

        # BCVA worse than 6/60 (legally blind)
        if gene == "NDP":
            blind = rng.random() < 0.90
        elif gene == "LRP5" and bone_disease:
            blind = rng.random() < 0.70
        elif rd and not surgery:
            blind = rng.random() < 0.55
        elif rd and surgery:
            blind = rng.random() < 0.30
        else:
            blind = rng.random() < 0.08

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis_years": age_dx,
            "retinal_detachment": rd,
            "avascular_peripheral_retina": avascular,
            "persistent_fetal_vasculature": pfv,
            "hearing_loss": hearing_loss,
            "bone_disease": bone_disease,
            "prophylactic_laser": laser,
            "surgical_intervention": surgery,
            "vitreous_synchysis": synchysis,
            "bcva_worse_than_6_60": blind,
            "systemic_involvement": entry["systemic_involvement"],
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(VITREORETINOPATHY_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    rd_count = sum(1 for p in all_patients if p["retinal_detachment"])
    avascular_count = sum(1 for p in all_patients if p["avascular_peripheral_retina"])
    hearing_count = sum(1 for p in all_patients if p["hearing_loss"])
    laser_count = sum(1 for p in all_patients if p["prophylactic_laser"])
    surgery_count = sum(1 for p in all_patients if p["surgical_intervention"])
    blind_count = sum(1 for p in all_patients if p["bcva_worse_than_6_60"])

    gene_summary = {}
    for idx, entry in enumerate(VITREORETINOPATHY_GENES):
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
            "vitreous_phenotype": entry["vitreous_phenotype"],
            "systemic_involvement": entry["systemic_involvement"],
            "retinal_detachment_risk_pct": entry["retinal_detachment_risk_pct"],
            "hearing_loss_type": entry["hearing_loss_type"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "rd_pct": round(100 * sum(1 for p in cohort if p["retinal_detachment"]) / len(cohort), 1),
            "avascular_pct": round(100 * sum(1 for p in cohort if p["avascular_peripheral_retina"]) / len(cohort), 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["hearing_loss"]) / len(cohort), 1),
            "laser_pct": round(100 * sum(1 for p in cohort if p["prophylactic_laser"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgical_intervention"]) / len(cohort), 1),
            "blind_pct": round(100 * sum(1 for p in cohort if p["bcva_worse_than_6_60"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Vitreoretinopathy-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Vitreoretinopathy Reference -- COL2A1/COL11A1/VCAN/FZD4/NDP/LRP5/TSPAN12/ZNF408",
        "genes_covered": [e["gene"] for e in VITREORETINOPATHY_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "retinal_detachment_pct": round(100 * rd_count / total, 1),
            "avascular_peripheral_retina_pct": round(100 * avascular_count / total, 1),
            "hearing_loss_pct": round(100 * hearing_count / total, 1),
            "prophylactic_laser_pct": round(100 * laser_count / total, 1),
            "surgical_intervention_pct": round(100 * surgery_count / total, 1),
            "bcva_worse_than_6_60_pct": round(100 * blind_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(VITREORETINOPATHY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "vitreous_phenotype": entry["vitreous_phenotype"],
            "systemic_involvement": entry["systemic_involvement"],
            "retinal_detachment_risk_pct": entry["retinal_detachment_risk_pct"],
            "hearing_loss_type": entry["hearing_loss_type"],
            "prophylactic_retinopexy_indicated": entry["prophylactic_retinopexy_indicated"],
            "onset_age": entry["onset_age"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "rd_pct": round(100 * sum(1 for p in cohort if p["retinal_detachment"]) / len(cohort), 1),
            "avascular_pct": round(100 * sum(1 for p in cohort if p["avascular_peripheral_retina"]) / len(cohort), 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["hearing_loss"]) / len(cohort), 1),
            "laser_pct": round(100 * sum(1 for p in cohort if p["prophylactic_laser"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgical_intervention"]) / len(cohort), 1),
            "blind_pct": round(100 * sum(1 for p in cohort if p["bcva_worse_than_6_60"]) / len(cohort), 1),
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
                "vitreous_phenotype": entry["vitreous_phenotype"],
                "systemic_involvement": entry["systemic_involvement"],
                "retinal_detachment_risk_pct": entry["retinal_detachment_risk_pct"],
                "hearing_loss_type": entry["hearing_loss_type"],
                "prophylactic_retinopexy_indicated": entry["prophylactic_retinopexy_indicated"],
                "onset_age": entry["onset_age"],
                "gene_family": entry["gene_family"],
            }
            for entry in VITREORETINOPATHY_GENES
        },
        "vitreoretinopathy_glossary": {
            "Stickler Syndrome — Type 1 vs Type 2 Vitreous Phenotypes and Genetic Subtypes": (
                "Stickler syndrome is the most common hereditary vitreoretinopathy. "
                "TWO VITREOUS PHENOTYPES (Spranger classification): "
                "TYPE 1 (COL2A1, ~80% of Stickler): OPTICALLY EMPTY VITREOUS — central vitreous cavity appears clear/empty on biomicroscopy "
                "with a pre-retinal membrane; gel completely collapsed. "
                "TYPE 2 (COL11A1, ~10-15% of Stickler): FIBRILLAR BEADED VITREOUS — sparse irregular fibrils visible on biomicroscopy. "
                "Genetic subtypes: STL1 (COL2A1), STL2 (COL11A1), STL3 (COL11A2 — no vitreous, COL11A2 not in vitreous), "
                "STL4/5/6 (COL9A1/A2/A3 — AR, no typical vitreous). "
                "Systemic features ALL types (except COL11A2 and COL9): midface hypoplasia, cleft palate, "
                "hearing loss (conductive + sensorineural), premature arthropathy. "
                "ROBIN SEQUENCE: Stickler most common cause — micrognathia + tongue base obstruction + U-shaped cleft palate. "
                "RETINAL DETACHMENT: 30-70% lifetime risk; prophylactic 360° retinopexy strongly recommended."
            ),
            "FEVR — Familial Exudative Vitreoretinopathy: Staging and Genes": (
                "FEVR is a hereditary disorder of retinal vascular development causing avascular peripheral retina. "
                "STAGING (Kashani-Hajieh classification): "
                "Stage 1: avascular peripheral retina only, asymptomatic; "
                "Stage 2A/2B: fibrovascular proliferation ± exudate, no RD; "
                "Stage 3: extraretinal fibrovascular proliferation (ridge) with plus disease; "
                "Stage 4A/4B: subtotal tractional RD, macula on/off; "
                "Stage 5: total RD ± phthisis. "
                "GENES (Wnt pathway): EVR1=FZD4 (most common, ~40-50%), EVR2=NDP (X-linked), "
                "EVR4=LRP5, EVR5=TSPAN12 (incomplete penetrance), EVR6=ZNF408 (PFV overlap). "
                "All four (FZD4/NDP/LRP5/TSPAN12) disrupt the SAME Norrin-FZD4-LRP5-TSPAN12 Wnt signaling complex. "
                "DDx from ROP: FEVR in TERM babies, family history, often bilateral but asymmetric. "
                "MANAGEMENT: laser photocoagulation of avascular zone (all stages 1-3); anti-VEGF adjunct; "
                "vitreoretinal surgery for Stage 4-5; screen ALL first-degree relatives by FFA."
            ),
            "Wagner Syndrome vs Stickler — The Pure Ocular Differential": (
                "CRITICAL DDx: Wagner syndrome (VCAN) and Stickler syndrome (COL2A1/COL11A1) both cause hereditary vitreoretinal degeneration "
                "but differ fundamentally in systemic involvement. "
                "WAGNER (VCAN): PURE OCULAR disease. "
                "No deafness, no joint disease, no midface abnormality, no cleft palate. "
                "Vitreous: synchysis (complete liquefaction) + fibrovascular VEILS (condensed membranes). "
                "Progressive choroidal atrophy (pericentral) + pigmentary retinopathy + ERG reduction. "
                "STICKLER (COL2A1/COL11A1): SYSTEMIC disease. "
                "Midface hypoplasia + cleft palate + sensorineural deafness + premature arthropathy. "
                "Type 1 vitreous (COL2A1): optically empty; Type 2 (COL11A1): fibrillar beaded. "
                "Diagnostic pitfall: family with mild or absent systemic features → check VCAN vs COL2A1 molecular testing. "
                "VCAN most variants are intronic splice-site (exon 7/8 donor/acceptor) — must specifically request VCAN sequencing."
            ),
            "Norrie Disease vs Retinoblastoma — Bilateral Leukocoria in Boys": (
                "Both present with BILATERAL LEUKOCORIA (white pupil reflex) in male infants — emergency DDx required. "
                "NORRIE DISEASE (NDP, XLR): "
                "X-linked (only hemizygous males blind from birth); carrier females normal; "
                "Fibrovascular pseudoglioma filling vitreous — NO CALCIFICATION on ultrasound; "
                "Family history: maternal uncle or maternal grandfather blind from infancy (X-linked pedigree); "
                "Associated sensorineural deafness (35%) and cognitive/psychiatric (25-35%). "
                "RETINOBLASTOMA (RB1): "
                "Autosomal dominant or sporadic; affects all sexes equally; "
                "Calcification in 90%+ of cases on ultrasound — PATHOGNOMONIC SIGN for Rb; "
                "Requires urgent oncological evaluation; genetic testing RB1; "
                "Treatment: chemotherapy, transpupillary thermotherapy, external beam, enucleation. "
                "KEY INVESTIGATIONS: B-scan ultrasound (calcification?), MRI orbits (extraocular extension), "
                "NDP gene sequencing, RB1 testing, family history. "
                "Never delay — retinoblastoma is potentially lethal."
            ),
            "Persistent Fetal Vasculature (PFV/PHPV) — ZNF408 and Hyaloid Remnants": (
                "Persistent Fetal Vasculature (PFV), formerly Persistent Hyperplastic Primary Vitreous (PHPV), "
                "is a developmental anomaly resulting from failure of hyaloid vasculature regression. "
                "ANATOMY: hyaloid artery runs from optic disc through Cloquet's canal to posterior lens capsule in fetal life; "
                "normally regresses completely by 8-9 months gestation. "
                "PFV spectrum: "
                "Anterior PFV: retrolental fibrovascular plaque, posterior lens opacity, elongated ciliary processes (dragged posteriorly); "
                "Posterior PFV: fibrovascular stalk from disc to lens (Bergmeister's papilla persistent); "
                "Combined. "
                "HEREDITARY PFV: ZNF408 variants cause hereditary PFV (bilateral cases with family history) "
                "combined with FEVR-like peripheral avascularity. "
                "Sporadic PFV: unilateral, no family history — commonest cause. "
                "DDx from cataract: PFV has retrolental stalk; from retinoblastoma: no calcification, "
                "family history, bilateral FEVR on FFA in ZNF408. "
                "Treatment: anterior vitrectomy + posterior capsulotomy for significant fibrovascular traction on lens; "
                "aggressive amblyopia treatment post-operatively."
            ),
            "Prophylactic Retinal Laser — Indication and Technique in Hereditary Vitreoretinopathy": (
                "Prophylactic laser photocoagulation is the primary preventive intervention for hereditary vitreoretinopathy. "
                "INDICATIONS: "
                "Stickler Syndrome: 360° barrier laser (retinopexy) around vitreous base condensations, lattice degeneration, "
                "round holes, retinal breaks — performed prophylactically even without detachment. "
                "FEVR: laser photocoagulation of avascular peripheral retina (Stages 1-2) to prevent fibrovascular progression. "
                "Wagner: laser to treat retinal breaks in vascularized zones adjacent to fibrovascular veils. "
                "TECHNIQUE: "
                "Stickler retinopexy: 2-3 rows of confluent photocoagulation burns; spot size 200-500 µm; "
                "360° around zone of vitreous base pathology; performed under scleral indentation. "
                "FEVR avascular zone: demarcation laser at vascular-avascular border. "
                "TIMING: Stickler — as early as possible after diagnosis, regardless of visual symptoms; "
                "repeat annually and when new breaks identified. "
                "Annual vitreoretinal examination for ALL family members (50% risk in AD, 50% carriers XLR). "
                "EVIDENCE: retrospective data supports prophylactic retinopexy in Stickler (reduces bilateral RD from >50% to ~5-7%)."
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
