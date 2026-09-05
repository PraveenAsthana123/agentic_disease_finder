#!/usr/bin/env python3
"""OI-Atlas — Complete 8-Gene Osteogenesis Imperfecta Atlas
COL1A1  (Collagen type I alpha-1 chain; 1464 aa; 17q21.33; AD;
          OI type I (mild, most common), II (perinatal lethal), III (progressively deforming), IV (moderate);
          Glycine substitutions in triple helix → collagen misfolding → ER stress → reduced secretion;
          More C-terminal the glycine substitution, more severe the phenotype;
          Most common OI gene worldwide — ~80% of all classic OI) ·
COL1A2  (Collagen type I alpha-2 chain; 1366 aa; 7q21.3; AD;
          OI type I–IV; qualitative and quantitative defects;
          Glycine substitutions often cause moderate-severe OI (OI III/IV);
          Null alleles (AR) → recessive OI with severe joint laxity, overlapping EDS phenotype;
          Splicing mutations → exon skipping → structurally defective collagen) ·
IFITM5  (Interferon-induced transmembrane protein 5 / BRIL; 138 aa; 11p15.5; AD;
          OI type V — UNIQUE: HYPERPLASTIC CALLUS pathognomonic; interosseous membrane calcification;
          c.-14C>T (5'UTR) creates new upstream AUG → extended N-terminal protein;
          BRIL gain-of-function → overactive mineralisation signal → massive callus at fracture sites;
          HYPERPLASTIC CALLUS MIMICS OSTEOSARCOMA — DO NOT BIOPSY without OI type V considered;
          Radial head dislocation characteristic; no dentinogenesis imperfecta) ·
SERPINF1 (Pigment epithelium-derived factor / PEDF; 418 aa; 17p13.3; AR;
          OI type VI — most severe OI at skeletal level: fish-scale lamellar bone biopsy PATHOGNOMONIC;
          Undetectable serum PEDF is the diagnostic biomarker;
          NORMAL AT BIRTH — fractures begin in first year; no DI; no blue sclerae;
          Bisphosphonates LESS EFFECTIVE than in other OI types — impaired osteoid mineralisation;
          Anti-RANKL therapy (denosumab) currently investigational) ·
CRTAP   (Cartilage-associated protein; 333 aa; 3p22.3; AR;
          OI type VII — rhizomelic shortening of humerus and femur PATHOGNOMONIC;
          CRTAP forms obligate complex with P3H1 (LEPRE1) + CyPB (PPIB) → Prolyl-3-hydroxylase complex;
          Hydroxylates Pro986 of alpha-1(I) collagen chain in ER; without hydroxylation → collagen misfolding;
          Often perinatally lethal; popcorn calcifications at metaphyses; white sclerae not blue) ·
P3H1    (Prolyl 3-hydroxylase 1 / LEPRE1; 736 aa; 1p34.2; AR;
          OI type VIII — West African founder p.Arg989Cys → 15-20% of severe/lethal OI in African Americans;
          Catalytic subunit of P3H1-CRTAP-CyPB complex; hydroxylates Pro986 of collagen-alpha1(I);
          Lethal to severe; popcorn calcification; white NOT blue sclerae;
          Histology: collagen fibrils thinner and irregular in EM) ·
FKBP10  (FK506-binding protein 10 / FKBP65; 582 aa; 17q21.2; AR;
          OI type XI and BRUCK SYNDROME 1 — OI + CONGENITAL JOINT CONTRACTURES pathognomonic;
          FKBP65 is an ER chaperone that facilitates prolyl cis-trans isomerisation of collagen → proper folding;
          LOF → unfolded collagen secreted → defective cross-linking → fragile bone + abnormal tendons;
          Lysyl pyridinoline/deoxypyridinoline crosslinks ABSENT in urine — diagnostic biomarker;
          Pterygia + knee, ankle, elbow, wrist contractures from birth;
          OI + contractures without BRUCK = FKBP10, not COL1) ·
WNT1    (Wnt family member 1; 370 aa; 12q13.12; AR for OI XV; AD for early-onset osteoporosis;
          OI type XV — biallelic: severe childhood OI with trabecular bone collapse;
          WNT1 → FZD4/LRP5 canonical Wnt → β-catenin → osteoblast survival and activity;
          NO dentinogenesis imperfecta; NO blue sclerae; distinguishes from COL1 OI;
          Cerebral involvement (leukoencephalopathy, ACC) in some biallelic cases;
          Heterozygous carriers: EARLY-ONSET OSTEOPOROSIS in 3rd-4th decade — fractures without OI severity;
          WNT1-OI responds to anti-sclerostin antibody romosozumab — investigational)
320-patient aggregate cohort (8 × 40, seeds 1374-1381)
"""

import random

SEED_BASE = 1374

OI_GENES = [
    # ── COL1A1 — OI type I/II/III/IV (most common) ──
    {
        "gene": "COL1A1",
        "protein": "Collagen Type I Alpha-1 Chain",
        "alias": (
            "COL1A1; OMIM gene 120150; OI type I #166200 / type II #166210 / type III #259420 / type IV #166220 (AD); "
            "17q21.33; 1464 aa (including signal peptide); ~142 kDa alpha-1(I) chain; "
            "pairs with two COL1A2 chains as [alpha-1(I)]2[alpha-1(II)]1 heterotrimer; "
            "most abundant structural protein in bone, skin, tendon, dentin; "
            "pathogenic variants: quantitative (haploinsufficiency via PTC → NMD → mild OI type I) or "
            "qualitative (glycine substitutions in triple helix Gly-X-Y repeats → procollagen misfolding → "
            "ER stress + delayed secretion → OI type II/III/IV); "
            "more C-terminal glycine substitution → more severe phenotype (disrupts more of the propagating helix fold); "
            "most common OI gene worldwide — ~80% of all classic OI alleles; "
            "de novo variants account for ~60% of severe (type II) cases"
        ),
        "aa": "1464 aa",
        "kDa": "~142 kDa",
        "locus": "17q21.33",
        "omim_gene": 120150,
        "omim_disease": 166200,
        "inheritance": (
            "AD — haploinsufficiency (PTC + NMD) → OI type I (mild); "
            "glycine substitutions → OI type II (lethal), III (severely deforming), IV (moderate); "
            "phenotypic severity correlates with triple-helix position and specific amino-acid substitution"
        ),
        "gene_class": (
            "COL1A1 encodes the alpha-1(I) chain of type I collagen. Two alpha-1(I) chains + one alpha-2(I) chain "
            "form a right-handed triple helix (Gly-X-Y repeats; Gly essential every 3rd position — any Gly→other "
            "disrupts propagation from C-terminus). Misfolded procollagen overloads ER → UPR → osteoblast apoptosis. "
            "Quantitative defects (one allele silenced) → reduced but structurally normal collagen → mild OI. "
            "Qualitative defects (dominant-negative misfolded chain poisons 50-75% of trimers) → severe OI."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Glycine substitution Gly→Arg (triple helix, C-terminal third — OI type III/IV)", 0.35),
            ("Premature termination codon (PTC) → NMD → haploinsufficiency (OI type I)", 0.30),
            ("Glycine substitution Gly→Cys (triple helix, C-terminal — OI type II/III)", 0.20),
            ("Splice site variant → exon skipping → in-frame deletion (OI type II/III)", 0.15),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.75,
            "dentinogenesis_imperfecta": 0.35,
            "hearing_loss": 0.40,
            "short_stature": 0.60,
            "scoliosis": 0.45,
            "joint_contractures": 0.00,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.25,
        },
        "hallmarks": [
            "Recurrent fractures from trivial trauma — frequency directly proportional to severity type",
            "BLUE SCLERAE: thin scleral collagen → choroidal pigment visible; most prominent in type I and II",
            "Dentinogenesis imperfecta (DI): opalescent teeth, prone to cracking; more common in OI III/IV",
            "C-TERMINAL GLYCINE RULE: more C-terminal the Gly substitution in the triple helix, more severe OI",
            "Haploinsufficiency (PTC→NMD) → OI type I (mild): normal stature, blue sclerae, low fracture rate",
            "Dominant-negative qualitative defects → OI type II (perinatally lethal) or III (severe deforming)",
            "Bisphosphonates (pamidronate IV or alendronate oral) reduce fracture rate 30-40%",
            "Telescoping intramedullary rods (Fassier-Duval) for long bones — prevent progressive deformity",
        ],
        "treatment_alerts": [
            "BISPHOSPHONATES: IV pamidronate or oral alendronate; start from first fracture; reduces fracture rate 30-40%",
            "TELESCOPING RODS (Fassier-Duval): preferred for growing children; standard rods migrate as child grows",
            "BASILAR INVAGINATION surveillance: annual MRI from age 6 if moderate-severe OI; urgent if cervical myelopathy signs",
            "HEARING AID: otosclerosis-like hearing loss in 40% of OI type I by 4th decade; audiogram every 2 years",
            "AVOID CONTACT SPORTS, trampolines, diving; cycling and swimming preferred for bone loading",
        ],
        "organ_system": "connective tissue (bone / skin / dentin / sclera)",
        "primary_treatment": "Bisphosphonates + orthopaedic stabilisation (telescoping rods); physio",
    },

    # ── COL1A2 — OI type I–IV (quantitative and qualitative) ──
    {
        "gene": "COL1A2",
        "protein": "Collagen Type I Alpha-2 Chain",
        "alias": (
            "COL1A2; OMIM gene 120160; OI type I–IV (AD); recessive OI/EDS overlap (AR); "
            "7q21.3; 1366 aa; ~130 kDa alpha-2(I) chain; "
            "single copy in [alpha-1(I)]2[alpha-2(I)] heterotrimer; "
            "COL1A2 glycine substitutions: dominant-negative OI type II/III/IV; "
            "premature termination codons: OI type I (milder than COL1A1 PTC due to different NMD efficiency); "
            "COL1A2 null alleles (AR, both alleles truncated): produce trimers of 3 × alpha-1(I) — "
            "homotrimers lack the normal alpha-2(I) contribution → EDS-like skin fragility + OI; "
            "some COL1A2 splicing variants → OI type III/IV with severe progressive deformity; "
            "second most common OI gene after COL1A1"
        ),
        "aa": "1366 aa",
        "kDa": "~130 kDa",
        "locus": "7q21.3",
        "omim_gene": 120160,
        "omim_disease": 166210,
        "inheritance": (
            "AD (glycine substitutions, splicing, haploinsufficiency) → OI type I–IV; "
            "AR (biallelic null alleles) → OI with EDS-like joint laxity and skin fragility (recessive COL1A2 OI); "
            "COL1A2 null → homotrimers [alpha-1(I)]3 assembled → structurally compromised but different from Gly-substitution OI"
        ),
        "gene_class": (
            "COL1A2 encodes the alpha-2(I) chain, the minor chain of type I collagen. Glycine substitutions disrupt "
            "the Gly-X-Y triple-helix, causing dominant-negative effect (misfolded chain poisons other trimers). "
            "Haploinsufficiency via NMD → mild OI type I. Biallelic null mutations produce functional homotrimers "
            "[alpha-1(I)]3 — these are secreted but have abnormal mechanical properties (poor fibril registration), "
            "causing OI with EDS-like skin involvement."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("Glycine substitution (triple helix, moderate C-terminal — OI type III/IV)", 0.40),
            ("Haploinsufficiency (PTC + NMD — OI type I, mild)", 0.25),
            ("Splice site → exon skipping → OI type II/III", 0.20),
            ("Biallelic null mutations → recessive OI + EDS-like joint hypermobility", 0.15),
        ],
        "age_onset_years_range": (0, 6),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.55,
            "dentinogenesis_imperfecta": 0.40,
            "hearing_loss": 0.35,
            "short_stature": 0.65,
            "scoliosis": 0.50,
            "joint_contractures": 0.00,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.20,
        },
        "hallmarks": [
            "OI type III/IV predominates in COL1A2 Gly-substitutions — moderate to severe progressive deformity",
            "Blue sclerae: present in ~55% (less universal than COL1A1 type I)",
            "Dentinogenesis imperfecta in 40% — opalescent/brown teeth, enamel fracture",
            "RECESSIVE COL1A2 OI: biallelic null → [alpha-1(I)]3 homotrimers → OI + skin hyperextensibility + joint laxity",
            "Scoliosis in 50% — progressive in weight-bearing; spinal rods if Cobb angle >50°",
            "Hearing loss by 4th decade in 35% — mixed conductive/sensorineural (otosclerosis-like stapes changes)",
            "More severe OI grade for same triple-helix position compared to COL1A1 (alpha-2 has fewer Gly-X-Y repeats)",
            "Bisphosphonates + physiotherapy standard; telescoping rods for femur/tibia if angulation progressive",
        ],
        "treatment_alerts": [
            "BISPHOSPHONATES: start early (first fracture/year 1); IV pamidronate if oral not tolerated",
            "RECESSIVE COL1A2: skin + joint surveillance — skin biopsies show abnormal collagen fibrils on EM",
            "SURGICAL PLANNING: EDS-like joint laxity in recessive form → different anaesthetic considerations (hypermobility)",
            "BASILAR INVAGINATION: annual MRI from age 6 in OI type III — sudden deterioration if untreated",
            "HEARING SURVEILLANCE: audiogram every 2 years from age 20; stapedectomy can improve conductive component",
        ],
        "organ_system": "connective tissue (bone / skin / dentin / sclera)",
        "primary_treatment": "Bisphosphonates + orthopaedic rods; recessive form: joint + skin surveillance",
    },

    # ── IFITM5 — OI type V ──
    {
        "gene": "IFITM5",
        "protein": "Interferon-Induced Transmembrane Protein 5 (BRIL)",
        "alias": (
            "IFITM5 (Bone-Restricted IFITM-Like / BRIL); OMIM gene 614757; OI type V #610967 (AD); "
            "11p15.5; 138 aa; ~16 kDa; interferon-induced transmembrane protein family member; "
            "virtually all OI type V caused by single recurrent c.-14C>T variant (5'UTR); "
            "creates new upstream AUG → extended N-terminal peptide (MSTATI leader) → gain-of-function; "
            "BRIL is expressed exclusively in osteoblasts and pre-osteoclasts — bone-restricted; "
            "overactive BRIL → enhanced mineralisation signalling → HYPERPLASTIC CALLUS at fracture sites; "
            "interosseous membrane calcification of forearm (between radius and ulna) PATHOGNOMONIC; "
            "radial head dislocation; no blue sclerae; no dentinogenesis imperfecta"
        ),
        "aa": "138 aa",
        "kDa": "~16 kDa",
        "locus": "11p15.5",
        "omim_gene": 614757,
        "omim_disease": 610967,
        "inheritance": (
            "AD — virtually all cases due to recurrent de novo or familial c.-14C>T gain-of-function variant; "
            "creates upstream start codon → extended protein; dominant gain-of-function mechanism; "
            "rare IFITM5 missense (p.Ser40Leu) causes severe OI type V with similar phenotype"
        ),
        "gene_class": (
            "IFITM5 (BRIL) is a member of the interferon-induced transmembrane protein family with bone-restricted "
            "expression in osteoblasts. The gain-of-function c.-14C>T mutation creates an N-terminal extension "
            "(MSTATI) that alters protein trafficking and signalling, leading to dysregulated mineralisation "
            "and massive callus formation at fracture sites. Unlike other OI types, collagen structure is normal — "
            "OI type V is a PRIMARY MINERALISATION DISORDER, not a collagen disorder."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("c.-14C>T (5'UTR) — recurrent de novo gain-of-function variant (>95% of OI type V)", 0.95),
            ("p.Ser40Leu missense — severe OI type V with similar hyperplastic callus phenotype", 0.05),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.52,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.05,
            "dentinogenesis_imperfecta": 0.02,
            "hearing_loss": 0.15,
            "short_stature": 0.55,
            "scoliosis": 0.40,
            "joint_contractures": 0.00,
            "hyperplastic_callus": 1.00,
            "basilar_invagination": 0.10,
        },
        "hallmarks": [
            "HYPERPLASTIC CALLUS at fracture sites — PATHOGNOMONIC; may be mistaken for osteosarcoma",
            "INTEROSSEOUS MEMBRANE CALCIFICATION of forearm — visible on plain X-ray: pathognomonic for OI type V",
            "Radial head dislocation — spontaneous or post-fracture, difficult surgical correction",
            "NO BLUE SCLERAE, NO DENTINOGENESIS IMPERFECTA — key DDx from COL1 OI types",
            "Primary mineralisation disorder — collagen biochemistry NORMAL (distinguishes from OI types I–IV)",
            "c.-14C>T recurrent variant: diagnose by targeted sequencing (not caught by standard OI panels)",
            "Bisphosphonates may WORSEN hyperplastic callus — use with caution; anti-RANKL investigational",
            "BRIL gain-of-function → consider anti-IFITM5 BRIL antibody (experimental)",
        ],
        "treatment_alerts": [
            "DO NOT BIOPSY HYPERPLASTIC CALLUS without OI type V considered — avoid unnecessary sarcoma workup",
            "BISPHOSPHONATES: use with caution in OI type V — may stimulate callus further; monitor closely",
            "INTEROSSEOUS MEMBRANE CALCIFICATION: passive stretching futile once calcified; surgery dangerous",
            "c.-14C>T STANDARD PANELS: many OI gene panels do not capture 5'UTR — request c.-14 specific assay",
            "RADIAL HEAD SURGERY: high re-dislocation rate; physiotherapy preferred; surgery only if severely symptomatic",
        ],
        "organ_system": "connective tissue (bone / mineralisation)",
        "primary_treatment": "Orthopaedic management; bisphosphonates with caution; hyperplastic callus monitoring",
    },

    # ── SERPINF1 — OI type VI ──
    {
        "gene": "SERPINF1",
        "protein": "Pigment Epithelium-Derived Factor (PEDF)",
        "alias": (
            "SERPINF1 (PEDF / EPC-1); OMIM gene 172860; OI type VI #613982 (AR); "
            "17p13.3; 418 aa; ~46 kDa; serine protease inhibitor superfamily member (non-inhibitory); "
            "PEDF is a potent anti-angiogenic and anti-apoptotic secreted factor; "
            "secreted by osteoblasts into bone matrix → regulates osteoclast activity + mineralisation; "
            "SERPINF1 LOF → UNDETECTABLE SERUM PEDF — the diagnostic biomarker for OI type VI; "
            "normal at birth; fractures begin at 4-12 months; no DI; no blue sclerae; "
            "FISH-SCALE LAMELLAR BONE biopsy pathognomonic — irregular lamellar structure under polarised light; "
            "bisphosphonates less effective than in COL1 OI types"
        ),
        "aa": "418 aa",
        "kDa": "~46 kDa",
        "locus": "17p13.3",
        "omim_gene": 172860,
        "omim_disease": 613982,
        "inheritance": (
            "AR — biallelic LOF (nonsense, frameshift, splice, large deletion); "
            "no AD cases described; heterozygous carriers clinically unaffected; "
            "carrier frequency and disease prevalence underestimated (born normal → late diagnosis)"
        ),
        "gene_class": (
            "SERPINF1 encodes PEDF, a member of the serine protease inhibitor superfamily that lacks protease-inhibitory "
            "activity. PEDF is produced by osteoblasts and secreted into bone matrix where it inhibits osteoclast "
            "differentiation and regulates mineralisation via integrin receptors. PEDF loss → unchecked osteoclast "
            "activity + defective osteoid mineralisation → accumulation of unmineralised osteoid (fish-scale lamellar "
            "bone on Goldner stain under polarised light). Serum PEDF is undetectable in OI type VI — this is the "
            "fastest diagnostic test before genetic confirmation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("Biallelic nonsense/frameshift → complete PEDF loss", 0.55),
            ("Biallelic splice-site variants → exon skipping → truncated PEDF", 0.25),
            ("Compound heterozygote: one nonsense + one missense (LOF)", 0.15),
            ("Large deletion encompassing SERPINF1 (contiguous gene syndrome)", 0.05),
        ],
        "age_onset_years_range": (0, 1),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.03,
            "dentinogenesis_imperfecta": 0.03,
            "hearing_loss": 0.10,
            "short_stature": 0.80,
            "scoliosis": 0.60,
            "joint_contractures": 0.00,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.20,
        },
        "hallmarks": [
            "FISH-SCALE LAMELLAR BONE on iliac crest biopsy under polarised light — PATHOGNOMONIC for OI type VI",
            "UNDETECTABLE SERUM PEDF — the diagnostic biomarker (fastest initial test; available before genetics)",
            "NORMAL AT BIRTH — first fractures at 4-12 months; born normal distinguishes from severe COL1 OI",
            "NO DENTINOGENESIS IMPERFECTA, NO BLUE SCLERAE — key DDx from COL1A1/A2 OI",
            "BISPHOSPHONATES LESS EFFECTIVE: mineralisation defect is PRIMARY — bisphosphonates reduce resorption but do not fix osteoid",
            "Scoliosis severe (60%) — often requires surgical fusion earlier than COL1 OI types",
            "Anti-RANKL (denosumab) investigational — targets osteoclast excess; preliminary data positive",
            "Vertebral fractures from standing/walking — bracing and wheelchair use common by early childhood",
        ],
        "treatment_alerts": [
            "SERUM PEDF: order immediately if OI type VI suspected (undetectable = diagnostic)",
            "BONE BIOPSY (iliac crest): request Goldner stain + polarised light — fish-scale lamellar pattern diagnoses OI type VI",
            "BISPHOSPHONATES: still first-line but lower efficacy; do NOT stop due to lack of response without specialist review",
            "DENOSUMAB: investigational; do not use off-label without specialist OI centre oversight",
            "SCOLIOSIS MONITORING: Cobb angle every 6 months from age 3; early posterior spinal fusion if >60°",
        ],
        "organ_system": "connective tissue (bone / mineralisation)",
        "primary_treatment": "Bisphosphonates (reduced efficacy) + orthopaedic support; anti-RANKL investigational",
    },

    # ── CRTAP — OI type VII ──
    {
        "gene": "CRTAP",
        "protein": "Cartilage-Associated Protein",
        "alias": (
            "CRTAP; OMIM gene 605497; OI type VII #610682 (AR); "
            "3p22.3; 333 aa; ~38 kDa; leucine-rich repeat protein; "
            "obligate component of the ER Prolyl 3-Hydroxylase complex: CRTAP + P3H1 (LEPRE1) + CyPB (PPIB); "
            "complex hydroxylates Pro986 in the alpha-1(I) chain of type I collagen → structural modification "
            "required for correct collagen folding, secretion, and fibril assembly; "
            "CRTAP LOF → P3H1 degraded (interdependent stability) → Pro986 under-hydroxylated → collagen misfolding; "
            "RHIZOMELIC SHORTENING of humerus and femur PATHOGNOMONIC (distinguishes from COL1 OI types); "
            "perinatally lethal to severe; white sclerae (NOT blue); popcorn metaphyseal calcifications"
        ),
        "aa": "333 aa",
        "kDa": "~38 kDa",
        "locus": "3p22.3",
        "omim_gene": 605497,
        "omim_disease": 610682,
        "inheritance": (
            "AR — biallelic LOF; CRTAP and P3H1 mutations often indistinguishable clinically; "
            "both cause OI type VII/VIII with rhizomelia and white sclerae; "
            "obligate heterozygous carriers unaffected"
        ),
        "gene_class": (
            "CRTAP is a scaffold protein that forms an obligate 1:1:1 trimeric complex with P3H1 (LEPRE1) and CyPB "
            "(PPIB) in the ER lumen. This Prolyl-3-Hydroxylase (P3H) complex catalyses post-translational "
            "hydroxylation of Pro986 in the alpha-1(I) chain of type I procollagen — a modification required for "
            "correct triple-helix folding. Without Pro986 hydroxylation, procollagen is retained in the ER → "
            "UPR activation → osteoblast stress → reduced collagen secretion and abnormal fibril assembly. "
            "CRTAP loss destabilises P3H1 (and vice versa), explaining why CRTAP and P3H1 mutations share the same phenotype."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Biallelic nonsense (PTC) → NMD → complete CRTAP loss", 0.50),
            ("Biallelic frameshift → unstable mRNA → complete loss", 0.30),
            ("Compound heterozygote: nonsense + splice-site", 0.15),
            ("Homozygous large deletion of CRTAP locus", 0.05),
        ],
        "age_onset_years_range": (0, 0),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.05,
            "dentinogenesis_imperfecta": 0.10,
            "hearing_loss": 0.15,
            "short_stature": 1.00,
            "scoliosis": 0.70,
            "joint_contractures": 0.05,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.30,
        },
        "hallmarks": [
            "RHIZOMELIC SHORTENING: proximal limb segments (humerus and femur) disproportionately short — PATHOGNOMONIC",
            "WHITE SCLERAE — NOT BLUE; key DDx from COL1A1/A2 OI (which often has blue sclerae)",
            "Perinatally lethal to severe; extreme skeletal fragility from in utero",
            "Popcorn calcifications at metaphyses on X-ray — growth plate cartilage herniation",
            "Collagen biochemistry: under-hydroxylation of Pro986 in alpha-1(I) — detectable on mass spectrometry",
            "CRTAP LOF destabilises P3H1 and vice versa — molecular phenotype of CRTAP = P3H1 mutations",
            "Bisphosphonates: standard of care but bone fragility often extreme; may need monthly IV pamidronate",
            "Respiratory complications from rib fractures and scoliosis — early ventilatory support",
        ],
        "treatment_alerts": [
            "RESPIRATORY MONITORING: rib cage fragility + scoliosis → restrictive lung disease; pulmonary function from age 5",
            "COLLAGEN BIOCHEMISTRY (mass spec): Pro986 under-hydroxylation confirms P3H complex defect — guides gene testing",
            "AVOID PHYSICAL MANIPULATION: extreme fragility; padded positioning, no routine physiotherapy manoeuvres",
            "PAMIDRONATE: IV infusion protocol; oral bisphosphonates often not feasible in severe forms",
            "GENETIC TESTING: CRTAP and P3H1 (LEPRE1) should be sequenced in parallel — clinically indistinguishable",
        ],
        "organ_system": "connective tissue (bone / collagen post-translational modification)",
        "primary_treatment": "IV pamidronate + respiratory support; extreme fragility care protocol",
    },

    # ── P3H1 — OI type VIII ──
    {
        "gene": "P3H1",
        "protein": "Prolyl 3-Hydroxylase 1 (LEPRE1)",
        "alias": (
            "P3H1 (LEPRE1 / LEPRECAN); OMIM gene 610339; OI type VIII #610915 (AR); "
            "1p34.2; 736 aa; ~85 kDa; prolyl hydroxylase; "
            "catalytic subunit of the P3H1-CRTAP-CyPB complex; hydroxylates Pro986 of alpha-1(I) collagen; "
            "WEST AFRICAN FOUNDER VARIANT: p.Arg989Cys (c.2965C>T) — responsible for 15-20% of severe/lethal OI "
            "in African American families; accounts for significant diagnostic delay in this population; "
            "lethal to severe phenotype; popcorn calcifications; white NOT blue sclerae; "
            "under-hydroxylation of collagen Pro986 directly demonstrable on mass spectrometry"
        ),
        "aa": "736 aa",
        "kDa": "~85 kDa",
        "locus": "1p34.2",
        "omim_gene": 610339,
        "omim_disease": 610915,
        "inheritance": (
            "AR — biallelic LOF; p.Arg989Cys West African founder (homozygous in many African-American pedigrees); "
            "obligate carriers (parents) unaffected; "
            "25% recurrence risk per pregnancy for carrier couples"
        ),
        "gene_class": (
            "P3H1 (LEPRE1) is the catalytic prolyl-3-hydroxylase enzyme in the obligate P3H1-CRTAP-CyPB ER complex. "
            "It hydroxylates Pro986 of the alpha-1(I) procollagen chain — a single post-translational modification "
            "critical for correct folding of the collagen triple helix. P3H1 LOF → Pro986 under-hydroxylation → "
            "collagen misfolding → ER retention → osteoblast UPR activation → reduced bone matrix secretion. "
            "P3H1 and CRTAP are mutually stabilising — P3H1 LOF destabilises CRTAP and vice versa, producing "
            "identical molecular phenotypes in both genes."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("p.Arg989Cys (West African founder) — homozygous (15-20% of severe OI in African Americans)", 0.45),
            ("p.Arg989Cys compound heterozygous with second LOF allele", 0.20),
            ("Other biallelic nonsense/frameshift (non-founder alleles)", 0.25),
            ("Splice-site variants → exon skipping → truncated P3H1", 0.10),
        ],
        "age_onset_years_range": (0, 0),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.05,
            "dentinogenesis_imperfecta": 0.10,
            "hearing_loss": 0.15,
            "short_stature": 1.00,
            "scoliosis": 0.70,
            "joint_contractures": 0.05,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.30,
        },
        "hallmarks": [
            "OI type VIII: severe to lethal skeletal fragility; overlaps with OI type VII (CRTAP) clinically",
            "WHITE SCLERAE — NOT BLUE; key DDx from COL1 OI types; helps distinguish from OI type I/III",
            "WEST AFRICAN FOUNDER p.Arg989Cys: most common severe OI allele in African Americans",
            "Popcorn calcifications at metaphyses — growth plate cartilage calcification islands on X-ray",
            "Collagen Pro986 under-hydroxylation on mass spectrometry — direct diagnostic evidence",
            "Extreme limb shortening + fractures; respiratory failure from rib deformity",
            "EM: thin, disorganised collagen fibrils in bone matrix (distinct from COL1A1/A2 OI fibrils)",
            "Bisphosphonates: standard care; combined with respiratory + nutritional support",
        ],
        "treatment_alerts": [
            "POPULATION-SPECIFIC: p.Arg989Cys must be in every OI panel for African-American patients — severe underdiagnosis",
            "COLLAGEN MASS SPECTROMETRY: Pro986 under-hydroxylation confirms P3H complex defect before genetics returns",
            "CRTAP MUST BE SEQUENCED ALONGSIDE P3H1: indistinguishable clinically — panel both simultaneously",
            "RESPIRATORY FAILURE: leading cause of death; NIV + physiotherapy chest management from infancy",
            "PAMIDRONATE IV: standard; monitor calcium closely in severe forms (milk-alkali / hypercalcaemia risk post-infusion)",
        ],
        "organ_system": "connective tissue (bone / collagen post-translational modification)",
        "primary_treatment": "IV pamidronate + respiratory + nutrition support; West African founder allele targeted testing",
    },

    # ── FKBP10 — OI type XI / Bruck syndrome ──
    {
        "gene": "FKBP10",
        "protein": "FK506-Binding Protein 10 (FKBP65)",
        "alias": (
            "FKBP10 (FKBP65); OMIM gene 607063; OI type XI #610968 / Bruck syndrome 1 #259450 (AR); "
            "17q21.2; 582 aa; ~65 kDa; FK506-binding protein family; ER-localised prolyl isomerase/chaperone; "
            "facilitates cis-trans prolyl isomerisation of procollagen during folding in the ER; "
            "FKBP10 LOF → defective procollagen folding → reduced hydroxylysyl pyridinoline (HP) crosslinks; "
            "DIAGNOSTIC BIOMARKER: absent or severely reduced lysyl pyridinoline (LP) and deoxypyridinoline (DPD) "
            "crosslinks in urine — distinguishes FKBP10 OI from all other OI types; "
            "BRUCK SYNDROME 1: OI + CONGENITAL JOINT CONTRACTURES (knees, ankles, elbows, wrists) + pterygia — "
            "contractures distinguish Bruck from standard OI type XI (FKBP10 only without contractures also described); "
            "moderate to severe OI; dentinogenesis imperfecta variable"
        ),
        "aa": "582 aa",
        "kDa": "~65 kDa",
        "locus": "17q21.2",
        "omim_gene": 607063,
        "omim_disease": 610968,
        "inheritance": (
            "AR — biallelic LOF; FKBP10 mutations cause OI type XI (without contractures) or "
            "Bruck syndrome 1 (with contractures + pterygia); "
            "variable expressivity within and between families — same variant may cause Bruck or OI type XI; "
            "Bruck syndrome 2 caused by PLOD2 (not FKBP10)"
        ),
        "gene_class": (
            "FKBP10 (FKBP65) is an ER-resident prolyl cis-trans isomerase and chaperone that associates with "
            "procollagen alpha chains during folding. It is required for correct post-translational modification "
            "of lysine residues in telopeptide regions of procollagen — which generates hydroxylysyl pyridinoline "
            "(HP) crosslinks in mature collagen fibrils. FKBP10 LOF → absent telopeptide crosslinks → "
            "mechanically inferior bone collagen + defective tendon/ligament matrix → OI + joint contractures. "
            "Urine HP/LP crosslinks are absent or drastically reduced — the most accessible diagnostic assay."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Biallelic nonsense/frameshift → complete FKBP65 loss (OI type XI or Bruck syndrome)", 0.55),
            ("Biallelic splice-site → exon skipping → truncated FKBP65", 0.25),
            ("Homozygous c.831dupC (Turkish/Middle Eastern founder allele)", 0.15),
            ("Compound heterozygous: nonsense + missense", 0.05),
        ],
        "age_onset_years_range": (0, 0),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.30,
            "dentinogenesis_imperfecta": 0.30,
            "hearing_loss": 0.20,
            "short_stature": 0.90,
            "scoliosis": 0.65,
            "joint_contractures": 0.70,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.20,
        },
        "hallmarks": [
            "BRUCK SYNDROME: OI + CONGENITAL JOINT CONTRACTURES (knees, ankles, elbows, wrists) + pterygia — pathognomonic combination",
            "ABSENT URINE LYSYL PYRIDINOLINE CROSSLINKS — the diagnostic biomarker; urine HP/LP crosslinks undetectable",
            "Contractures present at birth — distinguish Bruck from all other OI types; physiotherapy from day 1",
            "Moderate to severe fractures; bone architecture better preserved than CRTAP/P3H1 OI types",
            "Turkish/Middle Eastern c.831dupC founder allele — high index of suspicion in consanguineous families",
            "Variable expressivity: same FKBP10 variant may cause Bruck (with contractures) or OI XI (without)",
            "Collagen cross-linking defect in bone AND tendons/ligaments → joint instability after contracture release",
            "Scoliosis 65%; progressive; spinal fusion if severe Cobb angle",
        ],
        "treatment_alerts": [
            "URINE HP/LP CROSSLINKS: order before genetic testing — absent/reduced crosslinks diagnose FKBP10/PLOD2; distinguish from COL1 OI",
            "DISTINGUISH BRUCK 1 (FKBP10) from BRUCK 2 (PLOD2): same phenotype, different gene — panel both",
            "CONTRACTURE PHYSIO: early passive stretching from birth; serial casting for equinus; surgical release if functionally limiting",
            "POST-RELEASE INSTABILITY: joint laxity after contracture release (defective ligament crosslinks) → orthoses mandatory",
            "BISPHOSPHONATES: standard; bone fragility often moderate-severe; add vitamin D + calcium supplementation",
        ],
        "organ_system": "connective tissue (bone / collagen cross-linking / tendons)",
        "primary_treatment": "Bisphosphonates + contracture physiotherapy; serial casting; urine LP crosslinks diagnostic",
    },

    # ── WNT1 — OI type XV ──
    {
        "gene": "WNT1",
        "protein": "Wnt Family Member 1",
        "alias": (
            "WNT1; OMIM gene 164820; OI type XV #615220 (AR biallelic) / EOOP early-onset osteoporosis (AD, #615221); "
            "12q13.12; 370 aa; ~40 kDa; Wnt family secreted glycolipoprotein; "
            "signals via FZD4/LRP5/LRP6 receptors → β-catenin stabilisation → nucleus → osteoblast survival/differentiation; "
            "biallelic WNT1 LOF → OI type XV: severe trabecular bone loss from birth, multiple fractures, "
            "short stature, occasional cerebral involvement (leukoencephalopathy, agenesis of corpus callosum); "
            "heterozygous carriers → EARLY-ONSET OSTEOPOROSIS: fractures in 3rd-4th decade despite normal DXA-T-score; "
            "NO dentinogenesis imperfecta; NO blue sclerae — distinguishes from COL1 OI types; "
            "romosozumab (anti-sclerostin) investigational — promotes Wnt-like downstream signalling"
        ),
        "aa": "370 aa",
        "kDa": "~40 kDa",
        "locus": "12q13.12",
        "omim_gene": 164820,
        "omim_disease": 615220,
        "inheritance": (
            "AR (biallelic LOF) → OI type XV (severe childhood OI + occasional brain malformations); "
            "AD (heterozygous LOF) → early-onset osteoporosis (EOOP): fractures in 3rd-4th decade; "
            "same gene causes two distinct phenotypes depending on zygosity"
        ),
        "gene_class": (
            "WNT1 is a member of the Wnt family of secreted glycolipoproteins. It binds Frizzled receptors (FZD) "
            "and LRP5/LRP6 co-receptors on osteoblasts, triggering canonical Wnt/β-catenin signalling: GSK3 is "
            "inactivated → β-catenin escapes phosphorylation/degradation → translocates to nucleus → activates "
            "RUNX2, OSX, and other osteoblastogenic transcription factors. WNT1 LOF → osteoblasts fail to survive "
            "and differentiate adequately → severe trabecular bone loss. Brain expression of WNT1 explains the "
            "occasional neurological phenotype in biallelic cases (leukoencephalopathy, ACC)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Biallelic nonsense → OI type XV (severe OI + possible brain malformations)", 0.40),
            ("Biallelic frameshift → OI type XV", 0.30),
            ("Heterozygous LOF → early-onset osteoporosis (3rd-4th decade fractures)", 0.20),
            ("Compound heterozygous: nonsense + missense → OI type XV", 0.10),
        ],
        "age_onset_years_range": (0, 30),
        "sex_ratio_M": 0.50,
        "rates": {
            "fractures": 1.00,
            "blue_sclerae": 0.03,
            "dentinogenesis_imperfecta": 0.03,
            "hearing_loss": 0.10,
            "short_stature": 0.75,
            "scoliosis": 0.45,
            "joint_contractures": 0.00,
            "hyperplastic_callus": 0.00,
            "basilar_invagination": 0.10,
        },
        "hallmarks": [
            "NO DENTINOGENESIS IMPERFECTA, NO BLUE SCLERAE — key DDx from COL1A1/A2 OI",
            "EARLY-ONSET OSTEOPOROSIS in heterozygous carriers: fractures in 3rd-4th decade, often labelled 'idiopathic'",
            "OI TYPE XV (biallelic): severe trabecular bone loss; fractures from birth/infancy; DXA shows Z-score ≤ -6",
            "BRAIN MALFORMATIONS in ~20% of biallelic: leukoencephalopathy, agenesis of corpus callosum — MRI mandatory",
            "LRP5/LRP6 downstream pathway impaired — romosozumab (anti-sclerostin) investigational to rescue signalling",
            "Trabecular pattern severely disrupted on CT/HR-pQCT — distinct from COL1 OI which also affects cortical bone",
            "Heterozygous carriers: bone density low-normal on DXA; microarchitecture severely abnormal on HR-pQCT",
            "WNT1 should be in early-onset osteoporosis gene panels — often missed without family-history OI clue",
        ],
        "treatment_alerts": [
            "BRAIN MRI: mandatory in biallelic OI XV — leukoencephalopathy/ACC requires neurology co-management",
            "HETEROZYGOUS CARRIERS: DXA alone MISSES low bone quality — HR-pQCT or bone biopsy if DXA normal but fractures",
            "ROMOSOZUMAB: investigational in WNT1 OI — pro-osteogenic effect may rescue downstream signalling; specialist trial",
            "BISPHOSPHONATES: standard first-line; may be insufficient in severe OI XV (trabecular loss predominant)",
            "FAMILY SCREENING: heterozygous carrier parents → EOOP risk; proband's first-degree relatives need DXA and WNT1 testing",
        ],
        "organ_system": "connective tissue (bone / Wnt signalling / brain)",
        "primary_treatment": "Bisphosphonates + brain MRI surveillance; heterozygous family screening; romosozumab investigational",
    },
]


# ── cohort builder ──────────────────────────────────────────────────────────────

DIAGNOSTIC_ROUTES = [
    "Clinical OI features + OI gene panel (NGS)",
    "Biochemical collagen analysis (fibroblast / mass spec) → gene confirmation",
    "Bone biopsy pathology → gene testing",
    "Family cascade testing (affected sibling/parent → proband sequencing)",
    "Prenatal diagnosis (affected sibling → chorionic villus sampling)",
    "Incidental fracture workup (X-ray features → referral → genetic panel)",
]

TREATMENTS_SEEN = {
    "COL1A1": ["IV pamidronate + Fassier-Duval rods", "Oral alendronate + physio", "IV zoledronate + spinal fusion"],
    "COL1A2": ["IV pamidronate + telescoping rods", "Oral risedronate + physio", "IV pamidronate + spinal fusion"],
    "IFITM5": ["Orthopaedic management alone", "IV pamidronate with caution + orthopaedic", "Physio + NSAID for callus pain"],
    "SERPINF1": ["IV pamidronate (partial response) + spinal fusion", "IV zoledronate + respiratory support", "Denosumab trial (specialist)"],
    "CRTAP": ["Monthly IV pamidronate + NIV", "IV pamidronate + G-tube nutrition", "Palliative + respiratory support"],
    "P3H1": ["Monthly IV pamidronate + NIV", "IV pamidronate + nutrition support", "Palliative care (lethal form)"],
    "FKBP10": ["Bisphosphonate + serial casting + physio", "IV pamidronate + contracture surgery + orthoses", "Bisphosphonate + spinal fusion"],
    "WNT1": ["Bisphosphonates (first-line)", "Bisphosphonates + romosozumab (trial)", "Bisphosphonate + fracture fixation"],
}


def _build_cohort(gene_data):
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    n = gene_data["n_patients"]
    etiologies = gene_data["etiologies"]
    rates = gene_data["rates"]
    treatments = TREATMENTS_SEEN.get(gene, ["IV pamidronate", "Oral bisphosphonate", "Orthopaedic management"])

    cohort = []
    for i in range(n):
        # etiology
        r = rng.random()
        cumulative = 0.0
        chosen_etiology = etiologies[-1][0]
        for label, prob in etiologies:
            cumulative += prob
            if r < cumulative:
                chosen_etiology = label
                break

        # age at onset
        lo, hi = gene_data["age_onset_years_range"]
        if lo == hi:
            age_onset = lo
        else:
            age_onset = round(rng.uniform(lo, hi), 1)

        # age at diagnosis (delay varies by gene)
        delay_base = {
            "COL1A1": (0.2, 3.0),
            "COL1A2": (0.2, 4.0),
            "IFITM5": (0.5, 5.0),
            "SERPINF1": (0.3, 2.0),
            "CRTAP": (0.0, 0.5),
            "P3H1": (0.0, 0.5),
            "FKBP10": (0.0, 1.0),
            "WNT1": (0.0, 25.0),
        }.get(gene, (0.5, 3.0))
        delay = round(rng.uniform(*delay_base), 1)
        age_diag = round(age_onset + delay, 1)

        # sex
        p_male = gene_data["sex_ratio_M"]
        sex = "M" if rng.random() < p_male else "F"

        # clinical features
        features = {k: rng.random() < v for k, v in rates.items()}

        # treatment
        treatment = rng.choice(treatments)

        # diagnostic route
        diag_route = rng.choice(DIAGNOSTIC_ROUTES)

        cohort.append({
            "id": i + 1,
            "gene": gene,
            "etiology": chosen_etiology,
            "age_at_onset": age_onset,
            "age_at_diagnosis": age_diag,
            "sex": sex,
            "treatment_received": treatment,
            "diagnostic_route": diag_route,
            **{f"has_{k}": v for k, v in features.items()},
        })
    return cohort


# ── API functions ───────────────────────────────────────────────────────────────

def get_overview():
    """Atlas overview: gene list, aggregate stats, key DDx anchors."""
    genes_summary = []
    total_patients = 0
    total_fractures = 0
    total_blue_sclerae = 0
    total_di = 0
    total_hearing = 0
    total_short_stature = 0
    total_scoliosis = 0
    total_contractures = 0
    total_hyperplastic_callus = 0
    total_basilar = 0

    for gd in OI_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)
        total_patients += n

        fractures = sum(1 for p in cohort if p["has_fractures"])
        blue = sum(1 for p in cohort if p["has_blue_sclerae"])
        di = sum(1 for p in cohort if p["has_dentinogenesis_imperfecta"])
        hearing = sum(1 for p in cohort if p["has_hearing_loss"])
        short = sum(1 for p in cohort if p["has_short_stature"])
        scoliosis = sum(1 for p in cohort if p["has_scoliosis"])
        contractures = sum(1 for p in cohort if p["has_joint_contractures"])
        callus = sum(1 for p in cohort if p["has_hyperplastic_callus"])
        basilar = sum(1 for p in cohort if p["has_basilar_invagination"])

        total_fractures += fractures
        total_blue_sclerae += blue
        total_di += di
        total_hearing += hearing
        total_short_stature += short
        total_scoliosis += scoliosis
        total_contractures += contractures
        total_hyperplastic_callus += callus
        total_basilar += basilar

        non_lethal = [p for p in cohort if p["age_at_onset"] > 0 or p["age_at_diagnosis"] > 0]
        if non_lethal:
            avg_onset = round(sum(p["age_at_onset"] for p in non_lethal) / len(non_lethal), 1)
            avg_delay = round(sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in non_lethal) / len(non_lethal), 1)
        else:
            avg_onset = 0.0
            avg_delay = round(sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in cohort) / n, 1)

        genes_summary.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "n_patients": n,
            "fractures_pct": round(100 * fractures / n, 1),
            "blue_sclerae_pct": round(100 * blue / n, 1),
            "dentinogenesis_imperfecta_pct": round(100 * di / n, 1),
            "hearing_loss_pct": round(100 * hearing / n, 1),
            "short_stature_pct": round(100 * short / n, 1),
            "scoliosis_pct": round(100 * scoliosis / n, 1),
            "joint_contractures_pct": round(100 * contractures / n, 1),
            "hyperplastic_callus_pct": round(100 * callus / n, 1),
            "basilar_invagination_pct": round(100 * basilar / n, 1),
            "avg_age_at_onset": avg_onset,
            "avg_diagnosis_delay_years": avg_delay,
            "primary_organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "hallmarks": gd["hallmarks"][:4],
            "top_treatment_alert": gd["treatment_alerts"][0],
        })

    return {
        "atlas": "OI-Atlas",
        "subtitle": "Complete 8-Gene Osteogenesis Imperfecta Atlas",
        "api_path": "/api/oi-atlas/",
        "genes": [g["gene"] for g in OI_GENES],
        "total_patients": total_patients,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "fractures_pct": round(100 * total_fractures / total_patients, 1),
            "blue_sclerae_pct": round(100 * total_blue_sclerae / total_patients, 1),
            "dentinogenesis_imperfecta_pct": round(100 * total_di / total_patients, 1),
            "hearing_loss_pct": round(100 * total_hearing / total_patients, 1),
            "short_stature_pct": round(100 * total_short_stature / total_patients, 1),
            "scoliosis_pct": round(100 * total_scoliosis / total_patients, 1),
            "joint_contractures_pct": round(100 * total_contractures / total_patients, 1),
            "hyperplastic_callus_pct": round(100 * total_hyperplastic_callus / total_patients, 1),
            "basilar_invagination_pct": round(100 * total_basilar / total_patients, 1),
        },
        "genes_summary": genes_summary,
        "key_ddx_anchor": [
            "OI TYPE V (IFITM5): HYPERPLASTIC CALLUS at fracture sites may mimic osteosarcoma — DO NOT BIOPSY without OI type V considered; c.-14C>T is not in standard OI panels",
            "OI TYPE VI (SERPINF1): SERUM PEDF UNDETECTABLE — order immediately; NO blue sclerae, NO DI; bisphosphonates LESS EFFECTIVE; fish-scale bone biopsy pathognomonic",
            "OI TYPE VII/VIII (CRTAP/P3H1): RHIZOMELIC SHORTENING + WHITE SCLERAE (not blue) — key DDx from COL1 OI; mass spec Pro986 under-hydroxylation confirms P3H complex defect",
            "BRUCK SYNDROME (FKBP10): OI + CONGENITAL JOINT CONTRACTURES — pathognomonic combination; URINE LP CROSSLINKS ABSENT or severely reduced",
            "WNT1-OI XV: NO DI, NO BLUE SCLERAE; BRAIN MRI MANDATORY (leukoencephalopathy in 20%); heterozygous carriers get EARLY-ONSET OSTEOPOROSIS in 3rd-4th decade",
            "C-TERMINAL GLYCINE RULE (COL1A1/COL1A2): more C-terminal the Gly substitution in the triple helix, the more severe the OI phenotype (lethal > severely deforming > moderate)",
            "WEST AFRICAN FOUNDER P3H1 p.Arg989Cys: 15-20% of severe OI in African Americans — must be in every OI panel for this population; diagnose before assuming COL1 negative",
            "BISPHOSPHONATES ALL OI TYPES: first-line but LESS EFFECTIVE in SERPINF1 OI VI; USE WITH CAUTION in IFITM5 OI V (may worsen callus); standard in all others",
        ],
    }


def get_breakdown():
    """Per-gene detailed breakdown with cohort data."""
    result = []
    for gd in OI_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)

        sexes = {"M": sum(1 for p in cohort if p["sex"] == "M"),
                 "F": sum(1 for p in cohort if p["sex"] == "F")}
        etiology_counts = {}
        treatments = {}
        diagnostic_routes = {}

        for p in cohort:
            etiology_counts[p["etiology"]] = etiology_counts.get(p["etiology"], 0) + 1
            treatments[p["treatment_received"]] = treatments.get(p["treatment_received"], 0) + 1
            diagnostic_routes[p["diagnostic_route"]] = diagnostic_routes.get(p["diagnostic_route"], 0) + 1

        non_zero = [p for p in cohort if p["age_at_onset"] > 0 or p["age_at_diagnosis"] > 0]
        if non_zero:
            avg_onset = round(sum(p["age_at_onset"] for p in non_zero) / len(non_zero), 1)
            avg_delay = round(sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in non_zero) / len(non_zero), 1)
        else:
            avg_onset = 0.0
            avg_delay = round(sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in cohort) / n, 1)

        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            "n_patients": n,
            "sex_distribution": sexes,
            "avg_age_at_onset": avg_onset,
            "avg_diagnosis_delay_years": avg_delay,
            "fractures_pct": round(100 * sum(1 for p in cohort if p["has_fractures"]) / n, 1),
            "blue_sclerae_pct": round(100 * sum(1 for p in cohort if p["has_blue_sclerae"]) / n, 1),
            "dentinogenesis_imperfecta_pct": round(100 * sum(1 for p in cohort if p["has_dentinogenesis_imperfecta"]) / n, 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["has_hearing_loss"]) / n, 1),
            "short_stature_pct": round(100 * sum(1 for p in cohort if p["has_short_stature"]) / n, 1),
            "scoliosis_pct": round(100 * sum(1 for p in cohort if p["has_scoliosis"]) / n, 1),
            "joint_contractures_pct": round(100 * sum(1 for p in cohort if p["has_joint_contractures"]) / n, 1),
            "hyperplastic_callus_pct": round(100 * sum(1 for p in cohort if p["has_hyperplastic_callus"]) / n, 1),
            "basilar_invagination_pct": round(100 * sum(1 for p in cohort if p["has_basilar_invagination"]) / n, 1),
            "etiology_distribution": etiology_counts,
            "treatment_distribution": treatments,
            "diagnostic_route_distribution": diagnostic_routes,
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "primary_treatment": gd["primary_treatment"],
            "organ_system": gd["organ_system"],
        })
    return result


def get_definitions():
    """Key clinical definitions for OI conditions."""
    return {
        "definitions": [
            {
                "term": "Collagen Triple Helix Glycine Substitution Rule (COL1A1/COL1A2) — C-terminal Severity Gradient",
                "definition": (
                    "Type I collagen alpha chains contain Gly-X-Y repeat sequences (338 repeats per chain) that form "
                    "the right-handed triple helix, with propagation initiating from the C-terminus. Glycine is "
                    "mandatory every third position — even its smallest side chain (single H) is essential to fit "
                    "in the sterically restricted triple-helix core. Any Gly→other substitution introduces a bulky "
                    "side chain that distorts and arrests triple-helix propagation. "
                    "C-TERMINAL SEVERITY GRADIENT: Substitutions closer to the C-terminus affect a larger portion "
                    "of the propagating helix (more residues fail to fold) → more severely misfolded collagen → "
                    "dominant-negative effect on a higher fraction of trimers → more severe OI. N-terminal "
                    "substitutions affect fewer residues → milder OI types III/IV rather than II. "
                    "SPECIFIC AMINO ACID MATTERS: Gly→Cys and Gly→Arg are most severe (sulfhydryl bridges or "
                    "positive charge in helix core); Gly→Ala or Gly→Ser are milder. "
                    "HAPLOINSUFFICIENCY RULE: Premature termination codons → NMD of mutant transcript → 50% "
                    "normal collagen quantity → structurally normal collagen → OI type I (mild). "
                    "Clinical corollary: all OI type I patients should be tested for PTC alleles, not glycine substitutions."
                ),
            },
            {
                "term": "OI Type V (IFITM5/BRIL) — Hyperplastic Callus and the Osteosarcoma Mimic",
                "definition": (
                    "OI type V is caused exclusively by c.-14C>T in the 5'UTR of IFITM5, creating a new upstream "
                    "AUG that extends the N-terminus of BRIL (Bone-Restricted IFITM-Like protein). This is a "
                    "GAIN-OF-FUNCTION mutation — BRIL overactivation leads to dysregulated osteoblast mineralisation "
                    "signalling at fracture sites. "
                    "HYPERPLASTIC CALLUS: Fractures in OI type V heal with massively exuberant periosteal new bone "
                    "formation — the callus can grow to 5-10x the diameter of the original cortex. "
                    "OSTEOSARCOMA MIMIC: The rapid, irregular bone growth is radiologically and sometimes "
                    "histologically indistinguishable from osteosarcoma. DO NOT BIOPSY callus without OI type V "
                    "being considered — biopsy is not diagnostic and is harmful. "
                    "INTEROSSEOUS MEMBRANE CALCIFICATION: Calcification between radius and ulna → progressive "
                    "forearm rotation loss → fixed supination or pronation. Visible on plain X-ray. PATHOGNOMONIC. "
                    "KEY: c.-14C>T is not captured by standard OI NGS panels — specifically request 5'UTR sequencing "
                    "or IFITM5 targeted assay in any patient with hyperplastic callus or interosseous calcification."
                ),
            },
            {
                "term": "OI Type VI (SERPINF1/PEDF) — Undetectable PEDF and Fish-Scale Bone",
                "definition": (
                    "SERPINF1 encodes PEDF (Pigment Epithelium-Derived Factor), a secreted anti-angiogenic protein "
                    "with non-inhibitory serpin structure. PEDF is produced by osteoblasts into bone matrix where "
                    "it regulates osteoclast activity and osteoid mineralisation via integrin αvβ5 receptors. "
                    "DIAGNOSTIC BIOMARKER: Serum PEDF is undetectable (enzyme immunoassay <1 ng/mL) in all OI type VI "
                    "patients — the fastest available confirmatory test before genetic results return. "
                    "FISH-SCALE LAMELLAR BONE: Iliac crest biopsy stained with Goldner and examined under polarised "
                    "light shows irregular, non-parallel lamellar structure (resembling fish scales) rather than "
                    "the normal parallel arrangement of lamellar bone. Unmineralised osteoid accumulates. PATHOGNOMONIC. "
                    "BISPHOSPHONATES LESS EFFECTIVE: The primary defect is defective osteoid mineralisation — "
                    "bisphosphonates reduce resorption but cannot restore mineralisation; fracture rates improve "
                    "less than in COL1 OI types. Denosumab (anti-RANKL) is investigational at specialist OI centres. "
                    "NORMAL NEONATAL PHENOTYPE: Born without fractures; distinguishes from lethal OI type II (COL1A1/A2)."
                ),
            },
            {
                "term": "P3H Complex OI (CRTAP/P3H1) — Pro986 Under-Hydroxylation and Rhizomelic Shortening",
                "definition": (
                    "The ER Prolyl-3-Hydroxylase complex (P3H1-CRTAP-CyPB) performs a single post-translational "
                    "hydroxylation: Pro986 in the alpha-1(I) procollagen chain. This modification is required for "
                    "correct procollagen folding. P3H1 is the catalytic enzyme; CRTAP is the scaffold; CyPB (PPIB) "
                    "is the prolyl isomerase. All three are mutually stabilising — LOF in any component degrades "
                    "the others. "
                    "RHIZOMELIC SHORTENING (OI type VII/VIII): The mechanism by which P3H complex loss produces "
                    "disproportionate shortening of proximal limb segments (humerus, femur) is incompletely understood "
                    "but is consistent and distinctive. PATHOGNOMONIC for P3H complex OI (CRTAP type VII, P3H1 type VIII). "
                    "WHITE SCLERAE (NOT BLUE): Unlike COL1A1/A2 OI, collagen quantity is reduced and structurally "
                    "altered differently — sclerae are white/normal, never blue. Critical DDx from COL1 OI. "
                    "MASS SPECTROMETRY: Peptide mass spectrometry of collagen from skin biopsy or cell culture "
                    "demonstrates under-hydroxylation of Pro986 — practical early diagnostic tool. "
                    "WEST AFRICAN FOUNDER (P3H1 p.Arg989Cys): 15-20% of severe OI in African Americans. "
                    "Homozygous in many families; underdiagnosed — routine OI panels must include this allele."
                ),
            },
            {
                "term": "Bruck Syndrome (FKBP10) — OI with Congenital Joint Contractures and Absent Crosslinks",
                "definition": (
                    "FKBP10 encodes FKBP65, an ER-resident member of the FK506-binding protein family. FKBP65 "
                    "functions as a molecular chaperone that facilitates cis-trans isomerisation of proline residues "
                    "in the telopeptide domains of procollagen alpha chains during ER folding. This step is required "
                    "for the formation of hydroxylysyl pyridinoline (HP) and deoxypyridinoline (DPD) collagen "
                    "crosslinks in mature fibrillar collagen. "
                    "ABSENT CROSSLINKS: FKBP10 LOF → telopeptide proline incorrectly configured → crosslinking "
                    "enzyme (LH2, PLOD2) cannot act → HP/DPD crosslinks absent or severely reduced in urine. "
                    "URINE LP CROSSLINKS = DIAGNOSTIC BIOMARKER: absent/very low LP and DPD distinguishes "
                    "FKBP10/PLOD2 Bruck syndrome from all other OI types (COL1, SERPINF1, CRTAP, P3H1 all have "
                    "normal or elevated crosslinks). "
                    "BRUCK SYNDROME 1 (FKBP10) vs BRUCK SYNDROME 2 (PLOD2): Identical phenotype — OI + congenital "
                    "joint contractures + pterygia. Genetically distinct. Panel both in any Bruck phenotype. "
                    "JOINT CONTRACTURES: Present at birth (knees, ankles, elbows, wrists most commonly); require "
                    "physiotherapy serial casting from day 1; surgical release if functionally limiting (but joint "
                    "instability may result due to defective ligament crosslinks)."
                ),
            },
            {
                "term": "WNT1-OI XV — Biallelic Wnt1 Loss, Trabecular Bone Collapse, and Brain Involvement",
                "definition": (
                    "WNT1 encodes a secreted Wnt glycolipoprotein that activates canonical β-catenin signalling "
                    "via FZD/LRP5/LRP6 receptors in osteoblasts. WNT1→β-catenin→RUNX2/OSX transcription factor "
                    "cascade drives osteoblast survival and differentiation. Biallelic WNT1 LOF → insufficient "
                    "osteoblast function → profoundly low trabecular bone volume from birth → OI type XV. "
                    "NO BLUE SCLERAE, NO DENTINOGENESIS IMPERFECTA: Scleral and dentinal collagen is normal "
                    "(WNT1 is not a collagen gene). Absence of these classic OI features delays diagnosis. "
                    "BRAIN MALFORMATIONS (~20%): WNT1 is also expressed in the developing brain — biallelic LOF "
                    "causes leukoencephalopathy, agenesis of corpus callosum, or cerebellar hypoplasia in ~1 in 5 "
                    "OI XV patients. Brain MRI mandatory at diagnosis and if neurological symptoms develop. "
                    "HETEROZYGOUS CARRIERS → EARLY-ONSET OSTEOPOROSIS (EOOP): Heterozygous WNT1 LOF reduces "
                    "osteoblast efficiency by ~50% → bone mass falls faster than age-related decline → "
                    "fractures in 3rd-4th decade. DXA T-score may be only mildly low while microarchitecture "
                    "(HR-pQCT) is severely disrupted. WNT1 should be in EOOP gene panels. "
                    "ROMOSOZUMAB (anti-sclerostin): investigational in WNT1 OI — by blocking sclerostin (a Wnt "
                    "antagonist), it rescues some Wnt downstream signalling even when WNT1 ligand is absent."
                ),
            },
            {
                "term": "Bisphosphonate Therapy in OI — Gene-Specific Efficacy and Dosing Principles",
                "definition": (
                    "Bisphosphonates are the standard first-line pharmacotherapy for all OI types with recurrent "
                    "fractures. They act as osteoclast inhibitors (via nitrogen-containing bisphosphonate inhibition "
                    "of farnesyl pyrophosphate synthase → osteoclast apoptosis). "
                    "EFFICACY BY TYPE: Best established in COL1A1/A2 OI (types I-IV) — 30-40% fracture rate "
                    "reduction. Moderately effective in CRTAP/P3H1 OI (types VII/VIII). REDUCED EFFICACY in "
                    "SERPINF1 OI type VI (primary mineralisation defect unaddressed by reducing resorption). "
                    "USE WITH CAUTION in IFITM5 OI type V (hyperplastic callus may be exacerbated). "
                    "IV vs ORAL: IV pamidronate (Cycles: 3 consecutive days q3-4 months) preferred in severe OI "
                    "and small children; oral alendronate or risedronate acceptable in mild OI type I. "
                    "IV zoledronate (annual single infusion) increasingly used in adults. "
                    "DURATION: Treat through growth; consider holiday in adolescence (DXA monitoring); "
                    "resume if fracture frequency increases or DXA Z-score falls. "
                    "ADVERSE EFFECTS: Acute-phase reaction after first infusion (flu-like, 24-48 h); "
                    "osteonecrosis of jaw (ONJ) in prolonged high-dose exposure — dental review before and during therapy."
                ),
            },
            {
                "term": "Basilar Invagination in OI — Surgical Emergency and Surveillance",
                "definition": (
                    "Basilar invagination (BI) is the upward migration of the odontoid process (C2 dens) through "
                    "the foramen magnum — a late complication of softened skull base bone in moderate-severe OI. "
                    "MECHANISM: Softened occipital bone collapses around the atlas/axis as spinal loading increases "
                    "with weight-bearing. The dens rises into the posterior fossa → compresses medulla/cerebellum. "
                    "PREVALENCE: ~25% of OI type III, ~20% of OI type IV, ~30% of P3H complex OI. "
                    "PRESENTATION: Insidious — headache on Valsalva, neck pain, ataxia, dysarthria, apnoea, "
                    "pyramidal signs. Sudden deterioration can occur → medical emergency. "
                    "SURVEILLANCE: Annual brain/cervical MRI from age 6 in all moderate-severe OI (type III, IV, "
                    "CRTAP, P3H1). Urgent MRI for ANY new neurological symptom. "
                    "SURGICAL: Posterior craniovertebral junction decompression + occipitocervical fusion when "
                    "symptomatic BI present; requires specialised OI neurosurgical team. "
                    "BISPHOSPHONATES may slow skull base softening — another reason for early initiation."
                ),
            },
            {
                "term": "OI Gene Panel Diagnostic Approach — Tiered Testing and Molecular Diagnostics",
                "definition": (
                    "A modern OI diagnostic workup should be tiered: "
                    "TIER 1 — Phenotype-directed biochemical tests: "
                    "(1) Serum PEDF (SERPINF1 OI type VI — result same day); "
                    "(2) Urine LP/DPD crosslinks (FKBP10 Bruck syndrome — result 1-2 weeks); "
                    "(3) Collagen mass spectrometry from skin biopsy or cultured fibroblasts: "
                    "Pro986 under-hydroxylation → CRTAP/P3H1 (P3H complex); "
                    "overmodification of Gly-X-Y hydroxyproline → COL1A1/A2 qualitative defect; "
                    "TIER 2 — Next-generation sequencing OI gene panel: "
                    "Must include COL1A1, COL1A2, IFITM5 (with 5'UTR c.-14 assay), SERPINF1, CRTAP, P3H1 (LEPRE1), "
                    "FKBP10, WNT1 as minimum; many commercial panels add PPIB, SEC24D, TMEM38B, FAM46A, etc. "
                    "POPULATION-SPECIFIC: P3H1 p.Arg989Cys (West African founder) must be in panels for "
                    "African-American patients — severe OI underdiagnosis otherwise. "
                    "CASCADE: first-degree relatives of confirmed OI patients → carrier testing (AD OI) "
                    "or reproductive counselling (AR OI, 25% recurrence). "
                    "PRENATAL: chorionic villus sampling (CVS) in known familial variant; ultrasound after 18 weeks "
                    "may detect fractures/limb shortening in severe OI types."
                ),
            },
        ],
        "cascade_testing_note": (
            "CASCADE TESTING — OI PANEL: All first-degree relatives of confirmed OI probands should be offered "
            "genetic counselling and targeted testing. For AD OI (COL1A1/A2, IFITM5): 50% transmission per pregnancy. "
            "For AR OI (SERPINF1, CRTAP, P3H1, FKBP10, WNT1): both parents are obligate carriers; "
            "25% recurrence per pregnancy; preimplantation genetic testing (PGT) available for known familial variants."
        ),
    }
