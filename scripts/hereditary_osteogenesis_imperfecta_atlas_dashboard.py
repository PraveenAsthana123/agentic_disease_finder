#!/usr/bin/env python3
"""Hereditary-Osteogenesis-Imperfecta-Atlas — Complete 8-Gene OI & Collagen-I Bone Fragility Atlas
(COL1A1 · COL1A2 · IFITM5 · SERPINF1 · CRTAP · LEPRE1 · PPIB · FKBP10).

COL1A1   (Collagen type I alpha 1; 1464 aa; 138 kDa; 17q21.33; AD;
          OI types I / II / III / IV depending on variant class;
          NULL/frameshift → OI type I (mild, normal collagen structure, haploinsufficiency);
          GLYCINE SUBSTITUTION → types II–IV (structural triple-helix disruption);
          TYPE I: blue sclerae, hearing loss, minimal deformity — MILDEST AD OI;
          TYPE II: LETHAL PERINATALLY — crumpled long bones, rib fractures, stillbirth / early death;
          TYPE III: most severe survivable — wheelchair, severe deformity, scleral hue fades;
          TYPE IV: moderate, white sclerae; BISPHOSPHONATES cornerstone for types II–IV;
          seed SEED_BASE+0).
COL1A2   (Collagen type I alpha 2; 1366 aa; 129 kDa; 7q21.3; AD;
          Glycine substitutions in alpha-2 chain → OI types II–IV (triple-helix folding delay);
          Less common than COL1A1 but identical spectrum types II/III/IV;
          COL1A2 glycine substitutions closer to C-terminus often milder than N-terminus;
          EHLERS-DANLOS variant: biallelic splice → arthrochalasis EDS (severe joint laxity + OI);
          seed SEED_BASE+1).
IFITM5   (Interferon-induced transmembrane protein 5 / BRIL; 132 aa; 14 kDa; 11p15.5; AD;
          OI type V — recurrent de novo c.-14C>T promoter or p.Ser40Leu GOF;
          HYPERPLASTIC CALLUS (exuberant fracture callus) PATHOGNOMONIC;
          CALCIFICATION OF INTEROSSEOUS MEMBRANE (radius-ulna, tibia-fibula) PATHOGNOMONIC;
          DENSE METAPHYSEAL BAND on X-ray — highly specific for type V;
          bisphosphonates effective; collagen biochemistry NORMAL — not detected by collagen studies;
          seed SEED_BASE+2).
SERPINF1 (PEDF / Pigment Epithelium-Derived Factor; 418 aa; 46 kDa; 17p13.3; AR;
          OI type VI — FISH-SCALE BONE LAMELLAE ON BIOPSY PATHOGNOMONIC;
          COLLAGEN BIOCHEMISTRY NORMAL — missed by standard OI panel (no collagen defect);
          serum PEDF undetectable — diagnostic screening test;
          early childhood onset, progressive deformity;
          bisphosphonates + denosumab; anti-RANKL approach experimental;
          seed SEED_BASE+3).
CRTAP    (Cartilage-associated protein; 339 aa; 37 kDa; 3p22.3; AR;
          OI type VII — prolyl 3-hydroxylation complex (CRTAP + P3H1/LEPRE1 + PPIB);
          NULL/biallelic loss → LETHAL PERINATALLY or severe neonatal;
          RHIZOMELIA (shortening proximal limb segments) DISTINCTIVE FOR TYPE VII;
          OVERMODIFIED TYPE I COLLAGEN on electrophoresis — diagnostic signature;
          hypoplastic thorax → respiratory death; survival with intensive support;
          seed SEED_BASE+4).
LEPRE1   (Prolyl 3-hydroxylase 1 / P3H1; 736 aa; 84 kDa; 1p34.2; AR;
          OI type VIII — West African/Cameroonian founder p.Trp339Ter (c.1015G>T);
          NEONATAL LETHAL in severe forms — white sclerae, severe osteopenia;
          OVERMODIFICATION OF COLLAGEN alpha1(I) Pro986 PATHOGNOMONIC (electrophoresis);
          NOT detectable by standard DNA panel in West-African patients unless founder variant sought;
          BIOCHEMICAL DIAGNOSIS (collagen electrophoresis) essential before WES in high-risk groups;
          seed SEED_BASE+5).
PPIB     (Cyclophilin B / Peptidyl-prolyl cis-trans isomerase B; 212 aa; 24 kDa; 15q22.31; AR;
          OI type IX — third component of prolyl-3-hydroxylation ternary complex;
          OVERMODIFIED COLLAGEN ON SDS-PAGE (same signature as CRTAP/LEPRE1 — complex integrity);
          PPIB biallelic null → severe OI; hypomorphic = moderate;
          Normal P3H1 enzymatic activity (PPIB is isomerase/chaperone, not hydroxylase);
          Extremely rare; families reported from Middle East and Asia;
          seed SEED_BASE+6).
FKBP10   (FK506-binding protein 10; 582 aa; 65 kDa; 17q21.2; AR;
          OI type XI / Bruck syndrome 1;
          CONTRACTURES AT BIRTH + OI = BRUCK SYNDROME 1 PATHOGNOMONIC (OI + arthrogryposis);
          LYSYL HYDROXYLASE 2 activity reduced → defective pyridinoline cross-link formation;
          PTERYGIUM (webbing of skin at joints) in Bruck variant;
          Turkish founder c.831+1G>T; Omani/Arab families;
          bisphosphonates + contracture management; no specific therapy yet;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2550-2557).
"""

import random

SEED_BASE = 2550

OI_GENES = [
    # -- COL1A1 -- OI types I/II/III/IV -------------------------------------------------------
    {
        "gene": "COL1A1",
        "alt_name": (
            "COL1A1 (COL1A1-1464aa-17q21.33 / AD -- "
            "OI-TYPES-I-II-III-IV-NULL-TYPE-I-GLYCINE-SUBST-TYPE-II-IV -- "
            "TYPE-I-BLUE-SCLERA-HEARING-LOSS-MILDEST -- "
            "TYPE-II-LETHAL-PERINATAL-CRUMPLED-LONG-BONES -- "
            "BISPHOSPHONATE-CORNERSTONE-TYPES-II-III-IV)"
        ),
        "protein": (
            "COL1A1 -- 17q21.33 AD -- COL1A1-1464aa -- "
            "Collagen-alpha-1(I)-chain-138kDa-Triple-Helix-Gly-X-Y-Repeat -- "
            "Fibrillar-Collagen-Type-I-Most-Abundant-Body-Protein-Bone-Tendon-Skin -- "
            "Homotrimer-alpha1(I)2-alpha2(I)1-Staggered-D-Periodicity-67nm -- "
            "NULL-Frameshift-Haploinsufficiency-OI-Type-I-Mild -- "
            "Glycine-Substitution-Triple-Helix-Disruption-OI-Types-II-III-IV -- "
            "OMIM-Gene-120150-Disease-OI-Type-I-166200"
        ),
        "locus": "17q21.33",
        "protein_size": "1464 aa / 138 kDa",
        "inheritance": (
            "AD (haploinsufficiency for type I; dominant-negative glycine substitutions for types II–IV); "
            "TYPE I: blue sclerae + mild fractures + hearing loss (30–50% adults); MILDEST; "
            "TYPE II: LETHAL PERINATALLY — crumpled long bones, beaded ribs, severe osteopenia, stillbirth; "
            "TYPE III: most severe survivable — severe deformity, wheelchair, scleral hue fades to white with age; "
            "TYPE IV: moderate — normal-colour sclerae, mild–moderate deformity; "
            "Glycine substitutions near C-terminal propeptide cleavage site → milder type IV; "
            "N-terminal glycine substitutions near signal peptide → lethal type II"
        ),
        "disease_category": (
            "OI types I/II/III/IV (COL1A1) — most common genetic bone fragility disorder; "
            "null/frameshift = OI type I (mild haploinsufficiency); "
            "glycine substitution = OI types II-IV (structural triple-helix disruption); "
            "TYPE II LETHAL: crumpled long bones, beaded ribs — diagnosis at birth or antenatal; "
            "BISPHOSPHONATE therapy (pamidronate IV or alendronate PO) — cornerstone for types II-IV"
        ),
        "disease_pathway": (
            "Collagen type I forms the major structural scaffold of bone, tendon, and skin. "
            "The mature fibril is a heterotrimer [alpha1(I)]2[alpha2(I)]1 assembled from pro-collagen chains "
            "that fold into a right-handed triple helix requiring Gly-X-Y repeats (Gly every 3rd position). "
            "NULL COL1A1 alleles reduce total collagen I output by ~50% → OI type I (structurally normal but reduced). "
            "GLYCINE SUBSTITUTIONS within the triple helix introduce bulky side-chains, slowing zipper-like "
            "C→N terminal folding → prolonged endoplasmic-reticulum retention → excessive hydroxylation and "
            "glycosylation (overmodification) → secreted collagen has defective fibril cross-linking → "
            "severely fragile bone (types II–IV). "
            "Pamidronate IV (every 3 months for 3 days) increases bone mineral density and reduces fracture rate; "
            "zoledronate once-yearly increasingly used. Osteotomy + intramedullary rodding for severe deformity."
        ),
        "pathognomonic": (
            "TYPE I: Blue sclerae (thin sclera → choroidal pigment visible) + hearing loss (30-50%) + dentinogenesis imperfecta 50% — classic OI triad. "
            "TYPE II: LETHAL PERINATAL — crumpled accordion-like long bones + beaded ribs on X-ray — pathognomonic. "
            "TYPE III: Triangular face + barrel chest + progressive deformity — most severe survivable OI."
        ),
        "treatment": (
            "Bisphosphonates (pamidronate IV 3-day cycles q3mo, or zoledronate 1×/year): cornerstone — reduce fractures up to 50% in types II/III/IV. "
            "Intramedullary rodding (Fassier-Duval telescoping rod): prevents progressive bowing. "
            "Romosozumab/sclerostin antibody: phase 2 data positive for moderate-severe OI. "
            "Hematopoietic stem-cell transplant: early trials only. "
            "TYPE I: mild fracture management; hearing aids; no bisphosphonate in all cases. "
            "Physiotherapy + hydrotherapy — MANDATORY for ambulation."
        ),
        "key_features": [
            "Blue sclerae (type I, II) — choroid visible through thinned sclera; fades in type III with age",
            "Dentinogenesis imperfecta (50% of OI) — amber/brown brittle teeth, abrade rapidly",
            "Hearing loss (conductive 30%, sensorineural 20%) — ossicular OI in adults type I",
            "Ligamentous laxity + easy bruising — collagen I in all connective tissue",
            "Wormian bones on skull X-ray — multiple sutural ossification centres in calvarium",
            "TYPE II: beaded ribs (multiple rib fractures in utero) — pathognomonic on antenatal USS",
            "Intrauterine growth retardation + low fetal movement in type II",
            "Scoliosis progressive in type III — pulmonary restriction"
        ],
        "key_ddx": [
            "Child abuse (NAI) — CRITICAL DDx; metabolic bone disease screen + genetics before reporting abuse",
            "Rickets / vitamin D deficiency — check Ca/PO4/ALP/25-OH-D",
            "Hypophosphatasia — low ALP pathognomonic; PEDF normal; responds to enzyme therapy",
            "Idiopathic juvenile osteoporosis — no collagen defect; resolves puberty",
            "Ehlers-Danlos syndrome — skin/joint laxity predominates; OI features minor"
        ],
        "fracture_burden": "Type I: <5 fractures/year childhood; Type III: >10 fractures/year",
        "lethal_pct": 25,  # type II
        "blue_sclera_pct": 70,
        "hearing_impairment_pct": 40,
        "dentinogenesis_imperfecta_pct": 50,
        "on_bisphosphonate_pct": 65,
        "bone_mineral_density_zscore": -3.2,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- COL1A2 -- OI types II/III/IV / Arthrochalasis EDS ------------------------------------
    {
        "gene": "COL1A2",
        "alt_name": (
            "COL1A2 (COL1A2-1366aa-7q21.3 / AD -- "
            "OI-TYPES-II-III-IV-GLYCINE-SUBSTITUTION -- "
            "BIALLELIC-SPLICE-ARTHROCHALASIS-EDS-JOINT-LAXITY-PLUS-OI -- "
            "C-TERMINUS-GLYCINE-MILDER-N-TERMINUS-SEVERE -- "
            "COLLAGEN-ALPHA-2-CHAIN-STRUCTURAL)"
        ),
        "protein": (
            "COL1A2 -- 7q21.3 AD -- COL1A2-1366aa -- "
            "Collagen-alpha-2(I)-chain-129kDa-Triple-Helix-Component -- "
            "Fibrillar-Collagen-Type-I-Heterotrimer-alpha1(I)2-alpha2(I)1 -- "
            "Glycine-X-Y-Repeat-C-to-N-Terminal-Folding-Zipper -- "
            "Glycine-Substitution-Dominant-Negative-OI-Types-II-III-IV -- "
            "Biallelic-Splice-Arthrochalasis-EDS-Type-VIIB -- "
            "OMIM-Gene-120160-Disease-OI-Type-II-166220"
        ),
        "locus": "7q21.3",
        "protein_size": "1366 aa / 129 kDa",
        "inheritance": (
            "AD (glycine substitutions — OI types II-IV); "
            "Biallelic splice site variants (e.g. skipping of exon 6 removes pro-alpha2(I) N-propeptide cleavage site) → "
            "Arthrochalasis Ehlers-Danlos syndrome — SEVERE JOINT HYPERMOBILITY + OI features; "
            "AD glycine substitutions near C-terminus → milder type IV vs N-terminus → severe type II; "
            "COL1A2 null alleles do NOT cause OI (alpha2 haploinsufficiency is tolerated — alpha1 homotrimers form) — "
            "IMPORTANT: only glycine substitutions/structural → OI via dominant-negative"
        ),
        "disease_category": (
            "OI types II/III/IV (COL1A2 glycine substitutions) — identical clinical spectrum to COL1A1 types II-IV; "
            "Arthrochalasis EDS (biallelic COL1A2 splice) — SEVERE JOINT HYPERMOBILITY + hip dislocation + kyphoscoliosis + OI; "
            "null COL1A2 = NOT OI (alpha1(I) homotrimers compensate); "
            "C-terminus glycine substitution position predicts milder phenotype (type IV)"
        ),
        "disease_pathway": (
            "The [alpha1(I)]2[alpha2(I)]1 heterotrimer is the dominant collagen-I isoform in bone and skin. "
            "COL1A2 glycine substitutions disrupt the Gly-X-Y repeat in the alpha2 chain, slowing folding of the "
            "triplehelix from C to N terminus. The stalled helix undergoes excessive prolyl hydroxylation and "
            "glycosylation. Secreted fibres cannot pack correctly → defective D-period assembly → brittle bone. "
            "Biallelic skipping of exon 6 (arthrochalasis EDS) removes the pro-alpha2(I) N-propeptide cleavage site: "
            "uncleaved pro-chains cannot pack, generating lax connective tissue + bone fragility. "
            "Alpha2(I) null alleles (frameshift) are NOT OI because alpha1(I) homotrimers [alpha1(I)]3 form in "
            "alpha2 absence — functionally adequate, not dominant-negative. CRITICAL clinical distinction."
        ),
        "pathognomonic": (
            "Arthrochalasis EDS (biallelic COL1A2 splice): SEVERE CONGENITAL HIP DISLOCATION + extreme joint hypermobility + OI features — pathognomonic combination. "
            "AD OI type IV: moderate fractures + normal or light blue sclerae (not deep blue) + dentinogenesis imperfecta. "
            "COL1A2 null (frameshift only) → NO OI — normal skeleton, distinguishes from glycine substitutions."
        ),
        "treatment": (
            "AD OI types II–IV: same bisphosphonate/rodding approach as COL1A1. "
            "Arthrochalasis EDS: joint stabilisation (bracing, prolotherapy); surgical hip reduction neonatal if required. "
            "No collagen-specific therapies approved; TGF-β inhibition trials ongoing (fresolimumab, losartan). "
            "Physiotherapy mandatory for both OI and arthrochalasis EDS. "
            "Cardiac monitoring: aortic root dilation reported in arthrochalasis EDS — annual echo."
        ),
        "key_features": [
            "AD OI: identical spectrum to COL1A1 OI (blue sclerae, DI, hearing loss, fractures)",
            "Arthrochalasis EDS: extreme joint hypermobility + congenital bilateral hip dislocation at birth",
            "Kyphoscoliosis in arthrochalasis — progressive, may need spinal fusion",
            "Soft, velvety, hyperelastic skin in arthrochalasis variant",
            "COL1A2 null (frameshift) → NO OI — excess alpha1(I) homotrimers form (non-pathogenic)",
            "C-terminus glycine substitutions tend to milder OI type IV than N-terminus",
            "Dentinogenesis imperfecta 50% in AD OI (same as COL1A1)",
            "Wormian bones on skull X-ray (same as COL1A1 OI)"
        ],
        "key_ddx": [
            "COL1A1 OI — clinically indistinguishable; only molecular diagnosis separates",
            "Classical EDS (COL5A1/COL5A2) — joint laxity + skin fragility; no OI/DI",
            "Kyphoscoliotic EDS (PLOD1/FKBP14) — scoliosis + fragility; muscle hypotonia",
            "COL1A2 null carriers — completely normal (no haploinsufficiency OI for alpha2)"
        ],
        "fracture_burden": "Type II–IV: similar to COL1A1 equivalent types",
        "lethal_pct": 20,
        "blue_sclera_pct": 60,
        "hearing_impairment_pct": 35,
        "dentinogenesis_imperfecta_pct": 50,
        "on_bisphosphonate_pct": 60,
        "bone_mineral_density_zscore": -3.0,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- IFITM5 -- OI Type V ------------------------------------------------------------------
    {
        "gene": "IFITM5",
        "alt_name": (
            "IFITM5 (IFITM5-132aa-11p15.5 / AD -- "
            "OI-TYPE-V-HYPERPLASTIC-CALLUS-PATHOGNOMONIC -- "
            "CALCIFICATION-INTEROSSEOUS-MEMBRANE-PATHOGNOMONIC -- "
            "DENSE-METAPHYSEAL-BAND-X-RAY-SPECIFIC -- "
            "COLLAGEN-BIOCHEMISTRY-NORMAL-MISSES-ON-STANDARD-OI-PANEL)"
        ),
        "protein": (
            "IFITM5 -- 11p15.5 AD -- IFITM5-132aa -- "
            "BRIL-Bone-Restricted-IFITM-Like-Protein-14kDa-Transmembrane -- "
            "Interferon-Induced-Transmembrane-Protein-5-CD225-Domain -- "
            "Osteoblast-Mineralisation-Regulator-ER-Membrane-Localised -- "
            "Recurrent-De-Novo-c.-14C>T-5-UTR-Upstream-ATGATG-New-Start -- "
            "GOF-Gain-5-MALEP-N-Terminal-Extension-Dominant-Negative -- "
            "OMIM-Gene-614757-Disease-OI-Type-V-610967"
        ),
        "locus": "11p15.5",
        "protein_size": "132 aa / 14 kDa",
        "inheritance": (
            "AD (recurrent de novo, autosomal dominant with reduced penetrance reported); "
            "SINGLE RECURRENT MUTATION c.-14C>T (5′-UTR) in >95% of OI type V: "
            "creates new in-frame start codon → 5-aa N-terminal extension (MALEP) → "
            "dominant-negative effect on BRIL function in osteoblasts; "
            "rare p.Ser40Leu GOF variant also reported; "
            "COLLAGEN BIOCHEMISTRY COMPLETELY NORMAL — missed by type I collagen biochemical studies; "
            "clinically moderate OI spectrum, rarely lethal"
        ),
        "disease_category": (
            "OI type V (IFITM5) — unique radiological hallmarks not seen in any other OI type; "
            "hyperplastic callus after fractures (exuberant, tumour-like callus); "
            "calcification of interosseous membranes (radius-ulna, tibia-fibula); "
            "dense metaphyseal band distal radius/ulna on X-ray; "
            "radial head dislocation common; moderate OI; white sclerae"
        ),
        "disease_pathway": (
            "BRIL (bone-restricted IFITM-like protein, encoded by IFITM5) is an osteoblast membrane protein "
            "involved in bone mineralisation and ER calcium homeostasis. "
            "The recurrent c.-14C>T mutation in the 5′-UTR of IFITM5 creates an upstream AUG codon that "
            "adds 5 amino acids (MALEP sequence) to the N-terminus of BRIL. "
            "This N-terminal extension causes the mutant protein to mislocalize and act as a dominant-negative, "
            "disrupting normal BRIL function in osteoblasts. "
            "The consequence is defective mineralisation control, leading to abnormal callus formation after fractures "
            "(hyperplastic callus, sometimes mistaken for osteosarcoma) and progressive calcification of "
            "interosseous membranes restricting forearm/leg rotation. "
            "Collagen I structure is completely normal — standard collagen electrophoresis and type I collagen "
            "biochemistry are NORMAL, meaning OI type V is systematically missed if only collagen studies are ordered."
        ),
        "pathognomonic": (
            "HYPERPLASTIC CALLUS after fracture — exuberant, tumour-like callus (misdiagnosed as osteosarcoma) PATHOGNOMONIC. "
            "CALCIFICATION OF INTEROSSEOUS MEMBRANE (radius-ulna, tibia-fibula) on X-ray PATHOGNOMONIC. "
            "DENSE METAPHYSEAL BAND distal radius/ulna — specific for OI type V. "
            "RADIAL HEAD DISLOCATION in 50% — restricted forearm pronation-supination. "
            "NORMAL COLLAGEN BIOCHEMISTRY — distinguishes from COL1A1/COL1A2 OI."
        ),
        "treatment": (
            "Bisphosphonates (pamidronate IV / zoledronate): effective — reduces fracture rate and hyperplastic callus risk. "
            "Denosumab: reduces excessive bone resorption; use with caution (rebound fractures on cessation). "
            "Surgical: intramedullary rodding for long-bone deformity; forearm rotation — avoid forced reduction of "
            "calcified interosseous membrane. "
            "BIOPSY of hyperplastic callus if suspected osteosarcoma — ALWAYS consider OI type V first (shared radiological features); "
            "hyperplastic callus is benign OI reaction, NOT sarcoma."
        ),
        "key_features": [
            "Hyperplastic callus — exuberant fracture callus resembling soft-tissue mass (ruled out osteosarcoma before biopsy)",
            "Calcification of interosseous membranes — restricted forearm/ankle rotation on X-ray",
            "Dense metaphyseal band at distal radius — specific finding, visible from infancy",
            "Radial head dislocation (50%) — restricted pronation/supination",
            "White sclerae — distinguishes from types I/II/III",
            "Normal collagen biochemistry — OI type V missed if only collagen studies ordered",
            "Recurrent c.-14C>T mutation (>95% of cases) — targeted variant testing diagnostic",
            "Moderate severity — most patients ambulatory with aids"
        ],
        "key_ddx": [
            "Osteosarcoma — hyperplastic callus mimics sarcoma; ALWAYS check IFITM5 before biopsy",
            "Myositis ossificans — interosseous calcification differential; OI history distinguishes",
            "COL1A1/COL1A2 OI — white sclerae OI type IV; collagen study differentiates",
            "Fibrous dysplasia — expansile bone lesion; McCune-Albright GNAS testing"
        ],
        "fracture_burden": "Moderate — similar to OI type IV; fractures trigger pathognomonic hyperplastic callus",
        "lethal_pct": 0,
        "blue_sclera_pct": 0,
        "hearing_impairment_pct": 20,
        "dentinogenesis_imperfecta_pct": 15,
        "on_bisphosphonate_pct": 80,
        "bone_mineral_density_zscore": -2.8,
        "hyperplastic_callus_pct": 85,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- SERPINF1 -- OI Type VI ---------------------------------------------------------------
    {
        "gene": "SERPINF1",
        "alt_name": (
            "SERPINF1 (SERPINF1-418aa-17p13.3 / AR -- "
            "OI-TYPE-VI-FISH-SCALE-BONE-LAMELLAE-BIOPSY-PATHOGNOMONIC -- "
            "COLLAGEN-NORMAL-MISSED-STANDARD-OI-PANEL -- "
            "SERUM-PEDF-UNDETECTABLE-SCREENING-TEST -- "
            "ANTI-RANKL-DENOSUMAB-EXPERIMENTAL)"
        ),
        "protein": (
            "SERPINF1 -- 17p13.3 AR -- SERPINF1-418aa -- "
            "PEDF-Pigment-Epithelium-Derived-Factor-46kDa-Serpin-Superfamily -- "
            "Anti-Angiogenic-Neurotrophic-Anti-Tumour-Factor -- "
            "Bone-Osteoblast-Mineralisation-Regulator-NOT-Protease-Inhibitor -- "
            "Secreted-Glycoprotein-Retina-Bone-Widely-Expressed -- "
            "Biallelic-LOF-Absent-Serum-PEDF-Screening-Tool -- "
            "OMIM-Gene-172860-Disease-OI-Type-VI-613982"
        ),
        "locus": "17p13.3",
        "protein_size": "418 aa / 46 kDa",
        "inheritance": (
            "AR (biallelic LOF — null/missense); "
            "OI type VI: moderately severe — progressive fractures from 4–18 months; "
            "COLLAGEN TYPE I BIOCHEMISTRY COMPLETELY NORMAL — NOT an OI collagen disorder; "
            "SERUM PEDF LEVEL UNDETECTABLE by ELISA — simple diagnostic screening test; "
            "BONE BIOPSY shows FISH-SCALE BONE LAMELLAE (excess osteoid, mineralisation defect) PATHOGNOMONIC; "
            "not detectable by standard OI molecular panel; "
            "defective RANKL/osteoprotegerin axis — anti-RANKL approach rationale"
        ),
        "disease_category": (
            "OI type VI (SERPINF1) — a collagen-independent form of bone fragility; "
            "fish-scale bone lamellae on biopsy PATHOGNOMONIC; "
            "serum PEDF undetectable = screening test; "
            "collagen biochemistry normal — missed by standard OI gene panel if SERPINF1 excluded; "
            "denosumab + bisphosphonates; progressive deformity requiring rodding"
        ),
        "disease_pathway": (
            "PEDF (pigment epithelium-derived factor, encoded by SERPINF1) is a secreted glycoprotein of the serpin "
            "superfamily that lacks protease-inhibitor activity. In bone, PEDF is produced by osteoblasts and "
            "regulates mineralisation via modulation of the RANKL/OPG axis and direct effects on osteoid "
            "mineralisation machinery. "
            "Biallelic LOF of SERPINF1 → absent PEDF in serum and bone → defective secondary mineralisation, "
            "accumulation of unmineralised osteoid (excess osteoid volume), and disorganised lamellar bone "
            "architecture seen as fish-scale pattern on polarised light microscopy of iliac crest biopsy. "
            "Bone strength is severely impaired despite structurally normal collagen fibres. "
            "Excess RANKL-driven osteoclast activity secondary to PEDF absence — rationale for denosumab. "
            "PEDF also has anti-angiogenic and anti-tumour functions in retina, but ocular manifestations "
            "are not a prominent feature of OI type VI (distinct from eye disease uses of PEDF)."
        ),
        "pathognomonic": (
            "FISH-SCALE BONE LAMELLAE on iliac crest biopsy under polarised light microscopy — PATHOGNOMONIC for OI type VI. "
            "SERUM PEDF UNDETECTABLE by ELISA — simple, specific screening test before biopsy. "
            "COLLAGEN BIOCHEMISTRY NORMAL — missed by standard OI collagen electrophoresis. "
            "Progressive childhood onset (4–18 months) + moderate-severe fractures + white sclerae."
        ),
        "treatment": (
            "Bisphosphonates (pamidronate / zoledronate): moderately effective — reduce fractures. "
            "Denosumab (anti-RANKL monoclonal): superior to bisphosphonates in OI type VI — "
            "reduces osteoclast excess driven by PEDF absence; anti-RANKL mechanism specific to this type. "
            "Intramedullary rodding for progressive bowing. "
            "Monitor: serum PEDF (absent in affected, 50% in carriers) — diagnostic and therapeutic monitoring. "
            "Gene therapy (AAV-SERPINF1) preclinical. "
            "Avoid excessive fluoride — worsens mineralisation defect."
        ),
        "key_features": [
            "Fish-scale bone lamellae on biopsy (polarised light) — pathognomonic for type VI",
            "Serum PEDF undetectable — diagnostic screening test (normal in types I-V, VII-IX)",
            "Normal collagen I biochemistry — not detected by standard OI collagen studies",
            "Progressive fractures from 4–18 months — moderate-severe course",
            "White sclerae — unlike type I (blue), unlike type III (fades to white)",
            "Denosumab superiority over bisphosphonates (anti-RANKL rationale for PEDF absence)",
            "Excess osteoid volume on bone biopsy (mineralisation defect)",
            "No hyperplastic callus, no interosseous calcification (distinguishes from type V)"
        ],
        "key_ddx": [
            "OI types I–IV (COL1A1/COL1A2) — collagen biochemistry normal in type VI (key diagnostic difference)",
            "OI type V (IFITM5) — collagen also normal but hyperplastic callus + IOM calcification present",
            "Hypophosphatasia (ALPL) — low ALP pathognomonic; PEDF normal",
            "X-linked hypophosphatemia (PHEX) — low phosphate; elevated FGF23; distinct phosphopenic bone disease"
        ],
        "fracture_burden": "Moderate-severe — progressive from 4–18 months; multiple rib + long bone fractures",
        "lethal_pct": 2,
        "blue_sclera_pct": 5,
        "hearing_impairment_pct": 15,
        "dentinogenesis_imperfecta_pct": 10,
        "on_bisphosphonate_pct": 75,
        "bone_mineral_density_zscore": -3.5,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- CRTAP -- OI Type VII -----------------------------------------------------------------
    {
        "gene": "CRTAP",
        "alt_name": (
            "CRTAP (CRTAP-339aa-3p22.3 / AR -- "
            "OI-TYPE-VII-RHIZOMELIA-DISTINCTIVE-PATHOGNOMONIC -- "
            "OVERMODIFIED-COLLAGEN-ELECTROPHORESIS-DIAGNOSTIC -- "
            "LETHAL-NULL-PERINATALLY-HYPOPLASTIC-THORAX -- "
            "PROLYL-3-HYDROXYLATION-COMPLEX-COMPONENT)"
        ),
        "protein": (
            "CRTAP -- 3p22.3 AR -- CRTAP-339aa -- "
            "Cartilage-Associated-Protein-37kDa-ER-Resident -- "
            "Prolyl-3-Hydroxylation-Ternary-Complex-CRTAP-P3H1-CypB -- "
            "Required-for-P3H1-Stability-Not-Enzymatic-Activity-Itself -- "
            "Biallelic-Null-Lethal-Hypomorphic-Moderate-Severe-OI -- "
            "Rhizomelia-Proximal-Limb-Shortening-Type-VII-Distinguishing -- "
            "OMIM-Gene-605497-Disease-OI-Type-VII-610682"
        ),
        "locus": "3p22.3",
        "protein_size": "339 aa / 37 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "CRTAP null → OI type VII LETHAL PERINATALLY: rhizomelia + severe osteopenia + hypoplastic thorax → "
            "respiratory failure at birth / early infancy; "
            "hypomorphic (some residual protein) → survivable severe OI type VII: "
            "RHIZOMELIA (shortened proximal limbs: humeri/femora > distal) — distinctive feature; "
            "overmodified pro-alpha1(I) collagen on SDS-PAGE — diagnostic signature; "
            "multiple fractures at birth; white sclerae; no dentinogenesis imperfecta"
        ),
        "disease_category": (
            "OI type VII (CRTAP) — prolyl-3-hydroxylation complex defect; "
            "rhizomelia DISTINCTIVE for OI type VII (proximal limb shortening); "
            "overmodified collagen on electrophoresis = diagnostic; "
            "null = lethal; hypomorphic = moderate-severe survivable; "
            "white sclerae; normal collagen structure but abnormally modified"
        ),
        "disease_pathway": (
            "The prolyl 3-hydroxylation complex (CRTAP + P3H1/LEPRE1 + PPIB/CypB) hydroxylates a single conserved "
            "prolyl residue at position 986 of the pro-alpha1(I) collagen chain in the endoplasmic reticulum. "
            "This post-translational modification is required for efficient folding of the triple helix. "
            "CRTAP acts as a structural scaffold protein — it is required for P3H1 stability and ER localisation "
            "but is not itself the hydroxylase. "
            "Loss of CRTAP destabilises the entire ternary complex (P3H1 is degraded without CRTAP). "
            "Without prolyl 3-hydroxylation at Pro986, the pro-collagen folds more slowly, leading to excessive "
            "4-hydroxyproline and glycosylation (overmodification) — detectable as characteristic band-shift on "
            "SDS-PAGE of radiolabelled collagen. "
            "The modified collagen has abnormal fibril structure → severely fragile bone. "
            "RHIZOMELIA in OI type VII may reflect a role of CRTAP in cartilage-specific collagen modification "
            "beyond type I, linking to type II collagen in growth plates (CRTAP = cartilage-associated protein)."
        ),
        "pathognomonic": (
            "RHIZOMELIA (disproportionate shortening of proximal limb segments: humeri >> radius/ulna, femora >> tibia) PATHOGNOMONIC for OI type VII. "
            "OVERMODIFIED PRO-ALPHA1(I) COLLAGEN on SDS-PAGE (radiolabelled) — diagnostic band-shift. "
            "NULL = LETHAL PERINATALLY: rhizomelia + hypoplastic thorax + multiple rib fractures — die at birth."
        ),
        "treatment": (
            "Bisphosphonates: reduce fractures in survivable hypomorphic type VII. "
            "Respiratory support critical at birth: CPAP/ventilation for hypoplastic thorax in lethal type. "
            "Intramedullary rodding for progressive bowing. "
            "Genetic counselling: carrier frequency in some populations (Cree-Oji-Cree First Nations — founder p.Arg462X). "
            "No specific CRTAP-targeted therapy; supportive care cornerstone. "
            "Growth hormone trials in severe OI (including type VII) — limited evidence."
        ),
        "key_features": [
            "Rhizomelia — proximal limb shortening (humeri >> forearms; femora >> legs) — pathognomonic for type VII",
            "Overmodified collagen on SDS-PAGE — Pro986 under-hydroxylated → slow fold → over-glycosylated",
            "White sclerae — unlike type I (blue sclerae)",
            "No dentinogenesis imperfecta — collagen structure abnormal but not DI-causing",
            "Lethal perinatal in null: hypoplastic thorax + multiple fractures at birth",
            "Cree-Oji-Cree First Nations founder mutation (North America) — founder p.Arg462X",
            "CRTAP stabilises P3H1 — CRTAP null → secondary P3H1 degradation",
            "Multiple fractures at birth; barrel chest deformity in survivors"
        ],
        "key_ddx": [
            "LEPRE1/P3H1 OI type VIII — identical overmodified collagen; LEPRE1 enzyme activity distinguishes",
            "PPIB OI type IX — same ternary complex; collagen overmodified",
            "Achondroplasia (FGFR3) — rhizomelia without fractures; specific FGFR3 p.Gly380Arg",
            "Thanatophoric dysplasia (FGFR3 II) — rhizomelia + lethal + distinct X-ray (telephone receiver femur)"
        ],
        "fracture_burden": "Severe — multiple fractures at birth in lethal forms; multiple per year in survivors",
        "lethal_pct": 50,
        "blue_sclera_pct": 5,
        "hearing_impairment_pct": 10,
        "dentinogenesis_imperfecta_pct": 0,
        "on_bisphosphonate_pct": 70,
        "bone_mineral_density_zscore": -4.1,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- LEPRE1 -- OI Type VIII ---------------------------------------------------------------
    {
        "gene": "LEPRE1",
        "alt_name": (
            "LEPRE1 (LEPRE1-736aa-1p34.2 / AR -- "
            "OI-TYPE-VIII-WEST-AFRICAN-CAMEROONIAN-FOUNDER-pTrp339Ter -- "
            "OVERMODIFIED-COLLAGEN-ELECTROPHORESIS-DIAGNOSTIC -- "
            "NEONATAL-LETHAL-SEVERE-WHITE-SCLERA -- "
            "BIOCHEMICAL-DIAGNOSIS-ESSENTIAL-BEFORE-WES-WEST-AFRICA)"
        ),
        "protein": (
            "LEPRE1 -- 1p34.2 AR -- LEPRE1-736aa -- "
            "P3H1-Prolyl-3-Hydroxylase-1-84kDa-ER-Resident -- "
            "Collagen-Prolyl-3-Hydroxylase-Alpha-Ketoglutarate-Dependent-Dioxygenase -- "
            "Hydroxylates-Pro986-alpha1(I)-Collagen-Single-Critical-Site -- "
            "Forms-Ternary-Complex-CRTAP-LEPRE1-PPIB -- "
            "West-African-Founder-pTrp339Ter-c1015G>T-Pathognomonic -- "
            "OMIM-Gene-610339-Disease-OI-Type-VIII-610915"
        ),
        "locus": "1p34.2",
        "protein_size": "736 aa / 84 kDa",
        "inheritance": (
            "AR (biallelic LOF); "
            "WEST AFRICAN / CAMEROONIAN FOUNDER MUTATION p.Trp339Ter (c.1015G>T) — in >80% of West African OI type VIII; "
            "neonatal lethal or severe: white sclerae, severe osteopenia, multiple fractures at birth; "
            "OVERMODIFIED alpha1(I) collagen on SDS-PAGE at Pro986 — identical signature to CRTAP; "
            "BIOCHEMICAL COLLAGEN ELECTROPHORESIS ESSENTIAL in West African patients — diagnosis before WES "
            "because standard DNA panels may miss this founder; "
            "hypomorphic variants (p.Thr328Met) → survivable severe-moderate OI"
        ),
        "disease_category": (
            "OI type VIII (LEPRE1/P3H1) — prolyl 3-hydroxylase 1 catalytic subunit deficiency; "
            "West African/Cameroonian founder p.Trp339Ter in >80% affected; "
            "identical collagen overmodification to CRTAP (same complex); "
            "neonatal lethal severe form; white sclerae; "
            "biochemical diagnosis critical — targeted founder allele testing before WES in West African patients"
        ),
        "disease_pathway": (
            "P3H1 (encoded by LEPRE1) is the enzymatic component of the ternary prolyl-3-hydroxylation complex "
            "(CRTAP + P3H1 + CypB/PPIB) in the rough ER of fibroblasts and osteoblasts. "
            "P3H1 catalyses hydroxylation of Pro-986 in the alpha1(I) chain using alpha-ketoglutarate as co-substrate. "
            "This is the ONLY prolyl residue modified by 3-hydroxylation in collagen type I (vs the numerous 4-hydroxy "
            "prolines added by P4H). P3H1 null → complete loss of Pro986 3-hydroxylation → excessive folding delay → "
            "massive overmodification of all remaining 4-hydroxyproline and glycosylation sites → "
            "characteristic smear on SDS-PAGE. The fibres formed are structurally defective → very brittle bone. "
            "The West African founder mutation p.Trp339Ter (c.1015G>T) truncates the P3H1 protein, eliminating "
            "its catalytic activity. This founder allele is present in ~1/200 West African individuals, making "
            "OI type VIII the most common severe OI form in West and Central Africa. "
            "Standard WES/OI gene panels designed for European populations often do not prioritise LEPRE1 — "
            "biochemical diagnosis (collagen electrophoresis) in fibroblast culture identifies the characteristic "
            "band-shift and should guide molecular work-up in West African families before expensive WES."
        ),
        "pathognomonic": (
            "WEST AFRICAN FOUNDER p.Trp339Ter (c.1015G>T) — in >80% of West African OI type VIII; targeted allele testing confirms. "
            "OVERMODIFIED PRO-ALPHA1(I) COLLAGEN AT PRO986 on SDS-PAGE — identical to CRTAP (same complex). "
            "BIOCHEMICAL COLLAGEN ELECTROPHORESIS in fibroblasts — diagnostic before WES in West African patients. "
            "NEONATAL LETHAL FORM: white sclerae (NOT blue) + multiple fractures + severe osteopenia."
        ),
        "treatment": (
            "Bisphosphonates (pamidronate IV): standard in survivable forms — modest fracture reduction. "
            "Respiratory support at birth for severe/lethal type. "
            "Intramedullary rodding for deformity in survivors. "
            "Ethnic-specific founder allele testing: targeted p.Trp339Ter assay (fast, cheap) before WES "
            "in all West African families with severe childhood OI and white sclerae. "
            "Carrier testing for p.Trp339Ter in at-risk West African populations: "
            "preconception counselling, prenatal diagnosis available. "
            "No P3H1-specific enzyme replacement available."
        ),
        "key_features": [
            "West African/Cameroonian founder p.Trp339Ter (c.1015G>T) in >80% — targeted testing first",
            "Overmodified collagen SDS-PAGE — Pro986 underhydroxylated → same pattern as CRTAP type VII",
            "White sclerae — critical DDx from type I (blue sclerae)",
            "Neonatal lethal in null: severe osteopenia + multiple fractures at birth",
            "No dentinogenesis imperfecta (same as CRTAP type VII)",
            "Biochemical diagnosis (collagen electrophoresis) essential in West African patients before WES",
            "Carrier frequency ~1/200 West Africans — significant genetic burden",
            "Hypomorphic variants (p.Thr328Met) → survivable severe-moderate OI"
        ],
        "key_ddx": [
            "CRTAP OI type VII — rhizomelia distinguishes CRTAP from LEPRE1 (no rhizomelia in type VIII)",
            "COL1A1/COL1A2 OI type II/III — blue sclerae vs white sclerae distinguishes",
            "PPIB OI type IX — same ternary complex; rarer, Middle East/Asia families",
            "Thanatophoric dysplasia — lethal at birth but distinct skeletal dysmorphology"
        ],
        "fracture_burden": "Severe — multiple fractures at birth in lethal forms; severe per year in survivors",
        "lethal_pct": 55,
        "blue_sclera_pct": 3,
        "hearing_impairment_pct": 10,
        "dentinogenesis_imperfecta_pct": 0,
        "on_bisphosphonate_pct": 65,
        "bone_mineral_density_zscore": -4.3,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- PPIB -- OI Type IX -------------------------------------------------------------------
    {
        "gene": "PPIB",
        "alt_name": (
            "PPIB (PPIB-212aa-15q22.31 / AR -- "
            "OI-TYPE-IX-CYCLOPHILIN-B-PROLYL-ISOMERASE -- "
            "OVERMODIFIED-COLLAGEN-SAME-PATTERN-CRTAP-LEPRE1 -- "
            "TERNARY-COMPLEX-THIRD-COMPONENT -- "
            "MIDDLE-EAST-ASIA-FAMILIES-RARE)"
        ),
        "protein": (
            "PPIB -- 15q22.31 AR -- PPIB-212aa -- "
            "Cyclophilin-B-CypB-Peptidyl-Prolyl-Cis-Trans-Isomerase-24kDa-ER-Resident -- "
            "Catalyses-Cis-Trans-Isomerisation-Proline-Peptide-Bonds-Chaperone -- "
            "Third-Component-Prolyl-3-Hydroxylation-Ternary-Complex -- "
            "Cyclophilin-Family-FK506-Cyclosporine-Related-Not-Immunosuppressant-Function -- "
            "Biallelic-LOF-Overmodified-Collagen-OI-Type-IX -- "
            "OMIM-Gene-123841-Disease-OI-Type-IX-259440"
        ),
        "locus": "15q22.31",
        "protein_size": "212 aa / 24 kDa",
        "inheritance": (
            "AR (biallelic LOF — usually null or missense abolishing isomerase activity); "
            "OI type IX: severe to moderate — white sclerae, multiple fractures; "
            "OVERMODIFIED COLLAGEN on SDS-PAGE (same band-shift as CRTAP/LEPRE1) — "
            "ternary complex integrity disrupted even though PPIB is isomerase not hydroxylase; "
            "PPIB biallelic null → P3H1 complex destabilised → Pro986 underhydroxylated; "
            "extremely rare globally; reported families from Middle East (Lebanon, Jordan) and Asia"
        ),
        "disease_category": (
            "OI type IX (PPIB) — cyclophilin B / peptidyl-prolyl isomerase deficiency; "
            "third component of ternary prolyl-3-hydroxylation complex; "
            "overmodified collagen SDS-PAGE identical to CRTAP/LEPRE1 types VII/VIII; "
            "white sclerae + severe-moderate bone fragility; "
            "PPIB null destabilises P3H1 enzyme — complex integrity required for hydroxylase function"
        ),
        "disease_pathway": (
            "PPIB encodes Cyclophilin B (CypB), an ER-resident peptidyl-prolyl cis-trans isomerase (PPIase). "
            "CypB accelerates the slow rate-limiting cis-to-trans isomerisation of Xaa-Pro bonds in nascent "
            "pro-collagen chains as they fold. It is the third component of the CRTAP-P3H1-CypB ternary complex. "
            "Although PPIB itself is not the prolyl 3-hydroxylase, loss of PPIB destabilises P3H1 in the complex "
            "and impairs Pro986 hydroxylation — leading to the characteristic overmodification pattern. "
            "Separately, CypB has a chaperone role beyond Pro986 modification: it facilitates overall "
            "collagen chain folding speed. PPIB null → impaired folding chaperone activity → prolonged ER "
            "residence → global over-4-hydroxylation and glycosylation of collagen → defective fibril. "
            "Despite being in the cyclosporine-binding family, PPIB function in OI is independent of "
            "immunosuppressant pathways — cyclosporine does not treat OI type IX. "
            "The rarity of OI type IX reflects the lower frequency of PPIB biallelic mutations globally; "
            "consanguineous Middle Eastern and Asian families are the main source of reported cases."
        ),
        "pathognomonic": (
            "OVERMODIFIED PRO-ALPHA1(I) COLLAGEN on SDS-PAGE — identical band-shift to types VII (CRTAP) and VIII (LEPRE1); ternary complex test distinguishes. "
            "P3H1 ENZYME ACTIVITY NORMAL (PPIB is isomerase not hydroxylase) — distinguishes from LEPRE1 type VIII. "
            "CONSANGUINEOUS MIDDLE EASTERN / ASIAN FAMILY + SEVERE WHITE-SCLERAL OI + normal P4H + overmodified collagen → OI type IX."
        ),
        "treatment": (
            "Bisphosphonates: standard — pamidronate IV or zoledronate; reduces fractures. "
            "Intramedullary rodding for progressive long-bone deformity. "
            "Physiotherapy and hydrotherapy — essential for ambulation. "
            "No CypB-specific therapy available; cyclosporine does NOT treat OI type IX. "
            "Genetic counselling: consanguinity counselling; carrier detection by molecular testing. "
            "Supportive respiratory care if thoracic involvement present (less common than CRTAP type VII)."
        ),
        "key_features": [
            "Overmodified collagen SDS-PAGE — same pattern as types VII (CRTAP) and VIII (LEPRE1)",
            "P3H1 enzymatic activity NORMAL — PPIB is isomerase/chaperone not hydroxylase (DDx from LEPRE1)",
            "White sclerae — unlike OI type I (blue sclerae)",
            "No dentinogenesis imperfecta (no DI in ternary complex OI types)",
            "Severe–moderate OI; multiple fractures from infancy",
            "Consanguineous Middle East/Asia families — rare globally",
            "Complex integrity: PPIB null destabilises P3H1 (explains identical overmodification)",
            "Not cyclosporine-responsive (different molecular function from immunosuppressant pathway)"
        ],
        "key_ddx": [
            "CRTAP OI type VII — rhizomelia distinguishes CRTAP; no rhizomelia in PPIB",
            "LEPRE1 OI type VIII — identical overmodification; P3H1 enzyme activity LOW in LEPRE1, NORMAL in PPIB",
            "COL1A1/COL1A2 OI — collagen normal size on SDS-PAGE (no overmodification in structural collagen OI)",
            "FK506-binding protein disorders — FKBP10 shares name class but different disease (Bruck syndrome)"
        ],
        "fracture_burden": "Severe — multiple fractures from birth/infancy",
        "lethal_pct": 20,
        "blue_sclera_pct": 3,
        "hearing_impairment_pct": 10,
        "dentinogenesis_imperfecta_pct": 0,
        "on_bisphosphonate_pct": 72,
        "bone_mineral_density_zscore": -3.9,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 0,
        "nbs_indicated": False,
    },
    # -- FKBP10 -- OI Type XI / Bruck Syndrome 1 ---------------------------------------------
    {
        "gene": "FKBP10",
        "alt_name": (
            "FKBP10 (FKBP10-582aa-17q21.2 / AR -- "
            "OI-TYPE-XI-BRUCK-SYNDROME-1-CONTRACTURES-AT-BIRTH-PLUS-OI-PATHOGNOMONIC -- "
            "LYSYL-HYDROXYLASE-2-ACTIVITY-REDUCED-PYRIDINOLINE-CROSS-LINK-DEFECT -- "
            "TURKISH-FOUNDER-c831+1G>T-PTERYGIUM-JOINT-WEBBING -- "
            "FKBP65-COLLAGEN-CHAPERONE-ER)"
        ),
        "protein": (
            "FKBP10 -- 17q21.2 AR -- FKBP10-582aa -- "
            "FKBP65-FK506-Binding-Protein-65kDa-ER-Resident-Collagen-Chaperone -- "
            "Four-FKBP-Domains-Peptidyl-Prolyl-Isomerase-Activity -- "
            "Collagen-I-Secretion-Chaperone-Lysyl-Hydroxylase-2-LH2-Partner -- "
            "LH2-Activity-Reduced-Without-FKBP65-Defective-Pyridinoline-Cross-Links -- "
            "Turkish-Founder-c831+1G>T-Splice-Bruck-Syndrome-1 -- "
            "OMIM-Gene-607063-Disease-OI-Type-XI-Bruck-Syndrome-1-259450"
        ),
        "locus": "17q21.2",
        "protein_size": "582 aa / 65 kDa",
        "inheritance": (
            "AR (biallelic LOF — null, missense, splice); "
            "OI type XI / Bruck syndrome 1: "
            "CONTRACTURES AT BIRTH + OI (bone fragility) — PATHOGNOMONIC COMBINATION; "
            "PTERYGIUM (skin webbing at elbows/knees) in Bruck variant; "
            "Turkish founder c.831+1G>T splice variant; Omani/Arab consanguineous families; "
            "LYSYL HYDROXYLASE 2 (LH2/PLOD2) activity reduced — defective telopeptide lysine hydroxylation "
            "→ abnormal pyridinoline cross-links in collagen fibrils; "
            "moderate-severe OI; white sclerae; no dentinogenesis imperfecta"
        ),
        "disease_category": (
            "OI type XI / Bruck syndrome 1 (FKBP10) — FKBP65 collagen chaperone deficiency; "
            "contractures at birth + OI = Bruck syndrome 1 PATHOGNOMONIC; "
            "pterygium in classic Bruck; "
            "LH2 activity reduced (cross-link defect mechanism); "
            "Turkish founder c.831+1G>T; "
            "moderate-severe course; no hyperplastic callus; no rhizomelia"
        ),
        "disease_pathway": (
            "FKBP10 encodes FKBP65, an ER-resident chaperone with four FKBP-type peptidyl-prolyl isomerase domains. "
            "FKBP65 interacts directly with type I collagen chains and facilitates their folding and secretion. "
            "Critically, FKBP65 is required for normal activity of lysyl hydroxylase 2 (LH2, encoded by PLOD2). "
            "LH2 hydroxylates telopeptide lysine residues in collagen type I, which are the key sites for "
            "pyridinoline and lysyl-pyridinoline cross-link formation between fibrils. "
            "FKBP10 null → FKBP65 absent → LH2 activity reduced (secondary) → "
            "defective telopeptide hydroxylation → abnormal collagen cross-linking → structurally weak fibrils → "
            "bone fragility (OI) AND joint connective tissue weakness (contractures, pterygium). "
            "The JOINT CONTRACTURES at birth reflect failure of normal tendon/ligament collagen crosslink formation. "
            "Pterygium (skin webbing) occurs in the original Bruck syndrome description (Bruck 1897) — "
            "pterygia at elbows, knees, reflecting juxta-articular connective tissue shortening from birth. "
            "Collagen electrophoresis: TYPE I COLLAGEN SIZE NORMAL — no overmodification (unlike CRTAP/LEPRE1/PPIB); "
            "urinary total pyridinoline/deoxypyridinoline ratio may reflect cross-link defect."
        ),
        "pathognomonic": (
            "CONTRACTURES AT BIRTH + BONE FRAGILITY (OI) = BRUCK SYNDROME 1 PATHOGNOMONIC. "
            "PTERYGIUM (skin webbing at joint flexures: elbows, knees, ankles) in classic Bruck presentation. "
            "TURKISH FOUNDER c.831+1G>T splice mutation — targeted allele testing in consanguineous Turkish families with Bruck. "
            "NORMAL collagen SDS-PAGE SIZE (no overmodification — distinct from CRTAP/LEPRE1/PPIB types VII-IX)."
        ),
        "treatment": (
            "Bisphosphonates: reduce fractures (pamidronate / zoledronate). "
            "Contracture management: serial casting from neonatal period; surgical tendon release for severe contractures. "
            "Pterygium release surgery if function-limiting. "
            "Intramedullary rodding for progressive bowing. "
            "Physiotherapy essential — contractures worsen without active rehabilitation. "
            "Genetic counselling: targeted c.831+1G>T for Turkish families; WES for others. "
            "No FKBP65-specific or cross-link-correcting therapy approved; "
            "LH2/PLOD2 recombinant enzyme replacement in preclinical development."
        ),
        "key_features": [
            "Contractures at birth — rigid joint contractures (knees, elbows, ankles) present at delivery",
            "Pterygium — skin webbing at joint flexures (classic Bruck syndrome feature)",
            "Bone fragility (OI) — multiple fractures; moderate-severe course",
            "Normal collagen SDS-PAGE — no overmodification (LH2 cross-link defect, not folding defect)",
            "White sclerae — no blue sclerae (no type I OI features)",
            "No dentinogenesis imperfecta",
            "Turkish founder c.831+1G>T — dominant in consanguineous Turkish OI+contracture families",
            "FKBP65 loss → secondary LH2/PLOD2 activity reduction (chaperone-enzyme dependency)"
        ],
        "key_ddx": [
            "Bruck syndrome 2 (PLOD2/LH2 mutations directly) — identical phenotype; "
            "FKBP10 (Bruck 1) vs PLOD2 (Bruck 2) only by molecular testing",
            "Arthrogryposis multiplex congenita — contractures at birth without OI; many genetic causes (RYR1, BICD2)",
            "Kyphoscoliotic EDS (PLOD1/FKBP14) — scoliosis + fragility + hypotonia; less severe OI",
            "OI type V (IFITM5) — hyperplastic callus + IOM calcification; no contractures"
        ],
        "fracture_burden": "Moderate-severe — multiple fractures from infancy; worsens with contractures",
        "lethal_pct": 3,
        "blue_sclera_pct": 3,
        "hearing_impairment_pct": 15,
        "dentinogenesis_imperfecta_pct": 5,
        "on_bisphosphonate_pct": 75,
        "bone_mineral_density_zscore": -3.6,
        "hyperplastic_callus_pct": 0,
        "contractures_at_birth_pct": 92,
        "nbs_indicated": False,
    },
]


def _make_patients(gene_entry, n=40, seed=None):
    rng = random.Random(seed)
    patients = []
    for i in range(n):
        lethal = rng.random() < gene_entry["lethal_pct"] / 100
        fracture_count = rng.randint(2, 8) if not lethal else rng.randint(8, 30)
        bmd = gene_entry["bone_mineral_density_zscore"] + rng.gauss(0, 0.6)
        patients.append({
            "patient_id": f"{gene_entry['gene']}-{seed}-{i:03d}",
            "gene": gene_entry["gene"],
            "fracture_count_lifetime": fracture_count,
            "bone_mineral_density_zscore": round(bmd, 2),
            "lethal_perinatal": lethal,
            "blue_sclera": rng.random() < gene_entry["blue_sclera_pct"] / 100,
            "hearing_impairment": rng.random() < gene_entry["hearing_impairment_pct"] / 100,
            "dentinogenesis_imperfecta": rng.random() < gene_entry["dentinogenesis_imperfecta_pct"] / 100,
            "on_bisphosphonate": rng.random() < gene_entry["on_bisphosphonate_pct"] / 100,
            "hyperplastic_callus": rng.random() < gene_entry["hyperplastic_callus_pct"] / 100,
            "contractures_at_birth": rng.random() < gene_entry["contractures_at_birth_pct"] / 100,
            "wormian_bones": rng.random() < 0.55,
            "scoliosis": rng.random() < 0.40,
            "ambulation": rng.choice(
                ["independent", "ambulatory-aids", "wheelchair"]
                if not lethal
                else ["deceased-perinatal"]
            ),
        })
    return patients


def generate_overview():
    all_patients = []
    gene_summaries = []
    for idx, entry in enumerate(OI_GENES):
        pts = _make_patients(entry, n=40, seed=SEED_BASE + idx)
        all_patients.extend(pts)
        surviving = [p for p in pts if not p["lethal_perinatal"]]
        gene_summaries.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_name": entry["disease_category"].split(" — ")[0].strip(),
            "pathognomonic_short": entry["pathognomonic"].split(". ")[0][:120],
            "lethal_pct": entry["lethal_pct"],
            "blue_sclera_pct": entry["blue_sclera_pct"],
            "hearing_impairment_pct": entry["hearing_impairment_pct"],
            "dentinogenesis_imperfecta_pct": entry["dentinogenesis_imperfecta_pct"],
            "on_bisphosphonate_pct": entry["on_bisphosphonate_pct"],
            "hyperplastic_callus_pct": entry["hyperplastic_callus_pct"],
            "contractures_at_birth_pct": entry["contractures_at_birth_pct"],
            "avg_bmd_zscore": round(sum(p["bone_mineral_density_zscore"] for p in pts) / 40, 2),
            "avg_fractures": round(sum(p["fracture_count_lifetime"] for p in pts) / 40, 1),
            "wheelchair_pct": round(sum(1 for p in surviving if p["ambulation"] == "wheelchair") / max(len(surviving), 1) * 100, 1),
        })

    surviving = [p for p in all_patients if not p["lethal_perinatal"]]
    agg = {
        "lethal_pct": round(sum(1 for p in all_patients if p["lethal_perinatal"]) / 320 * 100, 1),
        "blue_sclera_pct": round(sum(1 for p in all_patients if p["blue_sclera"]) / 320 * 100, 1),
        "hearing_impairment_pct": round(sum(1 for p in all_patients if p["hearing_impairment"]) / 320 * 100, 1),
        "dentinogenesis_imperfecta_pct": round(sum(1 for p in all_patients if p["dentinogenesis_imperfecta"]) / 320 * 100, 1),
        "on_bisphosphonate_pct": round(sum(1 for p in all_patients if p["on_bisphosphonate"]) / 320 * 100, 1),
        "hyperplastic_callus_pct": round(sum(1 for p in all_patients if p["hyperplastic_callus"]) / 320 * 100, 1),
        "contractures_pct": round(sum(1 for p in all_patients if p["contractures_at_birth"]) / 320 * 100, 1),
        "avg_bmd_zscore": round(sum(p["bone_mineral_density_zscore"] for p in all_patients) / 320, 2),
        "avg_fractures": round(sum(p["fracture_count_lifetime"] for p in all_patients) / 320, 1),
        "wheelchair_pct": round(sum(1 for p in surviving if p["ambulation"] == "wheelchair") / max(len(surviving), 1) * 100, 1),
    }

    return {
        "title": "Hereditary Osteogenesis Imperfecta Atlas",
        "subtitle": "Complete 8-Gene Bone Fragility & Collagen-I Deficiency Reference",
        "genes": [e["gene"] for e in OI_GENES],
        "n_genes": 8,
        "total_patients": 320,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "disease_classes": [
            "COL1A1 — OI types I/II/III/IV (null/glycine-subst AD)",
            "COL1A2 — OI types II/III/IV / Arthrochalasis EDS (glycine-subst AD / biallelic splice AR)",
            "IFITM5 — OI type V (hyperplastic callus + IOM calcification — PATHOGNOMONIC)",
            "SERPINF1 — OI type VI (fish-scale lamellae on biopsy; PEDF absent — AR)",
            "CRTAP — OI type VII (rhizomelia; overmodified collagen — AR lethal/severe)",
            "LEPRE1 — OI type VIII (West African founder p.Trp339Ter; overmodified collagen — AR)",
            "PPIB — OI type IX (cyclophilin B; ternary complex; overmodified collagen — AR)",
            "FKBP10 — OI type XI / Bruck syndrome 1 (contractures at birth + OI — PATHOGNOMONIC)",
        ],
        "gene_summary": gene_summaries,
        "aggregate_metrics": agg,
        "clinical_pearls": [
            "COL1A1/COL1A2 NULL alleles (frameshift) → OI type I MILD (haploinsufficiency); GLYCINE SUBSTITUTION → severe types II-IV (dominant-negative) — ALWAYS classify variant type before prognosis.",
            "IFITM5 OI type V: HYPERPLASTIC CALLUS misdiagnosed as osteosarcoma — ALWAYS check IFITM5 c.-14C>T before biopsy; collagen biochemistry NORMAL.",
            "SERPINF1 OI type VI: serum PEDF UNDETECTABLE (ELISA) is a simple screening test; fish-scale lamellae on iliac crest biopsy PATHOGNOMONIC — denosumab superior to bisphosphonates.",
            "CRTAP type VII: RHIZOMELIA (proximal limb shortening) PATHOGNOMONIC — distinguishes from LEPRE1 (no rhizomelia) and PPIB (no rhizomelia).",
            "LEPRE1 type VIII: WEST AFRICAN FOUNDER p.Trp339Ter (>80% of West African OI VIII) — targeted founder allele testing before expensive WES in West African families.",
            "PPIB type IX: P3H1 ENZYME ACTIVITY NORMAL despite overmodified collagen — PPIB is isomerase/chaperone not hydroxylase; this distinguishes from LEPRE1 biochemically.",
            "FKBP10 / Bruck syndrome 1: CONTRACTURES AT BIRTH + OI = pathognomonic Bruck syndrome — PTERYGIUM (joint webbing) highly specific; Turkish founder c.831+1G>T.",
            "Child abuse (NAI) DDx: ALL children with multiple unexplained fractures MUST have metabolic bone disease screen + genetics before safeguarding report — OI can exactly mimic NAI radiologically.",
            "BISPHOSPHONATES (pamidronate IV or zoledronate): cornerstone for all moderate-severe OI — reduce fracture rate up to 50%; start as early as diagnosis in types II/III/IV.",
            "COL1A2 NULL alleles: frameshift → NO OI (alpha1(I) homotrimers compensate) — CRITICAL: this is why COL1A2 haploinsufficiency is NOT pathogenic for OI, unlike COL1A1.",
        ],
    }


def generate_breakdown():
    breakdowns = []
    for idx, entry in enumerate(OI_GENES):
        pts = _make_patients(entry, n=40, seed=SEED_BASE + idx)
        surviving = [p for p in pts if not p["lethal_perinatal"]]
        n_s = max(len(surviving), 1)
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
            "lethal_pct": round(sum(1 for p in pts if p["lethal_perinatal"]) / 40 * 100, 1),
            "blue_sclera_pct": round(sum(1 for p in pts if p["blue_sclera"]) / 40 * 100, 1),
            "hearing_impairment_pct": round(sum(1 for p in pts if p["hearing_impairment"]) / 40 * 100, 1),
            "dentinogenesis_imperfecta_pct": round(sum(1 for p in pts if p["dentinogenesis_imperfecta"]) / 40 * 100, 1),
            "on_bisphosphonate_pct": round(sum(1 for p in pts if p["on_bisphosphonate"]) / 40 * 100, 1),
            "hyperplastic_callus_pct": round(sum(1 for p in pts if p["hyperplastic_callus"]) / 40 * 100, 1),
            "contractures_at_birth_pct": round(sum(1 for p in pts if p["contractures_at_birth"]) / 40 * 100, 1),
            "avg_bmd_zscore": round(sum(p["bone_mineral_density_zscore"] for p in pts) / 40, 2),
            "avg_fractures": round(sum(p["fracture_count_lifetime"] for p in pts) / 40, 1),
            "wormian_bones_pct": round(sum(1 for p in pts if p["wormian_bones"]) / 40 * 100, 1),
            "scoliosis_pct": round(sum(1 for p in pts if p["scoliosis"]) / 40 * 100, 1),
            "wheelchair_pct": round(sum(1 for p in surviving if p["ambulation"] == "wheelchair") / n_s * 100, 1),
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
                "lethal_pct": entry["lethal_pct"],
                "blue_sclera_pct": entry["blue_sclera_pct"],
                "hearing_impairment_pct": entry["hearing_impairment_pct"],
                "dentinogenesis_imperfecta_pct": entry["dentinogenesis_imperfecta_pct"],
                "hyperplastic_callus_pct": entry["hyperplastic_callus_pct"],
                "contractures_at_birth_pct": entry["contractures_at_birth_pct"],
                "nbs_indicated": entry["nbs_indicated"],
            }
            for entry in OI_GENES
        },
        "oi_glossary": {
            "Collagen Triple Helix & Glycine Substitutions": (
                "Type I collagen is a heterotrimer [alpha1(I)]2[alpha2(I)]1 requiring Gly-X-Y repeats "
                "where every third residue MUST be glycine (smallest amino acid to fit helix centre). "
                "NULL alleles (frameshift/nonsense) → haploinsufficiency → OI type I (mild, normal structure). "
                "GLYCINE SUBSTITUTIONS → dominant-negative: bulky side-chain disrupts helix folding → "
                "overmodification → defective fibrils → OI types II-IV (severe). "
                "C-terminal glycine substitutions: milder (type IV). N-terminal: severe (type II/lethal)."
            ),
            "Prolyl 3-Hydroxylation Complex (CRTAP-LEPRE1-PPIB)": (
                "The ternary complex (CRTAP scaffold + P3H1/LEPRE1 enzyme + CypB/PPIB isomerase) "
                "hydroxylates a SINGLE prolyl residue (Pro986) in pro-alpha1(I) collagen. "
                "Loss of ANY component → Pro986 under-hydroxylated → slow fold → massive overmodification → "
                "IDENTICAL SDS-PAGE band-shift for all three (CRTAP, LEPRE1, PPIB OI). "
                "Distinguishing: RHIZOMELIA = CRTAP type VII; WEST AFRICAN FOUNDER = LEPRE1 type VIII; "
                "P3H1 ENZYME NORMAL = PPIB type IX."
            ),
            "OI Type V — IFITM5 Hyperplastic Callus & IOM Calcification": (
                "IFITM5 c.-14C>T is the single recurrent mutation in >95% OI type V — creates MALEP "
                "5-aa N-terminal extension on BRIL → dominant-negative osteoblast effect. "
                "HYPERPLASTIC CALLUS: exuberant fracture callus (biologically, not sarcoma) — "
                "may look alarming on X-ray/MRI; biopsy shows benign reactive bone. "
                "IOM CALCIFICATION: radius-ulna and tibia-fibula interosseous membranes calcify → "
                "restricted forearm/ankle rotation (X-ray diagnostic). "
                "COLLAGEN NORMAL — type V systematically MISSED if only collagen biochemistry ordered."
            ),
            "OI Type VI — SERPINF1/PEDF and Fish-Scale Lamellae": (
                "PEDF (SERPINF1) is a serpin-family secreted protein with no protease-inhibitor activity — "
                "regulates bone mineralisation and RANKL/OPG axis. "
                "Biallelic LOF → absent serum PEDF (undetectable ELISA) — simple diagnostic test. "
                "Bone biopsy shows FISH-SCALE LAMELLAE on polarised light — pathognomonic for type VI. "
                "Excess osteoid unmineralised → progressive deformity. "
                "DENOSUMAB (anti-RANKL) > bisphosphonates for type VI — mechanism-targeted."
            ),
            "Bruck Syndrome 1 (FKBP10) — OI + Congenital Contractures": (
                "FKBP65 (FKBP10) chaperones type I collagen in ER and supports LH2/PLOD2 "
                "telopeptide lysine hydroxylation. "
                "FKBP10 null → secondary LH2 activity drop → defective pyridinoline cross-links → "
                "OI (bone fragility) + CONTRACTURES (tendon/ligament cross-link failure) + PTERYGIUM. "
                "Bruck syndrome 1 = OI + arthrogryposis-like contractures + pterygium: PATHOGNOMONIC. "
                "Turkish founder c.831+1G>T: targeted testing in Turkish OI+contracture families. "
                "Normal collagen SDS-PAGE SIZE (no overmodification — cross-link defect not folding defect)."
            ),
            "Bisphosphonate Therapy in OI": (
                "Bisphosphonates (BPs) inhibit osteoclast function (farnesyl-pyrophosphate synthase). "
                "PAMIDRONATE IV (3-day cycles q3 months): established regimen — increases BMD, reduces fractures 30-50%. "
                "ZOLEDRONATE IV once yearly: equivalent efficacy, fewer infusions. "
                "ALENDRONATE PO: for mild OI type I in older children. "
                "START EARLY: infancy in moderate-severe types — maximal benefit during growth. "
                "DO NOT USE in OI type VI SERPINF1 as monotherapy: denosumab superior. "
                "Osteonecrosis of jaw (rare in paediatrics), rebound fractures with denosumab cessation."
            ),
            "Non-Accidental Injury (NAI) Mimicry": (
                "OI is the most important metabolic bone disease to exclude before reporting child abuse. "
                "OI type I can present with multiple fractures of varying age in a normal-appearing child. "
                "MANDATORY SCREEN before safeguarding: serum Ca/PO4/ALP/25-OH-D; urine Ca/Cr; "
                "COL1A1/COL1A2 molecular testing (especially in type I); collagen biochemistry. "
                "Blue sclerae, dentinogenesis imperfecta, family history: support OI. "
                "Absence of bruising, retinal haemorrhage, history of force: raises OI probability. "
                "GENETICS + CLINICAL MULTIDISCIPLINARY ASSESSMENT MANDATORY before NAI diagnosis."
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
    print("\n=== DEFINITIONS (COL1A1) ===")
    defn = generate_definitions()
    print(json.dumps(defn["gene_entries"]["COL1A1"], indent=2)[:1500])
