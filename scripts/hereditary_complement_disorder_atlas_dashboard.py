#!/usr/bin/env python3
"""Hereditary-Complement-Disorder-Atlas — Complete 8-Gene Complement Deficiency Atlas
(C1QB · C4A · C2 · C3 · C5 · CFH · SERPING1 · C9).

C1QB     (C1q B-chain; 226 aa; ~18 kDa subunit / trimeric C1q ~460 kDa; 1p36.12; AR;
          C1q Deficiency;
          MOST COMMON early classical-pathway complement deficiency worldwide;
          90% develop SLE-LIKE disease — malar rash, photosensitivity, arthritis, nephritis;
          HIGH NEGATIVE PREDICTIVE VALUE: if C1q intact, SLE less likely;
          seed SEED_BASE+0).
C4A      (C4A complement protein; 1744 aa; ~200 kDa; 6p21.3; AR biallelic null;
          C4A Deficiency — homozygous null allele C4A*Q0;
          SLE risk 4-fold increased; MLPA/gene-dosage MANDATORY to detect copy-number variation;
          Nephritis-predominant SLE phenotype; standard WES misses C4A CNV;
          seed SEED_BASE+1).
C2       (C2 complement protein; 752 aa; ~102 kDa; 6p21.3; AR;
          C2 Deficiency — MOST COMMON complement deficiency in Caucasian populations ~1:10,000;
          SLE-like disease 40% + RECURRENT ENCAPSULATED BACTERIAL INFECTIONS;
          HLA-DR15-DQ6 founder haplotype; CH50 absent; pneumococcal vaccine MANDATORY;
          seed SEED_BASE+2).
C3       (C3 complement protein; 1663 aa; ~185 kDa; 19p13.3; AR;
          C3 Deficiency — MOST SEVERE primary complement deficiency;
          ALL THREE PATHWAYS impaired; RECURRENT ENCAPSULATED BACTERIA;
          CH50 AND AP50 both absent; MPGN/C3NeF association;
          seed SEED_BASE+3).
C5       (C5 complement protein; 1676 aa; ~190 kDa; 9q33.2; AR;
          C5 Deficiency — Terminal pathway initiation deficiency;
          NARROW SPECTRUM: ONLY Neisseria meningitidis + N.gonorrhoeae PATHOGNOMONIC;
          Meningococcal vaccine ACWY+B MANDATORY; ECULIZUMAB PARADOXICALLY INCREASES Neisseria risk;
          seed SEED_BASE+4).
CFH      (Factor H; 1231 aa; ~155 kDa; 1q31.3; AD/AR;
          Factor H Deficiency — THREE DISTINCT PHENOTYPES by variant type:
          aHUS (AD heterozygous), AMD (common Y402H variant), MPGN/C3GN (AR complete loss);
          Eculizumab DRAMATICALLY EFFECTIVE for aHUS — reverses TMA;
          seed SEED_BASE+5).
SERPING1 (C1-Inhibitor/C1-INH; 500 aa; ~105 kDa; 11q12.1; AD;
          Hereditary Angioedema Type I and II — MOST CLINICALLY RECOGNISED hereditary complement disorder;
          BRADYKININ-MEDIATED (NOT histamine); LARYNGEAL ANGIOEDEMA FATAL;
          ACE INHIBITORS ABSOLUTELY CONTRAINDICATED;
          seed SEED_BASE+6).
C9       (C9 complement protein; 559 aa; ~63 kDa; 5p13.1; AR;
          C9 Deficiency — MOST COMMON complement deficiency in East Asian populations;
          Japan frequency 1:1,000; Japanese founder p.Arg95His 85% of Japanese cases;
          SELECTIVE NEISSERIA SUSCEPTIBILITY; meningococcal vaccine MANDATORY;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2350-2357).
"""

import random

SEED_BASE = 2350

COMPLEMENT_GENES = [
    # -- C1QB -- C1q Deficiency -------------------------------------------------------
    {
        "gene": "C1QB",
        "alt_name": (
            "C1QB (C1QB-226aa-1p36.12 / AR -- C1q-Deficiency -- "
            "MOST-COMMON-EARLY-COMPLEMENT-DEFICIENCY-WORLDWIDE -- "
            "90pct-SLE-LIKE-DISEASE-Malar-Rash-Photosensitivity-Arthritis-Nephritis-PATHOGNOMONIC -- "
            "HIGH-NEGATIVE-PREDICTIVE-VALUE-FOR-SLE-C1q-Intact-SLE-Less-Likely -- "
            "Classical-Pathway-Initiation-Failure-Apoptotic-Debris-Clearance-Defect)"
        ),
        "protein": (
            "C1QB -- 1p36.12 AR -- C1QB-226aa -- "
            "C1q-B-Chain-18kDa-Subunit-Trimeric-C1q-460kDa-Collagen-Stalk-Globular-Head -- "
            "C1q-Hexamer-Six-Globular-Heads-Bind-IgG-IgM-Fc-Regions-On-Immune-Complexes -- "
            "C1q-Also-Binds-Apoptotic-Cells-CRP-Pentraxins-Complement-Immune-Complex-Clearance -- "
            "C1QB-Deficiency-Entire-C1q-Absent-Heterotrimeric-Obligate-Assembly-Fails -- "
            "Failure-Apoptotic-Cell-Clearance-Nuclear-Antigen-Exposure-Anti-dsDNA-SLE -- "
            "SLE-90pct-Penetrance-Most-Severe-Complement-SLE-Link-Malar-Rash-Nephritis -- "
            "C1QA-C1QC-Also-Cause-C1q-Deficiency-Check-All-Three-Genes-If-C1q-Absent -- "
            "OMIM-Gene-120570-Disease-C1q-Deficiency-616430"
        ),
        "locus": "1p36.12",
        "protein_size": "226 aa / 18 kDa subunit (trimeric C1q ~460 kDa)",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "C1q is a heterotrimer of C1QA/C1QB/C1QC — deficiency in any one chain abolishes C1q; "
            "very rare: <100 cases reported worldwide; no ethnic enrichment reported; "
            "related: C1QA deficiency (186791), C1QC deficiency (120575) — same phenotype"
        ),
        "disease_category": "C1q Deficiency — Classical pathway complement deficiency with SLE-like disease",
        "disease_pathway": (
            "C1q is the recognition molecule of the classical complement pathway. "
            "C1q binds immune complexes (IgG/IgM-antigen) and apoptotic cells -> activates C1r/C1s -> "
            "cleaves C4 and C2 -> C3 convertase (C4b2a) -> C3 opsonisation and MAC formation. "
            "C1q ALSO binds CRP/pentraxins on apoptotic cells to mediate efferocytosis (apoptotic clearance). "
            "C1QB deficiency -> no C1q -> failure to opsonise/clear apoptotic cells -> "
            "nuclear antigens (dsDNA, histones, Sm) accumulate extracellularly -> "
            "anti-dsDNA/ANA antibody formation -> immune complex deposition -> SLE phenotype. "
            "SLE-like disease in 90% of C1q-deficient patients is the most penetrant complement-deficiency phenotype."
        ),
        "pathognomonic": (
            "SLE-LIKE DISEASE IN 90% OF C1QB-DEFICIENT PATIENTS — the most penetrant immunological phenotype of any complement deficiency. "
            "Features: malar rash, photosensitivity, arthritis, nephritis, oral ulcers — ANA and anti-dsDNA positive. "
            "HIGH NEGATIVE PREDICTIVE VALUE: if C1q level is intact and functional, SLE is less likely. "
            "Complement screening: CH50 very low (0-5% of normal); C1q level absent; C1q function absent. "
            "Encapsulated bacterial infections: LESS prominent than later-pathway deficiencies (C3 still functional for alternative/lectin). "
            "KEY diagnostic: very low CH50 + absent C1q protein level + SLE phenotype = C1QB/A/C sequencing."
        ),
        "treatment": (
            "No specific complement replacement therapy available in routine practice. "
            "SLE MANAGEMENT: hydroxychloroquine (HCQ) + corticosteroids + mycophenolate mofetil (MMF) standard. "
            "Nephritis: MMF + corticosteroids; consider belimumab (anti-BLYS) or voclosporin for refractory LN. "
            "Plasma infusion: C1q replacement attempted in acute episodes — experimental, not standard of care. "
            "COMPLEMENT REPLACEMENT (experimental): fresh frozen plasma contains C1q; may reduce SLE activity temporarily. "
            "Encapsulated bacterial prophylaxis: pneumococcal + H.influenzae B + meningococcal vaccines — less critical than C3/C5 deficiency but recommended. "
            "Annual urine albumin/creatinine (SLE nephritis screen); anti-dsDNA and C3/C4 monitoring."
        ),
        "key_features": [
            "Most common early complement deficiency worldwide",
            "SLE-like disease 90% penetrance (highest complement-SLE association)",
            "High negative predictive value: intact C1q makes SLE less likely",
            "CH50 very low (0-5%); C1q level absent",
            "Classical pathway only impaired (alternative/lectin intact — C3 still works)",
            "C1QA, C1QB, C1QC — any chain deficiency abolishes C1q",
        ],
        "key_ddx": (
            "Idiopathic SLE: C1q level normal initially; C1q consumption can occur in active SLE — distinguish by C1q GENE sequencing. "
            "C4A deficiency: SLE-like; C1q NORMAL; C4 level low; MLPA shows C4A null alleles. "
            "Acquired C1-INH deficiency / C1q deficiency secondary to lymphoma: C1q low but GENE intact; anti-C1q antibodies present; older age onset. "
            "C2 deficiency: SLE-like + encapsulated bacterial infections; CH50 absent; C1q NORMAL."
        ),
        "complement_pathway_affected": "Classical pathway initiation",
        "neisseria_risk": "Low — MAC pathway intact (alternative/lectin -> C3 -> C5-C9 still functional)",
        "sle_risk": "VERY HIGH — 90% of C1QB-deficient patients develop SLE-like disease",
        "ahus_risk": "None",
        "hae_risk": "None",
        "eculizumab_relevant": "No",
        "onset_age": "Childhood; SLE features emerge in first decade; recurrent infections less prominent",
        "key_vaccine": "Pneumococcal + H.influenzae B + Meningococcal (supportive; classical pathway impaired but alternative intact)",
    },
    # -- C4A -- C4A Deficiency -------------------------------------------------------
    {
        "gene": "C4A",
        "alt_name": (
            "C4A (C4A-1744aa-6p21.3 / AR-Biallelic-Null -- C4A-Deficiency -- "
            "SLE-RISK-4-FOLD-INCREASED-Biallelic-C4A-Null-Allele-C4AQ0 -- "
            "MLPA-GENE-DOSAGE-MANDATORY-CNV-Detection-Standard-WES-Misses -- "
            "NEPHRITIS-PREDOMINANT-SLE-Phenotype-C4-Level-Low-C1q-NORMAL -- "
            "Null-Allele-Prevalence-15-20pct-Caucasians-Heterozygous)"
        ),
        "protein": (
            "C4A -- 6p21.3 AR-Biallelic-Null -- C4A-1744aa -- "
            "C4A-Complement-Component-4A-200kDa-MHC-Class-III-Region-6p21.3 -- "
            "C4A-vs-C4B-Differ-At-4-Residues-C4A-Preferentially-Binds-Amino-Groups-Protein-Targets -- "
            "C4B-Preferentially-Binds-Hydroxyl-Groups-Carbohydrate-Targets -- "
            "C4A-Therefore-Better-Opsonises-Protein-Coated-Targets-Immune-Complexes -- "
            "C4A-Null-Allele-C4AQ0-Copy-Number-Variation-Not-Point-Mutation -- "
            "MLPA-Array-CGH-Required-CNV-Detection-Standard-WES-Cannot-Detect-CNV -- "
            "Two-Gene-Loci-C4A-C4B-Both-On-6p21.3-Variable-Copy-Number-1-6-Total -- "
            "OMIM-Gene-120810-Disease-C4A-Deficiency-614200"
        ),
        "locus": "6p21.3",
        "protein_size": "1744 aa / ~200 kDa",
        "inheritance": (
            "AR biallelic null (C4A*Q0 homozygous); copy number variation (CNV) — NOT a point mutation; "
            "heterozygous C4A null (C4AQ0/normal): present in 15-20% Caucasians — SLE risk modestly increased; "
            "biallelic C4A null (homozygous C4AQ0): 1-3% Caucasians — 4-fold SLE risk increase; "
            "MLPA or array CGH required — standard WES misses CNV"
        ),
        "disease_category": "C4A Deficiency — SLE-predisposing complement deficiency (classical pathway)",
        "disease_pathway": (
            "C4 is cleaved by C1s (classical pathway) or MASP-2 (lectin pathway) to C4a + C4b. "
            "C4b covalently binds target surfaces; C4b2a = classical C3 convertase. "
            "C4A (vs C4B isoform) preferentially binds protein-coated targets (immune complexes with IgG). "
            "C4A null -> reduced opsonisation of protein-rich immune complexes -> impaired clearance -> "
            "nuclear antigens persist -> anti-dsDNA/ANA antibodies -> SLE, especially nephritis. "
            "C4A null is the most common genetic predisposition to SLE in Caucasian populations. "
            "C4B isoform typically present (carbohydrate opsonisation partially preserved)."
        ),
        "pathognomonic": (
            "SLE PHENOTYPE — especially nephritis — with LOW C4 LEVEL and NORMAL C1q. "
            "C4 level low (both isotypes reduced if biallelic null); C1q NORMAL (distinguishes from C1QB deficiency). "
            "CH50: modestly reduced (40-60% of normal) — C4A null impairs but does not abolish classical pathway. "
            "MLPA/gene dosage: homozygous C4A null alleles PATHOGNOMONIC for genetic predisposition. "
            "KEY CLINICAL TRAP: Standard WES reports C4A wildtype — must request MLPA or gene dosage assay separately. "
            "Family testing: MLPA in first-degree relatives; heterozygous null = modestly elevated SLE risk."
        ),
        "treatment": (
            "SLE MANAGEMENT (same principles as idiopathic SLE): HCQ, corticosteroids, MMF, azathioprine. "
            "Lupus nephritis: MMF + corticosteroids; belimumab, voclosporin for refractory class III/IV. "
            "MLPA testing for family members — identify homozygous null relatives and counsel for SLE risk. "
            "Nephrotoxin avoidance: NSAIDs, aminoglycosides, contrast dye — renal monitoring essential. "
            "C4 level monitoring: tracks SLE disease activity (C4 consumed in active SLE — already low at baseline). "
            "Experimental: complement replacement not standard; FFP occasionally trialled in severe disease."
        ),
        "key_features": [
            "C4A null (C4AQ0) is the most common complement-gene predisposition to SLE in Caucasians",
            "SLE risk 4-fold elevated in biallelic C4A null",
            "C4 level low, C1q NORMAL — key distinguishing point from C1QB deficiency",
            "CH50 modestly reduced (40-60%), not zero",
            "CNV — MLPA or array CGH mandatory; standard WES MISSES IT",
            "Heterozygous null (15-20% Caucasians) — modestly elevated SLE risk",
        ],
        "key_ddx": (
            "C1QB deficiency: C1q level ABSENT; CH50 very low (0-5%); same SLE phenotype — different gene. "
            "Idiopathic SLE: normal C4A gene dosage; C4 low only during flares (consumption). "
            "C2 deficiency: CH50 ZERO; C2 absent; encapsulated bacterial infections also prominent. "
            "Acquired C4 consumption in SLE: C4A gene dosage normal; C4 normalises between flares."
        ),
        "complement_pathway_affected": "Classical and lectin pathway C3 convertase formation",
        "neisseria_risk": "Low — MAC pathway functional (C5-C9 intact)",
        "sle_risk": "HIGH — 4-fold increased with biallelic null",
        "ahus_risk": "None",
        "hae_risk": "None",
        "eculizumab_relevant": "No",
        "onset_age": "Childhood to young adulthood; SLE features predominate; nephritis often early complication",
        "key_vaccine": "Pneumococcal + H.influenzae B + Meningococcal (less urgent than C3/C5 deficiency)",
    },
    # -- C2 -- C2 Deficiency -------------------------------------------------------
    {
        "gene": "C2",
        "alt_name": (
            "C2 (C2-752aa-6p21.3 / AR -- C2-Deficiency -- "
            "MOST-COMMON-COMPLEMENT-DEFICIENCY-CAUCASIANS-1:10000 -- "
            "SLE-LIKE-40pct-PLUS-RECURRENT-PNEUMOCOCCAL-H.INFLUENZAE-INFECTIONS-COMBINED-PHENOTYPE -- "
            "HLA-DR15-DQ6-FOUNDER-HAPLOTYPE-Caucasian -- "
            "CH50-ABSENT-ZERO-Classical-Pathway-Fails-Pneumococcal-Vaccine-MANDATORY)"
        ),
        "protein": (
            "C2 -- 6p21.3 AR -- C2-752aa -- "
            "C2-Complement-Component-2-102kDa-Serine-Protease-Pro-Enzyme -- "
            "C2-Cleaved-By-C1s-Classical-Or-MASP-2-Lectin-Pathway -- "
            "C2b-Fragment-Binds-C4b-Forms-Classical-C3-Convertase-C4b2a -- "
            "C2-Absence-No-Classical-C3-Convertase-Formation-Lectin-Pathway-Also-C2-Dependent -- "
            "HLA-DR15-DQ6-Founder-Haplotype-28bp-Deletion-Exon-6-Most-Common-Caucasian-Variant -- "
            "Alternative-Pathway-INTACT-C3-Functional-Via-Factor-B-D-Properdin -- "
            "SLE-40pct-Encapsulated-Bacterial-Infections-Combined-Phenotype-Unique -- "
            "OMIM-Gene-Disease-Same-217000"
        ),
        "locus": "6p21.3",
        "protein_size": "752 aa / ~102 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; "
            "28-bp deletion exon 6 on HLA-DR15-DQ6 founder haplotype — accounts for most Caucasian C2 deficiency; "
            "frequency ~1:10,000 Caucasians (heterozygous carriers ~1:50); "
            "most common complement deficiency in Caucasian populations (not globally)"
        ),
        "disease_category": "C2 Deficiency — Classical pathway deficiency with dual phenotype (SLE + bacterial infections)",
        "disease_pathway": (
            "C2 is cleaved by C1s (classical) or MASP-2 (lectin) -> C2b binds C4b -> "
            "C4b2a = classical C3 convertase (opsonises targets via C3b; generates C3a anaphylatoxin). "
            "C2 deficiency -> no classical or lectin-driven C3 convertase -> "
            "TWO consequences: (1) Impaired immune complex/apoptotic debris clearance -> SLE-like disease (40%); "
            "(2) Impaired opsonisation of encapsulated bacteria (S.pneumoniae, H.influenzae, N.meningitidis) -> recurrent infections. "
            "Alternative pathway C3 activation INTACT (Factor B-dependent; C2 not involved) -> "
            "partial protection (encapsulated bacteria still resist alternative opsonisation better than gram-negatives)."
        ),
        "pathognomonic": (
            "SLE-LIKE DISEASE (40%) + RECURRENT ENCAPSULATED BACTERIAL INFECTIONS — combined dual phenotype PATHOGNOMONIC for C2 deficiency. "
            "CH50: ZERO (classical pathway abolished; screening test CH50 = 0). "
            "C2 level: UNDETECTABLE (protein level measurement confirms). "
            "C1q and C4 levels: NORMAL (pathway interrupted at C2 not at C1/C4). "
            "AP50 (alternative pathway): NORMAL (alternative pathway intact). "
            "HLA-DR15-DQ6 haplotype: present in majority of Caucasian patients — genetic context clue. "
            "Pneumococcal/H.influenzae B vaccination mandatory from diagnosis; penicillin prophylaxis if recurrent."
        ),
        "treatment": (
            "PNEUMOCOCCAL VACCINE (PCV13 + PPSV23): MANDATORY — pneumococcal disease is leading cause of morbidity. "
            "H.INFLUENZAE TYPE B (HiB) VACCINE: mandatory. "
            "MENINGOCOCCAL VACCINE ACWY + B: recommended (Neisseria susceptibility via classical pathway). "
            "Penicillin V prophylaxis (250-500 mg BD): for patients with recurrent encapsulated bacterial infections. "
            "SLE management: HCQ, corticosteroids, MMF — standard SLE protocols. "
            "Fresh frozen plasma: used as adjunct in severe acute infections — temporary C2 replacement. "
            "Annual complement screening (C2 level confirms diagnosis; CH50 confirms pathway absence)."
        ),
        "key_features": [
            "Most common complement deficiency in Caucasian populations (~1:10,000)",
            "Dual phenotype: SLE-like (40%) + recurrent encapsulated bacterial infections",
            "CH50 ZERO; AP50 normal; C1q and C4 NORMAL (C2 is the block)",
            "HLA-DR15-DQ6 founder haplotype (28-bp deletion exon 6)",
            "Alternative pathway intact — partial protection but insufficient for encapsulated bacteria",
            "Pneumococcal vaccine MANDATORY from diagnosis",
        ],
        "key_ddx": (
            "C1QB deficiency: CH50 very low; C1q absent; SLE dominant (90%); less bacterial infections. "
            "C3 deficiency: CH50 AND AP50 both absent; severe infections (gram-negative too); more severe phenotype. "
            "Acquired C2 consumption in SLE: C2 level low in active SLE — gene sequencing resolves. "
            "C4A deficiency: CH50 modestly reduced; C4 low; C2 normal; fewer bacterial infections."
        ),
        "complement_pathway_affected": "Classical and lectin pathway (C3 convertase formation); alternative pathway INTACT",
        "neisseria_risk": "Moderate — classical pathway impaired; alternative pathway partially compensates",
        "sle_risk": "HIGH — 40% of C2-deficient patients develop SLE-like disease",
        "ahus_risk": "None",
        "hae_risk": "None",
        "eculizumab_relevant": "No",
        "onset_age": "Childhood; infections in early childhood; SLE features in teens/young adults",
        "key_vaccine": "Pneumococcal (PCV13+PPSV23) + H.influenzae B + Meningococcal ACWY+B — ALL MANDATORY",
    },
    # -- C3 -- C3 Deficiency -------------------------------------------------------
    {
        "gene": "C3",
        "alt_name": (
            "C3 (C3-1663aa-19p13.3 / AR -- C3-Deficiency -- "
            "MOST-SEVERE-PRIMARY-COMPLEMENT-DEFICIENCY-ALL-THREE-PATHWAYS-IMPAIRED -- "
            "RECURRENT-ENCAPSULATED-BACTERIA-ALL-SPECIES-S.pneumoniae+H.influenzae+N.meningitidis -- "
            "CH50-AND-AP50-BOTH-ABSENT-ZERO-Unique-Pattern -- "
            "MPGN-C3-NEPHRITIC-FACTOR-Association-Membranoproliferative-Glomerulonephritis)"
        ),
        "protein": (
            "C3 -- 19p13.3 AR -- C3-1663aa -- "
            "C3-Complement-Component-3-185kDa-Central-Hub-All-Three-Complement-Pathways -- "
            "Alpha-Chain-110kDa-Beta-Chain-75kDa-Disulfide-Linked-Heterodimer -- "
            "Thioester-Bond-Alpha-Chain-C3b-Covalently-Binds-Surface-After-Cleavage -- "
            "C3b-Opsonin-CR1-Receptor-Phagocytosis-C3a-Anaphylatoxin-Mast-Cell-Degranulation -- "
            "C3-Convertase-From-ALL-Pathways-Classical-C4b2a-Alternative-C3bBb-Lectin-Same-As-Classical -- "
            "No-C3-No-C3b-No-Opsonisation-No-MAC-No-C5a-Chemotaxis-Total-Effector-Failure -- "
            "C3NeF-C3-Nephritic-Factor-Autoantibody-Stabilises-C3bBb-Secondary-C3-Consumption -- "
            "OMIM-Gene-120700-Disease-C3-Deficiency-613779"
        ),
        "locus": "19p13.3",
        "protein_size": "1663 aa / ~185 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; rare; "
            "any ethnicity — no founder enrichment reported; "
            "heterozygous carriers: C3 level ~50% — typically sufficient for complement function; "
            "C3 nephritic factor (C3NeF) is an ACQUIRED autoantibody that causes SECONDARY C3 depletion — "
            "not a primary C3 gene deficiency but phenotypically overlapping (check gene if C3NeF absent)"
        ),
        "disease_category": "C3 Deficiency — Most severe primary complement deficiency (all pathways impaired)",
        "disease_pathway": (
            "C3 is the central convergence point of all three complement pathways (classical, alternative, lectin). "
            "C3 cleavage by ANY C3 convertase generates: "
            "(1) C3b — covalently opsonises surfaces -> recognised by CR1/CR3 -> phagocytosis of bacteria; "
            "(2) C3a — anaphylatoxin -> mast cell degranulation -> inflammatory response; "
            "(3) C3b also feeds C5 convertases -> C5a generation (chemotaxis) + C5b-C9 MAC (bacteriolysis). "
            "C3 deficiency -> TOTAL LOSS of: opsonisation (ALL pathways), MAC formation, C3a/C5a generation. "
            "Result: profound susceptibility to ALL encapsulated bacteria; no discrimination between species. "
            "Additional: impaired immune complex clearance -> SLE-like features in ~10%; "
            "MPGN from immune complex deposition secondary to C3 consumption."
        ),
        "pathognomonic": (
            "RECURRENT INFECTIONS BY ALL ENCAPSULATED BACTERIA (S.pneumoniae, H.influenzae, N.meningitidis) PATHOGNOMONIC — no discrimination. "
            "Unlike late complement deficiency (C5-C9): C3 deficiency includes S.pneumoniae (which requires opsonisation, not MAC). "
            "CH50 ABSENT (ZERO) + AP50 ABSENT (ZERO) — unique pattern for C3 deficiency; C3 level differentiates from late complement. "
            "C3 level: UNDETECTABLE. C1q, C4, C2, Factor B all normal (pathway components intact, but no substrate). "
            "C3NeF (C3 nephritic factor) ABSENT in primary deficiency — if C3NeF positive, secondary C3 consumption (MPGN). "
            "MPGN type 2 (C3 glomerulopathy): immune complex deposition -> proteinuria + haematuria."
        ),
        "treatment": (
            "ENCAPSULATED BACTERIAL VACCINES — ALL MANDATORY: "
            "(1) Pneumococcal conjugate (PCV13) + polysaccharide (PPSV23); "
            "(2) H.influenzae type B (HiB); "
            "(3) Meningococcal ACWY + B (Bexsero/Trumenba). "
            "Antibiotic prophylaxis (penicillin V): strongly recommended — encapsulated bacterial risk is severe. "
            "PLASMA INFUSIONS: fresh frozen plasma as bridge therapy in acute severe infections — temporary C3 replacement. "
            "SLE features (~10%): HCQ + standard SLE immunosuppression. "
            "MPGN: ACE inhibitors for proteinuria; monitor eGFR; renal biopsy confirms C3 glomerulopathy pattern. "
            "Family screening: biallelic required; heterozygous carriers at 50% C3 — monitor."
        ),
        "key_features": [
            "Most severe primary complement deficiency — ALL THREE pathways impaired",
            "Recurrent infections: ALL encapsulated bacteria (S.pneumoniae, H.influenzae, N.meningitidis)",
            "CH50 ZERO + AP50 ZERO — C3 level (absent) resolves vs late complement deficiency (C3 normal)",
            "C3 level undetectable; C1q/C4/C2/Factor B all normal",
            "MPGN/C3 glomerulopathy association",
            "SLE-like features ~10% (impaired immune complex clearance)",
        ],
        "key_ddx": (
            "Late complement deficiency (C5-C9): ONLY Neisseria susceptibility (not S.pneumoniae); CH50 + AP50 absent; C3 NORMAL — key distinguishing point. "
            "Factor H/I deficiency (secondary C3 consumption): C3 LOW but C3 gene intact; C3NeF may be present; check CFH/CFI genes. "
            "C2 deficiency: AP50 NORMAL; C3 normal; less severe infection phenotype; SLE 40%. "
            "Acquired C3 depletion in SLE: C3 low but gene normal; ANA/dsDNA positive; C3 recovers with treatment."
        ),
        "complement_pathway_affected": "ALL pathways (classical + alternative + lectin) — C3 is the convergence point",
        "neisseria_risk": "HIGH — all MAC precursors absent (C3b -> C5 convertase required for C5-C9 MAC)",
        "sle_risk": "Moderate — 10% SLE-like disease",
        "ahus_risk": "Low — secondary MPGN/C3GN more typical",
        "hae_risk": "None",
        "eculizumab_relevant": "No — C5 downstream of C3 block; eculizumab not applicable",
        "onset_age": "Childhood; severe infections in first years of life; MPGN may present in second decade",
        "key_vaccine": "Pneumococcal + H.influenzae B + Meningococcal ACWY+B — ALL MANDATORY; antibiotic prophylaxis also",
    },
    # -- C5 -- C5 Deficiency -------------------------------------------------------
    {
        "gene": "C5",
        "alt_name": (
            "C5 (C5-1676aa-9q33.2 / AR -- C5-Deficiency -- "
            "TERMINAL-COMPLEMENT-PATHWAY-INITIATION-DEFICIENCY -- "
            "NARROW-SPECTRUM-ONLY-Neisseria-meningitidis+N.gonorrhoeae-PATHOGNOMONIC -- "
            "UNCOMMON-SEROGROUPS-W135-Y-X-N.meningitidis-Characteristic -- "
            "ECULIZUMAB-PARADOXICALLY-INCREASES-Neisseria-Risk-Vaccine-Prophylaxis-MANDATORY)"
        ),
        "protein": (
            "C5 -- 9q33.2 AR -- C5-1676aa -- "
            "C5-Complement-Component-5-190kDa-Terminal-Pathway-Initiator -- "
            "Alpha-Chain-115kDa-Beta-Chain-75kDa-Disulfide-Linked-Heterodimer -- "
            "C5-Cleaved-By-C5-Convertase-C4b2a3b-Classical-Or-C3bBbC3b-Alternative -- "
            "C5b-Initiates-MAC-C5b-6-7-8-9-Assembly-Transmembrane-Pore-Formation -- "
            "C5a-Potent-Anaphylatoxin-Chemotaxis-Factor-Neutrophil-Recruitment-Inflammation -- "
            "C5-Deficiency-No-MAC-No-C5a-Direct-Gram-Negative-Lysis-Abolished -- "
            "ECULIZUMAB-Anti-C5-Blocks-C5-Cleavage-Mimics-C5-Deficiency-For-Neisseria -- "
            "OMIM-Gene-120900-Disease-C5-Deficiency-609536"
        ),
        "locus": "9q33.2",
        "protein_size": "1676 aa / ~190 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "any ethnicity; no major founder enrichment; "
            "eculizumab pharmacologically mimics C5 deficiency — eculizumab-treated patients acquire same Neisseria risk"
        ),
        "disease_category": "C5 Deficiency — Terminal complement pathway deficiency (narrow spectrum: Neisseria only)",
        "disease_pathway": (
            "C5 convertases (classical C4b2a3b OR alternative C3bBb3b) cleave C5 into: "
            "(1) C5b — initiates MAC assembly (C5b + C6 + C7 + C8 + polymerised C9) -> transmembrane pore -> osmotic lysis of gram-negative bacteria; "
            "(2) C5a — potent anaphylatoxin; chemotaxis for neutrophils; mast cell degranulation. "
            "C5 deficiency -> NO MAC formation -> NO direct bacteriolysis of gram-negative bacteria. "
            "Opsonisation (C3b) INTACT -> encapsulated bacteria still partially cleared. "
            "NARROW SPECTRUM: N.meningitidis and N.gonorrhoeae possess sialylated LPS that resists opsonin-mediated killing -> MAC (direct lysis) is primary defence. "
            "S.pneumoniae/H.influenzae: cleared via opsonisation (C3b) despite absent MAC — so C5 deficiency patients NOT susceptible to these."
        ),
        "pathognomonic": (
            "RECURRENT NEISSERIA MENINGITIDIS (including UNCOMMON SEROGROUPS W135/Y/X) — NARROW SPECTRUM PATHOGNOMONIC. "
            "Disseminated Gonococcal Infection (DGI) also characteristic — N.gonorrhoeae requires MAC for clearance. "
            "NO susceptibility to S.pneumoniae or H.influenzae — distinguishes C5 deficiency from C3 deficiency. "
            "CH50: ABSENT (ZERO). AP50: ABSENT (ZERO). C3 level: NORMAL. C5 level: ABSENT. "
            "ECULIZUMAB ALERT: anti-C5 biologics (eculizumab, ravulizumab) BLOCK C5 -> same Neisseria risk as genetic C5 deficiency. "
            "MANDATORY before eculizumab: Meningococcal ACWY + B vaccination AND penicillin prophylaxis."
        ),
        "treatment": (
            "MENINGOCOCCAL VACCINE ACWY (MenACWY) + B-TYPE (Bexsero/Trumenba): MANDATORY — includes uncommon serogroups. "
            "Antibiotic prophylaxis: penicillin V (250-500 mg BD) — lifelong recommendation in high-risk periods. "
            "Family screening: identify other biallelic C5-deficient relatives. "
            "ECULIZUMAB-TREATED PATIENTS (not C5-deficient genetically but functionally equivalent): "
            "  -- MenACWY + MenB MINIMUM 2 WEEKS before first eculizumab dose; "
            "  -- Penicillin V prophylaxis for duration of eculizumab therapy; "
            "  -- Counsel patient on Neisseria symptoms (rash + fever = emergency). "
            "Meningococcal booster every 3-5 years (waning immunity). "
            "Gonococcal infection: standard antibiotic treatment; counsel for DGI risk."
        ),
        "key_features": [
            "Terminal complement pathway; only Neisseria meningitidis + gonorrhoeae — narrow spectrum PATHOGNOMONIC",
            "CH50 ZERO, AP50 ZERO, C3 NORMAL — key pattern",
            "Uncommon serogroups W135/Y/X meningitidis characteristic",
            "NO susceptibility to S.pneumoniae/H.influenzae (opsonisation via C3b intact)",
            "Eculizumab pharmacologically mimics C5 deficiency — vaccine mandatory before starting",
            "Disseminated gonococcal infection (DGI) characteristic",
        ],
        "key_ddx": (
            "C6/C7/C8/C9 deficiency: identical Neisseria phenotype; CH50 + AP50 absent; C5 NORMAL — check all terminal components. "
            "Properdin deficiency (CFP, X-linked): ONLY Neisseria; alternative pathway selectively impaired; AP50 absent, CH50 normal. "
            "C3 deficiency: ALL encapsulated bacteria (S.pneumoniae too); C3 absent; more severe phenotype. "
            "Eculizumab-treated (acquired functional C5 deficiency): drug history; C5 protein normal; eculizumab serum level detectable."
        ),
        "complement_pathway_affected": "Terminal pathway (MAC assembly); opsonisation intact",
        "neisseria_risk": "VERY HIGH — primary susceptibility; uncommon serogroups W135/Y/X characteristic",
        "sle_risk": "Low — ~5% SLE-like features",
        "ahus_risk": "None (eculizumab used for aHUS acts by blocking C5 — same gene product, therapeutic target)",
        "hae_risk": "None",
        "eculizumab_relevant": "YES — eculizumab is anti-C5; mimics C5 deficiency; VACCINE MANDATORY before prescribing",
        "onset_age": "Any age; young adults common for meningococcal presentation; family history may be absent (recessive)",
        "key_vaccine": "Meningococcal ACWY + B MANDATORY; boosters every 3-5 years; penicillin prophylaxis",
    },
    # -- CFH -- Factor H Deficiency -------------------------------------------------------
    {
        "gene": "CFH",
        "alt_name": (
            "CFH (CFH-1231aa-1q31.3 / AD-AR -- Factor-H-Deficiency -- "
            "THREE-DISTINCT-PHENOTYPES-Variant-Type-Determines-Disease -- "
            "aHUS-AD-Heterozygous-TMA-Eculizumab-DRAMATIC-RESPONSE-TRANSFORMS-OUTCOME -- "
            "AMD-Y402H-MOST-COMMON-GENETIC-RISK-FACTOR-Age-Related-Macular-Degeneration -- "
            "MPGN-C3GN-AR-Complete-Loss-Alternative-Pathway-Dysregulation)"
        ),
        "protein": (
            "CFH -- 1q31.3 AD-AR -- CFH-1231aa -- "
            "Factor-H-155kDa-Serum-Glycoprotein-20-Short-Consensus-Repeat-SCR-Domains -- "
            "ALTERNATIVE-PATHWAY-REGULATOR-Binds-C3b-On-Host-Cells-Not-Microbial-Surfaces -- "
            "SCR1-4-Cofactor-Factor-I-C3b-Inactivation-SCR7-20-Surface-Binding-Host-Recognition -- "
            "Factor-H-Distinguishes-Host-Endothelium-From-Pathogens-By-Surface-Polyanions-Heparan-Sulfate -- "
            "Deficiency-Loss-Surface-Discrimination-C3b-Deposits-On-Host-Endothelium-TMA -- "
            "Y402H-SCR7-Common-Variant-Reduces-Heparan-Sulfate-Binding-Retinal-RPE-AMD-Risk -- "
            "aHUS-AD-Incomplete-Penetrance-Heterozygous-Missense-SCR19-20-Endothelial-Binding-Domain -- "
            "OMIM-Gene-134370-Disease-aHUS-235400-AMD-603075"
        ),
        "locus": "1q31.3",
        "protein_size": "1231 aa / ~155 kDa",
        "inheritance": (
            "AD (aHUS phenotype — heterozygous missense, incomplete penetrance ~50%) OR "
            "AR (MPGN/C3GN — biallelic complete loss) OR "
            "Common variant (AMD — Y402H, p.Tyr402His, in SCR7; ~50% population-level risk modifier); "
            "penetrance varies by phenotype and co-existing risk factors (e.g. CFHR1/3 deletion for TMA risk)"
        ),
        "disease_category": "Factor H Deficiency — Three phenotypes: aHUS (AD/TMA), AMD (common variant), MPGN/C3GN (AR)",
        "disease_pathway": (
            "Factor H controls the alternative complement pathway on HOST cell surfaces (endothelium, blood cells). "
            "Mechanism: Factor H binds C3b on host surfaces (via heparan sulfate/sialic acid recognition in SCR7-20) -> "
            "acts as cofactor for Factor I -> C3b cleavage to iC3b (inactivation) + accelerates decay of C3bBb convertase. "
            "Without Factor H on host surfaces: C3b deposition on endothelium UNCHECKED -> "
            "C3bBb forms continuously on host endothelium -> C5 convertase -> sublytic MAC -> "
            "endothelial activation/damage -> THROMBOTIC MICROANGIOPATHY (TMA). "
            "aHUS (AD): one defective CFH allele -> insufficient Factor H on endothelium under stress -> TMA. "
            "AMD (Y402H): reduced binding to heparan sulfate in retinal pigment epithelium -> chronic inflammation -> drusen -> AMD. "
            "MPGN (AR): complete Factor H loss -> C3 consumption -> C3 deposition in glomeruli -> MPGN type 2."
        ),
        "pathognomonic": (
            "aHUS TRIAD: MICROANGIOPATHIC HAEMOLYTIC ANAEMIA (MAHA) + THROMBOCYTOPENIA + ACUTE KIDNEY INJURY — PATHOGNOMONIC for TMA. "
            "DISTINGUISHING aHUS from TTP: ADAMTS13 ACTIVITY NORMAL (>10%) in aHUS — ADAMTS13 <10% = TTP, NOT aHUS. "
            "DISTINGUISHING from Shiga-toxin HUS: stool culture/STEC serology negative in aHUS. "
            "Complement: C3 LOW (secondary consumption), C4 often normal (alternative pathway), CH50 low. "
            "CFH gene: heterozygous missense in SCR19-20 (endothelial binding domain) = aHUS-specific. "
            "AMD variant Y402H (p.Tyr402His): in SCR7; geographic atrophy + neovascular AMD. "
            "ECULIZUMAB: reverses active TMA in aHUS within days to weeks — DRAMATIC RESPONSE PATHOGNOMONIC."
        ),
        "treatment": (
            "aHUS (TMA PHENOTYPE): "
            "ECULIZUMAB (anti-C5, FDA 2011): FIRST-LINE — transforms outcomes; DRAMATIC TMA reversal. "
            "Ravulizumab (long-acting anti-C5): q8w dosing, non-inferior to eculizumab for aHUS. "
            "PLASMA EXCHANGE: bridge therapy until eculizumab available; temporarily replenishes Factor H. "
            "RENAL TRANSPLANT: perform only with prophylactic eculizumab (aHUS recurs post-transplant without it). "
            "MENINGOCOCCAL VACCINE ACWY + B: MANDATORY before eculizumab (blocks C5 -> Neisseria risk). "
            "AMD: anti-VEGF (ranibizumab/aflibercept) for neovascular AMD; monitoring for geographic atrophy. "
            "MPGN: ACE inhibitors; complement inhibition in research setting."
        ),
        "key_features": [
            "THREE PHENOTYPES: aHUS (AD), AMD common variant (Y402H), MPGN/C3GN (AR)",
            "aHUS: MAHA + thrombocytopenia + AKI — TMA without Shiga toxin or ADAMTS13 deficiency",
            "ADAMTS13 normal (>10%) in aHUS — key test to exclude TTP",
            "Eculizumab DRAMATICALLY reverses TMA in aHUS — transforms outcomes",
            "Y402H (p.Tyr402His): most common genetic risk factor for AMD (~50% population)",
            "C3 low (secondary consumption); C4 often normal in aHUS (alternative pathway)",
        ],
        "key_ddx": (
            "TTP: ADAMTS13 <10%; often autoimmune (anti-ADAMTS13); plasma exchange highly effective; eculizumab not first-line. "
            "STEC-HUS: positive stool E.coli O157:H7 or Shiga toxin serology; epidemic/children; no complement treatment. "
            "DIC: abnormal PT/APTT + fibrinogen low + D-dimer high — different coagulation profile. "
            "MPGN from C3 gene deficiency: C3 absent (gene defect); CFH gene normal; C3NeF may be present. "
            "Acquired anti-Factor H antibody: Factor H protein present but inhibited; anti-FH IgG detectable."
        ),
        "complement_pathway_affected": "Alternative pathway regulation (uncontrolled C3b deposition on host surfaces)",
        "neisseria_risk": "Moderate — MAC pathway functional in aHUS (C3b deposits but C5 intact); secondary Neisseria risk if eculizumab used",
        "sle_risk": "Low — ~10% SLE-like features (especially in MPGN/C3GN phenotype)",
        "ahus_risk": "VERY HIGH — primary disease phenotype (AD heterozygous missense SCR19-20)",
        "hae_risk": "None",
        "eculizumab_relevant": "YES — eculizumab FDA-approved for aHUS; VACCINE MANDATORY before starting",
        "onset_age": "aHUS: any age including children; triggered by infection/pregnancy; AMD: age >60; MPGN: teens to adults",
        "key_vaccine": "Meningococcal ACWY+B MANDATORY before eculizumab; update boosters q3-5y during treatment",
    },
    # -- SERPING1 -- Hereditary Angioedema -------------------------------------------------------
    {
        "gene": "SERPING1",
        "alt_name": (
            "SERPING1 (SERPING1-500aa-11q12.1 / AD -- Hereditary-Angioedema-HAE-Type-I-and-II -- "
            "MOST-CLINICALLY-RECOGNISED-HEREDITARY-COMPLEMENT-DISORDER-1:10000-1:50000 -- "
            "BRADYKININ-MEDIATED-NOT-Histamine-Antihistamines-And-Adrenaline-INEFFECTIVE -- "
            "LARYNGEAL-ANGIOEDEMA-FATAL-WITHOUT-Treatment-ACE-INHIBITORS-ABSOLUTELY-CONTRAINDICATED -- "
            "ABDOMINAL-ATTACKS-MISDIAGNOSED-Appendicitis-Unnecessary-Surgery-Classic-Trap)"
        ),
        "protein": (
            "SERPING1 -- 11q12.1 AD -- SERPING1-500aa -- "
            "C1-Inhibitor-C1-INH-105kDa-Serine-Protease-Inhibitor-Serpin-Family -- "
            "C1-INH-Inhibits-C1r-C1s-Classical-Pathway-Primary-Role-Complement -- "
            "C1-INH-ALSO-Inhibits-Kallikrein-Contact-Pathway-Factor-XIIa-Coagulation-Plasmin -- "
            "HAE-Type-I-85pct-Quantitative-Deficiency-C1-INH-Level-Low-Allelic-Loss-Haploinsufficiency -- "
            "HAE-Type-II-15pct-Qualitative-C1-INH-Present-But-Dysfunctional-Normal-Or-High-Antigen -- "
            "Kallikrein-Uninhibited-Bradykinin-Overproduced-Binds-BK2R-Endothelial-Vascular-Permeability -- "
            "Bradykinin-NOT-Histamine-Hence-Antihistamines-Fail-Adrenaline-Unreliable -- "
            "OMIM-Gene-606860-Disease-HAE1-106100-HAE2-Same-Locus"
        ),
        "locus": "11q12.1",
        "protein_size": "500 aa / ~105 kDa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency (HAE type I, 85%) or dominant negative dysfunctional C1-INH (HAE type II, 15%); "
            "one defective allele sufficient for disease (insufficient C1-INH from one normal allele); "
            "de novo mutations in ~25% (no family history); "
            "1:10,000-1:50,000 prevalence; most common hereditary complement disorder clinically recognised"
        ),
        "disease_category": "Hereditary Angioedema Type I/II — C1-INH deficiency (BRADYKININ-mediated; NOT histamine)",
        "disease_pathway": (
            "C1-INH (SERPING1) normally inhibits FOUR key proteases: "
            "(1) C1r/C1s — classical complement pathway; "
            "(2) Plasma kallikrein — contact activation pathway; "
            "(3) Factor XIIa — coagulation; "
            "(4) Plasmin — fibrinolysis. "
            "KEY PATHOMECHANISM: C1-INH deficiency -> UNINHIBITED PLASMA KALLIKREIN -> "
            "High-molecular-weight kininogen (HMK) cleavage -> excess BRADYKININ generation -> "
            "Bradykinin binds BK2R (bradykinin receptor 2) on endothelium -> "
            "increased vascular permeability -> non-pitting angioedema WITHOUT URTICARIA. "
            "Bradykinin-mediated angioedema: NO ITCH (no histamine); NO urticaria; responds to specific bradykinin-pathway drugs. "
            "Triggers: trauma, oestrogen, ACE inhibitors (block bradykinin breakdown -> attacks)."
        ),
        "pathognomonic": (
            "BRADYKININ-MEDIATED ANGIOEDEMA (non-pitting, no urticaria, no itch) — PATHOGNOMONIC for C1-INH deficiency/HAE. "
            "THREE SITES: (1) SKIN (non-pitting, non-urticarial swelling — face, extremities, genitals); "
            "(2) ABDOMEN (severe pain, vomiting, ascites — MISDIAGNOSED as appendicitis -> UNNECESSARY SURGERY); "
            "(3) LARYNX (airway obstruction -> FATAL without treatment). "
            "C4 LOW BETWEEN ATTACKS: C1-INH cannot stop C1s cleaving C4 -> C4 chronically low — reliable SCREENING test. "
            "C1q NORMAL: distinguishes from ACQUIRED C1-INH deficiency (in which C1q also low due to lymphoma/SLE). "
            "ABSOLUTE CONTRAINDICATION: ACE INHIBITORS block bradykinin breakdown -> accumulation -> fatal laryngeal attacks. "
            "ANTIHISTAMINES and ADRENALINE: INEFFECTIVE for bradykinin angioedema — wrong mechanism."
        ),
        "treatment": (
            "ACUTE ATTACK TREATMENT: "
            "C1-INH CONCENTRATE (Berinert 20 U/kg IV or Cinryze): first-line acute attack. "
            "ICATIBANT (bradykinin B2R antagonist, 30 mg SC self-injection): rapid acute relief — patient self-administration. "
            "Ecallantide (kallikrein inhibitor, 30 mg SC): acute attack. "
            "FRESH FROZEN PLASMA: emergency C1-INH source if concentrate unavailable. "
            "LONG-TERM PROPHYLAXIS: "
            "Lanadelumab (anti-plasma kallikrein, 300 mg SC q2-4w): most effective long-term prophylaxis. "
            "Garadacimab (anti-Factor XIIa, 200 mg SC monthly): emerging prophylaxis option. "
            "C1-INH concentrate (Cinryze 1000 U IV q3-4d): long-acting prophylaxis. "
            "ABSOLUTELY CONTRAINDICATED: ACE inhibitors (any), oestrogen-containing OCP/HRT in uncontrolled HAE. "
            "Patient education: SC self-injection kits + emergency medical alert card."
        ),
        "key_features": [
            "Bradykinin-mediated angioedema (NOT histamine) — antihistamines and adrenaline INEFFECTIVE",
            "Three sites: skin (non-pitting), abdomen (peritonism-like), larynx (FATAL)",
            "Abdominal attacks MISDIAGNOSED as appendicitis — surgical trap",
            "C4 LOW BETWEEN ATTACKS — reliable screening test; C1q NORMAL",
            "ACE inhibitors ABSOLUTELY CONTRAINDICATED (block bradykinin breakdown)",
            "Icatibant (bradykinin B2R antagonist) SC for acute attacks — patient self-injection",
        ],
        "key_ddx": (
            "Allergic angioedema (IgE/histamine): urticaria present; itch; antihistamine/adrenaline EFFECTIVE; C4 NORMAL. "
            "Acquired C1-INH deficiency: C1q LOW (also consumed) — lymphoma or SLE-associated; older onset; anti-C1-INH antibodies. "
            "ACE-inhibitor angioedema: drug history; no family history; C4 NORMAL; resolves with ACE-I cessation. "
            "HAE type III (FXII/Estrogen-dependent): normal C1-INH level AND function; triggered by oestrogen; FXII gene sequencing."
        ),
        "complement_pathway_affected": "Contact/kallikrein pathway (bradykinin excess); also classical pathway C1r/C1s inhibition lost",
        "neisseria_risk": "None — complement MAC pathway intact",
        "sle_risk": "None",
        "ahus_risk": "None",
        "hae_risk": "PRIMARY — HAE Type I (85%) and Type II (15%); laryngeal attacks FATAL without treatment",
        "eculizumab_relevant": "No",
        "onset_age": "Childhood to adolescence; mean first attack ~8-10 years; diagnosis often delayed 10+ years",
        "key_vaccine": "Standard vaccination schedule; no complement-specific vaccine requirement",
    },
    # -- C9 -- C9 Deficiency -------------------------------------------------------
    {
        "gene": "C9",
        "alt_name": (
            "C9 (C9-559aa-5p13.1 / AR -- C9-Deficiency -- "
            "MOST-COMMON-COMPLEMENT-DEFICIENCY-EAST-ASIAN-POPULATIONS-Japan-1:1000 -- "
            "JAPANESE-FOUNDER-p.Arg95His-85pct-Japanese-Cases -- "
            "SELECTIVE-NEISSERIA-MENINGITIDIS-SUSCEPTIBILITY-Serogroup-Y-Predominantly-Japan -- "
            "LOWER-SEVERITY-THAN-C5-C6-Some-Residual-MAC-C5b-C8-Partial-Activity)"
        ),
        "protein": (
            "C9 -- 5p13.1 AR -- C9-559aa -- "
            "C9-Complement-Component-9-63kDa-MAC-Pore-Forming-Subunit -- "
            "C9-Polymerises-12-18-Monomers-Around-C5b-678-Platform-Transmembrane-Pore -- "
            "MAC-C5b-C9-Pore-8-10nm-Diameter-Colloid-Osmotic-Lysis-Gram-Negative-Bacteria -- "
            "C9-Deficiency-Incomplete-MAC-C5b-C6-C7-C8-Still-Forms-Partial-Membrane-Destabilisation -- "
            "Lower-Severity-Than-C5-Deficiency-C5b-C8-Complex-Retains-Partial-Lytic-Activity -- "
            "p.Arg95His-Japanese-Founder-Variant-85pct-Japanese-C9-Deficiency-Cases -- "
            "Japan-Frequency-1:1000-Unique-Population-Enrichment-vs-1:1000000-Caucasians -- "
            "OMIM-Gene-120940-Disease-C9-Deficiency-613825"
        ),
        "locus": "5p13.1",
        "protein_size": "559 aa / ~63 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; "
            "p.Arg95His Japanese founder — 85% of Japanese C9 deficiency cases; "
            "Japan frequency ~1:1,000 (vs ~1:1,000,000 Caucasians) — extreme East Asian enrichment; "
            "heterozygous: intermediate CH50 (~50% normal); usually asymptomatic"
        ),
        "disease_category": "C9 Deficiency — Terminal MAC pore deficiency (East Asian enriched; Neisseria selective)",
        "disease_pathway": (
            "C9 is the polymerising subunit of the Membrane Attack Complex (MAC). "
            "Typically 12-18 C9 monomers assemble around the C5b-678 platform -> "
            "transmembrane pore (8-10 nm diameter) insertion into lipid bilayer -> "
            "ion flux + osmotic lysis of gram-negative bacteria (particularly Neisseria which lacks peptidoglycan protection). "
            "C9 deficiency -> incomplete MAC (C5b-C8 complex remains) -> "
            "PARTIAL MEMBRANE DESTABILISATION — C5b-C8 alone has some lytic activity (unlike C5/C6 deficiency). "
            "Clinical consequence: LOWER SEVERITY NEISSERIA SUSCEPTIBILITY than C5 deficiency. "
            "Selective: N.meningitidis (serogroup Y predominant in Japan) + N.gonorrhoeae. "
            "Not susceptible to encapsulated bacteria requiring opsonisation (C3b intact)."
        ),
        "pathognomonic": (
            "RECURRENT NEISSERIA MENINGITIDIS — SEROGROUP Y PREDOMINANTLY IN JAPAN — SELECTIVE SUSCEPTIBILITY PATHOGNOMONIC. "
            "East Asian ancestry (particularly Japanese) + recurrent meningococcal disease + CH50 absent = C9 deficiency until proven otherwise. "
            "CH50: ABSENT (ZERO) — MAC pore formation abolished. "
            "C9 LEVEL: ABSENT (protein measurement). "
            "All other complement proteins NORMAL (C1q, C4, C2, C3, C5, C6, C7, C8 all present and normal). "
            "AP50: ABSENT (alternative pathway also requires C9 for MAC formation). "
            "LOWER SEVERITY vs C5/C6 deficiency: C5b-C8 complex retains partial membrane-destabilising activity. "
            "p.Arg95His: Japanese-specific variant; test first in East Asian patients before full sequencing."
        ),
        "treatment": (
            "MENINGOCOCCAL VACCINE ACWY (quadrivalent, covers serogroups A/C/W135/Y): MANDATORY — especially serogroup Y (Japan). "
            "MENINGOCOCCAL B VACCINE (Bexsero/Trumenba): MANDATORY — coverage for serogroup B. "
            "BOOSTER every 3-5 years: meningococcal vaccine immunity wanes; re-boost throughout adulthood. "
            "Antibiotic prophylaxis (penicillin V): recommended during high-risk periods. "
            "Family screening: test siblings/first-degree relatives (p.Arg95His in Japanese families). "
            "C9 deficiency DOES NOT REQUIRE ECULIZUMAB: eculizumab blocks C5 (upstream of C9); C5 functional in C9 deficiency. "
            "Gonorrhoea: standard antibiotic treatment; counsel regarding DGI risk."
        ),
        "key_features": [
            "Most common complement deficiency in East Asia; Japan 1:1,000",
            "Japanese founder variant p.Arg95His (85% of Japanese cases)",
            "Selective Neisseria meningitidis susceptibility; serogroup Y predominant in Japan",
            "CH50 ZERO, AP50 ZERO; all other complement proteins NORMAL",
            "LOWER severity than C5 deficiency — C5b-C8 retains partial lytic activity without C9",
            "C9 deficiency does NOT need eculizumab (C5 functional; eculizumab targets C5, not C9)",
        ],
        "key_ddx": (
            "C5/C6/C7/C8 deficiency: identical Neisseria phenotype; all have CH50+AP50 absent; C9 NORMAL — check all terminal components. "
            "Properdin deficiency (X-linked, CFP): ONLY Neisseria; AP50 absent, CH50 normal — alternative pathway specific. "
            "C9 partial (heterozygous): intermediate CH50 (~50%); usually asymptomatic; p.Arg95His carrier in Japan. "
            "C3 deficiency: CH50+AP50 absent; C3 absent; ALL encapsulated bacteria susceptible (not narrow Neisseria only)."
        ),
        "complement_pathway_affected": "Terminal MAC pore formation (C5-C8 complex intact; C9 polymerisation absent)",
        "neisseria_risk": "HIGH — primary susceptibility; serogroup Y characteristic in Japanese; lower severity vs C5 deficiency",
        "sle_risk": "Low — ~5% SLE-like features",
        "ahus_risk": "None",
        "hae_risk": "None",
        "eculizumab_relevant": "No — C9 deficiency does not benefit from eculizumab (C5 functional and intact)",
        "onset_age": "Any age; young adults; recurrence typical; family history often identified after index case",
        "key_vaccine": "Meningococcal ACWY + B MANDATORY; booster every 3-5 years; penicillin prophylaxis in high-risk",
    },
]

PATIENTS_PER_GENE = 40


def _make_cohort(gene_entry, seed):
    """Generate 40 reproducible synthetic patients for a single complement gene."""
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    patients = []

    for i in range(PATIENTS_PER_GENE):
        age_at_dx = round(rng.uniform(0.5, 65.0), 1)
        sex = rng.choice(["M", "F", "F", "M"])

        # CH50 (% of normal total haemolytic complement) — gene-specific
        if gene == "C1QB":
            ch50_pct = round(rng.uniform(0.0, 5.0), 1)
        elif gene == "C4A":
            ch50_pct = round(rng.uniform(40.0, 60.0), 1)
        elif gene == "C2":
            ch50_pct = round(rng.uniform(0.0, 5.0), 1)
        elif gene == "C3":
            ch50_pct = round(rng.uniform(0.0, 3.0), 1)
        elif gene == "C5":
            ch50_pct = round(rng.uniform(0.0, 3.0), 1)
        elif gene == "CFH":
            ch50_pct = round(rng.uniform(10.0, 55.0), 1)
        elif gene == "SERPING1":
            ch50_pct = round(rng.uniform(70.0, 105.0), 1)
        elif gene == "C9":
            ch50_pct = round(rng.uniform(0.0, 5.0), 1)
        else:
            ch50_pct = round(rng.uniform(0.0, 10.0), 1)

        # AP50 (alternative pathway haemolytic)
        if gene in ("C3", "C5", "C9"):
            ap50_pct = round(rng.uniform(0.0, 3.0), 1)
        elif gene == "CFH":
            ap50_pct = round(rng.uniform(5.0, 40.0), 1)
        elif gene == "C2":
            ap50_pct = round(rng.uniform(85.0, 110.0), 1)
        elif gene == "SERPING1":
            ap50_pct = round(rng.uniform(85.0, 110.0), 1)
        else:
            ap50_pct = round(rng.uniform(80.0, 110.0), 1)

        # C1q level
        if gene == "C1QB":
            c1q_level = "absent"
        elif gene == "SERPING1":
            c1q_level = "normal"
        else:
            c1q_level = "normal"

        # C3 level
        if gene == "C3":
            c3_level = "absent"
        elif gene in ("CFH", "C2"):
            c3_level = rng.choice(["low", "low", "normal"])
        else:
            c3_level = "normal"

        # C4 level
        if gene in ("C1QB", "C2"):
            c4_level = rng.choice(["low", "absent", "absent"])
        elif gene == "C4A":
            c4_level = "low"
        elif gene == "SERPING1":
            c4_level = "low"
        else:
            c4_level = "normal"

        # Neisseria susceptibility
        if gene == "C5":
            neisseria_susceptibility = True
        elif gene == "C9":
            neisseria_susceptibility = True
        elif gene == "C3":
            neisseria_susceptibility = rng.random() < 0.80
        elif gene in ("C1QB", "C2"):
            neisseria_susceptibility = rng.random() < 0.10
        elif gene == "CFH":
            neisseria_susceptibility = rng.random() < 0.20
        else:
            neisseria_susceptibility = False

        # SLE features
        if gene == "C1QB":
            sle_features = rng.random() < 0.90
        elif gene == "C4A":
            sle_features = rng.random() < 0.70
        elif gene == "C2":
            sle_features = rng.random() < 0.40
        elif gene == "C3":
            sle_features = rng.random() < 0.10
        elif gene == "C5":
            sle_features = rng.random() < 0.05
        elif gene == "CFH":
            sle_features = rng.random() < 0.10
        elif gene == "SERPING1":
            sle_features = False
        elif gene == "C9":
            sle_features = rng.random() < 0.05
        else:
            sle_features = False

        # aHUS phenotype
        if gene == "CFH":
            aHUS_phenotype = rng.random() < 0.60
        else:
            aHUS_phenotype = False

        # Laryngeal angioedema (SERPING1 only)
        if gene == "SERPING1":
            laryngeal_angioedema = rng.random() < 0.40
        else:
            laryngeal_angioedema = False

        # Abdominal angioedema attack (SERPING1 only)
        if gene == "SERPING1":
            abdominal_attack = rng.random() < 0.65
        else:
            abdominal_attack = False

        # complement_pathway_affected
        pathway_map = {
            "C1QB": "classical",
            "C4A": "classical",
            "C2": "classical",
            "C3": "all_pathways",
            "C5": "terminal",
            "CFH": "regulatory",
            "SERPING1": "contact_kallikrein",
            "C9": "terminal",
        }
        complement_pathway_affected = pathway_map.get(gene, "unknown")

        # treatment assigned
        if gene == "C1QB":
            treatment = rng.choice([
                "HCQ + MMF (SLE management)",
                "HCQ + prednisolone",
                "HCQ + MMF + belimumab",
                "plasma infusion (experimental) + HCQ",
            ])
        elif gene == "C4A":
            treatment = rng.choice([
                "HCQ + MMF (SLE management)",
                "HCQ + prednisolone",
                "HCQ + MMF + voclosporin (LN)",
                "HCQ + azathioprine",
            ])
        elif gene == "C2":
            treatment = rng.choice([
                "penicillin prophylaxis + PCV13/PPSV23",
                "HCQ + MMF (SLE) + penicillin prophylaxis",
                "penicillin prophylaxis + HiB + MenACWY",
                "HCQ + prednisolone + antibiotic prophylaxis",
            ])
        elif gene == "C3":
            treatment = rng.choice([
                "penicillin prophylaxis + PCV13/PPSV23 + HiB + MenACWY+B",
                "plasma infusion bridge + antibiotic prophylaxis",
                "antibiotic prophylaxis + all encapsulated bacterial vaccines",
                "FFP acute + penicillin prophylaxis long-term",
            ])
        elif gene == "C5":
            treatment = rng.choice([
                "MenACWY + MenB + penicillin prophylaxis",
                "meningococcal vaccines + penicillin V 250mg BD",
                "MenACWY + MenB booster + antibiotic prophylaxis",
                "penicillin prophylaxis + all meningococcal vaccines",
            ])
        elif gene == "CFH":
            treatment = rng.choice([
                "eculizumab (aHUS) + MenACWY + MenB",
                "ravulizumab (aHUS) + meningococcal vaccination",
                "plasma exchange (bridge) + eculizumab",
                "anti-VEGF (AMD) + monitoring",
                "ACE inhibitor + eGFR monitoring (MPGN)",
            ])
        elif gene == "SERPING1":
            treatment = rng.choice([
                "C1-INH concentrate (Berinert) acute + lanadelumab prophylaxis",
                "icatibant SC acute + lanadelumab prophylaxis",
                "C1-INH concentrate acute + C1-INH concentrate prophylaxis (Cinryze)",
                "icatibant SC acute + garadacimab prophylaxis",
                "ecallantide acute + lanadelumab prophylaxis",
                "C1-INH concentrate + patient self-injection training",
            ])
        elif gene == "C9":
            treatment = rng.choice([
                "MenACWY + MenB + penicillin prophylaxis",
                "meningococcal vaccines + penicillin V 250mg BD",
                "MenACWY + MenB booster + antibiotic prophylaxis",
                "MenACWY (q3y booster) + MenB + penicillin V",
            ])
        else:
            treatment = "supportive + vaccines"

        # outcome
        if gene in ("C1QB", "C4A"):
            outcome = rng.choice([
                "controlled", "controlled", "remission", "remission",
                "end_stage_renal", "controlled",
            ])
        elif gene == "C2":
            outcome = rng.choice([
                "controlled", "controlled", "recurrent_infections",
                "remission", "controlled",
            ])
        elif gene == "C3":
            outcome = rng.choice([
                "recurrent_infections", "controlled", "controlled",
                "end_stage_renal", "recurrent_infections",
            ])
        elif gene == "C5":
            outcome = rng.choice([
                "controlled", "recurrent_infections", "controlled",
                "controlled", "fatal_without_treatment",
            ])
        elif gene == "CFH":
            outcome = rng.choice([
                "remission", "remission", "controlled",
                "end_stage_renal", "remission",
            ])
        elif gene == "SERPING1":
            outcome = rng.choice([
                "controlled", "controlled", "remission", "remission",
                "fatal_without_treatment", "controlled",
            ])
        elif gene == "C9":
            outcome = rng.choice([
                "controlled", "controlled", "recurrent_infections",
                "remission", "controlled",
            ])
        else:
            outcome = "controlled"

        # vaccine_status
        vaccine_status = rng.choice([
            "fully vaccinated (MenACWY+B + PCV13 + HiB)",
            "partially vaccinated (MenACWY only)",
            "unvaccinated at diagnosis",
            "fully vaccinated with boosters up to date",
            "vaccinated PCV13 + HiB only",
        ])

        # east_asian_ancestry flag (relevant for C9)
        east_asian_ancestry = (
            rng.random() < 0.85 if gene == "C9"
            else rng.random() < 0.10
        )

        patients.append({
            "patient_id": f"{gene}-{seed}-{i + 1:03d}",
            "gene": gene,
            "age_at_diagnosis_years": age_at_dx,
            "sex": sex,
            "ch50_pct": ch50_pct,
            "ap50_pct": ap50_pct,
            "c1q_level": c1q_level,
            "c3_level": c3_level,
            "c4_level": c4_level,
            "neisseria_susceptibility": neisseria_susceptibility,
            "sle_features": sle_features,
            "aHUS_phenotype": aHUS_phenotype,
            "laryngeal_angioedema": laryngeal_angioedema,
            "abdominal_attack": abdominal_attack,
            "complement_pathway_affected": complement_pathway_affected,
            "treatment": treatment,
            "outcome": outcome,
            "vaccine_status": vaccine_status,
            "east_asian_ancestry": east_asian_ancestry,
        })

    return patients


def generate_overview():
    """Return atlas-wide metrics + per-gene summary dict."""
    all_patients = []
    for idx, entry in enumerate(COMPLEMENT_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    neisseria_count = sum(1 for p in all_patients if p["neisseria_susceptibility"])
    sle_count = sum(1 for p in all_patients if p["sle_features"])
    ahus_count = sum(1 for p in all_patients if p["aHUS_phenotype"])
    laryngeal_count = sum(1 for p in all_patients if p["laryngeal_angioedema"])
    remission_count = sum(
        1 for p in all_patients if p["outcome"] in ("remission", "controlled")
    )
    fatal_count = sum(
        1 for p in all_patients if p["outcome"] == "fatal_without_treatment"
    )

    gene_summary = {}
    for idx, entry in enumerate(COMPLEMENT_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        n = len(cohort)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic_excerpt": entry["pathognomonic"][:300],
            "complement_pathway_affected": entry["complement_pathway_affected"],
            "n_patients": n,
            "avg_ch50_pct": round(sum(p["ch50_pct"] for p in cohort) / n, 1),
            "neisseria_susceptibility_pct": round(
                100 * sum(1 for p in cohort if p["neisseria_susceptibility"]) / n, 1
            ),
            "sle_features_pct": round(
                100 * sum(1 for p in cohort if p["sle_features"]) / n, 1
            ),
            "ahus_phenotype_pct": round(
                100 * sum(1 for p in cohort if p["aHUS_phenotype"]) / n, 1
            ),
            "laryngeal_angioedema_pct": round(
                100 * sum(1 for p in cohort if p["laryngeal_angioedema"]) / n, 1
            ),
            "remission_or_controlled_pct": round(
                100
                * sum(1 for p in cohort if p["outcome"] in ("remission", "controlled"))
                / n,
                1,
            ),
            "key_vaccine": entry["key_vaccine"],
            "eculizumab_relevant": entry["eculizumab_relevant"],
        }

    return {
        "atlas": "Hereditary-Complement-Disorder-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Complement Deficiency Reference — "
            "C1QB/C4A/C2/C3/C5/CFH/SERPING1/C9"
        ),
        "genes_covered": [e["gene"] for e in COMPLEMENT_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "neisseria_susceptibility_pct": round(100 * neisseria_count / total, 1),
            "sle_features_pct": round(100 * sle_count / total, 1),
            "ahus_phenotype_pct": round(100 * ahus_count / total, 1),
            "laryngeal_angioedema_pct": round(100 * laryngeal_count / total, 1),
            "remission_or_controlled_pct": round(100 * remission_count / total, 1),
            "fatal_without_treatment_pct": round(100 * fatal_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    """Return per-gene detailed breakdown with sample patients."""
    breakdown = []
    for idx, entry in enumerate(COMPLEMENT_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        n = len(cohort)
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
            "complement_pathway_affected": entry["complement_pathway_affected"],
            "neisseria_risk": entry["neisseria_risk"],
            "sle_risk": entry["sle_risk"],
            "ahus_risk": entry["ahus_risk"],
            "hae_risk": entry["hae_risk"],
            "eculizumab_relevant": entry["eculizumab_relevant"],
            "onset_age": entry["onset_age"],
            "key_vaccine": entry["key_vaccine"],
            "n_patients": n,
            "avg_ch50_pct": round(sum(p["ch50_pct"] for p in cohort) / n, 1),
            "avg_ap50_pct": round(sum(p["ap50_pct"] for p in cohort) / n, 1),
            "neisseria_susceptibility_pct": round(
                100 * sum(1 for p in cohort if p["neisseria_susceptibility"]) / n, 1
            ),
            "sle_features_pct": round(
                100 * sum(1 for p in cohort if p["sle_features"]) / n, 1
            ),
            "ahus_phenotype_pct": round(
                100 * sum(1 for p in cohort if p["aHUS_phenotype"]) / n, 1
            ),
            "laryngeal_angioedema_pct": round(
                100 * sum(1 for p in cohort if p["laryngeal_angioedema"]) / n, 1
            ),
            "abdominal_attack_pct": round(
                100 * sum(1 for p in cohort if p["abdominal_attack"]) / n, 1
            ),
            "remission_or_controlled_pct": round(
                100
                * sum(
                    1 for p in cohort if p["outcome"] in ("remission", "controlled")
                )
                / n,
                1,
            ),
            "fatal_without_treatment_pct": round(
                100
                * sum(1 for p in cohort if p["outcome"] == "fatal_without_treatment")
                / n,
                1,
            ),
            "c4_low_pct": round(
                100
                * sum(1 for p in cohort if p["c4_level"] in ("low", "absent"))
                / n,
                1,
            ),
            "c3_low_pct": round(
                100
                * sum(1 for p in cohort if p["c3_level"] in ("low", "absent"))
                / n,
                1,
            ),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    """Return gene_entries dict + complement_glossary dict."""
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
                "complement_pathway_affected": entry["complement_pathway_affected"],
                "neisseria_risk": entry["neisseria_risk"],
                "sle_risk": entry["sle_risk"],
                "ahus_risk": entry["ahus_risk"],
                "hae_risk": entry["hae_risk"],
                "eculizumab_relevant": entry["eculizumab_relevant"],
                "onset_age": entry["onset_age"],
                "key_vaccine": entry["key_vaccine"],
            }
            for entry in COMPLEMENT_GENES
        },
        "complement_glossary": {
            "Complement System Overview": (
                "The complement system comprises ~30 serum and cell-surface proteins forming a proteolytic cascade "
                "with three initiation pathways and one shared terminal pathway. "
                "CLASSICAL PATHWAY: C1q binds immune complexes/apoptotic cells -> C1r/C1s activation -> C4 cleavage -> C4b2a (C3 convertase). "
                "ALTERNATIVE PATHWAY: Spontaneous C3 hydrolysis (tick-over) + Factor B/D/Properdin -> C3bBb (C3 convertase); activated on microbial surfaces. "
                "LECTIN PATHWAY: MBL or Ficolins bind mannose/GlcNAc on microbes -> MASP-1/MASP-2 -> C4/C2 cleavage (same C3 convertase as classical). "
                "TERMINAL PATHWAY (shared): C3b generation -> C5 convertases -> C5 cleavage -> C5b + C5a; "
                "C5b + C6 + C7 + C8 + C9 (polymerised) -> MAC (C5b-C9) -> transmembrane pore -> bacteriolysis. "
                "THREE EFFECTOR FUNCTIONS: (1) Opsonisation (C3b -> CR1/CR3 -> phagocytosis); "
                "(2) Anaphylatoxins (C3a/C5a -> mast cells, neutrophil chemotaxis); "
                "(3) Direct bacteriolysis (MAC). "
                "REGULATION: Factor H/I (alternative pathway), C1-INH (classical/contact), CD46/DAF/CD59 (cell surfaces) — "
                "prevent host cell damage while targeting microbial surfaces."
            ),
            "CH50 and AP50 Interpretation": (
                "CH50 (Total Haemolytic Complement): measures CLASSICAL pathway function via sheep-red-blood-cell lysis assay. "
                "Normal range: 75-160 units/mL (lab-dependent). "
                "CH50 = ZERO: classical or terminal pathway component absent (C1q, C4, C2, C3, C5-C9). "
                "CH50 = LOW (not zero): partial deficiency or consumption (active SLE, aHUS, C4A heterozygous null). "
                "AP50 (Alternative Pathway Haemolytic Complement): measures alternative pathway via rabbit-red-blood-cell lysis. "
                "AP50 = ZERO + CH50 = ZERO: C3, C5, C6, C7, C8, or C9 deficiency (shared terminal pathway). "
                "AP50 = ZERO + CH50 = NORMAL: Properdin deficiency or Factor D deficiency (alternative pathway only). "
                "AP50 = NORMAL + CH50 = ZERO: C1, C2, or C4 deficiency (classical pathway only). "
                "INTERPRETATION ALGORITHM: "
                "1. Both zero -> deficiency at or downstream of C3 -> measure C3, C4, C5-C9 individually. "
                "2. CH50 only low/zero -> measure C1q, C4, C2 -> identify classical block. "
                "3. AP50 only low/zero -> measure Factor B, Factor D, Properdin -> alternative pathway block. "
                "4. Both normal -> complement likely intact; consider acquired consumption or functional assay issues."
            ),
            "SERPING1/HAE and Bradykinin vs Histamine": (
                "Hereditary Angioedema (HAE) from SERPING1 deficiency is BRADYKININ-MEDIATED — not histamine. "
                "This is the single most important clinical distinction for HAE management. "
                "BRADYKININ pathway: C1-INH deficiency -> uninhibited plasma kallikrein -> "
                "cleavage of HMK (high-MW kininogen) -> excess bradykinin -> BK2R activation on endothelium -> "
                "vascular permeability, non-pitting angioedema WITHOUT urticaria, WITHOUT itch. "
                "HISTAMINE pathway (allergic angioedema): IgE + allergen -> mast cell degranulation -> histamine -> "
                "urticaria + itch + angioedema; rapid onset (seconds-minutes); urticaria almost always present. "
                "CLINICAL CONSEQUENCE: "
                "Antihistamines (H1/H2 blockers): INEFFECTIVE in HAE — wrong mechanism. "
                "Adrenaline/epinephrine: UNRELIABLE in HAE (no specific mechanism). "
                "Correct treatment: C1-INH concentrate, icatibant (BK2R antagonist), or ecallantide (kallikrein inhibitor). "
                "ACE INHIBITORS: block bradykinin breakdown (kininase II = ACE) -> accumulation -> fatal laryngeal attacks -> ABSOLUTELY CONTRAINDICATED. "
                "C4 LOW BETWEEN ATTACKS: reliable screening test (C1-INH cannot stop C1s cleaving C4 even when asymptomatic). "
                "C1q NORMAL: distinguishes hereditary HAE from ACQUIRED C1-INH deficiency (lymphoma/SLE = C1q also low)."
            ),
            "Eculizumab and Neisseria Risk": (
                "Eculizumab (anti-C5 monoclonal antibody) blocks C5 cleavage -> no C5b -> no MAC formation. "
                "This PHARMACOLOGICALLY MIMICS C5 DEFICIENCY for the purpose of Neisseria susceptibility. "
                "MECHANISM: N.meningitidis and N.gonorrhoeae resist opsonin-mediated killing (sialylated LPS); "
                "they require MAC (direct lysis) for clearance. Blocking C5 -> no MAC -> bacteraemia/meningitis risk. "
                "RISK MAGNITUDE: 1,000-2,000 fold increased Neisseria meningitidis risk in eculizumab-treated patients vs general population. "
                "MANDATORY BEFORE ECULIZUMAB (any indication — PNH, aHUS, NMO, gMG): "
                "(1) Meningococcal conjugate ACWY (MenACWY) — minimum 2 weeks before first dose; "
                "(2) Meningococcal B (Bexsero/Trumenba) — 2 doses minimum; "
                "(3) Penicillin V prophylaxis 250-500 mg BD for duration of therapy; "
                "(4) Patient counselling: any fever/rash/headache/neck stiffness = emergency; "
                "(5) GP letter: patient on eculizumab — Neisseria risk elevated. "
                "BOOSTER every 3-5 years: meningococcal vaccine immunity wanes on chronic eculizumab. "
                "Ravulizumab (long-acting anti-C5, q8w): same risk profile; same vaccine/prophylaxis requirements."
            ),
            "Factor H and Three Phenotypes": (
                "CFH encodes Factor H — a single gene that causes THREE clinically distinct diseases depending on variant type and zygosity. "
                "PHENOTYPE 1 — aHUS (Atypical Haemolytic Uraemic Syndrome): "
                "  Variant: heterozygous missense in SCR19-20 (endothelial-binding domain); AD; incomplete penetrance ~50%. "
                "  Mechanism: insufficient Factor H on endothelial surfaces -> uncontrolled C3b deposition -> TMA. "
                "  Features: MAHA + thrombocytopenia + AKI; triggers = infection/pregnancy/complement stress. "
                "  Treatment: eculizumab/ravulizumab DRAMATICALLY reverses TMA; plasma exchange as bridge. "
                "PHENOTYPE 2 — Age-Related Macular Degeneration (AMD): "
                "  Variant: common SNP p.Tyr402His (Y402H) in SCR7; present in ~35-40% Europeans. "
                "  Mechanism: reduced Factor H binding to heparan sulfate in retinal RPE -> chronic complement activation -> drusen -> AMD. "
                "  Treatment: anti-VEGF (neovascular AMD); monitoring for geographic atrophy. "
                "PHENOTYPE 3 — MPGN/C3 Glomerulopathy: "
                "  Variant: biallelic (AR) complete loss-of-function. "
                "  Mechanism: complete Factor H absence -> C3 consumed (low C3) -> C3 deposition in glomeruli -> MPGN type 2/C3GN. "
                "  Treatment: ACE inhibitors + monitor; complement inhibition experimental. "
                "KEY DISTINGUISHING POINT: same gene CFH — variant TYPE and ZYGOSITY determines which phenotype."
            ),
            "C3 vs Late Complement Deficiency": (
                "A key clinical distinction: C3 deficiency has a BROADER susceptibility than late complement (C5-C9) deficiency. "
                "C3 DEFICIENCY (upstream hub): "
                "  -- ALL THREE pathways impaired (C3 is convergence point) -> no C3b opsonisation, no MAC. "
                "  -- Susceptible to ALL encapsulated bacteria: S.pneumoniae, H.influenzae, N.meningitidis. "
                "  -- CH50 ZERO + AP50 ZERO; C3 ABSENT. "
                "  -- More severe than late complement deficiency. "
                "LATE COMPLEMENT DEFICIENCY (C5-C9 — terminal pathway): "
                "  -- Opsonisation (C3b) INTACT -> S.pneumoniae and H.influenzae CLEARED by phagocytosis. "
                "  -- ONLY Neisseria species susceptible (require MAC for killing). "
                "  -- CH50 ZERO + AP50 ZERO; C3 NORMAL. "
                "DISTINGUISHING TEST: C3 LEVEL. "
                "  C3 absent + CH50/AP50 zero = C3 deficiency. "
                "  C3 normal + CH50/AP50 zero = late complement (C5-C9) deficiency. "
                "SPECTRUM OF INFECTION RISK (broadest to narrowest): "
                "  C3 deficiency (all encapsulated bacteria) > "
                "  C5 deficiency (Neisseria only) > "
                "  C9 deficiency (Neisseria, lower severity — C5b-C8 partial lysis retained). "
                "C3 NEPHRITIC FACTOR (C3NeF): acquired autoantibody that stabilises C3bBb -> secondary C3 consumption -> "
                "same low C3 level but C3 GENE NORMAL -> associated with MPGN type 2 and partial lipodystrophy."
            ),
            "Acquired Complement Deficiency DDx": (
                "Acquired complement deficiency must be distinguished from hereditary deficiency before genetic counselling. "
                "KEY PRINCIPLE: complement proteins can be CONSUMED in disease states — distinguish by checking if levels recover with treatment. "
                "ACQUIRED C1q DEFICIENCY: C1q LOW but GENE NORMAL; caused by lymphoma, SLE, myeloma. "
                "  -- Anti-C1q antibodies detectable; older onset; C1q rises with lymphoma treatment. "
                "  -- Distinguishes from hereditary C1QB deficiency (C1q genetically absent; younger onset; family history). "
                "ACQUIRED C4 DEPLETION in SLE: C4 low during active SLE (consumed); normalises with treatment. "
                "  -- C4A gene dosage normal on MLPA — distinguishes from hereditary C4A null. "
                "ACQUIRED C3 DEPLETION: "
                "  -- Active SLE: ANA/dsDNA positive; C3 recovers with treatment. "
                "  -- Factor H/I deficiency (secondary C3 consumption): C3 gene normal; Factor H/I level low. "
                "  -- MPGN + C3NeF: C3 low; C3NeF IgG autoantibody; C3 gene intact. "
                "ACQUIRED C1-INH DEFICIENCY: C1q also LOW (consumed); anti-C1-INH antibody; lymphoma/SLE associated. "
                "  -- Distinguishes from HAE (hereditary SERPING1 defect): C1q NORMAL in HAE, LOW in acquired. "
                "DIAGNOSTIC RULE: measure complement BETWEEN clinical episodes and after treatment to assess if levels normalise. "
                "Gene sequencing confirms hereditary deficiency when phenotype and family history are unclear."
            ),
            "Complement Deficiency Vaccination Protocol": (
                "Vaccination requirements differ by which complement pathway is deficient. "
                "CLASSICAL PATHWAY (C1QB, C4A, C2 deficiency): "
                "  Classical opsonisation impaired but alternative pathway partially compensates. "
                "  Vaccinate: Pneumococcal (PCV13 + PPSV23), H.influenzae type B, Meningococcal ACWY + B. "
                "  Penicillin prophylaxis: consider in C2 deficiency with recurrent infections. "
                "C3 DEFICIENCY (all pathways impaired — MOST COMPREHENSIVE VACCINATION): "
                "  MANDATORY: Pneumococcal (PCV13 + PPSV23) + H.influenzae B + Meningococcal ACWY + B. "
                "  MANDATORY antibiotic prophylaxis (penicillin V lifelong). "
                "  Boosters: Pneumococcal every 5 years (PPSV23); Meningococcal every 3-5 years. "
                "TERMINAL PATHWAY (C5, C9 deficiency — NEISSERIA ONLY): "
                "  MANDATORY: Meningococcal ACWY + B (BOTH types; covers uncommon serogroups). "
                "  Booster: every 3-5 years (waning immunity). "
                "  Penicillin prophylaxis: recommended. "
                "FACTOR H DEFICIENCY (CFH — for eculizumab-treated patients): "
                "  MANDATORY before eculizumab: MenACWY + MenB minimum 2 weeks before first dose. "
                "  Penicillin V during eculizumab therapy. "
                "SERPING1/HAE: Standard vaccination schedule — complement MAC intact; no encapsulated bacterial risk. "
                "GENERAL PRINCIPLES: "
                "  -- Live attenuated vaccines: safe (innate immunodeficiency, adaptive immunity intact). "
                "  -- Record ALL vaccinations; primary series before complement deficiency diagnosis if identified early. "
                "  -- Household contacts: ensure vaccinated (herd protection reduces index patient exposure risk)."
            ),
        },
    }


# Module-level aliases so api_backend.py can call atlas_.overview() / breakdown() / definitions()
overview = generate_overview
breakdown = generate_breakdown
definitions = generate_definitions


if __name__ == "__main__":
    import json

    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 — C1QB only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first gene entry — C1QB) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
    print("\n=== GLOSSARY (first entry) ===")
    first_gloss = next(iter(defn["complement_glossary"]))
    print(f"{first_gloss}:\n{defn['complement_glossary'][first_gloss][:600]}")
